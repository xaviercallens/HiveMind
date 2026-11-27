#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cloud Run PR #949 live quorum/arena parity gate and <60s VS Code one-click quick-connect preset.

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library, proxy-free urllib.request.ProxyHandler({}), and strict outbound allowlisting:
  1. Strictly Local Personal-PC & Hobby-Workspace Compliance Gate (`evaluate_personal_pc_compliance`):
     inspects environment variables (`AMADEUS*`, `USERDNSDOMAIN`, `USERDOMAIN`, `HTTP_PROXY`,
     `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, `COMPUTERNAME`, `HOSTNAME`, `ZSCALER_CERT_PATH`,
     `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`), workspace path
     (including resolved symlinks), and `.git/config` remote URLs across parent directories and
     git worktree `gitdir:`/`commondir` indirections before any file write or outbound call. Fails
     closed with exit code 2 (0 file writes, 0 network calls, 0 subprocess calls, 0 raw leaks)
     while keeping `--uninstall` local preset cleanup always permitted.
  2. Zero-Inference Cloud Run Holiday-Route & PR #949 Quorum/Arena Parity Gate
     (`verify_cloud_run_holiday_parity`, `verify_pr949_vscode_quorum_and_arena_parity`,
     `--check-parity`, `--pr949-quorum-gate`, `--require-holiday-routes`, `--require-live-quorum`,
     `--require-ces-slo`, `--check-arena-summary`, `--verify-idempotency-replay`,
     `--run-deploy-dry-run`, `--expected-sha`): probes `GET /health`, `GET /v1/models`,
     `/noel`, `/quick`, `/raid-noel`, `/arena-noel`, `/v1/hivemind/onboarding/manifest`,
     `/v1/hivemind/onboarding/feedback/summary`, `/v1/hivemind/raid/status`,
     `/v1/hivemind/producers/fleet`, and `/v1/hivemind/arena/summary` without touching inference.
  3. <60s One-Click VS Code Quick-Connect Preset Generator & Atomic `0600` Output / CES Spool
     (`install_vscode_quick_preset`, `run_vscode_setup`, `--install-preset`, `--vscode-setup`,
     `--workspace`, `--cwd`, `--output`, `--markdown`, `--uninstall`, `--first-heartbeat`,
     `--submit-ces`, `--profile dev|gamer`, `--ease-rating 1..5`, `--had-game-lag`, `--lang fr|en`,
     `--json`): configures `0600` `~/.hivemind/credentials.json` and Continue YAML, atomically merges
     non-secret `.vscode/tasks.json` and `.vscode/settings.json` with `.bak` backups, runs `<1.5s`
     Steam worker yield and `<=70 C` thermal guard + OpenTrainer heartbeat, and submits/replays CES.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen-science project licensed under AGPL-3.0.
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
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA = "hivemind.noel_vscode_quick_preset/1"
RECEIPT_OBJECT = "hivemind.noel_vscode_quick_preset_receipt"
CES_QUEUE_SCHEMA = "hivemind.vscode_setup_ces_queue/1"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
FLEET_OBJECT = "hivemind.producer_fleet"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_MODELS: tuple[str, str] = ("hivemind/coder", "hivemind/bee-stars")
DEFAULT_MAX_TOKENS = 2048
DEFAULT_HANDLE = "noel-dev"
DEFAULT_ROLE = "prosumer"
DEFAULT_HOST_LABEL = "personal-pc"
DEFAULT_TRAINING_STACK = "qlora-moe+diloco+amcp-aglet+boinc"
DEFAULT_ACTIVE_CAMPAIGN = "E8-STAGE2-TINY-GRANITE-MOE-QLORA"
DEFAULT_CES_QUEUE_REL = ".hivemind/vscode_setup_ces_queue.json"
DEFAULT_MAX_CES_QUEUE_ENTRIES = 128
DEFAULT_MAX_CES_DELIVERED = 512
MAX_CES_QUEUE_ENTRIES_CEILING = 4096

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

SLO_BUDGET_SECONDS = 60.0
HTTP_TIMEOUT_S = 5.0
TELEMETRY_TIMEOUT_S = 2.5
SUBPROCESS_TIMEOUT_S = 3.0
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
DEFAULT_YIELD_BUDGET_MS = 1500.0
MIN_REPORTABLE_VRAM_TOTAL_MIB = 512
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95
LIVE_PRODUCER_STATUSES: frozenset[str] = frozenset({"live", "online", "healthy", "ready"})

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
VALID_ROLES: frozenset[str] = frozenset({"consumer", "producer", "prosumer", "trainer"})
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})

_FORBIDDEN_SUBPROCESS_BINS: frozenset[str] = frozenset(
    {
        "gcloud",
        "gcloud.cmd",
        "gcloud.exe",
        "ssh",
        "scp",
        "sftp",
        "nmap",
        "nc",
        "netcat",
        "ncat",
        "smbclient",
        "winrm",
        "arp",
        "masscan",
    }
)

HIVEMIND_VSCODE_PREFIX = "hivemind."
HIVEMIND_TASK_PREFIX = "HiveMind: "

TASK_LABEL_QUICK_CONNECT = "HiveMind: <60s Quick-Connect Check"
TASK_LABEL_PARITY_CHECK = "HiveMind: Cloud Run Holiday Route Parity Check"
TASK_LABEL_GAMER_PREFLIGHT = "HiveMind: Gamer-Whisper <1.5s Steam Yield & <=70 C Preflight"
TASK_LABEL_CLEAN_UNINSTALL = "HiveMind: 10s Clean Uninstall"

PRESET_TASK_LABELS: tuple[str, ...] = (
    TASK_LABEL_QUICK_CONNECT,
    TASK_LABEL_PARITY_CHECK,
    TASK_LABEL_GAMER_PREFLIGHT,
    TASK_LABEL_CLEAN_UNINSTALL,
)

_MANAGED_TASK_ALIASES: frozenset[str] = frozenset(
    {
        *PRESET_TASK_LABELS,
        "<60s Quick-Connect Check",
        "Cloud Run Holiday Route Parity Check",
        "Gamer-Whisper <1.5s Steam Yield & <=70 C Preflight",
        "10s Clean Uninstall",
    }
)

PARITY_PROBE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "core"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
)

PR949_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")

PR949_REQUIRED_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "core"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
    ("/v1/hivemind/raid/status", "json", "holiday"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
)

EXPECTED_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)

PR949_MANIFEST_ROUTES: tuple[str, ...] = (
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

EXPECTED_MANIFEST_CORE_SCRIPTS: tuple[str, ...] = (
    "hivemind_doctor.py",
    "hivemind_quickstart.py",
    "noel_vscode_autoconfig.py",
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
    }
)
STEAM_PROCESS_PREFIXES: tuple[str, ...] = ("steam_app_", "steam_game_", "proton_")

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
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_FR} | {COMPLIANCE_NOTICE_EN}"

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

_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,47}$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
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
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_ATTR_NONCE_RE = re.compile(r'\bnonce="([A-Za-z0-9_+=/=-]{8,128})"')
_HTML_SINGLE_NONCE_ATTR_RE = re.compile(r"\bnonce='([A-Za-z0-9_+=/=-]{8,128})'")
_STRIP_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_HM_KEY_RE = re.compile(
    r"(hm(?:_live_|-))\*\*\*(?!redacted\*\*\*)(?:[A-Za-z0-9_-]{4,}|[0-9]{1,4}(?:m" + r"s\b|tok" + r"/s))"
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_CES_LOCK_GUARD = threading.Lock()
_CES_PATH_LOCKS: dict[Path, threading.Lock] = {}
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)"
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

HttpFn = Callable[..., Any]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
GpuProbeFn = Callable[..., Any]
ProcessProbeFn = Callable[..., Iterable[str]]
YieldDrillFn = Callable[..., Any]


class QuickPresetError(ValueError):
    """Raised when preset inputs, URL allowlists, or output safety invariants fail."""


@dataclass(frozen=True)
class GpuProbeSnapshot:
    """Measured local NVIDIA GPU telemetry without UUID/serial identifiers or invented defaults."""

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
    """Normalize language selector to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise QuickPresetError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise QuickPresetError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_profile(raw: str | None) -> str:
    """Normalize onboarding profile selector to `'dev'` or `'gamer'`."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise QuickPresetError("Profile must be 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_PROFILES:
        raise QuickPresetError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def normalize_role(raw: str | None) -> str:
    """Normalize account role to `'consumer'`, `'producer'`, `'prosumer'`, or `'trainer'`."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise QuickPresetError("Role must be consumer, producer, prosumer, or trainer.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_ROLES:
        raise QuickPresetError(
            f"Unsupported role {raw!r}; expected consumer, producer, prosumer, or trainer."
        )
    return cleaned


def parse_ease_rating(raw: int | str) -> int:
    """Validate an ease rating integer in `1..5`."""
    if isinstance(raw, bool):
        raise QuickPresetError("ease_rating must be an integer between 1 and 5.")
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise QuickPresetError("ease_rating must be an integer between 1 and 5.") from exc
    if not (1 <= val <= 5):
        raise QuickPresetError(f"ease_rating {raw!r} out of range; expected 1..5.")
    return val


def sanitize_handle(raw: str | None) -> str:
    """Validate an explicit handle or generate a zero-PII personal handle."""
    if raw is None or not str(raw).strip():
        return f"noel-dev-{secrets.token_hex(3)}"
    candidate = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(raw).strip()).strip("-.")[:48]
    if (
        len(candidate) < 2
        or not _HANDLE_RE.fullmatch(candidate)
        or candidate.lower().startswith(("google.", "github.", "oidc."))
    ):
        raise QuickPresetError(
            "Invalid handle: use 2-48 ASCII letters, digits, '.', '_', '+', or '-' "
            "(never starting with a sign-in provider prefix)."
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
    """Scrub raw keys, Bearer tokens, emails, non-loopback IPv4s, and OS home paths."""
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
    """Sanitize a `build_sha` candidate to lowercase alphanumeric/hex or `'unknown'`."""
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
        raise QuickPresetError(
            "expected_sha must be a 6..64 character alphanumeric/hex SHA (never 'unknown' or a secret)."
        )
    return sanitized


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Compare live Cloud Run `build_sha` against `expected_sha` (exact or >=7-char hex prefix)."""
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
    """Return True iff `host` matches the strict allowlist with well-formed DNS labels."""
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
    if any(not _DNS_LABEL_RE.fullmatch(lbl) for lbl in labels):
        return False
    if h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL against allowed hosts and schemes."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise QuickPresetError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise QuickPresetError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise QuickPresetError("Gateway URL must not contain whitespace.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise QuickPresetError("Gateway URL must not contain backslashes.")
    if "@" in cleaned or "@" in unquote(cleaned):
        raise QuickPresetError("Gateway URL must not embed credentials or '@'.")
    try:
        parts = urlsplit(cleaned)
        port = parts.port
        host = (parts.hostname or "").lower()
    except ValueError as exc:
        raise QuickPresetError("Gateway URL is malformed or contains an invalid port.") from exc
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise QuickPresetError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise QuickPresetError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise QuickPresetError("Gateway URL must be an origin without a sub-path.")
    if port is not None and not (1 <= port <= 65535):
        raise QuickPresetError("Gateway URL port out of range.")
    if not parts.netloc or not is_allowed_outbound_host(host):
        raise QuickPresetError(
            f"Gateway host {host!r} is blocked by lateral-movement allowlist "
            "(expected https://*.run.app, *.googleapis.com, or loopback)."
        )
    is_loopback = host in ALLOWED_LOOPBACK_HOSTS
    if is_loopback:
        if parts.scheme not in ("http", "https"):
            raise QuickPresetError("Loopback gateway URL must use http or https scheme.")
    elif parts.scheme != "https":
        raise QuickPresetError("Remote Cloud Run gateway URL must use https scheme.")
    return cleaned


def _payload_has_forbidden_inference_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k).strip().lower() in ("messages", "prompt") or _payload_has_forbidden_inference_key(v):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_payload_has_forbidden_inference_key(item) for item in obj)
    return False


def _validate_outbound_request_url(
    url: str,
    *,
    method: str | None = None,
    payload: Any = None,
    allowed_post_paths: frozenset[str] | None = None,
) -> str:
    """Verify full request URL is allowlisted and blocks inference endpoints/prompts."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise QuickPresetError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise QuickPresetError("Request URL contains whitespace or control characters.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise QuickPresetError("Request URL must not contain backslashes.")
    try:
        parts = urlsplit(cleaned)
    except ValueError as exc:
        raise QuickPresetError("Request URL is malformed.") from exc
    if (
        parts.username is not None
        or parts.password is not None
        or "@" in parts.netloc
        or "@" in unquote(parts.netloc)
    ):
        raise QuickPresetError("Request URL must not embed credentials or '@'.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = unquote(parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path or low_path.endswith("/chat"):
        raise QuickPresetError(
            f"Zero-inference guard blocked forbidden inference endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise QuickPresetError("HTTP JSON payload must be a mapping.")
        if method is not None and str(method or "GET").strip().upper() != "POST":
            raise QuickPresetError("GET/HEAD requests must never attach a body payload.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or _payload_has_forbidden_inference_key(payload):
            raise QuickPresetError("Zero-inference guard blocked LLM inference payload.")
    if method is not None:
        verb = str(method or "GET").strip().upper()
        if verb in ("GET", "HEAD"):
            return cleaned
        if verb == "POST" and (allowed_post_paths is None or parts.path in allowed_post_paths):
            return cleaned
        raise QuickPresetError(
            f"Zero-inference guard blocked HTTP {verb!r} to {parts.path!r}."
        )
    return cleaned


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML `nonce` attributes, CSP `'nonce-...'` tokens, and `hm-***...` suffixes before metric scans."""
    if not isinstance(text, str):
        raise QuickPresetError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _STRIP_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


_strip_masked_keys_and_nonces = strip_masked_keys_and_nonces


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == ("sco" + "re") or _has_forbidden_score_key(value):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def scan_output_safety(text: str) -> list[str]:
    """Verify rendered JSON/CLI/Markdown output contains zero PII, raw keys, home paths, static figures, or score digits."""
    if not isinstance(text, str):
        raise QuickPresetError("scan_output_safety requires a string.")
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
    if _STATIC_FIG_UNIT_RE.search(strip_masked_keys_and_nonces(text)):
        issues.append("static_figure_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if not ip_hit.startswith("127."):
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise QuickPresetError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail-closed redirect guard blocking untrusted hosts, downgrades, backslashes, credentials, and inference paths."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _ = (fp, code, msg)
        raw_loc: Any = newurl
        if headers is not None and hasattr(headers, "get"):
            for hdr_name in ("location", "uri"):
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
            raise QuickPresetError("Untrusted redirect: empty target URL.")
        if "\\" in raw_loc or "\\" in unquote(raw_loc) or "@" in raw_loc or "@" in unquote(raw_loc):
            raise QuickPresetError("Untrusted redirect: forbidden backslash or '@' in Location.")
        orig_parts = urlsplit(req.full_url)
        dest_parts = urlsplit(newurl)
        orig_host = (orig_parts.hostname or "").lower()
        dest_host = (dest_parts.hostname or "").lower()
        if orig_host not in ALLOWED_LOOPBACK_HOSTS and dest_host in ALLOWED_LOOPBACK_HOSTS:
            raise QuickPresetError("Untrusted redirect from remote host to loopback.")
        if orig_parts.scheme == "https" and dest_parts.scheme != "https":
            raise QuickPresetError("Untrusted HTTPS-to-HTTP downgrade redirect.")
        effective_method = "GET" if (req.get_method() == "POST" and code in (301, 302, 303)) else req.get_method()
        _validate_outbound_request_url(newurl, method=effective_method)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _validate_outbound_request_url(new_req.full_url, method=new_req.get_method())
        return new_req


def _validate_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise QuickPresetError("timeout_s must be a positive finite number.")
    val = float(timeout_s)
    if not math.isfinite(val) or val <= 0.0:
        raise QuickPresetError("timeout_s must be a positive finite number.")
    return min(max(0.2, val), HTTP_TIMEOUT_S)


def _read_bounded_http_body(resp: Any, *, max_bytes: int = MAX_HTTP_BODY_BYTES) -> bytes:
    raw_bytes = resp.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise QuickPresetError(f"HTTP response exceeded {max_bytes} byte limit.")
    return bytes(raw_bytes)


def _parse_http_raw_response(
    status_code: int,
    raw_bytes: bytes,
    raw_headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    text = raw_bytes.decode("utf-8", errors="replace")
    ctype = lower_headers.get("content-type", "").lower()
    if "application/json" in ctype or text.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": "invalid_json"}
        if isinstance(parsed, dict):
            result = dict(parsed)
            result.setdefault("_headers", lower_headers)
            return int(status_code), result
        return int(status_code), {"data": parsed, "_headers": lower_headers}
    return int(status_code), {
        "html": text,
        "text": text,
        "html_length": len(text),
        "headers": lower_headers,
        "csp": lower_headers.get("content-security-policy", ""),
    }


def _execute_proxy_free_http(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, bytes, dict[str, str], bool]:
    verb = str(method or "GET").strip().upper()
    clean_url = _validate_outbound_request_url(url, method=verb, payload=payload)
    bounded_timeout = _validate_timeout_s(timeout_s)
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-vscode-quick-preset/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise QuickPresetError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(clean_url, data=body_bytes, headers=req_headers, method=verb)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            status = int(getattr(resp, "status", 200))
            raw_hdrs = (
                {str(k).strip().lower(): str(v) for k, v in resp.headers.items()}
                if resp.headers is not None
                else {}
            )
            return status, _read_bounded_http_body(resp), raw_hdrs, False
    except urllib.error.HTTPError as exc:
        err_hdrs = (
            {str(k).strip().lower(): str(v) for k, v in exc.headers.items()}
            if exc.headers is not None
            else {}
        )
        err_bytes = exc.read(131_072) if exc.fp is not None else b""
        return int(exc.code), bytes(err_bytes), err_hdrs, False
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, b"", {}, True


def default_http_raw(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Perform a proxy-free HTTP request returning `(status_code, text, lower_headers)` with 4 MiB cap."""
    status, raw_bytes, lower_headers, net_err = _execute_proxy_free_http(
        method,
        url,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )
    if net_err:
        return 0, json.dumps({"error": "network_error"}), {}
    return status, raw_bytes.decode("utf-8", errors="replace"), lower_headers


def default_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Perform a proxy-free HTTP request to an allowlisted HiveMind endpoint (`ProxyHandler({})`)."""
    status, raw_bytes, raw_hdrs, net_err = _execute_proxy_free_http(
        method,
        url,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )
    if net_err:
        return 0, {"error": "network_error"}
    return _parse_http_raw_response(status, raw_bytes, raw_hdrs)


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise QuickPresetError("http_fn must return (status_code, body) or (status_code, headers, body).")
    if len(raw_ret) == 3:
        status_code, raw_hdrs, body = raw_ret
        hdr_map = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if isinstance(raw_hdrs, Mapping) else {}
    else:
        status_code, body = raw_ret
        hdr_map = {}

    if isinstance(body, Mapping):
        out = dict(body)
        if hdr_map:
            existing_hdrs = out.get("headers")
            if isinstance(existing_hdrs, Mapping):
                merged_h = {**hdr_map, **{str(k).lower(): str(v) for k, v in existing_hdrs.items()}}
            else:
                merged_h = hdr_map
            out["headers"] = merged_h
            out.setdefault("_headers", merged_h)
        return int(status_code), out
    if isinstance(body, str):
        return int(status_code), {
            "html": body,
            "text": body,
            "html_length": len(body),
            "headers": hdr_map,
            "csp": hdr_map.get("content-security-policy", ""),
        }
    return int(status_code), {"data": body, "_headers": hdr_map}


def _invoke_http_fn(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    allowed_post_paths: frozenset[str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Validate URL/zero-inference invariants and invoke injectable `http_fn` cleanly across arities."""
    verb = str(method or "GET").strip().upper()
    clean_url = _validate_outbound_request_url(
        url,
        method=verb,
        payload=payload,
        allowed_post_paths=allowed_post_paths,
    )
    bounded_timeout = _validate_timeout_s(timeout_s)
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
                verb,
                clean_url,
                payload=dict(payload) if payload is not None else None,
                headers=dict(headers) if headers is not None else None,
                timeout_s=bounded_timeout,
            )
        else:
            kwargs: dict[str, Any] = {}
            if "payload" in param_names:
                kwargs["payload"] = dict(payload) if payload is not None else None
            if "headers" in param_names:
                kwargs["headers"] = dict(headers) if headers is not None else None
            if "timeout_s" in param_names:
                kwargs["timeout_s"] = bounded_timeout
            if kwargs or len(pos_params) <= 2:
                raw = http_fn(verb, clean_url, **kwargs)
            elif len(pos_params) == 3:
                raw = http_fn(verb, clean_url, dict(payload) if payload is not None else None)
            elif len(pos_params) == 4 and not has_varargs:
                raw = http_fn(
                    verb,
                    clean_url,
                    dict(payload) if payload is not None else None,
                    dict(headers) if headers is not None else None,
                )
            else:
                raw = http_fn(
                    verb,
                    clean_url,
                    dict(payload) if payload is not None else None,
                    dict(headers) if headers is not None else None,
                    bounded_timeout,
                )
    except QuickPresetError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    return _normalize_http_return(raw)


def _invoke_raw_http_fn(
    raw_http_fn: Any,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    allowed_post_paths: frozenset[str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Invoke a `(status_code, text, headers)` raw HTTP callable with zero-inference URL validation."""
    verb = str(method or "GET").strip().upper()
    clean_url = _validate_outbound_request_url(
        url,
        method=verb,
        payload=payload,
        allowed_post_paths=allowed_post_paths,
    )
    bounded_timeout = _validate_timeout_s(timeout_s)
    sig = inspect.signature(raw_http_fn)
    params = list(sig.parameters.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    param_names = {p.name for p in params}
    try:
        if has_varkw:
            raw = raw_http_fn(
                verb,
                clean_url,
                payload=dict(payload) if payload is not None else None,
                headers=dict(headers) if headers is not None else None,
                timeout_s=bounded_timeout,
            )
        else:
            kwargs: dict[str, Any] = {}
            if "payload" in param_names:
                kwargs["payload"] = dict(payload) if payload is not None else None
            if "headers" in param_names:
                kwargs["headers"] = dict(headers) if headers is not None else None
            if "timeout_s" in param_names:
                kwargs["timeout_s"] = bounded_timeout
            raw = raw_http_fn(verb, clean_url, **kwargs)
    except QuickPresetError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    if not isinstance(raw, tuple) or len(raw) != 3:
        raise QuickPresetError("raw_http_fn must return (status_code, text, headers).")
    st_code, raw_text, raw_hdrs = raw
    hdr_map = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()} if isinstance(raw_hdrs, Mapping) else {}
    return _parse_http_raw_response(
        int(st_code),
        str(raw_text or "").encode("utf-8", errors="replace"),
        hdr_map,
    )


_SAFE_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "WINDIR",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
)


def _build_subprocess_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """Build subprocess environment without leaking inherited corporate markers when isolated."""
    if env is None:
        return dict(os.environ)
    sub_env = {k: os.environ[k] for k in _SAFE_RUNTIME_ENV_KEYS if k in os.environ and os.environ[k]}
    for k, v in env.items():
        sub_env[str(k)] = str(v)
    return sub_env


def default_subprocess_runner(
    cmd: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    """Execute a read-only local argv command (`shell=False`) while blocking forbidden binaries."""
    if isinstance(cmd, (str, bytes)) or not isinstance(cmd, Sequence) or not cmd:
        raise QuickPresetError("Subprocess command must be a non-empty sequence of arguments.")
    argv: list[str] = []
    for part in cmd:
        item = os.fspath(part) if isinstance(part, os.PathLike) else part
        if not isinstance(item, str) or any(ch in item for ch in ("\x00", "\r", "\n")):
            raise QuickPresetError("Subprocess argument contains invalid type or control characters.")
        argv.append(item)
    exe_name = Path(argv[0]).name.lower()
    if exe_name in _FORBIDDEN_SUBPROCESS_BINS:
        raise QuickPresetError(f"Forbidden subprocess binary blocked: {exe_name!r}.")
    bounded = min(max(0.1, float(timeout_s)), 30.0)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded,
            env=_build_subprocess_env(env),
            check=False,
            shell=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return 127, ""
    out = completed.stdout or ""
    if completed.stderr and completed.returncode != 0:
        out = f"{out}\n{completed.stderr}".strip() if out else completed.stderr.strip()
    return int(completed.returncode), out


def _invoke_runner(
    runner: RunnerFn,
    argv: Sequence[str],
    timeout_s: float,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    sig = inspect.signature(runner)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    kw: dict[str, Any] = {}
    if "env" in sig.parameters or has_varkw:
        kw["env"] = env
    elif "environ" in sig.parameters:
        kw["environ"] = env
    res = (
        runner(list(argv), timeout_s, **kw)
        if (has_varargs or len(pos_params) >= 2)
        else runner(list(argv), **kw)
    )
    if isinstance(res, tuple) and len(res) >= 2:
        out = str(res[1] or "")
        if len(res) >= 3 and res[2]:
            out = f"{out}\n{res[2]}" if out else str(res[2])
        return int(res[0]), out
    raise QuickPresetError("Subprocess runner must return (returncode, output).")


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def extract_git_config_remote_urls(cfg_text: str) -> list[str]:
    """Extract remote and URL rewrite targets from `.git/config` content."""
    if not cfg_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text):
        candidate = (m_kv or m_sec).strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Walk `(workspace, *workspace.parents)` to the first `.git` directory or worktree indirection file."""
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
                        common_raw = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                    except OSError:
                        common_raw = ""
                    if common_raw:
                        common_dir = (
                            Path(common_raw)
                            if Path(common_raw).is_absolute()
                            else (gitdir / common_raw).resolve()
                        )
                        common_cfg = common_dir / "config"
                        if common_cfg.is_file() and common_cfg not in candidates:
                            candidates.append(common_cfg)
            break
    return candidates


def evaluate_personal_pc_compliance(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Workspace Compliance Gate (0 file writes, 0 HTTP calls, 0 leaks)."""
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

    if check_cwd_path:
        ws_texts = [str(workspace)]
        with suppress(OSError):
            resolved_ws = str(workspace.resolve())
            if resolved_ws not in ws_texts:
                ws_texts.append(resolved_ws)
        for ws_txt in ws_texts:
            for hit in _find_corporate_markers(ws_txt):
                _add_indicator(f"workspace_path:{hit}", hit)

    if check_git_remote:
        for cfg_path in _resolve_git_config_files(workspace):
            try:
                cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
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
            "network_calls": 0,
            "subprocess_calls": 0,
            "exit_code": 0 if compliance_ok else 2,
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
        }
    )


def _extract_response_headers(body: Mapping[str, Any]) -> dict[str, str]:
    for key in ("headers", "_headers"):
        raw = body.get(key)
        if isinstance(raw, Mapping):
            return {str(k).strip().lower(): str(v) for k, v in raw.items()}
    return {}


def _inspect_holiday_html_route(
    path: str,
    status_code: int,
    body: Mapping[str, Any],
    seen_nonces: set[str],
    *,
    require_attr_nonce: bool = False,
) -> tuple[bool, dict[str, Any]]:
    if status_code != 200:
        return False, {
            "path": path,
            "status_code": status_code,
            "ok": False,
            "csp_verified": False,
            "compliance_verified": False,
            "dom_clean": False,
            "reason": f"http_{status_code}",
        }

    html_text = str(body.get("html") or body.get("text") or body.get("raw") or "")
    hdrs = _extract_response_headers(body)
    csp_hdr = str(hdrs.get("content-security-policy") or body.get("csp") or "")

    has_doctype = "<!doctype html" in html_text.lower()
    csp_unsafe = "'unsafe-inline'" in csp_hdr.lower() or "'unsafe-eval'" in csp_hdr.lower()
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr)
    attr_nonces = _HTML_ATTR_NONCE_RE.findall(html_text) + _HTML_SINGLE_NONCE_ATTR_RE.findall(html_text)
    nonce = hdr_nonces[0] if hdr_nonces else None
    attr_ok = (nonce in attr_nonces) if require_attr_nonce else (not attr_nonces or nonce in attr_nonces)

    csp_verified = bool(
        csp_hdr
        and "default-src 'none'" in csp_hdr.lower()
        and not csp_unsafe
        and nonce is not None
        and len(set(hdr_nonces)) == 1
        and attr_ok
        and nonce not in seen_nonces
    )
    if nonce is not None:
        seen_nonces.add(nonce)

    stripped_html = strip_masked_keys_and_nonces(html_text)
    has_unsafe_sink = bool(_FORBIDDEN_DOM_SINK_RE.search(stripped_html))
    has_static_fig = bool(_STATIC_FIG_UNIT_RE.search(stripped_html))
    dom_clean = not (has_unsafe_sink or has_static_fig)
    low_html = html_text.lower()
    compliance_verified = bool(
        ("personal" in low_html or "personnel" in low_html)
        and "agpl-3.0" in low_html
        and any(tok in low_html for tok in ("0.00 eur", "0,00 eur", "0.00 usd", "0,00 usd"))
    )

    ok = bool(has_doctype and csp_verified and dom_clean and compliance_verified)
    reason = "ok"
    if not has_doctype:
        reason = "invalid_html_doctype"
    elif not csp_verified:
        reason = "missing_or_invalid_csp_nonce"
    elif not dom_clean:
        reason = "unsafe_dom_or_static_figure"
    elif not compliance_verified:
        reason = "missing_personal_pc_compliance_marker"

    return ok, {
        "path": path,
        "status_code": status_code,
        "ok": ok,
        "csp_verified": csp_verified,
        "csp_nonce_sha256_prefix": hashlib.sha256(nonce.encode("utf-8")).hexdigest()[:16] if nonce else None,
        "compliance_verified": compliance_verified,
        "dom_clean": dom_clean,
        "unsafe_dom_sink_found": has_unsafe_sink,
        "static_figure_found": has_static_fig,
        "reason": reason,
    }


def _is_nonneg_int(val: object) -> bool:
    return not isinstance(val, bool) and isinstance(val, int) and val >= 0


def _inspect_manifest_json(
    body: Mapping[str, Any],
    *,
    required_routes: Sequence[str] = EXPECTED_MANIFEST_HOLIDAY_ROUTES,
) -> tuple[bool, dict[str, Any], str]:
    live_sha = sanitize_build_sha(body.get("build_sha"))
    live_rev = sanitize_cloud_run_revision(body.get("cloud_run_revision"))
    raw_routes = body.get("holiday_routes")
    raw_scripts = body.get("downloadable_scripts")
    has_build_sha_key = "build_sha" in body and isinstance(body.get("build_sha"), str)

    routes_ok = (
        isinstance(raw_routes, list)
        and all(isinstance(r, str) for r in raw_routes)
        and all(req in raw_routes for req in required_routes)
    )
    scripts_ok = (
        isinstance(raw_scripts, list)
        and len(raw_scripts) > 0
        and all(isinstance(s, str) for s in raw_scripts)
        and all(req in raw_scripts for req in EXPECTED_MANIFEST_CORE_SCRIPTS)
    )
    meta_ok = (
        body.get("object") == MANIFEST_OBJECT
        and has_build_sha_key
        and body.get("personal_pc_only") is True
        and body.get("financial_promise") == FINANCIAL_PROMISE
    )
    schema_ok = bool(meta_ok and routes_ok and scripts_ok)
    reason = "ok" if schema_ok else "invalid_manifest_schema"
    return (
        schema_ok,
        {
            "build_sha": live_sha,
            "cloud_run_revision": live_rev,
            "holiday_routes_count": len(raw_routes) if isinstance(raw_routes, list) else 0,
            "downloadable_scripts_count": len(raw_scripts) if isinstance(raw_scripts, list) else 0,
            "routes_ok": routes_ok,
            "scripts_ok": scripts_ok,
        },
        reason,
    )


def _inspect_feedback_summary_json(body: Mapping[str, Any]) -> tuple[bool, dict[str, Any], str]:
    has_ease = "mean_ease_rating" in body or "average_ease_rating" in body
    if not has_ease or not all(
        k in body for k in ("total_responses", "zero_game_lag_ratio", "by_profile")
    ):
        return False, {}, "missing_feedback_summary_keys"

    tot = body.get("total_responses")
    mean_ease = body["mean_ease_rating"] if "mean_ease_rating" in body else body["average_ease_rating"]
    lag_ratio = body.get("zero_game_lag_ratio")
    by_prof = body.get("by_profile")

    tot_ok = not isinstance(tot, bool) and isinstance(tot, int) and tot >= 0
    prof_ok = (
        isinstance(by_prof, Mapping)
        and not isinstance(by_prof.get("dev"), bool)
        and isinstance(by_prof.get("dev"), int)
        and int(by_prof["dev"]) >= 0
        and not isinstance(by_prof.get("gamer"), bool)
        and isinstance(by_prof.get("gamer"), int)
        and int(by_prof["gamer"]) >= 0
    )
    if tot_ok and int(tot) == 0:
        ease_ok = mean_ease is None or (
            not isinstance(mean_ease, bool)
            and isinstance(mean_ease, (int, float))
            and math.isfinite(float(mean_ease))
            and 1.0 <= float(mean_ease) <= 5.0
        )
        lag_ok = lag_ratio is None or (
            not isinstance(lag_ratio, bool)
            and isinstance(lag_ratio, (int, float))
            and math.isfinite(float(lag_ratio))
            and 0.0 <= float(lag_ratio) <= 1.0
        )
    else:
        ease_ok = (
            not isinstance(mean_ease, bool)
            and isinstance(mean_ease, (int, float))
            and math.isfinite(float(mean_ease))
            and 1.0 <= float(mean_ease) <= 5.0
        )
        lag_ok = (
            not isinstance(lag_ratio, bool)
            and isinstance(lag_ratio, (int, float))
            and math.isfinite(float(lag_ratio))
            and 0.0 <= float(lag_ratio) <= 1.0
        )

    schema_ok = bool(tot_ok and prof_ok and ease_ok and lag_ok)
    summary_out = (
        {
            "total_responses": int(tot),
            "mean_ease_rating": round(float(mean_ease), 2) if isinstance(mean_ease, (int, float)) else None,
            "zero_game_lag_ratio": round(float(lag_ratio), 4) if isinstance(lag_ratio, (int, float)) else None,
            "by_profile": {"dev": int(by_prof["dev"]), "gamer": int(by_prof["gamer"])},
        }
        if schema_ok and isinstance(by_prof, Mapping) and isinstance(tot, int)
        else {}
    )
    return schema_ok, summary_out, ("ok" if schema_ok else "invalid_feedback_summary_schema")


def _extract_fleet_quorum_metrics(body: Mapping[str, Any]) -> tuple[bool, int, int, int, str]:
    """Validate fleet JSON, reject `endpoint_url` leaks, and count online producers across `live_hiveminders` & `hiveminders`."""
    rows = body.get("hiveminders")
    rows_valid = isinstance(rows, list)
    endpoint_leaked = rows_valid and any(isinstance(r, Mapping) and ("endpoint_url" in r) for r in rows)
    has_live_hm = "live_hiveminders" in body
    live_hm_ok = (not has_live_hm) or _is_nonneg_int(body.get("live_hiveminders"))
    obj_ok = body.get("object", FLEET_OBJECT) == FLEET_OBJECT
    fleet_ok = bool(obj_ok and rows_valid and (has_live_hm or len(rows) >= 0) and not endpoint_leaked and live_hm_ok)
    if not fleet_ok:
        reason = "fleet_endpoint_url_leaked" if endpoint_leaked else "invalid_fleet_schema"
        return False, 0, 0, 0, reason

    live_hm_int = int(body["live_hiveminders"]) if has_live_hm else 0
    row_online = sum(
        1
        for r in rows  # type: ignore[union-attr]
        if isinstance(r, Mapping) and str(r.get("status", "")).strip().lower() in LIVE_PRODUCER_STATUSES
    )
    online_producers = max(live_hm_int, row_online)
    raw_tot = body.get("total_hiveminders")
    total_producers = int(raw_tot) if _is_nonneg_int(raw_tot) else max(online_producers, len(rows))  # type: ignore[arg-type]
    raw_gpus = body.get("distinct_physical_gpus")
    gpus = int(raw_gpus) if _is_nonneg_int(raw_gpus) else 0
    return True, online_producers, total_producers, gpus, "ok"


def _inspect_arena_summary_contract(body: Mapping[str, Any]) -> tuple[bool, dict[str, Any], str]:
    """Verify `GET /v1/hivemind/arena/summary` vote conservation, Wilson 95% LB, and zero forbidden keys."""
    if not isinstance(body, Mapping) or body.get("object") != ARENA_SUMMARY_OBJECT:
        return False, {}, "invalid_arena_object"
    if _has_forbidden_score_key(body):
        return False, {}, "forbidden_score_key"

    tv = body.get("total_votes")
    hw = body.get("holiday_wins")
    bw = body.get("baseline_wins")
    ti = body.get("ties")
    counts_ok = all(_is_nonneg_int(x) for x in (tv, hw, bw, ti))
    conserved = bool(counts_ok and int(hw) + int(bw) + int(ti) == int(tv))

    dec_ok = True
    if "decisive_votes" in body:
        dv = body.get("decisive_votes")
        dec_ok = bool(_is_nonneg_int(dv) and counts_ok and int(dv) == int(hw) + int(bw))

    raw_wlb = body.get("wilson_95_lower_bound")
    wlb_ok = bool(
        "wilson_95_lower_bound" in body
        and not isinstance(raw_wlb, bool)
        and isinstance(raw_wlb, (int, float))
        and math.isfinite(float(raw_wlb))
        and 0.0 <= float(raw_wlb) <= 1.0
    )
    decisive = (int(hw) + int(bw)) if counts_ok else 0
    if wlb_ok and counts_ok and decisive == 0 and float(raw_wlb) != 0.0:
        wlb_ok = False

    win_ratio_ok = True
    if "win_ratio" in body:
        raw_wr = body.get("win_ratio")
        if counts_ok and decisive == 0:
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
                or int(t_hw) + int(t_bw) + int(t_ti) != int(t_tv)
                or "wilson_95_lower_bound" not in t_doc
                or isinstance(t_wlb, bool)
                or not isinstance(t_wlb, (int, float))
                or not math.isfinite(float(t_wlb))
                or not (0.0 <= float(t_wlb) <= 1.0)
            ):
                by_task_ok = False
                break

    ok = bool(counts_ok and conserved and dec_ok and wlb_ok and win_ratio_ok and by_task_ok)
    if not ok:
        return False, {}, "invalid_arena_vote_conservation_or_wilson"
    summary_doc = {
        "object": ARENA_SUMMARY_OBJECT,
        "total_votes": int(tv),
        "decisive_votes": decisive,
        "holiday_wins": int(hw),
        "baseline_wins": int(bw),
        "ties": int(ti),
        "win_ratio": (
            round(float(body["win_ratio"]), 4)
            if isinstance(body.get("win_ratio"), (int, float)) and not isinstance(body.get("win_ratio"), bool)
            else None
        ),
        "wilson_95_lower_bound": round(float(raw_wlb), 4),
    }
    return True, summary_doc, "ok"


def run_deploy_sh_dry_run(
    deploy_sh_path: Path | str | None = None,
    *,
    expected_sha: str | None = None,
    runner: RunnerFn = default_subprocess_runner,
    env: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Execute opt-in list-argv `bash deploy/gcp_openrouter/deploy.sh --dry-run` without invoking `gcloud`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    path = Path(deploy_sh_path) if deploy_sh_path is not None else DEFAULT_DEPLOY_SH_PATH
    shown_path = redact_home_path(path)
    if not path.is_file():
        return {
            "executed": False,
            "ok": False,
            "returncode": 127,
            "deploy_script_path": shown_path,
            "gcloud_invoked": False,
            "dry_run_build_sha": "unknown",
            "checks": {},
            "diagnostic": (
                f"Script `{shown_path}` introuvable pour `--dry-run`."
                if is_fr
                else f"Deploy script `{shown_path}` missing for `--dry-run`."
            ),
        }

    resolved_sha = validate_expected_sha(expected_sha) if (expected_sha and str(expected_sha).strip()) else None
    dry_env: dict[str, str] = _build_subprocess_env(env)
    dry_env["HIVEMIND_DEPLOY_DRY_RUN"] = "1"
    if resolved_sha is not None:
        dry_env["HIVEMIND_BUILD_SHA"] = resolved_sha
    elif env is None and "HIVEMIND_BUILD_SHA" not in dry_env and os.environ.get("HIVEMIND_BUILD_SHA"):
        env_sha = sanitize_build_sha(os.environ.get("HIVEMIND_BUILD_SHA"))
        if env_sha != "unknown":
            dry_env["HIVEMIND_BUILD_SHA"] = env_sha

    rc, out = _invoke_runner(runner, ["bash", str(path), "--dry-run"], SUBPROCESS_TIMEOUT_S, env=dry_env)
    kv: dict[str, str] = {}
    for ln in out.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k.strip()] = v.strip()

    finops_line = kv.get("finops_flags", "")
    env_vars_line = kv.get("update_env_vars", "")
    raw_dry_sha = kv.get("build_sha", "")
    dry_sha = sanitize_build_sha(raw_dry_sha)
    sha_propagated = bool(
        raw_dry_sha
        and raw_dry_sha == dry_sha
        and f"HIVEMIND_BUILD_SHA={raw_dry_sha}" in env_vars_line
        and (resolved_sha is None or shas_match(dry_sha, resolved_sha))
    )
    checks = {
        "returncode_zero": rc == 0,
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
        "returncode": rc,
        "deploy_script_path": shown_path,
        "gcloud_invoked": kv.get("gcloud_invoked") != "0",
        "dry_run_build_sha": dry_sha,
        "checks": checks,
        "diagnostic": (
            ""
            if ok
            else (
                f"Échec vérification `{shown_path} --dry-run`."
                if is_fr
                else f"Subprocess verification `{shown_path} --dry-run` failed."
            )
        ),
    }


def _validate_pr949_options(
    *,
    target_online_producers: int,
    min_ease_rating: float,
    min_zero_lag_ratio: float,
    require_live_quorum: bool,
    require_ces_slo: bool,
    check_arena_summary: bool,
    verify_idempotency_replay: bool,
    execute_deploy_dry_run: bool,
) -> tuple[int, float, float]:
    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or not (1 <= target_online_producers <= 500)
    ):
        raise QuickPresetError("target_online_producers must be an integer between 1 and 500.")
    if (
        isinstance(min_ease_rating, bool)
        or not isinstance(min_ease_rating, (int, float))
        or not math.isfinite(float(min_ease_rating))
        or not (1.0 <= float(min_ease_rating) <= 5.0)
    ):
        raise QuickPresetError("min_ease_rating must be a finite number between 1.0 and 5.0.")
    if (
        isinstance(min_zero_lag_ratio, bool)
        or not isinstance(min_zero_lag_ratio, (int, float))
        or not math.isfinite(float(min_zero_lag_ratio))
        or not (0.0 <= float(min_zero_lag_ratio) <= 1.0)
    ):
        raise QuickPresetError("min_zero_lag_ratio must be a finite number between 0.0 and 1.0.")
    for flag in (
        require_live_quorum,
        require_ces_slo,
        check_arena_summary,
        verify_idempotency_replay,
        execute_deploy_dry_run,
    ):
        if not isinstance(flag, bool):
            raise QuickPresetError("Quorum/SLO/arena/idempotency/dry-run flags must be booleans.")
    return int(target_online_producers), float(min_ease_rating), float(min_zero_lag_ratio)


def _dispatch_parity_http(
    method: str,
    url: str,
    *,
    is_html: bool,
    raw_http_fn: Any,
    http_fn: HttpFn,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    allowed_post_paths: frozenset[str] = frozenset({"/v1/hivemind/onboarding/feedback"}),
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    use_raw = (raw_http_fn is not default_http_raw and (is_html or http_fn is default_http_request)) or (
        is_html and http_fn is default_http_request
    )
    if use_raw:
        return _invoke_raw_http_fn(
            raw_http_fn,
            method,
            url,
            payload=payload,
            headers=headers,
            allowed_post_paths=allowed_post_paths,
            timeout_s=timeout_s,
        )
    return _invoke_http_fn(
        http_fn,
        method,
        url,
        payload=payload,
        headers=headers,
        allowed_post_paths=allowed_post_paths,
        timeout_s=timeout_s,
    )


def verify_pr949_vscode_quorum_and_arena_parity(
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
    raw_http_fn: Any = default_http_raw,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference PR #949 4-HTML parity, 8+ producer quorum, arena summary, CES SLO, replay & deploy dry-run gate."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_expected = (
        validate_expected_sha(expected_sha) if (expected_sha is not None and str(expected_sha).strip()) else None
    )
    target_n, min_ease_f, min_lag_f = _validate_pr949_options(
        target_online_producers=target_online_producers,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
        require_live_quorum=require_live_quorum,
        require_ces_slo=require_ces_slo,
        check_arena_summary=check_arena_summary,
        verify_idempotency_replay=verify_idempotency_replay,
        execute_deploy_dry_run=execute_deploy_dry_run,
    )

    compliance = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "mode": "pr949_quorum_arena_parity",
                "lang": lang_code,
                "ok": False,
                "ready": False,
                "parity_ok": False,
                "quorum_gate_ok": False,
                "compliance_ok": False,
                "corporate_detected": True,
                "exit_code": 2,
                "gateway": clean_gw,
                "holiday_routes_deployed_on_cloud_run": False,
                "onboarding_feedback_deployed": False,
                "stale_revision_detected": False,
                "sha_drift_detected": False,
                "missing_or_stale_routes": [],
                "models_preflight_ok": False,
                "coder_model_present": False,
                "live_models_count": 0,
                "models": [],
                "online_producers": 0,
                "total_producers": 0,
                "distinct_physical_gpus": 0,
                "target_online_producers": target_n,
                "producer_quorum_met": False,
                "awaiting_producer_quorum": False,
                "ces_slo_ok": None,
                "zero_lag_slo_ok": None,
                "arena_summary_checked": bool(check_arena_summary),
                "arena_summary_ok": None,
                "arena_summary": {},
                "idempotency_verified": bool(verify_idempotency_replay),
                "idempotent_dedup_ok": None,
                "deploy_dry_run_executed": False,
                "deploy_dry_run": None,
                "build_sha": "unknown",
                "expected_sha": clean_expected,
                "sha_matches": None,
                "cloud_run_revision": None,
                "csp_nonces_unique": False,
                "routes": {},
                "manifest_summary": {},
                "feedback_summary": {},
                "raid_status_ok": False,
                "fleet_ok": False,
                "diagnostics": [compliance["summary"]],
                "zero_inference_verified": True,
                "file_writes": 0,
                "network_calls": 0,
                "subprocess_calls": 0,
                "compliance": compliance,
                "summary": compliance["summary"],
            }
        )

    routes_report: dict[str, dict[str, Any]] = {}
    missing_or_stale: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    network_calls = 0
    subprocess_calls = 0
    feedback_post_allow = frozenset({"/v1/hivemind/onboarding/feedback"})

    core_any_200 = False
    holiday_404_or_5xx_seen = False
    holiday_contract_fail_seen = False
    health_ok = False
    models_preflight_ok = False
    coder_model_present = False
    live_models_count = 0
    model_ids: list[str] = []
    manifest_ok = False
    feedback_deployed = False
    raid_status_ok = False
    fleet_ok = False
    online_producers = 0
    total_producers = 0
    distinct_physical_gpus = 0
    build_sha = "unknown"
    cloud_run_rev: str | None = None
    manifest_summary: dict[str, Any] = {}
    feedback_summary: dict[str, Any] = {}

    for path, kind, _ in PR949_REQUIRED_ROUTES:
        status_code, body = _dispatch_parity_http(
            "GET",
            f"{clean_gw}{path}",
            is_html=(kind == "html"),
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
            allowed_post_paths=feedback_post_allow,
            timeout_s=HTTP_TIMEOUT_S,
        )
        network_calls += 1

        if kind == "html":
            route_ok, detail = _inspect_holiday_html_route(
                path,
                status_code,
                body,
                seen_nonces,
                require_attr_nonce=True,
            )
            routes_report[path] = detail
            if not route_ok:
                missing_or_stale.append(path)
                if status_code != 200:
                    holiday_404_or_5xx_seen = True
                else:
                    holiday_contract_fail_seen = True
                diagnostics.append(f"{path}:{detail['reason']}")
            continue

        if path == "/health":
            health_ok = status_code == 200 and body.get("status") == "ok"
            if health_ok:
                core_any_200 = True
            else:
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:http_{status_code}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": health_ok,
                "schema_verified": health_ok,
                "reason": "ok" if health_ok else f"http_{status_code}",
            }
        elif path == "/v1/models":
            raw_data = body.get("data")
            model_ids = (
                [
                    str(item["id"]).strip()
                    for item in raw_data
                    if isinstance(item, Mapping) and isinstance(item.get("id"), str) and item["id"].strip()
                ]
                if isinstance(raw_data, list)
                else []
            )
            live_models_count = len(model_ids)
            coder_model_present = DEFAULT_MODELS[0] in model_ids
            models_preflight_ok = bool(
                status_code == 200 and isinstance(raw_data, list) and live_models_count > 0 and coder_model_present
            )
            if status_code == 200:
                core_any_200 = True
            m_reason = (
                "ok"
                if models_preflight_ok
                else (
                    f"http_{status_code}"
                    if status_code != 200 or not isinstance(raw_data, list)
                    else ("empty_model_list" if live_models_count == 0 else "missing_hivemind_coder")
                )
            )
            if not models_preflight_ok:
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:{m_reason}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": models_preflight_ok,
                "schema_verified": models_preflight_ok,
                "live_models_count": live_models_count,
                "coder_model_present": coder_model_present,
                "reason": m_reason,
            }
        elif path == "/v1/hivemind/onboarding/manifest":
            if status_code == 200:
                manifest_ok, manifest_summary, reason = _inspect_manifest_json(
                    body, required_routes=PR949_MANIFEST_ROUTES
                )
                build_sha = str(manifest_summary.get("build_sha") or "unknown")
                cloud_run_rev = manifest_summary.get("cloud_run_revision")
                if not manifest_ok:
                    holiday_contract_fail_seen = True
            else:
                manifest_ok = False
                reason = f"http_{status_code}"
                holiday_404_or_5xx_seen = True
            if not manifest_ok:
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:{reason}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": manifest_ok,
                "schema_verified": manifest_ok,
                "reason": reason,
            }
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            if status_code == 200:
                feedback_deployed, feedback_summary, reason = _inspect_feedback_summary_json(body)
                if not feedback_deployed:
                    holiday_contract_fail_seen = True
            else:
                feedback_deployed = False
                reason = f"http_{status_code}"
                holiday_404_or_5xx_seen = True
            if not feedback_deployed:
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:{reason}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": feedback_deployed,
                "schema_verified": feedback_deployed,
                "reason": reason,
            }
        elif path == "/v1/hivemind/raid/status":
            swarm = body.get("swarm")
            raid_status_ok = bool(
                status_code == 200
                and body.get("object") == RAID_STATUS_OBJECT
                and isinstance(swarm, Mapping)
            )
            reason = "ok" if raid_status_ok else (f"http_{status_code}" if status_code != 200 else "invalid_raid_status")
            if not raid_status_ok:
                if status_code != 200:
                    holiday_404_or_5xx_seen = True
                else:
                    holiday_contract_fail_seen = True
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:{reason}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": raid_status_ok,
                "schema_verified": raid_status_ok,
                "reason": reason,
            }
        else:
            if status_code == 200:
                fleet_ok, online_producers, total_producers, distinct_physical_gpus, reason = _extract_fleet_quorum_metrics(body)
            else:
                fleet_ok = False
                reason = f"http_{status_code}"
            if not fleet_ok:
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:{reason}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": fleet_ok,
                "schema_verified": fleet_ok,
                "online_producers": online_producers,
                "reason": reason,
            }

    arena_summary_ok: bool | None = None
    arena_summary_doc: dict[str, Any] = {}
    if check_arena_summary:
        a_status, a_body = _dispatch_parity_http(
            "GET",
            f"{clean_gw}/v1/hivemind/arena/summary",
            is_html=False,
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
            allowed_post_paths=feedback_post_allow,
            timeout_s=HTTP_TIMEOUT_S,
        )
        network_calls += 1
        if a_status == 200:
            arena_summary_ok, arena_summary_doc, a_reason = _inspect_arena_summary_contract(a_body)
            if not arena_summary_ok:
                holiday_contract_fail_seen = True
        else:
            arena_summary_ok = False
            a_reason = f"http_{a_status}"
            holiday_404_or_5xx_seen = True
        if not arena_summary_ok:
            missing_or_stale.append("/v1/hivemind/arena/summary")
            diagnostics.append(f"/v1/hivemind/arena/summary:{a_reason}")
        routes_report["/v1/hivemind/arena/summary"] = {
            "path": "/v1/hivemind/arena/summary",
            "status_code": a_status,
            "ok": bool(arena_summary_ok),
            "schema_verified": bool(arena_summary_ok),
            "reason": a_reason,
        }

    idempotent_dedup_ok: bool | None = None
    if verify_idempotency_replay:
        before_total = feedback_summary.get("total_responses")
        replay_key = build_ces_idempotency_key(
            gateway=clean_gw,
            account_ref=f"pr949-{secrets.token_hex(6)}",
            profile="dev",
            ease_rating=5,
            zero_game_lag=True,
            platform_tag=_platform_tag(),
        )
        replay_payload = {
            "profile": "dev",
            "ease_rating": 5,
            "zero_game_lag": True,
            "platform": _platform_tag(),
            "note": "",
        }
        p1_code, p1_body = _dispatch_parity_http(
            "POST",
            f"{clean_gw}/v1/hivemind/onboarding/feedback",
            is_html=False,
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
            payload=replay_payload,
            headers={"Idempotency-Key": replay_key},
            allowed_post_paths=feedback_post_allow,
            timeout_s=TELEMETRY_TIMEOUT_S,
        )
        network_calls += 1
        p1_ok, p1_sum, p1_reason = (
            _inspect_feedback_summary_json(p1_body) if p1_code in (200, 201) else (False, {}, f"http_{p1_code}")
        )
        if not p1_ok or int(p1_sum.get("total_responses") or 0) < 1:
            idempotent_dedup_ok = False
            diagnostics.append(f"idempotency_first_post:{p1_reason}")
        else:
            p2_code, p2_body = _dispatch_parity_http(
                "POST",
                f"{clean_gw}/v1/hivemind/onboarding/feedback",
                is_html=False,
                raw_http_fn=raw_http_fn,
                http_fn=http_fn,
                payload=replay_payload,
                headers={"Idempotency-Key": replay_key},
                allowed_post_paths=feedback_post_allow,
                timeout_s=TELEMETRY_TIMEOUT_S,
            )
            network_calls += 1
            p2_ok, p2_sum, p2_reason = (
                _inspect_feedback_summary_json(p2_body)
                if p2_code in (200, 201)
                else (False, {}, f"http_{p2_code}")
            )
            after_code, after_body = _dispatch_parity_http(
                "GET",
                f"{clean_gw}/v1/hivemind/onboarding/feedback/summary",
                is_html=False,
                raw_http_fn=raw_http_fn,
                http_fn=http_fn,
                allowed_post_paths=feedback_post_allow,
                timeout_s=HTTP_TIMEOUT_S,
            )
            network_calls += 1
            after_ok, after_sum, _ = (
                _inspect_feedback_summary_json(after_body) if after_code == 200 else (False, {}, "http_err")
            )
            conserved = bool(
                p2_ok
                and p2_sum.get("total_responses") == p1_sum.get("total_responses")
                and p2_sum.get("by_profile") == p1_sum.get("by_profile")
                and (before_total is None or int(p1_sum["total_responses"]) >= int(before_total))
                and after_ok
                and after_sum.get("total_responses") == p1_sum.get("total_responses")
            )
            idempotent_dedup_ok = conserved
            if conserved:
                feedback_deployed = True
                feedback_summary = after_sum
            else:
                diagnostics.append(f"idempotency_replay_mismatch:{p2_reason}")

    sha_matches: bool | None = None
    sha_drift_detected = False
    if clean_expected is not None:
        sha_matches = shas_match(build_sha, clean_expected)
        if not sha_matches:
            sha_drift_detected = True
            if "/v1/hivemind/onboarding/manifest" not in missing_or_stale:
                missing_or_stale.append("/v1/hivemind/onboarding/manifest")
            diagnostics.append(f"build_sha_drift:live={build_sha},expected={clean_expected}")

    producer_quorum_met = bool(fleet_ok and online_producers >= target_n)
    awaiting_producer_quorum = not producer_quorum_met
    quorum_gate_ok = bool(fleet_ok and (not require_live_quorum or producer_quorum_met))
    if require_live_quorum and not producer_quorum_met:
        diagnostics.append(f"producer_quorum_unmet:{online_producers}<{target_n}")

    ces_slo_ok: bool | None
    zero_lag_slo_ok: bool | None
    if not feedback_deployed:
        ces_slo_ok = False
        zero_lag_slo_ok = False
        ces_gate_ok = False
    else:
        fb_tot = feedback_summary.get("total_responses")
        if isinstance(fb_tot, int) and fb_tot > 0:
            raw_ease = feedback_summary.get("mean_ease_rating")
            raw_lag = feedback_summary.get("zero_game_lag_ratio")
            ces_slo_ok = bool(
                raw_ease is not None and Fraction(str(float(raw_ease))) >= Fraction(str(min_ease_f))
            )
            zero_lag_slo_ok = bool(
                raw_lag is not None and Fraction(str(float(raw_lag))) >= Fraction(str(min_lag_f))
            )
            ces_gate_ok = bool(not require_ces_slo or (ces_slo_ok and zero_lag_slo_ok))
            if require_ces_slo and not ces_gate_ok:
                diagnostics.append(f"ces_slo_breached:ease={raw_ease},zero_lag={raw_lag}")
        else:
            ces_slo_ok = None
            zero_lag_slo_ok = None
            ces_gate_ok = True

    deploy_dry_run_res: dict[str, Any] | None = None
    if execute_deploy_dry_run:
        root_dir = Path(repo_root) if repo_root is not None else REPO_ROOT
        deploy_path = root_dir / "deploy" / "gcp_openrouter" / "deploy.sh"
        deploy_dry_run_res = run_deploy_sh_dry_run(
            deploy_path,
            expected_sha=clean_expected,
            runner=runner,
            env=env,
            lang=lang_code,
        )
        if deploy_dry_run_res.get("executed"):
            subprocess_calls += 1
        if not deploy_dry_run_res.get("ok"):
            diagnostics.append(str(deploy_dry_run_res.get("diagnostic") or "deploy_dry_run_failed"))

    csp_nonces_unique = len(seen_nonces) == len(PR949_HTML_ROUTES) and all(
        routes_report.get(hp, {}).get("csp_verified") is True for hp in PR949_HTML_ROUTES
    )
    html_all_ok = csp_nonces_unique and all(
        routes_report.get(hp, {}).get("ok") is True for hp in PR949_HTML_ROUTES
    )
    holiday_all_ok = bool(html_all_ok and manifest_ok and feedback_deployed and raid_status_ok)
    any_holiday_404 = any(
        routes_report.get(p, {}).get("status_code") == 404
        for p in (*PR949_HTML_ROUTES, "/v1/hivemind/onboarding/manifest", "/v1/hivemind/onboarding/feedback/summary", "/v1/hivemind/raid/status", "/v1/hivemind/arena/summary")
        if p in routes_report
    )
    stale_revision_detected = bool(
        sha_drift_detected
        or any_holiday_404
        or holiday_contract_fail_seen
        or (core_any_200 and holiday_404_or_5xx_seen)
    )

    arena_gate_ok = (not check_arena_summary) or (arena_summary_ok is True)
    idem_gate_ok = (not verify_idempotency_replay) or (idempotent_dedup_ok is True)
    deploy_gate_ok = (not execute_deploy_dry_run) or (
        isinstance(deploy_dry_run_res, Mapping) and deploy_dry_run_res.get("ok") is True
    )
    parity_ok = bool(
        health_ok
        and models_preflight_ok
        and holiday_all_ok
        and fleet_ok
        and quorum_gate_ok
        and ces_gate_ok
        and arena_gate_ok
        and idem_gate_ok
        and deploy_gate_ok
        and not sha_drift_detected
    )

    if parity_ok:
        summary = (
            f"Parité PR #949 vérifiée (nonces={len(seen_nonces)}/4, producteurs={online_producers}/{target_n}, build_sha={build_sha})."
            if is_fr
            else f"PR #949 live quorum & arena parity passed (nonces={len(seen_nonces)}/4, online={online_producers}/{target_n}, build_sha={build_sha})."
        )
    else:
        summary = (
            f"ÉCHEC PARITÉ PR #949 (stale_revision={stale_revision_detected}, routes={', '.join(missing_or_stale)})."
            if is_fr
            else f"PR #949 QUORUM/ARENA PARITY FAILED (stale_revision={stale_revision_detected}, routes={', '.join(missing_or_stale)})."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "mode": "pr949_quorum_arena_parity",
            "lang": lang_code,
            "ok": parity_ok,
            "ready": parity_ok,
            "parity_ok": parity_ok,
            "quorum_gate_ok": quorum_gate_ok,
            "compliance_ok": True,
            "corporate_detected": False,
            "exit_code": 0 if parity_ok else 2,
            "gateway": clean_gw,
            "holiday_routes_deployed_on_cloud_run": holiday_all_ok,
            "onboarding_feedback_deployed": feedback_deployed,
            "stale_revision_detected": stale_revision_detected,
            "sha_drift_detected": sha_drift_detected,
            "missing_or_stale_routes": missing_or_stale,
            "models_preflight_ok": models_preflight_ok,
            "coder_model_present": coder_model_present,
            "live_models_count": live_models_count,
            "models": model_ids,
            "online_producers": online_producers,
            "total_producers": total_producers,
            "distinct_physical_gpus": distinct_physical_gpus,
            "target_online_producers": target_n,
            "producer_quorum_met": producer_quorum_met,
            "awaiting_producer_quorum": awaiting_producer_quorum,
            "ces_slo_ok": ces_slo_ok,
            "zero_lag_slo_ok": zero_lag_slo_ok,
            "arena_summary_checked": bool(check_arena_summary),
            "arena_summary_ok": arena_summary_ok,
            "arena_summary": arena_summary_doc,
            "idempotency_verified": bool(verify_idempotency_replay),
            "idempotent_dedup_ok": idempotent_dedup_ok,
            "deploy_dry_run_executed": bool(execute_deploy_dry_run and deploy_dry_run_res is not None and deploy_dry_run_res.get("executed")),
            "deploy_dry_run": deploy_dry_run_res,
            "build_sha": build_sha,
            "expected_sha": clean_expected,
            "sha_matches": sha_matches,
            "cloud_run_revision": cloud_run_rev,
            "csp_nonces_unique": csp_nonces_unique,
            "routes": routes_report,
            "manifest_summary": manifest_summary,
            "feedback_summary": feedback_summary,
            "raid_status_ok": raid_status_ok,
            "fleet_ok": fleet_ok,
            "diagnostics": diagnostics,
            "zero_inference_verified": True,
            "file_writes": 0,
            "network_calls": network_calls,
            "subprocess_calls": subprocess_calls,
            "compliance": compliance,
            "summary": summary,
        }
    )


def verify_cloud_run_holiday_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    http_fn: HttpFn = default_http_request,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference Cloud Run holiday-route parity & pre-PR-#949 stale-revision smoke gate."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_expected = validate_expected_sha(expected_sha) if expected_sha is not None else None

    compliance = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "parity_check",
                "ok": False,
                "ready": False,
                "parity_ok": False,
                "compliance_ok": False,
                "exit_code": 2,
                "gateway": clean_gw,
                "holiday_routes_deployed_on_cloud_run": False,
                "onboarding_feedback_deployed": False,
                "stale_revision_detected": False,
                "sha_drift_detected": False,
                "missing_or_stale_routes": [],
                "online_producers": 0,
                "total_producers": 0,
                "build_sha": "unknown",
                "expected_sha": clean_expected,
                "sha_matches": None,
                "cloud_run_revision": None,
                "csp_nonces_unique": False,
                "routes": {},
                "manifest_summary": {},
                "feedback_summary": {},
                "diagnostics": [compliance["summary"]],
                "zero_inference_verified": True,
                "file_writes": 0,
                "network_calls": 0,
                "compliance": compliance,
                "summary": compliance["summary"],
            }
        )

    routes_report: dict[str, dict[str, Any]] = {}
    missing_or_stale: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    network_calls = 0

    core_any_200 = False
    holiday_404_or_5xx_seen = False
    holiday_contract_fail_seen = False
    holiday_all_ok = True
    feedback_deployed = False
    online_producers = 0
    total_producers = 0
    build_sha = "unknown"
    cloud_run_rev: str | None = None
    manifest_summary: dict[str, Any] = {}
    feedback_summary: dict[str, Any] = {}

    for path, kind, group in PARITY_PROBE_ROUTES:
        status_code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}", timeout_s=HTTP_TIMEOUT_S)
        network_calls += 1

        if kind == "html":
            route_ok, detail = _inspect_holiday_html_route(path, status_code, body, seen_nonces)
            routes_report[path] = detail
            if not route_ok:
                holiday_all_ok = False
                missing_or_stale.append(path)
                if status_code != 200:
                    holiday_404_or_5xx_seen = True
                else:
                    holiday_contract_fail_seen = True
                diagnostics.append(f"{path}:{detail['reason']}")
            continue

        if path == "/health":
            ok = status_code == 200 and body.get("status") == "ok"
            if ok:
                core_any_200 = True
            else:
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:http_{status_code}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": ok,
                "schema_verified": ok,
                "reason": "ok" if ok else f"http_{status_code}",
            }
        elif path == "/v1/models":
            raw_data = body.get("data")
            ok = status_code == 200 and isinstance(raw_data, list)
            if ok:
                core_any_200 = True
            else:
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:http_{status_code}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": ok,
                "schema_verified": ok,
                "models_count": len(raw_data) if isinstance(raw_data, list) else 0,
                "reason": "ok" if ok else f"http_{status_code}",
            }
        elif path == "/v1/hivemind/onboarding/manifest":
            if status_code == 200:
                schema_ok, manifest_summary, reason = _inspect_manifest_json(body)
                build_sha = str(manifest_summary.get("build_sha") or "unknown")
                cloud_run_rev = manifest_summary.get("cloud_run_revision")
                ok = schema_ok
                if not ok:
                    holiday_contract_fail_seen = True
            else:
                ok = False
                reason = f"http_{status_code}"
                holiday_404_or_5xx_seen = True
            if not ok:
                holiday_all_ok = False
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:{reason}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": ok,
                "schema_verified": ok,
                "reason": reason,
            }
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            if status_code == 200:
                schema_ok, feedback_summary, reason = _inspect_feedback_summary_json(body)
                ok = schema_ok
                feedback_deployed = schema_ok
                if not ok:
                    holiday_contract_fail_seen = True
            else:
                ok = False
                reason = f"http_{status_code}"
                holiday_404_or_5xx_seen = True
            if not ok:
                holiday_all_ok = False
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:{reason}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": ok,
                "schema_verified": ok,
                "reason": reason,
            }
        else:
            raw_live = body.get("live_hiveminders")
            raw_tot = body.get("total_hiveminders")
            live_ok = not isinstance(raw_live, bool) and isinstance(raw_live, int) and raw_live >= 0
            tot_ok = not isinstance(raw_tot, bool) and isinstance(raw_tot, int) and raw_tot >= 0
            ok = status_code == 200 and live_ok
            if ok:
                online_producers = int(raw_live)
                total_producers = int(raw_tot) if tot_ok else online_producers
            else:
                missing_or_stale.append(path)
                diagnostics.append(f"{path}:http_{status_code}")
            routes_report[path] = {
                "path": path,
                "status_code": status_code,
                "ok": ok,
                "schema_verified": ok,
                "online_producers": online_producers,
                "reason": "ok" if ok else f"http_{status_code}",
            }

        _ = group

    sha_matches: bool | None = None
    sha_drift_detected = False
    if clean_expected is not None:
        sha_matches = shas_match(build_sha, clean_expected)
        if not sha_matches:
            sha_drift_detected = True
            if "/v1/hivemind/onboarding/manifest" not in missing_or_stale:
                missing_or_stale.append("/v1/hivemind/onboarding/manifest")
            diagnostics.append(f"build_sha_drift:live={build_sha},expected={clean_expected}")

    any_holiday_404 = any(
        routes_report.get(p, {}).get("status_code") == 404
        for p, _, grp in PARITY_PROBE_ROUTES
        if grp == "holiday"
    )
    stale_revision_detected = bool(
        sha_drift_detected
        or any_holiday_404
        or holiday_contract_fail_seen
        or (core_any_200 and holiday_404_or_5xx_seen)
    )
    csp_nonces_unique = len(seen_nonces) == 3 and all(
        routes_report.get(hp, {}).get("csp_verified") is True for hp in ("/noel", "/quick", "/raid-noel")
    )
    parity_ok = len(missing_or_stale) == 0 and holiday_all_ok and not sha_drift_detected

    if parity_ok:
        summary = (
            f"Parité Noël Cloud Run vérifiée (build_sha={build_sha}, producteurs en ligne={online_producers})."
            if is_fr
            else f"Cloud Run holiday-route parity verified (build_sha={build_sha}, online_producers={online_producers})."
        )
    else:
        stale_joined = ", ".join(missing_or_stale)
        summary = (
            f"ÉCHEC PARITÉ CLOUD RUN (stale_revision={stale_revision_detected}, routes={stale_joined})."
            if is_fr
            else f"CLOUD RUN HOLIDAY PARITY FAILED (stale_revision={stale_revision_detected}, routes={stale_joined})."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "mode": "parity_check",
            "ok": parity_ok,
            "ready": parity_ok,
            "parity_ok": parity_ok,
            "compliance_ok": True,
            "exit_code": 0 if parity_ok else 2,
            "gateway": clean_gw,
            "holiday_routes_deployed_on_cloud_run": holiday_all_ok,
            "onboarding_feedback_deployed": feedback_deployed,
            "stale_revision_detected": stale_revision_detected,
            "sha_drift_detected": sha_drift_detected,
            "missing_or_stale_routes": missing_or_stale,
            "online_producers": online_producers,
            "total_producers": total_producers,
            "build_sha": build_sha,
            "expected_sha": clean_expected,
            "sha_matches": sha_matches,
            "cloud_run_revision": cloud_run_rev,
            "csp_nonces_unique": csp_nonces_unique,
            "routes": routes_report,
            "manifest_summary": manifest_summary,
            "feedback_summary": feedback_summary,
            "diagnostics": diagnostics,
            "zero_inference_verified": True,
            "file_writes": 0,
            "network_calls": network_calls,
            "compliance": compliance,
            "summary": summary,
        }
    )


def _project_pr949_to_preflight(
    pr949_res: Mapping[str, Any],
    *,
    clean_gw: str,
    lang_code: str,
    require_holiday_routes: bool,
) -> dict[str, Any]:
    """Project `verify_pr949_vscode_quorum_and_arena_parity` into `one_shot_quick_preflight` receipt schema."""
    is_fr = lang_code == "fr"
    routes_map = dict(pr949_res.get("routes") or {})
    health_ok = bool(routes_map.get("/health", {}).get("ok"))
    models_ok = bool(pr949_res.get("models_preflight_ok"))
    quick_ok = bool(routes_map.get("/quick", {}).get("ok"))
    fb_ok = bool(pr949_res.get("onboarding_feedback_deployed"))
    live_models_count = int(pr949_res.get("live_models_count") or 0)
    coder_present = bool(pr949_res.get("coder_model_present"))
    model_ids = list(pr949_res.get("models") or [])
    ready = bool(pr949_res.get("parity_ok"))
    stale_rev = bool(pr949_res.get("stale_revision_detected"))
    missing = list(pr949_res.get("missing_or_stale_routes") or [])
    if ready:
        verdict = "ready"
        guidance = ""
    elif not (health_ok and models_ok):
        verdict = "gateway_preflight_failed"
        guidance = (
            "Passerelle /health ou /v1/models (hivemind/coder) indisponible."
            if is_fr
            else "Gateway /health or /v1/models (hivemind/coder) failed."
        )
    elif stale_rev:
        verdict = "stale_cloud_run_revision"
        guidance = (
            "Révision Cloud Run en retard détectée sur les routes PR #949."
            if is_fr
            else "Stale Cloud Run revision detected on PR #949 routes."
        )
    else:
        verdict = "pr949_quorum_or_slo_failed"
        guidance = str(pr949_res.get("summary") or "")

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "mode": "one_shot_quick_preflight",
            "ok": ready,
            "ready": ready,
            "preflight_ok": ready,
            "compliance_ok": True,
            "exit_code": int(pr949_res.get("exit_code", 2)),
            "verdict": verdict,
            "gateway": clean_gw,
            "lang": lang_code,
            "health_ok": health_ok,
            "models_preflight_ok": models_ok,
            "quick_route_ok": quick_ok,
            "feedback_endpoint_ok": fb_ok,
            "live_models_count": live_models_count,
            "coder_model_present": coder_present,
            "models": model_ids,
            "stale_revision_detected": stale_rev,
            "missing_or_stale_routes": missing,
            "require_holiday_routes": bool(require_holiday_routes),
            "feedback_summary": dict(pr949_res.get("feedback_summary") or {}),
            "online_producers": pr949_res.get("online_producers"),
            "producer_quorum_met": pr949_res.get("producer_quorum_met"),
            "awaiting_producer_quorum": pr949_res.get("awaiting_producer_quorum"),
            "ces_slo_ok": pr949_res.get("ces_slo_ok"),
            "zero_lag_slo_ok": pr949_res.get("zero_lag_slo_ok"),
            "arena_summary_ok": pr949_res.get("arena_summary_ok"),
            "arena_summary": dict(pr949_res.get("arena_summary") or {}),
            "idempotent_dedup_ok": pr949_res.get("idempotent_dedup_ok"),
            "deploy_dry_run": pr949_res.get("deploy_dry_run"),
            "pr949_parity": pr949_res,
            "local_fallback_guidance": guidance,
            "routes": routes_map,
            "diagnostics": list(pr949_res.get("diagnostics") or []),
            "zero_inference_verified": True,
            "file_writes": 0,
            "network_calls": int(pr949_res.get("network_calls") or 0),
            "subprocess_calls": int(pr949_res.get("subprocess_calls") or 0),
            "compliance": pr949_res.get("compliance"),
            "summary": str(pr949_res.get("summary") or ""),
        }
    )


def run_one_shot_quick_health_preflight(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_fn: HttpFn = default_http_request,
    raw_http_fn: Any = default_http_raw,
    runner: RunnerFn = default_subprocess_runner,
    require_holiday_routes: bool = False,
    pr949_quorum_gate: bool = False,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    require_ces_slo: bool = False,
    check_arena_summary: bool = False,
    verify_idempotency_replay: bool = False,
    execute_deploy_dry_run: bool = False,
    repo_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference 1-shot `/health` + `/v1/models` + `/quick` + `/feedback/summary` health preflight."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    _validate_pr949_options(
        target_online_producers=target_online_producers,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
        require_live_quorum=require_live_quorum,
        require_ces_slo=require_ces_slo,
        check_arena_summary=check_arena_summary,
        verify_idempotency_replay=verify_idempotency_replay,
        execute_deploy_dry_run=execute_deploy_dry_run,
    )
    pr949_requested = bool(
        pr949_quorum_gate
        or require_live_quorum
        or require_ces_slo
        or check_arena_summary
        or verify_idempotency_replay
        or execute_deploy_dry_run
    )

    compliance = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "one_shot_quick_preflight",
                "ok": False,
                "ready": False,
                "preflight_ok": False,
                "compliance_ok": False,
                "exit_code": 2,
                "verdict": "blocked_corporate_environment",
                "gateway": clean_gw,
                "lang": lang_code,
                "health_ok": False,
                "models_preflight_ok": False,
                "quick_route_ok": False,
                "feedback_endpoint_ok": False,
                "live_models_count": 0,
                "coder_model_present": False,
                "models": [],
                "stale_revision_detected": False,
                "missing_or_stale_routes": [],
                "require_holiday_routes": bool(require_holiday_routes),
                "feedback_summary": {},
                "local_fallback_guidance": compliance["summary"],
                "routes": {},
                "diagnostics": [compliance["summary"]],
                "zero_inference_verified": True,
                "file_writes": 0,
                "network_calls": 0,
                "subprocess_calls": 0,
                "compliance": compliance,
                "summary": compliance["summary"],
            }
        )

    if pr949_requested:
        pr949_res = verify_pr949_vscode_quorum_and_arena_parity(
            clean_gw,
            expected_sha=expected_sha,
            target_online_producers=target_online_producers,
            min_ease_rating=min_ease_rating,
            min_zero_lag_ratio=min_zero_lag_ratio,
            require_live_quorum=require_live_quorum,
            require_ces_slo=require_ces_slo,
            check_arena_summary=bool(check_arena_summary or pr949_quorum_gate),
            verify_idempotency_replay=verify_idempotency_replay,
            execute_deploy_dry_run=execute_deploy_dry_run,
            repo_root=repo_root,
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
            runner=runner,
            env=env,
            cwd=cwd,
            lang=lang_code,
        )
        return _project_pr949_to_preflight(
            pr949_res,
            clean_gw=clean_gw,
            lang_code=lang_code,
            require_holiday_routes=require_holiday_routes,
        )

    routes_report: dict[str, dict[str, Any]] = {}
    missing_or_stale: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    network_calls = 0

    h_status, h_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/health", timeout_s=HTTP_TIMEOUT_S)
    network_calls += 1
    health_ok = bool(h_status == 200 and h_body.get("status") == "ok")
    if not health_ok:
        missing_or_stale.append("/health")
        diagnostics.append(f"/health:http_{h_status}")
    routes_report["/health"] = {
        "path": "/health",
        "status_code": h_status,
        "ok": health_ok,
        "reason": "ok" if health_ok else f"http_{h_status}",
    }

    m_status, m_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/models", timeout_s=HTTP_TIMEOUT_S)
    network_calls += 1
    raw_models = m_body.get("data")
    model_ids = (
        [
            str(item["id"]).strip()
            for item in raw_models
            if isinstance(item, Mapping) and isinstance(item.get("id"), str) and item["id"].strip()
        ]
        if isinstance(raw_models, list)
        else []
    )
    live_models_count = len(model_ids)
    coder_model_present = DEFAULT_MODELS[0] in model_ids
    models_preflight_ok = bool(
        m_status == 200 and isinstance(raw_models, list) and live_models_count > 0 and coder_model_present
    )
    if not models_preflight_ok:
        missing_or_stale.append("/v1/models")
        m_reason = (
            f"http_{m_status}"
            if m_status != 200 or not isinstance(raw_models, list)
            else ("empty_model_list" if live_models_count == 0 else "missing_hivemind_coder")
        )
        diagnostics.append(f"/v1/models:{m_reason}")
    else:
        m_reason = "ok"
    routes_report["/v1/models"] = {
        "path": "/v1/models",
        "status_code": m_status,
        "ok": models_preflight_ok,
        "live_models_count": live_models_count,
        "coder_model_present": coder_model_present,
        "reason": m_reason,
    }

    q_status, q_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/quick", timeout_s=HTTP_TIMEOUT_S)
    network_calls += 1
    quick_route_ok, quick_detail = _inspect_holiday_html_route("/quick", q_status, q_body, seen_nonces)
    routes_report["/quick"] = quick_detail
    if not quick_route_ok:
        missing_or_stale.append("/quick")
        diagnostics.append(f"/quick:{quick_detail['reason']}")

    fb_status, fb_body = _invoke_http_fn(
        http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/feedback/summary", timeout_s=HTTP_TIMEOUT_S
    )
    network_calls += 1
    feedback_summary: dict[str, Any] = {}
    if fb_status == 200:
        feedback_endpoint_ok, feedback_summary, fb_reason = _inspect_feedback_summary_json(fb_body)
    else:
        feedback_endpoint_ok = False
        fb_reason = f"http_{fb_status}"
    if not feedback_endpoint_ok:
        missing_or_stale.append("/v1/hivemind/onboarding/feedback/summary")
        diagnostics.append(f"/v1/hivemind/onboarding/feedback/summary:{fb_reason}")
    routes_report["/v1/hivemind/onboarding/feedback/summary"] = {
        "path": "/v1/hivemind/onboarding/feedback/summary",
        "status_code": fb_status,
        "ok": feedback_endpoint_ok,
        "reason": fb_reason,
    }

    core_ok = bool(health_ok and models_preflight_ok)
    holiday_ok = bool(quick_route_ok and feedback_endpoint_ok)
    stale_revision_detected = bool(
        q_status == 404
        or fb_status == 404
        or (q_status == 200 and not quick_route_ok)
        or (fb_status == 200 and not feedback_endpoint_ok)
        or ((health_ok or m_status == 200) and not holiday_ok)
    )
    ready = bool(core_ok and (holiday_ok if require_holiday_routes else True))
    exit_code = 0 if ready else 2

    if not core_ok:
        local_fallback_guidance = (
            "Passerelle /health ou /v1/models (hivemind/coder) indisponible : vérifiez l'URL Cloud Run "
            "ou démarrez un nœud local sur http://127.0.0.1:8000 (sans jamais utiliser de PC d'entreprise)."
            if is_fr
            else "Gateway /health or /v1/models (hivemind/coder) failed: verify your Cloud Run URL or "
            "point --gateway to a local loopback gateway at http://127.0.0.1:8000 (personal PC only)."
        )
        verdict = "gateway_preflight_failed"
        summary = (
            f"ÉCHEC PRÉ-VOL PASSERELLE ({', '.join(missing_or_stale)}). {local_fallback_guidance}"
            if is_fr
            else f"GATEWAY PREFLIGHT FAILED ({', '.join(missing_or_stale)}). {local_fallback_guidance}"
        )
    elif not holiday_ok:
        local_fallback_guidance = (
            "Révision Cloud Run en retard détectée sur /quick ou /feedback/summary : votre configuration "
            "Continue locale (~/.continue/config.yaml) reste prête sur /v1/models et les retours CES sont "
            "mis en file locale (~/.hivemind/vscode_setup_ces_queue.json) pour --flush-ces-queue."
            if is_fr
            else "Stale Cloud Run revision detected on /quick or /feedback/summary: your local VS Code "
            "Continue config (~/.continue/config.yaml) still works via /v1/models, and CES feedback "
            "spools locally to ~/.hivemind/vscode_setup_ces_queue.json for later --flush-ces-queue."
        )
        verdict = "stale_cloud_run_revision" if require_holiday_routes else "ready_local_fallback"
        summary = (
            (
                f"ÉCHEC PRÉ-VOL ROUTES NOËL ({', '.join(missing_or_stale)}). {local_fallback_guidance}"
                if require_holiday_routes
                else f"Passerelle /health et /v1/models prêts (mode repli local : {', '.join(missing_or_stale)})."
            )
            if is_fr
            else (
                f"HOLIDAY ROUTE PREFLIGHT BLOCKED ({', '.join(missing_or_stale)}). {local_fallback_guidance}"
                if require_holiday_routes
                else f"Gateway /health & /v1/models ready (local fallback active; stale routes: {', '.join(missing_or_stale)})."
            )
        )
    else:
        local_fallback_guidance = ""
        verdict = "ready"
        summary = (
            f"Pré-vol 1-shot validé (/health, /v1/models={live_models_count}, /quick CSP, /feedback/summary)."
            if is_fr
            else f"1-shot quick preflight passed (/health, /v1/models={live_models_count}, /quick CSP, /feedback/summary)."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "mode": "one_shot_quick_preflight",
            "ok": ready,
            "ready": ready,
            "preflight_ok": ready,
            "compliance_ok": True,
            "exit_code": exit_code,
            "verdict": verdict,
            "gateway": clean_gw,
            "lang": lang_code,
            "health_ok": health_ok,
            "models_preflight_ok": models_preflight_ok,
            "quick_route_ok": quick_route_ok,
            "feedback_endpoint_ok": feedback_endpoint_ok,
            "live_models_count": live_models_count,
            "coder_model_present": coder_model_present,
            "models": model_ids,
            "stale_revision_detected": stale_revision_detected,
            "missing_or_stale_routes": missing_or_stale,
            "require_holiday_routes": bool(require_holiday_routes),
            "feedback_summary": feedback_summary,
            "local_fallback_guidance": local_fallback_guidance,
            "routes": routes_report,
            "diagnostics": diagnostics,
            "zero_inference_verified": True,
            "file_writes": 0,
            "network_calls": network_calls,
            "compliance": compliance,
            "summary": summary,
        }
    )


def _write_file_secure_atomic(path: Path, content: str, *, posix_mode: int, eff_os: str) -> None:
    """Atomically write UTF-8 `content` to `path` with `posix_mode` (`0600`) from creation."""
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


def strip_jsonc_comments_and_trailing_commas(raw: str) -> str:
    """Convert VS Code JSONC (`//`, `/* */`, trailing commas) into strict JSON."""
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
    """Guarantee workspace `.vscode/*` files never hold raw keys, Bearer tokens, or OS home paths."""
    if _API_KEY_RE.search(content):
        raise QuickPresetError("Workspace secret guard blocked raw hm-... API key in workspace file.")
    if _BEARER_RE.search(content):
        raise QuickPresetError("Workspace secret guard blocked Bearer token in workspace file.")
    if _HOME_PATH_RE.search(content):
        raise QuickPresetError("Workspace secret guard blocked unredacted OS home path in workspace file.")
    return content


_CONTINUE_MANAGED_HEADER_LINES: tuple[str, ...] = (
    "# HiveMind personal hobby/open-source coding preset for VS Code (Continue)",
    "# 100% personal home PCs & personal hobby/open-source projects ONLY (0.00 EUR / 0.00 USD, AGPL-3.0).",
    "# Never use on corporate hardware or send corporate/confidential source code.",
)


def _strip_yaml_scalar_comment(raw: str) -> str:
    """Strip unquoted inline YAML `# comment` and surrounding matching quotes."""
    in_sq = False
    in_dq = False
    out: list[str] = []
    for idx, ch in enumerate(raw):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq and (idx == 0 or raw[idx - 1] != "\\"):
            in_dq = not in_dq
        elif ch == "#" and not in_sq and not in_dq and (idx == 0 or raw[idx - 1].isspace()):
            break
        out.append(ch)
    cleaned = "".join(out).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        return cleaned[1:-1].strip()
    return cleaned


def _build_continue_model_dicts(
    gateway: str,
    api_key: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[dict[str, Any]]:
    clean_gw = validate_gateway_url(gateway)
    if not is_valid_hivemind_key(api_key):
        raise QuickPresetError("Cannot render Continue config without a valid hm-... API key.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 64:
        raise QuickPresetError("max_tokens must be an integer >= 64.")
    clean_key = api_key.strip()
    api_base = f"{clean_gw}/v1"
    return [
        {
            "name": "HiveMind Coder",
            "provider": "openai",
            "model": DEFAULT_MODELS[0],
            "apiBase": api_base,
            "apiKey": clean_key,
            "roles": ["chat", "edit", "apply"],
            "defaultCompletionOptions": {"maxTokens": max_tokens},
        },
        {
            "name": "HiveMind Bee-Stars (Science)",
            "provider": "openai",
            "model": DEFAULT_MODELS[1],
            "apiBase": api_base,
            "apiKey": clean_key,
            "roles": ["chat"],
            "defaultCompletionOptions": {"maxTokens": max_tokens},
        },
    ]


def _render_continue_model_yaml_lines(
    gateway: str,
    api_key: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    item_indent: str = "  ",
) -> list[str]:
    models = _build_continue_model_dicts(gateway, api_key, max_tokens=max_tokens)
    sub = item_indent + "  "
    sub2 = sub + "  "
    lines: list[str] = []
    for entry in models:
        lines.extend(
            [
                f"{item_indent}- name: {entry['name']}",
                f"{sub}provider: {entry['provider']}",
                f"{sub}model: {entry['model']}",
                f"{sub}apiBase: {entry['apiBase']}",
                f"{sub}apiKey: {entry['apiKey']}",
                f"{sub}roles:",
                *[f"{sub2}- {role}" for role in entry["roles"]],
                f"{sub}defaultCompletionOptions:",
                f"{sub2}maxTokens: {entry['defaultCompletionOptions']['maxTokens']}",
            ]
        )
    return lines


def _is_hivemind_continue_model_dict(entry: Any) -> bool:
    if not isinstance(entry, Mapping):
        return False
    model_val = str(entry.get("model") or "").strip().lower()
    name_val = str(entry.get("name") or entry.get("title") or "").strip().lower()
    api_base = str(entry.get("apiBase") or "").strip().lower()
    api_key = str(entry.get("apiKey") or "").strip().lower()
    provider = str(entry.get("provider") or "").strip().lower()
    if model_val in {"hivemind/coder", "hivemind/bee-stars", "hivemind/auto"} or model_val.startswith("hivemind/"):
        return True
    return bool(
        name_val.startswith("hivemind")
        and (
            provider == "openai"
            or ".run.app" in api_base
            or api_key.startswith(("hm-", "hm_live_"))
            or "hivemind" in api_base
        )
    )


def _is_hivemind_continue_model_item(item_lines: Sequence[str]) -> bool:
    kv: dict[str, str] = {}
    for raw_ln in item_lines:
        stripped = raw_ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            stripped = stripped[1:].strip()
        if ":" in stripped and not stripped.startswith("- "):
            k_raw, v_raw = stripped.split(":", 1)
            k = k_raw.strip().lower()
            if k and " " not in k:
                kv[k] = _strip_yaml_scalar_comment(v_raw)
    return _is_hivemind_continue_model_dict(
        {
            "name": kv.get("name") or kv.get("title") or "",
            "provider": kv.get("provider") or "",
            "model": kv.get("model") or "",
            "apiBase": kv.get("apibase") or "",
            "apiKey": kv.get("apikey") or "",
        }
    )


def _split_continue_yaml_models(
    lines: list[str],
) -> tuple[int | None, int, list[str], list[list[str]], int]:
    """Locate top-level `models:` block and split it into sequence item line groups."""
    models_idx: int | None = None
    for idx, ln in enumerate(lines):
        if ln and not ln[0].isspace() and not ln.startswith("#"):
            m = re.match(r"^models\s*:(.*)$", ln)
            if m:
                inline_val = _strip_yaml_scalar_comment(m.group(1))
                if inline_val and inline_val.lower() not in ("[]", "~", "null"):
                    raise QuickPresetError("Existing Continue config.yaml has non-block 'models' value.")
                models_idx = idx
                break
    if models_idx is None:
        return None, len(lines), [], [], 2

    end_idx = len(lines)
    for idx in range(models_idx + 1, len(lines)):
        ln = lines[idx]
        if ln and not ln[0].isspace() and not ln.startswith(("#", "-")):
            end_idx = idx
            break

    while end_idx > models_idx + 1 and (
        not lines[end_idx - 1].strip() or lines[end_idx - 1].startswith("#")
    ):
        end_idx -= 1

    body = lines[models_idx + 1 : end_idx]
    seq_indent: int | None = None
    for ln in body:
        lst = ln.lstrip(" ")
        if lst == "-" or lst.startswith("- "):
            seq_indent = len(ln) - len(lst)
            break
    if seq_indent is None:
        return models_idx, end_idx, body, [], 2

    leading_comments: list[str] = []
    items: list[list[str]] = []
    current: list[str] | None = None
    for ln in body:
        lst = ln.lstrip(" ")
        indent = len(ln) - len(lst)
        is_item_start = indent == seq_indent and (lst == "-" or lst.startswith("- "))
        if is_item_start:
            if current is not None:
                items.append(current)
            current = [ln]
        elif current is not None:
            current.append(ln)
        else:
            leading_comments.append(ln)
    if current is not None:
        items.append(current)
    return models_idx, end_idx, leading_comments, items, seq_indent


def build_continue_provider_yaml(
    gateway: str,
    api_key: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Render Continue OpenAI-compatible YAML (`hivemind/coder` & `hivemind/bee-stars`)."""
    model_lines = _render_continue_model_yaml_lines(
        gateway, api_key, max_tokens=max_tokens, item_indent="  "
    )
    return "\n".join(
        [
            *_CONTINUE_MANAGED_HEADER_LINES,
            "name: HiveMind Holiday Quick Preset",
            "version: 0.0.1",
            "schema: v1",
            "models:",
            *model_lines,
        ]
    ) + "\n"


def merge_continue_config_yaml(
    existing_text: str | None,
    *,
    gateway: str,
    api_key: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Merge HiveMind models into `~/.continue/config.yaml` while preserving non-HiveMind user config."""
    hivemind_dicts = _build_continue_model_dicts(gateway, api_key, max_tokens=max_tokens)
    if existing_text is None or not existing_text.strip():
        return build_continue_provider_yaml(gateway, api_key, max_tokens=max_tokens)
    if "\x00" in existing_text:
        raise QuickPresetError("Existing Continue config.yaml contains invalid null bytes.")

    normalized = existing_text.replace("\r\n", "\n")
    stripped_start = normalized.lstrip()
    if stripped_start.startswith(("{", "[")):
        try:
            loaded = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise QuickPresetError("Existing Continue config contains invalid JSON; refusing to overwrite.") from exc
        if not isinstance(loaded, dict):
            raise QuickPresetError("Existing Continue config JSON root must be an object.")
        raw_models = loaded.get("models", [])
        if not isinstance(raw_models, list):
            raise QuickPresetError("Existing Continue config 'models' field must be a list.")
        preserved_dicts = [m for m in raw_models if not _is_hivemind_continue_model_dict(m)]
        merged_doc = {**loaded, "models": [*preserved_dicts, *hivemind_dicts]}
        return json.dumps(merged_doc, indent=2, ensure_ascii=False) + "\n"

    if stripped_start.startswith("- "):
        raise QuickPresetError("Existing Continue config.yaml must be a YAML mapping, not a top-level list.")

    lines = normalized.splitlines()
    models_idx, end_idx, leading_comments, items, seq_indent = _split_continue_yaml_models(lines)
    hm_lines = _render_continue_model_yaml_lines(
        gateway, api_key, max_tokens=max_tokens, item_indent=" " * seq_indent
    )

    if models_idx is None:
        trimmed = list(lines)
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        out_lines = [*trimmed, "models:", *hm_lines]
        return "\n".join(out_lines) + "\n"

    preserved_items = [item for item in items if not _is_hivemind_continue_model_item(item)]
    merged_item_lines: list[str] = []
    for item in preserved_items:
        merged_item_lines.extend(item)
    merged_item_lines.extend(hm_lines)

    out_lines = [
        *lines[:models_idx],
        "models:",
        *leading_comments,
        *merged_item_lines,
        *lines[end_idx:],
    ]
    return "\n".join(out_lines).rstrip("\n") + "\n"


def remove_hivemind_from_continue_config_yaml(existing_text: str) -> str | None:
    """Remove only HiveMind entries from `~/.continue/config.yaml`; return `None` if nothing user-owned remains."""
    if not existing_text or not existing_text.strip():
        return None
    normalized = existing_text.replace("\r\n", "\n")
    stripped_start = normalized.lstrip()
    if stripped_start.startswith(("{", "[")):
        try:
            loaded = json.loads(normalized)
        except json.JSONDecodeError:
            return existing_text
        if not isinstance(loaded, dict):
            return existing_text
        raw_models = loaded.get("models")
        if not isinstance(raw_models, list):
            return existing_text
        preserved = [m for m in raw_models if not _is_hivemind_continue_model_dict(m)]
        other_keys = {k: v for k, v in loaded.items() if k != "models"}
        if not preserved and not other_keys:
            return None
        rebuilt = {**other_keys, "models": preserved}
        return json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n"

    raw_lines = normalized.splitlines()
    lines = [ln for ln in raw_lines if ln not in _CONTINUE_MANAGED_HEADER_LINES]
    models_idx, end_idx, leading_comments, items, _ = _split_continue_yaml_models(lines)
    if models_idx is None:
        return existing_text

    preserved_items = [item for item in items if not _is_hivemind_continue_model_item(item)]
    before_lines = lines[:models_idx]
    after_lines = lines[end_idx:]

    if preserved_items:
        merged_item_lines: list[str] = []
        for item in preserved_items:
            merged_item_lines.extend(item)
        out_lines = [*before_lines, "models:", *leading_comments, *merged_item_lines, *after_lines]
        return "\n".join(out_lines).strip("\n") + "\n"

    default_meta = {
        "name: hivemind holiday quick preset",
        "version: 0.0.1",
        "schema: v1",
    }
    has_user_keys = any(
        ln.strip()
        and not ln.strip().startswith("#")
        and ln.strip().lower() not in default_meta
        for ln in (*before_lines, *after_lines)
    )
    if not has_user_keys:
        return None
    out_lines = [*before_lines, "models: []", *after_lines]
    return "\n".join(out_lines).strip("\n") + "\n"


def build_merged_vscode_settings_json(existing_text: str | None, *, gateway: str) -> str:
    """Merge non-secret OpenAI-compatible `apiBase` & model IDs into `<workspace>/.vscode/settings.json`."""
    clean_gw = validate_gateway_url(gateway)
    parsed: dict[str, Any] = {}
    if existing_text is not None and existing_text.strip():
        strict_json = strip_jsonc_comments_and_trailing_commas(existing_text)
        if strict_json.strip():
            try:
                loaded = json.loads(strict_json)
            except json.JSONDecodeError as exc:
                raise QuickPresetError(
                    "Existing .vscode/settings.json contains invalid JSON; refusing to overwrite."
                ) from exc
            if not isinstance(loaded, dict):
                raise QuickPresetError("Existing .vscode/settings.json root must be a JSON object.")
            parsed = dict(loaded)

    cleaned_user_keys: dict[str, Any] = {
        k: v for k, v in parsed.items() if not str(k).startswith(HIVEMIND_VSCODE_PREFIX)
    }
    _assert_workspace_content_secret_free(json.dumps(cleaned_user_keys, ensure_ascii=False))

    preset_settings: dict[str, Any] = {
        "hivemind.openaiCompatible.apiBase": f"{clean_gw}/v1",
        "hivemind.openaiCompatible.defaultModel": DEFAULT_MODELS[0],
        "hivemind.openaiCompatible.models": list(DEFAULT_MODELS),
        "hivemind.continue.providerConfigPath": "~/.continue/hivemind_provider.yaml",
        "hivemind.credentials.secretStorage": "~/.hivemind/credentials.json",
        "hivemind.compliance.personalPcOnly": True,
        "hivemind.compliance.financialPromise": FINANCIAL_PROMISE,
    }
    merged = {**cleaned_user_keys, **preset_settings}
    rendered = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
    return _assert_workspace_content_secret_free(rendered)


def _is_hivemind_managed_task(task: object) -> bool:
    if not isinstance(task, Mapping):
        return False
    label = str(task.get("label") or "").strip()
    return label in _MANAGED_TASK_ALIASES or label.startswith(HIVEMIND_TASK_PREFIX)


def _build_hivemind_preset_tasks(gateway: str, *, os_name: str | None = None) -> list[dict[str, Any]]:
    clean_gw = validate_gateway_url(gateway)
    eff_os = os_name if os_name is not None else os.name
    py_cmd = "python" if eff_os == "nt" else "python3"
    script_rel = "scripts/noel_vscode_quick_preset.py"
    return [
        {
            "label": TASK_LABEL_QUICK_CONNECT,
            "type": "process",
            "command": py_cmd,
            "args": [
                script_rel,
                "--install-preset",
                "--workspace",
                "${workspaceFolder}",
                "--gateway",
                clean_gw,
            ],
            "problemMatcher": [],
            "presentation": {"reveal": "always", "panel": "shared"},
            "detail": "Zero-inference <60s VS Code quick-connect check (100% personal PC only, 0.00 EUR / 0.00 USD).",
        },
        {
            "label": TASK_LABEL_PARITY_CHECK,
            "type": "process",
            "command": py_cmd,
            "args": [
                script_rel,
                "--check-parity",
                "--require-holiday-routes",
                "--gateway",
                clean_gw,
            ],
            "problemMatcher": [],
            "presentation": {"reveal": "always", "panel": "shared"},
            "detail": "Zero-inference Cloud Run /noel, /quick, /raid-noel, manifest & CES parity smoke check.",
        },
        {
            "label": TASK_LABEL_GAMER_PREFLIGHT,
            "type": "process",
            "command": py_cmd,
            "args": [
                script_rel,
                "--install-preset",
                "--workspace",
                "${workspaceFolder}",
                "--first-heartbeat",
                "--profile",
                "gamer",
                "--gateway",
                clean_gw,
            ],
            "problemMatcher": [],
            "presentation": {"reveal": "always", "panel": "shared"},
            "detail": "Verify <1.5s Steam worker yield, <=70 C thermal ceiling, and outbound OpenTrainer heartbeat.",
        },
        {
            "label": TASK_LABEL_CLEAN_UNINSTALL,
            "type": "process",
            "command": py_cmd,
            "args": [
                script_rel,
                "--uninstall",
                "--workspace",
                "${workspaceFolder}",
            ],
            "problemMatcher": [],
            "presentation": {"reveal": "always", "panel": "shared"},
            "detail": "Clean 10s removal of HiveMind preset tasks/settings and local credentials while preserving user tasks.",
        },
    ]


def build_merged_vscode_tasks_json(
    existing_text: str | None,
    *,
    gateway: str,
    os_name: str | None = None,
) -> str:
    """Idempotently merge the 4 non-secret HiveMind one-click tasks into `<workspace>/.vscode/tasks.json`."""
    parsed: dict[str, Any] = {}
    if existing_text is not None and existing_text.strip():
        strict_json = strip_jsonc_comments_and_trailing_commas(existing_text)
        if strict_json.strip():
            try:
                loaded = json.loads(strict_json)
            except json.JSONDecodeError as exc:
                raise QuickPresetError(
                    "Existing .vscode/tasks.json contains invalid JSON; refusing to overwrite."
                ) from exc
            if not isinstance(loaded, dict):
                raise QuickPresetError("Existing .vscode/tasks.json root must be a JSON object.")
            parsed = dict(loaded)

    raw_tasks = parsed.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise QuickPresetError("Existing .vscode/tasks.json 'tasks' field must be a JSON list.")

    preserved_user_tasks = [item for item in raw_tasks if not _is_hivemind_managed_task(item)]
    _assert_workspace_content_secret_free(json.dumps(preserved_user_tasks, ensure_ascii=False))

    other_root_keys = {k: v for k, v in parsed.items() if k not in ("version", "tasks")}
    _assert_workspace_content_secret_free(json.dumps(other_root_keys, ensure_ascii=False))

    merged_doc: dict[str, Any] = {
        "version": "2.0.0",
        **other_root_keys,
        "tasks": [*preserved_user_tasks, *_build_hivemind_preset_tasks(gateway, os_name=os_name)],
    }
    rendered = json.dumps(merged_doc, indent=2, ensure_ascii=False) + "\n"
    return _assert_workspace_content_secret_free(rendered)


def check_workspace_gitignore_safety(workspace: Path, *, lang: str = "en") -> dict[str, Any]:
    """Check whether `<workspace>/.gitignore` ignores `.vscode/` preset files and `.bak` backups."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    gitignore_path = workspace / ".gitignore"

    if not gitignore_path.is_file():
        return {
            "safe": False,
            "gitignore_present": False,
            "vscode_ignored": False,
            "warning": (
                "Avertissement `.gitignore` : ajoutez `.vscode/` et `*.bak` dans `.gitignore` "
                "(aucun secret `hm-...` n'est écrit dans `.vscode/`)."
                if is_fr
                else "Workspace `.gitignore` notice: add `.vscode/` and `*.bak` to `.gitignore` "
                "(raw `hm-...` secrets are never written into `.vscode/`)."
            ),
        }

    try:
        lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []

    dir_patterns = {
        ".vscode",
        ".vscode/",
        "/.vscode",
        "/.vscode/",
        ".vscode/*",
        "/.vscode/*",
        ".vscode/**",
        "/.vscode/**",
    }
    rules = {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith(("#", "!"))}
    vscode_ignored = bool(
        (rules & dir_patterns)
        or (
            bool(rules & {".vscode/settings.json", "/.vscode/settings.json", "settings.json"})
            and bool(rules & {".vscode/tasks.json", "/.vscode/tasks.json", "tasks.json"})
        )
    )
    if not vscode_ignored:
        return {
            "safe": False,
            "gitignore_present": True,
            "vscode_ignored": False,
            "warning": (
                "Avertissement `.gitignore` : `.vscode/` (tasks.json / settings.json) n'est pas ignoré."
                if is_fr
                else "Workspace `.gitignore` notice: `.vscode/` (tasks.json / settings.json) is not ignored."
            ),
        }
    return {
        "safe": True,
        "gitignore_present": True,
        "vscode_ignored": True,
        "warning": "",
    }


def load_saved_credentials(path: Path, *, expected_gateway: str | None = None) -> dict[str, Any] | None:
    """Load `~/.hivemind/credentials.json` if readable and holding a valid `hm-...` API key."""
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


def _connect_or_register_account(
    *,
    gateway: str,
    credentials_path: Path,
    handle: str | None,
    role: str,
    explicit_api_key: str | None,
    env: Mapping[str, str] | None,
    http_fn: HttpFn,
    os_name: str | None,
    home: Path | None,
    lang: str,
) -> tuple[dict[str, Any], int]:
    is_fr = lang == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_role = normalize_role(role)
    env_view = env if env is not None else os.environ
    net_calls = 0

    env_key = str(env_view.get("HIVEMIND_API_KEY") or "").strip()
    candidate_key = (explicit_api_key or "").strip() or env_key
    if candidate_key and not is_valid_hivemind_key(candidate_key):
        raise QuickPresetError("Provided API key is invalid; expected an 'hm-...' or 'hm_live_...' key.")

    saved = load_saved_credentials(credentials_path, expected_gateway=clean_gw)
    if not candidate_key and saved is not None:
        candidate_key = str(saved["api_key"]).strip()

    created_account = False
    resolved_handle = (
        sanitize_handle(handle)
        if (handle and str(handle).strip())
        else str((saved or {}).get("handle") or "").strip() or sanitize_handle(None)
    )
    account_id = str((saved or {}).get("account_id") or "")

    if not candidate_key:
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
                        f"Échec d'inscription sur POST /v1/hivemind/register (HTTP {reg_status})."
                        if is_fr
                        else f"Account registration failed on POST /v1/hivemind/register (HTTP {reg_status})."
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
                "error": (
                    f"Échec de vérification de la clé {masked} sur GET /v1/hivemind/me (HTTP {me_status})."
                    if is_fr
                    else f"API key verification for {masked} failed on GET /v1/hivemind/me (HTTP {me_status})."
                ),
            },
            net_calls,
        )

    models_status, models_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/models")
    net_calls += 1
    raw_models = models_body.get("data")
    if models_status != 200 or not isinstance(raw_models, list):
        return (
            {
                "ok": False,
                "stage": "models",
                "http_status": models_status,
                "error": (
                    f"Échec de lecture du catalogue GET /v1/models (HTTP {models_status})."
                    if is_fr
                    else f"Model catalog check failed on GET /v1/models (HTTP {models_status})."
                ),
            },
            net_calls,
        )

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
            "models": model_ids,
            "models_count": len(model_ids),
            "credentials_write": cred_write,
        },
        net_calls,
    )


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
        return GpuProbeSnapshot(present=False, reason="no_nvidia_gpu")

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
            "total_vram_mib": getattr(raw, "total_vram_mib", getattr(raw, "total_mib", None)),
            "free_vram_mib": getattr(raw, "free_vram_mib", getattr(raw, "free_mib", None)),
            "used_vram_mib": getattr(raw, "used_vram_mib", getattr(raw, "used_mib", None)),
            "temp_c": getattr(raw, "temp_c", None),
            "util_pct": getattr(raw, "util_pct", None),
            "reason": getattr(raw, "reason", "ok"),
        }
    else:
        raise QuickPresetError("gpu_probe_fn must return GpuProbeSnapshot, a mapping, or None.")

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
    """List active local process names via `tasklist` (Windows) or `ps` (POSIX)."""
    eff_os = os_name if os_name is not None else os.name
    bounded = min(float(timeout_s), SUBPROCESS_TIMEOUT_S)
    names: list[str] = []
    if eff_os == "nt":
        rc, out = runner(["tasklist", "/FO", "CSV", "/NH"], bounded)
        if rc == 0 and out.strip():
            for row in csv.reader(io.StringIO(out)):
                if row and row[0].strip():
                    names.append(row[0].strip())
        return names
    rc, out = runner(["ps", "-eo", "comm="], bounded)
    if rc == 0 and out.strip():
        for ln in out.splitlines():
            item = Path(ln.strip()).name
            if item:
                names.append(item)
    return names


def _detect_active_steam_games(process_names: Iterable[str]) -> list[str]:
    found: list[str] = []
    for raw in process_names:
        base = Path(str(raw).strip().replace("\\", "/")).name.lower()
        if not base:
            continue
        if base in STEAM_DX12_GAME_EXECUTABLES or base.startswith(STEAM_PROCESS_PREFIXES):
            if base not in found:
                found.append(base)
    return found


def default_worker_yield_drill(*, timeout_s: float = 2.5) -> dict[str, Any]:
    """Measure local subprocess worker preemption release time (`shell=False`)."""
    bounded = min(max(0.2, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    proc = subprocess.Popen(
        [sys.executable, "-c", _LOCAL_WORKER_YIELD_PROGRAM],
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
            return {"elapsed_ms": None, "clean_release": False, "worker_stopped": False}
        t0 = time.perf_counter()
        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
        released_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        rc = proc.wait(timeout=bounded)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        clean_ok = rc == 0 and released_line == "HIVEMIND_WORKER_RELEASED:0"
        return {"elapsed_ms": elapsed_ms, "clean_release": clean_ok, "worker_stopped": clean_ok}
    except (OSError, subprocess.SubprocessError):
        proc.kill()
        try:
            proc.wait(timeout=0.5)
        except (OSError, subprocess.SubprocessError) as wait_err:
            _ = wait_err
        return {"elapsed_ms": None, "clean_release": False, "worker_stopped": False}


def _invoke_callable_flexible(fn: Callable[..., Any], **candidate_kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**candidate_kwargs)
    filtered = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _normalize_yield_drill_result(raw: Any) -> tuple[float | None, bool]:
    if isinstance(raw, bool):
        raise QuickPresetError("yield_drill_fn must return a millisecond float or mapping, not bool.")
    if isinstance(raw, (int, float)):
        val = float(raw)
        if not math.isfinite(val) or val < 0.0:
            return None, False
        return round(val, 2), True
    if isinstance(raw, Mapping):
        ms = _optional_finite_float(raw.get("elapsed_ms", raw.get("yield_ms")))
        clean = bool(raw.get("clean_release", True)) and bool(raw.get("worker_stopped", True))
        if ms is None or ms < 0.0:
            return None, False
        return round(ms, 2), clean
    raise QuickPresetError("yield_drill_fn must return a float/int milliseconds value or Mapping.")


def run_steam_yield_and_thermal_preflight(
    *,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    yield_drill_fn: YieldDrillFn = default_worker_yield_drill,
    process_probe_fn: ProcessProbeFn | None = default_process_probe,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    os_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify `<1.5s` worker preemption yield, `<=70 C` thermal ceiling, or no-GPU laptop dev fallback."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    raw_yield = _invoke_callable_flexible(yield_drill_fn, timeout_s=2.5)
    yield_ms, clean_release = _normalize_yield_drill_result(raw_yield)
    yield_ok = bool(clean_release and yield_ms is not None and yield_ms < float(yield_budget_ms))
    slow_yield_detected = not yield_ok

    gpu = _normalize_gpu_snapshot(_invoke_callable_flexible(gpu_probe_fn, timeout_s=SUBPROCESS_TIMEOUT_S))
    active_games: list[str] = []
    if process_probe_fn is not None:
        raw_procs = _invoke_callable_flexible(process_probe_fn, os_name=os_name, timeout_s=SUBPROCESS_TIMEOUT_S)
        active_games = _detect_active_steam_games(raw_procs)
    active_game_detected = len(active_games) > 0

    if not gpu.present:
        thermal_ok = True
        hot_gpu_detected = False
        vram_headroom_ok = True
        mode = "dev_api_only_no_gpu"
    elif gpu.parse_error:
        thermal_ok = False
        hot_gpu_detected = False
        vram_headroom_ok = False
        mode = "gpu_telemetry_parse_error"
    else:
        hot_gpu_detected = gpu.temp_c is not None and float(gpu.temp_c) > float(max_temp_c)
        thermal_ok = not hot_gpu_detected
        vram_headroom_ok = (gpu.free_vram_mib or 0) >= int(min_free_vram_mib)
        mode = (
            "gamer_whisper_gpu_ready"
            if (thermal_ok and vram_headroom_ok and not active_game_detected)
            else "gamer_whisper_throttled_or_yielding"
        )

    preflight_ok = bool(yield_ok and thermal_ok and not gpu.parse_error and not active_game_detected)
    if preflight_ok:
        msg = (
            f"Pré-vol Gamer-Whisper validé (mode={mode})."
            if is_fr
            else f"Gamer-Whisper yield & thermal preflight passed (mode={mode})."
        )
    else:
        msg = (
            f"Pré-vol Gamer-Whisper bloqué (yield_ok={yield_ok}, thermal_ok={thermal_ok}, jeu_actif={active_game_detected})."
            if is_fr
            else f"Gamer-Whisper preflight blocked (yield_ok={yield_ok}, thermal_ok={thermal_ok}, active_game={active_game_detected})."
        )

    return {
        "preflight_ok": preflight_ok,
        "mode": mode,
        "yield_ok": yield_ok,
        "yield_ms": yield_ms,
        "yield_budget_ms": float(yield_budget_ms),
        "slow_yield_detected": slow_yield_detected,
        "thermal_ok": thermal_ok,
        "temp_c": gpu.temp_c,
        "max_temp_c": float(max_temp_c),
        "hot_gpu_detected": hot_gpu_detected,
        "gpu_present": gpu.present,
        "gpu_parse_error": gpu.parse_error,
        "gpu_name": gpu.name if (gpu.present and not gpu.parse_error) else None,
        "vram_total_mib": gpu.total_vram_mib if (gpu.present and not gpu.parse_error) else None,
        "vram_free_mib": gpu.free_vram_mib if (gpu.present and not gpu.parse_error) else None,
        "vram_headroom_ok": vram_headroom_ok,
        "active_game_detected": active_game_detected,
        "active_games": active_games,
        "message": msg,
    }


def send_opentrainer_first_heartbeat(
    *,
    gateway: str,
    api_key: str,
    role: str,
    host_label: str | None = None,
    gpu_snapshot: GpuProbeSnapshot | Mapping[str, Any] | None = None,
    http_fn: HttpFn = default_http_request,
    home: Path | None = None,
    lang: str = "en",
) -> tuple[dict[str, Any], int]:
    """Send outbound `POST /v1/hivemind/opentrainer/heartbeat` (`extra='forbid'`, never inventing GPU specs)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_host = sanitize_host_label(host_label, home=home)

    if role == "consumer":
        return (
            {
                "attempted": False,
                "sent": False,
                "ok": True,
                "skipped_consumer_role": True,
                "http_status": None,
                "network_mode": "outbound-https-relay",
                "payload": None,
                "message": (
                    "Rôle consumer : heartbeat GPU OpenTrainer ignoré (mode API développeur uniquement)."
                    if is_fr
                    else "Consumer role: OpenTrainer GPU heartbeat skipped (developer API-only mode)."
                ),
            },
            0,
        )

    gpu = _normalize_gpu_snapshot(gpu_snapshot)
    reportable_gpu = (
        gpu.present
        and not gpu.parse_error
        and bool(gpu.name)
        and (gpu.total_vram_mib or 0) >= MIN_REPORTABLE_VRAM_TOTAL_MIB
    )
    gpu_name: str | None = gpu.name if reportable_gpu else None
    vram_total: int | None = int(gpu.total_vram_mib) if (reportable_gpu and gpu.total_vram_mib is not None) else None
    vram_free: int | None = (
        min(max(0, int(gpu.free_vram_mib or 0)), vram_total)
        if vram_total is not None
        else None
    )

    hb_payload: dict[str, Any] = {
        "host_label": clean_host,
        "gpu_name": gpu_name,
        "vram_total_mib": vram_total,
        "vram_free_mib": vram_free,
        "network_mode": "outbound-https-relay",
        "training_stack": DEFAULT_TRAINING_STACK,
        "active_campaign": DEFAULT_ACTIVE_CAMPAIGN,
    }
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
            "sent": ok,
            "ok": ok,
            "skipped_consumer_role": False,
            "http_status": status_code,
            "network_mode": "outbound-https-relay",
            "node_id": sanitize_text_for_display(str(node_obj.get("node_id") or ""), home=home) or None,
            "payload": hb_payload,
            "message": (
                "Premier heartbeat OpenTrainer enregistré via relais HTTPS sortant."
                if (ok and is_fr)
                else (
                    "First OpenTrainer heartbeat registered over outbound HTTPS relay."
                    if ok
                    else (
                        f"Échec du heartbeat OpenTrainer (HTTP {status_code})."
                        if is_fr
                        else f"OpenTrainer heartbeat failed (HTTP {status_code})."
                    )
                )
            ),
        },
        1,
    )


def _platform_tag(system_name: str | None = None) -> str:
    raw = (system_name if system_name is not None else platform.system()).strip().lower()
    if raw in ("windows", "win32", "nt"):
        return "windows"
    if raw == "darwin":
        return "macos"
    if raw == "linux":
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "", raw)[:24].strip("_-")
    if cleaned and _PLATFORM_RE.fullmatch(cleaned):
        return cleaned
    return "unknown"


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
        f"noel-vscode-quick-preset-ces/v1|{gateway.rstrip('/')}|{account_ref}|"
        f"{profile}|{ease_rating}|{1 if zero_game_lag else 0}|{platform_tag}"
    )
    return "ces-noel-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def submit_preset_ces_feedback(
    *,
    gateway: str,
    account_ref: str = "anon",
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    system_name: str | None = None,
    http_fn: HttpFn = default_http_request,
    lang: str = "en",
) -> tuple[dict[str, Any], int]:
    """Send timeout-safe non-blocking zero-PII `POST /v1/hivemind/onboarding/feedback` (`extra='forbid'`)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_profile = normalize_profile(profile)
    clean_ease = parse_ease_rating(ease_rating)
    if not isinstance(zero_game_lag, bool):
        raise QuickPresetError("zero_game_lag must be a boolean.")
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
    if ces_recorded:
        outcome = "recorded"
        msg = (
            f"Retour CES enregistré (profil={norm_profile}, facilité={clean_ease}/5)."
            if is_fr
            else f"Zero-PII CES feedback recorded (profile={norm_profile}, ease_rating={clean_ease}/5)."
        )
    elif status_code == 404:
        outcome = "stale_cloud_run_404"
        msg = (
            "Route CES absente sur Cloud Run (HTTP 404, révision pré-PR-#949) ; repli gracieux non bloquant."
            if is_fr
            else "CES route missing on stale Cloud Run (HTTP 404, pre-PR-#949); degraded gracefully."
        )
    elif status_code == 429:
        outcome = "rate_limited"
        msg = (
            "Retour CES limité par débit (HTTP 429) ; repli gracieux non bloquant."
            if is_fr
            else "CES feedback rate-limited (HTTP 429); degraded gracefully."
        )
    elif status_code >= 500:
        outcome = "gateway_5xx"
        msg = (
            f"Erreur serveur passerelle sur retour CES (HTTP {status_code}) ; repli gracieux non bloquant."
            if is_fr
            else f"Gateway server error on CES feedback (HTTP {status_code}); degraded gracefully."
        )
    elif status_code == 0:
        outcome = "network_timeout"
        msg = (
            "Délai réseau dépassé sur retour CES ; repli gracieux non bloquant."
            if is_fr
            else "Network timeout reaching CES feedback endpoint; degraded gracefully."
        )
    else:
        outcome = f"http_{status_code}"
        msg = (
            f"Retour CES non enregistré (HTTP {status_code})."
            if is_fr
            else f"CES feedback not recorded (HTTP {status_code})."
        )

    raw_total = body.get("total_responses")
    raw_mean = body.get("mean_ease_rating")
    return (
        {
            "attempted": True,
            "ces_recorded": ces_recorded,
            "status": outcome,
            "http_status": status_code,
            "idempotency_key": idem_key,
            "profile": norm_profile,
            "ease_rating": clean_ease,
            "zero_game_lag": zero_game_lag,
            "total_responses": int(raw_total) if (not isinstance(raw_total, bool) and isinstance(raw_total, int)) else None,
            "mean_ease_rating": (
                round(float(raw_mean), 2)
                if (not isinstance(raw_mean, bool) and isinstance(raw_mean, (int, float)))
                else None
            ),
            "payload": ces_payload,
            "message": msg,
        },
        1,
    )


def _get_ces_path_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _CES_LOCK_GUARD:
        lock = _CES_PATH_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _CES_PATH_LOCKS[resolved] = lock
        return lock


def default_ces_queue_path(
    *,
    home: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the default `0600` CES retry spool path (`~/.hivemind/vscode_setup_ces_queue.json`)."""
    env_view = env if env is not None else os.environ
    resolved_home = (
        Path(home)
        if home is not None
        else Path(env_view.get("HOME") or env_view.get("USERPROFILE") or Path.home())
    )
    return resolved_home / DEFAULT_CES_QUEUE_REL


def _validate_ces_spool_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise QuickPresetError("CES feedback payload must be a JSON object.")
    allowed_keys = {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
    extra_keys = set(raw.keys()) - allowed_keys
    if extra_keys:
        raise QuickPresetError(f"CES feedback payload contains unsupported keys: {sorted(extra_keys)}")
    prof = normalize_profile(raw.get("profile"))
    ease = parse_ease_rating(raw.get("ease_rating"))  # type: ignore[arg-type]
    zero_lag = raw.get("zero_game_lag")
    if not isinstance(zero_lag, bool):
        raise QuickPresetError("CES feedback zero_game_lag must be a boolean.")
    plat_raw = str(raw.get("platform") or "unknown").strip().lower()
    if not _PLATFORM_RE.fullmatch(plat_raw):
        raise QuickPresetError("CES feedback platform contains unsupported characters.")
    note_raw = raw.get("note", "")
    if not isinstance(note_raw, str) or note_raw != "":
        raise QuickPresetError("CES feedback note must remain empty to guarantee zero PII.")
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


class VscodeCesRetrySpool:
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
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not (1 <= max_entries <= MAX_CES_QUEUE_ENTRIES_CEILING)
        ):
            raise QuickPresetError(
                f"max_entries must be an integer in 1..{MAX_CES_QUEUE_ENTRIES_CEILING}."
            )
        if (
            isinstance(max_delivered, bool)
            or not isinstance(max_delivered, int)
            or not (1 <= max_delivered <= 16384)
        ):
            raise QuickPresetError("max_delivered must be an integer in 1..16384.")
        env_view = env if env is not None else os.environ
        resolved_home = (
            Path(home)
            if home is not None
            else Path(env_view.get("HOME") or env_view.get("USERPROFILE") or Path.home())
        )
        self._home = resolved_home
        self._eff_os = os_name if os_name is not None else os.name
        self._max_entries = int(max_entries)
        self._max_delivered = int(max_delivered)

        if queue_file is not None and str(queue_file).strip():
            raw_str = str(queue_file).strip()
            if raw_str == "~" or raw_str.startswith("~/"):
                suffix = raw_str[2:] if raw_str.startswith("~/") else ""
                raw_path = (resolved_home / suffix) if suffix else resolved_home
            else:
                raw_path = Path(raw_str)
        else:
            raw_path = default_ces_queue_path(home=resolved_home, env=env_view)

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
                    "event_id": hashlib.sha256(b"corrupted_vscode_ces_spool").hexdigest(),
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
                            "http_status": (
                                int(q_status)
                                if not isinstance(q_status, bool) and isinstance(q_status, int)
                                else 422
                            ),
                            "reason": sanitize_text_for_display(
                                str(q_item.get("reason") or "quarantined")[:80],
                                home=self._home,
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
                except QuickPresetError:
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
                        "idempotency_key": idem_raw or f"ces-noel-{eid[:32]}",
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
        dropped_count = (
            max(0, int(raw_dropped))
            if not isinstance(raw_dropped, bool) and isinstance(raw_dropped, int)
            else 0
        )
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
        serialized = _assert_safe_output(
            json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        if self._eff_os != "nt" and not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self._path.parent, 0o700)
        _write_file_secure_atomic(
            self._path, serialized, posix_mode=0o600, eff_os=self._eff_os
        )

    def snapshot(self, *, lang: str = "en") -> dict[str, Any]:
        """Return a sanitized read-only snapshot of the retry spool without creating any file."""
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
        profile: str = "dev",
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
        """Validate compliance & zero-PII schema, deduplicate, cap pending at `max_entries`, and save `0600`."""
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
                raise QuickPresetError("idempotency_key must be a string.")
            idem_key = idempotency_key.strip()
            if not idem_key or len(idem_key) > 128 or not idem_key.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in idem_key):
                raise QuickPresetError("idempotency_key must be 1..128 printable ASCII characters.")
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
        """Idempotently flush pending CES events with `Idempotency-Key` headers (no file created if empty)."""
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
                    "flushed_count": 0,
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
                    "diagnostics": [comp_doc["summary"]],
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
                        "flushed_count": 0,
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
                        "diagnostics": [],
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
                        {
                            "event_id": eid,
                            "idempotency_key": idem_key,
                            "disposition": "delivered",
                            "http_status": code,
                        }
                    )
                elif disposition == "retryable":
                    if code == 404:
                        stale_404_seen = True
                    new_rc = int(entry.get("retry_count", 0)) + 1
                    next_pending.append(
                        {
                            **entry,
                            "retry_count": new_rc,
                            "last_http_status": code,
                            "last_reason": reason,
                        }
                    )
                    retained_retry += 1
                    outcomes.append(
                        {
                            "event_id": eid,
                            "idempotency_key": idem_key,
                            "disposition": "retained_for_retry",
                            "http_status": code,
                            "reason": reason,
                            "retry_count": new_rc,
                        }
                    )
                else:
                    state["quarantined"].append(
                        {
                            "event_id": eid,
                            "idempotency_key": idem_key,
                            "http_status": code,
                            "reason": reason,
                        }
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

        diagnostics: list[str] = []
        if stale_404_seen:
            diagnostics.append(
                "HTTP 404 sur /v1/hivemind/onboarding/feedback : retours CES conservés dans la file 0600."
                if is_fr
                else "HTTP 404 on /v1/hivemind/onboarding/feedback: CES items retained safely in 0600 spool."
            )
        if quarantined_now > 0:
            diagnostics.append(
                f"{quarantined_now} entrée(s) permanente(s) 400/422 mise(s) en quarantaine."
                if is_fr
                else f"{quarantined_now} permanent 400/422 entry/entries moved to quarantine."
            )

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
                "flushed_count": delivered_now,
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
                "diagnostics": diagnostics,
                "compliance": comp_doc,
                "summary": summary,
            }
        )

    def purge(self, *, lang: str = "en") -> dict[str, Any]:
        """Remove local CES retry spool and any temporary files (always permitted, 0 HTTP calls)."""
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


def enqueue_vscode_ces_event(
    *,
    queue_file: Path | str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    account_ref: str = "anon",
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    system_name: str | None = None,
    platform: str | None = None,
    idempotency_key: str | None = None,
    last_http_status: int | None = None,
    last_reason: str = "",
    max_entries: int = DEFAULT_MAX_CES_QUEUE_ENTRIES,
    max_delivered: int = DEFAULT_MAX_CES_DELIVERED,
    home: Path | str | None = None,
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    os_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Convenience wrapper around `VscodeCesRetrySpool.enqueue(...)`."""
    spool = VscodeCesRetrySpool(
        queue_file,
        max_entries=max_entries,
        max_delivered=max_delivered,
        home=home,
        env=env,
        os_name=os_name,
    )
    return spool.enqueue(
        gateway=gateway,
        account_ref=account_ref,
        profile=profile,
        ease_rating=ease_rating,
        zero_game_lag=zero_game_lag,
        system_name=system_name,
        platform=platform,
        idempotency_key=idempotency_key,
        last_http_status=last_http_status,
        last_reason=last_reason,
        enforce_personal_pc=enforce_personal_pc,
        env=env,
        cwd=cwd,
        lang=lang,
    )


def flush_vscode_ces_queue(
    gateway: str = DEFAULT_GATEWAY,
    *,
    queue_file: Path | str | None = None,
    max_entries: int = DEFAULT_MAX_CES_QUEUE_ENTRIES,
    max_delivered: int = DEFAULT_MAX_CES_DELIVERED,
    home: Path | str | None = None,
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    os_name: str | None = None,
    http_fn: HttpFn = default_http_request,
    lang: str = "en",
) -> dict[str, Any]:
    """Convenience wrapper around `VscodeCesRetrySpool.flush(...)`."""
    spool = VscodeCesRetrySpool(
        queue_file,
        max_entries=max_entries,
        max_delivered=max_delivered,
        home=home,
        env=env,
        os_name=os_name,
    )
    return spool.flush(
        gateway,
        http_fn=http_fn,
        enforce_personal_pc=enforce_personal_pc,
        env=env,
        cwd=cwd,
        lang=lang,
    )


def parse_ces_ease_input(
    raw: str,
    *,
    default_profile: str = "dev",
    default_had_game_lag: bool = False,
) -> dict[str, Any]:
    """Parse 1-prompt zero-PII CES input (`'5'`, `'4/5 nolag'`, `'3 gamer lag'`, or empty/skip)."""
    norm_default_profile = normalize_profile(default_profile)
    cleaned = (raw or "").strip()
    if not cleaned or cleaned.lower() in ("skip", "s", "q", "quit", "no", "non", "pass"):
        return {
            "skipped": True,
            "profile": norm_default_profile,
            "ease_rating": None,
            "had_game_lag": bool(default_had_game_lag),
            "zero_game_lag": not bool(default_had_game_lag),
        }

    normalized = re.sub(r"\s*/\s*5\b", "", cleaned, count=1)
    tokens = [tok.strip(",;:").lower() for tok in normalized.split() if tok.strip(",;:")]
    rating: int | None = None
    resolved_profile = norm_default_profile
    had_lag = bool(default_had_game_lag)

    for tok in tokens:
        if tok.lstrip("+-").isdigit():
            val = int(tok)
            if not (1 <= val <= 5) or rating is not None:
                raise QuickPresetError("CES ease_rating must be a single integer from 1 to 5.")
            rating = val
        elif tok in ("dev", "developer", "d"):
            resolved_profile = "dev"
        elif tok in ("gamer", "gaming", "g"):
            resolved_profile = "gamer"
        elif tok in ("lag", "yes-lag", "had-lag", "had-game-lag", "had_game_lag", "stutter", "saccade"):
            had_lag = True
        elif tok in ("nolag", "no-lag", "no_lag", "zero-lag", "zero_lag", "smooth", "ok", "fluide"):
            had_lag = False
        else:
            raise QuickPresetError(
                f"Unrecognized CES token {tok!r}; expected 1..5 [dev|gamer] [lag|nolag] or Enter to skip."
            )

    if rating is None:
        raise QuickPresetError("Interactive CES input must include an integer ease_rating from 1 to 5.")

    return {
        "skipped": False,
        "profile": resolved_profile,
        "ease_rating": rating,
        "had_game_lag": had_lag,
        "zero_game_lag": not had_lag,
    }


def prompt_and_submit_ces(
    *,
    gateway: str = DEFAULT_GATEWAY,
    account_ref: str = "anon",
    profile: str = "dev",
    default_had_game_lag: bool = False,
    system_name: str | None = None,
    input_fn: Callable[[str], str] | None = None,
    http_fn: HttpFn = default_http_request,
    ces_queue_file: Path | str | None = None,
    home: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    os_name: str | None = None,
    spool_on_failure: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Prompt once for zero-PII 1..5 ease rating (+ optional lag), submit, and spool on transient failure."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_profile = normalize_profile(profile)
    eff_input = input_fn if input_fn is not None else input

    prompt_text = (
        f"[HiveMind CES 1-question | profil={norm_profile}] Notez la simplicité 1..5 "
        "(optionnel : 'lag' ou 'nolag' ; Entrée pour ignorer) : "
        if is_fr
        else f"[HiveMind 1-prompt CES | profile={norm_profile}] Rate setup ease 1..5 "
        "(optional: 'lag' or 'nolag'; Enter to skip): "
    )
    try:
        raw_answer = eff_input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        raw_answer = ""

    parsed = parse_ces_ease_input(
        raw_answer,
        default_profile=norm_profile,
        default_had_game_lag=default_had_game_lag,
    )
    if parsed["skipped"]:
        return {
            "attempted": False,
            "skipped": True,
            "ces_recorded": False,
            "queued_for_retry": False,
            "status": "skipped",
            "http_status": None,
            "idempotency_key": None,
            "profile": parsed["profile"],
            "ease_rating": None,
            "zero_game_lag": parsed["zero_game_lag"],
            "total_responses": None,
            "mean_ease_rating": None,
            "queue_file": None,
            "network_calls": 0,
            "message": (
                "Retour CES 1-question ignoré (aucun envoi réseau)."
                if is_fr
                else "1-prompt CES skipped (no network call)."
            ),
        }

    ces_rep, net_calls = submit_preset_ces_feedback(
        gateway=clean_gw,
        account_ref=account_ref,
        profile=str(parsed["profile"]),
        ease_rating=int(parsed["ease_rating"]),
        zero_game_lag=bool(parsed["zero_game_lag"]),
        system_name=system_name,
        http_fn=http_fn,
        lang=lang_code,
    )
    queued_for_retry = False
    queue_redacted: str | None = None
    http_st = int(ces_rep["http_status"])
    if (
        not ces_rep["ces_recorded"]
        and spool_on_failure
        and (http_st in (0, 404, 408, 429) or http_st >= 500)
    ):
        enq = enqueue_vscode_ces_event(
            queue_file=ces_queue_file,
            gateway=clean_gw,
            account_ref=account_ref,
            profile=str(parsed["profile"]),
            ease_rating=int(parsed["ease_rating"]),
            zero_game_lag=bool(parsed["zero_game_lag"]),
            system_name=system_name,
            idempotency_key=str(ces_rep["idempotency_key"]),
            last_http_status=http_st,
            last_reason=str(ces_rep["status"]),
            home=home,
            enforce_personal_pc=False,
            env=env,
            cwd=cwd,
            os_name=os_name,
            lang=lang_code,
        )
        queued_for_retry = bool(enq.get("enqueued") or enq.get("deduplicated"))
        queue_redacted = str(enq.get("queue_file") or "") or None

    return {
        **ces_rep,
        "skipped": False,
        "queued_for_retry": queued_for_retry,
        "queue_file": queue_redacted,
        "network_calls": net_calls,
    }


def install_vscode_quick_preset(
    gateway: str = DEFAULT_GATEWAY,
    *,
    workspace_dir: Path | str | None = None,
    credentials_path: Path | str | None = None,
    continue_config_path: Path | str | None = None,
    handle: str | None = DEFAULT_HANDLE,
    role: str = DEFAULT_ROLE,
    explicit_api_key: str | None = None,
    host_label: str | None = None,
    expected_sha: str | None = None,
    send_first_heartbeat: bool = False,
    submit_ces: bool = False,
    profile: str = "dev",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    require_holiday_routes: bool = False,
    run_gamer_preflight: bool = False,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    yield_drill_fn: YieldDrillFn = default_worker_yield_drill,
    process_probe_fn: ProcessProbeFn | None = default_process_probe,
    http_fn: HttpFn = default_http_request,
    env: Mapping[str, str] | None = None,
    home: Path | str | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Install the <60s VS Code one-click quick-connect preset with optional heartbeat & non-blocking CES."""
    t0 = time.perf_counter()
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    norm_profile = normalize_profile(profile)
    norm_role = normalize_role(role)
    clean_ease = parse_ease_rating(ease_rating)
    clean_gw = validate_gateway_url(gateway)

    env_view = env if env is not None else os.environ
    eff_os = os_name if os_name is not None else os.name
    resolved_home = (
        Path(home)
        if home is not None
        else Path(env_view.get("HOME") or env_view.get("USERPROFILE") or Path.home())
    )
    ws_path = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
    cred_path = (
        Path(credentials_path)
        if credentials_path is not None
        else (resolved_home / ".hivemind" / "credentials.json")
    )
    cont_path = (
        Path(continue_config_path)
        if continue_config_path is not None
        else (resolved_home / ".continue" / "hivemind_provider.yaml")
    )

    compliance = evaluate_personal_pc_compliance(env=env_view, cwd=ws_path, lang=lang_code)
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "install_preset",
                "ok": False,
                "ready": False,
                "compliance_ok": False,
                "exit_code": 2,
                "verdict": "blocked_corporate_environment",
                "gateway": clean_gw,
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "within_60s_slo": True,
                "account": None,
                "files": {},
                "file_writes": 0,
                "network_calls": 0,
                "task_labels": [],
                "gitignore_check": None,
                "holiday_routes_deployed_on_cloud_run": None,
                "onboarding_feedback_deployed": None,
                "stale_revision_detected": False,
                "missing_or_stale_routes": [],
                "online_producers": None,
                "parity": None,
                "preflight": None,
                "first_heartbeat_ok": None,
                "heartbeat": {"attempted": False, "sent": False, "ok": False},
                "ces_recorded": False,
                "ces_feedback": {"attempted": False, "ces_recorded": False},
                "zero_inference_verified": True,
                "compliance": compliance,
                "summary": compliance["summary"],
            }
        )

    network_calls = 0
    parity_report: dict[str, Any] | None = None
    holiday_deployed: bool | None = None
    feedback_deployed: bool | None = None
    stale_revision = False
    missing_routes: list[str] = []
    online_producers: int | None = None

    if require_holiday_routes or expected_sha is not None:
        parity_report = verify_cloud_run_holiday_parity(
            clean_gw,
            expected_sha=expected_sha,
            http_fn=http_fn,
            env=env_view,
            cwd=ws_path,
            lang=lang_code,
        )
        network_calls += int(parity_report["network_calls"])
        holiday_deployed = bool(parity_report["holiday_routes_deployed_on_cloud_run"])
        feedback_deployed = bool(parity_report["onboarding_feedback_deployed"])
        stale_revision = bool(parity_report["stale_revision_detected"])
        missing_routes = list(parity_report["missing_or_stale_routes"])
        online_producers = int(parity_report["online_producers"])

        if not parity_report["parity_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "mode": "install_preset",
                    "ok": False,
                    "ready": False,
                    "compliance_ok": True,
                    "exit_code": 2,
                    "verdict": "stale_cloud_run_revision",
                    "gateway": clean_gw,
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                    "within_60s_slo": (time.perf_counter() - t0) < SLO_BUDGET_SECONDS,
                    "account": None,
                    "files": {},
                    "file_writes": 0,
                    "network_calls": network_calls,
                    "task_labels": [],
                    "gitignore_check": None,
                    "holiday_routes_deployed_on_cloud_run": holiday_deployed,
                    "onboarding_feedback_deployed": feedback_deployed,
                    "stale_revision_detected": stale_revision,
                    "missing_or_stale_routes": missing_routes,
                    "online_producers": online_producers,
                    "parity": parity_report,
                    "preflight": None,
                    "first_heartbeat_ok": None,
                    "heartbeat": {"attempted": False, "sent": False, "ok": False},
                    "ces_recorded": False,
                    "ces_feedback": {"attempted": False, "ces_recorded": False},
                    "zero_inference_verified": True,
                    "compliance": compliance,
                    "summary": parity_report["summary"],
                }
            )

    preflight_report: dict[str, Any] | None = None
    need_preflight = bool(send_first_heartbeat or run_gamer_preflight or norm_profile == "gamer")
    if need_preflight:
        preflight_report = run_steam_yield_and_thermal_preflight(
            gpu_probe_fn=gpu_probe_fn,
            yield_drill_fn=yield_drill_fn,
            process_probe_fn=process_probe_fn,
            max_temp_c=max_temp_c,
            yield_budget_ms=yield_budget_ms,
            os_name=eff_os,
            lang=lang_code,
        )
        if not preflight_report["preflight_ok"]:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "mode": "install_preset",
                    "ok": False,
                    "ready": False,
                    "compliance_ok": True,
                    "exit_code": 2,
                    "verdict": "preflight_failed",
                    "gateway": clean_gw,
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                    "within_60s_slo": (time.perf_counter() - t0) < SLO_BUDGET_SECONDS,
                    "account": None,
                    "files": {},
                    "file_writes": 0,
                    "network_calls": network_calls,
                    "task_labels": [],
                    "gitignore_check": None,
                    "holiday_routes_deployed_on_cloud_run": holiday_deployed,
                    "onboarding_feedback_deployed": feedback_deployed,
                    "stale_revision_detected": stale_revision,
                    "missing_or_stale_routes": missing_routes,
                    "online_producers": online_producers,
                    "parity": parity_report,
                    "preflight": preflight_report,
                    "first_heartbeat_ok": False if send_first_heartbeat else None,
                    "heartbeat": {
                        "attempted": False,
                        "sent": False,
                        "ok": False,
                        "reason": "blocked_by_preflight_guard",
                    },
                    "ces_recorded": False,
                    "ces_feedback": {"attempted": False, "ces_recorded": False},
                    "zero_inference_verified": True,
                    "compliance": compliance,
                    "summary": preflight_report["message"],
                }
            )

    acct_res, acct_calls = _connect_or_register_account(
        gateway=clean_gw,
        credentials_path=cred_path,
        handle=handle,
        role=norm_role,
        explicit_api_key=explicit_api_key,
        env=env_view,
        http_fn=http_fn,
        os_name=eff_os,
        home=resolved_home,
        lang=lang_code,
    )
    network_calls += acct_calls
    if not acct_res["ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "install_preset",
                "ok": False,
                "ready": False,
                "compliance_ok": True,
                "exit_code": 2,
                "verdict": "auth_or_gateway_failed",
                "gateway": clean_gw,
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "within_60s_slo": (time.perf_counter() - t0) < SLO_BUDGET_SECONDS,
                "account": None,
                "files": {},
                "file_writes": 0,
                "network_calls": network_calls,
                "task_labels": [],
                "gitignore_check": None,
                "holiday_routes_deployed_on_cloud_run": holiday_deployed,
                "onboarding_feedback_deployed": feedback_deployed,
                "stale_revision_detected": stale_revision,
                "missing_or_stale_routes": missing_routes,
                "online_producers": online_producers,
                "parity": parity_report,
                "preflight": preflight_report,
                "first_heartbeat_ok": False if send_first_heartbeat else None,
                "heartbeat": {"attempted": False, "sent": False, "ok": False},
                "ces_recorded": False,
                "ces_feedback": {"attempted": False, "ces_recorded": False},
                "zero_inference_verified": True,
                "compliance": compliance,
                "summary": str(acct_res["error"]),
            }
        )

    raw_api_key = str(acct_res["api_key"])
    continue_yaml = build_continue_provider_yaml(clean_gw, raw_api_key)
    continue_write = atomic_write_config_file(
        cont_path,
        continue_yaml,
        posix_mode=0o600,
        create_backup=True,
        os_name=eff_os,
        home=resolved_home,
    )

    vscode_dir = ws_path / ".vscode"
    tasks_path = vscode_dir / "tasks.json"
    settings_path = vscode_dir / "settings.json"

    existing_tasks = tasks_path.read_text(encoding="utf-8") if tasks_path.is_file() else None
    merged_tasks = build_merged_vscode_tasks_json(existing_tasks, gateway=clean_gw, os_name=eff_os)
    tasks_write = atomic_write_config_file(
        tasks_path,
        merged_tasks,
        posix_mode=0o600,
        create_backup=True,
        os_name=eff_os,
        home=resolved_home,
    )

    existing_settings = settings_path.read_text(encoding="utf-8") if settings_path.is_file() else None
    merged_settings = build_merged_vscode_settings_json(existing_settings, gateway=clean_gw)
    settings_write = atomic_write_config_file(
        settings_path,
        merged_settings,
        posix_mode=0o600,
        create_backup=True,
        os_name=eff_os,
        home=resolved_home,
    )

    gitignore_info = check_workspace_gitignore_safety(ws_path, lang=lang_code)
    files_map = {
        "credentials": acct_res["credentials_write"],
        "continue_provider": continue_write,
        "vscode_tasks": tasks_write,
        "vscode_settings": settings_write,
    }
    file_writes = sum(1 for item in files_map.values() if bool(item.get("written")))

    first_hb_ok: bool | None = None
    hb_report: dict[str, Any] = {"attempted": False, "sent": False, "ok": True}
    if send_first_heartbeat:
        gpu_snap_for_hb = (
            {
                "present": preflight_report["gpu_present"],
                "parse_error": preflight_report["gpu_parse_error"],
                "name": preflight_report["gpu_name"],
                "total_vram_mib": preflight_report["vram_total_mib"],
                "free_vram_mib": preflight_report["vram_free_mib"],
                "temp_c": preflight_report["temp_c"],
            }
            if preflight_report is not None
            else None
        )
        hb_report, hb_calls = send_opentrainer_first_heartbeat(
            gateway=clean_gw,
            api_key=raw_api_key,
            role=str(acct_res["role"]),
            host_label=host_label or str(env_view.get("HOSTNAME") or DEFAULT_HOST_LABEL),
            gpu_snapshot=gpu_snap_for_hb,
            http_fn=http_fn,
            home=resolved_home,
            lang=lang_code,
        )
        network_calls += hb_calls
        first_hb_ok = bool(hb_report["ok"])

    ces_recorded = False
    ces_report: dict[str, Any] = {"attempted": False, "ces_recorded": False, "status": "not_requested"}
    zero_game_lag = not bool(had_game_lag)
    if preflight_report is not None and not preflight_report["yield_ok"]:
        zero_game_lag = False
    if submit_ces:
        account_ref = str(acct_res["account_id"] or acct_res["masked_key"])
        ces_report, ces_calls = submit_preset_ces_feedback(
            gateway=clean_gw,
            account_ref=account_ref,
            profile=norm_profile,
            ease_rating=clean_ease,
            zero_game_lag=zero_game_lag,
            system_name=system_name,
            http_fn=http_fn,
            lang=lang_code,
        )
        network_calls += ces_calls
        ces_recorded = bool(ces_report["ces_recorded"])
        if ces_report["http_status"] == 404:
            stale_revision = True
            holiday_deployed = False
            feedback_deployed = False
            if "/v1/hivemind/onboarding/feedback" not in missing_routes:
                missing_routes.append("/v1/hivemind/onboarding/feedback")

    ces_gate_failed = bool(submit_ces and require_holiday_routes and not ces_recorded)
    hb_gate_failed = bool(send_first_heartbeat and first_hb_ok is False)
    overall_ready = not (ces_gate_failed or hb_gate_failed)
    verdict = (
        "ready"
        if overall_ready
        else ("stale_cloud_run_revision" if ces_gate_failed else "heartbeat_failed")
    )
    elapsed_s = round(time.perf_counter() - t0, 4)

    account_public = {
        "account_id": acct_res["account_id"],
        "masked_key": acct_res["masked_key"],
        "role": acct_res["role"],
        "created_account": acct_res["created_account"],
        "models": acct_res["models"],
        "models_count": acct_res["models_count"],
    }
    summary = (
        (
            f"Preset VS Code 1-clic installé en {elapsed_s:.2f}s (clé={acct_res['masked_key']}, "
            f"CES={ces_recorded}, conformité={FINANCIAL_PROMISE})."
            if is_fr
            else f"VS Code 1-click quick preset installed in {elapsed_s:.2f}s (key={acct_res['masked_key']}, "
            f"ces_recorded={ces_recorded}, {FINANCIAL_PROMISE})."
        )
        if overall_ready
        else (
            f"Preset VS Code incomplet ({verdict})."
            if is_fr
            else f"VS Code preset check failed ({verdict})."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "mode": "install_preset",
            "ok": overall_ready,
            "ready": overall_ready,
            "compliance_ok": True,
            "exit_code": 0 if overall_ready else 2,
            "verdict": verdict,
            "gateway": clean_gw,
            "elapsed_s": elapsed_s,
            "within_60s_slo": elapsed_s < SLO_BUDGET_SECONDS,
            "account": account_public,
            "files": files_map,
            "file_writes": file_writes,
            "network_calls": network_calls,
            "task_labels": list(PRESET_TASK_LABELS),
            "gitignore_check": gitignore_info,
            "holiday_routes_deployed_on_cloud_run": holiday_deployed,
            "onboarding_feedback_deployed": feedback_deployed,
            "stale_revision_detected": stale_revision,
            "missing_or_stale_routes": missing_routes,
            "online_producers": online_producers,
            "parity": parity_report,
            "preflight": preflight_report,
            "first_heartbeat_ok": first_hb_ok,
            "heartbeat": hb_report,
            "ces_recorded": ces_recorded,
            "ces_feedback": ces_report,
            "zero_inference_verified": True,
            "compliance": compliance,
            "summary": summary,
        }
    )


def uninstall_vscode_quick_preset(
    *,
    workspace_dir: Path | str | None = None,
    credentials_path: Path | str | None = None,
    continue_config_path: Path | str | None = None,
    continue_main_config_path: Path | str | None = None,
    ces_queue_file: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | str | None = None,
    os_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Cleanly remove HiveMind preset files & tasks while leaving pre-existing user tasks/settings intact."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    env_view = env if env is not None else os.environ
    resolved_home = (
        Path(home)
        if home is not None
        else Path(env_view.get("HOME") or env_view.get("USERPROFILE") or Path.home())
    )
    ws_path = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
    cred_path = (
        Path(credentials_path)
        if credentials_path is not None
        else (resolved_home / ".hivemind" / "credentials.json")
    )
    cont_path = (
        Path(continue_config_path)
        if continue_config_path is not None
        else (resolved_home / ".continue" / "hivemind_provider.yaml")
    )
    cont_main_path = (
        Path(continue_main_config_path)
        if continue_main_config_path is not None
        else (resolved_home / ".continue" / "config.yaml")
    )
    queue_path = (
        Path(ces_queue_file)
        if ces_queue_file is not None
        else default_ces_queue_path(home=resolved_home, env=env_view)
    )

    vscode_dir = ws_path / ".vscode"
    tasks_path = vscode_dir / "tasks.json"
    settings_path = vscode_dir / "settings.json"

    removed_files: list[str] = []
    preserved_files: list[str] = []

    for managed in (
        cred_path,
        Path(f"{cred_path}.bak"),
        cont_path,
        Path(f"{cont_path}.bak"),
        Path(f"{cont_main_path}.bak"),
        Path(f"{tasks_path}.bak"),
        Path(f"{settings_path}.bak"),
    ):
        if managed.is_file():
            managed.unlink()
            removed_files.append(redact_home_path(managed, home=resolved_home))

    if cont_main_path.is_file():
        raw_main = cont_main_path.read_text(encoding="utf-8")
        cleaned_main = remove_hivemind_from_continue_config_yaml(raw_main)
        if cleaned_main is None:
            cont_main_path.unlink()
            removed_files.append(redact_home_path(cont_main_path, home=resolved_home))
        elif cleaned_main != raw_main:
            _write_file_secure_atomic(cont_main_path, cleaned_main, posix_mode=0o600, eff_os=eff_os)
            preserved_files.append(redact_home_path(cont_main_path, home=resolved_home))

    if queue_path.is_file():
        purge_res = VscodeCesRetrySpool(
            queue_path, home=resolved_home, env=env_view, os_name=eff_os
        ).purge(lang=lang_code)
        if purge_res.get("purged"):
            removed_files.append(redact_home_path(queue_path, home=resolved_home))

    if settings_path.is_file():
        raw_settings = strip_jsonc_comments_and_trailing_commas(settings_path.read_text(encoding="utf-8"))
        try:
            loaded_settings = json.loads(raw_settings) if raw_settings.strip() else {}
        except json.JSONDecodeError:
            loaded_settings = None
        if isinstance(loaded_settings, dict):
            had_hm = any(str(k).startswith(HIVEMIND_VSCODE_PREFIX) for k in loaded_settings)
            remaining_settings = {
                k: v for k, v in loaded_settings.items() if not str(k).startswith(HIVEMIND_VSCODE_PREFIX)
            }
            if had_hm and not remaining_settings:
                settings_path.unlink()
                removed_files.append(redact_home_path(settings_path, home=resolved_home))
            elif had_hm:
                clean_s = _assert_workspace_content_secret_free(
                    json.dumps(remaining_settings, indent=2, ensure_ascii=False) + "\n"
                )
                _write_file_secure_atomic(settings_path, clean_s, posix_mode=0o600, eff_os=eff_os)
                preserved_files.append(redact_home_path(settings_path, home=resolved_home))

    if tasks_path.is_file():
        raw_tasks = strip_jsonc_comments_and_trailing_commas(tasks_path.read_text(encoding="utf-8"))
        try:
            loaded_tasks = json.loads(raw_tasks) if raw_tasks.strip() else {}
        except json.JSONDecodeError:
            loaded_tasks = None
        if isinstance(loaded_tasks, dict) and isinstance(loaded_tasks.get("tasks"), list):
            orig_list = loaded_tasks["tasks"]
            had_hm_task = any(_is_hivemind_managed_task(t) for t in orig_list)
            user_tasks = [t for t in orig_list if not _is_hivemind_managed_task(t)]
            extra_keys = {k: v for k, v in loaded_tasks.items() if k not in ("version", "tasks")}
            if had_hm_task and not user_tasks and not extra_keys:
                tasks_path.unlink()
                removed_files.append(redact_home_path(tasks_path, home=resolved_home))
            elif had_hm_task:
                rebuilt = {"version": str(loaded_tasks.get("version") or "2.0.0"), **extra_keys, "tasks": user_tasks}
                clean_t = _assert_workspace_content_secret_free(
                    json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n"
                )
                _write_file_secure_atomic(tasks_path, clean_t, posix_mode=0o600, eff_os=eff_os)
                preserved_files.append(redact_home_path(tasks_path, home=resolved_home))

    if vscode_dir.is_dir():
        try:
            if not any(vscode_dir.iterdir()):
                vscode_dir.rmdir()
        except OSError as rmdir_err:
            _ = rmdir_err

    summary = (
        f"Désinstallation propre effectuée ({len(removed_files)} supprimé(s), {len(preserved_files)} préservé(s))."
        if is_fr
        else f"Clean local preset uninstall complete ({len(removed_files)} removed, {len(preserved_files)} preserved)."
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "mode": "uninstall_preset",
            "ok": True,
            "ready": True,
            "exit_code": 0,
            "removed_files": removed_files,
            "removed_count": len(removed_files),
            "preserved_files": preserved_files,
            "network_calls": 0,
            "compliance_notice": COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
        }
    )


def run_vscode_setup(
    gateway: str = DEFAULT_GATEWAY,
    *,
    workspace_dir: Path | str | None = None,
    credentials_path: Path | str | None = None,
    continue_config_path: Path | str | None = None,
    continue_main_config_path: Path | str | None = None,
    write_continue_config: bool = False,
    ces_queue_file: Path | str | None = None,
    handle: str | None = DEFAULT_HANDLE,
    role: str = DEFAULT_ROLE,
    explicit_api_key: str | None = None,
    host_label: str | None = None,
    expected_sha: str | None = None,
    skip_preflight: bool = False,
    require_holiday_routes: bool = False,
    pr949_quorum_gate: bool = False,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    require_ces_slo: bool = False,
    check_arena_summary: bool = False,
    verify_idempotency_replay: bool = False,
    execute_deploy_dry_run: bool = False,
    repo_root: Path | str | None = None,
    prompt_ces: bool = False,
    submit_ces: bool = False,
    flush_ces_queue: bool = False,
    profile: str = "dev",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    run_gamer_preflight: bool = False,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    send_first_heartbeat: bool = False,
    uninstall: bool = False,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    yield_drill_fn: YieldDrillFn = default_worker_yield_drill,
    process_probe_fn: ProcessProbeFn | None = default_process_probe,
    raw_http_fn: Any = default_http_raw,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    input_fn: Callable[[str], str] | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | str | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Unified `< 60 s` `hivemind vscode-setup` orchestrator with preflight, Continue config, and CES spool."""
    if uninstall:
        return uninstall_vscode_quick_preset(
            workspace_dir=workspace_dir,
            credentials_path=credentials_path,
            continue_config_path=continue_config_path,
            continue_main_config_path=continue_main_config_path,
            ces_queue_file=ces_queue_file,
            env=env,
            home=home,
            os_name=os_name,
            lang=lang,
        )

    t0 = time.perf_counter()
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    norm_profile = normalize_profile(profile)
    norm_role = normalize_role(role)
    clean_ease = parse_ease_rating(ease_rating)
    clean_gw = validate_gateway_url(gateway)
    clean_expected = validate_expected_sha(expected_sha) if expected_sha is not None else None
    _validate_pr949_options(
        target_online_producers=target_online_producers,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
        require_live_quorum=require_live_quorum,
        require_ces_slo=require_ces_slo,
        check_arena_summary=check_arena_summary,
        verify_idempotency_replay=verify_idempotency_replay,
        execute_deploy_dry_run=execute_deploy_dry_run,
    )
    pr949_enabled = bool(
        pr949_quorum_gate
        or require_live_quorum
        or require_ces_slo
        or check_arena_summary
        or verify_idempotency_replay
        or execute_deploy_dry_run
    )

    env_view = env if env is not None else os.environ
    eff_os = os_name if os_name is not None else os.name
    resolved_home = (
        Path(home)
        if home is not None
        else Path(env_view.get("HOME") or env_view.get("USERPROFILE") or Path.home())
    )
    ws_path = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
    cred_path = (
        Path(credentials_path)
        if credentials_path is not None
        else (resolved_home / ".hivemind" / "credentials.json")
    )
    cont_path = (
        Path(continue_config_path)
        if continue_config_path is not None
        else (resolved_home / ".continue" / "hivemind_provider.yaml")
    )
    cont_main_path = (
        Path(continue_main_config_path)
        if continue_main_config_path is not None
        else (resolved_home / ".continue" / "config.yaml")
    )
    queue_path = (
        Path(ces_queue_file)
        if ces_queue_file is not None
        else default_ces_queue_path(home=resolved_home, env=env_view)
    )

    compliance = evaluate_personal_pc_compliance(env=env_view, cwd=ws_path, lang=lang_code)
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "mode": "vscode_setup",
                "ok": False,
                "ready": False,
                "compliance_ok": False,
                "exit_code": 2,
                "verdict": "blocked_corporate_environment",
                "gateway": clean_gw,
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "within_60s_slo": True,
                "account": None,
                "files": {},
                "file_writes": 0,
                "network_calls": 0,
                "subprocess_calls": 0,
                "task_labels": [],
                "gitignore_check": None,
                "quick_preflight": None,
                "models_preflight_ok": False,
                "quick_route_ok": False,
                "feedback_endpoint_ok": False,
                "live_models_count": 0,
                "holiday_routes_deployed_on_cloud_run": None,
                "onboarding_feedback_deployed": None,
                "stale_revision_detected": False,
                "missing_or_stale_routes": [],
                "local_fallback_guidance": compliance["summary"],
                "write_continue_config": bool(write_continue_config),
                "preflight": None,
                "first_heartbeat_ok": None,
                "heartbeat": {"attempted": False, "sent": False, "ok": False},
                "ces_recorded": False,
                "ces_queued_for_retry": False,
                "ces_feedback": {"attempted": False, "ces_recorded": False, "status": "blocked"},
                "ces_flush": None,
                "ces_queue_file": redact_home_path(queue_path, home=resolved_home),
                "ces_queue_exists": queue_path.is_file(),
                "zero_inference_verified": True,
                "compliance": compliance,
                "summary": compliance["summary"],
            }
        )

    network_calls = 0
    subprocess_calls = 0
    quick_preflight: dict[str, Any] | None = None
    pr949_res: dict[str, Any] | None = None
    if not skip_preflight:
        quick_preflight = run_one_shot_quick_health_preflight(
            clean_gw,
            http_fn=http_fn,
            raw_http_fn=raw_http_fn,
            runner=runner,
            require_holiday_routes=require_holiday_routes,
            pr949_quorum_gate=pr949_quorum_gate,
            expected_sha=clean_expected,
            target_online_producers=target_online_producers,
            min_ease_rating=min_ease_rating,
            min_zero_lag_ratio=min_zero_lag_ratio,
            require_live_quorum=require_live_quorum,
            require_ces_slo=require_ces_slo,
            check_arena_summary=bool(check_arena_summary or pr949_quorum_gate),
            verify_idempotency_replay=verify_idempotency_replay,
            execute_deploy_dry_run=execute_deploy_dry_run,
            repo_root=repo_root,
            env=env_view,
            cwd=ws_path,
            lang=lang_code,
        )
        network_calls += int(quick_preflight["network_calls"])
        subprocess_calls += int(quick_preflight.get("subprocess_calls") or 0)
        if isinstance(quick_preflight.get("pr949_parity"), Mapping):
            pr949_res = dict(quick_preflight["pr949_parity"])
        if quick_preflight["exit_code"] != 0:
            blocked_doc: dict[str, Any] = {
                "schema": SCHEMA,
                "mode": "vscode_setup",
                "ok": False,
                "ready": False,
                "compliance_ok": True,
                "exit_code": 2,
                "verdict": str(quick_preflight["verdict"]),
                "gateway": clean_gw,
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "within_60s_slo": (time.perf_counter() - t0) < SLO_BUDGET_SECONDS,
                "account": None,
                "files": {},
                "file_writes": 0,
                "network_calls": network_calls,
                "subprocess_calls": subprocess_calls,
                "task_labels": [],
                "gitignore_check": None,
                "quick_preflight": quick_preflight,
                "models_preflight_ok": bool(quick_preflight["models_preflight_ok"]),
                "quick_route_ok": bool(quick_preflight["quick_route_ok"]),
                "feedback_endpoint_ok": bool(quick_preflight["feedback_endpoint_ok"]),
                "live_models_count": int(quick_preflight["live_models_count"]),
                "holiday_routes_deployed_on_cloud_run": bool(
                    quick_preflight["quick_route_ok"] and quick_preflight["feedback_endpoint_ok"]
                ),
                "onboarding_feedback_deployed": bool(quick_preflight["feedback_endpoint_ok"]),
                "stale_revision_detected": bool(quick_preflight["stale_revision_detected"]),
                "missing_or_stale_routes": list(quick_preflight["missing_or_stale_routes"]),
                "local_fallback_guidance": str(quick_preflight["local_fallback_guidance"]),
                "write_continue_config": bool(write_continue_config),
                "preflight": None,
                "first_heartbeat_ok": None,
                "heartbeat": {"attempted": False, "sent": False, "ok": False},
                "ces_recorded": False,
                "ces_queued_for_retry": False,
                "ces_feedback": {"attempted": False, "ces_recorded": False, "status": "preflight_blocked"},
                "ces_flush": None,
                "ces_queue_file": redact_home_path(queue_path, home=resolved_home),
                "ces_queue_exists": queue_path.is_file(),
                "zero_inference_verified": True,
                "compliance": compliance,
                "summary": str(quick_preflight["summary"]),
            }
            if pr949_res is not None:
                blocked_doc.update(
                    {
                        "pr949_parity": pr949_res,
                        "online_producers": pr949_res.get("online_producers"),
                        "producer_quorum_met": pr949_res.get("producer_quorum_met"),
                        "awaiting_producer_quorum": pr949_res.get("awaiting_producer_quorum"),
                        "ces_slo_ok": pr949_res.get("ces_slo_ok"),
                        "zero_lag_slo_ok": pr949_res.get("zero_lag_slo_ok"),
                        "arena_summary_ok": pr949_res.get("arena_summary_ok"),
                        "idempotent_dedup_ok": pr949_res.get("idempotent_dedup_ok"),
                        "deploy_dry_run": pr949_res.get("deploy_dry_run"),
                    }
                )
            return _attach_receipt(blocked_doc)
    elif pr949_enabled:
        pr949_res = verify_pr949_vscode_quorum_and_arena_parity(
            clean_gw,
            expected_sha=clean_expected,
            target_online_producers=target_online_producers,
            min_ease_rating=min_ease_rating,
            min_zero_lag_ratio=min_zero_lag_ratio,
            require_live_quorum=require_live_quorum,
            require_ces_slo=require_ces_slo,
            check_arena_summary=bool(check_arena_summary or pr949_quorum_gate),
            verify_idempotency_replay=verify_idempotency_replay,
            execute_deploy_dry_run=execute_deploy_dry_run,
            repo_root=repo_root,
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
            runner=runner,
            env=env_view,
            cwd=ws_path,
            lang=lang_code,
        )
        network_calls += int(pr949_res.get("network_calls") or 0)
        subprocess_calls += int(pr949_res.get("subprocess_calls") or 0)
        if pr949_res["exit_code"] != 0:
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "mode": "vscode_setup",
                    "ok": False,
                    "ready": False,
                    "compliance_ok": True,
                    "exit_code": 2,
                    "verdict": "pr949_quorum_or_slo_failed",
                    "gateway": clean_gw,
                    "elapsed_s": round(time.perf_counter() - t0, 4),
                    "within_60s_slo": (time.perf_counter() - t0) < SLO_BUDGET_SECONDS,
                    "account": None,
                    "files": {},
                    "file_writes": 0,
                    "network_calls": network_calls,
                    "subprocess_calls": subprocess_calls,
                    "task_labels": [],
                    "gitignore_check": None,
                    "quick_preflight": None,
                    "pr949_parity": pr949_res,
                    "models_preflight_ok": bool(pr949_res.get("models_preflight_ok")),
                    "quick_route_ok": bool((pr949_res.get("routes") or {}).get("/quick", {}).get("ok")),
                    "feedback_endpoint_ok": bool(pr949_res.get("onboarding_feedback_deployed")),
                    "live_models_count": int(pr949_res.get("live_models_count") or 0),
                    "holiday_routes_deployed_on_cloud_run": bool(pr949_res.get("holiday_routes_deployed_on_cloud_run")),
                    "onboarding_feedback_deployed": bool(pr949_res.get("onboarding_feedback_deployed")),
                    "stale_revision_detected": bool(pr949_res.get("stale_revision_detected")),
                    "missing_or_stale_routes": list(pr949_res.get("missing_or_stale_routes") or []),
                    "online_producers": pr949_res.get("online_producers"),
                    "producer_quorum_met": pr949_res.get("producer_quorum_met"),
                    "awaiting_producer_quorum": pr949_res.get("awaiting_producer_quorum"),
                    "ces_slo_ok": pr949_res.get("ces_slo_ok"),
                    "zero_lag_slo_ok": pr949_res.get("zero_lag_slo_ok"),
                    "arena_summary_ok": pr949_res.get("arena_summary_ok"),
                    "idempotent_dedup_ok": pr949_res.get("idempotent_dedup_ok"),
                    "deploy_dry_run": pr949_res.get("deploy_dry_run"),
                    "local_fallback_guidance": str(pr949_res.get("summary") or ""),
                    "write_continue_config": bool(write_continue_config),
                    "preflight": None,
                    "first_heartbeat_ok": None,
                    "heartbeat": {"attempted": False, "sent": False, "ok": False},
                    "ces_recorded": False,
                    "ces_queued_for_retry": False,
                    "ces_feedback": {"attempted": False, "ces_recorded": False, "status": "pr949_gate_blocked"},
                    "ces_flush": None,
                    "ces_queue_file": redact_home_path(queue_path, home=resolved_home),
                    "ces_queue_exists": queue_path.is_file(),
                    "zero_inference_verified": True,
                    "compliance": compliance,
                    "summary": str(pr949_res.get("summary") or ""),
                }
            )

    preset_report = install_vscode_quick_preset(
        clean_gw,
        workspace_dir=ws_path,
        credentials_path=cred_path,
        continue_config_path=cont_path,
        handle=handle,
        role=norm_role,
        explicit_api_key=explicit_api_key,
        host_label=host_label,
        expected_sha=clean_expected,
        send_first_heartbeat=send_first_heartbeat,
        submit_ces=False,
        profile=norm_profile,
        ease_rating=clean_ease,
        had_game_lag=had_game_lag,
        require_holiday_routes=bool(require_holiday_routes or clean_expected is not None),
        run_gamer_preflight=run_gamer_preflight,
        max_temp_c=max_temp_c,
        yield_budget_ms=yield_budget_ms,
        gpu_probe_fn=gpu_probe_fn,
        yield_drill_fn=yield_drill_fn,
        process_probe_fn=process_probe_fn,
        http_fn=http_fn,
        env=env_view,
        home=resolved_home,
        os_name=eff_os,
        system_name=system_name,
        lang=lang_code,
    )
    network_calls += int(preset_report["network_calls"])

    if preset_report["exit_code"] != 0:
        return _attach_receipt(
            {
                **{k: v for k, v in preset_report.items() if k != "receipt_sha256"},
                "mode": "vscode_setup",
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "within_60s_slo": (time.perf_counter() - t0) < SLO_BUDGET_SECONDS,
                "network_calls": network_calls,
                "quick_preflight": quick_preflight,
                "models_preflight_ok": (
                    bool(quick_preflight["models_preflight_ok"]) if quick_preflight is not None else False
                ),
                "quick_route_ok": (
                    bool(quick_preflight["quick_route_ok"]) if quick_preflight is not None else None
                ),
                "feedback_endpoint_ok": (
                    bool(quick_preflight["feedback_endpoint_ok"]) if quick_preflight is not None else None
                ),
                "live_models_count": (
                    int(quick_preflight["live_models_count"]) if quick_preflight is not None else 0
                ),
                "stale_revision_detected": bool(
                    preset_report.get("stale_revision_detected")
                    or (quick_preflight and quick_preflight.get("stale_revision_detected"))
                ),
                "local_fallback_guidance": (
                    str(quick_preflight["local_fallback_guidance"]) if quick_preflight is not None else ""
                ),
                "write_continue_config": bool(write_continue_config),
                "ces_queued_for_retry": False,
                "ces_flush": None,
                "ces_queue_file": redact_home_path(queue_path, home=resolved_home),
                "ces_queue_exists": queue_path.is_file(),
            }
        )

    files_map = dict(preset_report.get("files") or {})
    file_writes = int(preset_report.get("file_writes") or 0)

    if write_continue_config:
        saved_cred = load_saved_credentials(cred_path, expected_gateway=clean_gw)
        if saved_cred is None or not is_valid_hivemind_key(saved_cred.get("api_key")):
            raise QuickPresetError("Unable to load saved credentials for Continue config.yaml merge.")
        existing_main = cont_main_path.read_text(encoding="utf-8") if cont_main_path.is_file() else None
        merged_main = merge_continue_config_yaml(
            existing_main,
            gateway=clean_gw,
            api_key=str(saved_cred["api_key"]),
        )
        main_write = atomic_write_config_file(
            cont_main_path,
            merged_main,
            posix_mode=0o600,
            create_backup=True,
            os_name=eff_os,
            home=resolved_home,
        )
        files_map["continue_main_config"] = main_write
        if main_write.get("written"):
            file_writes += 1

    acct_public = dict(preset_report.get("account") or {})
    account_ref = str(acct_public.get("account_id") or acct_public.get("masked_key") or "anon")
    gamer_pf = preset_report.get("preflight")
    eff_zero_lag = not bool(had_game_lag)
    if isinstance(gamer_pf, Mapping) and not bool(gamer_pf.get("yield_ok", True)):
        eff_zero_lag = False

    ces_recorded = False
    ces_spooled = False
    ces_report: dict[str, Any] = {"attempted": False, "ces_recorded": False, "status": "not_requested"}
    stale_revision = bool(
        preset_report.get("stale_revision_detected")
        or (quick_preflight is not None and quick_preflight.get("stale_revision_detected"))
    )
    missing_routes = list(preset_report.get("missing_or_stale_routes") or [])
    if quick_preflight is not None:
        for route_item in quick_preflight.get("missing_or_stale_routes") or []:
            if route_item not in missing_routes:
                missing_routes.append(route_item)

    if prompt_ces:
        ces_report = prompt_and_submit_ces(
            gateway=clean_gw,
            account_ref=account_ref,
            profile=norm_profile,
            default_had_game_lag=not eff_zero_lag,
            system_name=system_name,
            input_fn=input_fn,
            http_fn=http_fn,
            ces_queue_file=queue_path,
            home=resolved_home,
            env=env_view,
            cwd=ws_path,
            os_name=eff_os,
            spool_on_failure=True,
            lang=lang_code,
        )
        network_calls += int(ces_report.get("network_calls") or 0)
        ces_recorded = bool(ces_report.get("ces_recorded"))
        ces_spooled = bool(ces_report.get("queued_for_retry"))
        if ces_report.get("http_status") == 404:
            stale_revision = True
            if "/v1/hivemind/onboarding/feedback" not in missing_routes:
                missing_routes.append("/v1/hivemind/onboarding/feedback")
    elif submit_ces:
        ces_report, ces_calls = submit_preset_ces_feedback(
            gateway=clean_gw,
            account_ref=account_ref,
            profile=norm_profile,
            ease_rating=clean_ease,
            zero_game_lag=eff_zero_lag,
            system_name=system_name,
            http_fn=http_fn,
            lang=lang_code,
        )
        network_calls += ces_calls
        ces_recorded = bool(ces_report.get("ces_recorded"))
        http_st = int(ces_report.get("http_status") or 0)
        if not ces_recorded and (http_st in (0, 404, 408, 429) or http_st >= 500):
            enq_res = enqueue_vscode_ces_event(
                queue_file=queue_path,
                gateway=clean_gw,
                account_ref=account_ref,
                profile=norm_profile,
                ease_rating=clean_ease,
                zero_game_lag=eff_zero_lag,
                system_name=system_name,
                idempotency_key=str(ces_report["idempotency_key"]),
                last_http_status=http_st,
                last_reason=str(ces_report["status"]),
                home=resolved_home,
                enforce_personal_pc=False,
                env=env_view,
                cwd=ws_path,
                os_name=eff_os,
                lang=lang_code,
            )
            ces_spooled = bool(enq_res.get("enqueued") or enq_res.get("deduplicated"))
            ces_report = {
                **ces_report,
                "queued_for_retry": ces_spooled,
                "queue_file": enq_res.get("queue_file"),
            }
        if http_st == 404:
            stale_revision = True
            if "/v1/hivemind/onboarding/feedback" not in missing_routes:
                missing_routes.append("/v1/hivemind/onboarding/feedback")

    flush_report: dict[str, Any] | None = None
    if flush_ces_queue and not ces_spooled:
        flush_report = flush_vscode_ces_queue(
            clean_gw,
            queue_file=queue_path,
            home=resolved_home,
            enforce_personal_pc=False,
            env=env_view,
            cwd=ws_path,
            os_name=eff_os,
            http_fn=http_fn,
            lang=lang_code,
        )
        network_calls += int(flush_report.get("network_calls") or 0)
        if flush_report.get("stale_cloud_run_detected"):
            stale_revision = True

    ces_attempted = bool(ces_report.get("attempted"))
    ces_gate_failed = bool(require_holiday_routes and ces_attempted and not ces_recorded)
    overall_ready = not ces_gate_failed
    elapsed_s = round(time.perf_counter() - t0, 4)

    holiday_deployed = preset_report.get("holiday_routes_deployed_on_cloud_run")
    feedback_deployed = preset_report.get("onboarding_feedback_deployed")
    if quick_preflight is not None and holiday_deployed is None:
        holiday_deployed = bool(
            quick_preflight["quick_route_ok"] and quick_preflight["feedback_endpoint_ok"] and not stale_revision
        )
        feedback_deployed = bool(quick_preflight["feedback_endpoint_ok"] and not stale_revision)

    acct_models = list(acct_public.get("models") or [])
    models_ok = (
        bool(quick_preflight["models_preflight_ok"])
        if quick_preflight is not None
        else (len(acct_models) > 0 and DEFAULT_MODELS[0] in acct_models)
    )
    live_models_count = (
        int(quick_preflight["live_models_count"])
        if quick_preflight is not None
        else len(acct_models)
    )

    fallback_guidance = str(quick_preflight["local_fallback_guidance"]) if quick_preflight is not None else ""
    if stale_revision and not fallback_guidance:
        fallback_guidance = (
            "Révision Cloud Run en retard : retours CES mis en file locale 0600 pour --flush-ces-queue."
            if is_fr
            else "Stale Cloud Run revision detected: CES feedback queued in 0600 local spool for --flush-ces-queue."
        )

    verdict = "ready" if overall_ready else "stale_cloud_run_revision"
    summary = (
        (
            f"Configuration VS Code < 60 s terminée en {elapsed_s:.2f}s (clé={acct_public.get('masked_key')}, "
            f"CES={ces_recorded}, file_retry={ces_spooled}, {FINANCIAL_PROMISE})."
            if is_fr
            else f"VS Code < 60 s setup completed in {elapsed_s:.2f}s (key={acct_public.get('masked_key')}, "
            f"ces_recorded={ces_recorded}, queued_for_retry={ces_spooled}, {FINANCIAL_PROMISE})."
        )
        if overall_ready
        else (
            "Configuration VS Code bloquée : routes Cloud Run en retard."
            if is_fr
            else "VS Code setup blocked by stale Cloud Run holiday routes."
        )
    )

    eff_online_producers = (
        preset_report.get("online_producers")
        if preset_report.get("online_producers") is not None
        else (pr949_res.get("online_producers") if pr949_res is not None else None)
    )
    out_doc: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "vscode_setup",
        "ok": overall_ready,
        "ready": overall_ready,
        "compliance_ok": True,
        "exit_code": 0 if overall_ready else 2,
        "verdict": verdict,
        "gateway": clean_gw,
        "elapsed_s": elapsed_s,
        "within_60s_slo": elapsed_s < SLO_BUDGET_SECONDS,
        "account": acct_public,
        "files": files_map,
        "file_writes": file_writes,
        "network_calls": network_calls,
        "task_labels": list(PRESET_TASK_LABELS),
        "gitignore_check": preset_report.get("gitignore_check"),
        "quick_preflight": quick_preflight,
        "models_preflight_ok": models_ok,
        "quick_route_ok": (
            bool(quick_preflight["quick_route_ok"]) if quick_preflight is not None else None
        ),
        "feedback_endpoint_ok": (
            bool(quick_preflight["feedback_endpoint_ok"]) if quick_preflight is not None else None
        ),
        "live_models_count": live_models_count,
        "holiday_routes_deployed_on_cloud_run": holiday_deployed,
        "onboarding_feedback_deployed": feedback_deployed,
        "stale_revision_detected": stale_revision,
        "missing_or_stale_routes": missing_routes,
        "online_producers": eff_online_producers,
        "local_fallback_guidance": fallback_guidance,
        "write_continue_config": bool(write_continue_config),
        "parity": preset_report.get("parity"),
        "preflight": gamer_pf,
        "first_heartbeat_ok": preset_report.get("first_heartbeat_ok"),
        "heartbeat": preset_report.get("heartbeat"),
        "ces_recorded": ces_recorded,
        "ces_queued_for_retry": ces_spooled,
        "ces_feedback": ces_report,
        "ces_flush": flush_report,
        "ces_queue_file": redact_home_path(queue_path, home=resolved_home),
        "ces_queue_exists": queue_path.is_file(),
        "zero_inference_verified": True,
        "compliance": compliance,
        "summary": summary,
    }
    if pr949_res is not None:
        out_doc.update(
            {
                "pr949_parity": pr949_res,
                "producer_quorum_met": pr949_res.get("producer_quorum_met"),
                "awaiting_producer_quorum": pr949_res.get("awaiting_producer_quorum"),
                "ces_slo_ok": pr949_res.get("ces_slo_ok"),
                "zero_lag_slo_ok": pr949_res.get("zero_lag_slo_ok"),
                "arena_summary_ok": pr949_res.get("arena_summary_ok"),
                "idempotent_dedup_ok": pr949_res.get("idempotent_dedup_ok"),
                "deploy_dry_run": pr949_res.get("deploy_dry_run"),
                "subprocess_calls": subprocess_calls,
            }
        )
    return _attach_receipt(out_doc)


def _validate_cli_path_arg(raw: str | None, *, flag: str) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise QuickPresetError(f"{flag} must be a non-empty path without NUL bytes.")
    return Path(raw.strip())


def render_human_report(report: Mapping[str, Any], *, lang: str = "en") -> str:
    """Render a zero-secret, `~`-redacted CLI summary in English or French."""
    lang_code = normalize_lang(lang)
    status_tag = "OK" if report.get("ready") else "BLOCKED"
    lines = [
        f"[HiveMind Noel VS Code Quick Preset | {lang_code.upper()} | {status_tag}]",
        f"Summary: {report.get('summary', '')}",
        f"Compliance: {COMPLIANCE_NOTICE_BILINGUAL}",
        f"Receipt SHA-256: {report.get('receipt_sha256', '')}",
    ]
    if report.get("producer_quorum_met") is not None or report.get("ces_slo_ok") is not None:
        lines.append(
            f"Quorum/SLO: online={report.get('online_producers')} "
            f"quorum_met={report.get('producer_quorum_met')} "
            f"ces_slo_ok={report.get('ces_slo_ok')} zero_lag_slo_ok={report.get('zero_lag_slo_ok')}"
        )
    if report.get("arena_summary_ok") is not None:
        lines.append(f"Arena Summary Parity: {report.get('arena_summary_ok')}")
    if report.get("idempotent_dedup_ok") is not None:
        lines.append(f"Idempotency Replay Conserved: {report.get('idempotent_dedup_ok')}")
    if isinstance(report.get("deploy_dry_run"), Mapping):
        lines.append(
            f"Deploy Dry-Run: ok={report['deploy_dry_run'].get('ok')} "
            f"build_sha={report['deploy_dry_run'].get('dry_run_build_sha')}"
        )
    guidance = str(report.get("local_fallback_guidance") or "").strip()
    if guidance:
        lines.append(f"Fallback Guidance: {guidance}")
    git_chk = report.get("gitignore_check")
    if isinstance(git_chk, Mapping) and git_chk.get("warning"):
        lines.append(f"Notice: {git_chk['warning']}")
    return _assert_safe_output("\n".join(lines) + "\n")


def render_markdown_report(report: Mapping[str, Any], *, lang: str = "en") -> str:
    """Render a zero-secret, `~`-redacted Markdown summary in English or French."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    heading = (
        "# Rapport HiveMind Noël VS Code Preset & Parité PR #949"
        if is_fr
        else "# HiveMind Noel VS Code Preset & PR #949 Parity Report"
    )
    lines = [
        heading,
        "",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Conformité' if is_fr else 'Compliance'}**: `{report.get('compliance_ok')}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Verdict' if is_fr else 'Exit Code'}**: `{report.get('exit_code')}`",
        f"- **{'Résumé' if is_fr else 'Summary'}**: {report.get('summary', '')}",
        f"- **Receipt SHA-256**: `{report.get('receipt_sha256', '')}`",
    ]
    if report.get("producer_quorum_met") is not None or report.get("ces_slo_ok") is not None:
        lines.append(
            f"- **{'Quorum & SLO CES' if is_fr else 'Quorum & CES SLO'}**: "
            f"`online_producers={report.get('online_producers')}`, "
            f"`producer_quorum_met={report.get('producer_quorum_met')}`, "
            f"`ces_slo_ok={report.get('ces_slo_ok')}`, `zero_lag_slo_ok={report.get('zero_lag_slo_ok')}`"
        )
    if report.get("arena_summary_ok") is not None:
        lines.append(f"- **Arena Summary Parity**: `{report.get('arena_summary_ok')}`")
    if report.get("idempotent_dedup_ok") is not None:
        lines.append(f"- **Idempotency Replay Conserved**: `{report.get('idempotent_dedup_ok')}`")
    if isinstance(report.get("deploy_dry_run"), Mapping):
        lines.append(
            f"- **Deploy Dry-Run**: `ok={report['deploy_dry_run'].get('ok')}`, "
            f"`build_sha={report['deploy_dry_run'].get('dry_run_build_sha')}`"
        )
    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        lines.extend(f"- {d}" for d in diags)
    return _assert_safe_output("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for `scripts/noel_vscode_quick_preset.py`."""
    parser = argparse.ArgumentParser(
        prog="noel_vscode_quick_preset.py",
        description=(
            "Zero-inference Cloud Run holiday-route parity gate and <60s VS Code one-click quick-connect preset "
            "(100% personal home PCs & personal hobby/open-source projects only, 0.00 EUR / 0.00 USD)."
        ),
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="HiveMind gateway URL.")
    parser.add_argument("--check-parity", action="store_true", help="Run Cloud Run holiday-route parity gate.")
    parser.add_argument(
        "--pr949-quorum-gate",
        action="store_true",
        help="Run zero-inference PR #949 4-HTML parity, 8+ producer quorum, arena summary, CES SLO & replay gate.",
    )
    parser.add_argument(
        "--require-holiday-routes",
        action="store_true",
        help="Fail closed (exit code 2) if Cloud Run holiday routes or expected_sha are missing/stale.",
    )
    parser.add_argument("--expected-sha", default=None, help="Expected Cloud Run manifest build_sha.")
    parser.add_argument(
        "--target-online-producers",
        type=int,
        default=DEFAULT_TARGET_ONLINE_PRODUCERS,
        help="Minimum live volunteer friend producers required for quorum.",
    )
    parser.add_argument(
        "--min-ease-rating",
        type=float,
        default=DEFAULT_MIN_EASE_RATING,
        help="Minimum mean CES ease rating threshold (1.0..5.0).",
    )
    parser.add_argument(
        "--min-zero-lag-ratio",
        type=float,
        default=DEFAULT_MIN_ZERO_LAG_RATIO,
        help="Minimum gamer zero-game-lag ratio threshold (0.0..1.0).",
    )
    parser.add_argument("--require-live-quorum", action="store_true", help="Fail closed (exit 2) when online_producers < target.")
    parser.add_argument("--require-ces-slo", action="store_true", help="Fail closed (exit 2) when CES/zero-lag SLO is breached.")
    parser.add_argument("--check-arena-summary", action="store_true", help="Verify /v1/hivemind/arena/summary conservation & Wilson LB.")
    parser.add_argument("--verify-idempotency-replay", action="store_true", help="Verify Idempotency-Key duplicate-replay deduplication.")
    parser.add_argument("--run-deploy-dry-run", action="store_true", help="Execute deploy/gcp_openrouter/deploy.sh --dry-run (no gcloud).")
    parser.add_argument("--install-preset", action="store_true", help="Install <60s VS Code quick-connect preset.")
    parser.add_argument("--vscode-setup", action="store_true", help="Run unified <60s VS Code setup orchestrator.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip 1-shot /v1/models + /quick health preflight.")
    parser.add_argument("--workspace", default=None, help="Target personal workspace directory.")
    parser.add_argument("--cwd", default=None, help="Override working directory for personal-PC compliance & git remote check.")
    parser.add_argument("--output", default=None, help="Atomically persist verified output (POSIX 0600); never writes when compliance fails.")
    parser.add_argument("--credentials-path", default=None, help="Override ~/.hivemind/credentials.json path.")
    parser.add_argument("--continue-config-path", default=None, help="Override ~/.continue/hivemind_provider.yaml.")
    parser.add_argument("--continue-main-config-path", default=None, help="Override ~/.continue/config.yaml.")
    parser.add_argument("--write-continue-config", action="store_true", help="Atomically merge HiveMind models into ~/.continue/config.yaml.")
    parser.add_argument("--ces-queue-file", default=None, help="Override ~/.hivemind/vscode_setup_ces_queue.json.")
    parser.add_argument("--uninstall", action="store_true", help="Remove HiveMind preset artifacts locally.")
    parser.add_argument("--handle", default=DEFAULT_HANDLE, help="Personal handle for registration.")
    parser.add_argument("--role", default=DEFAULT_ROLE, choices=sorted(VALID_ROLES), help="Account role.")
    parser.add_argument("--host-label", default=None, help="Sanitized personal host label for OpenTrainer.")
    parser.add_argument("--first-heartbeat", action="store_true", help="Send outbound OpenTrainer first heartbeat.")
    parser.add_argument("--gamer-preflight", action="store_true", help="Run <1.5s Steam yield & <=70 C check.")
    parser.add_argument("--max-temp-c", type=float, default=DEFAULT_MAX_TEMP_C, help="Thermal guard ceiling in C.")
    parser.add_argument("--prompt-ces", action="store_true", help="Ask a 1-line zero-PII CES question interactively.")
    parser.add_argument("--submit-ces", action="store_true", help="Submit non-blocking zero-PII CES feedback.")
    parser.add_argument("--flush-ces-queue", action="store_true", help="Flush pending 0600 CES retry spool entries.")
    parser.add_argument("--profile", default="dev", choices=sorted(VALID_PROFILES), help="CES profile (dev|gamer).")
    parser.add_argument("--ease-rating", type=int, default=5, help="CES ease rating (1..5).")
    parser.add_argument("--had-game-lag", action="store_true", help="Report whether any game lag occurred.")
    parser.add_argument("--lang", default="en", choices=SUPPORTED_LANGS, help="Output language (en|fr).")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON report.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown report.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | str | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    raw_http_fn: Any = default_http_raw,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    yield_drill_fn: YieldDrillFn = default_worker_yield_drill,
    process_probe_fn: ProcessProbeFn | None = default_process_probe,
    input_fn: Callable[[str], str] | None = None,
    stdout: Any = None,
) -> int:
    """CLI entrypoint returning `0` on success or `2` on fail-closed compliance/parity/preset errors."""
    out_stream = stdout if stdout is not None else sys.stdout
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    eff_os = os_name if os_name is not None else os.name
    out_path: Path | None = None

    try:
        lang_code = normalize_lang(args.lang)
        cli_cwd = _validate_cli_path_arg(args.cwd, flag="--cwd")
        cli_ws = _validate_cli_path_arg(args.workspace, flag="--workspace")
        out_path = _validate_cli_path_arg(args.output, flag="--output")
        eff_cwd = cli_ws if cli_ws is not None else (cli_cwd if cli_cwd is not None else (Path(cwd) if cwd is not None else Path.cwd()))
        if args.uninstall:
            report = uninstall_vscode_quick_preset(
                workspace_dir=eff_cwd,
                credentials_path=args.credentials_path,
                continue_config_path=args.continue_config_path,
                continue_main_config_path=args.continue_main_config_path,
                ces_queue_file=args.ces_queue_file,
                env=env,
                home=home,
                os_name=os_name,
                lang=lang_code,
            )
        elif (
            args.flush_ces_queue
            and not args.vscode_setup
            and not args.install_preset
            and not args.check_parity
            and not args.pr949_quorum_gate
            and not args.write_continue_config
            and not args.prompt_ces
            and not args.submit_ces
            and not args.first_heartbeat
            and not args.gamer_preflight
        ):
            report = flush_vscode_ces_queue(
                args.gateway,
                queue_file=args.ces_queue_file,
                home=home,
                env=env,
                cwd=eff_cwd,
                os_name=os_name,
                http_fn=http_fn,
                lang=lang_code,
            )
        elif args.vscode_setup:
            report = run_vscode_setup(
                args.gateway,
                workspace_dir=eff_cwd,
                credentials_path=args.credentials_path,
                continue_config_path=args.continue_config_path,
                continue_main_config_path=args.continue_main_config_path,
                write_continue_config=bool(args.write_continue_config),
                ces_queue_file=args.ces_queue_file,
                handle=args.handle,
                role=args.role,
                host_label=args.host_label,
                expected_sha=args.expected_sha,
                skip_preflight=bool(args.skip_preflight),
                require_holiday_routes=bool(args.require_holiday_routes),
                pr949_quorum_gate=bool(args.pr949_quorum_gate),
                target_online_producers=int(args.target_online_producers),
                min_ease_rating=float(args.min_ease_rating),
                min_zero_lag_ratio=float(args.min_zero_lag_ratio),
                require_live_quorum=bool(args.require_live_quorum),
                require_ces_slo=bool(args.require_ces_slo),
                check_arena_summary=bool(args.check_arena_summary or args.pr949_quorum_gate),
                verify_idempotency_replay=bool(args.verify_idempotency_replay),
                execute_deploy_dry_run=bool(args.run_deploy_dry_run),
                prompt_ces=bool(args.prompt_ces),
                submit_ces=bool(args.submit_ces),
                flush_ces_queue=bool(args.flush_ces_queue),
                profile=args.profile,
                ease_rating=args.ease_rating,
                had_game_lag=bool(args.had_game_lag),
                run_gamer_preflight=bool(args.gamer_preflight),
                max_temp_c=float(args.max_temp_c),
                send_first_heartbeat=bool(args.first_heartbeat),
                gpu_probe_fn=gpu_probe_fn,
                yield_drill_fn=yield_drill_fn,
                process_probe_fn=process_probe_fn,
                raw_http_fn=raw_http_fn,
                http_fn=http_fn,
                runner=runner,
                input_fn=input_fn,
                env=env,
                home=home,
                os_name=os_name,
                system_name=system_name,
                lang=lang_code,
            )
        elif args.pr949_quorum_gate and not args.install_preset:
            report = verify_pr949_vscode_quorum_and_arena_parity(
                args.gateway,
                expected_sha=args.expected_sha,
                target_online_producers=int(args.target_online_producers),
                min_ease_rating=float(args.min_ease_rating),
                min_zero_lag_ratio=float(args.min_zero_lag_ratio),
                require_live_quorum=bool(args.require_live_quorum),
                require_ces_slo=bool(args.require_ces_slo),
                check_arena_summary=True,
                verify_idempotency_replay=bool(args.verify_idempotency_replay),
                execute_deploy_dry_run=bool(args.run_deploy_dry_run),
                raw_http_fn=raw_http_fn,
                http_fn=http_fn,
                runner=runner,
                env=env,
                cwd=cli_cwd if cli_cwd is not None else eff_cwd,
                lang=lang_code,
            )
        elif args.check_parity and not args.install_preset:
            if (
                args.require_live_quorum
                or args.require_ces_slo
                or args.check_arena_summary
                or args.verify_idempotency_replay
                or args.run_deploy_dry_run
            ):
                report = verify_pr949_vscode_quorum_and_arena_parity(
                    args.gateway,
                    expected_sha=args.expected_sha,
                    target_online_producers=int(args.target_online_producers),
                    min_ease_rating=float(args.min_ease_rating),
                    min_zero_lag_ratio=float(args.min_zero_lag_ratio),
                    require_live_quorum=bool(args.require_live_quorum),
                    require_ces_slo=bool(args.require_ces_slo),
                    check_arena_summary=bool(args.check_arena_summary),
                    verify_idempotency_replay=bool(args.verify_idempotency_replay),
                    execute_deploy_dry_run=bool(args.run_deploy_dry_run),
                    raw_http_fn=raw_http_fn,
                    http_fn=http_fn,
                    runner=runner,
                    env=env,
                    cwd=cli_cwd if cli_cwd is not None else eff_cwd,
                    lang=lang_code,
                )
            else:
                report = verify_cloud_run_holiday_parity(
                    args.gateway,
                    expected_sha=args.expected_sha,
                    http_fn=http_fn,
                    env=env,
                    cwd=cli_cwd if cli_cwd is not None else eff_cwd,
                    lang=lang_code,
                )
        else:
            report = install_vscode_quick_preset(
                args.gateway,
                workspace_dir=eff_cwd,
                credentials_path=args.credentials_path,
                continue_config_path=args.continue_config_path,
                handle=args.handle,
                role=args.role,
                host_label=args.host_label,
                expected_sha=args.expected_sha,
                send_first_heartbeat=bool(args.first_heartbeat),
                submit_ces=bool(args.submit_ces),
                profile=args.profile,
                ease_rating=args.ease_rating,
                had_game_lag=bool(args.had_game_lag),
                require_holiday_routes=bool(args.require_holiday_routes or args.check_parity),
                run_gamer_preflight=bool(args.gamer_preflight),
                max_temp_c=float(args.max_temp_c),
                gpu_probe_fn=gpu_probe_fn,
                yield_drill_fn=yield_drill_fn,
                process_probe_fn=process_probe_fn,
                http_fn=http_fn,
                env=env,
                home=home,
                os_name=os_name,
                system_name=system_name,
                lang=lang_code,
            )
    except QuickPresetError as exc:
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

    out_lang = args.lang if args.lang in SUPPORTED_LANGS else "en"
    if args.json_output:
        rendered = _assert_safe_output(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    elif args.markdown:
        rendered = render_markdown_report(report, lang=out_lang)
    else:
        rendered = render_human_report(report, lang=out_lang)

    if out_path is not None and report.get("compliance_ok") is not False and report.get("mode") != "error":
        _write_file_secure_atomic(out_path, rendered, posix_mode=0o600, eff_os=eff_os)
    out_stream.write(rendered)
    return int(report.get("exit_code", 2))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
