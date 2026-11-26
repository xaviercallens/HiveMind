#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-Inference Cloud Run PR #949 CD Deploy-Bundle Gate & Unified Win11/VS Code Quickstart Bundle Verifier (`SLICE-SCALE-DYN-34`).

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment variables (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``,
     ``NODE_EXTRA_CA_CERTS``), workspace/home paths, and ``.git/config`` remote URLs (including
     worktree ``gitdir:`` / ``commondir`` indirection) BEFORE any network call, file write, or
     subprocess drill. Fails closed with exit code 2 (0 network calls, 0 file writes, 0 subprocess
     calls, 0 raw env/hostname/path leaks).
  2. Zero-Inference CD Post-Merge Cloud Run Deploy-Bundle & PR #949 Live-Revision Verifier
     (``verify_cd_deploy_bundle_and_live_revision``): audits
     ``deploy/gcp_openrouter/{deploy.sh,Dockerfile,entrypoint.sh,requirements.txt,server.py}``
     FinOps & supply-chain invariants (``--min-instances 0``, ``--max-instances 1``,
     ``--no-cpu-throttling``, ``--concurrency 80``, Litestream SHA-256 pin &
     ``HIVEMIND_ROUTER_REPLICA_URL``, Secret Manager ``HIVEMIND_ROUTER_SECRET``, 0 starter/bonus
     grants, ``HIVEMIND_BUILD_SHA`` injection), optionally runs list-argv
     ``bash deploy/gcp_openrouter/deploy.sh --dry-run`` plus an isolated ASGI ``build_app`` sanity
     check when FastAPI is available, computes deterministic CD bundle SHA-256 fingerprints across
     Cloud Run-deployed files, and probes ``/health``, ``/v1/models``, ``/noel``, ``/quick``,
     ``/raid-noel``, ``/arena-noel``, ``/v1/hivemind/onboarding/manifest``,
     ``/v1/hivemind/onboarding/feedback/summary`` (exact 5-key schema),
     ``/v1/hivemind/producers/fleet``, and ``/v1/hivemind/raid/status`` without ever calling
     ``POST /v1/chat/completions`` or prompting ``hivemind/coder``.
  3. Unified Win11 / POSIX / <60s VS Code Quickstart Bundle Verifier
     (``verify_unified_quickstart_bundle`` + bounded atomic ``0600`` ``BundleCesQueue``):
     verifies AGPL-3.0 headers, zero non-stdlib AST imports (allowing guarded optional ``httpx`` in
     ``hivemind_connect.py``), lateral-movement compliance, byte-for-byte SHA-256
     ``GET /scripts/<name>`` parity across manifest scripts, and ``join_hivemind.ps1`` /
     ``join_hivemind_campaign.sh`` launcher drift diagnostics; validates isolated ``< 60 s`` Dev VS
     Code ``0600`` Continue config & secret-free ``.vscode/settings.json`` + clean uninstall; runs
     a real subprocess ``< 1.5 s`` (1500.0 milliseconds) Steam/DX12 GPU yield & ``<= 70.0 C`` /
     ``>= 2048 MiB`` VRAM preflight; and optionally dispatches outbound
     ``POST /v1/hivemind/opentrainer/heartbeat`` and zero-PII ``Idempotency-Key``
     ``POST /v1/hivemind/onboarding/feedback`` via ``BundleCesQueue``.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen science under AGPL-3.0 (zero monetary value).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import hmac
import importlib
import importlib.util
import inspect
import io
import ipaddress
import json
import math
import os
import platform
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows relies on per-path RLock + atomic os.replace
    _fcntl = None  # type: ignore[assignment]

SCHEMA = "hivemind.noel_cd_deploy_and_quickstart_bundle_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_cd_deploy_and_quickstart_bundle_receipt"
CD_GATE_OBJECT = "hivemind.noel_cd_deploy_bundle_gate"
QUICKSTART_BUNDLE_OBJECT = "hivemind.noel_unified_quickstart_bundle"
UNIFIED_OBJECT = "hivemind.noel_cd_and_quickstart_unified_gate"
QUEUE_SCHEMA = "hivemind.bundle_ces_queue/1"

MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
FLEET_OBJECT = "hivemind.producer_fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
OPENTRAINER_HB_ACK_OBJECT = "hivemind.opentrainer_heartbeat_ack"

CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
AGPL_HEADER_LINE = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_REL = ".hivemind/bundle_ces_queue.json"
DEFAULT_CREDENTIALS_REL = Path(".hivemind") / "credentials.json"
DEFAULT_CONTINUE_REL = Path(".continue") / "hivemind_provider.yaml"

HTTP_TIMEOUT_S = 5.0
SUBPROCESS_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
MAX_SCRIPT_FILE_BYTES = 4 * 1024 * 1024

DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
DEFAULT_YIELD_BUDGET_MS = 1500.0
DEV_VSCODE_SLO_SECONDS = 60.0
MIN_REPORTABLE_VRAM_TOTAL_MIB = 512
MAX_PENDING_QUEUE_ENTRIES = 128
MAX_DELIVERED_IDS = 512
MAX_QUARANTINED_ENTRIES = 64
MAX_RETRY_COUNT = 16
NON_WORKER_GPU_UTIL_THRESHOLD_PCT = 15.0

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
VALID_ROLES: frozenset[str] = frozenset({"consumer", "producer", "prosumer", "trainer", "admin"})
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_DISPLAY_URLS: frozenset[str] = frozenset({"https://github.com/Vxtzq/CrowdGPT"})
ALLOWED_SOURCE_URL_WILDCARD_HOSTS: frozenset[str] = frozenset({"*.run.app", "*.googleapis.com"})
ALLOWED_DOC_PLACEHOLDERS: frozenset[str] = frozenset({"host", "gateway", "your_public_url", "127.0.0.1:port"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset(
    {
        "/v1/hivemind/opentrainer/heartbeat",
        "/v1/hivemind/onboarding/feedback",
        "/v1/hivemind/register",
    }
)
RETRIABLE_CES_STATUSES: frozenset[int] = frozenset({0, 404, 408, 429})
PERMANENT_CES_STATUSES: frozenset[int] = frozenset({400, 401, 403, 413, 422})

CLOUD_RUN_CORE_PATHS: tuple[str, ...] = (
    "deploy/gcp_openrouter/Dockerfile",
    "deploy/gcp_openrouter/deploy.sh",
    "deploy/gcp_openrouter/entrypoint.sh",
    "deploy/gcp_openrouter/requirements.txt",
    "deploy/gcp_openrouter/server.py",
)

HOLIDAY_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
CD_LIVE_PROBE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "models"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "feedback"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
    ("/v1/hivemind/raid/status", "json", "raid"),
)
ASGI_PROBE_ROUTES: tuple[tuple[str, str], ...] = (
    ("/health", "json"),
    ("/v1/models", "json"),
    ("/noel", "html"),
    ("/quick", "html"),
    ("/raid-noel", "html"),
    ("/arena-noel", "html"),
    ("/v1/hivemind/onboarding/manifest", "json"),
    ("/v1/hivemind/onboarding/feedback/summary", "json"),
    ("/v1/hivemind/producers/fleet", "json"),
    ("/v1/hivemind/raid/status", "json"),
)
EXPECTED_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid",
    "/raid-noel",
    "/v1/hivemind/arena/summary",
    "/v1/hivemind/arena/vote",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)
EXPECTED_MANIFEST_SCRIPTS: tuple[str, ...] = (
    "hivemind_boinc_idle_worker.py",
    "hivemind_connect.py",
    "hivemind_doctor.py",
    "hivemind_opentrainer_agent.py",
    "hivemind_producer_agent.py",
    "hivemind_producer_autostart.py",
    "hivemind_producer_shield.py",
    "hivemind_quickstart.py",
    "hivemind_volunteer_config.py",
    "hivemind_volunteer_preflight.py",
    "join_hivemind.ps1",
    "noel_canary_cohort_rehearsal.py",
    "noel_cloud_run_parity_gate.py",
    "noel_gamer_whisper_selftest.py",
    "noel_post_setup_probe.py",
    "noel_scale_load_harness.py",
    "noel_vscode_autoconfig.py",
)
QUICKSTART_CORE_PY_SCRIPTS: tuple[str, ...] = (
    "hivemind_boinc_idle_worker.py",
    "hivemind_connect.py",
    "hivemind_doctor.py",
    "hivemind_producer_autostart.py",
    "hivemind_quickstart.py",
    "hivemind_volunteer_config.py",
    "hivemind_volunteer_preflight.py",
    "noel_canary_cohort_rehearsal.py",
    "noel_cloud_run_parity_gate.py",
    "noel_gamer_whisper_selftest.py",
    "noel_post_setup_probe.py",
    "noel_scale_load_harness.py",
    "noel_vscode_autoconfig.py",
    "noel_vscode_quick_preset.py",
    "noel_deploy_smoke_and_win11_verifier.py",
    "noel_posix_launcher_and_smoke_verifier.py",
    "noel_cd_deploy_and_quickstart_bundle_verifier.py",
)
MANIFEST_DAEMON_PY_SCRIPTS: frozenset[str] = frozenset(
    {
        "hivemind_opentrainer_agent.py",
        "hivemind_producer_agent.py",
        "hivemind_producer_shield.py",
    }
)
EXPECTED_FEEDBACK_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "object",
        "total_responses",
        "mean_ease_rating",
        "zero_game_lag_ratio",
        "by_profile",
    }
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
        "starfield.exe",
        "witcher3.exe",
        "gta5.exe",
        "r5apex.exe",
        "palworld-win64-shipping.exe",
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

INSPECTED_ENV_KEYS: frozenset[str] = frozenset(
    {
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
    }
)
FORBIDDEN_LATERAL_IMPORTS: frozenset[str] = frozenset(
    {
        "paramiko",
        "fabric",
        "asyncssh",
        "telnetlib",
        "smbprotocol",
        "pywinrm",
        "impacket",
        "scapy",
        "nmap",
        "ftplib",
    }
)
FORBIDDEN_SUBPROCESS_BINARIES: frozenset[str] = frozenset(
    {
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

_CORP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("amadeus", re.compile(r"amadeus", re.IGNORECASE)),
    ("corp", re.compile(r"(?:^|[^a-z])corp(?:orate|net|sys|s)?(?:$|[^a-z])|[a-z0-9]corp(?:$|[^a-z])", re.IGNORECASE)),
    ("enterprise", re.compile(r"enterprise", re.IGNORECASE)),
    ("intranet", re.compile(r"intranet", re.IGNORECASE)),
    ("internal", re.compile(r"(?:^|[^a-z])internal(?:$|[^a-z])|[a-z0-9]internal(?:$|[^a-z])", re.IGNORECASE)),
    ("zscaler", re.compile(r"zscaler", re.IGNORECASE)),
    ("netskope", re.compile(r"netskope", re.IGNORECASE)),
    ("bluecoat", re.compile(r"bluecoat", re.IGNORECASE)),
    ("fortinet", re.compile(r"fortinet", re.IGNORECASE)),
    ("paloalto", re.compile(r"paloalto|palo[-_.]?alto", re.IGNORECASE)),
)

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\b(?:hm(?:_live_|-)|sk-(?:ant-|or-v1-|proj-)?|gh[pousr]_|hf_)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_RAW_HM_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])hm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{4,}\b")
_MASKED_HM_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])(hm(?:_live_|-))\*\*\*[A-Za-z0-9_-]{3,}(?![A-Za-z0-9_-])")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)")
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_STATIC_FIG_UNIT_RE = re.compile(r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)")
_HTML_NONCE_ATTR_RE = re.compile(r"""\bnonce\s*=\s*(?:"([^"]+)"|'([^']+)')""", re.IGNORECASE)
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+/=-]{8,128})'")
_UNSAFE_DOM_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|eval)\s*(?:=|\()|document\.write\s*\(|new\s+Function\s*\(",
    re.IGNORECASE,
)
_PS1_SH_UNSAFE_RE = re.compile(
    r"(?i)(?:\bInvoke-Expression\b|\biex\b|\.DownloadString\s*\(|cu"
    + r"rl[^\n]*\|\s*(?:ba)?sh|wg"
    + r"et[^\n]*\|\s*(?:ba)?sh|-EncodedCommand\b)"
)
_TROJAN_BIDI_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
_RUN_APP_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]+)*\.run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.googleapis\.com$")
_SAFE_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_]+\.(?:py|ps1)$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]+$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_CLOUD_RUN_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_VALUE_MARKERS: tuple[str, ...] = ("secret", "token", "password", "passwd", "apikey", "bearer", "aiza")

_SERVER_LOAD_LOCK = threading.Lock()
_SCRIPT_AUDIT_LOCK = threading.Lock()
_SCRIPT_AUDIT_CACHE: dict[tuple[str, str, bool, bool, bool], dict[str, Any]] = {}
_BASH_SYNTAX_CACHE: dict[tuple[str, str], bool] = {}
_HTTP_OPENER: urllib.request.OpenerDirector | None = None

HttpFn = Callable[..., Any]
RunnerFn = Callable[..., Any]


class VerifierError(ValueError):
    """Raised when verifier arguments, allowlists, queues, or safety invariants fail."""


BundleVerifierError = VerifierError


@dataclass(frozen=True)
class GpuSnapshot:
    """Measured local NVIDIA GPU telemetry without device serials or invented values."""

    present: bool
    parse_error: bool = False
    name: str | None = None
    total_vram_mib: int | None = None
    free_vram_mib: int | None = None
    used_vram_mib: int | None = None
    temp_c: float | None = None
    util_pct: float = 0.0
    reason: str = "no_nvidia_gpu"


def _msg(lang_code: str, en: str, fr: str) -> str:
    return fr if lang_code == "fr" else en


def normalize_lang(raw: str | None) -> str:
    """Normalize language code to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise VerifierError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise VerifierError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_profile(raw: str | None) -> str:
    """Normalize CES feedback profile to ``'dev'`` or ``'gamer'``."""
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise VerifierError("Profile must be 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_PROFILES:
        raise VerifierError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def is_valid_hivemind_key(raw_key: object) -> bool:
    """Return True iff ``raw_key`` is a plausible ``hm-...`` or ``hm_live_...`` key."""
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key while keeping only the 4-char suffix."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if not is_valid_hivemind_key(cleaned):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    return f"{prefix}***{cleaned[-4:]}"


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and ``hm-***...`` suffixes."""
    if not isinstance(text, str):
        raise VerifierError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def redact_home_path(
    target: Path | str,
    home: Path | str | None = None,
    workspace: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> str:
    """Replace OS user home prefixes with ``'~'`` and map workspace/repo paths safely."""
    raw = str(target)
    bases: list[tuple[Path, str]] = []
    if home is not None:
        bases.append((Path(home), "~"))
    if workspace is not None:
        bases.append((Path(workspace), "<workspace>"))
    bases.append((Path(repo_root) if repo_root is not None else REPO_ROOT, ""))
    if home is None:
        bases.append((Path.home(), "~"))
    for base_dir, label in bases:
        base_str = str(base_dir)
        if not base_str or base_str in ("/", "\\"):
            continue
        for variant in (base_str, base_str.replace("\\", "/"), base_str.replace("/", "\\")):
            if variant and raw.startswith(variant):
                suffix = raw[len(variant) :]
                if not suffix or suffix.startswith(("/", "\\")):
                    mapped = suffix.lstrip("/\\") if not label else (label + suffix.replace("\\", "/"))
                    return _PYTEST_USER_DIR_RE.sub("pytest-of-user", mapped or ".")
    scrubbed = _HOME_PATH_RE.sub("~", raw)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    return scrubbed.replace("\\", "/") if scrubbed != raw else raw


def sanitize_text_for_display(
    text: str,
    *,
    home: Path | str | None = None,
    workspace: Path | str | None = None,
) -> str:
    """Scrub raw keys, Bearer tokens, emails, non-loopback IPv4s, and OS home paths."""
    scrubbed = str(text)
    resolved_home = str(home if home is not None else Path.home())
    if resolved_home and resolved_home not in ("/", "\\"):
        for variant in (resolved_home, resolved_home.replace("\\", "/"), resolved_home.replace("/", "\\")):
            if variant and variant in scrubbed:
                scrubbed = scrubbed.replace(variant, "~")
    scrubbed = redact_home_path(scrubbed, home=home, workspace=workspace)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(lambda m: m.group(0) if m.group(0).startswith("127.") else "<redacted-ip>", scrubbed)
    return scrubbed


def _collect_json_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_json_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_json_keys(item))
    return found


def scan_output_safety(text: str) -> list[str]:
    """Verify rendered JSON, CLI, or Markdown has 0 PII, raw keys, score digits, or static figures."""
    if not isinstance(text, str):
        raise VerifierError("scan_output_safety requires a string.")
    issues: list[str] = []
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(text) or _RAW_HM_KEY_RE.search(text):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _HOME_PATH_RE.search(text):
        issues.append("unredacted_home_path_detected")
    if _FORBIDDEN_SCORE_KEY_RE.search(text):
        issues.append("forbidden_score_key_detected")
    if _STATIC_FIG_UNIT_RE.search(strip_masked_keys_and_nonces(text)):
        issues.append("static_figure_detected")
    if _TROJAN_BIDI_RE.search(text):
        issues.append("trojan_bidi_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if not ip_hit.startswith("127."):
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise VerifierError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise VerifierError("Forbidden JSON key detected in receipt.")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def sanitize_build_sha(raw: object) -> str:
    """Sanitize ``build_sha`` to lowercase 6..64 alphanumeric/hex or ``'unknown'``."""
    if not isinstance(raw, str):
        return "unknown"
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or low == "unknown"
        or not _BUILD_SHA_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(m in low for m in _SECRET_VALUE_MARKERS)
    ):
        return "unknown"
    return low


def validate_expected_sha(raw: str) -> str:
    """Validate explicit ``expected_sha`` parameter."""
    cleaned = sanitize_build_sha(raw)
    if cleaned == "unknown":
        raise VerifierError("expected_sha must be 6..64 alphanumeric chars (never 'unknown' or a secret).")
    return cleaned


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Compare live ``build_sha`` against ``expected_sha`` (exact or hex prefix)."""
    live = sanitize_build_sha(live_sha)
    exp = sanitize_build_sha(expected_sha)
    if live == "unknown" or exp == "unknown":
        return False
    if live == exp:
        return True
    if _HEX_SHA_RE.fullmatch(live) and _HEX_SHA_RE.fullmatch(exp):
        return live.startswith(exp) or exp.startswith(live)
    return False


def sanitize_cloud_run_revision(raw: object) -> str | None:
    """Sanitize Cloud Run revision identifier without exposing secrets."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or not _CLOUD_RUN_REVISION_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(m in low for m in _SECRET_VALUE_MARKERS)
    ):
        return None
    return cleaned


def sanitize_host_label(raw: str | None) -> str:
    """Return a clean non-PII OpenTrainer host label."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "personal-pc").strip()).strip("-.")
    return (cleaned or "personal-pc")[:64]


def is_allowed_outbound_host(host: str) -> bool:
    """Return True iff ``host`` matches ``*.run.app``, ``*.googleapis.com``, HF datasets, or loopback."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate gateway origin URL to ``https://*.run.app``, ``https://*.googleapis.com``, or loopback."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise VerifierError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise VerifierError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise VerifierError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise VerifierError("Gateway URL must not embed credentials, query strings, or fragments.")
    if parts.path not in ("", "/"):
        raise VerifierError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise VerifierError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise VerifierError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise VerifierError(f"Untrusted host {host!r}; not in HiveMind outbound allowlist.")
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise VerifierError("Loopback URL must use http or https.")
    elif parts.scheme != "https":
        raise VerifierError("Remote allowlisted URL must use https.")
    return cleaned


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Enforce allowlisted host, zero LLM inference/prompts, and allowlisted POST paths."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise VerifierError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise VerifierError("Request URL must not contain whitespace or control chars.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise VerifierError("Request URL must not embed credentials.")
    validate_gateway_url(f"{parts.scheme}://{parts.netloc}")
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise VerifierError(f"Zero-inference guard blocked inference endpoint {parts.path!r}.")
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise VerifierError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise VerifierError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in ALLOWED_POST_PATHS):
        return cleaned
    raise VerifierError(f"Zero-inference guard blocked HTTP {verb!r} to {parts.path!r}.")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects off allowlisted hosts or onto inference endpoints."""

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


def _get_http_opener() -> urllib.request.OpenerDirector:
    global _HTTP_OPENER
    with _SERVER_LOAD_LOCK:
        if _HTTP_OPENER is None:
            _HTTP_OPENER = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                _SafeRedirectHandler(),
            )
        return _HTTP_OPENER


def _parse_non_json_bytes(
    status_code: int,
    raw_bytes: bytes,
    hdrs: dict[str, str],
) -> tuple[int, dict[str, Any]]:
    try:
        text = raw_bytes.decode("utf-8")
        utf8_valid = True
    except UnicodeDecodeError:
        text = raw_bytes.decode("utf-8", errors="replace")
        utf8_valid = False
    return int(status_code), {
        "raw_bytes": raw_bytes,
        "utf8_valid": utf8_valid,
        "byte_length": len(raw_bytes),
        "html": text,
        "text": text,
        "html_length": len(text),
        "headers": hdrs,
        "csp": hdrs.get("content-security-policy", ""),
        "x_script_sha256": hdrs.get("x-script-sha256", ""),
    }


def _parse_http_raw_response(
    status_code: int,
    raw_bytes: bytes,
    raw_headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
        raise VerifierError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
    hdrs = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    ctype = hdrs.get("content-type", "").lower()
    looks_text = "text/html" in ctype or "text/plain" in ctype or "text/x-python" in ctype
    if "application/json" in ctype or (not looks_text and raw_bytes.lstrip().startswith((b"{", b"["))):
        text = raw_bytes.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": "invalid_json"}
        return (int(status_code), dict(parsed)) if isinstance(parsed, dict) else (int(status_code), {"data": parsed})
    return _parse_non_json_bytes(status_code, raw_bytes, hdrs)


def default_http_request(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib HTTP helper (``ProxyHandler({})``) enforcing host/path safety."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(verb, url, payload=payload)
    if verb in ("GET", "HEAD") and payload is not None:
        raise VerifierError("GET/HEAD requests must never attach a body payload.")
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/x-python;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-cd-quickstart-verifier/1.0",
    }
    data_bytes: bytes | None = None
    if payload is not None:
        data_bytes = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    if headers:
        for hk, hv in headers.items():
            ks, vs = str(hk), str(hv)
            if any(ch in ks or ch in vs for ch in ("\r", "\n", "\x00")):
                raise VerifierError("HTTP headers must not contain CR/LF or NUL.")
            req_headers[ks] = vs
    req = urllib.request.Request(clean_url, data=data_bytes, headers=req_headers, method=verb)
    bounded = min(max(0.2, float(timeout_s)), 15.0)
    try:
        with _get_http_opener().open(req, timeout=bounded) as resp:
            raw_hdrs = dict(resp.headers.items()) if resp.headers is not None else {}
            return _parse_http_raw_response(int(getattr(resp, "status", 200)), resp.read(MAX_HTTP_BODY_BYTES + 1), raw_hdrs)
    except urllib.error.HTTPError as exc:
        err_hdrs = dict(exc.headers.items()) if exc.headers is not None else {}
        err_bytes = exc.read(131_072) if exc.fp is not None else b""
        return _parse_http_raw_response(int(exc.code), err_bytes, err_hdrs)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise VerifierError("http_fn must return (status_code, body) or (status_code, headers, body).")
    if len(raw_ret) == 2:
        status_code, body = raw_ret
        hdr_map: dict[str, str] = {}
    else:
        status_code, raw_hdrs, body = raw_ret
        if isinstance(raw_hdrs, (str, bytes, bytearray)) and isinstance(body, Mapping):
            body, raw_hdrs = raw_hdrs, body
        hdr_map = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()} if isinstance(raw_hdrs, Mapping) else {}
    if isinstance(body, Mapping):
        out = dict(body)
        if hdr_map and ("raw_bytes" in out or "html" in out or "text" in out):
            existing = out.get("headers") if isinstance(out.get("headers"), Mapping) else {}
            merged = {**hdr_map, **{str(k).strip().lower(): str(v) for k, v in existing.items()}}
            out["headers"] = merged
            out.setdefault("csp", merged.get("content-security-policy", ""))
            out.setdefault("x_script_sha256", merged.get("x-script-sha256", ""))
        return int(status_code), out
    if isinstance(body, (bytes, bytearray)):
        return _parse_http_raw_response(int(status_code), bytes(body), hdr_map)
    if isinstance(body, str):
        ctype = hdr_map.get("content-type", "").lower()
        looks_text = "text/html" in ctype or "text/plain" in ctype or "text/x-python" in ctype
        if not looks_text and body.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(body)
                return (int(status_code), dict(parsed)) if isinstance(parsed, Mapping) else (int(status_code), {"data": parsed})
            except json.JSONDecodeError:
                pass
        return _parse_non_json_bytes(int(status_code), body.encode("utf-8"), hdr_map)
    return int(status_code), {"data": body}


def _invoke_http_fn(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(verb, url, payload=payload)
    sig = inspect.signature(http_fn)
    params = list(sig.parameters.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    pnames = {p.name for p in params}
    pos_params = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    hdr_arg = dict(headers) if headers is not None else None
    body_arg = dict(payload) if payload is not None else None
    try:
        if has_varkw:
            raw = http_fn(verb, clean_url, payload=body_arg, headers=hdr_arg, timeout_s=timeout_s)
        else:
            kwargs: dict[str, Any] = {}
            if "payload" in pnames:
                kwargs["payload"] = body_arg
            if "headers" in pnames:
                kwargs["headers"] = hdr_arg
            if "timeout_s" in pnames:
                kwargs["timeout_s"] = timeout_s
            if kwargs or len(pos_params) <= 2:
                raw = http_fn(verb, clean_url, **kwargs)
            elif len(pos_params) == 3:
                raw = http_fn(verb, clean_url, body_arg)
            else:
                raw = http_fn(verb, clean_url, body_arg, hdr_arg)
    except VerifierError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    return _normalize_http_return(raw)


def default_subprocess_runner(
    cmd: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> tuple[int, str]:
    """Execute a safe list-argv local command (``shell=False``, never invoking live ``gcloud``)."""
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise VerifierError("Subprocess command must be a non-empty sequence.")
    clean: list[str] = []
    for item in cmd:
        val = str(item)
        if any(ch in val for ch in ("\x00", "\r", "\n")):
            raise VerifierError("Subprocess argument contains invalid control characters.")
        clean.append(val)
    exe_name = Path(clean[0]).name.lower()
    if exe_name in FORBIDDEN_SUBPROCESS_BINARIES or exe_name in ("gcloud", "gcloud.cmd", "gcloud.exe"):
        raise VerifierError(f"Refused forbidden subprocess binary {exe_name!r}.")
    bounded = min(max(0.1, float(timeout_s)), 15.0)
    try:
        done = subprocess.run(
            clean,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded,
            check=False,
            shell=False,
            env=dict(env) if env is not None else None,
            cwd=str(cwd) if cwd is not None else None,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return 127, ""
    return int(done.returncode), (done.stdout or "") + (("\n" + done.stderr) if done.stderr else "")


def _invoke_runner(
    runner: RunnerFn,
    argv: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> tuple[int, str]:
    sig = inspect.signature(runner)
    params = list(sig.parameters.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    pnames = {p.name for p in params}
    pos_params = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    kwargs: dict[str, Any] = {}
    if has_varkw or "env" in pnames:
        if env is not None:
            kwargs["env"] = env
    if has_varkw or "cwd" in pnames:
        if cwd is not None:
            kwargs["cwd"] = cwd
    if has_varargs or len(pos_params) >= 2:
        res = runner(list(argv), timeout_s, **kwargs)
    else:
        res = runner(list(argv), **kwargs)
    if isinstance(res, tuple) and len(res) >= 2:
        return int(res[0]), str(res[1] or "")
    raise VerifierError("Subprocess runner must return (returncode, output).")


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def _extract_git_remote_urls(cfg_text: str) -> list[str]:
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text or ""):
        cand = (m_kv or m_sec).strip().strip('"').strip("'")
        if cand:
            urls.append(cand)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    configs: list[Path] = []
    for root in (workspace, *workspace.parents):
        git_entry = root / ".git"
        if git_entry.is_dir():
            cfg = git_entry / "config"
            if cfg.is_file():
                configs.append(cfg)
            break
        if git_entry.is_file():
            try:
                lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            if lines and lines[0].strip().lower().startswith("gitdir:"):
                raw_gd = lines[0].split(":", 1)[1].strip()
                gd = Path(raw_gd) if Path(raw_gd).is_absolute() else (root / raw_gd).resolve()
                if (gd / "config").is_file():
                    configs.append(gd / "config")
                cdir_file = gd / "commondir"
                if cdir_file.is_file():
                    with suppress(OSError):
                        crel = cdir_file.read_text(encoding="utf-8", errors="replace").strip()
                        if crel:
                            ccfg = (gd / crel).resolve() / "config"
                            if ccfg.is_file() and ccfg not in configs:
                                configs.append(ccfg)
            break
    return configs


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
    home: Path | str | None = None,
    extra_paths: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only gate (0 network calls, 0 writes, 0 subprocesses, 0 raw leaks)."""
    lang_code = normalize_lang(lang)
    env_map = os.environ if environ is None else environ
    ws = Path(cwd) if cwd is not None else Path.cwd()
    if home is not None:
        home_path = Path(home)
    elif environ is not None:
        env_home = str(env_map.get("HOME") or env_map.get("USERPROFILE") or "").strip()
        home_path = Path(env_home) if env_home else ws
    else:
        home_path = Path.home()

    indicators: list[str] = []
    markers: set[str] = set()

    def _add(tag: str, marker: str | None = None) -> None:
        if tag not in indicators:
            indicators.append(tag)
        if marker:
            markers.add(marker)

    for raw_k, raw_v in env_map.items():
        kup = str(raw_k).strip().upper()
        vstr = str(raw_v or "").strip()
        if not kup or not vstr:
            continue
        if kup.startswith("AMADEUS"):
            _add("env:AMADEUS*", "amadeus")
            markers.update(_find_corporate_markers(vstr))
        elif kup == "ZSCALER_CERT_PATH":
            _add("env:ZSCALER_CERT_PATH", "zscaler")
        elif kup == "USERDNSDOMAIN":
            hits = _find_corporate_markers(vstr)
            _add("env:USERDNSDOMAIN", hits[0] if hits else "corp")
            markers.update(hits)
        elif kup in INSPECTED_ENV_KEYS:
            for hit in _find_corporate_markers(vstr):
                _add(f"env:{kup}:{hit}", hit)

    path_targets: list[tuple[str, Path]] = [("workspace_path", ws), ("home_path", home_path)]
    for idx, extra in enumerate(extra_paths):
        if extra is not None:
            path_targets.append((f"extra_path_{idx}", Path(extra)))

    for label, p_obj in path_targets:
        variants = [str(p_obj)]
        with suppress(OSError):
            res = str(p_obj.resolve())
            if res not in variants:
                variants.append(res)
        for p_txt in variants:
            for hit in _find_corporate_markers(p_txt):
                _add(f"{label}:{hit}", hit)

    if check_git_remote:
        seen_cfgs: set[Path] = set()
        git_roots = [ws, *(Path(ep) for ep in extra_paths if ep is not None)]
        for root_dir in git_roots:
            for cfg in _resolve_git_config_files(root_dir):
                if cfg in seen_cfgs:
                    continue
                seen_cfgs.add(cfg)
                with suppress(OSError):
                    for rurl in _extract_git_remote_urls(cfg.read_text(encoding="utf-8", errors="replace")):
                        for hit in _find_corporate_markers(rurl):
                            _add(f"git_remote:{hit}", hit)

    ok = len(indicators) == 0
    notice = COMPLIANCE_NOTICE_FR if lang_code == "fr" else COMPLIANCE_NOTICE_EN
    summary = (
        _msg(
            lang_code,
            f"Personal-PC compliance verified locally ({FINANCIAL_PROMISE}). {notice}",
            f"Conformité PC personnel vérifiée localement ({FINANCIAL_PROMISE}). {notice}",
        )
        if ok
        else _msg(
            lang_code,
            f"COMPLIANCE REFUSED ({', '.join(indicators)}): corporate environment detected. {COMPLIANCE_NOTICE_BILINGUAL}",
            f"REFUS DE CONFORMITÉ ({', '.join(indicators)}) : environnement d'entreprise détecté. {COMPLIANCE_NOTICE_BILINGUAL}",
        )
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": "hivemind.personal_pc_compliance",
            "lang": lang_code,
            "compliance_ok": ok,
            "personal_pc_only": True,
            "financial_promise": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "exit_code": 0 if ok else 2,
            "indicators": indicators,
            "matched_markers": sorted(markers),
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "summary": summary,
            "compliance_notice": notice,
        }
    )


def audit_html_safety(html_text: str, csp_header: str = "") -> dict[str, Any]:
    """Audit HTML page for CSP default-src 'none', nonces, 0 DOM sinks, 0 static figures, and compliance banners."""
    attr_nonces = [m.group(1) or m.group(2) for m in _HTML_NONCE_ATTR_RE.finditer(html_text or "")]
    csp_nonces = _CSP_INLINE_NONCE_RE.findall(csp_header or "")
    csp_low = (csp_header or "").lower()
    has_default_src_none = "default-src 'none'" in csp_low
    csp_unsafe = "'unsafe-inline'" in csp_low or "'unsafe-eval'" in csp_low
    primary_nonce = csp_nonces[0] if csp_nonces else (attr_nonces[0] if attr_nonces else "")
    csp_matches_attr = bool(csp_nonces and len(set(csp_nonces)) == 1 and csp_nonces[0] in attr_nonces)
    unsafe_sinks = sorted({m.group(0).strip() for m in _UNSAFE_DOM_RE.finditer(html_text or "")})
    stripped = strip_masked_keys_and_nonces(html_text or "")
    static_figs = _STATIC_FIG_UNIT_RE.findall(stripped)
    low = stripped.lower()
    has_personal_banner = ("personal" in low or "personnel" in low) and ("0.00" in low or "0,00" in low)
    has_agpl = "agpl-3.0" in low or "agpl" in low
    return {
        "csp_present": bool(csp_header),
        "default_src_none": has_default_src_none,
        "csp_unsafe_absent": not csp_unsafe,
        "csp_matches_attr": csp_matches_attr,
        "nonce": primary_nonce,
        "nonce_count": len(attr_nonces),
        "unsafe_dom_sinks": unsafe_sinks,
        "static_figures": static_figs,
        "has_personal_pc_banner": has_personal_banner,
        "has_agpl_banner": has_agpl,
        "ok": bool(
            has_default_src_none
            and not csp_unsafe
            and csp_matches_attr
            and primary_nonce
            and not unsafe_sinks
            and not static_figs
            and has_personal_banner
        ),
    }


def _check_zero_grant_var(active_script: str, var_name: str) -> bool:
    has_zero = bool(re.search(rf"\$\{{{var_name}:-0(?:\.0+)?\}}|{var_name}=0(?:\.0+)?\b", active_script))
    has_nonzero = bool(
        re.search(rf"\$\{{{var_name}:-[1-9][0-9.]*\}}", active_script)
        or re.search(rf'(?:^|[,\s"\']){var_name}=[1-9][0-9.]*\b', active_script, re.MULTILINE)
    )
    return has_zero and not has_nonzero


def audit_deploy_bundle_finops(
    repo_root: Path | str | None = None,
    *,
    expected_sha: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Audit ``deploy/gcp_openrouter/{deploy.sh,Dockerfile,entrypoint.sh,requirements.txt,server.py}``."""
    lang_code = normalize_lang(lang)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    deploy_dir = root / "deploy" / "gcp_openrouter"

    files_present: dict[str, bool] = {}
    texts: dict[str, str] = {}
    for rel in CLOUD_RUN_CORE_PATHS:
        fp = root / rel
        name = Path(rel).name
        if fp.is_file():
            files_present[name] = True
            texts[name] = fp.read_text(encoding="utf-8", errors="replace")
        else:
            files_present[name] = False
            texts[name] = ""

    deploy_active = "\n".join(
        ln for ln in texts["deploy.sh"].splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    )
    dockerfile_txt = texts["Dockerfile"]
    entrypoint_active = "\n".join(
        ln for ln in texts["entrypoint.sh"].splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    )
    reqs_txt = texts["requirements.txt"]
    server_txt = texts["server.py"]

    server_ast_ok = False
    if files_present["server.py"]:
        try:
            stree = ast.parse(server_txt, filename="deploy/gcp_openrouter/server.py")
            has_build = any(isinstance(n, ast.FunctionDef) and n.name == "build_app" for n in ast.walk(stree))
            server_ast_ok = bool(
                has_build and "HIVEMIND_ROUTER_SECRET" in server_txt and "create_router_app" in server_txt
            )
        except SyntaxError:
            server_ast_ok = False

    conc_matches = re.findall(r"--concurrency(?:=|\s+)(\d+)\b", deploy_active)
    litestream_sha_match = re.search(r"ARG\s+LITESTREAM_SHA256=([0-9a-f]{64})\b", dockerfile_txt)
    invariants: dict[str, bool] = {
        "all_core_files_present": all(files_present.values()),
        "min_instances_zero": bool(
            re.search(r"--min-instances(?:=|\s+)0\b", deploy_active)
            and not re.search(r"--min-instances(?:=|\s+)[1-9]\d*\b", deploy_active)
        ),
        "max_instances_one": bool(
            re.search(r"--max-instances(?:=|\s+)1\b", deploy_active)
            and not re.search(r"--max-instances(?:=|\s+)(?:0|[2-9]|\d{2,})\b", deploy_active)
        ),
        "no_cpu_throttling": bool(
            re.search(r"--no-cpu-throttling\b", deploy_active)
            and not re.search(r"(?<!no-)--cpu-throttling\b", deploy_active)
        ),
        "concurrency_80": bool(conc_matches and all(v == "80" for v in conc_matches)),
        "litestream_sha256_pinned": bool(
            litestream_sha_match is not None
            and "LITESTREAM_VERSION=" in dockerfile_txt
            and "sha256sum -c -" in dockerfile_txt
            and "litestream-v" in dockerfile_txt
        ),
        "base_image_sha256_pinned": bool(re.search(r"FROM\s+python:3\.12-slim@sha256:[0-9a-f]{64}", dockerfile_txt)),
        "dockerfile_non_root_app": bool("USER app" in dockerfile_txt and "HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS=1" in dockerfile_txt),
        "litestream_replica_url": bool(
            "HIVEMIND_ROUTER_REPLICA_URL=" in deploy_active
            and "gcs://" in deploy_active
            and "HIVEMIND_ROUTER_REPLICA_URL" in entrypoint_active
            and "litestream restore" in entrypoint_active
            and "litestream replicate" in entrypoint_active
            and "--workers 1" in entrypoint_active
        ),
        "secret_manager_router_secret": bool(
            re.search(r"--set-secrets\s+[\"']?[^\"'\n]*HIVEMIND_ROUTER_SECRET=", deploy_active)
            and "HIVEMIND_ROUTER_SECRET" in server_txt
        ),
        "zero_starter_and_bonus_grants": bool(
            _check_zero_grant_var(deploy_active, "HIVEMIND_STARTER_GRANT_HMT")
            and _check_zero_grant_var(deploy_active, "HIVEMIND_REGISTRATION_BONUS_HMT")
            and _check_zero_grant_var(deploy_active, "HIVEMIND_ALLOW_BONUSES")
            and _check_zero_grant_var(deploy_active, "HIVEMIND_PARTNER_TESTING_GRANTS")
        ),
        "build_sha_injection": bool(
            "HIVEMIND_BUILD_SHA=" in deploy_active
            and "--update-env-vars" in deploy_active
            and "--set-env-vars" not in deploy_active
            and "sanitize_build_sha" in deploy_active
        ),
        "requirements_pinned": bool(
            files_present["requirements.txt"]
            and all(pkg in reqs_txt for pkg in ("fastapi==", "uvicorn", "pydantic==", "httpx=="))
        ),
        "server_py_contract": server_ast_ok,
    }
    failed_invariants = [k for k, v in invariants.items() if not v]
    finops_ok = len(failed_invariants) == 0
    sha_prefix = f"HIVEMIND_BUILD_SHA={expected_sha} " if expected_sha else ""
    dry_run_cmd = f"{sha_prefix}HIVEMIND_DEPLOY_DRY_RUN=1 bash deploy/gcp_openrouter/deploy.sh --dry-run"
    return {
        "deploy_dir": redact_home_path(deploy_dir, repo_root=root),
        "files_present": files_present,
        "finops_ok": finops_ok,
        "invariants": invariants,
        "failed_invariants": failed_invariants,
        "dry_run_command": dry_run_cmd,
        "litestream_sha256_prefix": litestream_sha_match.group(1)[:12] if litestream_sha_match else None,
        "note": _msg(
            lang_code,
            f"Cloud Run deploy bundle FinOps audit ({'PASS' if finops_ok else 'FAIL'}; 0 cloud mutations).",
            f"Audit FinOps du bundle Cloud Run ({'OK' if finops_ok else 'ÉCHEC'} ; 0 mutation cloud).",
        ),
    }


def compute_cd_bundle_fingerprint(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Compute deterministic SHA-256 fingerprint across Cloud Run-deployed files."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    router_dir = root / "src" / "hivemind" / "router"
    router_paths: list[str] = []
    if router_dir.is_dir():
        for entry in sorted(router_dir.glob("*.py")):
            if entry.is_file():
                router_paths.append(f"src/hivemind/router/{entry.name}")
        measured = router_dir / "measured_results.json"
        if measured.is_file():
            router_paths.append("src/hivemind/router/measured_results.json")

    script_paths = [f"scripts/{name}" for name in EXPECTED_MANIFEST_SCRIPTS]
    tracked = sorted(set(CLOUD_RUN_CORE_PATHS) | set(router_paths) | set(script_paths))

    file_hashes: dict[str, str] = {}
    missing_files: list[str] = []
    entries: list[dict[str, str]] = []
    for rel in tracked:
        fp = root / rel
        if fp.is_file():
            digest = hashlib.sha256(fp.read_bytes()).hexdigest()
        else:
            digest = "<missing>"
            missing_files.append(rel)
        file_hashes[rel] = digest
        entries.append({"path": rel, "sha256": digest})

    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    bundle_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "cd_bundle_sha256": bundle_sha256,
        "files_count": len(tracked),
        "router_files_count": len(router_paths),
        "manifest_scripts_count": len(script_paths),
        "missing_files": missing_files,
        "file_hashes": file_hashes,
    }


def run_deploy_sh_dry_run(
    deploy_script_path: Path | str,
    *,
    expected_sha: str | None = None,
    runner: RunnerFn = default_subprocess_runner,
    lang: str = "en",
) -> dict[str, Any]:
    """Execute list-argv ``bash deploy/gcp_openrouter/deploy.sh --dry-run`` without calling ``gcloud``."""
    lang_code = normalize_lang(lang)
    path = Path(deploy_script_path)
    if not path.is_file():
        return {
            "executed": False,
            "ok": False,
            "returncode": 127,
            "gcloud_invoked": False,
            "diagnostic": _msg(lang_code, "deploy.sh missing for dry-run.", "deploy.sh introuvable pour dry-run."),
        }
    argv = ["bash", str(path), "--dry-run"]
    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "HIVEMIND_DEPLOY_DRY_RUN": "1",
    }
    if expected_sha:
        clean_env["HIVEMIND_BUILD_SHA"] = expected_sha
    rc, out = _invoke_runner(runner, argv, SUBPROCESS_TIMEOUT_S, env=clean_env)
    kv: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    finops_line = kv.get("finops_flags", "")
    env_vars_line = kv.get("update_env_vars", "")
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
        "zero_grants_env": all(
            tok in env_vars_line
            for tok in (
                "HIVEMIND_STARTER_GRANT_HMT=0",
                "HIVEMIND_REGISTRATION_BONUS_HMT=0",
                "HIVEMIND_ALLOW_BONUSES=0",
                "HIVEMIND_PARTNER_TESTING_GRANTS=0",
            )
        ),
        "build_sha_injected": bool(
            sanitize_build_sha(kv.get("build_sha", "")) != "unknown"
            and f"HIVEMIND_BUILD_SHA={kv.get('build_sha', '')}" in env_vars_line
        ),
    }
    ok = all(checks.values())
    return {
        "executed": True,
        "ok": ok,
        "returncode": rc,
        "gcloud_invoked": False,
        "dry_run_build_sha": sanitize_build_sha(kv.get("build_sha", "")),
        "checks": checks,
        "diagnostic": "" if ok else _msg(lang_code, "deploy.sh --dry-run checks failed.", "Échec vérification deploy.sh --dry-run."),
    }


async def _asgi_get_request(app: Any, path: str) -> tuple[int, dict[str, str], bytes]:
    status_code = 500
    resp_headers: dict[str, str] = {}
    chunks: list[bytes] = []
    sent_req = False

    async def _receive() -> dict[str, Any]:
        nonlocal sent_req
        if not sent_req:
            sent_req = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def _send(message: Mapping[str, Any]) -> None:
        nonlocal status_code
        mtype = message.get("type")
        if mtype == "http.response.start":
            status_code = int(message.get("status", 500))
            for raw_k, raw_v in message.get("headers") or ():
                resp_headers[bytes(raw_k).decode("latin1").lower()] = bytes(raw_v).decode("latin1")
        elif mtype == "http.response.body":
            chunks.append(bytes(message.get("body") or b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8000"), (b"x-forwarded-for", b"127.0.0.1")],
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
    }
    await app(scope, _receive, _send)
    return status_code, resp_headers, b"".join(chunks)


def run_isolated_asgi_build_app_check(
    server_py_path: Path | str,
    *,
    build_sha: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run an isolated in-memory ASGI ``build_app`` sanity check when FastAPI is available."""
    lang_code = normalize_lang(lang)
    server_path = Path(server_py_path)
    if not server_path.is_file():
        return {"executed": False, "ok": False, "status": "server_py_missing", "statuses": {}, "issues": ["server.py missing"]}
    try:
        importlib.import_module("fastapi")
        importlib.import_module("hivemind.router.gateway")
    except ImportError:
        return {"executed": False, "ok": True, "status": "skipped_fastapi_unavailable", "statuses": {}, "issues": []}

    env_keys = (
        "HIVEMIND_ROUTER_SECRET",
        "HIVEMIND_ROUTER_DB",
        "HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS",
        "HIVEMIND_OPEN_REGISTRATION",
        "HIVEMIND_STARTER_GRANT_HMT",
        "HIVEMIND_REGISTRATION_BONUS_HMT",
        "HIVEMIND_ALLOW_BONUSES",
        "HIVEMIND_BUILD_SHA",
    )
    eff_sha = sanitize_build_sha(build_sha) if build_sha else "cd34testsha1"
    if eff_sha == "unknown":
        eff_sha = "cd34testsha1"
    with _SERVER_LOAD_LOCK:
        saved = {k: os.environ.get(k) for k in env_keys}
        try:
            os.environ["HIVEMIND_ROUTER_SECRET"] = secrets.token_hex(24)
            os.environ["HIVEMIND_ROUTER_DB"] = ":memory:"
            os.environ["HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS"] = "1"
            os.environ["HIVEMIND_OPEN_REGISTRATION"] = "1"
            os.environ["HIVEMIND_STARTER_GRANT_HMT"] = "0"
            os.environ["HIVEMIND_REGISTRATION_BONUS_HMT"] = "0"
            os.environ["HIVEMIND_ALLOW_BONUSES"] = "0"
            os.environ["HIVEMIND_BUILD_SHA"] = eff_sha
            spec = importlib.util.spec_from_file_location("hivemind_cd_bundle_server", server_path)
            if spec is None or spec.loader is None:
                return {"executed": False, "ok": False, "status": "spec_load_failed", "statuses": {}, "issues": ["spec load failed"]}
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            build_fn = getattr(mod, "build_app", None)
            if not callable(build_fn):
                return {"executed": False, "ok": False, "status": "missing_build_app", "statuses": {}, "issues": ["build_app missing"]}

            fail_closed_secret_ok = False
            try:
                build_fn({})
            except RuntimeError:
                fail_closed_secret_ok = True

            app = getattr(mod, "app", None) or build_fn(os.environ)

            statuses: dict[str, int] = {}
            issues: list[str] = []
            seen_nonces: set[str] = set()
            if not fail_closed_secret_ok:
                issues.append("server_build_app_failed_to_reject_missing_secret")

            async def _run_probes() -> None:
                for path, kind in ASGI_PROBE_ROUTES:
                    code, hdrs, raw_bytes = await _asgi_get_request(app, path)
                    statuses[path] = code
                    if code != 200:
                        issues.append(f"asgi_get_{path}_http_{code}")
                        continue
                    _, body = _parse_http_raw_response(code, raw_bytes, hdrs)
                    if path == "/health" and body.get("status") != "ok":
                        issues.append("asgi_health_not_ok")
                    elif path == "/v1/models" and not isinstance(body.get("data"), list):
                        issues.append("asgi_models_missing_data")
                    elif path == "/v1/hivemind/onboarding/manifest":
                        if body.get("object") != MANIFEST_OBJECT or not shas_match(str(body.get("build_sha") or ""), eff_sha):
                            issues.append("asgi_manifest_mismatch")
                    elif path == "/v1/hivemind/onboarding/feedback/summary":
                        if set(body.keys()) != EXPECTED_FEEDBACK_SUMMARY_KEYS or body.get("object") != FEEDBACK_SUMMARY_OBJECT:
                            issues.append("asgi_feedback_summary_schema")
                    elif kind == "html":
                        ha = audit_html_safety(str(body.get("html") or ""), str(body.get("csp") or ""))
                        nonce = str(ha.get("nonce") or "")
                        if not ha["ok"] or not nonce or nonce in seen_nonces:
                            issues.append(f"asgi_html_failed_{path}")
                        if nonce:
                            seen_nonces.add(nonce)

            asyncio.run(_run_probes())
        except Exception as exc:
            return {"executed": False, "ok": False, "status": "module_load_error", "statuses": {}, "issues": [f"{type(exc).__name__}"]}
        finally:
            for k, old_v in saved.items():
                if old_v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old_v
    ok = len(issues) == 0
    return {
        "executed": True,
        "ok": ok,
        "status": "passed" if ok else "failed",
        "fail_closed_secret_ok": fail_closed_secret_ok,
        "unique_csp_nonces": len(seen_nonces),
        "statuses": statuses,
        "issues": issues,
        "note": _msg(lang_code, "Isolated ASGI build_app sanity check complete.", "Sonde ASGI isolée build_app terminée."),
    }


def _evaluate_feedback_summary_payload(body: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    exact_keys = set(body.keys()) == EXPECTED_FEEDBACK_SUMMARY_KEYS
    obj_ok = body.get("object") == FEEDBACK_SUMMARY_OBJECT
    total = body.get("total_responses")
    total_ok = not isinstance(total, bool) and isinstance(total, int) and total >= 0
    ease = body.get("mean_ease_rating")
    lag = body.get("zero_game_lag_ratio")
    prof = body.get("by_profile")
    if total_ok and total == 0:
        ease_ok = ease is None
        lag_ok = lag is None
    else:
        ease_ok = not isinstance(ease, bool) and isinstance(ease, (int, float)) and 1.0 <= float(ease) <= 5.0
        lag_ok = not isinstance(lag, bool) and isinstance(lag, (int, float)) and 0.0 <= float(lag) <= 1.0
    prof_ok = (
        isinstance(prof, Mapping)
        and set(prof.keys()) == {"dev", "gamer"}
        and all(not isinstance(prof[k], bool) and isinstance(prof[k], int) and prof[k] >= 0 for k in ("dev", "gamer"))
        and total_ok
        and (prof["dev"] + prof["gamer"] == total)
    )
    ok = bool(exact_keys and obj_ok and total_ok and ease_ok and lag_ok and prof_ok)
    return ok, (dict(body) if ok else {})


def _build_dry_run_remediation_commands(expected_sha: str | None) -> dict[str, Any]:
    sha_arg = f",HIVEMIND_BUILD_SHA={expected_sha}" if expected_sha else ""
    redeploy_argv = [
        "gcloud",
        "run",
        "deploy",
        "hivemind-openrouter",
        "--source",
        "deploy/gcp_openrouter",
        "--region",
        "us-central1",
        "--min-instances",
        "0",
        "--max-instances",
        "1",
        "--no-cpu-throttling",
        "--concurrency",
        "80",
        "--no-traffic",
        f"--update-env-vars=HIVEMIND_STARTER_GRANT_HMT=0,HIVEMIND_REGISTRATION_BONUS_HMT=0,HIVEMIND_ALLOW_BONUSES=0{sha_arg}",
    ]
    rollback_argv = [
        "gcloud",
        "run",
        "services",
        "update-traffic",
        "hivemind-openrouter",
        "--region",
        "us-central1",
        "--to-revisions",
        "PREVIOUS_REVISION=100",
    ]
    return {
        "gcloud_invoked": False,
        "gcloud_calls": 0,
        "deploy_sh_dry_run": f"{f'HIVEMIND_BUILD_SHA={expected_sha} ' if expected_sha else ''}HIVEMIND_DEPLOY_DRY_RUN=1 bash deploy/gcp_openrouter/deploy.sh --dry-run",
        "redeploy_argv_dry_run": redeploy_argv,
        "redeploy_command_display": " ".join(shlex.quote(a) for a in redeploy_argv),
        "rollback_argv_dry_run": rollback_argv,
        "rollback_command_display": " ".join(shlex.quote(a) for a in rollback_argv),
    }


def _resolve_git_head_sha(root: Path, runner: RunnerFn) -> str | None:
    rc, out = _invoke_runner(runner, ["git", "-C", str(root), "rev-parse", "HEAD"], SUBPROCESS_TIMEOUT_S)
    if rc != 0:
        return None
    sha = sanitize_build_sha(out.strip())
    return None if sha == "unknown" else sha


def verify_cd_deploy_bundle_and_live_revision(
    gateway: str = DEFAULT_GATEWAY,
    *,
    repo_root: Path | str | None = None,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    run_deploy_dry_run: bool = True,
    run_asgi_stage_check: bool = True,
    require_holiday_routes: bool = False,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference CD Cloud Run deploy-bundle audit, dry-run/ASGI check & PR #949 live parity verifier."""
    lang_code = normalize_lang(lang)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    compliance = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd if cwd is not None else root,
        check_git_remote=check_git_remote,
        lang=lang_code,
        home=home,
        extra_paths=(root,),
    )
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": CD_GATE_OBJECT,
                "lang": lang_code,
                "ok": False,
                "exit_code": 2,
                "compliance_ok": False,
                "compliance": compliance,
                "network_calls": 0,
                "file_writes": 0,
                "subprocess_calls": 0,
                "inference_calls": 0,
                "gcloud_calls": 0,
                "summary": compliance["summary"],
            }
        )

    clean_gw = validate_gateway_url(gateway)
    if isinstance(target_online_producers, bool) or int(target_online_producers) < 1:
        raise VerifierError("target_online_producers must be >= 1.")

    subprocess_calls = 0
    resolved_expected_sha: str | None = validate_expected_sha(expected_sha) if expected_sha else None
    git_head_error = False
    if resolved_expected_sha is None and check_git_head:
        subprocess_calls += 1
        resolved_expected_sha = _resolve_git_head_sha(root, runner)
        if resolved_expected_sha is None:
            git_head_error = True

    finops_audit = audit_deploy_bundle_finops(root, expected_sha=resolved_expected_sha, lang=lang_code)
    bundle_fp = compute_cd_bundle_fingerprint(root)

    deploy_dry_run_res: dict[str, Any] = {"executed": False, "ok": True, "gcloud_invoked": False}
    if run_deploy_dry_run:
        subprocess_calls += 1
        deploy_dry_run_res = run_deploy_sh_dry_run(
            root / "deploy" / "gcp_openrouter" / "deploy.sh",
            expected_sha=resolved_expected_sha,
            runner=runner,
            lang=lang_code,
        )

    asgi_stage_res: dict[str, Any] = {"executed": False, "ok": True, "status": "skipped"}
    if run_asgi_stage_check:
        asgi_stage_res = run_isolated_asgi_build_app_check(
            root / "deploy" / "gcp_openrouter" / "server.py",
            build_sha=resolved_expected_sha,
            lang=lang_code,
        )

    route_results: dict[str, dict[str, Any]] = {}
    seen_nonces: set[str] = set()
    network_calls = 0
    fb_summary: dict[str, Any] | None = None

    for path, kind, category in CD_LIVE_PROBE_ROUTES:
        status, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        network_calls += 1
        entry: dict[str, Any] = {"status": status, "kind": kind, "category": category, "ok": False}

        if status == 200 and kind == "html":
            ha = audit_html_safety(str(body.get("html") or ""), str(body.get("csp") or ""))
            nonce = str(ha.get("nonce") or "")
            unique_nonce = bool(nonce and nonce not in seen_nonces)
            if nonce:
                seen_nonces.add(nonce)
            entry.update(ha)
            entry["unique_nonce"] = unique_nonce
            entry["ok"] = bool(ha["ok"] and unique_nonce)
        elif status == 200 and kind == "json":
            if path == "/health":
                entry["ok"] = str(body.get("status") or "").lower() == "ok"
            elif path == "/v1/models":
                models = body.get("data") if isinstance(body.get("data"), list) else []
                ids = [str(m.get("id") or "") for m in models if isinstance(m, Mapping)]
                has_coder = any(mid == "hivemind/coder" or mid.endswith("/hivemind/coder") for mid in ids)
                entry["model_count"] = len(ids)
                entry["has_hivemind_coder"] = has_coder
                entry["ok"] = bool(ids and has_coder)
            elif path == "/v1/hivemind/onboarding/manifest":
                hroutes = set(body.get("holiday_routes") or []) if isinstance(body.get("holiday_routes"), list) else set()
                live_sha = sanitize_build_sha(body.get("build_sha"))
                entry["build_sha"] = live_sha
                entry["cloud_run_revision"] = sanitize_cloud_run_revision(body.get("cloud_run_revision"))
                entry["missing_holiday_routes"] = sorted(set(EXPECTED_MANIFEST_HOLIDAY_ROUTES) - hroutes)
                entry["ok"] = bool(
                    body.get("object") == MANIFEST_OBJECT
                    and body.get("personal_pc_only") is True
                    and not entry["missing_holiday_routes"]
                )
            elif path == "/v1/hivemind/onboarding/feedback/summary":
                fb_ok, fb_doc = _evaluate_feedback_summary_payload(body)
                entry["exact_5_keys"] = set(body.keys()) == EXPECTED_FEEDBACK_SUMMARY_KEYS
                entry["summary"] = fb_doc
                entry["ok"] = fb_ok
                if fb_ok:
                    fb_summary = fb_doc
            elif path == "/v1/hivemind/producers/fleet":
                rows = body.get("hiveminders") if isinstance(body.get("hiveminders"), list) else []
                leaked = any(isinstance(r, Mapping) and "endpoint_url" in r for r in rows)
                entry["live_hiveminders"] = int(body.get("live_hiveminders") or 0)
                entry["total_hiveminders"] = int(body.get("total_hiveminders") or 0)
                entry["distinct_physical_gpus"] = int(body.get("distinct_physical_gpus") or 0)
                entry["endpoint_url_redacted"] = not leaked
                entry["ok"] = bool(body.get("object") == FLEET_OBJECT and not leaked)
            elif path == "/v1/hivemind/raid/status":
                comp = body.get("compliance") if isinstance(body.get("compliance"), Mapping) else {}
                entry["ok"] = bool(body.get("object") == RAID_STATUS_OBJECT and comp.get("personal_pc_only") is True)
        route_results[path] = entry

    pre_pr949_drift = [
        p
        for p, _, cat in CD_LIVE_PROBE_ROUTES
        if cat in ("holiday", "manifest", "feedback", "raid") and route_results[p]["status"] == 404
    ]
    manifest_info = route_results["/v1/hivemind/onboarding/manifest"]
    live_sha = str(manifest_info.get("build_sha") or "unknown")
    sha_match = (
        False
        if git_head_error
        else (True if resolved_expected_sha is None else (manifest_info["ok"] and shas_match(live_sha, resolved_expected_sha)))
    )
    sha_mismatch = bool(git_head_error or (resolved_expected_sha is not None and not sha_match))

    core_ok = bool(route_results["/health"]["ok"] and route_results["/v1/models"]["ok"])
    holiday_routes_ok = all(
        route_results[p]["ok"]
        for p in ("/noel", "/quick", "/raid-noel", "/arena-noel", "/v1/hivemind/onboarding/manifest", "/v1/hivemind/onboarding/feedback/summary", "/v1/hivemind/raid/status")
    )
    fleet_info = route_results["/v1/hivemind/producers/fleet"]
    fleet_ok = bool(fleet_info["ok"])
    stale_revision = bool(pre_pr949_drift)
    redeploy_required = bool(stale_revision or sha_mismatch or (require_holiday_routes and not holiday_routes_ok))

    local_bundle_ok = bool(
        finops_audit["finops_ok"]
        and deploy_dry_run_res["ok"]
        and asgi_stage_res["ok"]
        and not bundle_fp["missing_files"]
    )
    overall_ok = bool(local_bundle_ok and core_ok and holiday_routes_ok and fleet_ok and sha_match)
    hard_fail = bool(
        (not local_bundle_ok)
        or (not core_ok)
        or sha_mismatch
        or (require_holiday_routes and (not holiday_routes_ok or not fleet_ok))
    )
    exit_code = 2 if hard_fail else 0

    remediation = _build_dry_run_remediation_commands(resolved_expected_sha)
    summary = (
        _msg(
            lang_code,
            f"CD deploy bundle & PR #949 live revision verified (nonces={len(seen_nonces)}, sha={live_sha}).",
            f"Bundle CD & révision PR #949 Cloud Run vérifiés (nonces={len(seen_nonces)}, sha={live_sha}).",
        )
        if overall_ok
        else _msg(
            lang_code,
            f"CD deploy bundle / PR #949 revision drift detected (404_drift={len(pre_pr949_drift)}, sha_match={sha_match}, finops_ok={finops_audit['finops_ok']}).",
            f"Dérive bundle CD / PR #949 détectée (404={len(pre_pr949_drift)}, sha_match={sha_match}, finops_ok={finops_audit['finops_ok']}).",
        )
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": CD_GATE_OBJECT,
            "lang": lang_code,
            "gateway": clean_gw,
            "ok": overall_ok,
            "exit_code": exit_code,
            "compliance_ok": True,
            "compliance": compliance,
            "finops_audit": finops_audit,
            "deploy_dry_run": deploy_dry_run_res,
            "asgi_stage_check": asgi_stage_res,
            "cd_bundle_fingerprint": {k: v for k, v in bundle_fp.items() if k != "file_hashes"},
            "core_ok": core_ok,
            "holiday_routes_ok": holiday_routes_ok,
            "fleet_ok": fleet_ok,
            "expected_sha": resolved_expected_sha,
            "live_build_sha": live_sha,
            "cloud_run_revision": manifest_info.get("cloud_run_revision"),
            "sha_match": sha_match,
            "sha_mismatch": sha_mismatch,
            "stale_cloud_run_revision": stale_revision,
            "pre_pr949_drift_routes": pre_pr949_drift,
            "redeploy_required": redeploy_required,
            "remediation_dry_run": remediation,
            "unique_csp_nonces": sorted(seen_nonces),
            "unique_csp_nonce_count": len(seen_nonces),
            "feedback_summary": fb_summary,
            "live_hiveminders": int(fleet_info.get("live_hiveminders") or 0),
            "distinct_physical_gpus": int(fleet_info.get("distinct_physical_gpus") or 0),
            "routes": route_results,
            "network_calls": network_calls,
            "file_writes": 0,
            "subprocess_calls": subprocess_calls,
            "inference_calls": 0,
            "gcloud_calls": 0,
            "summary": summary,
        }
    )


def _exception_handles_import_error(handlers: Sequence[ast.ExceptHandler]) -> bool:
    for h in handlers:
        if h.type is None:
            return True
        if isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError", "Exception"):
            return True
        if isinstance(h.type, ast.Tuple):
            for elt in h.type.elts:
                if isinstance(elt, ast.Name) and elt.id in ("ImportError", "ModuleNotFoundError", "Exception"):
                    return True
    return False


def _audit_python_script_source(
    script_name: str,
    source_text: str,
    *,
    allow_optional_httpx: bool = False,
    allow_daemon_deps: bool = False,
    allow_local_probe_socket: bool = False,
) -> dict[str, Any]:
    """Audit AGPL-3.0 header, syntax, zero non-stdlib AST imports, and lateral-movement compliance."""
    src_digest = hashlib.sha256(source_text.encode("utf-8", errors="replace")).hexdigest()
    cache_key = (script_name, src_digest, allow_optional_httpx, allow_daemon_deps, allow_local_probe_socket)
    with _SCRIPT_AUDIT_LOCK:
        cached = _SCRIPT_AUDIT_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "non_stdlib_imports": list(cached["non_stdlib_imports"]), "violations": list(cached["violations"])}

    agpl_ok = AGPL_HEADER_LINE in source_text[:300]
    try:
        tree = ast.parse(source_text, filename=script_name)
    except SyntaxError as exc:
        return {
            "script": script_name,
            "ok": False,
            "agpl_header_ok": agpl_ok,
            "syntax_ok": False,
            "stdlib_ok": False,
            "lateral_movement_ok": False,
            "non_stdlib_imports": [],
            "violations": [f"syntax_error:line_{exc.lineno}"],
        }

    stdlib_names = getattr(sys, "stdlib_module_names", frozenset())
    non_stdlib: set[str] = set()
    unguarded_non_stdlib: set[str] = set()
    violations: list[str] = []
    if not agpl_ok:
        violations.append("missing_agpl_header")

    def _record_import(root_pkg: str, guarded: bool) -> None:
        if root_pkg in FORBIDDEN_LATERAL_IMPORTS:
            violations.append(f"forbidden_import:{root_pkg}")
        if root_pkg not in stdlib_names:
            non_stdlib.add(root_pkg)
            if not (allow_daemon_deps or (allow_optional_httpx and root_pkg == "httpx" and guarded)):
                unguarded_non_stdlib.add(root_pkg)

    doc_exprs: set[int] = set()
    stack: list[tuple[ast.AST, bool]] = [(tree, False)]
    while stack:
        node, guarded = stack.pop()
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                doc_exprs.add(id(node.body[0].value))

        if isinstance(node, ast.Import):
            for alias in node.names:
                _record_import(alias.name.split(".")[0], guarded)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            _record_import(node.module.split(".")[0], guarded)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                violations.append(f"forbidden_call:{node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in ("connect", "connect_ex", "create_connection", "sendto"):
                    if not (allow_local_probe_socket and node.func.attr == "create_connection"):
                        violations.append(f"raw_socket_call:{node.func.attr}")
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "system":
                    violations.append("forbidden_call:os.system")
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    violations.append("forbidden_shell_true")
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)) and node.args[0].elts:
                first = node.args[0].elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    cmd_bin = Path(first.value).name.lower()
                    if cmd_bin in FORBIDDEN_SUBPROCESS_BINARIES:
                        violations.append(f"forbidden_subprocess:{cmd_bin}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_exprs:
            text = node.value.strip()
            if "://" in text:
                for token in text.split():
                    if "://" not in token:
                        continue
                    cleaned = token.strip(",;()'\"<>")
                    if "=" in cleaned and not cleaned.startswith(("http://", "https://", "ws://", "wss://")):
                        cleaned = cleaned.split("=", 1)[1]
                    if cleaned.startswith(("^", "(", "[")) or cleaned.rstrip("/") in ALLOWED_DISPLAY_URLS:
                        continue
                    try:
                        parsed = urlparse(cleaned)
                    except ValueError:
                        continue
                    if parsed.scheme in ("http", "https", "ws", "wss", "ftp", "ssh", "smb"):
                        host = (parsed.hostname or "").lower()
                        is_plist_dtd = (
                            "-//Apple//DTD PLIST 1.0//EN" in text
                            and parsed.path == "/DTDs/PropertyList-1.0.dtd"
                            and host.endswith(".apple.com")
                        )
                        if host and not is_plist_dtd and host not in ALLOWED_DOC_PLACEHOLDERS and host not in ALLOWED_SOURCE_URL_WILDCARD_HOSTS and not is_allowed_outbound_host(host):
                            violations.append(f"unauthorized_url_host:line_{node.lineno}")
            if "." in text:
                for ip_hit in _IPV4_RE.findall(text):
                    try:
                        ip_obj = ipaddress.ip_address(ip_hit)
                    except ValueError:
                        continue
                    if not ip_obj.is_loopback and (ip_obj.is_private or ip_obj.is_link_local or not is_allowed_outbound_host(ip_hit)):
                        violations.append(f"non_loopback_ipv4:line_{node.lineno}")

        if isinstance(node, ast.Try):
            handles_imp = _exception_handles_import_error(node.handlers)
            for child in ast.iter_child_nodes(node):
                stack.append((child, guarded or (handles_imp and child in node.body)))
        else:
            for child in ast.iter_child_nodes(node):
                stack.append((child, guarded))

    stdlib_ok = len(unguarded_non_stdlib) == 0
    if not stdlib_ok:
        violations.append(f"non_stdlib_imports:{','.join(sorted(unguarded_non_stdlib))}")
    lateral_ok = not any(
        v.startswith(("forbidden_", "raw_socket_call:", "unauthorized_url_host:", "non_loopback_ipv4:"))
        for v in violations
    )
    result = {
        "script": script_name,
        "ok": agpl_ok and stdlib_ok and lateral_ok,
        "agpl_header_ok": agpl_ok,
        "syntax_ok": True,
        "stdlib_ok": stdlib_ok,
        "lateral_movement_ok": lateral_ok,
        "non_stdlib_imports": sorted(non_stdlib),
        "violations": violations,
    }
    with _SCRIPT_AUDIT_LOCK:
        if len(_SCRIPT_AUDIT_CACHE) < 256:
            _SCRIPT_AUDIT_CACHE[cache_key] = {**result, "non_stdlib_imports": list(result["non_stdlib_imports"]), "violations": list(result["violations"])}
    return result


def _extract_ps1_bootstrap_scripts(ps1_text: str) -> list[str]:
    m = re.search(r"\$AllowedBootstrapScripts\s*=\s*@\((.*?)\)", ps1_text, re.DOTALL)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def _extract_sh_bootstrap_scripts(sh_text: str) -> list[str]:
    m = re.search(r"ALLOWED_BOOTSTRAP_SCRIPTS=\((.*?)\)", sh_text, re.DOTALL)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def _audit_launcher_scripts(
    scripts_dir: Path,
    manifest_scripts: Sequence[str],
    *,
    runner: RunnerFn,
) -> tuple[dict[str, Any], int]:
    subprocess_calls = 0
    ps1_path = scripts_dir / "join_hivemind.ps1"
    sh_path = scripts_dir / "join_hivemind_campaign.sh"

    ps1_text = ps1_path.read_text(encoding="utf-8", errors="replace") if ps1_path.is_file() else ""
    sh_text = sh_path.read_text(encoding="utf-8", errors="replace") if sh_path.is_file() else ""

    ps1_agpl_ok = bool(ps1_text and "Licensed under the AGPL-3.0" in ps1_text[:300] and "0.00 EUR / 0.00 USD" in ps1_text)
    sh_agpl_ok = bool(sh_text and "Licensed under the AGPL-3.0" in sh_text[:300] and "0.00 EUR / 0.00 USD" in sh_text)
    ps1_safe = bool(ps1_text and not _PS1_SH_UNSAFE_RE.search(ps1_text) and not _STATIC_FIG_UNIT_RE.search(ps1_text))
    sh_safe = bool(sh_text and not _PS1_SH_UNSAFE_RE.search(sh_text) and not _STATIC_FIG_UNIT_RE.search(sh_text))

    bash_syntax_ok = False
    if sh_path.is_file():
        sh_digest = hashlib.sha256(sh_text.encode("utf-8", errors="replace")).hexdigest()
        b_key = (str(sh_path), sh_digest)
        cached_bash = _BASH_SYNTAX_CACHE.get(b_key) if runner is default_subprocess_runner else None
        if cached_bash is not None:
            bash_syntax_ok = cached_bash
        else:
            subprocess_calls += 1
            rc, _ = _invoke_runner(runner, ["bash", "-n", str(sh_path)], SUBPROCESS_TIMEOUT_S)
            bash_syntax_ok = rc == 0
            if runner is default_subprocess_runner:
                with _SCRIPT_AUDIT_LOCK:
                    if len(_BASH_SYNTAX_CACHE) < 64:
                        _BASH_SYNTAX_CACHE[b_key] = bash_syntax_ok

    ps1_allowed = _extract_ps1_bootstrap_scripts(ps1_text) if ps1_text else []
    sh_allowed = _extract_sh_bootstrap_scripts(sh_text) if sh_text else []
    man_set = set(manifest_scripts)
    ps1_set = set(ps1_allowed)
    sh_set = set(sh_allowed)

    ps1_not_in_manifest = sorted(ps1_set - man_set) if ps1_allowed else []
    manifest_not_in_ps1 = sorted(man_set - ps1_set) if ps1_allowed else []
    sh_not_in_manifest = sorted(sh_set - man_set) if sh_allowed else []
    manifest_not_in_sh = sorted(man_set - sh_set) if sh_allowed else []
    drift_detected = bool(ps1_not_in_manifest or manifest_not_in_ps1 or sh_not_in_manifest or manifest_not_in_sh)

    launcher_contracts_ok = bool(
        (not ps1_path.is_file() or (ps1_agpl_ok and ps1_safe and bool(ps1_allowed)))
        and (not sh_path.is_file() or (sh_agpl_ok and sh_safe and bash_syntax_ok and bool(sh_allowed)))
    )
    return {
        "contracts_ok": launcher_contracts_ok,
        "ps1_present": ps1_path.is_file(),
        "ps1_agpl_ok": ps1_agpl_ok,
        "ps1_safe": ps1_safe,
        "ps1_allowed_count": len(ps1_allowed),
        "sh_present": sh_path.is_file(),
        "sh_agpl_ok": sh_agpl_ok,
        "sh_safe": sh_safe,
        "sh_bash_syntax_ok": bash_syntax_ok,
        "sh_allowed_count": len(sh_allowed),
        "drift_detected": drift_detected,
        "ps1_not_in_manifest": ps1_not_in_manifest,
        "manifest_not_in_ps1": manifest_not_in_ps1,
        "sh_not_in_manifest": sh_not_in_manifest,
        "manifest_not_in_sh": manifest_not_in_sh,
    }, subprocess_calls


def compute_ces_idempotency_key(
    payload: Mapping[str, Any],
    *,
    account_ref: str = "",
    salt: str = "",
) -> str:
    """Compute a deterministic zero-PII ``Idempotency-Key`` for ``POST /v1/hivemind/onboarding/feedback``."""
    canonical = json.dumps(
        {
            "campaign": CAMPAIGN_ID,
            "account_ref_hash": hashlib.sha256(f"{account_ref}:{salt}".encode()).hexdigest()[:16],
            "profile": str(payload.get("profile") or ""),
            "ease_rating": int(payload.get("ease_rating") or 5),
            "zero_game_lag": bool(payload.get("zero_game_lag")),
            "platform": str(payload.get("platform") or ""),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return "dyn34-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:28]


def _validate_ces_item_payload(raw: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, Mapping):
        return None, "payload_not_mapping"
    try:
        profile = normalize_profile(raw.get("profile"))
    except VerifierError:
        return None, "invalid_profile"
    ease = raw.get("ease_rating")
    if isinstance(ease, bool) or not isinstance(ease, int) or not (1 <= ease <= 5):
        return None, "invalid_ease_rating"
    lag = raw.get("zero_game_lag")
    if not isinstance(lag, bool):
        return None, "invalid_zero_game_lag"
    plat = re.sub(r"[^A-Za-z0-9._-]+", "", str(raw.get("platform") or "linux"))[:32] or "linux"
    note_raw = str(raw.get("note") or "")
    if note_raw:
        return None, "non_empty_note_pii_risk"
    normalized = {"profile": profile, "ease_rating": ease, "zero_game_lag": lag, "platform": plat, "note": ""}
    if scan_output_safety(json.dumps(normalized, sort_keys=True)):
        return None, "unsafe_ces_payload"
    return normalized, None


class BundleCesQueue:
    """Atomic, bounded (<=128 pending / <=512 delivered), ``0600`` zero-PII CES feedback retry queue."""

    _PATH_LOCKS_GUARD = threading.Lock()
    _PATH_LOCKS: dict[Path, threading.RLock] = {}

    def __init__(
        self,
        queue_path: Path | str,
        *,
        max_pending: int = MAX_PENDING_QUEUE_ENTRIES,
        max_delivered: int = MAX_DELIVERED_IDS,
    ) -> None:
        self.queue_path = Path(queue_path)
        self.lock_path = self.queue_path.with_name(f"{self.queue_path.name}.lock")
        if isinstance(max_pending, bool) or not (1 <= int(max_pending) <= 4096):
            raise VerifierError("max_pending must be 1..4096.")
        if isinstance(max_delivered, bool) or not (1 <= int(max_delivered) <= 4096):
            raise VerifierError("max_delivered must be 1..4096.")
        self.max_pending = int(max_pending)
        self.max_delivered = int(max_delivered)

    @classmethod
    def _thread_lock(cls, target: Path) -> threading.RLock:
        key = target.resolve() if target.is_absolute() else (Path.cwd() / target).resolve()
        with cls._PATH_LOCKS_GUARD:
            lock = cls._PATH_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._PATH_LOCKS[key] = lock
            return lock

    @contextmanager
    def _locked(self, *, create_lock_file: bool) -> Iterator[None]:
        tlock = self._thread_lock(self.queue_path)
        with tlock:
            if not create_lock_file and not self.queue_path.exists():
                yield
                return
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if _fcntl is not None:
                    _fcntl.flock(fd, _fcntl.LOCK_EX)
                with suppress(OSError):
                    os.chmod(self.lock_path, 0o600)
                yield
            finally:
                if _fcntl is not None:
                    with suppress(OSError):
                        _fcntl.flock(fd, _fcntl.LOCK_UN)
                os.close(fd)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema": QUEUE_SCHEMA,
            "pending": [],
            "delivered_event_ids": [],
            "quarantined": [],
            "overflow_dropped": 0,
        }

    def _read_state_unlocked(self) -> tuple[dict[str, Any], list[str]]:
        if not self.queue_path.is_file():
            return self._empty_state(), []
        try:
            loaded = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_state(), ["corrupt_queue_json"]
        if not isinstance(loaded, dict) or loaded.get("schema") != QUEUE_SCHEMA:
            return self._empty_state(), ["invalid_queue_schema"]
        state = self._empty_state()
        anomalies: list[str] = []
        raw_delivered = loaded.get("delivered_event_ids")
        if isinstance(raw_delivered, list):
            seen_d: set[str] = set()
            clean_d: list[str] = []
            for eid in raw_delivered:
                s = str(eid or "").strip()
                if s and re.fullmatch(r"[A-Za-z0-9._-]{8,128}", s) and s not in seen_d:
                    seen_d.add(s)
                    clean_d.append(s)
            state["delivered_event_ids"] = clean_d[-self.max_delivered :]
        if isinstance(loaded.get("quarantined"), list):
            state["quarantined"] = [dict(q) for q in loaded["quarantined"] if isinstance(q, Mapping)][-MAX_QUARANTINED_ENTRIES:]
        state["overflow_dropped"] = int(loaded.get("overflow_dropped") or 0)

        raw_pending = loaded.get("pending")
        if isinstance(raw_pending, list):
            for entry in raw_pending:
                if not isinstance(entry, Mapping):
                    anomalies.append("quarantined_non_mapping")
                    continue
                eid = str(entry.get("event_id") or "").strip()
                if not re.fullmatch(r"[A-Za-z0-9._-]{8,128}", eid):
                    anomalies.append("quarantined_bad_event_id")
                    state["quarantined"].append({"reason": "bad_event_id"})
                    continue
                clean_payload, err = _validate_ces_item_payload(entry.get("payload") or {})
                if clean_payload is None:
                    anomalies.append(f"quarantined:{err}")
                    state["quarantined"].append({"event_id": eid, "reason": err})
                    continue
                state["pending"].append(
                    {
                        "event_id": eid,
                        "idempotency_key": str(entry.get("idempotency_key") or eid)[:128],
                        "payload": clean_payload,
                        "retry_count": max(0, int(entry.get("retry_count") or 0)),
                        "last_status": int(entry.get("last_status") or 0),
                    }
                )
        state["quarantined"] = state["quarantined"][-MAX_QUARANTINED_ENTRIES:]
        return state, anomalies

    def _write_state_atomic(self, state: Mapping[str, Any]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            os.chmod(self.queue_path.parent, 0o700)
        bounded = {
            "schema": QUEUE_SCHEMA,
            "pending": list(state.get("pending") or [])[-self.max_pending :],
            "delivered_event_ids": list(state.get("delivered_event_ids") or [])[-self.max_delivered :],
            "quarantined": list(state.get("quarantined") or [])[-MAX_QUARANTINED_ENTRIES:],
            "overflow_dropped": int(state.get("overflow_dropped") or 0),
        }
        raw_text = json.dumps(bounded, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        _assert_safe_output(raw_text)
        tmp_path = self.queue_path.with_name(f".{self.queue_path.name}.{os.getpid()}.tmp")
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(raw_text)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            with suppress(OSError):
                tmp_path.unlink()
            raise
        with suppress(OSError):
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, self.queue_path)
        with suppress(OSError):
            os.chmod(self.queue_path, 0o600)

    def snapshot(self) -> dict[str, Any]:
        """Return bounded queue summary without creating files when empty/missing."""
        with self._locked(create_lock_file=False):
            state, anomalies = self._read_state_unlocked()
            return {
                "exists": self.queue_path.is_file(),
                "pending_count": len(state["pending"]),
                "delivered_count": len(state["delivered_event_ids"]),
                "quarantined_count": len(state["quarantined"]),
                "overflow_dropped": state["overflow_dropped"],
                "anomalies": anomalies,
            }

    def enqueue(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        last_status: int = 0,
    ) -> dict[str, Any]:
        """Persist a validated zero-PII CES feedback item into the bounded ``0600`` spool."""
        clean_payload, err = _validate_ces_item_payload(payload)
        if clean_payload is None:
            raise VerifierError(f"Refusing invalid CES queue payload: {err}")
        eid = (idempotency_key or compute_ces_idempotency_key(clean_payload)).strip()[:128]
        with self._locked(create_lock_file=True):
            state, _ = self._read_state_unlocked()
            if eid in state["delivered_event_ids"] or any(p["event_id"] == eid for p in state["pending"]):
                return {"enqueued": False, "duplicate": True, "event_id": eid, "pending_count": len(state["pending"])}
            while len(state["pending"]) >= self.max_pending:
                state["pending"].pop(0)
                state["overflow_dropped"] = int(state.get("overflow_dropped") or 0) + 1
            state["pending"].append(
                {
                    "event_id": eid,
                    "idempotency_key": eid,
                    "payload": clean_payload,
                    "retry_count": 0,
                    "last_status": int(last_status),
                }
            )
            self._write_state_atomic(state)
            return {"enqueued": True, "duplicate": False, "event_id": eid, "pending_count": len(state["pending"])}

    def flush(self, gateway: str, *, http_fn: HttpFn = default_http_request) -> dict[str, Any]:
        """Replay queued CES items with ``Idempotency-Key``; do not create files if queue is missing."""
        clean_gw = validate_gateway_url(gateway)
        if not self.queue_path.is_file():
            return {
                "flushed_count": 0,
                "retained_count": 0,
                "quarantined_count": 0,
                "network_calls": 0,
                "file_writes": 0,
                "queue_file_exists": False,
            }
        with self._locked(create_lock_file=True):
            state, _ = self._read_state_unlocked()
            if not state["pending"] and not state["quarantined"] and not state["delivered_event_ids"]:
                return {
                    "flushed_count": 0,
                    "retained_count": 0,
                    "quarantined_count": 0,
                    "network_calls": 0,
                    "file_writes": 0,
                    "queue_file_exists": self.queue_path.is_file(),
                }
            remaining: list[dict[str, Any]] = []
            flushed = 0
            quarantined_now = 0
            calls = 0
            url = f"{clean_gw}/v1/hivemind/onboarding/feedback"
            for item in state["pending"]:
                eid = str(item["event_id"])
                if eid in state["delivered_event_ids"]:
                    continue
                status, _ = _invoke_http_fn(
                    http_fn,
                    "POST",
                    url,
                    payload=item["payload"],
                    headers={"Idempotency-Key": str(item["idempotency_key"])},
                )
                calls += 1
                if 200 <= status < 300:
                    flushed += 1
                    state["delivered_event_ids"].append(eid)
                elif status in PERMANENT_CES_STATUSES or (400 <= status < 500 and status not in RETRIABLE_CES_STATUSES):
                    quarantined_now += 1
                    state["quarantined"].append({"event_id": eid, "status": status, "reason": f"http_{status}"})
                else:
                    retries = int(item.get("retry_count") or 0) + 1
                    if retries >= MAX_RETRY_COUNT:
                        quarantined_now += 1
                        state["quarantined"].append({"event_id": eid, "status": status, "reason": "max_retries_exceeded"})
                    else:
                        updated = dict(item)
                        updated["retry_count"] = retries
                        updated["last_status"] = status
                        remaining.append(updated)
            state["pending"] = remaining[-self.max_pending :]
            state["delivered_event_ids"] = state["delivered_event_ids"][-self.max_delivered :]
            state["quarantined"] = state["quarantined"][-MAX_QUARANTINED_ENTRIES:]
            self._write_state_atomic(state)
            return {
                "flushed_count": flushed,
                "retained_count": len(state["pending"]),
                "quarantined_count": len(state["quarantined"]),
                "quarantined_delta": quarantined_now,
                "delivered_count": len(state["delivered_event_ids"]),
                "network_calls": calls,
                "file_writes": 1,
                "queue_file_exists": self.queue_path.is_file(),
            }

    def purge(self) -> dict[str, Any]:
        """Securely delete local CES queue and lock files (always permitted offline)."""
        removed = False
        with self._thread_lock(self.queue_path):
            for p in (self.queue_path, self.lock_path):
                if p.exists():
                    with suppress(OSError):
                        p.unlink()
                        removed = True
        return {"purged": True, "removed_files": removed, "queue_path": redact_home_path(self.queue_path)}


def _opt_float(raw: Any) -> float | None:
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


def default_gpu_probe(*, runner: RunnerFn = default_subprocess_runner, timeout_s: float = SUBPROCESS_TIMEOUT_S) -> GpuSnapshot:
    """Probe local NVIDIA GPU telemetry via ``nvidia-smi`` or return honest no-GPU fallback."""
    rc, stdout = _invoke_runner(
        runner,
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
        min(float(timeout_s), SUBPROCESS_TIMEOUT_S),
    )
    if rc != 0 or not stdout.strip():
        return GpuSnapshot(present=False, reason="dev_api_only_no_nvidia_gpu")
    gpus: list[GpuSnapshot] = []
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or not parts[0]:
            return GpuSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        tot, free, used, temp, util = (_opt_float(parts[i]) for i in range(1, 6))
        if tot is None or free is None or tot <= 0 or free < 0:
            return GpuSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        gpus.append(
            GpuSnapshot(
                present=True,
                parse_error=False,
                name=parts[0][:128],
                total_vram_mib=int(tot),
                free_vram_mib=int(free),
                used_vram_mib=int(used) if used is not None and used >= 0 else max(0, int(tot) - int(free)),
                temp_c=round(temp, 1) if temp is not None else None,
                util_pct=round(util, 1) if util is not None else 0.0,
                reason="ok",
            )
        )
    return max(gpus, key=lambda g: (g.free_vram_mib or 0, -(g.temp_c if g.temp_c is not None else 999.0)))


def _normalize_gpu_snapshot(raw: Any) -> GpuSnapshot:
    if raw is None:
        return GpuSnapshot(present=False, reason="dev_api_only_no_nvidia_gpu")
    if isinstance(raw, GpuSnapshot):
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
            "util_pct": getattr(raw, "util_pct", 0.0),
            "reason": getattr(raw, "reason", "ok"),
        }
    else:
        raise VerifierError("gpu_probe_fn must return GpuSnapshot, a mapping, or None.")
    if not bool(view.get("present", view.get("gpu_present", False))):
        return GpuSnapshot(present=False, reason=str(view.get("reason") or "dev_api_only_no_nvidia_gpu"))
    if bool(view.get("parse_error", False)):
        return GpuSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
    name = view.get("name") or view.get("gpu_name")
    tot = view.get("total_vram_mib", view.get("vram_total_mib", view.get("total_mib")))
    free = view.get("free_vram_mib", view.get("vram_free_mib", view.get("free_mib")))
    used = view.get("used_vram_mib", view.get("vram_used_mib", view.get("used_mib")))
    temp = _opt_float(view.get("temp_c"))
    util = _opt_float(view.get("util_pct"))
    if not name or tot is None or free is None:
        return GpuSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
    return GpuSnapshot(
        present=True,
        parse_error=False,
        name=str(name).strip()[:128],
        total_vram_mib=int(tot),
        free_vram_mib=int(free),
        used_vram_mib=int(used) if used is not None else max(0, int(tot) - int(free)),
        temp_c=round(temp, 1) if temp is not None else None,
        util_pct=round(util, 1) if util is not None else 0.0,
        reason=str(view.get("reason") or "ok"),
    )


def default_process_probe(*, runner: RunnerFn = default_subprocess_runner, os_name: str | None = None, timeout_s: float = SUBPROCESS_TIMEOUT_S) -> list[str]:
    """List local process executables via ``tasklist`` (Windows) or ``ps`` (POSIX)."""
    eff_os = os.name if os_name is None else os_name
    bounded = min(float(timeout_s), SUBPROCESS_TIMEOUT_S)
    procs: list[str] = []
    if eff_os == "nt":
        rc, out = _invoke_runner(runner, ["tasklist", "/FO", "CSV", "/NH"], bounded)
        if rc == 0 and out.strip():
            with suppress(csv.Error):
                for row in csv.reader(io.StringIO(out)):
                    if row and row[0].strip():
                        procs.append(row[0].strip().lower())
        return procs
    rc, out = _invoke_runner(runner, ["ps", "-eo", "comm="], bounded)
    if rc == 0 and out.strip():
        for line in out.splitlines():
            clean = line.strip().lower()
            if clean:
                procs.append(clean.rsplit("/", 1)[-1])
    return procs


def _detect_steam_dx12_activity(process_names: Iterable[str], *, non_worker_gpu_util_pct: float | None = None) -> dict[str, Any]:
    trunc = {e[:15] for e in STEAM_DX12_GAME_EXECUTABLES if len(e) > 15}
    matched: list[str] = []
    for proc in process_names:
        low = str(proc).strip().lower().replace("\\", "/")
        base = low.rsplit("/", 1)[-1]
        if base and (base in STEAM_DX12_GAME_EXECUTABLES or base in trunc or base.startswith(STEAM_PROCESS_PREFIXES) or "steamapps/common/" in low):
            if base not in matched:
                matched.append(base)
    gpu_high = bool(non_worker_gpu_util_pct is not None and math.isfinite(non_worker_gpu_util_pct) and non_worker_gpu_util_pct > NON_WORKER_GPU_UTIL_THRESHOLD_PCT)
    return {"game_detected": bool(matched or gpu_high), "matched_games": matched, "gpu_util_triggered": gpu_high}


def default_worker_yield_drill(*, clock_fn: Callable[[], float] = time.monotonic, trigger_exe: str = "cs2.exe", timeout_s: float = SUBPROCESS_TIMEOUT_S) -> dict[str, Any]:
    """Run a real subprocess worker start -> Steam trigger -> <1.5 s YIELD -> 0 MiB release drill."""
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
        ready = proc.stdout.readline().strip() if proc.stdout is not None else ""
        if ready != "HIVEMIND_WORKER_READY":
            proc.kill()
            proc.wait(timeout=1.0)
            return {"worker_started": False, "worker_stopped": False, "clean_release": False, "retained_worker_vram_mib": -1, "yield_latency_ms": None, "trigger_exe": trigger_exe}
        t0 = clock_fn()
        _detect_steam_dx12_activity([trigger_exe])
        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
            proc.stdin.close()
        rel = proc.stdout.readline().strip() if proc.stdout is not None else ""
        rc = proc.wait(timeout=bounded)
        elapsed_ms = round(max(0.0, (clock_fn() - t0) * 1000.0), 3)
        retained = int(rel.split(":", 1)[1].strip()) if rel.startswith("HIVEMIND_WORKER_RELEASED:") and rel.split(":", 1)[1].strip().isdigit() else -1
        return {
            "worker_started": True,
            "worker_stopped": proc.poll() is not None,
            "clean_release": bool(rc == 0 and proc.poll() is not None and retained == 0),
            "retained_worker_vram_mib": retained,
            "yield_latency_ms": elapsed_ms,
            "trigger_exe": trigger_exe,
            "method": "real_subprocess_yield_drill",
        }
    except (subprocess.TimeoutExpired, OSError):
        proc.kill()
        proc.wait(timeout=1.0)
        return {"worker_started": True, "worker_stopped": False, "clean_release": False, "retained_worker_vram_mib": -1, "yield_latency_ms": round(bounded * 1000.0, 3), "trigger_exe": trigger_exe}
    finally:
        if proc.poll() is None:
            with suppress(OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait(timeout=1.0)
        for st in (proc.stdin, proc.stdout, proc.stderr):
            if st is not None and not st.closed:
                st.close()


def _invoke_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _write_secure_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        os.chmod(path.parent, 0o700)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink()
        raise
    with suppress(OSError):
        os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)
    with suppress(OSError):
        os.chmod(path, 0o600)


def _run_isolated_vscode_quickstart_drill(
    *,
    clean_gw: str,
    home_dir: Path | None,
    workspace_dir: Path | None,
    api_key: str,
    handle: str,
    role: str,
    run_uninstall: bool,
    clock_fn: Callable[[], float],
) -> dict[str, Any]:
    t0 = clock_fn()
    file_writes = 0
    diagnostics: list[str] = []
    tmp_ctx = tempfile.TemporaryDirectory(prefix="hivemind-dyn34-qs-") if (home_dir is None or workspace_dir is None) else None
    try:
        eff_home = Path(home_dir) if home_dir is not None else (Path(tmp_ctx.name) / "home")  # type: ignore[union-attr]
        eff_ws = Path(workspace_dir) if workspace_dir is not None else (Path(tmp_ctx.name) / "ws")  # type: ignore[union-attr]
        eff_home.mkdir(parents=True, exist_ok=True)
        eff_ws.mkdir(parents=True, exist_ok=True)

        cred_path = eff_home / DEFAULT_CREDENTIALS_REL
        cont_path = eff_home / DEFAULT_CONTINUE_REL
        vscode_path = eff_ws / ".vscode" / "settings.json"

        preexisting_secret_leak = False
        existing_ws: dict[str, Any] = {}
        if vscode_path.is_file():
            pre_txt = vscode_path.read_text(encoding="utf-8", errors="replace")
            if _API_KEY_RE.search(pre_txt) or _RAW_HM_KEY_RE.search(pre_txt) or _BEARER_RE.search(pre_txt) or _HOME_PATH_RE.search(pre_txt):
                preexisting_secret_leak = True
                diagnostics.append("existing_vscode_settings_contains_secret_or_home_path")
            with suppress(json.JSONDecodeError):
                loaded = json.loads(pre_txt)
                if isinstance(loaded, dict):
                    existing_ws = loaded

        if not preexisting_secret_leak:
            cred_obj = {
                "handle": handle,
                "role": role,
                "gateway": clean_gw,
                "api_key": api_key,
                "personal_pc_only": True,
                "financial_promise": FINANCIAL_PROMISE,
            }
            _write_secure_atomic(cred_path, json.dumps(cred_obj, indent=2) + "\n")
            file_writes += 1

            cont_yaml = (
                "# HiveMind Personal Hobby Continue.dev Provider (0.00 EUR / 0.00 USD, AGPL-3.0)\n"
                "models:\n"
                "  - name: HiveMind Coder\n"
                "    provider: openai\n"
                "    model: hivemind/coder\n"
                f"    apiBase: {clean_gw}/v1\n"
                f"    apiKey: {api_key}\n"
            )
            _write_secure_atomic(cont_path, cont_yaml)
            file_writes += 1

            existing_ws.update(
                {
                    "hivemind.openaiCompatible.apiBase": f"{clean_gw}/v1",
                    "hivemind.openaiCompatible.models": ["hivemind/coder", "hivemind/bee-stars"],
                    "hivemind.personalPcOnly": True,
                    "hivemind.financialPromise": FINANCIAL_PROMISE,
                }
            )
            _write_secure_atomic(vscode_path, json.dumps(existing_ws, indent=2, sort_keys=True) + "\n")
            file_writes += 1

        cred_0600 = cred_path.is_file() and (os.name == "nt" or (cred_path.stat().st_mode & 0o777) == 0o600)
        cont_0600 = cont_path.is_file() and (os.name == "nt" or (cont_path.stat().st_mode & 0o777) == 0o600)
        cont_has_coder = cont_path.is_file() and ("hivemind/coder" in cont_path.read_text(encoding="utf-8", errors="replace"))
        ws_txt = vscode_path.read_text(encoding="utf-8", errors="replace") if vscode_path.is_file() else ""
        ws_secret_free = bool(
            not preexisting_secret_leak
            and ws_txt
            and api_key not in ws_txt
            and not _API_KEY_RE.search(ws_txt)
            and not _RAW_HM_KEY_RE.search(ws_txt)
            and not _BEARER_RE.search(ws_txt)
            and not _HOME_PATH_RE.search(ws_txt)
            and f"{clean_gw}/v1" in ws_txt
        )

        uninstall_clean = True
        if run_uninstall and not preexisting_secret_leak:
            for target in (cred_path, cont_path):
                if target.is_file():
                    target.unlink()
            uninstall_clean = (not cred_path.exists()) and (not cont_path.exists())
            if not uninstall_clean:
                diagnostics.append("uninstall_left_residue")

        elapsed_s = round(max(0.0, clock_fn() - t0), 4)
        slo_ok = elapsed_s < DEV_VSCODE_SLO_SECONDS
        ok = bool(slo_ok and cred_0600 and cont_0600 and cont_has_coder and ws_secret_free and uninstall_clean)
        return {
            "ok": ok,
            "slo_ok": slo_ok,
            "elapsed_s": elapsed_s,
            "credentials_0600_ok": cred_0600,
            "continue_0600_ok": cont_0600,
            "continue_has_hivemind_coder": cont_has_coder,
            "vscode_secret_free": ws_secret_free,
            "uninstall_ran": run_uninstall,
            "uninstall_clean": uninstall_clean,
            "file_writes": file_writes,
            "diagnostics": diagnostics,
        }
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _extract_script_payload(body: Mapping[str, Any]) -> tuple[bytes, bool, str, str]:
    raw_obj = body.get("raw_bytes")
    if isinstance(raw_obj, (bytes, bytearray)):
        raw_bytes = bytes(raw_obj)
        utf8_flag = body.get("utf8_valid")
        if isinstance(utf8_flag, bool) and not utf8_flag:
            return raw_bytes, False, "", str(body.get("x_script_sha256") or "").strip().lower()
        try:
            text = raw_bytes.decode("utf-8")
            utf8_valid = True
        except UnicodeDecodeError:
            text = ""
            utf8_valid = False
    else:
        raw_txt = body.get("text") if "text" in body else body.get("html")
        text = str(raw_txt) if isinstance(raw_txt, str) else ""
        raw_bytes = text.encode("utf-8")
        utf8_valid = isinstance(raw_txt, str)
    hdrs = body.get("headers") if isinstance(body.get("headers"), Mapping) else {}
    hdr_sha = str(hdrs.get("x-script-sha256") or body.get("x_script_sha256") or "").strip().lower()
    return raw_bytes, utf8_valid, text, hdr_sha


def _resolve_api_key(
    *,
    api_key: str | None,
    home_path: Path,
    environ: Mapping[str, str],
) -> tuple[str | None, str]:
    explicit = (api_key or "").strip()
    if explicit:
        return (explicit, "explicit_arg") if is_valid_hivemind_key(explicit) else (None, "invalid_explicit_key")
    env_key = str(environ.get("HIVEMIND_API_KEY") or "").strip()
    if env_key:
        return (env_key, "env") if is_valid_hivemind_key(env_key) else (None, "invalid_env_key")
    cred_file = home_path / DEFAULT_CREDENTIALS_REL
    if cred_file.is_file():
        with suppress(OSError, json.JSONDecodeError):
            data = json.loads(cred_file.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                saved = str(data.get("api_key") or "").strip()
                if is_valid_hivemind_key(saved):
                    return saved, "credentials_file"
    return None, "missing"


def verify_unified_quickstart_bundle(
    gateway: str = DEFAULT_GATEWAY,
    *,
    repo_root: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    home_dir: Path | str | None = None,
    verify_script_downloads: bool = True,
    send_opentrainer_heartbeat: bool = False,
    submit_ces: bool = False,
    flush_ces_queue: bool = False,
    queue_path: Path | str | None = None,
    profile: str = "gamer",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    host_label: str = "personal-pc",
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    gpu_probe_fn: Callable[..., Any] = default_gpu_probe,
    process_probe_fn: Callable[..., Any] = default_process_probe,
    worker_yield_fn: Callable[..., Any] = default_worker_yield_drill,
    clock_fn: Callable[[], float] = time.monotonic,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
    api_key: str | None = None,
    handle: str = "noel-bundle-probe",
    role: str | None = None,
    strict_launcher_parity: bool = False,
    run_uninstall: bool = True,
    check_git_remote: bool = True,
    idempotency_salt: str = "",
) -> dict[str, Any]:
    """Unified Win11 / POSIX / <60s VS Code Quickstart Bundle Verifier + Steam GPU yield & CES queue."""
    lang_code = normalize_lang(lang)
    norm_profile = normalize_profile(profile)
    eff_role = (role or ("prosumer" if norm_profile == "dev" else "producer")).strip().lower()
    if eff_role not in VALID_ROLES:
        raise VerifierError(f"Unsupported role {eff_role!r}.")
    if isinstance(ease_rating, bool) or not (1 <= int(ease_rating) <= 5):
        raise VerifierError("ease_rating must be an integer 1..5.")
    if isinstance(max_temp_c, bool) or not math.isfinite(float(max_temp_c)) or not (25.0 <= float(max_temp_c) <= 95.0):
        raise VerifierError("max_temp_c must be a finite float between 25.0 and 95.0.")
    if isinstance(min_free_vram_mib, bool) or int(min_free_vram_mib) < 256:
        raise VerifierError("min_free_vram_mib must be an integer >= 256.")
    if isinstance(yield_budget_ms, bool) or not math.isfinite(float(yield_budget_ms)) or float(yield_budget_ms) <= 0.0:
        raise VerifierError("yield_budget_ms must be > 0.0.")

    t_start = clock_fn()
    env_map = os.environ if environ is None else environ
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    ws_path = Path(workspace_dir) if workspace_dir is not None else None
    eff_home = Path(home_dir) if home_dir is not None else Path.home()

    compliance = evaluate_personal_pc_compliance(
        environ=env_map,
        cwd=cwd if cwd is not None else (ws_path if ws_path is not None else root),
        check_git_remote=check_git_remote,
        lang=lang_code,
        home=home_dir,
        extra_paths=(root, ws_path) if ws_path is not None else (root,),
    )
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": QUICKSTART_BUNDLE_OBJECT,
                "lang": lang_code,
                "ok": False,
                "exit_code": 2,
                "compliance_ok": False,
                "compliance": compliance,
                "network_calls": 0,
                "file_writes": 0,
                "subprocess_calls": 0,
                "inference_calls": 0,
                "summary": compliance["summary"],
            }
        )

    clean_gw = validate_gateway_url(gateway)
    scripts_dir = root / "scripts"
    network_calls = 0
    file_writes = 0
    subprocess_calls = 0
    diagnostics: list[str] = []

    # 1. Local AGPL-3.0, stdlib AST & lateral-movement compliance across quickstart bundle scripts
    local_audits: list[dict[str, Any]] = []
    daemon_deps: dict[str, list[str]] = {}
    for py_name in sorted(set(QUICKSTART_CORE_PY_SCRIPTS) | set(MANIFEST_DAEMON_PY_SCRIPTS)):
        sp = scripts_dir / py_name
        if not sp.is_file():
            continue
        aud = _audit_python_script_source(
            py_name,
            sp.read_text(encoding="utf-8", errors="replace"),
            allow_optional_httpx=(py_name == "hivemind_connect.py"),
            allow_daemon_deps=(py_name in MANIFEST_DAEMON_PY_SCRIPTS),
            allow_local_probe_socket=(py_name == "hivemind_volunteer_preflight.py"),
        )
        local_audits.append(aud)
        if py_name in MANIFEST_DAEMON_PY_SCRIPTS and aud["non_stdlib_imports"]:
            daemon_deps[py_name] = aud["non_stdlib_imports"]
        if not aud["ok"]:
            diagnostics.append(f"script_audit_failed:{py_name}:{'|'.join(aud['violations'])}")

    local_scripts_ok = bool(local_audits) and all(a["ok"] for a in local_audits)
    verified_local_py = {str(a["script"]) for a in local_audits if a.get("ok") and a.get("syntax_ok")}

    # 2. Byte-for-byte SHA-256 GET /scripts/<name> parity across manifest scripts
    manifest_scripts: list[str] = list(EXPECTED_MANIFEST_SCRIPTS)
    script_downloads: list[dict[str, Any]] = []
    downloads_ok = True
    manifest_ok = True
    if verify_script_downloads:
        m_code, m_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/manifest")
        network_calls += 1
        raw_scripts = m_body.get("downloadable_scripts") if isinstance(m_body, Mapping) else None
        if m_code != 200 or m_body.get("object") != MANIFEST_OBJECT or not isinstance(raw_scripts, list) or not raw_scripts:
            manifest_ok = False
            downloads_ok = False
            diagnostics.append(f"manifest_http_{m_code}")
        else:
            manifest_scripts = [str(s) for s in raw_scripts if isinstance(s, str)]
            missing_expected = [s for s in EXPECTED_MANIFEST_SCRIPTS if s not in manifest_scripts]
            if missing_expected:
                manifest_ok = False
                diagnostics.append(f"manifest_missing_scripts:{','.join(missing_expected)}")
            for sname in manifest_scripts:
                if not _SAFE_SCRIPT_NAME_RE.fullmatch(sname):
                    downloads_ok = False
                    diagnostics.append("unsafe_manifest_script_name")
                    continue
                sc_code, sc_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/scripts/{sname}")
                network_calls += 1
                if sc_code != 200:
                    downloads_ok = False
                    diagnostics.append(f"script_download_http_{sc_code}:{sname}")
                    script_downloads.append({"script": sname, "http_status": sc_code, "ok": False})
                    continue
                raw_bytes, utf8_valid, text, hdr_sha = _extract_script_payload(sc_body)
                remote_sha = hashlib.sha256(raw_bytes).hexdigest()
                agpl_ok = bool(utf8_valid and raw_bytes and AGPL_HEADER_LINE in text[:300])
                hdr_ok = (not hdr_sha) or (_SHA256_HEX_RE.fullmatch(hdr_sha) is not None and hmac.compare_digest(hdr_sha, remote_sha))
                local_fp = scripts_dir / sname
                local_sha = hashlib.sha256(local_fp.read_bytes()).hexdigest() if local_fp.is_file() else None
                local_match = local_sha is not None and hmac.compare_digest(remote_sha, local_sha)
                if sname.endswith(".py") and utf8_valid:
                    if local_match and sname in verified_local_py:
                        syntax_ok = True
                    else:
                        try:
                            ast.parse(text, filename=sname)
                            syntax_ok = True
                        except SyntaxError:
                            syntax_ok = False
                else:
                    syntax_ok = utf8_valid and bool(text.strip())
                row_ok = bool(raw_bytes and utf8_valid and agpl_ok and syntax_ok and hdr_ok and local_match)
                if not row_ok:
                    downloads_ok = False
                    diagnostics.append(f"script_parity_failed:{sname}")
                script_downloads.append(
                    {
                        "script": sname,
                        "http_status": sc_code,
                        "ok": row_ok,
                        "byte_length": len(raw_bytes),
                        "remote_sha256": remote_sha[:16],
                        "local_sha256_match": local_match,
                        "header_sha256_match": hdr_ok,
                        "agpl_header_ok": agpl_ok,
                    }
                )

    launcher_diag, launcher_sub_calls = _audit_launcher_scripts(scripts_dir, manifest_scripts, runner=runner)
    subprocess_calls += launcher_sub_calls
    if not launcher_diag["contracts_ok"]:
        diagnostics.append("launcher_contract_failed")
    if strict_launcher_parity and launcher_diag["drift_detected"]:
        diagnostics.append("launcher_manifest_drift_detected")

    # 3. Isolated < 60 s Dev VS Code 0600 Continue config & secret-free .vscode/settings.json + clean uninstall
    resolved_key, key_source = _resolve_api_key(api_key=api_key, home_path=eff_home, environ=env_map)
    drill_key = resolved_key or f"hm-bundle-{secrets.token_hex(8)}"
    vscode_res = _run_isolated_vscode_quickstart_drill(
        clean_gw=clean_gw,
        home_dir=Path(home_dir) if home_dir is not None else None,
        workspace_dir=ws_path,
        api_key=drill_key,
        handle=handle,
        role=eff_role,
        run_uninstall=run_uninstall,
        clock_fn=clock_fn,
    )
    file_writes += int(vscode_res["file_writes"])
    diagnostics.extend(vscode_res["diagnostics"])

    # 4. Real-subprocess < 1.5 s (1500.0 milliseconds) Steam/DX12 GPU yield & <= 70.0 C / >= 2048 MiB VRAM preflight
    gpu = _normalize_gpu_snapshot(_invoke_with_supported_kwargs(gpu_probe_fn))
    if gpu_probe_fn is default_gpu_probe:
        subprocess_calls += 1
    procs = _invoke_with_supported_kwargs(process_probe_fn)
    if process_probe_fn is default_process_probe:
        subprocess_calls += 1
    steam_info = _detect_steam_dx12_activity(procs or [], non_worker_gpu_util_pct=gpu.util_pct if gpu.present else None)
    yield_res = dict(_invoke_with_supported_kwargs(worker_yield_fn, clock_fn=clock_fn, yield_budget_ms=float(yield_budget_ms)))
    if worker_yield_fn is default_worker_yield_drill:
        subprocess_calls += 1

    if not gpu.present:
        gpu_status = "dev_api_only_no_nvidia_gpu"
        thermal_ok = True
        vram_ok = True
    elif gpu.parse_error:
        gpu_status = gpu.reason or "nvidia_smi_parse_error"
        thermal_ok = False
        vram_ok = False
        diagnostics.append(gpu_status)
    else:
        thermal_ok = bool(gpu.temp_c is not None and gpu.temp_c <= float(max_temp_c))
        vram_ok = bool(gpu.free_vram_mib is not None and gpu.free_vram_mib >= int(min_free_vram_mib))
        if not thermal_ok:
            diagnostics.append("gpu_temperature_above_quiet_ceiling")
        if not vram_ok:
            diagnostics.append("gpu_free_vram_below_reserve")
        gpu_status = "ok" if (thermal_ok and vram_ok) else ("hot_gpu" if not thermal_ok else "low_vram")

    raw_yield_ms = yield_res.get("yield_latency_ms")
    clean_rel = bool(yield_res.get("clean_release"))
    ret_vram = int(yield_res.get("retained_worker_vram_mib") if yield_res.get("retained_worker_vram_mib") is not None else -1)
    yield_slo_ok = bool(clean_rel and ret_vram == 0 and raw_yield_ms is not None and float(raw_yield_ms) < float(yield_budget_ms))
    if not yield_slo_ok:
        diagnostics.append("steam_yield_slo_exceeded")
    zero_game_lag = bool((not had_game_lag) and yield_slo_ok and thermal_ok and vram_ok)

    # 5. Optional outbound POST /v1/hivemind/opentrainer/heartbeat
    ot_res: dict[str, Any] = {"attempted": False, "ok": True, "status": "skipped"}
    if send_opentrainer_heartbeat:
        if resolved_key is None:
            ot_res = {"attempted": False, "ok": False, "status": "missing_api_key"}
            diagnostics.append("opentrainer_missing_api_key")
        else:
            hb_payload = {
                "host_label": sanitize_host_label(host_label),
                "gpu_name": gpu.name if (gpu.present and not gpu.parse_error) else None,
                "vram_total_mib": gpu.total_vram_mib if (gpu.present and not gpu.parse_error and (gpu.total_vram_mib or 0) >= MIN_REPORTABLE_VRAM_TOTAL_MIB) else None,
                "vram_free_mib": gpu.free_vram_mib if (gpu.present and not gpu.parse_error) else None,
                "network_mode": "outbound-https-relay",
                "training_stack": "qlora-moe+diloco+amcp-aglet+boinc",
                "active_campaign": CAMPAIGN_ID,
            }
            ot_status, ot_body = _invoke_http_fn(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
                payload=hb_payload,
                headers={"Authorization": f"Bearer {resolved_key}"},
            )
            network_calls += 1
            ot_ok = bool(ot_status == 200 and ot_body.get("object") == OPENTRAINER_HB_ACK_OBJECT)
            if not ot_ok:
                diagnostics.append(f"opentrainer_heartbeat_http_{ot_status}")
            ot_res = {
                "attempted": True,
                "http_status": ot_status,
                "ok": ot_ok,
                "network_mode": "outbound-https-relay",
                "node_id": ((ot_body.get("node") or {}).get("node_id") if isinstance(ot_body.get("node"), Mapping) else None),
            }

    # 6. Optional BundleCesQueue flush & zero-PII Idempotency-Key POST /v1/hivemind/onboarding/feedback
    q_path = Path(queue_path) if queue_path is not None else (eff_home / DEFAULT_QUEUE_REL)
    ces_queue = BundleCesQueue(q_path)
    flush_res: dict[str, Any] | None = None
    if flush_ces_queue:
        flush_res = ces_queue.flush(clean_gw, http_fn=http_fn)
        network_calls += int(flush_res["network_calls"])
        file_writes += int(flush_res["file_writes"])

    ces_res: dict[str, Any] = {"submitted": False, "status": "skipped", "spooled": False, "quarantined": False}
    if submit_ces:
        plat = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}.get(platform.system(), "linux")
        ces_payload = {
            "profile": norm_profile,
            "ease_rating": int(ease_rating),
            "zero_game_lag": zero_game_lag,
            "platform": plat,
            "note": "",
        }
        idem_key = compute_ces_idempotency_key(ces_payload, account_ref=handle or mask_api_key(drill_key), salt=idempotency_salt)
        c_status, c_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/onboarding/feedback",
            payload=ces_payload,
            headers={"Idempotency-Key": idem_key},
        )
        network_calls += 1
        if 200 <= c_status < 300:
            ces_res = {
                "submitted": True,
                "http_status": c_status,
                "status": "delivered",
                "spooled": False,
                "quarantined": False,
                "idempotency_key": idem_key,
                "total_responses": c_body.get("total_responses"),
            }
        elif c_status in PERMANENT_CES_STATUSES or (400 <= c_status < 500 and c_status not in RETRIABLE_CES_STATUSES):
            ces_res = {
                "submitted": True,
                "http_status": c_status,
                "status": "quarantined",
                "spooled": False,
                "quarantined": True,
                "idempotency_key": idem_key,
            }
            diagnostics.append(f"ces_rejected_http_{c_status}")
        else:
            ces_queue.enqueue(ces_payload, idempotency_key=idem_key, last_status=c_status)
            file_writes += 1
            ces_res = {
                "submitted": True,
                "http_status": c_status,
                "status": "spooled_for_retry",
                "spooled": True,
                "quarantined": False,
                "idempotency_key": idem_key,
            }

    elapsed_s = round(max(0.0, clock_fn() - t_start), 4)
    overall_ok = bool(
        local_scripts_ok
        and manifest_ok
        and downloads_ok
        and launcher_diag["contracts_ok"]
        and (not strict_launcher_parity or not launcher_diag["drift_detected"])
        and vscode_res["ok"]
        and thermal_ok
        and vram_ok
        and yield_slo_ok
        and ot_res["ok"]
        and not ces_res["quarantined"]
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": QUICKSTART_BUNDLE_OBJECT,
            "lang": lang_code,
            "gateway": clean_gw,
            "ok": overall_ok,
            "exit_code": 0 if overall_ok else 2,
            "compliance_ok": True,
            "compliance": compliance,
            "api_key_masked": mask_api_key(resolved_key) if resolved_key else "<ephemeral-dryrun>",
            "key_source": key_source,
            "local_scripts_ok": local_scripts_ok,
            "local_scripts_audited": len(local_audits),
            "manifest_daemon_script_deps": daemon_deps,
            "manifest_ok": manifest_ok,
            "downloads_ok": downloads_ok,
            "scripts_downloaded_count": len(script_downloads),
            "script_downloads": script_downloads,
            "launcher_diagnostics": launcher_diag,
            "vscode_quickstart": vscode_res,
            "gpu": {
                "present": gpu.present,
                "status": gpu_status,
                "name": gpu.name,
                "total_vram_mib": gpu.total_vram_mib,
                "free_vram_mib": gpu.free_vram_mib,
                "temp_c": gpu.temp_c,
                "thermal_ok": thermal_ok,
                "vram_ok": vram_ok,
            },
            "steam_preflight": {
                **steam_info,
                **yield_res,
                "yield_budget_ms": float(yield_budget_ms),
                "yield_slo_ok": yield_slo_ok,
                "zero_game_lag": zero_game_lag,
            },
            "opentrainer_heartbeat": ot_res,
            "ces_feedback": ces_res,
            "ces_queue_flush": flush_res,
            "ces_queue": ces_queue.snapshot(),
            "diagnostics": diagnostics,
            "elapsed_s": elapsed_s,
            "within_60s_slo": elapsed_s < DEV_VSCODE_SLO_SECONDS,
            "network_calls": network_calls,
            "file_writes": file_writes,
            "subprocess_calls": subprocess_calls,
            "inference_calls": 0,
            "summary": _msg(
                lang_code,
                f"Unified Win11/POSIX/VS Code quickstart bundle verified ({'OK' if overall_ok else 'FAILED'}; {FINANCIAL_PROMISE}).",
                f"Bundle quickstart Win11/POSIX/VS Code vérifié ({'OK' if overall_ok else 'ÉCHEC'} ; {FINANCIAL_PROMISE}).",
            ),
        }
    )


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render an FR/EN SHA-256-receipted Markdown summary without secrets or static figures."""
    lang_code = normalize_lang(str(report.get("lang") or "en"))
    lines = [
        f"# {_msg(lang_code, 'HiveMind CD Deploy-Bundle & Quickstart Verifier', 'Vérificateur Bundle CD & Quickstart HiveMind')}",
        "",
        f"- **Object**: `{report.get('object', '')}`",
        f"- **Status**: `{'PASS' if report.get('ok', report.get('compliance_ok')) else 'FAIL'}` (exit `{report.get('exit_code', 0)}`)",
        f"- **Personal-PC / FinOps**: `{FINANCIAL_PROMISE}` (`AGPL-3.0`, personal home PCs only)",
    ]
    if "live_build_sha" in report:
        lines.append(f"- **Build SHA**: `{report.get('live_build_sha')}` (stale=`{report.get('stale_cloud_run_revision')}`)")
    if isinstance(report.get("gpu"), Mapping):
        lines.append(f"- **GPU Status**: `{report['gpu'].get('status')}` | **Zero-Lag**: `{(report.get('steam_preflight') or {}).get('zero_game_lag')}`")
    lines.append(f"- **Receipt SHA-256**: `{report.get('receipt_sha256', '')}`")
    lines.append(f"- **Summary**: {report.get('summary', '')}")
    return _assert_safe_output("\n".join(lines) + "\n")


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN plain-text CLI summary without secrets or static figures."""
    lang_code = normalize_lang(str(report.get("lang") or "en"))
    status = "OK" if report.get("ok", report.get("compliance_ok")) else "FAIL"
    lines = [
        f"[HiveMind] {status} | exit={report.get('exit_code', 0)} | {FINANCIAL_PROMISE} | AGPL-3.0",
        str(report.get("summary") or ""),
    ]
    if report.get("diagnostics"):
        lines.append(f"{_msg(lang_code, 'Diagnostics', 'Diagnostics')}: {', '.join(str(d) for d in report['diagnostics'])}")
    lines.append(f"Receipt SHA-256: {report.get('receipt_sha256', '')}")
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Construct stdlib CLI parser for CD bundle gate, unified quickstart bundle, and CES queue ops."""
    p = argparse.ArgumentParser(description="Zero-Inference Cloud Run CD Deploy-Bundle & Unified Quickstart Bundle Verifier.")
    p.add_argument("--gateway", default=DEFAULT_GATEWAY)
    p.add_argument("--cd-gate", action="store_true", help="Run CD deploy-bundle FinOps & PR #949 live revision verifier.")
    p.add_argument("--quickstart-bundle", action="store_true", help="Run unified Win11/POSIX/VS Code quickstart bundle verifier.")
    p.add_argument("--enforce-personal-pc", action="store_true", help="Run strictly local Personal-PC compliance check only.")
    p.add_argument("--require-holiday-routes", action="store_true")
    p.add_argument("--expected-sha", default=None)
    p.add_argument("--check-git-head", action="store_true")
    p.add_argument("--skip-deploy-dry-run", action="store_true")
    p.add_argument("--skip-asgi-stage-check", action="store_true")
    p.add_argument("--skip-script-downloads", action="store_true")
    p.add_argument("--strict-launcher-parity", action="store_true")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--workspace-dir", default=None)
    p.add_argument("--home-dir", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--handle", default="noel-bundle-probe")
    p.add_argument("--send-opentrainer-heartbeat", action="store_true")
    p.add_argument("--submit-ces", action="store_true")
    p.add_argument("--flush-ces-queue", action="store_true")
    p.add_argument("--purge-queue", action="store_true")
    p.add_argument("--queue-path", default=None)
    p.add_argument("--profile", default="gamer")
    p.add_argument("--ease-rating", type=int, default=5)
    p.add_argument("--had-game-lag", action="store_true")
    p.add_argument("--host-label", default="personal-pc")
    p.add_argument("--max-temp-c", type=float, default=DEFAULT_MAX_TEMP_C)
    p.add_argument("--min-free-vram-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB)
    p.add_argument("--yield-budget-ms", type=float, default=DEFAULT_YIELD_BUDGET_MS)
    p.add_argument("--lang", default="en", choices=("en", "fr"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--markdown", action="store_true")
    return p


def _emit_report(payload: Mapping[str, Any], *, as_json: bool, as_markdown: bool) -> None:
    if as_json:
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        _assert_safe_output(text)
        print(text)
    elif as_markdown:
        print(render_markdown_report(payload), end="")
    else:
        print(render_cli_report(payload))


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | str | None = None,
) -> int:
    """CLI entrypoint returning ``0`` on pass or ``2`` on compliance/gate failure."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    eff_home = home if home is not None else args.home_dir
    if args.purge_queue:
        q = BundleCesQueue(args.queue_path or (Path(eff_home) if eff_home else Path.home()) / DEFAULT_QUEUE_REL)
        purged = _attach_receipt(
            {
                "schema": SCHEMA,
                "object": "hivemind.bundle_ces_queue.purge",
                "lang": args.lang,
                **q.purge(),
                "ok": True,
                "exit_code": 0,
                "summary": "Local bundle CES queue purged.",
            }
        )
        _emit_report(purged, as_json=args.json, as_markdown=args.markdown)
        return 0
    if args.enforce_personal_pc:
        comp = evaluate_personal_pc_compliance(environ=environ, cwd=cwd or args.workspace_dir, lang=args.lang, home=eff_home)
        _emit_report(comp, as_json=args.json, as_markdown=args.markdown)
        return int(comp["exit_code"])
    if args.cd_gate or args.require_holiday_routes:
        cd_res = verify_cd_deploy_bundle_and_live_revision(
            args.gateway,
            repo_root=args.repo_root,
            expected_sha=args.expected_sha,
            check_git_head=args.check_git_head,
            run_deploy_dry_run=not args.skip_deploy_dry_run,
            run_asgi_stage_check=not args.skip_asgi_stage_check,
            require_holiday_routes=args.require_holiday_routes,
            http_fn=http_fn,
            runner=runner,
            environ=environ,
            cwd=cwd or args.workspace_dir,
            home=eff_home,
            lang=args.lang,
        )
        _emit_report(cd_res, as_json=args.json, as_markdown=args.markdown)
        return int(cd_res["exit_code"])
    qs_res = verify_unified_quickstart_bundle(
        args.gateway,
        repo_root=args.repo_root,
        workspace_dir=args.workspace_dir,
        home_dir=eff_home,
        verify_script_downloads=not args.skip_script_downloads,
        send_opentrainer_heartbeat=args.send_opentrainer_heartbeat,
        submit_ces=args.submit_ces,
        flush_ces_queue=args.flush_ces_queue,
        queue_path=args.queue_path,
        profile=args.profile,
        ease_rating=args.ease_rating,
        had_game_lag=args.had_game_lag,
        host_label=args.host_label,
        max_temp_c=args.max_temp_c,
        min_free_vram_mib=args.min_free_vram_mib,
        yield_budget_ms=args.yield_budget_ms,
        http_fn=http_fn,
        runner=runner,
        environ=environ,
        cwd=cwd,
        lang=args.lang,
        api_key=args.api_key,
        handle=args.handle,
        strict_launcher_parity=args.strict_launcher_parity,
    )
    _emit_report(qs_res, as_json=args.json, as_markdown=args.markdown)
    return int(qs_res["exit_code"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
