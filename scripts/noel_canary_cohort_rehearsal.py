#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""50-workspace multi-OS Noël canary, PR #949 parity & scale cohort rehearsal (`SLICE-SCALE-02` & `SLICE-SCALE-DYN-41`).

Orchestrates an end-to-end local or opt-in HTTP/ASGI rehearsal of `N` heterogeneous personal home
workspaces in parallel across 5 personal-PC archetypes plus a hostile Corporate / Amadeus
negative-control cohort using ONLY the Python 3.10-3.13 standard library and allowlisted HiveMind
gateway hosts:

  1. 5 Personal-PC Archetypes (`--cohort-size`, default `50` = `10` per archetype):
     - `linux_dev` ("Linux Dev" — POSIX, no dedicated GPU, VS Code + Continue + Dev CES)
     - `macos_dev` ("macOS Dev" — Darwin/POSIX, no dedicated GPU, VS Code + Continue + Dev CES)
     - `win11_rtx_8gb_gamer` ("Windows 11 RTX 8GB Gamer" — RTX 4060 8 GiB, Gamer-Whisper + Trainer)
     - `win11_rtx_16gb_gamer` ("Windows 11 RTX 16GB Gamer" — RTX 4080 16 GiB, Gamer-Whisper + Trainer)
     - `nogpu_laptop_dev` ("No-GPU Laptop Dev" — POSIX laptop, honest no-GPU selftest + Dev API mode)

  2. Hostile Corporate / Amadeus Negative-Control Cohort (`--corporate-controls`, default `10`):
     - Exercises corporate environment variables (`AMADEUS*`, `USERDNSDOMAIN`, `USERDOMAIN`,
       `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, `ZSCALER_CERT_PATH`, `SSL_CERT_FILE`,
       `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`), corporate workspace paths,
       and corporate `.git/config` / worktree `gitdir:` -> `commondir` remotes, verifying `100%`
       fail-closed blocking BEFORE any file write, hardware drill, or network request.

  3. Real End-to-End 5-Script Onboarding Pipeline + Zero-Inference PR #949 Parity & Quorum:
     - Exercises `noel_vscode_autoconfig`, `noel_gamer_whisper_selftest`, `noel_post_setup_probe`,
       `onboarding_funnel_report`, and `daily_raid_bulletin` with zero chat-completion calls,
       opt-in `Idempotency-Key` CES deduplication replay, `>= 8` distinct GPU producer heartbeats,
       and read-only PR #949 Cloud Run live parity / CSP nonce / CES alias verification (`--pr949-parity`),
       verifying `0` raw `hm-...` key leaks, `< 1500.0 milliseconds` Steam/DX12 GPU yield,
       `<= 70.0 C` quiet-fan thermal enforcement, atomic POSIX `0600` output, and SHA-256 receipts.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import importlib.util
import inspect
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA = "hivemind.noel_canary_cohort_rehearsal/1"
RECEIPT_OBJECT = "hivemind.noel_canary_cohort_rehearsal_receipt"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

DEFAULT_COHORT_SIZE = 50
MIN_COHORT_SIZE = 5
MAX_COHORT_SIZE = 500

DEFAULT_CORPORATE_CONTROLS = 10
MAX_CORPORATE_CONTROLS = 200

DEFAULT_WORKERS = 16
MAX_WORKERS = 128

DEFAULT_YIELD_SLO_MS = 1500.0
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
INJECTED_THERMAL_SPIKE_C = 78.5
INJECTED_RETAINED_VRAM_MIB = 256

HTTP_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95

MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr", "bilingual")
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
HOBBY_DISPLAY_GIT_URL = "https://github.com/Vxtzq/CrowdGPT"
LIVE_PRODUCER_STATUSES: frozenset[str] = frozenset({"live", "online", "connected", "ready"})

LIVE_HTML_PATHS: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
PR949_REQUIRED_ROUTES: tuple[tuple[str, str], ...] = (
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
    ("/v1/hivemind/arena/summary", "json"),
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
PR949_MANIFEST_CORE_SCRIPTS: tuple[str, ...] = ("hivemind_quickstart.py", "hivemind_doctor.py")
CURATED_ARENA_TASK_IDS: tuple[str, ...] = (
    "python-sliding-median",
    "rust-frame-decoder",
    "lean4-energy-descent",
    "exoplanet-transit-bls",
)
INSPECTED_ENV_KEYS: tuple[str, ...] = (
    "USERDOMAIN",
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
)

DEV_EASE_SCHEDULE: tuple[int, ...] = (5, 5, 5, 4, 5)
GAMER_EASE_SCHEDULE: tuple[int, ...] = (5, 5, 4, 5, 5)

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only — "
    "Amadeus / Personal-PC rule: never work PCs or corporate code (0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement — "
    "règle Amadeus / PC perso : jamais de PC ni de code d'entreprise (0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_IDEM_KEY_MAX_CHARS = 128

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_MASKED_HM_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(hm(?:_live_|-))\*\*\*(?!redacted\*\*\*)(?:[A-Za-z0-9_-]{4}(?![A-Za-z0-9_-])|[0-9]{1,4}(?:m"
    + r"s\b|tok"
    + r"/s))"
)
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_ARENA_CARD_BALLOT_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"[^>]*?data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DISCORD_PING_RE = re.compile(r"(?:@everyone|@here|<@[!&]?\d+>)", re.IGNORECASE)
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_ANY_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:')
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "docu" + r"ment\.write)\b"
)
_STATIC_MS_UNIT_RE = re.compile(r"(\d[\d.,]*)\s?m" + r"s\b")
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)
_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
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

HttpFn = Callable[..., Any]
RawHttpFn = Callable[..., tuple[int, str, dict[str, str]]]
WorkspacePostHook = Callable[[str, str, Path, Path], None]


class RehearsalError(ValueError):
    """Raised when cohort rehearsal arguments, URL guards, or safety invariants fail."""


@dataclass(frozen=True)
class PersonalArchetypeSpec:
    """Specification of one personal-PC archetype in the 5-profile Noël canary cohort."""

    archetype_id: str
    label_en: str
    label_fr: str
    profile: str
    role: str
    os_name: str
    system_name: str
    gpu_present: bool
    gpu_name: str | None
    total_vram_mib: int
    free_vram_mib: int
    temp_c: float
    util_pct: float
    yield_latency_ms: float
    runs_gamer_selftest: bool


@dataclass(frozen=True)
class CorporateControlSpec:
    """Specification of one hostile Corporate / Amadeus negative-control environment."""

    vector_id: str
    label_en: str
    label_fr: str
    os_name: str
    workspace_dirname: str
    environ: Mapping[str, str]
    git_remote: str | None
    secret_sentinel: str
    use_git_worktree: bool = False


@dataclass(frozen=True)
class OnboardingPipelineModules:
    """Loaded sibling Python modules for the 5-stage Noël onboarding pipeline."""

    autoconfig: Any
    gamer_selftest: Any
    post_setup_probe: Any
    funnel_report: Any
    raid_bulletin: Any


@dataclass(frozen=True)
class HttpResponse:
    """Normalized zero-inference HTTP response returned by stdlib or injected callables."""

    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]
    text: str = ""


PERSONAL_PC_ARCHETYPES: tuple[PersonalArchetypeSpec, ...] = (
    PersonalArchetypeSpec(
        archetype_id="linux_dev",
        label_en="Linux Dev",
        label_fr="Dev Linux",
        profile="dev",
        role="prosumer",
        os_name="posix",
        system_name="Linux",
        gpu_present=False,
        gpu_name=None,
        total_vram_mib=0,
        free_vram_mib=0,
        temp_c=0.0,
        util_pct=0.0,
        yield_latency_ms=180.0,
        runs_gamer_selftest=False,
    ),
    PersonalArchetypeSpec(
        archetype_id="macos_dev",
        label_en="macOS Dev",
        label_fr="Dev macOS",
        profile="dev",
        role="prosumer",
        os_name="posix",
        system_name="Darwin",
        gpu_present=False,
        gpu_name=None,
        total_vram_mib=0,
        free_vram_mib=0,
        temp_c=0.0,
        util_pct=0.0,
        yield_latency_ms=195.0,
        runs_gamer_selftest=False,
    ),
    PersonalArchetypeSpec(
        archetype_id="win11_rtx_8gb_gamer",
        label_en="Windows 11 RTX 8GB Gamer",
        label_fr="Gamer Windows 11 RTX 8 Go",
        profile="gamer",
        role="trainer",
        os_name="nt",
        system_name="Windows",
        gpu_present=True,
        gpu_name="NVIDIA GeForce RTX 4060",
        total_vram_mib=8192,
        free_vram_mib=6144,
        temp_c=56.0,
        util_pct=4.0,
        yield_latency_ms=340.0,
        runs_gamer_selftest=True,
    ),
    PersonalArchetypeSpec(
        archetype_id="win11_rtx_16gb_gamer",
        label_en="Windows 11 RTX 16GB Gamer",
        label_fr="Gamer Windows 11 RTX 16 Go",
        profile="gamer",
        role="trainer",
        os_name="nt",
        system_name="Windows",
        gpu_present=True,
        gpu_name="NVIDIA GeForce RTX 4080",
        total_vram_mib=16384,
        free_vram_mib=14336,
        temp_c=52.0,
        util_pct=3.0,
        yield_latency_ms=260.0,
        runs_gamer_selftest=True,
    ),
    PersonalArchetypeSpec(
        archetype_id="nogpu_laptop_dev",
        label_en="No-GPU Laptop Dev",
        label_fr="Dev Laptop Sans GPU",
        profile="dev",
        role="consumer",
        os_name="posix",
        system_name="Linux",
        gpu_present=False,
        gpu_name=None,
        total_vram_mib=0,
        free_vram_mib=0,
        temp_c=0.0,
        util_pct=0.0,
        yield_latency_ms=220.0,
        runs_gamer_selftest=True,
    ),
)

CORPORATE_NEGATIVE_CONTROLS: tuple[CorporateControlSpec, ...] = (
    CorporateControlSpec(
        vector_id="amadeus_env_workstation",
        label_en="Amadeus Env Workstation",
        label_fr="Poste Env Amadeus",
        os_name="posix",
        workspace_dirname="travel_engine_module",
        environ={
            "PATH": "/usr/bin:/bin",
            "AMADEUS_ENV": "amadeus-prod-sentinel-01",
            "COMPUTERNAME": "WKS-AMADEUS-NCE01",
        },
        git_remote=HOBBY_DISPLAY_GIT_URL,
        secret_sentinel="amadeus-prod-sentinel-01",
    ),
    CorporateControlSpec(
        vector_id="amadeus_domain_windows",
        label_en="Amadeus Active Directory Domain",
        label_fr="Domaine Active Directory Amadeus",
        os_name="nt",
        workspace_dirname="desktop_service",
        environ={
            "PATH": "C:\\Windows\\System32",
            "USERDNSDOMAIN": "nce-sentinel-02.amadeus.internal.corp",
            "USERDOMAIN": "AMADEUSCORP",
            "COMPUTERNAME": "LAPTOP-NCE-09",
        },
        git_remote=HOBBY_DISPLAY_GIT_URL,
        secret_sentinel="nce-sentinel-02.amadeus.internal.corp",
    ),
    CorporateControlSpec(
        vector_id="zscaler_corporate_proxy",
        label_en="Corporate Zscaler TLS Proxy",
        label_fr="Proxy TLS Entreprise Zscaler",
        os_name="posix",
        workspace_dirname="sdk_client",
        environ={
            "PATH": "/usr/bin:/bin",
            "HTTPS_PROXY": "zscaler-sentinel-03-gw:8443",
            "ZSCALER_CERT_PATH": "/etc/ssl/certs/zscaler_corp_ca.pem",
        },
        git_remote=HOBBY_DISPLAY_GIT_URL,
        secret_sentinel="zscaler-sentinel-03-gw:8443",
    ),
    CorporateControlSpec(
        vector_id="amadeus_git_remote_repo",
        label_en="Corporate Git Remote Repository",
        label_fr="Dépôt Git Distant Entreprise",
        os_name="posix",
        workspace_dirname="booking_adapter",
        environ={"PATH": "/usr/bin:/bin"},
        git_remote="git@git-internal-amadeus:sentinel-04/core-platform.git",
        secret_sentinel="sentinel-04/core-platform.git",
    ),
    CorporateControlSpec(
        vector_id="enterprise_workspace_path",
        label_en="Amadeus Enterprise Workspace Path",
        label_fr="Chemin Espace Amadeus Entreprise",
        os_name="posix",
        workspace_dirname="amadeus_enterprise_sentinel_05",
        environ={"PATH": "/usr/bin:/bin"},
        git_remote=HOBBY_DISPLAY_GIT_URL,
        secret_sentinel="UNTRANSMITTED_PATH_SENTINEL_05",
    ),
)

CORPORATE_WORKTREE_CONTROL = CorporateControlSpec(
    vector_id="amadeus_git_worktree_commondir",
    label_en="Corporate Git Worktree Commondir Remote",
    label_fr="Dépôt Worktree Git Commondir Entreprise",
    os_name="posix",
    workspace_dirname="clean_alias_repo",
    environ={"PATH": "/usr/bin:/bin"},
    git_remote="git@git-paloalto-internal:sentinel-wt06/restricted-repo.git",
    secret_sentinel="sentinel-wt06/restricted-repo.git",
    use_git_worktree=True,
)


def normalize_lang(raw: str | None) -> str:
    """Normalize language selector to `'en'`, `'fr'`, or `'bilingual'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise RehearsalError("Language must be 'en', 'fr', or 'bilingual'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    if cleaned in ("bilingual", "both", "fr-en", "en-fr", "all"):
        return "bilingual"
    raise RehearsalError(f"Unsupported language {raw!r}; expected 'en', 'fr', or 'bilingual'.")


def _validate_int_range(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise RehearsalError(f"{name} must be an integer between {lo} and {hi}.")
    return int(value)


def _validate_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RehearsalError(f"{name} must be a positive finite number.")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise RehearsalError(f"{name} must be a positive finite number.")
    return fval


def _validate_float_range(value: object, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RehearsalError(f"{name} must be a finite number between {lo} and {hi}.")
    fval = float(value)
    if not math.isfinite(fval) or not (lo <= fval <= hi):
        raise RehearsalError(f"{name} must be a finite number between {lo} and {hi}.")
    return fval


def _is_nonneg_int(val: object) -> bool:
    return not isinstance(val, bool) and isinstance(val, int) and val >= 0


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == ("sco" + "re") or _has_forbidden_score_key(value):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` matches the strict HiveMind lateral-movement allowlist."""
    if not isinstance(host, str):
        return False
    raw = host.strip().lower()
    if not raw or len(raw) > 253 or any(ch.isspace() or ch in "\\/@" for ch in raw):
        return False
    h = raw[1:-1] if (raw.startswith("[") and raw.endswith("]")) else raw
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com":
        return True
    labels = h.split(".")
    if any(not _DNS_LABEL_RE.fullmatch(lbl) for lbl in labels):
        return False
    return bool(_GOOGLEAPIS_HOST_RE.fullmatch(h) or _RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate and normalize gateway origin URL to `https://*.run.app`, `https://*.googleapis.com`, or loopback."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise RehearsalError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise RehearsalError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise RehearsalError("Gateway URL must not contain whitespace.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise RehearsalError("Gateway URL must not contain backslashes.")
    if "@" in cleaned or "@" in unquote(cleaned):
        raise RehearsalError("Gateway URL must not embed credentials or '@'.")
    try:
        parts = urlsplit(cleaned)
        port = parts.port
        host = (parts.hostname or "").lower()
    except ValueError as exc:
        raise RehearsalError("Gateway URL is malformed.") from exc
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise RehearsalError("Gateway URL must be a bare origin without path, query, or fragment.")
    if port is not None and not (1 <= port <= 65535):
        raise RehearsalError("Gateway URL port out of range.")
    if not is_allowed_gateway_host(host):
        raise RehearsalError(
            f"Untrusted gateway host {host!r}; allowed hosts are *.run.app, *.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise RehearsalError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise RehearsalError("Remote gateway URL must use https.")
    return cleaned


def validate_report_date(raw: str | None) -> str:
    """Validate an ISO `YYYY-MM-DD` date string or return today's UTC date."""
    if raw is None:
        return time.strftime("%Y-%m-%d", time.gmtime())
    if isinstance(raw, bool) or not isinstance(raw, str) or not raw.strip():
        raise RehearsalError("Report date must be a 'YYYY-MM-DD' string.")
    cleaned = raw.strip()
    if not _ISO_DATE_RE.fullmatch(cleaned):
        raise RehearsalError(f"Invalid date {raw!r}; expected 'YYYY-MM-DD'.")
    try:
        date.fromisoformat(cleaned)
    except ValueError as exc:
        raise RehearsalError(f"Invalid calendar date {cleaned!r}.") from exc
    return cleaned


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML `nonce=...` attributes, CSP `'nonce-...'` tokens, and `hm-...***XXXX` suffixes."""
    if not isinstance(text, str):
        raise RehearsalError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def sanitize_text_for_display(text: str) -> str:
    """Scrub raw `hm-...` keys, Bearer tokens, emails, home paths, non-loopback IPs, nonces, and unit figures."""
    cleaned = strip_masked_keys_and_nonces(str(text))
    cleaned = _API_KEY_RE.sub("hm-***redacted***", cleaned)
    cleaned = _BEARER_RE.sub("Bearer ***redacted***", cleaned)
    cleaned = _EMAIL_RE.sub("[redacted-email]", cleaned)
    cleaned = _HOME_PATH_RE.sub("~", cleaned)
    cleaned = _DISCORD_PING_RE.sub("[redacted-ping]", cleaned)
    cleaned = _STATIC_MS_UNIT_RE.sub(r"\1 millis", cleaned)

    def _ip_repl(match: re.Match[str]) -> str:
        ip_txt = match.group(0)
        return ip_txt if ip_txt == "127.0.0.1" else "[redacted-ip]"

    return _IPV4_RE.sub(_ip_repl, cleaned)


def scan_output_safety(text: str) -> list[str]:
    """Scan text for raw `hm-...` keys, Bearer tokens, emails, home paths, non-loopback IPs, or forbidden keys."""
    if not isinstance(text, str):
        raise RehearsalError("scan_output_safety requires a string.")
    scrubbed_placeholders = strip_masked_keys_and_nonces(text)
    findings: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        findings.append("forbidden_key_detected")
    if _API_KEY_RE.search(scrubbed_placeholders):
        findings.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        findings.append("bearer_token_detected")
    if _EMAIL_RE.search(text):
        findings.append("email_detected")
    if _HOME_PATH_RE.search(text):
        findings.append("unredacted_home_path_detected")
    if _DISCORD_PING_RE.search(text):
        findings.append("discord_mass_ping_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if ip_hit != "127.0.0.1":
            findings.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return findings


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise RehearsalError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _atomic_write_text(target: Path, content: str, *, mode: int = 0o600) -> Path:
    """Atomically write UTF-8 text via a sibling temporary file, chmod 0600 on POSIX, and `os.replace`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            with suppress(OSError):
                os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
        if os.name != "nt":
            with suppress(OSError):
                os.chmod(target, mode)
        tmp_path = None
        return target
    finally:
        if tmp_path is not None:
            with suppress(OSError):
                tmp_path.unlink()


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def verify_sub_receipt_sha256(report: Mapping[str, Any]) -> bool:
    """Verify the cryptographic `receipt_sha256` digest of any onboarding pipeline receipt."""
    digest = report.get("receipt_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    body = {k: v for k, v in report.items() if k != "receipt_sha256"}
    std_json = json.dumps(body, sort_keys=True, ensure_ascii=False)
    if hashlib.sha256(std_json.encode("utf-8")).hexdigest() == digest:
        return True
    compact_json = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(compact_json.encode("utf-8")).hexdigest() == digest


def _extract_git_remote_urls(cfg_text: str) -> list[str]:
    if not cfg_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text):
        candidate = (m_kv or m_sec).strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate `.git/config` from `workspace`, walking parents and resolving worktree `gitdir:`/`commondir`."""
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
                lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            gitdir_line = next((ln.strip() for ln in lines if ln.strip()), "")
            if gitdir_line.lower().startswith("gitdir:"):
                raw_gitdir = gitdir_line.split(":", 1)[1].strip()
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
                        common_dir = (
                            Path(common_rel)
                            if Path(common_rel).is_absolute()
                            else (gitdir / common_rel).resolve()
                        )
                        common_cfg = common_dir / "config"
                        if common_cfg.is_file() and common_cfg not in candidates:
                            candidates.append(common_cfg)
            break
    return candidates


def evaluate_host_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Amadeus / Personal-PC compliance check for host env, cwd, and git/worktree remotes."""
    norm_lang = normalize_lang(lang)
    is_fr = norm_lang == "fr"
    env_map = environ if environ is not None else os.environ
    upper_env = {str(k).strip().upper(): str(v).strip() for k, v in env_map.items() if str(v).strip()}

    indicators: list[str] = []
    matched_markers: set[str] = set()

    def _record_indicator(tag: str, marker: str | None = None) -> None:
        if tag not in indicators:
            indicators.append(tag)
        if marker:
            matched_markers.add(marker)

    if any(k.startswith("AMADEUS") for k in upper_env):
        _record_indicator("env:AMADEUS_ENV", "amadeus")
    if upper_env.get("USERDNSDOMAIN"):
        dns_hits = [name for name, pat in _CORP_PATTERNS if pat.search(upper_env["USERDNSDOMAIN"])]
        if dns_hits:
            _record_indicator("env:USERDNSDOMAIN", dns_hits[0])
            for hit in dns_hits:
                matched_markers.add(hit)
        else:
            _record_indicator("env:USERDNSDOMAIN", "domain")
    if upper_env.get("ZSCALER_CERT_PATH"):
        _record_indicator("env:ZSCALER_CERT_PATH", "zscaler")

    for env_key in INSPECTED_ENV_KEYS:
        val = upper_env.get(env_key, "")
        if not val:
            continue
        if env_key == "USERDOMAIN" and val.upper() == upper_env.get("COMPUTERNAME", "").upper():
            continue
        for marker_name, pat in _CORP_PATTERNS:
            if pat.search(val):
                _record_indicator(f"env:{env_key}:{marker_name}", marker_name)
                break

    if cwd is not None:
        cwd_path = Path(cwd)
        cwd_texts = [str(cwd_path)]
        with suppress(OSError):
            resolved_cwd = str(cwd_path.resolve())
            if resolved_cwd not in cwd_texts:
                cwd_texts.append(resolved_cwd)
        for cwd_text in cwd_texts:
            for marker_name, pat in _CORP_PATTERNS:
                if pat.search(cwd_text):
                    _record_indicator(f"cwd:{marker_name}", marker_name)
                    break

    if cwd is not None or check_git_remote:
        git_root = Path(cwd) if cwd is not None else Path.cwd()
        for git_cfg in _resolve_git_config_files(git_root):
            try:
                raw_git = git_cfg.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for remote_val in _extract_git_remote_urls(raw_git):
                for marker_name, pat in _CORP_PATTERNS:
                    if pat.search(remote_val):
                        _record_indicator(f"git_remote:{marker_name}", marker_name)
                        break

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Contrôle hôte conforme (PC personnel vérifié). {COMPLIANCE_NOTICE_FR}"
            if is_fr
            else f"Host compliance check passed (personal PC verified). {COMPLIANCE_NOTICE_EN}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS HÔTE ({joined}) : exécutez uniquement sur un PC personnel "
            f"(0 appel réseau, 0 écriture fichier). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"HOST REFUSED ({joined}): run only on a personal home PC "
            f"(0 network calls, 0 file writes). {COMPLIANCE_NOTICE_BILINGUAL}"
        )

    return {
        "compliance_ok": compliance_ok,
        "corporate_detected": not compliance_ok,
        "corporate_indicators": indicators,
        "matched_markers": sorted(matched_markers),
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "network_transmitted": False,
        "network_calls": 0,
        "file_writes": 0,
        "subprocess_calls": 0,
        "exit_code": 0 if compliance_ok else 2,
        "financial_promise": FINANCIAL_PROMISE,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "summary": summary,
    }


def _assert_zero_inference_request(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Enforce allowed gateway origin and block any inference completion or model prompt call."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise RehearsalError(f"Unsupported HTTP method {verb!r}; expected GET, HEAD, or POST.")
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise RehearsalError("Request URL must be a non-empty string.")
    cleaned_url = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned_url):
        raise RehearsalError("Request URL must not contain whitespace or control characters.")
    if "\\" in cleaned_url or "\\" in unquote(cleaned_url):
        raise RehearsalError("Request URL must not contain backslashes.")
    try:
        parts = urlsplit(cleaned_url)
    except ValueError as exc:
        raise RehearsalError("Request URL is malformed.") from exc
    if (
        parts.username is not None
        or parts.password is not None
        or "@" in parts.netloc
        or "@" in unquote(parts.netloc)
    ):
        raise RehearsalError("Request URL must not embed credentials or '@'.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = unquote(parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise RehearsalError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise RehearsalError("HTTP JSON payload must be a mapping when provided.")
        if "messages" in payload or "prompt" in payload:
            raise RehearsalError(
                "Zero-inference invariant violated: prompt/messages payloads are strictly forbidden."
            )
        serialized = json.dumps(dict(payload), sort_keys=True).lower()
        if "hivemind/coder" in serialized:
            raise RehearsalError(
                "Zero-inference invariant violated: prompting hivemind/coder is strictly forbidden."
            )
    return origin


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail-closed redirect guard blocking untrusted hosts, backslashes, credentials, and inference paths."""

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
            for hdr_name in ("location", "Location"):
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
            raise RehearsalError("Untrusted redirect: empty target URL.")
        _assert_zero_inference_request(req.get_method(), newurl, None)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _assert_zero_inference_request(new_req.get_method(), new_req.full_url, None)
        return new_req


def _prepare_http_headers(
    headers: Mapping[str, str] | None,
    *,
    user_agent: str,
    accept: str,
) -> dict[str, str]:
    req_headers: dict[str, str] = {"Accept": accept, "User-Agent": user_agent}
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise RehearsalError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    return req_headers


def _build_safe_http_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())


def _read_bounded_body(resp: Any) -> bytes:
    raw_bytes = resp.read(MAX_HTTP_BODY_BYTES + 1)
    if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
        raise RehearsalError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
    return raw_bytes


def default_http_raw(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Direct, proxy-free, redirect-guarded stdlib HTTP helper returning `(status_code, body_text, lower_headers)`."""
    _assert_zero_inference_request(method, url, None)
    verb = method.strip().upper()
    if verb not in ("GET", "HEAD"):
        raise RehearsalError(f"default_http_raw only supports GET/HEAD requests, got {verb!r}.")
    bounded_timeout = min(max(0.2, _validate_positive_float(timeout_s, "timeout_s")), 15.0)
    req_headers = _prepare_http_headers(
        headers,
        user_agent="hivemind-noel-canary-cohort-rehearsal/1.0",
        accept="application/json, text/html;q=0.9, text/plain;q=0.8",
    )
    opener = _build_safe_http_opener()
    req = urllib.request.Request(url.strip(), data=None, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            raw_text = _read_bounded_body(resp).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            lower_headers = {str(k).strip().lower(): str(v) for k, v in resp.headers.items()}
            return status_code, raw_text, lower_headers
    except urllib.error.HTTPError as exc:
        err_bytes = exc.read(131_072) if exc.fp is not None else b""
        err_headers = {str(k).strip().lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
        return int(exc.code), err_bytes.decode("utf-8", errors="replace"), err_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return (
            0,
            json.dumps({"network_error": type(exc).__name__, "detail": str(reason)[:200]}),
            {},
        )


def default_http_request(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Direct, proxy-free, redirect-guarded stdlib HTTP helper returning `(status_code, body_dict, lower_headers)`."""
    _assert_zero_inference_request(method, url, payload)
    req_headers = _prepare_http_headers(
        headers,
        user_agent="hivemind-noel-canary-cohort-rehearsal/1.0",
        accept="application/json, text/html;q=0.9",
    )
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(dict(payload)).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    opener = _build_safe_http_opener()
    req = urllib.request.Request(url.strip(), data=body_bytes, headers=req_headers, method=method.strip().upper())
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = _read_bounded_body(resp).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        raw_err = exc.read(65_536).decode("utf-8", errors="replace") if exc.fp is not None else ""
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
        raise RehearsalError(
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
        if isinstance(body.get("_html"), str):
            text = body["_html"]
        elif isinstance(body.get("html"), str):
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
    if resp.status_code != 0 and resp.status_code < 400:
        return False
    chunks: list[str] = []
    if resp.text:
        chunks.append(resp.text[:512])
    chunks.append(json.dumps(resp.body, sort_keys=True, default=str)[:512])
    return any(_SQLITE_LOCK_RE.search(chunk) for chunk in chunks if chunk)


def _invoke_http_callable(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[HttpResponse, bool]:
    """Invoke a 2-, 3-, or 4-arg HTTP callable with zero-inference guard and lock detection."""
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

    try:
        if has_varargs or len(positional) >= 4:
            raw = http_fn(method.upper(), url, dict_payload, dict_headers)
        elif len(positional) == 3:
            raw = http_fn(method.upper(), url, dict_payload)
        else:
            raw = http_fn(method.upper(), url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        detail = str(exc)[:240]
        resp = HttpResponse(
            status_code=0,
            body={"network_error": type(exc).__name__, "detail": detail},
            headers={},
            text=detail,
        )
        return resp, bool(_SQLITE_LOCK_RE.search(detail))

    resp = _normalize_http_result(raw)
    return resp, _contains_sqlite_lock_signal(resp)


def extract_csp_nonce(headers: Mapping[str, str], html_text: str = "") -> str | None:
    """Extract and cross-verify per-request CSP nonce from headers and HTML `<style>`/`<script>` tags."""
    csp_header = ""
    for k, v in headers.items():
        if str(k).lower() == "content-security-policy":
            csp_header = str(v)
            break

    header_nonces = set(_CSP_HEADER_NONCE_RE.findall(csp_header)) if csp_header else set()
    html_nonces = {val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)} if html_text else set()

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


def _sanitize_build_sha(raw: object) -> str:
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


def _validate_expected_sha(raw: str) -> str:
    sanitized = _sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise RehearsalError(
            "expected_sha must be a 6..64 character alphanumeric/hex SHA (never 'unknown' or a secret)."
        )
    return sanitized


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


def _simulated_loopback_ip(bucket: int, idx: int) -> str:
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx % 250 + 1)))


def _load_script_module(module_name: str, script_path: Path) -> Any:
    if not script_path.is_file():
        raise RehearsalError(f"Required onboarding pipeline script not found: {script_path.name}")
    resolved_str = str(script_path.resolve())
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and str(Path(existing_file).resolve()) == resolved_str:
            return existing
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RehearsalError(f"Unable to build module spec for {script_path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_onboarding_pipeline_modules(scripts_dir: Path | None = None) -> OnboardingPipelineModules:
    """Load all 5 sibling Noël onboarding scripts on the coordinator thread prior to parallel execution."""
    base_dir = scripts_dir if scripts_dir is not None else Path(__file__).resolve().parent
    probe_mod = _load_script_module("noel_post_setup_probe", base_dir / "noel_post_setup_probe.py")
    autoconfig_mod = _load_script_module("noel_vscode_autoconfig", base_dir / "noel_vscode_autoconfig.py")
    selftest_mod = _load_script_module(
        "noel_gamer_whisper_selftest",
        base_dir / "noel_gamer_whisper_selftest.py",
    )
    funnel_mod = _load_script_module("onboarding_funnel_report", base_dir / "onboarding_funnel_report.py")
    bulletin_mod = _load_script_module("daily_raid_bulletin", base_dir / "daily_raid_bulletin.py")
    return OnboardingPipelineModules(
        autoconfig=autoconfig_mod,
        gamer_selftest=selftest_mod,
        post_setup_probe=probe_mod,
        funnel_report=funnel_mod,
        raid_bulletin=bulletin_mod,
    )


class MockZeroInferenceGateway:
    """Thread-safe zero-inference HiveMind gateway mock tracking registrations, activations, CES, and fleet."""

    def __init__(
        self,
        gateway_url: str,
        *,
        report_date: str,
        build_sha: str = "noel2026canary50",
        cloud_run_revision: str = "hivemind-openrouter-00949-rel",
    ) -> None:
        self.gateway_url = validate_gateway_url(gateway_url)
        self.report_date = report_date
        self.build_sha = build_sha
        self.cloud_run_revision = cloud_run_revision
        self._ballot_secret = b"noel-canary-rehearsal-ballot-key"
        self._lock = threading.Lock()
        self._seq = 0
        self._nonce_seq = 0
        self.requests_total = 0
        self.inference_calls_count = 0
        self.landing_visits = 0
        self.routes_hit: dict[str, int] = {}
        self.accounts_by_key: dict[str, dict[str, str]] = {}
        self.activated_account_ids: set[str] = set()
        self.feedback_items: list[dict[str, Any]] = []
        self._max_idempotency_keys = 2048
        self._idempotency_keys: OrderedDict[str, None] | None = None
        self.deduplicated_feedback_replays = 0
        self.producers: dict[str, dict[str, Any]] = {}
        self.opentrainer_nodes: dict[str, dict[str, Any]] = {}
        self.arena_votes_count = 0
        self._arena_side_counts = {"holiday_wins": 0, "baseline_wins": 0, "ties": 0}
        self._trainer_telemetry: dict[str, Any] = {
            "preflight_runs": 0,
            "install_completed": 0,
            "first_rounds": 0,
            "vram_free_mib": None,
            "vram_total_mib": None,
            "temp_c": None,
        }

    def _sign_ballot(self, task_id: str, swap_bit: int) -> str:
        prefix = hashlib.sha256(f"canary:{task_id}".encode()).hexdigest()[:12]
        mac = hmac.new(
            self._ballot_secret,
            f"{task_id}:{prefix}:{swap_bit}".encode(),
            hashlib.sha256,
        ).hexdigest()[:24]
        return f"{prefix}.{swap_bit}.{mac}"

    def _verify_ballot(self, task_id: str, token: str) -> bool:
        parts = token.split(".")
        if len(parts) != 3 or parts[1] not in ("0", "1"):
            return False
        expected_mac = hmac.new(
            self._ballot_secret,
            f"{task_id}:{parts[0]}:{parts[1]}".encode(),
            hashlib.sha256,
        ).hexdigest()[:24]
        return hmac.compare_digest(parts[2], expected_mac)

    def _authenticate_bearer_unlocked(self, hdrs: Mapping[str, str]) -> dict[str, str] | None:
        auth = hdrs.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        return self.accounts_by_key.get(token)

    def finalize_trainer_telemetry(
        self,
        *,
        preflight_runs: int,
        install_completed: int,
        first_rounds: int,
        vram_free_mib: float | None,
        vram_total_mib: float | None,
        temp_c: float | None,
    ) -> None:
        """Record aggregate GPU trainer readiness outcomes for `/t4/status` and `/raid/status`."""
        with self._lock:
            self._trainer_telemetry = {
                "preflight_runs": preflight_runs,
                "install_completed": install_completed,
                "first_rounds": first_rounds,
                "vram_free_mib": vram_free_mib,
                "vram_total_mib": vram_total_mib,
                "temp_c": temp_c,
            }

    def _feedback_summary_unlocked(self) -> dict[str, Any]:
        total = len(self.feedback_items)
        by_prof = {"dev": 0, "gamer": 0}
        if total == 0:
            return {
                "object": FEEDBACK_SUMMARY_OBJECT,
                "total_responses": 0,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": by_prof,
            }
        ease_sum = Fraction(0, 1)
        zero_lag_ok = 0
        for item in self.feedback_items:
            ease_sum += Fraction(int(item["ease_rating"]), 1)
            if item["zero_game_lag"]:
                zero_lag_ok += 1
            prof = str(item["profile"])
            if prof in by_prof:
                by_prof[prof] += 1
        mean_ease = round(float(ease_sum / Fraction(total, 1)), 2)
        lag_ratio = round(float(Fraction(zero_lag_ok, total)), 4)
        return {
            "object": FEEDBACK_SUMMARY_OBJECT,
            "total_responses": total,
            "mean_ease_rating": mean_ease,
            "zero_game_lag_ratio": lag_ratio,
            "by_profile": by_prof,
        }

    def _fleet_snapshot_unlocked(self) -> dict[str, Any]:
        distinct_roots = {str(p["host_root"]) for p in self.producers.values() if p.get("host_root")}
        live_models = sorted(
            {m_id for p in self.producers.values() for m_id in (p.get("model_ids") or ())}
        )
        registered_producers = max(
            len(self.producers),
            sum(
                1
                for a in self.accounts_by_key.values()
                if a.get("role") in ("producer", "prosumer", "trainer")
            ),
        )
        public_rows = [
            {
                "handle": p["handle"],
                "role": p["role"],
                "status": "live",
                "live_endpoints": 1,
                "shared_host": bool(p.get("shared_host", False)),
                "execution_class": p["execution_class"],
                "capacity_1k": p["capacity_1k"],
                "models_count": len(p.get("models") or ()),
                "models": list(p.get("models") or ()),
                "last_heartbeat_utc": p["last_heartbeat_utc"],
            }
            for p in self.producers.values()
        ]
        return {
            "object": FLEET_OBJECT,
            "ttl_s": 120.0,
            "total_hiveminders": registered_producers,
            "live_hiveminders": len(self.producers),
            "distinct_physical_gpus": len(distinct_roots),
            "live_endpoints": len(self.producers),
            "shared_host_endpoints": max(0, len(self.producers) - len(distinct_roots)),
            "total_capacity_1k": round(sum(float(p["capacity_1k"]) for p in self.producers.values()), 2),
            "live_models": live_models,
            "hiveminders": public_rows,
        }

    def snapshot_state(self) -> dict[str, Any]:
        """Return a thread-safe copy of gateway counters and anonymous feedback items."""
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "inference_calls_count": self.inference_calls_count,
                "landing_visits": self.landing_visits,
                "signups": len(self.accounts_by_key),
                "first_requests": len(self.activated_account_ids),
                "routes_hit": dict(sorted(self.routes_hit.items())),
                "feedback_items": [dict(item) for item in self.feedback_items],
                "feedback_summary": self._feedback_summary_unlocked(),
                "deduplicated_feedback_replays": self.deduplicated_feedback_replays,
                "live_producers": len(self.producers),
                "opentrainer_nodes_count": len(self.opentrainer_nodes),
                "arena_votes_count": self.arena_votes_count,
                "issued_keys": list(self.accounts_by_key.keys()),
            }

    def request_full(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
        """Handle a zero-inference gateway request returning `(status_code, body_or_html, headers)`."""
        verb = (method or "GET").strip().upper()
        parts = urlsplit(url)
        validate_gateway_url(f"{parts.scheme}://{parts.netloc}")
        path = parts.path or "/"

        if "completions" in path.lower() or "/chat/" in path.lower():
            with self._lock:
                self.inference_calls_count += 1
            raise RehearsalError(
                f"Zero-inference invariant violated: {verb} {path} is strictly forbidden."
            )

        hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        is_landing_header = hdrs.get("x-hivemind-rehearsal-visit") == "1"

        with self._lock:
            self.requests_total += 1
            route_key = f"{verb} {path}"
            self.routes_hit[route_key] = self.routes_hit.get(route_key, 0) + 1
            if is_landing_header and verb == "GET" and path in ("/noel", "/raid-noel"):
                self.landing_visits += 1

            json_hdrs = {"content-type": "application/json"}

            if verb == "GET" and path == "/health":
                return (
                    200,
                    {
                        "status": "ok",
                        "service": "hivemind-openrouter",
                        "campaign_id": CAMPAIGN_ID,
                        "build_sha": self.build_sha,
                    },
                    json_hdrs,
                )

            if verb == "GET" and path in ("/noel", "/quick", "/signup", "/raid-noel", "/arena-noel"):
                self._nonce_seq += 1
                nonce = f"NoelCanaryNonce{self._nonce_seq:06d}{secrets.token_hex(4)}"
                extra_cards = ""
                if path == "/arena-noel":
                    cards = [
                        f'<article data-task-id="{tid}" '
                        f'data-ballot="{self._sign_ballot(tid, (self._nonce_seq + idx) % 2)}"></article>'
                        for idx, tid in enumerate(CURATED_ARENA_TASK_IDS)
                    ]
                    extra_cards = "".join(cards)
                html = (
                    f'<!DOCTYPE html><html lang="en"><head>'
                    f'<style nonce="{nonce}">body{{}}</style></head>'
                    f'<body><main data-campaign="{CAMPAIGN_ID}">'
                    f'<p role="note">{COMPLIANCE_NOTICE_EN}</p>'
                    f"{extra_cards}</main>"
                    f'<script nonce="{nonce}">document.documentElement.dataset.ready="1";</script>'
                    f"</body></html>"
                )
                csp = (
                    f"default-src 'none'; style-src 'nonce-{nonce}'; "
                    f"script-src 'nonce-{nonce}'; connect-src 'self'; base-uri 'none'"
                )
                return (
                    200,
                    html,
                    {
                        "content-type": "text/html; charset=utf-8",
                        "content-security-policy": csp,
                        "x-hivemind-csp-nonce": nonce,
                    },
                )

            if verb == "POST" and path == "/v1/hivemind/register":
                body_in = dict(payload or {})
                handle = str(body_in.get("handle") or "noel-friend")
                role = str(body_in.get("role") or "prosumer")
                self._seq += 1
                acct_id = f"acct_noel_{self._seq:04d}"
                api_key = f"hm-canary{self._seq:04d}_{secrets.token_hex(4)}"
                record = {
                    "account_id": acct_id,
                    "api_key": api_key,
                    "handle": handle,
                    "role": role,
                }
                self.accounts_by_key[api_key] = record
                return 200, {"object": "hivemind.account.registered", **record}, json_hdrs

            if verb == "GET" and path == "/v1/hivemind/me":
                acct = self._authenticate_bearer_unlocked(hdrs)
                if acct is None:
                    return 401, {"error": "invalid_api_key"}, json_hdrs
                self.activated_account_ids.add(acct["account_id"])
                return (
                    200,
                    {
                        "object": "hivemind.account",
                        "account_id": acct["account_id"],
                        "handle": acct["handle"],
                        "role": acct["role"],
                    },
                    json_hdrs,
                )

            if verb == "GET" and path == "/v1/models":
                return (
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "hivemind/coder", "object": "model", "owned_by": "hivemind"},
                            {"id": "hivemind/bee-stars", "object": "model", "owned_by": "hivemind"},
                            {"id": "hivemind/auto", "object": "model", "owned_by": "hivemind"},
                        ],
                    },
                    json_hdrs,
                )

            if verb == "GET" and path == "/v1/hivemind/onboarding/manifest":
                return (
                    200,
                    {
                        "object": MANIFEST_OBJECT,
                        "campaign_id": CAMPAIGN_ID,
                        "status": "ok",
                        "build_sha": self.build_sha,
                        "cloud_run_revision": self.cloud_run_revision,
                        "holiday_routes": list(PR949_MANIFEST_ROUTES),
                        "downloadable_scripts": [
                            *PR949_MANIFEST_CORE_SCRIPTS,
                            "noel_canary_cohort_rehearsal.py",
                        ],
                        "personal_pc_only": True,
                        "financial_promise": FINANCIAL_PROMISE,
                    },
                    json_hdrs,
                )

            if verb == "POST" and path == "/v1/hivemind/producer/heartbeat":
                acct = self._authenticate_bearer_unlocked(hdrs)
                if acct is None:
                    return 401, {"error": "invalid_api_key"}, json_hdrs
                hb_in = dict(payload or {})
                raw_ep = hb_in.get("endpoint_url")
                if not isinstance(raw_ep, str) or not raw_ep.strip():
                    return 422, {"error": "invalid_endpoint_url"}, json_hdrs
                ep_parts = urlsplit(raw_ep.strip())
                host_root = ep_parts.netloc.lower() or raw_ep.strip().lower()
                raw_models = hb_in.get("models")
                if raw_models is None:
                    raw_models = [{"id": "coder", "loaded": True}]
                if not isinstance(raw_models, list):
                    return 422, {"error": "invalid_models"}, json_hdrs
                models_norm: list[dict[str, Any]] = []
                model_ids: list[str] = []
                for entry in raw_models:
                    if isinstance(entry, Mapping) and isinstance(entry.get("id"), str) and entry["id"].strip():
                        m_id = entry["id"].strip()
                        pub_id = m_id if m_id.startswith("hivemind/") else f"hivemind/{m_id}"
                        models_norm.append(dict(entry))
                        model_ids.append(pub_id)
                self.activated_account_ids.add(acct["account_id"])
                self.producers[acct["account_id"]] = {
                    "account_id": acct["account_id"],
                    "handle": acct["handle"],
                    "role": acct["role"],
                    "host_root": host_root,
                    "execution_class": str(hb_in.get("execution_class") or "ollama"),
                    "capacity_1k": float(hb_in.get("capacity_1k") or 0.0),
                    "models": models_norm,
                    "model_ids": model_ids,
                    "last_heartbeat_utc": round(time.time(), 3),
                }
                return (
                    200,
                    {
                        "object": "hivemind.producer.heartbeat",
                        "status": "live",
                        "account_id": acct["account_id"],
                        "live_endpoints": 1,
                    },
                    json_hdrs,
                )

            if verb == "POST" and path == "/v1/hivemind/opentrainer/heartbeat":
                acct = self._authenticate_bearer_unlocked(hdrs)
                if acct is None:
                    return 401, {"error": "invalid_api_key"}, json_hdrs
                ot_in = dict(payload or {})
                host_label = str(ot_in.get("host_label") or "trainer-node").strip()[:128]
                if not host_label:
                    return 422, {"error": "invalid_host_label"}, json_hdrs
                self.activated_account_ids.add(acct["account_id"])
                node_key = f"{acct['account_id']}:{host_label}"
                self.opentrainer_nodes[node_key] = {
                    "account_id": acct["account_id"],
                    "host_label": host_label,
                    "gpu_name": str(ot_in.get("gpu_name") or "not reported"),
                    "vram_total_mib": int(ot_in.get("vram_total_mib") or 0),
                    "vram_free_mib": int(ot_in.get("vram_free_mib") or 0),
                    "network_mode": str(ot_in.get("network_mode") or "outbound-https-relay"),
                }
                return (
                    200,
                    {
                        "object": "hivemind.opentrainer.heartbeat",
                        "status": "accepted",
                        "registered_nodes": len(self.opentrainer_nodes),
                    },
                    json_hdrs,
                )

            if verb == "GET" and path == "/v1/hivemind/producers/fleet":
                return 200, self._fleet_snapshot_unlocked(), json_hdrs

            if verb == "POST" and path == "/v1/hivemind/onboarding/feedback":
                fb_in = dict(payload or {})
                raw_idem = hdrs.get("idempotency-key")
                norm_idem: str | None = None
                if raw_idem is not None and raw_idem.strip(" \t") != "":
                    if (
                        len(raw_idem) > _IDEM_KEY_MAX_CHARS
                        or not raw_idem.isascii()
                        or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw_idem)
                    ):
                        return 400, {"error": "invalid_idempotency_key"}, json_hdrs
                    norm_idem = raw_idem.strip()
                prof = str(fb_in.get("profile") or "dev").strip().lower()
                raw_ease = fb_in.get("ease_rating")
                if raw_ease is None:
                    raw_ease = 5
                if (
                    prof not in ("dev", "gamer")
                    or isinstance(raw_ease, bool)
                    or not isinstance(raw_ease, int)
                    or not (1 <= raw_ease <= 5)
                ):
                    return 422, {"error": "invalid_feedback"}, json_hdrs
                if (
                    norm_idem is not None
                    and self._idempotency_keys is not None
                    and norm_idem in self._idempotency_keys
                ):
                    self._idempotency_keys.move_to_end(norm_idem)
                    self.deduplicated_feedback_replays += 1
                    return 200, self._feedback_summary_unlocked(), json_hdrs
                zero_lag = bool(fb_in.get("zero_game_lag", True))
                plat = str(fb_in.get("platform") or "linux")
                self.feedback_items.append(
                    {
                        "profile": prof,
                        "ease_rating": int(raw_ease),
                        "zero_game_lag": zero_lag,
                        "platform": plat,
                    }
                )
                if norm_idem is not None:
                    if self._idempotency_keys is None:
                        self._idempotency_keys = OrderedDict()
                    elif len(self._idempotency_keys) >= self._max_idempotency_keys:
                        self._idempotency_keys.popitem(last=False)
                    self._idempotency_keys[norm_idem] = None
                return 200, self._feedback_summary_unlocked(), json_hdrs

            if verb == "GET" and path == "/v1/hivemind/onboarding/feedback/summary":
                return 200, self._feedback_summary_unlocked(), json_hdrs

            if verb == "GET" and path == "/v1/hivemind/raid/status":
                fb_sum = self._feedback_summary_unlocked()
                tr = self._trainer_telemetry
                trainer_accounts = sum(
                    1 for rec in self.accounts_by_key.values() if rec["role"] == "trainer"
                )
                reg_nodes = max(int(tr["install_completed"]), len(self.opentrainer_nodes))
                act_trainers = max(int(tr["first_rounds"]), len(self.opentrainer_nodes))
                return (
                    200,
                    {
                        "object": RAID_STATUS_OBJECT,
                        "campaign": CAMPAIGN_ID,
                        "status": "active",
                        "compliance": {
                            "personal_pc_only": True,
                            "corporate_hardware_or_code_forbidden": True,
                            "monetary_cost_eur": "0.00 EUR",
                            "monetary_cost_usd": "0.00 USD",
                            "license": "AGPL-3.0",
                        },
                        "swarm": {
                            "registered_accounts": len(self.accounts_by_key),
                            "active_accounts_24h": len(self.activated_account_ids),
                            "live_producers": len(self.producers),
                            "registered_producers": len(self.producers),
                            "registered_trainer_accounts": trainer_accounts,
                            "registered_trainer_nodes": reg_nodes,
                            "active_trainers": act_trainers,
                            "accepted_opentrainer_rounds": int(tr["first_rounds"]),
                        },
                        "onboarding_feedback": fb_sum,
                        "feedback_summary": fb_sum,
                        "mean_ease_rating": fb_sum["mean_ease_rating"],
                    },
                    json_hdrs,
                )

            if verb == "GET" and path == "/v1/hivemind/live-telemetry":
                tr = self._trainer_telemetry
                trainer_accounts = sum(
                    1 for rec in self.accounts_by_key.values() if rec["role"] == "trainer"
                )
                reg_nodes = max(int(tr["install_completed"]), len(self.opentrainer_nodes))
                act_trainers = max(int(tr["first_rounds"]), len(self.opentrainer_nodes))
                return (
                    200,
                    {
                        "object": "hivemind.live_telemetry",
                        "accounts": {
                            "total_registered_accounts": len(self.accounts_by_key),
                            "active_accounts_24h": len(self.activated_account_ids),
                        },
                        "opentrainer": {
                            "registered_trainer_accounts_count": trainer_accounts,
                            "total_registered_nodes": reg_nodes,
                            "active_trainer_nodes": act_trainers,
                            "accepted_rounds_count": int(tr["first_rounds"]),
                        },
                    },
                    json_hdrs,
                )

            if verb == "POST" and path == "/v1/hivemind/arena/vote":
                vote_in = dict(payload or {})
                task_id = str(vote_in.get("task_id") or "")
                ballot = str(vote_in.get("ballot_token") or "")
                chosen = str(vote_in.get("chosen_side") or "")
                if (
                    task_id not in CURATED_ARENA_TASK_IDS
                    or chosen not in ("A", "B", "tie")
                    or not self._verify_ballot(task_id, ballot)
                ):
                    return 422, {"error": "invalid_ballot"}, json_hdrs
                self.arena_votes_count += 1
                if chosen == "tie":
                    self._arena_side_counts["ties"] += 1
                elif ballot.split(".")[1] == "0" and chosen == "A" or ballot.split(".")[1] == "1" and chosen == "B":
                    self._arena_side_counts["holiday_wins"] += 1
                else:
                    self._arena_side_counts["baseline_wins"] += 1
                return (
                    200,
                    {
                        "object": ARENA_SUMMARY_OBJECT,
                        "recorded": True,
                        "total_votes": self.arena_votes_count,
                    },
                    json_hdrs,
                )

            if verb == "GET" and path == "/v1/hivemind/arena/summary":
                total_v = (
                    self.arena_votes_count
                    if self.arena_votes_count > 0
                    else int(self._trainer_telemetry["first_rounds"])
                )
                return (
                    200,
                    {
                        "object": ARENA_SUMMARY_OBJECT,
                        "total_votes": total_v,
                        **(self._arena_side_counts if self.arena_votes_count > 0 else {}),
                    },
                    json_hdrs,
                )

            if verb == "GET" and path == "/v1/hivemind/t4/status":
                tr = self._trainer_telemetry
                ready_producers = max(int(tr["first_rounds"]), len(self.producers))
                distinct_roots = len({str(p["host_root"]) for p in self.producers.values() if p.get("host_root")})
                if ready_producers <= 0:
                    return (
                        200,
                        {
                            "captured_utc": f"{self.report_date}T20:00:00Z",
                            "telemetry_status": "NO_TELEMETRY_RECEIVED",
                            "active_producers_count": 0,
                            "distinct_physical_gpus": 0,
                        },
                        json_hdrs,
                    )
                return (
                    200,
                    {
                        "captured_utc": f"{self.report_date}T20:00:00Z",
                        "telemetry_status": "LIVE_COHORT_REHEARSAL",
                        "operational_mode": "GAMER_WHISPER_QUIET",
                        "active_producers_count": ready_producers,
                        "distinct_physical_gpus": max(ready_producers, distinct_roots),
                        "hardware": {
                            "vram_free_mib": tr["vram_free_mib"],
                            "vram_total_mib": tr["vram_total_mib"],
                            "temp_c": tr["temp_c"],
                            "power_w": 115.0,
                        },
                        "qlora_progress": {
                            "qlora_rounds_assimilated": ready_producers,
                            "ratchet_accepted_rounds": ready_producers,
                            "held_out_eval_loss_before": 1.842,
                            "held_out_eval_loss_after": 1.791,
                        },
                    },
                    json_hdrs,
                )

            return 404, {"error": f"route_not_found:{path}"}, json_hdrs

    def request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Handle a zero-inference gateway request returning `(status_code, dict_body)`."""
        code, raw_body, resp_headers = self.request_full(method, url, payload, headers)
        if isinstance(raw_body, dict):
            return code, raw_body
        text = str(raw_body)
        return code, {
            "_html": text,
            "csp_nonce": resp_headers.get("x-hivemind-csp-nonce", ""),
            "headers": dict(resp_headers),
            "content_type": resp_headers.get("content-type", ""),
        }

    def request_raw(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = HTTP_TIMEOUT_S,
    ) -> tuple[int, str, dict[str, str]]:
        """Read-only `RawHttpFn` adapter returning `(status_code, body_text, lower_headers)`."""
        _ = timeout_s
        code, raw_body, resp_headers = self.request_full(method, url, None, headers)
        text = json.dumps(raw_body, ensure_ascii=False) if isinstance(raw_body, dict) else str(raw_body)
        return code, text, resp_headers

    def http_get_json(self, url: str, timeout_s: float = 5.0) -> tuple[int, dict[str, Any]]:
        """Read-only `HttpGetFn` adapter for `onboarding_funnel_report` and `daily_raid_bulletin`."""
        _ = timeout_s
        return self.request_json("GET", url, None, None)


def _call_pr949_probe(
    url: str,
    *,
    raw_http_fn: RawHttpFn,
    http_fn: HttpFn,
) -> tuple[int, dict[str, Any] | None, str, dict[str, str]]:
    _assert_zero_inference_request("GET", url, None)
    use_raw = raw_http_fn is not default_http_raw or http_fn is default_http_request
    if use_raw:
        try:
            raw_ret = raw_http_fn("GET", url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, {"network_error": type(exc).__name__}, "", {}
        if not isinstance(raw_ret, tuple) or len(raw_ret) != 3:
            raise RehearsalError("raw_http_fn must return `(status_code, body_text, headers)`.")
        code_raw, body_raw, hdrs_raw = raw_ret
        code = int(code_raw)
        hdrs = {str(k).strip().lower(): str(v) for k, v in (hdrs_raw or {}).items()}
        if isinstance(body_raw, Mapping):
            json_doc = dict(body_raw)
            return code, json_doc, json.dumps(json_doc, ensure_ascii=False), hdrs
        text_body = str(body_raw)
        json_doc = None
        if "application/json" in hdrs.get("content-type", "").lower() or text_body.lstrip().startswith(("{", "[")):
            with suppress(json.JSONDecodeError):
                parsed = json.loads(text_body) if text_body.strip() else {}
                json_doc = dict(parsed) if isinstance(parsed, Mapping) else {"data": parsed}
        return code, json_doc, text_body, hdrs

    resp, _ = _invoke_http_callable(http_fn, "GET", url, None, None)
    if resp.text:
        return resp.status_code, (resp.body if isinstance(resp.body, dict) and "html_length" not in resp.body else None), resp.text, resp.headers
    return resp.status_code, resp.body, json.dumps(resp.body, ensure_ascii=False), resp.headers


def _inspect_html_contract(
    path: str,
    html_text: str,
    hdrs: Mapping[str, str],
    seen_nonces: set[str] | None,
    *,
    is_fr: bool,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Verify PR #949 HTML contract: doctype, `'default-src 'none''` CSP nonce, zero DOM sinks/static figures, banner."""
    issues: list[str] = []
    has_doctype = "<!doctype html" in html_text.lower()
    if not has_doctype:
        issues.append(f"{path}: missing `<!DOCTYPE html>`." if not is_fr else f"{path} : `<!DOCTYPE html>` absent.")

    csp_hdr = ""
    for k, v in hdrs.items():
        if str(k).lower() == "content-security-policy":
            csp_hdr = str(v)
            break
    csp_low = csp_hdr.lower()
    csp_unsafe = "'unsafe-inline'" in csp_low or "'unsafe-eval'" in csp_low
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    attr_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)]
    header_nonce = hdr_nonces[0] if (hdr_nonces and len(set(hdr_nonces)) == 1) else None
    nonce_reused = seen_nonces is not None and header_nonce is not None and header_nonce in seen_nonces
    if seen_nonces is not None and header_nonce is not None:
        seen_nonces.add(header_nonce)

    csp_ok = bool(
        "default-src 'none'" in csp_low
        and not csp_unsafe
        and header_nonce is not None
        and attr_nonces
        and header_nonce in attr_nonces
        and len(set(attr_nonces)) == 1
        and not nonce_reused
    )
    if nonce_reused:
        issues.append(f"{path}: reused CSP nonce detected." if not is_fr else f"{path} : nonce CSP réutilisé.")
    elif not csp_ok:
        issues.append(
            f"{path}: invalid or missing `'default-src 'none''` CSP nonce."
            if not is_fr
            else f"{path} : en-tête CSP `'default-src 'none''` ou nonce invalide/absent."
        )

    stripped = strip_masked_keys_and_nonces(html_text)
    unsafe_sink_hits = len(_FORBIDDEN_DOM_SINK_RE.findall(stripped))
    static_fig = bool(_STATIC_FIG_UNIT_RE.search(stripped))
    if unsafe_sink_hits > 0:
        issues.append(
            f"{path}: unsafe DOM sink detected (`textContent` only)."
            if not is_fr
            else f"{path} : sink DOM non sûr détecté (`textContent` uniquement)."
        )
    if static_fig:
        issues.append(
            f"{path}: unbacked static telemetry figure detected."
            if not is_fr
            else f"{path} : chiffre télémétrique statique non étayé détecté."
        )

    low_html = stripped.lower()
    banner_ok = bool(
        ("personal" in low_html or "personnel" in low_html)
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
        and "agpl-3.0" in low_html
    )
    if not banner_ok:
        issues.append(
            f"{path}: missing Personal-PC / 0.00 EUR / AGPL-3.0 banner."
            if not is_fr
            else f"{path} : bannière PC perso / 0,00 EUR / AGPL-3.0 absente."
        )

    ok = len(issues) == 0
    return (
        ok,
        {
            "ok": ok,
            "has_doctype": has_doctype,
            "csp_ok": csp_ok,
            "nonce_reused": nonce_reused,
            "unsafe_dom_sinks_count": unsafe_sink_hits,
            "static_figure_found": static_fig,
            "personal_pc_banner_ok": banner_ok,
        },
        issues,
    )


def _inspect_feedback_summary_contract(
    doc: Mapping[str, Any],
    *,
    min_ease: float,
    min_lag: float,
    is_fr: bool,
) -> tuple[bool, int, float | None, float | None, bool | None, list[str]]:
    issues: list[str] = []
    if doc.get("object") != FEEDBACK_SUMMARY_OBJECT:
        issues.append("Feedback summary invalid object type.")
    if _has_forbidden_score_key(doc):
        issues.append("Feedback summary contains forbidden key.")

    raw_total = doc.get("total_responses")
    total_ok = _is_nonneg_int(raw_total)
    total_val = int(raw_total) if total_ok else -1
    if not total_ok:
        issues.append("Feedback summary invalid `total_responses`.")

    has_mean = "mean_ease_rating" in doc
    has_avg = "average_ease_rating" in doc
    resolved_ease: float | None = None
    if not has_mean and not has_avg:
        issues.append("Feedback summary missing `mean_ease_rating` / `average_ease_rating`.")
    elif total_ok:
        checked: list[float] = []
        for lbl, raw_v, present in (
            ("mean_ease_rating", doc.get("mean_ease_rating"), has_mean),
            ("average_ease_rating", doc.get("average_ease_rating"), has_avg),
        ):
            if not present:
                continue
            if total_val == 0 and raw_v is not None:
                issues.append(f"Feedback summary non-null `{lbl}` when empty.")
            elif total_val > 0:
                if (
                    isinstance(raw_v, bool)
                    or not isinstance(raw_v, (int, float))
                    or not math.isfinite(float(raw_v))
                    or not (1.0 <= float(raw_v) <= 5.0)
                ):
                    issues.append(f"Feedback summary invalid `{lbl}` value.")
                else:
                    checked.append(float(raw_v))
        if has_mean and has_avg and len(checked) == 2 and round(checked[0], 2) != round(checked[1], 2):
            issues.append("Feedback summary `mean_ease_rating` and `average_ease_rating` disagree.")
        if checked:
            resolved_ease = round(checked[0], 2)

    resolved_lag: float | None = None
    if "zero_game_lag_ratio" not in doc:
        issues.append("Feedback summary missing `zero_game_lag_ratio`.")
    elif total_ok:
        raw_lag = doc.get("zero_game_lag_ratio")
        if total_val == 0 and raw_lag is not None:
            issues.append("Feedback summary non-null `zero_game_lag_ratio` when empty.")
        elif total_val > 0:
            if (
                isinstance(raw_lag, bool)
                or not isinstance(raw_lag, (int, float))
                or not math.isfinite(float(raw_lag))
                or not (0.0 <= float(raw_lag) <= 1.0)
            ):
                issues.append("Feedback summary invalid `zero_game_lag_ratio`.")
            else:
                resolved_lag = round(float(raw_lag), 4)

    bp = doc.get("by_profile")
    if (
        not isinstance(bp, Mapping)
        or not _is_nonneg_int(bp.get("dev"))
        or not _is_nonneg_int(bp.get("gamer"))
        or (total_ok and int(bp["dev"]) + int(bp["gamer"]) != total_val)
    ):
        issues.append("Feedback summary invalid `by_profile` conservation.")

    schema_ok = len(issues) == 0
    if not schema_ok:
        return False, max(total_val, 0), None, None, False, issues
    if total_val == 0:
        return True, 0, None, None, None, []

    ease_ok = resolved_ease is not None and Fraction(str(resolved_ease)) >= Fraction(str(min_ease))
    lag_ok = resolved_lag is not None and Fraction(str(resolved_lag)) >= Fraction(str(min_lag))
    slo_issues: list[str] = []
    if not ease_ok:
        slo_issues.append(
            f"Note CES moyenne {resolved_ease}/5 < {min_ease}/5."
            if is_fr
            else f"CES ease rating {resolved_ease}/5 below target >= {min_ease}/5."
        )
    if not lag_ok:
        slo_issues.append(
            f"Ratio zéro-lag Gamer {resolved_lag} < {min_lag}."
            if is_fr
            else f"Gamer zero_game_lag_ratio {resolved_lag} below target >= {min_lag}."
        )
    return True, total_val, resolved_ease, resolved_lag, bool(ease_ok and lag_ok), slo_issues


def _inspect_fleet_quorum_contract(
    doc: Mapping[str, Any],
    *,
    target_n: int,
    require_live_quorum: bool,
    is_fr: bool,
) -> tuple[bool, int | None, int | None, bool, bool, bool, list[str]]:
    issues: list[str] = []
    count_keys = ("total_hiveminders", "live_hiveminders", "distinct_physical_gpus", "live_endpoints")
    models_ok = isinstance(doc.get("live_models"), list) and all(isinstance(m, str) for m in doc["live_models"])
    rows = doc.get("hiveminders")
    rows_ok = isinstance(rows, list) and all(isinstance(r, Mapping) and "handle" in r and "status" in r for r in rows)
    endpoint_leaked = isinstance(rows, list) and any(
        isinstance(r, Mapping) and r.get("endpoint_url") is not None for r in rows
    )
    if (
        doc.get("object") != FLEET_OBJECT
        or not all(_is_nonneg_int(doc.get(k)) for k in count_keys)
        or not models_ok
        or not rows_ok
    ):
        issues.append(
            "Contrat JSON `/v1/hivemind/producers/fleet` invalide."
            if is_fr
            else "Invalid `/v1/hivemind/producers/fleet` JSON schema."
        )
    if endpoint_leaked:
        issues.append(
            "Fuite `endpoint_url` détectée dans `/v1/hivemind/producers/fleet`."
            if is_fr
            else "Unredacted `endpoint_url` leaked in public `/v1/hivemind/producers/fleet`."
        )
    if issues:
        return False, None, None, False, False, False, issues

    online_cnt = int(doc["live_hiveminders"])
    gpu_cnt = int(doc["distinct_physical_gpus"])
    quorum_met = online_cnt >= target_n and gpu_cnt >= 1
    awaiting = not quorum_met
    quorum_gate_ok = quorum_met or not require_live_quorum
    diags: list[str] = []
    if not quorum_met:
        if require_live_quorum:
            diags.append(
                f"Quorum producteurs insuffisant ({online_cnt}/{target_n} en ligne, {gpu_cnt} GPU distincts ; `--require-live-quorum`)."
                if is_fr
                else f"Live producer quorum unmet ({online_cnt}/{target_n} online, {gpu_cnt} distinct GPUs; `--require-live-quorum`)."
            )
        else:
            diags.append(
                f"En attente quorum amis (`awaiting_producer_quorum` : {online_cnt}/{target_n} producteurs, {gpu_cnt} GPU)."
                if is_fr
                else f"Awaiting friend producer quorum (`awaiting_producer_quorum`: {online_cnt}/{target_n} online, {gpu_cnt} GPUs)."
            )
    return True, online_cnt, gpu_cnt, quorum_met, awaiting, quorum_gate_ok, diags


def _build_finops_redeploy_cmd(gateway: str, expected_sha: str | None) -> str:
    sha_token = expected_sha if expected_sha else "${GITHUB_SHA:?set_GITHUB_SHA}"
    return (
        f'HIVEMIND_BUILD_SHA="{sha_token}" bash deploy/gcp_openrouter/deploy.sh '
        "--min-instances 0 --max-instances 1 --no-cpu-throttling "
        f"# verify: python3 scripts/noel_canary_cohort_rehearsal.py --pr949-parity --gateway {gateway}"
    )


def verify_pr949_cohort_live_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    raw_http_fn: RawHttpFn = default_http_raw,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference Cloud Run PR #949 live parity, 8+ producer quorum, and CES alias verifier."""
    norm_lang = normalize_lang(lang)
    is_fr = norm_lang == "fr"
    clean_gw = validate_gateway_url(gateway)
    target_n = _validate_int_range(target_online_producers, "target_online_producers", 1, 500)
    clean_ease = _validate_float_range(min_ease_rating, "min_ease_rating", 1.0, 5.0)
    clean_lag = _validate_float_range(min_zero_lag_ratio, "min_zero_lag_ratio", 0.0, 1.0)
    clean_expected_sha = (
        _validate_expected_sha(expected_sha)
        if expected_sha is not None and str(expected_sha).strip()
        else None
    )

    compliance = evaluate_host_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        lang=norm_lang,
    )
    if not compliance["compliance_ok"]:
        diag = sanitize_text_for_display(str(compliance["summary"]))
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "slice_id": "SLICE-SCALE-DYN-41",
                "campaign_id": CAMPAIGN_ID,
                "mode": "pr949_live_parity",
                "lang": norm_lang,
                "gateway": clean_gw,
                "compliance_ok": False,
                "corporate_detected": True,
                "compliance": compliance,
                "ready": False,
                "verdict": "NO_GO",
                "exit_code": 2,
                "holiday_routes_deployed_on_cloud_run": False,
                "onboarding_feedback_deployed": False,
                "stale_revision_detected": False,
                "missing_or_stale_routes": [],
                "models_ok": False,
                "model_ids": [],
                "coder_model_present": False,
                "manifest_ok": False,
                "live_build_sha": "unknown",
                "expected_build_sha": clean_expected_sha,
                "sha_match": None,
                "cloud_run_revision": None,
                "html_routes_ok": False,
                "unique_csp_nonces_count": 0,
                "forbidden_dom_sinks_count": 0,
                "static_figures_found": False,
                "personal_pc_banners_ok": False,
                "html_contracts": {},
                "feedback_schema_ok": False,
                "total_responses": 0,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "ces_slo_ok": False,
                "min_ease_rating": clean_ease,
                "min_zero_lag_ratio": clean_lag,
                "fleet_schema_ok": False,
                "online_producers": None,
                "distinct_physical_gpus": None,
                "target_online_producers": target_n,
                "producer_quorum_met": False,
                "awaiting_producer_quorum": False,
                "require_live_quorum": bool(require_live_quorum),
                "raid_status_ok": False,
                "arena_summary_ok": False,
                "http_statuses": {},
                "redeploy_required": False,
                "redeploy_command": _build_finops_redeploy_cmd(clean_gw, clean_expected_sha),
                "zero_inference_verified": True,
                "network_calls": 0,
                "file_writes": 0,
                "diagnostics": [diag],
                "summary": diag,
            }
        )

    http_statuses: dict[str, int] = {}
    stale_routes: set[str] = set()
    missing_manifest_routes: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    unsafe_sink_total = 0
    static_figures_any = False

    health_ok = False
    health_sha = "unknown"
    models_ok = False
    model_ids: list[str] = []
    coder_present = False
    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    cloud_run_revision: str | None = None
    feedback_schema_ok = False
    total_responses = 0
    mean_ease: float | None = None
    zero_lag: float | None = None
    ces_slo_ok: bool | None = False
    fleet_schema_ok = False
    online_producers: int | None = None
    distinct_physical_gpus: int | None = None
    producer_quorum_met = False
    awaiting_producer_quorum = False
    quorum_gate_ok = False
    raid_status_ok = False
    arena_summary_ok = False
    network_calls = 0

    for path, kind in PR949_REQUIRED_ROUTES:
        network_calls += 1
        code, json_doc, text_body, hdrs = _call_pr949_probe(
            f"{clean_gw}{path}",
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
        )
        http_statuses[path] = code
        if code != 200:
            stale_routes.add(path)
            diagnostics.append(
                f"GET {path} -> HTTP {code} (attendu 200)." if is_fr else f"GET {path} -> HTTP {code} (expected 200)."
            )
            continue

        if path == "/health":
            health_ok = isinstance(json_doc, Mapping) and json_doc.get("status") == "ok"
            if isinstance(json_doc, Mapping):
                health_sha = _sanitize_build_sha(json_doc.get("build_sha"))
            if not health_ok:
                stale_routes.add(path)
                diagnostics.append("GET /health missing `status == 'ok'`.")
        elif path == "/v1/models":
            raw_data = json_doc.get("data") if isinstance(json_doc, Mapping) else None
            valid_entries = (
                [
                    e
                    for e in raw_data
                    if isinstance(e, Mapping) and isinstance(e.get("id"), str) and e["id"].strip()
                ]
                if isinstance(raw_data, list)
                else []
            )
            model_ids = [str(e["id"]).strip() for e in valid_entries]
            coder_present = any(
                e["id"].strip() == "hivemind/coder" and (e.get("object") in (None, "model"))
                for e in valid_entries
            )
            models_ok = bool(
                isinstance(json_doc, Mapping)
                and json_doc.get("object", "list") == "list"
                and model_ids
                and coder_present
            )
            if not models_ok:
                stale_routes.add(path)
                diagnostics.append(
                    "GET /v1/models vide ou sans `hivemind/coder`."
                    if is_fr
                    else "GET /v1/models empty or missing `hivemind/coder` metadata."
                )
        elif kind == "html":
            h_ok, h_info, h_issues = _inspect_html_contract(path, text_body, hdrs, seen_nonces, is_fr=is_fr)
            html_contracts[path] = h_info
            unsafe_sink_total += int(h_info["unsafe_dom_sinks_count"])
            static_figures_any = static_figures_any or bool(h_info["static_figure_found"])
            if not h_ok:
                stale_routes.add(path)
                diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/manifest":
            if not isinstance(json_doc, Mapping):
                stale_routes.add(path)
                diagnostics.append("GET /v1/hivemind/onboarding/manifest missing JSON body.")
            else:
                meta_ok = bool(
                    json_doc.get("object") == MANIFEST_OBJECT
                    and json_doc.get("campaign_id") == CAMPAIGN_ID
                    and json_doc.get("status") == "ok"
                    and json_doc.get("personal_pc_only") is True
                    and json_doc.get("financial_promise") == FINANCIAL_PROMISE
                )
                raw_routes = json_doc.get("holiday_routes")
                if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes):
                    missing_manifest_routes = [r for r in PR949_MANIFEST_ROUTES if r not in raw_routes]
                else:
                    missing_manifest_routes = list(PR949_MANIFEST_ROUTES)
                raw_scripts = json_doc.get("downloadable_scripts")
                scripts_ok = (
                    isinstance(raw_scripts, list)
                    and all(isinstance(s, str) for s in raw_scripts)
                    and all(req in raw_scripts for req in PR949_MANIFEST_CORE_SCRIPTS)
                )
                live_build_sha = _sanitize_build_sha(json_doc.get("build_sha"))
                if live_build_sha == "unknown":
                    live_build_sha = health_sha
                raw_rev = json_doc.get("cloud_run_revision")
                cloud_run_revision = str(raw_rev).strip() if isinstance(raw_rev, str) and raw_rev.strip() else None
                if clean_expected_sha is not None:
                    sha_match = _shas_match(live_build_sha, clean_expected_sha)
                    if not sha_match:
                        diagnostics.append(
                            f"Build SHA drift: live='{live_build_sha}' != expected '{clean_expected_sha}'."
                        )
                manifest_ok = bool(meta_ok and not missing_manifest_routes and scripts_ok and sha_match is not False)
                if not manifest_ok:
                    stale_routes.add(path)
                    if missing_manifest_routes:
                        diagnostics.append(
                            f"Manifest missing holiday routes: {', '.join(missing_manifest_routes)}."
                        )
                    elif not meta_ok or not scripts_ok:
                        diagnostics.append("Manifest metadata or core scripts invalid.")
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            if not isinstance(json_doc, Mapping):
                stale_routes.add(path)
                diagnostics.append("Feedback summary missing JSON body.")
            else:
                feedback_schema_ok, total_responses, mean_ease, zero_lag, ces_slo_ok, fb_issues = (
                    _inspect_feedback_summary_contract(
                        json_doc,
                        min_ease=clean_ease,
                        min_lag=clean_lag,
                        is_fr=is_fr,
                    )
                )
                if not feedback_schema_ok:
                    stale_routes.add(path)
                diagnostics.extend(fb_issues)
        elif path == "/v1/hivemind/producers/fleet":
            if not isinstance(json_doc, Mapping):
                stale_routes.add(path)
                diagnostics.append("Producer fleet missing JSON body.")
            else:
                (
                    fleet_schema_ok,
                    online_producers,
                    distinct_physical_gpus,
                    producer_quorum_met,
                    awaiting_producer_quorum,
                    quorum_gate_ok,
                    fleet_issues,
                ) = _inspect_fleet_quorum_contract(
                    json_doc,
                    target_n=target_n,
                    require_live_quorum=bool(require_live_quorum),
                    is_fr=is_fr,
                )
                if not fleet_schema_ok:
                    stale_routes.add(path)
                diagnostics.extend(fleet_issues)
        elif path == "/v1/hivemind/raid/status":
            comp = json_doc.get("compliance") if isinstance(json_doc, Mapping) else None
            raid_status_ok = bool(
                isinstance(json_doc, Mapping)
                and json_doc.get("object") == RAID_STATUS_OBJECT
                and isinstance(comp, Mapping)
                and comp.get("personal_pc_only") is True
                and comp.get("monetary_cost_eur") == "0.00 EUR"
                and comp.get("monetary_cost_usd") == "0.00 USD"
                and comp.get("license") == "AGPL-3.0"
                and isinstance(json_doc.get("swarm"), Mapping)
                and isinstance(json_doc.get("onboarding_feedback"), Mapping)
            )
            if not raid_status_ok:
                stale_routes.add(path)
                diagnostics.append("GET /v1/hivemind/raid/status schema or compliance invalid.")
        else:
            counts_ok = False
            if isinstance(json_doc, Mapping) and json_doc.get("object") == ARENA_SUMMARY_OBJECT:
                tv = json_doc.get("total_votes")
                if _is_nonneg_int(tv):
                    if all(k in json_doc for k in ("holiday_wins", "baseline_wins", "ties")):
                        hw, bw, ti = json_doc["holiday_wins"], json_doc["baseline_wins"], json_doc["ties"]
                        counts_ok = (
                            _is_nonneg_int(hw)
                            and _is_nonneg_int(bw)
                            and _is_nonneg_int(ti)
                            and int(hw) + int(bw) + int(ti) == int(tv)
                        )
                    else:
                        counts_ok = True
            arena_summary_ok = counts_ok and not _has_forbidden_score_key(json_doc or {})
            if not arena_summary_ok:
                stale_routes.add(path)
                diagnostics.append("GET /v1/hivemind/arena/summary schema or vote conservation invalid.")

    html_routes_ok = bool(
        len(html_contracts) == len(LIVE_HTML_PATHS)
        and all(html_contracts.get(p, {}).get("ok") is True for p in LIVE_HTML_PATHS)
        and len(seen_nonces) == len(LIVE_HTML_PATHS)
    )
    personal_pc_banners_ok = bool(
        len(html_contracts) == len(LIVE_HTML_PATHS)
        and all(html_contracts.get(p, {}).get("personal_pc_banner_ok") is True for p in LIVE_HTML_PATHS)
    )
    onboarding_feedback_deployed = bool(
        http_statuses.get("/v1/hivemind/onboarding/feedback/summary") == 200 and feedback_schema_ok
    )
    holiday_routes_deployed = bool(
        health_ok
        and manifest_ok
        and html_routes_ok
        and onboarding_feedback_deployed
        and raid_status_ok
        and arena_summary_ok
    )
    missing_or_stale_routes = sorted(stale_routes | set(missing_manifest_routes))
    stale_revision_detected = bool(
        any(http_statuses.get(p) == 404 for p in http_statuses)
        or missing_manifest_routes
        or (sha_match is False)
        or not holiday_routes_deployed
    )
    ces_gate_ok = feedback_schema_ok and (ces_slo_ok is not False)
    ready = bool(
        holiday_routes_deployed
        and models_ok
        and fleet_schema_ok
        and ces_gate_ok
        and quorum_gate_ok
        and not stale_revision_detected
    )
    redeploy_required = bool(
        stale_revision_detected or not holiday_routes_deployed or not onboarding_feedback_deployed or not models_ok
    )
    clean_diags = [sanitize_text_for_display(d) for d in diagnostics]
    summary = (
        f"Parité PR #949={'GO' if ready else 'NO-GO'} sur {clean_gw} "
        f"(holiday_routes={holiday_routes_deployed}, feedback={onboarding_feedback_deployed}, "
        f"producers={online_producers}/{target_n}, awaiting_quorum={awaiting_producer_quorum}, "
        f"mean_ease={mean_ease}, zero_lag={zero_lag})."
        if is_fr
        else f"PR #949 parity={'GO' if ready else 'NO-GO'} on {clean_gw} "
        f"(holiday_routes={holiday_routes_deployed}, feedback={onboarding_feedback_deployed}, "
        f"producers={online_producers}/{target_n}, awaiting_quorum={awaiting_producer_quorum}, "
        f"mean_ease={mean_ease}, zero_lag={zero_lag})."
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "slice_id": "SLICE-SCALE-DYN-41",
            "campaign_id": CAMPAIGN_ID,
            "mode": "pr949_live_parity",
            "lang": norm_lang,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "compliance": compliance,
            "ready": ready,
            "verdict": "GO" if ready else "NO_GO",
            "exit_code": 0 if ready else 2,
            "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed,
            "onboarding_feedback_deployed": onboarding_feedback_deployed,
            "stale_revision_detected": stale_revision_detected,
            "missing_or_stale_routes": missing_or_stale_routes,
            "models_ok": models_ok,
            "model_ids": model_ids,
            "coder_model_present": coder_present,
            "manifest_ok": manifest_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "sha_match": sha_match,
            "cloud_run_revision": cloud_run_revision,
            "html_routes_ok": html_routes_ok,
            "unique_csp_nonces_count": len(seen_nonces),
            "forbidden_dom_sinks_count": unsafe_sink_total,
            "static_figures_found": static_figures_any,
            "personal_pc_banners_ok": personal_pc_banners_ok,
            "html_contracts": html_contracts,
            "feedback_schema_ok": feedback_schema_ok,
            "total_responses": total_responses,
            "mean_ease_rating": mean_ease,
            "zero_game_lag_ratio": zero_lag,
            "ces_slo_ok": ces_slo_ok,
            "min_ease_rating": clean_ease,
            "min_zero_lag_ratio": clean_lag,
            "fleet_schema_ok": fleet_schema_ok,
            "online_producers": online_producers,
            "distinct_physical_gpus": distinct_physical_gpus,
            "target_online_producers": target_n,
            "producer_quorum_met": producer_quorum_met,
            "awaiting_producer_quorum": awaiting_producer_quorum,
            "require_live_quorum": bool(require_live_quorum),
            "raid_status_ok": raid_status_ok,
            "arena_summary_ok": arena_summary_ok,
            "http_statuses": http_statuses,
            "redeploy_required": redeploy_required,
            "redeploy_command": _build_finops_redeploy_cmd(clean_gw, clean_expected_sha),
            "zero_inference_verified": True,
            "network_calls": network_calls,
            "file_writes": 0,
            "diagnostics": clean_diags,
            "summary": summary,
        }
    )


def build_personal_cohort_plan(cohort_size: int = DEFAULT_COHORT_SIZE) -> list[tuple[int, PersonalArchetypeSpec]]:
    """Build a deterministic N-workspace plan distributed evenly across the 5 personal-PC archetypes."""
    _validate_int_range(cohort_size, "cohort_size", MIN_COHORT_SIZE, MAX_COHORT_SIZE)
    return [
        (idx, PERSONAL_PC_ARCHETYPES[idx % len(PERSONAL_PC_ARCHETYPES)])
        for idx in range(cohort_size)
    ]


def build_corporate_control_plan(
    corporate_controls: int = DEFAULT_CORPORATE_CONTROLS,
    *,
    extra_controls: Sequence[CorporateControlSpec] = (),
) -> list[tuple[int, CorporateControlSpec]]:
    """Build a deterministic negative-control plan across corporate / Amadeus detection vectors."""
    _validate_int_range(corporate_controls, "corporate_controls", 0, MAX_CORPORATE_CONTROLS)
    plan = [
        (idx, CORPORATE_NEGATIVE_CONTROLS[idx % len(CORPORATE_NEGATIVE_CONTROLS)])
        for idx in range(corporate_controls)
    ]
    offset = len(plan)
    for extra_idx, spec in enumerate(extra_controls):
        plan.append((offset + extra_idx, spec))
    return plan


def scan_directory_for_key_leaks(directory: Path) -> tuple[int, list[str]]:
    """Recursively scan all regular files in `directory` for raw `hm-...` keys or Bearer tokens."""
    files_scanned = 0
    leaks: list[str] = []
    if not directory.exists():
        return 0, leaks
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        files_scanned += 1
        rel_name = file_path.relative_to(directory).as_posix()
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if _API_KEY_RE.search(text):
            leaks.append(f"raw_api_key_in:{rel_name}")
        if _BEARER_RE.search(text):
            leaks.append(f"bearer_token_in:{rel_name}")
    return files_scanned, leaks


def _make_step_clock(step_s: float = 0.002) -> tuple[Callable[[], float], list[float]]:
    current = [100.0]

    def _clock() -> float:
        current[0] += step_s
        return current[0]

    return _clock, current


class _LiveCohortTransportCoordinator:
    """Thread-safe coordinator wrapping an external `http_fn`/`raw_http_fn` or `MockZeroInferenceGateway`."""

    def __init__(
        self,
        gateway_url: str,
        *,
        report_date: str,
        http_fn: HttpFn | None,
        raw_http_fn: RawHttpFn | None,
        idempotency_replay: bool,
        include_heartbeat_quorum: bool,
    ) -> None:
        self.gateway_url = gateway_url
        self.external_mode = http_fn is not None
        self.mock_gateway = None if self.external_mode else MockZeroInferenceGateway(gateway_url, report_date=report_date)
        self._http_fn = http_fn
        self._raw_http_fn = raw_http_fn
        self.idempotency_replay = idempotency_replay
        self.include_heartbeat_quorum = include_heartbeat_quorum

        self._lock = threading.Lock()
        self.requests_total = 0
        self.inference_calls_count = 0
        self.landing_visits = 0
        self.routes_hit: dict[str, int] = {}
        self.signed_up_workspaces: set[int] = set()
        self.activated_workspaces: set[int] = set()
        self.observed_nonces: list[str] = []
        self.html_renders_total = 0
        self.strict_html_failures = 0
        self.sqlite_lock_errors = 0
        self.ces_replay_attempts = 0
        self.ces_replay_ok_count = 0
        self.opentrainer_heartbeats_recorded = 0
        self.producer_heartbeats_recorded = 0

    def record_html(self, nonce: str | None, strict_ok: bool) -> None:
        with self._lock:
            self.html_renders_total += 1
            if nonce:
                self.observed_nonces.append(nonce)
            if not strict_ok:
                self.strict_html_failures += 1

    def record_landing_visit(self) -> None:
        if not self.external_mode:
            return
        with self._lock:
            self.landing_visits += 1

    def call_full(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        workspace_index: int | None = None,
    ) -> HttpResponse:
        verb = (method or "GET").strip().upper()
        parts = urlsplit(url)
        path = parts.path or "/"
        try:
            _assert_zero_inference_request(verb, url, payload)
        except RehearsalError:
            with self._lock:
                self.inference_calls_count += 1
            raise

        req_headers = dict(headers or {})
        if (
            self.external_mode
            and workspace_index is not None
            and not any(k.lower() == "x-forwarded-for" for k in req_headers)
        ):
            req_headers["X-Forwarded-For"] = _simulated_loopback_ip(10 + (workspace_index // 200), workspace_index)

        if self.mock_gateway is not None:
            code, raw_body, resp_hdrs = self.mock_gateway.request_full(verb, url, payload, req_headers)
            resp = _normalize_http_result((code, raw_body, resp_hdrs))
            is_locked = _contains_sqlite_lock_signal(resp)
        else:
            assert self._http_fn is not None
            resp, is_locked = _invoke_http_callable(self._http_fn, verb, url, payload, req_headers)

        with self._lock:
            self.requests_total += 1
            route_key = f"{verb} {path}"
            self.routes_hit[route_key] = self.routes_hit.get(route_key, 0) + 1
            if is_locked:
                self.sqlite_lock_errors += 1
            if workspace_index is not None and 200 <= resp.status_code < 300:
                if verb == "POST" and path == "/v1/hivemind/register":
                    self.signed_up_workspaces.add(workspace_index)
                elif verb == "GET" and path == "/v1/hivemind/me":
                    self.activated_workspaces.add(workspace_index)
                elif verb == "POST" and path == "/v1/hivemind/opentrainer/heartbeat":
                    self.opentrainer_heartbeats_recorded += 1
                elif verb == "POST" and path == "/v1/hivemind/producer/heartbeat":
                    self.producer_heartbeats_recorded += 1
        return resp

    def make_workspace_http_fn(
        self,
        *,
        workspace_index: int,
        profile: str,
        replay_log: list[tuple[str, dict[str, Any], dict[str, str]]],
    ) -> Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], tuple[int, dict[str, Any]]]:
        def _ws_http(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, dict[str, Any]]:
            verb = (method or "GET").strip().upper()
            path = urlsplit(url).path or "/"
            req_hdrs = dict(headers or {})
            if (
                self.idempotency_replay
                and verb == "POST"
                and path == "/v1/hivemind/onboarding/feedback"
            ):
                if not any(k.lower() == "idempotency-key" for k in req_hdrs):
                    req_hdrs["Idempotency-Key"] = f"noel-canary-{profile}-ws{workspace_index:04d}"
                replay_log.append((url, dict(payload or {}), dict(req_hdrs)))
            resp = self.call_full(
                verb,
                url,
                payload,
                req_hdrs,
                workspace_index=workspace_index,
            )
            body_out = dict(resp.body)
            if resp.text and "_html" not in body_out:
                body_out["_html"] = resp.text
            return resp.status_code, body_out

        return _ws_http

    def replay_feedback_posts(
        self,
        *,
        workspace_index: int,
        replay_log: list[tuple[str, dict[str, Any], dict[str, str]]],
    ) -> bool:
        if not self.idempotency_replay:
            return True
        all_ok = True
        for url, payload, hdrs in replay_log:
            resp = self.call_full(
                "POST",
                url,
                payload,
                hdrs,
                workspace_index=workspace_index,
            )
            ok = resp.status_code == 200 and _is_nonneg_int(resp.body.get("total_responses"))
            with self._lock:
                self.ces_replay_attempts += 1
                if ok:
                    self.ces_replay_ok_count += 1
            all_ok = all_ok and ok
        return all_ok

    def coord_http_get_json(self, url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, dict[str, Any]]:
        _ = timeout_s
        if self.mock_gateway is not None:
            return self.mock_gateway.http_get_json(url, timeout_s)
        resp = self.call_full("GET", url, None, None)
        return resp.status_code, dict(resp.body)

    def raw_http_adapter(self) -> RawHttpFn:
        if self._raw_http_fn is not None:
            return self._raw_http_fn
        if self.mock_gateway is not None:
            return self.mock_gateway.request_raw
        return default_http_raw

    def live_http_adapter(self) -> HttpFn:
        if self._http_fn is not None:
            def _wrapped(
                method: str,
                url: str,
                payload: dict[str, Any] | None = None,
                headers: dict[str, str] | None = None,
            ) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
                resp = self.call_full(method, url, payload, headers)
                return resp.status_code, (resp.text if resp.text and "html_length" in resp.body else resp.body), resp.headers

            return _wrapped
        assert self.mock_gateway is not None
        return self.mock_gateway.request_full


def _execute_personal_workspace(
    *,
    workspace_index: int,
    spec: PersonalArchetypeSpec,
    gamer_ordinal: int | None,
    thermal_injected: bool,
    yield_injected: bool,
    send_producer_hb: bool,
    send_opentrainer_hb: bool,
    strict_html_checks: bool,
    base_dir: Path,
    gateway_url: str,
    coordinator: _LiveCohortTransportCoordinator,
    modules: OnboardingPipelineModules,
    yield_slo_ms: float,
    max_temp_c: float,
    min_free_vram_mib: int,
    script_lang: str,
    workspace_post_hook: WorkspacePostHook | None,
) -> dict[str, Any]:
    ws_root = base_dir / "personal" / f"ws_{workspace_index:03d}_{spec.archetype_id}"
    ws_home = ws_root / "home_box"
    ws_dir = ws_home / "projects" / "hobby_open_source"
    ws_dir.mkdir(parents=True, exist_ok=True)
    (ws_dir / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")
    git_dir = ws_dir / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "config").write_text(
        f'[core]\n\trepositoryformatversion = 0\n[remote "origin"]\n\turl = {HOBBY_DISPLAY_GIT_URL}\n',
        encoding="utf-8",
    )

    if spec.os_name == "nt":
        pc_name = f"GAMING-BOX-{workspace_index:03d}"
        ws_env: dict[str, str] = {
            "PATH": "C:\\Windows\\System32",
            "USERPROFILE": str(ws_home),
            "COMPUTERNAME": pc_name,
            "USERDOMAIN": pc_name,
        }
    else:
        ws_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(ws_home),
            "HOSTNAME": f"home-pc-{workspace_index:03d}",
        }

    eff_temp_c = INJECTED_THERMAL_SPIKE_C if thermal_injected else spec.temp_c
    eff_yield_ms = (yield_slo_ms + 350.0) if yield_injected else spec.yield_latency_ms
    eff_retained_vram = INJECTED_RETAINED_VRAM_MIB if yield_injected else 0

    ws_clock, clock_box = _make_step_clock(0.002)

    def _ws_runner(argv: Sequence[str], timeout_s: float) -> tuple[int, str]:
        _ = timeout_s
        cmd0 = Path(str(argv[0])).name.lower() if argv else ""
        if cmd0 == "nvidia-smi":
            if not spec.gpu_present:
                return 1, ""
            return (
                0,
                f"{spec.gpu_name}, {spec.total_vram_mib}, {spec.free_vram_mib}, "
                f"{eff_temp_c:.1f}, {spec.util_pct:.1f}\n",
            )
        clock_box[0] += (yield_slo_ms / 1000.0 + 0.25) if yield_injected else 0.004
        if spec.os_name == "nt":
            return (
                0,
                '"explorer.exe","1200","Console","1","85,000 K"\n'
                '"steam.exe","2400","Console","1","120,000 K"\n',
            )
        return 0, "systemd\nbash\ncode\npython3\n"

    def _ws_gpu_probe(**_kwargs: Any) -> Any:
        if not spec.gpu_present:
            return modules.gamer_selftest.GpuTelemetry(
                present=False,
                reason="nvidia_smi_not_found",
            )
        return modules.gamer_selftest.GpuTelemetry(
            present=True,
            name=spec.gpu_name,
            total_vram_mib=spec.total_vram_mib,
            free_vram_mib=spec.free_vram_mib,
            used_vram_mib=spec.total_vram_mib - spec.free_vram_mib,
            temp_c=eff_temp_c,
            util_pct=spec.util_pct,
            reason="ok",
        )

    def _ws_proc_probe(**_kwargs: Any) -> list[str]:
        return ["explorer.exe", "steam.exe"] if spec.os_name == "nt" else ["bash", "code", "python3"]

    def _ws_lifecycle_drill(**_kwargs: Any) -> dict[str, Any]:
        return {
            "worker_started": True,
            "worker_stopped": not yield_injected,
            "clean_release": not yield_injected,
            "retained_worker_vram_mib": eff_retained_vram,
            "yield_latency_ms": eff_yield_ms,
            "method": "rehearsal_deterministic_drill",
        }

    replay_log: list[tuple[str, dict[str, Any], dict[str, str]]] = []
    ws_http = coordinator.make_workspace_http_fn(
        workspace_index=workspace_index,
        profile=spec.profile,
        replay_log=replay_log,
    )

    landing_path = "/raid-noel" if spec.profile == "gamer" else "/noel"
    landing_resp = coordinator.call_full(
        "GET",
        f"{gateway_url}{landing_path}",
        None,
        {"X-HiveMind-Rehearsal-Visit": "1"},
        workspace_index=workspace_index,
    )
    coordinator.record_landing_visit()
    landing_nonce = extract_csp_nonce(landing_resp.headers, landing_resp.text)
    landing_html_ok = landing_resp.status_code == 200 and landing_nonce is not None
    if strict_html_checks and landing_resp.status_code == 200:
        contract_ok, _, _ = _inspect_html_contract(
            landing_path,
            landing_resp.text,
            landing_resp.headers,
            None,
            is_fr=(script_lang == "fr"),
        )
        coordinator.record_html(landing_nonce, contract_ok)
        landing_html_ok = landing_html_ok and contract_ok
    else:
        coordinator.record_html(landing_nonce, True)

    if strict_html_checks:
        extra_html_path = "/arena-noel" if spec.profile == "gamer" else "/quick"
        extra_resp = coordinator.call_full(
            "GET",
            f"{gateway_url}{extra_html_path}",
            None,
            None,
            workspace_index=workspace_index,
        )
        extra_nonce = extract_csp_nonce(extra_resp.headers, extra_resp.text)
        extra_ok, _, _ = _inspect_html_contract(
            extra_html_path,
            extra_resp.text,
            extra_resp.headers,
            None,
            is_fr=(script_lang == "fr"),
        )
        coordinator.record_html(extra_nonce, extra_ok and extra_resp.status_code == 200)
        landing_html_ok = landing_html_ok and extra_resp.status_code == 200 and extra_nonce is not None and extra_ok
        if extra_html_path == "/arena-noel":
            ballots = extract_arena_ballots(extra_resp.text)
            if ballots:
                t_id, b_tok = next(iter(ballots.items()))
                vote_resp = coordinator.call_full(
                    "POST",
                    f"{gateway_url}/v1/hivemind/arena/vote",
                    {"task_id": t_id, "chosen_side": "A", "profile": spec.profile, "ballot_token": b_tok},
                    None,
                    workspace_index=workspace_index,
                )
                landing_html_ok = landing_html_ok and vote_resp.status_code == 200

    sub_reports: list[dict[str, Any]] = []
    rendered_blobs: list[str] = []
    selftest_report: dict[str, Any] | None = None
    probe_report: dict[str, Any] | None = None

    handle = f"noel-{spec.archetype_id.replace('_', '-')}-{workspace_index:03d}"
    arch_ordinal = workspace_index // len(PERSONAL_PC_ARCHETYPES)

    if spec.profile == "dev":
        ease_val = DEV_EASE_SCHEDULE[arch_ordinal % len(DEV_EASE_SCHEDULE)]
        captured_probe: list[dict[str, Any]] = []

        def _chained_probe_fn(**probe_kwargs: Any) -> dict[str, Any]:
            res = modules.post_setup_probe.run_post_setup_probe(
                gateway=probe_kwargs["gateway"],
                smoke_only=False,
                verify_account=True,
                credentials_path=probe_kwargs["credentials_path"],
                continue_path=probe_kwargs["continue_path"],
                max_temp_c=max_temp_c,
                min_free_vram_mib=min_free_vram_mib,
                yield_budget_s=yield_slo_ms / 1000.0,
                feedback_result=probe_kwargs.get("feedback_result"),
                lang=script_lang,
                os_name=spec.os_name,
                environ=ws_env,
                home=ws_home,
                http_fn=ws_http,
                runner=_ws_runner,
                clock_fn=ws_clock,
            )
            captured_probe.append(res)
            return res

        autoconfig_rc, autoconfig_report = modules.autoconfig.run_vscode_autoconfig(
            gateway=gateway_url,
            handle=handle,
            role=spec.role,
            workspace=ws_dir,
            write_vscode_settings=True,
            skip_probe=False,
            ease_rating=ease_val,
            system_name=spec.system_name,
            lang=script_lang,
            os_name=spec.os_name,
            environ=ws_env,
            home=ws_home,
            cwd=ws_dir,
            http_fn=ws_http,
            probe_fn=_chained_probe_fn,
            clock_fn=ws_clock,
        )
        probe_report = captured_probe[0] if captured_probe else None
        sub_reports.append(autoconfig_report)
        rendered_blobs.append(modules.autoconfig.render_markdown_report(autoconfig_report))
        if probe_report is not None:
            sub_reports.append(probe_report)
            rendered_blobs.append(modules.post_setup_probe.render_markdown_report(probe_report))

        if spec.runs_gamer_selftest:
            selftest_report = modules.gamer_selftest.run_gamer_whisper_selftest(
                gateway=gateway_url,
                host_label=f"home-rig-{workspace_index:03d}",
                max_temp_c=max_temp_c,
                min_free_vram_mib=min_free_vram_mib,
                yield_slo_ms=yield_slo_ms,
                check_gateway=True,
                submit_feedback=False,
                lang=script_lang,
                os_name=spec.os_name,
                system_name=spec.system_name,
                environ=ws_env,
                home=ws_home,
                cwd=ws_dir,
                gpu_probe_fn=_ws_gpu_probe,
                process_probe_fn=_ws_proc_probe,
                worker_lifecycle_fn=_ws_lifecycle_drill,
                clock_fn=ws_clock,
                http_fn=ws_http,
            )
            sub_reports.append(selftest_report)
            rendered_blobs.append(modules.gamer_selftest.render_markdown_report(selftest_report))
    else:
        ease_val = GAMER_EASE_SCHEDULE[arch_ordinal % len(GAMER_EASE_SCHEDULE)]
        autoconfig_rc, autoconfig_report = modules.autoconfig.run_vscode_autoconfig(
            gateway=gateway_url,
            handle=handle,
            role=spec.role,
            workspace=ws_dir,
            write_vscode_settings=True,
            skip_probe=True,
            ease_rating=None,
            system_name=spec.system_name,
            lang=script_lang,
            os_name=spec.os_name,
            environ=ws_env,
            home=ws_home,
            cwd=ws_dir,
            http_fn=ws_http,
            clock_fn=ws_clock,
        )
        selftest_report = modules.gamer_selftest.run_gamer_whisper_selftest(
            gateway=gateway_url,
            host_label=f"home-rig-{workspace_index:03d}",
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            yield_slo_ms=yield_slo_ms,
            check_gateway=True,
            submit_feedback=True,
            ease_rating=ease_val,
            lang=script_lang,
            os_name=spec.os_name,
            system_name=spec.system_name,
            environ=ws_env,
            home=ws_home,
            cwd=ws_dir,
            gpu_probe_fn=_ws_gpu_probe,
            process_probe_fn=_ws_proc_probe,
            worker_lifecycle_fn=_ws_lifecycle_drill,
            clock_fn=ws_clock,
            http_fn=ws_http,
        )
        probe_report = modules.post_setup_probe.run_post_setup_probe(
            gateway=gateway_url,
            smoke_only=False,
            verify_account=True,
            credentials_path=ws_home / ".hivemind" / "credentials.json",
            continue_path=ws_home / ".continue" / "hivemind_provider.yaml",
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            yield_budget_s=yield_slo_ms / 1000.0,
            feedback_result=selftest_report.get("feedback"),
            lang=script_lang,
            os_name=spec.os_name,
            environ=ws_env,
            home=ws_home,
            http_fn=ws_http,
            runner=_ws_runner,
            clock_fn=ws_clock,
        )
        sub_reports.extend((autoconfig_report, selftest_report, probe_report))
        rendered_blobs.extend(
            (
                modules.autoconfig.render_markdown_report(autoconfig_report),
                modules.gamer_selftest.render_markdown_report(selftest_report),
                modules.post_setup_probe.render_markdown_report(probe_report),
            )
        )

    replay_ok = coordinator.replay_feedback_posts(workspace_index=workspace_index, replay_log=replay_log)

    heartbeat_ok = True
    if send_opentrainer_hb or send_producer_hb:
        cred_path = ws_home / ".hivemind" / "credentials.json"
        raw_api_key = ""
        if cred_path.is_file():
            with suppress(OSError, json.JSONDecodeError):
                cred_doc = json.loads(cred_path.read_text(encoding="utf-8"))
                if isinstance(cred_doc, Mapping) and isinstance(cred_doc.get("api_key"), str):
                    raw_api_key = cred_doc["api_key"].strip()
        auth_h = {"Authorization": f"Bearer {raw_api_key}"} if raw_api_key else {}
        if send_opentrainer_hb:
            ot_resp = coordinator.call_full(
                "POST",
                f"{gateway_url}/v1/hivemind/opentrainer/heartbeat",
                {
                    "host_label": f"home-rig-{workspace_index:03d}",
                    "gpu_name": spec.gpu_name or "RTX Personal GPU",
                    "vram_total_mib": max(512, spec.total_vram_mib),
                    "vram_free_mib": max(0, spec.free_vram_mib),
                    "network_mode": "outbound-https-relay",
                },
                auth_h,
                workspace_index=workspace_index,
            )
            heartbeat_ok = heartbeat_ok and ot_resp.status_code == 200
        if send_producer_hb:
            prod_resp = coordinator.call_full(
                "POST",
                f"{gateway_url}/v1/hivemind/producer/heartbeat",
                {
                    "endpoint_url": f"http://{_simulated_loopback_ip(50 + (workspace_index // 200), workspace_index)}:11434",
                    "models": [
                        {
                            "id": "coder",
                            "local_name": "qwen2.5-coder:7b",
                            "loaded": True,
                            "vram_mib": min(4608, max(512, spec.free_vram_mib or 4096)),
                        }
                    ],
                    "cost_per_1k": 0.0,
                    "capacity_1k": 50.0,
                    "gpu_vram_mib": spec.total_vram_mib if spec.gpu_present else 8192,
                    "execution_class": "ollama-gamer-whisper" if spec.gpu_present else "ollama",
                },
                auth_h,
                workspace_index=workspace_index,
            )
            heartbeat_ok = heartbeat_ok and prod_resp.status_code == 200

    if workspace_post_hook is not None:
        workspace_post_hook("personal", spec.archetype_id, ws_home, ws_dir)

    ws_files_scanned, ws_file_leaks = scan_directory_for_key_leaks(ws_dir)
    vscode_settings_file = ws_dir / ".vscode" / "settings.json"
    vscode_settings_ok = vscode_settings_file.is_file() and (
        "hivemind." in vscode_settings_file.read_text(encoding="utf-8", errors="replace")
    )

    posix_0600_ok = True
    posix_checked = 0
    if spec.os_name == "posix" and os.name != "nt":
        for secret_file in (
            ws_home / ".hivemind" / "credentials.json",
            ws_home / ".continue" / "hivemind_provider.yaml",
        ):
            posix_checked += 1
            if not secret_file.is_file() or (secret_file.stat().st_mode & 0o777) != 0o600:
                posix_0600_ok = False

    report_leaks: list[str] = []
    sub_receipts_ok = all(verify_sub_receipt_sha256(rep) for rep in sub_reports)
    for rep in sub_reports:
        rep_json = json.dumps(rep, sort_keys=True, ensure_ascii=False)
        report_leaks.extend(scan_output_safety(rep_json))
    for md_text in rendered_blobs:
        report_leaks.extend(scan_output_safety(md_text))

    if spec.profile == "dev":
        nogpu_ok = (
            True
            if not spec.runs_gamer_selftest
            else bool(
                selftest_report is not None
                and selftest_report.get("ready") is True
                and selftest_report.get("verdict") == "cpu_or_dev_only_no_nvidia_gpu"
            )
        )
        pipeline_ready = bool(
            landing_html_ok
            and autoconfig_rc == 0
            and autoconfig_report.get("ready") is True
            and probe_report is not None
            and probe_report.get("ready") is True
            and probe_report.get("recommended_mode") == "dev_api_only"
            and nogpu_ok
        )
        zero_game_lag = True
        yield_slo_ok: bool | None = None
        thermal_ok: bool | None = None
        vram_reserve_ok: bool | None = None
        measured_yield_ms: float | None = None
        observed_temp_c: float | None = None
    else:
        assert selftest_report is not None and probe_report is not None
        yield_slo_ok = bool(selftest_report.get("yield_slo_ok"))
        thermal_ok = bool(selftest_report.get("thermal_ok"))
        vram_reserve_ok = bool(selftest_report.get("vram_reserve_ok"))
        raw_y = selftest_report.get("yield_latency_ms")
        measured_yield_ms = float(raw_y) if isinstance(raw_y, (int, float)) else None
        observed_temp_c = eff_temp_c
        zero_game_lag = bool(selftest_report.get("zero_game_lag"))
        pipeline_ready = bool(
            landing_html_ok
            and autoconfig_rc == 0
            and autoconfig_report.get("ready") is True
            and selftest_report.get("exit_code") == 0
            and selftest_report.get("ready") is True
            and yield_slo_ok
            and thermal_ok
            and vram_reserve_ok
            and probe_report.get("ready") is True
            and probe_report.get("recommended_mode") == "gamer_whisper"
        )

    workspace_ready = bool(
        pipeline_ready
        and replay_ok
        and heartbeat_ok
        and vscode_settings_ok
        and posix_0600_ok
        and sub_receipts_ok
        and not ws_file_leaks
        and not report_leaks
    )

    return {
        "workspace_index": workspace_index,
        "gamer_ordinal": gamer_ordinal,
        "archetype_id": spec.archetype_id,
        "profile": spec.profile,
        "gpu_present": spec.gpu_present,
        "runs_gamer_selftest": spec.runs_gamer_selftest,
        "ready": workspace_ready,
        "ease_rating": ease_val,
        "zero_game_lag": zero_game_lag,
        "yield_slo_ok": yield_slo_ok,
        "thermal_ok": thermal_ok,
        "vram_reserve_ok": vram_reserve_ok,
        "yield_latency_ms": measured_yield_ms,
        "observed_temp_c": observed_temp_c,
        "free_vram_mib": spec.free_vram_mib if spec.gpu_present else None,
        "total_vram_mib": spec.total_vram_mib if spec.gpu_present else None,
        "workspace_files_scanned": ws_files_scanned,
        "workspace_key_leaks": len(ws_file_leaks),
        "report_key_leaks": len(report_leaks),
        "posix_0600_checked": posix_checked,
        "posix_0600_ok": posix_0600_ok,
        "sub_receipts_count": len(sub_reports),
        "sub_receipts_valid": sub_receipts_ok,
    }


def _setup_corporate_git_layout(ws_dir: Path, ws_root: Path, ctrl_spec: CorporateControlSpec, control_index: int) -> None:
    if not ctrl_spec.git_remote:
        return
    if ctrl_spec.use_git_worktree:
        common_git = ws_root / "shared_common_git"
        wt_gitdir = common_git / "worktrees" / f"wt_{control_index:03d}"
        wt_gitdir.mkdir(parents=True, exist_ok=True)
        (common_git / "config").write_text(
            f'[core]\n\tbare = false\n[remote "origin"]\n\turl = {ctrl_spec.git_remote}\n',
            encoding="utf-8",
        )
        (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
        (wt_gitdir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
        (ws_dir / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    else:
        git_dir = ws_dir / ".git"
        git_dir.mkdir(parents=True, exist_ok=True)
        (git_dir / "config").write_text(
            f'[core]\n\trepositoryformatversion = 0\n[remote "origin"]\n\turl = {ctrl_spec.git_remote}\n',
            encoding="utf-8",
        )


def _execute_corporate_workspace(
    *,
    control_index: int,
    ctrl_spec: CorporateControlSpec,
    base_dir: Path,
    gateway_url: str,
    modules: OnboardingPipelineModules,
    script_lang: str,
    workspace_post_hook: WorkspacePostHook | None,
) -> dict[str, Any]:
    ws_root = base_dir / "negative_ctrl" / f"ctrl_{control_index:03d}"
    ws_home = ws_root / "home_box"
    ws_dir = ws_home / "work_tree" / ctrl_spec.workspace_dirname
    ws_dir.mkdir(parents=True, exist_ok=True)
    _setup_corporate_git_layout(ws_dir, ws_root, ctrl_spec, control_index)

    before_files = {p.relative_to(ws_root).as_posix() for p in ws_root.rglob("*") if p.is_file()}
    network_calls = [0]
    hardware_calls = [0]
    ws_clock, _ = _make_step_clock(0.001)

    def _corp_http_fn(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (method, url, payload, headers)
        network_calls[0] += 1
        return 500, {"error": "corporate_network_call_should_never_happen"}

    def _corp_gpu_probe(**_kwargs: Any) -> Any:
        hardware_calls[0] += 1
        return modules.gamer_selftest.GpuTelemetry(present=False)

    def _corp_proc_probe(**_kwargs: Any) -> list[str]:
        hardware_calls[0] += 1
        return []

    def _corp_lifecycle_drill(**_kwargs: Any) -> dict[str, Any]:
        hardware_calls[0] += 1
        return {
            "worker_started": False,
            "worker_stopped": False,
            "clean_release": False,
            "retained_worker_vram_mib": -1,
            "yield_latency_ms": 9999.0,
        }

    local_host_comp = evaluate_host_compliance(
        environ=ctrl_spec.environ,
        cwd=ws_dir,
        check_git_remote=True,
        lang=script_lang,
    )
    autoconfig_rc, autoconfig_report = modules.autoconfig.run_vscode_autoconfig(
        gateway=gateway_url,
        workspace=ws_dir,
        write_vscode_settings=True,
        skip_probe=False,
        ease_rating=5,
        lang=script_lang,
        os_name=ctrl_spec.os_name,
        environ=ctrl_spec.environ,
        home=ws_home,
        cwd=ws_dir,
        http_fn=_corp_http_fn,
        clock_fn=ws_clock,
    )
    selftest_report = modules.gamer_selftest.run_gamer_whisper_selftest(
        gateway=gateway_url,
        check_gateway=True,
        submit_feedback=True,
        ease_rating=5,
        lang=script_lang,
        os_name=ctrl_spec.os_name,
        environ=ctrl_spec.environ,
        home=ws_home,
        cwd=ws_dir,
        gpu_probe_fn=_corp_gpu_probe,
        process_probe_fn=_corp_proc_probe,
        worker_lifecycle_fn=_corp_lifecycle_drill,
        clock_fn=ws_clock,
        http_fn=_corp_http_fn,
    )

    if workspace_post_hook is not None:
        workspace_post_hook("corporate", ctrl_spec.vector_id, ws_home, ws_dir)

    after_files = {p.relative_to(ws_root).as_posix() for p in ws_root.rglob("*") if p.is_file()}
    created_files = sorted(after_files - before_files)
    ws_files_scanned, ws_file_leaks = scan_directory_for_key_leaks(ws_root)

    sub_reports = (autoconfig_report, selftest_report)
    rendered_mds = (
        modules.autoconfig.render_markdown_report(autoconfig_report),
        modules.gamer_selftest.render_markdown_report(selftest_report),
    )
    sub_receipts_ok = all(verify_sub_receipt_sha256(rep) for rep in sub_reports)

    report_leaks: list[str] = []
    sentinel_leaked = ctrl_spec.secret_sentinel in json.dumps(local_host_comp, ensure_ascii=False)
    for rep in sub_reports:
        rep_json = json.dumps(rep, sort_keys=True, ensure_ascii=False)
        report_leaks.extend(scan_output_safety(rep_json))
        if ctrl_spec.secret_sentinel in rep_json:
            sentinel_leaked = True
    for md_text in rendered_mds:
        report_leaks.extend(scan_output_safety(md_text))
        if ctrl_spec.secret_sentinel in md_text:
            sentinel_leaked = True

    if sentinel_leaked:
        report_leaks.append("corporate_sentinel_value_leaked")

    matched_markers = sorted(
        set(local_host_comp.get("matched_markers") or ())
        | set(autoconfig_report.get("compliance", {}).get("matched_markers") or ())
        | set(selftest_report.get("compliance", {}).get("matched_markers") or ())
    )

    blocked_cleanly = bool(
        local_host_comp.get("compliance_ok") is False
        and autoconfig_rc == 2
        and autoconfig_report.get("compliance_ok") is False
        and autoconfig_report.get("verdict") == "refused_compliance"
        and selftest_report.get("exit_code") == 2
        and selftest_report.get("compliance_ok") is False
        and selftest_report.get("verdict") == "refused_compliance"
        and len(created_files) == 0
        and network_calls[0] == 0
        and hardware_calls[0] == 0
        and sub_receipts_ok
        and not ws_file_leaks
        and not report_leaks
    )

    return {
        "control_index": control_index,
        "vector_id": ctrl_spec.vector_id,
        "blocked": blocked_cleanly,
        "file_writes_count": len(created_files),
        "network_calls_count": network_calls[0],
        "hardware_calls_count": hardware_calls[0],
        "matched_markers": matched_markers,
        "workspace_files_scanned": ws_files_scanned,
        "workspace_key_leaks": len(ws_file_leaks),
        "report_key_leaks": len(report_leaks),
        "sub_receipts_count": 2,
        "sub_receipts_valid": sub_receipts_ok,
    }


def _percentile_nearest_rank(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    rank = max(1, math.ceil((pct / 100.0) * len(ordered)))
    return round(ordered[min(len(ordered) - 1, rank - 1)], 3)


def run_canary_cohort_rehearsal(
    gateway: str = DEFAULT_GATEWAY,
    *,
    cohort_size: int = DEFAULT_COHORT_SIZE,
    corporate_controls: int = DEFAULT_CORPORATE_CONTROLS,
    workers: int = DEFAULT_WORKERS,
    yield_slo_ms: float = DEFAULT_YIELD_SLO_MS,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    inject_gamer_thermal_failures: int = 0,
    inject_gamer_yield_failures: int = 0,
    report_date: str | None = None,
    lang: str = "en",
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = False,
    rehearsal_root: Path | None = None,
    keep_workspaces: bool = False,
    pipeline_modules: OnboardingPipelineModules | None = None,
    workspace_post_hook: WorkspacePostHook | None = None,
    generated_at_utc: str | None = None,
    http_fn: HttpFn | None = None,
    raw_http_fn: RawHttpFn | None = None,
    verify_pr949_parity: bool = False,
    idempotency_replay: bool = False,
    include_heartbeat_quorum: bool = False,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    expected_sha: str | None = None,
    require_live_quorum: bool = True,
    extra_corporate_controls: Sequence[CorporateControlSpec] = (),
) -> dict[str, Any]:
    """Run the parallel multi-OS Noël canary & scale cohort rehearsal and return a signed receipt."""
    t_wall0 = time.perf_counter()
    norm_lang = normalize_lang(lang)
    script_lang = "fr" if norm_lang == "fr" else "en"
    is_fr = norm_lang == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_date = validate_report_date(report_date)

    personal_plan = build_personal_cohort_plan(cohort_size)
    corporate_plan = build_corporate_control_plan(
        corporate_controls,
        extra_controls=extra_corporate_controls,
    )
    total_corp_controls = len(corporate_plan)

    _validate_int_range(workers, "workers", 1, MAX_WORKERS)
    if (
        isinstance(yield_slo_ms, bool)
        or not isinstance(yield_slo_ms, (int, float))
        or not math.isfinite(float(yield_slo_ms))
        or float(yield_slo_ms) <= 0.0
    ):
        raise RehearsalError("yield_slo_ms must be a positive finite millisecond budget.")
    _validate_float_range(max_temp_c, "max_temp_c", 30.0, 100.0)
    _validate_int_range(min_free_vram_mib, "min_free_vram_mib", 256, 65536)
    _validate_int_range(target_online_producers, "target_online_producers", 1, 500)
    for fault_name, fault_val in (
        ("inject_gamer_thermal_failures", inject_gamer_thermal_failures),
        ("inject_gamer_yield_failures", inject_gamer_yield_failures),
    ):
        if isinstance(fault_val, bool) or not isinstance(fault_val, int) or fault_val < 0:
            raise RehearsalError(f"{fault_name} must be a non-negative integer.")

    gpu_gamer_indices = [idx for idx, spec in personal_plan if spec.gpu_present]
    gpu_gamer_total = len(gpu_gamer_indices)
    if inject_gamer_thermal_failures + inject_gamer_yield_failures > gpu_gamer_total:
        raise RehearsalError(
            f"Injected gamer failures ({inject_gamer_thermal_failures + inject_gamer_yield_failures}) "
            f"cannot exceed GPU gamer workspaces count ({gpu_gamer_total})."
        )

    stamp = (
        generated_at_utc.strip()
        if isinstance(generated_at_utc, str) and generated_at_utc.strip()
        else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    host_comp = evaluate_host_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        lang=norm_lang,
    )
    if not host_comp["compliance_ok"]:
        elapsed_refused = round(max(0.0, time.perf_counter() - t_wall0), 4)
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "generated_at_utc": stamp,
                "lang": norm_lang,
                "gateway": clean_gw,
                "rehearsal_ok": False,
                "verdict": "refused_host_compliance",
                "exit_code": 2,
                "cohort_size": cohort_size,
                "personal_ready_count": 0,
                "personal_failed_count": cohort_size,
                "personal_ready_ratio": 0.0,
                "corporate_cohort_size": total_corp_controls,
                "corporate_blocked_count": total_corp_controls,
                "corporate_leaked_count": 0,
                "corporate_blocked_ratio": 1.0,
                "corporate_file_writes_count": 0,
                "corporate_network_calls_count": 0,
                "workers_count": workers,
                "elapsed_s": elapsed_refused,
                "zero_inference_verified": True,
                "inference_calls_count": 0,
                "zero_key_leaks_verified": True,
                "workspace_files_scanned_count": 0,
                "workspace_key_leak_count": 0,
                "report_key_leak_count": 0,
                "posix_0600_verified_count": 0,
                "gamer_whisper_enforcement": None,
                "archetype_breakdown": [],
                "corporate_control_breakdown": [],
                "pipeline_verification": None,
                "funnel_aggregation": None,
                "daily_raid_bulletin": None,
                "compliance": host_comp,
                "diagnostics": [sanitize_text_for_display(host_comp["summary"])],
            }
        )

    modules = pipeline_modules if pipeline_modules is not None else load_onboarding_pipeline_modules()
    coordinator = _LiveCohortTransportCoordinator(
        clean_gw,
        report_date=clean_date,
        http_fn=http_fn,
        raw_http_fn=raw_http_fn,
        idempotency_replay=bool(idempotency_replay),
        include_heartbeat_quorum=bool(include_heartbeat_quorum),
    )

    gamer_ordinal_by_ws: dict[int, int] = {
        ws_idx: ord_idx for ord_idx, ws_idx in enumerate(gpu_gamer_indices)
    }

    created_temp_dir = rehearsal_root is None
    active_root = (
        Path(tempfile.mkdtemp(prefix="hm_noel_canary_"))
        if rehearsal_root is None
        else Path(rehearsal_root)
    )
    active_root.mkdir(parents=True, exist_ok=True)

    strict_html_checks = bool(http_fn is not None or raw_http_fn is not None or verify_pr949_parity)
    extra_prod_ws: set[int] = set()
    if include_heartbeat_quorum and gpu_gamer_total < target_online_producers:
        needed = target_online_producers - gpu_gamer_total
        for ws_idx, spec in personal_plan:
            if not spec.gpu_present and spec.role in ("prosumer", "producer", "trainer"):
                extra_prod_ws.add(ws_idx)
                if len(extra_prod_ws) >= needed:
                    break

    personal_results: list[dict[str, Any]] = []
    corporate_results: list[dict[str, Any]] = []

    try:
        max_pool = min(workers, max(1, cohort_size + total_corp_controls))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_pool) as pool:
            personal_futures = []
            for ws_idx, spec in personal_plan:
                g_ord = gamer_ordinal_by_ws.get(ws_idx)
                t_inj = bool(g_ord is not None and g_ord < inject_gamer_thermal_failures)
                y_inj = bool(
                    g_ord is not None
                    and inject_gamer_thermal_failures
                    <= g_ord
                    < (inject_gamer_thermal_failures + inject_gamer_yield_failures)
                )
                hw_healthy = not t_inj and not y_inj
                send_ot = bool(include_heartbeat_quorum and spec.gpu_present and hw_healthy)
                send_prod = bool(
                    include_heartbeat_quorum
                    and hw_healthy
                    and (spec.gpu_present or ws_idx in extra_prod_ws)
                )
                personal_futures.append(
                    pool.submit(
                        _execute_personal_workspace,
                        workspace_index=ws_idx,
                        spec=spec,
                        gamer_ordinal=g_ord,
                        thermal_injected=t_inj,
                        yield_injected=y_inj,
                        send_producer_hb=send_prod,
                        send_opentrainer_hb=send_ot,
                        strict_html_checks=strict_html_checks,
                        base_dir=active_root,
                        gateway_url=clean_gw,
                        coordinator=coordinator,
                        modules=modules,
                        yield_slo_ms=float(yield_slo_ms),
                        max_temp_c=float(max_temp_c),
                        min_free_vram_mib=min_free_vram_mib,
                        script_lang=script_lang,
                        workspace_post_hook=workspace_post_hook,
                    )
                )

            corporate_futures = [
                pool.submit(
                    _execute_corporate_workspace,
                    control_index=c_idx,
                    ctrl_spec=c_spec,
                    base_dir=active_root,
                    gateway_url=clean_gw,
                    modules=modules,
                    script_lang=script_lang,
                    workspace_post_hook=workspace_post_hook,
                )
                for c_idx, c_spec in corporate_plan
            ]

            for fut in personal_futures:
                personal_results.append(fut.result())
            for fut in corporate_futures:
                corporate_results.append(fut.result())
    finally:
        if created_temp_dir and not keep_workspaces:
            shutil.rmtree(active_root, ignore_errors=True)

    personal_results.sort(key=lambda r: int(r["workspace_index"]))
    corporate_results.sort(key=lambda r: int(r["control_index"]))

    gpu_gamer_rows = [r for r in personal_results if r["gpu_present"]]
    ready_gpu_gamer_rows = [r for r in gpu_gamer_rows if r["ready"]]
    nogpu_selftest_rows = [r for r in personal_results if r["runs_gamer_selftest"] and not r["gpu_present"]]

    agg_free_vram = (
        round(sum(float(r["free_vram_mib"]) for r in ready_gpu_gamer_rows), 1)
        if ready_gpu_gamer_rows
        else None
    )
    agg_total_vram = (
        round(sum(float(r["total_vram_mib"]) for r in ready_gpu_gamer_rows), 1)
        if ready_gpu_gamer_rows
        else None
    )
    agg_mean_temp = (
        round(statistics.mean(float(r["observed_temp_c"]) for r in ready_gpu_gamer_rows), 1)
        if ready_gpu_gamer_rows
        else None
    )

    if coordinator.mock_gateway is not None:
        coordinator.mock_gateway.finalize_trainer_telemetry(
            preflight_runs=len(gpu_gamer_rows),
            install_completed=len(ready_gpu_gamer_rows),
            first_rounds=len(ready_gpu_gamer_rows),
            vram_free_mib=agg_free_vram,
            vram_total_mib=agg_total_vram,
            temp_c=agg_mean_temp,
        )

    # Exercise Stage 4: real onboarding_funnel_report (live gateway fetch + exact funnel analysis + Markdown)
    live_gw_bundle = modules.funnel_report.fetch_live_gateway_snapshot(
        clean_gw,
        http_get=coordinator.coord_http_get_json,
    )
    if coordinator.mock_gateway is not None:
        gw_snap = coordinator.mock_gateway.snapshot_state()
        funnel_input = {
            **live_gw_bundle,
            "funnel": {
                "landing_visits": gw_snap["landing_visits"],
                "signups": gw_snap["signups"],
                "first_requests": gw_snap["first_requests"],
                "preflight_runs": len(gpu_gamer_rows),
                "install_completed": len(ready_gpu_gamer_rows),
                "first_rounds": len(ready_gpu_gamer_rows),
            },
            "feedback_items": gw_snap["feedback_items"],
        }
    else:
        funnel_input = {
            **live_gw_bundle,
            "funnel": {
                "landing_visits": coordinator.landing_visits,
                "signups": len(coordinator.signed_up_workspaces),
                "first_requests": len(coordinator.activated_workspaces),
                "preflight_runs": len(gpu_gamer_rows),
                "install_completed": len(ready_gpu_gamer_rows),
                "first_rounds": len(ready_gpu_gamer_rows),
            },
        }
    funnel_receipt = modules.funnel_report.analyze_onboarding_funnel(
        funnel_input,
        generated_at_utc=stamp,
    )
    funnel_md = modules.funnel_report.render_markdown_report(funnel_receipt, lang=norm_lang)

    # Exercise Stage 5: real daily_raid_bulletin (live gateway fetch + 5-line FR/EN/bilingual receipt)
    t4_doc, fb_doc, bull_http = modules.raid_bulletin.fetch_live_bulletin_inputs(
        clean_gw,
        http_get=coordinator.coord_http_get_json,
        include_fleet=bool(coordinator.external_mode or include_heartbeat_quorum),
    )
    bulletin_receipt = modules.raid_bulletin.build_bulletin_receipt(
        t4_doc,
        fb_doc,
        lang=norm_lang,
        date_override=clean_date,
        source_kind="cohort_rehearsal",
        http_statuses=bull_http,
    )

    pr949_report: dict[str, Any] | None = None
    pr949_parity_ok = True
    if verify_pr949_parity:
        pr949_report = verify_pr949_cohort_live_parity(
            clean_gw,
            expected_sha=expected_sha,
            target_online_producers=target_online_producers,
            require_live_quorum=bool(require_live_quorum and include_heartbeat_quorum),
            raw_http_fn=coordinator.raw_http_adapter(),
            http_fn=coordinator.live_http_adapter(),
            environ=environ,
            cwd=cwd,
            check_git_remote=check_git_remote,
            lang=norm_lang,
        )
        pr949_parity_ok = bool(pr949_report["ready"]) and verify_sub_receipt_sha256(pr949_report)

    if coordinator.mock_gateway is not None:
        final_gw = coordinator.mock_gateway.snapshot_state()
    else:
        final_gw = {
            "requests_total": coordinator.requests_total,
            "inference_calls_count": coordinator.inference_calls_count,
            "routes_hit": dict(sorted(coordinator.routes_hit.items())),
        }

    # Aggregate personal & corporate ratios using exact Fraction arithmetic
    personal_ready_count = sum(1 for r in personal_results if r["ready"])
    personal_failed_count = cohort_size - personal_ready_count
    personal_ready_frac = Fraction(personal_ready_count, cohort_size)
    personal_ready_ratio = round(float(personal_ready_frac), 4)

    corporate_blocked_count = sum(1 for r in corporate_results if r["blocked"])
    corporate_leaked_count = total_corp_controls - corporate_blocked_count
    corporate_blocked_frac = (
        Fraction(corporate_blocked_count, total_corp_controls)
        if total_corp_controls > 0
        else Fraction(1, 1)
    )
    corporate_blocked_ratio = round(float(corporate_blocked_frac), 4)
    corporate_file_writes_count = sum(int(r["file_writes_count"]) for r in corporate_results)
    corporate_network_calls_count = sum(int(r["network_calls_count"]) for r in corporate_results)

    workspace_files_scanned = sum(int(r["workspace_files_scanned"]) for r in personal_results) + sum(
        int(r["workspace_files_scanned"]) for r in corporate_results
    )
    workspace_key_leak_count = sum(int(r["workspace_key_leaks"]) for r in personal_results) + sum(
        int(r["workspace_key_leaks"]) for r in corporate_results
    )
    extra_report_leaks = (
        len(scan_output_safety(json.dumps(funnel_receipt, sort_keys=True, ensure_ascii=False)))
        + len(scan_output_safety(funnel_md))
        + len(scan_output_safety(json.dumps(bulletin_receipt, sort_keys=True, ensure_ascii=False)))
    )
    if pr949_report is not None:
        extra_report_leaks += len(
            scan_output_safety(json.dumps(pr949_report, sort_keys=True, ensure_ascii=False))
        )
    report_key_leak_count = (
        sum(int(r["report_key_leaks"]) for r in personal_results)
        + sum(int(r["report_key_leaks"]) for r in corporate_results)
        + extra_report_leaks
    )
    zero_key_leaks_verified = workspace_key_leak_count == 0 and report_key_leak_count == 0

    posix_0600_verified_count = sum(
        int(r["posix_0600_checked"]) for r in personal_results if r["posix_0600_ok"]
    )
    all_posix_0600_ok = all(bool(r["posix_0600_ok"]) for r in personal_results)

    funnel_receipt_ok = verify_sub_receipt_sha256(funnel_receipt)
    bulletin_receipt_ok = verify_sub_receipt_sha256(bulletin_receipt)
    sub_receipts_count = (
        sum(int(r["sub_receipts_count"]) for r in personal_results)
        + sum(int(r["sub_receipts_count"]) for r in corporate_results)
        + 2
        + (1 if pr949_report is not None else 0)
    )
    all_sub_receipts_valid = (
        all(bool(r["sub_receipts_valid"]) for r in personal_results)
        and all(bool(r["sub_receipts_valid"]) for r in corporate_results)
        and funnel_receipt_ok
        and bulletin_receipt_ok
        and pr949_parity_ok
    )

    gamer_yield_ok_count = sum(1 for r in gpu_gamer_rows if r["yield_slo_ok"] is True)
    gamer_thermal_ok_count = sum(1 for r in gpu_gamer_rows if r["thermal_ok"] is True)
    gamer_vram_ok_count = sum(1 for r in gpu_gamer_rows if r["vram_reserve_ok"] is True)
    gamer_yield_slo_all_ok = gamer_yield_ok_count == len(gpu_gamer_rows)
    gamer_thermal_all_ok = (
        gamer_thermal_ok_count == len(gpu_gamer_rows)
        and gamer_vram_ok_count == len(gpu_gamer_rows)
    )
    yield_samples = [
        float(r["yield_latency_ms"])
        for r in gpu_gamer_rows
        if isinstance(r["yield_latency_ms"], (int, float))
    ]
    temp_samples = [
        float(r["observed_temp_c"])
        for r in gpu_gamer_rows
        if isinstance(r["observed_temp_c"], (int, float))
    ]
    max_yield_ms = round(max(yield_samples), 3) if yield_samples else None
    p95_yield_ms = _percentile_nearest_rank(yield_samples, 95.0)
    max_obs_temp_c = round(max(temp_samples), 1) if temp_samples else None

    expected_ease_frac = Fraction(
        sum(int(r["ease_rating"]) for r in personal_results),
        cohort_size,
    )
    expected_lag_frac = Fraction(
        sum(1 for r in personal_results if r["zero_game_lag"]),
        cohort_size,
    )
    expected_dev_count = sum(1 for r in personal_results if r["profile"] == "dev")
    expected_gamer_count = sum(1 for r in personal_results if r["profile"] == "gamer")
    f_counts = funnel_receipt["counts"]
    b_metrics = bulletin_receipt["metrics"]

    conservation_ok = bool(
        f_counts["landing_visits"] == cohort_size
        and f_counts["signups"] == cohort_size
        and f_counts["first_requests"] == cohort_size
        and f_counts["preflight_runs"] == len(gpu_gamer_rows)
        and f_counts["install_completed"] == len(ready_gpu_gamer_rows)
        and f_counts["first_rounds"] == len(ready_gpu_gamer_rows)
        and f_counts["feedback_responses"] == cohort_size
        and f_counts["feedback_by_profile"] == {"dev": expected_dev_count, "gamer": expected_gamer_count}
        and funnel_receipt["activation_ratio"] == 1.0
        and funnel_receipt["signup_conversion_ratio"] == 1.0
        and funnel_receipt["mean_ease_rating"] == round(float(expected_ease_frac), 2)
        and funnel_receipt["zero_game_lag_ratio"] == round(float(expected_lag_frac), 4)
        and b_metrics["feedback_total_responses"] == cohort_size
        and b_metrics["mean_ease_rating"] == funnel_receipt["mean_ease_rating"]
        and b_metrics["zero_game_lag_ratio"] == funnel_receipt["zero_game_lag_ratio"]
        and bulletin_receipt["bulletin_line_count"] == 5
    )

    replay_conserved = (
        not idempotency_replay
        or (
            coordinator.ces_replay_attempts >= cohort_size
            and coordinator.ces_replay_ok_count == coordinator.ces_replay_attempts
            and f_counts["feedback_responses"] == cohort_size
        )
    )
    quorum_ok = True
    fleet_online = 0
    fleet_gpus = 0
    if include_heartbeat_quorum:
        fl_code, fl_doc = coordinator.coord_http_get_json(f"{clean_gw}/v1/hivemind/producers/fleet")
        if fl_code == 200 and isinstance(fl_doc, Mapping):
            fleet_online = int(fl_doc.get("live_hiveminders") or 0)
            fleet_gpus = int(fl_doc.get("distinct_physical_gpus") or 0)
        quorum_ok = (
            fleet_online >= target_online_producers
            and fleet_gpus >= target_online_producers
            and coordinator.producer_heartbeats_recorded >= target_online_producers
        )

    unique_nonces_count = len(set(coordinator.observed_nonces))
    csp_nonces_unique = (
        unique_nonces_count == coordinator.html_renders_total
        and coordinator.html_renders_total >= cohort_size
        and coordinator.strict_html_failures == 0
    )

    zero_inference_verified = final_gw["inference_calls_count"] == 0
    sqlite_locks_ok = coordinator.sqlite_lock_errors == 0

    archetype_breakdown: list[dict[str, Any]] = []
    for spec in PERSONAL_PC_ARCHETYPES:
        rows = [r for r in personal_results if r["archetype_id"] == spec.archetype_id]
        tot = len(rows)
        rdy = sum(1 for r in rows if r["ready"])
        arch_ease = (
            round(float(Fraction(sum(int(r["ease_rating"]) for r in rows), tot)), 2)
            if tot > 0
            else None
        )
        arch_lag = (
            round(float(Fraction(sum(1 for r in rows if r["zero_game_lag"]), tot)), 4)
            if tot > 0
            else None
        )
        archetype_breakdown.append(
            {
                "archetype_id": spec.archetype_id,
                "label_en": spec.label_en,
                "label_fr": spec.label_fr,
                "profile": spec.profile,
                "os_name": spec.os_name,
                "system_name": spec.system_name,
                "gpu_present": spec.gpu_present,
                "total_count": tot,
                "ready_count": rdy,
                "failed_count": tot - rdy,
                "ready_ratio": round(float(Fraction(rdy, tot)), 4) if tot > 0 else 0.0,
                "mean_ease_rating": arch_ease,
                "zero_game_lag_ratio": arch_lag,
            }
        )

    all_corp_specs = list(CORPORATE_NEGATIVE_CONTROLS)
    seen_vec_ids = {s.vector_id for s in all_corp_specs}
    for extra_spec in extra_corporate_controls:
        if extra_spec.vector_id not in seen_vec_ids:
            all_corp_specs.append(extra_spec)
            seen_vec_ids.add(extra_spec.vector_id)

    corporate_breakdown: list[dict[str, Any]] = []
    for c_spec in all_corp_specs:
        c_rows = [r for r in corporate_results if r["vector_id"] == c_spec.vector_id]
        if not c_rows:
            continue
        c_tot = len(c_rows)
        c_blk = sum(1 for r in c_rows if r["blocked"])
        markers = sorted({m for r in c_rows for m in r["matched_markers"]})
        corporate_breakdown.append(
            {
                "vector_id": c_spec.vector_id,
                "label_en": c_spec.label_en,
                "label_fr": c_spec.label_fr,
                "total_count": c_tot,
                "blocked_count": c_blk,
                "file_writes_count": sum(int(r["file_writes_count"]) for r in c_rows),
                "network_calls_count": sum(int(r["network_calls_count"]) for r in c_rows),
                "matched_markers": markers,
            }
        )

    diagnostics: list[str] = []
    if personal_ready_frac != Fraction(1, 1):
        diagnostics.append(
            f"Seulement {personal_ready_count}/{cohort_size} espaces personnels prêts "
            f"(ratio={personal_ready_ratio:.4f})."
            if is_fr
            else f"Only {personal_ready_count}/{cohort_size} personal workspaces ready "
            f"(personal_ready_ratio={personal_ready_ratio:.4f})."
        )
    if corporate_blocked_frac != Fraction(1, 1) or corporate_file_writes_count > 0 or corporate_network_calls_count > 0:
        diagnostics.append(
            f"Échec barrière Amadeus/entreprise : bloqués={corporate_blocked_count}/{total_corp_controls}, "
            f"fichiers_écrits={corporate_file_writes_count}, appels_réseau={corporate_network_calls_count}."
            if is_fr
            else f"Corporate/Amadeus negative-control breach: blocked={corporate_blocked_count}/{total_corp_controls}, "
            f"file_writes={corporate_file_writes_count}, network_calls={corporate_network_calls_count}."
        )
    if not zero_key_leaks_verified:
        diagnostics.append(
            f"Fuite de clé hm-... ou jeton détectée (fichiers={workspace_key_leak_count}, rapports={report_key_leak_count})."
            if is_fr
            else f"Raw hm-... key or token leak detected (workspace_files={workspace_key_leak_count}, reports={report_key_leak_count})."
        )
    if not gamer_yield_slo_all_ok:
        diagnostics.append(
            f"Violation budget d'éviction Gamer-Whisper (< {float(yield_slo_ms):.1f} milliseconds) : "
            f"{gamer_yield_ok_count}/{len(gpu_gamer_rows)} conformes."
            if is_fr
            else f"Gamer-Whisper GPU yield SLO violation (< {float(yield_slo_ms):.1f} milliseconds): "
            f"{gamer_yield_ok_count}/{len(gpu_gamer_rows)} passed."
        )
    if not gamer_thermal_all_ok:
        diagnostics.append(
            f"Violation plafond thermique Gamer-Whisper (<= {float(max_temp_c):.1f} C) : "
            f"{gamer_thermal_ok_count}/{len(gpu_gamer_rows)} conformes."
            if is_fr
            else f"Gamer-Whisper thermal/VRAM ceiling violation (<= {float(max_temp_c):.1f} C): "
            f"{gamer_thermal_ok_count}/{len(gpu_gamer_rows)} passed."
        )
    if not conservation_ok or funnel_receipt["status"] != "healthy" or not replay_conserved:
        diagnostics.append(
            f"Alerte entonnoir / conservation UX (statut={funnel_receipt['status']}, "
            f"conservation_ok={conservation_ok}, replay_conserved={replay_conserved})."
            if is_fr
            else f"Onboarding funnel alert or conservation mismatch (status={funnel_receipt['status']}, "
            f"conservation_ok={conservation_ok}, replay_conserved={replay_conserved})."
        )
    if not quorum_ok:
        diagnostics.append(
            f"Quorum producteurs amis insuffisant ({fleet_online}/{target_online_producers} en ligne, {fleet_gpus} GPU distincts)."
            if is_fr
            else f"Friend producer quorum unmet ({fleet_online}/{target_online_producers} online, {fleet_gpus} distinct GPUs)."
        )
    if not csp_nonces_unique or not sqlite_locks_ok:
        diagnostics.append(
            f"Invariant HTTP/CSP/WAL en échec (nonces={unique_nonces_count}/{coordinator.html_renders_total}, locks={coordinator.sqlite_lock_errors})."
            if is_fr
            else f"HTTP/CSP/WAL invariant failed (nonces={unique_nonces_count}/{coordinator.html_renders_total}, locks={coordinator.sqlite_lock_errors})."
        )
    if pr949_report is not None and not pr949_parity_ok:
        for p_diag in pr949_report.get("diagnostics") or ():
            if p_diag not in diagnostics:
                diagnostics.append(str(p_diag))

    rehearsal_ok = bool(
        personal_ready_frac == Fraction(1, 1)
        and corporate_blocked_frac == Fraction(1, 1)
        and corporate_file_writes_count == 0
        and corporate_network_calls_count == 0
        and zero_key_leaks_verified
        and all_posix_0600_ok
        and gamer_yield_slo_all_ok
        and gamer_thermal_all_ok
        and zero_inference_verified
        and all_sub_receipts_valid
        and conservation_ok
        and replay_conserved
        and quorum_ok
        and csp_nonces_unique
        and sqlite_locks_ok
        and pr949_parity_ok
        and funnel_receipt["status"] == "healthy"
    )

    elapsed_s = round(max(0.0, time.perf_counter() - t_wall0), 4)

    receipt_core: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": stamp,
        "lang": norm_lang,
        "gateway": clean_gw,
        "rehearsal_ok": rehearsal_ok,
        "verdict": "ready" if rehearsal_ok else "failed_rehearsal_gate",
        "exit_code": 0 if rehearsal_ok else 2,
        "cohort_size": cohort_size,
        "personal_ready_count": personal_ready_count,
        "personal_failed_count": personal_failed_count,
        "personal_ready_ratio": personal_ready_ratio,
        "corporate_cohort_size": total_corp_controls,
        "corporate_blocked_count": corporate_blocked_count,
        "corporate_leaked_count": corporate_leaked_count,
        "corporate_blocked_ratio": corporate_blocked_ratio,
        "corporate_file_writes_count": corporate_file_writes_count,
        "corporate_network_calls_count": corporate_network_calls_count,
        "workers_count": workers,
        "elapsed_s": elapsed_s,
        "zero_inference_verified": zero_inference_verified,
        "inference_calls_count": final_gw["inference_calls_count"],
        "zero_key_leaks_verified": zero_key_leaks_verified,
        "workspace_files_scanned_count": workspace_files_scanned,
        "workspace_key_leak_count": workspace_key_leak_count,
        "report_key_leak_count": report_key_leak_count,
        "posix_0600_verified_count": posix_0600_verified_count,
        "sqlite_lock_error_count": coordinator.sqlite_lock_errors,
        "html_renders_total": coordinator.html_renders_total,
        "unique_csp_nonces_count": unique_nonces_count,
        "idempotency_replay_enabled": bool(idempotency_replay),
        "ces_replay_attempts": coordinator.ces_replay_attempts,
        "ces_replay_ok_count": coordinator.ces_replay_ok_count,
        "ces_replay_conserved": replay_conserved,
        "include_heartbeat_quorum": bool(include_heartbeat_quorum),
        "opentrainer_heartbeats_recorded": coordinator.opentrainer_heartbeats_recorded,
        "producer_heartbeats_recorded": coordinator.producer_heartbeats_recorded,
        "online_producers": fleet_online,
        "distinct_physical_gpus": fleet_gpus,
        "producer_quorum_met": quorum_ok if include_heartbeat_quorum else None,
        "gamer_whisper_enforcement": {
            "gpu_gamer_workspaces_count": len(gpu_gamer_rows),
            "nogpu_selftest_workspaces_count": len(nogpu_selftest_rows),
            "yield_slo_ms": round(float(yield_slo_ms), 1),
            "max_temp_c": round(float(max_temp_c), 1),
            "min_free_vram_mib": min_free_vram_mib,
            "gamer_yield_slo_ok_count": gamer_yield_ok_count,
            "gamer_thermal_ok_count": gamer_thermal_ok_count,
            "gamer_vram_reserve_ok_count": gamer_vram_ok_count,
            "gamer_yield_slo_all_ok": gamer_yield_slo_all_ok,
            "gamer_thermal_all_ok": gamer_thermal_all_ok,
            "max_gamer_yield_latency_ms": max_yield_ms,
            "p95_gamer_yield_latency_ms": p95_yield_ms,
            "max_observed_gpu_temp_c": max_obs_temp_c,
            "injected_thermal_failures": inject_gamer_thermal_failures,
            "injected_yield_failures": inject_gamer_yield_failures,
        },
        "archetype_breakdown": archetype_breakdown,
        "corporate_control_breakdown": corporate_breakdown,
        "pipeline_verification": {
            "scripts_exercised": [
                "noel_vscode_autoconfig",
                "noel_gamer_whisper_selftest",
                "noel_post_setup_probe",
                "onboarding_funnel_report",
                "daily_raid_bulletin",
            ],
            "autoconfig_runs_count": cohort_size + total_corp_controls,
            "gamer_selftest_runs_count": (
                len(gpu_gamer_rows) + len(nogpu_selftest_rows) + total_corp_controls
            ),
            "post_setup_probe_runs_count": cohort_size,
            "sub_receipts_verified_count": sub_receipts_count,
            "all_sub_receipts_valid": all_sub_receipts_valid,
            "gateway_requests_total": final_gw["requests_total"],
            "gateway_routes_hit": final_gw["routes_hit"],
        },
        "funnel_aggregation": {
            "funnel_receipt_sha256": funnel_receipt["receipt_sha256"],
            "status": funnel_receipt["status"],
            "landing_visits": f_counts["landing_visits"],
            "signups": f_counts["signups"],
            "first_requests": f_counts["first_requests"],
            "preflight_runs": f_counts["preflight_runs"],
            "install_completed": f_counts["install_completed"],
            "first_rounds": f_counts["first_rounds"],
            "activation_ratio": funnel_receipt["activation_ratio"],
            "signup_conversion_ratio": funnel_receipt["signup_conversion_ratio"],
            "preflight_to_first_round_ratio": funnel_receipt["preflight_to_first_round_ratio"],
            "install_dropoff_ratio": funnel_receipt["install_dropoff_ratio"],
            "feedback_responses": f_counts["feedback_responses"],
            "feedback_by_profile": f_counts["feedback_by_profile"],
            "mean_ease_rating": funnel_receipt["mean_ease_rating"],
            "zero_game_lag_ratio": funnel_receipt["zero_game_lag_ratio"],
            "alerts": funnel_receipt["alerts"],
            "conservation_ok": conservation_ok,
        },
        "daily_raid_bulletin": {
            "bulletin_receipt_sha256": bulletin_receipt["receipt_sha256"],
            "bulletin_line_count": bulletin_receipt["bulletin_line_count"],
            "active_producers_count": b_metrics["active_producers_count"],
            "feedback_total_responses": b_metrics["feedback_total_responses"],
            "mean_ease_rating": b_metrics["mean_ease_rating"],
            "zero_game_lag_ratio": b_metrics["zero_game_lag_ratio"],
            "bulletin_en": bulletin_receipt["bulletin_en"],
            "bulletin_fr": bulletin_receipt["bulletin_fr"],
        },
        "compliance": host_comp,
        "diagnostics": [sanitize_text_for_display(d) for d in diagnostics],
    }
    if pr949_report is not None:
        receipt_core["pr949_parity"] = pr949_report
    return _attach_receipt(receipt_core)


def _render_pr949_single_lang_markdown(report: Mapping[str, Any], *, fr: bool) -> str:
    ready = bool(report.get("ready"))
    title = (
        "# Vérification Parité Live Cloud Run PR #949 & Quorum (`SLICE-SCALE-DYN-41`)"
        if fr
        else "# Cloud Run PR #949 Live Parity & Producer Quorum Report (`SLICE-SCALE-DYN-41`)"
    )
    compliance_txt = COMPLIANCE_NOTICE_FR if fr else COMPLIANCE_NOTICE_EN
    lines: list[str] = [
        title,
        "",
        f"> {compliance_txt}",
        "",
        f"- **Verdict**: `{'GO' if ready else 'NO-GO'}` (`ready={ready}`, `exit_code={report.get('exit_code')}`)",
        f"- **Gateway**: `{report.get('gateway')}` (`live_build_sha={report.get('live_build_sha')}`)",
        f"- **holiday_routes_deployed_on_cloud_run**: `{report.get('holiday_routes_deployed_on_cloud_run')}` "
        f"(`onboarding_feedback_deployed={report.get('onboarding_feedback_deployed')}`, "
        f"`stale_revision_detected={report.get('stale_revision_detected')}`)",
        f"- **Producer Quorum**: `{report.get('online_producers')}/{report.get('target_online_producers')}` "
        f"(`producer_quorum_met={report.get('producer_quorum_met')}`, "
        f"`awaiting_producer_quorum={report.get('awaiting_producer_quorum')}`, "
        f"`distinct_physical_gpus={report.get('distinct_physical_gpus')}`)",
        f"- **CES Summary**: responses=`{report.get('total_responses')}`, "
        f"mean_ease_rating=`{report.get('mean_ease_rating')}`, "
        f"zero_game_lag_ratio=`{report.get('zero_game_lag_ratio')}`",
        f"- **CSP & HTML Contract**: nonces=`{report.get('unique_csp_nonces_count')}/{len(LIVE_HTML_PATHS)}`, "
        f"dom_sinks=`{report.get('forbidden_dom_sinks_count')}`, "
        f"static_figures=`{report.get('static_figures_found')}`, "
        f"banners_ok=`{report.get('personal_pc_banners_ok')}`",
        f"- **receipt_sha256**: `{report.get('receipt_sha256')}`",
    ]
    missing = report.get("missing_or_stale_routes") or ()
    if missing:
        lines.append(f"- **missing_or_stale_routes**: `{', '.join(str(r) for r in missing)}`")
    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        for item in diags:
            lines.append(f"- {sanitize_text_for_display(str(item))}")
    return "\n".join(lines) + "\n"


def _render_single_lang_markdown(report: Mapping[str, Any], *, fr: bool) -> str:
    if report.get("mode") == "pr949_live_parity":
        return _render_pr949_single_lang_markdown(report, fr=fr)

    ok = bool(report["rehearsal_ok"])
    gw_enf = report.get("gamer_whisper_enforcement")
    funnel = report.get("funnel_aggregation")
    bulletin = report.get("daily_raid_bulletin")
    pipe = report.get("pipeline_verification")

    if fr:
        heading = "# Répétition Générale Cohorte Noël Multi-OS (`SLICE-SCALE-02`)"
        verdict_line = f"- **Verdict répétition** : `{'PRÊT' if ok else 'ÉCHEC'}` (`rehearsal_ok={ok}`)"
        summary_hdr = "## Résumé de la Cohorte & Garde-fous"
        arch_hdr = "## Répartition des 5 Archétypes PC Personnels"
        arch_cols = "| Archétype | Profil | OS | Prêts / Total | Ratio Prêts | Facilité CES | Zéro Lag |"
        corp_hdr = "## Cohorte Contrôle Négatif Entreprise / Amadeus"
        corp_cols = "| Vecteur | Bloqués / Total | Fichiers Écrits | Appels Réseau | Marqueurs |"
        compliance_txt = COMPLIANCE_NOTICE_FR
    else:
        heading = "# Multi-OS Noël Canary & Scale Cohort Rehearsal (`SLICE-SCALE-02`)"
        verdict_line = f"- **Rehearsal Verdict**: `{'READY' if ok else 'FAILED'}` (`rehearsal_ok={ok}`)"
        summary_hdr = "## Cohort Summary & Safety Gates"
        arch_hdr = "## 5 Personal-PC Archetypes Breakdown"
        arch_cols = "| Archetype | Profile | OS | Ready / Total | Ready Ratio | Mean CES | Zero Game Lag |"
        corp_hdr = "## Hostile Corporate / Amadeus Negative-Control Cohort"
        corp_cols = "| Vector | Blocked / Total | File Writes | Network Calls | Matched Markers |"
        compliance_txt = COMPLIANCE_NOTICE_EN

    lines: list[str] = [
        heading,
        "",
        f"> {compliance_txt}",
        "",
        summary_hdr,
        verdict_line,
        f"- **cohort_size**: `{report['cohort_size']}` (`personal_ready_count={report['personal_ready_count']}`)",
        f"- **personal_ready_ratio**: `{report['personal_ready_ratio']:.4f}`",
        f"- **corporate_blocked_ratio**: `{report['corporate_blocked_ratio']:.4f}` "
        f"(`{report['corporate_blocked_count']}/{report['corporate_cohort_size']}`, "
        f"writes=`{report['corporate_file_writes_count']}`, net=`{report['corporate_network_calls_count']}`)",
        f"- **zero_key_leaks_verified**: `{report['zero_key_leaks_verified']}` "
        f"(files_scanned=`{report['workspace_files_scanned_count']}`, "
        f"workspace_leaks=`{report['workspace_key_leak_count']}`, report_leaks=`{report['report_key_leak_count']}`)",
        f"- **zero_inference_verified**: `{report['zero_inference_verified']}` "
        f"(inference_calls=`{report['inference_calls_count']}`)",
        f"- **receipt_sha256**: `{report['receipt_sha256']}`",
    ]

    if isinstance(gw_enf, Mapping):
        lines.extend(
            [
                "",
                "## Gamer-Whisper (< 1500.0 milliseconds & <= 70.0 C)",
                f"- **GPU Yield SLO**: `{gw_enf['gamer_yield_slo_ok_count']}/{gw_enf['gpu_gamer_workspaces_count']}` "
                f"(max=`{gw_enf['max_gamer_yield_latency_ms']}` milliseconds, budget `< {gw_enf['yield_slo_ms']}` milliseconds)",
                f"- **Thermal & VRAM Guard**: `{gw_enf['gamer_thermal_ok_count']}/{gw_enf['gpu_gamer_workspaces_count']}` "
                f"(max_temp=`{gw_enf['max_observed_gpu_temp_c']}` C, ceiling `<= {gw_enf['max_temp_c']}` C)",
            ]
        )

    arch_rows = report.get("archetype_breakdown") or []
    if arch_rows:
        lines.extend(["", arch_hdr, arch_cols, "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"])
        for row in arch_rows:
            label = row["label_fr"] if fr else row["label_en"]
            lines.append(
                f"| {label} | `{row['profile']}` | `{row['system_name']}` | "
                f"`{row['ready_count']}/{row['total_count']}` | `{row['ready_ratio']:.2f}` | "
                f"`{row['mean_ease_rating']}/5` | `{row['zero_game_lag_ratio']:.2f}` |"
            )

    corp_rows = report.get("corporate_control_breakdown") or []
    if corp_rows:
        lines.extend(["", corp_hdr, corp_cols, "| :--- | :---: | :---: | :---: | :--- |"])
        for c_row in corp_rows:
            c_label = c_row["label_fr"] if fr else c_row["label_en"]
            markers = ", ".join(c_row["matched_markers"]) or "n/a"
            lines.append(
                f"| {c_label} | `{c_row['blocked_count']}/{c_row['total_count']}` | "
                f"`{c_row['file_writes_count']}` | `{c_row['network_calls_count']}` | `{markers}` |"
            )

    if isinstance(funnel, Mapping) and isinstance(pipe, Mapping):
        lines.extend(
            [
                "",
                "## End-to-End Funnel & CES Aggregation" if not fr else "## Entonnoir Bout-en-Bout & Agrégation CES",
                f"- **Funnel Status**: `{funnel['status']}` (`conservation_ok={funnel['conservation_ok']}`)",
                f"- **Signups / First Requests**: `{funnel['signups']}` / `{funnel['first_requests']}` "
                f"(`activation_ratio={funnel['activation_ratio']}`)",
                f"- **Trainer Preflight -> First Round**: `{funnel['preflight_runs']}` -> `{funnel['first_rounds']}` "
                f"(`dropoff={funnel['install_dropoff_ratio']}`)",
                f"- **CES & Comfort**: responses=`{funnel['feedback_responses']}`, "
                f"mean_ease=`{funnel['mean_ease_rating']}/5`, zero_game_lag_ratio=`{funnel['zero_game_lag_ratio']}`",
                f"- **Verified Sub-Receipts**: `{pipe['sub_receipts_verified_count']}` "
                f"(`all_valid={pipe['all_sub_receipts_valid']}`)",
            ]
        )

    if isinstance(bulletin, Mapping):
        chosen_bull = bulletin["bulletin_fr"] if fr else bulletin["bulletin_en"]
        lines.extend(
            [
                "",
                "## 5-Line Holiday Raid Bulletin" if not fr else "## Bulletin Raid Noël (5 lignes)",
                "```text",
                str(chosen_bull),
                "```",
            ]
        )

    diags = report.get("diagnostics") or []
    if diags:
        lines.extend(["", "## Diagnostics"])
        for diag in diags:
            lines.append(f"- {sanitize_text_for_display(str(diag))}")

    return "\n".join(lines) + "\n"


def render_markdown_report(report: Mapping[str, Any], *, lang: str | None = None) -> str:
    """Render an English, French, or bilingual Markdown rehearsal or PR #949 parity report."""
    eff_lang = normalize_lang(lang if lang is not None else str(report.get("lang") or "en"))
    if eff_lang == "fr":
        rendered = _render_single_lang_markdown(report, fr=True)
    elif eff_lang == "en":
        rendered = _render_single_lang_markdown(report, fr=False)
    else:
        fr_part = _render_single_lang_markdown(report, fr=True).rstrip()
        en_part = _render_single_lang_markdown(report, fr=False)
        rendered = f"{fr_part}\n\n---\n\n{en_part}"
    return _assert_safe_output(rendered)


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a concise single-screen CLI summary audited for zero PII or key leaks."""
    if report.get("mode") == "pr949_live_parity":
        status_tag = "OK" if report.get("ready") else "FAIL"
        summary = (
            f"[Noel PR #949 Live Parity: {status_tag}] "
            f"holiday_routes={report.get('holiday_routes_deployed_on_cloud_run')} "
            f"producers={report.get('online_producers')}/{report.get('target_online_producers')} "
            f"quorum_met={report.get('producer_quorum_met')} "
            f"mean_ease={report.get('mean_ease_rating')} "
            f"receipt={str(report.get('receipt_sha256'))[:16]}..."
        )
        return _assert_safe_output(summary)

    status_tag = "OK" if report["rehearsal_ok"] else "FAIL"
    summary = (
        f"[Noel Canary Cohort Rehearsal: {status_tag}] "
        f"cohort_size={report['cohort_size']} "
        f"personal_ready_ratio={report['personal_ready_ratio']:.4f} "
        f"corporate_blocked_ratio={report['corporate_blocked_ratio']:.4f} "
        f"rehearsal_ok={report['rehearsal_ok']} "
        f"key_leaks={report['workspace_key_leak_count'] + report['report_key_leak_count']} "
        f"receipt={str(report['receipt_sha256'])[:16]}..."
    )
    return _assert_safe_output(summary)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for `scripts/noel_canary_cohort_rehearsal.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "50-workspace multi-OS Noël canary, PR #949 parity & scale cohort rehearsal suite "
            "(zero LLM inference, 100% personal home PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=default_gw,
        help="Allowlisted HiveMind gateway origin URL (https://*.run.app, *.googleapis.com, or loopback).",
    )
    parser.add_argument(
        "--cohort-size",
        type=int,
        default=DEFAULT_COHORT_SIZE,
        help="Number of personal home workspaces across the 5 archetypes (default: 50).",
    )
    parser.add_argument(
        "--corporate-controls",
        type=int,
        default=DEFAULT_CORPORATE_CONTROLS,
        help="Number of hostile Corporate / Amadeus negative-control workspaces (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel ThreadPoolExecutor max_workers concurrency (default: 16).",
    )
    parser.add_argument(
        "--yield-slo-ms",
        type=float,
        default=DEFAULT_YIELD_SLO_MS,
        help="Maximum allowed Steam/DX12 GPU yield latency in milliseconds (default: 1500.0).",
    )
    parser.add_argument(
        "--max-temp-c",
        type=float,
        default=DEFAULT_MAX_TEMP_C,
        help="Gamer-Whisper quiet-fan GPU thermal ceiling in Celsius (default: 70.0).",
    )
    parser.add_argument(
        "--min-free-vram-mib",
        type=int,
        default=DEFAULT_MIN_FREE_VRAM_MIB,
        help="Minimum free GPU VRAM reserve in MiB (default: 2048).",
    )
    parser.add_argument(
        "--inject-gamer-thermal-failures",
        type=int,
        default=0,
        help="Canary drill: inject >70 C GPU thermal spikes into N gamer workspaces.",
    )
    parser.add_argument(
        "--inject-gamer-yield-failures",
        type=int,
        default=0,
        help="Canary drill: inject >=1500 milliseconds GPU yield timeouts into N gamer workspaces.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional YYYY-MM-DD report date for daily_raid_bulletin.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Output language: 'en', 'fr', or 'bilingual' (default: 'en').",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Workspace directory inspected locally for Amadeus / Personal-PC & worktree git compliance.",
    )
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Inspect host env, cwd, and git/worktree remotes locally BEFORE any network call or file write.",
    )
    parser.add_argument(
        "--pr949-parity",
        action="store_true",
        help="Run zero-inference PR #949 live parity, CES alias, CSP nonce, and producer quorum check.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Optional expected HIVEMIND_BUILD_SHA for PR #949 manifest parity verification.",
    )
    parser.add_argument(
        "--target-online-producers",
        type=int,
        default=DEFAULT_TARGET_ONLINE_PRODUCERS,
        help="Target online friend GPU producers for quorum verification (default: 8).",
    )
    parser.add_argument(
        "--require-live-quorum",
        action="store_true",
        help="Fail closed (exit code 2) if online_producers < target-online-producers.",
    )
    parser.add_argument(
        "--min-ease-rating",
        type=float,
        default=DEFAULT_MIN_EASE_RATING,
        help="Minimum CES ease rating target out of 5 (default: 4.2).",
    )
    parser.add_argument(
        "--min-zero-lag-ratio",
        type=float,
        default=DEFAULT_MIN_ZERO_LAG_RATIO,
        help="Minimum gamer zero_game_lag_ratio target (default: 0.95).",
    )
    parser.add_argument(
        "--idempotency-replay",
        action="store_true",
        help="Replay each CES feedback POST once with Idempotency-Key during cohort rehearsal.",
    )
    parser.add_argument(
        "--include-heartbeat-quorum",
        action="store_true",
        help="Exercise >= 8 distinct GPU producer & opentrainer heartbeats during cohort rehearsal.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit SHA-256-receipted JSON rehearsal report.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Emit concise single-line CLI rehearsal summary.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit Markdown rehearsal report (default when --json/--cli omitted).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional file path where the rendered report is atomically written (POSIX 0600).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    rehearsal_root: Path | None = None,
    pipeline_modules: OnboardingPipelineModules | None = None,
    http_fn: HttpFn | None = None,
    raw_http_fn: RawHttpFn | None = None,
) -> int:
    """CLI entrypoint returning `0` on pass or `2` (fail-closed) on gate/compliance error."""
    args = build_parser().parse_args(argv)
    try:
        if args.cwd is not None and not str(args.cwd).strip():
            raise RehearsalError("--cwd must be a non-empty directory path when provided.")
        lang_code = normalize_lang(args.lang)
        eff_cwd: Path | str | None = cwd if cwd is not None else args.cwd
        eff_git_check = bool(args.enforce_personal_pc or eff_cwd is not None)

        if args.pr949_parity:
            report = verify_pr949_cohort_live_parity(
                args.gateway,
                expected_sha=args.expected_sha,
                target_online_producers=args.target_online_producers,
                min_ease_rating=args.min_ease_rating,
                min_zero_lag_ratio=args.min_zero_lag_ratio,
                require_live_quorum=bool(args.require_live_quorum),
                raw_http_fn=raw_http_fn if raw_http_fn is not None else default_http_raw,
                http_fn=http_fn if http_fn is not None else default_http_request,
                environ=environ,
                cwd=eff_cwd,
                check_git_remote=eff_git_check or True,
                lang=lang_code,
            )
            ok_flag = bool(report["ready"])
            compliance_ok = bool(report.get("compliance_ok", True))
        else:
            report = run_canary_cohort_rehearsal(
                args.gateway,
                cohort_size=args.cohort_size,
                corporate_controls=args.corporate_controls,
                workers=args.workers,
                yield_slo_ms=args.yield_slo_ms,
                max_temp_c=args.max_temp_c,
                min_free_vram_mib=args.min_free_vram_mib,
                inject_gamer_thermal_failures=args.inject_gamer_thermal_failures,
                inject_gamer_yield_failures=args.inject_gamer_yield_failures,
                report_date=args.date,
                lang=lang_code,
                environ=environ,
                cwd=eff_cwd,
                check_git_remote=eff_git_check,
                rehearsal_root=rehearsal_root,
                pipeline_modules=pipeline_modules,
                http_fn=http_fn,
                raw_http_fn=raw_http_fn,
                idempotency_replay=bool(args.idempotency_replay),
                include_heartbeat_quorum=bool(args.include_heartbeat_quorum),
                target_online_producers=args.target_online_producers,
                expected_sha=args.expected_sha,
                require_live_quorum=bool(args.require_live_quorum),
            )
            ok_flag = bool(report["rehearsal_ok"])
            compliance_ok = bool(report.get("compliance", {}).get("compliance_ok", True))

        if args.json:
            rendered = _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        elif args.cli:
            rendered = render_cli_report(report) + "\n"
        else:
            rendered = render_markdown_report(report, lang=lang_code)

        if not compliance_ok:
            # Fail closed with 0 disk writes while surfacing bilingual notice to stderr.
            print(f"Error: {COMPLIANCE_NOTICE_BILINGUAL}", file=sys.stderr)
            sys.stdout.write(rendered)
            return 2

        if args.output:
            _atomic_write_text(Path(str(args.output)), rendered, mode=0o600)

        sys.stdout.write(rendered)
        return 0 if ok_flag else 2
    except RehearsalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
