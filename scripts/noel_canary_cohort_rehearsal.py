#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""50-workspace multi-OS Noël canary & scale cohort rehearsal suite (`SLICE-SCALE-02`).

Orchestrates an end-to-end local rehearsal of `N=50` heterogeneous personal home workspaces in
parallel across 5 personal-PC archetypes plus a hostile Corporate / Amadeus negative-control cohort
using ONLY the Python 3.10-3.13 standard library and allowlisted HiveMind gateway hosts:

  1. 5 Personal-PC Archetypes (`--cohort-size`, default `50` = `10` per archetype):
     - `linux_dev` ("Linux Dev" — POSIX, no dedicated GPU, VS Code + Continue + Dev CES)
     - `macos_dev` ("macOS Dev" — Darwin/POSIX, no dedicated GPU, VS Code + Continue + Dev CES)
     - `win11_rtx_8gb_gamer` ("Windows 11 RTX 8GB Gamer" — RTX 4060 8 GiB, Gamer-Whisper + Trainer)
     - `win11_rtx_16gb_gamer` ("Windows 11 RTX 16GB Gamer" — RTX 4080 16 GiB, Gamer-Whisper + Trainer)
     - `nogpu_laptop_dev` ("No-GPU Laptop Dev" — POSIX laptop, honest no-GPU selftest + Dev API mode)

  2. Hostile Corporate / Amadeus Negative-Control Cohort (`--corporate-controls`, default `10`):
     - Exercises corporate environment variables (`AMADEUS_ENV`, `USERDNSDOMAIN`, `USERDOMAIN`,
       `HTTPS_PROXY` / `ZSCALER_CERT_PATH`), corporate workspace paths, and corporate `.git/config`
       remotes, verifying `100%` fail-closed blocking BEFORE any file write, hardware drill, or
       network request (`corporate_blocked_ratio == 1.0`).

  3. Real End-to-End Pipeline Execution in Isolated Temporary Home Directories:
     - `scripts/noel_vscode_autoconfig.py`
     - `scripts/noel_gamer_whisper_selftest.py`
     - `scripts/noel_post_setup_probe.py`
     - `scripts/onboarding_funnel_report.py`
     - `scripts/daily_raid_bulletin.py`
     with mocked zero-inference gateway transport (`0` calls to `/v1/chat/completions`), verifying
     `0` raw `hm-...` key leaks in workspace files or reports, `< 1500.0 ms` Steam/DX12 GPU yield,
     `<= 70.0 C` quiet-fan thermal enforcement, cryptographic SHA-256 receipts, and exact funnel
     conversion & CES rational conservation.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr", "bilingual")
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
HOBBY_DISPLAY_GIT_URL = "https://github.com/Vxtzq/CrowdGPT"

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

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DISCORD_PING_RE = re.compile(r"(?:@everyone|@here|<@[!&]?\d+>)", re.IGNORECASE)
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
    ("internal", re.compile(r"(?:^|[^a-z])internal(?:$|[^a-z])", re.IGNORECASE)),
    ("zscaler", re.compile(r"zscaler", re.IGNORECASE)),
    ("netskope", re.compile(r"netskope", re.IGNORECASE)),
    ("bluecoat", re.compile(r"bluecoat", re.IGNORECASE)),
    ("fortinet", re.compile(r"fortinet", re.IGNORECASE)),
    ("paloalto", re.compile(r"paloalto", re.IGNORECASE)),
)

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


@dataclass(frozen=True)
class OnboardingPipelineModules:
    """Loaded sibling Python modules for the 5-stage Noël onboarding pipeline."""

    autoconfig: Any
    gamer_selftest: Any
    post_setup_probe: Any
    funnel_report: Any
    raid_bulletin: Any


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


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` matches the strict HiveMind lateral-movement allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate and normalize gateway origin URL to `https://*.run.app` or loopback."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise RehearsalError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise RehearsalError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise RehearsalError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise RehearsalError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise RehearsalError("Gateway URL must be a bare origin without path, query, or fragment.")
    try:
        port = parts.port
    except ValueError as exc:
        raise RehearsalError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise RehearsalError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
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


def scan_output_safety(text: str) -> list[str]:
    """Scan text for raw `hm-...` keys, Bearer tokens, emails, home paths, non-loopback IPs, or forbidden keys."""
    if not isinstance(text, str):
        raise RehearsalError("scan_output_safety requires a string.")
    findings: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        findings.append("forbidden_key_detected")
    if _API_KEY_RE.search(text):
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


def evaluate_host_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Amadeus / Personal-PC compliance check for the rehearsal host."""
    norm_lang = normalize_lang(lang)
    is_fr = norm_lang == "fr"
    env_map = environ if environ is not None else os.environ
    upper_env = {str(k).upper(): str(v).strip() for k, v in env_map.items() if str(v).strip()}

    indicators: list[str] = []
    if any(k.startswith("AMADEUS") for k in upper_env):
        indicators.append("env:AMADEUS_ENV")
    if upper_env.get("USERDNSDOMAIN"):
        indicators.append("env:USERDNSDOMAIN")
    if upper_env.get("ZSCALER_CERT_PATH"):
        indicators.append("env:ZSCALER_CERT_PATH")

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
    summary = (
        f"Contrôle hôte conforme (PC personnel vérifié). {COMPLIANCE_NOTICE_FR}"
        if (compliance_ok and is_fr)
        else (
            f"Host compliance check passed (personal PC verified). {COMPLIANCE_NOTICE_EN}"
            if compliance_ok
            else (
                f"REFUS HÔTE ({', '.join(indicators)}) : exécutez uniquement sur un PC personnel. "
                f"{COMPLIANCE_NOTICE_FR}"
                if is_fr
                else f"HOST REFUSED ({', '.join(indicators)}): run only on a personal home PC. "
                f"{COMPLIANCE_NOTICE_EN}"
            )
        )
    )
    return {
        "compliance_ok": compliance_ok,
        "corporate_indicators": indicators,
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "financial_promise": FINANCIAL_PROMISE,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "summary": summary,
    }


def _load_script_module(module_name: str, script_path: Path) -> Any:
    if not script_path.is_file():
        raise RehearsalError(f"Required onboarding pipeline script not found: {script_path.name}")
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
    """Thread-safe zero-inference HiveMind gateway mock tracking registrations, activations, and CES."""

    def __init__(self, gateway_url: str, *, report_date: str) -> None:
        self.gateway_url = validate_gateway_url(gateway_url)
        self.report_date = report_date
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
        self._trainer_telemetry: dict[str, Any] = {
            "preflight_runs": 0,
            "install_completed": 0,
            "first_rounds": 0,
            "vram_free_mib": None,
            "vram_total_mib": None,
            "temp_c": None,
        }

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
                "object": "hivemind.onboarding_feedback_summary",
                "campaign_id": CAMPAIGN_ID,
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
            "object": "hivemind.onboarding_feedback_summary",
            "campaign_id": CAMPAIGN_ID,
            "total_responses": total,
            "mean_ease_rating": mean_ease,
            "zero_game_lag_ratio": lag_ratio,
            "by_profile": by_prof,
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
                "issued_keys": list(self.accounts_by_key.keys()),
            }

    def request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Handle a zero-inference gateway request from `autoconfig`, `selftest`, or `probe`."""
        verb = (method or "GET").strip().upper()
        parts = urlsplit(url)
        validate_gateway_url(f"{parts.scheme}://{parts.netloc}")
        path = parts.path or "/"

        if "completions" in path.lower():
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

            if verb == "GET" and path == "/health":
                return 200, {
                    "status": "ok",
                    "service": "hivemind-openrouter",
                    "campaign_id": CAMPAIGN_ID,
                    "build_sha": "noel2026canary50",
                }

            if verb == "GET" and path in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                self._nonce_seq += 1
                nonce = f"NoelCanaryNonce{self._nonce_seq:06d}"
                html = (
                    f'<!DOCTYPE html><html><head><style nonce="{nonce}">body{{}}</style></head>'
                    f'<body><main data-campaign="{CAMPAIGN_ID}"></main></body></html>'
                )
                return 200, {"_html": html, "csp_nonce": nonce}

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
                return 200, {"object": "hivemind.account.registered", **record}

            if verb == "GET" and path == "/v1/hivemind/me":
                auth = hdrs.get("authorization", "")
                token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
                acct = self.accounts_by_key.get(token)
                if acct is None:
                    return 401, {"error": "invalid_api_key"}
                self.activated_account_ids.add(acct["account_id"])
                return 200, {
                    "object": "hivemind.account",
                    "account_id": acct["account_id"],
                    "handle": acct["handle"],
                    "role": acct["role"],
                }

            if verb == "GET" and path == "/v1/models":
                return 200, {
                    "object": "list",
                    "data": [
                        {"id": "hivemind/coder"},
                        {"id": "hivemind/bee-stars"},
                        {"id": "hivemind/auto"},
                    ],
                }

            if verb == "POST" and path == "/v1/hivemind/onboarding/feedback":
                fb_in = dict(payload or {})
                prof = str(fb_in.get("profile") or "dev").strip().lower()
                rating = int(fb_in.get("ease_rating") or 5)
                zero_lag = bool(fb_in.get("zero_game_lag", True))
                plat = str(fb_in.get("platform") or "linux")
                self.feedback_items.append(
                    {
                        "profile": prof,
                        "ease_rating": rating,
                        "zero_game_lag": zero_lag,
                        "platform": plat,
                    }
                )
                summary = self._feedback_summary_unlocked()
                return 200, {**summary, "recorded": True}

            if verb == "GET" and path == "/v1/hivemind/onboarding/feedback/summary":
                return 200, self._feedback_summary_unlocked()

            if verb == "GET" and path == "/v1/hivemind/raid/status":
                fb_sum = self._feedback_summary_unlocked()
                tr = self._trainer_telemetry
                trainer_accounts = sum(
                    1 for rec in self.accounts_by_key.values() if rec["role"] == "trainer"
                )
                return 200, {
                    "object": "hivemind.raid_noel.status",
                    "campaign": CAMPAIGN_ID,
                    "status": "active",
                    "swarm": {
                        "registered_accounts": len(self.accounts_by_key),
                        "active_accounts_24h": len(self.activated_account_ids),
                        "registered_trainer_accounts": trainer_accounts,
                        "registered_trainer_nodes": int(tr["install_completed"]),
                        "active_trainers": int(tr["first_rounds"]),
                        "accepted_opentrainer_rounds": int(tr["first_rounds"]),
                    },
                    "onboarding_feedback": fb_sum,
                    "feedback_summary": fb_sum,
                    "mean_ease_rating": fb_sum["mean_ease_rating"],
                }

            if verb == "GET" and path == "/v1/hivemind/live-telemetry":
                tr = self._trainer_telemetry
                trainer_accounts = sum(
                    1 for rec in self.accounts_by_key.values() if rec["role"] == "trainer"
                )
                return 200, {
                    "object": "hivemind.live_telemetry",
                    "accounts": {
                        "total_registered_accounts": len(self.accounts_by_key),
                        "active_accounts_24h": len(self.activated_account_ids),
                    },
                    "opentrainer": {
                        "registered_trainer_accounts_count": trainer_accounts,
                        "total_registered_nodes": int(tr["install_completed"]),
                        "active_trainer_nodes": int(tr["first_rounds"]),
                        "accepted_rounds_count": int(tr["first_rounds"]),
                    },
                }

            if verb == "GET" and path == "/v1/hivemind/arena/summary":
                return 200, {
                    "object": "hivemind.blind_ab_arena.summary",
                    "total_votes": int(self._trainer_telemetry["first_rounds"]),
                }

            if verb == "GET" and path == "/v1/hivemind/t4/status":
                tr = self._trainer_telemetry
                ready_producers = int(tr["first_rounds"])
                if ready_producers <= 0:
                    return 200, {
                        "captured_utc": f"{self.report_date}T20:00:00Z",
                        "telemetry_status": "NO_TELEMETRY_RECEIVED",
                        "active_producers_count": 0,
                        "distinct_physical_gpus": 0,
                    }
                return 200, {
                    "captured_utc": f"{self.report_date}T20:00:00Z",
                    "telemetry_status": "LIVE_COHORT_REHEARSAL",
                    "operational_mode": "GAMER_WHISPER_QUIET",
                    "active_producers_count": ready_producers,
                    "distinct_physical_gpus": ready_producers,
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
                }

            return 404, {"error": f"route_not_found:{path}"}

    def http_get_json(self, url: str, timeout_s: float = 5.0) -> tuple[int, dict[str, Any]]:
        """Read-only `HttpGetFn` adapter for `onboarding_funnel_report` and `daily_raid_bulletin`."""
        _ = timeout_s
        return self.request_json("GET", url, None, None)


def build_personal_cohort_plan(cohort_size: int = DEFAULT_COHORT_SIZE) -> list[tuple[int, PersonalArchetypeSpec]]:
    """Build a deterministic N-workspace plan distributed evenly across the 5 personal-PC archetypes."""
    if (
        isinstance(cohort_size, bool)
        or not isinstance(cohort_size, int)
        or not (MIN_COHORT_SIZE <= cohort_size <= MAX_COHORT_SIZE)
    ):
        raise RehearsalError(
            f"cohort_size must be an integer between {MIN_COHORT_SIZE} and {MAX_COHORT_SIZE}."
        )
    return [
        (idx, PERSONAL_PC_ARCHETYPES[idx % len(PERSONAL_PC_ARCHETYPES)])
        for idx in range(cohort_size)
    ]


def build_corporate_control_plan(
    corporate_controls: int = DEFAULT_CORPORATE_CONTROLS,
) -> list[tuple[int, CorporateControlSpec]]:
    """Build a deterministic negative-control plan across corporate / Amadeus detection vectors."""
    if (
        isinstance(corporate_controls, bool)
        or not isinstance(corporate_controls, int)
        or not (0 <= corporate_controls <= MAX_CORPORATE_CONTROLS)
    ):
        raise RehearsalError(
            f"corporate_controls must be an integer between 0 and {MAX_CORPORATE_CONTROLS}."
        )
    return [
        (idx, CORPORATE_NEGATIVE_CONTROLS[idx % len(CORPORATE_NEGATIVE_CONTROLS)])
        for idx in range(corporate_controls)
    ]


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


def _execute_personal_workspace(
    *,
    workspace_index: int,
    spec: PersonalArchetypeSpec,
    gamer_ordinal: int | None,
    thermal_injected: bool,
    yield_injected: bool,
    base_dir: Path,
    gateway_url: str,
    mock_gateway: MockZeroInferenceGateway,
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

    landing_path = "/raid-noel" if spec.profile == "gamer" else "/noel"
    landing_code, landing_doc = mock_gateway.request_json(
        "GET",
        f"{gateway_url}{landing_path}",
        None,
        {"X-HiveMind-Rehearsal-Visit": "1"},
    )
    landing_ok = landing_code == 200 and bool(landing_doc.get("csp_nonce"))

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
                http_fn=mock_gateway.request_json,
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
            http_fn=mock_gateway.request_json,
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
                http_fn=mock_gateway.request_json,
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
            http_fn=mock_gateway.request_json,
            clock_fn=ws_clock,
        )
        selftest_report = modules.gamer_selftest.run_gamer_whisper_selftest(
            gateway=gateway_url,
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
            http_fn=mock_gateway.request_json,
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
            http_fn=mock_gateway.request_json,
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
            landing_ok
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
            landing_ok
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
    if ctrl_spec.git_remote:
        git_dir = ws_dir / ".git"
        git_dir.mkdir(parents=True, exist_ok=True)
        (git_dir / "config").write_text(
            f'[core]\n\trepositoryformatversion = 0\n[remote "origin"]\n\turl = {ctrl_spec.git_remote}\n',
            encoding="utf-8",
        )

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
    sentinel_leaked = False
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
        set(autoconfig_report.get("compliance", {}).get("matched_markers") or ())
        | set(selftest_report.get("compliance", {}).get("matched_markers") or ())
    )

    blocked_cleanly = bool(
        autoconfig_rc == 2
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
    rehearsal_root: Path | None = None,
    keep_workspaces: bool = False,
    pipeline_modules: OnboardingPipelineModules | None = None,
    workspace_post_hook: WorkspacePostHook | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run the parallel multi-OS Noël canary & scale cohort rehearsal and return a signed receipt."""
    t_wall0 = time.perf_counter()
    norm_lang = normalize_lang(lang)
    script_lang = "fr" if norm_lang == "fr" else "en"
    is_fr = norm_lang == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_date = validate_report_date(report_date)

    personal_plan = build_personal_cohort_plan(cohort_size)
    corporate_plan = build_corporate_control_plan(corporate_controls)

    if isinstance(workers, bool) or not isinstance(workers, int) or not (1 <= workers <= MAX_WORKERS):
        raise RehearsalError(f"workers must be an integer between 1 and {MAX_WORKERS}.")
    if (
        isinstance(yield_slo_ms, bool)
        or not isinstance(yield_slo_ms, (int, float))
        or not math.isfinite(float(yield_slo_ms))
        or float(yield_slo_ms) <= 0.0
    ):
        raise RehearsalError("yield_slo_ms must be a positive finite millisecond budget.")
    if (
        isinstance(max_temp_c, bool)
        or not isinstance(max_temp_c, (int, float))
        or not math.isfinite(float(max_temp_c))
        or not (30.0 <= float(max_temp_c) <= 100.0)
    ):
        raise RehearsalError("max_temp_c must be a finite temperature ceiling in [30.0, 100.0] C.")
    if (
        isinstance(min_free_vram_mib, bool)
        or not isinstance(min_free_vram_mib, int)
        or not (256 <= min_free_vram_mib <= 65536)
    ):
        raise RehearsalError("min_free_vram_mib must be an integer between 256 and 65536.")
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

    host_comp = evaluate_host_compliance(environ=environ, lang=norm_lang)
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
                "corporate_cohort_size": corporate_controls,
                "corporate_blocked_count": corporate_controls,
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
                "diagnostics": [host_comp["summary"]],
            }
        )

    modules = pipeline_modules if pipeline_modules is not None else load_onboarding_pipeline_modules()
    mock_gateway = MockZeroInferenceGateway(clean_gw, report_date=clean_date)

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

    personal_results: list[dict[str, Any]] = []
    corporate_results: list[dict[str, Any]] = []

    try:
        max_pool = min(workers, max(1, cohort_size + corporate_controls))
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
                personal_futures.append(
                    pool.submit(
                        _execute_personal_workspace,
                        workspace_index=ws_idx,
                        spec=spec,
                        gamer_ordinal=g_ord,
                        thermal_injected=t_inj,
                        yield_injected=y_inj,
                        base_dir=active_root,
                        gateway_url=clean_gw,
                        mock_gateway=mock_gateway,
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

    mock_gateway.finalize_trainer_telemetry(
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
        http_get=mock_gateway.http_get_json,
    )
    gw_snap = mock_gateway.snapshot_state()
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
    funnel_receipt = modules.funnel_report.analyze_onboarding_funnel(
        funnel_input,
        generated_at_utc=stamp,
    )
    funnel_md = modules.funnel_report.render_markdown_report(funnel_receipt, lang=norm_lang)

    # Exercise Stage 5: real daily_raid_bulletin (live gateway fetch + 5-line FR/EN/bilingual receipt)
    t4_doc, fb_doc, bull_http = modules.raid_bulletin.fetch_live_bulletin_inputs(
        clean_gw,
        http_get=mock_gateway.http_get_json,
    )
    bulletin_receipt = modules.raid_bulletin.build_bulletin_receipt(
        t4_doc,
        fb_doc,
        lang=norm_lang,
        date_override=clean_date,
        source_kind="cohort_rehearsal",
        http_statuses=bull_http,
    )

    final_gw = mock_gateway.snapshot_state()

    # Aggregate personal & corporate ratios using exact Fraction arithmetic
    personal_ready_count = sum(1 for r in personal_results if r["ready"])
    personal_failed_count = cohort_size - personal_ready_count
    personal_ready_frac = Fraction(personal_ready_count, cohort_size)
    personal_ready_ratio = round(float(personal_ready_frac), 4)

    corporate_blocked_count = sum(1 for r in corporate_results if r["blocked"])
    corporate_leaked_count = corporate_controls - corporate_blocked_count
    corporate_blocked_frac = (
        Fraction(corporate_blocked_count, corporate_controls)
        if corporate_controls > 0
        else Fraction(1, 1)
    )
    corporate_blocked_ratio = round(float(corporate_blocked_frac), 4)
    corporate_file_writes_count = sum(int(r["file_writes_count"]) for r in corporate_results)
    corporate_network_calls_count = sum(int(r["network_calls_count"]) for r in corporate_results)

    # Leak & receipt verification across workspace files, sub-reports, funnel report, and raid bulletin
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
    )
    all_sub_receipts_valid = (
        all(bool(r["sub_receipts_valid"]) for r in personal_results)
        and all(bool(r["sub_receipts_valid"]) for r in corporate_results)
        and funnel_receipt_ok
        and bulletin_receipt_ok
    )

    # Gamer-Whisper hardware SLO summaries
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

    # Exact rational conservation verification against onboarding_funnel_report & daily_raid_bulletin
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

    zero_inference_verified = final_gw["inference_calls_count"] == 0

    # Per-archetype breakdown
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

    corporate_breakdown: list[dict[str, Any]] = []
    for c_spec in CORPORATE_NEGATIVE_CONTROLS:
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
            f"Échec barrière Amadeus/entreprise : bloqués={corporate_blocked_count}/{corporate_controls}, "
            f"fichiers_écrits={corporate_file_writes_count}, appels_réseau={corporate_network_calls_count}."
            if is_fr
            else f"Corporate/Amadeus negative-control breach: blocked={corporate_blocked_count}/{corporate_controls}, "
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
            f"Violation budget d'éviction Gamer-Whisper (< {float(yield_slo_ms):.1f} ms) : "
            f"{gamer_yield_ok_count}/{len(gpu_gamer_rows)} conformes."
            if is_fr
            else f"Gamer-Whisper GPU yield SLO violation (< {float(yield_slo_ms):.1f} ms): "
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
    if not conservation_ok or funnel_receipt["status"] != "healthy":
        diagnostics.append(
            f"Alerte entonnoir / conservation UX (statut={funnel_receipt['status']}, "
            f"conservation_ok={conservation_ok})."
            if is_fr
            else f"Onboarding funnel alert or conservation mismatch (status={funnel_receipt['status']}, "
            f"conservation_ok={conservation_ok})."
        )

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
        and funnel_receipt["status"] == "healthy"
    )

    elapsed_s = round(max(0.0, time.perf_counter() - t_wall0), 4)

    return _attach_receipt(
        {
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
            "corporate_cohort_size": corporate_controls,
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
                "autoconfig_runs_count": cohort_size + corporate_controls,
                "gamer_selftest_runs_count": (
                    len(gpu_gamer_rows) + len(nogpu_selftest_rows) + corporate_controls
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
            "diagnostics": diagnostics,
        }
    )


def _render_single_lang_markdown(report: Mapping[str, Any], *, fr: bool) -> str:
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
                "## Gamer-Whisper (< 1500.0 ms & <= 70.0 C)",
                f"- **GPU Yield SLO**: `{gw_enf['gamer_yield_slo_ok_count']}/{gw_enf['gpu_gamer_workspaces_count']}` "
                f"(max=`{gw_enf['max_gamer_yield_latency_ms']}` ms, budget `< {gw_enf['yield_slo_ms']}` ms)",
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
            lines.append(f"- {diag}")

    return "\n".join(lines) + "\n"


def render_markdown_report(report: Mapping[str, Any], *, lang: str | None = None) -> str:
    """Render an English, French, or bilingual Markdown rehearsal report audited for zero leaks."""
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
            "50-workspace multi-OS Noël canary & scale cohort rehearsal suite "
            "(zero LLM inference, 100% personal home PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=default_gw,
        help="Allowlisted HiveMind gateway origin URL (https://*.run.app or loopback).",
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
        help="Canary drill: inject >=1500 ms GPU yield timeouts into N gamer workspaces.",
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
        help="Optional file path where the rendered report is also written.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    rehearsal_root: Path | None = None,
    pipeline_modules: OnboardingPipelineModules | None = None,
) -> int:
    """CLI entrypoint returning `0` when `rehearsal_ok` is True or `2` (fail-closed) otherwise."""
    args = build_parser().parse_args(argv)
    try:
        lang_code = normalize_lang(args.lang)
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
            rehearsal_root=rehearsal_root,
            pipeline_modules=pipeline_modules,
        )
        if args.json:
            rendered = _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        elif args.cli:
            rendered = render_cli_report(report) + "\n"
        else:
            rendered = render_markdown_report(report, lang=lang_code)

        if args.output:
            out_path = Path(str(args.output))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered, encoding="utf-8")

        sys.stdout.write(rendered)
        return 0 if report["rehearsal_ok"] else 2
    except RehearsalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
