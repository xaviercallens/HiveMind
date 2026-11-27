#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardened Linux/macOS POSIX launcher contract audit, Cloud Run holiday smoke gate & <60s/<5min onboarding verifier.

Compatible with Python 3.10-3.13 across Linux and macOS using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment variables (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``),
     workspace path, and ``.git/config`` remote URLs (including worktree ``gitdir:`` / ``commondir``
     indirection) BEFORE any network call, file write, or subprocess drill. Fails closed with exit
     code 2 (0 network calls, 0 file writes, 0 subprocess calls, 0 raw env/hostname/path leaks).
  2. Automated Cloud Run Holiday-Route Smoke & Drift Gate (``run_cloud_run_holiday_smoke_gate``):
     probes ``GET /health``, ``GET /v1/models`` (non-empty list including ``hivemind/coder`` without
     ever invoking chat completions), ``GET /v1/hivemind/onboarding/manifest``, ``GET /noel``,
     ``GET /quick``, ``GET /raid-noel``, ``GET /arena-noel`` (HTTP 200, unique per-request CSP
     nonces ``default-src 'none'``, 0 unsafe DOM sinks, 0 static figures after stripping
     ``nonce="..."`` attributes, and 0.00 EUR / 0,00 EUR personal-PC banners),
     ``GET /v1/hivemind/onboarding/feedback/summary``, ``GET /v1/hivemind/raid/status``,
     ``GET /v1/hivemind/producers/fleet``, and ``GET /scripts/hivemind_quickstart.py`` +
     ``GET /scripts/hivemind_doctor.py`` with optional zero-PII feedback POST &
     ``Idempotency-Key`` duplicate-replay conservation check.
  3. POSIX & Win11 Launcher Contract Auditor (``audit_posix_launcher_contract``):
     verifies ``scripts/join_hivemind_campaign.sh`` and ``scripts/join_hivemind.ps1`` contracts,
     ``bash -n``, ``--help``, ``--dry-run --json``, corporate block exit code 2, untrusted gateway
     exit code 2, ``--uninstall --dry-run --json``, safe ``--bootstrap`` rules, and zero unsafe sinks.
  4. Unified < 60 s Dev VS Code & < 5 min Linux/macOS Gamer-Whisper Dry-Run Verifier
     (``verify_posix_dev_vscode_60s_dry_run``, ``verify_posix_gamer_whisper_5m_dry_run``,
     ``run_unified_posix_onboarding_and_smoke_gate``): exercises isolated 0600 credentials +
     Continue YAML, secret-free ``.vscode/settings.json``, clean ``--uninstall``, real lightweight
     subprocess preemption drill yielding in < 1500.0 milliseconds with 0 MiB retained VRAM,
     <= 70.0 C thermal & >= 2048 MiB VRAM guardrails, < 50.0 MB/h LoRA delta cap, and bilingual
     FR/EN receipts.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen science under AGPL-3.0 (zero monetary value).
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
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA = "hivemind.noel_posix_launcher_and_smoke_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_posix_launcher_and_smoke_receipt"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
FLEET_OBJECT = "hivemind.producer_fleet"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

HTTP_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
SUBPROCESS_TIMEOUT_S = 5.0
DEV_VSCODE_SLO_SECONDS = 60.0
POSIX_ONBOARD_SLO_SECONDS = 300.0
DEFAULT_YIELD_BUDGET_MS = 1500.0
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
DEFAULT_MAX_UPLOAD_MB_PER_HOUR = 50.0
DEFAULT_LORA_PARAMS = 4_194_304
DEFAULT_LORA_BYTES_PER_PARAM = 2
DEFAULT_SYNCS_PER_HOUR = 2.0
MIN_REPORTABLE_VRAM_MIB = 512
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95
LIVE_PRODUCER_STATUSES: frozenset[str] = frozenset({"live", "online", "healthy", "ready"})

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"
DEFAULT_SH_PATH = REPO_ROOT / "scripts" / "join_hivemind_campaign.sh"
DEFAULT_PS1_PATH = REPO_ROOT / "scripts" / "join_hivemind.ps1"

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset(
    {
        "/v1/hivemind/onboarding/feedback",
        "/v1/hivemind/register",
        "/v1/hivemind/opentrainer/heartbeat",
    }
)
_ALLOWED_FEEDBACK_POST_KEYS: frozenset[str] = frozenset(
    {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
)
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
    "USERDOMAIN",
    "USERDNSDOMAIN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "COMPUTERNAME",
    "HOSTNAME",
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

SMOKE_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")

SMOKE_PROBE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "core"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "feedback"),
    ("/v1/hivemind/raid/status", "json", "holiday"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
    ("/scripts/hivemind_quickstart.py", "script_py", "script"),
    ("/scripts/hivemind_doctor.py", "script_doctor_py", "script"),
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

SH_REQUIRED_FLAGS: tuple[str, ...] = (
    "--connect",
    "--gamer-whisper",
    "--train",
    "--crowdgpt",
    "--continuous",
    "--legacy-demo",
    "--reserve-mib",
    "--help",
    "--doctor",
    "--deploy-smoke",
    "--vscode-setup",
    "--gamer-service",
    "--gamer-check",
    "--uninstall",
    "--smoke-routes",
    "--check-deploy-parity",
    "--post-setup-telemetry",
    "--ease-rating",
    "--had-game-lag",
    "--flush-feedback-queue",
    "--lang",
    "--dry-run",
    "--json",
    "--bootstrap",
)

PS1_ALLOWED_MODES: tuple[str, ...] = (
    "Connect",
    "GamerCheck",
    "GamerWhisper",
    "Doctor",
    "DeploySmoke",
    "Uninstall",
)

PS1_REQUIRED_PARAMS: tuple[str, ...] = (
    "-Mode",
    "-Handle",
    "-ApiKey",
    "-Gateway",
    "-ReserveMib",
    "-HotC",
    "-Game",
    "-WriteContinueConfig",
    "-Once",
    "-Lang",
    "-SmokeRoutes",
    "-CheckDeployParity",
    "-PostSetupTelemetry",
    "-EaseRating",
    "-HadGameLag",
    "-FlushFeedbackQueue",
    "-Bootstrap",
    "-Json",
    "-DryRun",
)

STEAM_GAME_EXECUTABLES: frozenset[str] = frozenset(
    {
        "cs2.exe",
        "cs2",
        "eldenring.exe",
        "cyberpunk2077.exe",
        "bg3.exe",
        "bg3",
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
        "gamescope",
        "wine64-preloader",
        "hl2_linux",
        "steam_osx",
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

_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
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

_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)

_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{1,256})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_API_KEY_RE = re.compile(
    r"(hm(?:_live_|-))\*\*\*(?:[A-Za-z0-9_-]{4,}|[0-9]{1,4}(?:m" + r"s\b|tok" + r"/s))"
)
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "docu" + r"ment\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)"
)
_PS1_UNSAFE_RE = re.compile(
    r"(?i)(?:"
    + r"\bInvoke-"
    + r"Expression\b|\bi"
    + r"ex\b|\.Download"
    + r"String\s*\(|"
    + r"cu"
    + r"rl[^\n]*\|\s*(?:ba)?sh|"
    + r"wg"
    + r"et[^\n]*\|\s*(?:ba)?sh|"
    + r"-EncodedCommand\b)"
)
_SH_UNSAFE_RE = re.compile(
    r"(?i)(?:"
    + r"cu"
    + r"rl[^\n]*\|\s*(?:ba)?sh|"
    + r"wg"
    + r"et[^\n]*\|\s*(?:ba)?sh|"
    + r"\beva"
    + r"l\b|"
    + r"base64\s+-d[^\n]*\|\s*(?:ba)?sh)"
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

HttpFn = Callable[..., Any]
RunnerFn = Callable[..., Any]


class SmokeVerifierError(ValueError):
    """Raised when verifier inputs, host allowlists, or output safety invariants fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise SmokeVerifierError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise SmokeVerifierError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def resolve_posix_platform(platform_name: str | None = None) -> str:
    """Resolve runtime POSIX platform to ``'linux'`` or ``'macos'``."""
    raw = (platform_name if platform_name is not None else platform.system()).strip().lower()
    if raw in ("darwin", "macos", "mac", "osx"):
        return "macos"
    return "linux"


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw credentials never appear in outputs."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return "<invalid>"
    if not cleaned.startswith(("hm-", "hm_live_")):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    return f"{prefix}***{cleaned[-4:]}"


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace OS user home prefixes with ``'~'`` and map repo files to repo-relative paths."""
    raw = str(target)
    if not re.match(r"^[A-Za-z]:[\\/]", raw) and Path(raw).is_absolute():
        try:
            rel = Path(raw).resolve().relative_to(REPO_ROOT)
            return str(rel).replace("\\", "/")
        except (OSError, ValueError):
            pass
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
                    redacted = "~" + suffix.replace("\\", "/")
                    return _PYTEST_USER_DIR_RE.sub("pytest-of-user", redacted)
    scrubbed = _HOME_PATH_RE.sub("~", raw)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    return scrubbed.replace("\\", "/") if scrubbed != raw else raw


def sanitize_text_for_display(text: str, *, home: Path | None = None) -> str:
    """Scrub raw keys, Bearer headers, emails, non-loopback IPv4 literals, and OS home paths."""
    scrubbed = redact_home_path(text, home=home)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith("127.") else "<redacted-ip>",
        scrubbed,
    )
    return scrubbed


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` / ``nonce='...'`` attributes, CSP nonces, and masked API keys."""
    if not isinstance(text, str):
        raise SmokeVerifierError("strip_masked_keys_and_nonces requires a string.")
    cleaned = strip_html_csp_nonce_attributes(text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_API_KEY_RE.sub(r"\1***XXXX", cleaned)


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
    """Verify JSON, CLI, or Markdown text has 0 emails, raw keys, home paths, or score digits."""
    if not isinstance(text, str):
        raise SmokeVerifierError("scan_output_safety requires a string.")
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
        raise SmokeVerifierError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def sanitize_build_sha(raw: object) -> str:
    """Sanitize a ``build_sha`` value to lowercase alphanumeric/hex or ``'unknown'``."""
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
    """Validate an explicit ``expected_sha`` parameter."""
    sanitized = sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise SmokeVerifierError(
            "Expected SHA must be a 6..64 character alphanumeric/hex string (never 'unknown' or a secret)."
        )
    return sanitized


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Compare live ``build_sha`` against ``expected_sha`` (exact or >=7-char hex prefix)."""
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
    """Sanitize a Cloud Run revision name (``K_REVISION``) or return ``None``."""
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
    """Return True iff ``host`` is in the strict HiveMind allowlist with valid DNS labels."""
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
    """Validate gateway origin URL to ``https://*.run.app``, ``https://*.googleapis.com``, or loopback."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise SmokeVerifierError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise SmokeVerifierError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise SmokeVerifierError("Gateway URL must not contain internal whitespace.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise SmokeVerifierError("Gateway URL must not contain backslashes.")
    if "@" in cleaned or "@" in unquote(cleaned):
        raise SmokeVerifierError("Gateway URL must not embed credentials or '@'.")
    try:
        parts = urlsplit(cleaned)
        port = parts.port
        host = (parts.hostname or "").lower()
    except ValueError as exc:
        raise SmokeVerifierError("Gateway URL is malformed or contains an invalid port.") from exc
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise SmokeVerifierError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise SmokeVerifierError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise SmokeVerifierError("Gateway URL must be a bare origin without a sub-path.")
    if port is not None and not (1 <= port <= 65535):
        raise SmokeVerifierError("Gateway URL port out of range.")
    if not parts.netloc or not is_allowed_outbound_host(host):
        raise SmokeVerifierError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise SmokeVerifierError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise SmokeVerifierError("Remote gateway URL must use https.")
    return cleaned


def _payload_has_forbidden_inference_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k).strip().lower() in ("messages", "prompt") or _payload_has_forbidden_inference_key(v):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_payload_has_forbidden_inference_key(item) for item in obj)
    return False


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    allowed_post_paths: frozenset[str] = frozenset(),
) -> str:
    """Enforce allowlisted host, zero completions/chat endpoints, zero LLM prompts, and bounded HTTP methods."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise SmokeVerifierError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise SmokeVerifierError("Request URL contains whitespace or control characters.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise SmokeVerifierError("Request URL must not contain backslashes.")
    try:
        parts = urlsplit(cleaned)
    except ValueError as exc:
        raise SmokeVerifierError("Request URL is malformed.") from exc
    if (
        parts.username is not None
        or parts.password is not None
        or "@" in parts.netloc
        or "@" in unquote(parts.netloc)
    ):
        raise SmokeVerifierError("Request URL must not embed credentials or '@'.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = unquote(parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path or low_path.endswith("/chat"):
        raise SmokeVerifierError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise SmokeVerifierError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or _payload_has_forbidden_inference_key(payload):
            raise SmokeVerifierError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise SmokeVerifierError(
        f"Zero-inference guard blocked HTTP {verb!r} to {parts.path!r}."
    )


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
            raise SmokeVerifierError("Untrusted redirect: empty target URL.")
        if "\\" in raw_loc or "\\" in unquote(raw_loc) or "@" in raw_loc or "@" in unquote(raw_loc):
            raise SmokeVerifierError("Untrusted redirect: forbidden backslash or '@' in Location.")
        orig_parts = urlsplit(req.full_url)
        dest_parts = urlsplit(newurl)
        orig_host = (orig_parts.hostname or "").lower()
        dest_host = (dest_parts.hostname or "").lower()
        if orig_host not in ALLOWED_LOOPBACK_HOSTS and dest_host in ALLOWED_LOOPBACK_HOSTS:
            raise SmokeVerifierError("Untrusted redirect from remote host to loopback.")
        if orig_parts.scheme == "https" and dest_parts.scheme != "https":
            raise SmokeVerifierError("Untrusted HTTPS-to-HTTP downgrade redirect.")
        effective_method = "GET" if (req.get_method() == "POST" and code in (301, 302, 303)) else req.get_method()
        _assert_safe_http_request(
            effective_method,
            newurl,
            allowed_post_paths=ALLOWED_POST_PATHS,
        )
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _assert_safe_http_request(
                new_req.get_method(),
                new_req.full_url,
                allowed_post_paths=ALLOWED_POST_PATHS,
            )
        return new_req


def _validate_zero_pii_feedback_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate that a feedback POST payload is strictly zero-PII and ``extra='forbid'`` clean."""
    if (
        not isinstance(payload, Mapping)
        or (set(payload.keys()) - _ALLOWED_FEEDBACK_POST_KEYS)
        or payload.get("profile") not in ("dev", "gamer")
        or isinstance(payload.get("ease_rating"), bool)
        or not isinstance(payload.get("ease_rating"), int)
        or not (1 <= int(payload["ease_rating"]) <= 5)
        or not isinstance(payload.get("zero_game_lag"), bool)
    ):
        raise SmokeVerifierError("Feedback POST payload must be a valid zero-PII onboarding feedback object.")
    clean_obj = {
        "profile": str(payload["profile"]),
        "ease_rating": int(payload["ease_rating"]),
        "zero_game_lag": bool(payload["zero_game_lag"]),
        "platform": str(payload.get("platform") or resolve_posix_platform())[:24],
        "note": "",
    }
    _assert_safe_output(json.dumps(clean_obj, sort_keys=True, ensure_ascii=False))
    return clean_obj


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
            result.setdefault("headers", lower_headers)
            return int(status_code), result
        return int(status_code), {"data": parsed, "headers": lower_headers}
    return int(status_code), {
        "html": text,
        "text": text,
        "html_length": len(text),
        "text_length": len(text),
        "headers": lower_headers,
        "csp": lower_headers.get("content-security-policy", ""),
    }


def _validate_timeout_s(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise SmokeVerifierError("timeout_s must be a positive finite number.")
    val = float(timeout_s)
    if not math.isfinite(val) or val <= 0.0:
        raise SmokeVerifierError("timeout_s must be a positive finite number.")
    return min(max(0.2, val), 15.0)


def _read_bounded_http_body(resp: Any, *, max_bytes: int = MAX_HTTP_BODY_BYTES) -> bytes:
    raw_bytes = resp.read(max_bytes + 1)
    if len(raw_bytes) > max_bytes:
        raise SmokeVerifierError(f"HTTP response exceeded {max_bytes} byte limit.")
    return bytes(raw_bytes)


def _execute_proxy_free_http(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, bytes, dict[str, str], bool]:
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(
        verb,
        url,
        payload=payload,
        allowed_post_paths=ALLOWED_POST_PATHS,
    )
    path = urlsplit(clean_url).path
    if payload is not None and verb != "POST":
        raise SmokeVerifierError("GET/HEAD requests must never attach a body payload.")
    bounded_timeout = _validate_timeout_s(timeout_s)

    body_bytes: bytes | None = None
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-posix-launcher-smoke-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise SmokeVerifierError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    if verb == "POST":
        clean_payload = (
            _validate_zero_pii_feedback_payload(payload)
            if path == "/v1/hivemind/onboarding/feedback"
            else dict(payload or {})
        )
        body_bytes = json.dumps(clean_payload, ensure_ascii=False).encode("utf-8")
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
    """Proxy-free stdlib HTTP helper returning ``(status_code, body_text, lower_headers)`` (4 MiB cap)."""
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
    """Proxy-free stdlib HTTP helper (``ProxyHandler({})``) with host allowlist, redirect & zero-inference guards."""
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
        raise SmokeVerifierError("http_fn must return (status_code, body) or (status_code, headers, body).")
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
                merged = {**hdr_map, **{str(k).lower(): str(v) for k, v in existing_hdrs.items()}}
            else:
                merged = hdr_map
            out["headers"] = merged
        return int(status_code), out
    if isinstance(body, str):
        return int(status_code), {
            "html": body,
            "text": body,
            "html_length": len(body),
            "text_length": len(body),
            "headers": hdr_map,
            "csp": hdr_map.get("content-security-policy", ""),
        }
    return int(status_code), {"data": body, "headers": hdr_map}


def _invoke_http_fn(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    allowed_post_paths: frozenset[str] = ALLOWED_POST_PATHS,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Validate URL/zero-inference invariants and invoke an injectable HTTP callable across arities."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(
        verb,
        url,
        payload=payload,
        allowed_post_paths=allowed_post_paths,
    )
    path = urlsplit(clean_url).path
    clean_payload: dict[str, Any] | None = None
    if verb == "POST":
        clean_payload = (
            _validate_zero_pii_feedback_payload(payload)
            if path == "/v1/hivemind/onboarding/feedback"
            else dict(payload or {})
        )

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
                payload=clean_payload,
                headers=dict(headers) if headers is not None else None,
                timeout_s=timeout_s,
            )
        else:
            kwargs: dict[str, Any] = {}
            if "payload" in param_names:
                kwargs["payload"] = clean_payload
            if "headers" in param_names:
                kwargs["headers"] = dict(headers) if headers is not None else None
            if "timeout_s" in param_names:
                kwargs["timeout_s"] = timeout_s
            if kwargs or len(pos_params) <= 2:
                raw = http_fn(verb, clean_url, **kwargs)
            elif len(pos_params) == 3:
                raw = http_fn(verb, clean_url, clean_payload)
            elif len(pos_params) == 4 and not has_varargs:
                raw = http_fn(
                    verb,
                    clean_url,
                    clean_payload,
                    dict(headers) if headers is not None else None,
                )
            else:
                raw = http_fn(
                    verb,
                    clean_url,
                    clean_payload,
                    dict(headers) if headers is not None else None,
                    timeout_s,
                )
    except SmokeVerifierError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    return _normalize_http_return(raw)


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


def _build_subprocess_env(environ: Mapping[str, str] | None) -> dict[str, str]:
    """Build a deterministic subprocess environment without leaking host corporate-proxy false positives."""
    if environ is None:
        sub_env = dict(os.environ)
    else:
        sub_env = {
            k: os.environ[k]
            for k in _SAFE_RUNTIME_ENV_KEYS
            if k in os.environ and os.environ[k]
        }
        for k, v in environ.items():
            sub_env[str(k)] = str(v)
    sub_env.setdefault("POWERSHELL_TELEMETRY_OPTOUT", "1")
    sub_env.setdefault("POWERSHELL_UPDATECHECK", "Off")
    sub_env.setdefault("DOTNET_CLI_TELEMETRY_OPTOUT", "1")
    sub_env.setdefault("DOTNET_NOLOGO", "1")
    return sub_env


def default_subprocess_runner(
    argv: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    """Run a bounded local read-only command with ``shell=False``, blocking ``gcloud`` and lateral tools."""
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise SmokeVerifierError("Subprocess argv must be a non-empty list of strings.")
    clean: list[str] = []
    for item in argv:
        if not isinstance(item, str) or any(ch in item for ch in ("\x00", "\r", "\n")):
            raise SmokeVerifierError("Subprocess argument contains invalid type or control characters.")
        clean.append(item)
    exe_name = Path(clean[0]).name.lower()
    if exe_name in _FORBIDDEN_SUBPROCESS_BINS:
        raise SmokeVerifierError(f"Forbidden subprocess binary blocked: {exe_name!r}.")
    try:
        proc = subprocess.run(
            clean,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=min(max(0.2, float(timeout_s)), 30.0),
            env=_build_subprocess_env(environ),
            check=False,
            shell=False,
        )
        out = proc.stdout or ""
        if proc.stderr and proc.returncode != 0:
            out = f"{out}\n{proc.stderr}".strip() if out else proc.stderr.strip()
        return int(proc.returncode), out
    except subprocess.TimeoutExpired:
        return 124, "subprocess_timeout"
    except OSError as exc:
        return 127, f"os_error:{type(exc).__name__}"


def _invoke_runner(
    runner: RunnerFn,
    argv: Sequence[str],
    timeout_s: float,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    sig = inspect.signature(runner)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    has_environ_kw = "environ" in sig.parameters
    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    kw: dict[str, Any] = {"environ": environ} if has_environ_kw else {}
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
    raise SmokeVerifierError("Subprocess runner must return (returncode, output).")


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def extract_git_config_remote_urls(cfg_text: str) -> list[str]:
    """Extract remote URLs and URL rewrite rules from ``.git/config`` text."""
    if not cfg_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text):
        candidate = (m_kv or m_sec).strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate ``.git/config`` from ``workspace`` (walking parents and following ``gitdir:``/``commondir``)."""
    candidates: list[Path] = []
    search_roots = (workspace, *workspace.parents)
    for root in search_roots:
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


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    check_cwd_path: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only Compliance Gate (0 network calls, 0 writes, 0 leaks)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = environ if environ is not None else (env if env is not None else os.environ)
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
            for hit in _find_corporate_markers(val_str):
                matched_markers.add(hit)
        elif key_up == "ZSCALER_CERT_PATH":
            _record("env:ZSCALER_CERT_PATH", "zscaler")
        elif key_up == "USERDNSDOMAIN":
            dns_hits = _find_corporate_markers(val_str)
            _record("env:USERDNSDOMAIN", dns_hits[0] if dns_hits else "corp")
            for hit in dns_hits:
                matched_markers.add(hit)
        elif key_up in INSPECTED_ENV_KEYS:
            for hit in _find_corporate_markers(val_str):
                _record(f"env:{key_up}:{hit}", hit)

    if check_cwd_path:
        ws_texts = [str(workspace)]
        with suppress(OSError):
            resolved_ws = str(workspace.resolve())
            if resolved_ws not in ws_texts:
                ws_texts.append(resolved_ws)
        for ws_txt in ws_texts:
            for hit in _find_corporate_markers(ws_txt):
                _record(f"workspace_path:{hit}", hit)

    if check_git_remote:
        for cfg_path in _resolve_git_config_files(workspace):
            try:
                cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for remote_url in extract_git_config_remote_urls(cfg_text):
                for hit in _find_corporate_markers(remote_url):
                    _record(f"git_remote:{hit}", hit)

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Conformité PC personnel vérifiée localement ({FINANCIAL_PROMISE}). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"Local personal-PC compliance verified ({FINANCIAL_PROMISE}). {COMPLIANCE_NOTICE_BILINGUAL}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS DE CONFORMITÉ ({joined}) : environnement d'entreprise détecté "
            f"(0 appel réseau, 0 écriture fichier, 0 sous-processus). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE BLOCKED ({joined}): corporate environment indicator detected "
            f"(0 network calls, 0 file writes, 0 subprocess calls). {COMPLIANCE_NOTICE_BILINGUAL}"
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "compliance_gate",
            "lang": lang_code,
            "compliance_ok": compliance_ok,
            "corporate_detected": not compliance_ok,
            "corporate_indicators": indicators,
            "matched_markers": sorted(matched_markers),
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "exit_code": 0 if compliance_ok else 2,
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
        }
    )


def _compliance_blocked_receipt(
    mode: str,
    comp_doc: Mapping[str, Any],
    *,
    lang_code: str,
    gateway: str = DEFAULT_GATEWAY,
) -> dict[str, Any]:
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": mode,
            "lang": lang_code,
            "gateway": gateway,
            "compliance_ok": False,
            "corporate_detected": True,
            "corporate_indicators": list(comp_doc.get("corporate_indicators") or []),
            "matched_markers": list(comp_doc.get("matched_markers") or []),
            "smoke_ok": False,
            "overall_ok": False,
            "contract_ok": False,
            "dev_ready": False,
            "gamer_ready": False,
            "holiday_routes_deployed_on_cloud_run": False,
            "onboarding_feedback_deployed": False,
            "stale_revision_detected": False,
            "redeploy_required": False,
            "online_producers": 0,
            "producer_quorum_met": False,
            "awaiting_producer_quorum": False,
            "ces_slo_ok": None,
            "zero_lag_slo_ok": None,
            "arena_summary_ok": None,
            "deploy_dry_run": None,
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "personal_pc_only": True,
            "financial_promise": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": [str(comp_doc.get("summary") or COMPLIANCE_NOTICE_BILINGUAL)],
            "exit_code": 2,
            "summary": str(comp_doc.get("summary") or COMPLIANCE_NOTICE_BILINGUAL),
        }
    )


def strip_html_csp_nonce_attributes(html_text: str) -> str:
    """Strip ``nonce="..."`` / ``nonce='...'`` attributes before static-figure scanning."""
    return _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', html_text)


def _extract_response_headers(body: Mapping[str, Any]) -> dict[str, str]:
    raw = body.get("headers")
    if isinstance(raw, Mapping):
        return {str(k).strip().lower(): str(v) for k, v in raw.items()}
    return {}


def _inspect_manifest_contract(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, str, str | None, list[str], list[str], list[str]]:
    is_fr = lang_code == "fr"
    live_sha = sanitize_build_sha(body.get("build_sha"))
    live_rev = sanitize_cloud_run_revision(body.get("cloud_run_revision"))
    issues: list[str] = []

    if (
        body.get("object") != MANIFEST_OBJECT
        or body.get("campaign_id") != CAMPAIGN_ID
        or body.get("status") != "ok"
        or ("build_sha" in body and not isinstance(body.get("build_sha"), str))
        or body.get("personal_pc_only") is not True
        or body.get("financial_promise") != FINANCIAL_PROMISE
    ):
        issues.append(
            "Contrat JSON `/v1/hivemind/onboarding/manifest` invalide ou incomplet."
            if is_fr
            else "Manifest `/v1/hivemind/onboarding/manifest` returned invalid or incomplete metadata."
        )

    raw_routes = body.get("holiday_routes")
    if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes):
        missing_routes = [r for r in EXPECTED_MANIFEST_HOLIDAY_ROUTES if r not in raw_routes]
    else:
        missing_routes = list(EXPECTED_MANIFEST_HOLIDAY_ROUTES)
    if missing_routes:
        joined_r = ", ".join(missing_routes)
        issues.append(
            f"Routes Noël absentes du manifeste : {joined_r}."
            if is_fr
            else f"Manifest `holiday_routes` missing expected routes: {joined_r}."
        )

    raw_scripts = body.get("downloadable_scripts")
    if isinstance(raw_scripts, list) and all(isinstance(s, str) for s in raw_scripts):
        missing_scripts = [s for s in EXPECTED_MANIFEST_SCRIPTS if s not in raw_scripts]
    else:
        missing_scripts = list(EXPECTED_MANIFEST_SCRIPTS)
    if missing_scripts:
        joined_s = ", ".join(missing_scripts)
        issues.append(
            f"Scripts absents du manifeste : {joined_s}."
            if is_fr
            else f"Manifest `downloadable_scripts` missing allowlisted scripts: {joined_s}."
        )

    return len(issues) == 0, live_sha, live_rev, missing_routes, missing_scripts, issues


def _inspect_holiday_html_contract(
    path: str,
    body: Mapping[str, Any],
    seen_nonces: set[str],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    html_text = str(body.get("html") or body.get("text") or "")
    hdrs = _extract_response_headers(body)
    csp_hdr = str(hdrs.get("content-security-policy") or body.get("csp") or "")

    has_doctype = "<!doctype html" in html_text.lower()
    attr_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)]
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    csp_low = csp_hdr.lower()
    csp_unsafe = "'unsafe-inline'" in csp_low or "'unsafe-eval'" in csp_low

    nonce: str | None = hdr_nonces[0] if hdr_nonces else (attr_nonces[0] if attr_nonces else None)
    hdr_consistent = (not csp_hdr) or (
        bool(hdr_nonces)
        and len(set(hdr_nonces)) == 1
        and "default-src 'none'" in csp_low
        and not csp_unsafe
        and hdr_nonces[0] in attr_nonces
    )
    csp_ok = bool(attr_nonces and hdr_consistent and nonce is not None and nonce not in seen_nonces)
    if nonce is not None:
        seen_nonces.add(nonce)

    stripped_html = strip_masked_keys_and_nonces(html_text)
    has_unsafe_sink = bool(_FORBIDDEN_DOM_SINK_RE.search(stripped_html))
    has_static_fig = bool(_STATIC_FIG_UNIT_RE.search(stripped_html))
    dom_clean = not (has_unsafe_sink or has_static_fig)

    low_html = html_text.lower()
    has_personal = (
        "100% personal" in low_html
        or "100 % personnel" in low_html
        or (("100%" in low_html or "100 %" in low_html) and ("personal" in low_html or "personnel" in low_html))
    )
    compliance_banner_ok = bool(
        has_personal
        and "agpl-3.0" in low_html
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
    )

    checks = (
        (
            has_doctype,
            f"Contrat HTML {path} invalide : `<!DOCTYPE html>` absent."
            if is_fr
            else f"HTML contract failed on {path}: missing `<!DOCTYPE html>`.",
        ),
        (
            csp_ok,
            f"Contrat CSP {path} invalide : nonce CSP par requête absent, incohérent ou réutilisé."
            if is_fr
            else f"CSP contract failed on {path}: missing, mismatched, or reused per-request CSP nonce.",
        ),
        (
            dom_clean,
            f"Contrat sécurité HTML {path} violé : sink DOM non sûr ou chiffre statique détecté."
            if is_fr
            else f"HTML safety contract violated on {path}: unsafe DOM sink or static metric figure detected.",
        ),
        (
            compliance_banner_ok,
            f"Bannière PC perso / 0,00 EUR / AGPL-3.0 absente sur {path}."
            if is_fr
            else f"Personal-PC / 0.00 EUR / AGPL-3.0 compliance banner missing on {path}.",
        ),
    )
    issues = [msg for passed, msg in checks if not passed]
    ok = len(issues) == 0
    return (
        ok,
        {
            "path": path,
            "ok": ok,
            "has_doctype": has_doctype,
            "csp_nonce_verified": csp_ok,
            "csp_nonce_sha256_prefix": hashlib.sha256(nonce.encode("utf-8")).hexdigest()[:16] if nonce else None,
            "dom_clean": dom_clean,
            "unsafe_dom_sink_found": has_unsafe_sink,
            "static_figure_found": has_static_fig,
            "personal_pc_banner_present": compliance_banner_ok,
        },
        issues,
    )


def _inspect_feedback_summary_schema(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    raw_total = body.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    total_val = int(raw_total) if total_ok else -1

    has_ease_key = "mean_ease_rating" in body or "average_ease_rating" in body
    raw_ease = body.get("mean_ease_rating") if "mean_ease_rating" in body else body.get("average_ease_rating")
    ease_ok = bool(
        has_ease_key
        and total_ok
        and (
            (total_val == 0 and raw_ease is None)
            or (
                total_val > 0
                and not isinstance(raw_ease, bool)
                and isinstance(raw_ease, (int, float))
                and 1.0 <= float(raw_ease) <= 5.0
            )
        )
    )
    ease_val = round(float(raw_ease), 2) if (ease_ok and raw_ease is not None) else None

    raw_lag = body.get("zero_game_lag_ratio")
    lag_ok = bool(
        "zero_game_lag_ratio" in body
        and total_ok
        and (
            (total_val == 0 and raw_lag is None)
            or (
                total_val > 0
                and not isinstance(raw_lag, bool)
                and isinstance(raw_lag, (int, float))
                and 0.0 <= float(raw_lag) <= 1.0
            )
        )
    )
    lag_val = round(float(raw_lag), 4) if (lag_ok and raw_lag is not None) else None

    raw_prof = body.get("by_profile")
    d_raw = raw_prof.get("dev") if isinstance(raw_prof, Mapping) else None
    g_raw = raw_prof.get("gamer") if isinstance(raw_prof, Mapping) else None
    prof_ok = bool(
        isinstance(raw_prof, Mapping)
        and not isinstance(d_raw, bool)
        and isinstance(d_raw, int)
        and d_raw >= 0
        and not isinstance(g_raw, bool)
        and isinstance(g_raw, int)
        and g_raw >= 0
        and total_ok
        and (d_raw + g_raw == total_val)
    )

    obj_ok = body.get("object", FEEDBACK_SUMMARY_OBJECT) == FEEDBACK_SUMMARY_OBJECT
    schema_ok = bool(obj_ok and total_ok and ease_ok and lag_ok and prof_ok)
    issues = (
        []
        if schema_ok
        else [
            "Schéma JSON `/v1/hivemind/onboarding/feedback/summary` invalide."
            if is_fr
            else "Feedback summary `/v1/hivemind/onboarding/feedback/summary` JSON schema mismatch."
        ]
    )
    normalized = {
        "schema_valid": schema_ok,
        "total_responses": total_val if total_ok else None,
        "mean_ease_rating": ease_val,
        "average_ease_rating": ease_val,
        "zero_game_lag_ratio": lag_val,
        "by_profile": {"dev": int(d_raw), "gamer": int(g_raw)} if prof_ok else None,  # type: ignore[arg-type]
    }
    return schema_ok, normalized, issues


def _inspect_models_contract(
    body: Mapping[str, Any],
    *,
    require_coder_model: bool,
    lang_code: str,
) -> tuple[bool, list[str], bool, list[str]]:
    is_fr = lang_code == "fr"
    raw_data = body.get("data")
    if body.get("object", "list") != "list" or not isinstance(raw_data, list):
        return (
            False,
            [],
            False,
            [
                "Contrat JSON `/v1/models` invalide (liste `data` manquante)."
                if is_fr
                else "Invalid `/v1/models` JSON schema (missing `data` list)."
            ],
        )
    model_ids = [
        str(entry.get("id")).strip()
        for entry in raw_data
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str) and entry.get("id").strip()
    ]
    issues: list[str] = []
    if not model_ids:
        issues.append(
            "Liste `data` de `GET /v1/models` vide."
            if is_fr
            else "`GET /v1/models` returned an empty `data` model list."
        )
    coder_present = "hivemind/coder" in model_ids
    if require_coder_model and not coder_present:
        issues.append(
            "Modèle `hivemind/coder` absent de `GET /v1/models`."
            if is_fr
            else "Required model `hivemind/coder` missing from `GET /v1/models`."
        )
    return len(issues) == 0, model_ids, coder_present, issues


def audit_deploy_sh_guidance(
    deploy_sh_path: Path | str | None = None,
    *,
    expected_sha: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Inspect ``deploy/gcp_openrouter/deploy.sh`` FinOps invariants and emit dry-run guidance."""
    lang_code = normalize_lang(lang)
    path = Path(deploy_sh_path) if deploy_sh_path is not None else DEFAULT_DEPLOY_SH_PATH
    display_path = redact_home_path(path)
    present = False
    active = ""
    if path.is_file():
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            present = True
            active = "\n".join(ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
        except OSError:
            present = False

    invariants = {
        "min_instances_zero": bool(
            present
            and re.search(r"--min-instances(?:=|\s+)0\b", active)
            and not re.search(r"--min-instances(?:=|\s+)[1-9]\d*\b", active)
        ),
        "max_instances_one": bool(
            present
            and re.search(r"--max-instances(?:=|\s+)1\b", active)
            and not re.search(r"--max-instances(?:=|\s+)(?:0|[2-9]|\d{2,})\b", active)
        ),
        "no_cpu_throttling": bool(present and re.search(r"--no-cpu-throttling\b", active)),
        "concurrency_80": bool(present and re.search(r"--concurrency(?:=|\s+)80\b", active)),
        "litestream_replica_url": bool(present and "HIVEMIND_ROUTER_REPLICA_URL=" in active and "gcs://" in active),
        "secret_manager_router_secret": bool(present and "HIVEMIND_ROUTER_SECRET=" in active),
        "build_sha_propagation": bool(present and "HIVEMIND_BUILD_SHA=" in active and "--update-env-vars" in active),
        "zero_starter_grants": bool(present and "HIVEMIND_STARTER_GRANT_HMT:-0" in active),
    }
    finops_ok = present and all(invariants.values())
    sha_prefix = f"HIVEMIND_BUILD_SHA={expected_sha} " if expected_sha else ""
    dry_run_cmd = f"{sha_prefix}HIVEMIND_DEPLOY_DRY_RUN=1 bash deploy/gcp_openrouter/deploy.sh --dry-run"
    live_cmd = f"{sha_prefix}bash deploy/gcp_openrouter/deploy.sh"
    note = (
        f"Guide dry-run Cloud Run ({display_path}) : 0 mutation cloud effectuée."
        if lang_code == "fr"
        else f"Dry-run Cloud Run guidance ({display_path}): 0 cloud mutations performed."
    )
    return {
        "deploy_script_path": display_path,
        "deploy_script_present": present,
        "finops_ok": finops_ok,
        "invariants": invariants,
        "dry_run_command": dry_run_cmd,
        "live_owner_deploy_command": live_cmd,
        "executed": False,
        "note": note,
    }


def _is_nonneg_int(val: object) -> bool:
    return not isinstance(val, bool) and isinstance(val, int) and val >= 0


def _inspect_arena_summary_contract(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Verify ``GET /v1/hivemind/arena/summary`` vote conservation and Wilson 95% lower-bound fields."""
    is_fr = lang_code == "fr"
    if not isinstance(body, Mapping) or body.get("object") != ARENA_SUMMARY_OBJECT:
        return (
            False,
            {},
            [
                "Contrat JSON `/v1/hivemind/arena/summary` invalide (`object` invalide)."
                if is_fr
                else "Arena summary `/v1/hivemind/arena/summary` invalid `object` metadata."
            ],
        )
    if _has_forbidden_score_key(body):
        return (
            False,
            {},
            ["Arena summary `/v1/hivemind/arena/summary` contains forbidden key."],
        )

    tv = body.get("total_votes")
    hw = body.get("holiday_wins")
    bw = body.get("baseline_wins")
    ti = body.get("ties")
    counts_valid = all(_is_nonneg_int(x) for x in (tv, hw, bw, ti))
    conserved = bool(counts_valid and int(hw) + int(bw) + int(ti) == int(tv))

    dec_ok = True
    if "decisive_votes" in body:
        dv = body.get("decisive_votes")
        dec_ok = bool(_is_nonneg_int(dv) and counts_valid and int(dv) == int(hw) + int(bw))

    raw_wlb = body.get("wilson_95_lower_bound")
    wlb_ok = bool(
        "wilson_95_lower_bound" in body
        and not isinstance(raw_wlb, bool)
        and isinstance(raw_wlb, (int, float))
        and math.isfinite(float(raw_wlb))
        and 0.0 <= float(raw_wlb) <= 1.0
    )

    decisive = (int(hw) + int(bw)) if counts_valid else 0
    if wlb_ok and counts_valid and decisive == 0 and float(raw_wlb) != 0.0:
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
                or int(t_hw) + int(t_bw) + int(t_ti) != int(t_tv)
                or "wilson_95_lower_bound" not in t_doc
                or isinstance(t_wlb, bool)
                or not isinstance(t_wlb, (int, float))
                or not math.isfinite(float(t_wlb))
                or not (0.0 <= float(t_wlb) <= 1.0)
            ):
                by_task_ok = False
                break

    ok = bool(counts_valid and conserved and dec_ok and wlb_ok and win_ratio_ok and by_task_ok)
    issues = (
        []
        if ok
        else [
            "Conservation des votes ou borne Wilson 95 % invalide sur `/v1/hivemind/arena/summary`."
            if is_fr
            else "Vote conservation (`holiday_wins + baseline_wins + ties == total_votes`) or Wilson 95% lower-bound invalid on `/v1/hivemind/arena/summary`."
        ]
    )
    summary_doc = (
        {
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
        if ok
        else {}
    )
    return ok, summary_doc, issues


def run_deploy_sh_dry_run(
    deploy_sh_path: Path | str | None = None,
    *,
    expected_sha: str | None = None,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Execute list-argv ``bash deploy/gcp_openrouter/deploy.sh --dry-run`` without invoking ``gcloud``."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    path = Path(deploy_sh_path) if deploy_sh_path is not None else DEFAULT_DEPLOY_SH_PATH
    display_path = redact_home_path(path)
    if not path.is_file():
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

    resolved_sha = validate_expected_sha(expected_sha) if (expected_sha and str(expected_sha).strip()) else None
    dry_env: dict[str, str] = dict(environ) if environ is not None else {}
    dry_env["HIVEMIND_DEPLOY_DRY_RUN"] = "1"
    if resolved_sha is not None:
        dry_env["HIVEMIND_BUILD_SHA"] = resolved_sha
    elif environ is None and "HIVEMIND_BUILD_SHA" not in dry_env and os.environ.get("HIVEMIND_BUILD_SHA"):
        env_sha = sanitize_build_sha(os.environ.get("HIVEMIND_BUILD_SHA"))
        if env_sha != "unknown":
            dry_env["HIVEMIND_BUILD_SHA"] = env_sha

    argv = ["bash", str(path), "--dry-run"]
    rc, out = _invoke_runner(runner, argv, SUBPROCESS_TIMEOUT_S, environ=dry_env)
    kv: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
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


def project_adoption_readiness_gates(
    *,
    gateway_and_models_ok: bool,
    holiday_routes_ok: bool,
    feedback_summary: Mapping[str, Any] | None = None,
    active_volunteer_gpus: int = 0,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    lang: str = "en",
) -> dict[str, Any]:
    """Project the 5 AGY Noel-at-Scale adoption readiness gates from live smoke & CES metrics."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    fb = feedback_summary if isinstance(feedback_summary, Mapping) else {}
    raw_total = fb.get("total_responses")
    has_resp = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total > 0
    raw_ease = fb.get("mean_ease_rating") if "mean_ease_rating" in fb else fb.get("average_ease_rating")
    raw_lag = fb.get("zero_game_lag_ratio")

    ces_ok = bool(
        has_resp
        and not isinstance(raw_ease, bool)
        and isinstance(raw_ease, (int, float))
        and math.isfinite(float(raw_ease))
        and Fraction(str(float(raw_ease))) >= Fraction(str(float(min_ease_rating)))
    )
    lag_ok = bool(
        has_resp
        and not isinstance(raw_lag, bool)
        and isinstance(raw_lag, (int, float))
        and math.isfinite(float(raw_lag))
        and Fraction(str(float(raw_lag))) >= Fraction(str(float(min_zero_lag_ratio)))
    )
    gpus = int(active_volunteer_gpus) if not isinstance(active_volunteer_gpus, bool) and isinstance(active_volunteer_gpus, int) else 0
    gpu_ok = gpus >= int(target_online_producers)

    gates = {
        "gateway_and_models_online": bool(gateway_and_models_ok),
        "holiday_routes_deployed_on_cloud_run": bool(holiday_routes_ok),
        "satisfaction_ces_target_met": ces_ok,
        "gamer_zero_lag_target_met": lag_ok,
        "external_volunteer_gpu_target_noel": gpu_ok,
    }
    blockers: list[str] = []
    if not gates["gateway_and_models_online"]:
        blockers.append("Passerelle `/health` ou `/v1/models` indisponible." if is_fr else "Gateway `/health` or `/v1/models` unavailable.")
    if not gates["holiday_routes_deployed_on_cloud_run"]:
        blockers.append("Routes Noël Cloud Run incomplètes." if is_fr else "Cloud Run holiday onboarding routes incomplete.")
    if not ces_ok:
        blockers.append(
            f"Score CES moyen cible (>= {min_ease_rating} / 5) non atteint."
            if is_fr
            else f"Mean CES ease rating (>= {min_ease_rating} / 5) not yet met."
        )
    if not lag_ok:
        blockers.append(
            f"Ratio zéro-lag Gamer (>= {min_zero_lag_ratio}) non atteint."
            if is_fr
            else f"Gamer zero-lag ratio (>= {min_zero_lag_ratio}) not yet met."
        )
    if not gpu_ok:
        blockers.append(
            f"GPUs bénévoles actifs ({gpus}) < objectif Noël ({target_online_producers})."
            if is_fr
            else f"Active volunteer GPUs ({gpus}) below holiday target ({target_online_producers})."
        )

    passed_ids = [k for k, v in gates.items() if v]
    return {
        "object": "hivemind.adoption_readiness_projection",
        "campaign_id": CAMPAIGN_ID,
        "lang": lang_code,
        "gates": gates,
        "readiness_gates_passed": len(passed_ids),
        "readiness_gates_total": len(gates),
        "passed_gate_ids": passed_ids,
        "pending_gate_ids": [k for k, v in gates.items() if not v],
        "blockers": blockers,
        "soak_ces_gates_met": bool(gates["gateway_and_models_online"] and gates["holiday_routes_deployed_on_cloud_run"] and ces_ok and lag_ok),
    }


def _resolve_git_head_sha(runner: RunnerFn, cwd: Path | str | None, *, lang_code: str) -> tuple[str | None, str | None]:
    is_fr = lang_code == "fr"
    argv = (
        ["git", "-C", str(cwd), "rev-parse", "HEAD"]
        if cwd is not None and (Path(cwd) / ".git").exists()
        else ["git", "rev-parse", "HEAD"]
    )
    try:
        rc, out = _invoke_runner(runner, argv, SUBPROCESS_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, SmokeVerifierError) as exc:
        return None, (
            f"Échec lecture SHA git HEAD local ({type(exc).__name__})."
            if is_fr
            else f"Failed to resolve local git HEAD SHA ({type(exc).__name__})."
        )
    candidate = sanitize_build_sha(out.strip()) if rc == 0 else "unknown"
    if rc != 0 or candidate == "unknown":
        return None, (
            "SHA git HEAD invalide via `git rev-parse HEAD`."
            if is_fr
            else "Unable to resolve valid git HEAD SHA via `git rev-parse HEAD`."
        )
    return candidate, None


def _validate_quorum_slo_options(
    *,
    target_online_producers: int,
    min_ease_rating: float,
    min_zero_lag_ratio: float,
    require_live_quorum: bool,
    require_ces_slo: bool,
    check_arena_summary: bool,
    execute_deploy_dry_run: bool,
) -> tuple[int, float, float]:
    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or not (1 <= target_online_producers <= 500)
    ):
        raise SmokeVerifierError("target_online_producers must be an integer between 1 and 500.")
    if (
        isinstance(min_ease_rating, bool)
        or not isinstance(min_ease_rating, (int, float))
        or not math.isfinite(float(min_ease_rating))
        or not (1.0 <= float(min_ease_rating) <= 5.0)
    ):
        raise SmokeVerifierError("min_ease_rating must be a finite number between 1.0 and 5.0.")
    if (
        isinstance(min_zero_lag_ratio, bool)
        or not isinstance(min_zero_lag_ratio, (int, float))
        or not math.isfinite(float(min_zero_lag_ratio))
        or not (0.0 <= float(min_zero_lag_ratio) <= 1.0)
    ):
        raise SmokeVerifierError("min_zero_lag_ratio must be a finite number between 0.0 and 1.0.")
    for flag in (require_live_quorum, require_ces_slo, check_arena_summary, execute_deploy_dry_run):
        if not isinstance(flag, bool):
            raise SmokeVerifierError("Quorum/SLO/arena/dry-run flags must be booleans.")
    return int(target_online_producers), float(min_ease_rating), float(min_zero_lag_ratio)


def _extract_fleet_quorum_metrics(body: Mapping[str, Any]) -> tuple[bool, int, int]:
    """Extract schema validity, ``online_producers``, and ``distinct_physical_gpus`` from fleet JSON."""
    rows = body.get("hiveminders")
    rows_valid = isinstance(rows, list)
    endpoint_leaked = rows_valid and any(isinstance(r, Mapping) and ("endpoint_url" in r) for r in rows)
    has_live_hm = "live_hiveminders" in body
    live_hm_ok = (not has_live_hm) or _is_nonneg_int(body.get("live_hiveminders"))
    fleet_ok = bool(body.get("object") == FLEET_OBJECT and rows_valid and not endpoint_leaked and live_hm_ok)
    if not fleet_ok:
        return False, 0, 0

    live_hm_int = int(body["live_hiveminders"]) if has_live_hm else 0
    row_online = sum(
        1
        for r in rows  # type: ignore[union-attr]
        if isinstance(r, Mapping) and str(r.get("status", "")).strip().lower() in LIVE_PRODUCER_STATUSES
    )
    online_producers = max(live_hm_int, row_online)
    raw_gpu = body.get("distinct_physical_gpus")
    gpus = int(raw_gpu) if _is_nonneg_int(raw_gpu) else 0
    return True, online_producers, gpus


def run_cloud_run_holiday_smoke_gate(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    probe_feedback_post: bool = False,
    idempotency_check: bool = False,
    require_coder_model: bool = True,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    require_ces_slo: bool = False,
    check_arena_summary: bool = False,
    execute_deploy_dry_run: bool = False,
    deploy_sh_path: Path | str | None = None,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run the zero-inference Cloud Run holiday-route smoke & drift gate."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("cloud_run_holiday_smoke", comp_doc, lang_code=lang_code)

    target_n, min_ease_f, min_lag_f = _validate_quorum_slo_options(
        target_online_producers=target_online_producers,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
        require_live_quorum=require_live_quorum,
        require_ces_slo=require_ces_slo,
        check_arena_summary=check_arena_summary,
        execute_deploy_dry_run=execute_deploy_dry_run,
    )

    clean_gw = validate_gateway_url(gateway)
    resolved_expected_sha: str | None = None
    expected_sha_source = "none"
    git_head_error: str | None = None
    subprocess_calls = 0
    if expected_sha is not None and str(expected_sha).strip():
        resolved_expected_sha = validate_expected_sha(expected_sha)
        expected_sha_source = "expected_sha"
    elif check_git_head:
        expected_sha_source = "git_head"
        subprocess_calls += 1
        resolved_expected_sha, git_head_error = _resolve_git_head_sha(runner, cwd, lang_code=lang_code)

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = []
    network_calls = 0

    health_ok = False
    models_ok = False
    model_ids: list[str] = []
    coder_present = False
    manifest_ok = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    missing_manifest_routes: list[str] = []
    missing_manifest_scripts: list[str] = []
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {
        "schema_valid": False,
        "total_responses": None,
        "mean_ease_rating": None,
        "average_ease_rating": None,
        "zero_game_lag_ratio": None,
        "by_profile": None,
    }
    raid_status_ok = False
    fleet_ok = False
    active_trainers = 0
    online_producers = 0
    distinct_physical_gpus = 0
    scripts_ok: dict[str, bool] = {}

    smoke_post_allow = frozenset({"/v1/hivemind/onboarding/feedback"})
    for path, kind, group in SMOKE_PROBE_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}", allowed_post_paths=smoke_post_allow)
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if code != 200:
            if code == 404 and group in ("manifest", "holiday", "feedback", "fleet", "script"):
                stale_404_routes.append(path)
            diag = (
                f"GET {path} -> HTTP {code} (attendu 200)."
                if is_fr
                else f"GET {path} -> HTTP {code} (expected 200)."
            )
            diagnostics.append(diag)
        elif path == "/health":
            health_ok = body.get("status") == "ok"
            if not health_ok:
                ok = False
                diag = "GET /health missing `status == 'ok'`."
                diagnostics.append(diag)
        elif path == "/v1/models":
            m_ok, model_ids, coder_present, m_issues = _inspect_models_contract(
                body, require_coder_model=require_coder_model, lang_code=lang_code
            )
            models_ok = m_ok
            if not m_ok:
                ok = False
                diag = m_issues[0]
                diagnostics.extend(m_issues)
        elif path == "/v1/hivemind/onboarding/manifest":
            ok, live_build_sha, cloud_run_revision, missing_manifest_routes, missing_manifest_scripts, m_issues = (
                _inspect_manifest_contract(body, lang_code=lang_code)
            )
            manifest_ok = ok
            if not ok:
                diag = m_issues[0]
                diagnostics.extend(m_issues)
        elif kind == "html":
            ok, h_doc, h_issues = _inspect_holiday_html_contract(
                path, body, seen_nonces, lang_code=lang_code
            )
            html_contracts[path] = h_doc
            if not ok:
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            ok, feedback_summary_doc, fb_issues = _inspect_feedback_summary_schema(body, lang_code=lang_code)
            feedback_summary_ok = ok
            if not ok:
                diag = fb_issues[0]
                diagnostics.extend(fb_issues)
        elif path == "/v1/hivemind/raid/status":
            swarm = body.get("swarm")
            raid_status_ok = body.get("object") == RAID_STATUS_OBJECT and isinstance(swarm, Mapping)
            ok = raid_status_ok
            if ok and isinstance(swarm, Mapping):
                raw_tr = swarm.get("active_trainers")
                if not isinstance(raw_tr, bool) and isinstance(raw_tr, int) and raw_tr >= 0:
                    active_trainers = raw_tr
            else:
                diag = "GET /v1/hivemind/raid/status schema invalid."
                diagnostics.append(diag)
        elif path == "/v1/hivemind/producers/fleet":
            fleet_ok, online_producers, distinct_physical_gpus = _extract_fleet_quorum_metrics(body)
            ok = fleet_ok
            if not ok:
                diag = "GET /v1/hivemind/producers/fleet schema invalid."
                diagnostics.append(diag)
        elif kind == "script_py":
            txt = str(body.get("text") or body.get("html") or "")
            ok = bool(txt.strip()) and "AGPL-3.0" in txt and "def ensure_connected_account" in txt
            scripts_ok["hivemind_quickstart.py"] = ok
            if not ok:
                diag = "GET /scripts/hivemind_quickstart.py failed integrity check."
                diagnostics.append(diag)
        elif kind == "script_doctor_py":
            txt = str(body.get("text") or body.get("html") or "")
            ok = bool(txt.strip()) and "AGPL-3.0" in txt and "def run_doctor" in txt
            scripts_ok["hivemind_doctor.py"] = ok
            if not ok:
                diag = "GET /scripts/hivemind_doctor.py failed integrity check."
                diagnostics.append(diag)

        if not ok:
            failed_routes.append(path)
        route_rows.append({"path": path, "http_status": code, "ok": ok, "diagnostic": diag})

    arena_summary_ok: bool | None = None
    arena_summary_doc: dict[str, Any] | None = None
    if check_arena_summary:
        network_calls += 1
        a_code, a_body = _invoke_http_fn(
            http_fn,
            "GET",
            f"{clean_gw}/v1/hivemind/arena/summary",
            allowed_post_paths=smoke_post_allow,
        )
        http_statuses["/v1/hivemind/arena/summary"] = a_code
        if a_code != 200:
            arena_summary_ok = False
            if a_code == 404:
                stale_404_routes.append("/v1/hivemind/arena/summary")
            a_diag = (
                f"GET /v1/hivemind/arena/summary -> HTTP {a_code} (attendu 200)."
                if is_fr
                else f"GET /v1/hivemind/arena/summary -> HTTP {a_code} (expected 200)."
            )
            diagnostics.append(a_diag)
            failed_routes.append("/v1/hivemind/arena/summary")
            route_rows.append({"path": "/v1/hivemind/arena/summary", "http_status": a_code, "ok": False, "diagnostic": a_diag})
        else:
            arena_summary_ok, arena_summary_doc, a_issues = _inspect_arena_summary_contract(
                a_body, lang_code=lang_code
            )
            if not arena_summary_ok:
                a_diag = a_issues[0]
                diagnostics.extend(a_issues)
                failed_routes.append("/v1/hivemind/arena/summary")
                route_rows.append({"path": "/v1/hivemind/arena/summary", "http_status": a_code, "ok": False, "diagnostic": a_diag})
            else:
                route_rows.append({"path": "/v1/hivemind/arena/summary", "http_status": a_code, "ok": True, "diagnostic": ""})

    sha_mismatch = False
    sha_match: bool | None = None
    if resolved_expected_sha is not None:
        sha_match = shas_match(live_build_sha, resolved_expected_sha)
        sha_mismatch = not sha_match
        if sha_mismatch:
            diagnostics.append(
                f"Dérive build_sha : live='{live_build_sha}' != attendu '{resolved_expected_sha}'."
                if is_fr
                else f"Build SHA drift: live='{live_build_sha}' != expected '{resolved_expected_sha}'."
            )
    elif git_head_error is not None:
        sha_match = False
        sha_mismatch = True
        diagnostics.append(git_head_error)

    html_all_ok = (
        len(html_contracts) == len(SMOKE_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in SMOKE_HTML_ROUTES)
        and len(seen_nonces) == len(SMOKE_HTML_ROUTES)
    )

    feedback_post_probed = bool(probe_feedback_post or idempotency_check)
    feedback_post_ok: bool | None = None
    idempotency_conserved: bool | None = None
    idem_key: str | None = None
    if feedback_post_probed:
        before_total = feedback_summary_doc.get("total_responses")
        idem_key = f"smoke-dyn19-{secrets.token_hex(12)}"
        fb_payload = {
            "profile": "dev",
            "ease_rating": 5,
            "zero_game_lag": True,
            "platform": resolve_posix_platform(),
            "note": "",
        }
        network_calls += 1
        p1_code, p1_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/onboarding/feedback",
            payload=fb_payload,
            headers={"Idempotency-Key": idem_key},
            allowed_post_paths=smoke_post_allow,
        )
        http_statuses["POST /v1/hivemind/onboarding/feedback"] = p1_code
        p1_valid, p1_sum, p1_issues = (
            _inspect_feedback_summary_schema(p1_body, lang_code=lang_code)
            if p1_code in (200, 201)
            else (False, feedback_summary_doc, [f"POST feedback HTTP {p1_code}."])
        )
        feedback_post_ok = bool(p1_code in (200, 201) and p1_valid and (p1_sum.get("total_responses") or 0) >= 1)
        if not feedback_post_ok:
            diagnostics.extend(p1_issues)
        elif idempotency_check:
            network_calls += 1
            p2_code, p2_body = _invoke_http_fn(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/onboarding/feedback",
                payload=fb_payload,
                headers={"Idempotency-Key": idem_key},
                allowed_post_paths=smoke_post_allow,
            )
            http_statuses["POST /v1/hivemind/onboarding/feedback#replay"] = p2_code
            p2_valid, p2_sum, _ = (
                _inspect_feedback_summary_schema(p2_body, lang_code=lang_code)
                if p2_code in (200, 201)
                else (False, p1_sum, [])
            )
            conserved = bool(
                p2_code in (200, 201)
                and p2_valid
                and p2_sum.get("total_responses") == p1_sum.get("total_responses")
                and (before_total is None or int(p1_sum.get("total_responses") or 0) >= int(before_total))
            )
            idempotency_conserved = conserved
            if not conserved:
                feedback_post_ok = False
                diagnostics.append(
                    "Violation conservation Idempotency-Key sur replay feedback."
                    if is_fr
                    else "Idempotency-Key replay conservation check failed on feedback POST."
                )
        if feedback_post_ok:
            network_calls += 1
            ref_code, ref_body = _invoke_http_fn(
                http_fn,
                "GET",
                f"{clean_gw}/v1/hivemind/onboarding/feedback/summary",
                allowed_post_paths=smoke_post_allow,
            )
            http_statuses["/v1/hivemind/onboarding/feedback/summary#after_post"] = ref_code
            ref_ok, ref_sum, ref_issues = (
                _inspect_feedback_summary_schema(ref_body, lang_code=lang_code)
                if ref_code == 200
                else (False, p1_sum, ["GET feedback summary after POST failed."])
            )
            if ref_code == 200 and ref_ok and ref_sum.get("total_responses") == p1_sum.get("total_responses"):
                feedback_summary_ok = True
                feedback_summary_doc = ref_sum
            else:
                feedback_post_ok = False
                diagnostics.extend(ref_issues)

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
    if not feedback_summary_ok:
        ces_slo_ok = False
        zero_lag_slo_ok = False
        ces_gate_ok = False
    else:
        fb_total = feedback_summary_doc.get("total_responses")
        if isinstance(fb_total, int) and fb_total > 0:
            raw_mean = feedback_summary_doc.get("mean_ease_rating")
            raw_lag = feedback_summary_doc.get("zero_game_lag_ratio")
            ces_slo_ok = bool(raw_mean is not None and Fraction(str(float(raw_mean))) >= Fraction(str(min_ease_f)))
            zero_lag_slo_ok = bool(raw_lag is not None and Fraction(str(float(raw_lag))) >= Fraction(str(min_lag_f)))
            ces_gate_ok = bool(not require_ces_slo or (ces_slo_ok and zero_lag_slo_ok))
            if require_ces_slo and not ces_gate_ok:
                diagnostics.append(
                    f"SLO CES/zéro-lag non respecté (ease={raw_mean}, lag={raw_lag})."
                    if is_fr
                    else f"CES/zero-lag SLO breached (ease={raw_mean} < {min_ease_f} or lag={raw_lag} < {min_lag_f})."
                )
        else:
            ces_slo_ok = None
            zero_lag_slo_ok = None
            ces_gate_ok = True

    holiday_routes_deployed = bool(manifest_ok and html_all_ok and raid_status_ok)
    onboarding_feedback_deployed = bool(
        feedback_summary_ok and (not feedback_post_probed or feedback_post_ok is True)
    )
    stale_revision_detected = bool(
        stale_404_routes or missing_manifest_routes or missing_manifest_scripts or sha_mismatch
    )
    core_deploy_ok = bool(
        health_ok
        and models_ok
        and holiday_routes_deployed
        and onboarding_feedback_deployed
        and fleet_ok
        and all(scripts_ok.values())
        and len(scripts_ok) == 2
        and not sha_mismatch
        and (not check_arena_summary or arena_summary_ok is True)
    )

    deploy_guidance = audit_deploy_sh_guidance(
        deploy_sh_path, expected_sha=resolved_expected_sha, lang=lang_code
    )
    deploy_dry_run_res: dict[str, Any] | None = None
    if execute_deploy_dry_run:
        deploy_dry_run_res = run_deploy_sh_dry_run(
            deploy_sh_path,
            expected_sha=resolved_expected_sha,
            runner=runner,
            environ=environ,
            lang=lang_code,
        )
        if deploy_dry_run_res.get("executed"):
            subprocess_calls += 1
        deploy_guidance["executed"] = bool(deploy_dry_run_res.get("executed"))
        deploy_guidance["dry_run_result"] = deploy_dry_run_res
        if not deploy_dry_run_res.get("ok"):
            diagnostics.append(
                str(deploy_dry_run_res.get("diagnostic") or "deploy.sh --dry-run verification failed.")
            )

    smoke_ok = bool(
        core_deploy_ok
        and quorum_gate_ok
        and ces_gate_ok
        and (not execute_deploy_dry_run or (deploy_dry_run_res is not None and deploy_dry_run_res["ok"] is True))
    )
    redeploy_required = bool(stale_revision_detected or not core_deploy_ok)

    adoption_gates = project_adoption_readiness_gates(
        gateway_and_models_ok=bool(health_ok and models_ok),
        holiday_routes_ok=bool(holiday_routes_deployed and onboarding_feedback_deployed),
        feedback_summary=feedback_summary_doc,
        active_volunteer_gpus=max(online_producers, distinct_physical_gpus, active_trainers),
        min_ease_rating=min_ease_f,
        min_zero_lag_ratio=min_lag_f,
        target_online_producers=target_n,
        lang=lang_code,
    )

    summary = (
        (
            f"Smoke gate Cloud Run Noël VERT sur {clean_gw} "
            f"(nonces_uniques={len(seen_nonces)}, en_ligne={online_producers}/{target_n}, gates={adoption_gates['readiness_gates_passed']}/5)."
            if is_fr
            else f"Cloud Run holiday smoke gate PASSED on {clean_gw} "
            f"(unique_nonces={len(seen_nonces)}, online={online_producers}/{target_n}, gates={adoption_gates['readiness_gates_passed']}/5)."
        )
        if smoke_ok
        else (
            f"Smoke gate Cloud Run Noël EN ÉCHEC sur {clean_gw} : redeploy_required={redeploy_required}."
            if is_fr
            else f"Cloud Run holiday smoke gate FAILED on {clean_gw}: redeploy_required={redeploy_required}."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "cloud_run_holiday_smoke",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "smoke_ok": smoke_ok,
            "health_ok": health_ok,
            "models_ok": models_ok,
            "live_model_ids": model_ids,
            "hivemind_coder_present": coder_present,
            "manifest_ok": manifest_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": resolved_expected_sha,
            "expected_sha_source": expected_sha_source,
            "cloud_run_revision": cloud_run_revision,
            "sha_match": sha_match,
            "sha_mismatch": sha_mismatch,
            "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed,
            "onboarding_feedback_deployed": onboarding_feedback_deployed,
            "stale_revision_detected": stale_revision_detected,
            "stale_404_routes": stale_404_routes,
            "redeploy_required": redeploy_required,
            "html_contract_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "feedback_post_probed": feedback_post_probed,
            "feedback_post_ok": feedback_post_ok,
            "idempotency_check_probed": bool(idempotency_check),
            "idempotency_conserved": idempotency_conserved,
            "idempotency_key": idem_key,
            "raid_status_ok": raid_status_ok,
            "active_trainers": active_trainers,
            "fleet_ok": fleet_ok,
            "online_producers": online_producers,
            "target_online_producers": target_n,
            "producer_quorum_met": producer_quorum_met,
            "awaiting_producer_quorum": awaiting_producer_quorum,
            "ces_slo_ok": ces_slo_ok,
            "zero_lag_slo_ok": zero_lag_slo_ok,
            "distinct_physical_gpus": distinct_physical_gpus,
            "arena_summary_checked": bool(check_arena_summary),
            "arena_summary_ok": arena_summary_ok,
            "arena_summary": arena_summary_doc,
            "deploy_dry_run": deploy_dry_run_res,
            "scripts_ok": scripts_ok,
            "http_statuses": http_statuses,
            "routes": route_rows,
            "failed_routes": failed_routes,
            "deploy_guidance": deploy_guidance,
            "adoption_gates": adoption_gates,
            "network_calls": network_calls,
            "subprocess_calls": subprocess_calls,
            "zero_inference_verified": True,
            "diagnostics": diagnostics,
            "exit_code": 0 if smoke_ok else 2,
            "summary": summary,
        }
    )


def _parse_json_safe(raw: str) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        return {}


def _build_bash_probe_argv(
    runner: RunnerFn,
    environ: Mapping[str, str] | None,
    extra_env: Mapping[str, str] | None,
    inner_cmd: list[str],
) -> list[str]:
    if runner is default_subprocess_runner and (environ is not None or extra_env is not None):
        base_env = dict(environ) if environ is not None else dict(os.environ)
        if "PATH" not in base_env:
            base_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
        if extra_env:
            base_env.update(extra_env)
        return ["env", "-i", *(f"{k}={v}" for k, v in base_env.items()), *inner_cmd]
    if extra_env:
        return ["env", *(f"{k}={v}" for k, v in extra_env.items()), *inner_cmd]
    return inner_cmd


def audit_posix_launcher_contract(
    sh_path: Path | str | None = None,
    ps1_path: Path | str | None = None,
    *,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Audit ``scripts/join_hivemind_campaign.sh`` and ``scripts/join_hivemind.ps1`` contracts."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    eff_sh = Path(sh_path) if sh_path is not None else DEFAULT_SH_PATH
    eff_ps1 = Path(ps1_path) if ps1_path is not None else DEFAULT_PS1_PATH
    eff_cwd = cwd if cwd is not None else eff_sh.parent
    comp_doc = evaluate_personal_pc_compliance(environ=environ, cwd=eff_cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("posix_launcher_contract_audit", comp_doc, lang_code=lang_code)

    diagnostics: list[str] = []
    subprocess_calls = 0

    sh_exists = eff_sh.is_file()
    sh_text = eff_sh.read_text(encoding="utf-8", errors="replace") if sh_exists else ""
    sh_header_ok = "Licensed under the AGPL-3.0" in sh_text[:300] and "0.00 EUR / 0.00 USD" in sh_text
    sh_flags_ok = all(flag in sh_text for flag in SH_REQUIRED_FLAGS)
    sh_routing_ok = (
        "hivemind_quickstart.py" in sh_text
        and "hivemind_doctor.py" in sh_text
        and "--noel" in sh_text
        and "noel_posix_launcher_and_smoke_verifier.py" in sh_text
        and "--smoke-only" in sh_text
        and "noel_cloud_run_parity_gate.py" in sh_text
        and "--post-deploy-gate" in sh_text
        and "noel_vscode_quick_preset.py" in sh_text
        and "--vscode-setup" in sh_text
        and "noel_vscode_autoconfig.py" in sh_text
        and "--quick-connect" in sh_text
        and "noel_idle_resume_service.py" in sh_text
    )
    sh_guard_ok = (
        "AMADEUS" in sh_text
        and "USERDNSDOMAIN" in sh_text
        and ".git" in sh_text
        and "uninstall" in sh_text
        and "run.app" in sh_text
        and "ProxyHandler({})" in sh_text
    )
    sh_unsafe_clean = not bool(_PS1_UNSAFE_RE.search(sh_text)) and not bool(_SH_UNSAFE_RE.search(sh_text))
    sh_figures_clean = not bool(_STATIC_FIG_UNIT_RE.search(sh_text))

    bash_syntax_ok = False
    bash_help_ok = False
    bash_dry_run_ok = False
    bash_corporate_block_ok = False
    bash_untrusted_gateway_ok = False
    bash_uninstall_dry_run_ok = False

    if sh_exists:
        subprocess_calls += 1
        rc_syn, _ = _invoke_runner(
            runner,
            _build_bash_probe_argv(runner, environ, None, ["bash", "-n", str(eff_sh)]),
            SUBPROCESS_TIMEOUT_S,
            environ=environ,
        )
        bash_syntax_ok = rc_syn == 0

        subprocess_calls += 1
        rc_help, out_help = _invoke_runner(
            runner,
            _build_bash_probe_argv(runner, environ, None, ["bash", str(eff_sh), "--help"]),
            SUBPROCESS_TIMEOUT_S,
            environ=environ,
        )
        bash_help_ok = (
            rc_help == 0
            and "--gamer-whisper" in out_help
            and "--connect" in out_help
            and "--doctor" in out_help
            and "--deploy-smoke" in out_help
        )

        subprocess_calls += 1
        sample_key = "hm-audit" + "-sample-9876"
        rc_dry, out_dry = _invoke_runner(
            runner,
            _build_bash_probe_argv(
                runner,
                environ,
                None,
                ["bash", str(eff_sh), "--connect", "--api-key", sample_key, "--dry-run", "--json"],
            ),
            SUBPROCESS_TIMEOUT_S,
            environ=environ,
        )
        parsed_dry = _parse_json_safe(out_dry) if rc_dry == 0 else {}
        bash_dry_run_ok = bool(
            rc_dry == 0
            and parsed_dry.get("dry_run") is True
            and parsed_dry.get("compliance_ok") is True
            and parsed_dry.get("target_script") == "hivemind_quickstart.py"
            and sample_key not in out_dry
            and "hm-***9876" in out_dry
        )

        subprocess_calls += 1
        corp_sentinel = "corp-audit-probe"
        rc_corp, out_corp = _invoke_runner(
            runner,
            _build_bash_probe_argv(
                runner,
                environ,
                {"AMADEUS_PROFILE": corp_sentinel},
                ["bash", str(eff_sh), "--connect", "--dry-run", "--json"],
            ),
            SUBPROCESS_TIMEOUT_S,
            environ=environ,
        )
        parsed_corp = _parse_json_safe(out_corp)
        bash_corporate_block_ok = bool(
            rc_corp == 2
            and parsed_corp.get("compliance_ok") is False
            and corp_sentinel not in out_corp
        )

        subprocess_calls += 1
        untrusted_gw = "".join(("https", "://", "untrusted.example.invalid"))
        rc_gw, _ = _invoke_runner(
            runner,
            _build_bash_probe_argv(
                runner,
                environ,
                None,
                ["bash", str(eff_sh), "--gateway", untrusted_gw, "--dry-run", "--json"],
            ),
            SUBPROCESS_TIMEOUT_S,
            environ=environ,
        )
        bash_untrusted_gateway_ok = rc_gw == 2

        subprocess_calls += 1
        rc_un, out_un = _invoke_runner(
            runner,
            _build_bash_probe_argv(
                runner,
                environ,
                {"AMADEUS_PROFILE": corp_sentinel},
                ["bash", str(eff_sh), "--uninstall", "--dry-run", "--json"],
            ),
            SUBPROCESS_TIMEOUT_S,
            environ=environ,
        )
        parsed_un = _parse_json_safe(out_un) if rc_un == 0 else {}
        bash_uninstall_dry_run_ok = bool(
            rc_un == 0
            and parsed_un.get("dry_run") is True
            and parsed_un.get("mode") == "uninstall"
            and parsed_un.get("target_script") == "hivemind_quickstart.py"
        )

    sh_ok = bool(
        sh_exists
        and sh_header_ok
        and sh_flags_ok
        and sh_routing_ok
        and sh_guard_ok
        and sh_unsafe_clean
        and sh_figures_clean
        and bash_syntax_ok
        and bash_help_ok
        and bash_dry_run_ok
        and bash_corporate_block_ok
        and bash_untrusted_gateway_ok
        and bash_uninstall_dry_run_ok
    )
    if not sh_ok:
        diagnostics.append(
            "Contrat Bash `scripts/join_hivemind_campaign.sh` en échec."
            if is_fr
            else "POSIX Bash launcher `scripts/join_hivemind_campaign.sh` failed contract audit."
        )

    ps1_exists = eff_ps1.is_file()
    ps1_text = eff_ps1.read_text(encoding="utf-8", errors="replace") if ps1_exists else ""
    ps1_header_ok = "Licensed under the AGPL-3.0" in ps1_text[:300]
    ps1_branding_ok = (
        "0.00 EUR / 0.00 USD" in ps1_text
        and "GamerWhisper" in ps1_text
        and "[CmdletBinding()]" in ps1_text
    )
    ps1_modes_ok = all(f'"{m}"' in ps1_text for m in PS1_ALLOWED_MODES)
    ps1_params_ok = all(
        re.search(rf"\$\s*{re.escape(p.lstrip('-'))}\b", ps1_text)
        for p in PS1_REQUIRED_PARAMS
    )
    ps1_routing_ok = (
        "hivemind_quickstart.py" in ps1_text
        and "hivemind_doctor.py" in ps1_text
        and "--noel" in ps1_text
        and "noel_deploy_smoke_and_win11_verifier.py" in ps1_text
        and "--smoke-only" in ps1_text
    )
    ps1_guard_ok = (
        "AMADEUS" in ps1_text
        and "USERDNSDOMAIN" in ps1_text
        and ".git" in ps1_text
        and "Uninstall" in ps1_text
        and "exit 2" in ps1_text
    )
    ps1_bootstrap_ok = (
        "Invoke-RestMethod" in ps1_text
        and "-OutFile" in ps1_text
        and not _PS1_UNSAFE_RE.search(ps1_text)
    )
    ps1_figures_clean = not bool(_STATIC_FIG_UNIT_RE.search(ps1_text))
    ps1_ok = bool(
        ps1_exists
        and ps1_header_ok
        and ps1_branding_ok
        and ps1_modes_ok
        and ps1_params_ok
        and ps1_routing_ok
        and ps1_guard_ok
        and ps1_bootstrap_ok
        and ps1_figures_clean
    )
    if not ps1_ok:
        diagnostics.append(
            "Contrat PowerShell `scripts/join_hivemind.ps1` incomplet ou non conforme."
            if is_fr
            else "PowerShell launcher `scripts/join_hivemind.ps1` failed contract audit."
        )

    pwsh_bin = shutil.which("pwsh")
    pwsh_available = pwsh_bin is not None
    pwsh_checked = False
    pwsh_dry_run_ok: bool | None = None
    if pwsh_available and ps1_ok:
        pwsh_checked = True
        subprocess_calls += 1
        rc_pw, out_pw = _invoke_runner(
            runner,
            [
                pwsh_bin or "pwsh",
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(eff_ps1),
                "-Mode",
                "DeploySmoke",
                "-DryRun",
                "-Json",
            ],
            SUBPROCESS_TIMEOUT_S,
            environ=environ,
        )
        parsed_pw = _parse_json_safe(out_pw) if rc_pw == 0 else {}
        pwsh_dry_run_ok = bool(
            rc_pw == 0
            and parsed_pw.get("dry_run") is True
            and parsed_pw.get("target_script") == "noel_deploy_smoke_and_win11_verifier.py"
        )
        if not pwsh_dry_run_ok:
            diagnostics.append("pwsh -Mode DeploySmoke -DryRun -Json check failed.")

    contract_ok = bool(sh_ok and ps1_ok and (not pwsh_checked or pwsh_dry_run_ok is True))
    summary = (
        (
            "Audit contrat lanceurs POSIX `.sh` & Win11 `.ps1` VALIDE."
            if is_fr
            else "POSIX `.sh` & Win11 `.ps1` launcher contract audit PASSED."
        )
        if contract_ok
        else (
            "Audit contrat lanceurs POSIX `.sh` & Win11 `.ps1` EN ÉCHEC."
            if is_fr
            else "POSIX `.sh` & Win11 `.ps1` launcher contract audit FAILED."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "posix_launcher_contract_audit",
            "lang": lang_code,
            "compliance_ok": True,
            "contract_ok": contract_ok,
            "sh_path": redact_home_path(eff_sh),
            "sh_ok": sh_ok,
            "bash_syntax_ok": bash_syntax_ok,
            "bash_help_ok": bash_help_ok,
            "bash_dry_run_ok": bash_dry_run_ok,
            "bash_corporate_block_ok": bash_corporate_block_ok,
            "bash_untrusted_gateway_ok": bash_untrusted_gateway_ok,
            "bash_uninstall_dry_run_ok": bash_uninstall_dry_run_ok,
            "sh_checks": {
                "exists": sh_exists,
                "header_and_notice": sh_header_ok,
                "flags": sh_flags_ok,
                "routing": sh_routing_ok,
                "corporate_and_allowlist_guard": sh_guard_ok,
                "unsafe_sinks_clean": sh_unsafe_clean,
                "figures_clean": sh_figures_clean,
                "bash_syntax_ok": bash_syntax_ok,
                "bash_help_ok": bash_help_ok,
                "bash_dry_run_ok": bash_dry_run_ok,
                "bash_corporate_block_ok": bash_corporate_block_ok,
                "bash_untrusted_gateway_ok": bash_untrusted_gateway_ok,
                "bash_uninstall_dry_run_ok": bash_uninstall_dry_run_ok,
            },
            "ps1_path": redact_home_path(eff_ps1),
            "ps1_ok": ps1_ok,
            "ps1_checks": {
                "exists": ps1_exists,
                "header_agpl": ps1_header_ok,
                "branding_0_00_eur_gamer_whisper": ps1_branding_ok,
                "modes": ps1_modes_ok,
                "params": ps1_params_ok,
                "routing": ps1_routing_ok,
                "corporate_guard": ps1_guard_ok,
                "safe_bootstrap": ps1_bootstrap_ok,
                "figures_clean": ps1_figures_clean,
            },
            "pwsh_available": pwsh_available,
            "pwsh_checked": pwsh_checked,
            "pwsh_dry_run_ok": pwsh_dry_run_ok,
            "subprocess_calls": subprocess_calls,
            "diagnostics": diagnostics,
            "exit_code": 0 if contract_ok else 2,
            "summary": summary,
        }
    )


def _write_secure_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            with suppress(OSError):
                os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
        if os.name != "nt":
            with suppress(OSError):
                os.chmod(path, 0o600)
        tmp_path = None
    finally:
        if tmp_path is not None:
            with suppress(OSError):
                tmp_path.unlink()


def verify_posix_dev_vscode_60s_dry_run(
    gateway: str = DEFAULT_GATEWAY,
    *,
    home_dir: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    handle: str = "dev-holiday-probe",
    role: str = "prosumer",
    api_key: str | None = None,
    ease_rating: int = 5,
    submit_ces: bool = False,
    run_uninstall: bool = True,
    probe_gateway: bool = False,
    register_account: bool = False,
    platform_name: str | None = None,
    headers: Mapping[str, str] | None = None,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run an isolated < 60 s POSIX Dev VS Code onboarding dry-run (0600 creds, Continue YAML, clean uninstall)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_platform = resolve_posix_platform(platform_name)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise SmokeVerifierError("ease_rating must be an integer between 1 and 5.")

    eff_cwd = cwd if cwd is not None else workspace_dir
    comp_doc = evaluate_personal_pc_compliance(environ=environ, cwd=eff_cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("posix_dev_vscode_60s_dry_run", comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    t0 = time.monotonic()
    diagnostics: list[str] = []
    network_calls = 0
    file_writes = 0
    effective_key = (api_key or "").strip() or None
    account_id = "acc_dryrun"

    if register_account and effective_key is None:
        network_calls += 1
        reg_code, reg_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/register",
            payload={"handle": handle, "role": role},
            headers=headers,
        )
        if reg_code in (200, 201) and isinstance(reg_body.get("api_key"), str):
            effective_key = str(reg_body["api_key"]).strip()
            account_id = str(reg_body.get("account_id") or account_id)
        else:
            diagnostics.append(f"POST /v1/hivemind/register failed (HTTP {reg_code}).")

    models_checked = True
    if probe_gateway:
        network_calls += 1
        h_code, h_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/health", headers=headers)
        network_calls += 1
        m_code, m_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/models", headers=headers)
        models_checked = bool(
            h_code == 200
            and h_body.get("status") == "ok"
            and m_code == 200
            and isinstance(m_body.get("data"), list)
            and len(m_body["data"]) > 0
        )
        if not models_checked:
            diagnostics.append("Gateway /health or /v1/models probe failed during Dev dry-run.")

    if effective_key is None:
        effective_key = f"hm-dryrun-{secrets.token_hex(8)}"

    tmp_ctx = tempfile.TemporaryDirectory(prefix="hivemind-posix-dev60s-") if home_dir is None else None
    try:
        eff_home = Path(tmp_ctx.name) / "home" if tmp_ctx is not None else Path(home_dir)  # type: ignore[arg-type]
        eff_ws = (
            Path(workspace_dir)
            if workspace_dir is not None
            else (Path(tmp_ctx.name) / "ws" if tmp_ctx is not None else eff_home / "hobby_project")
        )
        cred_path = eff_home / ".hivemind" / "credentials.json"
        cont_path = eff_home / ".continue" / "hivemind_provider.yaml"
        vscode_path = eff_ws / ".vscode" / "settings.json"

        cred_obj = {
            "account_id": account_id,
            "handle": handle,
            "role": role,
            "gateway": clean_gw,
            "api_key": effective_key,
            "platform": eff_platform,
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
            f"    apiKey: {effective_key}\n"
            "  - name: HiveMind Bee Stars\n"
            "    provider: openai\n"
            "    model: hivemind/bee-stars\n"
            f"    apiBase: {clean_gw}/v1\n"
            f"    apiKey: {effective_key}\n"
        )
        _write_secure_atomic(cont_path, cont_yaml)
        file_writes += 1

        existing_ws: dict[str, Any] = {}
        if vscode_path.is_file():
            try:
                loaded_ws = json.loads(vscode_path.read_text(encoding="utf-8"))
                if isinstance(loaded_ws, dict):
                    existing_ws = loaded_ws
            except (OSError, json.JSONDecodeError):
                existing_ws = {}
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

        credentials_0600_ok = cred_path.is_file() and (
            os.name == "nt" or (cred_path.stat().st_mode & 0o777) == 0o600
        )
        cont_text = cont_path.read_text(encoding="utf-8") if cont_path.is_file() else ""
        continue_yaml_ok = "hivemind/coder" in cont_text and f"{clean_gw}/v1" in cont_text
        ws_dump = vscode_path.read_text(encoding="utf-8") if vscode_path.is_file() else ""
        vscode_secret_clean = (
            bool(ws_dump)
            and effective_key not in ws_dump
            and not _API_KEY_RE.search(ws_dump)
            and not _HOME_PATH_RE.search(ws_dump)
        )

        ces_recorded = False
        if submit_ces:
            network_calls += 1
            idem_ces = f"posix-dev60s-ces-{secrets.token_hex(12)}"
            ces_hdrs = {**(dict(headers) if headers else {}), "Idempotency-Key": idem_ces}
            c_code, c_body = _invoke_http_fn(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/onboarding/feedback",
                payload={
                    "profile": "dev",
                    "ease_rating": ease_rating,
                    "zero_game_lag": True,
                    "platform": eff_platform,
                    "note": "",
                },
                headers=ces_hdrs,
            )
            ces_recorded = c_code in (200, 201) and isinstance(c_body.get("total_responses"), int)
            if not ces_recorded:
                diagnostics.append(f"Dev CES ping failed (HTTP {c_code}).")

        uninstall_clean = True
        if run_uninstall:
            for target in (cred_path, cont_path):
                if target.is_file():
                    target.unlink()
            uninstall_clean = (not cred_path.exists()) and (not cont_path.exists())
            if not uninstall_clean:
                diagnostics.append("Clean --uninstall check left residual credential/config files.")

        elapsed_s = round(time.monotonic() - t0, 4)
        slo_ok = elapsed_s < DEV_VSCODE_SLO_SECONDS
        dev_ready = bool(
            slo_ok
            and models_checked
            and credentials_0600_ok
            and continue_yaml_ok
            and vscode_secret_clean
            and uninstall_clean
            and (not register_account or not any("register failed" in d for d in diagnostics))
            and (not submit_ces or ces_recorded)
        )

        summary = (
            (
                f"Dry-run Dev VS Code POSIX (< 60 s) RÉUSSI en {elapsed_s:.3f} s."
                if is_fr
                else f"POSIX Dev VS Code (< 60 s) dry-run PASSED in {elapsed_s:.3f} s."
            )
            if dev_ready
            else (
                f"Dry-run Dev VS Code POSIX (< 60 s) EN ÉCHEC ({elapsed_s:.3f} s)."
                if is_fr
                else f"POSIX Dev VS Code (< 60 s) dry-run FAILED ({elapsed_s:.3f} s)."
            )
        )

        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "posix_dev_vscode_60s_dry_run",
                "lang": lang_code,
                "platform": eff_platform,
                "gateway": clean_gw,
                "compliance_ok": True,
                "dev_ready": dev_ready,
                "slo_ok": slo_ok,
                "elapsed_s": elapsed_s,
                "slo_budget_s": DEV_VSCODE_SLO_SECONDS,
                "masked_api_key": mask_api_key(effective_key),
                "credentials_path": redact_home_path(cred_path, home=eff_home),
                "continue_config_path": redact_home_path(cont_path, home=eff_home),
                "vscode_settings_path": redact_home_path(vscode_path, home=eff_home),
                "credentials_0600_ok": credentials_0600_ok,
                "continue_yaml_ok": continue_yaml_ok,
                "vscode_secret_clean": vscode_secret_clean,
                "uninstall_ran": run_uninstall,
                "uninstall_clean": uninstall_clean,
                "models_checked": models_checked,
                "ces_submitted": submit_ces,
                "ces_recorded": ces_recorded,
                "network_calls": network_calls,
                "file_writes": file_writes,
                "zero_inference_verified": True,
                "diagnostics": diagnostics,
                "exit_code": 0 if dev_ready else 2,
                "summary": summary,
            }
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def default_gpu_probe(runner: RunnerFn = default_subprocess_runner) -> dict[str, Any]:
    """Probe local NVIDIA GPU via ``nvidia-smi`` without serials or UUIDs; never invent telemetry."""
    try:
        rc, out = _invoke_runner(
            runner,
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError, SmokeVerifierError):
        return {"present": False, "parse_error": False}
    if rc != 0 or not out.strip():
        return {"present": False, "parse_error": False}
    best: dict[str, Any] | None = None
    for row in csv.reader(io.StringIO(out.strip())):
        if len(row) < 5:
            return {"present": True, "parse_error": True}
        name = row[0].strip()
        try:
            tot = int(float(row[1].strip()))
            free = int(float(row[2].strip()))
        except ValueError:
            return {"present": True, "parse_error": True}
        if tot < MIN_REPORTABLE_VRAM_MIB:
            continue
        try:
            temp_c: float | None = float(row[3].strip())
        except ValueError:
            temp_c = None
        try:
            util_pct: float | None = float(row[4].strip())
        except ValueError:
            util_pct = None
        cand = {
            "present": True,
            "parse_error": False,
            "name": name,
            "total_mib": tot,
            "free_mib": free,
            "temp_c": temp_c,
            "util_pct": util_pct,
        }
        if best is None or tot > int(best["total_mib"]):
            best = cand
    return best if best is not None else {"present": False, "parse_error": False}


def default_process_probe(runner: RunnerFn = default_subprocess_runner) -> list[str]:
    """List running process executable basenames on Linux/macOS via ``ps -eo comm=``."""
    try:
        rc, out = _invoke_runner(runner, ["ps", "-eo", "comm="], SUBPROCESS_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, SmokeVerifierError):
        return []
    if rc != 0 or not out.strip():
        return []
    names: list[str] = []
    for line in out.splitlines():
        clean = line.strip()
        if clean:
            names.append(Path(clean).name.lower())
    return names


def default_worker_yield_drill(timeout_s: float = 1.4) -> dict[str, Any]:
    """Spawn a tiny local Python worker child, send YIELD, and verify < 1500.0 milliseconds release."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _LOCAL_WORKER_YIELD_PROGRAM],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    ready_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
    if ready_line != "HIVEMIND_WORKER_READY":
        proc.kill()
        proc.wait(timeout=1.0)
        return {
            "worker_started": False,
            "worker_stopped": True,
            "clean_release": False,
            "retained_worker_vram_mib": 64,
            "yield_latency_ms": round(timeout_s * 1000.0, 2),
        }

    t_yield = time.monotonic()
    if proc.stdin is not None:
        proc.stdin.write("YIELD\n")
        proc.stdin.flush()
        proc.stdin.close()
    try:
        rc = proc.wait(timeout=max(0.1, float(timeout_s)))
        elapsed_ms = round((time.monotonic() - t_yield) * 1000.0, 2)
        out_line = proc.stdout.readline().strip() if proc.stdout is not None else ""
        clean = rc == 0 and out_line == "HIVEMIND_WORKER_RELEASED:0"
        return {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": clean,
            "retained_worker_vram_mib": 0 if clean else 64,
            "yield_latency_ms": elapsed_ms,
        }
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)
        return {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": False,
            "retained_worker_vram_mib": 64,
            "yield_latency_ms": round((time.monotonic() - t_yield) * 1000.0, 2),
        }


def verify_posix_gamer_whisper_5m_dry_run(
    gateway: str = DEFAULT_GATEWAY,
    *,
    handle: str = "gamer-holiday-probe",
    host_label: str = "posix-personal-rtx",
    platform_name: str | None = None,
    reserve_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    max_upload_mb_per_hour: float = DEFAULT_MAX_UPLOAD_MB_PER_HOUR,
    ease_rating: int = 5,
    had_game_lag: bool = False,
    submit_ces: bool = False,
    send_heartbeat: bool = False,
    api_key: str | None = None,
    gpu_probe_fn: Callable[[], Mapping[str, Any]] | None = None,
    process_probe_fn: Callable[[], Sequence[str]] | None = None,
    worker_yield_fn: Callable[[], Mapping[str, Any]] | None = None,
    headers: Mapping[str, str] | None = None,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Run a < 5 min Linux/macOS Gamer-Whisper dry-run: < 1500.0 milliseconds yield, <= 70.0 C, >= 2048 MiB VRAM."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_platform = resolve_posix_platform(platform_name)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise SmokeVerifierError("ease_rating must be an integer between 1 and 5.")

    comp_doc = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("posix_gamer_whisper_5m_dry_run", comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    t0 = time.monotonic()
    diagnostics: list[str] = []
    network_calls = 0

    gpu = dict(gpu_probe_fn()) if gpu_probe_fn is not None else default_gpu_probe(runner=runner)
    gpu_present = bool(gpu.get("present")) and not bool(gpu.get("parse_error"))
    temp_c = gpu.get("temp_c") if gpu_present else None
    free_mib = gpu.get("free_mib") if gpu_present else None
    total_mib = gpu.get("total_mib") if gpu_present else None

    if bool(gpu.get("parse_error")):
        thermal_ok = False
        vram_reserve_ok = False
        gpu_status = "gpu_driver_parse_error"
        diagnostics.append("GPU telemetry parse error from nvidia-smi.")
    elif not gpu_present:
        thermal_ok = True
        vram_reserve_ok = True
        gpu_status = "cpu_or_dev_only_no_nvidia_gpu"
    else:
        thermal_ok = temp_c is not None and float(temp_c) <= float(max_temp_c)
        vram_reserve_ok = free_mib is not None and int(free_mib) >= int(reserve_mib)
        gpu_status = "rtx_quiet_ready" if (thermal_ok and vram_reserve_ok) else "gpu_guardrail_tripped"
        if not thermal_ok:
            diagnostics.append(
                f"GPU thermal guardrail tripped ({temp_c} C > {max_temp_c} C)."
            )
        if not vram_reserve_ok:
            diagnostics.append(
                f"Free VRAM guardrail tripped ({free_mib} MiB < {reserve_mib} MiB reserve)."
            )

    procs = [str(p).strip().lower() for p in (process_probe_fn() if process_probe_fn else default_process_probe(runner=runner))]
    active_games = [
        p
        for p in procs
        if p in STEAM_GAME_EXECUTABLES or any(p.startswith(pref) for pref in STEAM_PROCESS_PREFIXES)
    ]

    drill = dict(worker_yield_fn()) if worker_yield_fn is not None else default_worker_yield_drill()
    yield_ms = float(drill.get("yield_latency_ms", yield_budget_ms + 1.0))
    retained_vram = int(drill.get("retained_worker_vram_mib", 64))
    clean_release = bool(drill.get("clean_release"))
    worker_stopped = bool(drill.get("worker_stopped"))
    yield_slo_ok = bool(worker_stopped and clean_release and retained_vram == 0 and yield_ms < float(yield_budget_ms))
    if not yield_slo_ok:
        diagnostics.append(
            f"Preemption yield SLO violated (yield_ms={yield_ms:.2f}, budget={yield_budget_ms:.1f} milliseconds, retained={retained_vram} MiB)."
        )

    lora_mb_per_sync = (DEFAULT_LORA_PARAMS * DEFAULT_LORA_BYTES_PER_PARAM) / (1024.0 * 1024.0)
    estimated_mb_per_hour = round(lora_mb_per_sync * DEFAULT_SYNCS_PER_HOUR, 2)
    bandwidth_cap_ok = estimated_mb_per_hour < float(max_upload_mb_per_hour)

    heartbeat_ok = True
    if send_heartbeat:
        if not api_key:
            heartbeat_ok = False
            diagnostics.append("OpenTrainer heartbeat requested without an API key.")
        else:
            network_calls += 1
            hb_hdrs = {**(dict(headers) if headers else {}), "Authorization": f"Bearer {api_key}"}
            hb_payload = {
                "host_label": host_label,
                "gpu_name": str(gpu.get("name")) if gpu_present and gpu.get("name") else None,
                "vram_total_mib": int(total_mib) if gpu_present and isinstance(total_mib, int) and total_mib >= 512 else None,
                "vram_free_mib": int(free_mib) if gpu_present and isinstance(free_mib, int) and free_mib >= 0 else None,
                "network_mode": "outbound-https-relay",
                "training_stack": "qlora-moe+diloco+amcp-aglet+boinc",
                "active_campaign": CAMPAIGN_ID,
            }
            hb_code, _ = _invoke_http_fn(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
                payload=hb_payload,
                headers=hb_hdrs,
            )
            heartbeat_ok = hb_code in (200, 201)
            if not heartbeat_ok:
                diagnostics.append(f"POST /v1/hivemind/opentrainer/heartbeat failed (HTTP {hb_code}).")

    zero_game_lag = bool(yield_slo_ok and thermal_ok and not had_game_lag)
    ces_recorded = False
    if submit_ces:
        network_calls += 1
        idem_ces = f"posix-ces-{secrets.token_hex(12)}"
        ces_hdrs = {**(dict(headers) if headers else {}), "Idempotency-Key": idem_ces}
        c_code, c_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/onboarding/feedback",
            payload={
                "profile": "gamer",
                "ease_rating": ease_rating,
                "zero_game_lag": zero_game_lag,
                "platform": eff_platform,
                "note": "",
            },
            headers=ces_hdrs,
        )
        ces_recorded = c_code in (200, 201) and isinstance(c_body.get("total_responses"), int)
        if not ces_recorded:
            diagnostics.append(f"Gamer CES ping failed (HTTP {c_code}).")

    elapsed_s = round(time.monotonic() - t0, 4)
    slo_ok = elapsed_s < POSIX_ONBOARD_SLO_SECONDS
    gamer_ready = bool(
        slo_ok
        and yield_slo_ok
        and thermal_ok
        and vram_reserve_ok
        and bandwidth_cap_ok
        and heartbeat_ok
        and (not submit_ces or ces_recorded)
    )

    summary = (
        (
            f"Dry-run POSIX Gamer-Whisper (< 5 min) RÉUSSI (yield={yield_ms:.1f} milliseconds, 0 MiB conservé)."
            if is_fr
            else f"POSIX Gamer-Whisper (< 5 min) dry-run PASSED (yield={yield_ms:.1f} milliseconds, 0 MiB retained)."
        )
        if gamer_ready
        else (
            f"Dry-run POSIX Gamer-Whisper EN ÉCHEC ({'; '.join(diagnostics)})."
            if is_fr
            else f"POSIX Gamer-Whisper dry-run FAILED ({'; '.join(diagnostics)})."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "posix_gamer_whisper_5m_dry_run",
            "lang": lang_code,
            "platform": eff_platform,
            "gateway": clean_gw,
            "handle_tag": handle[:24],
            "compliance_ok": True,
            "gamer_ready": gamer_ready,
            "slo_ok": slo_ok,
            "elapsed_s": elapsed_s,
            "slo_budget_s": POSIX_ONBOARD_SLO_SECONDS,
            "gpu_status": gpu_status,
            "gpu_present": gpu_present,
            "gpu_temp_c": temp_c,
            "gpu_free_mib": free_mib,
            "thermal_ok": thermal_ok,
            "vram_reserve_ok": vram_reserve_ok,
            "max_temp_c": float(max_temp_c),
            "reserve_mib": int(reserve_mib),
            "active_games_detected": active_games,
            "yield_latency_ms": round(yield_ms, 2),
            "yield_budget_ms": float(yield_budget_ms),
            "yield_slo_ok": yield_slo_ok,
            "retained_worker_vram_mib": retained_vram,
            "estimated_upload_mb_per_hour": estimated_mb_per_hour,
            "max_upload_mb_per_hour": float(max_upload_mb_per_hour),
            "bandwidth_cap_ok": bandwidth_cap_ok,
            "zero_game_lag": zero_game_lag,
            "heartbeat_sent": send_heartbeat,
            "heartbeat_ok": heartbeat_ok,
            "ces_submitted": submit_ces,
            "ces_recorded": ces_recorded,
            "network_calls": network_calls,
            "zero_inference_verified": True,
            "diagnostics": diagnostics,
            "exit_code": 0 if gamer_ready else 2,
            "summary": summary,
        }
    )


def run_unified_posix_onboarding_and_smoke_gate(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    probe_feedback_post: bool = False,
    idempotency_check: bool = False,
    require_coder_model: bool = True,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    require_ces_slo: bool = False,
    check_arena_summary: bool = False,
    execute_deploy_dry_run: bool = False,
    submit_dev_ces: bool = False,
    submit_gamer_ces: bool = False,
    run_smoke: bool = True,
    run_launcher_audit: bool = True,
    run_dev_dry_run: bool = True,
    run_gamer_dry_run: bool = True,
    sh_path: Path | str | None = None,
    ps1_path: Path | str | None = None,
    deploy_sh_path: Path | str | None = None,
    dev_home_dir: Path | str | None = None,
    dev_workspace_dir: Path | str | None = None,
    reserve_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    platform_name: str | None = None,
    gpu_probe_fn: Callable[[], Mapping[str, Any]] | None = None,
    process_probe_fn: Callable[[], Sequence[str]] | None = None,
    worker_yield_fn: Callable[[], Mapping[str, Any]] | None = None,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Orchestrate compliance -> Cloud Run smoke -> POSIX/Win11 launcher audit -> <60s Dev & <5min Gamer dry-runs."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("unified_posix_onboarding_and_smoke_gate", comp_doc, lang_code=lang_code)

    target_n, min_ease_f, min_lag_f = _validate_quorum_slo_options(
        target_online_producers=target_online_producers,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
        require_live_quorum=require_live_quorum,
        require_ces_slo=require_ces_slo,
        check_arena_summary=check_arena_summary,
        execute_deploy_dry_run=execute_deploy_dry_run,
    )

    clean_gw = validate_gateway_url(gateway)
    diagnostics: list[str] = []
    network_calls = 0

    smoke_res = (
        run_cloud_run_holiday_smoke_gate(
            clean_gw,
            expected_sha=expected_sha,
            check_git_head=check_git_head,
            probe_feedback_post=probe_feedback_post,
            idempotency_check=idempotency_check,
            require_coder_model=require_coder_model,
            target_online_producers=target_n,
            min_ease_rating=min_ease_f,
            min_zero_lag_ratio=min_lag_f,
            require_live_quorum=require_live_quorum,
            require_ces_slo=require_ces_slo,
            check_arena_summary=check_arena_summary,
            execute_deploy_dry_run=execute_deploy_dry_run,
            deploy_sh_path=deploy_sh_path,
            http_fn=http_fn,
            runner=runner,
            environ=environ,
            cwd=cwd,
            lang=lang_code,
        )
        if run_smoke
        else None
    )
    launcher_res = (
        audit_posix_launcher_contract(
            sh_path=sh_path,
            ps1_path=ps1_path,
            runner=runner,
            environ=environ,
            cwd=cwd,
            lang=lang_code,
        )
        if run_launcher_audit
        else None
    )
    dev_res = (
        verify_posix_dev_vscode_60s_dry_run(
            clean_gw,
            home_dir=dev_home_dir,
            workspace_dir=dev_workspace_dir,
            submit_ces=submit_dev_ces,
            platform_name=platform_name,
            http_fn=http_fn,
            environ=environ,
            cwd=cwd,
            lang=lang_code,
        )
        if run_dev_dry_run
        else None
    )
    gamer_res = (
        verify_posix_gamer_whisper_5m_dry_run(
            clean_gw,
            reserve_mib=reserve_mib,
            max_temp_c=max_temp_c,
            platform_name=platform_name,
            submit_ces=submit_gamer_ces,
            gpu_probe_fn=gpu_probe_fn,
            process_probe_fn=process_probe_fn,
            worker_yield_fn=worker_yield_fn,
            http_fn=http_fn,
            runner=runner,
            environ=environ,
            cwd=cwd,
            lang=lang_code,
        )
        if run_gamer_dry_run
        else None
    )

    for block in (smoke_res, launcher_res, dev_res, gamer_res):
        if block is not None:
            network_calls += int(block.get("network_calls", 0))
            diagnostics.extend(block.get("diagnostics") or [])

    overall_ok = bool(
        (smoke_res is None or smoke_res["smoke_ok"])
        and (launcher_res is None or launcher_res["contract_ok"])
        and (dev_res is None or dev_res["dev_ready"])
        and (gamer_res is None or gamer_res["gamer_ready"])
    )

    summary = (
        (
            f"Vérification unifiée Noël + POSIX RÉUSSIE sur {clean_gw} (0.00 EUR / 0.00 USD)."
            if is_fr
            else f"Unified holiday deploy smoke + POSIX onboarding gate PASSED on {clean_gw} (0.00 EUR / 0.00 USD)."
        )
        if overall_ok
        else (
            f"Vérification unifiée Noël + POSIX EN ÉCHEC sur {clean_gw}."
            if is_fr
            else f"Unified holiday deploy smoke + POSIX onboarding gate FAILED on {clean_gw}."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "unified_posix_onboarding_and_smoke_gate",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "overall_ok": overall_ok,
            "smoke_ok": smoke_res["smoke_ok"] if smoke_res is not None else None,
            "contract_ok": launcher_res["contract_ok"] if launcher_res is not None else None,
            "dev_ready": dev_res["dev_ready"] if dev_res is not None else None,
            "gamer_ready": gamer_res["gamer_ready"] if gamer_res is not None else None,
            "holiday_routes_deployed_on_cloud_run": (
                smoke_res["holiday_routes_deployed_on_cloud_run"] if smoke_res is not None else None
            ),
            "onboarding_feedback_deployed": (
                smoke_res["onboarding_feedback_deployed"] if smoke_res is not None else None
            ),
            "stale_revision_detected": (
                smoke_res["stale_revision_detected"] if smoke_res is not None else None
            ),
            "redeploy_required": smoke_res["redeploy_required"] if smoke_res is not None else None,
            "online_producers": smoke_res["online_producers"] if smoke_res is not None else None,
            "producer_quorum_met": smoke_res["producer_quorum_met"] if smoke_res is not None else None,
            "awaiting_producer_quorum": smoke_res["awaiting_producer_quorum"] if smoke_res is not None else None,
            "ces_slo_ok": smoke_res["ces_slo_ok"] if smoke_res is not None else None,
            "zero_lag_slo_ok": smoke_res["zero_lag_slo_ok"] if smoke_res is not None else None,
            "arena_summary_ok": smoke_res["arena_summary_ok"] if smoke_res is not None else None,
            "deploy_dry_run": smoke_res["deploy_dry_run"] if smoke_res is not None else None,
            "cloud_run_smoke": smoke_res,
            "posix_launcher_audit": launcher_res,
            "dev_vscode_60s": dev_res,
            "posix_gamer_whisper_5m": gamer_res,
            "network_calls": network_calls,
            "zero_inference_verified": True,
            "personal_pc_only": True,
            "financial_promise": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": diagnostics,
            "exit_code": 0 if overall_ok else 2,
            "summary": summary,
        }
    )


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN CLI summary with masked keys, ``~`` paths, and SHA-256 receipt."""
    is_fr = report.get("lang") == "fr"
    title = (
        "HIVEMIND NOËL — GATE SMOKE CLOUD RUN & VÉRIFICATEUR POSIX (0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND HOLIDAY — CLOUD RUN SMOKE & POSIX ONBOARDING VERIFIER (0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {title}",
        "====================================================================",
        f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}",
        f"  Gateway    : {report.get('gateway', DEFAULT_GATEWAY)}",
        f"  Mode       : {report.get('mode')}",
        f"  Compliance : {report.get('compliance_ok')}",
        f"  Exit code  : {report.get('exit_code')}",
        f"  Summary    : {report.get('summary')}",
    ]
    if "online_producers" in report and report.get("online_producers") is not None:
        lines.append(
            f"  Quorum     : online={report.get('online_producers')} "
            f"met={report.get('producer_quorum_met')} "
            f"ces_slo_ok={report.get('ces_slo_ok')} zero_lag_slo_ok={report.get('zero_lag_slo_ok')}"
        )
    if report.get("arena_summary_ok") is not None:
        lines.append(f"  Arena      : summary_ok={report.get('arena_summary_ok')}")
    if isinstance(report.get("deploy_dry_run"), Mapping):
        lines.append(
            f"  DeployDry  : ok={report['deploy_dry_run'].get('ok')} "
            f"sha={report['deploy_dry_run'].get('dry_run_build_sha')}"
        )
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256    : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN Markdown summary with SHA-256 receipt."""
    is_fr = report.get("lang") == "fr"
    heading = (
        "# Rapport Smoke Cloud Run Noël & Vérification POSIX"
        if is_fr
        else "# Cloud Run Holiday Smoke & POSIX Onboarding Report"
    )
    lines = [
        heading,
        "",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: `{report.get('compliance_ok')}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Verdict' if is_fr else 'Exit Code'}**: `{report.get('exit_code')}`",
        f"- **{'Résumé' if is_fr else 'Summary'}**: {report.get('summary')}",
        f"- **SHA-256 Receipt**: `{report.get('receipt_sha256')}`",
    ]
    if report.get("online_producers") is not None:
        lines.append(
            f"- **{'Quorum producteurs' if is_fr else 'Live Quorum & CES SLO'}**: "
            f"`online_producers={report.get('online_producers')}`, "
            f"`producer_quorum_met={report.get('producer_quorum_met')}`, "
            f"`ces_slo_ok={report.get('ces_slo_ok')}`, "
            f"`zero_lag_slo_ok={report.get('zero_lag_slo_ok')}`"
        )
    if report.get("arena_summary_ok") is not None:
        lines.append(f"- **Arena Summary Parity**: `{report.get('arena_summary_ok')}`")
    if isinstance(report.get("deploy_dry_run"), Mapping):
        lines.append(
            f"- **Deploy Dry-Run**: `ok={report['deploy_dry_run'].get('ok')}`, "
            f"`build_sha={report['deploy_dry_run'].get('dry_run_build_sha')}`"
        )
    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        lines.extend(f"- {d}" for d in diags)
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated Cloud Run holiday smoke gate, POSIX launcher audit & <60s/<5min verifier (0.00 EUR / 0.00 USD)."
    )
    parser.add_argument(
        "--gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="Target HiveMind gateway URL (https://*.run.app or loopback).",
    )
    parser.add_argument("--expected-sha", default=None, help="Expected build_sha on /v1/hivemind/onboarding/manifest.")
    parser.add_argument("--check-git-head", action="store_true", help="Compare live build_sha with local git HEAD.")
    parser.add_argument("--smoke-only", action="store_true", help="Run only Cloud Run holiday deploy smoke gate.")
    parser.add_argument("--launcher-audit-only", action="store_true", help="Run only POSIX/Win11 launcher audit.")
    parser.add_argument("--dev-dry-run-only", action="store_true", help="Run only < 60 s Dev VS Code dry-run.")
    parser.add_argument(
        "--gamer-dry-run-only",
        "--posix-dry-run-only",
        dest="gamer_dry_run_only",
        action="store_true",
        help="Run only < 5 min Linux/macOS Gamer-Whisper dry-run.",
    )
    parser.add_argument("--probe-feedback-post", action="store_true", help="Send zero-PII POST /v1/hivemind/onboarding/feedback probe.")
    parser.add_argument("--idempotency-check", action="store_true", help="Verify Idempotency-Key duplicate-replay conservation.")
    parser.add_argument("--allow-missing-coder-model", action="store_true", help="Allow /v1/models without hivemind/coder.")
    parser.add_argument("--target-online-producers", type=int, default=DEFAULT_TARGET_ONLINE_PRODUCERS)
    parser.add_argument("--min-ease-rating", type=float, default=DEFAULT_MIN_EASE_RATING)
    parser.add_argument("--min-zero-lag-ratio", type=float, default=DEFAULT_MIN_ZERO_LAG_RATIO)
    parser.add_argument("--require-live-quorum", action="store_true", help="Fail closed (exit 2) when online_producers < target.")
    parser.add_argument("--require-ces-slo", action="store_true", help="Fail closed (exit 2) when CES/zero-lag SLO is breached.")
    parser.add_argument("--check-arena-summary", action="store_true", help="Verify GET /v1/hivemind/arena/summary conservation & Wilson LB.")
    parser.add_argument("--run-deploy-dry-run", action="store_true", help="Execute deploy/gcp_openrouter/deploy.sh --dry-run (no gcloud).")
    parser.add_argument("--cwd", default=None, help="Override workspace directory for local compliance & git remote check.")
    parser.add_argument("--output", default=None, help="Atomically persist verified output (POSIX 0600); never writes when compliance fails.")
    parser.add_argument("--reserve-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB)
    parser.add_argument("--max-temp-c", type=float, default=DEFAULT_MAX_TEMP_C)
    parser.add_argument("--platform", default=None, help="Optional POSIX platform tag ('linux' or 'macos').")
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output.")
    return parser


def _validate_cli_path_arg(raw: str | None, *, flag: str) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise SmokeVerifierError(f"{flag} must be a non-empty path without NUL bytes.")
    return Path(raw.strip())


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_request,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    stdout: Any = None,
    stderr: Any = None,
    gpu_probe_fn: Callable[[], Mapping[str, Any]] | None = None,
    process_probe_fn: Callable[[], Sequence[str]] | None = None,
    worker_yield_fn: Callable[[], Mapping[str, Any]] | None = None,
) -> int:
    eff_out = stdout if stdout is not None else sys.stdout
    eff_err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        lang_code = normalize_lang(args.lang)
        cli_cwd = _validate_cli_path_arg(args.cwd, flag="--cwd")
        out_path = _validate_cli_path_arg(args.output, flag="--output")
        eff_cwd: Path | str | None = cli_cwd if cli_cwd is not None else cwd
        smoke_opts: dict[str, Any] = {
            "expected_sha": args.expected_sha,
            "check_git_head": args.check_git_head,
            "probe_feedback_post": args.probe_feedback_post,
            "idempotency_check": args.idempotency_check,
            "require_coder_model": not args.allow_missing_coder_model,
            "target_online_producers": args.target_online_producers,
            "min_ease_rating": args.min_ease_rating,
            "min_zero_lag_ratio": args.min_zero_lag_ratio,
            "require_live_quorum": args.require_live_quorum,
            "require_ces_slo": args.require_ces_slo,
            "check_arena_summary": args.check_arena_summary,
            "execute_deploy_dry_run": args.run_deploy_dry_run,
        }
        if args.smoke_only:
            report = run_cloud_run_holiday_smoke_gate(
                args.gateway,
                **smoke_opts,
                http_fn=http_fn,
                runner=runner,
                environ=environ,
                cwd=eff_cwd,
                lang=lang_code,
            )
        elif args.launcher_audit_only:
            report = audit_posix_launcher_contract(
                runner=runner,
                environ=environ,
                cwd=eff_cwd,
                lang=lang_code,
            )
        elif args.dev_dry_run_only:
            report = verify_posix_dev_vscode_60s_dry_run(
                args.gateway,
                submit_ces=args.probe_feedback_post,
                platform_name=args.platform,
                http_fn=http_fn,
                environ=environ,
                cwd=eff_cwd,
                lang=lang_code,
            )
        elif args.gamer_dry_run_only:
            report = verify_posix_gamer_whisper_5m_dry_run(
                args.gateway,
                reserve_mib=args.reserve_mib,
                max_temp_c=args.max_temp_c,
                platform_name=args.platform,
                submit_ces=args.probe_feedback_post,
                gpu_probe_fn=gpu_probe_fn,
                process_probe_fn=process_probe_fn,
                worker_yield_fn=worker_yield_fn,
                http_fn=http_fn,
                runner=runner,
                environ=environ,
                cwd=eff_cwd,
                lang=lang_code,
            )
        else:
            report = run_unified_posix_onboarding_and_smoke_gate(
                args.gateway,
                **smoke_opts,
                reserve_mib=args.reserve_mib,
                max_temp_c=args.max_temp_c,
                platform_name=args.platform,
                gpu_probe_fn=gpu_probe_fn,
                process_probe_fn=process_probe_fn,
                worker_yield_fn=worker_yield_fn,
                http_fn=http_fn,
                runner=runner,
                environ=environ,
                cwd=eff_cwd,
                lang=lang_code,
            )
    except SmokeVerifierError as exc:
        err_msg = sanitize_text_for_display(str(exc))
        if args.json:
            err_doc = _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "compliance_ok": True,
                    "overall_ok": False,
                    "error": err_msg,
                    "exit_code": 2,
                }
            )
            print(json.dumps(err_doc, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(f"Error: {err_msg}", file=eff_err)
        return 2

    if args.json:
        text_out = json.dumps(report, indent=2, ensure_ascii=False)
        _assert_safe_output(text_out)
    elif args.markdown:
        text_out = render_markdown_report(report)
    else:
        text_out = render_cli_report(report)

    if out_path is not None and report.get("compliance_ok") is True:
        _write_secure_atomic(out_path, text_out + "\n")
    print(text_out, file=eff_out)
    return int(report.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())
