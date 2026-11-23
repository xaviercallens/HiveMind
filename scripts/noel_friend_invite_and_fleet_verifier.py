#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Friend-invite quick-launch generator, Cloud Run drift preflight, and live fleet activation verifier (`SLICE-SCALE-DYN-05`).

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library and proxy-free `urllib.request.ProxyHandler({})`:
  1. Personal-PC & Hobby-Only Compliance Gate (`evaluate_personal_pc_compliance`):
     strictly local inspection of `USERDNSDOMAIN`, `USERDOMAIN`, `HTTP_PROXY`, `HTTPS_PROXY`,
     `COMPUTERNAME`, `HOSTNAME`, workspace path, and `.git/config` / worktree remote URLs for
     corporate markers (`amadeus`, `corp`, `enterprise`, `intranet`, `internal`, `zscaler`,
     `netskope`, `bluecoat`, `fortinet`, `paloalto`) before writing invite files, probing Cloud Run,
     sending heartbeats, or posting CES feedback. Refuses closed with exit code 2 (`0` file writes,
     `0` network calls, `0` local env/path value leaks) and surfaces the bilingual FR/EN 100%
     personal home PC / `0.00 EUR / 0.00 USD` symbolic-points notice.
  2. Zero-Inference Cloud Run Revision-Drift Preflight & Friend-Invite Quick-Launch Generator
     (`check_cloud_run_invite_readiness`, `generate_friend_invite_pack`): probes `GET /health`,
     `GET /v1/hivemind/onboarding/manifest`, `GET /noel`, `GET /quick`, `GET /raid-noel`,
     `GET /v1/hivemind/onboarding/feedback/summary`, and `GET /v1/hivemind/producers/fleet` without
     ever touching GPU inference (`POST /v1/chat/completions` strictly blocked), detects
     `holiday_routes_deployed_on_cloud_run` vs `revision_drift_detected`, and renders concise FR/EN
     Signal/WhatsApp/Discord/Markdown invite cards with safe two-step download-then-run snippets
     (never curl-pipe-shell) plus an automatic hold/fallback notice when drift is detected.
  3. `<1.5s` Steam GPU Yield / `<=70.0 C` Preflight & Live Fleet + CES Activation Verifier
     (`verify_live_fleet_and_ces_activation`): runs a deterministic `<1.5s` Steam/DX12 yield &
     `<=70.0 C` quiet-fan preflight (or honest `'dev_api_only_no_nvidia_gpu'` status when no NVIDIA
     GPU is present without inventing fake GPU specs), optionally sends outbound
     `POST /v1/hivemind/opentrainer/heartbeat` (`OpenTrainerHeartbeatRequest` `extra='forbid'`),
     verifies live fleet progress against `GET /v1/hivemind/producers/fleet` +
     `GET /v1/hivemind/raid/status`, and submits (or safely degrades/spools on HTTP 404/429/5xx)
     zero-PII `POST /v1/hivemind/onboarding/feedback` (`OnboardingFeedbackRequest` `extra='forbid'`).
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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows fallback uses per-path RLock + atomic os.replace
    _fcntl = None  # type: ignore[assignment]

SCHEMA = "hivemind.noel_friend_invite_and_fleet_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_friend_invite_fleet_receipt"
READINESS_OBJECT = "hivemind.noel_cloud_run_invite_readiness"
INVITE_PACK_OBJECT = "hivemind.noel_friend_invite_pack"
ACTIVATION_OBJECT = "hivemind.noel_live_fleet_ces_activation"
PREFLIGHT_OBJECT = "hivemind.noel_steam_yield_thermal_preflight"
SPOOL_SCHEMA = "hivemind.noel_ces_feedback_spool/1"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_GUILD_TAG = "Team Sophia / Friends"
DEFAULT_CHANNEL = "signal"
DEFAULT_HOST_LABEL = "personal-pc"
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
DEFAULT_YIELD_BUDGET_MS = 1500.0
NON_WORKER_GPU_UTIL_THRESHOLD_PCT = 15.0
MIN_REPORTABLE_VRAM_TOTAL_MIB = 512
HTTP_TIMEOUT_S = 5.0
SUBPROCESS_TIMEOUT_S = 3.0
DEFAULT_MAX_SPOOL_ENTRIES = 128
DEFAULT_CREDENTIALS_REL = ".hivemind/credentials.json"
DEFAULT_CES_SPOOL_REL = ".hivemind/ces_feedback_spool.json"

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
VALID_CHANNELS: frozenset[str] = frozenset({"signal", "whatsapp", "discord", "markdown"})
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})

INVITE_READINESS_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
)

HOLIDAY_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel")
HOLIDAY_REQUIRED_ROUTES: tuple[str, ...] = (
    "/v1/hivemind/onboarding/manifest",
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/feedback/summary",
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

ALLOWED_ZERO_INFERENCE_POST_PATHS: frozenset[str] = frozenset(
    {
        "/v1/hivemind/opentrainer/heartbeat",
        "/v1/hivemind/producer/heartbeat",
        "/v1/hivemind/onboarding/feedback",
        "/v1/hivemind/register",
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

_INSPECTED_ENV_KEYS: tuple[str, ...] = (
    "USERDOMAIN",
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
    "100% personal home PCs & personal hobby/open-source projects only — "
    "Amadeus / Personal-PC rule: never work PCs or corporate code (0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement — "
    "règle Amadeus / PC perso : jamais de PC ni de code d'entreprise (0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_FR} | {COMPLIANCE_NOTICE_EN}"

SYMBOLIC_POINTS_NOTICE_EN = (
    "Symbolic non-monetary compute-reciprocity points only (0.00 EUR / 0.00 USD, zero commercial value)."
)
SYMBOLIC_POINTS_NOTICE_FR = (
    "Points symboliques non monétaires de réciprocité de calcul uniquement (0,00 EUR / 0,00 USD, sans valeur marchande)."
)
SYMBOLIC_POINTS_NOTICE_BILINGUAL = f"{SYMBOLIC_POINTS_NOTICE_FR} | {SYMBOLIC_POINTS_NOTICE_EN}"

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

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r'\bnonce="([A-Za-z0-9_+=/=-]{8,128})"')
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)"
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

_ALLOWED_FEEDBACK_KEYS: frozenset[str] = frozenset(
    {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
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

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{3,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{4,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PATH_PREFIX_HOME_RE = re.compile(
    r"^.*?(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

HttpFn = Callable[..., Any]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
GpuProbeFn = Callable[..., Any]
ProcessProbeFn = Callable[..., Iterable[str]]
WorkerYieldFn = Callable[..., Any]
ClockFn = Callable[[], float]

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


class InviteFleetVerifierError(ValueError):
    """Raised when arguments, host allowlists, zero-inference guards, or output safety checks fail."""


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


def _get_path_lock(resolved_path: Path) -> threading.RLock:
    key = str(resolved_path)
    with _PATH_LOCKS_GUARD:
        existing = _PATH_LOCKS.get(key)
        if existing is None:
            existing = threading.RLock()
            _PATH_LOCKS[key] = existing
        return existing


def normalize_lang(raw: str | None) -> str:
    """Normalize language selector to `'fr'` or `'en'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise InviteFleetVerifierError("Language must be 'fr' or 'en'.")
    cleaned = (raw or "fr").strip().lower().replace("_", "-")
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    raise InviteFleetVerifierError(f"Unsupported language {raw!r}; expected 'fr' or 'en'.")


def normalize_channel(raw: str | None) -> str:
    """Normalize friend-invite output channel to `'signal'`, `'whatsapp'`, `'discord'`, or `'markdown'`."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise InviteFleetVerifierError("Channel must be signal, whatsapp, discord, or markdown.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_CHANNELS:
        raise InviteFleetVerifierError(
            f"Unsupported channel {raw!r}; expected signal, whatsapp, discord, or markdown."
        )
    return cleaned


def normalize_profile(raw: str | None) -> str:
    """Normalize contributor onboarding profile to `'dev'` or `'gamer'`."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise InviteFleetVerifierError("Profile must be 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_PROFILES:
        raise InviteFleetVerifierError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def is_valid_hivemind_key(raw_key: object) -> bool:
    """Return True iff `raw_key` is a structurally valid `hm-...` or `hm_live_...` API key."""
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def mask_api_key(raw_key: object) -> str:
    """Mask any `hm-...` or `hm_live_...` token without leaking secret characters."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<redacted-key>"
    cleaned = raw_key.strip()
    if cleaned.startswith("hm_live_"):
        return "hm_live_***REDACTED***"
    if cleaned.startswith("hm-"):
        return "hm-***REDACTED***"
    return "<redacted-key>"


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Redact local user home directories to `'~'` or repo-relative paths so usernames never leak."""
    raw = str(target)
    if home is not None:
        explicit_home = str(home)
        if explicit_home and explicit_home not in ("/", "\\"):
            for variant in (
                explicit_home,
                explicit_home.replace("\\", "/"),
                explicit_home.replace("/", "\\"),
            ):
                if variant and raw.startswith(variant):
                    suffix = raw[len(variant) :]
                    if not suffix or suffix.startswith(("/", "\\")):
                        return "~" + suffix.replace("\\", "/")

    if not re.match(r"^[A-Za-z]:[\\/]", raw) and Path(raw).is_absolute():
        try:
            rel = Path(raw).resolve().relative_to(REPO_ROOT)
            return str(rel).replace("\\", "/")
        except (OSError, ValueError):
            pass

    default_home = str(Path.home())
    if default_home and default_home not in ("/", "\\"):
        for variant in (
            default_home,
            default_home.replace("\\", "/"),
            default_home.replace("/", "\\"),
        ):
            if variant and raw.startswith(variant):
                suffix = raw[len(variant) :]
                if not suffix or suffix.startswith(("/", "\\")):
                    return "~" + suffix.replace("\\", "/")

    prefixed = _PATH_PREFIX_HOME_RE.sub("~", raw)
    if prefixed != raw:
        return prefixed.replace("\\", "/")
    return raw


def sanitize_text_for_display(text: str, *, home: Path | None = None) -> str:
    """Scrub `hm-...`/`hm_live_...` tokens, Bearer credentials, emails, non-loopback IPs, and home paths."""
    scrubbed = str(text)
    if home is not None:
        explicit_home = str(home)
        if explicit_home and explicit_home not in ("/", "\\"):
            for variant in (
                explicit_home,
                explicit_home.replace("\\", "/"),
                explicit_home.replace("/", "\\"),
            ):
                if variant:
                    scrubbed = scrubbed.replace(variant, "~")
    default_home = str(Path.home())
    if default_home and default_home not in ("/", "\\"):
        for variant in (
            default_home,
            default_home.replace("\\", "/"),
            default_home.replace("/", "\\"),
        ):
            if variant:
                scrubbed = scrubbed.replace(variant, "~")
    scrubbed = _HOME_PATH_RE.sub("~", scrubbed).replace("\\Users\\", "/Users/")
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0) == "127.0.0.1" else "<redacted-ip>",
        scrubbed,
    )
    scrubbed = _FORBIDDEN_KEY_RE.sub('"metric_val": 0', scrubbed)
    return scrubbed


def scan_output_safety(text: str) -> list[str]:
    """Verify CLI, invite card, Markdown, or JSON output has zero PII, raw keys, home paths, or `'score': <digit>`."""
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
        raise InviteFleetVerifierError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _find_corporate_markers(value: str) -> list[str]:
    if not value:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(value)]


def sanitize_guild_tag(raw: str | None, *, home: Path | None = None) -> str:
    """Sanitize friendly guild tag (`2..64` printable chars, zero secrets/PII, zero corporate markers)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_GUILD_TAG
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise InviteFleetVerifierError("guild_tag must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise InviteFleetVerifierError("guild_tag must not contain control characters.")
    if _find_corporate_markers(raw):
        raise InviteFleetVerifierError(
            "guild_tag must not reference corporate or employer identifiers (Amadeus / Personal-PC rule)."
        )
    scrubbed = sanitize_text_for_display(raw.strip(), home=home)
    cleaned = re.sub(r"[^A-Za-z0-9À-ÿ /_.+#&-]+", "", scrubbed).strip()[:64]
    if len(cleaned) < 2:
        return DEFAULT_GUILD_TAG
    return cleaned


def sanitize_host_label(raw: str | None, *, home: Path | None = None) -> str:
    """Produce a clean `1..64` ASCII personal host label free of PII, secrets, IPs, and home paths."""
    if raw is None or isinstance(raw, bool) or not str(raw).strip():
        return DEFAULT_HOST_LABEL
    if _find_corporate_markers(str(raw)):
        raise InviteFleetVerifierError(
            "host_label must not contain corporate identifiers (Amadeus / Personal-PC rule)."
        )
    scrubbed = sanitize_text_for_display(str(raw).strip(), home=home)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", scrubbed).strip("-._")[:64]
    return cleaned if cleaned else DEFAULT_HOST_LABEL


def sanitize_build_sha(raw: object) -> str:
    """Sanitize a build SHA candidate to lowercase alphanumeric/hex or `'unknown'`."""
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
    """Validate an explicit `--expected-sha` argument."""
    sanitized = sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise InviteFleetVerifierError(
            "Expected SHA must be a 6..64 character alphanumeric/hex string (never 'unknown' or a secret)."
        )
    return sanitized


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Compare live Cloud Run `build_sha` against expected SHA (exact or >=7-char hex prefix)."""
    clean_live = sanitize_build_sha(live_sha)
    clean_exp = sanitize_build_sha(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_RE.fullmatch(clean_live) and _HEX_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def sanitize_cloud_run_revision(raw: object) -> str | None:
    """Sanitize a Cloud Run revision name (`K_REVISION`) or return `None`."""
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


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is `*.run.app`, `*.googleapis.com`, or loopback (`127.0.0.1`/`localhost`/`::1`)."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h) or _GOOGLEAPIS_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app`, `https://*.googleapis.com`, or loopback."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise InviteFleetVerifierError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise InviteFleetVerifierError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise InviteFleetVerifierError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise InviteFleetVerifierError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise InviteFleetVerifierError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise InviteFleetVerifierError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise InviteFleetVerifierError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise InviteFleetVerifierError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise InviteFleetVerifierError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app, https://*.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise InviteFleetVerifierError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise InviteFleetVerifierError("Remote Cloud Run / Google APIs URL must use https.")
    return cleaned


def _extract_git_remote_urls(workspace: Path) -> list[str]:
    """Read `.git/config` (including git worktree `gitdir:` + `commondir` indirection) strictly on local disk."""
    git_entry = workspace / ".git"
    cfg_paths: list[Path] = [git_entry / "config"]
    try:
        if git_entry.is_file():
            lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines and lines[0].strip().lower().startswith("gitdir:"):
                raw_gitdir = lines[0].split(":", 1)[1].strip()
                gitdir = (
                    Path(raw_gitdir)
                    if Path(raw_gitdir).is_absolute()
                    else (workspace / raw_gitdir).resolve()
                )
                commondir_file = gitdir / "commondir"
                common_rel = (
                    commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                    if commondir_file.is_file()
                    else "."
                )
                cfg_paths = [gitdir / "config", (gitdir / common_rel).resolve() / "config"]
        raw_texts = [p.read_text(encoding="utf-8", errors="replace") for p in cfg_paths if p.is_file()]
    except OSError:
        return []
    return [
        (m1 or m2).strip().strip('"')
        for txt in raw_texts
        for m1, m2 in _GIT_REMOTE_URL_RE.findall(txt)
        if (m1 or m2).strip().strip('"')
    ]


def evaluate_personal_pc_compliance(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "fr",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only Compliance Gate (0 network calls, 0 file writes, 0 value leaks)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = env if env is not None else os.environ
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    upper_env: dict[str, str] = {
        str(k).upper(): str(v).strip() for k, v in env_map.items() if str(v).strip()
    }

    indicators: list[str] = []
    matched_markers: set[str] = set()

    if any(k.startswith("AMADEUS") for k in upper_env):
        indicators.append("env:AMADEUS_ENV")
        matched_markers.add("amadeus")

    if upper_env.get("ZSCALER_CERT_PATH"):
        indicators.append("env:ZSCALER_CERT_PATH")
        matched_markers.add("zscaler")

    dns_val = upper_env.get("USERDNSDOMAIN", "")
    if dns_val:
        indicators.append("env:USERDNSDOMAIN")
        dns_hits = _find_corporate_markers(dns_val)
        matched_markers.update(dns_hits or ["corp"])

    for env_key in _INSPECTED_ENV_KEYS:
        val = upper_env.get(env_key, "")
        hits = _find_corporate_markers(val)
        if hits:
            indicators.append(f"env:{env_key}")
            matched_markers.update(hits)

    cwd_hits = _find_corporate_markers(str(workspace))
    for hit in cwd_hits:
        tag = f"workspace_path:{hit}"
        if tag not in indicators:
            indicators.append(tag)
        matched_markers.add(hit)

    git_hits = sorted(
        {
            hit
            for remote_url in _extract_git_remote_urls(workspace)
            for hit in _find_corporate_markers(remote_url)
        }
    )
    for hit in git_hits:
        indicators.append(f"git_remote:{hit}")
        matched_markers.add(hit)

    compliance_ok = len(indicators) == 0
    joined = ", ".join(indicators)
    summary = (
        (
            f"Conformité PC personnel vérifiée localement ({FINANCIAL_PROMISE}). "
            f"{COMPLIANCE_NOTICE_BILINGUAL} | {SYMBOLIC_POINTS_NOTICE_BILINGUAL}"
            if is_fr
            else f"Personal home PC compliance verified locally ({FINANCIAL_PROMISE}). "
            f"{COMPLIANCE_NOTICE_BILINGUAL} | {SYMBOLIC_POINTS_NOTICE_BILINGUAL}"
        )
        if compliance_ok
        else (
            f"REFUS DE CONFORMITÉ ({joined}) : environnement professionnel / d'entreprise détecté "
            f"(0 écriture fichier, 0 appel réseau). {COMPLIANCE_NOTICE_BILINGUAL} | {SYMBOLIC_POINTS_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE BLOCKED ({joined}): corporate machine or git remote indicator detected "
            f"(0 file writes, 0 network calls). {COMPLIANCE_NOTICE_BILINGUAL} | {SYMBOLIC_POINTS_NOTICE_BILINGUAL}"
        )
    )

    return _attach_receipt(
        {
            "compliance_ok": compliance_ok,
            "status": "compliant" if compliance_ok else "blocked_corporate_environment",
            "corporate_detected": not compliance_ok,
            "corporate_indicators": indicators,
            "matched_markers": sorted(matched_markers),
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "symbolic_points_only": True,
            "network_calls": 0,
            "file_writes": 0,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "symbolic_points_notice": SYMBOLIC_POINTS_NOTICE_BILINGUAL,
            "summary": summary,
        }
    )


def _compliance_blocked_report(
    mode: str,
    gateway: str,
    compliance_doc: Mapping[str, Any],
    *,
    lang_code: str = "fr",
) -> dict[str, Any]:
    """Build a fail-closed report when `evaluate_personal_pc_compliance` detects a corporate environment."""
    safe_gw = DEFAULT_GATEWAY
    with suppress(InviteFleetVerifierError):
        safe_gw = validate_gateway_url(gateway)
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": mode,
            "lang": lang_code,
            "gateway": safe_gw,
            "compliance_ok": False,
            "exit_code": 2,
            "invite_ready": False,
            "share_approved": False,
            "invite_hold_active": True,
            "holiday_routes_deployed_on_cloud_run": False,
            "revision_drift_detected": False,
            "activation_ok": False,
            "network_calls": 0,
            "file_writes": 0,
            "zero_inference_verified": True,
            "diagnostics": [str(compliance_doc["summary"])],
            "compliance": dict(compliance_doc),
        }
    )


def _detect_platform_tag(system_name: str | None = None) -> str:
    raw = (system_name if system_name is not None else platform.system()).strip().lower()
    if raw in ("windows", "win32", "nt"):
        return "windows"
    if raw == "darwin":
        return "macos"
    if raw == "linux":
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-_")[:24]
    return cleaned if (cleaned and _PLATFORM_RE.fullmatch(cleaned)) else "unknown"


def validate_onboarding_feedback_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate zero-PII CES feedback payload strictly matching `OnboardingFeedbackRequest` (`extra='forbid'`)."""
    if not isinstance(payload, Mapping):
        raise InviteFleetVerifierError("Feedback payload must be a JSON object.")
    extra_keys = set(payload.keys()) - _ALLOWED_FEEDBACK_KEYS
    if extra_keys:
        joined = ", ".join(sorted(str(k) for k in extra_keys))
        raise InviteFleetVerifierError(
            f"Forbidden extra keys in feedback payload (`extra='forbid'`): {joined}."
        )
    clean_profile = normalize_profile(payload.get("profile"))  # type: ignore[arg-type]
    raw_ease = payload.get("ease_rating")
    if isinstance(raw_ease, bool) or not isinstance(raw_ease, int) or not (1 <= raw_ease <= 5):
        raise InviteFleetVerifierError("ease_rating must be an integer between 1 and 5.")
    raw_lag = payload.get("zero_game_lag")
    if not isinstance(raw_lag, bool):
        raise InviteFleetVerifierError("zero_game_lag must be a boolean.")
    raw_note = payload.get("note", "")
    if not isinstance(raw_note, str) or raw_note != "":
        raise InviteFleetVerifierError("CES feedback `note` must be empty ('') to enforce zero-PII.")
    raw_platform = payload.get("platform", "unknown")
    if (
        not isinstance(raw_platform, str)
        or not _PLATFORM_RE.fullmatch(raw_platform)
        or any(m in raw_platform.lower() for m in _SECRET_VALUE_MARKERS)
    ):
        raise InviteFleetVerifierError("CES feedback `platform` must be a clean 1..24 char ASCII identifier.")
    entry = {
        "profile": clean_profile,
        "ease_rating": int(raw_ease),
        "zero_game_lag": bool(raw_lag),
        "platform": raw_platform,
        "note": "",
    }
    _assert_safe_output(json.dumps(entry, sort_keys=True))
    return entry


def validate_opentrainer_heartbeat_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate outbound heartbeat payload strictly matching `OpenTrainerHeartbeatRequest` (`extra='forbid'`)."""
    if not isinstance(payload, Mapping):
        raise InviteFleetVerifierError("OpenTrainer heartbeat payload must be a JSON object.")
    extra_keys = set(payload.keys()) - _ALLOWED_OPENTRAINER_HB_KEYS
    if extra_keys:
        joined = ", ".join(sorted(str(k) for k in extra_keys))
        raise InviteFleetVerifierError(
            f"Forbidden extra keys in OpenTrainer heartbeat payload (`extra='forbid'`): {joined}."
        )
    clean_host = sanitize_host_label(payload.get("host_label"))  # type: ignore[arg-type]
    net_mode = payload.get("network_mode", "outbound-https-relay")
    if net_mode != "outbound-https-relay":
        raise InviteFleetVerifierError("OpenTrainer heartbeat `network_mode` must be 'outbound-https-relay'.")

    raw_gpu = payload.get("gpu_name")
    clean_gpu: str | None = None
    if raw_gpu is not None:
        if isinstance(raw_gpu, bool) or not isinstance(raw_gpu, str) or not raw_gpu.strip():
            raise InviteFleetVerifierError("gpu_name must be a non-empty string or None.")
        clean_gpu = sanitize_text_for_display(raw_gpu.strip())[:128]

    raw_tot = payload.get("vram_total_mib")
    clean_tot: int | None = None
    if raw_tot is not None:
        if isinstance(raw_tot, bool) or not isinstance(raw_tot, int) or raw_tot < MIN_REPORTABLE_VRAM_TOTAL_MIB:
            raise InviteFleetVerifierError(f"vram_total_mib must be None or int >= {MIN_REPORTABLE_VRAM_TOTAL_MIB}.")
        clean_tot = int(raw_tot)

    raw_free = payload.get("vram_free_mib")
    clean_free: int | None = None
    if raw_free is not None:
        if isinstance(raw_free, bool) or not isinstance(raw_free, int) or raw_free < 0:
            raise InviteFleetVerifierError("vram_free_mib must be None or int >= 0.")
        clean_free = min(int(raw_free), clean_tot) if clean_tot is not None else int(raw_free)

    validated: dict[str, Any] = {
        "host_label": clean_host,
        "gpu_name": clean_gpu,
        "vram_total_mib": clean_tot,
        "vram_free_mib": clean_free,
        "network_mode": "outbound-https-relay",
    }
    if "training_stack" in payload and isinstance(payload["training_stack"], str) and payload["training_stack"].strip():
        validated["training_stack"] = sanitize_text_for_display(payload["training_stack"].strip())[:128]
    if "active_campaign" in payload and isinstance(payload["active_campaign"], str) and payload["active_campaign"].strip():
        validated["active_campaign"] = sanitize_text_for_display(payload["active_campaign"].strip())[:128]

    _assert_safe_output(json.dumps(validated, sort_keys=True))
    return validated


def _assert_zero_inference_request(method: str, url: str) -> tuple[str, str]:
    """Enforce allowed gateway origin, zero LLM inference (`completions` blocked), and method/path allowlist."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise InviteFleetVerifierError(f"Unsupported HTTP method {verb!r}.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    path = parts.path or "/"
    low_path = path.lower()
    if "completions" in low_path or "hivemind/coder" in low_path:
        raise InviteFleetVerifierError(
            f"Zero-inference invariant violated: {verb} {path} is strictly forbidden."
        )
    if verb == "POST" and path not in ALLOWED_ZERO_INFERENCE_POST_PATHS:
        raise InviteFleetVerifierError(
            f"Zero-inference invariant violated: HTTP POST {path} is not in the allowlist."
        )
    return origin, path


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib HTTP helper (`ProxyHandler({})`) for zero-inference GETs, heartbeats, and CES POSTs."""
    verb = (method or "").strip().upper()
    _, path = _assert_zero_inference_request(verb, url)
    if payload is not None and verb != "POST":
        raise InviteFleetVerifierError("GET/HEAD requests never send JSON request bodies.")

    validated_payload: dict[str, Any] | None = None
    if verb == "POST":
        if path == "/v1/hivemind/onboarding/feedback":
            validated_payload = validate_onboarding_feedback_payload(payload)
        elif path == "/v1/hivemind/opentrainer/heartbeat":
            validated_payload = validate_opentrainer_heartbeat_payload(payload)
        elif payload is not None:
            validated_payload = dict(payload)

    body_bytes = (
        json.dumps(validated_payload, sort_keys=True).encode("utf-8")
        if validated_payload is not None
        else None
    )

    clean_headers: dict[str, str] = {}
    for hk, hv in (headers or {}).items():
        ks = str(hk).strip()
        vs = str(hv).strip()
        if any(ord(c) < 32 or ord(c) == 127 for c in ks + vs):
            raise InviteFleetVerifierError("HTTP headers must not contain control characters.")
        clean_headers[ks] = vs

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-friend-invite-fleet-verifier/1.0",
        **({"Content-Type": "application/json"} if body_bytes is not None else {}),
        **clean_headers,
    }

    bounded_timeout = min(max(0.2, float(timeout_s)), 15.0)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            raw = resp.read(524_288).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            content_type = resp_headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        raw_err = exc.read(65_536).decode("utf-8", errors="replace")
        try:
            parsed_err = json.loads(raw_err) if raw_err.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": sanitize_text_for_display(raw_err[:200])}
        return int(exc.code), (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {
            "network_error": type(exc).__name__,
            "detail": sanitize_text_for_display(str(reason)[:180]),
        }

    if not raw.strip():
        return status_code, {
            "content_type": content_type,
            "text_length": 0,
            "html_length": 0,
            "headers": resp_headers,
        }
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return status_code, {
            "content_type": content_type,
            "text_length": len(raw),
            "html_length": len(raw),
            "html": raw,
            "headers": resp_headers,
        }
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def _invoke_http_callable(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Validate URL & zero-inference guard, then invoke a 2-, 3-, or 4-arg HTTP callable safely."""
    verb = (method or "").strip().upper()
    _, path = _assert_zero_inference_request(verb, url)

    clean_payload: dict[str, Any] | None = payload
    if verb == "POST" and path == "/v1/hivemind/onboarding/feedback":
        clean_payload = validate_onboarding_feedback_payload(payload)
    elif verb == "POST" and path == "/v1/hivemind/opentrainer/heartbeat":
        clean_payload = validate_opentrainer_heartbeat_payload(payload)

    sig = inspect.signature(http_fn)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name != "timeout_s"
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    extra_kw: dict[str, Any] = {}
    if timeout_s is not None and "timeout_s" in sig.parameters:
        extra_kw["timeout_s"] = timeout_s

    try:
        if has_varargs or len(positional) >= 4:
            raw_res = http_fn(verb, url, clean_payload, headers, **extra_kw)
        elif len(positional) == 3:
            raw_res = http_fn(verb, url, clean_payload, **extra_kw)
        else:
            raw_res = http_fn(verb, url, **extra_kw)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {
            "network_error": type(exc).__name__,
            "detail": sanitize_text_for_display(str(exc)[:180]),
        }

    raw_hdrs = (
        raw_res[2]
        if (isinstance(raw_res, tuple) and len(raw_res) == 3 and isinstance(raw_res[2], Mapping))
        else {}
    )
    resp_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()}
    status_code, body = int(raw_res[0]), raw_res[1]

    if isinstance(body, Mapping):
        norm_body = {**({"headers": resp_headers} if resp_headers else {}), **dict(body)}
    elif isinstance(body, str):
        norm_body = {
            "text_length": len(body),
            "html_length": len(body),
            "html": body,
            **({"headers": resp_headers} if resp_headers else {}),
        }
    else:
        norm_body = {"data": body}
    return status_code, norm_body


def _write_file_secure_atomic(path: Path, content: str, *, posix_mode: int = 0o600) -> None:
    """Atomically write UTF-8 `content` to `path` with `0600` POSIX permissions from file creation."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt" and path.parent.name == ".hivemind":
        with suppress(OSError):
            os.chmod(path.parent, 0o700)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp_path, flags, posix_mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(tmp_path, posix_mode)
        os.replace(tmp_path, path)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def _validate_manifest_body(
    body: Mapping[str, Any],
    *,
    lang: str,
) -> tuple[bool, str, str | None, list[str], list[str], list[str]]:
    """Validate `/v1/hivemind/onboarding/manifest` `build_sha`, `holiday_routes`, and `downloadable_scripts`."""
    is_fr = lang == "fr"
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
            f"Routes Noël absentes dans `holiday_routes` : {joined_r}."
            if is_fr
            else f"Manifest `holiday_routes` missing required endpoints: {joined_r}."
        )

    raw_scripts = body.get("downloadable_scripts")
    if isinstance(raw_scripts, list) and all(isinstance(s, str) for s in raw_scripts):
        missing_scripts = [s for s in EXPECTED_MANIFEST_SCRIPTS if s not in raw_scripts]
    else:
        missing_scripts = list(EXPECTED_MANIFEST_SCRIPTS)
    if missing_scripts:
        joined_s = ", ".join(missing_scripts)
        issues.append(
            f"Scripts absents dans `downloadable_scripts` : {joined_s}."
            if is_fr
            else f"Manifest `downloadable_scripts` missing allowlisted scripts: {joined_s}."
        )

    return len(issues) == 0, live_sha, live_rev, missing_routes, missing_scripts, issues


def _inspect_holiday_html_contract(
    path: str,
    body: Mapping[str, Any],
    seen_nonces: set[str],
    *,
    lang: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Inspect `/noel`, `/quick`, and `/raid-noel` per-request CSP nonces, DOM safety, and 0.00 EUR/USD copy."""
    is_fr = lang == "fr"
    html_text = str(body.get("html") or "")
    hdrs = body.get("headers") if isinstance(body.get("headers"), Mapping) else {}
    csp_hdr = str(hdrs.get("content-security-policy") or body.get("csp") or "")

    has_doctype = "<!doctype html" in html_text.lower()
    attr_nonces = _HTML_ATTR_NONCE_RE.findall(html_text)
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    nonce = hdr_nonces[0] if hdr_nonces else (attr_nonces[0] if attr_nonces else None)

    csp_ok = bool(
        attr_nonces
        and nonce is not None
        and set(attr_nonces) == {nonce}
        and (
            not csp_hdr
            or (hdr_nonces and set(hdr_nonces) == {nonce} and "default-src 'none'" in csp_hdr.lower())
        )
        and nonce not in seen_nonces
    )
    if nonce is not None:
        seen_nonces.add(nonce)

    sanitized_html = _HTML_ATTR_NONCE_RE.sub('nonce=""', html_text)
    dom_clean = not (_FORBIDDEN_DOM_SINK_RE.search(sanitized_html) or _STATIC_FIG_UNIT_RE.search(sanitized_html))
    low_html = html_text.lower()
    compliance_banner_ok = (
        (
            "100% personal" in low_html
            or "100 % personnel" in low_html
            or "100 % sur pc personnel" in low_html
            or "100 % pc perso" in low_html
        )
        and "agpl-3.0" in low_html
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
    )

    checks = (
        (
            has_doctype,
            f"Contrat HTML {path} invalide : `<!DOCTYPE html>` absent."
            if is_fr
            else f"HTML contract failed on {path}: missing `<!DOCTYPE html>` document.",
        ),
        (
            csp_ok,
            f"Contrat CSP {path} invalide : nonce CSP par requête absent, incohérent ou réutilisé."
            if is_fr
            else f"CSP contract failed on {path}: missing, mismatched, or reused per-request CSP nonce.",
        ),
        (
            dom_clean,
            f"Sécurité DOM {path} violée : sink HTML non sûr ou figure statique détectée."
            if is_fr
            else f"HTML safety violated on {path}: unsafe DOM sink or static metric figure detected.",
        ),
        (
            compliance_banner_ok,
            f"Mention conformité PC perso / 0,00 EUR absente sur {path}."
            if is_fr
            else f"Personal-PC / 0.00 EUR compliance notice missing on {path}.",
        ),
    )
    issues = [msg for passed, msg in checks if not passed]
    contract_ok = len(issues) == 0
    return (
        contract_ok,
        {
            "path": path,
            "ok": contract_ok,
            "has_doctype": has_doctype,
            "csp_nonce_verified": csp_ok,
            "csp_nonce_sha256_prefix": hashlib.sha256(nonce.encode("utf-8")).hexdigest()[:16] if nonce else None,
            "dom_clean": dom_clean,
            "personal_pc_banner_present": compliance_banner_ok,
        },
        issues,
    )


def _inspect_feedback_summary_schema(
    body: Mapping[str, Any],
    *,
    lang: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate `GET /v1/hivemind/onboarding/feedback/summary` schema and conservation invariants."""
    is_fr = lang == "fr"
    raw_total = body.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    total_val = int(raw_total) if total_ok else -1

    has_ease = "average_ease_rating" in body or "mean_ease_rating" in body
    raw_ease = (
        body.get("average_ease_rating")
        if "average_ease_rating" in body
        else body.get("mean_ease_rating")
    )
    ease_ok = bool(
        has_ease
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
    schema_ok = bool(total_ok and ease_ok and lag_ok and prof_ok)
    issues = (
        []
        if schema_ok
        else [
            "Schéma `/v1/hivemind/onboarding/feedback/summary` invalide."
            if is_fr
            else "Feedback summary `/v1/hivemind/onboarding/feedback/summary` schema mismatch."
        ]
    )
    return (
        schema_ok,
        {
            "schema_valid": schema_ok,
            "total_responses": total_val if total_ok else None,
            "average_ease_rating": ease_val,
            "mean_ease_rating": ease_val,
            "zero_game_lag_ratio": lag_val,
            "by_profile": {"dev": int(d_raw), "gamer": int(g_raw)} if prof_ok else None,  # type: ignore[arg-type]
        },
        issues,
    )


def _inspect_producer_fleet_schema(
    body: Mapping[str, Any],
    *,
    lang: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate `GET /v1/hivemind/producers/fleet` zero-inference connectivity summary."""
    is_fr = lang == "fr"
    obj_ok = body.get("object") in (FLEET_OBJECT, None)
    int_fields = ("total_hiveminders", "live_hiveminders", "distinct_physical_gpus", "live_endpoints")
    counts_ok = all(
        not isinstance(body.get(field), bool)
        and isinstance(body.get(field), int)
        and int(body[field]) >= 0
        for field in int_fields
    )
    models_ok = isinstance(body.get("live_models"), list)
    schema_ok = bool(obj_ok and counts_ok and models_ok)
    issues = (
        []
        if schema_ok
        else [
            "Schéma `/v1/hivemind/producers/fleet` invalide."
            if is_fr
            else "Producer fleet `/v1/hivemind/producers/fleet` schema mismatch."
        ]
    )
    normalized = (
        {
            "schema_valid": True,
            "total_hiveminders": int(body["total_hiveminders"]),
            "live_hiveminders": int(body["live_hiveminders"]),
            "distinct_physical_gpus": int(body["distinct_physical_gpus"]),
            "live_endpoints": int(body["live_endpoints"]),
            "live_models_count": len(body["live_models"]),
        }
        if schema_ok
        else {
            "schema_valid": False,
            "total_hiveminders": None,
            "live_hiveminders": None,
            "distinct_physical_gpus": None,
            "live_endpoints": None,
            "live_models_count": None,
        }
    )
    return schema_ok, normalized, issues


def _http_route_failure_diag(path: str, status_code: int, *, lang: str) -> str:
    is_fr = lang == "fr"
    if status_code == 404:
        return (
            f"GET {path} a répondu HTTP 404 : révision Cloud Run antérieure à PR #949 (dérive de révision détectée)."
            if is_fr
            else f"GET {path} returned HTTP 404: Cloud Run revision predates PR #949 (revision drift detected)."
        )
    if status_code == 429:
        return (
            f"GET {path} limité par quota HTTP 429 pendant la vérification pré-invitation."
            if is_fr
            else f"GET {path} rate-limited with HTTP 429 during invite readiness preflight."
        )
    if status_code == 0:
        return (
            f"GET {path} injoignable (délai réseau dépassé ou passerelle hors ligne)."
            if is_fr
            else f"GET {path} unreachable (gateway network/timeout error)."
        )
    return (
        f"GET {path} en échec avec HTTP {status_code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} failed with HTTP {status_code} (expected HTTP 200)."
    )


def check_cloud_run_invite_readiness(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    http_fn: HttpFn = default_http_json,
    lang: str = "fr",
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Zero-inference Cloud Run revision-drift preflight across all 7 onboarding, holiday, CES & fleet routes."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        return _compliance_blocked_report(
            "check_drift",
            gateway,
            comp_doc,
            lang_code=lang_code,
        )

    clean_gw = validate_gateway_url(gateway)
    resolved_expected_sha = (
        validate_expected_sha(expected_sha)
        if (expected_sha is not None and str(expected_sha).strip())
        else None
    )

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = []

    manifest_valid = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    missing_manifest_routes: list[str] = []
    missing_manifest_scripts: list[str] = []

    seen_nonces: set[str] = set()
    seen_nonces_raw: list[str] = []
    html_contracts: dict[str, dict[str, Any]] = {}
    html_contract_ok = True

    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {
        "schema_valid": False,
        "total_responses": None,
        "average_ease_rating": None,
        "mean_ease_rating": None,
        "zero_game_lag_ratio": None,
        "by_profile": None,
    }
    producer_fleet_ok = False
    producer_fleet_doc: dict[str, Any] = {
        "schema_valid": False,
        "total_hiveminders": None,
        "live_hiveminders": None,
        "distinct_physical_gpus": None,
        "live_endpoints": None,
        "live_models_count": None,
    }

    for path, expected_kind, group in INVITE_READINESS_ROUTES:
        code, body = _invoke_http_callable(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if code == 200 and path == "/health":
            if body.get("status") != "ok":
                ok = False
                diag = (
                    "GET /health a répondu HTTP 200 sans `status == 'ok'`."
                    if is_fr
                    else "GET /health returned HTTP 200 without `status == 'ok'`."
                )
                diagnostics.append(diag)
        elif code == 200 and path == "/v1/hivemind/onboarding/manifest":
            (
                manifest_valid,
                live_build_sha,
                cloud_run_revision,
                missing_manifest_routes,
                missing_manifest_scripts,
                manifest_issues,
            ) = _validate_manifest_body(body, lang=lang_code)
            if not manifest_valid:
                ok = False
                diag = manifest_issues[0]
                diagnostics.extend(manifest_issues)
        elif code == 200 and expected_kind == "html":
            before_count = len(seen_nonces)
            h_ok, h_doc, h_issues = _inspect_holiday_html_contract(
                path,
                body,
                seen_nonces,
                lang=lang_code,
            )
            if len(seen_nonces) > before_count and h_doc.get("csp_nonce_sha256_prefix"):
                seen_nonces_raw.append(str(h_doc["csp_nonce_sha256_prefix"]))
            html_contracts[path] = h_doc
            html_contract_ok = html_contract_ok and h_ok
            if not h_ok:
                ok = False
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        elif code == 200 and path == "/v1/hivemind/onboarding/feedback/summary":
            feedback_summary_ok, feedback_summary_doc, fb_issues = _inspect_feedback_summary_schema(
                body,
                lang=lang_code,
            )
            if not feedback_summary_ok:
                ok = False
                diag = fb_issues[0]
                diagnostics.extend(fb_issues)
        elif code == 200 and path == "/v1/hivemind/producers/fleet":
            producer_fleet_ok, producer_fleet_doc, fl_issues = _inspect_producer_fleet_schema(
                body,
                lang=lang_code,
            )
            if not producer_fleet_ok:
                ok = False
                diag = fl_issues[0]
                diagnostics.extend(fl_issues)

        if code != 200 and expected_kind == "html":
            html_contract_ok = False

        if not ok:
            failed_routes.append(path)
            if code == 404:
                stale_404_routes.append(path)
            if not diag:
                diag = _http_route_failure_diag(path, code, lang=lang_code)
                diagnostics.append(diag)

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "expected_kind": expected_kind,
                "group": group,
                "http_status": code,
                "ok": ok,
                "diagnostic": sanitize_text_for_display(diag),
            }
        )

    sha_match: bool | None = None
    sha_drift = False
    if resolved_expected_sha is not None:
        sha_match = shas_match(live_build_sha, resolved_expected_sha)
        sha_drift = not sha_match
        if sha_drift:
            rev_label = cloud_run_revision or ("inconnue" if is_fr else "unknown")
            diagnostics.append(
                f"Dérive de SHA Cloud Run : `build_sha`='{live_build_sha}' (révision={rev_label}) "
                f"!= attendu '{resolved_expected_sha}'."
                if is_fr
                else f"Cloud Run SHA drift: live `build_sha`='{live_build_sha}' (revision={rev_label}) "
                f"!= expected '{resolved_expected_sha}'."
            )

    html_all_ok = bool(html_contract_ok and len(html_contracts) == len(HOLIDAY_HTML_ROUTES))
    csp_nonces_unique = bool(html_all_ok and len(seen_nonces) == len(HOLIDAY_HTML_ROUTES))
    holiday_routes_deployed = bool(
        all(http_statuses.get(p) == 200 for p in HOLIDAY_REQUIRED_ROUTES)
        and manifest_valid
        and html_all_ok
        and feedback_summary_ok
    )
    revision_drift_detected = bool(
        not holiday_routes_deployed
        or len(stale_404_routes) > 0
        or sha_drift
    )

    routes_verified_count = sum(1 for row in route_rows if row["ok"])
    routes_total_count = len(route_rows)
    invite_ready = bool(
        routes_verified_count == routes_total_count
        and holiday_routes_deployed
        and producer_fleet_ok
        and csp_nonces_unique
        and not revision_drift_detected
    )

    if revision_drift_detected:
        hold_msg = (
            "MISE EN ATTENTE D'INVITATION : dérive de révision Cloud Run détectée (routes 404 ou SHA obsolète) — "
            "ne partagez pas les liens `/noel`, `/quick` ou `/raid-noel` avant le redéploiement."
            if is_fr
            else "INVITE HOLD ACTIVE: Cloud Run revision drift detected (stale 404 routes or SHA mismatch) — "
            "do not share `/noel`, `/quick`, or `/raid-noel` links until Cloud Run cutover completes."
        )
        diagnostics.insert(0, hold_msg)

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": READINESS_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "check_drift",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "invite_ready": invite_ready,
            "share_approved": invite_ready,
            "invite_hold_active": not invite_ready,
            "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed,
            "revision_drift_detected": revision_drift_detected,
            "stale_404_routes": stale_404_routes,
            "failed_routes": failed_routes,
            "routes_verified_count": routes_verified_count,
            "routes_total_count": routes_total_count,
            "http_statuses": http_statuses,
            "manifest_valid": manifest_valid,
            "live_build_sha": live_build_sha,
            "expected_build_sha": resolved_expected_sha,
            "sha_match": sha_match,
            "sha_drift": sha_drift,
            "cloud_run_revision": cloud_run_revision,
            "missing_manifest_routes": missing_manifest_routes,
            "missing_manifest_scripts": missing_manifest_scripts,
            "html_contract_ok": html_all_ok,
            "html_contracts": html_contracts,
            "csp_nonces_unique": csp_nonces_unique,
            "unique_csp_nonces_count": len(seen_nonces),
            "csp_nonce_prefixes": seen_nonces_raw,
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "producer_fleet_ok": producer_fleet_ok,
            "producer_fleet": producer_fleet_doc,
            "network_calls": routes_total_count,
            "file_writes": 0,
            "zero_inference_verified": True,
            "routes": route_rows,
            "diagnostics": [sanitize_text_for_display(d) for d in diagnostics],
            "compliance": dict(comp_doc),
        }
    )


def _build_two_step_snippets(clean_gw: str, *, hold_active: bool) -> dict[str, Any]:
    """Build safe two-step download-then-run snippets (never curl-pipe-shell) pointing to gateway scripts."""
    dev_step1 = f'curl -fsSL "{clean_gw}/scripts/noel_vscode_autoconfig.py" -o noel_vscode_autoconfig.py'
    dev_step2 = f'python3 noel_vscode_autoconfig.py --gateway "{clean_gw}" --quick-connect'
    gamer_step1 = f'Invoke-WebRequest -Uri "{clean_gw}/scripts/join_hivemind.ps1" -OutFile join_hivemind.ps1'
    gamer_step2 = (
        f'powershell -ExecutionPolicy Bypass -File .\\join_hivemind.ps1 -Mode GamerWhisper -Gateway "{clean_gw}"'
    )
    uninstall_cmd = "python3 noel_vscode_autoconfig.py --uninstall"
    local_dev_fallback = f'python3 scripts/hivemind_quickstart.py --gateway "{clean_gw}" --write-continue-config'
    return {
        "curl_pipe_shell_used": False,
        "download_then_run_enforced": True,
        "hold_active": hold_active,
        "dev_vscode_60s": {
            "step1_download": dev_step1,
            "step2_run": dev_step2,
            "combined_two_step": f"{dev_step1}\n{dev_step2}",
            "local_repo_fallback": local_dev_fallback,
        },
        "gamer_win11_5min": {
            "step1_download": gamer_step1,
            "step2_run": gamer_step2,
            "combined_two_step": f"{gamer_step1}\n{gamer_step2}",
        },
        "uninstall_10s_command": uninstall_cmd,
    }


def _render_invite_card_for_channel(
    *,
    clean_gw: str,
    clean_guild: str,
    channel: str,
    lang: str,
    holiday_deployed: bool,
    drift_detected: bool,
    hold_active: bool,
    stale_routes: Sequence[str],
    snippets: Mapping[str, Any],
) -> str:
    """Render a concise Signal, WhatsApp, Discord, or Markdown invite card in FR or EN."""
    is_fr = lang == "fr"
    dev_s1 = snippets["dev_vscode_60s"]["step1_download"]
    dev_s2 = snippets["dev_vscode_60s"]["step2_run"]
    gam_s1 = snippets["gamer_win11_5min"]["step1_download"]
    gam_s2 = snippets["gamer_win11_5min"]["step2_run"]
    uninst = snippets["uninstall_10s_command"]
    dev_fallback = snippets["dev_vscode_60s"]["local_repo_fallback"]

    stale_note = ", ".join(stale_routes) if stale_routes else "/noel, /quick, /raid-noel"

    if is_fr:
        header = f"🎄 HiveMind Raid GPU d'Hiver — [{clean_guild}]"
        compliance_line = (
            "⚖️ Conformité Amadeus / PC perso : 100 % PC personnels à domicile & projets hobby/open-source "
            "uniquement (jamais de PC ni de code d'entreprise) — 0,00 EUR / 0.00 USD, points symboliques "
            "sans valeur marchande, licence AGPL-3.0."
        )
        physics_line = (
            "⚡ Transparence électrique & jeu : ~60-120 W en charge douce (~0,12-0,24 kWh pour 2h), "
            "deltas LoRA <50 MB/h en HTTPS sortant (0 port entrant), plafond silencieux <=70.0 C, "
            f"cession auto Steam/DX12 en <1.5s, désinstallation propre `--uninstall` en 10s (`{uninst}`)."
        )
        if drift_detected or hold_active:
            links_block = (
                f"⏸️ [EN ATTENTE / HOLD — DÉRIVE CLOUD RUN DÉTECTÉE] : `{stale_note}` non déployé ou SHA en retard. "
                f"Ne partagez PAS les liens `/noel`, `/quick` ou `/raid-noel` à l'aveugle ! "
                f"Repli vérifié en attendant le cutover : {clean_gw}/signup ou script local `{dev_fallback}`."
            )
        else:
            links_block = (
                f"🔗 Pages vérifiées en direct : {clean_gw}/noel | {clean_gw}/quick | {clean_gw}/raid-noel"
            )
        dev_title = "👩‍💻 Dev VS Code (<60s — téléchargement puis exécution en 2 étapes, jamais de pipe shell) :"
        gam_title = "🎮 Gamer Windows 11 RTX (<5min — mode Gamer-Whisper silencieux <=70.0 C, 2 étapes) :"
    else:
        header = f"🎄 HiveMind Winter GPU Raid — [{clean_guild}]"
        compliance_line = (
            "⚖️ Amadeus / Personal-PC Compliance: 100% personal home PCs & personal hobby/open-source "
            "projects only (never work PCs or corporate code) — 0.00 EUR / 0.00 USD, symbolic non-monetary "
            "points only, AGPL-3.0."
        )
        physics_line = (
            "⚡ Honest Wh electricity & gaming note: ~60-120 W gentle load (~0.12-0.24 kWh per 2h), "
            "<50 MB/h LoRA deltas over outbound HTTPS (0 open ports), <=70.0 C quiet-fan cap, "
            f"<1.5s automatic Steam/DX12 yield, clean `--uninstall` in 10s (`{uninst}`)."
        )
        if drift_detected or hold_active:
            links_block = (
                f"⏸️ [HOLD / FALLBACK — CLOUD RUN DRIFT DETECTED]: `{stale_note}` returned 404 or SHA mismatch. "
                f"Do NOT share stale `/noel`, `/quick`, or `/raid-noel` links blindly! "
                f"Safe fallback until Cloud Run cutover: {clean_gw}/signup or local `{dev_fallback}`."
            )
        else:
            links_block = (
                f"🔗 Verified live hubs: {clean_gw}/noel | {clean_gw}/quick | {clean_gw}/raid-noel"
            )
        dev_title = "👩‍💻 Dev VS Code (<60s — safe 2-step download-then-run, never curl-pipe-shell):"
        gam_title = "🎮 Windows 11 RTX Gamer-Whisper (<5min — quiet <=70.0 C, safe 2-step):"

    _ = holiday_deployed
    if channel == "discord":
        text = (
            f"**{header}**\n"
            f"{compliance_line}\n"
            f"{physics_line}\n"
            f"{links_block}\n"
            f"**{dev_title}**\n"
            f"```sh\n{dev_s1}\n{dev_s2}\n```\n"
            f"**{gam_title}**\n"
            f"```powershell\n{gam_s1}\n{gam_s2}\n```"
        )
    elif channel == "markdown":
        text = (
            f"# {header}\n\n"
            f"- {compliance_line}\n"
            f"- {physics_line}\n"
            f"- {links_block}\n\n"
            f"## {dev_title}\n\n"
            f"```sh\n{dev_s1}\n{dev_s2}\n```\n\n"
            f"## {gam_title}\n\n"
            f"```powershell\n{gam_s1}\n{gam_s2}\n```\n"
        )
    else:
        # Compact Signal / WhatsApp card
        text = (
            f"{header}\n"
            f"{compliance_line}\n"
            f"{physics_line}\n"
            f"{links_block}\n"
            f"{dev_title}\n"
            f"  1) {dev_s1}\n"
            f"  2) {dev_s2}\n"
            f"{gam_title}\n"
            f"  1) {gam_s1}\n"
            f"  2) {gam_s2}"
        )

    return _assert_safe_output(sanitize_text_for_display(text))


def generate_friend_invite_pack(
    gateway: str = DEFAULT_GATEWAY,
    *,
    guild_tag: str = DEFAULT_GUILD_TAG,
    channel: str = DEFAULT_CHANNEL,
    lang: str = "fr",
    readiness: Mapping[str, Any] | None = None,
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Render concise FR/EN Signal/WhatsApp/Discord/Markdown friend-invite cards with drift hold guard."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        return _compliance_blocked_report(
            "generate_invite",
            gateway,
            comp_doc,
            lang_code=lang_code,
        )

    clean_gw = validate_gateway_url(gateway)
    clean_channel = normalize_channel(channel)
    clean_guild = sanitize_guild_tag(guild_tag)

    if readiness is not None:
        if not isinstance(readiness, Mapping):
            raise InviteFleetVerifierError("readiness must be a mapping returned by check_cloud_run_invite_readiness.")
        holiday_deployed = bool(readiness.get("holiday_routes_deployed_on_cloud_run", False))
        drift_detected = bool(
            readiness.get("revision_drift_detected", not holiday_deployed)
        )
        raw_stale = readiness.get("stale_404_routes")
        stale_routes = (
            [str(r) for r in raw_stale if isinstance(r, str)]
            if isinstance(raw_stale, Sequence) and not isinstance(raw_stale, (str, bytes))
            else []
        )
        live_sha = sanitize_build_sha(readiness.get("live_build_sha"))
        invite_ready = bool(readiness.get("invite_ready", holiday_deployed and not drift_detected))
    else:
        holiday_deployed = True
        drift_detected = False
        stale_routes = []
        live_sha = "unknown"
        invite_ready = True

    hold_active = bool(drift_detected or not holiday_deployed or not invite_ready)
    snippets = _build_two_step_snippets(clean_gw, hold_active=hold_active)

    cards: dict[str, str] = {}
    for ch_name in ("signal", "whatsapp", "discord", "markdown"):
        cards[ch_name] = _render_invite_card_for_channel(
            clean_gw=clean_gw,
            clean_guild=clean_guild,
            channel=ch_name,
            lang=lang_code,
            holiday_deployed=holiday_deployed,
            drift_detected=drift_detected,
            hold_active=hold_active,
            stale_routes=stale_routes,
            snippets=snippets,
        )

    hold_notice = ""
    if hold_active:
        hold_notice = (
            "HOLD ACTIF : révision Cloud Run en retard (`revision_drift_detected=True`) — "
            "les liens `/noel`, `/quick` et `/raid-noel` sont retenus jusqu'au cutover vérifié."
            if is_fr
            else "HOLD ACTIVE: Cloud Run revision drift detected (`revision_drift_detected=True`) — "
            "`/noel`, `/quick`, and `/raid-noel` links are held back until verified cutover."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": INVITE_PACK_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "generate_invite",
            "lang": lang_code,
            "gateway": clean_gw,
            "guild_tag": clean_guild,
            "channel": clean_channel,
            "compliance_ok": True,
            "holiday_routes_deployed_on_cloud_run": holiday_deployed,
            "revision_drift_detected": drift_detected,
            "stale_404_routes": stale_routes,
            "live_build_sha": live_sha,
            "invite_ready": not hold_active,
            "share_approved": not hold_active,
            "invite_hold_active": hold_active,
            "hold_notice": hold_notice,
            "fallback_signup_url": f"{clean_gw}/signup",
            "verified_holiday_urls": (
                []
                if hold_active
                else [f"{clean_gw}/noel", f"{clean_gw}/quick", f"{clean_gw}/raid-noel"]
            ),
            "honest_energy_spec": {
                "gpu_power_watts_range": "~60-120 W",
                "lora_delta_bandwidth": "<50 MB/h",
                "max_temp_c": DEFAULT_MAX_TEMP_C,
                "steam_yield_budget_s": "<1.5s",
                "uninstall_duration": "10s",
                "financial_promise": FINANCIAL_PROMISE,
            },
            "snippets": snippets,
            "invite_card": cards[clean_channel],
            "cards": cards,
            "zero_inference_verified": True,
            "compliance": dict(comp_doc),
        }
    )


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
        raise InviteFleetVerifierError("Subprocess command must be a non-empty sequence of string arguments.")
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
        raise InviteFleetVerifierError("gpu_probe_fn must return GpuProbeSnapshot, a mapping, or None.")

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
    """Run a lightweight local dummy-worker preemption drill (`shell=False`, `<= 3s` timeout)."""
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


def run_steam_yield_and_thermal_preflight(
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
    lang: str = "fr",
) -> dict[str, Any]:
    """Run the deterministic `<1.5s` Steam/DX12 GPU yield & `<=70.0 C` quiet-fan preflight."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name

    if (
        isinstance(max_temp_c, bool)
        or not math.isfinite(float(max_temp_c))
        or not (25.0 <= float(max_temp_c) <= 95.0)
    ):
        raise InviteFleetVerifierError("max_temp_c must be a finite temperature between 25.0 and 95.0 C.")
    if isinstance(min_free_vram_mib, bool) or int(min_free_vram_mib) < 256:
        raise InviteFleetVerifierError("min_free_vram_mib must be an integer >= 256 MiB.")
    if (
        isinstance(yield_budget_ms, bool)
        or not math.isfinite(float(yield_budget_ms))
        or float(yield_budget_ms) <= 0.0
    ):
        raise InviteFleetVerifierError("yield_budget_ms must be a positive finite budget.")

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
        retained_mib = int(raw_ret) if raw_ret is not None else (0 if clean_release else -1)
        yield_latency_ms = round(rep_ms, 3) if rep_ms is not None else (measured_ms if worker_stopped else None)
        drill_method = str(raw_drill.get("method") or "custom_worker_yield_fn")
    else:
        raise InviteFleetVerifierError("worker_yield_fn must return a mapping or numeric latency.")

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
        gpu_ready = False
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
                f"Aucun GPU NVIDIA détecté — statut honnête 'dev_api_only_no_nvidia_gpu' "
                f"(API Dev prête sans GPU fictif, cession worker vérifiée en {lat_s_str} < {budget_s:.2f}s)."
                if is_fr
                else f"No NVIDIA GPU detected — honest 'dev_api_only_no_nvidia_gpu' status "
                f"(Dev API ready without fake GPU specs, worker yield verified in {lat_s_str} < {budget_s:.2f}s)."
            )
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
                f"(< {reserve_mib} MiB) : worker en pause pour garder les ventilateurs silencieux."
                if is_fr
                else f"{gpu.name} at {gpu.temp_c:.1f} C (> {cap_c:.1f} C) and {gpu.free_vram_mib} MiB free "
                f"(< {reserve_mib} MiB): worker paused to keep fans quiet and protect VRAM."
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
            "object": PREFLIGHT_OBJECT,
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


def load_saved_credentials(path: Path, *, expected_gateway: str | None = None) -> dict[str, Any] | None:
    """Load `~/.hivemind/credentials.json` if readable and holding a valid `hm-...` / `hm_live_...` key."""
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


@contextmanager
def _locked_spool_file(spool_path: Path) -> Iterator[None]:
    rlock = _get_path_lock(spool_path)
    with rlock:
        lock_fd: int | None = None
        spool_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt" and spool_path.parent.name == ".hivemind":
            with suppress(OSError):
                os.chmod(spool_path.parent, 0o700)
        if _fcntl is not None:
            lock_path = spool_path.with_name(f".{spool_path.name}.lock")
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            with suppress(OSError):
                os.chmod(lock_path, 0o600)
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
        try:
            yield
        finally:
            if lock_fd is not None:
                try:
                    if _fcntl is not None:
                        _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)


def _read_ces_spool_unlocked(spool_path: Path) -> dict[str, Any]:
    if not spool_path.is_file():
        return {"schema": SPOOL_SCHEMA, "pending": [], "delivered_event_ids": []}
    try:
        loaded = json.loads(spool_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": SPOOL_SCHEMA, "pending": [], "delivered_event_ids": []}
    if not isinstance(loaded, dict):
        return {"schema": SPOOL_SCHEMA, "pending": [], "delivered_event_ids": []}
    pending = loaded.get("pending") if isinstance(loaded.get("pending"), list) else []
    delivered = loaded.get("delivered_event_ids") if isinstance(loaded.get("delivered_event_ids"), list) else []
    return {
        "schema": SPOOL_SCHEMA,
        "pending": [dict(x) for x in pending if isinstance(x, Mapping)],
        "delivered_event_ids": [str(x) for x in delivered if isinstance(x, str)],
    }


def enqueue_ces_spool_item(
    spool_path: Path,
    payload: Mapping[str, Any],
    *,
    reason: str,
    home: Path | None = None,
) -> dict[str, Any]:
    """Atomically spool a zero-PII CES feedback item (`0600` permissions) when Cloud Run returns 404/429/5xx."""
    clean_payload = validate_onboarding_feedback_payload(payload)
    canonical = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"))
    event_id = hashlib.sha256(f"hivemind.noel.ces.spool/1:{canonical}".encode()).hexdigest()

    with _locked_spool_file(spool_path):
        state = _read_ces_spool_unlocked(spool_path)
        existing_ids = {str(item.get("event_id")) for item in state["pending"]} | set(
            state["delivered_event_ids"]
        )
        deduplicated = event_id in existing_ids
        if not deduplicated:
            state["pending"].append(
                {
                    "event_id": event_id,
                    "payload": clean_payload,
                    "reason": sanitize_text_for_display(reason, home=home)[:64],
                }
            )
            if len(state["pending"]) > DEFAULT_MAX_SPOOL_ENTRIES:
                state["pending"] = state["pending"][-DEFAULT_MAX_SPOOL_ENTRIES:]
        rendered = _assert_safe_output(json.dumps(state, indent=2, sort_keys=True) + "\n")
        _write_file_secure_atomic(spool_path, rendered, posix_mode=0o600)
        pending_count = len(state["pending"])

    return {
        "spooled": True,
        "deduplicated": deduplicated,
        "event_id": event_id,
        "pending_count": pending_count,
        "spool_file": redact_home_path(spool_path, home=home),
    }


def flush_ces_spool(
    gateway: str = DEFAULT_GATEWAY,
    *,
    spool_path: Path,
    http_fn: HttpFn = default_http_json,
    home: Path | None = None,
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "fr",
) -> dict[str, Any]:
    """Flush locally spooled zero-PII CES items once Cloud Run cutover exposes `/v1/hivemind/onboarding/feedback`."""
    lang_code = normalize_lang(lang)
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("flush_ces_spool", gateway, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    delivered = 0
    retained: list[dict[str, Any]] = []
    network_calls = 0

    with _locked_spool_file(spool_path):
        state = _read_ces_spool_unlocked(spool_path)
        delivered_ids = list(state["delivered_event_ids"])
        for entry in state["pending"]:
            ev_id = str(entry.get("event_id") or "")
            raw_pl = entry.get("payload")
            try:
                clean_pl = validate_onboarding_feedback_payload(
                    raw_pl if isinstance(raw_pl, Mapping) else None
                )
            except InviteFleetVerifierError:
                continue
            network_calls += 1
            code, _ = _invoke_http_callable(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/onboarding/feedback",
                payload=clean_pl,
                headers={"Idempotency-Key": ev_id} if ev_id else None,
            )
            if code in (200, 201):
                delivered += 1
                if ev_id and ev_id not in delivered_ids:
                    delivered_ids.append(ev_id)
            elif code in (0, 404, 408, 429) or code >= 500:
                retained.append({"event_id": ev_id, "payload": clean_pl, "reason": f"http_{code}"})

        state["pending"] = retained
        state["delivered_event_ids"] = delivered_ids[-512:]
        rendered = _assert_safe_output(json.dumps(state, indent=2, sort_keys=True) + "\n")
        _write_file_secure_atomic(spool_path, rendered, posix_mode=0o600)

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": "hivemind.noel_ces_spool_flush",
            "mode": "flush_ces_spool",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "flush_ok": len(retained) == 0,
            "delivered_count": delivered,
            "remaining_pending_count": len(retained),
            "spool_file": redact_home_path(spool_path, home=home),
            "network_calls": network_calls,
            "zero_inference_verified": True,
        }
    )


def verify_live_fleet_and_ces_activation(
    gateway: str = DEFAULT_GATEWAY,
    *,
    api_key: str | None = None,
    credentials_path: Path | str | None = None,
    send_opentrainer_heartbeat: bool = False,
    submit_ces: bool = False,
    profile: str = "gamer",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    host_label: str = DEFAULT_HOST_LABEL,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_yield_fn: WorkerYieldFn = default_worker_yield_drill,
    http_fn: HttpFn = default_http_json,
    clock_fn: ClockFn = time.monotonic,
    lang: str = "fr",
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    ces_spool_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run `<1.5s` Steam yield / `<=70.0 C` preflight, optional OpenTrainer heartbeat, live fleet check, and CES ping."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        return _compliance_blocked_report(
            "verify_activation",
            gateway,
            comp_doc,
            lang_code=lang_code,
        )

    clean_gw = validate_gateway_url(gateway)
    clean_profile = normalize_profile(profile)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise InviteFleetVerifierError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(had_game_lag, bool):
        raise InviteFleetVerifierError("had_game_lag must be a boolean.")
    if (
        isinstance(target_online_producers, bool)
        or not isinstance(target_online_producers, int)
        or target_online_producers < 1
    ):
        raise InviteFleetVerifierError("target_online_producers must be an integer >= 1.")

    resolved_home = home if home is not None else Path.home()
    env_view = env if env is not None else os.environ
    clean_host_label = sanitize_host_label(host_label, home=resolved_home)

    cred_target = (
        Path(credentials_path)
        if (credentials_path is not None and str(credentials_path).strip())
        else (resolved_home / DEFAULT_CREDENTIALS_REL)
    )
    spool_target = (
        Path(ces_spool_path)
        if (ces_spool_path is not None and str(ces_spool_path).strip())
        else (resolved_home / DEFAULT_CES_SPOOL_REL)
    )

    env_key = str(env_view.get("HIVEMIND_API_KEY") or "").strip()
    resolved_key = (api_key or "").strip() or env_key
    if resolved_key and not is_valid_hivemind_key(resolved_key):
        raise InviteFleetVerifierError(
            "Provided API key is invalid; expected an 'hm-...' or 'hm_live_...' key."
        )
    if not resolved_key:
        saved_cred = load_saved_credentials(cred_target, expected_gateway=clean_gw)
        if saved_cred is not None:
            resolved_key = str(saved_cred["api_key"]).strip()

    has_key = bool(resolved_key)
    masked_key = mask_api_key(resolved_key) if has_key else None
    auth_headers = {"Authorization": f"Bearer {resolved_key}"} if has_key else None

    preflight = run_steam_yield_and_thermal_preflight(
        max_temp_c=max_temp_c,
        min_free_vram_mib=min_free_vram_mib,
        yield_budget_ms=yield_budget_ms,
        had_game_lag=had_game_lag,
        gpu_probe_fn=gpu_probe_fn,
        process_probe_fn=process_probe_fn,
        worker_yield_fn=worker_yield_fn,
        clock_fn=clock_fn,
        os_name=os_name,
        home=resolved_home,
        lang=lang_code,
    )

    honest_gpu_name: str | None = None
    honest_vram_total: int | None = None
    honest_vram_free: int | None = None
    if (
        preflight["gpu_present"]
        and preflight["gpu_name"]
        and isinstance(preflight["vram_total_mib"], int)
        and preflight["vram_total_mib"] >= MIN_REPORTABLE_VRAM_TOTAL_MIB
        and isinstance(preflight["vram_free_mib"], int)
        and preflight["vram_free_mib"] >= 0
    ):
        honest_gpu_name = str(preflight["gpu_name"])[:128]
        honest_vram_total = int(preflight["vram_total_mib"])
        honest_vram_free = min(int(preflight["vram_free_mib"]), honest_vram_total)

    diagnostics: list[str] = []
    network_calls = 0
    file_writes = 0

    ot_sent = False
    ot_accepted = False
    ot_http_status: int | None = None
    ot_node_id: str | None = None
    ot_payload: dict[str, Any] | None = None
    ot_ok = True

    if not send_opentrainer_heartbeat:
        ot_status = "not_requested"
    elif not has_key:
        ot_status = "missing_api_key"
        ot_ok = False
        diagnostics.append(
            "Clé `hm-...` requise pour envoyer `POST /v1/hivemind/opentrainer/heartbeat`."
            if is_fr
            else "Valid `hm-...` API key required to send `POST /v1/hivemind/opentrainer/heartbeat`."
        )
    elif not preflight["yield_slo_ok"]:
        ot_status = "blocked_yield_slo"
        ot_ok = False
        diagnostics.append(
            "Heartbeat OpenTrainer bloqué : le test de cession Steam `<1.5s` a échoué."
            if is_fr
            else "OpenTrainer heartbeat held back: `<1.5s` Steam GPU yield preflight failed."
        )
    else:
        ot_payload = validate_opentrainer_heartbeat_payload(
            {
                "host_label": clean_host_label,
                "gpu_name": honest_gpu_name,
                "vram_total_mib": honest_vram_total,
                "vram_free_mib": honest_vram_free,
                "network_mode": "outbound-https-relay",
            }
        )
        ot_sent = True
        network_calls += 1
        ot_http_status, ot_body = _invoke_http_callable(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
            payload=ot_payload,
            headers=auth_headers,
        )
        ot_accepted = ot_http_status in (200, 201)
        ot_ok = ot_accepted
        node_doc = ot_body.get("node") if isinstance(ot_body.get("node"), Mapping) else {}
        if isinstance(node_doc.get("node_id"), str) and node_doc["node_id"].strip():
            ot_node_id = sanitize_text_for_display(node_doc["node_id"].strip(), home=resolved_home)
        if ot_accepted:
            ot_status = "accepted"
        else:
            ot_status = f"rejected_http_{ot_http_status}"
            diagnostics.append(
                f"Échec sur POST /v1/hivemind/opentrainer/heartbeat (HTTP {ot_http_status})."
                if is_fr
                else f"Outbound heartbeat failed on POST /v1/hivemind/opentrainer/heartbeat (HTTP {ot_http_status})."
            )

    network_calls += 1
    fleet_code, fleet_body = _invoke_http_callable(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/producers/fleet",
    )
    fleet_ok = fleet_code == 200
    if not fleet_ok:
        diagnostics.append(
            f"Échec de lecture GET /v1/hivemind/producers/fleet (HTTP {fleet_code})."
            if is_fr
            else f"Fleet check failed on GET /v1/hivemind/producers/fleet (HTTP {fleet_code})."
        )

    network_calls += 1
    raid_code, raid_body = _invoke_http_callable(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/raid/status",
    )
    raid_ok = raid_code == 200
    stale_raid_route_404 = raid_code == 404
    if not raid_ok:
        diagnostics.append(
            f"Échec de lecture GET /v1/hivemind/raid/status (HTTP {raid_code})."
            if is_fr
            else f"Raid status check failed on GET /v1/hivemind/raid/status (HTTP {raid_code})."
        )

    swarm = raid_body.get("swarm") if isinstance(raid_body.get("swarm"), Mapping) else {}

    def _safe_int(val: Any, default: int = 0) -> int:
        if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
            return val
        return default

    live_hiveminders = (
        _safe_int(fleet_body.get("live_hiveminders"))
        if "live_hiveminders" in fleet_body
        else _safe_int(swarm.get("live_producers"))
    )
    total_hiveminders = _safe_int(
        fleet_body.get("total_hiveminders"),
        _safe_int(swarm.get("registered_producers")),
    )
    distinct_gpus = _safe_int(fleet_body.get("distinct_physical_gpus"))
    live_endpoints = _safe_int(fleet_body.get("live_endpoints"))
    raw_models = fleet_body.get("live_models")
    live_models = (
        [str(m) for m in raw_models if isinstance(m, str) and m.strip()]
        if isinstance(raw_models, list)
        else []
    )
    active_trainers = _safe_int(swarm.get("active_trainers"))
    registered_trainer_nodes = _safe_int(swarm.get("registered_trainer_nodes"))
    target_met = live_hiveminders >= target_online_producers

    effective_zero_lag = bool(preflight["zero_game_lag"])
    ces_doc: dict[str, Any]
    if not submit_ces:
        ces_doc = {
            "attempted": False,
            "submitted": False,
            "spooled": False,
            "degraded": False,
            "non_blocking": True,
            "status": "not_requested",
            "http_status": None,
            "profile": clean_profile,
            "ease_rating": int(ease_rating),
            "zero_game_lag": effective_zero_lag,
            "average_ease_rating": None,
            "mean_ease_rating": None,
            "total_responses": None,
            "zero_game_lag_ratio": None,
            "spool_file": None,
            "zero_pii_payload": None,
        }
    else:
        ces_payload = validate_onboarding_feedback_payload(
            {
                "profile": clean_profile,
                "ease_rating": int(ease_rating),
                "zero_game_lag": effective_zero_lag,
                "platform": _detect_platform_tag(system_name),
                "note": "",
            }
        )
        network_calls += 1
        ces_code, ces_body = _invoke_http_callable(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/onboarding/feedback",
            payload=ces_payload,
        )
        ces_recorded = ces_code in (200, 201)
        raw_avg = (
            ces_body.get("average_ease_rating")
            if "average_ease_rating" in ces_body
            else ces_body.get("mean_ease_rating")
        )
        avg_ease = (
            round(float(raw_avg), 2)
            if isinstance(raw_avg, (int, float)) and not isinstance(raw_avg, bool)
            else None
        )
        raw_tot_resp = ces_body.get("total_responses")
        tot_resp = (
            int(raw_tot_resp)
            if isinstance(raw_tot_resp, int) and not isinstance(raw_tot_resp, bool)
            else None
        )
        raw_lag_r = ces_body.get("zero_game_lag_ratio")
        lag_ratio = (
            round(float(raw_lag_r), 4)
            if isinstance(raw_lag_r, (int, float)) and not isinstance(raw_lag_r, bool)
            else None
        )

        spooled_meta: dict[str, Any] | None = None
        if ces_recorded:
            ces_status = "recorded"
        elif ces_code in (0, 404, 408, 429) or ces_code >= 500:
            ces_status = "spooled_stale_404" if ces_code == 404 else f"spooled_http_{ces_code}"
            spooled_meta = enqueue_ces_spool_item(
                spool_target,
                ces_payload,
                reason=ces_status,
                home=resolved_home,
            )
            file_writes += 1
            diagnostics.append(
                f"Retour CES mis en file locale sans bloquer (`{ces_status}`) — sera rejoué après cutover Cloud Run."
                if is_fr
                else f"CES feedback safely spooled locally without blocking (`{ces_status}`) — ready for post-cutover flush."
            )
        else:
            ces_status = f"rejected_http_{ces_code}"
            diagnostics.append(
                f"Retour CES refusé sans bloquer (HTTP {ces_code})."
                if is_fr
                else f"CES feedback rejected without blocking (HTTP {ces_code})."
            )

        ces_doc = {
            "attempted": True,
            "submitted": ces_recorded,
            "spooled": spooled_meta is not None,
            "degraded": not ces_recorded,
            "stale_cloud_run_404": ces_code == 404,
            "non_blocking": True,
            "status": ces_status,
            "http_status": ces_code,
            "profile": clean_profile,
            "ease_rating": int(ease_rating),
            "zero_game_lag": effective_zero_lag,
            "average_ease_rating": avg_ease,
            "mean_ease_rating": avg_ease,
            "total_responses": tot_resp,
            "zero_game_lag_ratio": lag_ratio,
            "spool_file": spooled_meta["spool_file"] if spooled_meta is not None else None,
            "event_id": spooled_meta["event_id"] if spooled_meta is not None else None,
            "zero_pii_payload": ces_payload,
        }

    activation_ok = bool(
        preflight["preflight_ok"]
        and preflight["yield_slo_ok"]
        and ot_ok
        and fleet_ok
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": ACTIVATION_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "verify_activation",
            "lang": lang_code,
            "gateway": clean_gw,
            "profile": clean_profile,
            "compliance_ok": True,
            "activation_ok": activation_ok,
            "credentials_path": redact_home_path(cred_target, home=resolved_home),
            "api_key_present": has_key,
            "masked_api_key": masked_key,
            "yield_slo_ok": bool(preflight["yield_slo_ok"]),
            "thermal_ok": bool(preflight["thermal_ok"]),
            "vram_reserve_ok": bool(preflight["vram_reserve_ok"]),
            "zero_game_lag": effective_zero_lag,
            "preflight": preflight,
            "opentrainer_heartbeat": {
                "requested": bool(send_opentrainer_heartbeat),
                "sent": ot_sent,
                "accepted": ot_accepted,
                "http_status": ot_http_status,
                "status": ot_status,
                "node_id": ot_node_id,
                "host_label": clean_host_label,
                "network_mode": "outbound-https-relay",
                "gpu_name": honest_gpu_name,
                "vram_total_mib": honest_vram_total,
                "vram_free_mib": honest_vram_free,
                "payload": ot_payload,
            },
            "fleet": {
                "fleet_route_ok": fleet_ok,
                "fleet_http_status": fleet_code,
                "raid_route_ok": raid_ok,
                "raid_http_status": raid_code,
                "stale_raid_route_404": stale_raid_route_404,
                "revision_drift_detected": bool(fleet_code == 404 or raid_code == 404),
                "live_hiveminders": live_hiveminders,
                "total_hiveminders": total_hiveminders,
                "distinct_physical_gpus": distinct_gpus,
                "live_endpoints": live_endpoints,
                "live_models": live_models,
                "active_trainers": active_trainers,
                "registered_trainer_nodes": registered_trainer_nodes,
                "target_online_producers": target_online_producers,
                "target_producers_met": target_met,
                "producers_remaining_to_target": max(0, target_online_producers - live_hiveminders),
            },
            "ces_feedback": ces_doc,
            "network_calls": network_calls,
            "file_writes": file_writes,
            "zero_inference_verified": True,
            "diagnostics": [sanitize_text_for_display(d, home=resolved_home) for d in diagnostics],
            "compliance": dict(comp_doc),
        }
    )


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a concise, secret-redacted human CLI summary."""
    lines = [
        f"HiveMind Noel Friend-Invite & Fleet Verifier [{report.get('mode')}]",
        f"Gateway: {report.get('gateway')} | Compliance OK: {report.get('compliance_ok')}",
        COMPLIANCE_NOTICE_BILINGUAL,
    ]
    if "invite_card" in report and isinstance(report.get("invite_card"), str):
        lines.append("")
        lines.append(str(report["invite_card"]))
    elif isinstance(report.get("invite_pack"), Mapping):
        lines.append("")
        lines.append(str(report["invite_pack"].get("invite_card") or ""))
    if isinstance(report.get("fleet"), Mapping):
        fl = report["fleet"]
        lines.append(
            f"Fleet: live_hiveminders={fl.get('live_hiveminders')}/{fl.get('target_online_producers')} "
            f"| distinct_gpus={fl.get('distinct_physical_gpus')} | active_trainers={fl.get('active_trainers')}"
        )
    for diag in report.get("diagnostics") or ():
        lines.append(f"  - {diag}")
    lines.append(f"Receipt SHA-256: {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines) + "\n")


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a sanitized Markdown summary report."""
    lines = [
        f"# HiveMind Noel Friend-Invite & Fleet Report (`{report.get('mode')}`)",
        "",
        f"- **Gateway**: `{report.get('gateway')}`",
        f"- **Compliance OK**: `{report.get('compliance_ok')}`",
        f"- **Rule**: {COMPLIANCE_NOTICE_BILINGUAL}",
        f"- **Receipt SHA-256**: `{report.get('receipt_sha256')}`",
    ]
    if "holiday_routes_deployed_on_cloud_run" in report:
        lines.append(
            f"- **Holiday Routes Deployed**: `{report.get('holiday_routes_deployed_on_cloud_run')}` "
            f"| **Revision Drift**: `{report.get('revision_drift_detected')}`"
        )
    if isinstance(report.get("fleet"), Mapping):
        fl = report["fleet"]
        lines.append(
            f"- **Live Fleet**: `live_hiveminders={fl.get('live_hiveminders')}`, "
            f"`distinct_physical_gpus={fl.get('distinct_physical_gpus')}`, "
            f"`active_trainers={fl.get('active_trainers')}`"
        )
    if "invite_card" in report and isinstance(report.get("invite_card"), str):
        lines.extend(["", "## Invite Card", "", str(report["invite_card"])])
    return _assert_safe_output("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for `scripts/noel_friend_invite_and_fleet_verifier.py`."""
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind Noel friend-invite quick-launch generator, Cloud Run revision-drift preflight, "
            "and <1.5s Steam GPU yield / live fleet / CES activation verifier."
        )
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="Cloud Run or loopback gateway origin URL.")
    parser.add_argument("--check-drift", action="store_true", help="Probe zero-inference Cloud Run readiness & drift.")
    parser.add_argument("--generate-invite", action="store_true", help="Render friend-invite quick-launch card.")
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        choices=("signal", "whatsapp", "discord", "markdown"),
        help="Invite card format channel.",
    )
    parser.add_argument("--guild-tag", default=DEFAULT_GUILD_TAG, help="Friendly symbolic guild tag.")
    parser.add_argument("--expected-sha", default=None, help="Expected Cloud Run build_sha.")
    parser.add_argument("--output-invite", default=None, help="Optional path to write the rendered invite card.")
    parser.add_argument(
        "--verify-activation",
        action="store_true",
        help="Run <1.5s Steam GPU yield / <=70.0 C preflight + live fleet + optional CES activation verifier.",
    )
    parser.add_argument(
        "--send-opentrainer-heartbeat",
        action="store_true",
        help="Send outbound POST /v1/hivemind/opentrainer/heartbeat.",
    )
    parser.add_argument("--submit-ces", action="store_true", help="Submit zero-PII CES + zero_game_lag feedback.")
    parser.add_argument(
        "--flush-ces-spool",
        action="store_true",
        help="Flush locally spooled zero-PII CES items after Cloud Run cutover.",
    )
    parser.add_argument("--profile", default="gamer", choices=("dev", "gamer"), help="Onboarding profile.")
    parser.add_argument("--ease-rating", type=int, default=5, help="CES ease rating from 1 to 5.")
    parser.add_argument("--had-game-lag", action="store_true", help="Report observed game lag during session.")
    parser.add_argument(
        "--target-online-producers",
        type=int,
        default=DEFAULT_TARGET_ONLINE_PRODUCERS,
        help="Target online HiveMinder producers (default: 8).",
    )
    parser.add_argument(
        "--max-temp-c",
        type=float,
        default=DEFAULT_MAX_TEMP_C,
        help="Quiet-fan GPU temperature cap in Celsius (default: 70.0).",
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
        help="Maximum Steam/DX12 worker yield latency budget in ms (default: 1500.0).",
    )
    parser.add_argument("--api-key", default=None, help="Optional hm-... / hm_live_... API key.")
    parser.add_argument("--credentials-file", default=None, help="Optional path to credentials.json.")
    parser.add_argument("--ces-spool-file", default=None, help="Optional path to local CES feedback spool JSON.")
    parser.add_argument("--host-label", default=DEFAULT_HOST_LABEL, help="Personal PC host label.")
    parser.add_argument("--lang", default="fr", help="Output language ('fr' or 'en').")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON receipt report.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown summary report.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    gpu_probe_fn: GpuProbeFn = default_gpu_probe,
    process_probe_fn: ProcessProbeFn = default_process_probe,
    worker_yield_fn: WorkerYieldFn = default_worker_yield_drill,
    clock_fn: ClockFn = time.monotonic,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    out_stream: Any = None,
    err_stream: Any = None,
) -> int:
    """CLI entrypoint returning `0` on success, `1` on drift/activation failure, and `2` on compliance/arg refusal."""
    out = out_stream if out_stream is not None else sys.stdout
    err = err_stream if err_stream is not None else sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    try:
        lang_code = normalize_lang(args.lang)
    except InviteFleetVerifierError as exc:
        err.write(sanitize_text_for_display(str(exc), home=home) + "\n")
        return 2

    resolved_home = home if home is not None else Path.home()
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        blocked = _compliance_blocked_report(
            "cli_compliance_gate",
            args.gateway,
            comp_doc,
            lang_code=lang_code,
        )
        if args.as_json:
            out.write(_assert_safe_output(json.dumps(blocked, indent=2, ensure_ascii=False)) + "\n")
        else:
            err.write(render_cli_report(blocked))
        return 2

    try:
        clean_gw = validate_gateway_url(args.gateway)
        want_drift = bool(args.check_drift)
        want_invite = bool(args.generate_invite or args.output_invite)
        want_act = bool(
            args.verify_activation or args.send_opentrainer_heartbeat or args.submit_ces
        )
        want_flush = bool(args.flush_ces_spool)
        if not (want_drift or want_invite or want_act or want_flush):
            want_drift = True
            want_invite = True

        spool_target = (
            Path(args.ces_spool_file)
            if args.ces_spool_file
            else (resolved_home / DEFAULT_CES_SPOOL_REL)
        )

        readiness_doc: dict[str, Any] | None = None
        invite_doc: dict[str, Any] | None = None
        activation_doc: dict[str, Any] | None = None
        flush_doc: dict[str, Any] | None = None
        file_writes = 0
        network_calls = 0

        if want_drift:
            readiness_doc = check_cloud_run_invite_readiness(
                clean_gw,
                expected_sha=args.expected_sha,
                http_fn=http_fn,
                lang=lang_code,
                enforce_personal_pc=False,
                env=env,
                cwd=cwd,
            )
            network_calls += int(readiness_doc["network_calls"])

        if want_invite:
            invite_doc = generate_friend_invite_pack(
                clean_gw,
                guild_tag=args.guild_tag,
                channel=args.channel,
                lang=lang_code,
                readiness=readiness_doc,
                enforce_personal_pc=False,
                env=env,
                cwd=cwd,
            )
            if args.output_invite:
                invite_path = Path(args.output_invite)
                _write_file_secure_atomic(
                    invite_path,
                    str(invite_doc["invite_card"]) + "\n",
                    posix_mode=0o600,
                )
                file_writes += 1

        if want_flush:
            flush_doc = flush_ces_spool(
                clean_gw,
                spool_path=spool_target,
                http_fn=http_fn,
                home=resolved_home,
                enforce_personal_pc=False,
                env=env,
                cwd=cwd,
                lang=lang_code,
            )
            network_calls += int(flush_doc["network_calls"])

        if want_act:
            activation_doc = verify_live_fleet_and_ces_activation(
                clean_gw,
                api_key=args.api_key,
                credentials_path=args.credentials_file,
                send_opentrainer_heartbeat=bool(args.send_opentrainer_heartbeat),
                submit_ces=bool(args.submit_ces),
                profile=args.profile,
                ease_rating=args.ease_rating,
                had_game_lag=bool(args.had_game_lag),
                host_label=args.host_label,
                target_online_producers=args.target_online_producers,
                max_temp_c=args.max_temp_c,
                min_free_vram_mib=args.min_free_vram_mib,
                yield_budget_ms=args.yield_budget_ms,
                gpu_probe_fn=gpu_probe_fn,
                process_probe_fn=process_probe_fn,
                worker_yield_fn=worker_yield_fn,
                http_fn=http_fn,
                clock_fn=clock_fn,
                lang=lang_code,
                enforce_personal_pc=False,
                env=env,
                cwd=cwd,
                home=resolved_home,
                os_name=os_name,
                system_name=system_name,
                ces_spool_path=spool_target,
            )
            network_calls += int(activation_doc["network_calls"])
            file_writes += int(activation_doc["file_writes"])

        active_docs = [d for d in (readiness_doc, invite_doc, activation_doc, flush_doc) if d is not None]
        if len(active_docs) == 1 and file_writes == 0:
            final_report = active_docs[0]
        else:
            final_report = _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "campaign_id": CAMPAIGN_ID,
                    "mode": "combined_workflow",
                    "lang": lang_code,
                    "gateway": clean_gw,
                    "compliance_ok": True,
                    "invite_ready": (
                        bool(readiness_doc["invite_ready"])
                        if readiness_doc is not None
                        else (bool(invite_doc["invite_ready"]) if invite_doc is not None else True)
                    ),
                    "holiday_routes_deployed_on_cloud_run": (
                        bool(readiness_doc["holiday_routes_deployed_on_cloud_run"])
                        if readiness_doc is not None
                        else None
                    ),
                    "revision_drift_detected": (
                        bool(readiness_doc["revision_drift_detected"])
                        if readiness_doc is not None
                        else False
                    ),
                    "activation_ok": (
                        bool(activation_doc["activation_ok"]) if activation_doc is not None else None
                    ),
                    "output_invite_path": (
                        redact_home_path(args.output_invite, home=resolved_home)
                        if args.output_invite
                        else None
                    ),
                    "readiness": readiness_doc,
                    "invite_pack": invite_doc,
                    "activation": activation_doc,
                    "ces_spool_flush": flush_doc,
                    "fleet": activation_doc["fleet"] if activation_doc is not None else None,
                    "diagnostics": [
                        *(readiness_doc["diagnostics"] if readiness_doc is not None else []),
                        *(activation_doc["diagnostics"] if activation_doc is not None else []),
                    ],
                    "network_calls": network_calls,
                    "file_writes": file_writes,
                    "zero_inference_verified": True,
                    "compliance": dict(comp_doc),
                }
            )
    except InviteFleetVerifierError as exc:
        err.write(sanitize_text_for_display(str(exc), home=resolved_home) + "\n")
        return 2

    if args.as_json:
        out.write(_assert_safe_output(json.dumps(final_report, indent=2, ensure_ascii=False)) + "\n")
    elif args.markdown:
        out.write(render_markdown_report(final_report))
    else:
        out.write(render_cli_report(final_report))

    if readiness_doc is not None and readiness_doc["revision_drift_detected"]:
        return 1
    if activation_doc is not None and not activation_doc["activation_ok"]:
        return 1
    if flush_doc is not None and not flush_doc["flush_ok"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
