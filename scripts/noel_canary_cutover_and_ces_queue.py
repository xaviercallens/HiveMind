#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cloud Run canary cutover verifier and atomic, idempotent CES feedback retry queue (`SLICE-SCALE-DYN-04`).

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library and proxy-free `urllib.request.ProxyHandler({})`:
  1. Personal-PC & Hobby-Only Compliance Gate (`evaluate_personal_pc_compliance`):
     strictly local inspection of `USERDNSDOMAIN`, `USERDOMAIN`, `HTTP_PROXY`, `HTTPS_PROXY`,
     `COMPUTERNAME`, `HOSTNAME`, and `.git/config` / worktree remote URLs for corporate markers
     (`amadeus`, `corp`, `enterprise`, `intranet`, `internal`, `zscaler`, `netskope`, `bluecoat`,
     `fortinet`, `paloalto`) before any queue write, canary check, or feedback flush, refusing
     closed with exit code 2 (`0` network calls, `0` file writes, `0` env/path value leaks) while
     keeping `--purge-queue` local cleanup always permitted.
  2. Dry-Run FinOps-Safe Cloud Run Canary Cutover & Rollback Verifier (`audit_deploy_script_finops`,
     `build_canary_traffic_plan`, `verify_canary_cutover`): audits `deploy/gcp_openrouter/deploy.sh`
     FinOps invariants (`--min-instances 0`, `--max-instances 1`, `--no-cpu-throttling`,
     `--concurrency 80`, Litestream GCS replica, Secret Manager `HIVEMIND_ROUTER_SECRET`, and `0`
     starter/bonus grants), probes zero-inference canary routes (`GET /health`,
     `GET /v1/hivemind/onboarding/manifest`, `GET /noel`, `GET /quick`, `GET /raid-noel`,
     `GET /arena-noel`, `GET /v1/hivemind/onboarding/feedback/summary`, and
     `GET /v1/hivemind/producers/fleet`) without ever calling `POST /v1/chat/completions` or
     executing `gcloud`, and emits `promote_to_100_pct=True` or `rollback_triggered=True` with
     sanitized `gcloud run services update-traffic` rollback commands.
  3. Atomic, Bounded, Idempotent Local CES & Steam Zero-Game-Lag Feedback Retry Queue
     (`CesFeedbackRetryQueue`, `enqueue_ces_feedback`, `flush_ces_feedback_queue`,
     `cutover_and_flush_ces_queue`, `purge_ces_feedback_queue`): persists zero-PII
     `OnboardingFeedbackRequest`-conforming items (`profile`, `ease_rating`, `zero_game_lag`,
     `platform`, `note=''`) in a `0600`-permissioned bounded JSON spool (`max_entries=128`),
     deduplicates by deterministic SHA-256 `event_id` across `pending` and `delivered_event_ids`,
     sends `Idempotency-Key: <event_id>` on flush, retains items with bounded `retry_count` &
     deterministic exponential backoff across stale Cloud Run `404`, `408`, `429`, `5xx`, or
     timeout responses, and quarantines permanent `400`/`422` errors.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows fallback uses per-path RLock + atomic os.replace
    _fcntl = None  # type: ignore[assignment]

SCHEMA = "hivemind.noel_canary_cutover_and_ces_queue/1"
RECEIPT_OBJECT = "hivemind.noel_canary_cutover_ces_receipt"
QUEUE_SCHEMA = "hivemind.ces_feedback_retry_queue/1"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_CLOUD_RUN_SERVICE = "hivemind-openrouter"
DEFAULT_CLOUD_RUN_REGION = "us-central1"
DEFAULT_CANDIDATE_TAG = "noel-canary"
DEFAULT_CANARY_PERCENT = 10
DEFAULT_MAX_QUEUE_ENTRIES = 128
MAX_QUEUE_ENTRIES_CEILING = 4096
MAX_DELIVERED_LEDGER_MINIMUM = 512
MAX_RETRY_COUNT = 16
DEFAULT_BASE_BACKOFF_S = 1.0
DEFAULT_MAX_BACKOFF_S = 120.0
HTTP_TIMEOUT_S = 5.0

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_SCRIPT_REL = "deploy/gcp_openrouter/deploy.sh"
DEFAULT_QUEUE_REL = ".hivemind/ces_feedback_queue.json"

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})

CANARY_ZERO_INFERENCE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
)

CANARY_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")

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
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

SYMBOLIC_POINTS_NOTICE_EN = (
    "Symbolic non-monetary compute-reciprocity points only (0.00 EUR / 0.00 USD, zero commercial value)."
)
SYMBOLIC_POINTS_NOTICE_FR = (
    "Points symboliques non monétaires de réciprocité de calcul uniquement (0,00 EUR / 0,00 USD, sans valeur marchande)."
)
SYMBOLIC_POINTS_NOTICE_BILINGUAL = f"{SYMBOLIC_POINTS_NOTICE_EN} | {SYMBOLIC_POINTS_NOTICE_FR}"

_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_CLOUD_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_SAFE_CANDIDATE_TAG_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")

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
    r'^\s*(?:push)?(?:url|insteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r'\bnonce="([A-Za-z0-9_+=/=-]{8,128})"')
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)"
)
_ALLOWED_FEEDBACK_KEYS: frozenset[str] = frozenset(
    {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
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

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


class CanaryQueueError(ValueError):
    """Raised when canary cutover or CES retry queue arguments, guards, or safety checks fail."""


def _get_path_lock(resolved_path: Path) -> threading.RLock:
    key = str(resolved_path)
    with _PATH_LOCKS_GUARD:
        existing = _PATH_LOCKS.get(key)
        if existing is None:
            existing = threading.RLock()
            _PATH_LOCKS[key] = existing
        return existing


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise CanaryQueueError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise CanaryQueueError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


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
    """Scrub `hm-...`/`hm_live_...` tokens, Bearer credentials, emails, IPs, and home paths."""
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
    """Verify JSON or Markdown output contains zero PII, raw keys, home paths, non-loopback IPs, or `'score': <digit>`."""
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
        raise CanaryQueueError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


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
    """Validate an explicit `--expected-sha` / `target_sha` argument."""
    sanitized = sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise CanaryQueueError(
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
        raise CanaryQueueError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise CanaryQueueError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise CanaryQueueError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise CanaryQueueError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise CanaryQueueError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise CanaryQueueError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise CanaryQueueError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise CanaryQueueError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise CanaryQueueError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app, https://*.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise CanaryQueueError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise CanaryQueueError("Remote Cloud Run / Google APIs URL must use https.")
    return cleaned


def _detect_default_platform() -> str:
    raw = (sys.platform or "unknown").strip().lower()
    if raw.startswith("linux"):
        return "linux"
    if raw.startswith(("win", "cygwin", "msys")):
        return "windows"
    if raw.startswith("darwin"):
        return "darwin"
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-_")[:24]
    return cleaned if (cleaned and _PLATFORM_RE.fullmatch(cleaned)) else "unknown"


def validate_ces_feedback_entry(
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    platform: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Validate and normalize a zero-PII CES feedback item strictly matching `OnboardingFeedbackRequest` (`extra='forbid'`)."""
    if isinstance(profile, bool) or not isinstance(profile, str):
        raise CanaryQueueError("Feedback profile must be 'dev' or 'gamer'.")
    clean_profile = profile.strip().lower()
    if clean_profile not in VALID_PROFILES:
        raise CanaryQueueError(f"Invalid feedback profile {profile!r}; expected 'dev' or 'gamer'.")
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise CanaryQueueError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise CanaryQueueError("zero_game_lag must be a boolean.")
    if platform is not None and (isinstance(platform, bool) or not isinstance(platform, str)):
        raise CanaryQueueError("platform must be a string or None.")
    if note is not None and (isinstance(note, bool) or not isinstance(note, str)):
        raise CanaryQueueError("note must be a string.")

    clean_platform = _detect_default_platform()
    if platform is not None and platform.strip():
        raw_plat = platform.strip().lower()
        if raw_plat.isascii() and len(raw_plat) <= 32:
            cand = re.sub(r"[^a-z0-9_-]+", "-", raw_plat).strip("-_")[:24]
            if cand and _PLATFORM_RE.fullmatch(cand) and not any(m in cand for m in _SECRET_VALUE_MARKERS):
                clean_platform = cand

    entry = {
        "profile": clean_profile,
        "ease_rating": int(ease_rating),
        "zero_game_lag": bool(zero_game_lag),
        "platform": clean_platform,
        "note": "",
    }
    _assert_safe_output(json.dumps(entry, sort_keys=True))
    return entry


def validate_feedback_payload_dict(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Enforce `OnboardingFeedbackRequest` (`extra='forbid'`, `note == ''`, zero PII) on a payload mapping."""
    if not isinstance(payload, Mapping):
        raise CanaryQueueError("Feedback payload must be a JSON object.")
    extra_keys = set(payload.keys()) - _ALLOWED_FEEDBACK_KEYS
    if extra_keys:
        joined = ", ".join(sorted(str(k) for k in extra_keys))
        raise CanaryQueueError(f"Forbidden extra keys in feedback payload (`extra='forbid'`): {joined}.")
    raw_note = payload.get("note", "")
    if not isinstance(raw_note, str) or raw_note != "":
        raise CanaryQueueError("CES feedback payload `note` must be empty ('') to enforce zero-PII.")
    raw_platform = payload.get("platform")
    if (
        not isinstance(raw_platform, str)
        or not _PLATFORM_RE.fullmatch(raw_platform)
        or any(m in raw_platform.lower() for m in _SECRET_VALUE_MARKERS)
    ):
        raise CanaryQueueError("CES feedback payload `platform` must be a clean 1..24 char ASCII identifier.")
    validated = validate_ces_feedback_entry(
        profile=payload.get("profile"),  # type: ignore[arg-type]
        ease_rating=payload.get("ease_rating"),  # type: ignore[arg-type]
        zero_game_lag=payload.get("zero_game_lag"),  # type: ignore[arg-type]
        platform=raw_platform,
        note="",
    )
    return validated


def _assert_zero_inference_request(
    method: str,
    url: str,
    *,
    allow_feedback_post: bool = False,
) -> str:
    """Enforce allowed gateway origin, zero LLM inference (`completions` blocked), and method allowlist."""
    verb = (method or "").strip().upper()
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "hivemind/coder" in low_path:
        raise CanaryQueueError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    if verb in ("GET", "HEAD") or (
        allow_feedback_post and verb == "POST" and parts.path == "/v1/hivemind/onboarding/feedback"
    ):
        return origin
    raise CanaryQueueError(
        f"Zero-inference invariant violated: HTTP {verb} {parts.path} is forbidden."
    )


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib HTTP helper (`ProxyHandler({})`) for zero-inference canary GETs and zero-PII CES POSTs."""
    verb = (method or "").strip().upper()
    is_feedback_post = verb == "POST" and urlsplit(url).path == "/v1/hivemind/onboarding/feedback"
    _assert_zero_inference_request(verb, url, allow_feedback_post=is_feedback_post)
    if payload is not None and not is_feedback_post:
        raise CanaryQueueError("Zero-inference canary verifier never sends request payloads on GET/HEAD.")
    body_bytes = (
        json.dumps(validate_feedback_payload_dict(payload), sort_keys=True).encode("utf-8")
        if is_feedback_post
        else None
    )

    clean_extra_headers: dict[str, str] = {}
    for hk, hv in (headers or {}).items():
        ks = str(hk).strip()
        vs = str(hv).strip()
        if any(ord(c) < 32 or ord(c) == 127 for c in ks + vs):
            raise CanaryQueueError("HTTP headers must not contain control characters.")
        clean_extra_headers[ks] = vs

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-canary-ces-queue/1.0",
        **({"Content-Type": "application/json"} if body_bytes is not None else {}),
        **clean_extra_headers,
    }

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
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
    allow_feedback_post: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Invoke a 2-arg `(method, url)` or 4-arg `(method, url, payload, headers)` HTTP callable safely."""
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, allow_feedback_post=allow_feedback_post)
    clean_payload = (
        validate_feedback_payload_dict(payload)
        if (allow_feedback_post and verb == "POST")
        else None
    )
    sig = inspect.signature(http_fn)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    try:
        if has_varargs or len(positional) >= 4:
            raw_res = http_fn(verb, url, clean_payload, headers)
        elif len(positional) == 3:
            raw_res = http_fn(verb, url, clean_payload)
        else:
            raw_res = http_fn(verb, url)
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


def _find_corporate_markers(value: str) -> list[str]:
    if not value:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(value)]


def _extract_git_remote_urls(workspace: Path) -> list[str]:
    """Read `.git/config` (including git worktree `gitdir:` + `commondir` indirection) locally."""
    git_entry = workspace / ".git"
    cfg_paths: list[Path] = [git_entry / "config"]
    try:
        if git_entry.is_file():
            lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines and lines[0].strip().lower().startswith("gitdir:"):
                raw_gitdir = lines[0].split(":", 1)[1].strip()
                gitdir = Path(raw_gitdir) if Path(raw_gitdir).is_absolute() else (workspace / raw_gitdir).resolve()
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
    lang: str = "en",
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
            f"REFUS DE CONFORMITÉ ({joined}) : environnement d'entreprise détecté "
            f"(0 appel réseau, 0 écriture fichier). {COMPLIANCE_NOTICE_BILINGUAL} | {SYMBOLIC_POINTS_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE BLOCKED ({joined}): corporate environment indicator detected "
            f"(0 network calls, 0 file writes). {COMPLIANCE_NOTICE_BILINGUAL} | {SYMBOLIC_POINTS_NOTICE_BILINGUAL}"
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
    queue_file: Path | str | None = None,
    home: Path | None = None,
    lang_code: str = "en",
) -> dict[str, Any]:
    """Build a fail-closed report when `evaluate_personal_pc_compliance` detects a corporate environment."""
    safe_gw = DEFAULT_GATEWAY
    try:
        safe_gw = validate_gateway_url(gateway)
    except CanaryQueueError:
        safe_gw = DEFAULT_GATEWAY
    redacted_q = (
        redact_home_path(queue_file, home=home)
        if queue_file is not None
        else f"~/{DEFAULT_QUEUE_REL}"
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": mode,
            "lang": lang_code,
            "gateway": safe_gw,
            "queue_file": redacted_q,
            "compliance_ok": False,
            "canary_ok": False,
            "promote_to_100_pct": False,
            "rollback_triggered": False,
            "plan_ready": False,
            "enqueued": False,
            "deduplicated": False,
            "flush_executed": False,
            "flush_ok": False,
            "flushed_count": 0,
            "delivered_count": 0,
            "network_calls": 0,
            "file_writes": 0,
            "zero_inference_verified": True,
            "diagnostics": [str(compliance_doc["summary"])],
            "compliance": dict(compliance_doc),
        }
    )


def _validate_cloud_identifier(value: str, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise CanaryQueueError(f"Invalid Cloud Run {label}: expected string.")
    cleaned = value.strip()
    low = cleaned.lower()
    if (
        not _SAFE_CLOUD_TOKEN_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        raise CanaryQueueError(f"Invalid Cloud Run {label}: {sanitize_text_for_display(cleaned)!r}.")
    return cleaned


def _validate_candidate_tag(value: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise CanaryQueueError("Candidate tag must be a lowercase DNS label string.")
    cleaned = value.strip().lower()
    if (
        not _SAFE_CANDIDATE_TAG_RE.fullmatch(cleaned)
        or cleaned.endswith("-")
        or "---" in cleaned
        or cleaned.startswith(("hm-", "sk-"))
        or any(marker in cleaned for marker in _SECRET_VALUE_MARKERS)
    ):
        raise CanaryQueueError(f"Invalid Cloud Run candidate traffic tag: {sanitize_text_for_display(value)!r}.")
    return cleaned


def _validate_canary_percent(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 99):
        raise CanaryQueueError("canary_percent must be an integer between 1 and 99.")
    return int(value)


def _zero_grants_invariant_holds(active_script: str) -> bool:
    """Verify starter, registration bonus, allow_bonuses, and partner grants default strictly to `0`."""
    required_defaults = (
        ("STARTER_GRANT_HMT", "HIVEMIND_STARTER_GRANT_HMT"),
        ("REGISTRATION_BONUS_HMT", "HIVEMIND_REGISTRATION_BONUS_HMT"),
        ("ALLOW_BONUSES", "HIVEMIND_ALLOW_BONUSES"),
        ("PARTNER_TESTING_GRANTS", "HIVEMIND_PARTNER_TESTING_GRANTS"),
    )
    for local_var, env_var in required_defaults:
        default_zero_pat = re.compile(
            rf'(?:{local_var}=\"\$\{{{env_var}:-0\}}\"|{env_var}=0\b)'
        )
        nonzero_pat = re.compile(
            rf'(?:{local_var}=\"\$\{{{env_var}:-[1-9]\d*(?:\.\d+)?\}}\"|{env_var}=[1-9]\d*(?:\.\d+)?\b|{local_var}=[\"\']?[1-9]\d*)'
        )
        if not default_zero_pat.search(active_script) or nonzero_pat.search(active_script):
            return False
    return True


def audit_deploy_script_finops(
    deploy_script_path: Path | str | None = None,
    *,
    lang: str = "en",
) -> dict[str, Any]:
    """Audit `deploy/gcp_openrouter/deploy.sh` FinOps, Litestream, Secret Manager, and 0-grant invariants."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    target_path = (
        Path(deploy_script_path)
        if deploy_script_path is not None and str(deploy_script_path).strip()
        else (REPO_ROOT / DEFAULT_DEPLOY_SCRIPT_REL)
    )
    display_path = redact_home_path(target_path)

    script_present = False
    active_script = ""
    diagnostics: list[str] = []
    try:
        if target_path.is_file():
            raw_script = target_path.read_text(encoding="utf-8")
            script_present = True
            active_lines = [
                ln for ln in raw_script.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
            ]
            active_script = "\n".join(active_lines)
    except OSError:
        script_present = False

    if not script_present:
        diagnostics.append(
            f"Script de déploiement introuvable ou illisible : `{display_path}`."
            if is_fr
            else f"Deploy script missing or unreadable: `{display_path}`."
        )

    invariant_specs: tuple[tuple[str, str, bool, str, str], ...] = (
        (
            "min_instances_zero",
            "--min-instances 0",
            bool(
                script_present
                and re.search(r"--min-instances(?:=|\s+)0\b", active_script)
                and not re.search(r"--min-instances(?:=|\s+)[1-9]\d*\b", active_script)
            ),
            "Scale-to-zero FinOps cap (--min-instances 0) preserves 0 idle Cloud Run spend.",
            "Plafond FinOps scale-to-zero (--min-instances 0) garantissant 0 coût au repos.",
        ),
        (
            "max_instances_one",
            "--max-instances 1",
            bool(
                script_present
                and re.search(r"--max-instances(?:=|\s+)1\b", active_script)
                and not re.search(r"--max-instances(?:=|\s+)(?:0|[2-9]|\d{2,})\b", active_script)
            ),
            "Single-instance SQLite WAL + Litestream writer lock (--max-instances 1).",
            "Instance unique SQLite WAL + Litestream (--max-instances 1).",
        ),
        (
            "no_cpu_throttling",
            "--no-cpu-throttling",
            bool(
                script_present
                and re.search(r"--no-cpu-throttling\b", active_script)
                and not re.search(r"(?<!no-)--cpu-throttling\b", active_script)
            ),
            "Continuous CPU allocation (--no-cpu-throttling) so Litestream flushes WAL between requests.",
            "Allocation CPU continue (--no-cpu-throttling) pour la réplication Litestream.",
        ),
        (
            "concurrency_80",
            "--concurrency 80",
            bool(
                script_present
                and re.search(r"--concurrency(?:=|\s+)80\b", active_script)
                and not re.search(r"--concurrency(?:=|\s+)(?!80\b)\d+\b", active_script)
            ),
            "Bounded request concurrency (--concurrency 80) on single Cloud Run instance.",
            "Concurrence bornée (--concurrency 80) sur l'instance unique Cloud Run.",
        ),
        (
            "litestream_gcs_replica",
            "HIVEMIND_ROUTER_REPLICA_URL=gcs://",
            bool(
                script_present
                and "HIVEMIND_ROUTER_REPLICA_URL=gcs://" in active_script
            ),
            "Durable Litestream SQLite GCS replica configured (HIVEMIND_ROUTER_REPLICA_URL=gcs://...).",
            "Réplication durable SQLite Litestream vers GCS (HIVEMIND_ROUTER_REPLICA_URL=gcs://...).",
        ),
        (
            "secret_manager_router_secret",
            "--set-secrets HIVEMIND_ROUTER_SECRET=...",
            bool(
                script_present
                and re.search(r"--set-secrets\s+[\"']?[^\"'\n]*HIVEMIND_ROUTER_SECRET=", active_script)
                and "ENV_VARS=\"HIVEMIND_ROUTER_SECRET=" not in active_script
                and ",HIVEMIND_ROUTER_SECRET=" not in active_script
            ),
            "Router HMAC secret mounted strictly via Secret Manager (HIVEMIND_ROUTER_SECRET).",
            "Secret HMAC routeur monté exclusivement via Secret Manager (HIVEMIND_ROUTER_SECRET).",
        ),
        (
            "zero_starter_and_bonus_grants",
            "STARTER/BONUS/PARTNER_GRANTS=0",
            bool(script_present and _zero_grants_invariant_holds(active_script)),
            "Zero starter/registration/partner bonus grants (0.00 EUR / 0.00 USD reciprocity invariant).",
            "Zéro bonus de départ ou partenaire (invariant 0,00 EUR / 0,00 USD).",
        ),
    )

    invariants: list[dict[str, Any]] = []
    failed_invariants: list[str] = []
    for inv_id, expected_tok, passed, desc_en, desc_fr in invariant_specs:
        invariants.append(
            {
                "id": inv_id,
                "expected": expected_tok,
                "passed": passed,
                "description": desc_fr if is_fr else desc_en,
            }
        )
        if not passed:
            failed_invariants.append(inv_id)
            if script_present:
                diagnostics.append(
                    f"Invariant FinOps manquant ou violé dans `{display_path}` : `{expected_tok}` ({inv_id})."
                    if is_fr
                    else f"Missing or violated FinOps invariant in `{display_path}`: `{expected_tok}` ({inv_id})."
                )

    finops_ok = bool(script_present and not failed_invariants)
    return _attach_receipt(
        {
            "deploy_script": display_path,
            "deploy_script_present": script_present,
            "finops_ok": finops_ok,
            "invariants_verified_count": sum(1 for row in invariants if row["passed"]),
            "invariants_total_count": len(invariants),
            "checks": {row["id"]: bool(row["passed"]) for row in invariants},
            "invariants": invariants,
            "failed_invariants": failed_invariants,
            "diagnostics": diagnostics,
        }
    )


def build_canary_traffic_plan(
    *,
    gateway: str = DEFAULT_GATEWAY,
    candidate_tag: str = DEFAULT_CANDIDATE_TAG,
    canary_percent: int = DEFAULT_CANARY_PERCENT,
    previous_revision: str | None = None,
    target_sha: str | None = None,
    service_name: str = DEFAULT_CLOUD_RUN_SERVICE,
    region: str = DEFAULT_CLOUD_RUN_REGION,
    deploy_script_path: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Build a dry-run FinOps-safe Cloud Run canary deploy, traffic-split, promotion, and rollback plan."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_tag = _validate_candidate_tag(candidate_tag)
    clean_pct = _validate_canary_percent(canary_percent)
    stable_pct = 100 - clean_pct
    clean_service = _validate_cloud_identifier(service_name, "service name")
    clean_region = _validate_cloud_identifier(region, "region")

    has_raw_rev = previous_revision is not None and (
        not isinstance(previous_revision, str) or bool(previous_revision.strip())
    )
    clean_rev = sanitize_cloud_run_revision(previous_revision) if has_raw_rev else None
    if has_raw_rev and clean_rev is None:
        raise CanaryQueueError(
            "Invalid Cloud Run previous_revision; expected safe K_REVISION identifier without secrets."
        )

    resolved_sha = (
        validate_expected_sha(target_sha)
        if (target_sha is not None and str(target_sha).strip())
        else None
    )
    sha_expr = resolved_sha if resolved_sha is not None else "$(git rev-parse --short=12 HEAD)"
    rev_expr = clean_rev if clean_rev is not None else "${PREVIOUS_K_REVISION:?set_PREVIOUS_K_REVISION}"

    finops_audit = audit_deploy_script_finops(deploy_script_path=deploy_script_path, lang=lang_code)
    finops_ok = bool(finops_audit["finops_ok"])

    stage_canary_deploy_command = (
        f'HIVEMIND_BUILD_SHA="{sha_expr}" '
        'GCP_PROJECT="${GCP_PROJECT:?set_GCP_PROJECT}" '
        'HIVEMIND_ROUTER_ADMIN_KEY_HASH="${HIVEMIND_ROUTER_ADMIN_KEY_HASH:?set_ADMIN_HASH}" '
        f'REGION="{clean_region}" '
        f"bash {DEFAULT_DEPLOY_SCRIPT_REL} && "
        f"gcloud run services update {clean_service} "
        '--project "${GCP_PROJECT}" '
        f'--region "{clean_region}" '
        "--min-instances 0 --max-instances 1 --no-cpu-throttling --concurrency 80 "
        f'--update-env-vars "HIVEMIND_BUILD_SHA={sha_expr}" && '
        f"gcloud run services update-traffic {clean_service} "
        '--project "${GCP_PROJECT}" '
        f'--region "{clean_region}" '
        f'--set-tags "{clean_tag}=LATEST" '
        f'--to-revisions "{rev_expr}=100"'
    )
    canary_traffic_split_command = (
        f"gcloud run services update-traffic {clean_service} "
        '--project "${GCP_PROJECT:?set_GCP_PROJECT}" '
        f'--region "{clean_region}" '
        f'--to-tags "{clean_tag}={clean_pct}"'
    )
    promote_100_pct_command = (
        f"gcloud run services update-traffic {clean_service} "
        '--project "${GCP_PROJECT:?set_GCP_PROJECT}" '
        f'--region "{clean_region}" '
        f'--to-tags "{clean_tag}=100" && '
        f"gcloud run services update {clean_service} "
        '--project "${GCP_PROJECT}" '
        f'--region "{clean_region}" '
        "--min-instances 0 --max-instances 1 --no-cpu-throttling --concurrency 80"
    )
    rollback_command = (
        f"gcloud run services update-traffic {clean_service} "
        '--project "${GCP_PROJECT:?set_GCP_PROJECT}" '
        f'--region "{clean_region}" '
        f'--to-revisions "{rev_expr}=100" '
        f'--remove-tags "{clean_tag}" && '
        f"gcloud run services update {clean_service} "
        '--project "${GCP_PROJECT}" '
        f'--region "{clean_region}" '
        "--min-instances 0 --max-instances 1 --no-cpu-throttling --concurrency 80"
    )
    sha_arg = f" --expected-sha {resolved_sha}" if resolved_sha is not None else ""
    rev_arg = f" --previous-revision {clean_rev}" if clean_rev is not None else ""
    verify_canary_command = (
        f"python3 scripts/noel_canary_cutover_and_ces_queue.py --gateway {clean_gw} "
        f"--verify-canary --canary-percent {clean_pct}{sha_arg}{rev_arg}"
    )

    summary = (
        f"Plan canari Cloud Run dry-run prêt (`{clean_tag}`={clean_pct}% / `{rev_expr}`={stable_pct}%, "
        f"FinOps={finops_ok}, 0 appel gcloud)."
        if is_fr
        else f"Dry-run Cloud Run canary plan ready (`{clean_tag}`={clean_pct}% / `{rev_expr}`={stable_pct}%, "
        f"FinOps={finops_ok}, 0 gcloud calls)."
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "canary_traffic_plan",
            "lang": lang_code,
            "gateway": clean_gw,
            "service_name": clean_service,
            "region": clean_region,
            "candidate_tag": clean_tag,
            "canary_percent": clean_pct,
            "stable_percent": stable_pct,
            "traffic_split_pct": {"candidate": clean_pct, "stable": stable_pct},
            "previous_revision": clean_rev,
            "revision_pinned": clean_rev is not None,
            "target_build_sha": resolved_sha,
            "finops_ok": finops_ok,
            "plan_ready": finops_ok,
            "finops_audit": finops_audit,
            "dry_run_only": True,
            "gcloud_executed": False,
            "cloud_state_mutated": False,
            "zero_inference_verified": True,
            "stage_canary_deploy_command": stage_canary_deploy_command,
            "canary_traffic_split_command": canary_traffic_split_command,
            "promote_100_pct_command": promote_100_pct_command,
            "rollback_command": rollback_command,
            "verify_canary_command": verify_canary_command,
            "diagnostics": list(finops_audit["diagnostics"]),
            "summary": summary,
            "compliance": {
                "personal_pc_only": True,
                "amadeus_corporate_prohibition": True,
                "financial_promise": FINANCIAL_PROMISE,
                "monetary_cost": FINANCIAL_PROMISE,
                "license": "AGPL-3.0",
                "notice_en": COMPLIANCE_NOTICE_EN,
                "notice_fr": COMPLIANCE_NOTICE_FR,
                "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            },
        }
    )


def _validate_manifest_body(
    body: Mapping[str, Any],
    *,
    lang: str,
) -> tuple[bool, str, str | None, list[str], list[str], list[str]]:
    """Validate `/v1/hivemind/onboarding/manifest` build_sha, holiday_routes, and script allowlist."""
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
    """Inspect `/noel`, `/quick`, `/raid-noel`, and `/arena-noel` CSP nonces, DOM safety, and 0.00 EUR/USD copy."""
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
        and (not csp_hdr or (hdr_nonces and set(hdr_nonces) == {nonce} and "default-src 'none'" in csp_hdr.lower()))
        and nonce not in seen_nonces
    )
    if nonce is not None:
        seen_nonces.add(nonce)

    dom_clean = not (_FORBIDDEN_DOM_SINK_RE.search(html_text) or _STATIC_FIG_UNIT_RE.search(html_text))
    low_html = html_text.lower()
    compliance_banner_ok = (
        (
            "100% personal" in low_html
            or "100 % personnel" in low_html
            or "100 % sur pc personnel" in low_html
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
    raw_ease = body.get("average_ease_rating") if "average_ease_rating" in body else body.get("mean_ease_rating")
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
            f"GET {path} a répondu HTTP 404 : révision Cloud Run en retard sur `main` (rollback/cutover requis)."
            if is_fr
            else f"GET {path} returned HTTP 404: Cloud Run revision lags behind `main` (rollback/cutover required)."
        )
    if status_code == 429:
        return (
            f"GET {path} limité par quota HTTP 429 pendant la vérification canari."
            if is_fr
            else f"GET {path} rate-limited with HTTP 429 during canary verification."
        )
    if status_code == 0:
        return (
            f"GET {path} injoignable (erreur réseau/timeout vers la passerelle)."
            if is_fr
            else f"GET {path} unreachable (gateway network/timeout error)."
        )
    return (
        f"GET {path} en échec avec HTTP {status_code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} failed with HTTP {status_code} (expected HTTP 200)."
    )


def verify_canary_cutover(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    previous_revision: str | None = None,
    canary_percent: int = DEFAULT_CANARY_PERCENT,
    candidate_tag: str = DEFAULT_CANDIDATE_TAG,
    service_name: str = DEFAULT_CLOUD_RUN_SERVICE,
    region: str = DEFAULT_CLOUD_RUN_REGION,
    deploy_script_path: Path | str | None = None,
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify FinOps invariants and all 8 zero-inference canary routes; decide promotion vs rollback."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        return _compliance_blocked_report(
            "verify_canary",
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
    traffic_plan = build_canary_traffic_plan(
        gateway=clean_gw,
        candidate_tag=candidate_tag,
        canary_percent=canary_percent,
        previous_revision=previous_revision,
        target_sha=resolved_expected_sha,
        service_name=service_name,
        region=region,
        deploy_script_path=deploy_script_path,
        lang=lang_code,
    )
    finops_ok = bool(traffic_plan["finops_ok"])

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = list(traffic_plan["diagnostics"])

    manifest_valid = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    missing_manifest_routes: list[str] = []
    missing_manifest_scripts: list[str] = []

    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    html_contract_ok = True
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {
        "schema_valid": False,
        "total_responses": None,
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

    for path, expected_kind, group in CANARY_ZERO_INFERENCE_ROUTES:
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
            h_ok, h_doc, h_issues = _inspect_holiday_html_contract(
                path,
                body,
                seen_nonces,
                lang=lang_code,
            )
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
            rev_label = cloud_run_revision or ("aucune" if is_fr else "none")
            diagnostics.append(
                f"Dérive de SHA canari : `build_sha`='{live_build_sha}' (révision={rev_label}) "
                f"!= attendu '{resolved_expected_sha}'."
                if is_fr
                else f"Canary SHA drift: live `build_sha`='{live_build_sha}' (revision={rev_label}) "
                f"!= expected '{resolved_expected_sha}'."
            )

    html_all_ok = bool(html_contract_ok and len(html_contracts) == len(CANARY_HTML_ROUTES))
    routes_verified_count = sum(1 for row in route_rows if row["ok"])
    routes_total_count = len(route_rows)
    canary_ok = bool(
        finops_ok
        and routes_verified_count == routes_total_count
        and manifest_valid
        and html_all_ok
        and feedback_summary_ok
        and producer_fleet_ok
        and not sha_drift
    )
    promote_to_100_pct = canary_ok
    rollback_triggered = not canary_ok

    if rollback_triggered:
        diagnostics.insert(
            0,
            "ROLLBACK DÉCLENCHÉ : le canari Cloud Run a échoué — restaurez immédiatement le trafic via `gcloud run services update-traffic`."
            if is_fr
            else "ROLLBACK TRIGGERED: Cloud Run canary verification failed — restore stable traffic via `gcloud run services update-traffic`."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "verify_canary",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": bool(comp_doc["compliance_ok"]),
            "finops_ok": finops_ok,
            "canary_ok": canary_ok,
            "promote_to_100_pct": promote_to_100_pct,
            "rollback_triggered": rollback_triggered,
            "stale_cloud_run_detected": len(stale_404_routes) > 0,
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
            "unique_csp_nonces_count": len(seen_nonces),
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "producer_fleet_ok": producer_fleet_ok,
            "producer_fleet": producer_fleet_doc,
            "dry_run_only": True,
            "gcloud_executed": False,
            "cloud_state_mutated": False,
            "zero_inference_verified": True,
            "canary_percent": int(traffic_plan["canary_percent"]),
            "candidate_tag": str(traffic_plan["candidate_tag"]),
            "previous_revision": traffic_plan["previous_revision"],
            "promote_100_pct_command": str(traffic_plan["promote_100_pct_command"]),
            "rollback_command": str(traffic_plan["rollback_command"]),
            "traffic_plan": traffic_plan,
            "routes": route_rows,
            "diagnostics": [sanitize_text_for_display(d) for d in diagnostics],
            "compliance": dict(comp_doc),
        }
    )


def compute_deterministic_backoff_s(
    retry_count: int,
    *,
    base_backoff_s: float = DEFAULT_BASE_BACKOFF_S,
    max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
) -> float:
    """Compute bounded, jitter-free exponential retry backoff in seconds (`base * 2**(retry_count - 1)`)."""
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        raise CanaryQueueError("retry_count must be a non-negative integer.")
    if base_backoff_s <= 0.0 or max_backoff_s < base_backoff_s:
        raise CanaryQueueError("Invalid backoff bounds.")
    if retry_count == 0:
        return 0.0
    bounded_step = min(retry_count, MAX_RETRY_COUNT)
    raw = float(base_backoff_s) * float(2 ** (bounded_step - 1))
    return round(min(float(max_backoff_s), raw), 3)


def compute_ces_event_id(
    payload: Mapping[str, Any],
    *,
    idempotency_key: str | None = None,
) -> str:
    """Compute a deterministic 64-char SHA-256 `event_id` from caller `idempotency_key` or canonical payload."""
    clean_payload = validate_feedback_payload_dict(payload)
    if idempotency_key is not None:
        if isinstance(idempotency_key, bool) or not isinstance(idempotency_key, str):
            raise CanaryQueueError("idempotency_key must be a string.")
        cleaned_key = idempotency_key.strip()
        if len(cleaned_key) > 256 or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned_key):
            raise CanaryQueueError("idempotency_key must be 1..256 printable characters without control bytes.")
        if cleaned_key:
            if _SHA256_HEX_RE.fullmatch(cleaned_key):
                return cleaned_key
            return hashlib.sha256(f"hivemind.ces.idempotency/1:{cleaned_key}".encode()).hexdigest()
    canonical = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"hivemind.ces.payload/1:{canonical}".encode()).hexdigest()


def _classify_flush_http_status(status_code: int) -> tuple[str, str]:
    """Classify HTTP outcome into `'delivered'`, `'retryable'`, or `'permanent_quarantine'`."""
    if 200 <= status_code < 300:
        return "delivered", "ok"
    if status_code == 404:
        return "retryable", "stale_cloud_run_404"
    if status_code == 408:
        return "retryable", "request_timeout_408"
    if status_code == 429:
        return "retryable", "rate_limited_429"
    if status_code == 0:
        return "retryable", "network_timeout_or_unreachable"
    if status_code >= 500:
        return "retryable", f"gateway_{status_code}"
    return "permanent_quarantine", f"permanent_http_{status_code}"


class CesFeedbackRetryQueue:
    """Atomic, `0600`-permissioned, bounded, idempotent local CES & zero-game-lag feedback retry spool."""

    def __init__(
        self,
        queue_file: Path | str | None = None,
        *,
        max_entries: int = DEFAULT_MAX_QUEUE_ENTRIES,
        home: Path | None = None,
        base_backoff_s: float = DEFAULT_BASE_BACKOFF_S,
        max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
    ) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not (1 <= max_entries <= MAX_QUEUE_ENTRIES_CEILING)
        ):
            raise CanaryQueueError(
                f"max_entries must be an integer between 1 and {MAX_QUEUE_ENTRIES_CEILING}."
            )
        if base_backoff_s <= 0.0 or max_backoff_s < base_backoff_s:
            raise CanaryQueueError("Invalid retry backoff configuration.")
        self._home = home
        resolved_home = home if home is not None else Path.home()
        raw_path = (
            Path(queue_file)
            if (queue_file is not None and str(queue_file).strip())
            else (resolved_home / DEFAULT_QUEUE_REL)
        )
        if str(raw_path).startswith("~/") or str(raw_path) == "~":
            suffix = str(raw_path)[2:] if str(raw_path).startswith("~/") else ""
            raw_path = (resolved_home / suffix) if suffix else resolved_home
        self._path = raw_path.resolve()
        self._max_entries = int(max_entries)
        self._max_delivered_ids = max(MAX_DELIVERED_LEDGER_MINIMUM, self._max_entries * 8)
        self._base_backoff_s = float(base_backoff_s)
        self._max_backoff_s = float(max_backoff_s)
        self._rlock = _get_path_lock(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def redacted_path(self) -> str:
        return redact_home_path(self._path, home=self._home)

    @contextmanager
    def _locked_spool(self, *, create_parent: bool, lock_if_exists: bool = False) -> Iterator[None]:
        with self._rlock:
            lock_fd: int | None = None
            should_flock = create_parent or (lock_if_exists and self._path.is_file())
            if should_flock:
                parent = self._path.parent
                parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if parent.name == ".hivemind":
                    with suppress(OSError):
                        os.chmod(parent, 0o700)
                if _fcntl is not None:
                    lock_path = self._path.with_name(f".{self._path.name}.lock")
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

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema": QUEUE_SCHEMA,
            "max_entries": self._max_entries,
            "dropped_overflow_count": 0,
            "pending": [],
            "delivered_event_ids": [],
            "quarantined": [],
        }

    def _load_state_unlocked(self) -> tuple[dict[str, Any], bool]:
        """Load and sanitize spool state; returns `(state, mutated_during_recovery)`."""
        if not self._path.is_file():
            return self._empty_state(), False
        try:
            raw_text = self._path.read_text(encoding="utf-8")
            parsed = json.loads(raw_text)
        except (OSError, json.JSONDecodeError):
            recovered = self._empty_state()
            recovered["quarantined"] = [
                {
                    "event_id": hashlib.sha256(b"corrupted_spool_recovery").hexdigest(),
                    "reason": "corrupted_queue_json_recovered",
                    "http_status": 0,
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
                if isinstance(item, str) and _SHA256_HEX_RE.fullmatch(item) and item not in seen_delivered:
                    delivered.append(item)
                    seen_delivered.add(item)
                else:
                    mutated = True
        if len(delivered) > self._max_delivered_ids:
            delivered = delivered[-self._max_delivered_ids :]
            mutated = True

        quarantined: list[dict[str, Any]] = []
        raw_quar = parsed.get("quarantined")
        if isinstance(raw_quar, list):
            for q_item in raw_quar[-self._max_entries :]:
                if isinstance(q_item, Mapping) and isinstance(q_item.get("event_id"), str):
                    q_eid = q_item["event_id"]
                    if _SHA256_HEX_RE.fullmatch(q_eid):
                        quarantined.append(
                            {
                                "event_id": q_eid,
                                "reason": sanitize_text_for_display(
                                    str(q_item.get("reason", "quarantined"))[:80],
                                    home=self._home,
                                ),
                                "http_status": (
                                    int(q_item["http_status"])
                                    if not isinstance(q_item.get("http_status"), bool)
                                    and isinstance(q_item.get("http_status"), int)
                                    else 422
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
                    if isinstance(raw_eid, str) and _SHA256_HEX_RE.fullmatch(raw_eid)
                    else hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode("utf-8")).hexdigest()
                )
                if eid in seen_delivered or eid in seen_pending:
                    mutated = True
                    continue
                try:
                    clean_payload = validate_feedback_payload_dict(entry.get("payload"))  # type: ignore[arg-type]
                except CanaryQueueError:
                    mutated = True
                    quarantined.append(
                        {
                            "event_id": eid,
                            "reason": "tampered_or_invalid_local_payload",
                            "http_status": 422,
                        }
                    )
                    quarantined = quarantined[-self._max_entries :]
                    continue

                raw_rc = entry.get("retry_count", 0)
                rc = (
                    min(max(0, int(raw_rc)), MAX_RETRY_COUNT)
                    if not isinstance(raw_rc, bool) and isinstance(raw_rc, int)
                    else 0
                )
                raw_bo = entry.get("backoff_seconds", 0.0)
                bo = (
                    round(max(0.0, float(raw_bo)), 3)
                    if not isinstance(raw_bo, bool) and isinstance(raw_bo, (int, float))
                    else 0.0
                )
                raw_next = entry.get("next_attempt_epoch", 0.0)
                nxt = (
                    round(max(0.0, float(raw_next)), 3)
                    if not isinstance(raw_next, bool) and isinstance(raw_next, (int, float))
                    else 0.0
                )
                raw_created = entry.get("enqueued_epoch", 0.0)
                created = (
                    round(max(0.0, float(raw_created)), 3)
                    if not isinstance(raw_created, bool) and isinstance(raw_created, (int, float))
                    else 0.0
                )
                raw_status = entry.get("last_http_status")
                last_status = (
                    int(raw_status)
                    if not isinstance(raw_status, bool) and isinstance(raw_status, int)
                    else None
                )
                last_reason = sanitize_text_for_display(
                    str(entry.get("last_reason") or ""),
                    home=self._home,
                )[:80]

                pending.append(
                    {
                        "event_id": eid,
                        "payload": clean_payload,
                        "enqueued_epoch": created,
                        "retry_count": rc,
                        "backoff_seconds": bo,
                        "next_attempt_epoch": nxt,
                        "last_http_status": last_status,
                        "last_reason": last_reason,
                    }
                )
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
                "schema": QUEUE_SCHEMA,
                "max_entries": self._max_entries,
                "dropped_overflow_count": dropped_count,
                "pending": pending,
                "delivered_event_ids": delivered,
                "quarantined": quarantined[-self._max_entries :],
            },
            mutated,
        )

    def _atomic_save_unlocked(self, state: Mapping[str, Any]) -> None:
        """Persist `state` atomically via `0600` temp file + `fsync` + `os.replace`."""
        doc = {
            "schema": QUEUE_SCHEMA,
            "max_entries": self._max_entries,
            "dropped_overflow_count": int(state.get("dropped_overflow_count", 0)),
            "pending": list(state.get("pending", [])),
            "delivered_event_ids": list(state.get("delivered_event_ids", []))[-self._max_delivered_ids :],
            "quarantined": list(state.get("quarantined", []))[-self._max_entries :],
        }
        serialized = _assert_safe_output(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False)) + "\n"
        tmp_path = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with suppress(OSError):
                os.chmod(tmp_path, 0o600)
            os.write(fd, serialized.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, self._path)
        with suppress(OSError):
            os.chmod(self._path, 0o600)

    def snapshot(self, *, lang: str = "en") -> dict[str, Any]:
        """Return a read-only sanitized snapshot of the local CES retry queue."""
        lang_code = normalize_lang(lang)
        with self._locked_spool(create_parent=False):
            state, _ = self._load_state_unlocked()
            exists = self._path.is_file()
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "queue_snapshot",
                "lang": lang_code,
                "queue_file": self.redacted_path,
                "queue_exists": exists,
                "max_entries": self._max_entries,
                "pending_count": len(state["pending"]),
                "delivered_ledger_count": len(state["delivered_event_ids"]),
                "quarantined_count": len(state["quarantined"]),
                "dropped_overflow_count": int(state["dropped_overflow_count"]),
                "pending_event_ids": [item["event_id"] for item in state["pending"]],
            }
        )

    def enqueue(
        self,
        *,
        profile: str = "dev",
        ease_rating: int = 5,
        zero_game_lag: bool = True,
        platform: str | None = None,
        note: str = "",
        idempotency_key: str | None = None,
        enforce_personal_pc: bool = True,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        now_epoch: float | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Validate compliance & schema, deduplicate by SHA-256 `event_id`, and atomically persist (`0600`)."""
        lang_code = normalize_lang(lang)
        comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if enforce_personal_pc and not comp_doc["compliance_ok"]:
            return _compliance_blocked_report(
                "enqueue_ces",
                DEFAULT_GATEWAY,
                comp_doc,
                queue_file=self._path,
                home=self._home,
                lang_code=lang_code,
            )

        clean_payload = validate_ces_feedback_entry(
            profile=profile,
            ease_rating=ease_rating,
            zero_game_lag=zero_game_lag,
            platform=platform,
            note=note,
        )
        event_id = compute_ces_event_id(clean_payload, idempotency_key=idempotency_key)
        ts = round(float(now_epoch if now_epoch is not None else time.time()), 3)

        with self._locked_spool(create_parent=True):
            state, recovered = self._load_state_unlocked()
            delivered_ids: list[str] = state["delivered_event_ids"]
            pending: list[dict[str, Any]] = state["pending"]

            if event_id in delivered_ids:
                if recovered or not self._path.is_file():
                    self._atomic_save_unlocked(state)
                return _attach_receipt(
                    {
                        "schema": SCHEMA,
                        "object": RECEIPT_OBJECT,
                        "campaign_id": CAMPAIGN_ID,
                        "mode": "enqueue_ces",
                        "lang": lang_code,
                        "compliance_ok": bool(comp_doc["compliance_ok"]),
                        "queue_file": self.redacted_path,
                        "enqueued": False,
                        "deduplicated": True,
                        "duplicate_state": "already_delivered",
                        "evicted_oldest": False,
                        "event_id": event_id,
                        "payload": clean_payload,
                        "pending_count": len(pending),
                        "delivered_ledger_count": len(delivered_ids),
                        "quarantined_count": len(state["quarantined"]),
                        "dropped_overflow_count": int(state["dropped_overflow_count"]),
                        "network_calls": 0,
                        "compliance": dict(comp_doc),
                    }
                )

            if any(item["event_id"] == event_id for item in pending):
                if recovered or not self._path.is_file():
                    self._atomic_save_unlocked(state)
                return _attach_receipt(
                    {
                        "schema": SCHEMA,
                        "object": RECEIPT_OBJECT,
                        "campaign_id": CAMPAIGN_ID,
                        "mode": "enqueue_ces",
                        "lang": lang_code,
                        "compliance_ok": bool(comp_doc["compliance_ok"]),
                        "queue_file": self.redacted_path,
                        "enqueued": False,
                        "deduplicated": True,
                        "duplicate_state": "already_pending",
                        "evicted_oldest": False,
                        "event_id": event_id,
                        "payload": clean_payload,
                        "pending_count": len(pending),
                        "delivered_ledger_count": len(delivered_ids),
                        "quarantined_count": len(state["quarantined"]),
                        "dropped_overflow_count": int(state["dropped_overflow_count"]),
                        "network_calls": 0,
                        "compliance": dict(comp_doc),
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
                    "enqueued_epoch": ts,
                    "retry_count": 0,
                    "backoff_seconds": 0.0,
                    "next_attempt_epoch": ts,
                    "last_http_status": None,
                    "last_reason": "",
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
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "enqueue_ces",
                "lang": lang_code,
                "compliance_ok": bool(comp_doc["compliance_ok"]),
                "queue_file": self.redacted_path,
                "enqueued": True,
                "deduplicated": False,
                "duplicate_state": "none",
                "evicted_oldest": evicted_oldest,
                "event_id": event_id,
                "payload": clean_payload,
                "pending_count": pending_count,
                "delivered_ledger_count": delivered_count,
                "quarantined_count": quarantined_count,
                "dropped_overflow_count": dropped_count,
                "network_calls": 0,
                "compliance": dict(comp_doc),
            }
        )

    def flush(
        self,
        gateway: str = DEFAULT_GATEWAY,
        *,
        enforce_personal_pc: bool = True,
        env: Mapping[str, str] | None = None,
        cwd: Path | str | None = None,
        http_fn: HttpFn = default_http_json,
        respect_backoff: bool = False,
        now_epoch: float | None = None,
        lang: str = "en",
    ) -> dict[str, Any]:
        """Idempotently flush queued CES feedback items with `Idempotency-Key: <event_id>`."""
        lang_code = normalize_lang(lang)
        comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if enforce_personal_pc and not comp_doc["compliance_ok"]:
            return _compliance_blocked_report(
                "flush_ces_queue",
                gateway,
                comp_doc,
                queue_file=self._path,
                home=self._home,
                lang_code=lang_code,
            )

        clean_gw = validate_gateway_url(gateway)
        current_epoch = round(float(now_epoch if now_epoch is not None else time.time()), 3)
        post_url = f"{clean_gw}/v1/hivemind/onboarding/feedback"

        delivered_now = 0
        retained_retry = 0
        quarantined_now = 0
        deferred_backoff = 0
        network_calls = 0
        stale_404_seen = False
        outcomes: list[dict[str, Any]] = []

        with self._locked_spool(create_parent=False, lock_if_exists=True):
            if not self._path.is_file():
                return _attach_receipt(
                    {
                        "schema": SCHEMA,
                        "object": RECEIPT_OBJECT,
                        "campaign_id": CAMPAIGN_ID,
                        "mode": "flush_ces_queue",
                        "lang": lang_code,
                        "gateway": clean_gw,
                        "compliance_ok": bool(comp_doc["compliance_ok"]),
                        "queue_file": self.redacted_path,
                        "flush_ok": True,
                        "flushed_count": 0,
                        "delivered_count": 0,
                        "retained_retry_count": 0,
                        "quarantined_count": 0,
                        "deferred_backoff_count": 0,
                        "pending_count": 0,
                        "delivered_ledger_count": 0,
                        "total_quarantined_count": 0,
                        "stale_cloud_run_detected": False,
                        "network_calls": 0,
                        "zero_inference_verified": True,
                        "outcomes": [],
                        "diagnostics": [],
                        "compliance": dict(comp_doc),
                    }
                )

            state, mutated = self._load_state_unlocked()
            quarantined_now += len(state["quarantined"]) if mutated else 0
            delivered_set = set(state["delivered_event_ids"])
            next_pending: list[dict[str, Any]] = []

            for entry in state["pending"]:
                eid = str(entry["event_id"])
                if eid in delivered_set:
                    mutated = True
                    continue

                next_due = float(entry.get("next_attempt_epoch", 0.0))
                if respect_backoff and int(entry.get("retry_count", 0)) > 0 and current_epoch < next_due:
                    next_pending.append(entry)
                    deferred_backoff += 1
                    outcomes.append(
                        {
                            "event_id": eid,
                            "disposition": "deferred_backoff",
                            "http_status": entry.get("last_http_status"),
                            "retry_count": int(entry.get("retry_count", 0)),
                            "backoff_seconds": float(entry.get("backoff_seconds", 0.0)),
                        }
                    )
                    continue

                code, _ = _invoke_http_callable(
                    http_fn,
                    "POST",
                    post_url,
                    payload=dict(entry["payload"]),
                    headers={"Idempotency-Key": eid},
                    allow_feedback_post=True,
                )
                network_calls += 1
                disposition, reason = _classify_flush_http_status(code)
                mutated = True

                if disposition == "delivered":
                    state["delivered_event_ids"].append(eid)
                    delivered_set.add(eid)
                    delivered_now += 1
                    outcomes.append(
                        {
                            "event_id": eid,
                            "disposition": "delivered",
                            "http_status": code,
                            "retry_count": int(entry.get("retry_count", 0)),
                        }
                    )
                elif disposition == "retryable":
                    if code == 404:
                        stale_404_seen = True
                    new_rc = min(int(entry.get("retry_count", 0)) + 1, MAX_RETRY_COUNT)
                    backoff_s = compute_deterministic_backoff_s(
                        new_rc,
                        base_backoff_s=self._base_backoff_s,
                        max_backoff_s=self._max_backoff_s,
                    )
                    updated_entry = {
                        **entry,
                        "retry_count": new_rc,
                        "backoff_seconds": backoff_s,
                        "next_attempt_epoch": round(current_epoch + backoff_s, 3),
                        "last_http_status": code,
                        "last_reason": reason,
                    }
                    next_pending.append(updated_entry)
                    retained_retry += 1
                    outcomes.append(
                        {
                            "event_id": eid,
                            "disposition": "retained_for_retry",
                            "reason": reason,
                            "http_status": code,
                            "retry_count": new_rc,
                            "backoff_seconds": backoff_s,
                        }
                    )
                else:
                    state["quarantined"].append(
                        {
                            "event_id": eid,
                            "reason": reason,
                            "http_status": code,
                        }
                    )
                    state["quarantined"] = state["quarantined"][-self._max_entries :]
                    quarantined_now += 1
                    outcomes.append(
                        {
                            "event_id": eid,
                            "disposition": "quarantined",
                            "reason": reason,
                            "http_status": code,
                        }
                    )

            state["pending"] = next_pending
            state["delivered_event_ids"] = state["delivered_event_ids"][-self._max_delivered_ids :]
            if mutated:
                self._atomic_save_unlocked(state)

            pending_after = len(state["pending"])
            delivered_total = len(state["delivered_event_ids"])
            quarantined_total = len(state["quarantined"])

        is_fr = lang_code == "fr"
        diagnostics: list[str] = []
        if stale_404_seen:
            diagnostics.append(
                "HTTP 404 détecté sur POST /v1/hivemind/onboarding/feedback : révision Cloud Run en retard, "
                "retours CES conservés en sécurité dans la file locale."
                if is_fr
                else "HTTP 404 on POST /v1/hivemind/onboarding/feedback: Cloud Run lags behind `main`; "
                "CES items safely retained in local retry spool."
            )
        elif retained_retry > 0:
            diagnostics.append(
                f"{retained_retry} retour(s) CES conservé(s) avec backoff exponentiel déterministe."
                if is_fr
                else f"{retained_retry} CES item(s) retained with deterministic exponential backoff."
            )
        if quarantined_now > 0:
            diagnostics.append(
                f"{quarantined_now} entrée(s) permanente(s) 4xx mise(s) en quarantaine."
                if is_fr
                else f"{quarantined_now} permanent 4xx item(s) moved to quarantine."
            )

        flush_ok = retained_retry == 0 and quarantined_now == 0
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "flush_ces_queue",
                "lang": lang_code,
                "gateway": clean_gw,
                "compliance_ok": bool(comp_doc["compliance_ok"]),
                "queue_file": self.redacted_path,
                "flush_ok": flush_ok,
                "flushed_count": delivered_now,
                "delivered_count": delivered_now,
                "retained_retry_count": retained_retry,
                "quarantined_count": quarantined_now,
                "deferred_backoff_count": deferred_backoff,
                "pending_count": pending_after,
                "delivered_ledger_count": delivered_total,
                "total_quarantined_count": quarantined_total,
                "stale_cloud_run_detected": stale_404_seen,
                "network_calls": network_calls,
                "zero_inference_verified": True,
                "outcomes": outcomes,
                "diagnostics": diagnostics,
                "compliance": dict(comp_doc),
            }
        )

    def purge(self, *, lang: str = "en") -> dict[str, Any]:
        """Remove local CES retry spool and lock/temp artifacts (always permitted, 0 network calls)."""
        lang_code = normalize_lang(lang)
        is_fr = lang_code == "fr"
        removed = False
        with self._rlock:
            parent = self._path.parent
            if self._path.exists():
                self._path.unlink(missing_ok=True)
                removed = True
            if parent.is_dir():
                for extra in parent.glob(f".{self._path.name}.*"):
                    extra.unlink(missing_ok=True)
        summary = (
            f"File locale CES purgée (`{self.redacted_path}`, 0 appel réseau)."
            if is_fr
            else f"Local CES retry queue purged (`{self.redacted_path}`, 0 network calls)."
        )
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "purge_queue",
                "lang": lang_code,
                "queue_file": self.redacted_path,
                "purged": removed,
                "already_absent": not removed,
                "pending_count": 0,
                "network_calls": 0,
                "summary": summary,
                "compliance": {
                    "personal_pc_only": True,
                    "amadeus_corporate_prohibition": True,
                    "financial_promise": FINANCIAL_PROMISE,
                    "monetary_cost": FINANCIAL_PROMISE,
                    "license": "AGPL-3.0",
                    "notice_en": COMPLIANCE_NOTICE_EN,
                    "notice_fr": COMPLIANCE_NOTICE_FR,
                    "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
                },
            }
        )


def enqueue_ces_feedback(
    *,
    queue_file: Path | str | None = None,
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    platform: str | None = None,
    note: str = "",
    idempotency_key: str | None = None,
    max_entries: int = DEFAULT_MAX_QUEUE_ENTRIES,
    home: Path | None = None,
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    now_epoch: float | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Convenience wrapper to enqueue a zero-PII CES feedback event into `CesFeedbackRetryQueue`."""
    queue = CesFeedbackRetryQueue(queue_file, max_entries=max_entries, home=home)
    return queue.enqueue(
        profile=profile,
        ease_rating=ease_rating,
        zero_game_lag=zero_game_lag,
        platform=platform,
        note=note,
        idempotency_key=idempotency_key,
        enforce_personal_pc=enforce_personal_pc,
        env=env,
        cwd=cwd,
        now_epoch=now_epoch,
        lang=lang,
    )


def flush_ces_feedback_queue(
    gateway: str = DEFAULT_GATEWAY,
    *,
    queue_file: Path | str | None = None,
    max_entries: int = DEFAULT_MAX_QUEUE_ENTRIES,
    home: Path | None = None,
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    respect_backoff: bool = False,
    now_epoch: float | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Convenience wrapper to flush queued CES feedback items to `POST /v1/hivemind/onboarding/feedback`."""
    queue = CesFeedbackRetryQueue(queue_file, max_entries=max_entries, home=home)
    return queue.flush(
        gateway,
        enforce_personal_pc=enforce_personal_pc,
        env=env,
        cwd=cwd,
        http_fn=http_fn,
        respect_backoff=respect_backoff,
        now_epoch=now_epoch,
        lang=lang,
    )


def purge_ces_feedback_queue(
    *,
    queue_file: Path | str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Convenience wrapper to purge the local CES feedback retry spool (always permitted locally)."""
    queue = CesFeedbackRetryQueue(queue_file, home=home)
    return queue.purge(lang=lang)


def cutover_and_flush_ces_queue(
    gateway: str = DEFAULT_GATEWAY,
    *,
    queue_file: Path | str | None = None,
    expected_sha: str | None = None,
    previous_revision: str | None = None,
    canary_percent: int = DEFAULT_CANARY_PERCENT,
    candidate_tag: str = DEFAULT_CANDIDATE_TAG,
    deploy_script_path: Path | str | None = None,
    max_entries: int = DEFAULT_MAX_QUEUE_ENTRIES,
    home: Path | None = None,
    enforce_personal_pc: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    now_epoch: float | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify Cloud Run canary cutover first; promote & flush CES queue iff canary passes, else hold & trigger rollback."""
    lang_code = normalize_lang(lang)
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        return _compliance_blocked_report(
            "cutover_and_flush",
            gateway,
            comp_doc,
            queue_file=queue_file,
            home=home,
            lang_code=lang_code,
        )

    canary_rep = verify_canary_cutover(
        gateway,
        expected_sha=expected_sha,
        previous_revision=previous_revision,
        canary_percent=canary_percent,
        candidate_tag=candidate_tag,
        deploy_script_path=deploy_script_path,
        enforce_personal_pc=enforce_personal_pc,
        env=env,
        cwd=cwd,
        http_fn=http_fn,
        lang=lang_code,
    )
    queue = CesFeedbackRetryQueue(queue_file, max_entries=max_entries, home=home)
    if not canary_rep["promote_to_100_pct"]:
        snap = queue.snapshot(lang=lang_code)
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "cutover_and_flush",
                "lang": lang_code,
                "gateway": canary_rep["gateway"],
                "queue_file": queue.redacted_path,
                "compliance_ok": True,
                "canary_ok": False,
                "promote_to_100_pct": False,
                "rollback_triggered": True,
                "flush_executed": False,
                "flush_ok": False,
                "flushed_count": 0,
                "pending_count": snap["pending_count"],
                "rollback_command": canary_rep["rollback_command"],
                "canary_report": canary_rep,
                "flush_report": None,
                "diagnostics": list(canary_rep["diagnostics"]),
                "compliance": dict(comp_doc),
            }
        )

    flush_rep = queue.flush(
        gateway,
        enforce_personal_pc=False,
        env=env,
        cwd=cwd,
        http_fn=http_fn,
        respect_backoff=False,
        now_epoch=now_epoch,
        lang=lang_code,
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "cutover_and_flush",
            "lang": lang_code,
            "gateway": canary_rep["gateway"],
            "queue_file": queue.redacted_path,
            "compliance_ok": True,
            "canary_ok": True,
            "promote_to_100_pct": True,
            "rollback_triggered": False,
            "flush_executed": True,
            "flush_ok": bool(flush_rep["flush_ok"]),
            "flushed_count": int(flush_rep["flushed_count"]),
            "pending_count": int(flush_rep["pending_count"]),
            "promote_100_pct_command": canary_rep["promote_100_pct_command"],
            "rollback_command": canary_rep["rollback_command"],
            "canary_report": canary_rep,
            "flush_report": flush_rep,
            "diagnostics": list(canary_rep["diagnostics"]) + list(flush_rep["diagnostics"]),
            "compliance": dict(comp_doc),
        }
    )


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a sanitized bilingual FR/EN Markdown summary with SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    mode = str(report.get("mode", "verify_canary"))
    header = (
        "# Vérificateur Canari Cloud Run & File CES Idempotente HiveMind"
        if is_fr
        else "# HiveMind Cloud Run Canary Cutover Verifier & Idempotent CES Queue"
    )
    lines: list[str] = [
        header,
        "",
        f"> **Compliance / Conformité**: {COMPLIANCE_NOTICE_BILINGUAL}",
        f"> **Points**: {SYMBOLIC_POINTS_NOTICE_BILINGUAL}",
        "",
        f"- **Mode**: `{mode}`",
    ]
    if "gateway" in report:
        lines.append(f"- **Gateway**: `{report.get('gateway')}`")
    if "queue_file" in report:
        lines.append(f"- **Queue File**: `{report.get('queue_file')}`")
    if "compliance_ok" in report:
        lines.append(f"- **Compliance OK**: `{report.get('compliance_ok')}`")
    if "promote_to_100_pct" in report:
        lines.append(
            f"- **Promote to 100%**: `{report.get('promote_to_100_pct')}` "
            f"(`rollback_triggered={report.get('rollback_triggered')}`)"
        )
    if "flushed_count" in report:
        lines.append(
            f"- **CES Flushed**: `{report.get('flushed_count')}` "
            f"(`pending={report.get('pending_count')}`)"
        )
    if "enqueued" in report and mode == "enqueue_ces":
        lines.append(
            f"- **Enqueued**: `{report.get('enqueued')}` "
            f"(`deduplicated={report.get('deduplicated')}`, `event_id={report.get('event_id')}`)"
        )
    if report.get("rollback_triggered") and report.get("rollback_command"):
        lines.extend(
            [
                "",
                "## Rollback Command (Dry-Run)" if not is_fr else "## Commande de Rollback (Dry-Run)",
                "",
                "```bash",
                str(report["rollback_command"]),
                "```",
            ]
        )
    elif mode == "canary_traffic_plan":
        lines.extend(
            [
                "",
                "## Canary Traffic Plan (Dry-Run)" if not is_fr else "## Plan de Trafic Canari (Dry-Run)",
                "",
                "```bash",
                str(report.get("canary_traffic_split_command", "")),
                str(report.get("promote_100_pct_command", "")),
                str(report.get("rollback_command", "")),
                "```",
            ]
        )
    diags = report.get("diagnostics")
    if isinstance(diags, Sequence) and diags:
        lines.extend(["", "## Diagnostics", ""])
        for item in diags:
            lines.append(f"- {sanitize_text_for_display(str(item))}")
    lines.extend(["", f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`"])
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_canary_cutover_and_ces_queue.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind Cloud Run canary cutover verifier and atomic, idempotent CES feedback retry queue "
            "(100% personal home PCs only, 0.00 EUR / 0.00 USD, AGPL-3.0)."
        )
    )
    parser.add_argument("--gateway", default=default_gw, help="HiveMind Cloud Run or loopback gateway URL.")
    parser.add_argument("--verify-canary", action="store_true", help="Verify FinOps & zero-inference canary routes.")
    parser.add_argument("--print-canary-plan", action="store_true", help="Print dry-run Cloud Run canary traffic plan.")
    parser.add_argument("--canary-percent", type=int, default=DEFAULT_CANARY_PERCENT, help="Canary traffic percentage (1..99).")
    parser.add_argument("--candidate-tag", default=DEFAULT_CANDIDATE_TAG, help="Cloud Run candidate revision tag.")
    parser.add_argument("--previous-revision", default=None, help="Previous known-good Cloud Run K_REVISION.")
    parser.add_argument("--expected-sha", default=None, help="Expected live HIVEMIND_BUILD_SHA.")
    parser.add_argument("--deploy-script", default="", help="Optional path to deploy/gcp_openrouter/deploy.sh.")
    parser.add_argument("--queue-file", default="", help="Local CES spool file (default ~/.hivemind/ces_feedback_queue.json).")
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_QUEUE_ENTRIES, help="Max pending queue size.")
    parser.add_argument("--enqueue-ces", action="store_true", help="Enqueue a zero-PII CES feedback item locally.")
    parser.add_argument("--flush-ces-queue", action="store_true", help="Flush pending CES items idempotently.")
    parser.add_argument("--cutover-and-flush", action="store_true", help="Verify canary cutover then flush CES queue.")
    parser.add_argument("--purge-queue", action="store_true", help="Delete local CES queue spool (always allowed).")
    parser.add_argument("--respect-backoff", action="store_true", help="Defer items whose backoff window has not elapsed.")
    parser.add_argument("--profile", choices=("dev", "gamer"), default="dev", help="Onboarding cohort profile.")
    parser.add_argument("--ease-rating", type=int, default=5, help="Ease rating 1..5.")
    parser.add_argument("--had-game-lag", action="store_true", help="Report game lag (sets zero_game_lag=False).")
    parser.add_argument("--platform", default=None, help="Optional ASCII platform identifier.")
    parser.add_argument("--idempotency-key", default=None, help="Optional deterministic idempotency key.")
    parser.add_argument("--lang", default="en", help="Output language ('fr' or 'en').")
    parser.add_argument("--json", action="store_true", help="Emit JSON output with SHA-256 receipt.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output (default).")
    return parser


def _emit(
    report: Mapping[str, Any],
    *,
    as_json: bool,
    ok: bool,
    out_stream: Any,
) -> int:
    rendered = (
        _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False))
        if as_json
        else render_markdown_report(report)
    )
    print(rendered, file=out_stream)
    return 0 if ok else 2


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | None = None,
    now_epoch: float | None = None,
    out_stream: Any = None,
    err_stream: Any = None,
) -> int:
    """CLI entrypoint returning `0` on verified promotion/queue operation or `2` (fail-closed)."""
    out = out_stream if out_stream is not None else sys.stdout
    err = err_stream if err_stream is not None else sys.stderr
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) and exc.code != 0 else (0 if exc.code == 0 else 2)

    queue_arg: Path | str | None = args.queue_file if args.queue_file else None
    deploy_arg: Path | str | None = args.deploy_script if args.deploy_script else None

    try:
        lang_code = normalize_lang(args.lang)

        if args.purge_queue:
            purge_rep = purge_ces_feedback_queue(queue_file=queue_arg, home=home, lang=lang_code)
            return _emit(purge_rep, as_json=bool(args.json), ok=True, out_stream=out)

        comp = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if not comp["compliance_ok"]:
            print(comp["summary"], file=err)
            mode_label = (
                "cutover_and_flush"
                if args.cutover_and_flush
                else (
                    "enqueue_ces"
                    if args.enqueue_ces
                    else (
                        "flush_ces_queue"
                        if args.flush_ces_queue
                        else ("canary_traffic_plan" if args.print_canary_plan else "verify_canary")
                    )
                )
            )
            blocked = _compliance_blocked_report(
                mode_label,
                args.gateway,
                comp,
                queue_file=queue_arg,
                home=home,
                lang_code=lang_code,
            )
            return _emit(blocked, as_json=bool(args.json), ok=False, out_stream=out)

        if args.print_canary_plan and not (
            args.verify_canary or args.cutover_and_flush or args.enqueue_ces or args.flush_ces_queue
        ):
            plan = build_canary_traffic_plan(
                gateway=args.gateway,
                candidate_tag=args.candidate_tag,
                canary_percent=args.canary_percent,
                previous_revision=args.previous_revision,
                target_sha=args.expected_sha,
                deploy_script_path=deploy_arg,
                lang=lang_code,
            )
            return _emit(plan, as_json=bool(args.json), ok=bool(plan["plan_ready"]), out_stream=out)

        enq_rep: dict[str, Any] | None = None
        if args.enqueue_ces:
            enq_rep = enqueue_ces_feedback(
                queue_file=queue_arg,
                profile=args.profile,
                ease_rating=args.ease_rating,
                zero_game_lag=not bool(args.had_game_lag),
                platform=args.platform,
                idempotency_key=args.idempotency_key,
                max_entries=args.max_entries,
                home=home,
                enforce_personal_pc=False,
                env=env,
                cwd=cwd,
                now_epoch=now_epoch,
                lang=lang_code,
            )
            if not (args.flush_ces_queue or args.cutover_and_flush or args.verify_canary):
                return _emit(enq_rep, as_json=bool(args.json), ok=True, out_stream=out)

        if args.cutover_and_flush:
            cf_rep = cutover_and_flush_ces_queue(
                args.gateway,
                queue_file=queue_arg,
                expected_sha=args.expected_sha,
                previous_revision=args.previous_revision,
                canary_percent=args.canary_percent,
                candidate_tag=args.candidate_tag,
                deploy_script_path=deploy_arg,
                max_entries=args.max_entries,
                home=home,
                enforce_personal_pc=False,
                env=env,
                cwd=cwd,
                http_fn=http_fn,
                now_epoch=now_epoch,
                lang=lang_code,
            )
            if enq_rep is not None:
                cf_rep = _attach_receipt({**cf_rep, "enqueue_report": enq_rep})
            ok = bool(cf_rep["promote_to_100_pct"] and cf_rep["flush_ok"])
            return _emit(cf_rep, as_json=bool(args.json), ok=ok, out_stream=out)

        if args.flush_ces_queue and not args.verify_canary:
            fl_rep = flush_ces_feedback_queue(
                args.gateway,
                queue_file=queue_arg,
                max_entries=args.max_entries,
                home=home,
                enforce_personal_pc=False,
                env=env,
                cwd=cwd,
                http_fn=http_fn,
                respect_backoff=bool(args.respect_backoff),
                now_epoch=now_epoch,
                lang=lang_code,
            )
            if enq_rep is not None:
                fl_rep = _attach_receipt({**fl_rep, "enqueue_report": enq_rep})
            return _emit(fl_rep, as_json=bool(args.json), ok=bool(fl_rep["flush_ok"]), out_stream=out)

        ver_rep = verify_canary_cutover(
            args.gateway,
            expected_sha=args.expected_sha,
            previous_revision=args.previous_revision,
            canary_percent=args.canary_percent,
            candidate_tag=args.candidate_tag,
            deploy_script_path=deploy_arg,
            enforce_personal_pc=False,
            env=env,
            cwd=cwd,
            http_fn=http_fn,
            lang=lang_code,
        )
        return _emit(ver_rep, as_json=bool(args.json), ok=bool(ver_rep["promote_to_100_pct"]), out_stream=out)
    except CanaryQueueError as exc:
        print(f"Error: {sanitize_text_for_display(str(exc), home=home)}", file=err)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
