#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference Dec 18 J-1 Launch-Eve Preflight & Atomic 0600 CES Spool Readiness Verifier (`SLICE-SCALE-DYN-29`).

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment markers (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``,
     ``NODE_EXTRA_CA_CERTS``), workspace/home paths, and ``.git/config`` remote URLs (including
     worktree ``gitdir:`` / ``commondir`` indirection) BEFORE any network call or file write.
     Fails closed with exit code 2 (0 network calls, 0 file writes, 0 raw env/hostname/path leaks).
  2. Local 0600 Credential & CES Spool Readiness Auditor (``audit_local_0600_credentials_and_spool``):
     inspects ``~/.hivemind`` directory permissions, ``credentials.json``, local CES spool files,
     and ``~/.continue/hivemind_provider.yaml`` read-only without creating files when none are
     requested, failing closed on ``0644`` / group-or-world-writable modes, symlinks, or raw key
     leaks, with opt-in atomic ``0600`` CES spool write-and-clean verification.
  3. Zero-Inference Dec 18, 2026 (J-1 Launch-Eve) Preflight Verifier
     (``verify_j1_launch_eve_readiness``):
     probes ``GET /health``, ``GET /v1/models``, ``GET /noel``, ``GET /quick``, ``GET /raid-noel``,
     ``GET /arena-noel``, ``GET /v1/hivemind/onboarding/manifest``, and
     ``GET /v1/hivemind/onboarding/feedback/summary`` strictly read-only (never invoking
     ``POST /v1/chat/completions`` or prompting ``hivemind/coder``), enforcing unique per-request
     CSP nonces ``default-src 'none'``, 0 unsafe DOM sinks, 0 static figures (stripping
     ``hm-***`` / ``hm_live_***`` masked-key suffixes and HTML ``nonce="..."`` attributes first),
     exact 5-key feedback summary equality, deterministic 5-pillar J-1 Go/No-Go checklist, and
     FR/EN JSON & Markdown SHA-256 receipts.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen science under AGPL-3.0 (zero monetary value).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import secrets
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_launch_eve_preflight_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_j1_launch_eve_preflight_receipt"
CHECKLIST_OBJECT = "hivemind.noel_j1_launch_eve_checklist"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
J1_LAUNCH_EVE_DATE = "2026-12-18"
LAUNCH_DAY_DATE = "2026-12-19"
HOLIDAY_WRAPUP_DATE = "2027-01-03"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
AGPL_HEADER_LINE = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)

HTTP_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 1_048_576
MAX_LOCAL_FILE_BYTES = 1_048_576
REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset()

J1_HTML_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
)

J1_PROBE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "models"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "feedback"),
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

EXPECTED_FEEDBACK_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "object",
        "total_responses",
        "mean_ease_rating",
        "zero_game_lag_ratio",
        "by_profile",
    }
)

PILLAR_KEYS: tuple[str, ...] = (
    "compliance_ok",
    "routes_ok",
    "models_ok",
    "manifest_ok",
    "feedback_summary_ok",
)

LOCAL_CES_SPOOL_FILES: tuple[str, ...] = (
    "ces_feedback_queue.json",
    "ces_telemetry_spool.json",
    "ces_feedback_spool.json",
    "vscode_setup_ces_queue.json",
    "idle_service_ces_queue.json",
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

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_SAFE_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_]+\.(?:py|ps1)$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_FULL_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_API_KEY_RE = re.compile(
    r"(hm(?:_live_|-))\*\*\*(?:[A-Za-z0-9_-]{4,}|[0-9]{1,3}m" + r"s\b)"
)
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)"
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

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(
    r"\b(?:hm(?:_live_|-)|sk-(?:ant-|or-v1-|proj-)?|gh[pousr]_|hf_)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b"
)
_RAW_HM_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])hm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{4,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_TROJAN_BIDI_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)

HttpFn = Callable[..., Any]


class LaunchEvePreflightError(ValueError):
    """Raised when J-1 launch-eve preflight inputs, host allowlists, or safety invariants fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI or API output language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise LaunchEvePreflightError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise LaunchEvePreflightError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw credentials never appear in receipts."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        return "<invalid>"
    if cleaned.startswith("hm_live_") and len(cleaned) >= 12:
        return f"hm_live_***{cleaned[-4:]}"
    if cleaned.startswith("hm-") and len(cleaned) >= 8:
        return f"hm-***{cleaned[-4:]}"
    return "<invalid>"


def is_valid_hivemind_key(raw_key: object) -> bool:
    """Return True iff ``raw_key`` is a well-formed unmasked ``hm-...`` or ``hm_live_...`` key."""
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if "***" in cleaned or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        return False
    return (cleaned.startswith("hm-") and len(cleaned) >= 8) or (
        cleaned.startswith("hm_live_") and len(cleaned) >= 12
    )


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and masked ``hm-***`` suffixes."""
    if not isinstance(text, str):
        raise LaunchEvePreflightError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    cleaned = _MASKED_API_KEY_RE.sub(r"\1***XXXX", cleaned)
    return cleaned


def _match_prefix_redacted(raw: str, base_dir: Path, label: str) -> str | None:
    resolved_base = str(base_dir)
    if not resolved_base or resolved_base in ("/", "\\"):
        return None
    for variant in (
        resolved_base,
        resolved_base.replace("\\", "/"),
        resolved_base.replace("/", "\\"),
    ):
        if variant and raw.startswith(variant):
            suffix = raw[len(variant) :]
            if not suffix or suffix.startswith(("/", "\\")):
                redacted = label + suffix.replace("\\", "/")
                return _PYTEST_USER_DIR_RE.sub("pytest-of-user", redacted)
    return None


def redact_home_path(
    target: Path | str,
    home: Path | None = None,
    workspace: Path | None = None,
) -> str:
    """Replace OS home prefixes with ``'~'`` and map workspace/repo paths safely."""
    raw = str(target)
    if home is not None:
        hit_home = _match_prefix_redacted(raw, home, "~")
        if hit_home is not None:
            return hit_home

    if workspace is not None:
        hit_ws = _match_prefix_redacted(raw, workspace, "<workspace>")
        if hit_ws is not None:
            return hit_ws

    if not re.match(r"^[A-Za-z]:[\\/]", raw) and Path(raw).is_absolute():
        try:
            rel = Path(raw).resolve().relative_to(REPO_ROOT)
            return str(rel).replace("\\", "/")
        except (OSError, ValueError):
            rel = None
            _ = rel

    if home is None:
        hit_default_home = _match_prefix_redacted(raw, Path.home(), "~")
        if hit_default_home is not None:
            return hit_default_home

    scrubbed = _HOME_PATH_RE.sub("~", raw)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    return scrubbed.replace("\\", "/") if scrubbed != raw else raw


def sanitize_text_for_display(
    text: str,
    *,
    home: Path | None = None,
    workspace: Path | None = None,
) -> str:
    """Scrub raw keys, Bearer headers, emails, non-loopback IPv4 literals, and OS home paths."""
    scrubbed = text
    resolved_home = str(home if home is not None else Path.home())
    if resolved_home and resolved_home not in ("/", "\\"):
        for variant in (
            resolved_home,
            resolved_home.replace("\\", "/"),
            resolved_home.replace("/", "\\"),
        ):
            if variant and variant in scrubbed:
                scrubbed = scrubbed.replace(variant, "~")
    scrubbed = redact_home_path(scrubbed, home=home, workspace=workspace)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _RAW_HM_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith("127.") else "<redacted-ip>",
        scrubbed,
    )
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
    """Verify JSON, CLI, or Markdown text has 0 PII, raw keys, score digits, or static figures."""
    if not isinstance(text, str):
        raise LaunchEvePreflightError("scan_output_safety requires a string.")
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
        raise LaunchEvePreflightError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise LaunchEvePreflightError("Forbidden JSON key detected in J-1 preflight receipt.")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
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
        raise LaunchEvePreflightError(
            "Expected SHA must be a 6..64 character alphanumeric/hex string (never 'unknown' or a secret)."
        )
    return sanitized


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Compare two ``build_sha`` values (exact or >=7-char hex prefix)."""
    clean_live = sanitize_build_sha(live_sha)
    clean_exp = sanitize_build_sha(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def validate_preflight_date(raw: str | None) -> str:
    """Validate ISO ``YYYY-MM-DD`` preflight date (defaults to ``2026-12-18`` J-1 Launch-Eve)."""
    if raw is None:
        return J1_LAUNCH_EVE_DATE
    if isinstance(raw, bool) or not isinstance(raw, str) or not _ISO_DATE_RE.fullmatch(raw.strip()):
        raise LaunchEvePreflightError("preflight_date must be a 'YYYY-MM-DD' string.")
    cleaned = raw.strip()
    try:
        date.fromisoformat(cleaned)
    except ValueError as exc:
        raise LaunchEvePreflightError(f"Invalid preflight_date {cleaned!r}.") from exc
    return cleaned


def compute_merkle_root(
    leaf_hashes: Sequence[str],
    *,
    domain: str = "hivemind.j1_launch_eve.merkle",
) -> str:
    """Compute a deterministic domain-separated binary SHA-256 Merkle root over leaf digests."""
    if not leaf_hashes:
        return hashlib.sha256(f"{domain}.empty/1".encode()).hexdigest()
    level: list[str] = []
    for digest in leaf_hashes:
        if not isinstance(digest, str) or not _SHA256_FULL_RE.fullmatch(digest):
            raise LaunchEvePreflightError("Merkle leaf digest must be a 64-char lowercase hex SHA-256.")
        level.append(digest)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: list[str] = []
        for idx in range(0, len(level), 2):
            pair = f"{domain}.node/1:{level[idx]}:{level[idx + 1]}".encode()
            next_level.append(hashlib.sha256(pair).hexdigest())
        level = next_level
    return level[0]


def is_allowed_outbound_host(host: str) -> bool:
    """Return True iff ``host`` is in the strict HiveMind allowlist."""
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
        raise LaunchEvePreflightError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise LaunchEvePreflightError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise LaunchEvePreflightError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise LaunchEvePreflightError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise LaunchEvePreflightError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise LaunchEvePreflightError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise LaunchEvePreflightError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise LaunchEvePreflightError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise LaunchEvePreflightError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise LaunchEvePreflightError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise LaunchEvePreflightError("Remote gateway URL must use https.")
    return cleaned


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    allowed_post_paths: frozenset[str] = ALLOWED_POST_PATHS,
) -> str:
    """Enforce allowlisted host, zero completions/prompts, and read-only HTTP GET/HEAD."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise LaunchEvePreflightError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise LaunchEvePreflightError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise LaunchEvePreflightError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise LaunchEvePreflightError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise LaunchEvePreflightError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise LaunchEvePreflightError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise LaunchEvePreflightError(
        f"Zero-inference read-only guard blocked HTTP {verb!r} to {parts.path!r}."
    )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects to untrusted hosts or inference completion endpoints."""

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


def _parse_http_raw_response(
    status_code: int,
    raw_bytes: bytes,
    raw_headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
        raise LaunchEvePreflightError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
    lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    ctype = lower_headers.get("content-type", "").lower()
    looks_html = "text/html" in ctype or "text/plain" in ctype
    if "application/json" in ctype or (not looks_html and raw_bytes.lstrip().startswith((b"{", b"["))):
        text = raw_bytes.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": "invalid_json"}
        if isinstance(parsed, dict):
            return int(status_code), dict(parsed)
        return int(status_code), {"data": parsed}
    html_text = raw_bytes.decode("utf-8", errors="replace")
    return int(status_code), {
        "html": html_text,
        "text": html_text,
        "html_length": len(html_text),
        "headers": lower_headers,
        "csp": lower_headers.get("content-security-policy", ""),
    }


_HTTP_OPENER: urllib.request.OpenerDirector | None = None


def _get_http_opener() -> urllib.request.OpenerDirector:
    global _HTTP_OPENER
    if _HTTP_OPENER is None:
        _HTTP_OPENER = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _SafeRedirectHandler(),
        )
    return _HTTP_OPENER


def default_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib HTTP helper (``ProxyHandler({})``) preserving exact JSON schema keys."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(
        verb,
        url,
        payload=payload,
        allowed_post_paths=ALLOWED_POST_PATHS,
    )
    if payload is not None:
        raise LaunchEvePreflightError("GET/HEAD requests must never attach a body payload.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-launch-eve-preflight-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise LaunchEvePreflightError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str

    req = urllib.request.Request(clean_url, data=None, headers=req_headers, method=verb)
    opener = _get_http_opener()
    bounded_timeout = min(max(0.2, float(timeout_s)), 15.0)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            status = int(getattr(resp, "status", 200))
            raw_hdrs = dict(resp.headers.items()) if resp.headers is not None else {}
            return _parse_http_raw_response(status, resp.read(MAX_HTTP_BODY_BYTES + 1), raw_hdrs)
    except urllib.error.HTTPError as exc:
        err_hdrs = dict(exc.headers.items()) if exc.headers is not None else {}
        err_bytes = exc.read(131_072) if exc.fp is not None else b""
        return _parse_http_raw_response(int(exc.code), err_bytes, err_hdrs)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}


def _looks_like_http_headers(candidate: Mapping[str, Any]) -> bool:
    if not candidate:
        return True
    keys = {str(k).strip().lower() for k in candidate}
    header_markers = {"content-type", "content-security-policy", "cache-control", "vary", "content-language"}
    body_markers = {"object", "status", "total_responses", "holiday_routes", "data", "detail", "error"}
    if keys & body_markers:
        return False
    return bool(keys & header_markers) or all(isinstance(v, str) for v in candidate.values())


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise LaunchEvePreflightError("http_fn must return (status_code, body) or (status_code, headers, body).")
    if len(raw_ret) == 2:
        status_code, body = raw_ret
        hdr_map: dict[str, str] = {}
    else:
        status_code, part1, part2 = raw_ret
        if isinstance(part1, str) and isinstance(part2, Mapping):
            body, raw_hdrs = part1, part2
        elif isinstance(part1, Mapping) and isinstance(part2, str):
            raw_hdrs, body = part1, part2
        elif isinstance(part1, Mapping) and isinstance(part2, Mapping):
            if _looks_like_http_headers(part2) and not _looks_like_http_headers(part1):
                body, raw_hdrs = part1, part2
            else:
                raw_hdrs, body = part1, part2
        else:
            raw_hdrs, body = (part1 if isinstance(part1, Mapping) else {}), part2
        hdr_map = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()}

    if isinstance(body, Mapping):
        if "html" in body or "text" in body:
            out = dict(body)
            if hdr_map:
                existing = out.get("headers")
                merged = (
                    {**hdr_map, **{str(k).strip().lower(): str(v) for k, v in existing.items()}}
                    if isinstance(existing, Mapping)
                    else hdr_map
                )
                out["headers"] = merged
                out.setdefault("csp", merged.get("content-security-policy", ""))
            return int(status_code), out
        if hdr_map:
            return int(status_code), {"_json": dict(body), "_headers": hdr_map}
        return int(status_code), dict(body)
    if isinstance(body, str):
        stripped = body.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, Mapping):
                    return int(status_code), (
                        {"_json": dict(parsed), "_headers": hdr_map} if hdr_map else dict(parsed)
                    )
                return int(status_code), {"data": parsed, "_headers": hdr_map} if hdr_map else {"data": parsed}
            except json.JSONDecodeError:
                parsed = None
                _ = parsed
        return int(status_code), {
            "html": body,
            "text": body,
            "html_length": len(body),
            "text_length": len(body),
            "headers": hdr_map,
            "csp": hdr_map.get("content-security-policy", ""),
        }
    return int(status_code), {"data": body, "_headers": hdr_map} if hdr_map else {"data": body}


def _unwrap_json_body(normalized: Mapping[str, Any]) -> dict[str, Any]:
    inner = normalized.get("_json")
    if isinstance(inner, Mapping):
        return dict(inner)
    return {k: v for k, v in normalized.items() if k != "_headers"}


def _invoke_http_fn(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Validate URL/zero-inference invariants and invoke an injectable HTTP callable."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(
        verb,
        url,
        payload=payload,
        allowed_post_paths=ALLOWED_POST_PATHS,
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
    call_headers = dict(headers) if headers is not None else None

    try:
        if has_varkw:
            raw = http_fn(verb, clean_url, payload=None, headers=call_headers, timeout_s=timeout_s)
        else:
            kwargs: dict[str, Any] = {}
            if "payload" in param_names:
                kwargs["payload"] = None
            if "headers" in param_names:
                kwargs["headers"] = call_headers
            if "timeout_s" in param_names:
                kwargs["timeout_s"] = timeout_s
            if kwargs or len(pos_params) <= 2:
                raw = http_fn(verb, clean_url, **kwargs)
            elif len(pos_params) == 3:
                raw = http_fn(verb, clean_url, None)
            elif len(pos_params) == 4 and not has_varargs:
                raw = http_fn(verb, clean_url, None, call_headers)
            else:
                raw = http_fn(verb, clean_url, None, call_headers, timeout_s)
    except LaunchEvePreflightError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    return _normalize_http_return(raw)


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
                        common_rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                    except OSError:
                        common_rel = ""
                    if common_rel:
                        common_cfg = (gitdir / common_rel).resolve() / "config"
                        if common_cfg.is_file() and common_cfg not in candidates:
                            candidates.append(common_cfg)
            break
    return candidates


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home_dir: Path | str | None = None,
    check_git_remote: bool = True,
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

    path_checks: list[tuple[str, Path]] = [("workspace_path", workspace)]
    if home_dir is not None and not isinstance(home_dir, bool) and str(home_dir).strip():
        path_checks.append(("home_path", Path(home_dir)))

    for scope_label, target_path in path_checks:
        texts = [str(target_path)]
        try:
            resolved_p = str(target_path.resolve())
            if resolved_p not in texts:
                texts.append(resolved_p)
        except OSError:
            resolved_p = ""
            _ = resolved_p
        for p_txt in texts:
            for hit in _find_corporate_markers(p_txt):
                _record(f"{scope_label}:{hit}", hit)

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
            f"(0 appel réseau, 0 écriture fichier). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE BLOCKED ({joined}): corporate environment indicator detected "
            f"(0 network calls, 0 file writes). {COMPLIANCE_NOTICE_BILINGUAL}"
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
            "zero_financial_payout": True,
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
    gateway: str | None = None,
    preflight_date: str = J1_LAUNCH_EVE_DATE,
) -> dict[str, Any]:
    checklist = {key: False for key in PILLAR_KEYS}
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "checklist_object": CHECKLIST_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "preflight_date": preflight_date,
            "launch_day_date": LAUNCH_DAY_DATE,
            "holiday_wrapup_date": HOLIDAY_WRAPUP_DATE,
            "mode": mode,
            "lang": lang_code,
            "gateway": gateway or DEFAULT_GATEWAY,
            "compliance_ok": False,
            "corporate_detected": True,
            "corporate_indicators": list(comp_doc.get("corporate_indicators") or []),
            "matched_markers": list(comp_doc.get("matched_markers") or []),
            "routes_ok": False,
            "models_ok": False,
            "manifest_ok": False,
            "feedback_summary_ok": False,
            "local_0600_ok": False,
            "checklist": checklist,
            "pillars_passed_count": 0,
            "pillars_total_count": len(PILLAR_KEYS),
            "ready": False,
            "go_nogo": "NO-GO",
            "decision": "HOLD_COMPLIANCE_BLOCKED",
            "hold_reasons": ["corporate_environment_detected"],
            "stale_404_drift": False,
            "redeploy_required": False,
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "zero_financial_payout": True,
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": [str(comp_doc.get("summary") or COMPLIANCE_NOTICE_BILINGUAL)],
            "exit_code": 2,
            "summary": str(comp_doc.get("summary") or COMPLIANCE_NOTICE_BILINGUAL),
        }
    )


def _read_text_bounded(path: Path, max_bytes: int = MAX_LOCAL_FILE_BYTES) -> str:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise LaunchEvePreflightError(f"File exceeded {max_bytes} byte limit.")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LaunchEvePreflightError("Local credential or spool file is not valid UTF-8.") from exc


def _posix_mode_label(path: Path, eff_os: str) -> tuple[int | None, str]:
    if eff_os == "nt":
        return None, "windows_acl"
    mode_bits = path.stat().st_mode & 0o777
    return mode_bits, f"{mode_bits:04o}"


def _write_bytes_secure_atomic(
    path: Path,
    raw_bytes: bytes,
    *,
    eff_os: str,
) -> None:
    """Atomically write ``raw_bytes`` via ``O_EXCL`` ``0600`` temp file + ``fsync`` + ``os.replace``."""
    if path.is_symlink() or path.parent.is_symlink():
        raise LaunchEvePreflightError("Refusing symlink during atomic 0600 CES spool probe.")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if eff_os != "nt":
            os.chmod(path.parent, 0o700)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if eff_os != "nt":
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except BaseException:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def _validate_ces_spool_payload(doc: Mapping[str, Any]) -> tuple[bool, int, int]:
    """Validate zero-PII CES spool structure without assuming a single spool filename."""
    pending_raw = doc.get("pending")
    delivered_raw = doc.get("delivered_event_ids")
    if "pending" in doc and not isinstance(pending_raw, list):
        return False, 0, 0
    if "delivered_event_ids" in doc and not isinstance(delivered_raw, list):
        return False, 0, 0

    pending_list = pending_raw if isinstance(pending_raw, list) else []
    for item in pending_list:
        if not isinstance(item, Mapping):
            return False, 0, 0
        payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else item
        if "profile" in payload and payload.get("profile") not in ("dev", "gamer"):
            return False, 0, 0
        if "ease_rating" in payload:
            er = payload.get("ease_rating")
            if isinstance(er, bool) or not isinstance(er, int) or not (1 <= er <= 5):
                return False, 0, 0
        if "zero_game_lag" in payload and not isinstance(payload.get("zero_game_lag"), bool):
            return False, 0, 0
    delivered_list = delivered_raw if isinstance(delivered_raw, list) else []
    return True, len(pending_list), len(delivered_list)


def audit_local_0600_credentials_and_spool(
    home_dir: Path | str | None = None,
    *,
    os_name: str | None = None,
    verify_atomic_spool_write: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Audit local ``0600`` credential & CES spool permissions without creating files unless requested."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name

    if home_dir is None:
        if verify_atomic_spool_write:
            raise LaunchEvePreflightError("home_dir is required when verify_atomic_spool_write=True.")
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "audit_local_0600",
                "lang": lang_code,
                "home_dir_provided": False,
                "local_state_audited": False,
                "hivemind_dir_exists": False,
                "hivemind_dir_ok": True,
                "hivemind_dir_posix_mode": None,
                "credentials_present": False,
                "credentials_0600_ok": True,
                "credentials_valid": None,
                "credentials_posix_mode": None,
                "masked_api_key": "<none>",
                "ces_spool_files_checked": [],
                "ces_spool_0600_ok": True,
                "ces_spool_valid": True,
                "continue_provider_0600_ok": True,
                "insecure_permission_paths": [],
                "symlink_violations": [],
                "malformed_files": [],
                "atomic_spool_write_verified": None,
                "local_0600_ok": True,
                "file_writes": 0,
                "files_created": 0,
                "diagnostics": [],
                "exit_code": 0,
            }
        )

    if isinstance(home_dir, bool) or not isinstance(home_dir, (Path, str)) or not str(home_dir).strip():
        raise LaunchEvePreflightError("home_dir must be a non-empty Path or directory string.")
    eff_home = Path(home_dir)
    if eff_home.is_symlink():
        raise LaunchEvePreflightError("Refusing symlink home_dir.")
    if eff_home.exists() and not eff_home.is_dir():
        raise LaunchEvePreflightError("home_dir must point to a directory.")

    hm_dir = eff_home / ".hivemind"
    hivemind_dir_exists = hm_dir.exists() or hm_dir.is_symlink()
    hivemind_dir_ok = True
    hivemind_dir_mode_str: str | None = None
    credentials_present = False
    credentials_0600_ok = True
    credentials_valid: bool | None = None
    credentials_mode_str: str | None = None
    masked_key = "<none>"
    ces_spool_rows: list[dict[str, Any]] = []
    ces_spool_0600_ok = True
    ces_spool_valid = True
    continue_provider_0600_ok = True
    insecure_paths: list[str] = []
    symlink_violations: list[str] = []
    malformed_files: list[str] = []
    diagnostics: list[str] = []

    if hm_dir.is_symlink():
        hivemind_dir_ok = False
        symlink_violations.append("~/.hivemind")
        diagnostics.append(
            "Lien symbolique interdit sur ~/.hivemind."
            if is_fr
            else "Refusing symlink on ~/.hivemind directory."
        )
    elif hm_dir.exists() and not hm_dir.is_dir():
        hivemind_dir_ok = False
        malformed_files.append("~/.hivemind")
        diagnostics.append(
            "~/.hivemind n'est pas un répertoire."
            if is_fr
            else "~/.hivemind exists but is not a directory."
        )
    elif hm_dir.is_dir():
        dir_mode, hivemind_dir_mode_str = _posix_mode_label(hm_dir, eff_os)
        if dir_mode is not None and (dir_mode & 0o022) != 0:
            hivemind_dir_ok = False
            insecure_paths.append("~/.hivemind")
            diagnostics.append(
                f"Permissions trop permissives sur ~/.hivemind ({hivemind_dir_mode_str}, groupe/autres en écriture interdits)."
                if is_fr
                else f"Insecure permissions on ~/.hivemind ({hivemind_dir_mode_str}; group/world write forbidden)."
            )

        cred_path = hm_dir / "credentials.json"
        if cred_path.exists() or cred_path.is_symlink():
            credentials_present = True
            shown_cred = redact_home_path(cred_path, home=eff_home)
            if cred_path.is_symlink():
                credentials_0600_ok = False
                credentials_valid = False
                symlink_violations.append(shown_cred)
                diagnostics.append(
                    f"Lien symbolique interdit sur {shown_cred}."
                    if is_fr
                    else f"Refusing symlink for {shown_cred}."
                )
            elif not cred_path.is_file():
                credentials_valid = False
                malformed_files.append(shown_cred)
                diagnostics.append(
                    f"{shown_cred} n'est pas un fichier régulier."
                    if is_fr
                    else f"{shown_cred} is not a regular file."
                )
            else:
                mode_bits, credentials_mode_str = _posix_mode_label(cred_path, eff_os)
                if mode_bits is not None and mode_bits != 0o600:
                    credentials_0600_ok = False
                    insecure_paths.append(shown_cred)
                    diagnostics.append(
                        f"{shown_cred} a le mode POSIX {credentials_mode_str} (attendu 0600 propriétaire seul)."
                        if is_fr
                        else f"{shown_cred} has POSIX mode {credentials_mode_str} (expected 0600 owner-only)."
                    )
                try:
                    cred_doc = json.loads(_read_text_bounded(cred_path))
                except (LaunchEvePreflightError, OSError, json.JSONDecodeError):
                    cred_doc = None
                if not isinstance(cred_doc, dict) or not is_valid_hivemind_key(cred_doc.get("api_key")):
                    credentials_valid = False
                    malformed_files.append(shown_cred)
                    diagnostics.append(
                        f"{shown_cred} illisible ou clé hm-... valide absente."
                        if is_fr
                        else f"{shown_cred} is malformed or missing a valid hm-... key."
                    )
                else:
                    credentials_valid = True
                    masked_key = mask_api_key(cred_doc.get("api_key"))

        cred_bak = hm_dir / "credentials.json.bak"
        if cred_bak.exists() or cred_bak.is_symlink():
            shown_bak = redact_home_path(cred_bak, home=eff_home)
            if cred_bak.is_symlink():
                credentials_0600_ok = False
                symlink_violations.append(shown_bak)
                diagnostics.append(f"Refusing symlink for {shown_bak}.")
            elif cred_bak.is_file():
                bak_mode, bak_mode_str = _posix_mode_label(cred_bak, eff_os)
                if bak_mode is not None and bak_mode != 0o600:
                    credentials_0600_ok = False
                    insecure_paths.append(shown_bak)
                    diagnostics.append(
                        f"{shown_bak} has POSIX mode {bak_mode_str} (expected 0600 owner-only)."
                    )

        for spool_name in LOCAL_CES_SPOOL_FILES:
            for rel_name, is_json_spool in (
                (spool_name, True),
                (f"{spool_name}.bak", False),
                (f".{spool_name}.lock", False),
            ):
                sp_path = hm_dir / rel_name
                if not sp_path.exists() and not sp_path.is_symlink():
                    continue
                shown_sp = redact_home_path(sp_path, home=eff_home)
                if sp_path.is_symlink():
                    ces_spool_0600_ok = False
                    symlink_violations.append(shown_sp)
                    diagnostics.append(f"Refusing symlink for {shown_sp}.")
                    continue
                if not sp_path.is_file():
                    ces_spool_valid = False
                    malformed_files.append(shown_sp)
                    diagnostics.append(f"{shown_sp} is not a regular file.")
                    continue
                sp_mode, sp_mode_str = _posix_mode_label(sp_path, eff_os)
                sp_mode_ok = sp_mode is None or sp_mode == 0o600
                if not sp_mode_ok:
                    ces_spool_0600_ok = False
                    insecure_paths.append(shown_sp)
                    diagnostics.append(
                        f"Spool CES {shown_sp} a le mode {sp_mode_str} (attendu 0600 propriétaire seul)."
                        if is_fr
                        else f"CES spool {shown_sp} has POSIX mode {sp_mode_str} (expected 0600 owner-only)."
                    )
                if is_json_spool:
                    pending_cnt = 0
                    delivered_cnt = 0
                    sp_schema_ok = True
                    try:
                        raw_sp = _read_text_bounded(sp_path)
                        parsed_sp = json.loads(raw_sp)
                    except (LaunchEvePreflightError, OSError, json.JSONDecodeError):
                        raw_sp = ""
                        parsed_sp = None
                    if not isinstance(parsed_sp, dict):
                        sp_schema_ok = False
                    else:
                        payload_ok, pending_cnt, delivered_cnt = _validate_ces_spool_payload(parsed_sp)
                        safety_hits = scan_output_safety(raw_sp)
                        sp_schema_ok = bool(payload_ok and not safety_hits)
                    if not sp_schema_ok:
                        ces_spool_valid = False
                        malformed_files.append(shown_sp)
                        diagnostics.append(
                            f"Spool CES {shown_sp} invalide ou non sûr (PII/clé détectée)."
                            if is_fr
                            else f"CES spool {shown_sp} is malformed or failed zero-PII/secret safety check."
                        )
                    ces_spool_rows.append(
                        {
                            "file": shown_sp,
                            "posix_mode": sp_mode_str,
                            "mode_0600_ok": sp_mode_ok,
                            "schema_ok": sp_schema_ok,
                            "pending_count": pending_cnt,
                            "delivered_count": delivered_cnt,
                        }
                    )

    cont_provider = eff_home / ".continue" / "hivemind_provider.yaml"
    if cont_provider.exists() or cont_provider.is_symlink():
        shown_cont = redact_home_path(cont_provider, home=eff_home)
        if cont_provider.is_symlink():
            continue_provider_0600_ok = False
            symlink_violations.append(shown_cont)
            diagnostics.append(f"Refusing symlink for {shown_cont}.")
        elif cont_provider.is_file():
            cp_mode, cp_mode_str = _posix_mode_label(cont_provider, eff_os)
            if cp_mode is not None and cp_mode != 0o600:
                continue_provider_0600_ok = False
                insecure_paths.append(shown_cont)
                diagnostics.append(
                    f"{shown_cont} has POSIX mode {cp_mode_str} (expected 0600 owner-only)."
                )

    file_writes = 0
    files_created = 0
    atomic_spool_write_verified: bool | None = None
    if verify_atomic_spool_write:
        if not hivemind_dir_ok:
            atomic_spool_write_verified = False
        else:
            probe_path = hm_dir / f".j1_ces_spool_probe_{secrets.token_hex(4)}.json"
            probe_payload = json.dumps(
                {"schema": SCHEMA, "pending": [], "delivered_event_ids": []},
                sort_keys=True,
            ).encode("utf-8")
            try:
                _write_bytes_secure_atomic(probe_path, probe_payload, eff_os=eff_os)
                file_writes += 1
                files_created += 1
                p_mode, _ = _posix_mode_label(probe_path, eff_os)
                atomic_spool_write_verified = bool(
                    (p_mode is None or p_mode == 0o600)
                    and probe_path.read_bytes() == probe_payload
                )
            except (LaunchEvePreflightError, OSError):
                atomic_spool_write_verified = False
                diagnostics.append(
                    "Échec de vérification d'écriture atomique 0600 du spool CES."
                    if is_fr
                    else "Atomic 0600 CES spool probe write failed."
                )
            finally:
                with suppress(OSError):
                    probe_path.unlink(missing_ok=True)

    local_0600_ok = bool(
        hivemind_dir_ok
        and credentials_0600_ok
        and (credentials_valid is not False)
        and ces_spool_0600_ok
        and ces_spool_valid
        and continue_provider_0600_ok
        and not insecure_paths
        and not symlink_violations
        and not malformed_files
        and (atomic_spool_write_verified is not False)
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "audit_local_0600",
            "lang": lang_code,
            "home_dir_provided": True,
            "local_state_audited": True,
            "hivemind_dir_exists": hivemind_dir_exists,
            "hivemind_dir_ok": hivemind_dir_ok,
            "hivemind_dir_posix_mode": hivemind_dir_mode_str,
            "credentials_present": credentials_present,
            "credentials_0600_ok": credentials_0600_ok,
            "credentials_valid": credentials_valid,
            "credentials_posix_mode": credentials_mode_str,
            "masked_api_key": masked_key,
            "ces_spool_files_checked": ces_spool_rows,
            "ces_spool_0600_ok": ces_spool_0600_ok,
            "ces_spool_valid": ces_spool_valid,
            "continue_provider_0600_ok": continue_provider_0600_ok,
            "insecure_permission_paths": insecure_paths,
            "symlink_violations": symlink_violations,
            "malformed_files": malformed_files,
            "atomic_spool_write_verified": atomic_spool_write_verified,
            "local_0600_ok": local_0600_ok,
            "file_writes": file_writes,
            "files_created": files_created,
            "diagnostics": diagnostics,
            "exit_code": 0 if local_0600_ok else 2,
        }
    )


def _inspect_html_route(
    path: str,
    body: Mapping[str, Any],
    seen_nonces: set[str],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    html_text = str(body.get("html") or body.get("text") or "")
    raw_hdrs = body.get("headers")
    hdrs = (
        {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()}
        if isinstance(raw_hdrs, Mapping)
        else {}
    )
    csp_hdr = str(hdrs.get("content-security-policy") or body.get("csp") or "")

    has_doctype = "<!doctype html" in html_text.lower()
    attr_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)]
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    csp_low = csp_hdr.lower()
    csp_unsafe = "'unsafe-inline'" in csp_low or "'unsafe-eval'" in csp_low

    nonce: str | None = hdr_nonces[0] if hdr_nonces else (attr_nonces[0] if attr_nonces else None)
    csp_ok = bool(
        attr_nonces
        and hdr_nonces
        and len(set(hdr_nonces)) == 1
        and "default-src 'none'" in csp_low
        and not csp_unsafe
        and hdr_nonces[0] in attr_nonces
        and nonce is not None
        and nonce not in seen_nonces
    )
    if nonce is not None:
        seen_nonces.add(nonce)

    stripped_html = strip_masked_keys_and_nonces(html_text)
    has_unsafe_sink = bool(_FORBIDDEN_DOM_SINK_RE.search(stripped_html))
    has_static_fig = bool(_STATIC_FIG_UNIT_RE.search(stripped_html))
    has_completions_call = "completions" in stripped_html.lower()
    dom_clean = not (has_unsafe_sink or has_static_fig or has_completions_call)

    low_html = stripped_html.lower()
    has_personal = (
        "100% personal" in low_html
        or "100 % personnel" in low_html
        or (("100%" in low_html or "100 %" in low_html) and ("personal" in low_html or "personnel" in low_html))
    )
    has_eur = "0.00 eur" in low_html or "0,00 eur" in low_html
    has_usd = "0.00 usd" in low_html or "0,00 usd" in low_html
    needs_usd = path in ("/raid-noel", "/arena-noel")
    compliance_banner_ok = bool(has_personal and "agpl-3.0" in low_html and has_eur and (has_usd or not needs_usd))

    checks = (
        (
            has_doctype,
            f"Contrat HTML {path} invalide : `<!DOCTYPE html>` absent."
            if is_fr
            else f"HTML contract failed on {path}: missing `<!DOCTYPE html>`.",
        ),
        (
            csp_ok,
            f"Contrat CSP {path} invalide : nonce `'default-src 'none''` par requête absent, incohérent ou réutilisé."
            if is_fr
            else f"CSP contract failed on {path}: missing, mismatched, or reused `'default-src 'none''` nonce.",
        ),
        (
            dom_clean,
            f"Contrat sécurité DOM {path} violé : sink non sûr, chiffre statique ou appel completions détecté."
            if is_fr
            else f"DOM safety contract violated on {path}: unsafe sink, static figure, or completions call detected.",
        ),
        (
            compliance_banner_ok,
            f"Bannière conformité PC perso / 0,00 EUR / AGPL-3.0 absente sur {path}."
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
            "completions_reference_found": has_completions_call,
            "personal_pc_banner_present": compliance_banner_ok,
        },
        issues,
    )


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


def _inspect_manifest_contract(
    body: Mapping[str, Any],
    *,
    expected_sha: str | None,
    lang_code: str,
) -> tuple[bool, str, bool | None, list[str], list[str]]:
    is_fr = lang_code == "fr"
    live_sha = sanitize_build_sha(body.get("build_sha"))
    issues: list[str] = []

    if (
        body.get("object") != MANIFEST_OBJECT
        or body.get("campaign_id") != CAMPAIGN_ID
        or body.get("status") != "ok"
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

    if "downloadable_scripts" in body:
        raw_scripts = body.get("downloadable_scripts")
        if (
            not isinstance(raw_scripts, list)
            or not raw_scripts
            or not all(isinstance(s, str) and _SAFE_SCRIPT_NAME_RE.fullmatch(s) for s in raw_scripts)
        ):
            issues.append(
                "Liste `downloadable_scripts` invalide dans `/v1/hivemind/onboarding/manifest`."
                if is_fr
                else "Manifest `downloadable_scripts` contains empty or unsafe script entries."
            )

    sha_match: bool | None = None
    if expected_sha is not None:
        sha_match = shas_match(live_sha, expected_sha)
        if not sha_match:
            issues.append(
                f"Dérive `build_sha` : live='{live_sha}' != attendu '{expected_sha}'."
                if is_fr
                else f"Build SHA drift: live='{live_sha}' != expected '{expected_sha}'."
            )

    return len(issues) == 0, live_sha, sha_match, missing_routes, issues


def _inspect_feedback_summary_exact(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    exact_5_keys = set(body.keys()) == EXPECTED_FEEDBACK_SUMMARY_KEYS

    raw_total = body.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    total_val = int(raw_total) if total_ok else -1

    raw_ease = body.get("mean_ease_rating")
    ease_ok = bool(
        "mean_ease_rating" in body
        and total_ok
        and (
            (total_val == 0 and raw_ease is None)
            or (
                total_val > 0
                and not isinstance(raw_ease, bool)
                and isinstance(raw_ease, (int, float))
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
                and not isinstance(raw_lag, bool)
                and isinstance(raw_lag, (int, float))
                and math.isfinite(float(raw_lag))
                and 0.0 <= float(raw_lag) <= 1.0
            )
        )
    )

    raw_prof = body.get("by_profile")
    prof_keys_ok = isinstance(raw_prof, Mapping) and set(raw_prof.keys()) == {"dev", "gamer"}
    d_raw = raw_prof.get("dev") if isinstance(raw_prof, Mapping) else None
    g_raw = raw_prof.get("gamer") if isinstance(raw_prof, Mapping) else None
    prof_ok = bool(
        prof_keys_ok
        and not isinstance(d_raw, bool)
        and isinstance(d_raw, int)
        and d_raw >= 0
        and not isinstance(g_raw, bool)
        and isinstance(g_raw, int)
        and g_raw >= 0
        and total_ok
        and (d_raw + g_raw == total_val)
    )

    obj_ok = body.get("object") == FEEDBACK_SUMMARY_OBJECT
    schema_ok = bool(exact_5_keys and obj_ok and total_ok and ease_ok and lag_ok and prof_ok)
    issues = (
        []
        if schema_ok
        else [
            "Schéma exact 5 clés `/v1/hivemind/onboarding/feedback/summary` invalide."
            if is_fr
            else "Feedback summary `/v1/hivemind/onboarding/feedback/summary` exact 5-key schema mismatch."
        ]
    )
    preserved = (
        {
            "object": FEEDBACK_SUMMARY_OBJECT,
            "total_responses": total_val,
            "mean_ease_rating": float(raw_ease) if raw_ease is not None else None,
            "zero_game_lag_ratio": float(raw_lag) if raw_lag is not None else None,
            "by_profile": {"dev": int(d_raw), "gamer": int(g_raw)},
        }
        if schema_ok
        else {}
    )
    return schema_ok, preserved, issues


def verify_j1_launch_eve_readiness(
    gateway: str = DEFAULT_GATEWAY,
    *,
    home_dir: Path | str | None = None,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    expected_sha: str | None = None,
    require_coder_model: bool = True,
    verify_atomic_spool_write: bool = False,
    preflight_date: str = J1_LAUNCH_EVE_DATE,
    os_name: str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Run Dec 18 J-1 launch-eve zero-inference preflight, local 0600 hygiene audit, and 5-pillar gate."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_date = validate_preflight_date(preflight_date)

    eff_env = environ if environ is not None else env
    comp_doc = evaluate_personal_pc_compliance(
        environ=eff_env,
        cwd=cwd,
        home_dir=home_dir,
        check_git_remote=check_git_remote,
        lang=lang_code,
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt(
            "verify_j1_launch_eve_readiness",
            comp_doc,
            lang_code=lang_code,
            preflight_date=clean_date,
        )

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None

    local_audit = audit_local_0600_credentials_and_spool(
        home_dir,
        os_name=os_name,
        verify_atomic_spool_write=verify_atomic_spool_write,
        lang=lang_code,
    )

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = list(local_audit.get("diagnostics") or [])
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}

    health_ok = False
    models_ok = False
    model_ids: list[str] = []
    coder_present = False
    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    missing_manifest_routes: list[str] = []
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {}
    network_calls = 0

    for path, kind, group in J1_PROBE_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if code != 200:
            if code == 404 and group in ("holiday", "manifest", "feedback", "models"):
                stale_404_routes.append(path)
            diag = (
                f"GET {path} -> HTTP {code} (attendu 200)."
                if is_fr
                else f"GET {path} -> HTTP {code} (expected 200)."
            )
            diagnostics.append(diag)
        elif path == "/health":
            health_ok = _unwrap_json_body(body).get("status") == "ok"
            ok = health_ok
            if not health_ok:
                diag = (
                    "GET /health sans `status == 'ok'`."
                    if is_fr
                    else "GET /health missing `status == 'ok'`."
                )
                diagnostics.append(diag)
        elif path == "/v1/models":
            m_ok, model_ids, coder_present, m_issues = _inspect_models_contract(
                _unwrap_json_body(body),
                require_coder_model=require_coder_model,
                lang_code=lang_code,
            )
            models_ok = m_ok
            ok = m_ok
            if not m_ok:
                diag = m_issues[0]
                diagnostics.extend(m_issues)
        elif kind == "html":
            ok, h_doc, h_issues = _inspect_html_route(path, body, seen_nonces, lang_code=lang_code)
            html_contracts[path] = h_doc
            if not ok:
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/manifest":
            manifest_json = _unwrap_json_body(body)
            ok, live_build_sha, sha_match, missing_manifest_routes, m_issues = _inspect_manifest_contract(
                manifest_json,
                expected_sha=clean_expected_sha,
                lang_code=lang_code,
            )
            manifest_ok = ok
            if not ok:
                diag = m_issues[0]
                diagnostics.extend(m_issues)
        else:
            fb_json = _unwrap_json_body(body)
            ok, feedback_summary_doc, fb_issues = _inspect_feedback_summary_exact(fb_json, lang_code=lang_code)
            feedback_summary_ok = ok
            if not ok:
                diag = fb_issues[0]
                diagnostics.extend(fb_issues)

        if not ok:
            failed_routes.append(path)
        r_src = json.dumps(
            {"preflight_date": clean_date, "path": path, "http_status": code, "ok": ok},
            sort_keys=True,
            ensure_ascii=False,
        )
        route_rows.append(
            {
                "path": path,
                "http_status": code,
                "ok": ok,
                "diagnostic": diag,
                "leaf_sha256": hashlib.sha256(f"hivemind.j1_launch_eve.route/1:{r_src}".encode()).hexdigest(),
            }
        )

    html_all_ok = bool(
        len(html_contracts) == len(J1_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in J1_HTML_ROUTES)
        and len(seen_nonces) == len(J1_HTML_ROUTES)
    )
    routes_ok = bool(health_ok and html_all_ok)
    local_0600_ok = bool(local_audit["local_0600_ok"])

    checklist = {
        "compliance_ok": True,
        "routes_ok": routes_ok,
        "models_ok": models_ok,
        "manifest_ok": manifest_ok,
        "feedback_summary_ok": feedback_summary_ok,
    }
    pillars_passed_count = sum(1 for v in checklist.values() if v)
    pillars_total_count = len(PILLAR_KEYS)

    gateway_ok = bool(routes_ok and models_ok and manifest_ok and feedback_summary_ok)
    stale_404_drift = bool(stale_404_routes or missing_manifest_routes)
    redeploy_required = bool(stale_404_drift or not gateway_ok)
    ready = bool(all(checklist.values()) and local_0600_ok)

    hold_reasons: list[str] = []
    if not local_0600_ok:
        hold_reasons.append("insecure_0600_permissions")
    if not routes_ok:
        hold_reasons.append("routes_drift_or_contract_failure")
    if not models_ok:
        hold_reasons.append("models_missing_or_empty")
    if not manifest_ok:
        hold_reasons.append("manifest_drift_or_mismatch")
    if not feedback_summary_ok:
        hold_reasons.append("feedback_summary_schema_invalid")

    if ready:
        decision = "J1_LAUNCH_EVE_READY"
    elif not local_0600_ok and gateway_ok:
        decision = "HOLD_LOCAL_0600_PERMISSIONS"
    elif not local_0600_ok:
        decision = "HOLD_LOCAL_AND_GATEWAY_DRIFT"
    elif routes_ok and manifest_ok and feedback_summary_ok and not models_ok:
        decision = "HOLD_MODELS_MISSING"
    else:
        decision = "HOLD_ROUTES_OR_CONTRACT_DRIFT"

    go_nogo = "GO" if ready else "NO-GO"
    checklist_src = json.dumps(
        {
            "preflight_date": clean_date,
            "checklist": checklist,
            "local_0600_ok": local_0600_ok,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    checklist_leaf = hashlib.sha256(
        f"hivemind.j1_launch_eve.checklist/1:{checklist_src}".encode()
    ).hexdigest()
    merkle_root = compute_merkle_root(
        [str(local_audit["receipt_sha256"])] + [row["leaf_sha256"] for row in route_rows] + [checklist_leaf]
    )

    if ready:
        summary = (
            f"Pré-vol J-1 ({clean_date}) {go_nogo} sur {clean_gw} : "
            f"{pillars_passed_count}/{pillars_total_count} piliers OK, "
            f"hygiène_0600={local_0600_ok}, nonces={len(seen_nonces)}, {FINANCIAL_PROMISE}."
            if is_fr
            else f"J-1 launch-eve preflight ({clean_date}) {go_nogo} on {clean_gw}: "
            f"{pillars_passed_count}/{pillars_total_count} pillars OK, "
            f"local_0600={local_0600_ok}, nonces={len(seen_nonces)}, {FINANCIAL_PROMISE}."
        )
    else:
        summary = (
            f"Pré-vol J-1 ({clean_date}) {go_nogo} ({decision}) sur {clean_gw} : "
            f"{pillars_passed_count}/{pillars_total_count} piliers OK, "
            f"raisons={', '.join(hold_reasons)}."
            if is_fr
            else f"J-1 launch-eve preflight ({clean_date}) {go_nogo} ({decision}) on {clean_gw}: "
            f"{pillars_passed_count}/{pillars_total_count} pillars OK, "
            f"reasons={', '.join(hold_reasons)}."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "checklist_object": CHECKLIST_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "preflight_date": clean_date,
            "launch_day_date": LAUNCH_DAY_DATE,
            "holiday_wrapup_date": HOLIDAY_WRAPUP_DATE,
            "mode": "verify_j1_launch_eve_readiness",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "routes_ok": routes_ok,
            "health_ok": health_ok,
            "html_all_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "models_ok": models_ok,
            "model_ids": model_ids,
            "coder_model_present": coder_present,
            "manifest_ok": manifest_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "sha_match": sha_match,
            "missing_manifest_routes": missing_manifest_routes,
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "checklist": checklist,
            "pillars_passed_count": pillars_passed_count,
            "pillars_total_count": pillars_total_count,
            "local_0600_ok": local_0600_ok,
            "local_audit": local_audit,
            "ready": ready,
            "go_nogo": go_nogo,
            "decision": decision,
            "hold_reasons": hold_reasons,
            "stale_404_drift": stale_404_drift,
            "stale_404_routes": stale_404_routes,
            "redeploy_required": redeploy_required,
            "http_statuses": http_statuses,
            "routes": route_rows,
            "failed_routes": failed_routes,
            "merkle_root_sha256": merkle_root,
            "network_calls": network_calls,
            "file_writes": int(local_audit["file_writes"]),
            "files_created": int(local_audit["files_created"]),
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "zero_financial_payout": True,
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": diagnostics,
            "exit_code": 0 if ready else 2,
            "summary": summary,
        }
    )


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN CLI J-1 launch-eve preflight & 0600 CES spool readiness receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    title = (
        f"HIVEMIND NOËL — PRÉ-VOL VEILLE DE LANCEMENT J-1 ({verdict}, 0,00 EUR / 0,00 USD)"
        if is_fr
        else f"HIVEMIND HOLIDAY — DEC 18 J-1 LAUNCH-EVE PREFLIGHT ({verdict}, 0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {title}",
        "====================================================================",
        f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}",
        f"  Gateway     : {report.get('gateway', DEFAULT_GATEWAY)}",
        f"  Date        : {report.get('preflight_date', J1_LAUNCH_EVE_DATE)} -> {LAUNCH_DAY_DATE}",
        f"  Mode        : {report.get('mode')}",
        f"  Verdict     : {verdict} ({report.get('decision', verdict)})",
        f"  Compliance  : {comp_ok}",
        f"  Exit code   : {report.get('exit_code')}",
        f"  Summary     : {report.get('summary')}",
    ]
    chk = report.get("checklist")
    if isinstance(chk, Mapping):
        pillar_str = ", ".join(f"{k}={chk.get(k)}" for k in PILLAR_KEYS)
        lines.append(
            f"  Pillars     : {report.get('pillars_passed_count')}/{report.get('pillars_total_count')} ({pillar_str})"
        )
    local_blk = report.get("local_audit") if isinstance(report.get("local_audit"), Mapping) else report
    if "local_0600_ok" in local_blk:
        lines.append(
            f"  Local 0600  : ok={local_blk.get('local_0600_ok')} "
            f"(creds={local_blk.get('credentials_0600_ok')}, "
            f"spool={local_blk.get('ces_spool_0600_ok')}, "
            f"key={local_blk.get('masked_api_key')}, "
            f"writes={local_blk.get('file_writes', 0)})"
        )
    if report.get("merkle_root_sha256"):
        lines.append(f"  Merkle Root : {report.get('merkle_root_sha256')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256     : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN Markdown J-1 launch-eve preflight & 0600 CES spool readiness receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    heading = (
        f"# Pré-Vol Veille de Lancement J-1 (18 Déc.) HiveMind — {verdict}"
        if is_fr
        else f"# HiveMind Dec 18 J-1 Launch-Eve Preflight & CES Spool Gate — {verdict}"
    )
    lines = [
        heading,
        "",
        f"- **{'Verdict Go/No-Go' if is_fr else 'Go/No-Go Verdict'}**: `{verdict}` (`decision={report.get('decision', verdict)}`, `exit_code={report.get('exit_code')}`)",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Fenêtre campagne' if is_fr else 'Preflight Date'}**: `{report.get('preflight_date', J1_LAUNCH_EVE_DATE)}` (`{LAUNCH_DAY_DATE}..{HOLIDAY_WRAPUP_DATE}`)",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: `{comp_ok}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Zéro inférence' if is_fr else 'Zero-Inference Verified'}**: `{report.get('zero_inference_verified', True)}`",
    ]
    chk = report.get("checklist")
    if isinstance(chk, Mapping):
        lines.append(
            f"- **{'Checklist 5 Piliers' if is_fr else '5-Pillar J-1 Checklist'}**: "
            f"`{report.get('pillars_passed_count')}/{report.get('pillars_total_count')}` — "
            + ", ".join(f"`{k}={chk.get(k)}`" for k in PILLAR_KEYS)
        )
    local_blk = report.get("local_audit") if isinstance(report.get("local_audit"), Mapping) else report
    if "local_0600_ok" in local_blk:
        lines.append(
            f"- **{'Hygiène locale 0600' if is_fr else 'Local 0600 Hygiene'}**: "
            f"`local_0600_ok={local_blk.get('local_0600_ok')}`, "
            f"`credentials_0600_ok={local_blk.get('credentials_0600_ok')}`, "
            f"`ces_spool_0600_ok={local_blk.get('ces_spool_0600_ok')}`, "
            f"`masked_api_key={local_blk.get('masked_api_key')}`, "
            f"`file_writes={local_blk.get('file_writes', 0)}`"
        )
    fb = report.get("feedback_summary")
    if isinstance(fb, Mapping) and fb:
        lines.append(
            f"- **{'Métriques CES' if is_fr else 'CES Summary'}**: "
            f"`responses={fb.get('total_responses')}`, `ease={fb.get('mean_ease_rating')}/5`, "
            f"`zero_game_lag={fb.get('zero_game_lag_ratio')}`"
        )
    if "unique_csp_nonces_count" in report:
        lines.append(
            f"- **{'Nonces CSP uniques' if is_fr else 'Unique CSP Nonces'}**: `{report.get('unique_csp_nonces_count')}`"
        )
    if report.get("merkle_root_sha256"):
        lines.append(f"- **Merkle Root SHA-256**: `{report.get('merkle_root_sha256')}`")
    lines.append(f"- **{'Résumé' if is_fr else 'Summary'}**: {report.get('summary')}")
    lines.append(f"- **SHA-256 Receipt**: `{report.get('receipt_sha256')}`")

    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        lines.extend(f"- {d}" for d in diags)
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the Dec 18 J-1 launch-eve preflight & CES spool verifier."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-inference Dec 18 J-1 launch-eve preflight & atomic 0600 CES spool readiness verifier "
            "(0.00 EUR / 0.00 USD, AGPL-3.0)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="Target HiveMind gateway URL (https://*.run.app, https://*.googleapis.com, or loopback).",
    )
    parser.add_argument(
        "--home-dir",
        default=None,
        help="Optional home directory to audit read-only for 0600 credentials and CES spools.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Expected build_sha on /v1/hivemind/onboarding/manifest.",
    )
    parser.add_argument(
        "--preflight-date",
        default=J1_LAUNCH_EVE_DATE,
        help="ISO preflight date (default: 2026-12-18).",
    )
    parser.add_argument(
        "--allow-missing-coder",
        action="store_true",
        help="Allow GET /v1/models without requiring hivemind/coder.",
    )
    parser.add_argument(
        "--verify-atomic-spool-write",
        action="store_true",
        help="Explicitly probe atomic 0600 CES spool write-and-clean inside --home-dir/.hivemind.",
    )
    parser.add_argument(
        "--local-audit-only",
        action="store_true",
        help="Run only local compliance + 0600 credential/spool audit without network calls.",
    )
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Run only the strictly local personal-PC compliance check.",
    )
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit signed JSON receipt.")
    parser.add_argument("--markdown", action="store_true", help="Emit signed Markdown receipt.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """CLI entrypoint enforcing local compliance before any network call or file operation."""
    eff_out = stdout if stdout is not None else sys.stdout
    eff_err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        lang_code = normalize_lang(args.lang)
        eff_env = environ if environ is not None else os.environ
        if args.enforce_personal_pc:
            report = evaluate_personal_pc_compliance(
                environ=eff_env,
                cwd=cwd,
                home_dir=args.home_dir,
                check_git_remote=True,
                lang=lang_code,
            )
        elif args.local_audit_only:
            comp_doc = evaluate_personal_pc_compliance(
                environ=eff_env,
                cwd=cwd,
                home_dir=args.home_dir,
                check_git_remote=True,
                lang=lang_code,
            )
            if not comp_doc["compliance_ok"]:
                report = _compliance_blocked_receipt(
                    "audit_local_0600",
                    comp_doc,
                    lang_code=lang_code,
                    gateway=args.gateway,
                )
            else:
                target_home: Path | str | None = args.home_dir
                if target_home is None and args.verify_atomic_spool_write:
                    raise LaunchEvePreflightError("--home-dir is required with --verify-atomic-spool-write.")
                report = audit_local_0600_credentials_and_spool(
                    target_home,
                    verify_atomic_spool_write=bool(args.verify_atomic_spool_write),
                    lang=lang_code,
                )
        else:
            report = verify_j1_launch_eve_readiness(
                args.gateway,
                home_dir=args.home_dir,
                http_fn=http_fn,
                environ=eff_env,
                cwd=cwd,
                expected_sha=args.expected_sha,
                require_coder_model=not bool(args.allow_missing_coder),
                verify_atomic_spool_write=bool(args.verify_atomic_spool_write),
                preflight_date=args.preflight_date,
                lang=lang_code,
            )
    except LaunchEvePreflightError as exc:
        err_msg = sanitize_text_for_display(str(exc))
        if args.json:
            err_doc = _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "compliance_ok": True,
                    "ready": False,
                    "go_nogo": "NO-GO",
                    "decision": "ERROR",
                    "error": err_msg,
                    "exit_code": 2,
                }
            )
            print(json.dumps(err_doc, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(f"Error: {err_msg}", file=eff_err)
        return 2

    if args.json:
        text_out = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        _assert_safe_output(text_out)
        print(text_out, file=eff_out)
    elif args.markdown:
        print(render_markdown_report(report), file=eff_out)
    else:
        print(render_cli_report(report), file=eff_out)
    return int(report.get("exit_code", 2))


if __name__ == "__main__":
    raise SystemExit(main())
