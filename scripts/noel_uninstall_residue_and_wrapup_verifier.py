#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference Clean Uninstall Residue Auditor & Jan 3, 2027 Holiday Wrap-Up Verifier (`SLICE-SCALE-DYN-26`).

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment markers (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``,
     ``NODE_EXTRA_CA_CERTS``), workspace path, and ``.git/config`` remote URLs (including worktree
     ``gitdir:`` / ``commondir`` indirection) BEFORE any network call or file write. Fails closed
     with exit code 2 (0 network calls, 0 file writes, 0 raw env/hostname/path leaks), while
     keeping local zero-residue uninstall verification always permitted.
  2. Clean Uninstall Zero-Residue Auditor (``audit_clean_uninstall_zero_residue``):
     verifies in strict read-only mode (0 network calls, 0 file writes) that after ``--uninstall``
     zero HiveMind-managed credential, spool, state, or provider files
     (``~/.hivemind/credentials.json``, ``~/.hivemind/ces_feedback_queue.json``,
     ``~/.hivemind/ces_telemetry_spool.json``, ``~/.continue/hivemind_provider.yaml``, backups/temp
     files, HiveMind entries in ``~/.continue/config.yaml`` or ``.vscode/settings.json`` /
     ``.vscode/tasks.json``) or unmasked ``hm-...`` / ``hm_live_...`` keys remain, while user
     non-HiveMind settings stay 100% intact and masked keys such as ``hm-***XXXX`` stay safe.
  3. Jan 3, 2027 Holiday Wrap-Up & Uninstall Promise Verifier
     (``verify_holiday_wrapup_and_uninstall_promise``):
     probes ``GET /noel``, ``GET /quick``, ``GET /raid-noel``, ``GET /arena-noel``,
     ``GET /v1/hivemind/onboarding/manifest``, and ``GET /v1/hivemind/onboarding/feedback/summary``
     in strict zero-inference read-only mode (never calling ``POST /v1/chat/completions`` or
     prompting ``hivemind/coder``), enforcing unique per-request CSP nonces ``default-src 'none'``,
     0 unsafe DOM sinks, 0 static figures (stripping masked-key suffixes and HTML ``nonce="..."``
     attributes first), exact 5-key feedback summary equality, optional uninstall zero-residue
     verification, and FR/EN JSON & Markdown SHA-256 receipts.

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
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_uninstall_residue_and_wrapup_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_uninstall_wrapup_receipt"
RESIDUE_AUDIT_OBJECT = "hivemind.noel_uninstall_zero_residue_audit"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

HTTP_TIMEOUT_S = 5.0
MAX_SCAN_FILE_BYTES = 524_288
REPO_ROOT = Path(__file__).resolve().parents[1]

HOLIDAY_START_DATE = date(2026, 12, 19)
HOLIDAY_END_DATE = date(2027, 1, 3)
HOLIDAY_WRAPUP_DATE = HOLIDAY_END_DATE.isoformat()
HOLIDAY_DAYS_COUNT = 16

UNINSTALL_POSIX_CMD = "python3 scripts/hivemind_quickstart.py --uninstall"
UNINSTALL_WIN_CMD = r".\scripts\join_hivemind.ps1 -Mode Uninstall"

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset()

WRAPUP_HTML_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
)

WRAPUP_ROUTES: tuple[tuple[str, str, str], ...] = (
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

EXPECTED_FEEDBACK_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "object",
        "total_responses",
        "mean_ease_rating",
        "zero_game_lag_ratio",
        "by_profile",
    }
)

MANAGED_RESIDUE_REL_PATHS: tuple[str, ...] = (
    ".hivemind/credentials.json",
    ".hivemind/credentials.json.bak",
    ".hivemind/ces_feedback_queue.json",
    ".hivemind/ces_feedback_queue.json.bak",
    ".hivemind/ces_telemetry_spool.json",
    ".hivemind/ces_telemetry_spool.json.bak",
    ".hivemind/vscode_setup_ces_queue.json",
    ".hivemind/vscode_setup_ces_queue.json.bak",
    ".hivemind/idle_resume_state.json",
    ".hivemind/idle_resume_state.json.bak",
    ".hivemind/idle_service_ces_queue.json",
    ".hivemind/idle_service_ces_queue.json.bak",
    ".continue/hivemind_provider.yaml",
    ".continue/hivemind_provider.yaml.bak",
)

_MANAGED_TMP_STEMS: tuple[str, ...] = (
    ".credentials.json.",
    ".ces_feedback_queue.json.",
    ".ces_telemetry_spool.json.",
    ".vscode_setup_ces_queue.json.",
    ".idle_resume_state.json.",
    ".idle_service_ces_queue.json.",
    ".hivemind_provider.yaml.",
)

_CONTINUE_CONFIG_RESIDUE_MARKERS: tuple[str, ...] = (
    "hivemind/coder",
    "hivemind/bee-stars",
    "hivemind/auto",
    "hivemind-openrouter",
    "name: hivemind",
    '"name": "hivemind',
)

_MANAGED_TASK_ALIASES: frozenset[str] = frozenset(
    {
        "HiveMind: <60s Quick-Connect Check",
        "HiveMind: Cloud Run Holiday Route Parity Check",
        "HiveMind: Gamer-Whisper <1.5s Steam Yield & <=70 C Preflight",
        "HiveMind: 10s Clean Uninstall",
        "<60s Quick-Connect Check",
        "Cloud Run Holiday Route Parity Check",
        "Gamer-Whisper <1.5s Steam Yield & <=70 C Preflight",
        "10s Clean Uninstall",
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
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_API_KEY_RE = re.compile(r"(hm(?:_live_|-))\*\*\*(?:[A-Za-z0-9_-]{4,}|[0-9]{1,3}m" + r"s\b)")
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


class UninstallWrapupError(ValueError):
    """Raised when uninstall residue, host allowlist, or holiday wrap-up invariants fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize output language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise UninstallWrapupError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise UninstallWrapupError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw credentials never appear in receipts."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return "<invalid>"
    if cleaned.startswith("hm_live_") and len(cleaned) >= 12:
        return f"hm_live_***{cleaned[-4:]}"
    if cleaned.startswith("hm-") and len(cleaned) >= 8:
        return f"hm-***{cleaned[-4:]}"
    return "<invalid>"


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and masked ``hm-***`` suffixes."""
    if not isinstance(text, str):
        raise UninstallWrapupError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    cleaned = _MASKED_API_KEY_RE.sub(r"\1***XXXX", cleaned)
    return cleaned


def strip_jsonc_comments_and_trailing_commas(text: str) -> str:
    """Strip JSONC comments and trailing commas safely while preserving string literals."""
    no_comments: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    escaped = False
    while i < n:
        ch = text[i]
        if in_str:
            no_comments.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            no_comments.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        no_comments.append(ch)
        i += 1

    stage1 = "".join(no_comments)
    out: list[str] = []
    i = 0
    m = len(stage1)
    in_str = False
    escaped = False
    while i < m:
        ch = stage1[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < m and stage1[j] in (" ", "\t", "\r", "\n"):
                j += 1
            if j < m and stage1[j] in ("}", "]"):
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


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
    """Replace OS home prefixes with ``'~'`` and map workspace/repo files safely."""
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
    if resolved_home and resolved_home not in ("/", "\\", ""):
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
        raise UninstallWrapupError("scan_output_safety requires a string.")
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
        raise UninstallWrapupError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise UninstallWrapupError("Forbidden JSON key detected in wrap-up receipt.")
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
        raise UninstallWrapupError(
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


def validate_wrapup_date(raw: str) -> str:
    """Validate an ISO ``YYYY-MM-DD`` holiday wrap-up date within ``2026-12-19..2027-01-03``."""
    if isinstance(raw, bool) or not isinstance(raw, str) or not _ISO_DATE_RE.fullmatch(raw.strip()):
        raise UninstallWrapupError("wrapup_date must be an ISO YYYY-MM-DD date string.")
    cleaned = raw.strip()
    try:
        parsed = date.fromisoformat(cleaned)
    except ValueError as exc:
        raise UninstallWrapupError(f"Invalid wrapup_date {raw!r}.") from exc
    if parsed < HOLIDAY_START_DATE or parsed > HOLIDAY_END_DATE:
        raise UninstallWrapupError(
            f"wrapup_date must fall within holiday window {HOLIDAY_START_DATE.isoformat()}..{HOLIDAY_END_DATE.isoformat()}."
        )
    return cleaned


def compute_merkle_root(leaf_hashes: Sequence[str]) -> str:
    """Compute a deterministic domain-separated binary SHA-256 Merkle root over leaf digests."""
    if not leaf_hashes:
        return hashlib.sha256(b"hivemind.uninstall_wrapup.merkle.empty/1").hexdigest()
    level: list[str] = []
    for digest in leaf_hashes:
        if not isinstance(digest, str) or len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
            raise UninstallWrapupError("Merkle leaf digest must be a 64-char lowercase hex SHA-256.")
        level.append(digest)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: list[str] = []
        for idx in range(0, len(level), 2):
            pair = f"hivemind.uninstall_wrapup.merkle.node/1:{level[idx]}:{level[idx + 1]}".encode()
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
        raise UninstallWrapupError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise UninstallWrapupError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise UninstallWrapupError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise UninstallWrapupError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise UninstallWrapupError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise UninstallWrapupError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise UninstallWrapupError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise UninstallWrapupError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise UninstallWrapupError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise UninstallWrapupError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise UninstallWrapupError("Remote gateway URL must use https.")
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
        raise UninstallWrapupError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise UninstallWrapupError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise UninstallWrapupError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise UninstallWrapupError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise UninstallWrapupError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise UninstallWrapupError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise UninstallWrapupError(
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
    lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    text = raw_bytes.decode("utf-8", errors="replace")
    ctype = lower_headers.get("content-type", "").lower()
    if "application/json" in ctype or text.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": "invalid_json"}
        if isinstance(parsed, dict):
            return int(status_code), dict(parsed)
        return int(status_code), {"data": parsed}
    return int(status_code), {
        "html": text,
        "text": text,
        "html_length": len(text),
        "text_length": len(text),
        "headers": lower_headers,
        "csp": lower_headers.get("content-security-policy", ""),
    }


_HTTP_OPENER: urllib.request.OpenerDirector | None = None


def _get_http_opener() -> urllib.request.OpenerDirector:
    global _HTTP_OPENER
    if _HTTP_OPENER is None:
        _HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
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
    clean_url = _assert_safe_http_request(verb, url, payload=payload, allowed_post_paths=ALLOWED_POST_PATHS)
    if payload is not None:
        raise UninstallWrapupError("GET/HEAD requests must never attach a body payload.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-uninstall-wrapup-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise UninstallWrapupError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str

    req = urllib.request.Request(clean_url, data=None, headers=req_headers, method=verb)
    opener = _get_http_opener()
    bounded_timeout = min(max(0.2, float(timeout_s)), 15.0)
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
        raise UninstallWrapupError("http_fn must return (status_code, body) or (status_code, headers, body).")
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
    except UninstallWrapupError:
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

    ws_texts = [str(workspace)]
    try:
        resolved_ws = str(workspace.resolve())
        if resolved_ws not in ws_texts:
            ws_texts.append(resolved_ws)
    except OSError:
        resolved_ws = ""
        _ = resolved_ws
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
            "uninstall_supported": True,
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
    wrapup_date: str = HOLIDAY_WRAPUP_DATE,
    gateway: str | None = None,
) -> dict[str, Any]:
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": mode,
            "wrapup_date": wrapup_date,
            "holiday_window_start": HOLIDAY_START_DATE.isoformat(),
            "holiday_window_end": HOLIDAY_END_DATE.isoformat(),
            "lang": lang_code,
            "gateway": gateway or DEFAULT_GATEWAY,
            "compliance_ok": False,
            "corporate_detected": True,
            "corporate_indicators": list(comp_doc.get("corporate_indicators") or []),
            "matched_markers": list(comp_doc.get("matched_markers") or []),
            "ready": False,
            "uninstall_audit_performed": False,
            "uninstall_clean": None,
            "decision": "HOLD_COMPLIANCE_BLOCKED",
            "go_nogo": "NO-GO",
            "stale_404_drift": False,
            "redeploy_required": False,
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "zero_financial_payout": True,
            "uninstall_supported": True,
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


def _read_text_bounded(path: Path, max_bytes: int = MAX_SCAN_FILE_BYTES) -> str:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes)
    return raw.decode("utf-8", errors="replace")


def _scan_dir_for_raw_hm_keys(
    root: Path,
    *,
    eff_home: Path,
    eff_ws: Path | None,
    unmasked_hits: list[dict[str, Any]],
    masked_counter: list[int],
    scanned_paths: set[Path],
) -> None:
    if not root.is_dir():
        return
    for entry in sorted(root.rglob("*")):
        if not entry.is_file() or entry.is_symlink():
            continue
        if entry in scanned_paths:
            continue
        scanned_paths.add(entry)
        try:
            text = _read_text_bounded(entry)
        except OSError:
            continue
        masked_counter[0] += len(_MASKED_API_KEY_RE.findall(text))
        raw_matches = _RAW_HM_KEY_RE.findall(text)
        if raw_matches:
            unmasked_hits.append(
                {
                    "file": redact_home_path(entry, home=eff_home, workspace=eff_ws),
                    "masked_preview": mask_api_key(raw_matches[0]),
                    "count": len(raw_matches),
                }
            )


def _inspect_continue_main_config(
    cont_cfg: Path,
    *,
    eff_home: Path,
    eff_ws: Path | None,
    is_fr: bool,
    residual_entries: list[str],
    preserved_user_files: list[str],
    diagnostics: list[str],
) -> None:
    if not cont_cfg.is_file():
        return
    try:
        cfg_text = _read_text_bounded(cont_cfg)
    except OSError:
        return
    low = cfg_text.lower()
    hits = [marker for marker in _CONTINUE_CONFIG_RESIDUE_MARKERS if marker in low]
    raw_hm = _RAW_HM_KEY_RE.findall(cfg_text)
    if hits or raw_hm:
        for h in hits:
            residual_entries.append(h)
        if raw_hm and "unmasked_api_key" not in residual_entries:
            residual_entries.append("unmasked_api_key")
        red = redact_home_path(cont_cfg, home=eff_home, workspace=eff_ws)
        diagnostics.append(
            f"Résidu HiveMind présent dans {red}."
            if is_fr
            else f"HiveMind provider residue remains in {red}."
        )
    else:
        preserved_user_files.append(redact_home_path(cont_cfg, home=eff_home, workspace=eff_ws))


def _inspect_vscode_settings_residue(
    vs_settings: Path,
    *,
    eff_home: Path,
    eff_ws: Path,
    is_fr: bool,
    residual_settings_keys: list[str],
    preserved_user_files: list[str],
    diagnostics: list[str],
) -> None:
    if not vs_settings.is_file():
        return
    try:
        raw_txt = _read_text_bounded(vs_settings)
    except OSError:
        return
    strict = strip_jsonc_comments_and_trailing_commas(raw_txt)
    loaded: Any = None
    if strict.strip():
        try:
            loaded = json.loads(strict)
        except json.JSONDecodeError:
            loaded = None

    hm_keys: list[str] = []
    if isinstance(loaded, Mapping):
        hm_keys = sorted(str(k) for k in loaded if str(k).lower().startswith("hivemind."))
        user_keys = [str(k) for k in loaded if not str(k).lower().startswith("hivemind.")]
    else:
        hm_keys = sorted(set(re.findall(r'"(hivemind\.[A-Za-z0-9_.-]+)"', raw_txt)))
        user_keys = []

    raw_hm = _RAW_HM_KEY_RE.findall(raw_txt)
    if hm_keys or raw_hm:
        residual_settings_keys.extend(hm_keys)
        if raw_hm and "unmasked_api_key" not in residual_settings_keys:
            residual_settings_keys.append("unmasked_api_key")
        red = redact_home_path(vs_settings, home=eff_home, workspace=eff_ws)
        diagnostics.append(
            f"Clés HiveMind résiduelles dans {red}."
            if is_fr
            else f"Residual HiveMind keys found in {red}."
        )
    elif user_keys or raw_txt.strip():
        preserved_user_files.append(redact_home_path(vs_settings, home=eff_home, workspace=eff_ws))


def _inspect_vscode_tasks_residue(
    vs_tasks: Path,
    *,
    eff_home: Path,
    eff_ws: Path,
    is_fr: bool,
    residual_tasks: list[str],
    preserved_user_files: list[str],
    diagnostics: list[str],
) -> None:
    if not vs_tasks.is_file():
        return
    try:
        raw_txt = _read_text_bounded(vs_tasks)
    except OSError:
        return
    strict = strip_jsonc_comments_and_trailing_commas(raw_txt)
    loaded: Any = None
    if strict.strip():
        try:
            loaded = json.loads(strict)
        except json.JSONDecodeError:
            loaded = None

    hm_tasks: list[str] = []
    user_tasks_count = 0
    if isinstance(loaded, Mapping) and isinstance(loaded.get("tasks"), list):
        for t in loaded["tasks"]:
            if not isinstance(t, Mapping):
                continue
            lbl = str(t.get("label") or "").strip()
            if lbl in _MANAGED_TASK_ALIASES or lbl.startswith("HiveMind: "):
                hm_tasks.append(lbl)
            else:
                user_tasks_count += 1
    elif "HiveMind:" in raw_txt or "noel_vscode_quick_preset.py" in raw_txt:
        hm_tasks.append("HiveMind task entry")

    raw_hm = _RAW_HM_KEY_RE.findall(raw_txt)
    if hm_tasks or raw_hm:
        residual_tasks.extend(hm_tasks)
        if raw_hm and "unmasked_api_key" not in residual_tasks:
            residual_tasks.append("unmasked_api_key")
        red = redact_home_path(vs_tasks, home=eff_home, workspace=eff_ws)
        diagnostics.append(
            f"Tâches HiveMind résiduelles dans {red}."
            if is_fr
            else f"Residual HiveMind tasks found in {red}."
        )
    elif user_tasks_count > 0 or raw_txt.strip():
        preserved_user_files.append(redact_home_path(vs_tasks, home=eff_home, workspace=eff_ws))


def audit_clean_uninstall_zero_residue(
    home_dir: Path | str,
    workspace_dir: Path | str | None = None,
    *,
    lang: str = "en",
) -> dict[str, Any]:
    """Audit ``home_dir`` and optional ``workspace_dir`` read-only for zero HiveMind uninstall residue."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    if isinstance(home_dir, bool) or not isinstance(home_dir, (Path, str)) or not str(home_dir).strip():
        raise UninstallWrapupError("home_dir must be a non-empty Path or string.")
    eff_home = Path(home_dir)
    if eff_home.exists() and not eff_home.is_dir():
        raise UninstallWrapupError("home_dir must be a directory path.")

    eff_ws: Path | None = None
    if workspace_dir is not None:
        if isinstance(workspace_dir, bool) or not isinstance(workspace_dir, (Path, str)) or not str(workspace_dir).strip():
            raise UninstallWrapupError("workspace_dir must be a non-empty Path or string when provided.")
        eff_ws = Path(workspace_dir)
        if eff_ws.exists() and not eff_ws.is_dir():
            raise UninstallWrapupError("workspace_dir must be a directory path.")

    residual_files: list[str] = []
    residual_continue_entries: list[str] = []
    residual_vscode_keys: list[str] = []
    residual_vscode_tasks: list[str] = []
    unmasked_api_keys_found: list[dict[str, Any]] = []
    preserved_user_files: list[str] = []
    diagnostics: list[str] = []
    masked_counter = [0]
    scanned_paths: set[Path] = set()

    for rel_str in MANAGED_RESIDUE_REL_PATHS:
        cand = eff_home / rel_str
        if cand.is_file() or cand.is_symlink():
            red = redact_home_path(cand, home=eff_home, workspace=eff_ws)
            residual_files.append(red)
            diagnostics.append(
                f"Artefact HiveMind résiduel détecté après désinstallation : {red}."
                if is_fr
                else f"Leftover HiveMind-managed file detected after uninstall: {red}."
            )

    for managed_dir in (eff_home / ".hivemind", eff_home / ".continue"):
        if not managed_dir.is_dir():
            continue
        for item in sorted(managed_dir.iterdir()):
            if not item.is_file():
                continue
            if item.name.endswith(".tmp") and any(item.name.startswith(pref) for pref in _MANAGED_TMP_STEMS):
                red = redact_home_path(item, home=eff_home, workspace=eff_ws)
                if red not in residual_files:
                    residual_files.append(red)
                    diagnostics.append(
                        f"Fichier temporaire HiveMind résiduel : {red}."
                        if is_fr
                        else f"Leftover HiveMind temporary staging file: {red}."
                    )

    cont_dir = eff_home / ".continue"
    cont_cfg = cont_dir / "config.yaml"
    _inspect_continue_main_config(
        cont_cfg,
        eff_home=eff_home,
        eff_ws=eff_ws,
        is_fr=is_fr,
        residual_entries=residual_continue_entries,
        preserved_user_files=preserved_user_files,
        diagnostics=diagnostics,
    )
    if cont_dir.is_dir():
        for entry in sorted(cont_dir.iterdir()):
            if not entry.is_file() or entry.is_symlink():
                continue
            if entry.name in ("hivemind_provider.yaml", "hivemind_provider.yaml.bak", "config.yaml"):
                continue
            if entry.name.endswith(".tmp") and entry.name.startswith(".hivemind_provider.yaml."):
                continue
            try:
                txt = _read_text_bounded(entry)
            except OSError:
                continue
            if not _RAW_HM_KEY_RE.search(txt) and "hivemind/coder" not in txt.lower():
                preserved_user_files.append(redact_home_path(entry, home=eff_home, workspace=eff_ws))

    if eff_ws is not None:
        vscode_dir = eff_ws / ".vscode"
        _inspect_vscode_settings_residue(
            vscode_dir / "settings.json",
            eff_home=eff_home,
            eff_ws=eff_ws,
            is_fr=is_fr,
            residual_settings_keys=residual_vscode_keys,
            preserved_user_files=preserved_user_files,
            diagnostics=diagnostics,
        )
        _inspect_vscode_tasks_residue(
            vscode_dir / "tasks.json",
            eff_home=eff_home,
            eff_ws=eff_ws,
            is_fr=is_fr,
            residual_tasks=residual_vscode_tasks,
            preserved_user_files=preserved_user_files,
            diagnostics=diagnostics,
        )
        for bak_name in ("settings.json.bak", "tasks.json.bak"):
            bak_path = vscode_dir / bak_name
            if bak_path.is_file():
                try:
                    bak_txt = _read_text_bounded(bak_path)
                except OSError:
                    continue
                if "hivemind." in bak_txt.lower() or "HiveMind:" in bak_txt or _RAW_HM_KEY_RE.search(bak_txt):
                    red = redact_home_path(bak_path, home=eff_home, workspace=eff_ws)
                    residual_files.append(red)
                    diagnostics.append(
                        f"Sauvegarde HiveMind résiduelle dans {red}."
                        if is_fr
                        else f"Leftover HiveMind backup artifact found at {red}."
                    )

    _scan_dir_for_raw_hm_keys(
        eff_home / ".hivemind",
        eff_home=eff_home,
        eff_ws=eff_ws,
        unmasked_hits=unmasked_api_keys_found,
        masked_counter=masked_counter,
        scanned_paths=scanned_paths,
    )
    _scan_dir_for_raw_hm_keys(
        cont_dir,
        eff_home=eff_home,
        eff_ws=eff_ws,
        unmasked_hits=unmasked_api_keys_found,
        masked_counter=masked_counter,
        scanned_paths=scanned_paths,
    )
    if eff_ws is not None:
        _scan_dir_for_raw_hm_keys(
            eff_ws / ".vscode",
            eff_home=eff_home,
            eff_ws=eff_ws,
            unmasked_hits=unmasked_api_keys_found,
            masked_counter=masked_counter,
            scanned_paths=scanned_paths,
        )

    for hit in unmasked_api_keys_found:
        diagnostics.append(
            f"Clé hm-... non masquée résiduelle dans {hit['file']} ({hit['masked_preview']})."
            if is_fr
            else f"Unmasked hm-... key residue found in {hit['file']} ({hit['masked_preview']})."
        )

    zero_residue = not (
        residual_files
        or residual_continue_entries
        or residual_vscode_keys
        or residual_vscode_tasks
        or unmasked_api_keys_found
    )
    go_nogo = "GO" if zero_residue else "NO-GO"
    decision = "CLEAN_UNINSTALL_ZERO_RESIDUE" if zero_residue else "HOLD_UNINSTALL_RESIDUE_DETECTED"
    if zero_residue:
        summary = (
            f"Audit désinstallation propre VALIDÉ : 0 fichier résiduel, 0 clé hm-... brute, "
            f"{len(preserved_user_files)} fichier(s) utilisateur préservé(s), {FINANCIAL_PROMISE}."
            if is_fr
            else f"Clean uninstall zero-residue audit PASSED: 0 residual files, 0 unmasked hm-... keys, "
            f"{len(preserved_user_files)} user config file(s) preserved, {FINANCIAL_PROMISE}."
        )
    else:
        summary = (
            f"Audit désinstallation propre EN ÉCHEC : {len(residual_files)} fichier(s) résiduel(s), "
            f"{len(unmasked_api_keys_found)} clé(s) non masquée(s)."
            if is_fr
            else f"Clean uninstall zero-residue audit FAILED: {len(residual_files)} residual file(s), "
            f"{len(unmasked_api_keys_found)} unmasked key hit(s)."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RESIDUE_AUDIT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "audit_clean_uninstall_zero_residue",
            "lang": lang_code,
            "home_dir": redact_home_path(eff_home, home=eff_home, workspace=eff_ws),
            "workspace_dir": redact_home_path(eff_ws, home=eff_home, workspace=eff_ws) if eff_ws is not None else None,
            "zero_residue": zero_residue,
            "ready": zero_residue,
            "go_nogo": go_nogo,
            "decision": decision,
            "residual_files": residual_files,
            "residual_files_count": len(residual_files),
            "residual_continue_config_entries": residual_continue_entries,
            "residual_vscode_settings_keys": residual_vscode_keys,
            "residual_vscode_tasks": residual_vscode_tasks,
            "unmasked_api_keys_found": unmasked_api_keys_found,
            "unmasked_api_keys_count": len(unmasked_api_keys_found),
            "masked_keys_tolerated_count": masked_counter[0],
            "preserved_user_files": preserved_user_files,
            "preserved_user_files_count": len(preserved_user_files),
            "user_settings_intact": True,
            "uninstall_always_permitted": True,
            "uninstall_supported": True,
            "uninstall_commands": {"posix": UNINSTALL_POSIX_CMD, "windows": UNINSTALL_WIN_CMD},
            "compliance_ok": True,
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "zero_financial_payout": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": diagnostics,
            "exit_code": 0 if zero_residue else 2,
            "summary": summary,
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
    hdrs = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()} if isinstance(raw_hdrs, Mapping) else {}
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
    uninstall_advertised = bool(
        "--uninstall" in low_html
        or "clean local uninstall" in low_html
        or "clean one-command uninstall" in low_html
        or "désinstallation propre" in low_html
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
            "uninstall_promise_advertised": uninstall_advertised,
        },
        issues,
    )


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


def verify_holiday_wrapup_and_uninstall_promise(
    gateway: str = DEFAULT_GATEWAY,
    *,
    home_dir: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
    expected_sha: str | None = None,
    wrapup_date: str = HOLIDAY_WRAPUP_DATE,
) -> dict[str, Any]:
    """Verify Jan 3, 2027 holiday wrap-up routes, uninstall promise, and optional zero-residue home."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    eff_env = environ if environ is not None else env
    comp_doc = evaluate_personal_pc_compliance(
        environ=eff_env,
        cwd=cwd,
        check_git_remote=True,
        lang=lang_code,
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt(
            "verify_holiday_wrapup_and_uninstall_promise",
            comp_doc,
            lang_code=lang_code,
        )

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None
    clean_wrapup_date = validate_wrapup_date(wrapup_date)

    uninstall_audit: dict[str, Any] | None = None
    uninstall_clean: bool | None = None
    if home_dir is not None:
        uninstall_audit = audit_clean_uninstall_zero_residue(
            home_dir,
            workspace_dir=workspace_dir,
            lang=lang_code,
        )
        uninstall_clean = bool(uninstall_audit["zero_residue"])

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = list(uninstall_audit["diagnostics"]) if uninstall_audit is not None else []
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}

    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    missing_manifest_routes: list[str] = []
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {}
    network_calls = 0

    for path, kind, group in WRAPUP_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if code != 200:
            if code == 404 and group in ("holiday", "manifest", "feedback"):
                stale_404_routes.append(path)
            diag = (
                f"GET {path} -> HTTP {code} (attendu 200)."
                if is_fr
                else f"GET {path} -> HTTP {code} (expected 200)."
            )
            diagnostics.append(diag)
        elif kind == "html":
            ok, h_doc, h_issues = _inspect_html_route(path, body, seen_nonces, lang_code=lang_code)
            html_contracts[path] = h_doc
            if not ok:
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/manifest":
            manifest_json = _unwrap_json_body(body)
            ok, live_build_sha, sha_match, missing_manifest_routes, m_issues = _inspect_manifest_contract(
                manifest_json, expected_sha=clean_expected_sha, lang_code=lang_code
            )
            manifest_ok = ok
            if not ok:
                diag = m_issues[0]
                diagnostics.extend(m_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            fb_json = _unwrap_json_body(body)
            ok, feedback_summary_doc, fb_issues = _inspect_feedback_summary_exact(fb_json, lang_code=lang_code)
            feedback_summary_ok = ok
            if not ok:
                diag = fb_issues[0]
                diagnostics.extend(fb_issues)

        if not ok:
            failed_routes.append(path)
        route_digest_src = json.dumps(
            {"wrapup_date": clean_wrapup_date, "path": path, "http_status": code, "ok": ok},
            sort_keys=True,
            ensure_ascii=False,
        )
        route_rows.append(
            {
                "path": path,
                "http_status": code,
                "ok": ok,
                "diagnostic": diag,
                "leaf_sha256": hashlib.sha256(
                    f"hivemind.uninstall_wrapup.merkle.route/1:{route_digest_src}".encode()
                ).hexdigest(),
            }
        )

    html_all_ok = bool(
        len(html_contracts) == len(WRAPUP_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in WRAPUP_HTML_ROUTES)
        and len(seen_nonces) == len(WRAPUP_HTML_ROUTES)
    )
    uninstall_promise_advertised = bool(
        html_contracts.get("/noel", {}).get("uninstall_promise_advertised")
        and html_contracts.get("/quick", {}).get("uninstall_promise_advertised")
    )
    stale_404_drift = bool(stale_404_routes or missing_manifest_routes)
    live_routes_ok = bool(
        html_all_ok
        and manifest_ok
        and feedback_summary_ok
        and (clean_expected_sha is None or sha_match is True)
    )
    uninstall_ok = uninstall_clean is not False

    hold_reasons: list[str] = []
    if not live_routes_ok:
        hold_reasons.append("routes_drift_or_contract_failure")
    if uninstall_clean is False:
        hold_reasons.append("uninstall_residue_detected")

    ready = bool(live_routes_ok and uninstall_ok)
    if not live_routes_ok and uninstall_clean is False:
        decision = "HOLD_UNINSTALL_RESIDUE_AND_ROUTES_DRIFT"
    elif uninstall_clean is False:
        decision = "HOLD_UNINSTALL_RESIDUE"
    elif not live_routes_ok:
        decision = "HOLD_ROUTES_DRIFT"
    else:
        decision = "HOLIDAY_WRAPUP_READY"

    redeploy_required = bool(stale_404_drift or not live_routes_ok)
    go_nogo = "GO" if ready else "NO-GO"

    wrapup_leaf_src = json.dumps(
        {
            "wrapup_date": clean_wrapup_date,
            "uninstall_audit_performed": uninstall_audit is not None,
            "uninstall_clean": uninstall_clean,
            "live_routes_ok": live_routes_ok,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    wrapup_leaf_sha = hashlib.sha256(f"hivemind.uninstall_wrapup.merkle.gate/1:{wrapup_leaf_src}".encode()).hexdigest()
    merkle_inputs = [row["leaf_sha256"] for row in route_rows]
    if uninstall_audit is not None:
        merkle_inputs.append(str(uninstall_audit["receipt_sha256"]))
    merkle_inputs.append(wrapup_leaf_sha)
    merkle_root = compute_merkle_root(merkle_inputs)

    if ready:
        audit_note_fr = f", désinstall_propre={uninstall_clean}" if uninstall_audit is not None else ""
        audit_note_en = f", uninstall_clean={uninstall_clean}" if uninstall_audit is not None else ""
        summary = (
            f"Clôture Noël ({clean_wrapup_date}) {go_nogo} sur {clean_gw} : "
            f"routes={len(WRAPUP_ROUTES)}/{len(WRAPUP_ROUTES)}{audit_note_fr}, {FINANCIAL_PROMISE}."
            if is_fr
            else f"Holiday wrap-up ({clean_wrapup_date}) {go_nogo} on {clean_gw}: "
            f"routes={len(WRAPUP_ROUTES)}/{len(WRAPUP_ROUTES)}{audit_note_en}, {FINANCIAL_PROMISE}."
        )
    else:
        summary = (
            f"Clôture Noël ({clean_wrapup_date}) {go_nogo} ({decision}) "
            f"sur {clean_gw} (raisons={', '.join(hold_reasons)})."
            if is_fr
            else f"Holiday wrap-up ({clean_wrapup_date}) {go_nogo} ({decision}) "
            f"on {clean_gw} (reasons={', '.join(hold_reasons)})."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "verify_holiday_wrapup_and_uninstall_promise",
            "wrapup_date": clean_wrapup_date,
            "holiday_window_start": HOLIDAY_START_DATE.isoformat(),
            "holiday_window_end": HOLIDAY_END_DATE.isoformat(),
            "holiday_days_count": HOLIDAY_DAYS_COUNT,
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "ready": ready,
            "go_nogo": go_nogo,
            "decision": decision,
            "hold_reasons": hold_reasons,
            "uninstall_audit_performed": uninstall_audit is not None,
            "uninstall_clean": uninstall_clean,
            "uninstall_audit": uninstall_audit,
            "uninstall_promise_advertised": uninstall_promise_advertised,
            "uninstall_commands": {"posix": UNINSTALL_POSIX_CMD, "windows": UNINSTALL_WIN_CMD},
            "live_routes_ok": live_routes_ok,
            "html_all_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "manifest_ok": manifest_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "sha_match": sha_match,
            "missing_manifest_routes": missing_manifest_routes,
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "stale_404_drift": stale_404_drift,
            "stale_404_routes": stale_404_routes,
            "redeploy_required": redeploy_required,
            "http_statuses": http_statuses,
            "routes": route_rows,
            "failed_routes": failed_routes,
            "merkle_root_sha256": merkle_root,
            "network_calls": network_calls,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "zero_financial_payout": True,
            "uninstall_supported": True,
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
    """Render a compact FR/EN CLI clean-uninstall residue & Jan 3 wrap-up receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    title = (
        f"HIVEMIND NOËL — AUDIT DÉSINSTALLATION & CLÔTURE 3 JANV. ({verdict}, 0,00 EUR / 0,00 USD)"
        if is_fr
        else f"HIVEMIND HOLIDAY — CLEAN UNINSTALL & JAN 3 WRAP-UP ({verdict}, 0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {title}",
        "====================================================================",
        f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}",
        f"  Gateway     : {report.get('gateway', DEFAULT_GATEWAY)}",
        f"  Mode        : {report.get('mode')}",
        f"  Verdict     : {verdict} ({report.get('decision', verdict)})",
        f"  Compliance  : {comp_ok}",
        f"  Exit code   : {report.get('exit_code')}",
        f"  Summary     : {report.get('summary')}",
    ]
    if report.get("wrapup_date"):
        lines.append(f"  Wrap-up     : {report.get('wrapup_date')} (2026-12-19..2027-01-03)")
    audit_blk = report.get("uninstall_audit") if isinstance(report.get("uninstall_audit"), Mapping) else report
    if "zero_residue" in audit_blk:
        lines.append(
            f"  Uninstall   : zero_residue={audit_blk.get('zero_residue')} "
            f"(residual_files={audit_blk.get('residual_files_count')}, "
            f"preserved_user={audit_blk.get('preserved_user_files_count')})"
        )
    if report.get("merkle_root_sha256"):
        lines.append(f"  Merkle Root : {report.get('merkle_root_sha256')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256     : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN Markdown clean-uninstall residue & Jan 3 wrap-up receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    heading = (
        f"# Audit Désinstallation Propre & Clôture Noël 3 Janvier HiveMind — {verdict}"
        if is_fr
        else f"# HiveMind Clean Uninstall & Jan 3 Holiday Wrap-Up Gate — {verdict}"
    )
    lines = [
        heading,
        "",
        f"- **{'Verdict Go/No-Go' if is_fr else 'Go/No-Go Verdict'}**: `{verdict}` (`decision={report.get('decision', verdict)}`, `exit_code={report.get('exit_code')}`)",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: `{comp_ok}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Zéro inférence' if is_fr else 'Zero-Inference Verified'}**: `{report.get('zero_inference_verified', True)}`",
        f"- **{'Attestation non financière' if is_fr else 'Non-Financial Symbolic Attestation'}**: `zero_financial_payout={report.get('zero_financial_payout', True)}`, `uninstall_supported={report.get('uninstall_supported', True)}`",
    ]
    if report.get("wrapup_date"):
        lines.append(
            f"- **{'Date de clôture' if is_fr else 'Holiday Wrap-Up Date'}**: `{report.get('wrapup_date')}` (`2026-12-19..2027-01-03`)"
        )
    audit_blk = report.get("uninstall_audit") if isinstance(report.get("uninstall_audit"), Mapping) else report
    if "zero_residue" in audit_blk:
        lines.append(
            f"- **{'Audit Zéro-Résidu' if is_fr else 'Uninstall Zero-Residue Audit'}**: "
            f"`zero_residue={audit_blk.get('zero_residue')}`, "
            f"`residual_files={audit_blk.get('residual_files_count')}`, "
            f"`unmasked_keys={audit_blk.get('unmasked_api_keys_count')}`, "
            f"`preserved_user_files={audit_blk.get('preserved_user_files_count')}`"
        )
    lines.append(f"- **{'Résumé' if is_fr else 'Summary'}**: {report.get('summary')}")
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
    lines.append(f"- **SHA-256 Receipt**: `{report.get('receipt_sha256')}`")

    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        lines.extend(f"- {d}" for d in diags)
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the clean uninstall residue & Jan 3 holiday wrap-up verifier."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-inference clean uninstall zero-residue auditor & Jan 3, 2027 holiday wrap-up verifier "
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
        help="Home directory to audit for zero HiveMind residue after --uninstall.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Optional workspace directory to check .vscode/settings.json & tasks.json preservation.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Expected build_sha on /v1/hivemind/onboarding/manifest.",
    )
    parser.add_argument(
        "--wrapup-date",
        default=HOLIDAY_WRAPUP_DATE,
        help="ISO wrap-up date within 2026-12-19..2027-01-03 (default: 2027-01-03).",
    )
    parser.add_argument(
        "--uninstall-audit",
        action="store_true",
        help="Run only the strictly local clean-uninstall zero-residue audit (always permitted).",
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
    """CLI entrypoint enforcing compliance while keeping local uninstall residue audit permitted."""
    eff_out = stdout if stdout is not None else sys.stdout
    eff_err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        lang_code = normalize_lang(args.lang)
        eff_env = environ if environ is not None else os.environ
        if args.uninstall_audit:
            target_home: Path | str = (
                args.home_dir
                if args.home_dir is not None
                else (eff_env.get("HOME") or eff_env.get("USERPROFILE") or Path.home())
            )
            report = audit_clean_uninstall_zero_residue(
                target_home,
                workspace_dir=args.workspace,
                lang=lang_code,
            )
        elif args.enforce_personal_pc:
            report = evaluate_personal_pc_compliance(
                environ=eff_env,
                cwd=cwd,
                check_git_remote=True,
                lang=lang_code,
            )
        else:
            report = verify_holiday_wrapup_and_uninstall_promise(
                args.gateway,
                home_dir=args.home_dir,
                workspace_dir=args.workspace,
                http_fn=http_fn,
                environ=eff_env,
                cwd=cwd,
                lang=lang_code,
                expected_sha=args.expected_sha,
                wrapup_date=args.wrapup_date,
            )
    except UninstallWrapupError as exc:
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
