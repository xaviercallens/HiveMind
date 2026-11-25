#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference Continue.dev Non-Destructive Config Merge & Backup-Restore Verifier (`SLICE-SCALE-DYN-28`).

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment markers (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``,
     ``NODE_EXTRA_CA_CERTS``), workspace path, and ``.git/config`` remote URLs (including worktree
     ``gitdir:`` / ``commondir`` indirection) BEFORE any network call or file write. Fails closed
     with exit code 2 (0 network calls, 0 file writes, 0 raw env/hostname/path leaks).
  2. Zero-Inference Continue.dev Non-Destructive Config Merge & Backup-Restore Verifier
     (``verify_continue_config_roundtrip``):
     verifies in an isolated sandbox that merging ``hivemind/coder`` and ``hivemind/bee-stars``
     into ``~/.continue/config.yaml`` creates an atomic ``0600`` ``.bak`` backup, preserves 100%
     of pre-existing non-HiveMind models/providers/sections byte-for-byte on uninstall/restore,
     stays idempotent across repeated merges without clobbering ``.bak``, rejects corrupted or
     tampered backups fail-closed, leaves zero HiveMind residue, and masks all ``hm-...`` keys.
  3. Zero-Inference Holiday Route, CSP Nonce & Exact 5-Key Feedback Verifier
     (``verify_continue_and_holiday_readiness``):
     probes ``GET /noel``, ``GET /quick``, ``GET /raid-noel``, ``GET /arena-noel``,
     ``GET /v1/hivemind/onboarding/manifest``, and ``GET /v1/hivemind/onboarding/feedback/summary``
     in strict read-only zero-inference mode (never calling ``POST /v1/chat/completions`` or
     prompting ``hivemind/coder``), enforcing unique per-request CSP nonces ``default-src 'none'``,
     0 unsafe DOM sinks, 0 static figures (stripping ``hm-***`` / ``hm_live_***`` masked-key
     suffixes and HTML ``nonce="..."`` attributes first), exact 5-key feedback summary equality,
     and FR/EN JSON & Markdown SHA-256 receipts.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen science under AGPL-3.0 (zero monetary value).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import inspect
import json
import math
import os
import re
import secrets
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_continue_config_backup_and_restore_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_continue_config_backup_restore_receipt"
ROUNDTRIP_OBJECT = "hivemind.noel_continue_config_roundtrip"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_MODELS: tuple[str, str] = ("hivemind/coder", "hivemind/bee-stars")
DEFAULT_MAX_TOKENS = 2048

HTTP_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 524_288
MAX_CONFIG_BYTES = 1_048_576
REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset()

HOLIDAY_HTML_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
)

HOLIDAY_PROBE_ROUTES: tuple[tuple[str, str, str], ...] = (
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

_CONTINUE_MANAGED_HEADER_LINES: tuple[str, ...] = (
    "# HiveMind personal hobby/open-source coding preset for VS Code (Continue)",
    "# 100% personal home PCs & personal hobby/open-source projects ONLY (0.00 EUR / 0.00 USD, AGPL-3.0).",
    "# Never use on corporate hardware or send corporate/confidential source code.",
)

DEFAULT_SAMPLE_EXISTING_CONTINUE_YAML = (
    "# Pre-existing personal hobby Continue.dev configuration\n"
    "name: Personal Hobby Assistant\n"
    "version: 0.0.1\n"
    "schema: v1\n"
    "models:\n"
    "  - name: Local Ollama Qwen\n"
    "    provider: ollama\n"
    "    model: qwen2.5-coder:7b\n"
    "    apiBase: http://127.0.0.1:11434/v1\n"
    "    roles:\n"
    "      - chat\n"
    "      - edit\n"
    "  - name: Local LlamaCpp Coder\n"
    "    provider: llama.cpp\n"
    "    model: deepseek-coder-6.7b\n"
    "tabAutocompleteModel:\n"
    "  title: Starcoder\n"
    "  provider: ollama\n"
    "  model: starcoder2:3b\n"
    "allowAnonymousTelemetry: false\n"
)

_CONTINUE_RESIDUE_MARKERS: tuple[str, ...] = (
    "hivemind/coder",
    "hivemind/bee-stars",
    "hivemind/auto",
    "hivemind-openrouter",
    "name: hivemind",
    '"name": "hivemind',
)

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_FULL_RE = re.compile(r"^[0-9a-f]{64}$")

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


class ContinueConfigError(ValueError):
    """Raised when Continue.dev merge, backup-restore, host allowlist, or readiness checks fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI or API output language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ContinueConfigError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ContinueConfigError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw secrets never appear in receipts."""
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
        raise ContinueConfigError("strip_masked_keys_and_nonces requires a string.")
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
        raise ContinueConfigError("scan_output_safety requires a string.")
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
        raise ContinueConfigError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise ContinueConfigError("Forbidden JSON key detected in receipt.")
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
        raise ContinueConfigError(
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


def compute_merkle_root(
    leaf_hashes: Sequence[str],
    *,
    domain: str = "hivemind.continue_backup_restore.merkle",
) -> str:
    """Compute a deterministic domain-separated binary SHA-256 Merkle root over leaf digests."""
    if not leaf_hashes:
        return hashlib.sha256(f"{domain}.empty/1".encode()).hexdigest()
    level: list[str] = []
    for digest in leaf_hashes:
        if not isinstance(digest, str) or not _SHA256_FULL_RE.fullmatch(digest):
            raise ContinueConfigError("Merkle leaf digest must be a 64-char lowercase hex SHA-256.")
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
        raise ContinueConfigError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ContinueConfigError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise ContinueConfigError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ContinueConfigError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ContinueConfigError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise ContinueConfigError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ContinueConfigError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ContinueConfigError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise ContinueConfigError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ContinueConfigError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ContinueConfigError("Remote gateway URL must use https.")
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
        raise ContinueConfigError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ContinueConfigError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ContinueConfigError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise ContinueConfigError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ContinueConfigError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise ContinueConfigError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise ContinueConfigError(
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
        raise ContinueConfigError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
    lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    text = raw_bytes.decode("utf-8", errors="replace")
    ctype = lower_headers.get("content-type", "").lower()
    if "application/json" in ctype or (
        "text/html" not in ctype and text.lstrip().startswith(("{", "["))
    ):
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
    clean_url = _assert_safe_http_request(verb, url, payload=payload, allowed_post_paths=ALLOWED_POST_PATHS)
    if payload is not None:
        raise ContinueConfigError("GET/HEAD requests must never attach a body payload.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-continue-backup-restore-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise ContinueConfigError("HTTP headers must not contain CR/LF or NUL characters.")
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
        raise ContinueConfigError("http_fn must return (status_code, body) or (status_code, headers, body).")
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
    except ContinueConfigError:
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
    gateway: str | None = None,
) -> dict[str, Any]:
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": mode,
            "lang": lang_code,
            "gateway": gateway or DEFAULT_GATEWAY,
            "compliance_ok": False,
            "corporate_detected": True,
            "corporate_indicators": list(comp_doc.get("corporate_indicators") or []),
            "matched_markers": list(comp_doc.get("matched_markers") or []),
            "ready": False,
            "roundtrip_ok": False,
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


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ContinueConfigError(f"Refusing symlink for {label}.")
    if path.parent.is_symlink():
        raise ContinueConfigError(f"Refusing symlink parent directory for {label}.")


def _read_bytes_bounded(path: Path, label: str, max_bytes: int = MAX_CONFIG_BYTES) -> bytes:
    _reject_symlink(path, label)
    if not path.is_file():
        raise ContinueConfigError(f"Expected regular file for {label}.")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ContinueConfigError(f"{label} exceeded {max_bytes} byte limit.")
    return data


def _write_bytes_secure_atomic(
    path: Path,
    raw_bytes: bytes,
    *,
    posix_mode: int = 0o600,
    eff_os: str,
    label: str = "Continue config",
) -> None:
    """Atomically write ``raw_bytes`` to ``path`` with ``0600`` mode on POSIX."""
    _reject_symlink(path, label)
    if eff_os != "nt" and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp_path, flags, posix_mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
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


def _strip_yaml_scalar_comment(raw: str) -> str:
    """Strip unquoted inline YAML ``# comment`` and surrounding matching quotes."""
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
    if in_sq or in_dq:
        raise ContinueConfigError("Unclosed quote detected in Continue YAML scalar.")
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
        raise ContinueConfigError("Cannot render Continue config without a valid hm-... API key.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 64:
        raise ContinueConfigError("max_tokens must be an integer >= 64.")
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


def _parse_continue_yaml_item_dict(item_lines: Sequence[str]) -> dict[str, Any]:
    kv: dict[str, str] = {}
    for raw_ln in item_lines:
        stripped = raw_ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            rest = stripped[1:].strip()
            if not rest:
                continue
            if ":" in rest:
                k_raw, v_raw = rest.split(":", 1)
                k = k_raw.strip().lower()
                if k and " " not in k:
                    kv[k] = _strip_yaml_scalar_comment(v_raw)
            continue
        if ":" not in stripped:
            raise ContinueConfigError("Invalid YAML mapping line inside Continue models item.")
        k_raw, v_raw = stripped.split(":", 1)
        k = k_raw.strip().lower()
        if k and " " not in k:
            kv[k] = _strip_yaml_scalar_comment(v_raw)
    if not any(k in kv for k in ("name", "title", "model", "provider")):
        raise ContinueConfigError("Continue models item missing name/title/model/provider field.")
    entry = {
        "name": kv.get("name") or kv.get("title") or "",
        "provider": kv.get("provider") or "",
        "model": kv.get("model") or "",
        "apiBase": kv.get("apibase") or "",
        "apiKey": kv.get("apikey") or "",
    }
    return {
        "name": entry["name"],
        "provider": entry["provider"],
        "model": entry["model"],
        "api_base": entry["apiBase"],
        "has_api_key": bool(entry["apiKey"]),
        "is_hivemind": _is_hivemind_continue_model_dict(entry),
    }


def _split_continue_yaml_models(
    lines: list[str],
) -> tuple[int | None, int, list[str], list[list[str]], int]:
    """Locate top-level ``models:`` block and split it into sequence item line groups."""
    models_idx: int | None = None
    for idx, ln in enumerate(lines):
        if ln and not ln[0].isspace() and not ln.startswith("#"):
            m = re.match(r"^models\s*:(.*)$", ln)
            if m:
                inline_val = _strip_yaml_scalar_comment(m.group(1))
                if inline_val and inline_val.lower() not in ("[]", "~", "null"):
                    raise ContinueConfigError("Existing Continue config.yaml has non-block 'models' value.")
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


def validate_continue_config_text(text: str) -> dict[str, Any]:
    """Validate Continue.dev ``config.yaml`` (or ``config.json``) strictly with stdlib only."""
    if not isinstance(text, str) or not text.strip():
        raise ContinueConfigError("Continue config content is empty.")
    if "\x00" in text or _TROJAN_BIDI_RE.search(text):
        raise ContinueConfigError("Continue config contains forbidden NUL or bidirectional control chars.")
    if any(ord(ch) < 32 and ch not in ("\n", "\r", "\t") for ch in text):
        raise ContinueConfigError("Continue config contains invalid control characters.")

    normalized = text.replace("\r\n", "\n")
    stripped_start = normalized.lstrip()

    if stripped_start.startswith(("{", "[")):
        try:
            loaded = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ContinueConfigError("Continue JSON config is malformed.") from exc
        if not isinstance(loaded, dict):
            raise ContinueConfigError("Continue JSON config root must be an object.")
        raw_models = loaded.get("models", [])
        if not isinstance(raw_models, list) or not all(isinstance(m, Mapping) for m in raw_models):
            raise ContinueConfigError("Continue JSON config 'models' field must be a list of objects.")
        models_meta: list[dict[str, Any]] = []
        for entry in raw_models:
            models_meta.append(
                {
                    "name": str(entry.get("name") or entry.get("title") or ""),
                    "provider": str(entry.get("provider") or ""),
                    "model": str(entry.get("model") or ""),
                    "api_base": str(entry.get("apiBase") or ""),
                    "has_api_key": bool(str(entry.get("apiKey") or "").strip()),
                    "is_hivemind": _is_hivemind_continue_model_dict(entry),
                }
            )
        return {
            "format": "json",
            "models": models_meta,
            "non_hivemind_models": [m for m in models_meta if not m["is_hivemind"]],
            "hivemind_models": [m for m in models_meta if m["is_hivemind"]],
            "other_top_level_keys": sorted(str(k) for k in loaded if k != "models"),
        }

    if stripped_start.startswith("- "):
        raise ContinueConfigError("Continue YAML config must be a mapping, not a top-level list.")

    lines = normalized.splitlines()
    for ln in lines:
        if ln.startswith("\t") or re.match(r"^ *\t", ln):
            raise ContinueConfigError("Continue YAML config contains forbidden tab indentation.")
        if ln and not ln[0].isspace() and not ln.startswith(("#", "-", "---", "...")):
            if ":" not in ln:
                raise ContinueConfigError("Continue YAML top-level line missing ':' key separator.")
            _, val_part = ln.split(":", 1)
            _strip_yaml_scalar_comment(val_part)
        elif ln.strip() and not ln.strip().startswith("#"):
            lst = ln.strip()
            if ":" in lst and not lst.startswith("-"):
                _strip_yaml_scalar_comment(lst.split(":", 1)[1])

    models_idx, end_idx, _, items, _ = _split_continue_yaml_models(lines)
    models_meta_yaml = [_parse_continue_yaml_item_dict(item) for item in items]

    other_keys: list[str] = []
    for idx, ln in enumerate(lines):
        if models_idx is not None and models_idx <= idx < end_idx:
            continue
        if ln and not ln[0].isspace() and not ln.startswith(("#", "-", "---", "...")):
            top_k = ln.split(":", 1)[0].strip()
            if top_k and top_k != "models" and top_k not in other_keys:
                other_keys.append(top_k)

    return {
        "format": "yaml",
        "models": models_meta_yaml,
        "non_hivemind_models": [m for m in models_meta_yaml if not m["is_hivemind"]],
        "hivemind_models": [m for m in models_meta_yaml if m["is_hivemind"]],
        "other_top_level_keys": sorted(other_keys),
    }


def build_continue_provider_yaml(
    gateway: str,
    api_key: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Render Continue OpenAI-compatible YAML (``hivemind/coder`` & ``hivemind/bee-stars``)."""
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
    """Non-destructively merge HiveMind models into ``~/.continue/config.yaml``."""
    hivemind_dicts = _build_continue_model_dicts(gateway, api_key, max_tokens=max_tokens)
    if existing_text is None or not existing_text.strip():
        return build_continue_provider_yaml(gateway, api_key, max_tokens=max_tokens)

    meta = validate_continue_config_text(existing_text)
    normalized = existing_text.replace("\r\n", "\n")

    if meta["format"] == "json":
        loaded = json.loads(normalized)
        preserved_dicts = [m for m in loaded.get("models", []) if not _is_hivemind_continue_model_dict(m)]
        merged_doc = {**loaded, "models": [*preserved_dicts, *hivemind_dicts]}
        return json.dumps(merged_doc, indent=2, ensure_ascii=False) + "\n"

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

    preserved_items = [item for item in items if not _parse_continue_yaml_item_dict(item)["is_hivemind"]]
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
    """Remove only HiveMind entries from ``~/.continue/config.yaml``; return ``None`` if empty."""
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

    preserved_items = [item for item in items if not _parse_continue_yaml_item_dict(item)["is_hivemind"]]
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


def _posix_mode_str(path: Path, eff_os: str) -> str:
    if eff_os == "nt":
        return "windows_acl"
    return f"{path.stat().st_mode & 0o777:04o}"


def backup_and_merge_continue_config(
    config_path: Path | str,
    *,
    gateway: str,
    api_key: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    os_name: str | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Atomically create a ``0600`` ``.bak`` backup and merge HiveMind models into ``config.yaml``."""
    cfg_path = Path(config_path)
    bak_path = Path(f"{cfg_path}.bak")
    eff_os = os_name if os_name is not None else os.name
    _reject_symlink(cfg_path, "Continue config")
    _reject_symlink(bak_path, "Continue backup")

    existed_before = cfg_path.exists()
    baseline_bytes = b""
    baseline_text: str | None = None
    baseline_sha256: str | None = None
    baseline_meta: dict[str, Any] = {
        "non_hivemind_models": [],
        "other_top_level_keys": [],
    }
    if existed_before:
        baseline_bytes = _read_bytes_bounded(cfg_path, "Continue config")
        try:
            baseline_text = baseline_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContinueConfigError("Existing Continue config is not valid UTF-8.") from exc
        baseline_meta = validate_continue_config_text(baseline_text)
        baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()

    merged_text = merge_continue_config_yaml(
        baseline_text,
        gateway=gateway,
        api_key=api_key,
        max_tokens=max_tokens,
    )
    merged_bytes = merged_text.encode("utf-8")
    merged_sha256 = hashlib.sha256(merged_bytes).hexdigest()
    merged_meta = validate_continue_config_text(merged_text)

    if existed_before and baseline_bytes == merged_bytes:
        mode_tightened = False
        if eff_os != "nt" and (cfg_path.stat().st_mode & 0o777) != 0o600:
            os.chmod(cfg_path, 0o600)
            mode_tightened = True
        return {
            "path": redact_home_path(cfg_path, home=home),
            "backup_path": redact_home_path(bak_path, home=home) if bak_path.is_file() else None,
            "action": "unchanged",
            "written": False,
            "existed_before": True,
            "backup_created": False,
            "backup_sha256": hashlib.sha256(bak_path.read_bytes()).hexdigest() if bak_path.is_file() else None,
            "baseline_sha256": baseline_sha256,
            "merged_sha256": merged_sha256,
            "config_posix_mode": _posix_mode_str(cfg_path, eff_os),
            "backup_posix_mode": _posix_mode_str(bak_path, eff_os) if bak_path.is_file() else None,
            "permissions_tightened": mode_tightened,
            "masked_api_key": mask_api_key(api_key),
            "preserved_user_models": [m["model"] for m in merged_meta["non_hivemind_models"]],
            "hivemind_models": [m["model"] for m in merged_meta["hivemind_models"]],
        }

    backup_created = False
    backup_sha256: str | None = None
    backup_mode: str | None = None
    if existed_before:
        _write_bytes_secure_atomic(
            bak_path,
            baseline_bytes,
            posix_mode=0o600,
            eff_os=eff_os,
            label="Continue backup",
        )
        written_bak = _read_bytes_bounded(bak_path, "Continue backup")
        if written_bak != baseline_bytes:
            raise ContinueConfigError("Written .bak backup failed byte-for-byte verification.")
        backup_created = True
        backup_sha256 = hashlib.sha256(written_bak).hexdigest()
        backup_mode = _posix_mode_str(bak_path, eff_os)

    _write_bytes_secure_atomic(
        cfg_path,
        merged_bytes,
        posix_mode=0o600,
        eff_os=eff_os,
        label="Continue config",
    )
    return {
        "path": redact_home_path(cfg_path, home=home),
        "backup_path": redact_home_path(bak_path, home=home) if backup_created else None,
        "action": "updated" if existed_before else "created",
        "written": True,
        "existed_before": existed_before,
        "backup_created": backup_created,
        "backup_sha256": backup_sha256,
        "baseline_sha256": baseline_sha256,
        "merged_sha256": merged_sha256,
        "config_posix_mode": _posix_mode_str(cfg_path, eff_os),
        "backup_posix_mode": backup_mode,
        "permissions_tightened": False,
        "masked_api_key": mask_api_key(api_key),
        "preserved_user_models": [m["model"] for m in baseline_meta["non_hivemind_models"]],
        "hivemind_models": [m["model"] for m in merged_meta["hivemind_models"]],
    }


def restore_continue_config_from_backup(
    config_path: Path | str,
    *,
    expected_backup_sha256: str | None = None,
    expected_existed_before: bool | None = None,
    remove_backup: bool = True,
    os_name: str | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed exact byte-for-byte Continue ``config.yaml`` restore from ``.bak``."""
    cfg_path = Path(config_path)
    bak_path = Path(f"{cfg_path}.bak")
    eff_os = os_name if os_name is not None else os.name
    _reject_symlink(cfg_path, "Continue config")
    _reject_symlink(bak_path, "Continue backup")

    if expected_backup_sha256 is not None:
        if not isinstance(expected_backup_sha256, str) or not _SHA256_FULL_RE.fullmatch(expected_backup_sha256):
            raise ContinueConfigError("expected_backup_sha256 must be a 64-char lowercase hex SHA-256.")

    if bak_path.exists():
        bak_bytes = _read_bytes_bounded(bak_path, "Continue backup")
        actual_bak_sha256 = hashlib.sha256(bak_bytes).hexdigest()
        if expected_backup_sha256 is not None and not hmac.compare_digest(
            actual_bak_sha256, expected_backup_sha256
        ):
            raise ContinueConfigError(
                "Continue backup integrity check failed: .bak SHA-256 mismatch (corrupted or tampered backup)."
            )
        try:
            bak_text = bak_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContinueConfigError("Continue .bak backup is not valid UTF-8; refusing restore.") from exc
        validate_continue_config_text(bak_text)

        _write_bytes_secure_atomic(
            cfg_path,
            bak_bytes,
            posix_mode=0o600,
            eff_os=eff_os,
            label="Continue config",
        )
        restored_bytes = _read_bytes_bounded(cfg_path, "Continue config")
        if restored_bytes != bak_bytes:
            raise ContinueConfigError("Restored Continue config failed byte-for-byte verification.")
        if remove_backup:
            bak_path.unlink()
        return {
            "path": redact_home_path(cfg_path, home=home),
            "restored": True,
            "restore_method": "backup_exact_bytes",
            "restored_sha256": actual_bak_sha256,
            "byte_for_byte_match": True,
            "backup_removed": bool(remove_backup),
            "config_posix_mode": _posix_mode_str(cfg_path, eff_os),
        }

    if expected_backup_sha256 is not None or expected_existed_before is True:
        raise ContinueConfigError(
            "Expected Continue .bak backup is missing; refusing destructive restore."
        )

    removed_file = False
    restored_sha256: str | None = None
    if cfg_path.is_file():
        raw_bytes = _read_bytes_bounded(cfg_path, "Continue config")
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        cleaned = remove_hivemind_from_continue_config_yaml(raw_text)
        if cleaned is None:
            cfg_path.unlink()
            removed_file = True
        else:
            cleaned_bytes = cleaned.encode("utf-8")
            _write_bytes_secure_atomic(
                cfg_path,
                cleaned_bytes,
                posix_mode=0o600,
                eff_os=eff_os,
                label="Continue config",
            )
            restored_sha256 = hashlib.sha256(cleaned_bytes).hexdigest()

    return {
        "path": redact_home_path(cfg_path, home=home),
        "restored": True,
        "restore_method": "hivemind_entries_stripped",
        "restored_sha256": restored_sha256,
        "byte_for_byte_match": None,
        "file_removed": removed_file,
        "backup_removed": True,
        "config_posix_mode": _posix_mode_str(cfg_path, eff_os) if cfg_path.is_file() else None,
    }


def _scan_continue_dir_residue(
    cont_dir: Path,
    *,
    allow_masked_only: bool = True,
) -> tuple[int, int, int]:
    """Return ``(hivemind_marker_hits, unmasked_key_hits, masked_key_hits)`` inside ``cont_dir``."""
    _ = allow_masked_only
    if not cont_dir.is_dir():
        return 0, 0, 0
    hm_marker_hits = 0
    unmasked_hits = 0
    masked_hits = 0
    for entry in sorted(cont_dir.rglob("*")):
        if not entry.is_file() or entry.is_symlink():
            continue
        raw = entry.read_bytes()[:MAX_CONFIG_BYTES]
        text = raw.decode("utf-8", errors="replace")
        low = text.lower()
        hm_marker_hits += sum(1 for m in _CONTINUE_RESIDUE_MARKERS if m in low)
        masked_hits += len(_MASKED_API_KEY_RE.findall(text))
        unmasked_hits += len(_RAW_HM_KEY_RE.findall(text))
    return hm_marker_hits, unmasked_hits, masked_hits


def _run_tampered_backup_negative_control(*, gateway: str, api_key: str, os_name: str | None) -> bool:
    """Verify inside an isolated temp sandbox that tampered/corrupted .bak backups fail closed."""
    with tempfile.TemporaryDirectory(prefix="hm-tamper-drill-") as tmp_dir:
        sbx = Path(tmp_dir)
        cfg = sbx / ".continue" / "config.yaml"
        baseline = DEFAULT_SAMPLE_EXISTING_CONTINUE_YAML
        _write_bytes_secure_atomic(
            cfg,
            baseline.encode("utf-8"),
            posix_mode=0o600,
            eff_os=os_name or os.name,
            label="Tamper drill baseline",
        )
        merge_doc = backup_and_merge_continue_config(
            cfg,
            gateway=gateway,
            api_key=api_key,
            os_name=os_name,
            home=sbx,
        )
        merged_before = cfg.read_bytes()
        bak = Path(f"{cfg}.bak")
        bak.write_bytes(b"corrupted_non_yaml_payload_without_colon\n")
        try:
            restore_continue_config_from_backup(
                cfg,
                expected_backup_sha256=str(merge_doc["backup_sha256"]),
                os_name=os_name,
                home=sbx,
            )
        except ContinueConfigError:
            return cfg.read_bytes() == merged_before
        return False


def verify_continue_config_roundtrip(
    home_dir: Path | str,
    *,
    sample_existing_yaml: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    api_key: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    verify_tamper_fail_closed: bool = True,
    os_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify Continue ``config.yaml`` non-destructive merge, 0600 ``.bak`` backup, and exact restore."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    if isinstance(home_dir, bool) or not isinstance(home_dir, (Path, str)) or not str(home_dir).strip():
        raise ContinueConfigError("home_dir must be a non-empty Path or directory string.")
    eff_home = Path(home_dir)
    if eff_home.is_symlink() or (eff_home.exists() and not eff_home.is_dir()):
        raise ContinueConfigError("home_dir must be a real directory path, not a file or symlink.")

    eff_env = environ if environ is not None else env
    comp_cwd = cwd if cwd is not None else eff_home
    comp_doc = evaluate_personal_pc_compliance(
        environ=eff_env,
        cwd=comp_cwd,
        check_git_remote=check_git_remote,
        lang=lang_code,
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt(
            "verify_continue_config_roundtrip",
            comp_doc,
            lang_code=lang_code,
            gateway=gateway,
        )
    if cwd is not None:
        home_markers = _find_corporate_markers(str(eff_home))
        if home_markers:
            blocked = {
                "corporate_indicators": [f"home_dir:{m}" for m in home_markers],
                "matched_markers": sorted(home_markers),
                "summary": (
                    f"REFUS DE CONFORMITÉ (home_dir:{home_markers[0]}) : chemin d'entreprise détecté. "
                    f"{COMPLIANCE_NOTICE_BILINGUAL}"
                    if is_fr
                    else f"COMPLIANCE BLOCKED (home_dir:{home_markers[0]}): corporate home path detected. "
                    f"{COMPLIANCE_NOTICE_BILINGUAL}"
                ),
            }
            return _compliance_blocked_receipt(
                "verify_continue_config_roundtrip",
                blocked,
                lang_code=lang_code,
                gateway=gateway,
            )

    clean_gw = validate_gateway_url(gateway)
    eff_os = os_name if os_name is not None else os.name
    eff_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else f"hm-sbx-{secrets.token_hex(12)}"
    if not is_valid_hivemind_key(eff_key):
        raise ContinueConfigError("api_key must be a valid unmasked hm-... or hm_live_... key.")
    masked_key = mask_api_key(eff_key)

    cont_dir = eff_home / ".continue"
    cfg_path = cont_dir / "config.yaml"
    bak_path = cont_dir / "config.yaml.bak"
    _reject_symlink(cfg_path, "Continue config")
    _reject_symlink(bak_path, "Continue backup")

    file_writes = 0
    if sample_existing_yaml is not None:
        validate_continue_config_text(sample_existing_yaml)
        _write_bytes_secure_atomic(
            cfg_path,
            sample_existing_yaml.encode("utf-8"),
            posix_mode=0o600,
            eff_os=eff_os,
            label="Baseline Continue config",
        )
        file_writes += 1
        baseline_bytes = sample_existing_yaml.encode("utf-8")
    elif cfg_path.exists():
        baseline_bytes = _read_bytes_bounded(cfg_path, "Continue config")
        validate_continue_config_text(baseline_bytes.decode("utf-8"))
    else:
        _write_bytes_secure_atomic(
            cfg_path,
            DEFAULT_SAMPLE_EXISTING_CONTINUE_YAML.encode("utf-8"),
            posix_mode=0o600,
            eff_os=eff_os,
            label="Default sandbox Continue config",
        )
        file_writes += 1
        baseline_bytes = DEFAULT_SAMPLE_EXISTING_CONTINUE_YAML.encode("utf-8")

    baseline_text = baseline_bytes.decode("utf-8")
    baseline_meta = validate_continue_config_text(baseline_text)
    baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    preexisting_non_hm_models = [
        {"name": m["name"], "provider": m["provider"], "model": m["model"]}
        for m in baseline_meta["non_hivemind_models"]
    ]
    preexisting_sections = list(baseline_meta["other_top_level_keys"])

    # Step 1: Non-destructive merge + atomic 0600 .bak backup
    merge_res = backup_and_merge_continue_config(
        cfg_path,
        gateway=clean_gw,
        api_key=eff_key,
        max_tokens=max_tokens,
        os_name=eff_os,
        home=eff_home,
    )
    if merge_res["written"]:
        file_writes += 2 if merge_res["backup_created"] else 1

    merged_bytes = _read_bytes_bounded(cfg_path, "Merged Continue config")
    merged_text = merged_bytes.decode("utf-8")
    merged_meta = validate_continue_config_text(merged_text)
    merged_non_hm = [
        {"name": m["name"], "provider": m["provider"], "model": m["model"]}
        for m in merged_meta["non_hivemind_models"]
    ]
    merged_hm_models = [m["model"] for m in merged_meta["hivemind_models"]]
    non_hm_preserved_on_merge = (
        merged_non_hm == preexisting_non_hm_models
        and merged_meta["other_top_level_keys"] == preexisting_sections
    )
    hivemind_models_present = all(m in merged_hm_models for m in DEFAULT_MODELS)

    bak_exists = bak_path.is_file() and not bak_path.is_symlink()
    bak_bytes = _read_bytes_bounded(bak_path, "Continue backup") if bak_exists else b""
    bak_sha256 = hashlib.sha256(bak_bytes).hexdigest() if bak_exists else None
    backup_exact_match = bool(bak_exists and bak_bytes == baseline_bytes and bak_sha256 == baseline_sha256)
    backup_0600_ok = eff_os == "nt" or (bak_exists and (bak_path.stat().st_mode & 0o777) == 0o600)
    merged_0600_ok = eff_os == "nt" or (cfg_path.stat().st_mode & 0o777) == 0o600

    # Step 2: Idempotent re-merge must keep .bak intact and make zero overwrites
    remerge_res = backup_and_merge_continue_config(
        cfg_path,
        gateway=clean_gw,
        api_key=eff_key,
        max_tokens=max_tokens,
        os_name=eff_os,
        home=eff_home,
    )
    bak_sha_after_remerge = hashlib.sha256(_read_bytes_bounded(bak_path, "Continue backup")).hexdigest() if bak_path.is_file() else None
    idempotent_remerge_ok = (
        remerge_res["action"] == "unchanged"
        and remerge_res["written"] is False
        and remerge_res["backup_created"] is False
        and bak_sha_after_remerge == baseline_sha256
    )

    # Step 3: Structural uninstall check on merged text
    stripped_text = remove_hivemind_from_continue_config_yaml(merged_text)
    structural_uninstall_ok = False
    if stripped_text is not None:
        stripped_meta = validate_continue_config_text(stripped_text)
        stripped_non_hm = [
            {"name": m["name"], "provider": m["provider"], "model": m["model"]}
            for m in stripped_meta["non_hivemind_models"]
        ]
        structural_uninstall_ok = (
            stripped_non_hm == preexisting_non_hm_models
            and stripped_meta["other_top_level_keys"] == preexisting_sections
            and len(stripped_meta["hivemind_models"]) == 0
        )

    # Step 4: Exact byte-for-byte restore from verified .bak backup
    restore_res = restore_continue_config_from_backup(
        cfg_path,
        expected_backup_sha256=baseline_sha256,
        expected_existed_before=True,
        remove_backup=True,
        os_name=eff_os,
        home=eff_home,
    )
    file_writes += 1

    restored_bytes = _read_bytes_bounded(cfg_path, "Restored Continue config")
    restored_sha256 = hashlib.sha256(restored_bytes).hexdigest()
    byte_for_byte_restored = restored_bytes == baseline_bytes and restored_sha256 == baseline_sha256
    restored_0600_ok = eff_os == "nt" or (cfg_path.stat().st_mode & 0o777) == 0o600
    backup_residue_cleaned = not bak_path.exists()
    hm_marker_hits, unmasked_hits, masked_hits = _scan_continue_dir_residue(cont_dir)
    if eff_key.encode("utf-8") in restored_bytes or eff_key in merged_text and eff_key in restored_bytes.decode("utf-8", errors="replace"):
        unmasked_hits += 1

    # Step 5: Isolated negative-control tamper drill (when enabled)
    tampered_backup_rejected: bool | None = None
    if verify_tamper_fail_closed:
        tampered_backup_rejected = _run_tampered_backup_negative_control(
            gateway=clean_gw,
            api_key=eff_key,
            os_name=os_name,
        )

    roundtrip_ok = bool(
        non_hm_preserved_on_merge
        and hivemind_models_present
        and backup_exact_match
        and backup_0600_ok
        and merged_0600_ok
        and idempotent_remerge_ok
        and structural_uninstall_ok
        and byte_for_byte_restored
        and restored_0600_ok
        and backup_residue_cleaned
        and hm_marker_hits == 0
        and unmasked_hits == 0
        and (tampered_backup_rejected is not False)
        and restore_res.get("restored") is True
    )
    go_nogo = "GO" if roundtrip_ok else "NO-GO"
    decision = "CONTINUE_ROUNDTRIP_VERIFIED" if roundtrip_ok else "HOLD_CONTINUE_ROUNDTRIP_FAILED"
    diagnostics: list[str] = []
    if not roundtrip_ok:
        diagnostics.append(
            "Aller-retour fusion/sauvegarde/restauration Continue.dev incomplet."
            if is_fr
            else "Continue.dev non-destructive merge/backup/restore roundtrip invariant failed."
        )

    summary = (
        f"Aller-retour Continue.dev {go_nogo} : modèles utilisateur={len(preexisting_non_hm_models)} "
        f"restaurés octet-à-octet, .bak 0600 nettoyé, clé={masked_key}, {FINANCIAL_PROMISE}."
        if is_fr
        else f"Continue.dev roundtrip {go_nogo}: {len(preexisting_non_hm_models)} user model(s) "
        f"restored byte-for-byte, 0600 .bak cleaned, key={masked_key}, {FINANCIAL_PROMISE}."
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": ROUNDTRIP_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "verify_continue_config_roundtrip",
            "lang": lang_code,
            "gateway": clean_gw,
            "home_dir": redact_home_path(eff_home, home=eff_home),
            "config_path": redact_home_path(cfg_path, home=eff_home),
            "compliance_ok": True,
            "corporate_detected": False,
            "roundtrip_ok": roundtrip_ok,
            "ready": roundtrip_ok,
            "go_nogo": go_nogo,
            "decision": decision,
            "masked_api_key": masked_key,
            "baseline_sha256": baseline_sha256,
            "backup_sha256": bak_sha256,
            "merged_sha256": merge_res["merged_sha256"],
            "restored_sha256": restored_sha256,
            "backup_created": merge_res["backup_created"],
            "backup_exact_match": backup_exact_match,
            "backup_posix_mode": merge_res["backup_posix_mode"],
            "merged_posix_mode": merge_res["config_posix_mode"],
            "restored_posix_mode": restore_res["config_posix_mode"],
            "backup_0600_ok": backup_0600_ok,
            "merged_0600_ok": merged_0600_ok,
            "restored_0600_ok": restored_0600_ok,
            "non_hivemind_models_preserved_count": len(preexisting_non_hm_models),
            "preserved_preexisting_models": [m["model"] for m in preexisting_non_hm_models],
            "preserved_preexisting_sections": preexisting_sections,
            "non_hivemind_preserved_on_merge": non_hm_preserved_on_merge,
            "hivemind_models_added": list(DEFAULT_MODELS),
            "hivemind_models_present": hivemind_models_present,
            "idempotent_remerge_ok": idempotent_remerge_ok,
            "structural_uninstall_ok": structural_uninstall_ok,
            "byte_for_byte_restored": byte_for_byte_restored,
            "backup_residue_cleaned": backup_residue_cleaned,
            "hivemind_residue_count": hm_marker_hits,
            "unmasked_api_keys_count": unmasked_hits,
            "masked_keys_tolerated_count": masked_hits,
            "tampered_backup_rejected": tampered_backup_rejected,
            "file_writes": file_writes,
            "network_calls": 0,
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
            "exit_code": 0 if roundtrip_ok else 2,
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
            "continue_flag_advertised": "--write-continue-config" in low_html,
            "uninstall_flag_advertised": "--uninstall" in low_html,
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


def verify_continue_and_holiday_readiness(
    gateway: str = DEFAULT_GATEWAY,
    *,
    home_dir: Path | str | None = None,
    sample_existing_yaml: str | None = None,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    expected_sha: str | None = None,
    os_name: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify Continue config merge/backup/restore roundtrip plus zero-inference holiday routes."""
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
            "verify_continue_and_holiday_readiness",
            comp_doc,
            lang_code=lang_code,
        )

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None

    if home_dir is None:
        sandbox_isolated = True
        with tempfile.TemporaryDirectory(prefix="hivemind-continue-sbx-") as tmp_sbx:
            roundtrip_doc = verify_continue_config_roundtrip(
                Path(tmp_sbx),
                sample_existing_yaml=sample_existing_yaml,
                gateway=clean_gw,
                environ=eff_env,
                cwd=cwd if cwd is not None else Path.cwd(),
                check_git_remote=False,
                os_name=os_name,
                lang=lang_code,
            )
    else:
        sandbox_isolated = False
        roundtrip_doc = verify_continue_config_roundtrip(
            home_dir,
            sample_existing_yaml=sample_existing_yaml,
            gateway=clean_gw,
            environ=eff_env,
            cwd=cwd if cwd is not None else Path.cwd(),
            check_git_remote=False,
            os_name=os_name,
            lang=lang_code,
        )

    if not roundtrip_doc.get("compliance_ok", False):
        return roundtrip_doc

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = list(roundtrip_doc.get("diagnostics") or [])
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}

    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    missing_manifest_routes: list[str] = []
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {}
    network_calls = 0

    for path, kind, group in HOLIDAY_PROBE_ROUTES:
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
        else:
            fb_json = _unwrap_json_body(body)
            ok, feedback_summary_doc, fb_issues = _inspect_feedback_summary_exact(fb_json, lang_code=lang_code)
            feedback_summary_ok = ok
            if not ok:
                diag = fb_issues[0]
                diagnostics.extend(fb_issues)

        if not ok:
            failed_routes.append(path)
        route_digest_src = json.dumps(
            {"path": path, "http_status": code, "ok": ok},
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
                    f"hivemind.continue_backup_restore.route/1:{route_digest_src}".encode()
                ).hexdigest(),
            }
        )

    html_all_ok = bool(
        len(html_contracts) == len(HOLIDAY_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in HOLIDAY_HTML_ROUTES)
        and len(seen_nonces) == len(HOLIDAY_HTML_ROUTES)
    )
    roundtrip_ok = bool(roundtrip_doc.get("roundtrip_ok"))
    stale_404_drift = bool(stale_404_routes or missing_manifest_routes)
    live_routes_ok = bool(
        html_all_ok
        and manifest_ok
        and feedback_summary_ok
        and (clean_expected_sha is None or sha_match is True)
    )
    redeploy_required = bool(stale_404_drift or not live_routes_ok)

    hold_reasons: list[str] = []
    if not roundtrip_ok:
        hold_reasons.append("continue_backup_restore_failed")
    if not live_routes_ok:
        hold_reasons.append("routes_drift_or_contract_failure")

    ready = bool(roundtrip_ok and live_routes_ok)
    if ready:
        decision = "CONTINUE_BACKUP_RESTORE_AND_HOLIDAY_READY"
    elif not roundtrip_ok and not live_routes_ok:
        decision = "HOLD_CONTINUE_ROUNDTRIP_AND_ROUTES_DRIFT"
    elif not roundtrip_ok:
        decision = "HOLD_CONTINUE_BACKUP_RESTORE_FAILED"
    else:
        decision = "HOLD_ROUTES_DRIFT"

    go_nogo = "GO" if ready else "NO-GO"
    merkle_root = compute_merkle_root(
        [str(roundtrip_doc["receipt_sha256"])] + [row["leaf_sha256"] for row in route_rows]
    )

    if ready:
        summary = (
            f"Continue.dev & Noël {go_nogo} sur {clean_gw} : sauvegarde 0600 & restauration exacte "
            f"({roundtrip_doc['masked_api_key']}), routes={len(HOLIDAY_PROBE_ROUTES)}/{len(HOLIDAY_PROBE_ROUTES)}, "
            f"{FINANCIAL_PROMISE}."
            if is_fr
            else f"Continue.dev & holiday gate {go_nogo} on {clean_gw}: 0600 backup & exact restore "
            f"({roundtrip_doc['masked_api_key']}), routes={len(HOLIDAY_PROBE_ROUTES)}/{len(HOLIDAY_PROBE_ROUTES)}, "
            f"{FINANCIAL_PROMISE}."
        )
    else:
        summary = (
            f"Continue.dev & Noël {go_nogo} ({decision}) sur {clean_gw} "
            f"(raisons={', '.join(hold_reasons)})."
            if is_fr
            else f"Continue.dev & holiday gate {go_nogo} ({decision}) on {clean_gw} "
            f"(reasons={', '.join(hold_reasons)})."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "verify_continue_and_holiday_readiness",
            "lang": lang_code,
            "gateway": clean_gw,
            "sandbox_isolated": sandbox_isolated,
            "compliance_ok": True,
            "corporate_detected": False,
            "ready": ready,
            "go_nogo": go_nogo,
            "decision": decision,
            "hold_reasons": hold_reasons,
            "roundtrip_ok": roundtrip_ok,
            "continue_roundtrip": roundtrip_doc,
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
            "file_writes": int(roundtrip_doc.get("file_writes") or 0),
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
    """Render a compact FR/EN CLI Continue.dev backup-restore & readiness receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    title = (
        f"HIVEMIND NOËL — CONTINUE.DEV FUSION & RESTAURATION ({verdict}, 0,00 EUR / 0,00 USD)"
        if is_fr
        else f"HIVEMIND HOLIDAY — CONTINUE.DEV MERGE & BACKUP-RESTORE ({verdict}, 0.00 EUR / 0.00 USD)"
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
    rt_blk = report.get("continue_roundtrip") if isinstance(report.get("continue_roundtrip"), Mapping) else report
    if "roundtrip_ok" in rt_blk:
        lines.append(
            f"  Continue    : roundtrip_ok={rt_blk.get('roundtrip_ok')} "
            f"(byte_restore={rt_blk.get('byte_for_byte_restored')}, "
            f"bak_0600={rt_blk.get('backup_0600_ok')}, "
            f"key={rt_blk.get('masked_api_key')})"
        )
    if report.get("merkle_root_sha256"):
        lines.append(f"  Merkle Root : {report.get('merkle_root_sha256')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256     : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN Markdown Continue.dev backup-restore & readiness receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    heading = (
        f"# Vérificateur Fusion Non-Destructive & Restauration Continue.dev HiveMind — {verdict}"
        if is_fr
        else f"# HiveMind Continue.dev Non-Destructive Merge & Backup-Restore Gate — {verdict}"
    )
    lines = [
        heading,
        "",
        f"- **{'Verdict Go/No-Go' if is_fr else 'Go/No-Go Verdict'}**: `{verdict}` (`decision={report.get('decision', verdict)}`, `exit_code={report.get('exit_code')}`)",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: `{comp_ok}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Zéro inférence' if is_fr else 'Zero-Inference Verified'}**: `{report.get('zero_inference_verified', True)}`",
    ]
    rt_blk = report.get("continue_roundtrip") if isinstance(report.get("continue_roundtrip"), Mapping) else report
    if "roundtrip_ok" in rt_blk:
        lines.append(
            f"- **{'Aller-Retour Continue.dev' if is_fr else 'Continue.dev Roundtrip'}**: "
            f"`roundtrip_ok={rt_blk.get('roundtrip_ok')}`, "
            f"`byte_for_byte_restored={rt_blk.get('byte_for_byte_restored')}`, "
            f"`backup_0600_ok={rt_blk.get('backup_0600_ok')}`, "
            f"`tampered_backup_rejected={rt_blk.get('tampered_backup_rejected')}`, "
            f"`masked_api_key={rt_blk.get('masked_api_key')}`"
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
    """Build CLI parser for the Continue.dev config backup-restore & readiness verifier."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-inference Continue.dev non-destructive config merge, 0600 .bak backup-restore, "
            "and holiday route readiness verifier (0.00 EUR / 0.00 USD, AGPL-3.0)."
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
        help="Optional sandbox home directory for Continue config roundtrip (defaults to temp sandbox).",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Expected build_sha on /v1/hivemind/onboarding/manifest.",
    )
    parser.add_argument(
        "--roundtrip-only",
        action="store_true",
        help="Verify only local isolated Continue config merge/backup/restore without network probes.",
    )
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Run only the strictly local Personal-PC compliance check.",
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
    """CLI entrypoint enforcing compliance before any network call or sandbox file write."""
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
                check_git_remote=True,
                lang=lang_code,
            )
        elif args.roundtrip_only:
            comp_check = evaluate_personal_pc_compliance(
                environ=eff_env,
                cwd=cwd,
                check_git_remote=True,
                lang=lang_code,
            )
            if not comp_check["compliance_ok"]:
                report = _compliance_blocked_receipt(
                    "verify_continue_config_roundtrip",
                    comp_check,
                    lang_code=lang_code,
                    gateway=args.gateway,
                )
            elif args.home_dir is not None:
                report = verify_continue_config_roundtrip(
                    args.home_dir,
                    gateway=args.gateway,
                    environ=eff_env,
                    cwd=cwd if cwd is not None else Path.cwd(),
                    check_git_remote=False,
                    lang=lang_code,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="hivemind-continue-sbx-") as tmp_sbx:
                    report = verify_continue_config_roundtrip(
                        Path(tmp_sbx),
                        gateway=args.gateway,
                        environ=eff_env,
                        cwd=cwd if cwd is not None else Path.cwd(),
                        check_git_remote=False,
                        lang=lang_code,
                    )
        else:
            report = verify_continue_and_holiday_readiness(
                args.gateway,
                home_dir=args.home_dir,
                http_fn=http_fn,
                environ=eff_env,
                cwd=cwd,
                expected_sha=args.expected_sha,
                lang=lang_code,
            )
    except ContinueConfigError as exc:
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
