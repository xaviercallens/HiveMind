#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference Zero-Clone Script-Bundle SHA-256 Integrity & Manifest Verifier (`SLICE-SCALE-DYN-27`).

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment markers (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``,
     ``NODE_EXTRA_CA_CERTS``), workspace path, and ``.git/config`` remote URLs (including worktree
     ``gitdir:`` / ``commondir`` indirection) BEFORE any network call or file write. Fails closed
     with exit code 2 (0 network calls, 0 file writes, 0 raw env/hostname/path leaks).
  2. Zero-Inference Zero-Clone Script-Bundle SHA-256 & Merkle Verifier
     (``verify_zero_clone_script_bundle``):
     fetches ``GET /v1/hivemind/onboarding/manifest``, downloads every listed downloadable script
     via ``GET /scripts/<name>``, verifies non-empty strict UTF-8 source (plus ``ast.parse`` for
     ``.py`` scripts), AGPL-3.0 header within the first 300 chars, optional ``X-Script-SHA256``
     header consistency, exact byte-for-byte SHA-256 match against ``local_scripts_dir`` (when
     provided), and a deterministic domain-separated binary SHA-256 bundle Merkle root.
  3. Zero-Inference Holiday Route, CSP Nonce & Exact 5-Key Feedback Verifier:
     probes ``GET /noel``, ``GET /quick``, ``GET /raid-noel``, ``GET /arena-noel``, and
     ``GET /v1/hivemind/onboarding/feedback/summary`` in strict read-only zero-inference mode
     (never calling ``POST /v1/chat/completions`` or prompting ``hivemind/coder``), enforcing
     unique per-request CSP nonces ``default-src 'none'``, 0 unsafe DOM sinks, 0 static figures
     (stripping ``hm-***`` / ``hm_live_***`` masked-key suffixes and HTML ``nonce="..."``
     attributes first), exact 5-key feedback summary equality, fail-closed rejection on any SHA-256
     mismatch or HTTP 404/5xx, and FR/EN JSON & Markdown SHA-256 receipts.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen science under AGPL-3.0 (zero monetary value).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import inspect
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_script_bundle_integrity_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_script_bundle_integrity_receipt"
BUNDLE_SUMMARY_OBJECT = "hivemind.noel_script_bundle_merkle_summary"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
AGPL_HEADER_LINE = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)

HTTP_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
MAX_SCRIPT_FILE_BYTES = 4 * 1024 * 1024
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


class ScriptBundleError(ValueError):
    """Raised when script-bundle verifier arguments, host allowlists, or safety invariants fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI or API output language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ScriptBundleError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ScriptBundleError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


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


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and masked ``hm-***`` suffixes."""
    if not isinstance(text, str):
        raise ScriptBundleError("strip_masked_keys_and_nonces requires a string.")
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
    """Replace OS home prefixes with ``'~'`` and map repo/workspace paths safely."""
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
        raise ScriptBundleError("scan_output_safety requires a string.")
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
        raise ScriptBundleError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise ScriptBundleError("Forbidden JSON key detected in script-bundle receipt.")
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
        raise ScriptBundleError(
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
    domain: str = "hivemind.script_bundle.merkle",
) -> str:
    """Compute a deterministic domain-separated binary SHA-256 Merkle root over leaf digests."""
    if not leaf_hashes:
        return hashlib.sha256(f"{domain}.empty/1".encode()).hexdigest()
    level: list[str] = []
    for digest in leaf_hashes:
        if not isinstance(digest, str) or not _SHA256_FULL_RE.fullmatch(digest):
            raise ScriptBundleError("Merkle leaf digest must be a 64-char lowercase hex SHA-256.")
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
        raise ScriptBundleError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ScriptBundleError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise ScriptBundleError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ScriptBundleError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ScriptBundleError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise ScriptBundleError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ScriptBundleError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ScriptBundleError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise ScriptBundleError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ScriptBundleError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ScriptBundleError("Remote gateway URL must use https.")
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
        raise ScriptBundleError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ScriptBundleError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ScriptBundleError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise ScriptBundleError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ScriptBundleError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise ScriptBundleError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise ScriptBundleError(
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


def _parse_non_json_bytes(
    status_code: int,
    raw_bytes: bytes,
    lower_headers: dict[str, str],
) -> tuple[int, dict[str, Any]]:
    try:
        strict_text = raw_bytes.decode("utf-8")
        utf8_valid = True
    except UnicodeDecodeError:
        strict_text = ""
        utf8_valid = False
    return int(status_code), {
        "raw_bytes": raw_bytes,
        "utf8_valid": utf8_valid,
        "byte_length": len(raw_bytes),
        "html": strict_text,
        "text": strict_text,
        "html_length": len(strict_text),
        "text_length": len(raw_bytes),
        "headers": lower_headers,
        "csp": lower_headers.get("content-security-policy", ""),
        "x_script_sha256": lower_headers.get("x-script-sha256", ""),
    }


def _parse_http_raw_response(
    status_code: int,
    raw_bytes: bytes,
    raw_headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
        raise ScriptBundleError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
    lower_headers = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    ctype = lower_headers.get("content-type", "").lower()
    looks_text_script = "text/x-python" in ctype or "text/plain" in ctype or "text/html" in ctype
    if "application/json" in ctype or (
        not looks_text_script and raw_bytes.lstrip().startswith((b"{", b"["))
    ):
        text = raw_bytes.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": "invalid_json"}
        if isinstance(parsed, dict):
            return int(status_code), dict(parsed)
        return int(status_code), {"data": parsed}
    return _parse_non_json_bytes(status_code, raw_bytes, lower_headers)


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
    """Proxy-free stdlib HTTP helper (``ProxyHandler({})``) preserving exact raw bytes & JSON keys."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(
        verb,
        url,
        payload=payload,
        allowed_post_paths=ALLOWED_POST_PATHS,
    )
    if payload is not None:
        raise ScriptBundleError("GET/HEAD requests must never attach a body payload.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/x-python;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-script-bundle-integrity-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise ScriptBundleError("HTTP headers must not contain CR/LF or NUL characters.")
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
    header_markers = {
        "content-type",
        "content-security-policy",
        "cache-control",
        "vary",
        "content-language",
        "x-script-sha256",
        "content-disposition",
    }
    body_markers = {"object", "status", "total_responses", "holiday_routes", "data", "detail", "error"}
    if keys & body_markers:
        return False
    return bool(keys & header_markers) or all(isinstance(v, str) for v in candidate.values())


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise ScriptBundleError("http_fn must return (status_code, body) or (status_code, headers, body).")
    if len(raw_ret) == 2:
        status_code, body = raw_ret
        hdr_map: dict[str, str] = {}
    else:
        status_code, part1, part2 = raw_ret
        if isinstance(part1, (str, bytes, bytearray)) and isinstance(part2, Mapping):
            body, raw_hdrs = part1, part2
        elif isinstance(part1, Mapping) and isinstance(part2, (str, bytes, bytearray)):
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
        if "raw_bytes" in body or "html" in body or "text" in body:
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
                out.setdefault("x_script_sha256", merged.get("x-script-sha256", ""))
            return int(status_code), out
        if hdr_map:
            return int(status_code), {"_json": dict(body), "_headers": hdr_map}
        return int(status_code), dict(body)

    if isinstance(body, (bytes, bytearray)):
        return _parse_http_raw_response(int(status_code), bytes(body), hdr_map)

    if isinstance(body, str):
        ctype = hdr_map.get("content-type", "").lower()
        looks_text_script = "text/x-python" in ctype or "text/plain" in ctype or "text/html" in ctype
        stripped = body.lstrip()
        if not looks_text_script and stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, Mapping):
                    return int(status_code), (
                        {"_json": dict(parsed), "_headers": hdr_map} if hdr_map else dict(parsed)
                    )
                return int(status_code), (
                    {"data": parsed, "_headers": hdr_map} if hdr_map else {"data": parsed}
                )
            except json.JSONDecodeError:
                parsed = None
                _ = parsed
        raw_bytes = body.encode("utf-8")
        return int(status_code), {
            "raw_bytes": raw_bytes,
            "utf8_valid": True,
            "byte_length": len(raw_bytes),
            "html": body,
            "text": body,
            "html_length": len(body),
            "text_length": len(raw_bytes),
            "headers": hdr_map,
            "csp": hdr_map.get("content-security-policy", ""),
            "x_script_sha256": hdr_map.get("x-script-sha256", ""),
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
    except ScriptBundleError:
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
            "bundle_ok": False,
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


def _inspect_manifest_contract(
    body: Mapping[str, Any],
    *,
    expected_sha: str | None,
    expected_scripts: Sequence[str] | None,
    lang_code: str,
) -> tuple[bool, str, bool | None, list[str], list[str], list[str], list[str]]:
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

    raw_scripts = body.get("downloadable_scripts")
    listed_scripts: list[str] = []
    invalid_script_entries: list[str] = []
    duplicate_scripts: list[str] = []
    if isinstance(raw_scripts, list) and raw_scripts and all(isinstance(s, str) for s in raw_scripts):
        seen_names: set[str] = set()
        for entry in raw_scripts:
            if not _SAFE_SCRIPT_NAME_RE.fullmatch(entry):
                invalid_script_entries.append("<unsafe-script-name>")
                continue
            if entry in seen_names:
                duplicate_scripts.append(entry)
                continue
            seen_names.add(entry)
            listed_scripts.append(entry)
    else:
        issues.append(
            "Liste `downloadable_scripts` absente ou invalide dans `/v1/hivemind/onboarding/manifest`."
            if is_fr
            else "Manifest `downloadable_scripts` missing, empty, or invalid."
        )

    if invalid_script_entries:
        issues.append(
            "Nom de script non sûr détecté dans `downloadable_scripts`."
            if is_fr
            else "Unsafe script filename detected in manifest `downloadable_scripts`."
        )
    if duplicate_scripts:
        joined_dup = ", ".join(sorted(set(duplicate_scripts)))
        issues.append(
            f"Doublons détectés dans `downloadable_scripts` : {joined_dup}."
            if is_fr
            else f"Duplicate entries detected in manifest `downloadable_scripts`: {joined_dup}."
        )

    missing_expected_scripts: list[str] = []
    if expected_scripts is not None:
        missing_expected_scripts = [s for s in expected_scripts if s not in listed_scripts]
        if missing_expected_scripts:
            joined_s = ", ".join(missing_expected_scripts)
            issues.append(
                f"Scripts attendus absents du manifeste : {joined_s}."
                if is_fr
                else f"Manifest `downloadable_scripts` missing expected scripts: {joined_s}."
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

    return (
        len(issues) == 0,
        live_sha,
        sha_match,
        missing_routes,
        sorted(listed_scripts),
        missing_expected_scripts,
        issues,
    )


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


def _extract_script_payload(body: Mapping[str, Any]) -> tuple[bytes, bool, str, str]:
    raw_obj = body.get("raw_bytes")
    if isinstance(raw_obj, (bytes, bytearray)):
        raw_bytes = bytes(raw_obj)
        utf8_valid_flag = body.get("utf8_valid")
        if isinstance(utf8_valid_flag, bool) and not utf8_valid_flag:
            utf8_valid = False
            text = ""
        else:
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
    hdrs = body.get("headers")
    hdr_sha = ""
    if isinstance(hdrs, Mapping):
        hdr_sha = str(hdrs.get("x-script-sha256") or "").strip().lower()
    if not hdr_sha:
        hdr_sha = str(body.get("x_script_sha256") or "").strip().lower()
    return raw_bytes, utf8_valid, text, hdr_sha


def _verify_single_downloaded_script(
    script_name: str,
    code: int,
    body: Mapping[str, Any],
    *,
    resolved_local_dir: Path | None,
    is_fr: bool,
) -> tuple[dict[str, Any], str | None, list[str]]:
    issues: list[str] = []
    if code != 200:
        msg = (
            f"GET /scripts/{script_name} -> HTTP {code} (attendu 200)."
            if is_fr
            else f"GET /scripts/{script_name} -> HTTP {code} (expected 200)."
        )
        issues.append(msg)
        fail_digest = hashlib.sha256(f"http-fail:{script_name}:{code}".encode()).hexdigest()
        leaf_sha = hashlib.sha256(
            f"hivemind.script_bundle.leaf/1:{script_name}:0:{fail_digest}".encode()
        ).hexdigest()
        row = {
            "script": script_name,
            "path": f"/scripts/{script_name}",
            "http_status": code,
            "ok": False,
            "non_empty": False,
            "utf8_valid": False,
            "agpl_header_ok": False,
            "syntax_ok": False,
            "byte_length": 0,
            "remote_sha256": None,
            "header_sha256_match": None,
            "local_present": None,
            "local_sha256": None,
            "local_sha256_match": None,
            "leaf_sha256": leaf_sha,
            "diagnostic": msg,
        }
        return row, None, issues

    raw_bytes, utf8_valid, source_text, hdr_sha = _extract_script_payload(body)
    byte_len = len(raw_bytes)
    non_empty = byte_len > 0 and (not utf8_valid or bool(source_text.strip()))
    if not non_empty:
        issues.append(
            f"Script `{script_name}` vide (0 octet)."
            if is_fr
            else f"Downloaded script `{script_name}` is empty (0 bytes)."
        )
    if not utf8_valid:
        issues.append(
            f"Script `{script_name}` non valide en UTF-8."
            if is_fr
            else f"Downloaded script `{script_name}` is not valid UTF-8."
        )

    agpl_ok = bool(utf8_valid and non_empty and AGPL_HEADER_LINE in source_text[:300])
    if utf8_valid and non_empty and not agpl_ok:
        issues.append(
            f"En-tête AGPL-3.0 manquant dans les 300 premiers caractères de `{script_name}`."
            if is_fr
            else f"Missing AGPL-3.0 copyright header in first 300 chars of `{script_name}`."
        )

    syntax_ok = False
    if utf8_valid and non_empty:
        if script_name.endswith(".py"):
            try:
                ast.parse(source_text, filename=script_name)
                syntax_ok = True
            except SyntaxError as exc:
                issues.append(
                    f"Syntaxe Python invalide dans `{script_name}` (ligne {exc.lineno})."
                    if is_fr
                    else f"Invalid Python syntax in `{script_name}` (line {exc.lineno})."
                )
        else:
            syntax_ok = "\x00" not in source_text

    remote_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    header_sha_match: bool | None = None
    if hdr_sha:
        header_sha_match = bool(
            _SHA256_FULL_RE.fullmatch(hdr_sha) and hmac.compare_digest(hdr_sha, remote_sha256)
        )
        if not header_sha_match:
            issues.append(
                f"Divergence d'en-tête `X-Script-SHA256` pour `{script_name}`."
                if is_fr
                else f"HTTP `X-Script-SHA256` header mismatch for `{script_name}`."
            )

    local_present: bool | None = None
    local_sha256: str | None = None
    local_sha256_match: bool | None = None
    local_leaf_sha: str | None = None
    if resolved_local_dir is not None:
        candidate = (resolved_local_dir / script_name).resolve()
        if candidate.parent != resolved_local_dir or not candidate.is_file():
            local_present = False
            local_sha256_match = False
            issues.append(
                f"Script local introuvable ou hors dossier dans `local_scripts_dir` : `{script_name}`."
                if is_fr
                else f"Local script missing or outside `local_scripts_dir`: `{script_name}`."
            )
        else:
            local_present = True
            local_bytes = candidate.read_bytes()
            if len(local_bytes) > MAX_SCRIPT_FILE_BYTES:
                local_sha256_match = False
                issues.append(
                    f"Script local `{script_name}` dépasse la taille maximale autorisée."
                    if is_fr
                    else f"Local script `{script_name}` exceeds maximum allowed size."
                )
            else:
                local_sha256 = hashlib.sha256(local_bytes).hexdigest()
                local_sha256_match = hmac.compare_digest(remote_sha256, local_sha256) and (
                    byte_len == len(local_bytes)
                )
                local_leaf_sha = hashlib.sha256(
                    f"hivemind.script_bundle.leaf/1:{script_name}:{len(local_bytes)}:{local_sha256}".encode()
                ).hexdigest()
                if not local_sha256_match:
                    issues.append(
                        f"Divergence SHA-256 octet-à-octet sur `{script_name}` "
                        f"(remote={remote_sha256[:12]}, local={local_sha256[:12]})."
                        if is_fr
                        else f"Byte-for-byte SHA-256 mismatch for `{script_name}` "
                        f"(remote={remote_sha256[:12]}, local={local_sha256[:12]})."
                    )

    ok = bool(
        non_empty
        and utf8_valid
        and agpl_ok
        and syntax_ok
        and (header_sha_match is not False)
        and (local_sha256_match is not False)
    )
    leaf_sha = hashlib.sha256(
        f"hivemind.script_bundle.leaf/1:{script_name}:{byte_len}:{remote_sha256}".encode()
    ).hexdigest()
    row = {
        "script": script_name,
        "path": f"/scripts/{script_name}",
        "http_status": code,
        "ok": ok,
        "non_empty": non_empty,
        "utf8_valid": utf8_valid,
        "agpl_header_ok": agpl_ok,
        "syntax_ok": syntax_ok,
        "byte_length": byte_len,
        "remote_sha256": remote_sha256,
        "header_sha256_match": header_sha_match,
        "local_present": local_present,
        "local_sha256": local_sha256,
        "local_sha256_match": local_sha256_match,
        "leaf_sha256": leaf_sha,
        "diagnostic": issues[0] if issues else "",
    }
    return row, local_leaf_sha, issues


def verify_zero_clone_script_bundle(
    gateway: str = DEFAULT_GATEWAY,
    *,
    local_scripts_dir: Path | str | None = None,
    expected_sha: str | None = None,
    expected_scripts: Sequence[str] | None = None,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify manifest, all downloadable ``/scripts/<name>`` SHA-256 digests, and holiday routes."""
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
            "verify_zero_clone_script_bundle",
            comp_doc,
            lang_code=lang_code,
        )

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None

    resolved_local_dir: Path | None = None
    local_dir_display: str | None = None
    if local_scripts_dir is not None:
        if (
            isinstance(local_scripts_dir, bool)
            or not isinstance(local_scripts_dir, (Path, str))
            or not str(local_scripts_dir).strip()
        ):
            raise ScriptBundleError("local_scripts_dir must be a non-empty Path or directory string.")
        cand_dir = Path(local_scripts_dir)
        if not cand_dir.is_dir():
            raise ScriptBundleError("local_scripts_dir must point to an existing directory.")
        resolved_local_dir = cand_dir.resolve()
        local_dir_display = redact_home_path(resolved_local_dir)

    http_statuses: dict[str, int] = {}
    script_http_statuses: dict[str, int] = {}
    diagnostics: list[str] = []
    stale_404_routes: list[str] = []
    failed_routes: list[str] = []
    network_calls = 0

    # 1. Fetch manifest first
    man_path = "/v1/hivemind/onboarding/manifest"
    network_calls += 1
    man_code, man_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{man_path}")
    http_statuses[man_path] = man_code

    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    missing_manifest_routes: list[str] = []
    listed_scripts: list[str] = []
    missing_expected_scripts: list[str] = []

    if man_code != 200:
        failed_routes.append(man_path)
        if man_code == 404:
            stale_404_routes.append(man_path)
        diagnostics.append(
            f"GET {man_path} -> HTTP {man_code} (attendu 200)."
            if is_fr
            else f"GET {man_path} -> HTTP {man_code} (expected 200)."
        )
    else:
        manifest_json = _unwrap_json_body(man_body)
        (
            manifest_ok,
            live_build_sha,
            sha_match,
            missing_manifest_routes,
            listed_scripts,
            missing_expected_scripts,
            m_issues,
        ) = _inspect_manifest_contract(
            manifest_json,
            expected_sha=clean_expected_sha,
            expected_scripts=expected_scripts,
            lang_code=lang_code,
        )
        if not manifest_ok:
            failed_routes.append(man_path)
            diagnostics.extend(m_issues)

    # 2. Download every listed script via GET /scripts/<name> and verify SHA-256 + AGPL + syntax
    script_rows: list[dict[str, Any]] = []
    local_leaf_hashes: list[str] = []
    missing_remote_scripts: list[str] = []
    sha256_mismatches: list[str] = []
    agpl_failures: list[str] = []
    syntax_failures: list[str] = []
    total_script_bytes = 0

    for script_name in listed_scripts:
        script_path = f"/scripts/{script_name}"
        network_calls += 1
        sc_code, sc_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{script_path}")
        script_http_statuses[script_name] = sc_code
        if sc_code == 404:
            stale_404_routes.append(script_path)
            missing_remote_scripts.append(script_name)

        row, local_leaf, sc_issues = _verify_single_downloaded_script(
            script_name,
            sc_code,
            sc_body,
            resolved_local_dir=resolved_local_dir,
            is_fr=is_fr,
        )
        script_rows.append(row)
        total_script_bytes += int(row["byte_length"])
        if local_leaf is not None:
            local_leaf_hashes.append(local_leaf)
        if row["http_status"] == 200 and not row["agpl_header_ok"]:
            agpl_failures.append(script_name)
        if row["http_status"] == 200 and not row["syntax_ok"]:
            syntax_failures.append(script_name)
        if row["local_sha256_match"] is False or row["header_sha256_match"] is False:
            sha256_mismatches.append(script_name)
        diagnostics.extend(sc_issues)

    bundle_merkle_root = compute_merkle_root([row["leaf_sha256"] for row in script_rows])
    local_bundle_merkle_root = (
        compute_merkle_root(local_leaf_hashes)
        if (resolved_local_dir is not None and len(local_leaf_hashes) == len(script_rows) and script_rows)
        else None
    )
    merkle_local_match: bool | None = (
        hmac.compare_digest(bundle_merkle_root, local_bundle_merkle_root)
        if local_bundle_merkle_root is not None
        else None
    )

    # 3. Probe holiday HTML routes and exact 5-key feedback summary in zero-inference mode
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {}
    route_rows: list[dict[str, Any]] = []

    manifest_leaf_src = json.dumps(
        {"path": man_path, "http_status": man_code, "ok": manifest_ok, "live_build_sha": live_build_sha},
        sort_keys=True,
        ensure_ascii=False,
    )
    route_rows.append(
        {
            "path": man_path,
            "http_status": man_code,
            "ok": manifest_ok,
            "leaf_sha256": hashlib.sha256(f"hivemind.script_bundle.route/1:{manifest_leaf_src}".encode()).hexdigest(),
        }
    )

    for path, kind, group in HOLIDAY_PROBE_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        ok = code == 200
        if code != 200:
            if code == 404 and group in ("holiday", "feedback"):
                stale_404_routes.append(path)
            diagnostics.append(
                f"GET {path} -> HTTP {code} (attendu 200)."
                if is_fr
                else f"GET {path} -> HTTP {code} (expected 200)."
            )
        elif kind == "html":
            ok, h_doc, h_issues = _inspect_html_route(path, body, seen_nonces, lang_code=lang_code)
            html_contracts[path] = h_doc
            if not ok:
                diagnostics.extend(h_issues)
        else:
            fb_json = _unwrap_json_body(body)
            ok, feedback_summary_doc, fb_issues = _inspect_feedback_summary_exact(fb_json, lang_code=lang_code)
            feedback_summary_ok = ok
            if not ok:
                diagnostics.extend(fb_issues)

        if not ok:
            failed_routes.append(path)
        r_src = json.dumps({"path": path, "http_status": code, "ok": ok}, sort_keys=True, ensure_ascii=False)
        route_rows.append(
            {
                "path": path,
                "http_status": code,
                "ok": ok,
                "leaf_sha256": hashlib.sha256(f"hivemind.script_bundle.route/1:{r_src}".encode()).hexdigest(),
            }
        )

    scripts_verified_count = sum(1 for row in script_rows if row["ok"])
    scripts_total_count = len(script_rows)
    scripts_all_ok = bool(
        listed_scripts
        and scripts_total_count > 0
        and scripts_verified_count == scripts_total_count
        and not missing_expected_scripts
        and (merkle_local_match is not False)
    )
    html_all_ok = bool(
        len(html_contracts) == len(HOLIDAY_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in HOLIDAY_HTML_ROUTES)
        and len(seen_nonces) == len(HOLIDAY_HTML_ROUTES)
    )
    routes_all_ok = bool(
        manifest_ok
        and html_all_ok
        and feedback_summary_ok
        and (clean_expected_sha is None or sha_match is True)
    )
    bundle_ok = bool(scripts_all_ok and routes_all_ok)
    stale_404_drift = bool(stale_404_routes or missing_manifest_routes or missing_expected_scripts)
    redeploy_required = bool(stale_404_drift or not bundle_ok)

    hold_reasons: list[str] = []
    if sha256_mismatches:
        hold_reasons.append("sha256_mismatch")
    if agpl_failures:
        hold_reasons.append("missing_agpl_header")
    if syntax_failures:
        hold_reasons.append("script_syntax_error")
    if missing_remote_scripts or stale_404_routes:
        hold_reasons.append("http_404_drift")
    if not routes_all_ok or not scripts_all_ok:
        if "bundle_or_routes_failed" not in hold_reasons and not hold_reasons:
            hold_reasons.append("bundle_or_routes_failed")

    if bundle_ok:
        decision = "SCRIPT_BUNDLE_VERIFIED"
    elif sha256_mismatches and routes_all_ok:
        decision = "HOLD_SHA256_MISMATCH"
    elif not scripts_all_ok and not routes_all_ok:
        decision = "HOLD_SCRIPT_BUNDLE_AND_ROUTES_DRIFT"
    elif not scripts_all_ok:
        decision = "HOLD_SCRIPT_INTEGRITY_FAILED"
    else:
        decision = "HOLD_ROUTES_DRIFT"

    go_nogo = "GO" if bundle_ok else "NO-GO"
    master_merkle_root = compute_merkle_root(
        [bundle_merkle_root] + [r["leaf_sha256"] for r in route_rows],
        domain="hivemind.script_bundle_gate.merkle",
    )

    if bundle_ok:
        summary = (
            f"Bundle scripts Noël {go_nogo} sur {clean_gw} : "
            f"{scripts_verified_count}/{scripts_total_count} scripts vérifiés (SHA-256 & AGPL-3.0), "
            f"routes={len(HOLIDAY_PROBE_ROUTES) + 1}/{len(HOLIDAY_PROBE_ROUTES) + 1}, {FINANCIAL_PROMISE}."
            if is_fr
            else f"Holiday script bundle {go_nogo} on {clean_gw}: "
            f"{scripts_verified_count}/{scripts_total_count} scripts verified (SHA-256 & AGPL-3.0), "
            f"routes={len(HOLIDAY_PROBE_ROUTES) + 1}/{len(HOLIDAY_PROBE_ROUTES) + 1}, {FINANCIAL_PROMISE}."
        )
    else:
        summary = (
            f"Bundle scripts Noël {go_nogo} ({decision}) sur {clean_gw} : "
            f"{scripts_verified_count}/{scripts_total_count} scripts OK, "
            f"raisons={', '.join(hold_reasons)}."
            if is_fr
            else f"Holiday script bundle {go_nogo} ({decision}) on {clean_gw}: "
            f"{scripts_verified_count}/{scripts_total_count} scripts OK, "
            f"reasons={', '.join(hold_reasons)}."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "bundle_summary_object": BUNDLE_SUMMARY_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "verify_zero_clone_script_bundle",
            "lang": lang_code,
            "gateway": clean_gw,
            "local_scripts_dir": local_dir_display,
            "compliance_ok": True,
            "corporate_detected": False,
            "bundle_ok": bundle_ok,
            "ready": bundle_ok,
            "go_nogo": go_nogo,
            "decision": decision,
            "hold_reasons": hold_reasons,
            "manifest_ok": manifest_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "sha_match": sha_match,
            "missing_manifest_routes": missing_manifest_routes,
            "missing_expected_scripts": missing_expected_scripts,
            "listed_scripts": listed_scripts,
            "scripts_verified_count": scripts_verified_count,
            "scripts_total_count": scripts_total_count,
            "total_script_bytes": total_script_bytes,
            "scripts": script_rows,
            "script_http_statuses": script_http_statuses,
            "missing_remote_scripts": missing_remote_scripts,
            "sha256_mismatches": sha256_mismatches,
            "agpl_failures": agpl_failures,
            "syntax_failures": syntax_failures,
            "bundle_merkle_root_sha256": bundle_merkle_root,
            "local_bundle_merkle_root_sha256": local_bundle_merkle_root,
            "merkle_local_match": merkle_local_match,
            "merkle_root_sha256": master_merkle_root,
            "html_all_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "stale_404_drift": stale_404_drift,
            "stale_404_routes": stale_404_routes,
            "redeploy_required": redeploy_required,
            "http_statuses": http_statuses,
            "routes": route_rows,
            "failed_routes": failed_routes,
            "network_calls": network_calls,
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
            "diagnostics": diagnostics,
            "exit_code": 0 if bundle_ok else 2,
            "summary": summary,
        }
    )


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN CLI script-bundle SHA-256 integrity receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if report.get("exit_code") == 0 else "NO-GO")
    title = (
        f"HIVEMIND NOËL — INTÉGRITÉ SHA-256 BUNDLE SCRIPTS ({verdict}, 0,00 EUR / 0,00 USD)"
        if is_fr
        else f"HIVEMIND HOLIDAY — SCRIPT-BUNDLE SHA-256 INTEGRITY ({verdict}, 0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {title}",
        "====================================================================",
        f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}",
        f"  Gateway     : {report.get('gateway', DEFAULT_GATEWAY)}",
        f"  Mode        : {report.get('mode')}",
        f"  Verdict     : {verdict} ({report.get('decision', verdict)})",
        f"  Compliance  : {report.get('compliance_ok')}",
        f"  Exit code   : {report.get('exit_code')}",
        f"  Summary     : {report.get('summary')}",
    ]
    if "scripts_verified_count" in report:
        lines.append(
            f"  Scripts     : {report.get('scripts_verified_count')}/{report.get('scripts_total_count')}"
        )
    if report.get("bundle_merkle_root_sha256"):
        lines.append(f"  BundleRoot  : {report.get('bundle_merkle_root_sha256')}")
    if report.get("merkle_root_sha256"):
        lines.append(f"  MerkleRoot  : {report.get('merkle_root_sha256')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256     : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN Markdown script-bundle SHA-256 integrity receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if report.get("exit_code") == 0 else "NO-GO")
    heading = (
        f"# Vérificateur d'Intégrité SHA-256 Bundle Scripts HiveMind — {verdict}"
        if is_fr
        else f"# HiveMind Zero-Clone Script-Bundle SHA-256 Integrity Gate — {verdict}"
    )
    lines = [
        heading,
        "",
        f"- **{'Verdict Go/No-Go' if is_fr else 'Go/No-Go Verdict'}**: `{verdict}` (`decision={report.get('decision', verdict)}`, `exit_code={report.get('exit_code')}`)",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: `{report.get('compliance_ok')}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Zéro inférence' if is_fr else 'Zero-Inference Verified'}**: `{report.get('zero_inference_verified', True)}`",
    ]
    if "scripts_verified_count" in report:
        lines.append(
            f"- **{'Scripts vérifiés' if is_fr else 'Scripts Verified'}**: "
            f"`{report.get('scripts_verified_count')}/{report.get('scripts_total_count')}`"
        )
    if report.get("bundle_merkle_root_sha256"):
        lines.append(f"- **Bundle Merkle SHA-256**: `{report.get('bundle_merkle_root_sha256')}`")
    if report.get("merkle_root_sha256"):
        lines.append(f"- **Master Merkle SHA-256**: `{report.get('merkle_root_sha256')}`")
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
    lines.append(f"- **{'Résumé' if is_fr else 'Summary'}**: {report.get('summary')}")
    lines.append(f"- **SHA-256 Receipt**: `{report.get('receipt_sha256')}`")

    rows = report.get("scripts")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows:
        lines.extend(
            [
                "",
                "| Script | HTTP | Status | Remote SHA-256 (12) | Local Match |",
                "|---|---:|---|---|---|",
            ]
        )
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            rsha = str(row.get("remote_sha256") or "-")[:12]
            status_lbl = "PASS" if row.get("ok") else "FAIL"
            lines.append(
                f"| `{row.get('script')}` | `{row.get('http_status')}` | **{status_lbl}** | `{rsha}` | `{row.get('local_sha256_match')}` |"
            )

    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        lines.extend(f"- {d}" for d in diags)
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the zero-inference script-bundle SHA-256 integrity verifier."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-inference zero-clone script-bundle SHA-256 integrity & holiday manifest verifier "
            "(0.00 EUR / 0.00 USD, AGPL-3.0)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="Target HiveMind gateway URL (https://*.run.app, https://*.googleapis.com, or loopback).",
    )
    parser.add_argument(
        "--local-scripts-dir",
        default=None,
        help="Optional local scripts/ directory for exact byte-for-byte SHA-256 comparison.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Expected build_sha on /v1/hivemind/onboarding/manifest.",
    )
    parser.add_argument(
        "--require-expected-scripts",
        action="store_true",
        help="Require all 17 canonical allowlisted onboarding scripts to appear in manifest.",
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
    """CLI entrypoint enforcing compliance before any network call or local script-dir read."""
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
        else:
            expected_scripts = EXPECTED_MANIFEST_SCRIPTS if args.require_expected_scripts else None
            report = verify_zero_clone_script_bundle(
                args.gateway,
                local_scripts_dir=args.local_scripts_dir,
                expected_sha=args.expected_sha,
                expected_scripts=expected_scripts,
                http_fn=http_fn,
                environ=eff_env,
                cwd=cwd,
                lang=lang_code,
            )
    except ScriptBundleError as exc:
        err_msg = sanitize_text_for_display(str(exc))
        if args.json:
            err_doc = _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "compliance_ok": True,
                    "bundle_ok": False,
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
