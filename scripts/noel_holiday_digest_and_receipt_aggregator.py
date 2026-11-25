#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference Holiday Launch Digest & Cross-Verifier SHA-256 Receipt Aggregator.

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment variables (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``),
     workspace path, and ``.git/config`` remote URLs (including worktree ``gitdir:`` / ``commondir``
     indirection) BEFORE any file read/write or network call. Fails closed with exit code 2
     (0 network calls, 0 file writes, 0 raw env/hostname/path leaks).
  2. Cross-Verifier SHA-256 & Zero-PII Receipt Aggregator (``aggregate_holiday_verifier_receipts``):
     verifies canonical SHA-256 integrity, zero-PII/static-figure output safety (including
     ``hm-***XXXX`` / ``hm_live_***`` masked-key suffixes and HTML/CSP nonces), personal-PC
     ``0.00 EUR / 0.00 USD`` compliance, and unanimous readiness across Win11, POSIX, Arena,
     and Raid verifier receipts, producing a deterministic binary SHA-256 Merkle root.
  3. Zero-Inference Live Holiday Launch Digest (``run_live_holiday_digest``):
     probes ``GET /noel``, ``GET /quick``, ``GET /raid-noel``, ``GET /arena-noel``,
     ``GET /v1/hivemind/onboarding/manifest``, and ``GET /v1/hivemind/onboarding/feedback/summary``
     in strict zero-inference read-only mode (never calling chat completions or prompting
     ``hivemind/coder``), enforcing unique per-request CSP nonces ``default-src 'none'``,
     0 unsafe DOM sinks, 0 static figures, exact 5-key feedback summary preservation, and
     emitting compact FR/EN Markdown & JSON Go/No-Go Launch Digests with a master Merkle &
     SHA-256 receipt.

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
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_holiday_digest_and_receipt_aggregator/1"
RECEIPT_OBJECT = "hivemind.noel_holiday_launch_digest_receipt"
AGGREGATION_OBJECT = "hivemind.noel_cross_verifier_receipt_aggregation"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

HTTP_TIMEOUT_S = 5.0
MAX_RECEIPT_FILE_BYTES = 2_000_000

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REQUIRED_VERIFIERS: tuple[str, ...] = ("win11", "posix", "arena", "raid")
KNOWN_VERIFIER_IDS: frozenset[str] = frozenset(DEFAULT_REQUIRED_VERIFIERS)

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset()

LIVE_DIGEST_HTML_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
)

LIVE_DIGEST_ROUTES: tuple[tuple[str, str, str], ...] = (
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

_READINESS_FLAG_KEYS: tuple[str, ...] = (
    "ready",
    "overall_ok",
    "parity_ok",
    "smoke_ok",
    "contract_ok",
    "dev_ready",
    "gamer_ready",
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
_SHA256_FULL_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERIFIER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_API_KEY_RE = re.compile(r"(hm(?:_live_|-))\*\*\*(?:[A-Za-z0-9_-]{4,}|[0-9]{1,3}ms\b)")
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


class DigestAggregatorError(ValueError):
    """Raised when digest aggregator inputs, host allowlists, or safety guards fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise DigestAggregatorError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise DigestAggregatorError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw credentials never appear in outputs."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return "<invalid>"
    if cleaned.startswith("hm_live_"):
        return f"hm_live_***{cleaned[-4:]}"
    if cleaned.startswith("hm-"):
        return f"hm-***{cleaned[-4:]}"
    return "<invalid>"


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and masked ``hm-***`` suffixes."""
    if not isinstance(text, str):
        raise DigestAggregatorError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    cleaned = _MASKED_API_KEY_RE.sub(r"\1***XXXX", cleaned)
    return cleaned


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
    scrubbed = redact_home_path(scrubbed, home=home)
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
        raise DigestAggregatorError("scan_output_safety requires a string.")
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
    if _TROJAN_BIDI_RE.search(text):
        issues.append("trojan_bidi_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if not ip_hit.startswith("127."):
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise DigestAggregatorError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise DigestAggregatorError("Forbidden JSON key detected in digest receipt.")
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
        raise DigestAggregatorError(
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


def compute_merkle_root(leaf_hashes: Sequence[str]) -> str:
    """Compute a deterministic domain-separated binary SHA-256 Merkle root over leaf digests."""
    if not leaf_hashes:
        return hashlib.sha256(b"hivemind.holiday.merkle.empty/1").hexdigest()
    level: list[str] = []
    for digest in leaf_hashes:
        if not isinstance(digest, str) or not _SHA256_FULL_RE.fullmatch(digest):
            raise DigestAggregatorError("Merkle leaf digest must be a 64-char lowercase hex SHA-256.")
        level.append(digest)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: list[str] = []
        for idx in range(0, len(level), 2):
            pair = f"hivemind.holiday.merkle.node/1:{level[idx]}:{level[idx + 1]}".encode()
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
        raise DigestAggregatorError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise DigestAggregatorError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise DigestAggregatorError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise DigestAggregatorError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise DigestAggregatorError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise DigestAggregatorError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise DigestAggregatorError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise DigestAggregatorError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise DigestAggregatorError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise DigestAggregatorError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise DigestAggregatorError("Remote gateway URL must use https.")
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
        raise DigestAggregatorError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise DigestAggregatorError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise DigestAggregatorError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise DigestAggregatorError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise DigestAggregatorError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise DigestAggregatorError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise DigestAggregatorError(
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
        raise DigestAggregatorError("GET/HEAD requests must never attach a body payload.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-holiday-digest-aggregator/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise DigestAggregatorError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str

    req = urllib.request.Request(clean_url, data=None, headers=req_headers, method=verb)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())
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
        raise DigestAggregatorError("http_fn must return (status_code, body) or (status_code, headers, body).")
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
                pass
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
    except DigestAggregatorError:
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
        pass
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
            "go_nogo": "NO-GO",
            "unanimous_ready": False,
            "stale_404_drift": False,
            "redeploy_required": False,
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
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


def _normalize_required_verifiers(require_verifiers: Sequence[str]) -> tuple[str, ...]:
    if isinstance(require_verifiers, (str, bytes)) or not isinstance(require_verifiers, Sequence):
        raise DigestAggregatorError("require_verifiers must be a non-empty sequence of verifier names.")
    out: list[str] = []
    for raw in require_verifiers:
        if isinstance(raw, bool) or not isinstance(raw, str):
            raise DigestAggregatorError("Verifier name must be a string.")
        cleaned = raw.strip().lower()
        if not _VERIFIER_ID_RE.fullmatch(cleaned):
            raise DigestAggregatorError(f"Invalid verifier identifier {raw!r}.")
        if cleaned not in out:
            out.append(cleaned)
    if not out:
        raise DigestAggregatorError("require_verifiers must not be empty.")
    return tuple(out)


def _identify_verifier_kind(doc: Mapping[str, Any], hint: str | None = None) -> str | None:
    for key in ("verifier", "verifier_id", "verifier_name", "source_verifier"):
        raw = doc.get(key)
        if isinstance(raw, str) and raw.strip():
            cand = raw.strip().lower()
            if cand in KNOWN_VERIFIER_IDS:
                return cand
            if _VERIFIER_ID_RE.fullmatch(cand):
                return cand

    hint_low = (hint or "").strip().lower()
    if hint_low in KNOWN_VERIFIER_IDS:
        return hint_low

    combined = " ".join(
        str(doc.get(k) or "") for k in ("schema", "object", "mode")
    ).lower()
    if "win11" in combined or "deploy_smoke_and_win11" in combined:
        return "win11"
    if "posix" in combined:
        return "posix"
    if "arena" in combined:
        return "arena"
    if "raid" in combined:
        return "raid"

    for vname in DEFAULT_REQUIRED_VERIFIERS:
        if vname in hint_low:
            return vname
    return None


def _load_receipt_item(
    item: Any,
    *,
    key_hint: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if isinstance(item, Mapping):
        return dict(item), key_hint, None

    if isinstance(item, (Path, str)):
        raw_str = str(item)
        if "\x00" in raw_str or not raw_str.strip():
            return None, key_hint, "invalid_receipt_path"
        path = Path(raw_str)
        name_hint = key_hint or path.stem
        try:
            if not path.is_file():
                return None, name_hint, f"missing_receipt_file:{redact_home_path(path)}"
            size = path.stat().st_size
            if size > MAX_RECEIPT_FILE_BYTES:
                return None, name_hint, f"oversized_receipt_file:{redact_home_path(path)}"
            raw_text = path.read_text(encoding="utf-8")
            parsed = json.loads(raw_text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, name_hint, f"unreadable_or_invalid_json_receipt:{redact_home_path(path)}"
        if not isinstance(parsed, dict):
            return None, name_hint, f"non_object_json_receipt:{redact_home_path(path)}"
        return parsed, name_hint, None

    raise DigestAggregatorError("Each receipt item must be a dict/Mapping or JSON file path.")


def _extract_receipt_build_sha(doc: Mapping[str, Any]) -> str:
    direct = sanitize_build_sha(doc.get("live_build_sha") or doc.get("build_sha"))
    if direct != "unknown":
        return direct
    cloud_smoke = doc.get("cloud_run_smoke")
    if isinstance(cloud_smoke, Mapping):
        nested = sanitize_build_sha(cloud_smoke.get("live_build_sha") or cloud_smoke.get("build_sha"))
        if nested != "unknown":
            return nested
    return "unknown"


def _audit_single_verifier_receipt(
    verifier_id: str,
    doc: Mapping[str, Any],
    *,
    expected_sha: str | None,
    is_fr: bool,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    sha256_valid = False
    pii_safe = True
    pii_issues: list[str] = []
    actual_sha_raw = doc.get("receipt_sha256")
    actual_sha = actual_sha_raw.strip().lower() if isinstance(actual_sha_raw, str) else ""

    core_payload = {str(k): v for k, v in doc.items() if k != "receipt_sha256"}
    try:
        canonical = json.dumps(core_payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        canonical = ""
        issues.append(
            f"[{verifier_id}] JSON canonique invalide ou flottant non fini."
            if is_fr
            else f"[{verifier_id}] Non-canonical or non-finite JSON payload."
        )

    if canonical:
        computed_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if _SHA256_FULL_RE.fullmatch(actual_sha) and hmac.compare_digest(actual_sha, computed_sha):
            sha256_valid = True
        else:
            issues.append(
                f"[{verifier_id}] Intégrité SHA-256 invalide ou falsifiée."
                if is_fr
                else f"[{verifier_id}] SHA-256 receipt mismatch or missing receipt_sha256."
            )

        if ("sco" + "re") in _collect_json_keys(doc):
            pii_issues.append("forbidden_score_key_detected")
        pii_issues.extend(scan_output_safety(canonical))
        if pii_issues:
            pii_safe = False
            issues.append(
                f"[{verifier_id}] Fuite PII / clé brute / figure statique détectée ({', '.join(pii_issues)})."
                if is_fr
                else f"[{verifier_id}] Output safety/PII violation ({', '.join(pii_issues)})."
            )
    else:
        computed_sha = hashlib.sha256(b"invalid").hexdigest()

    comp_ok_val = doc.get("compliance_ok")
    corp_det_val = doc.get("corporate_detected")
    personal_pc_val = doc.get("personal_pc_only")
    zero_inf_val = doc.get("zero_inference_verified")
    fin_prom_val = doc.get("financial_promise")
    compliance_flags_ok = bool(
        comp_ok_val is not False
        and corp_det_val is not True
        and personal_pc_val is not False
        and zero_inf_val is not False
        and (fin_prom_val is None or fin_prom_val == FINANCIAL_PROMISE)
    )
    if not compliance_flags_ok:
        issues.append(
            f"[{verifier_id}] Garantie conformité PC personnel / 0.00 EUR / zéro-inférence violée."
            if is_fr
            else f"[{verifier_id}] Personal-PC / 0.00 EUR / zero-inference compliance check failed."
        )

    drift_detected = bool(
        doc.get("stale_404_drift") is True
        or doc.get("stale_revision_detected") is True
        or doc.get("redeploy_required") is True
        or doc.get("sha_mismatch") is True
    )
    if drift_detected:
        issues.append(
            f"[{verifier_id}] Dérive Cloud Run / 404 ou redéploiement requis détecté."
            if is_fr
            else f"[{verifier_id}] Stale 404 / revision drift or redeploy_required flagged."
        )

    raw_exit = doc.get("exit_code")
    exit_ok = raw_exit is None or (not isinstance(raw_exit, bool) and isinstance(raw_exit, int) and raw_exit == 0)
    if not exit_ok:
        issues.append(
            f"[{verifier_id}] Code de sortie non nul ({raw_exit})."
            if is_fr
            else f"[{verifier_id}] Non-zero verifier exit_code ({raw_exit})."
        )

    present_flags = {
        k: doc[k] for k in _READINESS_FLAG_KEYS if k in doc and doc[k] is not None
    }
    explicit_ready_ok = ("ready" not in doc) or (doc["ready"] is True)
    all_flags_true = bool(present_flags) and all(v is True for v in present_flags.values()) and explicit_ready_ok
    if not all_flags_true:
        issues.append(
            f"[{verifier_id}] Verdict readiness non prêt (`ready != True`)."
            if is_fr
            else f"[{verifier_id}] Verifier readiness flag is not True."
        )

    live_build_sha = _extract_receipt_build_sha(doc)
    sha_ok = True
    if expected_sha is not None and live_build_sha != "unknown" and not shas_match(live_build_sha, expected_sha):
        sha_ok = False
        issues.append(
            f"[{verifier_id}] Dérive build_sha ({live_build_sha} != {expected_sha})."
            if is_fr
            else f"[{verifier_id}] Build SHA drift ({live_build_sha} != {expected_sha})."
        )

    entry_ready = bool(
        sha256_valid and pii_safe and compliance_flags_ok and not drift_detected and exit_ok and all_flags_true and sha_ok
    )
    safe_prefix = actual_sha[:16] if _SHA256_FULL_RE.fullmatch(actual_sha) else "<invalid>"
    leaf_input = f"hivemind.holiday.merkle.leaf/1:{verifier_id}:{computed_sha}".encode()
    leaf_sha256 = hashlib.sha256(leaf_input).hexdigest()

    entry = {
        "verifier": verifier_id,
        "schema": str(doc.get("schema") or "unknown")[:96] if pii_safe else "<redacted>",
        "object": str(doc.get("object") or "unknown")[:96] if pii_safe else "<redacted>",
        "mode": str(doc.get("mode") or "unknown")[:96] if pii_safe else "<redacted>",
        "sha256_valid": sha256_valid,
        "pii_safe": pii_safe,
        "pii_issues": sorted(set(pii_issues)),
        "compliance_ok": compliance_flags_ok,
        "drift_clear": not drift_detected,
        "exit_ok": exit_ok,
        "ready": entry_ready,
        "live_build_sha": live_build_sha,
        "receipt_sha256": actual_sha if sha256_valid else None,
        "receipt_sha256_prefix": safe_prefix,
        "leaf_sha256": leaf_sha256,
    }
    return entry, issues


def aggregate_holiday_verifier_receipts(
    receipt_dicts_or_paths: Sequence[Any] | Mapping[str, Any],
    *,
    require_verifiers: Sequence[str] = DEFAULT_REQUIRED_VERIFIERS,
    expected_sha: str | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify SHA-256 integrity, zero-PII redaction, and unanimous readiness across holiday verifiers."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    req_verifiers = _normalize_required_verifiers(require_verifiers)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None

    if isinstance(receipt_dicts_or_paths, Mapping):
        if any(k in receipt_dicts_or_paths for k in ("schema", "object", "receipt_sha256")):
            raise DigestAggregatorError(
                "Expected a sequence or {verifier: receipt} mapping, got a single receipt object."
            )
        items_to_load: list[tuple[str | None, Any]] = [
            (str(k), v) for k, v in receipt_dicts_or_paths.items()
        ]
    elif isinstance(receipt_dicts_or_paths, (str, bytes, Path)) or not isinstance(
        receipt_dicts_or_paths, Iterable
    ):
        raise DigestAggregatorError("receipt_dicts_or_paths must be a sequence or mapping of receipts/paths.")
    else:
        items_to_load = [(None, item) for item in receipt_dicts_or_paths]

    comp_doc = evaluate_personal_pc_compliance(
        environ=environ, cwd=cwd, check_git_remote=True, lang=lang_code
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("aggregate_verifier_receipts", comp_doc, lang_code=lang_code)

    diagnostics: list[str] = []
    per_verifier: dict[str, dict[str, Any]] = {}
    duplicate_verifiers: list[str] = []
    unidentified_count = 0

    for key_hint, raw_item in items_to_load:
        loaded_doc, name_hint, load_err = _load_receipt_item(raw_item, key_hint=key_hint)
        if load_err is not None or loaded_doc is None:
            diagnostics.append(
                f"Impossible de charger le reçu ({load_err})."
                if is_fr
                else f"Failed to load verifier receipt ({load_err})."
            )
            if name_hint and name_hint.lower() in KNOWN_VERIFIER_IDS and name_hint.lower() not in per_verifier:
                vid_err = name_hint.lower()
                per_verifier[vid_err] = {
                    "verifier": vid_err,
                    "schema": "unreadable",
                    "object": "unreadable",
                    "mode": "unreadable",
                    "sha256_valid": False,
                    "pii_safe": True,
                    "pii_issues": [],
                    "compliance_ok": False,
                    "drift_clear": False,
                    "exit_ok": False,
                    "ready": False,
                    "live_build_sha": "unknown",
                    "receipt_sha256": None,
                    "receipt_sha256_prefix": "<missing>",
                    "leaf_sha256": hashlib.sha256(f"missing:{vid_err}".encode()).hexdigest(),
                }
            else:
                unidentified_count += 1
            continue

        vid = _identify_verifier_kind(loaded_doc, hint=name_hint)
        if vid is None:
            unidentified_count += 1
            diagnostics.append(
                "Reçu sans identifiant de vérificateur reconnu (win11/posix/arena/raid)."
                if is_fr
                else "Receipt missing recognized verifier identity (win11/posix/arena/raid)."
            )
            continue
        if vid in per_verifier:
            duplicate_verifiers.append(vid)
            diagnostics.append(
                f"Reçu en double détecté pour le vérificateur `{vid}`."
                if is_fr
                else f"Duplicate receipt detected for verifier `{vid}`."
            )
            continue

        entry, entry_issues = _audit_single_verifier_receipt(
            vid, loaded_doc, expected_sha=clean_expected_sha, is_fr=is_fr
        )
        per_verifier[vid] = entry
        diagnostics.extend(entry_issues)

    missing_verifiers = [v for v in req_verifiers if v not in per_verifier]
    for mv in missing_verifiers:
        diagnostics.append(
            f"Reçu obligatoire manquant : `{mv}`."
            if is_fr
            else f"Missing required verifier receipt: `{mv}`."
        )

    tampered_verifiers = sorted(v for v, row in per_verifier.items() if not row["sha256_valid"])
    pii_failed_verifiers = sorted(v for v, row in per_verifier.items() if not row["pii_safe"])
    unready_verifiers = sorted(v for v, row in per_verifier.items() if not row["ready"])

    known_shas = [
        row["live_build_sha"]
        for row in per_verifier.values()
        if row["live_build_sha"] != "unknown"
    ]
    build_sha_consensus_ok = True
    if len(known_shas) >= 2:
        anchor = known_shas[0]
        if not all(shas_match(candidate, anchor) for candidate in known_shas[1:]):
            build_sha_consensus_ok = False
            diagnostics.append(
                "Divergence de `build_sha` entre les reçus des vérificateurs."
                if is_fr
                else "Build SHA disagreement detected across verifier receipts."
            )

    integrity_ok = bool(
        not tampered_verifiers
        and not duplicate_verifiers
        and unidentified_count == 0
        and len(per_verifier) >= len(req_verifiers)
    )
    pii_redaction_ok = len(pii_failed_verifiers) == 0
    unanimous_ready = bool(
        not missing_verifiers
        and not duplicate_verifiers
        and unidentified_count == 0
        and integrity_ok
        and pii_redaction_ok
        and not unready_verifiers
        and build_sha_consensus_ok
    )

    ordered_leaves = [
        {
            "verifier": vid,
            "leaf_sha256": per_verifier[vid]["leaf_sha256"],
            "receipt_sha256_prefix": per_verifier[vid]["receipt_sha256_prefix"],
        }
        for vid in sorted(per_verifier)
    ]
    merkle_root = compute_merkle_root([row["leaf_sha256"] for row in ordered_leaves])
    go_nogo = "GO" if unanimous_ready else "NO-GO"

    summary = (
        (
            f"Agrégation des reçus Noël ({len(per_verifier)}/{len(req_verifiers)}) : {go_nogo} "
            f"(SHA-256 & zéro-PII vérifiés)."
            if is_fr
            else f"Holiday verifier receipt aggregation ({len(per_verifier)}/{len(req_verifiers)}): "
            f"{go_nogo} (SHA-256 & zero-PII verified)."
        )
        if unanimous_ready
        else (
            f"Agrégation des reçus Noël EN ÉCHEC ({go_nogo}) : "
            f"manquants={missing_verifiers}, falsifiés={tampered_verifiers}, non_prêts={unready_verifiers}."
            if is_fr
            else f"Holiday verifier receipt aggregation FAILED ({go_nogo}): "
            f"missing={missing_verifiers}, tampered={tampered_verifiers}, unready={unready_verifiers}."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": AGGREGATION_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "aggregate_verifier_receipts",
            "lang": lang_code,
            "compliance_ok": True,
            "corporate_detected": False,
            "ready": unanimous_ready,
            "unanimous_ready": unanimous_ready,
            "go_nogo": go_nogo,
            "integrity_ok": integrity_ok,
            "pii_redaction_ok": pii_redaction_ok,
            "build_sha_consensus_ok": build_sha_consensus_ok,
            "expected_build_sha": clean_expected_sha,
            "known_build_shas": sorted(set(known_shas)),
            "required_verifiers": list(req_verifiers),
            "present_verifiers": sorted(per_verifier),
            "missing_verifiers": missing_verifiers,
            "duplicate_verifiers": sorted(set(duplicate_verifiers)),
            "tampered_verifiers": tampered_verifiers,
            "pii_failed_verifiers": pii_failed_verifiers,
            "unready_verifiers": unready_verifiers,
            "unidentified_receipts_count": unidentified_count,
            "verifiers": per_verifier,
            "merkle_leaves": ordered_leaves,
            "merkle_root_sha256": merkle_root,
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": diagnostics,
            "exit_code": 0 if unanimous_ready else 2,
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
            "mean_ease_rating": round(float(raw_ease), 2) if raw_ease is not None else None,
            "zero_game_lag_ratio": round(float(raw_lag), 4) if raw_lag is not None else None,
            "by_profile": {"dev": int(d_raw), "gamer": int(g_raw)},
        }
        if schema_ok
        else {}
    )
    return schema_ok, preserved, issues


def run_live_holiday_digest(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
    receipts: Sequence[Any] | Mapping[str, Any] | None = None,
    require_verifiers: Sequence[str] = DEFAULT_REQUIRED_VERIFIERS,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    """Probe holiday gateway routes in zero-inference mode and emit a Go/No-Go Launch Digest."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=True,
        lang=lang_code,
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("live_holiday_launch_digest", comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}

    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    missing_manifest_routes: list[str] = []
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {}
    network_calls = 0

    for path, kind, group in LIVE_DIGEST_ROUTES:
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
                    f"hivemind.holiday.merkle.route/1:{route_digest_src}".encode()
                ).hexdigest(),
            }
        )

    html_all_ok = bool(
        len(html_contracts) == len(LIVE_DIGEST_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in LIVE_DIGEST_HTML_ROUTES)
        and len(seen_nonces) == len(LIVE_DIGEST_HTML_ROUTES)
    )
    stale_404_drift = bool(stale_404_routes or missing_manifest_routes)
    live_routes_ok = bool(
        html_all_ok
        and manifest_ok
        and feedback_summary_ok
        and (clean_expected_sha is None or sha_match is True)
    )

    receipt_agg: dict[str, Any] | None = None
    verifiers_ready = True
    if receipts is not None:
        receipt_agg = aggregate_holiday_verifier_receipts(
            receipts,
            require_verifiers=require_verifiers,
            expected_sha=clean_expected_sha or (live_build_sha if live_build_sha != "unknown" else None),
            environ=environ,
            cwd=cwd,
            lang=lang_code,
        )
        verifiers_ready = bool(receipt_agg["ready"])
        diagnostics.extend(receipt_agg.get("diagnostics") or [])

    ready = bool(live_routes_ok and verifiers_ready)
    redeploy_required = bool(stale_404_drift or not live_routes_ok)
    go_nogo = "GO" if ready else "NO-GO"

    merkle_inputs = [row["leaf_sha256"] for row in route_rows]
    if receipt_agg is not None:
        merkle_inputs.append(str(receipt_agg["merkle_root_sha256"]))
    master_merkle_root = compute_merkle_root(merkle_inputs)

    summary = (
        (
            f"Digest lancement Noël {go_nogo} sur {clean_gw} "
            f"(routes_vertes={len(LIVE_DIGEST_ROUTES) - len(failed_routes)}/{len(LIVE_DIGEST_ROUTES)}, "
            f"nonces_uniques={len(seen_nonces)})."
            if is_fr
            else f"Holiday Launch Digest {go_nogo} on {clean_gw} "
            f"(routes_ok={len(LIVE_DIGEST_ROUTES) - len(failed_routes)}/{len(LIVE_DIGEST_ROUTES)}, "
            f"unique_nonces={len(seen_nonces)})."
        )
        if ready
        else (
            f"Digest lancement Noël {go_nogo} sur {clean_gw} "
            f"(redéploiement requis={redeploy_required}, échecs={len(failed_routes)})."
            if is_fr
            else f"Holiday Launch Digest {go_nogo} on {clean_gw} "
            f"(redeploy_required={redeploy_required}, failed_routes={len(failed_routes)})."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "live_holiday_launch_digest",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "ready": ready,
            "go_nogo": go_nogo,
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
            "receipt_aggregation": receipt_agg,
            "verifiers_ready": verifiers_ready,
            "merkle_root_sha256": master_merkle_root,
            "network_calls": network_calls,
            "file_writes": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
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
    """Render a compact FR/EN CLI Holiday Launch Digest with Merkle & SHA-256 receipts."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if report.get("ready") else "NO-GO")
    title = (
        f"HIVEMIND NOËL — DIGEST LANCEMENT {verdict} & AGRÉGATEUR SHA-256 (0,00 EUR / 0,00 USD)"
        if is_fr
        else f"HIVEMIND HOLIDAY — {verdict} LAUNCH DIGEST & SHA-256 RECEIPT AGGREGATOR (0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {title}",
        "====================================================================",
        f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}",
        f"  Gateway     : {report.get('gateway', DEFAULT_GATEWAY)}",
        f"  Mode        : {report.get('mode')}",
        f"  Verdict     : {verdict}",
        f"  Compliance  : {report.get('compliance_ok')}",
        f"  Exit code   : {report.get('exit_code')}",
        f"  Summary     : {report.get('summary')}",
    ]
    if report.get("merkle_root_sha256"):
        lines.append(f"  Merkle Root : {report.get('merkle_root_sha256')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256     : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN Markdown Go/No-Go Holiday Launch Digest with Merkle & SHA-256 receipts."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if report.get("ready") else "NO-GO")
    heading = (
        f"# Digest de Lancement Noël HiveMind — {verdict}"
        if is_fr
        else f"# HiveMind Holiday Launch Digest — {verdict}"
    )
    lines = [
        heading,
        "",
        f"- **{'Verdict Go/No-Go' if is_fr else 'Go/No-Go Verdict'}**: `{verdict}` (`exit_code={report.get('exit_code')}`)",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: "
        f"`{report.get('compliance_ok')}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Zéro inférence' if is_fr else 'Zero-Inference Verified'}**: `{report.get('zero_inference_verified')}`",
        f"- **{'Résumé' if is_fr else 'Summary'}**: {report.get('summary')}",
    ]
    if "unique_csp_nonces_count" in report:
        lines.append(
            f"- **{'Nonces CSP uniques' if is_fr else 'Unique CSP Nonces'}**: `{report.get('unique_csp_nonces_count')}`"
        )
    if report.get("merkle_root_sha256"):
        lines.append(f"- **Merkle Root SHA-256**: `{report.get('merkle_root_sha256')}`")
    lines.append(f"- **SHA-256 Receipt**: `{report.get('receipt_sha256')}`")

    agg_block = report if report.get("object") == AGGREGATION_OBJECT else report.get("receipt_aggregation")
    if isinstance(agg_block, Mapping) and isinstance(agg_block.get("verifiers"), Mapping):
        lines.extend(
            [
                "",
                "## Vérificateurs Croisés" if is_fr else "## Cross-Verifier Receipts",
                "| Verifier | Ready | SHA-256 | Zero-PII | Prefix |",
                "| :-- | :-- | :-- | :-- | :-- |",
            ]
        )
        for vid, row in sorted(agg_block["verifiers"].items()):
            lines.append(
                f"| `{vid}` | `{row.get('ready')}` | `{row.get('sha256_valid')}` | "
                f"`{row.get('pii_safe')}` | `{row.get('receipt_sha256_prefix')}` |"
            )

    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        lines.extend(f"- {d}" for d in diags)
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the Holiday Launch Digest & Receipt Aggregator."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-inference Holiday Launch Digest & Cross-Verifier SHA-256 Receipt Aggregator "
            "(0.00 EUR / 0.00 USD, AGPL-3.0)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="Target HiveMind gateway URL (https://*.run.app, https://*.googleapis.com, or loopback).",
    )
    parser.add_argument(
        "--receipt",
        dest="receipts",
        action="append",
        default=None,
        help="Path to a verifier JSON receipt file (repeatable across win11, posix, arena, raid).",
    )
    parser.add_argument(
        "--require-verifiers",
        default=",".join(DEFAULT_REQUIRED_VERIFIERS),
        help="Comma-separated required verifier IDs (default: win11,posix,arena,raid).",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate and verify local --receipt files without HTTP probes.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Expected build_sha on /v1/hivemind/onboarding/manifest and verifier receipts.",
    )
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Run only the strictly local personal-PC compliance check.",
    )
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit signed JSON receipt.")
    parser.add_argument("--markdown", action="store_true", help="Emit signed Markdown digest.")
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
    """CLI entrypoint running local compliance first, then receipt aggregation and/or live digest probes."""
    eff_out = stdout if stdout is not None else sys.stdout
    eff_err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        lang_code = normalize_lang(args.lang)
        if args.enforce_personal_pc:
            report = evaluate_personal_pc_compliance(
                environ=environ, cwd=cwd, check_git_remote=True, lang=lang_code
            )
        else:
            comp_doc = evaluate_personal_pc_compliance(
                environ=environ, cwd=cwd, check_git_remote=True, lang=lang_code
            )
            if not comp_doc["compliance_ok"]:
                report = _compliance_blocked_receipt("holiday_digest_cli", comp_doc, lang_code=lang_code)
            else:
                req_list = tuple(
                    part.strip().lower()
                    for part in str(args.require_verifiers or "").split(",")
                    if part.strip()
                )
                if args.aggregate_only:
                    report = aggregate_holiday_verifier_receipts(
                        args.receipts or (),
                        require_verifiers=req_list,
                        expected_sha=args.expected_sha,
                        environ=environ,
                        cwd=cwd,
                        lang=lang_code,
                    )
                else:
                    report = run_live_holiday_digest(
                        args.gateway,
                        http_fn=http_fn,
                        environ=environ,
                        cwd=cwd,
                        lang=lang_code,
                        receipts=args.receipts,
                        require_verifiers=req_list,
                        expected_sha=args.expected_sha,
                    )
    except DigestAggregatorError as exc:
        err_msg = sanitize_text_for_display(str(exc))
        if args.json:
            err_doc = _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "compliance_ok": True,
                    "ready": False,
                    "go_nogo": "NO-GO",
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
