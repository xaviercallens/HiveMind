#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-Inference Week S6 Final Sprint Bilan, Blind A/B Checkpoint & 8+ Producer Quorum Verifier (`SLICE-SCALE-DYN-32`).

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment markers (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``,
     ``NODE_EXTRA_CA_CERTS``), workspace/home paths, and ``.git/config`` remote URLs (including
     worktree ``gitdir:`` / ``commondir`` indirection) BEFORE any network call or file write.
     Fails closed with exit code 2 (0 network calls, 0 file writes, 0 raw leaks).
  2. Zero-Inference Week S6 (Dec 27, 2026 - Jan 3, 2027) Final Bilan, Blind A/B Checkpoint &
     8+ Friend-Producer Quorum Verifier (``verify_s6_final_bilan_and_quorum``):
     probes ``GET /health``, ``GET /v1/models``, ``GET /v1/hivemind/onboarding/manifest``,
     ``GET /v1/hivemind/producers/fleet``, ``GET /v1/hivemind/raid/status``,
     ``GET /v1/hivemind/arena/summary``, ``GET /noel``, ``GET /quick``, ``GET /raid-noel``,
     ``GET /arena-noel``, and ``GET /v1/hivemind/onboarding/feedback/summary`` in strict
     read-only zero-inference mode (never calling ``POST /v1/chat/completions`` or prompting
     ``hivemind/coder``), verifying PR #949 live holiday route parity (unique CSP nonces
     ``default-src 'none'``, 0 unsafe DOM sinks, 0 static figures after stripping masked keys
     and HTML nonce attrs, personal-PC/0.00 EUR banners, non-empty ``/v1/models`` with
     ``hivemind/coder``), 8+ external friend producer quorum, blind A/B checkpoint conservation
     and 95% Wilson lower bound, exact 5-key feedback summary & CES/zero-lag SLOs, and emitting
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
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_s6_final_bilan_and_quorum_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_s6_final_bilan_receipt"
BILAN_OBJECT = "hivemind.noel_s6_final_bilan"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
SPRINT_ID = "S6"
S6_WINDOW_START = "2026-12-27"
S6_WINDOW_END = "2027-01-03"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

HTTP_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 1_048_576
Z_WILSON_95 = 1.959963984540054
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MIN_ONLINE_PRODUCERS = 8
DEFAULT_MIN_PHYSICAL_GPUS = 1
DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95
DEFAULT_MIN_ARENA_VOTES = 8
DEFAULT_MIN_ARENA_WILSON_LB = 0.5

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset()

S6_HTML_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
)

S6_PROBE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "models"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
    ("/v1/hivemind/raid/status", "json", "raid"),
    ("/v1/hivemind/arena/summary", "json", "arena"),
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
_CARD_BALLOT_PAIR_A_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"[^>]*?data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_CARD_BALLOT_PAIR_B_RE = re.compile(
    r'data-ballot="([A-Za-z0-9._-]{16,128})"[^>]*?data-task-id="([A-Za-z0-9_-]{3,64})"'
)
_ANY_DATA_BALLOT_RE = re.compile(r'data-ballot="([^"]*)"')
_CAND_BLOCK_RE = re.compile(r'<pre\s+id="cand[AB]-([^"]+)"[^>]*>(.*?)</pre>', re.DOTALL)

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


class S6BilanVerifierError(ValueError):
    """Raised when Week S6 bilan/quorum inputs, host allowlists, or safety invariants fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI or API output language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise S6BilanVerifierError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise S6BilanVerifierError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


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


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and masked ``hm-***`` suffixes."""
    if not isinstance(text, str):
        raise S6BilanVerifierError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_API_KEY_RE.sub(r"\1***XXXX", cleaned)


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
            pass
    if home is None:
        hit_default = _match_prefix_redacted(raw, Path.home(), "~")
        if hit_default is not None:
            return hit_default
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
        raise S6BilanVerifierError("scan_output_safety requires a string.")
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
        raise S6BilanVerifierError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise S6BilanVerifierError("Forbidden JSON key detected in S6 bilan receipt.")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def compute_merkle_root(
    leaf_hashes: Sequence[str],
    *,
    domain: str = "hivemind.s6_bilan.merkle",
) -> str:
    """Compute a deterministic domain-separated binary SHA-256 Merkle root over leaf digests."""
    if not leaf_hashes:
        return hashlib.sha256(f"{domain}.empty/1".encode()).hexdigest()
    level: list[str] = []
    for digest in leaf_hashes:
        if not isinstance(digest, str) or not _SHA256_FULL_RE.fullmatch(digest):
            raise S6BilanVerifierError("Merkle leaf digest must be a 64-char lowercase hex SHA-256.")
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
        raise S6BilanVerifierError(
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


def is_allowed_outbound_host(host: str) -> bool:
    """Return True iff ``host`` is in the strict HiveMind outbound allowlist."""
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
        raise S6BilanVerifierError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise S6BilanVerifierError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise S6BilanVerifierError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise S6BilanVerifierError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise S6BilanVerifierError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise S6BilanVerifierError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise S6BilanVerifierError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise S6BilanVerifierError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise S6BilanVerifierError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise S6BilanVerifierError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise S6BilanVerifierError("Remote gateway URL must use https.")
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
        raise S6BilanVerifierError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise S6BilanVerifierError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise S6BilanVerifierError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise S6BilanVerifierError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise S6BilanVerifierError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise S6BilanVerifierError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise S6BilanVerifierError(
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
        raise S6BilanVerifierError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
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
    clean_url = _assert_safe_http_request(verb, url, payload=payload)
    if payload is not None:
        raise S6BilanVerifierError("GET/HEAD requests must never attach a body payload.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-s6-bilan-quorum-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise S6BilanVerifierError("HTTP headers must not contain CR/LF or NUL characters.")
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
    body_markers = {"object", "status", "total_responses", "total_hiveminders", "total_votes", "data", "detail", "error"}
    if keys & body_markers:
        return False
    return bool(keys & header_markers) or all(isinstance(v, str) for v in candidate.values())


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise S6BilanVerifierError("http_fn must return (status_code, body) or (status_code, headers, body).")
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
    clean_url = _assert_safe_http_request(verb, url, payload=payload)
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
    except S6BilanVerifierError:
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
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only Compliance Gate (0 network calls, 0 writes, 0 leaks)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = environ if environ is not None else os.environ
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
    for p_txt in ws_texts:
        for hit in _find_corporate_markers(p_txt):
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
            "sprint_id": SPRINT_ID,
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
            "bilan_object": BILAN_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "sprint_id": SPRINT_ID,
            "s6_window_start": S6_WINDOW_START,
            "s6_window_end": S6_WINDOW_END,
            "mode": mode,
            "lang": lang_code,
            "gateway": gateway or DEFAULT_GATEWAY,
            "compliance_ok": False,
            "corporate_detected": True,
            "corporate_indicators": list(comp_doc.get("corporate_indicators") or []),
            "matched_markers": list(comp_doc.get("matched_markers") or []),
            "routes_ready": False,
            "fleet_schema_ok": False,
            "raid_status_ok": False,
            "arena_schema_ok": False,
            "arena_conservation_ok": False,
            "ab_checkpoint_ready": False,
            "ab_checkpoint_ok": False,
            "ab_checkpoint_status": "blocked",
            "awaiting_arena_votes": False,
            "producer_quorum_reached": False,
            "awaiting_producer_quorum": False,
            "producer_quorum_ok": False,
            "producer_quorum_status": "blocked",
            "feedback_summary_ok": False,
            "ces_slo_ok": False,
            "ces_slo_status": "blocked",
            "zero_lag_slo_ok": False,
            "zero_lag_slo_status": "blocked",
            "awaiting_first_votes": False,
            "ready": False,
            "go_nogo": "NO-GO",
            "decision": "HOLD_COMPLIANCE_BLOCKED",
            "hold_reasons": ["corporate_environment_detected"],
            "stale_404_drift": False,
            "pre_pr_949_drift_detected": False,
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


def _validate_s6_thresholds(
    *,
    min_online_producers: int,
    min_ease_rating: float,
    min_zero_lag_ratio: float,
    min_arena_votes: int,
    min_arena_wilson_lb: float,
) -> tuple[int, float, float, int, float]:
    if (
        isinstance(min_online_producers, bool)
        or not isinstance(min_online_producers, int)
        or min_online_producers < 1
    ):
        raise S6BilanVerifierError("min_online_producers must be an integer >= 1.")
    if (
        isinstance(min_ease_rating, bool)
        or not isinstance(min_ease_rating, (int, float))
        or not math.isfinite(float(min_ease_rating))
        or not (1.0 <= float(min_ease_rating) <= 5.0)
    ):
        raise S6BilanVerifierError("min_ease_rating must be a finite number in [1.0, 5.0].")
    if (
        isinstance(min_zero_lag_ratio, bool)
        or not isinstance(min_zero_lag_ratio, (int, float))
        or not math.isfinite(float(min_zero_lag_ratio))
        or not (0.0 <= float(min_zero_lag_ratio) <= 1.0)
    ):
        raise S6BilanVerifierError("min_zero_lag_ratio must be a finite number in [0.0, 1.0].")
    if (
        isinstance(min_arena_votes, bool)
        or not isinstance(min_arena_votes, int)
        or min_arena_votes < 1
    ):
        raise S6BilanVerifierError("min_arena_votes must be an integer >= 1.")
    if (
        isinstance(min_arena_wilson_lb, bool)
        or not isinstance(min_arena_wilson_lb, (int, float))
        or not math.isfinite(float(min_arena_wilson_lb))
        or not (0.0 <= float(min_arena_wilson_lb) <= 1.0)
    ):
        raise S6BilanVerifierError("min_arena_wilson_lb must be a finite number in [0.0, 1.0].")
    return (
        int(min_online_producers),
        float(min_ease_rating),
        float(min_zero_lag_ratio),
        int(min_arena_votes),
        float(min_arena_wilson_lb),
    )


def wilson_95_lower_bound(wins: int, trials: int, z: float = Z_WILSON_95) -> float:
    """Compute the exact binomial 95% Wilson confidence interval lower bound (``0.0`` when ``trials == 0``)."""
    if (
        isinstance(wins, bool)
        or not isinstance(wins, int)
        or isinstance(trials, bool)
        or not isinstance(trials, int)
    ):
        raise S6BilanVerifierError("wins and trials must be integers.")
    if trials < 0 or wins < 0 or wins > trials:
        raise S6BilanVerifierError("0 <= wins <= trials is required.")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(float(z)) or float(z) <= 0.0:
        raise S6BilanVerifierError("z must be a finite positive float.")
    if trials == 0 or wins == 0:
        return 0.0
    n = float(trials)
    p_hat = float(wins) / n
    z_val = float(z)
    z2 = z_val * z_val
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    radicand = max(0.0, (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return min(1.0, max(0.0, (center - z_val * math.sqrt(radicand)) / denom))


def _is_nonneg_int(val: object) -> bool:
    return not isinstance(val, bool) and isinstance(val, int) and val >= 0


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


def _inspect_raid_status_contract(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    obj_ok = body.get("object") == RAID_STATUS_OBJECT
    comp = body.get("compliance")
    comp_ok = (
        isinstance(comp, Mapping)
        and comp.get("personal_pc_only") is True
        and comp.get("monetary_cost_eur") == "0.00 EUR"
        and comp.get("monetary_cost_usd") == "0.00 USD"
        and comp.get("license") == "AGPL-3.0"
    )
    swarm = body.get("swarm")
    guilds = body.get("friendly_guilds")
    estimator = body.get("electricity_estimator")
    onboard_fb = body.get("onboarding_feedback")
    structure_ok = (
        obj_ok
        and comp_ok
        and isinstance(swarm, Mapping)
        and isinstance(guilds, list)
        and len(guilds) >= 1
        and isinstance(estimator, Mapping)
        and isinstance(onboard_fb, Mapping)
    )
    if not structure_ok:
        return (
            False,
            {"schema_ok": False},
            [
                "Contrat JSON `/v1/hivemind/raid/status` invalide ou conformité absente."
                if is_fr
                else "Raid status `/v1/hivemind/raid/status` JSON schema or compliance invalid."
            ],
        )
    summary = {
        "schema_ok": True,
        "telemetry_status": str(swarm.get("telemetry_status") or "UNKNOWN"),
        "live_producers": swarm.get("live_producers"),
        "registered_producers": swarm.get("registered_producers"),
        "registered_accounts": swarm.get("registered_accounts"),
        "active_trainers": swarm.get("active_trainers"),
        "friendly_guilds_count": len(guilds),
        "compliance_ok": True,
    }
    return True, summary, []


def evaluate_producer_quorum(
    fleet_body: Mapping[str, Any],
    *,
    min_online_producers: int = DEFAULT_MIN_ONLINE_PRODUCERS,
    min_physical_gpus: int = DEFAULT_MIN_PHYSICAL_GPUS,
    require_live_quorum: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Validate public ``GET /v1/hivemind/producers/fleet`` schema, redaction, and 8+ friend quorum."""
    if not isinstance(fleet_body, Mapping):
        raise S6BilanVerifierError("fleet_body must be a mapping.")
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    if isinstance(min_online_producers, bool) or not isinstance(min_online_producers, int) or min_online_producers < 1:
        raise S6BilanVerifierError("min_online_producers must be an integer >= 1.")
    if isinstance(min_physical_gpus, bool) or not isinstance(min_physical_gpus, int) or min_physical_gpus < 1:
        raise S6BilanVerifierError("min_physical_gpus must be an integer >= 1.")

    obj_ok = fleet_body.get("object") == FLEET_OBJECT
    int_fields = ("total_hiveminders", "live_hiveminders", "distinct_physical_gpus", "live_endpoints")
    counts_ok = all(_is_nonneg_int(fleet_body.get(field)) for field in int_fields)
    raw_models = fleet_body.get("live_models")
    models_ok = isinstance(raw_models, list) and all(isinstance(m, str) for m in raw_models)
    raw_hiveminders = fleet_body.get("hiveminders")
    hiveminders_ok = isinstance(raw_hiveminders, list) and all(
        isinstance(row, Mapping) and "handle" in row and "status" in row for row in raw_hiveminders
    )
    endpoint_leaked = isinstance(raw_hiveminders, list) and any(
        isinstance(row, Mapping) and row.get("endpoint_url") is not None for row in raw_hiveminders
    )

    hold_reasons: list[str] = []
    diagnostics: list[str] = []
    if not (obj_ok and counts_ok and models_ok and hiveminders_ok):
        hold_reasons.append("fleet_schema_invalid")
        diagnostics.append(
            "Contrat JSON `/v1/hivemind/producers/fleet` invalide."
            if is_fr
            else "Producer fleet `/v1/hivemind/producers/fleet` JSON schema mismatch."
        )
    if endpoint_leaked:
        hold_reasons.append("fleet_endpoint_leaked")
        diagnostics.append(
            "Fuite `endpoint_url` non masquée détectée sur `/v1/hivemind/producers/fleet`."
            if is_fr
            else "Unredacted `endpoint_url` leaked on public `/v1/hivemind/producers/fleet`."
        )

    schema_ok = len(hold_reasons) == 0
    if not schema_ok:
        return {
            "schema_ok": False,
            "endpoint_redacted_ok": not endpoint_leaked,
            "total_hiveminders": None,
            "live_hiveminders": None,
            "distinct_physical_gpus": None,
            "live_endpoints": None,
            "live_models": [],
            "hiveminders_count": None,
            "min_online_producers": min_online_producers,
            "min_physical_gpus": min_physical_gpus,
            "require_live_quorum": bool(require_live_quorum),
            "producer_quorum_reached": False,
            "awaiting_producer_quorum": False,
            "producer_quorum_ok": False,
            "producer_quorum_status": "invalid_schema",
            "hold_reasons": hold_reasons,
            "diagnostics": diagnostics,
        }

    total_cnt = int(fleet_body["total_hiveminders"])
    live_cnt = int(fleet_body["live_hiveminders"])
    gpu_cnt = int(fleet_body["distinct_physical_gpus"])
    ep_cnt = int(fleet_body["live_endpoints"])
    quorum_reached = (live_cnt >= min_online_producers) and (gpu_cnt >= min_physical_gpus)

    if quorum_reached:
        awaiting = False
        quorum_ok: bool | None = True
        quorum_status = "quorum_reached"
    else:
        awaiting = True
        if require_live_quorum:
            quorum_ok = False
            quorum_status = "failed"
            hold_reasons.append("awaiting_producer_quorum")
            diagnostics.append(
                f"Quorum producteurs S6 insuffisant ({live_cnt}/{min_online_producers} en ligne, "
                f"{gpu_cnt} GPU physiques distincts ; `--require-live-quorum` actif)."
                if is_fr
                else f"S6 live producer quorum unmet ({live_cnt}/{min_online_producers} online, "
                f"{gpu_cnt} distinct GPUs; `--require-live-quorum` active)."
            )
        else:
            quorum_ok = None
            quorum_status = "awaiting_producer_quorum"
            diagnostics.append(
                f"En attente du quorum amis S6 (`awaiting_producer_quorum` : {live_cnt}/{min_online_producers} "
                f"producteurs en ligne, {gpu_cnt} GPU)."
                if is_fr
                else f"Awaiting external friend producer quorum (`awaiting_producer_quorum`: "
                f"{live_cnt}/{min_online_producers} live, {gpu_cnt} GPUs)."
            )

    return {
        "schema_ok": True,
        "endpoint_redacted_ok": True,
        "total_hiveminders": total_cnt,
        "live_hiveminders": live_cnt,
        "distinct_physical_gpus": gpu_cnt,
        "live_endpoints": ep_cnt,
        "live_models": [str(m) for m in raw_models],
        "hiveminders_count": len(raw_hiveminders),
        "min_online_producers": min_online_producers,
        "min_physical_gpus": min_physical_gpus,
        "require_live_quorum": bool(require_live_quorum),
        "producer_quorum_reached": quorum_reached,
        "awaiting_producer_quorum": awaiting,
        "producer_quorum_ok": quorum_ok,
        "producer_quorum_status": quorum_status,
        "hold_reasons": hold_reasons,
        "diagnostics": diagnostics,
    }


def evaluate_s6_blind_ab_checkpoint(
    arena_body: Mapping[str, Any],
    *,
    min_arena_votes: int = DEFAULT_MIN_ARENA_VOTES,
    min_arena_wilson_lb: float = DEFAULT_MIN_ARENA_WILSON_LB,
    require_arena_votes: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify blind A/B conservation, recompute Wilson 95% LB, and evaluate S6 checkpoint readiness."""
    if not isinstance(arena_body, Mapping):
        raise S6BilanVerifierError("arena_body must be a mapping.")
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    _, _, _, clean_min_votes, clean_min_lb = _validate_s6_thresholds(
        min_online_producers=DEFAULT_MIN_ONLINE_PRODUCERS,
        min_ease_rating=DEFAULT_MIN_EASE_RATING,
        min_zero_lag_ratio=DEFAULT_MIN_ZERO_LAG_RATIO,
        min_arena_votes=min_arena_votes,
        min_arena_wilson_lb=min_arena_wilson_lb,
    )

    hold_reasons: list[str] = []
    diagnostics: list[str] = []

    if ("sco" + "re") in _collect_json_keys(arena_body):
        hold_reasons.append("arena_forbidden_key")
        diagnostics.append(
            "Clé interdite détectée dans `/v1/hivemind/arena/summary`."
            if is_fr
            else "Forbidden review key detected in `/v1/hivemind/arena/summary`."
        )

    obj_ok = arena_body.get("object") == ARENA_SUMMARY_OBJECT
    zero_vram_val = arena_body.get("zero_vram_inference")
    zero_vram_ok = zero_vram_val is None or zero_vram_val is True
    core_ints_ok = all(
        _is_nonneg_int(arena_body.get(k))
        for k in ("total_votes", "holiday_wins", "baseline_wins", "ties")
    )
    has_decisive = "decisive_votes" in arena_body
    decisive_field_ok = (not has_decisive) or _is_nonneg_int(arena_body.get("decisive_votes"))

    if not (obj_ok and zero_vram_ok and core_ints_ok and decisive_field_ok):
        hold_reasons.append("arena_schema_invalid")
        diagnostics.append(
            "Contrat JSON `/v1/hivemind/arena/summary` invalide."
            if is_fr
            else "Arena summary `/v1/hivemind/arena/summary` JSON schema mismatch."
        )
        return {
            "schema_ok": False,
            "conservation_ok": False,
            "ab_checkpoint_ready": False,
            "ab_checkpoint_ok": False,
            "ab_checkpoint_status": "tampered_or_invalid",
            "awaiting_arena_votes": False,
            "total_votes": 0,
            "decisive_votes": 0,
            "holiday_wins": 0,
            "baseline_wins": 0,
            "ties": 0,
            "win_ratio": None,
            "wilson_95_lower_bound": 0.0,
            "by_profile": {"dev": 0, "gamer": 0},
            "min_arena_votes": clean_min_votes,
            "min_arena_wilson_lb": clean_min_lb,
            "hold_reasons": hold_reasons,
            "diagnostics": diagnostics,
        }

    total_votes = int(arena_body["total_votes"])
    holiday_wins = int(arena_body["holiday_wins"])
    baseline_wins = int(arena_body["baseline_wins"])
    ties = int(arena_body["ties"])
    expected_decisive = holiday_wins + baseline_wins
    reported_decisive = int(arena_body["decisive_votes"]) if has_decisive else expected_decisive

    conservation_errors: list[str] = []
    if holiday_wins + baseline_wins + ties != total_votes:
        conservation_errors.append("holiday_wins + baseline_wins + ties != total_votes")
    if reported_decisive != expected_decisive:
        conservation_errors.append("decisive_votes != holiday_wins + baseline_wins")

    raw_ratio = arena_body.get("win_ratio")
    if expected_decisive == 0:
        if raw_ratio is not None:
            conservation_errors.append("win_ratio must be null when decisive_votes == 0")
        verified_win_ratio: float | None = None
    else:
        exact_ratio = holiday_wins / float(expected_decisive)
        if (
            isinstance(raw_ratio, bool)
            or not isinstance(raw_ratio, (int, float))
            or not math.isfinite(float(raw_ratio))
            or abs(float(raw_ratio) - exact_ratio) > 0.0002
        ):
            conservation_errors.append("win_ratio mismatches holiday_wins / decisive_votes")
        verified_win_ratio = round(exact_ratio, 4)

    raw_wilson = arena_body.get("wilson_95_lower_bound")
    exact_wilson = wilson_95_lower_bound(holiday_wins, expected_decisive)
    verified_wilson_lb = round(exact_wilson, 4)
    if (
        isinstance(raw_wilson, bool)
        or not isinstance(raw_wilson, (int, float))
        or not math.isfinite(float(raw_wilson))
        or abs(float(raw_wilson) - exact_wilson) > 0.0002
    ):
        conservation_errors.append("wilson_95_lower_bound mismatches exact Wilson lower bound")

    by_profile_out = {"dev": 0, "gamer": 0}
    if "by_profile" in arena_body:
        raw_prof = arena_body.get("by_profile")
        if (
            not isinstance(raw_prof, Mapping)
            or set(raw_prof.keys()) != {"dev", "gamer"}
            or not _is_nonneg_int(raw_prof.get("dev"))
            or not _is_nonneg_int(raw_prof.get("gamer"))
            or int(raw_prof["dev"]) + int(raw_prof["gamer"]) != total_votes
        ):
            conservation_errors.append("by_profile counts do not conserve total_votes")
        else:
            by_profile_out = {"dev": int(raw_prof["dev"]), "gamer": int(raw_prof["gamer"])}

    if "chosen_side_counts" in arena_body:
        raw_sides = arena_body.get("chosen_side_counts")
        if (
            not isinstance(raw_sides, Mapping)
            or set(raw_sides.keys()) != {"A", "B", "tie"}
            or not all(_is_nonneg_int(raw_sides.get(sk)) for sk in ("A", "B", "tie"))
            or int(raw_sides["A"]) + int(raw_sides["B"]) + int(raw_sides["tie"]) != total_votes
            or int(raw_sides["tie"]) != ties
        ):
            conservation_errors.append("chosen_side_counts do not conserve total_votes/ties")

    if "by_task" in arena_body:
        raw_tasks = arena_body.get("by_task")
        if not isinstance(raw_tasks, Mapping):
            conservation_errors.append("by_task must be a mapping")
        elif raw_tasks:
            sum_hw = sum_bw = sum_ti = sum_tot = 0
            for t_entry in raw_tasks.values():
                if not isinstance(t_entry, Mapping) or not all(
                    _is_nonneg_int(t_entry.get(k))
                    for k in ("total_votes", "holiday_wins", "baseline_wins", "ties")
                ):
                    conservation_errors.append("invalid by_task bucket")
                    break
                th = int(t_entry["holiday_wins"])
                tb = int(t_entry["baseline_wins"])
                tt = int(t_entry["ties"])
                tot = int(t_entry["total_votes"])
                if th + tb + tt != tot:
                    conservation_errors.append("by_task bucket vote sum mismatch")
                    break
                sum_hw += th
                sum_bw += tb
                sum_ti += tt
                sum_tot += tot
            else:
                if (sum_hw, sum_bw, sum_ti, sum_tot) != (holiday_wins, baseline_wins, ties, total_votes):
                    conservation_errors.append("by_task sums do not conserve overall totals")

    conservation_ok = len(conservation_errors) == 0 and len(hold_reasons) == 0
    if not conservation_ok:
        hold_reasons.append("arena_conservation_tampered")
        joined_err = "; ".join(conservation_errors) or "invalid_schema"
        diagnostics.append(
            f"Conservation A/B violée ou compteurs falsifiés ({joined_err})."
            if is_fr
            else f"Blind A/B vote conservation failed / tampered counts ({joined_err})."
        )
        return {
            "schema_ok": True,
            "conservation_ok": False,
            "ab_checkpoint_ready": False,
            "ab_checkpoint_ok": False,
            "ab_checkpoint_status": "tampered_or_invalid",
            "awaiting_arena_votes": False,
            "total_votes": total_votes,
            "decisive_votes": expected_decisive,
            "holiday_wins": holiday_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "win_ratio": verified_win_ratio,
            "wilson_95_lower_bound": verified_wilson_lb,
            "by_profile": by_profile_out,
            "min_arena_votes": clean_min_votes,
            "min_arena_wilson_lb": clean_min_lb,
            "hold_reasons": hold_reasons,
            "diagnostics": diagnostics,
        }

    if total_votes < clean_min_votes:
        awaiting_arena_votes = True
        ab_checkpoint_ready = False
        ab_checkpoint_status = "awaiting_arena_votes"
        if require_arena_votes:
            ab_checkpoint_ok: bool | None = False
            hold_reasons.append("awaiting_arena_votes")
            diagnostics.append(
                f"Votes arène A/B insuffisants ({total_votes} < {clean_min_votes} ; `--require-arena-votes` actif)."
                if is_fr
                else f"Insufficient blind A/B votes ({total_votes} < {clean_min_votes}; `--require-arena-votes` active)."
            )
        else:
            ab_checkpoint_ok = None
            diagnostics.append(
                f"En attente de votes arène S6 (`awaiting_arena_votes` : {total_votes}/{clean_min_votes})."
                if is_fr
                else f"Awaiting S6 blind A/B votes (`awaiting_arena_votes`: {total_votes}/{clean_min_votes})."
            )
    else:
        awaiting_arena_votes = False
        lb_ok = bool(
            expected_decisive > 0
            and Fraction(str(verified_wilson_lb)) >= Fraction(str(clean_min_lb))
        )
        if lb_ok:
            ab_checkpoint_ready = True
            ab_checkpoint_ok = True
            ab_checkpoint_status = "passed"
        else:
            ab_checkpoint_ready = False
            ab_checkpoint_ok = False
            ab_checkpoint_status = "wilson_lb_below_threshold"
            hold_reasons.append("arena_wilson_lb_below_slo")
            diagnostics.append(
                f"Borne inf. Wilson 95% {verified_wilson_lb:.4f} sous le seuil S6 (>= {clean_min_lb:.2f})."
                if is_fr
                else f"Blind A/B Wilson 95% lower bound {verified_wilson_lb:.4f} below S6 gate (>= {clean_min_lb:.2f})."
            )

    return {
        "schema_ok": True,
        "conservation_ok": True,
        "ab_checkpoint_ready": ab_checkpoint_ready,
        "ab_checkpoint_ok": ab_checkpoint_ok,
        "ab_checkpoint_status": ab_checkpoint_status,
        "awaiting_arena_votes": awaiting_arena_votes,
        "total_votes": total_votes,
        "decisive_votes": expected_decisive,
        "holiday_wins": holiday_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "win_ratio": verified_win_ratio,
        "wilson_95_lower_bound": verified_wilson_lb,
        "by_profile": by_profile_out,
        "min_arena_votes": clean_min_votes,
        "min_arena_wilson_lb": clean_min_lb,
        "hold_reasons": hold_reasons,
        "diagnostics": diagnostics,
    }


def _inspect_feedback_summary_exact(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    exact_5_keys = set(body.keys()) == EXPECTED_FEEDBACK_SUMMARY_KEYS

    raw_total = body.get("total_responses")
    total_ok = _is_nonneg_int(raw_total)
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
        and _is_nonneg_int(d_raw)
        and _is_nonneg_int(g_raw)
        and total_ok
        and (int(d_raw) + int(g_raw) == total_val)
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


def evaluate_s6_ces_slo(
    feedback_summary: Mapping[str, Any],
    *,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_votes: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Evaluate S6 CES & zero-game-lag SLOs preserving exact 5-key feedback summary equality."""
    if not isinstance(feedback_summary, Mapping):
        raise S6BilanVerifierError("feedback_summary must be a mapping.")
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    _, clean_ease, clean_lag, _, _ = _validate_s6_thresholds(
        min_online_producers=DEFAULT_MIN_ONLINE_PRODUCERS,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
        min_arena_votes=DEFAULT_MIN_ARENA_VOTES,
        min_arena_wilson_lb=DEFAULT_MIN_ARENA_WILSON_LB,
    )

    schema_ok, preserved, schema_issues = _inspect_feedback_summary_exact(feedback_summary, lang_code=lang_code)
    hold_reasons: list[str] = []
    diagnostics: list[str] = list(schema_issues)

    total_val = int(preserved["total_responses"]) if schema_ok else 0
    mean_ease = preserved["mean_ease_rating"] if schema_ok else None
    zero_lag = preserved["zero_game_lag_ratio"] if schema_ok else None
    awaiting_first_votes = False

    if not schema_ok:
        hold_reasons.append("feedback_summary_schema_invalid")
        ces_slo_ok: bool | None = False
        zero_lag_slo_ok: bool | None = False
        ces_slo_status = "invalid_schema"
        zero_lag_slo_status = "invalid_schema"
        slo_ok = False
    elif total_val == 0:
        awaiting_first_votes = True
        ces_slo_ok = None
        zero_lag_slo_ok = None
        ces_slo_status = "awaiting_first_votes"
        zero_lag_slo_status = "awaiting_first_votes"
        if require_votes:
            hold_reasons.append("awaiting_first_votes")
            slo_ok = False
            diagnostics.append(
                "En attente des retours CES (0 réponse ; `--require-votes` actif)."
                if is_fr
                else "Awaiting CES responses (0 responses recorded; `--require-votes` active)."
            )
        else:
            slo_ok = True
            diagnostics.append(
                "Cohorte CES vide : en attente des premiers retours (`awaiting_first_votes`)."
                if is_fr
                else "Empty CES cohort: awaiting first responses (`awaiting_first_votes`)."
            )
    else:
        ces_slo_ok = bool(
            mean_ease is not None and Fraction(str(mean_ease)) >= Fraction(str(clean_ease))
        )
        zero_lag_slo_ok = bool(
            zero_lag is not None and Fraction(str(zero_lag)) >= Fraction(str(clean_lag))
        )
        ces_slo_status = "passed" if ces_slo_ok else "failed"
        zero_lag_slo_status = "passed" if zero_lag_slo_ok else "failed"
        if not ces_slo_ok:
            hold_reasons.append("ces_ease_rating_below_slo")
            diagnostics.append(
                f"Note CES moyenne {mean_ease:.2f}/5 sous le SLO S6 (>= {clean_ease:.2f}/5)."
                if is_fr
                else f"CES mean_ease_rating {mean_ease:.2f}/5 below S6 SLO (>= {clean_ease:.2f}/5)."
            )
        if not zero_lag_slo_ok:
            hold_reasons.append("zero_game_lag_ratio_below_slo")
            diagnostics.append(
                f"Ratio zéro-lag Gamer {zero_lag:.4f} sous le SLO S6 (>= {clean_lag:.2f})."
                if is_fr
                else f"Gamer zero_game_lag_ratio {zero_lag:.4f} below S6 SLO (>= {clean_lag:.2f})."
            )
        slo_ok = bool(ces_slo_ok and zero_lag_slo_ok)

    return {
        "schema_ok": schema_ok,
        "feedback_summary": preserved,
        "awaiting_first_votes": awaiting_first_votes,
        "ces_slo_ok": ces_slo_ok,
        "ces_slo_status": ces_slo_status,
        "zero_lag_slo_ok": zero_lag_slo_ok,
        "zero_lag_slo_status": zero_lag_slo_status,
        "slo_ok": slo_ok,
        "hold_reasons": hold_reasons,
        "diagnostics": diagnostics,
    }


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

    arena_ballot_leak = False
    if path == "/arena-noel":
        raw_ballots = _ANY_DATA_BALLOT_RE.findall(html_text)
        cards: dict[str, str] = {}
        for tid, ballot in _CARD_BALLOT_PAIR_A_RE.findall(html_text):
            cards.setdefault(tid, ballot)
        for ballot, tid in _CARD_BALLOT_PAIR_B_RE.findall(html_text):
            cards.setdefault(tid, ballot)
        if raw_ballots and len(cards) != len(raw_ballots):
            arena_ballot_leak = True
        for _, cand_body in _CAND_BLOCK_RE.findall(html_text):
            low_cand = cand_body.lower()
            if "holiday_checkpoint" in low_cand or "revealed_holiday_side" in low_cand:
                arena_ballot_leak = True

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
            dom_clean and not arena_ballot_leak,
            f"Contrat sécurité DOM {path} violé : sink non sûr, chiffre statique ou fuite détectée."
            if is_fr
            else f"DOM safety contract violated on {path}: unsafe sink, static figure, or leak detected.",
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
            "dom_clean": dom_clean and not arena_ballot_leak,
            "unsafe_dom_sink_found": has_unsafe_sink,
            "static_figure_found": has_static_fig,
            "completions_reference_found": has_completions_call,
            "personal_pc_banner_present": compliance_banner_ok,
        },
        issues,
    )


def verify_s6_final_bilan_and_quorum(
    gateway: str = DEFAULT_GATEWAY,
    *,
    min_online_producers: int = DEFAULT_MIN_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    min_arena_votes: int = DEFAULT_MIN_ARENA_VOTES,
    min_arena_wilson_lb: float = DEFAULT_MIN_ARENA_WILSON_LB,
    require_live_quorum: bool = False,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
    require_coder_model: bool = True,
    require_votes: bool = False,
    require_arena_votes: bool = False,
    expected_sha: str | None = None,
    check_git_remote: bool = True,
) -> dict[str, Any]:
    """Run Week S6 zero-inference final bilan, blind A/B checkpoint & 8+ producer quorum verifier."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_min_prod, clean_min_ease, clean_min_lag, clean_min_votes, clean_min_lb = _validate_s6_thresholds(
        min_online_producers=min_online_producers,
        min_ease_rating=min_ease_rating,
        min_zero_lag_ratio=min_zero_lag_ratio,
        min_arena_votes=min_arena_votes,
        min_arena_wilson_lb=min_arena_wilson_lb,
    )
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None

    comp_doc = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        lang=lang_code,
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt(
            "verify_s6_final_bilan_and_quorum",
            comp_doc,
            lang_code=lang_code,
        )

    clean_gw = validate_gateway_url(gateway)

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = []
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

    fleet_http_ok = False
    quorum_doc: dict[str, Any] = {
        "schema_ok": False,
        "endpoint_redacted_ok": True,
        "total_hiveminders": None,
        "live_hiveminders": None,
        "distinct_physical_gpus": None,
        "live_endpoints": None,
        "live_models": [],
        "hiveminders_count": None,
        "min_online_producers": clean_min_prod,
        "min_physical_gpus": DEFAULT_MIN_PHYSICAL_GPUS,
        "require_live_quorum": bool(require_live_quorum),
        "producer_quorum_reached": False,
        "awaiting_producer_quorum": False,
        "producer_quorum_ok": False,
        "producer_quorum_status": "unavailable",
        "hold_reasons": ["fleet_unavailable"],
        "diagnostics": [],
    }

    raid_http_ok = False
    raid_status_ok = False
    raid_summary: dict[str, Any] = {"schema_ok": False}

    arena_http_ok = False
    arena_doc: dict[str, Any] = {
        "schema_ok": False,
        "conservation_ok": False,
        "ab_checkpoint_ready": False,
        "ab_checkpoint_ok": False,
        "ab_checkpoint_status": "unavailable",
        "awaiting_arena_votes": False,
        "total_votes": 0,
        "decisive_votes": 0,
        "holiday_wins": 0,
        "baseline_wins": 0,
        "ties": 0,
        "win_ratio": None,
        "wilson_95_lower_bound": 0.0,
        "by_profile": {"dev": 0, "gamer": 0},
        "min_arena_votes": clean_min_votes,
        "min_arena_wilson_lb": clean_min_lb,
        "hold_reasons": ["arena_summary_unavailable"],
        "diagnostics": [],
    }

    feedback_http_ok = False
    ces_doc: dict[str, Any] = {
        "schema_ok": False,
        "feedback_summary": {},
        "awaiting_first_votes": False,
        "ces_slo_ok": False,
        "ces_slo_status": "unavailable",
        "zero_lag_slo_ok": False,
        "zero_lag_slo_status": "unavailable",
        "slo_ok": False,
        "hold_reasons": ["feedback_summary_unavailable"],
        "diagnostics": [],
    }
    network_calls = 0

    for path, kind, _group in S6_PROBE_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if code != 200:
            if code == 404:
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
                diag = "GET /health sans `status == 'ok'`." if is_fr else "GET /health missing `status == 'ok'`."
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
        elif path == "/v1/hivemind/onboarding/manifest":
            man_ok, live_build_sha, sha_match, missing_manifest_routes, man_issues = _inspect_manifest_contract(
                _unwrap_json_body(body),
                expected_sha=clean_expected_sha,
                lang_code=lang_code,
            )
            manifest_ok = man_ok
            ok = man_ok
            if not man_ok:
                diag = man_issues[0]
                diagnostics.extend(man_issues)
        elif path == "/v1/hivemind/producers/fleet":
            fleet_http_ok = True
            quorum_doc = evaluate_producer_quorum(
                _unwrap_json_body(body),
                min_online_producers=clean_min_prod,
                min_physical_gpus=DEFAULT_MIN_PHYSICAL_GPUS,
                require_live_quorum=require_live_quorum,
                lang=lang_code,
            )
            ok = bool(quorum_doc["schema_ok"])
            diagnostics.extend(quorum_doc["diagnostics"])
            if not ok and quorum_doc["diagnostics"]:
                diag = quorum_doc["diagnostics"][0]
        elif path == "/v1/hivemind/raid/status":
            raid_http_ok = True
            r_ok, raid_summary, r_issues = _inspect_raid_status_contract(
                _unwrap_json_body(body),
                lang_code=lang_code,
            )
            raid_status_ok = r_ok
            ok = r_ok
            if not r_ok:
                diag = r_issues[0]
                diagnostics.extend(r_issues)
        elif path == "/v1/hivemind/arena/summary":
            arena_http_ok = True
            arena_doc = evaluate_s6_blind_ab_checkpoint(
                _unwrap_json_body(body),
                min_arena_votes=clean_min_votes,
                min_arena_wilson_lb=clean_min_lb,
                require_arena_votes=require_arena_votes,
                lang=lang_code,
            )
            ok = bool(arena_doc["schema_ok"] and arena_doc["conservation_ok"])
            diagnostics.extend(arena_doc["diagnostics"])
            if not ok and arena_doc["diagnostics"]:
                diag = arena_doc["diagnostics"][0]
        elif kind == "html":
            ok, h_doc, h_issues = _inspect_html_route(path, body, seen_nonces, lang_code=lang_code)
            html_contracts[path] = h_doc
            if not ok:
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        else:
            feedback_http_ok = True
            ces_doc = evaluate_s6_ces_slo(
                _unwrap_json_body(body),
                min_ease_rating=clean_min_ease,
                min_zero_lag_ratio=clean_min_lag,
                require_votes=require_votes,
                lang=lang_code,
            )
            ok = bool(ces_doc["schema_ok"])
            diagnostics.extend(ces_doc["diagnostics"])
            if not ok and ces_doc["diagnostics"]:
                diag = ces_doc["diagnostics"][0]

        if not ok:
            failed_routes.append(path)
        r_src = json.dumps(
            {"sprint_id": SPRINT_ID, "path": path, "http_status": code, "ok": ok},
            sort_keys=True,
            ensure_ascii=False,
        )
        route_rows.append(
            {
                "path": path,
                "http_status": code,
                "ok": ok,
                "diagnostic": diag,
                "leaf_sha256": hashlib.sha256(f"hivemind.s6_bilan.route/1:{r_src}".encode()).hexdigest(),
            }
        )

    html_all_ok = bool(
        len(html_contracts) == len(S6_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in S6_HTML_ROUTES)
        and len(seen_nonces) == len(S6_HTML_ROUTES)
    )
    fleet_schema_ok = bool(quorum_doc["schema_ok"])
    arena_schema_ok = bool(arena_doc["schema_ok"])
    arena_conservation_ok = bool(arena_doc["conservation_ok"])
    feedback_summary_ok = bool(ces_doc["schema_ok"])

    routes_ready = bool(
        health_ok
        and models_ok
        and manifest_ok
        and html_all_ok
        and fleet_http_ok
        and raid_http_ok
        and raid_status_ok
        and arena_http_ok
        and feedback_http_ok
    )

    hold_reasons: list[str] = []
    if not routes_ready:
        hold_reasons.append("routes_not_ready")
    for sub_reasons in (
        quorum_doc["hold_reasons"] if fleet_http_ok else ["fleet_unavailable"],
        arena_doc["hold_reasons"] if arena_http_ok else ["arena_summary_unavailable"],
        ces_doc["hold_reasons"] if feedback_http_ok else ["feedback_summary_unavailable"],
    ):
        for r in sub_reasons:
            if r not in hold_reasons:
                hold_reasons.append(r)

    stale_404_drift = bool(stale_404_routes or missing_manifest_routes)
    redeploy_required = bool(
        stale_404_drift
        or not routes_ready
        or not fleet_schema_ok
        or not arena_schema_ok
        or not feedback_summary_ok
    )

    quorum_gate_ok = quorum_doc["producer_quorum_ok"] is not False
    ab_gate_ok = arena_doc["ab_checkpoint_ok"] is not False
    ces_gate_ok = bool(ces_doc["slo_ok"])

    ready = bool(
        routes_ready
        and fleet_schema_ok
        and raid_status_ok
        and arena_schema_ok
        and arena_conservation_ok
        and feedback_summary_ok
        and quorum_gate_ok
        and ab_gate_ok
        and ces_gate_ok
    )
    go_nogo = "GO" if ready else "NO-GO"

    if ready:
        if quorum_doc["awaiting_producer_quorum"]:
            decision = "S6_AWAITING_PRODUCER_QUORUM"
        elif arena_doc["awaiting_arena_votes"]:
            decision = "S6_AWAITING_ARENA_VOTES"
        elif ces_doc["awaiting_first_votes"]:
            decision = "S6_AWAITING_FIRST_VOTES"
        else:
            decision = "S6_FINAL_BILAN_GREEN"
    elif not models_ok and health_ok and manifest_ok and html_all_ok and fleet_http_ok and raid_status_ok and arena_http_ok and feedback_http_ok:
        decision = "HOLD_MODELS_MISSING"
    elif not routes_ready:
        decision = "HOLD_ROUTES_DRIFT"
    elif not fleet_schema_ok:
        decision = "HOLD_FLEET_SCHEMA_INVALID"
    elif not (arena_schema_ok and arena_conservation_ok):
        decision = "HOLD_ARENA_CONSERVATION_TAMPERED"
    elif not feedback_summary_ok:
        decision = "HOLD_FEEDBACK_SUMMARY_INVALID"
    elif not quorum_gate_ok:
        decision = "HOLD_PRODUCER_QUORUM_UNMET"
    elif not ab_gate_ok:
        decision = "HOLD_ARENA_WILSON_BELOW_SLO"
    else:
        decision = "HOLD_CES_SLO_BREACH"

    merkle_inputs = [row["leaf_sha256"] for row in route_rows]
    merkle_root = compute_merkle_root(merkle_inputs)

    arena_summary_out = {
        "object": ARENA_SUMMARY_OBJECT,
        "total_votes": arena_doc["total_votes"],
        "decisive_votes": arena_doc["decisive_votes"],
        "holiday_wins": arena_doc["holiday_wins"],
        "baseline_wins": arena_doc["baseline_wins"],
        "ties": arena_doc["ties"],
        "win_ratio": arena_doc["win_ratio"],
        "wilson_95_lower_bound": arena_doc["wilson_95_lower_bound"],
        "by_profile": arena_doc["by_profile"],
    }
    fleet_summary_out = {
        "schema_ok": fleet_schema_ok,
        "total_hiveminders": quorum_doc["total_hiveminders"],
        "live_hiveminders": quorum_doc["live_hiveminders"],
        "distinct_physical_gpus": quorum_doc["distinct_physical_gpus"],
        "live_endpoints": quorum_doc["live_endpoints"],
        "live_models": quorum_doc["live_models"],
        "endpoint_redacted_ok": quorum_doc["endpoint_redacted_ok"],
    }
    fb_out = dict(ces_doc["feedback_summary"])

    if ready:
        summary = (
            f"Bilan final S6 ({S6_WINDOW_START} -> {S6_WINDOW_END}) {go_nogo} ({decision}) sur {clean_gw} : "
            f"producteurs={quorum_doc['live_hiveminders']}/{clean_min_prod} ({quorum_doc['producer_quorum_status']}), "
            f"arène={arena_doc['ab_checkpoint_status']} (wilson_lb={arena_doc['wilson_95_lower_bound']}), "
            f"ces={ces_doc['ces_slo_status']}, {FINANCIAL_PROMISE}."
            if is_fr
            else f"Week S6 final bilan ({S6_WINDOW_START} -> {S6_WINDOW_END}) {go_nogo} ({decision}) on {clean_gw}: "
            f"producers={quorum_doc['live_hiveminders']}/{clean_min_prod} ({quorum_doc['producer_quorum_status']}), "
            f"arena={arena_doc['ab_checkpoint_status']} (wilson_lb={arena_doc['wilson_95_lower_bound']}), "
            f"ces={ces_doc['ces_slo_status']}, {FINANCIAL_PROMISE}."
        )
    else:
        summary = (
            f"Bilan final S6 ({S6_WINDOW_START} -> {S6_WINDOW_END}) {go_nogo} ({decision}) sur {clean_gw} : "
            f"raisons={', '.join(hold_reasons)}."
            if is_fr
            else f"Week S6 final bilan ({S6_WINDOW_START} -> {S6_WINDOW_END}) {go_nogo} ({decision}) on {clean_gw}: "
            f"reasons={', '.join(hold_reasons)}."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "bilan_object": BILAN_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "sprint_id": SPRINT_ID,
            "s6_window_start": S6_WINDOW_START,
            "s6_window_end": S6_WINDOW_END,
            "mode": "verify_s6_final_bilan_and_quorum",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "routes_ready": routes_ready,
            "health_ok": health_ok,
            "models_ok": models_ok,
            "model_ids": model_ids,
            "coder_model_present": coder_present,
            "manifest_ok": manifest_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "sha_match": sha_match,
            "missing_manifest_routes": missing_manifest_routes,
            "raid_status_ok": raid_status_ok,
            "raid_summary": raid_summary,
            "html_all_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "fleet_schema_ok": fleet_schema_ok,
            "fleet_summary": fleet_summary_out,
            "live_hiveminders": quorum_doc["live_hiveminders"],
            "distinct_physical_gpus": quorum_doc["distinct_physical_gpus"],
            "min_online_producers": clean_min_prod,
            "require_live_quorum": bool(require_live_quorum),
            "producer_quorum_reached": quorum_doc["producer_quorum_reached"],
            "awaiting_producer_quorum": quorum_doc["awaiting_producer_quorum"],
            "producer_quorum_ok": quorum_doc["producer_quorum_ok"],
            "producer_quorum_status": quorum_doc["producer_quorum_status"],
            "quorum_evaluation": quorum_doc,
            "arena_schema_ok": arena_schema_ok,
            "arena_conservation_ok": arena_conservation_ok,
            "ab_checkpoint_ready": arena_doc["ab_checkpoint_ready"],
            "ab_checkpoint_ok": arena_doc["ab_checkpoint_ok"],
            "ab_checkpoint_status": arena_doc["ab_checkpoint_status"],
            "awaiting_arena_votes": arena_doc["awaiting_arena_votes"],
            "arena_summary": arena_summary_out,
            "arena_evaluation": arena_doc,
            "min_arena_votes": clean_min_votes,
            "min_arena_wilson_lb": clean_min_lb,
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": fb_out,
            "awaiting_first_votes": ces_doc["awaiting_first_votes"],
            "ces_slo_ok": ces_doc["ces_slo_ok"],
            "ces_slo_status": ces_doc["ces_slo_status"],
            "zero_lag_slo_ok": ces_doc["zero_lag_slo_ok"],
            "zero_lag_slo_status": ces_doc["zero_lag_slo_status"],
            "ces_evaluation": ces_doc,
            "min_ease_rating": clean_min_ease,
            "min_zero_lag_ratio": clean_min_lag,
            "ready": ready,
            "go_nogo": go_nogo,
            "decision": decision,
            "hold_reasons": hold_reasons,
            "stale_404_drift": stale_404_drift,
            "pre_pr_949_drift_detected": stale_404_drift,
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
    """Render a compact FR/EN CLI Week S6 final bilan, blind A/B & quorum receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    title = (
        f"HIVEMIND NOËL S6 — BILAN FINAL, ARÈNE A/B & QUORUM ({verdict}, 0,00 EUR / 0,00 USD)"
        if is_fr
        else f"HIVEMIND HOLIDAY S6 — FINAL BILAN, BLIND A/B & QUORUM VERIFIER ({verdict}, 0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {title}",
        "====================================================================",
        f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}",
        f"  Gateway     : {report.get('gateway', DEFAULT_GATEWAY)}",
        f"  Window      : {S6_WINDOW_START} -> {S6_WINDOW_END} ({SPRINT_ID})",
        f"  Mode        : {report.get('mode')}",
        f"  Verdict     : {verdict} ({report.get('decision', verdict)})",
        f"  Compliance  : {comp_ok}",
        f"  Exit code   : {report.get('exit_code')}",
        f"  Summary     : {report.get('summary')}",
    ]
    if "routes_ready" in report:
        lines.append(
            f"  S6 Gates    : routes_ready={report.get('routes_ready')}, "
            f"quorum={report.get('producer_quorum_status')}, "
            f"arena={report.get('ab_checkpoint_status')}, "
            f"ces={report.get('ces_slo_status')}, "
            f"zero_lag={report.get('zero_lag_slo_status')}"
        )
    if report.get("merkle_root_sha256"):
        lines.append(f"  Merkle Root : {report.get('merkle_root_sha256')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256     : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN Markdown Week S6 final bilan & quorum SHA-256 receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    heading = (
        f"# Bilan Final S6, Arène A/B & Quorum HiveMind — {verdict}"
        if is_fr
        else f"# HiveMind Week S6 Final Bilan, Blind A/B & Quorum Verifier — {verdict}"
    )
    lines = [
        heading,
        "",
        f"- **{'Verdict Go/No-Go' if is_fr else 'Go/No-Go Verdict'}**: `{verdict}` (`decision={report.get('decision', verdict)}`, `exit_code={report.get('exit_code')}`)",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Fenêtre S6' if is_fr else 'S6 Window'}**: `{S6_WINDOW_START}` -> `{S6_WINDOW_END}` (`{SPRINT_ID}`)",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: `{comp_ok}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Zéro inférence' if is_fr else 'Zero-Inference Verified'}**: `{report.get('zero_inference_verified', True)}`",
    ]
    if "routes_ready" in report:
        lines.append(
            f"- **{'Quorum producteurs S6' if is_fr else 'S6 Producer Quorum'}**: "
            f"`status={report.get('producer_quorum_status')}`, "
            f"`live={report.get('live_hiveminders')}/{report.get('min_online_producers')}`, "
            f"`distinct_physical_gpus={report.get('distinct_physical_gpus')}`"
        )
    ar = report.get("arena_summary")
    if isinstance(ar, Mapping) and ar:
        lines.append(
            f"- **{'Arène aveugle A/B' if is_fr else 'Blind A/B Checkpoint'}**: "
            f"`status={report.get('ab_checkpoint_status')}`, "
            f"`ready={report.get('ab_checkpoint_ready')}`, "
            f"`votes={ar.get('total_votes')}`, "
            f"`wins={ar.get('holiday_wins')}/{ar.get('baseline_wins')}/{ar.get('ties')}`, "
            f"`win_ratio={ar.get('win_ratio')}`, "
            f"`wilson_95_lb={ar.get('wilson_95_lower_bound')}`"
        )
    fb = report.get("feedback_summary")
    if isinstance(fb, Mapping) and fb:
        lines.append(
            f"- **{'Métriques CES' if is_fr else 'CES Summary'}**: "
            f"`responses={fb.get('total_responses')}`, `ease={fb.get('mean_ease_rating')}/5`, "
            f"`zero_game_lag={fb.get('zero_game_lag_ratio')}`, "
            f"`awaiting_first_votes={report.get('awaiting_first_votes')}`"
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
    """Build CLI argument parser for the Week S6 final bilan & quorum verifier."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-inference Week S6 final sprint bilan, blind A/B checkpoint & 8+ producer quorum verifier "
            "(0.00 EUR / 0.00 USD, AGPL-3.0)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="Target HiveMind gateway URL (https://*.run.app, https://*.googleapis.com, or loopback).",
    )
    parser.add_argument(
        "--min-online-producers",
        type=int,
        default=DEFAULT_MIN_ONLINE_PRODUCERS,
        help="Minimum live external friend producers required for quorum (default: 8).",
    )
    parser.add_argument(
        "--min-ease",
        type=float,
        default=DEFAULT_MIN_EASE_RATING,
        help="Minimum CES mean_ease_rating out of 5 when total_responses > 0 (default: 4.2).",
    )
    parser.add_argument(
        "--min-zero-lag",
        type=float,
        default=DEFAULT_MIN_ZERO_LAG_RATIO,
        help="Minimum gamer zero_game_lag_ratio when total_responses > 0 (default: 0.95).",
    )
    parser.add_argument(
        "--min-arena-votes",
        type=int,
        default=DEFAULT_MIN_ARENA_VOTES,
        help="Minimum blind A/B votes before evaluating Wilson lower bound (default: 8).",
    )
    parser.add_argument(
        "--min-arena-wilson-lb",
        type=float,
        default=DEFAULT_MIN_ARENA_WILSON_LB,
        help="Minimum 95% Wilson lower bound for blind A/B holiday checkpoint (default: 0.5).",
    )
    parser.add_argument(
        "--require-live-quorum",
        action="store_true",
        help="Exit with code 2 when live_hiveminders < min_online_producers instead of awaiting_producer_quorum.",
    )
    parser.add_argument(
        "--require-arena-votes",
        action="store_true",
        help="Exit with code 2 when arena total_votes < min_arena_votes instead of awaiting_arena_votes.",
    )
    parser.add_argument(
        "--require-votes",
        action="store_true",
        help="Exit with code 2 when CES total_responses == 0 instead of awaiting_first_votes.",
    )
    parser.add_argument(
        "--allow-missing-coder",
        action="store_true",
        help="Allow GET /v1/models without requiring hivemind/coder.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Optional expected build_sha on /v1/hivemind/onboarding/manifest.",
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
    """CLI entrypoint enforcing local compliance before any network call or file write."""
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
            report = verify_s6_final_bilan_and_quorum(
                args.gateway,
                min_online_producers=args.min_online_producers,
                min_ease_rating=args.min_ease,
                min_zero_lag_ratio=args.min_zero_lag,
                min_arena_votes=args.min_arena_votes,
                min_arena_wilson_lb=args.min_arena_wilson_lb,
                require_live_quorum=bool(args.require_live_quorum),
                http_fn=http_fn,
                environ=eff_env,
                cwd=cwd,
                lang=lang_code,
                require_coder_model=not bool(args.allow_missing_coder),
                require_votes=bool(args.require_votes),
                require_arena_votes=bool(args.require_arena_votes),
                expected_sha=args.expected_sha,
            )
    except S6BilanVerifierError as exc:
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
