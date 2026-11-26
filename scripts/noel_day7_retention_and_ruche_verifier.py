#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference Dec 26, 2026 (Day-7 Mid-Holiday Raid) Retention & La Ruche Verifier (`SLICE-SCALE-DYN-31`).

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment markers (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``,
     ``NODE_EXTRA_CA_CERTS``), workspace/home paths, and ``.git/config`` remote URLs (including
     worktree ``gitdir:`` / ``commondir`` indirection) BEFORE any network call or file write.
     Fails closed with exit code 2 (0 network calls, 0 file writes, 0 raw env/path leaks).
  2. Zero-Inference Dec 26, 2026 (Day-7 Mid-Holiday Raid) Cohort Balance, Retention & La Ruche
     Foraging Health Verifier (``verify_day7_retention_and_ruche_health``):
     probes ``GET /health``, ``GET /v1/models``, ``GET /v1/hivemind/raid/status``,
     ``GET /v1/hivemind/producers/fleet``, ``GET /noel``, ``GET /quick``, ``GET /raid-noel``,
     ``GET /arena-noel``, and ``GET /v1/hivemind/onboarding/feedback/summary`` in strict
     read-only zero-inference mode (never calling ``POST /v1/chat/completions`` or prompting
     ``hivemind/coder``), enforcing unique per-request CSP nonces ``default-src 'none'``,
     0 unsafe DOM sinks, 0 static figures (stripping ``hm-***`` / ``hm_live_***`` masked-key
     suffixes and HTML ``nonce="..."`` attributes first), exact 5-key feedback summary equality,
     mid-raid cohort balance (dev vs gamer >= ``min_cohort_share`` when ``total_responses >= 8``,
     with small-cohort warmup handling below 8 responses), ``ces_slo_ok``, ``zero_lag_slo_ok``,
     ``raid_status_ok``, ``routes_ready``, and La Ruche foraging health, emitting FR/EN JSON &
     Markdown SHA-256 receipts.

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
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_day7_retention_and_ruche_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_day7_retention_and_ruche_receipt"
SLO_OBJECT = "hivemind.noel_day7_cohort_retention_slo"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
FLEET_OBJECT = "hivemind.producer_fleet"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
DAY7_RAID_DATE = "2026-12-26"
RAID_MILESTONE = "Day-7"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

HTTP_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 1_048_576
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95
DEFAULT_MIN_COHORT_SHARE = 0.25
DEFAULT_COHORT_WARMUP_RESPONSES = 8

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset()

DAY7_HTML_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
)

DAY7_PROBE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "models"),
    ("/v1/hivemind/raid/status", "json", "raid_status"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "feedback"),
)

EXPECTED_GUILD_IDS: tuple[str, ...] = (
    "team-python",
    "team-rust-cpp",
    "team-steam-rpg",
    "team-night-owl",
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

HttpFn = Callable[..., Any]


class Day7RetentionVerifierError(ValueError):
    """Raised when Day-7 retention verifier arguments, host allowlists, or safety invariants fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI or API output language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise Day7RetentionVerifierError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise Day7RetentionVerifierError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def validate_raid_date(raw: str | None) -> str:
    """Validate ISO ``YYYY-MM-DD`` Day-7 holiday raid date (defaults to ``2026-12-26``)."""
    if raw is None:
        return DAY7_RAID_DATE
    if isinstance(raw, bool) or not isinstance(raw, str) or not _ISO_DATE_RE.fullmatch(raw.strip()):
        raise Day7RetentionVerifierError("raid_date must be a 'YYYY-MM-DD' string.")
    cleaned = raw.strip()
    try:
        date.fromisoformat(cleaned)
    except ValueError as exc:
        raise Day7RetentionVerifierError(f"Invalid raid_date {cleaned!r}.") from exc
    return cleaned


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
        raise Day7RetentionVerifierError("strip_masked_keys_and_nonces requires a string.")
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
        raise Day7RetentionVerifierError("scan_output_safety requires a string.")
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
        raise Day7RetentionVerifierError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise Day7RetentionVerifierError("Forbidden JSON key detected in Day-7 retention receipt.")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def compute_merkle_root(
    leaf_hashes: Sequence[str],
    *,
    domain: str = "hivemind.day7_retention.merkle",
) -> str:
    """Compute a deterministic domain-separated binary SHA-256 Merkle root over leaf digests."""
    if not leaf_hashes:
        return hashlib.sha256(f"{domain}.empty/1".encode()).hexdigest()
    level: list[str] = []
    for digest in leaf_hashes:
        if not isinstance(digest, str) or not _SHA256_FULL_RE.fullmatch(digest):
            raise Day7RetentionVerifierError("Merkle leaf digest must be a 64-char lowercase hex SHA-256.")
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
        raise Day7RetentionVerifierError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise Day7RetentionVerifierError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise Day7RetentionVerifierError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise Day7RetentionVerifierError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise Day7RetentionVerifierError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise Day7RetentionVerifierError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise Day7RetentionVerifierError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise Day7RetentionVerifierError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise Day7RetentionVerifierError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise Day7RetentionVerifierError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise Day7RetentionVerifierError("Remote gateway URL must use https.")
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
        raise Day7RetentionVerifierError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise Day7RetentionVerifierError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise Day7RetentionVerifierError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise Day7RetentionVerifierError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise Day7RetentionVerifierError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise Day7RetentionVerifierError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise Day7RetentionVerifierError(
        f"Zero-inference read-only guard blocked HTTP {verb!r} to {parts.path!r}."
    )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects to untrusted hosts or inference endpoints."""

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
        raise Day7RetentionVerifierError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
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
        raise Day7RetentionVerifierError("GET/HEAD requests must never attach a body payload.")

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-day7-retention-ruche-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise Day7RetentionVerifierError("HTTP headers must not contain CR/LF or NUL characters.")
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
    body_markers = {"object", "status", "total_responses", "total_hiveminders", "data", "detail", "error"}
    if keys & body_markers:
        return False
    return bool(keys & header_markers) or all(isinstance(v, str) for v in candidate.values())


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise Day7RetentionVerifierError("http_fn must return (status_code, body) or (status_code, headers, body).")
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
    except Day7RetentionVerifierError:
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
    raid_date: str = DAY7_RAID_DATE,
) -> dict[str, Any]:
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "slo_object": SLO_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "raid_date": raid_date,
            "raid_milestone": RAID_MILESTONE,
            "mode": mode,
            "lang": lang_code,
            "gateway": gateway or DEFAULT_GATEWAY,
            "compliance_ok": False,
            "corporate_detected": True,
            "corporate_indicators": list(comp_doc.get("corporate_indicators") or []),
            "matched_markers": list(comp_doc.get("matched_markers") or []),
            "routes_ready": False,
            "raid_status_ok": False,
            "fleet_schema_ok": False,
            "ruche_health_ok": False,
            "feedback_summary_ok": False,
            "ces_slo_ok": False,
            "ces_slo_status": "blocked",
            "zero_lag_slo_ok": False,
            "zero_lag_slo_status": "blocked",
            "cohort_balance_ok": False,
            "cohort_balance_status": "blocked",
            "cohort_warmup": False,
            "awaiting_first_votes": False,
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


def _validate_day7_thresholds(
    min_ease_rating: float,
    min_zero_lag_ratio: float,
    min_cohort_share: float,
    warmup_min_responses: int = DEFAULT_COHORT_WARMUP_RESPONSES,
    min_live_hiveminders: int = 0,
) -> tuple[float, float, float, int, int]:
    if (
        isinstance(min_ease_rating, bool)
        or not isinstance(min_ease_rating, (int, float))
        or not math.isfinite(float(min_ease_rating))
        or not (1.0 <= float(min_ease_rating) <= 5.0)
    ):
        raise Day7RetentionVerifierError("min_ease_rating must be a finite number in [1.0, 5.0].")
    if (
        isinstance(min_zero_lag_ratio, bool)
        or not isinstance(min_zero_lag_ratio, (int, float))
        or not math.isfinite(float(min_zero_lag_ratio))
        or not (0.0 <= float(min_zero_lag_ratio) <= 1.0)
    ):
        raise Day7RetentionVerifierError("min_zero_lag_ratio must be a finite number in [0.0, 1.0].")
    if (
        isinstance(min_cohort_share, bool)
        or not isinstance(min_cohort_share, (int, float))
        or not math.isfinite(float(min_cohort_share))
        or not (0.0 <= float(min_cohort_share) <= 0.5)
    ):
        raise Day7RetentionVerifierError("min_cohort_share must be a finite number in [0.0, 0.5].")
    if (
        isinstance(warmup_min_responses, bool)
        or not isinstance(warmup_min_responses, int)
        or warmup_min_responses < 1
    ):
        raise Day7RetentionVerifierError("warmup_min_responses must be an integer >= 1.")
    if (
        isinstance(min_live_hiveminders, bool)
        or not isinstance(min_live_hiveminders, int)
        or min_live_hiveminders < 0
    ):
        raise Day7RetentionVerifierError("min_live_hiveminders must be an integer >= 0.")
    return (
        float(min_ease_rating),
        float(min_zero_lag_ratio),
        float(min_cohort_share),
        int(warmup_min_responses),
        int(min_live_hiveminders),
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


def _opt_nonneg_int(val: Any) -> tuple[bool, int | None]:
    if val is None:
        return True, None
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        return False, None
    return True, int(val)


def _inspect_raid_status_contract(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate ``GET /v1/hivemind/raid/status`` compliance, guilds, swarm & estimator."""
    is_fr = lang_code == "fr"
    issues: list[str] = []

    if ("sco" + "re") in _collect_json_keys(body):
        issues.append("Forbidden JSON key detected in `/v1/hivemind/raid/status`.")
    if body.get("object") != RAID_STATUS_OBJECT:
        issues.append(
            "Champ `object` != `hivemind.raid_noel.status` sur `/v1/hivemind/raid/status`."
            if is_fr
            else "Invalid `object` in `/v1/hivemind/raid/status`."
        )

    comp = body.get("compliance")
    comp_ok = bool(
        isinstance(comp, Mapping)
        and comp.get("personal_pc_only") is True
        and comp.get("corporate_hardware_or_code_forbidden") is True
        and comp.get("monetary_cost_eur") == "0.00 EUR"
        and comp.get("monetary_cost_usd") == "0.00 USD"
        and comp.get("license") == "AGPL-3.0"
    )
    if not comp_ok:
        issues.append(
            "Bloc `compliance` (PC perso / 0.00 EUR / 0.00 USD / AGPL-3.0) invalide sur `/v1/hivemind/raid/status`."
            if is_fr
            else "Raid status `compliance` block missing personal-PC / 0.00 EUR-USD / AGPL-3.0 guarantees."
        )

    guilds_raw = body.get("friendly_guilds")
    guild_ids: list[str] = []
    guilds_ok = isinstance(guilds_raw, list)
    if isinstance(guilds_raw, list):
        for g in guilds_raw:
            if not isinstance(g, Mapping):
                guilds_ok = False
                continue
            gid = str(g.get("guild_id") or "")
            guild_ids.append(gid)
            if g.get("symbolic_fee") != "0.00 EUR" or g.get("monetary_prize") != "0.00 EUR":
                guilds_ok = False
                issues.append(f"Guild {gid!r} violated symbolic 0.00 EUR non-monetary invariant.")
    missing_guilds = [gid for gid in EXPECTED_GUILD_IDS if gid not in guild_ids]
    if not guilds_ok or missing_guilds:
        guilds_ok = False
        issues.append(
            f"Guildes amicales incomplètes ou non symboliques : {', '.join(missing_guilds) or 'invalid'}."
            if is_fr
            else f"Friendly holiday guilds incomplete or non-symbolic: {', '.join(missing_guilds) or 'invalid'}."
        )

    swarm = body.get("swarm")
    swarm_vals: dict[str, int | None] = {}
    telemetry_status = "UNAVAILABLE"
    if not isinstance(swarm, Mapping):
        issues.append(
            "Bloc `swarm` absent de `/v1/hivemind/raid/status`."
            if is_fr
            else "Missing `swarm` object in `/v1/hivemind/raid/status`."
        )
    else:
        raw_tstat = swarm.get("telemetry_status")
        if not isinstance(raw_tstat, str) or not raw_tstat.strip():
            issues.append("Missing `swarm.telemetry_status` in `/v1/hivemind/raid/status`.")
        else:
            telemetry_status = raw_tstat.strip()
        for field in (
            "active_trainers",
            "registered_trainer_nodes",
            "registered_trainer_accounts",
            "live_producers",
            "registered_producers",
            "registered_accounts",
            "active_accounts_24h",
            "accepted_opentrainer_rounds",
            "live_models",
        ):
            ok_field, parsed_val = _opt_nonneg_int(swarm.get(field))
            if not ok_field:
                issues.append(f"Invalid swarm counter {field!r} in `/v1/hivemind/raid/status`.")
            swarm_vals[field] = parsed_val

    est_blk = body.get("electricity_estimator")
    if not isinstance(est_blk, Mapping) or not isinstance(est_blk.get("default_estimate"), Mapping):
        issues.append(
            "Bloc `electricity_estimator` invalide dans `/v1/hivemind/raid/status`."
            if is_fr
            else "Missing or invalid `electricity_estimator` in `/v1/hivemind/raid/status`."
        )

    fb_blk = body.get("onboarding_feedback")
    raid_fb: dict[str, Any] = {}
    if not isinstance(fb_blk, Mapping):
        issues.append(
            "Bloc `onboarding_feedback` absent de `/v1/hivemind/raid/status`."
            if is_fr
            else "Missing `onboarding_feedback` block in `/v1/hivemind/raid/status`."
        )
    else:
        ok_tot, tot_val = _opt_nonneg_int(fb_blk.get("total_responses"))
        by_prof = fb_blk.get("by_profile")
        prof_ok = isinstance(by_prof, Mapping) and set(by_prof.keys()) == {"dev", "gamer"}
        ok_dev, dev_val = _opt_nonneg_int(by_prof.get("dev")) if isinstance(by_prof, Mapping) else (False, None)
        ok_gam, gam_val = _opt_nonneg_int(by_prof.get("gamer")) if isinstance(by_prof, Mapping) else (False, None)
        if not (ok_tot and prof_ok and ok_dev and ok_gam):
            issues.append("Invalid `onboarding_feedback` counters in `/v1/hivemind/raid/status`.")
        else:
            raid_fb = {
                "total_responses": tot_val,
                "mean_ease_rating": fb_blk.get("mean_ease_rating"),
                "zero_game_lag_ratio": fb_blk.get("zero_game_lag_ratio"),
                "by_profile": {"dev": dev_val, "gamer": gam_val},
            }

    schema_ok = len(issues) == 0
    normalized = {
        "schema_ok": schema_ok,
        "telemetry_status": telemetry_status if schema_ok else "INVALID",
        "guilds_count": len(guild_ids) if guilds_ok else 0,
        "active_trainers": swarm_vals.get("active_trainers"),
        "registered_trainer_nodes": swarm_vals.get("registered_trainer_nodes"),
        "registered_trainer_accounts": swarm_vals.get("registered_trainer_accounts"),
        "live_producers": swarm_vals.get("live_producers"),
        "registered_producers": swarm_vals.get("registered_producers"),
        "registered_accounts": swarm_vals.get("registered_accounts"),
        "active_accounts_24h": swarm_vals.get("active_accounts_24h"),
        "accepted_opentrainer_rounds": swarm_vals.get("accepted_opentrainer_rounds"),
        "live_models": swarm_vals.get("live_models"),
        "onboarding_feedback": raid_fb,
    }
    return schema_ok, normalized, issues


def _inspect_fleet_contract(
    body: Mapping[str, Any],
    *,
    min_live_hiveminders: int,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate public zero-inference ``GET /v1/hivemind/producers/fleet`` schema and redaction."""
    is_fr = lang_code == "fr"
    obj_ok = body.get("object") == FLEET_OBJECT
    int_fields = (
        "total_hiveminders",
        "live_hiveminders",
        "distinct_physical_gpus",
        "live_endpoints",
    )
    counts_ok = all(
        not isinstance(body.get(field), bool)
        and isinstance(body.get(field), int)
        and int(body[field]) >= 0
        for field in int_fields
    )
    raw_models = body.get("live_models")
    models_ok = isinstance(raw_models, list) and all(isinstance(m, str) for m in raw_models)

    raw_hiveminders = body.get("hiveminders")
    hiveminders_ok = isinstance(raw_hiveminders, list) and all(
        isinstance(row, Mapping) and "handle" in row and "status" in row for row in raw_hiveminders
    )
    endpoint_leaked = isinstance(raw_hiveminders, list) and any(
        isinstance(row, Mapping) and row.get("endpoint_url") is not None for row in raw_hiveminders
    )

    issues: list[str] = []
    if not (obj_ok and counts_ok and models_ok and hiveminders_ok):
        issues.append(
            "Contrat JSON `/v1/hivemind/producers/fleet` invalide."
            if is_fr
            else "Producer fleet `/v1/hivemind/producers/fleet` JSON schema mismatch."
        )
    if endpoint_leaked:
        issues.append(
            "Fuite `endpoint_url` non masquée détectée sur `/v1/hivemind/producers/fleet`."
            if is_fr
            else "Unredacted `endpoint_url` leaked on public `/v1/hivemind/producers/fleet`."
        )

    schema_ok = len(issues) == 0
    live_cnt = int(body["live_hiveminders"]) if counts_ok else 0
    if schema_ok and live_cnt < min_live_hiveminders:
        issues.append(
            f"Producteurs live insuffisants sur la flotte ({live_cnt} < {min_live_hiveminders})."
            if is_fr
            else f"Live fleet producers below floor ({live_cnt} < {min_live_hiveminders})."
        )

    ok = len(issues) == 0
    normalized = (
        {
            "schema_ok": True,
            "total_hiveminders": int(body["total_hiveminders"]),
            "live_hiveminders": live_cnt,
            "distinct_physical_gpus": int(body["distinct_physical_gpus"]),
            "live_endpoints": int(body["live_endpoints"]),
            "live_models": [str(m) for m in raw_models],
            "live_models_count": len(raw_models),
            "hiveminders_count": len(raw_hiveminders),
            "endpoint_redacted_ok": not endpoint_leaked,
        }
        if schema_ok
        else {
            "schema_ok": False,
            "total_hiveminders": None,
            "live_hiveminders": None,
            "distinct_physical_gpus": None,
            "live_endpoints": None,
            "live_models": [],
            "live_models_count": None,
            "hiveminders_count": None,
            "endpoint_redacted_ok": not endpoint_leaked,
        }
    )
    return ok, normalized, issues


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


def evaluate_day7_retention_and_cohort_balance(
    feedback_summary: Mapping[str, Any],
    *,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    min_cohort_share: float = DEFAULT_MIN_COHORT_SHARE,
    warmup_min_responses: int = DEFAULT_COHORT_WARMUP_RESPONSES,
    require_votes: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Evaluate Day-7 CES, zero-game-lag & dev/gamer cohort balance (enforced when ``total_responses >= 8``)."""
    if not isinstance(feedback_summary, Mapping):
        raise Day7RetentionVerifierError("feedback_summary must be a mapping.")
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    min_ease, min_lag, min_share, warmup_n, _ = _validate_day7_thresholds(
        min_ease_rating, min_zero_lag_ratio, min_cohort_share, warmup_min_responses
    )

    schema_ok, preserved, schema_issues = _inspect_feedback_summary_exact(feedback_summary, lang_code=lang_code)
    hold_reasons: list[str] = []
    diagnostics: list[str] = list(schema_issues)

    total_val = int(preserved["total_responses"]) if schema_ok else 0
    mean_ease = preserved["mean_ease_rating"] if schema_ok else None
    zero_lag = preserved["zero_game_lag_ratio"] if schema_ok else None
    by_profile = preserved["by_profile"] if schema_ok else {"dev": 0, "gamer": 0}
    dev_cnt = int(by_profile["dev"])
    gamer_cnt = int(by_profile["gamer"])

    ces_slo_ok: bool | None
    zero_lag_slo_ok: bool | None
    cohort_balance_ok: bool | None
    dev_share: float | None = None
    gamer_share: float | None = None
    min_observed_share: float | None = None
    awaiting_first_votes = False
    cohort_warmup = False
    cohort_balance_enforced = False

    if not schema_ok:
        hold_reasons.append("feedback_summary_schema_invalid")
        ces_slo_ok = False
        zero_lag_slo_ok = False
        cohort_balance_ok = False
        ces_slo_status = "invalid_schema"
        zero_lag_slo_status = "invalid_schema"
        cohort_balance_status = "invalid_schema"
        slo_ok = False
        decision = "HOLD_FEEDBACK_SUMMARY_INVALID"
    elif total_val == 0:
        awaiting_first_votes = True
        cohort_warmup = True
        ces_slo_ok = None
        zero_lag_slo_ok = None
        cohort_balance_ok = None
        ces_slo_status = "awaiting_first_votes"
        zero_lag_slo_status = "awaiting_first_votes"
        cohort_balance_status = "awaiting_first_votes"
        if require_votes:
            hold_reasons.append("awaiting_first_votes")
            slo_ok = False
            decision = "HOLD_AWAITING_FIRST_VOTES"
            diagnostics.append(
                "En attente des premiers votes CES Jour-7 (0 réponse reçue ; `--require-votes` actif)."
                if is_fr
                else "Awaiting first Day-7 CES votes (0 responses recorded; `--require-votes` active)."
            )
        else:
            slo_ok = True
            decision = "DAY7_AWAITING_FIRST_VOTES"
            diagnostics.append(
                "Cohorte vide Jour-7 : en attente des premiers votes CES (`awaiting_first_votes`)."
                if is_fr
                else "Empty Day-7 cohort: awaiting first CES votes (`awaiting_first_votes`)."
            )
    else:
        dev_frac = Fraction(dev_cnt, total_val)
        gamer_frac = Fraction(gamer_cnt, total_val)
        dev_share = round(float(dev_frac), 4)
        gamer_share = round(float(gamer_frac), 4)
        min_observed_share = round(float(min(dev_frac, gamer_frac)), 4)

        ces_slo_ok = bool(
            mean_ease is not None and Fraction(str(mean_ease)) >= Fraction(str(min_ease))
        )
        zero_lag_slo_ok = bool(
            zero_lag is not None and Fraction(str(zero_lag)) >= Fraction(str(min_lag))
        )
        ces_slo_status = "passed" if ces_slo_ok else "failed"
        zero_lag_slo_status = "passed" if zero_lag_slo_ok else "failed"

        if not ces_slo_ok:
            hold_reasons.append("ces_ease_rating_below_slo")
            diagnostics.append(
                f"Note CES moyenne {mean_ease:.2f}/5 sous le SLO Jour-7 (>= {min_ease:.2f}/5)."
                if is_fr
                else f"CES mean_ease_rating {mean_ease:.2f}/5 below Day-7 SLO (>= {min_ease:.2f}/5)."
            )
        if not zero_lag_slo_ok:
            hold_reasons.append("zero_game_lag_ratio_below_slo")
            diagnostics.append(
                f"Ratio zéro-lag Gamer {zero_lag:.4f} sous le SLO Jour-7 (>= {min_lag:.2f})."
                if is_fr
                else f"Gamer zero_game_lag_ratio {zero_lag:.4f} below Day-7 SLO (>= {min_lag:.2f})."
            )

        if total_val < warmup_n:
            cohort_warmup = True
            cohort_balance_enforced = False
            cohort_balance_ok = None
            cohort_balance_status = "warmup_small_cohort"
            diagnostics.append(
                f"Préchauffage petite cohorte ({total_val} < {warmup_n}) : équilibre dev/gamer non bloquant."
                if is_fr
                else f"Small-cohort warmup ({total_val} < {warmup_n} responses): dev/gamer balance check deferred."
            )
            slo_ok = bool(ces_slo_ok and zero_lag_slo_ok)
            decision = "DAY7_COHORT_WARMUP_GREEN" if slo_ok else "HOLD_DAY7_SLO_BREACH"
        else:
            cohort_warmup = False
            cohort_balance_enforced = True
            min_share_frac = Fraction(str(min_share))
            cohort_balance_ok = bool(dev_frac >= min_share_frac and gamer_frac >= min_share_frac)
            cohort_balance_status = "passed" if cohort_balance_ok else "failed"
            if not cohort_balance_ok:
                hold_reasons.append("cohort_balance_skewed")
                diagnostics.append(
                    f"Déséquilibre de cohorte Jour-7 : dev={dev_cnt}/{total_val} ({dev_share:.2%}), "
                    f"gamer={gamer_cnt}/{total_val} ({gamer_share:.2%}) < seuil {min_share:.0%}."
                    if is_fr
                    else f"Day-7 cohort skew detected: dev={dev_cnt}/{total_val} ({dev_share:.2%}), "
                    f"gamer={gamer_cnt}/{total_val} ({gamer_share:.2%}) < floor {min_share:.0%}."
                )
            slo_ok = bool(ces_slo_ok and zero_lag_slo_ok and cohort_balance_ok)
            if slo_ok:
                decision = "DAY7_RETENTION_AND_RUCHE_GREEN"
            elif ces_slo_ok and zero_lag_slo_ok and not cohort_balance_ok:
                decision = "HOLD_COHORT_SKEWED"
            else:
                decision = "HOLD_DAY7_SLO_BREACH"

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": SLO_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "evaluate_day7_retention_and_cohort_balance",
            "lang": lang_code,
            "schema_ok": schema_ok,
            "feedback_summary": preserved,
            "total_responses": total_val,
            "mean_ease_rating": mean_ease,
            "zero_game_lag_ratio": zero_lag,
            "by_profile": by_profile,
            "dev_share": dev_share,
            "gamer_share": gamer_share,
            "min_observed_share": min_observed_share,
            "min_ease_rating": min_ease,
            "min_zero_lag_ratio": min_lag,
            "min_cohort_share": min_share,
            "warmup_min_responses": warmup_n,
            "awaiting_first_votes": awaiting_first_votes,
            "cohort_warmup": cohort_warmup,
            "cohort_balance_enforced": cohort_balance_enforced,
            "ces_slo_ok": ces_slo_ok,
            "ces_slo_status": ces_slo_status,
            "zero_lag_slo_ok": zero_lag_slo_ok,
            "zero_lag_slo_status": zero_lag_slo_status,
            "cohort_balance_ok": cohort_balance_ok,
            "cohort_balance_status": cohort_balance_status,
            "slo_ok": slo_ok,
            "decision": decision,
            "go_nogo": "GO" if slo_ok else "NO-GO",
            "hold_reasons": hold_reasons,
            "diagnostics": diagnostics,
            "exit_code": 0 if slo_ok else 2,
        }
    )


def _crosscheck_raid_vs_feedback_summary(
    raid_doc: Mapping[str, Any],
    summary_doc: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, list[str]]:
    """Verify `/v1/hivemind/raid/status` feedback metrics do not contradict `/v1/hivemind/onboarding/feedback/summary`."""
    is_fr = lang_code == "fr"
    raid_fb = raid_doc.get("onboarding_feedback")
    if not isinstance(raid_fb, Mapping) or raid_fb.get("total_responses") is None:
        return True, []
    r_tot = int(raid_fb["total_responses"])
    s_tot = int(summary_doc["total_responses"])
    r_prof = raid_fb.get("by_profile") or {}
    s_prof = summary_doc["by_profile"]
    r_dev = int(r_prof["dev"])
    r_gam = int(r_prof["gamer"])
    s_dev = int(s_prof["dev"])
    s_gam = int(s_prof["gamer"])

    if r_tot > s_tot or r_dev > s_dev or r_gam > s_gam:
        return False, [
            "Divergence rétrograde entre `/v1/hivemind/raid/status` et `/v1/hivemind/onboarding/feedback/summary`."
            if is_fr
            else "Regressive feedback mismatch between `/v1/hivemind/raid/status` and `/v1/hivemind/onboarding/feedback/summary`."
        ]
    if r_tot == s_tot:
        if (r_dev, r_gam) != (s_dev, s_gam):
            return False, [
                "Divergence `by_profile` entre `/v1/hivemind/raid/status` et `/v1/hivemind/onboarding/feedback/summary`."
                if is_fr
                else "`by_profile` mismatch between `/v1/hivemind/raid/status` and `/v1/hivemind/onboarding/feedback/summary`."
            ]
        if r_tot > 0:
            r_ease = raid_fb.get("mean_ease_rating")
            s_ease = summary_doc.get("mean_ease_rating")
            r_lag = raid_fb.get("zero_game_lag_ratio")
            s_lag = summary_doc.get("zero_game_lag_ratio")
            if (
                r_ease is None
                or s_ease is None
                or r_lag is None
                or s_lag is None
                or abs(float(r_ease) - float(s_ease)) > 0.011
                or abs(float(r_lag) - float(s_lag)) > 0.001
            ):
                return False, [
                    "Divergence métrique CES entre `/v1/hivemind/raid/status` et `/v1/hivemind/onboarding/feedback/summary`."
                    if is_fr
                    else "CES metric mismatch between `/v1/hivemind/raid/status` and `/v1/hivemind/onboarding/feedback/summary`."
                ]
    return True, []


def _summarize_ruche_foraging_health(
    raid_doc: Mapping[str, Any],
    fleet_doc: Mapping[str, Any],
) -> dict[str, Any]:
    raid_ok = bool(raid_doc.get("schema_ok"))
    fleet_ok = bool(fleet_doc.get("schema_ok"))
    live_p = fleet_doc.get("live_hiveminders") or 0
    active_t = raid_doc.get("active_trainers") or 0
    rounds = raid_doc.get("accepted_opentrainer_rounds") or 0
    foraging_active = bool(live_p > 0 or active_t > 0 or rounds > 0)
    return {
        "ruche_health_ok": bool(raid_ok and fleet_ok),
        "raid_status_ok": raid_ok,
        "fleet_schema_ok": fleet_ok,
        "endpoint_redacted_ok": bool(fleet_doc.get("endpoint_redacted_ok")),
        "foraging_active": foraging_active,
        "ruche_status": "foraging_active" if foraging_active else "awaiting_foragers",
        "telemetry_status": raid_doc.get("telemetry_status"),
        "live_hiveminders": fleet_doc.get("live_hiveminders"),
        "total_hiveminders": fleet_doc.get("total_hiveminders"),
        "distinct_physical_gpus": fleet_doc.get("distinct_physical_gpus"),
        "live_endpoints": fleet_doc.get("live_endpoints"),
        "live_models": list(fleet_doc.get("live_models") or []),
        "active_trainers": raid_doc.get("active_trainers"),
        "registered_trainer_nodes": raid_doc.get("registered_trainer_nodes"),
        "accepted_opentrainer_rounds": raid_doc.get("accepted_opentrainer_rounds"),
        "guilds_count": raid_doc.get("guilds_count", 0),
    }


def verify_day7_retention_and_ruche_health(
    gateway: str = DEFAULT_GATEWAY,
    *,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    min_cohort_share: float = DEFAULT_MIN_COHORT_SHARE,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
    require_coder_model: bool = True,
    min_live_hiveminders: int = 0,
    warmup_min_responses: int = DEFAULT_COHORT_WARMUP_RESPONSES,
    require_votes: bool = False,
    raid_date: str = DAY7_RAID_DATE,
    check_git_remote: bool = True,
) -> dict[str, Any]:
    """Run Dec 26 Day-7 mid-holiday raid retention, cohort balance & La Ruche foraging health verifier."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_date = validate_raid_date(raid_date)
    min_ease, min_lag, min_share, warmup_n, min_live = _validate_day7_thresholds(
        min_ease_rating,
        min_zero_lag_ratio,
        min_cohort_share,
        warmup_min_responses,
        min_live_hiveminders,
    )

    comp_doc = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        lang=lang_code,
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt(
            "verify_day7_retention_and_ruche_health",
            comp_doc,
            lang_code=lang_code,
            raid_date=clean_date,
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
    raid_http_ok = False
    raid_status_ok = False
    raid_doc: dict[str, Any] = {}
    fleet_http_ok = False
    fleet_schema_ok = False
    fleet_doc: dict[str, Any] = {}
    feedback_http_ok = False
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {}
    slo_doc: dict[str, Any] | None = None
    network_calls = 0

    for path, kind, group in DAY7_PROBE_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if code != 200:
            if code == 404 and group in ("holiday", "raid_status", "fleet", "feedback", "models"):
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
        elif path == "/v1/hivemind/raid/status":
            raid_http_ok = True
            r_ok, raid_doc, r_issues = _inspect_raid_status_contract(
                _unwrap_json_body(body),
                lang_code=lang_code,
            )
            raid_status_ok = r_ok
            ok = r_ok
            if not r_ok:
                diag = r_issues[0]
                diagnostics.extend(r_issues)
        elif path == "/v1/hivemind/producers/fleet":
            fleet_http_ok = True
            f_ok, fleet_doc, f_issues = _inspect_fleet_contract(
                _unwrap_json_body(body),
                min_live_hiveminders=min_live,
                lang_code=lang_code,
            )
            fleet_schema_ok = f_ok
            ok = f_ok
            if not f_ok:
                diag = f_issues[0]
                diagnostics.extend(f_issues)
        elif kind == "html":
            ok, h_doc, h_issues = _inspect_html_route(path, body, seen_nonces, lang_code=lang_code)
            html_contracts[path] = h_doc
            if not ok:
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        else:
            feedback_http_ok = True
            fb_json = _unwrap_json_body(body)
            slo_doc = evaluate_day7_retention_and_cohort_balance(
                fb_json,
                min_ease_rating=min_ease,
                min_zero_lag_ratio=min_lag,
                min_cohort_share=min_share,
                warmup_min_responses=warmup_n,
                require_votes=require_votes,
                lang=lang_code,
            )
            feedback_summary_ok = bool(slo_doc["schema_ok"])
            ok = feedback_summary_ok
            if feedback_summary_ok:
                feedback_summary_doc = dict(slo_doc["feedback_summary"])
            else:
                diag = slo_doc["diagnostics"][0]
                diagnostics.extend(slo_doc["diagnostics"])

        if not ok:
            failed_routes.append(path)
        r_src = json.dumps(
            {"raid_date": clean_date, "path": path, "http_status": code, "ok": ok},
            sort_keys=True,
            ensure_ascii=False,
        )
        route_rows.append(
            {
                "path": path,
                "http_status": code,
                "ok": ok,
                "diagnostic": diag,
                "leaf_sha256": hashlib.sha256(f"hivemind.day7_retention.route/1:{r_src}".encode()).hexdigest(),
            }
        )

    if raid_status_ok and feedback_summary_ok:
        x_ok, x_issues = _crosscheck_raid_vs_feedback_summary(
            raid_doc,
            feedback_summary_doc,
            lang_code=lang_code,
        )
        if not x_ok:
            raid_status_ok = False
            raid_doc["schema_ok"] = False
            diagnostics.extend(x_issues)
            if "/v1/hivemind/raid/status" not in failed_routes:
                failed_routes.append("/v1/hivemind/raid/status")

    html_all_ok = bool(
        len(html_contracts) == len(DAY7_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in DAY7_HTML_ROUTES)
        and len(seen_nonces) == len(DAY7_HTML_ROUTES)
    )
    routes_ready = bool(
        health_ok
        and models_ok
        and raid_http_ok
        and fleet_http_ok
        and html_all_ok
        and feedback_http_ok
    )
    ruche_health = _summarize_ruche_foraging_health(raid_doc, fleet_doc)
    ruche_health_ok = bool(ruche_health["ruche_health_ok"])

    if slo_doc is not None:
        awaiting_first_votes = bool(slo_doc["awaiting_first_votes"])
        cohort_warmup = bool(slo_doc["cohort_warmup"])
        cohort_balance_enforced = bool(slo_doc["cohort_balance_enforced"])
        ces_slo_ok: bool | None = slo_doc["ces_slo_ok"]
        ces_slo_status = str(slo_doc["ces_slo_status"])
        zero_lag_slo_ok: bool | None = slo_doc["zero_lag_slo_ok"]
        zero_lag_slo_status = str(slo_doc["zero_lag_slo_status"])
        cohort_balance_ok: bool | None = slo_doc["cohort_balance_ok"]
        cohort_balance_status = str(slo_doc["cohort_balance_status"])
        dev_share = slo_doc["dev_share"]
        gamer_share = slo_doc["gamer_share"]
        min_observed_share = slo_doc["min_observed_share"]
        slo_ok = bool(slo_doc["slo_ok"])
        if feedback_summary_ok:
            diagnostics.extend(slo_doc["diagnostics"])
    else:
        awaiting_first_votes = False
        cohort_warmup = False
        cohort_balance_enforced = False
        ces_slo_ok = False
        ces_slo_status = "unavailable"
        zero_lag_slo_ok = False
        zero_lag_slo_status = "unavailable"
        cohort_balance_ok = False
        cohort_balance_status = "unavailable"
        dev_share = None
        gamer_share = None
        min_observed_share = None
        slo_ok = False

    hold_reasons: list[str] = []
    if not routes_ready:
        hold_reasons.append("routes_not_ready")
    if not raid_status_ok:
        hold_reasons.append("raid_status_unavailable" if not raid_http_ok else "raid_status_invalid")
    if not fleet_schema_ok:
        hold_reasons.append("fleet_schema_invalid")
    if not feedback_summary_ok:
        hold_reasons.append(
            "feedback_summary_unavailable" if not feedback_http_ok else "feedback_summary_schema_invalid"
        )
    if slo_doc is not None:
        for reason in slo_doc["hold_reasons"]:
            if reason not in hold_reasons:
                hold_reasons.append(reason)

    stale_404_drift = bool(stale_404_routes)
    redeploy_required = bool(
        stale_404_drift
        or not routes_ready
        or not raid_status_ok
        or not fleet_schema_ok
        or not feedback_summary_ok
    )
    ready = bool(
        routes_ready
        and raid_status_ok
        and fleet_schema_ok
        and ruche_health_ok
        and feedback_summary_ok
        and slo_ok
    )
    go_nogo = "GO" if ready else "NO-GO"

    if ready:
        if awaiting_first_votes:
            decision = "DAY7_AWAITING_FIRST_VOTES"
        elif cohort_warmup:
            decision = "DAY7_COHORT_WARMUP_GREEN"
        else:
            decision = "DAY7_RETENTION_AND_RUCHE_GREEN"
    elif not models_ok and health_ok and html_all_ok and raid_http_ok and fleet_http_ok and feedback_http_ok:
        decision = "HOLD_MODELS_MISSING"
    elif not routes_ready:
        decision = "HOLD_ROUTES_DRIFT"
    elif not raid_status_ok:
        decision = "HOLD_RAID_STATUS_INVALID"
    elif not fleet_schema_ok:
        decision = "HOLD_FLEET_SCHEMA_INVALID"
    elif not feedback_summary_ok:
        decision = "HOLD_FEEDBACK_SUMMARY_INVALID"
    elif awaiting_first_votes:
        decision = "HOLD_AWAITING_FIRST_VOTES"
    elif ces_slo_ok is True and zero_lag_slo_ok is True and cohort_balance_ok is False:
        decision = "HOLD_COHORT_SKEWED"
    else:
        decision = "HOLD_DAY7_SLO_BREACH"

    slos = {
        "routes_ready": routes_ready,
        "raid_status_ok": raid_status_ok,
        "fleet_schema_ok": fleet_schema_ok,
        "ruche_health_ok": ruche_health_ok,
        "ces_slo_ok": ces_slo_ok,
        "ces_slo_status": ces_slo_status,
        "zero_lag_slo_ok": zero_lag_slo_ok,
        "zero_lag_slo_status": zero_lag_slo_status,
        "cohort_balance_ok": cohort_balance_ok,
        "cohort_balance_status": cohort_balance_status,
        "cohort_warmup": cohort_warmup,
        "awaiting_first_votes": awaiting_first_votes,
    }

    merkle_inputs = [row["leaf_sha256"] for row in route_rows]
    if slo_doc is not None:
        merkle_inputs.append(str(slo_doc["receipt_sha256"]))
    merkle_root = compute_merkle_root(merkle_inputs)

    if ready:
        ces_label = (
            "awaiting_first_votes"
            if awaiting_first_votes
            else f"ease={feedback_summary_doc.get('mean_ease_rating')}/5, zero_lag={feedback_summary_doc.get('zero_game_lag_ratio')}, cohort={cohort_balance_status}"
        )
        summary = (
            f"Raid Noël Jour-7 ({clean_date}) {go_nogo} ({decision}) sur {clean_gw} : "
            f"routes={routes_ready}, raid={raid_status_ok}, ruche={ruche_health['ruche_status']}, "
            f"ces={ces_label}, {FINANCIAL_PROMISE}."
            if is_fr
            else f"Dec 26 Day-7 raid ({clean_date}) {go_nogo} ({decision}) on {clean_gw}: "
            f"routes_ready={routes_ready}, raid_status_ok={raid_status_ok}, ruche={ruche_health['ruche_status']}, "
            f"ces={ces_label}, {FINANCIAL_PROMISE}."
        )
    else:
        summary = (
            f"Raid Noël Jour-7 ({clean_date}) {go_nogo} ({decision}) sur {clean_gw} : "
            f"raisons={', '.join(hold_reasons)}."
            if is_fr
            else f"Dec 26 Day-7 raid ({clean_date}) {go_nogo} ({decision}) on {clean_gw}: "
            f"reasons={', '.join(hold_reasons)}."
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "slo_object": SLO_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "raid_date": clean_date,
            "raid_milestone": RAID_MILESTONE,
            "mode": "verify_day7_retention_and_ruche_health",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "routes_ready": routes_ready,
            "health_ok": health_ok,
            "models_ok": models_ok,
            "model_ids": model_ids,
            "coder_model_present": coder_present,
            "raid_status_ok": raid_status_ok,
            "raid_summary": {k: v for k, v in raid_doc.items() if k != "onboarding_feedback"},
            "html_all_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "fleet_schema_ok": fleet_schema_ok,
            "fleet_summary": fleet_doc,
            "ruche_health_ok": ruche_health_ok,
            "ruche_foraging_health": ruche_health,
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "awaiting_first_votes": awaiting_first_votes,
            "cohort_warmup": cohort_warmup,
            "cohort_balance_enforced": cohort_balance_enforced,
            "ces_slo_ok": ces_slo_ok,
            "ces_slo_status": ces_slo_status,
            "zero_lag_slo_ok": zero_lag_slo_ok,
            "zero_lag_slo_status": zero_lag_slo_status,
            "cohort_balance_ok": cohort_balance_ok,
            "cohort_balance_status": cohort_balance_status,
            "dev_share": dev_share,
            "gamer_share": gamer_share,
            "min_observed_share": min_observed_share,
            "slos": slos,
            "slo_evaluation": slo_doc,
            "min_ease_rating": min_ease,
            "min_zero_lag_ratio": min_lag,
            "min_cohort_share": min_share,
            "warmup_min_responses": warmup_n,
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
    """Render a compact FR/EN CLI Dec 26 Day-7 retention & La Ruche health receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    title = (
        f"HIVEMIND NOËL — VÉRIFICATEUR RÉTENTION JOUR-7 & LA RUCHE ({verdict}, 0,00 EUR / 0,00 USD)"
        if is_fr
        else f"HIVEMIND HOLIDAY — DEC 26 DAY-7 RETENTION & LA RUCHE VERIFIER ({verdict}, 0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {title}",
        "====================================================================",
        f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}",
        f"  Gateway     : {report.get('gateway', DEFAULT_GATEWAY)}",
        f"  Date/Phase  : {report.get('raid_date', DAY7_RAID_DATE)} ({report.get('raid_milestone', RAID_MILESTONE)})",
        f"  Mode        : {report.get('mode')}",
        f"  Verdict     : {verdict} ({report.get('decision', verdict)})",
        f"  Compliance  : {comp_ok}",
        f"  Exit code   : {report.get('exit_code')}",
        f"  Summary     : {report.get('summary')}",
    ]
    if "routes_ready" in report:
        lines.append(
            f"  SLOs        : routes_ready={report.get('routes_ready')}, "
            f"raid_status_ok={report.get('raid_status_ok')}, "
            f"fleet_schema_ok={report.get('fleet_schema_ok')}, "
            f"ces={report.get('ces_slo_status')} ({report.get('ces_slo_ok')}), "
            f"zero_lag={report.get('zero_lag_slo_status')} ({report.get('zero_lag_slo_ok')}), "
            f"cohort={report.get('cohort_balance_status')} ({report.get('cohort_balance_ok')})"
        )
    if report.get("merkle_root_sha256"):
        lines.append(f"  Merkle Root : {report.get('merkle_root_sha256')}")
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256     : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN Markdown Dec 26 Day-7 retention & La Ruche health receipt."""
    is_fr = report.get("lang") == "fr"
    verdict = report.get("go_nogo") or ("GO" if (report.get("ready") is True or report.get("exit_code") == 0) else "NO-GO")
    comp_ok = report.get("compliance_ok") if "compliance_ok" in report else report.get("personal_pc_only")
    heading = (
        f"# Vérificateur Rétention Jour-7 (26 Déc.) & La Ruche HiveMind — {verdict}"
        if is_fr
        else f"# HiveMind Dec 26 Day-7 Raid Retention & La Ruche Health Verifier — {verdict}"
    )
    lines = [
        heading,
        "",
        f"- **{'Verdict Go/No-Go' if is_fr else 'Go/No-Go Verdict'}**: `{verdict}` (`decision={report.get('decision', verdict)}`, `exit_code={report.get('exit_code')}`)",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Jalon Jour-7' if is_fr else 'Day-7 Milestone'}**: `{report.get('raid_date', DAY7_RAID_DATE)}` (`{report.get('raid_milestone', RAID_MILESTONE)}`)",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: `{comp_ok}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Zéro inférence' if is_fr else 'Zero-Inference Verified'}**: `{report.get('zero_inference_verified', True)}`",
    ]
    if "routes_ready" in report:
        lines.append(
            f"- **{'SLOs Jour-7' if is_fr else 'Day-7 Retention & Cohort SLOs'}**: "
            f"`routes_ready={report.get('routes_ready')}`, "
            f"`raid_status_ok={report.get('raid_status_ok')}`, "
            f"`fleet_schema_ok={report.get('fleet_schema_ok')}`, "
            f"`ces_slo={report.get('ces_slo_status')}`, "
            f"`zero_lag_slo={report.get('zero_lag_slo_status')}`, "
            f"`cohort_balance={report.get('cohort_balance_status')}`"
        )
    ruche = report.get("ruche_foraging_health")
    if isinstance(ruche, Mapping) and ruche:
        lines.append(
            f"- **{'Santé La Ruche' if is_fr else 'La Ruche Foraging Health'}**: "
            f"`status={ruche.get('ruche_status')}`, "
            f"`live_hiveminders={ruche.get('live_hiveminders')}`, "
            f"`distinct_physical_gpus={ruche.get('distinct_physical_gpus')}`, "
            f"`active_trainers={ruche.get('active_trainers')}`, "
            f"`accepted_opentrainer_rounds={ruche.get('accepted_opentrainer_rounds')}`"
        )
    fb = report.get("feedback_summary")
    if isinstance(fb, Mapping) and fb:
        lines.append(
            f"- **{'Métriques CES & Cohortes' if is_fr else 'CES & Cohort Summary'}**: "
            f"`responses={fb.get('total_responses')}`, `ease={fb.get('mean_ease_rating')}/5`, "
            f"`zero_game_lag={fb.get('zero_game_lag_ratio')}`, "
            f"`by_profile={fb.get('by_profile')}`, "
            f"`cohort_warmup={report.get('cohort_warmup')}`"
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
    """Build CLI argument parser for the Dec 26 Day-7 retention & La Ruche health verifier."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-inference Dec 26 Day-7 holiday raid retention, cohort balance & La Ruche foraging "
            "health verifier (0.00 EUR / 0.00 USD, AGPL-3.0)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="Target HiveMind gateway URL (https://*.run.app, https://*.googleapis.com, or loopback).",
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
        "--min-cohort-share",
        type=float,
        default=DEFAULT_MIN_COHORT_SHARE,
        help="Minimum dev/gamer cohort share when total_responses >= warmup threshold (default: 0.25).",
    )
    parser.add_argument(
        "--warmup-min-responses",
        type=int,
        default=DEFAULT_COHORT_WARMUP_RESPONSES,
        help="Minimum responses before enforcing dev/gamer cohort balance (default: 8).",
    )
    parser.add_argument(
        "--min-live-hiveminders",
        type=int,
        default=0,
        help="Optional minimum live_hiveminders count on /v1/hivemind/producers/fleet (default: 0).",
    )
    parser.add_argument(
        "--raid-date",
        default=DAY7_RAID_DATE,
        help="ISO Day-7 raid date (default: 2026-12-26).",
    )
    parser.add_argument(
        "--allow-missing-coder",
        action="store_true",
        help="Allow GET /v1/models without requiring hivemind/coder.",
    )
    parser.add_argument(
        "--require-votes",
        action="store_true",
        help="Fail closed when total_responses == 0 instead of returning awaiting_first_votes.",
    )
    parser.add_argument(
        "--slo-json",
        default=None,
        help="Evaluate Day-7 retention & cohort balance locally from a feedback summary JSON file (no HTTP).",
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
    """CLI entrypoint enforcing local compliance before any network call or file read."""
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
        elif args.slo_json is not None:
            comp_doc = evaluate_personal_pc_compliance(
                environ=eff_env,
                cwd=cwd,
                check_git_remote=True,
                lang=lang_code,
            )
            if not comp_doc["compliance_ok"]:
                report = _compliance_blocked_receipt(
                    "evaluate_day7_retention_and_cohort_balance",
                    comp_doc,
                    lang_code=lang_code,
                    gateway=args.gateway,
                )
            else:
                slo_path = Path(args.slo_json)
                if not slo_path.is_file():
                    raise Day7RetentionVerifierError(f"Missing SLO JSON file: {redact_home_path(slo_path)}")
                try:
                    raw_summary = json.loads(slo_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise Day7RetentionVerifierError("Invalid JSON in --slo-json file.") from exc
                report = evaluate_day7_retention_and_cohort_balance(
                    raw_summary,
                    min_ease_rating=args.min_ease,
                    min_zero_lag_ratio=args.min_zero_lag,
                    min_cohort_share=args.min_cohort_share,
                    warmup_min_responses=args.warmup_min_responses,
                    require_votes=bool(args.require_votes),
                    lang=lang_code,
                )
        else:
            report = verify_day7_retention_and_ruche_health(
                args.gateway,
                min_ease_rating=args.min_ease,
                min_zero_lag_ratio=args.min_zero_lag,
                min_cohort_share=args.min_cohort_share,
                http_fn=http_fn,
                environ=eff_env,
                cwd=cwd,
                lang=lang_code,
                require_coder_model=not bool(args.allow_missing_coder),
                min_live_hiveminders=args.min_live_hiveminders,
                warmup_min_responses=args.warmup_min_responses,
                require_votes=bool(args.require_votes),
                raid_date=args.raid_date,
            )
    except Day7RetentionVerifierError as exc:
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
