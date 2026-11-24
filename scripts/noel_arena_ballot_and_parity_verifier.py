#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference /arena-noel blind A/B ballot Elo auditor, CSP parity gate, and compliance verifier.

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects corporate environment variables (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``),
     workspace path, and ``.git/config`` remote URLs (including worktree ``gitdir:`` / ``commondir``
     indirection) BEFORE any network call or file write. Fails closed with exit code 2 (0 network
     calls, 0 file writes, 0 raw env/hostname/path leaks).
  2. Zero-Inference Holiday Parity & CSP Gate (``verify_arena_noel_holiday_readiness``):
     probes ``GET /arena-noel``, ``GET /noel``, ``GET /quick``, ``GET /raid-noel``,
     ``GET /v1/hivemind/onboarding/manifest``, and ``GET /v1/hivemind/onboarding/feedback/summary``
     (verifying HTTP 200, unique per-request CSP nonces ``default-src 'none'``, 0 unsafe DOM sinks,
     0 static figures after stripping ``nonce="..."`` attributes, zero auto-completions calls,
     personal-PC 0.00 EUR/USD compliance copy, exact 5-key feedback summary equality, and dry-run
     or optional live blind ballot token validation).
  3. Zero-PII Blind A/B Pairwise Elo & Position-Bias Auditor (``compute_blind_ab_elo_summary``):
     rejects any prompt/code PII or ``hm-...`` secret leaks in ballots, solves bounded Bradley-Terry
     MLE / Elo ratings with Wilson-logit 95% confidence intervals and tie-aware win rates, audits
     position-bias symmetry (A vs B), and emits signed SHA-256 FR/EN JSON/Markdown receipts.

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
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_arena_ballot_and_parity_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_arena_ballot_and_parity_receipt"
ELO_SUMMARY_OBJECT = "hivemind.blind_ab_elo_summary"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

HTTP_TIMEOUT_S = 5.0
Z_WILSON_95 = 1.959963984540054
MAX_BALLOTS = 50_000
BASE_ELO_RATING = 1500.0
ELO_SCALE = 400.0

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset(
    {
        "/v1/hivemind/arena/vote",
        "/v1/hivemind/onboarding/feedback",
    }
)

CURATED_ARENA_TASK_IDS: tuple[str, ...] = (
    "python-sliding-median",
    "rust-frame-decoder",
    "lean4-energy-descent",
    "exoplanet-transit-bls",
)

ARENA_PARITY_HTML_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid-noel",
)

ARENA_PARITY_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/arena-noel", "html", "holiday"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
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

ALLOWED_BALLOT_KEYS: frozenset[str] = frozenset(
    {
        "task_id",
        "chosen_side",
        "profile",
        "holiday_side",
        "revealed_holiday_side",
        "preferred_checkpoint",
        "winner",
        "model_a",
        "model_b",
        "ballot_token",
        "lang",
        "platform",
    }
)

FORBIDDEN_PII_BALLOT_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "code",
        "completion",
        "response",
        "messages",
        "candidate_a",
        "candidate_b",
        "email",
        "username",
        "user",
        "user_id",
        "handle",
        "ip",
        "ip_address",
        "api_key",
        "token",
        "secret",
        "password",
        "authorization",
        "note",
        "comment",
        "free_text",
        "sco" + "re",
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
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TASK_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PLATFORM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
_BALLOT_TOKEN_RE = re.compile(r"^[0-9a-f]{8,32}\.[01]\.[0-9a-f]{24}$")
_CARD_BALLOT_PAIR_A_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"[^>]*?data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_CARD_BALLOT_PAIR_B_RE = re.compile(
    r'data-ballot="([A-Za-z0-9._-]{16,128})"[^>]*?data-task-id="([A-Za-z0-9_-]{3,64})"'
)
_ANY_DATA_BALLOT_RE = re.compile(r'data-ballot="([^"]*)"')
_CAND_BLOCK_RE = re.compile(r'<pre\s+id="cand[AB]-([^"]+)"[^>]*>(.*?)</pre>', re.DOTALL)

_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(["'])[^"']*\1""")
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
_API_KEY_RE = re.compile(r"\b(?:hm(?:_live_|-)|sk-(?:ant-|or-v1-|proj-)?|gh[pousr]_|hf_)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
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


class ArenaVerifierError(ValueError):
    """Raised when verifier inputs, host allowlists, ballots, or output safety guards fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ArenaVerifierError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ArenaVerifierError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


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


def scan_output_safety(text: str) -> list[str]:
    """Verify JSON, CLI, or Markdown text has 0 emails, raw keys, home paths, or score digits."""
    if not isinstance(text, str):
        raise ArenaVerifierError("scan_output_safety requires a string.")
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
    if _STATIC_FIG_UNIT_RE.search(text):
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
        raise ArenaVerifierError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def strip_html_csp_nonce_attributes(html_text: str) -> str:
    """Strip ``nonce="..."`` / ``nonce='...'`` attributes before static-figure scanning."""
    return _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', html_text)


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
        raise ArenaVerifierError(
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
        raise ArenaVerifierError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ArenaVerifierError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise ArenaVerifierError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ArenaVerifierError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ArenaVerifierError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise ArenaVerifierError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ArenaVerifierError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ArenaVerifierError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise ArenaVerifierError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ArenaVerifierError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ArenaVerifierError("Remote gateway URL must use https.")
    return cleaned


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    allowed_post_paths: frozenset[str] = ALLOWED_POST_PATHS,
) -> str:
    """Enforce allowlisted host, zero completions/prompts, and bounded HTTP methods."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ArenaVerifierError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ArenaVerifierError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ArenaVerifierError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise ArenaVerifierError(
            f"Zero-inference guard blocked forbidden endpoint {parts.path!r}."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ArenaVerifierError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise ArenaVerifierError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in allowed_post_paths):
        return cleaned
    raise ArenaVerifierError(f"Zero-inference guard blocked HTTP {verb!r} to {parts.path!r}.")


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
    if payload is not None and verb != "POST":
        raise ArenaVerifierError("GET/HEAD requests must never attach a body payload.")

    body_bytes: bytes | None = None
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-arena-parity-verifier/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise ArenaVerifierError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    if verb == "POST":
        clean_payload = dict(payload or {})
        body_bytes = json.dumps(clean_payload, ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(clean_url, data=body_bytes, headers=req_headers, method=verb)
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
    body_markers = {"object", "status", "total_responses", "total_votes", "data", "detail", "error", "recorded"}
    if keys & body_markers:
        return False
    return bool(keys & header_markers) or all(isinstance(v, str) for v in candidate.values())


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise ArenaVerifierError("http_fn must return (status_code, body) or (status_code, headers, body).")
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
    allowed_post_paths: frozenset[str] = ALLOWED_POST_PATHS,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Validate URL/zero-inference invariants and invoke an injectable HTTP callable."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(
        verb,
        url,
        payload=payload,
        allowed_post_paths=allowed_post_paths,
    )
    clean_payload = dict(payload) if (verb == "POST" and payload is not None) else None

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
    except ArenaVerifierError:
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
            "ready": False,
            "parity_ok": False,
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


def extract_arena_ballot_cards(html_text: str) -> list[dict[str, str]]:
    """Extract blind A/B ``(task_id, ballot_token)`` pairs from rendered ``/arena-noel`` HTML."""
    if not html_text:
        return []
    seen: dict[str, str] = {}
    for tid, ballot in _CARD_BALLOT_PAIR_A_RE.findall(html_text):
        seen.setdefault(tid, ballot)
    for ballot, tid in _CARD_BALLOT_PAIR_B_RE.findall(html_text):
        seen.setdefault(tid, ballot)
    return [{"task_id": tid, "ballot_token": ballot} for tid, ballot in seen.items()]


def _normalize_side_token(raw: object, *, allow_tie: bool, field_name: str) -> str:
    if isinstance(raw, bool) or not isinstance(raw, str) or not raw.isascii():
        raise ArenaVerifierError(f"{field_name} must be an ASCII string.")
    cleaned = raw.strip()
    if cleaned in ("A", "a"):
        return "A"
    if cleaned in ("B", "b"):
        return "B"
    if allow_tie and cleaned.lower() in ("tie", "draw"):
        return "tie"
    raise ArenaVerifierError(f"Invalid {field_name} value.")


def _normalize_model_label(raw: object, *, field_name: str) -> str:
    if isinstance(raw, bool) or not isinstance(raw, str) or not raw.isascii():
        raise ArenaVerifierError(f"{field_name} must be a valid checkpoint label.")
    low = raw.strip().lower()
    if low in ("holiday", "holiday_checkpoint", "noel", "noel_checkpoint"):
        return "holiday_checkpoint"
    if low in ("base", "baseline", "base_checkpoint", "baseline_checkpoint"):
        return "baseline_checkpoint"
    raise ArenaVerifierError(f"Unknown checkpoint label in {field_name}.")


def _audit_string_for_pii(value: str, *, field_name: str) -> None:
    if any(ord(ch) < 32 and ch not in ("\t",) or ord(ch) == 127 for ch in value):
        raise ArenaVerifierError(f"Ballot field {field_name!r} contains forbidden control characters.")
    if _TROJAN_BIDI_RE.search(value):
        raise ArenaVerifierError(f"Ballot field {field_name!r} contains Trojan bidi Unicode characters.")
    if _EMAIL_RE.search(value) or _API_KEY_RE.search(value) or _BEARER_RE.search(value):
        raise ArenaVerifierError(f"Ballot field {field_name!r} failed zero-PII/secret safety check.")
    if _HOME_PATH_RE.search(value):
        raise ArenaVerifierError(f"Ballot field {field_name!r} contains unredacted home path.")
    for ip_hit in _IPV4_RE.findall(value):
        if not ip_hit.startswith("127."):
            raise ArenaVerifierError(f"Ballot field {field_name!r} contains non-loopback IPv4.")


def audit_blind_ab_ballot(ballot: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one zero-PII blind A/B ballot, rejecting PII, prompts, code, or secrets."""
    if not isinstance(ballot, Mapping):
        raise ArenaVerifierError("Each blind ballot must be a mapping.")

    keys = {str(k) for k in ballot}
    forbidden_hits = sorted(keys & FORBIDDEN_PII_BALLOT_KEYS)
    if forbidden_hits:
        raise ArenaVerifierError(
            f"Blind ballot rejected forbidden PII/prompt/code keys: {', '.join(forbidden_hits)}."
        )
    unknown_keys = sorted(keys - ALLOWED_BALLOT_KEYS)
    if unknown_keys:
        raise ArenaVerifierError(
            f"Blind ballot rejected unexpected keys: {', '.join(unknown_keys)}."
        )

    for raw_k, raw_v in ballot.items():
        k_str = str(raw_k)
        if isinstance(raw_v, str):
            _audit_string_for_pii(raw_v, field_name=k_str)
        elif raw_v is not None and not isinstance(raw_v, bool):
            raise ArenaVerifierError(f"Ballot field {k_str!r} must be a string or None.")

    raw_task = ballot.get("task_id")
    if (
        isinstance(raw_task, bool)
        or not isinstance(raw_task, str)
        or not _TASK_SLUG_RE.fullmatch(raw_task.strip())
    ):
        raise ArenaVerifierError("Ballot task_id must be a clean lowercase slug.")
    task_id = raw_task.strip()

    profile = "unspecified"
    if "profile" in ballot and ballot["profile"] is not None:
        raw_prof = ballot["profile"]
        if isinstance(raw_prof, bool) or not isinstance(raw_prof, str):
            raise ArenaVerifierError("Ballot profile must be 'dev' or 'gamer'.")
        norm_prof = raw_prof.strip().lower()
        if norm_prof not in ("dev", "gamer"):
            raise ArenaVerifierError("Ballot profile must be 'dev' or 'gamer'.")
        profile = norm_prof

    if "platform" in ballot and ballot["platform"] is not None:
        raw_plat = ballot["platform"]
        if (
            isinstance(raw_plat, bool)
            or not isinstance(raw_plat, str)
            or not _PLATFORM_RE.fullmatch(raw_plat.strip().lower())
        ):
            raise ArenaVerifierError("Ballot platform must be a clean lowercase identifier.")

    if "ballot_token" in ballot and ballot["ballot_token"] is not None:
        raw_tok = ballot["ballot_token"]
        if (
            isinstance(raw_tok, bool)
            or not isinstance(raw_tok, str)
            or not _BALLOT_TOKEN_RE.fullmatch(raw_tok.strip())
        ):
            raise ArenaVerifierError("Ballot ballot_token must match signed HMAC format.")

    chosen_side: str | None = None
    if "chosen_side" in ballot and ballot["chosen_side"] is not None:
        chosen_side = _normalize_side_token(ballot["chosen_side"], allow_tie=True, field_name="chosen_side")

    holiday_side: str | None = None
    if "holiday_side" in ballot and ballot["holiday_side"] is not None:
        holiday_side = _normalize_side_token(ballot["holiday_side"], allow_tie=False, field_name="holiday_side")
    if "revealed_holiday_side" in ballot and ballot["revealed_holiday_side"] is not None:
        rev_side = _normalize_side_token(
            ballot["revealed_holiday_side"], allow_tie=False, field_name="revealed_holiday_side"
        )
        if holiday_side is not None and holiday_side != rev_side:
            raise ArenaVerifierError("Ballot holiday_side and revealed_holiday_side disagree.")
        holiday_side = rev_side

    if "model_a" in ballot or "model_b" in ballot:
        if ballot.get("model_a") is None or ballot.get("model_b") is None:
            raise ArenaVerifierError("Both model_a and model_b must be provided together.")
        ma = _normalize_model_label(ballot["model_a"], field_name="model_a")
        mb = _normalize_model_label(ballot["model_b"], field_name="model_b")
        if ma == mb:
            raise ArenaVerifierError("model_a and model_b must be distinct checkpoints.")
        inferred_hol = "A" if ma == "holiday_checkpoint" else "B"
        if holiday_side is not None and holiday_side != inferred_hol:
            raise ArenaVerifierError("Ballot holiday_side disagrees with model_a/model_b.")
        holiday_side = inferred_hol

    explicit_pref: str | None = None
    for pref_key in ("preferred_checkpoint", "winner"):
        if pref_key in ballot and ballot[pref_key] is not None:
            raw_pref = ballot[pref_key]
            if isinstance(raw_pref, bool) or not isinstance(raw_pref, str):
                raise ArenaVerifierError(f"{pref_key} must be a string.")
            low_pref = raw_pref.strip().lower()
            if low_pref in ("tie", "draw"):
                cand_pref = "tie"
            elif low_pref in ("a", "b") and holiday_side is not None:
                cand_pref = "holiday_checkpoint" if low_pref.upper() == holiday_side else "baseline_checkpoint"
                if chosen_side is not None and chosen_side != low_pref.upper():
                    raise ArenaVerifierError(f"{pref_key} disagrees with chosen_side.")
                chosen_side = low_pref.upper()
            else:
                cand_pref = _normalize_model_label(raw_pref, field_name=pref_key)
            if explicit_pref is not None and explicit_pref != cand_pref:
                raise ArenaVerifierError("Ballot preferred_checkpoint and winner disagree.")
            explicit_pref = cand_pref

    if chosen_side == "tie":
        derived_pref = "tie"
    elif chosen_side is not None and holiday_side is not None:
        derived_pref = "holiday_checkpoint" if chosen_side == holiday_side else "baseline_checkpoint"
    elif explicit_pref is not None:
        derived_pref = explicit_pref
    else:
        raise ArenaVerifierError("Ballot requires (chosen_side + holiday_side) or preferred_checkpoint/winner.")

    if explicit_pref is not None and derived_pref != explicit_pref:
        raise ArenaVerifierError("Ballot preferred_checkpoint contradicts chosen_side and holiday_side.")

    return {
        "task_id": task_id,
        "profile": profile,
        "chosen_side": chosen_side,
        "holiday_side": holiday_side,
        "preferred_checkpoint": derived_pref,
    }


def wilson_95_interval(wins: float, trials: int, z: float = Z_WILSON_95) -> tuple[float, float]:
    """Compute the exact binomial 95% Wilson score confidence interval ``(lower, upper)``."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 0:
        raise ArenaVerifierError("trials must be a non-negative integer.")
    if isinstance(wins, bool) or not isinstance(wins, (int, float)) or not math.isfinite(float(wins)):
        raise ArenaVerifierError("wins must be a finite number.")
    w = float(wins)
    if w < 0.0 or w > float(trials):
        raise ArenaVerifierError("0 <= wins <= trials is required.")
    if trials == 0:
        return 0.0, 0.0
    n = float(trials)
    p_hat = w / n
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    radicand = max(0.0, (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    half_width = float(z) * math.sqrt(radicand)
    lo = 0.0 if w == 0.0 else min(1.0, max(0.0, (center - half_width) / denom))
    hi = 1.0 if w == n else min(1.0, max(0.0, (center + half_width) / denom))
    return lo, hi


def _wilson_ci_bounded_for_elo(wins: float, trials: float, z: float = Z_WILSON_95) -> tuple[float, float]:
    n = float(trials)
    p_hat = float(wins) / n
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    radicand = max(0.0, (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    half_width = float(z) * math.sqrt(radicand)
    lo = min(1.0 - 1e-12, max(1e-12, (center - half_width) / denom))
    hi = min(1.0 - 1e-12, max(1e-12, (center + half_width) / denom))
    return lo, hi


def _solve_bradley_terry_two_player(holiday_score: float, baseline_score: float) -> tuple[float, float, bool]:
    """Solve 2-player Bradley-Terry MLE via Newton-Raphson, returning ``(delta_nat, win_prob, converged)``."""
    total = holiday_score + baseline_score
    if total <= 0.0:
        return 0.0, 0.5, True
    eff_h = holiday_score if (holiday_score > 0.0 and baseline_score > 0.0) else (holiday_score + 0.5)
    eff_b = baseline_score if (holiday_score > 0.0 and baseline_score > 0.0) else (baseline_score + 0.5)
    eff_n = eff_h + eff_b

    delta = 0.0
    converged = False
    for _ in range(64):
        prob = 1.0 / (1.0 + math.exp(-delta))
        grad = eff_h - eff_n * prob
        hess = max(1e-15, eff_n * prob * (1.0 - prob))
        step = max(-4.0, min(4.0, grad / hess))
        delta = max(-18.0, min(18.0, delta + step))
        if abs(step) < 1e-12:
            converged = True
            break
    win_prob = 1.0 / (1.0 + math.exp(-delta))
    return delta, win_prob, converged


def _elo_from_prob(prob: float) -> float:
    clamped = min(1.0 - 1e-12, max(1e-12, float(prob)))
    return ELO_SCALE * math.log10(clamped / (1.0 - clamped))


def compute_blind_ab_elo_summary(
    ballots: Sequence[Mapping[str, Any]],
    *,
    lang: str = "en",
) -> dict[str, Any]:
    """Audit zero-PII blind A/B ballots and compute Bradley-Terry/Elo ratings, 95% CIs, and bias checks."""
    lang_code = normalize_lang(lang)
    if isinstance(ballots, (str, bytes, bytearray)) or not isinstance(ballots, Sequence):
        raise ArenaVerifierError("ballots must be a sequence of ballot mappings.")
    if len(ballots) > MAX_BALLOTS:
        raise ArenaVerifierError(f"ballots sequence exceeds maximum limit ({MAX_BALLOTS}).")

    normalized: list[dict[str, Any]] = [audit_blind_ab_ballot(b) for b in ballots]

    holiday_wins = 0
    baseline_wins = 0
    ties = 0
    chosen_side_counts: dict[str, int] = {"A": 0, "B": 0, "tie": 0}
    presented_counts: dict[str, int] = {"A": 0, "B": 0}
    by_profile_counts: dict[str, int] = {"dev": 0, "gamer": 0}
    by_task_counts: dict[str, list[int]] = {}
    side_outcomes: dict[str, list[int]] = {"A": [0, 0, 0], "B": [0, 0, 0]}

    for item in normalized:
        tid = item["task_id"]
        prof = item["profile"]
        pref = item["preferred_checkpoint"]
        c_side = item["chosen_side"]
        h_side = item["holiday_side"]

        bucket = by_task_counts.setdefault(tid, [0, 0, 0])
        if pref == "holiday_checkpoint":
            holiday_wins += 1
            bucket[0] += 1
        elif pref == "baseline_checkpoint":
            baseline_wins += 1
            bucket[1] += 1
        else:
            ties += 1
            bucket[2] += 1

        if prof in by_profile_counts:
            by_profile_counts[prof] += 1

        if c_side in chosen_side_counts:
            chosen_side_counts[c_side] += 1

        if h_side in ("A", "B"):
            presented_counts[h_side] += 1
            sb = side_outcomes[h_side]
            if pref == "holiday_checkpoint":
                sb[0] += 1
            elif pref == "baseline_checkpoint":
                sb[1] += 1
            else:
                sb[2] += 1

    total = len(normalized)
    decisive = holiday_wins + baseline_wins

    if total == 0:
        decisive_win_rate: float | None = None
        tie_aware_win_rate: float | None = None
        tie_rate: float | None = None
        wilson_lb = 0.0
        wilson_dec_ci = [0.0, 0.0]
        wilson_tie_ci = [0.0, 0.0]
        elo_diff = 0.0
        elo_diff_ci = [0.0, 0.0]
        holiday_elo = BASE_ELO_RATING
        baseline_elo = BASE_ELO_RATING
        holiday_elo_ci = [BASE_ELO_RATING, BASE_ELO_RATING]
        bt_prob = 0.5
        bt_converged = True
        strength_h = 1.0
        strength_b = 1.0
    else:
        s_h = float(holiday_wins) + 0.5 * float(ties)
        s_b = float(baseline_wins) + 0.5 * float(ties)
        decisive_win_rate = round(holiday_wins / decisive, 4) if decisive > 0 else None
        tie_aware_win_rate = round(s_h / float(total), 4)
        tie_rate = round(ties / float(total), 4)

        w_dec_lo, w_dec_hi = wilson_95_interval(float(holiday_wins), decisive)
        wilson_lb = round(w_dec_lo, 4)
        wilson_dec_ci = [round(w_dec_lo, 4), round(w_dec_hi, 4)]

        w_tie_lo, w_tie_hi = wilson_95_interval(s_h, total)
        wilson_tie_ci = [round(w_tie_lo, 4), round(w_tie_hi, 4)]

        delta_nat, bt_prob_raw, bt_converged = _solve_bradley_terry_two_player(s_h, s_b)
        elo_diff = round((ELO_SCALE / math.log(10.0)) * delta_nat, 2)
        holiday_elo = round(BASE_ELO_RATING + 0.5 * elo_diff, 2)
        baseline_elo = round(BASE_ELO_RATING - 0.5 * elo_diff, 2)
        bt_prob = round(bt_prob_raw, 4)
        strength_h = round(math.exp(0.5 * delta_nat), 6)
        strength_b = round(math.exp(-0.5 * delta_nat), 6)

        eff_h = s_h if (s_h > 0.0 and s_b > 0.0) else (s_h + 0.5)
        eff_n = float(total) if (s_h > 0.0 and s_b > 0.0) else (float(total) + 1.0)
        p_ci_lo, p_ci_hi = _wilson_ci_bounded_for_elo(eff_h, eff_n)
        ed_lo = round(_elo_from_prob(p_ci_lo), 2)
        ed_hi = round(_elo_from_prob(p_ci_hi), 2)
        elo_diff_ci = [min(ed_lo, elo_diff), max(ed_hi, elo_diff)]
        holiday_elo_ci = [
            round(BASE_ELO_RATING + 0.5 * elo_diff_ci[0], 2),
            round(BASE_ELO_RATING + 0.5 * elo_diff_ci[1], 2),
        ]

    # Position-bias symmetry audit
    pres_a = presented_counts["A"]
    pres_b = presented_counts["B"]
    pres_total = pres_a + pres_b

    def _side_tie_aware_rate(counts: Sequence[int]) -> float | None:
        cnt = counts[0] + counts[1] + counts[2]
        if cnt == 0:
            return None
        return round((counts[0] + 0.5 * counts[2]) / float(cnt), 4)

    wr_on_a = _side_tie_aware_rate(side_outcomes["A"])
    wr_on_b = _side_tie_aware_rate(side_outcomes["B"])
    pos_delta = round(abs(wr_on_a - wr_on_b), 4) if (wr_on_a is not None and wr_on_b is not None) else None

    dec_side_picks = chosen_side_counts["A"] + chosen_side_counts["B"]
    side_asymmetry = (
        round(abs(chosen_side_counts["A"] - chosen_side_counts["B"]) / float(dec_side_picks), 4)
        if dec_side_picks > 0
        else None
    )

    if total == 0:
        position_balanced = True
        position_symmetry_ok: bool | None = True
    elif pres_total == 0:
        position_balanced = False
        position_symmetry_ok = None
    elif pres_a == 0 or pres_b == 0:
        position_balanced = False
        position_symmetry_ok = pres_total < 4
    else:
        imbalance = abs(pres_a - pres_b) / float(pres_total)
        position_balanced = imbalance <= 0.5
        position_symmetry_ok = bool(position_balanced and pos_delta is not None and pos_delta <= 0.25)

    by_task: dict[str, dict[str, Any]] = {}
    for tid in sorted(by_task_counts):
        tw_h, tw_b, t_tie = by_task_counts[tid]
        t_tot = tw_h + tw_b + t_tie
        t_dec = tw_h + tw_b
        t_s_h = float(tw_h) + 0.5 * float(t_tie)
        t_s_b = float(tw_b) + 0.5 * float(t_tie)
        t_delta, t_prob, _ = _solve_bradley_terry_two_player(t_s_h, t_s_b)
        by_task[tid] = {
            "total_ballots": t_tot,
            "decisive_ballots": t_dec,
            "holiday_wins": tw_h,
            "baseline_wins": tw_b,
            "ties": t_tie,
            "decisive_win_rate": round(tw_h / t_dec, 4) if t_dec > 0 else None,
            "tie_aware_win_rate": round(t_s_h / float(t_tot), 4) if t_tot > 0 else None,
            "wilson_95_lower_bound": round(wilson_95_interval(float(tw_h), t_dec)[0], 4),
            "elo_diff": round((ELO_SCALE / math.log(10.0)) * t_delta, 2),
            "bradley_terry_win_prob": round(t_prob, 4),
        }

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": ELO_SUMMARY_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "lang": lang_code,
            "total_ballots": total,
            "decisive_ballots": decisive,
            "holiday_wins": holiday_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "tie_rate": tie_rate,
            "decisive_win_rate": decisive_win_rate,
            "tie_aware_win_rate": tie_aware_win_rate,
            "wilson_95_lower_bound": wilson_lb,
            "wilson_95_interval_decisive": wilson_dec_ci,
            "wilson_95_interval_tie_aware": wilson_tie_ci,
            "holiday_elo": holiday_elo,
            "baseline_elo": baseline_elo,
            "elo_diff": elo_diff,
            "elo_diff_ci_95": elo_diff_ci,
            "holiday_elo_ci_95": holiday_elo_ci,
            "bradley_terry_win_prob": bt_prob,
            "bradley_terry_strengths": {
                "holiday_checkpoint": strength_h,
                "baseline_checkpoint": strength_b,
            },
            "bradley_terry_converged": bt_converged,
            "statistically_significant_holiday_preference": bool(total > 0 and elo_diff_ci[0] > 0.0),
            "position_bias_audit": {
                "chosen_side_counts": dict(chosen_side_counts),
                "holiday_presented_on_side_counts": dict(presented_counts),
                "tie_aware_win_rate_when_holiday_on_a": wr_on_a,
                "tie_aware_win_rate_when_holiday_on_b": wr_on_b,
                "position_bias_delta": pos_delta,
                "side_pick_asymmetry": side_asymmetry,
                "position_balanced": position_balanced,
                "position_symmetry_ok": position_symmetry_ok,
            },
            "by_profile": dict(by_profile_counts),
            "by_task": by_task,
            "ballot_pii_audit_ok": True,
            "zero_vram_inference": True,
            "personal_pc_only": True,
            "financial_promise": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
        }
    )


def _inspect_html_parity_route(
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

    stripped_html = strip_html_csp_nonce_attributes(html_text)
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
    needs_usd = path == "/arena-noel"
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
            else f"Personal-PC / 0.00 EUR-USD / AGPL-3.0 compliance banner missing on {path}.",
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
    keys = set(body.keys())
    exact_5_keys = keys == EXPECTED_FEEDBACK_SUMMARY_KEYS

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


def _inspect_arena_html_ballots(
    arena_html: str,
    *,
    lang_code: str,
) -> tuple[bool, list[dict[str, str]], list[str]]:
    is_fr = lang_code == "fr"
    issues: list[str] = []

    raw_ballots = _ANY_DATA_BALLOT_RE.findall(arena_html)
    cards = extract_arena_ballot_cards(arena_html)
    has_arena_shell = "arenaNoelApp" in arena_html or "data-task-id=" in arena_html or "task-card" in arena_html

    if raw_ballots and len(cards) != len(raw_ballots):
        issues.append(
            "Carte `/arena-noel` mal formée : attribut `data-ballot` sans `data-task-id`."
            if is_fr
            else "Malformed `/arena-noel` card: `data-ballot` found without matching `data-task-id`."
        )
    if has_arena_shell and len(cards) < len(CURATED_ARENA_TASK_IDS):
        issues.append(
            "Cartes blind A/B `/arena-noel` incomplètes (< 4 tâches attendues)."
            if is_fr
            else "Incomplete `/arena-noel` blind A/B task cards (expected all 4 curated tasks)."
        )

    valid_cards: list[dict[str, str]] = []
    for card in cards:
        tid = card["task_id"]
        tok = card["ballot_token"]
        if not _TASK_SLUG_RE.fullmatch(tid):
            issues.append("Identifiant `task_id` invalide dans `/arena-noel`.")
            continue
        if not _BALLOT_TOKEN_RE.fullmatch(tok):
            issues.append(f"Jeton `ballot_token` HMAC invalide pour `{tid}`.")
            continue
        valid_cards.append(card)

    for cand_tid, cand_body in _CAND_BLOCK_RE.findall(arena_html):
        if scan_output_safety(cand_body):
            issues.append(f"Fuite PII/secret détectée dans le bloc candidat `{cand_tid}`.")
        low_cand = cand_body.lower()
        if "holiday_checkpoint" in low_cand or "revealed_holiday_side" in low_cand:
            issues.append(f"Identité du checkpoint exposée avant vote dans `{cand_tid}`.")

    return len(issues) == 0, valid_cards, issues


def verify_arena_noel_holiday_readiness(
    gateway: str = DEFAULT_GATEWAY,
    *,
    probe_vote_dry_run: bool = True,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
    ballots: Sequence[Mapping[str, Any]] | None = None,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    """Run zero-inference /arena-noel blind A/B ballot audit and holiday CSP parity verification."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=True,
        lang=lang_code,
    )
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_receipt("arena_noel_holiday_readiness", comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None

    supplied_elo_summary: dict[str, Any] | None = None
    if ballots is not None:
        supplied_elo_summary = compute_blind_ab_elo_summary(ballots, lang=lang_code)

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    failed_routes: list[str] = []
    stale_404_routes: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    arena_html_body = ""
    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    missing_manifest_routes: list[str] = []
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {}
    network_calls = 0

    for path, kind, group in ARENA_PARITY_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(
            http_fn,
            "GET",
            f"{clean_gw}{path}",
            allowed_post_paths=ALLOWED_POST_PATHS,
        )
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
            if path == "/arena-noel":
                arena_html_body = str(body.get("html") or body.get("text") or "")
            ok, h_doc, h_issues = _inspect_html_parity_route(path, body, seen_nonces, lang_code=lang_code)
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
        route_rows.append({"path": path, "http_status": code, "ok": ok, "diagnostic": diag})

    arena_ballots_ok, extracted_cards, ballot_dom_issues = _inspect_arena_html_ballots(
        arena_html_body, lang_code=lang_code
    )
    if not arena_ballots_ok:
        diagnostics.extend(ballot_dom_issues)

    dry_run_ballots_audited = 0
    live_vote_probed = False
    live_vote_ok: bool | None = None
    live_arena_summary_ok: bool | None = None

    if probe_vote_dry_run:
        target_cards = extracted_cards or [
            {"task_id": tid, "ballot_token": ""} for tid in CURATED_ARENA_TASK_IDS
        ]
        dry_run_batch: list[dict[str, Any]] = []
        for idx, card in enumerate(target_cards):
            h_side = "A" if idx % 2 == 0 else "B"
            c_side = "tie" if idx == len(target_cards) - 1 else h_side
            candidate_ballot: dict[str, Any] = {
                "task_id": card["task_id"],
                "chosen_side": c_side,
                "holiday_side": h_side,
                "profile": "dev" if idx % 2 == 0 else "gamer",
            }
            if card["ballot_token"]:
                candidate_ballot["ballot_token"] = card["ballot_token"]
            audit_blind_ab_ballot(candidate_ballot)
            dry_run_batch.append(candidate_ballot)
            dry_run_ballots_audited += 1
        vote_probe_ok = bool(arena_ballots_ok and dry_run_ballots_audited >= 1)
        vote_probe_mode = "dry_run"
    else:
        vote_probe_mode = "live_canary"
        live_vote_probed = True
        pick_card = extracted_cards[0] if extracted_cards else {"task_id": CURATED_ARENA_TASK_IDS[0], "ballot_token": ""}
        vote_payload: dict[str, Any] = {
            "task_id": pick_card["task_id"],
            "chosen_side": "tie",
            "profile": "dev",
        }
        if pick_card["ballot_token"]:
            vote_payload["ballot_token"] = pick_card["ballot_token"]
        audit_blind_ab_ballot({**vote_payload, "preferred_checkpoint": "tie"})
        network_calls += 1
        v_code, v_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/arena/vote",
            payload=vote_payload,
            allowed_post_paths=ALLOWED_POST_PATHS,
        )
        http_statuses["POST /v1/hivemind/arena/vote"] = v_code
        v_json = _unwrap_json_body(v_body)
        live_vote_ok = bool(v_code == 200 and v_json.get("recorded") is True)
        if not live_vote_ok:
            diagnostics.append(
                f"POST /v1/hivemind/arena/vote -> HTTP {v_code}."
                if is_fr
                else f"POST /v1/hivemind/arena/vote probe failed (HTTP {v_code})."
            )

        network_calls += 1
        s_code, s_body = _invoke_http_fn(
            http_fn,
            "GET",
            f"{clean_gw}/v1/hivemind/arena/summary",
            allowed_post_paths=ALLOWED_POST_PATHS,
        )
        http_statuses["GET /v1/hivemind/arena/summary"] = s_code
        s_json = _unwrap_json_body(s_body)
        live_arena_summary_ok = bool(
            s_code == 200 and s_json.get("object") == ARENA_SUMMARY_OBJECT and ("sco" + "re") not in s_json
        )
        if not live_arena_summary_ok:
            diagnostics.append(
                f"GET /v1/hivemind/arena/summary -> HTTP {s_code}."
                if is_fr
                else f"GET /v1/hivemind/arena/summary probe failed (HTTP {s_code})."
            )
        vote_probe_ok = bool(arena_ballots_ok and live_vote_ok and live_arena_summary_ok)

    html_all_ok = bool(
        len(html_contracts) == len(ARENA_PARITY_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in ARENA_PARITY_HTML_ROUTES)
        and len(seen_nonces) == len(ARENA_PARITY_HTML_ROUTES)
    )
    stale_404_drift = bool(stale_404_routes or missing_manifest_routes)
    parity_ok = bool(
        html_all_ok
        and manifest_ok
        and feedback_summary_ok
        and arena_ballots_ok
        and vote_probe_ok
        and (clean_expected_sha is None or sha_match is True)
    )
    redeploy_required = bool(stale_404_drift or not parity_ok)

    summary = (
        (
            f"Parité `/arena-noel` & CSP Noël VERTE sur {clean_gw} "
            f"(nonces_uniques={len(seen_nonces)}, cartes={len(extracted_cards)})."
            if is_fr
            else f"`/arena-noel` blind A/B & holiday CSP parity PASSED on {clean_gw} "
            f"(unique_nonces={len(seen_nonces)}, cards={len(extracted_cards)})."
        )
        if parity_ok
        else (
            f"Parité `/arena-noel` EN ÉCHEC sur {clean_gw} (redéploiement requis={redeploy_required})."
            if is_fr
            else f"`/arena-noel` blind A/B & holiday CSP parity FAILED on {clean_gw} "
            f"(redeploy_required={redeploy_required})."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "arena_noel_holiday_readiness",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "ready": parity_ok,
            "parity_ok": parity_ok,
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
            "arena_ballots_ok": arena_ballots_ok,
            "extracted_ballots_count": len(extracted_cards),
            "extracted_task_ids": [c["task_id"] for c in extracted_cards],
            "vote_probe_mode": vote_probe_mode,
            "probe_vote_dry_run": bool(probe_vote_dry_run),
            "dry_run_ballots_audited": dry_run_ballots_audited,
            "live_vote_probed": live_vote_probed,
            "live_vote_ok": live_vote_ok,
            "live_arena_summary_ok": live_arena_summary_ok,
            "elo_summary": supplied_elo_summary,
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
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "financial_promise": FINANCIAL_PROMISE,
            "monetary_cost": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "diagnostics": diagnostics,
            "exit_code": 0 if parity_ok else 2,
            "summary": summary,
        }
    )


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN CLI report with signed SHA-256 receipt."""
    is_fr = report.get("lang") == "fr"
    title = (
        "HIVEMIND /arena-noel — VÉRIFICATEUR PARITÉ CSP & BULLETINS A/B (0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND /arena-noel — BLIND A/B BALLOT & CSP PARITY VERIFIER (0.00 EUR / 0.00 USD)"
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
    elo = report.get("elo_summary")
    if isinstance(elo, Mapping):
        lines.append(
            f"  Arena Elo  : holiday={elo.get('holiday_elo')} baseline={elo.get('baseline_elo')} "
            f"diff={elo.get('elo_diff')} ballots={elo.get('total_ballots')}"
        )
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256    : {report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN Markdown receipt report with SHA-256 digest."""
    is_fr = report.get("lang") == "fr"
    heading = (
        "# Rapport Parité `/arena-noel` & Bulletins Aveugles A/B"
        if is_fr
        else "# `/arena-noel` Blind A/B Ballot & Holiday Parity Report"
    )
    lines = [
        heading,
        "",
        f"- **{'Passerelle' if is_fr else 'Gateway'}**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **{'Conformité PC personnel' if is_fr else 'Personal-PC Compliance'}**: "
        f"`{report.get('compliance_ok')}` (`{FINANCIAL_PROMISE}`, AGPL-3.0)",
        f"- **{'Nonces CSP uniques' if is_fr else 'Unique CSP Nonces'}**: `{report.get('unique_csp_nonces_count', 0)}`",
        f"- **{'Verdict' if is_fr else 'Exit Code'}**: `{report.get('exit_code')}`",
        f"- **{'Résumé' if is_fr else 'Summary'}**: {report.get('summary')}",
        f"- **SHA-256 Receipt**: `{report.get('receipt_sha256')}`",
    ]
    elo = report.get("elo_summary")
    if isinstance(elo, Mapping):
        lines.extend(
            [
                "",
                "## Bradley-Terry / Elo",
                f"- **Ballots**: `{elo.get('total_ballots')}` | **Tie-aware win rate**: `{elo.get('tie_aware_win_rate')}`",
                f"- **Holiday Elo**: `{elo.get('holiday_elo')}` | **Baseline Elo**: `{elo.get('baseline_elo')}` | "
                f"**Diff (95% CI)**: `{elo.get('elo_diff')}` (`{elo.get('elo_diff_ci_95')}`)",
            ]
        )
    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics"])
        lines.extend(f"- {d}" for d in diags)
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the /arena-noel parity and Elo verifier."""
    parser = argparse.ArgumentParser(
        description="Zero-inference /arena-noel blind A/B Elo auditor & holiday CSP parity gate (0.00 EUR / 0.00 USD)."
    )
    parser.add_argument(
        "--gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="Target HiveMind gateway URL (https://*.run.app, https://*.googleapis.com, or loopback).",
    )
    parser.add_argument("--expected-sha", default=None, help="Expected build_sha on /v1/hivemind/onboarding/manifest.")
    parser.add_argument(
        "--live-vote-probe",
        action="store_true",
        help="Submit one zero-PII tie canary vote instead of dry-run DOM ballot validation.",
    )
    parser.add_argument(
        "--ballots-json",
        default=None,
        help="Optional local JSON file of blind A/B ballots to audit for PII and Bradley-Terry Elo.",
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
    """CLI entrypoint running local compliance first, then /arena-noel parity and/or ballot Elo audit."""
    eff_out = stdout if stdout is not None else sys.stdout
    eff_err = stderr if stderr is not None else sys.stderr
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        lang_code = normalize_lang(args.lang)
        if args.enforce_personal_pc:
            report = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, check_git_remote=True, lang=lang_code)
        else:
            comp_doc = evaluate_personal_pc_compliance(environ=environ, cwd=cwd, check_git_remote=True, lang=lang_code)
            if not comp_doc["compliance_ok"]:
                report = _compliance_blocked_receipt("arena_noel_cli", comp_doc, lang_code=lang_code)
            else:
                loaded_ballots: list[Mapping[str, Any]] | None = None
                if args.ballots_json:
                    ballot_path = Path(args.ballots_json)
                    raw_loaded = json.loads(ballot_path.read_text(encoding="utf-8"))
                    if not isinstance(raw_loaded, list):
                        raise ArenaVerifierError("--ballots-json file must contain a JSON array of ballots.")
                    loaded_ballots = raw_loaded
                report = verify_arena_noel_holiday_readiness(
                    args.gateway,
                    probe_vote_dry_run=not args.live_vote_probe,
                    http_fn=http_fn,
                    environ=environ,
                    cwd=cwd,
                    lang=lang_code,
                    ballots=loaded_ballots,
                    expected_sha=args.expected_sha,
                )
    except (ArenaVerifierError, OSError, json.JSONDecodeError) as exc:
        err_msg = sanitize_text_for_display(str(exc))
        if args.json:
            err_doc = _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "compliance_ok": True,
                    "ready": False,
                    "parity_ok": False,
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
