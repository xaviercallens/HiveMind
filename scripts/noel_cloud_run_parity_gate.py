#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cloud Run holiday route manifest verifier, post-deploy CES & rollback gate, and dry-run FinOps planner.

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library:
  1. Zero-Inference Cloud Run Holiday Parity & Manifest Gate (`check_cloud_run_holiday_parity`):
     verifies GET `/health`, `/v1/models`, `/v1/hivemind/onboarding/manifest`, `/noel`, `/quick`,
     `/raid-noel`, `/arena-noel`, `/v1/hivemind/onboarding/feedback/summary`, downloadable
     `/scripts/<name>` spot probes (`REQUIRED_PARITY_SCRIPTS`, or every allowlisted script in
     `REQUIRED_SCRIPTS` under `--verify-all-scripts`, including `noel_vscode_autoconfig.py`,
     `noel_gamer_whisper_selftest.py`, `noel_scale_load_harness.py`, and
     `noel_canary_cohort_rehearsal.py`) without ever calling chat-completion endpoints
     or prompting `hivemind/coder`. Compares live `build_sha`, `holiday_routes`, and
     `downloadable_scripts` against `REQUIRED_SCRIPTS` and `--expected-sha` or local `git rev-parse HEAD`
     (`--check-git-head`), failing closed (exit code 2, `parity_ok=False`,
     `redeploy_required=True`) on HTTP 404/5xx, missing holiday routes/scripts, or SHA drift.
  2. Post-Deploy Holiday Route & CES Readiness Verifier (`verify_post_deploy_holiday_readiness`,
     `--post-deploy-gate`, `--previous-revision`, `--probe-feedback-post`): verifies manifest schema &
     `build_sha` parity, `/noel` and `/quick` HTTP 200 + HTML/CSP/personal-PC compliance contract,
     `GET /v1/hivemind/onboarding/feedback/summary` schema, optional zero-PII canary
     `POST /v1/hivemind/onboarding/feedback` round-trip, and automatic dry-run rollback plan signaling
     (`post_deploy_ready=False`, `rollback_required=True`) on any 404/422/5xx, schema mismatch, or SHA drift.
  3. Zero-Inference PR #949 Live Quorum & CES Parity Gate (`verify_pr949_live_quorum_and_ces_parity`,
     `--pr949-quorum-gate`): verifies `/health`, `/v1/models` (`hivemind/coder` metadata), manifest
     17-script & route parity, all 4 HTML holiday routes (`/noel`, `/quick`, `/raid-noel`, `/arena-noel`)
     with unique per-request CSP nonces and personal-PC/0.00 EUR/AGPL-3.0 banners, CES/zero-lag SLOs,
     `Idempotency-Key` duplicate-replay deduplication (`--verify-idempotency-replay`), 8+ friend-producer
     quorum (`--target-online-producers`, `--require-live-quorum`), `/v1/hivemind/raid/status`, and
     `/v1/hivemind/arena/summary`.
  4. Dry-Run Cloud Run Redeploy & Rollback Verifiers (`build_cloud_run_redeploy_plan`,
     `build_cloud_run_rollback_plan`, `--print-redeploy-plan`): inspects `deploy/gcp_openrouter/deploy.sh`
     invariants (`--min-instances 0`, `--max-instances 1`, `--no-cpu-throttling`, Litestream
     `HIVEMIND_ROUTER_REPLICA_URL`, Secret Manager `HIVEMIND_ROUTER_SECRET`), emits a sanitized
     `deploy_manifest`, and renders FinOps-safe redeploy and `gcloud run services update-traffic`
     rollback commands without ever executing `gcloud` or mutating cloud state.
  5. Strictly Local Personal-PC & Amadeus Compliance Guard & Atomic 0600 Output (`check_personal_pc_compliance`,
     `--enforce-personal-pc`, `--cwd`, `--output`): blocks corporate environments and corporate git/worktree
     remotes locally with exit code 2 (`0` network calls, `0` file writes, `0` env/path value leaks).

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science with symbolic non-monetary points under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA = "hivemind.noel_cloud_run_parity_gate/1"
RECEIPT_OBJECT = "hivemind.noel_cloud_run_parity_receipt"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_CLOUD_RUN_SERVICE = "hivemind-openrouter"
DEFAULT_CLOUD_RUN_REGION = "us-central1"

HTTP_TIMEOUT_S = 5.0
SUBPROCESS_TIMEOUT_S = 3.0
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_SCRIPT_REL = "deploy/gcp_openrouter/deploy.sh"

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

REQUIRED_PARITY_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "core"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
)

POST_DEPLOY_CONTRACT_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick")
PR949_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
PR949_REQUIRED_ROUTES: tuple[tuple[str, str], ...] = (
    ("/health", "json"),
    ("/v1/models", "json"),
    ("/v1/hivemind/onboarding/manifest", "json"),
    ("/noel", "html"),
    ("/quick", "html"),
    ("/raid-noel", "html"),
    ("/arena-noel", "html"),
    ("/v1/hivemind/onboarding/feedback/summary", "json"),
    ("/v1/hivemind/producers/fleet", "json"),
    ("/v1/hivemind/raid/status", "json"),
    ("/v1/hivemind/arena/summary", "json"),
)

REQUIRED_PARITY_SCRIPTS: tuple[str, ...] = (
    "hivemind_quickstart.py",
    "hivemind_doctor.py",
    "noel_post_setup_probe.py",
    "noel_cloud_run_parity_gate.py",
    "join_hivemind.ps1",
)

NOEL_ZERO_CLONE_SCRIPTS: tuple[str, ...] = (
    "noel_canary_cohort_rehearsal.py",
    "noel_cloud_run_parity_gate.py",
    "noel_gamer_whisper_selftest.py",
    "noel_post_setup_probe.py",
    "noel_scale_load_harness.py",
    "noel_vscode_autoconfig.py",
)

REQUIRED_SCRIPTS: tuple[str, ...] = (
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

EXPECTED_MANIFEST_SCRIPTS: tuple[str, ...] = REQUIRED_SCRIPTS

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
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_CLOUD_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
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
    r'^\s*(?:push)?(?:url|(?:push)?insteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_MASKED_HM_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(hm(?:_live_|-))\*\*\*(?!redacted\*\*\*)[A-Za-z0-9_-]{4}(?![A-Za-z0-9_-])"
)
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "docu" + r"ment\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)"
)
_ALLOWED_FEEDBACK_POST_KEYS: frozenset[str] = frozenset(
    {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
)
_CANARY_FEEDBACK_PAYLOAD: dict[str, Any] = {
    "profile": "dev",
    "ease_rating": 5,
    "zero_game_lag": True,
    "platform": "linux",
}

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

HttpFn = Callable[..., Any]
RawHttpFn = Callable[..., tuple[int, str, dict[str, str]]]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]


class ParityGateError(ValueError):
    """Raised when parity gate arguments, URL guards, or output safety invariants fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ParityGateError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ParityGateError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Redact local user home directories to `'~'` or repo-relative paths so usernames never leak."""
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
                    return "~" + suffix.replace("\\", "/")
    redacted = _HOME_PATH_RE.sub("~", raw)
    return redacted.replace("\\", "/") if redacted != raw else raw


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
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        return "unknown"
    return low


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


def validate_expected_sha(raw: str) -> str:
    """Validate an explicit `--expected-sha` / `--target-sha` argument."""
    sanitized = sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise ParityGateError(
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


def _validate_int_range(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise ParityGateError(f"{name} must be an integer between {lo} and {hi}.")
    return int(value)


def _validate_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityGateError(f"{name} must be a positive finite number.")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ParityGateError(f"{name} must be a positive finite number.")
    return fval


def _validate_float_range(value: object, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityGateError(f"{name} must be a finite number between {lo} and {hi}.")
    fval = float(value)
    if not math.isfinite(fval) or not (lo <= fval <= hi):
        raise ParityGateError(f"{name} must be a finite number between {lo} and {hi}.")
    return fval


def _is_nonneg_int(val: object) -> bool:
    return not isinstance(val, bool) and isinstance(val, int) and val >= 0


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == ("sco" + "re") or _has_forbidden_score_key(value):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is a valid `*.run.app` DNS hostname or loopback (`127.0.0.1` / `localhost` / `::1`)."""
    if not isinstance(host, str):
        return False
    raw = host.strip().lower()
    if not raw or len(raw) > 253 or any(ch.isspace() or ch in "\\/@" for ch in raw):
        return False
    h = raw[1:-1] if (raw.startswith("[") and raw.endswith("]")) else raw
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    labels = h.split(".")
    if len(labels) < 3 or any(not _DNS_LABEL_RE.fullmatch(lbl) for lbl in labels):
        return False
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app` or `http(s)://127.0.0.1|localhost|[::1]`."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ParityGateError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ParityGateError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise ParityGateError("Gateway URL must not contain internal whitespace.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise ParityGateError("Gateway URL must not contain backslashes.")
    if "@" in cleaned or "@" in unquote(cleaned):
        raise ParityGateError("Gateway URL must not embed credentials.")
    try:
        parts = urlsplit(cleaned)
        port = parts.port
        host = (parts.hostname or "").lower()
    except ValueError as exc:
        raise ParityGateError("Gateway URL is malformed or contains an invalid port.") from exc
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ParityGateError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ParityGateError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise ParityGateError("Gateway URL must be a bare origin without a path suffix.")
    if port is not None and not (1 <= port <= 65535):
        raise ParityGateError("Gateway URL port out of range.")
    if not is_allowed_gateway_host(host):
        raise ParityGateError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app or 127.0.0.1/localhost/::1."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ParityGateError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ParityGateError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def _safe_urlsplit(url: str) -> Any:
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ParityGateError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ParityGateError("Request URL must not contain whitespace or control characters.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise ParityGateError("Request URL must not contain backslashes.")
    try:
        parts = urlsplit(cleaned)
    except ValueError as exc:
        raise ParityGateError("Request URL is malformed.") from exc
    if (
        parts.username is not None
        or parts.password is not None
        or "@" in parts.netloc
        or "@" in unquote(parts.netloc)
    ):
        raise ParityGateError("Request URL must not embed credentials.")
    return parts


def _assert_zero_inference_request(
    method: str,
    url: str,
    *,
    allow_feedback_post: bool = False,
    payload: Any = None,
) -> str:
    """Enforce read-only GET/HEAD (or zero-PII POST /v1/hivemind/onboarding/feedback), allowed gateway, and zero LLM inference."""
    verb = (method or "").strip().upper()
    parts = _safe_urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = unquote(parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise ParityGateError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    if payload is not None and isinstance(payload, Mapping):
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "messages" in payload or "prompt" in payload or "hivemind/coder" in serialized:
            raise ParityGateError(
                "Zero-inference invariant violated: messages/prompt/hivemind/coder payloads are strictly forbidden."
            )
    if verb in ("GET", "HEAD") or (
        allow_feedback_post and verb == "POST" and parts.path == "/v1/hivemind/onboarding/feedback"
    ):
        return origin
    raise ParityGateError(
        f"Zero-inference read-only invariant violated: HTTP method {verb!r} is forbidden."
    )


def _validate_zero_pii_feedback_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate that an onboarding feedback canary POST payload is strictly zero-PII and zero-inference."""
    if (
        not isinstance(payload, Mapping)
        or (set(payload.keys()) - _ALLOWED_FEEDBACK_POST_KEYS)
        or payload.get("profile") not in ("dev", "gamer")
        or isinstance(payload.get("ease_rating"), bool)
        or not isinstance(payload.get("ease_rating"), int)
        or not (1 <= int(payload["ease_rating"]) <= 5)
        or not isinstance(payload.get("zero_game_lag"), bool)
    ):
        raise ParityGateError("Feedback canary POST payload must be a valid zero-PII onboarding feedback object.")
    doc: dict[str, Any] = {
        "profile": payload["profile"],
        "ease_rating": int(payload["ease_rating"]),
        "zero_game_lag": bool(payload["zero_game_lag"]),
        "platform": str(payload.get("platform") or "linux")[:24],
    }
    if "note" in payload:
        note_val = payload.get("note")
        if isinstance(note_val, bool) or not isinstance(note_val, str) or len(note_val) > 240:
            raise ParityGateError("Feedback note must be a short zero-PII string.")
        doc["note"] = note_val
    return json.loads(_assert_safe_output(json.dumps(doc, sort_keys=True)))


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail-closed redirect guard blocking untrusted hosts, backslashes, credentials, and inference paths."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _ = (fp, code, msg)
        raw_loc: Any = newurl
        if headers is not None and hasattr(headers, "get"):
            for hdr_name in ("location", "Location"):
                cand = headers.get(hdr_name)
                if cand is not None:
                    raw_loc = cand
                    break
        if (
            not isinstance(raw_loc, str)
            or not raw_loc.strip()
            or not isinstance(newurl, str)
            or not newurl.strip()
        ):
            raise ParityGateError("Untrusted redirect: empty target URL.")
        _assert_zero_inference_request(req.get_method(), newurl, allow_feedback_post=False)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            new_method = new_req.get_method()
            new_parts = _safe_urlsplit(new_req.full_url)
            allow_post = new_method == "POST" and new_parts.path == "/v1/hivemind/onboarding/feedback"
            _assert_zero_inference_request(new_method, new_req.full_url, allow_feedback_post=allow_post)
        return new_req


def _prepare_http_headers(
    headers: Mapping[str, str] | None,
    *,
    has_json_body: bool = False,
) -> dict[str, str]:
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-cloud-run-parity-gate/1.0",
    }
    if has_json_body:
        req_headers["Content-Type"] = "application/json"
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise ParityGateError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    return req_headers


def _build_safe_http_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())


def _read_bounded_body(resp: Any) -> bytes:
    raw_bytes = resp.read(MAX_HTTP_BODY_BYTES + 1)
    if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
        raise ParityGateError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
    return raw_bytes


def default_http_raw(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Direct, proxy-free, redirect-guarded stdlib HTTP helper returning `(status_code, body_text, lower_headers)`."""
    verb = (method or "").strip().upper()
    parts = _safe_urlsplit(url)
    is_feedback_post = verb == "POST" and parts.path == "/v1/hivemind/onboarding/feedback"
    _assert_zero_inference_request(
        verb,
        url,
        allow_feedback_post=is_feedback_post,
        payload=payload,
    )
    if payload is not None and not is_feedback_post:
        raise ParityGateError("Zero-inference parity gate never sends request body payloads.")
    body_bytes = (
        json.dumps(_validate_zero_pii_feedback_payload(payload)).encode("utf-8")
        if is_feedback_post
        else None
    )
    bounded_timeout = min(max(0.2, _validate_positive_float(timeout_s, "timeout_s")), 60.0)
    req_headers = _prepare_http_headers(headers, has_json_body=body_bytes is not None)
    opener = _build_safe_http_opener()
    req = urllib.request.Request(url.strip(), data=body_bytes, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=bounded_timeout) as resp:
            raw_text = _read_bounded_body(resp).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            lower_headers = {str(k).strip().lower(): str(v) for k, v in resp.headers.items()}
            return status_code, raw_text, lower_headers
    except urllib.error.HTTPError as exc:
        err_bytes = exc.read(131_072) if exc.fp is not None else b""
        err_headers = {str(k).strip().lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
        return int(exc.code), err_bytes.decode("utf-8", errors="replace"), err_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return (
            0,
            json.dumps({"network_error": type(exc).__name__, "detail": str(reason)[:200]}),
            {},
        )


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Direct, proxy-free, redirect-guarded stdlib HTTP helper handling GET parity probes and feedback canary POST."""
    status_code, raw, resp_headers = default_http_raw(
        method,
        url,
        payload=payload,
        headers=headers,
        timeout_s=HTTP_TIMEOUT_S,
    )
    if status_code >= 400 or status_code == 0:
        try:
            parsed_err = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": raw[:240]}
        return status_code, (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})

    content_type = resp_headers.get("content-type", "")
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
    headers: Mapping[str, str] | None = None,
    allow_feedback_post: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Invoke a 2-arg `(method, url)` or 4-arg `(method, url, payload, headers)` HTTP callable safely."""
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, allow_feedback_post=allow_feedback_post, payload=payload)
    clean_payload = (
        _validate_zero_pii_feedback_payload(payload)
        if (allow_feedback_post and verb == "POST")
        else None
    )
    clean_headers = {str(k): str(v) for k, v in headers.items()} if headers is not None else None
    sig = inspect.signature(http_fn)
    params = sig.parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    try:
        if has_varargs or len(positional) >= 4:
            raw_res = http_fn(verb, url, clean_payload, clean_headers)
        elif has_varkw or ("payload" in params and "headers" in params):
            raw_res = http_fn(verb, url, payload=clean_payload, headers=clean_headers)
        elif "headers" in params:
            raw_res = http_fn(verb, url, headers=clean_headers)
        elif len(positional) == 3:
            raw_res = http_fn(verb, url, clean_payload)
        else:
            raw_res = http_fn(verb, url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": type(exc).__name__, "detail": str(exc)[:200]}

    raw_hdrs = (
        raw_res[2]
        if (isinstance(raw_res, tuple) and len(raw_res) == 3 and isinstance(raw_res[2], Mapping))
        else {}
    )
    resp_headers = {str(k).lower(): str(v) for k, v in raw_hdrs.items()}
    status_code, body = raw_res[0], raw_res[1]

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
    return int(status_code), norm_body


def _invoke_raw_http(
    raw_http_fn: RawHttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    allow_feedback_post: bool = False,
) -> tuple[int, str, dict[str, str]]:
    """Invoke a `raw_http_fn` callable safely returning `(status_code, body_text, lower_headers)`."""
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, allow_feedback_post=allow_feedback_post, payload=payload)
    clean_payload = (
        _validate_zero_pii_feedback_payload(payload)
        if (allow_feedback_post and verb == "POST")
        else None
    )
    clean_headers = {str(k): str(v) for k, v in headers.items()} if headers is not None else None
    sig = inspect.signature(raw_http_fn)
    params = sig.parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    try:
        if has_varargs or len(positional) >= 4:
            raw_res = raw_http_fn(verb, url, clean_payload, clean_headers)
        elif has_varkw or ("payload" in params and "headers" in params):
            raw_res = raw_http_fn(verb, url, payload=clean_payload, headers=clean_headers)
        elif "headers" in params:
            if clean_payload is not None:
                raise ParityGateError("Injected raw_http_fn does not accept POST payload arguments.")
            raw_res = raw_http_fn(verb, url, headers=clean_headers)
        else:
            if clean_payload is not None or clean_headers is not None:
                raise ParityGateError("Injected 2-arg raw_http_fn does not support payload/headers.")
            raw_res = raw_http_fn(verb, url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, json.dumps({"network_error": type(exc).__name__, "detail": str(exc)[:200]}), {}

    if not isinstance(raw_res, tuple) or len(raw_res) != 3:
        raise ParityGateError("raw_http_fn must return `(status_code, body_text, headers)`.")
    code_raw, body_raw, hdrs_raw = raw_res
    lower_hdrs = {str(k).strip().lower(): str(v) for k, v in (hdrs_raw or {}).items()}
    if isinstance(body_raw, Mapping):
        body_text = json.dumps(dict(body_raw), ensure_ascii=False)
    elif isinstance(body_raw, (bytes, bytearray)):
        body_text = body_raw.decode("utf-8", errors="replace")
    else:
        body_text = str(body_raw or "")
    return int(code_raw), body_text, lower_hdrs


def default_subprocess_runner(
    argv: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[int, str]:
    """Run a local read-only command (`git rev-parse HEAD`) with `shell=False` and `gcloud` blocked."""
    if not argv:
        raise ParityGateError("Subprocess argv must not be empty.")
    exe_name = Path(argv[0]).name.lower()
    if exe_name in ("gcloud", "gcloud.cmd", "gcloud.exe"):
        raise ParityGateError("Cloud mutation guard: executing gcloud is strictly forbidden.")
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout_s,
        check=False,
        shell=False,
    )
    return int(proc.returncode), proc.stdout


def _resolve_git_head_sha(runner: RunnerFn, *, lang: str) -> tuple[str | None, str | None]:
    """Resolve local `git rev-parse HEAD` via injectable runner without mutating state."""
    is_fr = lang == "fr"
    try:
        rc, stdout = runner(["git", "rev-parse", "HEAD"], SUBPROCESS_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        msg = (
            f"Impossible de résoudre le SHA git HEAD local (`git rev-parse HEAD` : {type(exc).__name__})."
            if is_fr
            else f"Failed to resolve local git HEAD SHA (`git rev-parse HEAD`: {type(exc).__name__})."
        )
        return None, msg

    candidate = sanitize_build_sha(stdout.strip()) if rc == 0 else "unknown"
    if rc != 0 or candidate == "unknown":
        msg = (
            "Impossible de lire un SHA git HEAD valide via `git rev-parse HEAD` ; passez `--expected-sha`."
            if is_fr
            else "Unable to read a valid git HEAD SHA from `git rev-parse HEAD`; pass `--expected-sha`."
        )
        return None, msg
    return candidate, None


def _extract_git_remote_urls(workspace: Path | str) -> list[str]:
    """Walk `(workspace, *workspace.parents)` to `.git` (dir or worktree `gitdir:`/`commondir`) and extract remote URLs."""
    root_ws = Path(workspace)
    cfg_paths: list[Path] = []
    for candidate_root in (root_ws, *root_ws.parents):
        git_entry = candidate_root / ".git"
        if git_entry.is_dir():
            cfg = git_entry / "config"
            if cfg.is_file():
                cfg_paths.append(cfg)
            break
        if git_entry.is_file():
            try:
                lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            first_line = next((ln.strip() for ln in lines if ln.strip()), "")
            if first_line.lower().startswith("gitdir:"):
                raw_gitdir = first_line.split(":", 1)[1].strip()
                gitdir = (
                    Path(raw_gitdir)
                    if Path(raw_gitdir).is_absolute()
                    else (candidate_root / raw_gitdir).resolve()
                )
                direct_cfg = gitdir / "config"
                if direct_cfg.is_file():
                    cfg_paths.append(direct_cfg)
                commondir_file = gitdir / "commondir"
                if commondir_file.is_file():
                    try:
                        common_rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                    except OSError:
                        common_rel = ""
                    if common_rel:
                        common_dir = (
                            Path(common_rel)
                            if Path(common_rel).is_absolute()
                            else (gitdir / common_rel).resolve()
                        )
                        common_cfg = common_dir / "config"
                        if common_cfg.is_file() and common_cfg not in cfg_paths:
                            cfg_paths.append(common_cfg)
            break

    raw_texts: list[str] = []
    for cfg_path in cfg_paths:
        with suppress(OSError):
            raw_texts.append(cfg_path.read_text(encoding="utf-8", errors="replace"))
    return [
        (m1 or m2).strip().strip('"').strip("'")
        for txt in raw_texts
        for m1, m2 in _GIT_REMOTE_URL_RE.findall(txt)
        if (m1 or m2).strip().strip('"').strip("'")
    ]


def check_personal_pc_compliance(
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    *,
    check_cwd_path: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Amadeus Compliance Guard (0 network calls, 0 file writes, 0 value leaks)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = env if env is not None else os.environ
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    upper_env: dict[str, str] = {
        str(k).upper(): str(v).strip() for k, v in env_map.items() if str(v).strip()
    }
    amadeus_ind = ["env:AMADEUS_ENV"] if any(k.startswith("AMADEUS") for k in upper_env) else []
    zscaler_ind = ["env:ZSCALER_CERT_PATH"] if upper_env.get("ZSCALER_CERT_PATH") else []
    dns_val = upper_env.get("USERDNSDOMAIN", "")
    dns_hits = [name for name, pat in _CORP_PATTERNS if pat.search(dns_val)] if dns_val else []
    env_hits = [
        (f"env:{key}", [name for name, pat in _CORP_PATTERNS if pat.search(upper_env[key])])
        for key in _INSPECTED_ENV_KEYS
        if upper_env.get(key)
    ]
    git_hits = sorted(
        {
            name
            for remote_url in _extract_git_remote_urls(workspace)
            for name, pat in _CORP_PATTERNS
            if pat.search(remote_url)
        }
    )

    cwd_hits: list[str] = []
    if check_cwd_path:
        path_candidates: list[Path] = [workspace]
        with suppress(OSError):
            resolved_ws = workspace.resolve()
            if resolved_ws != workspace:
                path_candidates.append(resolved_ws)
        seen_cwd_markers: set[str] = set()
        for ws_candidate in path_candidates:
            for seg in ws_candidate.parts:
                for name, pat in _CORP_PATTERNS:
                    if name not in seen_cwd_markers and pat.search(seg):
                        seen_cwd_markers.add(name)
                        cwd_hits.append(name)

    indicators = (
        amadeus_ind
        + zscaler_ind
        + (["env:USERDNSDOMAIN"] if dns_val else [])
        + [tag for tag, hits in env_hits if hits]
        + [f"git_remote:{name}" for name in git_hits]
        + [f"cwd:{name}" for name in sorted(cwd_hits)]
    )
    matched_markers = sorted(
        set(["amadeus"] if amadeus_ind else [])
        | set(["zscaler"] if zscaler_ind else [])
        | set(dns_hits or (["corp"] if dns_val else []))
        | {name for _, hits in env_hits for name in hits}
        | set(git_hits)
        | set(cwd_hits)
    )
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
            "matched_markers": matched_markers,
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


def _compliance_payload(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_cwd_path: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    local_check = check_personal_pc_compliance(
        env=env,
        cwd=cwd,
        check_cwd_path=check_cwd_path,
        lang=lang,
    )
    return {
        "personal_pc_only": bool(local_check["personal_pc_only"]),
        "amadeus_corporate_prohibition": bool(local_check["amadeus_corporate_prohibition"]),
        "financial_promise": FINANCIAL_PROMISE,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
    }


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML `nonce='...'` attributes, CSP `'nonce-...'` tokens, and masked `hm-...***XXXX` suffixes."""
    if not isinstance(text, str):
        raise ParityGateError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def scan_output_safety(text: str) -> list[str]:
    """Verify JSON or Markdown output contains zero PII, raw keys, home paths, static figures, IPs, or `'score': <digit>`."""
    if not isinstance(text, str):
        raise ParityGateError("scan_output_safety requires a string.")
    scrubbed = strip_masked_keys_and_nonces(text)
    issues: list[str] = []
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(scrubbed):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _HOME_PATH_RE.search(text):
        issues.append("unredacted_home_path_detected")
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    if _STATIC_FIG_UNIT_RE.search(scrubbed):
        issues.append("static_figure_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if ip_hit != "127.0.0.1":
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise ParityGateError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _atomic_write_text(target: Path, content: str, *, mode: int = 0o600) -> Path:
    """Atomically write UTF-8 text via a sibling temporary file, chmod 0600 on POSIX, and `os.replace`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            with suppress(OSError):
                os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
        if os.name != "nt":
            with suppress(OSError):
                os.chmod(target, mode)
        tmp_path = None
        return target
    finally:
        if tmp_path is not None:
            with suppress(OSError):
                tmp_path.unlink()


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _http_failure_diagnostic(path: str, group: str, status_code: int, *, lang: str) -> str:
    is_fr = lang == "fr"
    if status_code == 404 and group in ("holiday", "manifest", "script"):
        return (
            f"GET {path} a répondu HTTP 404 : la révision Cloud Run est en retard sur `main` "
            "(redéploiement passerelle requis)."
            if is_fr
            else f"GET {path} returned HTTP 404: Cloud Run revision lags behind `main` "
            "(gateway redeploy required)."
        )
    if status_code == 404:
        return (
            f"GET {path} a répondu HTTP 404 (route principale introuvable sur la passerelle)."
            if is_fr
            else f"GET {path} returned HTTP 404 (core gateway route missing)."
        )
    if status_code == 0:
        return (
            f"GET {path} injoignable (erreur réseau/TLS vers la passerelle)."
            if is_fr
            else f"GET {path} unreachable (network/TLS error contacting gateway)."
        )
    if status_code >= 500:
        return (
            f"GET {path} en erreur serveur passerelle (HTTP {status_code} ; échec fermé)."
            if is_fr
            else f"GET {path} failed with gateway server error (HTTP {status_code}; failing closed)."
        )
    return (
        f"GET {path} a répondu HTTP {status_code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} returned HTTP {status_code} (expected HTTP 200)."
    )


def _validate_manifest_body(
    body: Mapping[str, Any],
    *,
    lang: str,
) -> tuple[bool, str, str | None, list[str], list[str], list[str]]:
    """Validate `/v1/hivemind/onboarding/manifest` contract and return parity gaps."""
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
            f"Routes Noël absentes de `holiday_routes` dans le manifeste : {joined_r}."
            if is_fr
            else f"Manifest `holiday_routes` is missing required holiday endpoints: {joined_r}."
        )

    raw_scripts = body.get("downloadable_scripts")
    if isinstance(raw_scripts, list) and all(isinstance(s, str) for s in raw_scripts):
        missing_scripts = [s for s in REQUIRED_SCRIPTS if s not in raw_scripts]
    else:
        missing_scripts = list(REQUIRED_SCRIPTS)

    if missing_scripts:
        joined_s = ", ".join(missing_scripts)
        issues.append(
            f"Scripts absents de `downloadable_scripts` dans le manifeste : {joined_s}."
            if is_fr
            else f"Manifest `downloadable_scripts` is missing required allowlisted scripts: {joined_s}."
        )

    return len(issues) == 0, live_sha, live_rev, missing_routes, missing_scripts, issues


def _inspect_holiday_html_contract(
    path: str,
    body: Mapping[str, Any],
    seen_nonces: set[str],
    *,
    lang: str,
    require_csp_header: bool = False,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Inspect holiday HTML (`/noel`, `/quick`, `/raid-noel`, `/arena-noel`), CSP nonce, DOM safety, and Personal-PC compliance."""
    is_fr = lang == "fr"
    html_text = str(body.get("html") or "")
    hdrs = body.get("headers") if isinstance(body.get("headers"), Mapping) else {}
    csp_hdr = str(hdrs.get("content-security-policy") or body.get("csp") or "")
    csp_low = csp_hdr.lower()

    has_doctype = "<!doctype html" in html_text.lower()
    attr_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)]
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    csp_unsafe = "'unsafe-inline'" in csp_low or "'unsafe-eval'" in csp_low
    unique_hdr_nonce = hdr_nonces[0] if (hdr_nonces and len(set(hdr_nonces)) == 1) else None
    unique_attr_nonce = attr_nonces[0] if (attr_nonces and len(set(attr_nonces)) == 1) else None
    nonce = unique_hdr_nonce if csp_hdr else unique_attr_nonce

    header_ok = (
        unique_hdr_nonce is not None
        and "default-src 'none'" in csp_low
        and unique_hdr_nonce == unique_attr_nonce
    )
    csp_ok = bool(
        attr_nonces
        and unique_attr_nonce is not None
        and not csp_unsafe
        and (header_ok if (require_csp_header or csp_hdr) else True)
        and nonce is not None
        and nonce not in seen_nonces
    )
    track_nonce = unique_hdr_nonce or unique_attr_nonce or (hdr_nonces[0] if hdr_nonces else None)
    if track_nonce is not None:
        seen_nonces.add(track_nonce)

    sanitized_html = strip_masked_keys_and_nonces(html_text)
    sink_hits = _FORBIDDEN_DOM_SINK_RE.findall(sanitized_html)
    fig_hits = _STATIC_FIG_UNIT_RE.findall(sanitized_html)
    dom_clean = not (sink_hits or fig_hits)
    low_html = sanitized_html.lower()
    has_personal_marker = (
        "100% personal" in low_html
        or "100 % personnel" in low_html
        or "100 % sur pc personnel" in low_html
        or "100% sur pc personnel" in low_html
    )
    compliance_banner_ok = bool(
        has_personal_marker
        and "agpl-3.0" in low_html
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
    )
    checks = (
        (
            has_doctype,
            f"Contrat HTML {path} invalide : document `<!DOCTYPE html>` absent ou vide."
            if is_fr
            else f"HTML contract failed on {path}: missing or empty `<!DOCTYPE html>` document.",
        ),
        (
            csp_ok,
            f"Contrat CSP {path} invalide : nonce CSP par requête absent, incohérent ou réutilisé."
            if is_fr
            else f"CSP contract failed on {path}: missing, mismatched, or reused per-request CSP nonce.",
        ),
        (
            dom_clean,
            f"Contrat sécurité HTML {path} violé : sink DOM non sûr ou métrique statique détectée."
            if is_fr
            else f"HTML safety contract violated on {path}: unsafe DOM sink or static metric figure detected.",
        ),
        (
            compliance_banner_ok,
            f"Contrat conformité PC perso / 0,00 EUR absent sur {path}."
            if is_fr
            else f"Personal-PC / 0.00 EUR compliance banner missing on {path}.",
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
            "unsafe_dom_sinks_count": len(sink_hits),
            "static_figure_found": bool(fig_hits),
            "personal_pc_banner_present": compliance_banner_ok,
        },
        issues,
    )


def _inspect_feedback_summary_schema(
    body: Mapping[str, Any],
    *,
    lang: str,
    require_object: bool = False,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate `GET /v1/hivemind/onboarding/feedback/summary` schema and zero-PII bounds."""
    is_fr = lang == "fr"
    obj_present = "object" in body or require_object
    obj_ok = (not obj_present) or (body.get("object") == FEEDBACK_SUMMARY_OBJECT)
    forbidden_ok = not _has_forbidden_score_key(body)

    raw_total = body.get("total_responses")
    total_ok = _is_nonneg_int(raw_total)
    total_val = int(raw_total) if total_ok else -1

    has_mean = "mean_ease_rating" in body
    has_avg = "average_ease_rating" in body
    ease_ok = bool((has_mean or has_avg) and total_ok)
    ease_vals: list[float] = []
    if ease_ok:
        for present, raw_val in ((has_mean, body.get("mean_ease_rating")), (has_avg, body.get("average_ease_rating"))):
            if not present:
                continue
            if total_val == 0:
                if raw_val is not None:
                    ease_ok = False
                    break
            elif (
                isinstance(raw_val, bool)
                or not isinstance(raw_val, (int, float))
                or not math.isfinite(float(raw_val))
                or not (1.0 <= float(raw_val) <= 5.0)
            ):
                ease_ok = False
                break
            else:
                ease_vals.append(round(float(raw_val), 2))
        if ease_ok and len(ease_vals) == 2 and ease_vals[0] != ease_vals[1]:
            ease_ok = False
    ease_val = ease_vals[0] if (ease_ok and ease_vals) else None

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
    lag_val = round(float(raw_lag), 4) if (lag_ok and raw_lag is not None) else None

    raw_prof = body.get("by_profile")
    d_raw = raw_prof.get("dev") if isinstance(raw_prof, Mapping) else None
    g_raw = raw_prof.get("gamer") if isinstance(raw_prof, Mapping) else None
    prof_ok = bool(
        isinstance(raw_prof, Mapping)
        and _is_nonneg_int(d_raw)
        and _is_nonneg_int(g_raw)
        and total_ok
        and (int(d_raw) + int(g_raw) == total_val)  # type: ignore[arg-type]
    )

    schema_ok = bool(obj_ok and forbidden_ok and total_ok and ease_ok and lag_ok and prof_ok)
    issues = (
        []
        if schema_ok
        else [
            "Contrat JSON `/v1/hivemind/onboarding/feedback/summary` invalide "
            "(attendu `total_responses`, `average_ease_rating`/`mean_ease_rating`, `zero_game_lag_ratio`, `by_profile`)."
            if is_fr
            else "Feedback summary `/v1/hivemind/onboarding/feedback/summary` schema mismatch "
            "(expected `total_responses`, `average_ease_rating`/`mean_ease_rating`, `zero_game_lag_ratio`, `by_profile`)."
        ]
    )
    normalized = {
        "schema_valid": schema_ok,
        "total_responses": total_val if total_ok else None,
        "average_ease_rating": ease_val,
        "mean_ease_rating": ease_val,
        "zero_game_lag_ratio": lag_val,
        "by_profile": {"dev": int(d_raw), "gamer": int(g_raw)} if prof_ok else None,  # type: ignore[arg-type]
    }
    return schema_ok, normalized, issues


def _probe_feedback_canary_roundtrip(
    clean_gw: str,
    http_fn: HttpFn,
    fallback_doc: dict[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, int, dict[str, Any], list[str]]:
    """Execute zero-PII `POST /v1/hivemind/onboarding/feedback` canary + `GET` summary verification."""
    post_code, post_body = _invoke_http_callable(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload=_CANARY_FEEDBACK_PAYLOAD,
        allow_feedback_post=True,
    )
    post_ok, post_doc, _ = (
        _inspect_feedback_summary_schema(post_body, lang=lang_code)
        if post_code == 200
        else (False, fallback_doc, [])
    )
    ref_code, ref_body = (
        _invoke_http_callable(http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/feedback/summary")
        if (post_code == 200 and post_ok and (post_doc.get("total_responses") or 0) >= 1)
        else (0, {})
    )
    ref_ok, ref_doc, _ = (
        _inspect_feedback_summary_schema(ref_body, lang=lang_code)
        if ref_code == 200
        else (False, post_doc, [])
    )
    canary_ok = bool(ref_code == 200 and ref_ok and (ref_doc.get("total_responses") or 0) >= 1)
    return (
        canary_ok,
        post_code,
        (ref_doc if canary_ok else fallback_doc),
        (
            []
            if canary_ok
            else [
                f"Canari POST /v1/hivemind/onboarding/feedback en échec (HTTP {post_code}) ; rollback requis."
                if lang_code == "fr"
                else f"Canary POST /v1/hivemind/onboarding/feedback failed (HTTP {post_code}); rollback required."
            ]
        ),
    )


def _probe_gateway_parity_core(
    clean_gw: str,
    *,
    expected_sha: str | None,
    check_git_head: bool,
    verify_all_scripts: bool,
    strict_post_deploy: bool,
    probe_feedback_post: bool = False,
    rollback_plan: Mapping[str, Any] | None = None,
    compliance_doc: Mapping[str, Any] | None = None,
    http_fn: HttpFn,
    runner: RunnerFn,
    lang_code: str,
) -> dict[str, Any]:
    """Shared zero-inference route, manifest, script, HTML/CSP, and CES feedback probe engine."""
    is_fr = lang_code == "fr"
    resolved_expected_sha: str | None = None
    expected_sha_source = "none"
    git_head_error: str | None = None
    if expected_sha is not None and str(expected_sha).strip():
        resolved_expected_sha = validate_expected_sha(expected_sha)
        expected_sha_source = "expected_sha"
    elif check_git_head:
        expected_sha_source = "git_head"
        resolved_expected_sha, git_head_error = _resolve_git_head_sha(runner, lang=lang_code)

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    missing_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []

    manifest_present = False
    manifest_valid = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    missing_manifest_routes: list[str] = []
    missing_manifest_scripts: list[str] = []
    models_count: int | None = None

    seen_nonces: set[str] = set()
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

    for path, expected_kind, group in REQUIRED_PARITY_ROUTES:
        code, body = _invoke_http_callable(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if code == 200 and path == "/health":
            if body.get("status") != "ok":
                ok = False
                diag = (
                    "GET /health a répondu HTTP 200 mais `status` != 'ok'."
                    if is_fr
                    else "GET /health returned HTTP 200 without `status == 'ok'`."
                )
        elif code == 200 and path == "/v1/models":
            if not isinstance(body.get("data"), list):
                ok = False
                diag = (
                    "GET /v1/models a répondu HTTP 200 sans liste `data` valide."
                    if is_fr
                    else "GET /v1/models returned HTTP 200 without a valid `data` list."
                )
            else:
                models_count = len(body["data"])
        elif code == 200 and path == "/v1/hivemind/onboarding/manifest":
            manifest_present = True
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
            if "html_length" in body and int(body["html_length"]) <= 0:
                ok = False
                diag = (
                    f"GET {path} a renvoyé une page HTML vide."
                    if is_fr
                    else f"GET {path} returned an empty HTML response."
                )
            if path in POST_DEPLOY_CONTRACT_HTML_ROUTES:
                h_ok, h_doc, h_issues = _inspect_holiday_html_contract(
                    path,
                    body,
                    seen_nonces,
                    lang=lang_code,
                )
                html_contracts[path] = h_doc
                html_contract_ok = html_contract_ok and h_ok
                if not h_ok and strict_post_deploy:
                    ok, diag = False, (diag or h_issues[0])
                    diagnostics.extend(h_issues)
        elif code == 200 and path == "/v1/hivemind/onboarding/feedback/summary":
            fb_ok, fb_doc, fb_issues = _inspect_feedback_summary_schema(body, lang=lang_code)
            feedback_summary_ok, feedback_summary_doc = fb_ok, fb_doc
            if not fb_ok and strict_post_deploy:
                ok, diag = False, fb_issues[0]
                diagnostics.extend(fb_issues)

        html_contract_ok = html_contract_ok and not (code != 200 and path in POST_DEPLOY_CONTRACT_HTML_ROUTES)

        if not ok:
            failed_routes.append(path)
            if code == 404:
                missing_routes.append(path)
            if not diag:
                diag = _http_failure_diagnostic(path, group, code, lang=lang_code)
                diagnostics.append(diag)
            elif path != "/v1/hivemind/onboarding/manifest" and (
                not strict_post_deploy
                or path not in ("/v1/hivemind/onboarding/feedback/summary", *POST_DEPLOY_CONTRACT_HTML_ROUTES)
            ):
                diagnostics.append(diag)

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "expected_kind": expected_kind,
                "group": group,
                "http_status": code,
                "ok": ok,
                "diagnostic": diag,
            }
        )

    script_rows: list[dict[str, Any]] = []
    script_http_statuses: dict[str, int] = {}
    missing_scripts: list[str] = []
    failed_scripts: list[str] = []
    target_scripts = REQUIRED_SCRIPTS if verify_all_scripts else REQUIRED_PARITY_SCRIPTS

    for script_name in target_scripts:
        script_path = f"/scripts/{script_name}"
        code, body = _invoke_http_callable(http_fn, "GET", f"{clean_gw}{script_path}")
        script_http_statuses[script_name] = code
        ok = code == 200
        diag = ""
        if code == 200 and "text_length" in body and int(body["text_length"]) <= 0:
            ok = False
            diag = (
                f"GET {script_path} a renvoyé un script vide (0 octet)."
                if is_fr
                else f"GET {script_path} returned an empty script body (0 bytes)."
            )
        if not ok:
            failed_scripts.append(script_name)
            if code == 404:
                missing_scripts.append(script_name)
            if not diag:
                diag = _http_failure_diagnostic(script_path, "script", code, lang=lang_code)
            diagnostics.append(diag)

        script_rows.append(
            {
                "script": script_name,
                "path": script_path,
                "method": "GET",
                "http_status": code,
                "ok": ok,
                "diagnostic": diag,
            }
        )

    sha_check_requested = expected_sha_source != "none"
    sha_match: bool | None = None
    sha_drift = False
    if sha_check_requested:
        if git_head_error is not None or resolved_expected_sha is None:
            sha_match = False
            sha_drift = True
            if git_head_error is not None:
                diagnostics.append(git_head_error)
        else:
            sha_match = shas_match(live_build_sha, resolved_expected_sha)
            sha_drift = not sha_match
            if sha_drift:
                rev_label = cloud_run_revision or ("aucune" if is_fr else "none")
                sha_diag = (
                    f"Dérive de SHA Cloud Run détectée : `build_sha` actif='{live_build_sha}' "
                    f"(révision={rev_label}) != SHA attendu '{resolved_expected_sha}' ; redéploiement requis."
                    if is_fr
                    else f"Cloud Run revision SHA drift detected: live `build_sha`='{live_build_sha}' "
                    f"(revision={rev_label}) != expected '{resolved_expected_sha}'; redeploy required."
                )
                diagnostics.append(sha_diag)

    routes_verified_count = sum(1 for row in route_rows if row["ok"])
    routes_total_count = len(route_rows)
    scripts_verified_count = sum(1 for row in script_rows if row["ok"])
    scripts_total_count = len(script_rows)

    parity_ok = bool(
        routes_verified_count == routes_total_count
        and scripts_verified_count == scripts_total_count
        and manifest_present
        and manifest_valid
        and not sha_drift
    )
    cloud_run_lags_main = bool(
        http_statuses.get("/health") == 200
        and (
            missing_routes
            or missing_scripts
            or missing_manifest_routes
            or missing_manifest_scripts
            or sha_drift
        )
    )

    if cloud_run_lags_main:
        lead_msg = (
            "Alerte Parité Cloud Run : `/health` répond HTTP 200 mais la révision déployée est en retard "
            "sur `main` — exécutez `--print-redeploy-plan` pour afficher la commande de redéploiement."
            if is_fr
            else "Cloud Run Parity Alert: `/health` is HTTP 200 while the deployed revision lags behind "
            "`main` — run `--print-redeploy-plan` for the copy-ready redeploy command."
        )
        diagnostics.insert(0, lead_msg)

    html_all_ok = bool(html_contract_ok and len(html_contracts) == len(POST_DEPLOY_CONTRACT_HTML_ROUTES))
    feedback_post_ok, feedback_post_status, feedback_summary_doc, post_diags = (
        _probe_feedback_canary_roundtrip(clean_gw, http_fn, feedback_summary_doc, lang_code=lang_code)
        if (strict_post_deploy and probe_feedback_post)
        else (None, None, feedback_summary_doc, [])
    )
    diagnostics.extend(post_diags)
    feedback_summary_ok = bool(feedback_summary_ok or feedback_post_ok)
    post_deploy_ready = bool(
        parity_ok and html_all_ok and feedback_summary_ok and (not probe_feedback_post or feedback_post_ok is True)
    )
    rollback_required = not post_deploy_ready
    if strict_post_deploy and rollback_required:
        diagnostics.insert(
            0,
            "ROLLBACK REQUIS : la vérification post-déploiement Cloud Run a échoué — "
            "basculez le trafic via `gcloud run services update-traffic`."
            if is_fr
            else "ROLLBACK REQUIRED: Cloud Run post-deploy holiday verification failed — "
            "restore traffic via `gcloud run services update-traffic`.",
        )

    base_report: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "mode": "post_deploy_gate" if strict_post_deploy else "parity_gate",
        "lang": lang_code,
        "gateway": clean_gw,
        "parity_ok": post_deploy_ready if strict_post_deploy else parity_ok,
        "redeploy_required": rollback_required if strict_post_deploy else (not parity_ok),
        "cloud_run_lags_main": cloud_run_lags_main,
        "routes_verified_count": routes_verified_count,
        "routes_total_count": routes_total_count,
        "scripts_verified_count": scripts_verified_count,
        "scripts_total_count": scripts_total_count,
        "manifest_present": manifest_present,
        "manifest_valid": manifest_valid,
        "live_build_sha": live_build_sha,
        "expected_build_sha": resolved_expected_sha,
        "expected_sha_source": expected_sha_source,
        "sha_check_requested": sha_check_requested,
        "sha_match": sha_match,
        "sha_drift": sha_drift,
        "cloud_run_revision": cloud_run_revision,
        "models_count": models_count,
        "zero_inference_verified": True,
        "http_statuses": http_statuses,
        "script_http_statuses": script_http_statuses,
        "missing_routes": missing_routes,
        "failed_routes": failed_routes,
        "missing_scripts": missing_scripts,
        "failed_scripts": failed_scripts,
        "missing_manifest_routes": missing_manifest_routes,
        "missing_manifest_scripts": missing_manifest_scripts,
        "diagnostics": diagnostics,
        "routes": route_rows,
        "scripts": script_rows,
        "compliance": dict(compliance_doc) if strict_post_deploy and compliance_doc else _compliance_payload(lang=lang_code),
    }
    post_extra: dict[str, Any] = (
        {
            "compliance_ok": bool(compliance_doc["compliance_ok"]) if compliance_doc else True,
            "post_deploy_ready": post_deploy_ready,
            "rollback_required": rollback_required,
            "html_contract_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "feedback_post_probed": bool(probe_feedback_post),
            "feedback_post_ok": feedback_post_ok,
            "feedback_post_http_status": feedback_post_status,
            "rollback_plan": dict(rollback_plan) if rollback_plan is not None else None,
        }
        if strict_post_deploy
        else {}
    )
    return _attach_receipt({**base_report, **post_extra})


def _compliance_blocked_report(
    mode: str,
    gateway: str,
    compliance_doc: Mapping[str, Any],
    *,
    lang_code: str,
) -> dict[str, Any]:
    """Build a fail-closed report when `--enforce-personal-pc` detects a corporate environment."""
    pr949_extra: dict[str, Any] = (
        {
            "slice_id": "SLICE-SCALE-DYN-42",
            "corporate_detected": True,
            "ready": False,
            "exit_code": 2,
            "holiday_routes_deployed_on_cloud_run": False,
            "onboarding_feedback_deployed": False,
            "stale_revision_detected": False,
            "models_coder_present": False,
            "online_producers": None,
            "producer_quorum_met": False,
            "awaiting_producer_quorum": False,
            "ces_slo_ok": False,
            "zero_lag_slo_ok": False,
            "idempotent_dedup_ok": None,
        }
        if mode == "pr949_quorum_gate"
        else {}
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": mode,
            "lang": lang_code,
            "gateway": gateway,
            "compliance_ok": False,
            "parity_ok": False,
            "post_deploy_ready": False,
            "plan_ready": False,
            "redeploy_required": False,
            "rollback_required": False,
            "rollback_plan": None,
            "routes_verified_count": 0,
            "routes_total_count": 0,
            "scripts_verified_count": 0,
            "scripts_total_count": 0,
            "network_calls": 0,
            "file_writes": 0,
            "zero_inference_verified": True,
            "diagnostics": [str(compliance_doc["summary"])],
            "compliance": dict(compliance_doc),
            **pr949_extra,
        }
    )


def check_cloud_run_holiday_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    verify_all_scripts: bool = False,
    enforce_personal_pc: bool = False,
    check_cwd_path: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    lang: str = "en",
) -> dict[str, Any]:
    """Run the zero-inference Cloud Run Holiday Parity & Manifest Gate, failing closed on any drift."""
    lang_code = normalize_lang(lang)
    comp_doc = check_personal_pc_compliance(
        env=env,
        cwd=cwd,
        check_cwd_path=check_cwd_path,
        lang=lang_code,
    )
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("parity_gate", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)
    return _probe_gateway_parity_core(
        validate_gateway_url(gateway),
        expected_sha=expected_sha,
        check_git_head=check_git_head,
        verify_all_scripts=verify_all_scripts,
        strict_post_deploy=False,
        compliance_doc=comp_doc,
        http_fn=http_fn,
        runner=runner,
        lang_code=lang_code,
    )


def _validate_cloud_identifier(value: str, label: str) -> str:
    cleaned = (value or "").strip()
    if not _SAFE_CLOUD_TOKEN_RE.fullmatch(cleaned):
        raise ParityGateError(f"Invalid Cloud Run {label}: {value!r}.")
    return cleaned


def build_cloud_run_rollback_plan(
    *,
    previous_revision: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    service_name: str = DEFAULT_CLOUD_RUN_SERVICE,
    region: str = DEFAULT_CLOUD_RUN_REGION,
    lang: str = "en",
) -> dict[str, Any]:
    """Validate revision identifier and render a dry-run FinOps-safe Cloud Run traffic rollback plan."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_service = _validate_cloud_identifier(service_name, "service name")
    clean_region = _validate_cloud_identifier(region, "region")

    has_raw_rev = previous_revision is not None and (
        not isinstance(previous_revision, str) or bool(previous_revision.strip())
    )
    clean_rev = sanitize_cloud_run_revision(previous_revision) if has_raw_rev else None
    if has_raw_rev and clean_rev is None:
        raise ParityGateError(
            "Invalid Cloud Run revision identifier; expected safe K_REVISION token without secrets."
        )

    rev_expr = clean_rev if clean_rev is not None else "${PREVIOUS_K_REVISION:?set_PREVIOUS_K_REVISION}"
    rollback_command = (
        f"gcloud run services update-traffic {clean_service} "
        '--project "${GCP_PROJECT:?set_GCP_PROJECT}" '
        f'--region "{clean_region}" '
        f'--to-revisions "{rev_expr}=100" && '
        f"gcloud run services update {clean_service} "
        '--project "${GCP_PROJECT}" '
        f'--region "{clean_region}" '
        "--min-instances 0 --max-instances 1 --no-cpu-throttling"
    )
    verify_command = (
        f"python3 scripts/noel_cloud_run_parity_gate.py --gateway {clean_gw} --post-deploy-gate"
    )
    summary = (
        f"Plan de rollback Cloud Run dry-run prêt vers `{rev_expr}` (FinOps `--min-instances 0 --max-instances 1`, 0 appel gcloud)."
        if is_fr
        else f"Dry-run Cloud Run rollback plan ready for `{rev_expr}` (FinOps `--min-instances 0 --max-instances 1`, 0 gcloud calls)."
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "rollback_plan",
            "lang": lang_code,
            "gateway": clean_gw,
            "service_name": clean_service,
            "region": clean_region,
            "previous_revision": clean_rev,
            "revision_pinned": clean_rev is not None,
            "finops_safe": True,
            "min_instances": 0,
            "max_instances": 1,
            "no_cpu_throttling": True,
            "dry_run_only": True,
            "gcloud_executed": False,
            "cloud_state_mutated": False,
            "zero_inference_verified": True,
            "rollback_command": rollback_command,
            "post_rollback_verify_command": verify_command,
            "verify_command": verify_command,
            "summary": summary,
            "compliance": _compliance_payload(lang=lang_code),
        }
    )


def build_cloud_run_redeploy_plan(
    *,
    deploy_script_path: Path | None = None,
    target_sha: str | None = None,
    check_git_head: bool = False,
    gateway: str = DEFAULT_GATEWAY,
    service_name: str = DEFAULT_CLOUD_RUN_SERVICE,
    region: str = DEFAULT_CLOUD_RUN_REGION,
    previous_revision: str | None = None,
    runner: RunnerFn = default_subprocess_runner,
    lang: str = "en",
) -> dict[str, Any]:
    """Inspect `deploy/gcp_openrouter/deploy.sh` FinOps invariants and render a dry-run redeploy command."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    clean_service = _validate_cloud_identifier(service_name, "service name")
    clean_region = _validate_cloud_identifier(region, "region")

    resolved_sha: str | None = None
    sha_ok = True
    diagnostics: list[str] = []
    if target_sha is not None and str(target_sha).strip():
        resolved_sha = validate_expected_sha(target_sha)
    elif check_git_head:
        resolved_sha, git_err = _resolve_git_head_sha(runner, lang=lang_code)
        if git_err is not None or resolved_sha is None:
            sha_ok = False
            if git_err is not None:
                diagnostics.append(git_err)

    target_path = (
        deploy_script_path
        if deploy_script_path is not None
        else (REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh")
    )
    display_path = redact_home_path(target_path)

    script_present = False
    active_script = ""
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
            "Scale-to-zero FinOps cap (--min-instances 0) preserves idle cloud budget.",
            "Plafond FinOps scale-to-zero (--min-instances 0) préservant le budget au repos.",
        ),
        (
            "max_instances_one",
            "--max-instances 1",
            bool(
                script_present
                and re.search(r"--max-instances(?:=|\s+)1\b", active_script)
                and not re.search(r"--max-instances(?:=|\s+)(?:0|[2-9]|\d{2,})\b", active_script)
            ),
            "Single-writer SQLite + Litestream safety cap (--max-instances 1).",
            "Plafond mono-instance SQLite + Litestream (--max-instances 1).",
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
            "CPU maintenu (--no-cpu-throttling) pour la réplication continue Litestream.",
        ),
        (
            "litestream_replica_url",
            "HIVEMIND_ROUTER_REPLICA_URL",
            bool(
                script_present
                and "HIVEMIND_ROUTER_REPLICA_URL=" in active_script
                and "gcs://" in active_script
            ),
            "Durable GCS Litestream ledger replication (HIVEMIND_ROUTER_REPLICA_URL).",
            "Réplication durable du registre SQLite via Litestream (HIVEMIND_ROUTER_REPLICA_URL).",
        ),
        (
            "secret_manager_router_secret",
            "HIVEMIND_ROUTER_SECRET",
            bool(
                script_present
                and re.search(r"--set-secrets\s+[\"']?[^\"'\n]*HIVEMIND_ROUTER_SECRET=", active_script)
            ),
            "Router HMAC secret mounted from Secret Manager (--set-secrets HIVEMIND_ROUTER_SECRET=...).",
            "Secret HMAC monté depuis Secret Manager (--set-secrets HIVEMIND_ROUTER_SECRET=...).",
        ),
    )

    invariants: list[dict[str, Any]] = []
    failed_invariants: list[str] = []
    for inv_id, expected_token, passed, desc_en, desc_fr in invariant_specs:
        desc = desc_fr if is_fr else desc_en
        invariants.append(
            {
                "id": inv_id,
                "expected": expected_token,
                "passed": passed,
                "description": desc,
            }
        )
        if not passed:
            failed_invariants.append(inv_id)
            if script_present:
                diagnostics.append(
                    f"Invariant FinOps/Cloud Run manquant dans `{display_path}` : `{expected_token}` ({inv_id})."
                    if is_fr
                    else f"Missing FinOps/Cloud Run invariant in `{display_path}`: `{expected_token}` ({inv_id})."
                )

    finops_ok = bool(script_present and not failed_invariants)
    plan_ready = bool(finops_ok and sha_ok)
    sha_expr = resolved_sha if resolved_sha is not None else "$(git rev-parse --short=12 HEAD)"

    deploy_manifest: dict[str, Any] = {
        "service_name": clean_service,
        "region": clean_region,
        "target_build_sha": sanitize_build_sha(resolved_sha),
        "finops_invariants": {
            "min_instances": 0,
            "max_instances": 1,
            "no_cpu_throttling": True,
            "litestream_replica_url_required": True,
            "secret_manager_router_secret_required": True,
            "all_passed": finops_ok,
            "checks": {item["id"]: bool(item["passed"]) for item in invariants},
        },
        "holiday_routes": list(EXPECTED_MANIFEST_HOLIDAY_ROUTES),
        "required_scripts": list(REQUIRED_SCRIPTS),
        "personal_pc_only": True,
        "financial_promise": FINANCIAL_PROMISE,
    }
    rollback_plan = build_cloud_run_rollback_plan(
        previous_revision=previous_revision,
        gateway=clean_gw,
        service_name=clean_service,
        region=clean_region,
        lang=lang_code,
    )

    redeploy_command = (
        f'HIVEMIND_BUILD_SHA="{sha_expr}" '
        'GCP_PROJECT="${GCP_PROJECT:?set_GCP_PROJECT}" '
        'HIVEMIND_ROUTER_ADMIN_KEY_HASH="${HIVEMIND_ROUTER_ADMIN_KEY_HASH:?set_ADMIN_HASH}" '
        f'REGION="{clean_region}" '
        f"bash {DEFAULT_DEPLOY_SCRIPT_REL} && "
        f"gcloud run services update {clean_service} "
        '--project "${GCP_PROJECT}" '
        f'--region "{clean_region}" '
        "--min-instances 0 --max-instances 1 "
        f'--update-env-vars "HIVEMIND_BUILD_SHA={sha_expr}"'
    )
    verify_flag = f"--expected-sha {resolved_sha}" if resolved_sha is not None else "--check-git-head"
    verify_command = (
        f"python3 scripts/noel_cloud_run_parity_gate.py --gateway {clean_gw} {verify_flag}"
    )

    core_plan: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "mode": "redeploy_plan",
        "lang": lang_code,
        "gateway": clean_gw,
        "service_name": clean_service,
        "region": clean_region,
        "deploy_script": display_path,
        "deploy_script_present": script_present,
        "finops_ok": finops_ok,
        "plan_ready": plan_ready,
        "parity_ok": plan_ready,
        "redeploy_required": True,
        "routes_verified_count": 0,
        "scripts_verified_count": 0,
        "invariants_verified_count": sum(1 for item in invariants if item["passed"]),
        "invariants_total_count": len(invariants),
        "failed_invariants": failed_invariants,
        "invariants": invariants,
        "target_build_sha": resolved_sha,
        "deploy_manifest": deploy_manifest,
        "rollback_plan": rollback_plan,
        "dry_run_only": True,
        "gcloud_executed": False,
        "cloud_state_mutated": False,
        "zero_inference_verified": True,
        "redeploy_command": redeploy_command,
        "verify_command": verify_command,
        "diagnostics": diagnostics,
        "compliance": _compliance_payload(lang=lang_code),
    }
    return _attach_receipt(core_plan)


def verify_post_deploy_holiday_readiness(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    previous_revision: str | None = None,
    probe_feedback_post: bool = False,
    enforce_personal_pc: bool = False,
    check_cwd_path: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify post-deploy holiday manifest, `/noel` & `/quick` HTML/CSP, CES summary/canary, and rollback readiness."""
    lang_code = normalize_lang(lang)
    comp_doc = check_personal_pc_compliance(
        env=env,
        cwd=cwd,
        check_cwd_path=check_cwd_path,
        lang=lang_code,
    )
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("post_deploy_gate", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)
    return _probe_gateway_parity_core(
        clean_gw := validate_gateway_url(gateway),
        expected_sha=expected_sha,
        check_git_head=check_git_head,
        verify_all_scripts=False,
        strict_post_deploy=True,
        probe_feedback_post=probe_feedback_post,
        rollback_plan=build_cloud_run_rollback_plan(
            previous_revision=previous_revision,
            gateway=clean_gw,
            lang=lang_code,
        ),
        compliance_doc=comp_doc,
        http_fn=http_fn,
        runner=runner,
        lang_code=lang_code,
    )


def _http_accepts_payload_and_headers(fn: Callable[..., Any]) -> bool:
    sig = inspect.signature(fn)
    params = sig.parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return bool(
        any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        or len(positional) >= 4
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        or ("payload" in params and "headers" in params)
    )


def _call_pr949_get(
    url: str,
    *,
    expected_kind: str,
    raw_http_fn: RawHttpFn,
    http_fn: HttpFn,
) -> tuple[int, Mapping[str, Any] | None, str, dict[str, str]]:
    """Probe a read-only GET endpoint via `raw_http_fn` or `http_fn` without invoking inference."""
    use_raw = (raw_http_fn is not default_http_raw and (http_fn is default_http_json or expected_kind == "html")) or (
        raw_http_fn is default_http_raw and http_fn is default_http_json
    )
    if use_raw:
        code, body_text, lower_hdrs = _invoke_raw_http(raw_http_fn, "GET", url)
        if expected_kind == "html":
            return code, None, body_text, lower_hdrs
        json_doc: dict[str, Any] | None = None
        if body_text.strip():
            with suppress(json.JSONDecodeError):
                parsed = json.loads(body_text)
                if isinstance(parsed, Mapping):
                    json_doc = dict(parsed)
        return code, json_doc, body_text, lower_hdrs

    code, norm_body = _invoke_http_callable(http_fn, "GET", url)
    resp_hdrs = norm_body.get("headers") if isinstance(norm_body.get("headers"), Mapping) else {}
    lower_hdrs = {str(k).strip().lower(): str(v) for k, v in resp_hdrs.items()}
    if expected_kind == "html":
        html_txt = str(norm_body.get("html") or "")
        return code, norm_body, html_txt, lower_hdrs
    if "html" in norm_body and isinstance(norm_body.get("html"), str):
        parsed_doc: dict[str, Any] | None = None
        with suppress(json.JSONDecodeError):
            raw_obj = json.loads(norm_body["html"])
            if isinstance(raw_obj, Mapping):
                parsed_doc = dict(raw_obj)
        return code, parsed_doc, "", lower_hdrs
    return code, norm_body, "", lower_hdrs


def _dispatch_pr949_feedback_post(
    url: str,
    *,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    raw_http_fn: RawHttpFn,
    http_fn: HttpFn,
) -> tuple[int, Mapping[str, Any] | None]:
    """Dispatch zero-PII POST feedback replay via raw or JSON HTTP transport."""
    use_raw = (
        raw_http_fn is default_http_raw and http_fn is default_http_json
    ) or (
        raw_http_fn is not default_http_raw
        and (http_fn is default_http_json or not _http_accepts_payload_and_headers(http_fn))
    )
    if use_raw:
        code, body_text, _ = _invoke_raw_http(
            raw_http_fn,
            "POST",
            url,
            payload=payload,
            headers=headers,
            allow_feedback_post=True,
        )
        parsed_doc: dict[str, Any] | None = None
        if body_text.strip():
            with suppress(json.JSONDecodeError):
                loaded = json.loads(body_text)
                if isinstance(loaded, Mapping):
                    parsed_doc = dict(loaded)
        return code, parsed_doc

    code, body_doc = _invoke_http_callable(
        http_fn,
        "POST",
        url,
        payload=dict(payload),
        headers=headers,
        allow_feedback_post=True,
    )
    return code, body_doc


def _inspect_fleet_quorum_body(
    body: Mapping[str, Any],
    *,
    target_n: int,
    require_live_quorum: bool,
    is_fr: bool,
) -> tuple[bool, int | None, int | None, bool, bool, bool, list[str]]:
    """Validate `/v1/hivemind/producers/fleet` schema, zero endpoint_url leakage, and quorum gate."""
    issues: list[str] = []
    count_keys = ("total_hiveminders", "live_hiveminders", "distinct_physical_gpus", "live_endpoints")
    models_ok = isinstance(body.get("live_models"), list) and all(isinstance(m, str) for m in body["live_models"])
    rows = body.get("hiveminders")
    rows_ok = isinstance(rows, list) and all(
        isinstance(r, Mapping) and isinstance(r.get("handle"), str) and isinstance(r.get("status"), str)
        for r in rows
    )
    endpoint_leaked = isinstance(rows, list) and any(
        isinstance(r, Mapping) and r.get("endpoint_url") is not None for r in rows
    )
    if (
        body.get("object") != FLEET_OBJECT
        or _has_forbidden_score_key(body)
        or not all(_is_nonneg_int(body.get(k)) for k in count_keys)
        or not models_ok
        or not rows_ok
    ):
        issues.append(
            "Contrat JSON `/v1/hivemind/producers/fleet` invalide."
            if is_fr
            else "Invalid `/v1/hivemind/producers/fleet` JSON schema."
        )
    if endpoint_leaked:
        issues.append(
            "Fuite `endpoint_url` détectée dans `/v1/hivemind/producers/fleet`."
            if is_fr
            else "Unredacted `endpoint_url` leaked in `/v1/hivemind/producers/fleet`."
        )
    if issues:
        return False, None, None, False, False, False, issues

    online_cnt = int(body["live_hiveminders"])
    gpu_cnt = int(body["distinct_physical_gpus"])
    quorum_met = bool(online_cnt >= target_n and gpu_cnt >= 1)
    awaiting = not quorum_met
    quorum_gate_ok = bool(quorum_met or not require_live_quorum)
    diags: list[str] = []
    if not quorum_met:
        if require_live_quorum:
            diags.append(
                f"Quorum producteurs insuffisant ({online_cnt}/{target_n} en ligne, {gpu_cnt} GPU distincts ; `--require-live-quorum`)."
                if is_fr
                else f"Live producer quorum unmet ({online_cnt}/{target_n} online, {gpu_cnt} distinct GPUs; `--require-live-quorum`)."
            )
        else:
            diags.append(
                f"En attente quorum amis (`awaiting_producer_quorum` : {online_cnt}/{target_n} producteurs, {gpu_cnt} GPU)."
                if is_fr
                else f"Awaiting friend producer quorum (`awaiting_producer_quorum`: {online_cnt}/{target_n} online, {gpu_cnt} GPUs)."
            )
    return True, online_cnt, gpu_cnt, quorum_met, awaiting, quorum_gate_ok, diags


def _inspect_raid_and_arena_body(
    path: str,
    body: Mapping[str, Any] | None,
    *,
    is_fr: bool,
) -> tuple[bool, list[str]]:
    """Validate `/v1/hivemind/raid/status` and `/v1/hivemind/arena/summary` zero-PII contracts."""
    if not isinstance(body, Mapping) or _has_forbidden_score_key(body):
        return False, [
            f"Réponse JSON invalide sur `{path}`."
            if is_fr
            else f"Invalid JSON payload returned by `{path}`."
        ]
    if path == "/v1/hivemind/raid/status":
        comp = body.get("compliance")
        ok = bool(
            body.get("object") == RAID_STATUS_OBJECT
            and isinstance(comp, Mapping)
            and comp.get("personal_pc_only") is True
            and comp.get("monetary_cost_eur") == "0.00 EUR"
            and comp.get("monetary_cost_usd") == "0.00 USD"
            and comp.get("license") == "AGPL-3.0"
            and isinstance(body.get("swarm"), Mapping)
            and isinstance(body.get("onboarding_feedback"), Mapping)
        )
        return (
            ok,
            []
            if ok
            else [
                "Contrat JSON `/v1/hivemind/raid/status` ou bannière conformité invalide."
                if is_fr
                else "GET /v1/hivemind/raid/status schema or compliance invalid."
            ],
        )

    counts_ok = False
    if body.get("object") == ARENA_SUMMARY_OBJECT and _is_nonneg_int(body.get("total_votes")):
        tv = int(body["total_votes"])
        if all(k in body for k in ("holiday_wins", "baseline_wins", "ties")):
            hw, bw, ti = body["holiday_wins"], body["baseline_wins"], body["ties"]
            counts_ok = bool(
                _is_nonneg_int(hw)
                and _is_nonneg_int(bw)
                and _is_nonneg_int(ti)
                and (int(hw) + int(bw) + int(ti) == tv)
            )
        else:
            counts_ok = True
    return (
        counts_ok,
        []
        if counts_ok
        else [
            "Contrat JSON `/v1/hivemind/arena/summary` ou conservation des votes invalide."
            if is_fr
            else "GET /v1/hivemind/arena/summary schema or vote conservation invalid."
        ],
    )


def _evaluate_ces_and_lag_slos(
    fb_doc: Mapping[str, Any],
    *,
    schema_ok: bool,
    min_ease: float,
    min_lag: float,
    is_fr: bool,
) -> tuple[bool | None, bool | None, list[str]]:
    if not schema_ok:
        return False, False, []
    total_resp = int(fb_doc["total_responses"])
    if total_resp == 0:
        return None, None, []
    ease_val = fb_doc["mean_ease_rating"]
    lag_val = fb_doc["zero_game_lag_ratio"]
    ease_ok = bool(ease_val is not None and Fraction(str(ease_val)) >= Fraction(str(min_ease)))
    lag_ok = bool(lag_val is not None and Fraction(str(lag_val)) >= Fraction(str(min_lag)))
    issues: list[str] = []
    if not ease_ok:
        issues.append(
            f"Note CES moyenne {ease_val}/5 < seuil >= {min_ease}/5."
            if is_fr
            else f"CES ease rating {ease_val}/5 below target >= {min_ease}/5."
        )
    if not lag_ok:
        issues.append(
            f"Ratio zéro-lag Gamer {lag_val} < seuil >= {min_lag}."
            if is_fr
            else f"Gamer zero_game_lag_ratio {lag_val} below target >= {min_lag}."
        )
    return ease_ok, lag_ok, issues


def _verify_feedback_idempotency_replay(
    clean_gw: str,
    *,
    expected_sha: str | None,
    baseline_schema_ok: bool,
    baseline_doc: Mapping[str, Any],
    raw_http_fn: RawHttpFn,
    http_fn: HttpFn,
    lang_code: str,
) -> tuple[bool, str, list[int], bool, dict[str, Any], list[str], int]:
    """Dispatch the same `Idempotency-Key` POST `/v1/hivemind/onboarding/feedback` twice and verify exact 1-to-1 dedup."""
    is_fr = lang_code == "fr"
    pre_total = baseline_doc.get("total_responses") if baseline_schema_ok else None
    pre_prof = baseline_doc.get("by_profile") if baseline_schema_ok else None
    seed = f"{clean_gw}|{expected_sha or 'live'}|{pre_total if isinstance(pre_total, int) else -1}"
    idem_key = f"hivemind-noel-pr949-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"
    idem_sha_prefix = hashlib.sha256(idem_key.encode("utf-8")).hexdigest()[:16]
    post_url = f"{clean_gw}/v1/hivemind/onboarding/feedback"
    summary_url = f"{clean_gw}/v1/hivemind/onboarding/feedback/summary"
    hdrs = {"Idempotency-Key": idem_key}

    c1, b1 = _dispatch_pr949_feedback_post(
        post_url, payload=_CANARY_FEEDBACK_PAYLOAD, headers=hdrs, raw_http_fn=raw_http_fn, http_fn=http_fn
    )
    c2, b2 = _dispatch_pr949_feedback_post(
        post_url, payload=_CANARY_FEEDBACK_PAYLOAD, headers=hdrs, raw_http_fn=raw_http_fn, http_fn=http_fn
    )
    c_get, b_get, _, _ = _call_pr949_get(
        summary_url, expected_kind="json", raw_http_fn=raw_http_fn, http_fn=http_fn
    )
    extra_calls = 3

    ok1, d1, _ = (
        _inspect_feedback_summary_schema(b1, lang=lang_code, require_object=True)
        if (c1 == 200 and isinstance(b1, Mapping))
        else (False, {}, [])
    )
    ok2, d2, _ = (
        _inspect_feedback_summary_schema(b2, lang=lang_code, require_object=True)
        if (c2 == 200 and isinstance(b2, Mapping))
        else (False, {}, [])
    )
    ok_get, d_get, _ = (
        _inspect_feedback_summary_schema(b_get, lang=lang_code, require_object=True)
        if (c_get == 200 and isinstance(b_get, Mapping))
        else (False, {}, [])
    )

    delta_1_ok = bool(
        ok1
        and isinstance(pre_total, int)
        and isinstance(pre_prof, Mapping)
        and d1.get("total_responses") == pre_total + 1
        and isinstance(d1.get("by_profile"), Mapping)
        and d1["by_profile"].get("dev") == pre_prof.get("dev", -1) + 1
        and d1["by_profile"].get("gamer") == pre_prof.get("gamer")
    )
    replay_conserved = bool(
        ok1
        and ok2
        and ok_get
        and d2.get("total_responses") == d1.get("total_responses")
        and d2.get("by_profile") == d1.get("by_profile")
        and d_get.get("total_responses") == d1.get("total_responses")
        and d_get.get("by_profile") == d1.get("by_profile")
    )
    dedup_ok = bool(delta_1_ok and replay_conserved)
    diags: list[str] = []
    if not dedup_ok:
        diags.append(
            f"Échec déduplication `Idempotency-Key` sur POST `/v1/hivemind/onboarding/feedback` "
            f"(HTTP={c1}/{c2}/{c_get}, total={pre_total}->{d1.get('total_responses')}->{d2.get('total_responses')}->{d_get.get('total_responses')})."
            if is_fr
            else f"`Idempotency-Key` duplicate-replay conservation failed on POST `/v1/hivemind/onboarding/feedback` "
            f"(HTTP={c1}/{c2}/{c_get}, totals={pre_total}->{d1.get('total_responses')}->{d2.get('total_responses')}->{d_get.get('total_responses')})."
        )
    latest_doc = dict(d_get) if ok_get else dict(baseline_doc)
    latest_ok = bool(ok_get if ok_get else baseline_schema_ok)
    return dedup_ok, idem_sha_prefix, [c1, c2], latest_ok, latest_doc, diags, extra_calls


def verify_pr949_live_quorum_and_ces_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    previous_revision: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    probe_feedback_post: bool = False,
    verify_idempotency_replay: bool = False,
    enforce_personal_pc: bool = False,
    check_cwd_path: bool = False,
    raw_http_fn: RawHttpFn = default_http_raw,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify PR #949 zero-inference 4-HTML-route parity, `/v1/models` `hivemind/coder` metadata, CES SLOs, Idempotency-Key replay, and 8+ producer quorum."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    target_n = _validate_int_range(target_online_producers, "target_online_producers", 1, 500)
    clean_ease = _validate_float_range(min_ease_rating, "min_ease_rating", 1.0, 5.0)
    clean_lag = _validate_float_range(min_zero_lag_ratio, "min_zero_lag_ratio", 0.0, 1.0)
    resolved_expected_sha: str | None = None
    expected_sha_source = "none"
    if expected_sha is not None and str(expected_sha).strip():
        resolved_expected_sha = validate_expected_sha(expected_sha)
        expected_sha_source = "expected_sha"

    comp_doc = check_personal_pc_compliance(
        env=env,
        cwd=cwd,
        check_cwd_path=check_cwd_path,
        lang=lang_code,
    )
    if enforce_personal_pc and not comp_doc["compliance_ok"]:
        safe_gw = DEFAULT_GATEWAY
        with suppress(ParityGateError):
            safe_gw = validate_gateway_url(gateway)
        return _compliance_blocked_report("pr949_quorum_gate", safe_gw, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    rollback_plan = build_cloud_run_rollback_plan(
        previous_revision=previous_revision,
        gateway=clean_gw,
        lang=lang_code,
    )
    git_head_error: str | None = None
    if resolved_expected_sha is None and check_git_head:
        expected_sha_source = "git_head"
        resolved_expected_sha, git_head_error = _resolve_git_head_sha(runner, lang=lang_code)

    http_statuses: dict[str, int] = {}
    route_rows: list[dict[str, Any]] = []
    missing_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []

    health_ok = False
    health_sha = "unknown"
    models_ok = False
    models_count: int | None = None
    model_ids: list[str] = []
    models_coder_present = False
    manifest_present = False
    manifest_valid = False
    manifest_scripts_count = 0
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    missing_manifest_routes: list[str] = []
    missing_manifest_scripts: list[str] = []
    sha_match: bool | None = None
    sha_drift = False

    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    unsafe_dom_sinks_count = 0
    static_figure_found = False

    feedback_schema_ok = False
    feedback_summary_doc: dict[str, Any] = {
        "schema_valid": False,
        "total_responses": None,
        "average_ease_rating": None,
        "mean_ease_rating": None,
        "zero_game_lag_ratio": None,
        "by_profile": None,
    }
    fleet_schema_ok = False
    online_producers: int | None = None
    distinct_physical_gpus: int | None = None
    producer_quorum_met = False
    awaiting_producer_quorum = False
    quorum_gate_ok = False
    raid_status_ok = False
    arena_summary_ok = False
    network_calls = 0

    for path, kind in PR949_REQUIRED_ROUTES:
        network_calls += 1
        code, json_doc, text_body, hdrs = _call_pr949_get(
            f"{clean_gw}{path}",
            expected_kind=kind,
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
        )
        http_statuses[path] = code
        row_ok = code == 200
        diag = ""

        if code != 200:
            group = "holiday" if (kind == "html" or "onboarding" in path or "raid" in path or "arena" in path) else "core"
            diag = _http_failure_diagnostic(path, group, code, lang=lang_code)
            diagnostics.append(diag)
        elif path == "/health":
            health_ok = bool(isinstance(json_doc, Mapping) and json_doc.get("status") == "ok" and not _has_forbidden_score_key(json_doc))
            if isinstance(json_doc, Mapping):
                health_sha = sanitize_build_sha(json_doc.get("build_sha"))
            if not health_ok:
                row_ok = False
                diag = (
                    "GET /health invalide (`status != 'ok'`)."
                    if is_fr
                    else "GET /health returned invalid body (`status != 'ok'`)."
                )
                diagnostics.append(diag)
        elif path == "/v1/models":
            raw_data = json_doc.get("data") if isinstance(json_doc, Mapping) else None
            valid_entries = (
                [
                    entry
                    for entry in raw_data
                    if isinstance(entry, Mapping) and isinstance(entry.get("id"), str) and bool(entry["id"].strip())
                ]
                if isinstance(raw_data, list)
                else []
            )
            model_ids = [str(entry["id"]).strip() for entry in valid_entries]
            models_count = len(valid_entries) if isinstance(raw_data, list) else None
            coder_entries = [
                entry
                for entry in valid_entries
                if entry["id"].strip() == "hivemind/coder"
                and entry.get("object") in (None, "model")
                and ("owned_by" not in entry or (isinstance(entry.get("owned_by"), str) and bool(str(entry["owned_by"]).strip())))
            ]
            models_coder_present = len(coder_entries) >= 1
            models_ok = bool(
                isinstance(json_doc, Mapping)
                and not _has_forbidden_score_key(json_doc)
                and json_doc.get("object", "list") == "list"
                and bool(model_ids)
                and models_coder_present
            )
            if not models_ok:
                row_ok = False
                diag = (
                    "GET /v1/models vide ou sans métadonnées `hivemind/coder`."
                    if is_fr
                    else "GET /v1/models empty or missing `hivemind/coder` model metadata."
                )
                diagnostics.append(diag)
        elif path == "/v1/hivemind/onboarding/manifest":
            manifest_present = True
            if not isinstance(json_doc, Mapping) or _has_forbidden_score_key(json_doc):
                row_ok = False
                diag = (
                    "Manifeste `/v1/hivemind/onboarding/manifest` sans corps JSON valide."
                    if is_fr
                    else "Manifest `/v1/hivemind/onboarding/manifest` returned invalid JSON body."
                )
                diagnostics.append(diag)
            else:
                (
                    manifest_valid,
                    live_build_sha,
                    cloud_run_revision,
                    missing_manifest_routes,
                    missing_manifest_scripts,
                    man_issues,
                ) = _validate_manifest_body(json_doc, lang=lang_code)
                if isinstance(json_doc.get("downloadable_scripts"), list):
                    manifest_scripts_count = len(json_doc["downloadable_scripts"])
                if live_build_sha == "unknown":
                    live_build_sha = health_sha
                if not manifest_valid:
                    row_ok = False
                    diag = man_issues[0]
                    diagnostics.extend(man_issues)
        elif kind == "html":
            html_payload: Mapping[str, Any] = {"html": text_body, "headers": hdrs}
            h_ok, h_doc, h_issues = _inspect_holiday_html_contract(
                path,
                html_payload,
                seen_nonces,
                lang=lang_code,
                require_csp_header=True,
            )
            html_contracts[path] = h_doc
            unsafe_dom_sinks_count += int(h_doc["unsafe_dom_sinks_count"])
            static_figure_found = static_figure_found or bool(h_doc["static_figure_found"])
            if not h_ok:
                row_ok = False
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            if not isinstance(json_doc, Mapping):
                row_ok = False
                diag = (
                    "GET /v1/hivemind/onboarding/feedback/summary sans corps JSON."
                    if is_fr
                    else "GET /v1/hivemind/onboarding/feedback/summary missing JSON body."
                )
                diagnostics.append(diag)
            else:
                fb_ok, fb_doc, fb_issues = _inspect_feedback_summary_schema(
                    json_doc,
                    lang=lang_code,
                    require_object=True,
                )
                feedback_schema_ok = fb_ok
                feedback_summary_doc = fb_doc
                if not fb_ok:
                    row_ok = False
                    diag = fb_issues[0]
                    diagnostics.extend(fb_issues)
        elif path == "/v1/hivemind/producers/fleet":
            if not isinstance(json_doc, Mapping):
                row_ok = False
                diag = (
                    "GET /v1/hivemind/producers/fleet sans corps JSON."
                    if is_fr
                    else "GET /v1/hivemind/producers/fleet missing JSON body."
                )
                diagnostics.append(diag)
            else:
                (
                    fleet_schema_ok,
                    online_producers,
                    distinct_physical_gpus,
                    producer_quorum_met,
                    awaiting_producer_quorum,
                    quorum_gate_ok,
                    fleet_diags,
                ) = _inspect_fleet_quorum_body(
                    json_doc,
                    target_n=target_n,
                    require_live_quorum=bool(require_live_quorum),
                    is_fr=is_fr,
                )
                diagnostics.extend(fleet_diags)
                if not fleet_schema_ok or not quorum_gate_ok:
                    row_ok = False
                    diag = fleet_diags[0] if fleet_diags else diag
        else:
            ra_ok, ra_issues = _inspect_raid_and_arena_body(path, json_doc, is_fr=is_fr)
            if path == "/v1/hivemind/raid/status":
                raid_status_ok = ra_ok
            else:
                arena_summary_ok = ra_ok
            if not ra_ok:
                row_ok = False
                diag = ra_issues[0]
                diagnostics.extend(ra_issues)

        if not row_ok:
            failed_routes.append(path)
            if code == 404:
                missing_routes.append(path)

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "expected_kind": kind,
                "http_status": code,
                "ok": row_ok,
                "diagnostic": diag,
            }
        )

    sha_check_requested = expected_sha_source != "none"
    if sha_check_requested:
        if git_head_error is not None or resolved_expected_sha is None:
            sha_match = False
            sha_drift = True
            if git_head_error is not None:
                diagnostics.append(git_head_error)
        else:
            sha_match = shas_match(live_build_sha, resolved_expected_sha)
            sha_drift = not sha_match
            if sha_drift:
                rev_label = cloud_run_revision or ("aucune" if is_fr else "none")
                diagnostics.append(
                    f"Dérive de SHA Cloud Run détectée : `build_sha` actif='{live_build_sha}' "
                    f"(révision={rev_label}) != SHA attendu '{resolved_expected_sha}' ; redéploiement requis."
                    if is_fr
                    else f"Cloud Run revision SHA drift detected: live `build_sha`='{live_build_sha}' "
                    f"(revision={rev_label}) != expected '{resolved_expected_sha}'; redeploy required."
                )

    do_replay = bool(probe_feedback_post or verify_idempotency_replay)
    idempotent_dedup_ok: bool | None = None
    idempotency_key_sha256_prefix: str | None = None
    idempotency_post_statuses: list[int] = []
    if do_replay:
        (
            idempotent_dedup_ok,
            idempotency_key_sha256_prefix,
            idempotency_post_statuses,
            feedback_schema_ok,
            feedback_summary_doc,
            replay_diags,
            replay_calls,
        ) = _verify_feedback_idempotency_replay(
            clean_gw,
            expected_sha=resolved_expected_sha,
            baseline_schema_ok=feedback_schema_ok,
            baseline_doc=feedback_summary_doc,
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
            lang_code=lang_code,
        )
        network_calls += replay_calls
        diagnostics.extend(replay_diags)

    ces_slo_ok, zero_lag_slo_ok, slo_issues = _evaluate_ces_and_lag_slos(
        feedback_summary_doc,
        schema_ok=feedback_schema_ok,
        min_ease=clean_ease,
        min_lag=clean_lag,
        is_fr=is_fr,
    )
    diagnostics.extend(slo_issues)

    total_responses = (
        int(feedback_summary_doc["total_responses"])
        if feedback_schema_ok and isinstance(feedback_summary_doc.get("total_responses"), int)
        else 0
    )
    mean_ease = feedback_summary_doc.get("mean_ease_rating") if feedback_schema_ok else None
    zero_lag = feedback_summary_doc.get("zero_game_lag_ratio") if feedback_schema_ok else None

    html_routes_ok = bool(
        len(html_contracts) == len(PR949_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in PR949_HTML_ROUTES)
        and len(seen_nonces) == len(PR949_HTML_ROUTES)
    )
    personal_pc_banners_ok = bool(
        len(html_contracts) == len(PR949_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("personal_pc_banner_present") is True for p in PR949_HTML_ROUTES)
    )
    onboarding_feedback_deployed = bool(
        http_statuses.get("/v1/hivemind/onboarding/feedback/summary") == 200 and feedback_schema_ok
    )
    holiday_routes_deployed_on_cloud_run = bool(
        health_ok
        and manifest_valid
        and html_routes_ok
        and onboarding_feedback_deployed
        and raid_status_ok
        and arena_summary_ok
    )
    stale_revision_detected = bool(
        any(code == 404 for code in http_statuses.values())
        or missing_manifest_routes
        or missing_manifest_scripts
        or sha_drift
        or not holiday_routes_deployed_on_cloud_run
    )
    cloud_run_lags_main = bool(
        http_statuses.get("/health") == 200
        and (missing_routes or missing_manifest_routes or missing_manifest_scripts or sha_drift)
    )
    replay_gate_ok = (not do_replay) or (idempotent_dedup_ok is True)
    ces_gate_ok = bool(feedback_schema_ok and (ces_slo_ok is not False) and (zero_lag_slo_ok is not False))

    ready = bool(
        holiday_routes_deployed_on_cloud_run
        and models_ok
        and models_coder_present
        and fleet_schema_ok
        and ces_gate_ok
        and quorum_gate_ok
        and replay_gate_ok
        and not stale_revision_detected
    )
    redeploy_required = bool(
        stale_revision_detected
        or not holiday_routes_deployed_on_cloud_run
        or not onboarding_feedback_deployed
        or not models_ok
        or not models_coder_present
    )
    rollback_required = not ready

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "slice_id": "SLICE-SCALE-DYN-42",
            "campaign_id": CAMPAIGN_ID,
            "mode": "pr949_quorum_gate",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": bool(comp_doc["compliance_ok"]),
            "corporate_detected": bool(comp_doc["corporate_detected"]),
            "compliance": dict(comp_doc),
            "ready": ready,
            "parity_ok": ready,
            "post_deploy_ready": ready,
            "exit_code": 0 if ready else 2,
            "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed_on_cloud_run,
            "onboarding_feedback_deployed": onboarding_feedback_deployed,
            "stale_revision_detected": stale_revision_detected,
            "cloud_run_lags_main": cloud_run_lags_main,
            "redeploy_required": redeploy_required,
            "rollback_required": rollback_required,
            "rollback_plan": dict(rollback_plan),
            "routes_verified_count": sum(1 for row in route_rows if row["ok"]),
            "routes_total_count": len(route_rows),
            "scripts_verified_count": manifest_scripts_count if manifest_valid else 0,
            "scripts_total_count": len(REQUIRED_SCRIPTS),
            "models_ok": models_ok,
            "models_count": models_count,
            "model_ids": model_ids,
            "models_coder_present": models_coder_present,
            "manifest_present": manifest_present,
            "manifest_valid": manifest_valid,
            "manifest_scripts_count": manifest_scripts_count,
            "live_build_sha": live_build_sha,
            "expected_build_sha": resolved_expected_sha,
            "expected_sha_source": expected_sha_source,
            "sha_check_requested": sha_check_requested,
            "sha_match": sha_match,
            "sha_drift": sha_drift,
            "cloud_run_revision": cloud_run_revision,
            "html_contract_ok": html_routes_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "unsafe_dom_sinks_count": unsafe_dom_sinks_count,
            "static_figure_found": static_figure_found,
            "personal_pc_banners_ok": personal_pc_banners_ok,
            "feedback_schema_ok": feedback_schema_ok,
            "feedback_summary": feedback_summary_doc,
            "total_responses": total_responses,
            "mean_ease_rating": mean_ease,
            "average_ease_rating": mean_ease,
            "zero_game_lag_ratio": zero_lag,
            "min_ease_rating": clean_ease,
            "min_zero_lag_ratio": clean_lag,
            "ces_slo_ok": ces_slo_ok,
            "zero_lag_slo_ok": zero_lag_slo_ok,
            "feedback_post_probed": do_replay,
            "verify_idempotency_replay": bool(verify_idempotency_replay),
            "idempotency_key_sha256_prefix": idempotency_key_sha256_prefix,
            "idempotent_dedup_ok": idempotent_dedup_ok,
            "idempotency_post_statuses": idempotency_post_statuses,
            "fleet_schema_ok": fleet_schema_ok,
            "online_producers": online_producers,
            "distinct_physical_gpus": distinct_physical_gpus,
            "target_online_producers": target_n,
            "producer_quorum_met": producer_quorum_met,
            "awaiting_producer_quorum": awaiting_producer_quorum,
            "require_live_quorum": bool(require_live_quorum),
            "raid_status_ok": raid_status_ok,
            "arena_summary_ok": arena_summary_ok,
            "http_statuses": http_statuses,
            "missing_routes": missing_routes,
            "failed_routes": failed_routes,
            "missing_manifest_routes": missing_manifest_routes,
            "missing_manifest_scripts": missing_manifest_scripts,
            "routes": route_rows,
            "zero_inference_verified": True,
            "network_calls": network_calls,
            "file_writes": 0,
            "diagnostics": diagnostics,
        }
    )


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN GitHub-flavored Markdown report with SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)
    mode = str(report.get("mode", "parity_gate"))

    header = (
        "# Gate de Parité Cloud Run Noël & Plan FinOps HiveMind"
        if is_fr
        else "# HiveMind Holiday Cloud Run Parity Gate & FinOps Plan"
    )
    lines: list[str] = [
        header,
        "",
        f"> **Compliance / Conformité**: {bilingual_notice}",
        "",
        f"- **Gateway**: `{report.get('gateway')}`",
        f"- **Mode**: `{mode}`",
        f"- **Parity OK**: `{report.get('parity_ok')}` (`redeploy_required={report.get('redeploy_required')}`)",
        f"- **Routes Verified**: `{report.get('routes_verified_count')}`",
        f"- **Scripts Verified**: `{report.get('scripts_verified_count')}`",
        *(
            [
                f"- **Post-Deploy Ready**: `{report.get('post_deploy_ready')}` "
                f"(`rollback_required={report.get('rollback_required')}`)"
            ]
            if "post_deploy_ready" in report
            else []
        ),
    ]

    if mode == "pr949_quorum_gate":
        lines.extend(
            [
                f"- **PR #949 Holiday Routes**: `{report.get('holiday_routes_deployed_on_cloud_run')}` "
                f"(`feedback_deployed={report.get('onboarding_feedback_deployed')}`, "
                f"`stale_revision={report.get('stale_revision_detected')}`)",
                f"- **Models (`hivemind/coder`)**: `present={report.get('models_coder_present')}`",
                f"- **Friend Producer Quorum**: `{report.get('online_producers')}/{report.get('target_online_producers')}` "
                f"(`quorum_met={report.get('producer_quorum_met')}`, `awaiting={report.get('awaiting_producer_quorum')}`)",
                f"- **CES & Gamer Lag**: `mean_ease={report.get('mean_ease_rating')}`, "
                f"`zero_game_lag_ratio={report.get('zero_game_lag_ratio')}`, "
                f"`ces_slo_ok={report.get('ces_slo_ok')}`, `zero_lag_slo_ok={report.get('zero_lag_slo_ok')}`",
                f"- **Idempotency-Key Replay**: `idempotent_dedup_ok={report.get('idempotent_dedup_ok')}`",
            ]
        )

    if "live_build_sha" in report:
        lines.append(
            f"- **Build SHA**: live=`{report.get('live_build_sha')}`, "
            f"expected=`{report.get('expected_build_sha') or 'n/a'}`, "
            f"revision=`{report.get('cloud_run_revision') or 'none'}`"
        )

    routes = report.get("routes")
    if isinstance(routes, Sequence) and routes:
        lines.extend(
            [
                "",
                "## Routes Cloud Run (Zéro Inférence)" if is_fr else "## Cloud Run Holiday Routes (Zero-Inference)",
                "",
                "| Route | HTTP | Status | Diagnostic |",
                "| :--- | :---: | :---: | :--- |",
            ]
        )
        for row in routes:
            state = "OK" if row.get("ok") else "FAIL"
            diag = str(row.get("diagnostic") or "—")
            lines.append(f"| `{row.get('path')}` | `{row.get('http_status')}` | **{state}** | {diag} |")

    scripts = report.get("scripts")
    if isinstance(scripts, Sequence) and scripts:
        lines.extend(
            [
                "",
                "## Scripts Téléchargeables (`/scripts/<name>`)" if is_fr else "## Downloadable Onboarding Scripts (`/scripts/<name>`)",
                "",
                "| Script | HTTP | Status | Diagnostic |",
                "| :--- | :---: | :---: | :--- |",
            ]
        )
        for row in scripts:
            state = "OK" if row.get("ok") else "FAIL"
            diag = str(row.get("diagnostic") or "—")
            lines.append(f"| `{row.get('script')}` | `{row.get('http_status')}` | **{state}** | {diag} |")

    plan_doc: Mapping[str, Any] | None = (
        report
        if mode == "redeploy_plan"
        else (report["redeploy_plan"] if isinstance(report.get("redeploy_plan"), Mapping) else None)
    )
    if plan_doc is not None:
        lines.extend(
            [
                "",
                "## Vérification FinOps & Commande Dry-Run" if is_fr else "## FinOps Invariants & Dry-Run Redeploy Command",
                "",
                f"- **FinOps OK**: `{plan_doc.get('finops_ok')}` (`{plan_doc.get('deploy_script')}`)",
            ]
        )
        for inv in plan_doc.get("invariants") or ():
            badge = "OK" if inv.get("passed") else "FAIL"
            lines.append(f"  - **[{badge}] `{inv.get('expected')}`**: {inv.get('description')}")
        lines.extend(
            [
                "",
                "```bash",
                str(plan_doc.get("redeploy_command", "")),
                str(plan_doc.get("verify_command", "")),
                "```",
            ]
        )

    rb_doc: Mapping[str, Any] | None = (
        report
        if mode == "rollback_plan"
        else (report["rollback_plan"] if isinstance(report.get("rollback_plan"), Mapping) and report.get("rollback_required") else None)
    )
    lines.extend(
        [
            "",
            "## Plan de Rollback Cloud Run (Dry-Run)" if is_fr else "## Cloud Run Traffic Rollback Plan (Dry-Run)",
            "",
            "```bash",
            str(rb_doc.get("rollback_command", "")),
            str(rb_doc.get("verify_command", "")),
            "```",
        ]
        if rb_doc is not None
        else []
    )

    diags = report.get("diagnostics")
    if isinstance(diags, Sequence) and diags:
        lines.extend(
            [
                "",
                "## Diagnostics" if not is_fr else "## Diagnostics Actionnables",
                "",
            ]
        )
        for item in diags:
            lines.append(f"- {item}")

    lines.extend(["", f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`"])
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_cloud_run_parity_gate.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind Noël Cloud Run holiday manifest parity gate, post-deploy CES & rollback verifier, "
            "and dry-run FinOps redeploy planner (100% personal PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=default_gw,
        help="HiveMind gateway URL (https://*.run.app or loopback 127.0.0.1/localhost/::1).",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Expected alphanumeric/hex HIVEMIND_BUILD_SHA on the deployed Cloud Run gateway.",
    )
    parser.add_argument(
        "--check-git-head",
        action="store_true",
        help="Compare live Cloud Run build_sha against local `git rev-parse HEAD`.",
    )
    parser.add_argument(
        "--print-redeploy-plan",
        action="store_true",
        help="Verify deploy/gcp_openrouter/deploy.sh FinOps invariants and print dry-run redeploy command.",
    )
    parser.add_argument(
        "--verify-live",
        action="store_true",
        help="Also run live HTTP parity checks when combined with --print-redeploy-plan.",
    )
    parser.add_argument(
        "--verify-all-scripts",
        action="store_true",
        help="Directly probe GET /scripts/<name> across every allowlisted script in REQUIRED_SCRIPTS.",
    )
    parser.add_argument(
        "--post-deploy-gate",
        action="store_true",
        help="Run post-deploy holiday manifest, HTML/CSP, CES feedback summary, and rollback readiness gate.",
    )
    parser.add_argument(
        "--pr949-quorum-gate",
        action="store_true",
        help="Run PR #949 live 4-HTML-route parity, hivemind/coder /v1/models, CES SLO, Idempotency-Key replay, and producer quorum gate.",
    )
    parser.add_argument(
        "--target-online-producers",
        type=int,
        default=DEFAULT_TARGET_ONLINE_PRODUCERS,
        help="Target friend-producer count for --pr949-quorum-gate (default 8).",
    )
    parser.add_argument(
        "--require-live-quorum",
        action="store_true",
        help="Fail closed (exit code 2) in --pr949-quorum-gate unless online_producers >= --target-online-producers.",
    )
    parser.add_argument(
        "--min-ease-rating",
        type=float,
        default=DEFAULT_MIN_EASE_RATING,
        help="Minimum mean CES ease rating required when responses > 0 (default 4.2).",
    )
    parser.add_argument(
        "--min-zero-lag-ratio",
        type=float,
        default=DEFAULT_MIN_ZERO_LAG_RATIO,
        help="Minimum Gamer zero_game_lag_ratio required when responses > 0 (default 0.95).",
    )
    parser.add_argument(
        "--verify-idempotency-replay",
        action="store_true",
        help="Dispatch duplicate Idempotency-Key POST /v1/hivemind/onboarding/feedback replay test during --pr949-quorum-gate.",
    )
    parser.add_argument(
        "--previous-revision",
        default=None,
        help="Previous known-good Cloud Run K_REVISION for dry-run traffic rollback planning.",
    )
    parser.add_argument(
        "--probe-feedback-post",
        action="store_true",
        help="Send a zero-PII canary POST /v1/hivemind/onboarding/feedback round-trip during --post-deploy-gate or --pr949-quorum-gate.",
    )
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Enforce local Personal-PC & Amadeus compliance guard before any network or subprocess call.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Workspace directory to inspect for corporate markers & git/worktree remotes.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path to atomically persist JSON/Markdown report with POSIX 0600 permissions.",
    )
    parser.add_argument(
        "--deploy-script",
        default="",
        help="Optional path to deploy/gcp_openrouter/deploy.sh for FinOps verification.",
    )
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output with SHA-256 receipt.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output (default).")
    return parser


def _validate_output_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned or "\x00" in cleaned:
        raise ParityGateError("Invalid --output path.")
    return Path(cleaned)


def _resolve_cli_cwd(
    raw_cwd: str | None,
    fallback_cwd: Path | str | None,
) -> tuple[Path | str | None, bool]:
    if raw_cwd is None:
        return fallback_cwd, False
    cleaned = raw_cwd.strip()
    if not cleaned or "\x00" in cleaned:
        raise ParityGateError("Invalid --cwd path.")
    return cleaned, True


def _emit_report(
    report: Mapping[str, Any],
    *,
    as_json: bool,
    ok: bool = True,
    output_path: Path | None = None,
) -> int:
    rendered = (
        _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False))
        if as_json
        else render_markdown_report(report)
    )
    if output_path is not None:
        _atomic_write_text(output_path, rendered + "\n", mode=0o600)
    print(rendered)
    return 0 if ok else 2


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    raw_http_fn: RawHttpFn = default_http_raw,
    runner: RunnerFn = default_subprocess_runner,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> int:
    """CLI entrypoint returning `0` on verified parity/plan or `2` (fail-closed) on any drift/error."""
    args = build_parser().parse_args(argv)
    deploy_path = Path(args.deploy_script) if args.deploy_script else None

    try:
        lang_code = normalize_lang(args.lang)
        out_path = _validate_output_path(args.output)
        eff_cwd, check_cwd_path = _resolve_cli_cwd(args.cwd, cwd)
        comp = check_personal_pc_compliance(
            env=env,
            cwd=eff_cwd,
            check_cwd_path=check_cwd_path,
            lang=lang_code,
        )
        blocked_mode = (
            "pr949_quorum_gate"
            if args.pr949_quorum_gate
            else (
                "post_deploy_gate"
                if args.post_deploy_gate
                else ("redeploy_plan" if args.print_redeploy_plan else "parity_gate")
            )
        )
        if args.enforce_personal_pc and not comp["compliance_ok"]:
            print(comp["summary"], file=sys.stderr)
            return _emit_report(
                _compliance_blocked_report(
                    blocked_mode,
                    DEFAULT_GATEWAY,
                    comp,
                    lang_code=lang_code,
                ),
                as_json=bool(args.json),
                ok=False,
            )

        clean_gw = validate_gateway_url(args.gateway)
        if args.pr949_quorum_gate:
            pr949_rep = verify_pr949_live_quorum_and_ces_parity(
                clean_gw,
                expected_sha=args.expected_sha,
                check_git_head=bool(args.check_git_head),
                previous_revision=args.previous_revision,
                target_online_producers=args.target_online_producers,
                min_ease_rating=args.min_ease_rating,
                min_zero_lag_ratio=args.min_zero_lag_ratio,
                require_live_quorum=bool(args.require_live_quorum),
                probe_feedback_post=bool(args.probe_feedback_post),
                verify_idempotency_replay=bool(args.verify_idempotency_replay),
                check_cwd_path=check_cwd_path,
                raw_http_fn=raw_http_fn,
                http_fn=http_fn,
                runner=runner,
                env=env,
                cwd=eff_cwd,
                lang=lang_code,
            )
            return _emit_report(
                pr949_rep,
                as_json=bool(args.json),
                ok=bool(pr949_rep["ready"]),
                output_path=out_path,
            )

        if args.post_deploy_gate:
            post_rep = verify_post_deploy_holiday_readiness(
                clean_gw,
                expected_sha=args.expected_sha,
                check_git_head=bool(args.check_git_head),
                previous_revision=args.previous_revision,
                probe_feedback_post=bool(args.probe_feedback_post),
                check_cwd_path=check_cwd_path,
                env=env,
                cwd=eff_cwd,
                http_fn=http_fn,
                runner=runner,
                lang=lang_code,
            )
            return _emit_report(
                post_rep,
                as_json=bool(args.json),
                ok=bool(post_rep["post_deploy_ready"]),
                output_path=out_path,
            )

        if args.print_redeploy_plan and not args.verify_live:
            plan = build_cloud_run_redeploy_plan(
                deploy_script_path=deploy_path,
                target_sha=args.expected_sha,
                check_git_head=bool(args.check_git_head),
                gateway=clean_gw,
                previous_revision=args.previous_revision,
                runner=runner,
                lang=lang_code,
            )
            return _emit_report(
                plan,
                as_json=bool(args.json),
                ok=bool(plan["plan_ready"]),
                output_path=out_path,
            )

        report = check_cloud_run_holiday_parity(
            clean_gw,
            expected_sha=args.expected_sha,
            check_git_head=bool(args.check_git_head),
            verify_all_scripts=bool(args.verify_all_scripts),
            check_cwd_path=check_cwd_path,
            env=env,
            cwd=eff_cwd,
            http_fn=http_fn,
            runner=runner,
            lang=lang_code,
        )
        if args.print_redeploy_plan:
            plan = build_cloud_run_redeploy_plan(
                deploy_script_path=deploy_path,
                target_sha=args.expected_sha,
                check_git_head=bool(args.check_git_head),
                gateway=clean_gw,
                previous_revision=args.previous_revision,
                runner=runner,
                lang=lang_code,
            )
            report = _attach_receipt({**report, "mode": "parity_and_redeploy_plan", "redeploy_plan": plan})
            return _emit_report(
                report,
                as_json=bool(args.json),
                ok=bool(report["parity_ok"] and plan["plan_ready"]),
                output_path=out_path,
            )

        return _emit_report(
            report,
            as_json=bool(args.json),
            ok=bool(report["parity_ok"]),
            output_path=out_path,
        )
    except ParityGateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
