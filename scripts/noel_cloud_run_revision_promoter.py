#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Idempotent Cloud Run holiday revision promoter, active-training safety lock, and /noel+/feedback canary.

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Workspace Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects environment variables (``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``,
     ``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``,
     ``ZSCALER_CERT_PATH``), workspace paths, and ``.git/config`` remote URLs (including worktree
     ``gitdir:`` / ``commondir`` indirection) BEFORE any network call, pre-promote check, or
     rollout command generation. Fails closed with exit code 2 (0 HTTP calls, 0 subprocess calls,
     0 raw env/hostname/path leaks).
  2. Zero-Inference Revision Drift & Pre-Promote FinOps/Gateway Gate (``detect_revision_drift``,
     ``verify_pre_promote_gateway_and_finops``): probes ``GET /health``,
     ``GET /v1/hivemind/onboarding/manifest``, ``GET /noel``, ``GET /quick``, ``GET /raid-noel``,
     ``GET /arena-noel``, and ``GET /v1/hivemind/onboarding/feedback/summary`` for pre-PR-#949 HTTP
     404 drift, ``build_sha`` mismatch, or missing holiday routes/scripts; audits
     ``deploy/gcp_openrouter/deploy.sh`` FinOps invariants (``--min-instances 0``,
     ``--max-instances 1``, ``--no-cpu-throttling``, ``--concurrency 80``, Litestream
     ``HIVEMIND_ROUTER_REPLICA_URL``, Secret Manager ``HIVEMIND_ROUTER_SECRET``, 0 starter/bonus
     grants) and runs a zero-inference ASGI pre-promote sanity probe when available.
  3. Active-Training Safety Lock (``check_active_training_safety_lock``): inspects read-only
     ``GET /v1/hivemind/t4/status``, ``GET /v1/hivemind/raid/status``,
     ``GET /v1/hivemind/producers/fleet``, plus optional local training lockfile / worker process
     state so live traffic shifts defer safely (``promotion_deferred_training_active=True``,
     exit code 3 with ``--defer-on-active-training``) while ``--no-traffic-only`` staging stays safe.
  4. Idempotent Cloud Run Revision Promoter & Canary Rollback Gate (``build_idempotent_rollout_plan``,
     ``run_post_deploy_holiday_canary``, ``promote_holiday_revision``): constructs sanitized list-argv
     ``gcloud`` commands (no ``shell=True``), short-circuits idempotently when live ``build_sha``
     matches target SHA and holiday/feedback routes are HTTP 200 healthy, defaults safely to
     dry-run (``execute=False``), verifies ``/noel`` & ``/quick`` CSP nonces/DOM safety/0.00 EUR
     banners (stripping ``nonce="..."`` attributes before static-figure checks) plus feedback schema
     and optional ``Idempotency-Key`` canary POST, and triggers automatic rollback on canary failure.
  5. Deploy-on-Merge Manifest Diff & Expanded Smoke Suite (``compute_deploy_manifest_fingerprint``,
     ``detect_deploy_manifest_diff``, ``verify_post_deploy_smoke_suite``,
     ``project_adoption_readiness_gates``, ``orchestrate_deploy_on_merge``): computes deterministic
     SHA-256 manifest digests across Cloud Run-deployed files, verifies ``deploy.sh``
     ``HIVEMIND_BUILD_SHA`` propagation, inspects read-only ``git diff`` against live ``build_sha``,
     verifies ``/v1/models`` (non-empty + ``hivemind/coder`` metadata without ``/v1/chat/completions``)
     and all 4 holiday HTML routes (``/noel``, ``/quick``, ``/raid-noel``, ``/arena-noel``), projects
     readiness gates, and emits zero-PII ``incident_alert`` payloads on failure or rollback.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen-science project licensed under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_cloud_run_revision_promoter/1"
RECEIPT_OBJECT = "hivemind.noel_cloud_run_revision_promoter_receipt"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"

DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_CLOUD_RUN_SERVICE = "hivemind-openrouter"
DEFAULT_CLOUD_RUN_REGION = "us-central1"
DEFAULT_CANDIDATE_TAG = "noel-holiday-canary"
DEFAULT_DEPLOY_SOURCE_DIR = "deploy/gcp_openrouter"

HTTP_TIMEOUT_S = 5.0
SUBPROCESS_TIMEOUT_S = 5.0
GCLOUD_DEPLOY_TIMEOUT_S = 300.0
GCLOUD_TRAFFIC_TIMEOUT_S = 60.0

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_SCRIPT_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"
DEFAULT_SERVER_PY_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "server.py"

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})

DRIFT_PROBE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
)

PRE_PROMOTE_PROBE_ROUTES: tuple[tuple[str, str], ...] = (
    ("/health", "json"),
    ("/noel", "html"),
    ("/quick", "html"),
    ("/v1/hivemind/onboarding/manifest", "json"),
    ("/v1/hivemind/onboarding/feedback/summary", "json"),
)

CANARY_PROBE_ROUTES: tuple[tuple[str, str], ...] = (
    ("/health", "json"),
    ("/v1/hivemind/onboarding/manifest", "json"),
    ("/noel", "html"),
    ("/quick", "html"),
    ("/v1/hivemind/onboarding/feedback/summary", "json"),
)

SMOKE_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")

SMOKE_SUITE_PROBE_ROUTES: tuple[tuple[str, str], ...] = (
    ("/health", "json"),
    ("/v1/models", "json"),
    ("/v1/hivemind/onboarding/manifest", "json"),
    ("/noel", "html"),
    ("/quick", "html"),
    ("/raid-noel", "html"),
    ("/arena-noel", "html"),
    ("/v1/hivemind/onboarding/feedback/summary", "json"),
)

CLOUD_RUN_CORE_PATHS: tuple[str, ...] = (
    "deploy/gcp_openrouter/Dockerfile",
    "deploy/gcp_openrouter/deploy.sh",
    "deploy/gcp_openrouter/entrypoint.sh",
    "deploy/gcp_openrouter/requirements.txt",
    "deploy/gcp_openrouter/server.py",
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
    "USERDNSDOMAIN",
    "USERDOMAIN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "COMPUTERNAME",
    "HOSTNAME",
    "ZSCALER_CERT_PATH",
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
_CLOUD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_CLOUD_TAG_RE = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_CLOUD_PROJECT_RE = re.compile(r"^[a-z][a-z0-9:._-]{2,62}$")
_SAFE_REL_SOURCE_RE = re.compile(r"^(?:\.|[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)$")

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
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)

_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{1,256})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(["'])[^"']*\1""")
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\.write)\b"
)
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + "s\\b|tok" + "/s|HTTP" + " Reqs|Comple" + "tions|\\bs/" + r"round)"
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

_ALLOWED_CANARY_FEEDBACK_KEYS: frozenset[str] = frozenset(
    {"profile", "ease_rating", "zero_game_lag", "platform", "note"}
)
_TRAINING_WORKER_MARKERS: tuple[str, ...] = (
    "e8_qlora_moe_round",
    "hivemind_qlora",
    "hivemind_opentrainer",
    "opentrainer_agent",
    "opentrainer_worker",
    "hivemind_boinc_idle_worker",
    "boinc_client",
    "qlora_train",
    "diloco_worker",
    "swarm_trainer",
)
_SAFE_READONLY_GIT_SUBCOMMANDS: frozenset[str] = frozenset({"rev-parse", "status", "log", "show", "diff"})

_SERVER_LOAD_LOCK = threading.Lock()

HttpFn = Callable[..., Any]
RunnerFn = Callable[..., Any]


class RevisionPromoterError(ValueError):
    """Raised when promoter arguments, Cloud identifiers, host allowlists, or output safety fail."""


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise RevisionPromoterError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise RevisionPromoterError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def mask_api_key(raw_key: object) -> str:
    """Mask an `hm-...` or `hm_live_...` API key so raw secrets never appear in outputs."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        return "<invalid>"
    if not cleaned.startswith(("hm-", "hm_live_")):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    return f"{prefix}***{cleaned[-4:]}"


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace user home directory prefixes with `'~'` and scrub temporary user segments."""
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
                    redacted_home = "~" + suffix.replace("\\", "/")
                    return _PYTEST_USER_DIR_RE.sub("pytest-of-user", redacted_home)
    scrubbed = _HOME_PATH_RE.sub("~", raw)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    return scrubbed.replace("\\", "/") if scrubbed != raw else raw


def sanitize_text_for_display(text: str, *, home: Path | None = None) -> str:
    """Scrub raw keys, Bearer tokens, emails, non-loopback IPv4s, and OS home paths."""
    scrubbed = redact_home_path(text, home=home)
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", scrubbed)
    scrubbed = _API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), scrubbed)
    scrubbed = _EMAIL_RE.sub("<redacted-email>", scrubbed)
    scrubbed = _IPV4_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith("127.") else "<redacted-ip>",
        scrubbed,
    )
    return scrubbed


def scan_output_safety(text: str) -> list[str]:
    """Verify JSON or Markdown output contains zero PII, raw keys, home paths, or `'score': <digit>`."""
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
    for ip_hit in _IPV4_RE.findall(text):
        if not ip_hit.startswith("127."):
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise RevisionPromoterError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def sanitize_build_sha(raw: object) -> str:
    """Sanitize a `build_sha` candidate to lowercase alphanumeric/hex or `'unknown'`."""
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


def validate_target_sha(raw: str) -> str:
    """Validate an explicit `--expected-sha` / `target_sha` value."""
    sanitized = sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise RevisionPromoterError(
            "Target SHA must be a 6..64 character alphanumeric/hex string (never 'unknown' or a secret)."
        )
    return sanitized


def shas_match(live_sha: str, target_sha: str) -> bool:
    """Compare live Cloud Run `build_sha` against `target_sha` (exact or >=7-char hex prefix)."""
    clean_live = sanitize_build_sha(live_sha)
    clean_exp = sanitize_build_sha(target_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
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


def validate_cloud_identifier(raw: str, label: str = "identifier") -> str:
    """Strictly validate Cloud Run service name or region identifier."""
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise RevisionPromoterError(f"Cloud Run {label} must be a non-empty string.")
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or not _CLOUD_ID_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        raise RevisionPromoterError(f"Invalid Cloud Run {label}: {raw!r}.")
    return cleaned


def validate_candidate_tag(raw: str) -> str:
    """Strictly validate Cloud Run traffic tag (lowercase RFC1035 label)."""
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise RevisionPromoterError("Candidate tag must be a non-empty string.")
    cleaned = raw.strip()
    if (
        not cleaned
        or not _CLOUD_TAG_RE.fullmatch(cleaned)
        or any(marker in cleaned for marker in _SECRET_VALUE_MARKERS)
    ):
        raise RevisionPromoterError(f"Invalid Cloud Run candidate tag: {raw!r}.")
    return cleaned


def validate_gcp_project(raw: str | None) -> str | None:
    """Validate optional GCP project id without allowing shell metacharacters or secrets."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise RevisionPromoterError("GCP project id must be a string or None.")
    cleaned = raw.strip()
    if not cleaned:
        return None
    if not _CLOUD_PROJECT_RE.fullmatch(cleaned) or any(m in cleaned.lower() for m in _SECRET_VALUE_MARKERS):
        raise RevisionPromoterError(f"Invalid GCP project identifier: {raw!r}.")
    return cleaned


def validate_deploy_source_dir(raw: str) -> str:
    """Validate relative deploy source path to prevent traversal or shell injection."""
    if isinstance(raw, bool) or not isinstance(raw, str) or not raw.strip():
        raise RevisionPromoterError("Deploy source dir must be a non-empty relative path.")
    cleaned = raw.strip()
    if ".." in cleaned.split("/") or not _SAFE_REL_SOURCE_RE.fullmatch(cleaned):
        raise RevisionPromoterError(f"Unsafe deploy source dir: {raw!r}.")
    return cleaned


def is_allowed_outbound_host(host: str) -> bool:
    """Return True iff `host` is in the strict HiveMind outbound allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app`, `https://*.googleapis.com`, or loopback."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise RevisionPromoterError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise RevisionPromoterError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise RevisionPromoterError("Gateway URL must not contain internal whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise RevisionPromoterError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise RevisionPromoterError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise RevisionPromoterError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise RevisionPromoterError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise RevisionPromoterError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise RevisionPromoterError(
            f"Untrusted host {host!r}; expected https://*.run.app, https://*.googleapis.com, "
            "datasets-server.huggingface.co, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise RevisionPromoterError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise RevisionPromoterError("Remote gateway URL must use https.")
    return cleaned


def _assert_zero_inference_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    allow_feedback_post: bool = False,
) -> str:
    """Validate allowlisted host, forbid `/v1/chat/completions` and `hivemind/coder`, and allow only read-only GET/HEAD or canary feedback POST."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise RevisionPromoterError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise RevisionPromoterError("Request URL contains whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise RevisionPromoterError("Request URL must not embed credentials.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path:
        raise RevisionPromoterError(
            f"Zero-inference guard blocked forbidden inference endpoint {parts.path!r}."
        )
    if payload is not None:
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload:
            raise RevisionPromoterError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (
        allow_feedback_post and verb == "POST" and parts.path == "/v1/hivemind/onboarding/feedback"
    ):
        return cleaned
    raise RevisionPromoterError(
        f"Zero-inference read-only guard blocked HTTP {verb!r} to {parts.path!r}."
    )


def _validate_zero_pii_feedback_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate that a canary feedback POST payload is strictly zero-PII and `extra='forbid'` compliant."""
    if (
        not isinstance(payload, Mapping)
        or (set(payload.keys()) - _ALLOWED_CANARY_FEEDBACK_KEYS)
        or payload.get("profile") not in ("dev", "gamer")
        or isinstance(payload.get("ease_rating"), bool)
        or not isinstance(payload.get("ease_rating"), int)
        or not (1 <= int(payload["ease_rating"]) <= 5)
        or not isinstance(payload.get("zero_game_lag"), bool)
    ):
        raise RevisionPromoterError("Canary feedback POST payload must be a valid zero-PII feedback object.")
    clean_obj = {
        "profile": str(payload["profile"]),
        "ease_rating": int(payload["ease_rating"]),
        "zero_game_lag": bool(payload["zero_game_lag"]),
        "platform": str(payload.get("platform") or "linux")[:24],
        "note": "",
    }
    _assert_safe_output(json.dumps(clean_obj, sort_keys=True, ensure_ascii=False))
    return clean_obj


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
            result = dict(parsed)
            result.setdefault("_headers", lower_headers)
            return int(status_code), result
        return int(status_code), {"data": parsed, "_headers": lower_headers}
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
    """Proxy-free stdlib HTTP helper (`ProxyHandler({})`) enforcing strict host allowlist and zero-inference."""
    verb = str(method or "GET").strip().upper()
    is_feedback_post = verb == "POST" and urlsplit(url).path == "/v1/hivemind/onboarding/feedback"
    clean_url = _assert_zero_inference_request(
        verb, url, payload=payload, allow_feedback_post=is_feedback_post
    )
    if payload is not None and not is_feedback_post:
        raise RevisionPromoterError("Zero-inference HTTP helper refuses payloads outside onboarding feedback.")

    body_bytes: bytes | None = None
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-cloud-run-revision-promoter/1.0",
    }
    if headers:
        for hk, hv in headers.items():
            req_headers[str(hk)] = str(hv)
    if is_feedback_post:
        body_bytes = json.dumps(_validate_zero_pii_feedback_payload(payload), ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(clean_url, data=body_bytes, headers=req_headers, method=verb)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
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


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise RevisionPromoterError("http_fn must return (status_code, body) or (status_code, headers, body).")
    if len(raw_ret) == 3:
        status_code, raw_hdrs, body = raw_ret
        hdr_map = {str(k).lower(): str(v) for k, v in raw_hdrs.items()} if isinstance(raw_hdrs, Mapping) else {}
    else:
        status_code, body = raw_ret
        hdr_map = {}

    if isinstance(body, Mapping):
        out = dict(body)
        if hdr_map:
            existing_hdrs = out.get("headers")
            if isinstance(existing_hdrs, Mapping):
                merged = {**hdr_map, **{str(k).lower(): str(v) for k, v in existing_hdrs.items()}}
            else:
                merged = hdr_map
            out["headers"] = merged
            out.setdefault("_headers", merged)
        return int(status_code), out
    if isinstance(body, str):
        return int(status_code), {
            "html": body,
            "text": body,
            "html_length": len(body),
            "text_length": len(body),
            "headers": hdr_map,
            "csp": hdr_map.get("content-security-policy", ""),
        }
    return int(status_code), {"data": body, "_headers": hdr_map}


def _invoke_http_fn(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    allow_feedback_post: bool = False,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Validate URL/zero-inference invariants and invoke an injectable HTTP callable across arities."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_zero_inference_request(
        verb, url, payload=payload, allow_feedback_post=allow_feedback_post
    )
    clean_payload = (
        _validate_zero_pii_feedback_payload(payload)
        if (allow_feedback_post and verb == "POST")
        else None
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
    except RevisionPromoterError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    return _normalize_http_return(raw)


def _validate_argv_sequence(argv: Sequence[str], *, allowed_binaries: frozenset[str]) -> list[str]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise RevisionPromoterError("Command argv must be a non-empty list of strings.")
    clean: list[str] = []
    for item in argv:
        if not isinstance(item, str) or any(ch in item for ch in ("\x00", "\r", "\n")):
            raise RevisionPromoterError("Command argument contains invalid type or control characters.")
        clean.append(item)
    exe_name = Path(clean[0]).name.lower()
    if exe_name not in allowed_binaries:
        raise RevisionPromoterError(f"Executable {exe_name!r} is not in allowlist {sorted(allowed_binaries)}.")
    return clean


def default_git_runner(
    argv: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[int, str]:
    """Run a local read-only `git` command (`shell=False`) without mutating repository state."""
    clean = _validate_argv_sequence(argv, allowed_binaries=frozenset({"git", "git.exe"}))
    idx = 1
    if len(clean) > 3 and clean[1] == "-C" and clean[2] and not clean[2].startswith("-"):
        idx = 3
    subcmd = clean[idx] if len(clean) > idx else ""
    if subcmd not in _SAFE_READONLY_GIT_SUBCOMMANDS:
        raise RevisionPromoterError(f"Only read-only git commands are permitted, got {subcmd!r}.")
    if subcmd == "diff":
        for extra in clean[idx + 1 :]:
            if extra.startswith("-") and extra != "--name-only":
                raise RevisionPromoterError(
                    f"Only `--name-only` flag is permitted for read-only git diff, got {extra!r}."
                )
    try:
        proc = subprocess.run(
            clean,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=min(max(0.5, float(timeout_s)), 15.0),
            check=False,
            shell=False,
        )
        return int(proc.returncode), proc.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{type(exc).__name__}"


def default_gcloud_runner(
    argv: Sequence[str],
    timeout_s: float = GCLOUD_DEPLOY_TIMEOUT_S,
) -> tuple[int, str]:
    """Execute a validated list-argv `gcloud` command (`shell=False`)."""
    clean = _validate_argv_sequence(argv, allowed_binaries=frozenset({"gcloud", "gcloud.cmd", "gcloud.exe"}))
    try:
        proc = subprocess.run(
            clean,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=min(max(1.0, float(timeout_s)), 900.0),
            check=False,
            shell=False,
        )
        combined = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return int(proc.returncode), combined
    except subprocess.TimeoutExpired:
        return 124, "gcloud_timeout"
    except OSError as exc:
        return 127, f"gcloud_os_error:{type(exc).__name__}"


def _invoke_runner(
    runner: RunnerFn,
    argv: Sequence[str],
    timeout_s: float,
) -> tuple[int, str]:
    sig = inspect.signature(runner)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    pos_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    res = runner(list(argv), timeout_s) if (has_varargs or len(pos_params) >= 2) else runner(list(argv))
    if isinstance(res, tuple) and len(res) >= 2:
        out = str(res[1] or "")
        if len(res) >= 3 and res[2]:
            out = f"{out}\n{res[2]}" if out else str(res[2])
        return int(res[0]), out
    raise RevisionPromoterError("Subprocess runner must return (returncode, output).")


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def extract_git_config_remote_urls(cfg_text: str) -> list[str]:
    """Extract remote and URL rewrite targets from `.git/config` content."""
    if not cfg_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text):
        candidate = (m_kv or m_sec).strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate `.git/config` inside `workspace` (supporting git worktree `gitdir:` indirection)."""
    git_entry = workspace / ".git"
    candidates: list[Path] = []
    if git_entry.is_dir():
        cfg = git_entry / "config"
        if cfg.is_file():
            candidates.append(cfg)
    elif git_entry.is_file():
        try:
            first_lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            first_lines = []
        if first_lines and first_lines[0].strip().lower().startswith("gitdir:"):
            raw_gitdir = first_lines[0].split(":", 1)[1].strip()
            gitdir = Path(raw_gitdir) if Path(raw_gitdir).is_absolute() else (workspace / raw_gitdir).resolve()
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
    return candidates


def evaluate_personal_pc_compliance(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Workspace Compliance Gate (0 network calls, 0 subprocesses, 0 leaks)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = env if env is not None else os.environ
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    indicators: list[str] = []
    matched_markers: set[str] = set()

    def _add_indicator(tag: str, marker: str | None = None) -> None:
        if tag not in indicators:
            indicators.append(tag)
        if marker:
            matched_markers.add(marker)

    for raw_k, raw_v in env_map.items():
        key_up = str(raw_k).strip().upper()
        val_str = str(raw_v or "").strip()
        if not key_up:
            continue
        if key_up.startswith("AMADEUS") and val_str:
            _add_indicator("env:AMADEUS*", "amadeus")
            for hit in _find_corporate_markers(val_str):
                matched_markers.add(hit)
            continue
        if key_up == "ZSCALER_CERT_PATH" and val_str:
            _add_indicator("env:ZSCALER_CERT_PATH", "zscaler")
            continue
        if key_up == "USERDNSDOMAIN" and val_str:
            dns_hits = _find_corporate_markers(val_str)
            _add_indicator("env:USERDNSDOMAIN", dns_hits[0] if dns_hits else "corp")
            for hit in dns_hits:
                matched_markers.add(hit)
            continue
        if key_up in INSPECTED_ENV_KEYS and val_str:
            for hit in _find_corporate_markers(val_str):
                _add_indicator(f"env:{key_up}:{hit}", hit)

    ws_texts = [str(workspace)]
    try:
        resolved_ws = str(workspace.resolve())
        if resolved_ws not in ws_texts:
            ws_texts.append(resolved_ws)
    except OSError:
        pass
    for ws_txt in ws_texts:
        for hit in _find_corporate_markers(ws_txt):
            _add_indicator(f"workspace_path:{hit}", hit)

    for cfg_path in _resolve_git_config_files(workspace):
        try:
            cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            cfg_text = ""
        for remote_url in extract_git_config_remote_urls(cfg_text):
            for hit in _find_corporate_markers(remote_url):
                _add_indicator(f"git_remote:{hit}", hit)

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Conformité PC personnel vérifiée localement ({FINANCIAL_PROMISE}). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"Local personal-PC compliance gate passed ({FINANCIAL_PROMISE}). {COMPLIANCE_NOTICE_BILINGUAL}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS DE CONFORMITÉ ({joined}) : environnement d'entreprise détecté "
            f"(0 appel HTTP, 0 sous-processus, 0 commande générée). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE GATE BLOCKED ({joined}): corporate machine, proxy, or repository detected "
            f"(0 outbound HTTP calls, 0 subprocess calls, 0 commands generated). {COMPLIANCE_NOTICE_BILINGUAL}"
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
            "subprocess_calls": 0,
            "file_writes": 0,
            "exit_code": 0 if compliance_ok else 2,
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            "summary": summary,
        }
    )


def _compliance_blocked_report(
    mode: str,
    gateway: str,
    comp_doc: Mapping[str, Any],
    *,
    lang_code: str,
) -> dict[str, Any]:
    """Return a strictly fail-closed report with zero network, subprocess, or rollout activity."""
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
            "pre_promote_ok": False,
            "finops_ok": False,
            "drift_detected": False,
            "deploy_needed": False,
            "canary_ok": False,
            "smoke_ok": False,
            "plan_ready": False,
            "promoted": False,
            "idempotent_noop": False,
            "already_at_target_revision": False,
            "promotion_deferred_training_active": False,
            "rollback_triggered": False,
            "incident_alert_emitted": False,
            "incident_alert": None,
            "readiness_gates_passed": 0,
            "readiness_gates_total": 5,
            "dry_run_only": True,
            "gcloud_executed": False,
            "gcloud_calls": 0,
            "cloud_state_mutated": False,
            "network_calls": 0,
            "subprocess_calls": 0,
            "file_writes": 0,
            "zero_inference_verified": True,
            "commands": [],
            "exit_code": 2,
            "diagnostics": [str(comp_doc.get("summary") or COMPLIANCE_NOTICE_BILINGUAL)],
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
            "summary": str(comp_doc.get("summary") or COMPLIANCE_NOTICE_BILINGUAL),
        }
    )


def _extract_response_headers(body: Mapping[str, Any]) -> dict[str, str]:
    for key in ("headers", "_headers"):
        raw = body.get(key)
        if isinstance(raw, Mapping):
            return {str(k).strip().lower(): str(v) for k, v in raw.items()}
    return {}


def _inspect_manifest_contract(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, str, str | None, list[str], list[str], list[str]]:
    is_fr = lang_code == "fr"
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
            f"Routes Noël absentes du manifeste : {joined_r}."
            if is_fr
            else f"Manifest `holiday_routes` missing expected routes: {joined_r}."
        )

    raw_scripts = body.get("downloadable_scripts")
    if isinstance(raw_scripts, list) and all(isinstance(s, str) for s in raw_scripts):
        missing_scripts = [s for s in EXPECTED_MANIFEST_SCRIPTS if s not in raw_scripts]
    else:
        missing_scripts = list(EXPECTED_MANIFEST_SCRIPTS)
    if missing_scripts:
        joined_s = ", ".join(missing_scripts)
        issues.append(
            f"Scripts absents du manifeste : {joined_s}."
            if is_fr
            else f"Manifest `downloadable_scripts` missing allowlisted scripts: {joined_s}."
        )

    return len(issues) == 0, live_sha, live_rev, missing_routes, missing_scripts, issues


def _inspect_feedback_summary_contract(
    body: Mapping[str, Any],
    *,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    is_fr = lang_code == "fr"
    raw_total = body.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    total_val = int(raw_total) if total_ok else -1

    has_ease_key = "mean_ease_rating" in body or "average_ease_rating" in body
    raw_ease = body.get("mean_ease_rating") if "mean_ease_rating" in body else body.get("average_ease_rating")
    ease_ok = bool(
        has_ease_key
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

    obj_ok = body.get("object", FEEDBACK_SUMMARY_OBJECT) == FEEDBACK_SUMMARY_OBJECT
    schema_ok = bool(obj_ok and total_ok and ease_ok and lag_ok and prof_ok)
    issues = (
        []
        if schema_ok
        else [
            "Schéma JSON `/v1/hivemind/onboarding/feedback/summary` invalide."
            if is_fr
            else "Feedback summary `/v1/hivemind/onboarding/feedback/summary` JSON schema mismatch."
        ]
    )
    normalized = {
        "schema_valid": schema_ok,
        "total_responses": total_val if total_ok else None,
        "mean_ease_rating": ease_val,
        "zero_game_lag_ratio": lag_val,
        "by_profile": {"dev": int(d_raw), "gamer": int(g_raw)} if prof_ok else None,  # type: ignore[arg-type]
    }
    return schema_ok, normalized, issues


def strip_html_csp_nonce_attributes(html_text: str) -> str:
    """Strip `nonce="..."` / `nonce='...'` attributes before static-figure scanning."""
    return _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', html_text)


def _inspect_holiday_html_contract(
    path: str,
    body: Mapping[str, Any],
    seen_nonces: set[str],
    *,
    strict_contract: bool,
    lang_code: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Inspect holiday HTML page for doctype, unique CSP nonce, DOM safety, and 0.00 EUR compliance banner."""
    is_fr = lang_code == "fr"
    html_text = str(body.get("html") or body.get("text") or "")
    hdrs = _extract_response_headers(body)
    csp_hdr = str(hdrs.get("content-security-policy") or body.get("csp") or "")

    has_doctype = "<!doctype html" in html_text.lower()
    has_html = bool(html_text.strip()) and ("<html" in html_text.lower() or has_doctype)
    attr_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)]
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    csp_low = csp_hdr.lower()
    csp_unsafe = "'unsafe-inline'" in csp_low or "'unsafe-eval'" in csp_low

    nonce: str | None = None
    if hdr_nonces:
        nonce = hdr_nonces[0]
    elif attr_nonces:
        nonce = attr_nonces[0]

    hdr_consistent = (not csp_hdr) or (
        bool(hdr_nonces)
        and len(set(hdr_nonces)) == 1
        and "default-src 'none'" in csp_low
        and not csp_unsafe
        and hdr_nonces[0] in attr_nonces
    )
    csp_ok = bool(attr_nonces and hdr_consistent and nonce is not None and nonce not in seen_nonces)
    if nonce is not None:
        seen_nonces.add(nonce)

    stripped_html = strip_html_csp_nonce_attributes(html_text)
    dom_clean = not (_FORBIDDEN_DOM_SINK_RE.search(stripped_html) or _STATIC_FIG_UNIT_RE.search(stripped_html))

    low_html = html_text.lower()
    has_personal_pc_phrase = (
        "100% personal" in low_html
        or "100 % personnel" in low_html
        or (("100%" in low_html or "100 %" in low_html) and ("personal" in low_html or "personnel" in low_html))
    )
    compliance_banner_ok = bool(
        has_personal_pc_phrase
        and "agpl-3.0" in low_html
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
    )

    if not strict_contract:
        basic_ok = bool(has_html and (not csp_hdr or (not csp_unsafe and hdr_consistent)) and dom_clean)
        issues = (
            []
            if basic_ok
            else [
                f"Page HTML {path} invalide ou sink DOM non sûr."
                if is_fr
                else f"Holiday HTML page {path} invalid or unsafe DOM/CSP detected."
            ]
        )
        return (
            basic_ok,
            {
                "path": path,
                "ok": basic_ok,
                "has_doctype": has_doctype,
                "csp_nonce_verified": csp_ok,
                "dom_clean": dom_clean,
                "personal_pc_banner_present": compliance_banner_ok,
            },
            issues,
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
            f"Contrat CSP {path} invalide : nonce par requête manquant, incohérent ou réutilisé."
            if is_fr
            else f"CSP contract failed on {path}: missing, mismatched, or reused per-request CSP nonce.",
        ),
        (
            dom_clean,
            f"Contrat sécurité HTML {path} violé : sink DOM interdit ou chiffre statique détecté."
            if is_fr
            else f"HTML safety violated on {path}: forbidden DOM sink or static metric figure detected.",
        ),
        (
            compliance_banner_ok,
            f"Bannière PC personnel / 0,00 EUR / AGPL-3.0 absente sur {path}."
            if is_fr
            else f"Personal-PC / 0.00 EUR / AGPL-3.0 compliance banner missing on {path}.",
        ),
    )
    issues = [msg for ok_flag, msg in checks if not ok_flag]
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


def _resolve_git_head_sha(runner: RunnerFn, *, lang_code: str) -> tuple[str | None, str | None]:
    is_fr = lang_code == "fr"
    try:
        rc, out = _invoke_runner(runner, ["git", "rev-parse", "HEAD"], SUBPROCESS_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, RevisionPromoterError) as exc:
        return None, (
            f"Échec lecture SHA git HEAD local ({type(exc).__name__})."
            if is_fr
            else f"Failed to resolve local git HEAD SHA ({type(exc).__name__})."
        )
    candidate = sanitize_build_sha(out.strip()) if rc == 0 else "unknown"
    if rc != 0 or candidate == "unknown":
        return None, (
            "SHA git HEAD invalide via `git rev-parse HEAD`."
            if is_fr
            else "Unable to resolve a valid SHA from `git rev-parse HEAD`."
        )
    return candidate, None


def detect_revision_drift(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    git_runner: RunnerFn = default_git_runner,
    http_fn: HttpFn = default_http_request,
    lang: str = "en",
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Probe Cloud Run holiday & manifest routes (zero inference) to detect pre-PR-#949 404 or SHA drift."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("drift_detection", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)

    resolved_expected_sha: str | None = None
    expected_sha_source = "none"
    git_head_error: str | None = None
    subprocess_calls = 0
    if expected_sha is not None and str(expected_sha).strip():
        resolved_expected_sha = validate_target_sha(expected_sha)
        expected_sha_source = "expected_sha"
    elif check_git_head:
        expected_sha_source = "git_head"
        subprocess_calls = 1
        resolved_expected_sha, git_head_error = _resolve_git_head_sha(git_runner, lang_code=lang_code)

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    pre_pr_949_404_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []

    manifest_present = False
    manifest_valid = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    missing_manifest_routes: list[str] = []
    missing_manifest_scripts: list[str] = []
    seen_nonces: set[str] = set()
    feedback_summary_ok = False
    network_calls = 0

    for path, expected_kind, group in DRIFT_PROBE_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
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
        elif code == 200 and path == "/v1/hivemind/onboarding/manifest":
            manifest_present = True
            (
                manifest_valid,
                live_build_sha,
                cloud_run_revision,
                missing_manifest_routes,
                missing_manifest_scripts,
                manifest_issues,
            ) = _inspect_manifest_contract(body, lang_code=lang_code)
            if not manifest_valid:
                ok = False
                diag = manifest_issues[0]
                diagnostics.extend(manifest_issues)
        elif code == 200 and expected_kind == "html":
            h_ok, _h_doc, h_issues = _inspect_holiday_html_contract(
                path, body, seen_nonces, strict_contract=False, lang_code=lang_code
            )
            if not h_ok:
                ok = False
                diag = h_issues[0]
                diagnostics.extend(h_issues)
        elif code == 200 and path == "/v1/hivemind/onboarding/feedback/summary":
            fb_ok, _fb_doc, fb_issues = _inspect_feedback_summary_contract(body, lang_code=lang_code)
            feedback_summary_ok = fb_ok
            if not fb_ok:
                ok = False
                diag = fb_issues[0]
                diagnostics.extend(fb_issues)

        if not ok:
            failed_routes.append(path)
            if code == 404 and group in ("manifest", "holiday"):
                pre_pr_949_404_routes.append(path)
            if not diag:
                if code == 404:
                    diag = (
                        f"GET {path} -> HTTP 404 (dérive révision pré-PR-#949)."
                        if is_fr
                        else f"GET {path} -> HTTP 404 (pre-PR-#949 revision drift)."
                    )
                elif code == 0:
                    diag = (
                        f"GET {path} injoignable (erreur réseau/TLS)."
                        if is_fr
                        else f"GET {path} unreachable (network/TLS error)."
                    )
                else:
                    diag = (
                        f"GET {path} -> HTTP {code} (attendu 200)."
                        if is_fr
                        else f"GET {path} -> HTTP {code} (expected 200)."
                    )
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

    sha_check_requested = expected_sha_source != "none"
    sha_match: bool | None = None
    sha_mismatch = False
    if sha_check_requested:
        if git_head_error is not None or resolved_expected_sha is None:
            sha_match = False
            sha_mismatch = True
            if git_head_error is not None:
                diagnostics.append(git_head_error)
        else:
            sha_match = shas_match(live_build_sha, resolved_expected_sha)
            sha_mismatch = not sha_match
            if sha_mismatch:
                diagnostics.append(
                    f"Dérive SHA : `build_sha` en direct='{live_build_sha}' != attendu '{resolved_expected_sha}'."
                    if is_fr
                    else f"Build SHA mismatch: live `build_sha`='{live_build_sha}' != expected '{resolved_expected_sha}'."
                )

    pre_pr_949_drift_detected = bool(pre_pr_949_404_routes)
    missing_holiday_routes = sorted(set(pre_pr_949_404_routes) | set(missing_manifest_routes))
    missing_scripts = list(missing_manifest_scripts)
    drift_detected = bool(
        failed_routes
        or pre_pr_949_drift_detected
        or missing_holiday_routes
        or missing_scripts
        or sha_mismatch
        or not manifest_present
        or not manifest_valid
        or not feedback_summary_ok
    )

    summary = (
        (
            f"Aucune dérive détectée sur {clean_gw} (SHA={live_build_sha})."
            if is_fr
            else f"No revision drift detected on {clean_gw} (SHA={live_build_sha})."
        )
        if not drift_detected
        else (
            f"Dérive détectée sur {clean_gw} : pre_pr_949_404={len(pre_pr_949_404_routes)}, "
            f"sha_mismatch={sha_mismatch}, routes_ko={len(failed_routes)}."
            if is_fr
            else f"Revision drift detected on {clean_gw}: pre_pr_949_404={len(pre_pr_949_404_routes)}, "
            f"sha_mismatch={sha_mismatch}, failed_routes={len(failed_routes)}."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "drift_detection",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "drift_detected": drift_detected,
            "pre_pr_949_drift_detected": pre_pr_949_drift_detected,
            "pre_pr_949_404_routes": pre_pr_949_404_routes,
            "sha_check_requested": sha_check_requested,
            "sha_match": sha_match,
            "sha_mismatch": sha_mismatch,
            "live_build_sha": live_build_sha,
            "expected_build_sha": resolved_expected_sha,
            "expected_sha_source": expected_sha_source,
            "cloud_run_revision": cloud_run_revision,
            "manifest_present": manifest_present,
            "manifest_valid": manifest_valid,
            "feedback_summary_ok": feedback_summary_ok,
            "missing_holiday_routes": missing_holiday_routes,
            "missing_manifest_routes": missing_manifest_routes,
            "missing_scripts": missing_scripts,
            "failed_routes": failed_routes,
            "routes_verified_count": sum(1 for r in route_rows if r["ok"]),
            "routes_total_count": len(route_rows),
            "http_statuses": http_statuses,
            "routes": route_rows,
            "network_calls": network_calls,
            "subprocess_calls": subprocess_calls,
            "zero_inference_verified": True,
            "diagnostics": diagnostics,
            "compliance": {
                "personal_pc_only": True,
                "amadeus_corporate_prohibition": True,
                "financial_promise": FINANCIAL_PROMISE,
                "license": "AGPL-3.0",
                "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            },
            "exit_code": 0 if not drift_detected else 2,
            "summary": summary,
        }
    )


def _check_zero_grant_default(active_script: str, var_name: str) -> bool:
    """Verify deploy.sh keeps a grant/bonus variable strictly at 0 without non-zero defaults or overrides."""
    has_zero_binding = bool(
        re.search(rf'\$\{{{var_name}:-0(?:\.0+)?\}}|{var_name}=0(?:\.0+)?\b', active_script)
    )
    has_nonzero_default = bool(
        re.search(rf'\$\{{{var_name}:-[1-9][0-9.]*\}}', active_script)
        or re.search(rf'(?:^|[,\s"\']){var_name}=[1-9][0-9.]*\b', active_script, re.MULTILINE)
    )
    return has_zero_binding and not has_nonzero_default


def _audit_deploy_finops(deploy_script_path: Path, *, lang_code: str) -> dict[str, Any]:
    is_fr = lang_code == "fr"
    display_path = redact_home_path(deploy_script_path)
    script_present = False
    active_script = ""
    diagnostics: list[str] = []

    try:
        if deploy_script_path.is_file():
            raw_script = deploy_script_path.read_text(encoding="utf-8")
            script_present = True
            active_script = "\n".join(
                ln for ln in raw_script.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
            )
    except OSError:
        script_present = False

    if not script_present:
        diagnostics.append(
            f"Script de déploiement introuvable : `{display_path}`."
            if is_fr
            else f"Deploy script missing or unreadable: `{display_path}`."
        )

    conc_matches = re.findall(r"--concurrency(?:=|\s+)(\d+)\b", active_script)
    concurrency_ok = bool(script_present and conc_matches and all(val == "80" for val in conc_matches))

    zero_grants_ok = bool(
        script_present
        and _check_zero_grant_default(active_script, "HIVEMIND_STARTER_GRANT_HMT")
        and _check_zero_grant_default(active_script, "HIVEMIND_REGISTRATION_BONUS_HMT")
        and _check_zero_grant_default(active_script, "HIVEMIND_ALLOW_BONUSES")
        and _check_zero_grant_default(active_script, "HIVEMIND_PARTNER_TESTING_GRANTS")
    )

    specs: tuple[tuple[str, str, bool, str, str], ...] = (
        (
            "min_instances_zero",
            "--min-instances 0",
            bool(
                script_present
                and re.search(r"--min-instances(?:=|\s+)0\b", active_script)
                and not re.search(r"--min-instances(?:=|\s+)[1-9]\d*\b", active_script)
            ),
            "Scale-to-zero FinOps cap (--min-instances 0).",
            "Plafond FinOps scale-to-zero (--min-instances 0).",
        ),
        (
            "max_instances_one",
            "--max-instances 1",
            bool(
                script_present
                and re.search(r"--max-instances(?:=|\s+)1\b", active_script)
                and not re.search(r"--max-instances(?:=|\s+)(?:0|[2-9]|\d{2,})\b", active_script)
            ),
            "Single-writer SQLite + Litestream cap (--max-instances 1).",
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
            "Continuous CPU allocation (--no-cpu-throttling) for Litestream replication.",
            "CPU maintenu (--no-cpu-throttling) pour la réplication Litestream.",
        ),
        (
            "concurrency_80",
            "--concurrency 80",
            concurrency_ok,
            "Bounded request concurrency (--concurrency 80).",
            "Concurrence bornée (--concurrency 80).",
        ),
        (
            "litestream_replica_url",
            "HIVEMIND_ROUTER_REPLICA_URL",
            bool(script_present and "HIVEMIND_ROUTER_REPLICA_URL=" in active_script and "gcs://" in active_script),
            "Durable GCS Litestream ledger replication (HIVEMIND_ROUTER_REPLICA_URL).",
            "Réplication GCS Litestream durable (HIVEMIND_ROUTER_REPLICA_URL).",
        ),
        (
            "secret_manager_router_secret",
            "HIVEMIND_ROUTER_SECRET",
            bool(
                script_present
                and re.search(r"--set-secrets\s+[\"']?[^\"'\n]*HIVEMIND_ROUTER_SECRET=", active_script)
            ),
            "Secret Manager mounted HIVEMIND_ROUTER_SECRET.",
            "Secret HIVEMIND_ROUTER_SECRET monté depuis Secret Manager.",
        ),
        (
            "zero_starter_and_bonus_grants",
            "0 starter/bonus grants",
            zero_grants_ok,
            "Zero starter/registration/bonus HMT grants at deploy.",
            "Zéro crédit starter/bonus HMT au déploiement.",
        ),
    )

    invariants: list[dict[str, Any]] = []
    failed_invariants: list[str] = []
    for inv_id, expected_tok, passed, desc_en, desc_fr in specs:
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
                    f"Invariant FinOps manquant dans `{display_path}` : `{expected_tok}`."
                    if is_fr
                    else f"Missing FinOps invariant in `{display_path}`: `{expected_tok}`."
                )

    return {
        "deploy_script": display_path,
        "deploy_script_present": script_present,
        "finops_ok": bool(script_present and not failed_invariants),
        "invariants": invariants,
        "failed_invariants": failed_invariants,
        "diagnostics": diagnostics,
    }


def _load_build_app_from_server_py(server_py_path: Path) -> tuple[bool, Callable[..., Any] | None, list[str]]:
    """Inspect `deploy/gcp_openrouter/server.py` safely and load `build_app` if runtime deps exist."""
    issues: list[str] = []
    display_path = redact_home_path(server_py_path)
    if not server_py_path.is_file():
        return False, None, [f"Server entrypoint missing: `{display_path}`."]

    source = server_py_path.read_text(encoding="utf-8", errors="replace")
    try:
        import ast

        tree = ast.parse(source, filename=display_path)
    except SyntaxError as exc:
        return False, None, [f"SyntaxError in `{display_path}`: {exc.msg}."]

    has_build_app = any(
        isinstance(node, ast.FunctionDef) and node.name == "build_app" for node in ast.walk(tree)
    )
    if not has_build_app or "HIVEMIND_ROUTER_SECRET" not in source or "create_router_app" not in source:
        issues.append(f"`{display_path}` is missing required `build_app` / `HIVEMIND_ROUTER_SECRET` contract.")
        return False, None, issues

    try:
        importlib.import_module("fastapi")
        importlib.import_module("hivemind.router.gateway")
    except ImportError:
        return True, None, []

    env_keys = (
        "HIVEMIND_ROUTER_SECRET",
        "HIVEMIND_ROUTER_DB",
        "HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS",
        "HIVEMIND_OPEN_REGISTRATION",
        "HIVEMIND_STARTER_GRANT_HMT",
        "HIVEMIND_REGISTRATION_BONUS_HMT",
        "HIVEMIND_ALLOW_BONUSES",
    )
    with _SERVER_LOAD_LOCK:
        saved = {k: os.environ.get(k) for k in env_keys}
        try:
            os.environ["HIVEMIND_ROUTER_SECRET"] = secrets.token_hex(24)
            os.environ["HIVEMIND_ROUTER_DB"] = ":memory:"
            os.environ["HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS"] = "1"
            os.environ["HIVEMIND_OPEN_REGISTRATION"] = "1"
            os.environ["HIVEMIND_STARTER_GRANT_HMT"] = "0"
            os.environ["HIVEMIND_REGISTRATION_BONUS_HMT"] = "0"
            os.environ["HIVEMIND_ALLOW_BONUSES"] = "0"
            spec = importlib.util.spec_from_file_location("hivemind_prepromote_server", server_py_path)
            if spec is None or spec.loader is None:
                return False, None, [f"Unable to load module spec from `{display_path}`."]
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:
            return False, None, [f"Failed loading `{display_path}`: {type(exc).__name__}."]
        finally:
            for k, old_v in saved.items():
                if old_v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old_v

    build_fn = getattr(mod, "build_app", None)
    if not callable(build_fn):
        return False, None, [f"`{display_path}` has no callable `build_app`."]
    return True, build_fn, []


async def _asgi_get_request(app: Any, path: str) -> tuple[int, dict[str, str], bytes]:
    """Pure-stdlib ASGI 3.0 GET fallback when `fastapi.testclient` is absent."""
    status_code = 500
    resp_headers: dict[str, str] = {}
    chunks: list[bytes] = []
    sent_req = False

    async def _receive() -> dict[str, Any]:
        nonlocal sent_req
        if not sent_req:
            sent_req = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def _send(message: Mapping[str, Any]) -> None:
        nonlocal status_code
        mtype = message.get("type")
        if mtype == "http.response.start":
            status_code = int(message.get("status", 500))
            for raw_k, raw_v in message.get("headers") or ():
                resp_headers[bytes(raw_k).decode("latin1").lower()] = bytes(raw_v).decode("latin1")
        elif mtype == "http.response.body":
            chunks.append(bytes(message.get("body") or b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8000"), (b"x-forwarded-for", b"127.0.0.1")],
        "client": ("127.0.0.1", 8000),
        "server": ("127.0.0.1", 8000),
    }
    await app(scope, _receive, _send)
    return status_code, resp_headers, b"".join(chunks)


def _probe_asgi_pre_promote_routes(
    app: Any,
    *,
    lang_code: str,
) -> tuple[bool, dict[str, int], list[str], str]:
    """Run zero-inference GET sanity probes across core + holiday routes on an ASGI app."""
    is_fr = lang_code == "fr"
    statuses: dict[str, int] = {}
    issues: list[str] = []
    seen_nonces: set[str] = set()
    transport_used = "asgi_stdlib"

    def _inspect_probe_result(path: str, expected_kind: str, code: int, hdrs: dict[str, str], raw_bytes: bytes) -> None:
        statuses[path] = code
        if code != 200:
            issues.append(
                f"Pre-promote GET {path} -> HTTP {code}." if not is_fr else f"Pré-promotion GET {path} -> HTTP {code}."
            )
            return
        parsed_status, norm_body = _parse_http_raw_response(code, raw_bytes, hdrs)
        _ = parsed_status
        if path == "/health" and norm_body.get("status") != "ok":
            issues.append("Pre-promote GET /health missing `status == 'ok'`.")
        elif path == "/v1/hivemind/onboarding/manifest":
            ok_m, _sha, _rev, _mr, _ms, m_issues = _inspect_manifest_contract(norm_body, lang_code=lang_code)
            if not ok_m:
                issues.extend(m_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            ok_f, _fb, f_issues = _inspect_feedback_summary_contract(norm_body, lang_code=lang_code)
            if not ok_f:
                issues.extend(f_issues)
        elif expected_kind == "html":
            ok_h, _hd, h_issues = _inspect_holiday_html_contract(
                path, norm_body, seen_nonces, strict_contract=True, lang_code=lang_code
            )
            if not ok_h:
                issues.extend(h_issues)

    testclient_cls: Any = None
    try:
        testclient_cls = getattr(importlib.import_module("fastapi.testclient"), "TestClient", None)
    except ImportError:
        testclient_cls = None

    if testclient_cls is not None:
        transport_used = "fastapi_testclient"
        with testclient_cls(app) as client:
            for path, expected_kind in PRE_PROMOTE_PROBE_ROUTES:
                _assert_zero_inference_request("GET", f"http://127.0.0.1:8000{path}")
                resp = client.get(path, headers={"X-Forwarded-For": "127.0.0.1"})
                hdrs = {str(k).lower(): str(v) for k, v in resp.headers.items()}
                _inspect_probe_result(path, expected_kind, resp.status_code, hdrs, resp.content)
    else:
        for path, expected_kind in PRE_PROMOTE_PROBE_ROUTES:
            _assert_zero_inference_request("GET", f"http://127.0.0.1:8000{path}")
            code, hdrs, raw_bytes = asyncio.run(_asgi_get_request(app, path))
            _inspect_probe_result(path, expected_kind, code, hdrs, raw_bytes)

    return len(issues) == 0, statuses, issues, transport_used


def verify_pre_promote_gateway_and_finops(
    *,
    deploy_script_path: Path | str | None = None,
    server_py_path: Path | str | None = None,
    app_factory: Callable[..., Any] | None = None,
    run_app_probe: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Audit `deploy/gcp_openrouter/deploy.sh` FinOps invariants and run optional ASGI pre-promote sanity probe."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("pre_promote_check", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    deploy_path = Path(deploy_script_path) if deploy_script_path is not None else DEFAULT_DEPLOY_SCRIPT_PATH
    server_path = Path(server_py_path) if server_py_path is not None else DEFAULT_SERVER_PY_PATH

    finops = _audit_deploy_finops(deploy_path, lang_code=lang_code)
    diagnostics = list(finops["diagnostics"])

    server_static_ok, build_app_fn, server_issues = _load_build_app_from_server_py(server_path)
    diagnostics.extend(server_issues)

    app_probe_executed = False
    app_probe_ok = server_static_ok
    app_probe_statuses: dict[str, int] = {}
    app_probe_transport = "none"

    if run_app_probe:
        factory = app_factory if app_factory is not None else build_app_fn
        if factory is not None:
            probe_env = {
                "HIVEMIND_ROUTER_SECRET": secrets.token_hex(24),
                "HIVEMIND_ROUTER_DB": ":memory:",
                "HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS": "1",
                "HIVEMIND_OPEN_REGISTRATION": "1",
                "HIVEMIND_STARTER_GRANT_HMT": "0",
                "HIVEMIND_REGISTRATION_BONUS_HMT": "0",
                "HIVEMIND_ALLOW_BONUSES": "0",
                "HIVEMIND_BUILD_SHA": sanitize_build_sha((env or os.environ).get("HIVEMIND_BUILD_SHA")),
            }
            try:
                sig = inspect.signature(factory)
                req_pos = [
                    p
                    for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and p.default is inspect.Parameter.empty
                ]
                app = factory(probe_env) if req_pos else factory()
                if app is None:
                    raise RevisionPromoterError("app_factory returned None.")
                app_probe_executed = True
                app_probe_ok, app_probe_statuses, probe_issues, app_probe_transport = _probe_asgi_pre_promote_routes(
                    app, lang_code=lang_code
                )
                diagnostics.extend(probe_issues)
            except Exception as exc:
                app_probe_executed = True
                app_probe_ok = False
                diagnostics.append(
                    f"Échec sonde pré-promotion ASGI ({type(exc).__name__})."
                    if is_fr
                    else f"Pre-promote ASGI sanity probe failed ({type(exc).__name__})."
                )
        elif app_factory is None and server_static_ok:
            app_probe_transport = "skipped_runtime_unavailable"
            app_probe_ok = True

    pre_promote_ok = bool(finops["finops_ok"] and server_static_ok and app_probe_ok)
    summary = (
        (
            f"Vérification pré-promotion & FinOps OK (finops={finops['finops_ok']}, asgi_probe={app_probe_executed})."
            if is_fr
            else f"Pre-promote FinOps & gateway check passed (finops={finops['finops_ok']}, asgi_probe={app_probe_executed})."
        )
        if pre_promote_ok
        else (
            f"Échec pré-promotion / FinOps : {', '.join(diagnostics[:3])}"
            if is_fr
            else f"Pre-promote / FinOps verification failed: {', '.join(diagnostics[:3])}"
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "pre_promote_check",
            "lang": lang_code,
            "compliance_ok": True,
            "pre_promote_ok": pre_promote_ok,
            "finops_ok": finops["finops_ok"],
            "deploy_script": finops["deploy_script"],
            "deploy_script_present": finops["deploy_script_present"],
            "invariants": finops["invariants"],
            "failed_invariants": finops["failed_invariants"],
            "server_py": redact_home_path(server_path),
            "server_py_ok": server_static_ok,
            "app_probe_executed": app_probe_executed,
            "app_probe_ok": app_probe_ok,
            "app_probe_transport": app_probe_transport,
            "app_probe_statuses": app_probe_statuses,
            "network_calls": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "diagnostics": diagnostics,
            "compliance": {
                "personal_pc_only": True,
                "amadeus_corporate_prohibition": True,
                "financial_promise": FINANCIAL_PROMISE,
                "license": "AGPL-3.0",
                "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
            },
            "exit_code": 0 if pre_promote_ok else 2,
            "summary": summary,
        }
    )


def _inspect_local_lockfile(lock_file_path: Path | str | None) -> tuple[bool, dict[str, Any]]:
    """Inspect optional local training lockfile (JSON payload or exclusive flock)."""
    if lock_file_path is None:
        return False, {"checked": False, "exists": False, "active": False}
    lp = Path(lock_file_path)
    if not lp.exists():
        return False, {"checked": True, "path": redact_home_path(lp), "exists": False, "active": False}

    flock_held = False
    try:
        with lp.open("r", encoding="utf-8", errors="replace") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                flock_held = True
            raw = fh.read(4096).strip()
    except OSError:
        return True, {"checked": True, "path": redact_home_path(lp), "exists": True, "active": True, "reason": "unreadable_lockfile"}

    active = True
    if raw.startswith("{"):
        try:
            doc = json.loads(raw)
            if isinstance(doc, Mapping) and (doc.get("active") is False or doc.get("released") is True or doc.get("training_active") is False):
                active = flock_held
        except json.JSONDecodeError:
            active = True

    return active, {
        "checked": True,
        "path": redact_home_path(lp),
        "exists": True,
        "flock_held": flock_held,
        "active": active,
    }


def _inspect_local_training_processes(process_probe_fn: Callable[[], Iterable[str]] | None) -> tuple[bool, list[str]]:
    if process_probe_fn is None:
        return False, []
    matched: list[str] = []
    for raw_proc in process_probe_fn() or ():
        name = str(raw_proc or "").strip()
        low = name.lower()
        if any(marker in low for marker in _TRAINING_WORKER_MARKERS):
            matched.append(sanitize_text_for_display(name)[:80])
    return len(matched) > 0, matched


def check_active_training_safety_lock(
    gateway: str = DEFAULT_GATEWAY,
    *,
    lock_file_path: Path | str | None = None,
    process_probe_fn: Callable[[], Iterable[str]] | None = None,
    http_fn: HttpFn = default_http_request,
    allow_during_training: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Inspect read-only T4/raid/fleet endpoints and optional local lock/worker state before live traffic shift."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("training_safety_lock", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    active_reasons: list[str] = []
    network_calls = 0

    network_calls += 1
    t4_code, t4_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/t4/status")
    network_calls += 1
    raid_code, raid_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/raid/status")
    network_calls += 1
    fleet_code, fleet_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/producers/fleet")

    if t4_code == 200:
        op_mode = str(t4_body.get("operational_mode") or "")
        gov = t4_body.get("governor") if isinstance(t4_body.get("governor"), Mapping) else {}
        boinc_mode = str(gov.get("boinc_gpu_mode") or "never").strip().lower()
        hold_present = bool(gov.get("hold_file_present", False))
        gov_state = str(gov.get("state") or "").strip().lower()
        qlora = t4_body.get("qlora_progress") if isinstance(t4_body.get("qlora_progress"), Mapping) else {}
        hw = t4_body.get("hardware") if isinstance(t4_body.get("hardware"), Mapping) else {}
        res_procs = hw.get("resident_processes") if isinstance(hw.get("resident_processes"), list) else []

        if op_mode == "EUROPEAN_NIGHT_QLORA_MOE_TRAINING":
            active_reasons.append("t4_status:EUROPEAN_NIGHT_QLORA_MOE_TRAINING")
        if boinc_mode == "always" and not hold_present:
            active_reasons.append("t4_status:boinc_gpu_mode=always")
        if gov_state in ("training", "qlora_training", "boinc_training", "active_training") or gov_state.startswith("training_"):
            active_reasons.append(f"t4_status:governor_state={gov_state}")
        if t4_body.get("training_active") is True or qlora.get("training_active") is True or qlora.get("active_round") is True:
            active_reasons.append("t4_status:qlora_training_active")
        for proc in res_procs:
            pname = str(proc.get("name") if isinstance(proc, Mapping) else proc).lower()
            if any(tok in pname for tok in ("qlora", "opentrainer", "boinc_client", "hivemind_qlora")):
                active_reasons.append("t4_status:resident_training_process")
                break

    if raid_code == 200:
        swarm = raid_body.get("swarm") if isinstance(raid_body.get("swarm"), Mapping) else {}
        raw_trainers = swarm.get("active_trainers")
        if isinstance(raw_trainers, int) and not isinstance(raw_trainers, bool) and raw_trainers > 0:
            active_reasons.append(f"raid_status:active_trainers={raw_trainers}")
        if raid_body.get("training_active") is True:
            active_reasons.append("raid_status:training_active")

    if fleet_code == 200:
        if fleet_body.get("training_active") is True:
            active_reasons.append("fleet:training_active")
        for row in fleet_body.get("hiveminders") or ():
            if not isinstance(row, Mapping) or row.get("status") != "live":
                continue
            exec_cls = str(row.get("execution_class") or "").lower()
            if row.get("training_active") is True or any(
                tok in exec_cls for tok in ("qlora", "opentrainer", "boinc", "diloco", "train")
            ):
                active_reasons.append("fleet:live_training_producer")
                break

    lockfile_active, lockfile_state = _inspect_local_lockfile(lock_file_path)
    if lockfile_active:
        active_reasons.append("local_lockfile:active")

    proc_active, matched_procs = _inspect_local_training_processes(process_probe_fn)
    if proc_active:
        active_reasons.append(f"local_process:{matched_procs[0]}")

    training_active = len(active_reasons) > 0
    safety_lock_engaged = bool(training_active and not allow_during_training)
    live_traffic_shift_safe = bool((not training_active) or allow_during_training)

    summary = (
        (
            f"Verrou entraînement actif engagé ({', '.join(active_reasons)}) : bascule trafic live différée/bloquée."
            if is_fr
            else f"Active-training safety lock engaged ({', '.join(active_reasons)}): live traffic shift unsafe."
        )
        if safety_lock_engaged
        else (
            "Aucun round d'entraînement bloquant détecté ; bascule Cloud Run sûre."
            if is_fr
            else "No blocking active training rounds detected; Cloud Run traffic promotion safe."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "training_safety_lock",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "training_active": training_active,
            "allow_during_training": bool(allow_during_training),
            "safety_lock_engaged": safety_lock_engaged,
            "live_traffic_shift_safe": live_traffic_shift_safe,
            "no_traffic_staging_safe": True,
            "active_reasons": active_reasons,
            "endpoints": {
                "/v1/hivemind/t4/status": t4_code,
                "/v1/hivemind/raid/status": raid_code,
                "/v1/hivemind/producers/fleet": fleet_code,
            },
            "lockfile": lockfile_state,
            "matched_training_processes": matched_procs,
            "network_calls": network_calls,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "exit_code": 0 if live_traffic_shift_safe else 3,
            "summary": summary,
        }
    )


def _format_cmd_display(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in argv)


def build_idempotent_rollout_plan(
    *,
    gateway: str = DEFAULT_GATEWAY,
    target_sha: str | None = None,
    previous_revision: str | None = None,
    candidate_revision: str | None = None,
    candidate_tag: str = DEFAULT_CANDIDATE_TAG,
    service_name: str = DEFAULT_CLOUD_RUN_SERVICE,
    region: str = DEFAULT_CLOUD_RUN_REGION,
    gcp_project: str | None = None,
    source_dir: str = DEFAULT_DEPLOY_SOURCE_DIR,
    no_traffic_only: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Sanitize all Cloud Run identifiers and construct list-argv gcloud rollout & rollback commands."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("rollout_plan", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    clean_service = validate_cloud_identifier(service_name, "service_name")
    clean_region = validate_cloud_identifier(region, "region")
    clean_tag = validate_candidate_tag(candidate_tag)
    clean_project = validate_gcp_project(gcp_project if gcp_project is not None else (env or {}).get("GCP_PROJECT"))
    clean_source = validate_deploy_source_dir(source_dir)

    clean_sha: str | None = None
    if target_sha is not None and str(target_sha).strip():
        clean_sha = validate_target_sha(target_sha)

    clean_prev_rev: str | None = None
    if previous_revision is not None and str(previous_revision).strip():
        clean_prev_rev = sanitize_cloud_run_revision(previous_revision)
        if clean_prev_rev is None:
            raise RevisionPromoterError(f"Invalid previous Cloud Run revision: {previous_revision!r}.")

    clean_cand_rev: str | None = None
    if candidate_revision is not None and str(candidate_revision).strip():
        clean_cand_rev = sanitize_cloud_run_revision(candidate_revision)
        if clean_cand_rev is None:
            raise RevisionPromoterError(f"Invalid candidate Cloud Run revision: {candidate_revision!r}.")

    proj_args = ["--project", clean_project] if clean_project else []
    env_pairs = [
        "HIVEMIND_STARTER_GRANT_HMT=0",
        "HIVEMIND_REGISTRATION_BONUS_HMT=0",
        "HIVEMIND_ALLOW_BONUSES=0",
        "HIVEMIND_PARTNER_TESTING_GRANTS=0",
    ]
    if clean_sha is not None:
        env_pairs.insert(0, f"HIVEMIND_BUILD_SHA={clean_sha}")

    deploy_no_traffic_argv = [
        "gcloud",
        "run",
        "deploy",
        clean_service,
        *proj_args,
        "--source",
        clean_source,
        "--region",
        clean_region,
        "--no-traffic",
        "--tag",
        clean_tag,
        "--min-instances",
        "0",
        "--max-instances",
        "1",
        "--no-cpu-throttling",
        "--concurrency",
        "80",
        "--update-env-vars",
        ",".join(env_pairs),
        "--format",
        "json",
    ]

    promo_rev_token = clean_cand_rev or f"{clean_service}-{clean_tag}"
    promote_traffic_argv = [
        "gcloud",
        "run",
        "services",
        "update-traffic",
        clean_service,
        *proj_args,
        "--region",
        clean_region,
        "--to-revisions",
        f"{promo_rev_token}=100",
    ]

    rb_rev_token = clean_prev_rev or "PREVIOUS_K_REVISION"
    rollback_traffic_argv = [
        "gcloud",
        "run",
        "services",
        "update-traffic",
        clean_service,
        *proj_args,
        "--region",
        clean_region,
        "--to-revisions",
        f"{rb_rev_token}=100",
    ]

    planned_steps = (
        [deploy_no_traffic_argv]
        if no_traffic_only
        else [deploy_no_traffic_argv, promote_traffic_argv]
    )

    summary = (
        f"Plan de déploiement idempotent prêt ({len(planned_steps)} commande(s) gcloud dry-run, 0 exécution)."
        if is_fr
        else f"Idempotent Cloud Run rollout plan ready ({len(planned_steps)} dry-run gcloud command(s), 0 executed)."
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "rollout_plan",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "plan_ready": True,
            "service_name": clean_service,
            "region": clean_region,
            "gcp_project": clean_project,
            "candidate_tag": clean_tag,
            "target_build_sha": clean_sha,
            "previous_revision": clean_prev_rev,
            "candidate_revision": clean_cand_rev,
            "no_traffic_only": bool(no_traffic_only),
            "deploy_no_traffic_argv": deploy_no_traffic_argv,
            "promote_traffic_argv": promote_traffic_argv,
            "rollback_traffic_argv": rollback_traffic_argv,
            "rollback_revision_pinned": clean_prev_rev is not None,
            "commands": planned_steps,
            "commands_display": [_format_cmd_display(cmd) for cmd in planned_steps],
            "rollback_display": _format_cmd_display(rollback_traffic_argv),
            "shell_false_enforced": True,
            "dry_run_only": True,
            "gcloud_executed": False,
            "gcloud_calls": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
            "zero_inference_verified": True,
            "exit_code": 0,
            "summary": summary,
        }
    )


def run_post_deploy_holiday_canary(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    probe_feedback_post: bool = False,
    http_fn: HttpFn = default_http_request,
    lang: str = "en",
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """Verify live post-deploy /noel, /quick, manifest, feedback summary, and optional Idempotency-Key POST."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("post_deploy_canary", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = (
        validate_target_sha(expected_sha)
        if expected_sha is not None and str(expected_sha).strip()
        else None
    )

    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    http_statuses: dict[str, int] = {}
    diagnostics: list[str] = []
    network_calls = 0

    health_ok = False
    manifest_ok = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {
        "schema_valid": False,
        "total_responses": None,
        "mean_ease_rating": None,
        "zero_game_lag_ratio": None,
        "by_profile": None,
    }

    for path, expected_kind in CANARY_PROBE_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        if code != 200:
            diagnostics.append(
                f"Canari GET {path} en échec (HTTP {code})."
                if is_fr
                else f"Canary GET {path} failed (HTTP {code})."
            )
            continue

        if path == "/health":
            health_ok = body.get("status") == "ok"
            if not health_ok:
                diagnostics.append("Canary GET /health returned 200 without `status == 'ok'`.")
        elif path == "/v1/hivemind/onboarding/manifest":
            ok_m, live_build_sha, cloud_run_revision, _mr, _ms, m_issues = _inspect_manifest_contract(
                body, lang_code=lang_code
            )
            manifest_ok = ok_m
            diagnostics.extend(m_issues)
        elif expected_kind == "html":
            ok_h, h_doc, h_issues = _inspect_holiday_html_contract(
                path, body, seen_nonces, strict_contract=True, lang_code=lang_code
            )
            html_contracts[path] = h_doc
            diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            fb_ok, fb_doc, fb_issues = _inspect_feedback_summary_contract(body, lang_code=lang_code)
            feedback_summary_ok = fb_ok
            feedback_summary_doc = fb_doc
            diagnostics.extend(fb_issues)

    sha_ok = True
    if clean_expected_sha is not None:
        sha_ok = shas_match(live_build_sha, clean_expected_sha)
        if not sha_ok:
            diagnostics.append(
                f"Canari SHA incorrect : direct='{live_build_sha}' != attendu '{clean_expected_sha}'."
                if is_fr
                else f"Canary build SHA mismatch: live='{live_build_sha}' != expected '{clean_expected_sha}'."
            )

    html_all_ok = (
        len(html_contracts) == 2
        and all(html_contracts.get(p, {}).get("ok") is True for p in ("/noel", "/quick"))
        and len(seen_nonces) == 2
    )

    feedback_post_ok: bool | None = None
    feedback_post_http_status: int | None = None
    idem_key: str | None = None
    if probe_feedback_post:
        idem_seed = f"noel-cloud-run-promoter-canary/v1|{clean_gw}|{clean_expected_sha or 'live'}"
        idem_key = "canary-noel-" + hashlib.sha256(idem_seed.encode("utf-8")).hexdigest()[:32]
        canary_payload = {
            "profile": "dev",
            "ease_rating": 5,
            "zero_game_lag": True,
            "platform": "linux",
            "note": "",
        }
        network_calls += 1
        post_code, post_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/onboarding/feedback",
            payload=canary_payload,
            headers={"Idempotency-Key": idem_key},
            allow_feedback_post=True,
        )
        feedback_post_http_status = post_code
        post_valid, post_sum, _ = (
            _inspect_feedback_summary_contract(post_body, lang_code=lang_code)
            if post_code in (200, 201)
            else (False, feedback_summary_doc, [])
        )
        if post_code in (200, 201) and post_valid and (post_sum.get("total_responses") or 0) >= 1:
            network_calls += 1
            ref_code, ref_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/feedback/summary")
            http_statuses["/v1/hivemind/onboarding/feedback/summary#after_post"] = ref_code
            ref_ok, ref_sum, ref_issues = (
                _inspect_feedback_summary_contract(ref_body, lang_code=lang_code)
                if ref_code == 200
                else (False, post_sum, ["Canary GET feedback summary after POST failed."])
            )
            feedback_post_ok = bool(ref_code == 200 and ref_ok and (ref_sum.get("total_responses") or 0) >= 1)
            if feedback_post_ok:
                feedback_summary_ok = True
                feedback_summary_doc = ref_sum
            else:
                diagnostics.extend(ref_issues)
        else:
            feedback_post_ok = False
            diagnostics.append(
                f"Canari POST /v1/hivemind/onboarding/feedback en échec (HTTP {post_code})."
                if is_fr
                else f"Canary POST /v1/hivemind/onboarding/feedback failed (HTTP {post_code})."
            )

    canary_ok = bool(
        health_ok
        and manifest_ok
        and sha_ok
        and html_all_ok
        and feedback_summary_ok
        and (not probe_feedback_post or feedback_post_ok is True)
    )
    summary = (
        (
            f"Canari post-déploiement Cloud Run VERT sur {clean_gw} (nonces CSP uniques={len(seen_nonces)})."
            if is_fr
            else f"Post-deploy Cloud Run holiday canary PASSED on {clean_gw} (unique CSP nonces={len(seen_nonces)})."
        )
        if canary_ok
        else (
            f"Canari post-déploiement Cloud Run EN ÉCHEC sur {clean_gw} : rollback requis."
            if is_fr
            else f"Post-deploy Cloud Run holiday canary FAILED on {clean_gw}: rollback required."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "post_deploy_canary",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "canary_ok": canary_ok,
            "rollback_required": not canary_ok,
            "health_ok": health_ok,
            "manifest_ok": manifest_ok,
            "sha_ok": sha_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "cloud_run_revision": cloud_run_revision,
            "html_contract_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "feedback_post_probed": bool(probe_feedback_post),
            "feedback_post_ok": feedback_post_ok,
            "feedback_post_http_status": feedback_post_http_status,
            "feedback_post_idempotency_key": idem_key,
            "http_statuses": http_statuses,
            "network_calls": network_calls,
            "zero_inference_verified": True,
            "diagnostics": diagnostics,
            "exit_code": 0 if canary_ok else 2,
            "summary": summary,
        }
    )


def _extract_deployed_revision(gcloud_output: str, fallback: str) -> str:
    """Parse newly deployed revision name from `gcloud run deploy --format json` or CLI output."""
    text = (gcloud_output or "").strip()
    if text.startswith("{"):
        try:
            doc = json.loads(text)
            if isinstance(doc, Mapping):
                status_blk = doc.get("status") if isinstance(doc.get("status"), Mapping) else {}
                for candidate in (
                    status_blk.get("latestCreatedRevisionName"),
                    doc.get("latestCreatedRevisionName"),
                    doc.get("candidate_revision"),
                    doc.get("revision"),
                    doc.get("revision_name"),
                ):
                    rev = sanitize_cloud_run_revision(candidate)
                    if rev is not None:
                        return rev
        except json.JSONDecodeError:
            pass
    m = re.search(r"revision\s*\[([A-Za-z0-9._-]+)\]", text, re.IGNORECASE)
    if m:
        rev = sanitize_cloud_run_revision(m.group(1))
        if rev is not None:
            return rev
    return fallback


def promote_holiday_revision(
    gateway: str = DEFAULT_GATEWAY,
    *,
    target_sha: str | None = None,
    previous_revision: str | None = None,
    candidate_tag: str = DEFAULT_CANDIDATE_TAG,
    service_name: str = DEFAULT_CLOUD_RUN_SERVICE,
    region: str = DEFAULT_CLOUD_RUN_REGION,
    gcp_project: str | None = None,
    deploy_script_path: Path | str | None = None,
    server_py_path: Path | str | None = None,
    app_factory: Callable[..., Any] | None = None,
    lock_file_path: Path | str | None = None,
    process_probe_fn: Callable[[], Iterable[str]] | None = None,
    check_git_head: bool = False,
    execute: bool = False,
    no_traffic_only: bool = False,
    allow_during_training: bool = False,
    defer_on_active_training: bool = False,
    probe_feedback_post: bool = False,
    gcloud_runner: RunnerFn = default_gcloud_runner,
    http_fn: HttpFn = default_http_request,
    git_runner: RunnerFn = default_git_runner,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Orchestrate compliance -> pre-promote -> drift -> safety lock -> idempotent rollout -> canary/rollback."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    # 1. Strictly local compliance gate BEFORE any network call, pre-promote check, or rollout command generation
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("revision_promoter", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    clean_tag = validate_candidate_tag(candidate_tag)
    clean_service = validate_cloud_identifier(service_name, "service_name")
    clean_region = validate_cloud_identifier(region, "region")
    clean_project = validate_gcp_project(gcp_project if gcp_project is not None else (env or {}).get("GCP_PROJECT"))

    # 2. Pre-promote FinOps & zero-inference FastAPI sanity check
    pre_promote = verify_pre_promote_gateway_and_finops(
        deploy_script_path=deploy_script_path,
        server_py_path=server_py_path,
        app_factory=app_factory,
        env=env,
        cwd=cwd,
        lang=lang_code,
    )
    if not pre_promote["pre_promote_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "revision_promoter",
                "lang": lang_code,
                "gateway": clean_gw,
                "verdict": "pre_promote_failed",
                "compliance_ok": True,
                "pre_promote_ok": False,
                "finops_ok": pre_promote["finops_ok"],
                "drift_detected": False,
                "idempotent_noop": False,
                "already_at_target_revision": False,
                "promotion_deferred_training_active": False,
                "staged_no_traffic_only": False,
                "promoted": False,
                "canary_ok": False,
                "rollback_triggered": False,
                "dry_run_only": not execute,
                "gcloud_executed": False,
                "gcloud_calls": 0,
                "cloud_state_mutated": False,
                "network_calls": 0,
                "zero_inference_verified": True,
                "pre_promote": pre_promote,
                "commands": [],
                "diagnostics": list(pre_promote.get("diagnostics") or []),
                "compliance": comp_doc,
                "exit_code": 2,
                "summary": pre_promote["summary"],
            }
        )

    # 3. Zero-inference revision drift detection
    drift = detect_revision_drift(
        clean_gw,
        expected_sha=target_sha,
        check_git_head=check_git_head,
        git_runner=git_runner,
        http_fn=http_fn,
        lang=lang_code,
        env=env,
        cwd=cwd,
    )
    network_calls = int(drift.get("network_calls", 0))
    resolved_sha = drift.get("expected_build_sha")
    effective_prev_rev = (
        sanitize_cloud_run_revision(previous_revision)
        if (previous_revision is not None and str(previous_revision).strip())
        else drift.get("cloud_run_revision")
    )
    if previous_revision is not None and str(previous_revision).strip() and effective_prev_rev is None:
        raise RevisionPromoterError(f"Invalid previous Cloud Run revision: {previous_revision!r}.")

    # 4. Idempotent short-circuit when live build_sha == target_sha and all holiday/feedback routes are healthy
    already_at_target = bool(
        drift.get("sha_check_requested")
        and drift.get("sha_match") is True
        and not drift.get("drift_detected")
    )
    if already_at_target:
        canary_verify = run_post_deploy_holiday_canary(
            clean_gw,
            expected_sha=resolved_sha,
            probe_feedback_post=probe_feedback_post,
            http_fn=http_fn,
            lang=lang_code,
            env=env,
            cwd=cwd,
        )
        network_calls += int(canary_verify.get("network_calls", 0))
        if canary_verify["canary_ok"]:
            summary = (
                f"No-op idempotent : {clean_gw} sert déjà le SHA cible `{resolved_sha}` "
                "(0 mutation gcloud)."
                if is_fr
                else f"Idempotent no-op: {clean_gw} is already serving target SHA `{resolved_sha}` "
                "with all holiday/feedback routes healthy (0 gcloud mutations)."
            )
            return _attach_receipt(
                {
                    "schema": SCHEMA,
                    "object": RECEIPT_OBJECT,
                    "campaign_id": CAMPAIGN_ID,
                    "mode": "revision_promoter",
                    "lang": lang_code,
                    "gateway": clean_gw,
                    "verdict": "idempotent_noop",
                    "compliance_ok": True,
                    "pre_promote_ok": True,
                    "finops_ok": True,
                    "drift_detected": False,
                    "idempotent_noop": True,
                    "already_at_target_revision": True,
                    "promotion_deferred_training_active": False,
                    "staged_no_traffic_only": False,
                    "promoted": False,
                    "canary_ok": True,
                    "rollback_triggered": False,
                    "dry_run_only": not execute,
                    "gcloud_executed": False,
                    "gcloud_calls": 0,
                    "cloud_state_mutated": False,
                    "network_calls": network_calls,
                    "zero_inference_verified": True,
                    "target_build_sha": resolved_sha,
                    "live_build_sha": drift.get("live_build_sha"),
                    "cloud_run_revision": drift.get("cloud_run_revision"),
                    "previous_revision": effective_prev_rev,
                    "pre_promote": pre_promote,
                    "drift": drift,
                    "canary": canary_verify,
                    "commands": [],
                    "executed_commands": [],
                    "diagnostics": [],
                    "compliance": comp_doc,
                    "exit_code": 0,
                    "summary": summary,
                }
            )

    # 5. Active-training safety lock check
    safety_lock = check_active_training_safety_lock(
        clean_gw,
        lock_file_path=lock_file_path,
        process_probe_fn=process_probe_fn,
        http_fn=http_fn,
        allow_during_training=allow_during_training,
        env=env,
        cwd=cwd,
        lang=lang_code,
    )
    network_calls += int(safety_lock.get("network_calls", 0))

    if safety_lock["training_active"] and not allow_during_training and not no_traffic_only:
        deferred = bool(defer_on_active_training)
        exit_code = 3 if deferred else 2
        verdict = "deferred_active_training" if deferred else "blocked_active_training"
        msg = (
            f"Promotion trafic différée (verrou entraînement actif : {', '.join(safety_lock['active_reasons'])} ; "
            "utilisez --no-traffic-only ou attendez la fin de la ronde)."
            if is_fr
            else f"Live traffic promotion deferred (active training lock: {', '.join(safety_lock['active_reasons'])}; "
            "use --no-traffic-only to stage safely without shifting traffic)."
        )
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "revision_promoter",
                "lang": lang_code,
                "gateway": clean_gw,
                "verdict": verdict,
                "compliance_ok": True,
                "pre_promote_ok": True,
                "finops_ok": True,
                "drift_detected": bool(drift.get("drift_detected")),
                "idempotent_noop": False,
                "already_at_target_revision": False,
                "promotion_deferred_training_active": deferred,
                "safety_lock_blocked": not deferred,
                "staged_no_traffic_only": False,
                "promoted": False,
                "canary_ok": False,
                "rollback_triggered": False,
                "dry_run_only": not execute,
                "gcloud_executed": False,
                "gcloud_calls": 0,
                "cloud_state_mutated": False,
                "network_calls": network_calls,
                "zero_inference_verified": True,
                "target_build_sha": resolved_sha,
                "previous_revision": effective_prev_rev,
                "pre_promote": pre_promote,
                "drift": drift,
                "safety_lock": safety_lock,
                "commands": [],
                "executed_commands": [],
                "diagnostics": [msg],
                "compliance": comp_doc,
                "exit_code": exit_code,
                "summary": msg,
            }
        )

    # 6. Build sanitized list-argv gcloud rollout plan
    rollout_plan = build_idempotent_rollout_plan(
        gateway=clean_gw,
        target_sha=resolved_sha,
        previous_revision=effective_prev_rev,
        candidate_tag=clean_tag,
        service_name=clean_service,
        region=clean_region,
        gcp_project=clean_project,
        no_traffic_only=no_traffic_only,
        env=env,
        cwd=cwd,
        lang=lang_code,
    )

    if not execute:
        summary = (
            f"Dry-run terminé : {len(rollout_plan['commands'])} commande(s) gcloud planifiée(s) "
            f"(dérive={drift['drift_detected']}, 0 mutation Cloud Run)."
            if is_fr
            else f"Dry-run complete: {len(rollout_plan['commands'])} gcloud command(s) planned "
            f"(drift={drift['drift_detected']}, 0 Cloud Run mutations)."
        )
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "revision_promoter",
                "lang": lang_code,
                "gateway": clean_gw,
                "verdict": "dry_run_plan_ready",
                "compliance_ok": True,
                "pre_promote_ok": True,
                "finops_ok": True,
                "drift_detected": bool(drift.get("drift_detected")),
                "idempotent_noop": False,
                "already_at_target_revision": False,
                "promotion_deferred_training_active": False,
                "staged_no_traffic_only": False,
                "promoted": False,
                "canary_ok": False,
                "rollback_triggered": False,
                "plan_ready": True,
                "dry_run_only": True,
                "gcloud_executed": False,
                "gcloud_calls": 0,
                "cloud_state_mutated": False,
                "network_calls": network_calls,
                "zero_inference_verified": True,
                "target_build_sha": resolved_sha,
                "live_build_sha": drift.get("live_build_sha"),
                "previous_revision": effective_prev_rev,
                "pre_promote": pre_promote,
                "drift": drift,
                "safety_lock": safety_lock,
                "rollout_plan": rollout_plan,
                "commands": rollout_plan["commands"],
                "executed_commands": [],
                "diagnostics": list(drift.get("diagnostics") or []),
                "compliance": comp_doc,
                "exit_code": 0,
                "summary": summary,
            }
        )

    # 7. Execute Step A: --no-traffic --tag deployment
    executed_commands: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    deploy_argv = list(rollout_plan["deploy_no_traffic_argv"])
    deploy_rc, deploy_out = _invoke_runner(gcloud_runner, deploy_argv, GCLOUD_DEPLOY_TIMEOUT_S)
    executed_commands.append(
        {
            "step": "deploy_no_traffic",
            "argv": deploy_argv,
            "returncode": deploy_rc,
            "output_preview": sanitize_text_for_display(deploy_out)[:240],
        }
    )
    if deploy_rc != 0:
        msg = (
            f"Échec du déploiement Cloud Run `--no-traffic` (code={deploy_rc})."
            if is_fr
            else f"Cloud Run `--no-traffic` deploy failed (rc={deploy_rc})."
        )
        diagnostics.append(msg)
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "revision_promoter",
                "lang": lang_code,
                "gateway": clean_gw,
                "verdict": "deploy_failed",
                "compliance_ok": True,
                "pre_promote_ok": True,
                "finops_ok": True,
                "drift_detected": bool(drift.get("drift_detected")),
                "idempotent_noop": False,
                "already_at_target_revision": False,
                "promotion_deferred_training_active": False,
                "staged_no_traffic_only": False,
                "promoted": False,
                "canary_ok": False,
                "rollback_triggered": False,
                "dry_run_only": False,
                "gcloud_executed": True,
                "gcloud_calls": len(executed_commands),
                "cloud_state_mutated": False,
                "network_calls": network_calls,
                "zero_inference_verified": True,
                "target_build_sha": resolved_sha,
                "previous_revision": effective_prev_rev,
                "pre_promote": pre_promote,
                "drift": drift,
                "safety_lock": safety_lock,
                "rollout_plan": rollout_plan,
                "executed_commands": executed_commands,
                "diagnostics": diagnostics,
                "compliance": comp_doc,
                "exit_code": 2,
                "summary": msg,
            }
        )

    candidate_rev = _extract_deployed_revision(deploy_out, fallback=f"{clean_service}-{clean_tag}")

    if no_traffic_only:
        summary = (
            f"Révision `{candidate_rev}` déployée avec `--no-traffic --tag {clean_tag}` (0 % trafic live déplacé)."
            if is_fr
            else f"Staged revision `{candidate_rev}` with `--no-traffic --tag {clean_tag}` (0% live traffic shifted)."
        )
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "revision_promoter",
                "lang": lang_code,
                "gateway": clean_gw,
                "verdict": "staged_no_traffic_only",
                "compliance_ok": True,
                "pre_promote_ok": True,
                "finops_ok": True,
                "drift_detected": bool(drift.get("drift_detected")),
                "idempotent_noop": False,
                "already_at_target_revision": False,
                "promotion_deferred_training_active": False,
                "staged_no_traffic_only": True,
                "traffic_shifted": False,
                "promoted": False,
                "canary_ok": False,
                "canary_skipped_no_traffic_only": True,
                "rollback_triggered": False,
                "dry_run_only": False,
                "gcloud_executed": True,
                "gcloud_calls": len(executed_commands),
                "cloud_state_mutated": True,
                "network_calls": network_calls,
                "zero_inference_verified": True,
                "target_build_sha": resolved_sha,
                "candidate_revision": candidate_rev,
                "candidate_tag": clean_tag,
                "previous_revision": effective_prev_rev,
                "pre_promote": pre_promote,
                "drift": drift,
                "safety_lock": safety_lock,
                "rollout_plan": rollout_plan,
                "executed_commands": executed_commands,
                "diagnostics": diagnostics,
                "compliance": comp_doc,
                "exit_code": 0,
                "summary": summary,
            }
        )

    # 8. Execute Step B: Traffic promotion (`--to-revisions <candidate_rev>=100`)
    final_plan = build_idempotent_rollout_plan(
        gateway=clean_gw,
        target_sha=resolved_sha,
        previous_revision=effective_prev_rev,
        candidate_revision=candidate_rev,
        candidate_tag=clean_tag,
        service_name=clean_service,
        region=clean_region,
        gcp_project=clean_project,
        no_traffic_only=False,
        env=env,
        cwd=cwd,
        lang=lang_code,
    )
    promote_argv = list(final_plan["promote_traffic_argv"])
    promote_rc, promote_out = _invoke_runner(gcloud_runner, promote_argv, GCLOUD_TRAFFIC_TIMEOUT_S)
    executed_commands.append(
        {
            "step": "promote_traffic",
            "argv": promote_argv,
            "returncode": promote_rc,
            "output_preview": sanitize_text_for_display(promote_out)[:240],
        }
    )

    canary_report: dict[str, Any] | None = None
    if promote_rc != 0:
        diagnostics.append(
            f"Échec `update-traffic` vers `{candidate_rev}` (code={promote_rc})."
            if is_fr
            else f"`update-traffic` to `{candidate_rev}` failed (rc={promote_rc})."
        )
        canary_passed = False
    else:
        canary_report = run_post_deploy_holiday_canary(
            clean_gw,
            expected_sha=resolved_sha,
            probe_feedback_post=probe_feedback_post,
            http_fn=http_fn,
            lang=lang_code,
            env=env,
            cwd=cwd,
        )
        network_calls += int(canary_report.get("network_calls", 0))
        canary_passed = bool(canary_report["canary_ok"])
        diagnostics.extend(canary_report.get("diagnostics") or [])

    if canary_passed:
        summary = (
            f"Révision `{candidate_rev}` promue à 100 % et validée par le canari Noël sur {clean_gw}."
            if is_fr
            else f"Promoted revision `{candidate_rev}` to 100% traffic and passed post-deploy holiday canary on {clean_gw}."
        )
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "revision_promoter",
                "lang": lang_code,
                "gateway": clean_gw,
                "verdict": "promoted",
                "compliance_ok": True,
                "pre_promote_ok": True,
                "finops_ok": True,
                "drift_detected": bool(drift.get("drift_detected")),
                "idempotent_noop": False,
                "already_at_target_revision": False,
                "promotion_deferred_training_active": False,
                "staged_no_traffic_only": False,
                "traffic_shifted": True,
                "promoted": True,
                "canary_ok": True,
                "rollback_triggered": False,
                "dry_run_only": False,
                "gcloud_executed": True,
                "gcloud_calls": len(executed_commands),
                "cloud_state_mutated": True,
                "network_calls": network_calls,
                "zero_inference_verified": True,
                "target_build_sha": resolved_sha,
                "candidate_revision": candidate_rev,
                "candidate_tag": clean_tag,
                "previous_revision": effective_prev_rev,
                "pre_promote": pre_promote,
                "drift": drift,
                "safety_lock": safety_lock,
                "rollout_plan": final_plan,
                "canary": canary_report,
                "executed_commands": executed_commands,
                "diagnostics": diagnostics,
                "compliance": comp_doc,
                "exit_code": 0,
                "summary": summary,
            }
        )

    # 9. Automatic sanitized rollback on canary/traffic failure
    rollback_succeeded = False
    rollback_argv: list[str] | None = None
    if effective_prev_rev is not None:
        rollback_argv = list(final_plan["rollback_traffic_argv"])
        rb_rc, rb_out = _invoke_runner(gcloud_runner, rollback_argv, GCLOUD_TRAFFIC_TIMEOUT_S)
        rollback_succeeded = rb_rc == 0
        executed_commands.append(
            {
                "step": "rollback_traffic",
                "argv": rollback_argv,
                "returncode": rb_rc,
                "output_preview": sanitize_text_for_display(rb_out)[:240],
            }
        )
    else:
        diagnostics.append(
            "Rollback automatique demandé mais `--previous-revision` non fourni et aucune révision antérieure découverte."
            if is_fr
            else "Automatic rollback requested, but `--previous-revision` was not provided and no live revision was discovered."
        )

    summary = (
        f"ROLLBACK DÉCLENCHÉ vers `{effective_prev_rev or 'unknown'}` après échec du canari post-déploiement "
        f"(succeeded={rollback_succeeded})."
        if is_fr
        else f"ROLLBACK TRIGGERED to `{effective_prev_rev or 'unknown'}` after post-deploy canary failure "
        f"(succeeded={rollback_succeeded})."
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "revision_promoter",
            "lang": lang_code,
            "gateway": clean_gw,
            "verdict": "rolled_back_after_canary_failure",
            "compliance_ok": True,
            "pre_promote_ok": True,
            "finops_ok": True,
            "drift_detected": bool(drift.get("drift_detected")),
            "idempotent_noop": False,
            "already_at_target_revision": False,
            "promotion_deferred_training_active": False,
            "staged_no_traffic_only": False,
            "traffic_shifted": promote_rc == 0,
            "promoted": False,
            "canary_ok": False,
            "rollback_triggered": True,
            "rollback_succeeded": rollback_succeeded,
            "rollback_revision": effective_prev_rev,
            "rollback_command_argv": rollback_argv,
            "dry_run_only": False,
            "gcloud_executed": True,
            "gcloud_calls": len(executed_commands),
            "cloud_state_mutated": True,
            "network_calls": network_calls,
            "zero_inference_verified": True,
            "target_build_sha": resolved_sha,
            "candidate_revision": candidate_rev,
            "candidate_tag": clean_tag,
            "previous_revision": effective_prev_rev,
            "pre_promote": pre_promote,
            "drift": drift,
            "safety_lock": safety_lock,
            "rollout_plan": final_plan,
            "canary": canary_report,
            "executed_commands": executed_commands,
            "diagnostics": diagnostics,
            "compliance": comp_doc,
            "exit_code": 2,
            "summary": summary,
        }
    )


def _is_cloud_run_deployed_path(rel_path: str) -> bool:
    """Return True iff `rel_path` belongs to the Cloud Run OpenRouter build context."""
    cleaned = rel_path.strip().replace("\\", "/").lstrip("./")
    if not cleaned:
        return False
    if cleaned in CLOUD_RUN_CORE_PATHS:
        return True
    if cleaned.startswith("src/hivemind/router/") and (
        cleaned.endswith(".py") or cleaned == "src/hivemind/router/measured_results.json"
    ):
        return True
    return cleaned in {f"scripts/{name}" for name in EXPECTED_MANIFEST_SCRIPTS}


def _check_deploy_sh_build_sha_propagation(deploy_sh_path: Path) -> bool:
    """Verify `deploy/gcp_openrouter/deploy.sh` propagates `HIVEMIND_BUILD_SHA` via `--update-env-vars`."""
    if not deploy_sh_path.is_file():
        return False
    try:
        raw_text = deploy_sh_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    active_lines = [ln for ln in raw_text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    active_text = "\n".join(active_lines)
    propagates_var = bool(
        re.search(r"(?:ENV_VARS=[^\n]*|--update-env-vars[^\n]*)HIVEMIND_BUILD_SHA=", active_text)
    )
    return bool(propagates_var and "--update-env-vars" in active_text and "--set-env-vars" not in active_text)


def compute_deploy_manifest_fingerprint(
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Compute deterministic SHA-256 manifest digest across Cloud Run-deployed files."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    router_dir = root / "src" / "hivemind" / "router"
    router_files: list[str] = []
    if router_dir.is_dir():
        for entry in sorted(router_dir.glob("*.py")):
            if entry.is_file():
                router_files.append(f"src/hivemind/router/{entry.name}")
        measured = router_dir / "measured_results.json"
        if measured.is_file():
            router_files.append("src/hivemind/router/measured_results.json")

    script_rel_paths = sorted(f"scripts/{name}" for name in EXPECTED_MANIFEST_SCRIPTS)
    tracked_paths = sorted(set(CLOUD_RUN_CORE_PATHS) | set(router_files) | set(script_rel_paths))

    file_hashes: dict[str, str] = {}
    missing_files: list[str] = []
    manifest_entries: list[dict[str, str]] = []
    for rel in tracked_paths:
        target = root / rel
        if target.is_file():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        else:
            digest = "<missing>"
            missing_files.append(rel)
        file_hashes[rel] = digest
        manifest_entries.append({"path": rel, "sha256": digest})

    manifest_canonical = json.dumps(manifest_entries, sort_keys=True, ensure_ascii=False)
    manifest_sha256 = hashlib.sha256(manifest_canonical.encode("utf-8")).hexdigest()
    deploy_sh_ok = _check_deploy_sh_build_sha_propagation(root / "deploy" / "gcp_openrouter" / "deploy.sh")

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "manifest_fingerprint",
            "repo_root": redact_home_path(root),
            "manifest_sha256": manifest_sha256,
            "files_count": len(tracked_paths),
            "router_py_count": sum(1 for p in router_files if p.endswith(".py")),
            "scripts_count": len(script_rel_paths),
            "missing_files": missing_files,
            "deploy_sh_build_sha_propagation_ok": deploy_sh_ok,
            "tracked_paths": tracked_paths,
            "file_hashes": file_hashes,
            "zero_inference_verified": True,
        }
    )


def _resolve_git_head_sha_in_repo(
    runner: RunnerFn,
    root: Path,
    *,
    lang_code: str,
) -> tuple[str | None, str | None]:
    is_fr = lang_code == "fr"
    try:
        rc, out = _invoke_runner(runner, ["git", "-C", str(root), "rev-parse", "HEAD"], SUBPROCESS_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, RevisionPromoterError) as exc:
        return None, (
            f"Échec lecture SHA git HEAD local ({type(exc).__name__})."
            if is_fr
            else f"Failed to resolve local git HEAD SHA ({type(exc).__name__})."
        )
    candidate = sanitize_build_sha(out.strip()) if rc == 0 else "unknown"
    if rc != 0 or candidate == "unknown":
        return None, (
            "SHA git HEAD invalide via `git rev-parse HEAD`."
            if is_fr
            else "Unable to resolve a valid SHA from `git rev-parse HEAD`."
        )
    return candidate, None


def detect_deploy_manifest_diff(
    gateway: str = DEFAULT_GATEWAY,
    *,
    repo_root: Path | str | None = None,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    force_deploy: bool = False,
    http_fn: HttpFn = default_http_request,
    git_runner: RunnerFn = default_git_runner,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Detect pre-PR-#949 drift, manifest mismatches, and Cloud Run-scoped git diffs vs live `build_sha`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    eff_cwd = cwd if cwd is not None else repo_root
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=eff_cwd, lang=lang_code)
    if comp_doc["compliance_ok"] and repo_root is not None and cwd is not None:
        comp_repo = evaluate_personal_pc_compliance(env=env, cwd=repo_root, lang=lang_code)
        if not comp_repo["compliance_ok"]:
            comp_doc = comp_repo
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("deploy_manifest_diff", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    manifest = compute_deploy_manifest_fingerprint(repo_root=root)

    subprocess_calls = 0
    resolved_target_sha: str | None = None
    expected_sha_source = "none"
    git_head_error: str | None = None
    if expected_sha is not None and str(expected_sha).strip():
        resolved_target_sha = validate_target_sha(expected_sha)
        expected_sha_source = "expected_sha"
    else:
        subprocess_calls += 1
        head_sha, head_err = _resolve_git_head_sha_in_repo(git_runner, root, lang_code=lang_code)
        if head_sha is not None:
            resolved_target_sha = head_sha
            expected_sha_source = "git_head"
        elif check_git_head:
            expected_sha_source = "git_head"
            git_head_error = head_err

    drift = detect_revision_drift(
        clean_gw,
        expected_sha=resolved_target_sha,
        check_git_head=False,
        git_runner=git_runner,
        http_fn=http_fn,
        lang=lang_code,
        env=env,
        cwd=eff_cwd,
    )
    network_calls = int(drift.get("network_calls", 0))
    live_build_sha = str(drift.get("live_build_sha") or "unknown")
    diagnostics = list(drift.get("diagnostics") or [])

    if git_head_error is not None:
        diagnostics.append(git_head_error)

    deploy_sh_prop_ok = bool(manifest["deploy_sh_build_sha_propagation_ok"])
    if not deploy_sh_prop_ok:
        diagnostics.append(
            "Le script `deploy/gcp_openrouter/deploy.sh` ne propage pas `HIVEMIND_BUILD_SHA` via `--update-env-vars`."
            if is_fr
            else "`deploy/gcp_openrouter/deploy.sh` is missing `HIVEMIND_BUILD_SHA` propagation via `--update-env-vars`."
        )

    sha_mismatch = bool(drift.get("sha_mismatch")) or (git_head_error is not None)
    sha_match: bool | None = False if git_head_error is not None else drift.get("sha_match")

    git_diff_checked = False
    git_diff_ok: bool | None = None
    changed_files: list[str] = []
    cloud_run_changed_files: list[str] = []
    non_cloud_run_changed_files: list[str] = []

    if (
        resolved_target_sha is not None
        and live_build_sha != "unknown"
        and not shas_match(live_build_sha, resolved_target_sha)
    ):
        git_diff_checked = True
        subprocess_calls += 1
        try:
            diff_rc, diff_out = _invoke_runner(
                git_runner,
                ["git", "-C", str(root), "diff", "--name-only", f"{live_build_sha}..{resolved_target_sha}"],
                SUBPROCESS_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError, RevisionPromoterError) as exc:
            diff_rc, diff_out = 127, f"{type(exc).__name__}"

        if diff_rc == 0:
            git_diff_ok = True
            seen_paths: set[str] = set()
            for raw_line in diff_out.splitlines():
                rel = raw_line.strip().replace("\\", "/").lstrip("./")
                if not rel or rel in seen_paths:
                    continue
                seen_paths.add(rel)
                changed_files.append(rel)
                if _is_cloud_run_deployed_path(rel):
                    cloud_run_changed_files.append(rel)
                else:
                    non_cloud_run_changed_files.append(rel)
        else:
            git_diff_ok = False
            diagnostics.append(
                f"Impossible de calculer `git diff --name-only` entre `{live_build_sha}` et `{resolved_target_sha}`."
                if is_fr
                else f"Unable to inspect read-only `git diff --name-only` between `{live_build_sha}` and `{resolved_target_sha}`."
            )

    routes_unhealthy = bool(
        drift.get("failed_routes")
        or not drift.get("manifest_present")
        or not drift.get("manifest_valid")
        or not drift.get("feedback_summary_ok")
    )
    missing_routes_or_scripts = bool(drift.get("missing_holiday_routes") or drift.get("missing_scripts"))
    docs_or_non_runtime_only_diff = bool(
        sha_mismatch
        and not drift.get("pre_pr_949_drift_detected")
        and not routes_unhealthy
        and not missing_routes_or_scripts
        and not manifest.get("missing_files")
        and git_diff_ok is True
        and len(cloud_run_changed_files) == 0
    )

    trigger_reasons: list[str] = []
    if force_deploy:
        trigger_reasons.append("force_deploy")
    if drift.get("pre_pr_949_drift_detected"):
        trigger_reasons.append("pre_pr_949_404_drift")
    if missing_routes_or_scripts:
        trigger_reasons.append("missing_routes_or_scripts")
    if routes_unhealthy and not drift.get("pre_pr_949_drift_detected"):
        trigger_reasons.append("unhealthy_live_routes")
    if manifest.get("missing_files"):
        trigger_reasons.append("missing_local_manifest_files")
    if cloud_run_changed_files:
        trigger_reasons.append("cloud_run_files_changed")
    if sha_mismatch and not docs_or_non_runtime_only_diff:
        trigger_reasons.append("sha_mismatch")

    deploy_needed = bool(trigger_reasons)
    idempotent_skip = not deploy_needed
    skip_reason: str | None = None
    if idempotent_skip:
        if docs_or_non_runtime_only_diff:
            skip_reason = "non_cloud_run_diff_only"
        elif sha_match is True:
            skip_reason = "already_at_target_sha"
        else:
            skip_reason = "live_healthy_no_drift"

    summary = (
        (
            f"Déploiement Cloud Run REQUIS sur {clean_gw} ({', '.join(trigger_reasons)})."
            if is_fr
            else f"Cloud Run deploy NEEDED on {clean_gw} ({', '.join(trigger_reasons)})."
        )
        if deploy_needed
        else (
            f"Skip idempotent sur {clean_gw} ({skip_reason}) : aucun changement Cloud Run requis."
            if is_fr
            else f"Idempotent skip on {clean_gw} ({skip_reason}): no Cloud Run redeploy needed."
        )
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "deploy_manifest_diff",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "deploy_needed": deploy_needed,
            "idempotent_skip": idempotent_skip,
            "skip_reason": skip_reason,
            "force_deploy": bool(force_deploy),
            "trigger_reasons": trigger_reasons,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_fingerprint": manifest,
            "deploy_sh_build_sha_propagation_ok": deploy_sh_prop_ok,
            "drift_detected": bool(drift.get("drift_detected")),
            "pre_pr_949_drift_detected": bool(drift.get("pre_pr_949_drift_detected")),
            "pre_pr_949_404_routes": list(drift.get("pre_pr_949_404_routes") or []),
            "missing_holiday_routes": list(drift.get("missing_holiday_routes") or []),
            "missing_scripts": list(drift.get("missing_scripts") or []),
            "live_build_sha": live_build_sha,
            "expected_build_sha": resolved_target_sha,
            "expected_sha_source": expected_sha_source,
            "cloud_run_revision": drift.get("cloud_run_revision"),
            "sha_match": sha_match,
            "sha_mismatch": sha_mismatch,
            "git_diff_checked": git_diff_checked,
            "git_diff_ok": git_diff_ok,
            "changed_files": changed_files,
            "cloud_run_changed_files": cloud_run_changed_files,
            "non_cloud_run_changed_files": non_cloud_run_changed_files,
            "drift": drift,
            "network_calls": network_calls,
            "subprocess_calls": subprocess_calls,
            "zero_inference_verified": True,
            "diagnostics": diagnostics,
            "exit_code": 0 if deploy_sh_prop_ok else 2,
            "summary": summary,
        }
    )


def _inspect_models_contract(
    body: Mapping[str, Any],
    *,
    require_coder_model: bool,
    lang_code: str,
) -> tuple[bool, list[str], bool, dict[str, Any] | None, list[str]]:
    is_fr = lang_code == "fr"
    issues: list[str] = []
    obj_ok = body.get("object", "list") == "list"
    raw_data = body.get("data")
    if not obj_ok or not isinstance(raw_data, list):
        issues.append(
            "Contrat JSON `/v1/models` invalide (champ `data` manquant)."
            if is_fr
            else "Invalid `/v1/models` JSON schema (missing `data` list)."
        )
        return False, [], False, None, issues

    model_ids: list[str] = []
    coder_meta: dict[str, Any] | None = None
    for entry in raw_data:
        if not isinstance(entry, Mapping):
            continue
        raw_id = entry.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            continue
        clean_id = raw_id.strip()
        model_ids.append(clean_id)
        if clean_id == "hivemind/coder" and coder_meta is None:
            coder_meta = {
                "present": True,
                "id": "hivemind/coder",
                "object": str(entry.get("object") or "model"),
                "owned_by": str(entry.get("owned_by") or "hivemind"),
                "loaded": entry.get("loaded"),
            }

    if not model_ids:
        issues.append(
            "Liste `data` de `GET /v1/models` vide (aucun modèle servi)."
            if is_fr
            else "`GET /v1/models` returned an empty `data` model list."
        )
    coder_present = coder_meta is not None
    if require_coder_model and not coder_present:
        issues.append(
            "Modèle `hivemind/coder` absent de `GET /v1/models`."
            if is_fr
            else "Required model `hivemind/coder` missing from `GET /v1/models`."
        )
    return len(issues) == 0, model_ids, coder_present, coder_meta, issues


def project_adoption_readiness_gates(
    smoke_report: Mapping[str, Any] | None = None,
    *,
    gateway_and_models_ok: bool | None = None,
    holiday_routes_ok: bool | None = None,
    feedback_summary: Mapping[str, Any] | None = None,
    active_volunteer_gpus: int = 0,
    lang: str = "en",
) -> dict[str, Any]:
    """Project the 5 objective Noel-at-Scale adoption readiness gates from smoke & CES metrics."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    rep = smoke_report if isinstance(smoke_report, Mapping) else {}
    gw_ok = (
        bool(gateway_and_models_ok)
        if gateway_and_models_ok is not None
        else bool(rep.get("health_ok") and rep.get("models_ok"))
    )
    hol_ok = (
        bool(holiday_routes_ok)
        if holiday_routes_ok is not None
        else bool(rep.get("manifest_ok") and rep.get("html_contract_ok") and rep.get("feedback_summary_ok"))
    )

    eff_fb: Mapping[str, Any] = (
        feedback_summary
        if isinstance(feedback_summary, Mapping)
        else (rep.get("feedback_summary") if isinstance(rep.get("feedback_summary"), Mapping) else {})
    )
    raw_total = eff_fb.get("total_responses")
    has_resp = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total > 0
    raw_ease = eff_fb.get("mean_ease_rating") if "mean_ease_rating" in eff_fb else eff_fb.get("average_ease_rating")
    raw_lag = eff_fb.get("zero_game_lag_ratio")

    ces_ok = bool(
        has_resp
        and not isinstance(raw_ease, bool)
        and isinstance(raw_ease, (int, float))
        and float(raw_ease) >= 4.2
    )
    lag_ok = bool(
        has_resp
        and not isinstance(raw_lag, bool)
        and isinstance(raw_lag, (int, float))
        and float(raw_lag) >= 0.95
    )

    eff_gpus = (
        int(active_volunteer_gpus)
        if not isinstance(active_volunteer_gpus, bool) and isinstance(active_volunteer_gpus, int)
        else 0
    )
    gpu_ok = eff_gpus >= 8

    gates = {
        "gateway_and_models_online": gw_ok,
        "holiday_routes_deployed_on_cloud_run": hol_ok,
        "satisfaction_ces_target_met": ces_ok,
        "gamer_zero_lag_target_met": lag_ok,
        "external_volunteer_gpu_target_noel": gpu_ok,
    }
    blockers: list[str] = []
    if not gw_ok:
        blockers.append(
            "Passerelle Cloud Run `/health` ou `/v1/models` indisponible."
            if is_fr
            else "Cloud Run gateway `/health` or `/v1/models` check failed."
        )
    if not hol_ok:
        blockers.append(
            "Routes de Noël (`/noel`, `/quick`, `/raid-noel`, `/arena-noel`, `/feedback/summary`) incomplètes."
            if is_fr
            else "Holiday onboarding routes (`/noel`, `/quick`, `/raid-noel`, `/arena-noel`, `/feedback/summary`) incomplete."
        )
    if not ces_ok:
        blockers.append(
            "Score CES moyen cible (>= 4.2 / 5) non encore atteint."
            if is_fr
            else "Mean CES ease target (>= 4.2 / 5) not yet met."
        )
    if not lag_ok:
        blockers.append(
            "Ratio zéro-lag Gamer cible (>= 0.95) non encore atteint."
            if is_fr
            else "Gamer zero-lag comfort ratio (>= 0.95) not yet met."
        )
    if not gpu_ok:
        blockers.append(
            f"GPUs bénévoles externes actifs ({eff_gpus}) inférieurs à l'objectif Noël (>= 8)."
            if is_fr
            else f"Active external volunteer GPUs ({eff_gpus}) below holiday target (>= 8)."
        )

    passed_ids = [k for k, v in gates.items() if v]
    pending_ids = [k for k, v in gates.items() if not v]
    passed_count = len(passed_ids)

    return {
        "schema": SCHEMA,
        "object": "hivemind.adoption_readiness_projection",
        "campaign_id": CAMPAIGN_ID,
        "lang": lang_code,
        "gates": gates,
        "readiness_gates_passed": passed_count,
        "readiness_gates_total": len(gates),
        "passed_gate_ids": passed_ids,
        "pending_gate_ids": pending_ids,
        "blockers": blockers,
        "healthy_rollout_floor_met": passed_count >= 2,
        "soak_ces_gates_met": bool(gw_ok and hol_ok and ces_ok and lag_ok),
    }


def _build_incident_alert(
    *,
    gateway: str,
    verdict: str,
    failure_categories: Sequence[str],
    failed_routes: Sequence[str],
    diagnostics: Sequence[str],
    rollback_triggered: bool,
    rollback_succeeded: bool | None,
    rollback_revision: str | None,
    live_build_sha: str,
    expected_build_sha: str | None,
    summary: str,
) -> dict[str, Any]:
    clean_diags = [sanitize_text_for_display(str(item)) for item in diagnostics if str(item).strip()]
    alert = {
        "object": "hivemind.cloud_run_incident_alert",
        "campaign_id": CAMPAIGN_ID,
        "severity": "critical",
        "gateway": gateway,
        "verdict": verdict,
        "failure_categories": sorted({str(c) for c in failure_categories if str(c).strip()}),
        "failed_routes": list(failed_routes),
        "rollback_triggered": bool(rollback_triggered),
        "rollback_succeeded": rollback_succeeded,
        "rollback_revision": rollback_revision,
        "live_build_sha": sanitize_build_sha(live_build_sha),
        "expected_build_sha": expected_build_sha,
        "diagnostics": clean_diags,
        "summary": sanitize_text_for_display(summary),
        "personal_pc_only": True,
        "financial_promise": FINANCIAL_PROMISE,
        "zero_pii_verified": True,
    }
    _assert_safe_output(json.dumps(alert, sort_keys=True, ensure_ascii=False))
    return alert


def verify_post_deploy_smoke_suite(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    probe_feedback_post: bool = False,
    require_coder_model: bool = False,
    previous_revision: str | None = None,
    active_volunteer_gpus: int = 0,
    http_fn: HttpFn = default_http_request,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify /health, /v1/models, manifest, all 4 holiday HTML pages, CES summary, and readiness projection."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("post_deploy_smoke_suite", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = (
        validate_target_sha(expected_sha)
        if expected_sha is not None and str(expected_sha).strip()
        else None
    )
    clean_prev_rev: str | None = None
    if previous_revision is not None and str(previous_revision).strip():
        clean_prev_rev = sanitize_cloud_run_revision(previous_revision)
        if clean_prev_rev is None:
            raise RevisionPromoterError(f"Invalid previous Cloud Run revision: {previous_revision!r}.")

    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    http_statuses: dict[str, int] = {}
    failed_routes: list[str] = []
    failure_categories: list[str] = []
    diagnostics: list[str] = []
    network_calls = 0

    health_ok = False
    models_ok = False
    live_model_ids: list[str] = []
    coder_present = False
    coder_metadata: dict[str, Any] | None = None
    manifest_ok = False
    live_build_sha = "unknown"
    cloud_run_revision: str | None = None
    feedback_summary_ok = False
    feedback_summary_doc: dict[str, Any] = {
        "schema_valid": False,
        "total_responses": None,
        "mean_ease_rating": None,
        "zero_game_lag_ratio": None,
        "by_profile": None,
    }

    for path, expected_kind in SMOKE_SUITE_PROBE_ROUTES:
        network_calls += 1
        code, body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}{path}")
        http_statuses[path] = code
        if code != 200:
            failed_routes.append(path)
            if code == 404:
                failure_categories.append("stale_404_drift")
            else:
                failure_categories.append("http_error")
            diagnostics.append(
                f"Smoke GET {path} en échec (HTTP {code})."
                if is_fr
                else f"Smoke GET {path} failed (HTTP {code})."
            )
            continue

        if path == "/health":
            health_ok = body.get("status") == "ok"
            if not health_ok:
                failed_routes.append(path)
                failure_categories.append("health_contract_failed")
                diagnostics.append("Smoke GET /health returned HTTP 200 without `status == 'ok'`.")
        elif path == "/v1/models":
            m_ok, live_model_ids, coder_present, coder_metadata, m_issues = _inspect_models_contract(
                body, require_coder_model=require_coder_model, lang_code=lang_code
            )
            models_ok = m_ok
            if not m_ok:
                failed_routes.append(path)
                failure_categories.append("empty_models_list" if not live_model_ids else "models_contract_failed")
                diagnostics.extend(m_issues)
        elif path == "/v1/hivemind/onboarding/manifest":
            ok_m, live_build_sha, cloud_run_revision, _mr, _ms, man_issues = _inspect_manifest_contract(
                body, lang_code=lang_code
            )
            manifest_ok = ok_m
            if not ok_m:
                failed_routes.append(path)
                failure_categories.append("manifest_contract_failed")
                diagnostics.extend(man_issues)
        elif expected_kind == "html":
            ok_h, h_doc, h_issues = _inspect_holiday_html_contract(
                path, body, seen_nonces, strict_contract=True, lang_code=lang_code
            )
            html_contracts[path] = h_doc
            if not ok_h:
                failed_routes.append(path)
                if not h_doc.get("csp_nonce_verified"):
                    failure_categories.append("csp_nonce_violation")
                if not h_doc.get("dom_clean"):
                    failure_categories.append("unsafe_dom_or_static_figure")
                if not h_doc.get("personal_pc_banner_present"):
                    failure_categories.append("missing_compliance_banner")
                diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            fb_ok, fb_doc, fb_issues = _inspect_feedback_summary_contract(body, lang_code=lang_code)
            feedback_summary_ok = fb_ok
            feedback_summary_doc = fb_doc
            if not fb_ok:
                failed_routes.append(path)
                failure_categories.append("feedback_summary_schema_failed")
                diagnostics.extend(fb_issues)

    sha_ok = True
    if clean_expected_sha is not None:
        sha_ok = shas_match(live_build_sha, clean_expected_sha)
        if not sha_ok:
            failure_categories.append("sha_mismatch")
            diagnostics.append(
                f"Smoke SHA incorrect : direct='{live_build_sha}' != attendu '{clean_expected_sha}'."
                if is_fr
                else f"Smoke build SHA mismatch: live='{live_build_sha}' != expected '{clean_expected_sha}'."
            )

    html_all_ok = (
        len(html_contracts) == len(SMOKE_HTML_ROUTES)
        and all(html_contracts.get(p, {}).get("ok") is True for p in SMOKE_HTML_ROUTES)
        and len(seen_nonces) == len(SMOKE_HTML_ROUTES)
    )
    if len(html_contracts) == len(SMOKE_HTML_ROUTES) and len(seen_nonces) != len(SMOKE_HTML_ROUTES):
        failure_categories.append("reused_csp_nonce")

    feedback_post_ok: bool | None = None
    feedback_post_http_status: int | None = None
    idem_key: str | None = None
    if probe_feedback_post:
        idem_seed = f"noel-smoke-suite/v1|{clean_gw}|{clean_expected_sha or 'live'}"
        idem_key = "smoke-noel-" + hashlib.sha256(idem_seed.encode("utf-8")).hexdigest()[:32]
        smoke_payload = {
            "profile": "dev",
            "ease_rating": 5,
            "zero_game_lag": True,
            "platform": "linux",
            "note": "",
        }
        network_calls += 1
        post_code, post_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/onboarding/feedback",
            payload=smoke_payload,
            headers={"Idempotency-Key": idem_key},
            allow_feedback_post=True,
        )
        feedback_post_http_status = post_code
        post_valid, post_sum, _ = (
            _inspect_feedback_summary_contract(post_body, lang_code=lang_code)
            if post_code in (200, 201)
            else (False, feedback_summary_doc, [])
        )
        if post_code in (200, 201) and post_valid and (post_sum.get("total_responses") or 0) >= 1:
            network_calls += 1
            ref_code, ref_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/onboarding/feedback/summary")
            http_statuses["/v1/hivemind/onboarding/feedback/summary#after_post"] = ref_code
            ref_ok, ref_sum, ref_issues = (
                _inspect_feedback_summary_contract(ref_body, lang_code=lang_code)
                if ref_code == 200
                else (False, post_sum, ["Smoke GET feedback summary after POST failed."])
            )
            feedback_post_ok = bool(ref_code == 200 and ref_ok and (ref_sum.get("total_responses") or 0) >= 1)
            if feedback_post_ok:
                feedback_summary_ok = True
                feedback_summary_doc = ref_sum
            else:
                failure_categories.append("feedback_post_failed")
                diagnostics.extend(ref_issues)
        else:
            feedback_post_ok = False
            failure_categories.append("feedback_post_failed")
            diagnostics.append(
                f"Smoke POST /v1/hivemind/onboarding/feedback en échec (HTTP {post_code})."
                if is_fr
                else f"Smoke POST /v1/hivemind/onboarding/feedback failed (HTTP {post_code})."
            )

    smoke_ok = bool(
        health_ok
        and models_ok
        and manifest_ok
        and sha_ok
        and html_all_ok
        and feedback_summary_ok
        and (not probe_feedback_post or feedback_post_ok is True)
    )
    readiness_projection = project_adoption_readiness_gates(
        gateway_and_models_ok=bool(health_ok and models_ok),
        holiday_routes_ok=bool(manifest_ok and html_all_ok and feedback_summary_ok),
        feedback_summary=feedback_summary_doc,
        active_volunteer_gpus=active_volunteer_gpus,
        lang=lang_code,
    )

    summary = (
        (
            f"Suite smoke post-déploiement Cloud Run VERTE sur {clean_gw} "
            f"(nonces CSP uniques={len(seen_nonces)}, gates={readiness_projection['readiness_gates_passed']}/5)."
            if is_fr
            else f"Post-deploy Cloud Run smoke suite PASSED on {clean_gw} "
            f"(unique CSP nonces={len(seen_nonces)}, readiness_gates={readiness_projection['readiness_gates_passed']}/5)."
        )
        if smoke_ok
        else (
            f"Suite smoke post-déploiement Cloud Run EN ÉCHEC sur {clean_gw} : alerte & rollback requis."
            if is_fr
            else f"Post-deploy Cloud Run smoke suite FAILED on {clean_gw}: incident alert & rollback required."
        )
    )

    effective_rb_rev = clean_prev_rev or cloud_run_revision
    incident_alert: dict[str, Any] | None = None
    if not smoke_ok:
        incident_alert = _build_incident_alert(
            gateway=clean_gw,
            verdict="smoke_suite_failed",
            failure_categories=failure_categories,
            failed_routes=failed_routes,
            diagnostics=diagnostics,
            rollback_triggered=False,
            rollback_succeeded=None,
            rollback_revision=effective_rb_rev,
            live_build_sha=live_build_sha,
            expected_build_sha=clean_expected_sha,
            summary=summary,
        )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "mode": "post_deploy_smoke_suite",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "smoke_ok": smoke_ok,
            "canary_ok": smoke_ok,
            "rollback_required": not smoke_ok,
            "health_ok": health_ok,
            "models_ok": models_ok,
            "live_models_count": len(live_model_ids),
            "live_model_ids": live_model_ids,
            "hivemind_coder_present": coder_present,
            "coder_model_metadata": coder_metadata,
            "manifest_ok": manifest_ok,
            "sha_ok": sha_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "cloud_run_revision": cloud_run_revision,
            "html_contract_ok": html_all_ok,
            "html_contracts": html_contracts,
            "unique_csp_nonces_count": len(seen_nonces),
            "feedback_summary_ok": feedback_summary_ok,
            "feedback_summary": feedback_summary_doc,
            "feedback_post_probed": bool(probe_feedback_post),
            "feedback_post_ok": feedback_post_ok,
            "feedback_post_http_status": feedback_post_http_status,
            "feedback_post_idempotency_key": idem_key,
            "http_statuses": http_statuses,
            "failed_routes": failed_routes,
            "failure_categories": sorted(set(failure_categories)),
            "readiness_projection": readiness_projection,
            "readiness_gates_passed": readiness_projection["readiness_gates_passed"],
            "readiness_gates_total": readiness_projection["readiness_gates_total"],
            "readiness_gates": readiness_projection["gates"],
            "incident_alert_emitted": incident_alert is not None,
            "incident_alert": incident_alert,
            "network_calls": network_calls,
            "zero_inference_verified": True,
            "diagnostics": diagnostics,
            "exit_code": 0 if smoke_ok else 2,
            "summary": summary,
        }
    )


def _build_merge_receipt(
    *,
    lang_code: str,
    gateway: str,
    verdict: str,
    manifest_diff: Mapping[str, Any],
    pre_promote: Mapping[str, Any],
    comp_doc: Mapping[str, Any],
    network_calls: int,
    execute: bool,
    exit_code: int,
    summary: str,
    diagnostics: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a standardized deploy-on-merge receipt with SHA-256 signature."""
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "mode": "deploy_on_merge",
        "lang": lang_code,
        "gateway": gateway,
        "verdict": verdict,
        "compliance_ok": True,
        "pre_promote_ok": bool(pre_promote.get("pre_promote_ok")),
        "finops_ok": bool(pre_promote.get("finops_ok")),
        "deploy_needed": bool(manifest_diff.get("deploy_needed")),
        "drift_detected": bool(manifest_diff.get("drift_detected")),
        "manifest_sha256": manifest_diff.get("manifest_sha256"),
        "idempotent_noop": False,
        "already_at_target_revision": False,
        "promotion_deferred_training_active": False,
        "staged_no_traffic_only": False,
        "promoted": False,
        "canary_ok": False,
        "smoke_ok": False,
        "rollback_triggered": False,
        "incident_alert_emitted": False,
        "incident_alert": None,
        "readiness_gates_passed": 0,
        "readiness_gates_total": 5,
        "dry_run_only": not execute,
        "gcloud_executed": False,
        "gcloud_calls": 0,
        "cloud_state_mutated": False,
        "network_calls": network_calls,
        "zero_inference_verified": True,
        "target_build_sha": manifest_diff.get("expected_build_sha"),
        "manifest_diff": manifest_diff,
        "pre_promote": pre_promote,
        "commands": [],
        "executed_commands": [],
        "diagnostics": list(diagnostics),
        "compliance": comp_doc,
        "exit_code": exit_code,
        "summary": summary,
    }
    if extra:
        payload.update(extra)
    return _attach_receipt(payload)


def orchestrate_deploy_on_merge(
    gateway: str = DEFAULT_GATEWAY,
    *,
    repo_root: Path | str | None = None,
    target_sha: str | None = None,
    previous_revision: str | None = None,
    candidate_tag: str = DEFAULT_CANDIDATE_TAG,
    service_name: str = DEFAULT_CLOUD_RUN_SERVICE,
    region: str = DEFAULT_CLOUD_RUN_REGION,
    gcp_project: str | None = None,
    deploy_script_path: Path | str | None = None,
    server_py_path: Path | str | None = None,
    app_factory: Callable[..., Any] | None = None,
    lock_file_path: Path | str | None = None,
    process_probe_fn: Callable[[], Iterable[str]] | None = None,
    check_git_head: bool = False,
    force_deploy: bool = False,
    execute: bool = False,
    no_traffic_only: bool = False,
    allow_during_training: bool = False,
    defer_on_active_training: bool = False,
    probe_feedback_post: bool = False,
    require_coder_model: bool = False,
    active_volunteer_gpus: int = 0,
    gcloud_runner: RunnerFn = default_gcloud_runner,
    http_fn: HttpFn = default_http_request,
    git_runner: RunnerFn = default_git_runner,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """End-to-end deploy-on-merge orchestrator: compliance -> manifest diff -> FinOps -> lock -> rollout -> smoke -> rollback."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"

    eff_cwd = cwd if cwd is not None else repo_root
    comp_doc = evaluate_personal_pc_compliance(env=env, cwd=eff_cwd, lang=lang_code)
    if comp_doc["compliance_ok"] and repo_root is not None and cwd is not None:
        comp_repo = evaluate_personal_pc_compliance(env=env, cwd=repo_root, lang=lang_code)
        if not comp_repo["compliance_ok"]:
            comp_doc = comp_repo
    if not comp_doc["compliance_ok"]:
        return _compliance_blocked_report("deploy_on_merge", DEFAULT_GATEWAY, comp_doc, lang_code=lang_code)

    clean_gw = validate_gateway_url(gateway)
    clean_tag = validate_candidate_tag(candidate_tag)
    clean_service = validate_cloud_identifier(service_name, "service_name")
    clean_region = validate_cloud_identifier(region, "region")
    clean_project = validate_gcp_project(gcp_project if gcp_project is not None else (env or {}).get("GCP_PROJECT"))

    if previous_revision is not None and str(previous_revision).strip():
        explicit_prev_rev = sanitize_cloud_run_revision(previous_revision)
        if explicit_prev_rev is None:
            raise RevisionPromoterError(f"Invalid previous Cloud Run revision: {previous_revision!r}.")
    else:
        explicit_prev_rev = None

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    eff_deploy_path = (
        Path(deploy_script_path)
        if deploy_script_path is not None
        else (
            root / "deploy" / "gcp_openrouter" / "deploy.sh"
            if (root / "deploy" / "gcp_openrouter" / "deploy.sh").is_file() or repo_root is None
            else DEFAULT_DEPLOY_SCRIPT_PATH
        )
    )
    eff_server_path = (
        Path(server_py_path)
        if server_py_path is not None
        else (
            root / "deploy" / "gcp_openrouter" / "server.py"
            if (root / "deploy" / "gcp_openrouter" / "server.py").is_file() or repo_root is None
            else DEFAULT_SERVER_PY_PATH
        )
    )

    # 2. Manifest fingerprint & read-only git diff against live build_sha
    manifest_diff = detect_deploy_manifest_diff(
        clean_gw,
        repo_root=root,
        expected_sha=target_sha,
        check_git_head=check_git_head,
        force_deploy=force_deploy,
        http_fn=http_fn,
        git_runner=git_runner,
        env=env,
        cwd=eff_cwd,
        lang=lang_code,
    )
    network_calls = int(manifest_diff.get("network_calls", 0))
    resolved_sha = manifest_diff.get("expected_build_sha")
    effective_prev_rev = explicit_prev_rev or manifest_diff.get("cloud_run_revision")

    # 3. Pre-promote FinOps & zero-inference FastAPI sanity check
    pre_promote = verify_pre_promote_gateway_and_finops(
        deploy_script_path=eff_deploy_path,
        server_py_path=eff_server_path,
        app_factory=app_factory,
        env=env,
        cwd=eff_cwd,
        lang=lang_code,
    )
    if not pre_promote["pre_promote_ok"] or not manifest_diff["deploy_sh_build_sha_propagation_ok"]:
        pre_diags = list(pre_promote.get("diagnostics") or [])
        if not manifest_diff["deploy_sh_build_sha_propagation_ok"]:
            pre_diags.extend(
                d for d in manifest_diff.get("diagnostics") or [] if "HIVEMIND_BUILD_SHA" in str(d)
            )
        summary_msg = pre_diags[0] if pre_diags else pre_promote["summary"]
        alert = _build_incident_alert(
            gateway=clean_gw,
            verdict="pre_promote_failed",
            failure_categories=["pre_promote_finops_failed"],
            failed_routes=[],
            diagnostics=pre_diags,
            rollback_triggered=False,
            rollback_succeeded=None,
            rollback_revision=effective_prev_rev,
            live_build_sha=str(manifest_diff.get("live_build_sha") or "unknown"),
            expected_build_sha=resolved_sha,
            summary=summary_msg,
        )
        return _build_merge_receipt(
            lang_code=lang_code,
            gateway=clean_gw,
            verdict="pre_promote_failed",
            manifest_diff=manifest_diff,
            pre_promote=pre_promote,
            comp_doc=comp_doc,
            network_calls=network_calls,
            execute=execute,
            exit_code=2,
            summary=summary_msg,
            diagnostics=pre_diags,
            extra={
                "pre_promote_ok": False,
                "incident_alert_emitted": True,
                "incident_alert": alert,
                "previous_revision": effective_prev_rev,
            },
        )

    # 4. Idempotent no-op skip when deploy_needed is False -> run expanded post-deploy smoke suite directly
    if not manifest_diff["deploy_needed"]:
        smoke_expected_sha = (
            resolved_sha if manifest_diff.get("skip_reason") == "already_at_target_sha" else None
        )
        smoke_res = verify_post_deploy_smoke_suite(
            clean_gw,
            expected_sha=smoke_expected_sha,
            probe_feedback_post=probe_feedback_post,
            require_coder_model=require_coder_model,
            previous_revision=effective_prev_rev,
            active_volunteer_gpus=active_volunteer_gpus,
            http_fn=http_fn,
            env=env,
            cwd=eff_cwd,
            lang=lang_code,
        )
        network_calls += int(smoke_res.get("network_calls", 0))
        if smoke_res["smoke_ok"]:
            skip_tag = str(manifest_diff.get("skip_reason") or "live_healthy_no_drift")
            summary = (
                f"Skip idempotent deploy-on-merge ({skip_tag}) sur {clean_gw} ; "
                f"smoke suite verte (gates={smoke_res['readiness_gates_passed']}/5, 0 mutation gcloud)."
                if is_fr
                else f"Deploy-on-merge idempotent no-op ({skip_tag}) on {clean_gw}; "
                f"smoke suite passed (readiness_gates={smoke_res['readiness_gates_passed']}/5, 0 gcloud mutations)."
            )
            return _build_merge_receipt(
                lang_code=lang_code,
                gateway=clean_gw,
                verdict="idempotent_noop",
                manifest_diff=manifest_diff,
                pre_promote=pre_promote,
                comp_doc=comp_doc,
                network_calls=network_calls,
                execute=execute,
                exit_code=0,
                summary=summary,
                extra={
                    "idempotent_noop": True,
                    "already_at_target_revision": bool(manifest_diff.get("sha_match")),
                    "skip_reason": skip_tag,
                    "canary_ok": True,
                    "smoke_ok": True,
                    "readiness_projection": smoke_res["readiness_projection"],
                    "readiness_gates_passed": smoke_res["readiness_gates_passed"],
                    "readiness_gates_total": smoke_res["readiness_gates_total"],
                    "live_build_sha": smoke_res.get("live_build_sha"),
                    "cloud_run_revision": smoke_res.get("cloud_run_revision"),
                    "previous_revision": effective_prev_rev,
                    "smoke_suite": smoke_res,
                    "canary": smoke_res,
                },
            )
        return _build_merge_receipt(
            lang_code=lang_code,
            gateway=clean_gw,
            verdict="smoke_suite_failed",
            manifest_diff=manifest_diff,
            pre_promote=pre_promote,
            comp_doc=comp_doc,
            network_calls=network_calls,
            execute=execute,
            exit_code=2,
            summary=smoke_res["summary"],
            diagnostics=list(smoke_res.get("diagnostics") or []),
            extra={
                "already_at_target_revision": bool(manifest_diff.get("sha_match")),
                "incident_alert_emitted": bool(smoke_res.get("incident_alert_emitted")),
                "incident_alert": smoke_res.get("incident_alert"),
                "readiness_projection": smoke_res["readiness_projection"],
                "readiness_gates_passed": smoke_res["readiness_gates_passed"],
                "readiness_gates_total": smoke_res["readiness_gates_total"],
                "live_build_sha": smoke_res.get("live_build_sha"),
                "previous_revision": effective_prev_rev,
                "smoke_suite": smoke_res,
                "canary": smoke_res,
            },
        )

    # 5. Active-training safety lock before live rollout
    safety_lock = check_active_training_safety_lock(
        clean_gw,
        lock_file_path=lock_file_path,
        process_probe_fn=process_probe_fn,
        http_fn=http_fn,
        allow_during_training=allow_during_training,
        env=env,
        cwd=eff_cwd,
        lang=lang_code,
    )
    network_calls += int(safety_lock.get("network_calls", 0))

    if safety_lock["training_active"] and not allow_during_training and not no_traffic_only:
        deferred = bool(defer_on_active_training)
        exit_code = 3 if deferred else 2
        verdict = "deferred_active_training" if deferred else "blocked_active_training"
        msg = (
            f"Déploiement merge différé (verrou entraînement actif : {', '.join(safety_lock['active_reasons'])})."
            if is_fr
            else f"Deploy-on-merge deferred (active training lock: {', '.join(safety_lock['active_reasons'])}; "
            "use --no-traffic-only to stage safely without shifting traffic)."
        )
        alert = (
            None
            if deferred
            else _build_incident_alert(
                gateway=clean_gw,
                verdict=verdict,
                failure_categories=["active_training_safety_lock"],
                failed_routes=[],
                diagnostics=[msg],
                rollback_triggered=False,
                rollback_succeeded=None,
                rollback_revision=effective_prev_rev,
                live_build_sha=str(manifest_diff.get("live_build_sha") or "unknown"),
                expected_build_sha=resolved_sha,
                summary=msg,
            )
        )
        return _build_merge_receipt(
            lang_code=lang_code,
            gateway=clean_gw,
            verdict=verdict,
            manifest_diff=manifest_diff,
            pre_promote=pre_promote,
            comp_doc=comp_doc,
            network_calls=network_calls,
            execute=execute,
            exit_code=exit_code,
            summary=msg,
            diagnostics=[msg],
            extra={
                "promotion_deferred_training_active": deferred,
                "safety_lock_blocked": not deferred,
                "incident_alert_emitted": alert is not None,
                "incident_alert": alert,
                "previous_revision": effective_prev_rev,
                "safety_lock": safety_lock,
            },
        )

    # 6. Build sanitized list-argv rollout plan (dry-run by default unless --execute)
    rollout_plan = build_idempotent_rollout_plan(
        gateway=clean_gw,
        target_sha=resolved_sha,
        previous_revision=effective_prev_rev,
        candidate_tag=clean_tag,
        service_name=clean_service,
        region=clean_region,
        gcp_project=clean_project,
        no_traffic_only=no_traffic_only,
        env=env,
        cwd=eff_cwd,
        lang=lang_code,
    )

    if not execute:
        summary = (
            f"Dry-run deploy-on-merge prêt : {len(rollout_plan['commands'])} commande(s) gcloud planifiée(s) "
            f"(deploy_needed=True, manifest={manifest_diff['manifest_sha256'][:12]}, 0 mutation Cloud Run)."
            if is_fr
            else f"Deploy-on-merge dry-run ready: {len(rollout_plan['commands'])} gcloud command(s) planned "
            f"(deploy_needed=True, manifest={manifest_diff['manifest_sha256'][:12]}, 0 Cloud Run mutations)."
        )
        return _build_merge_receipt(
            lang_code=lang_code,
            gateway=clean_gw,
            verdict="dry_run_plan_ready",
            manifest_diff=manifest_diff,
            pre_promote=pre_promote,
            comp_doc=comp_doc,
            network_calls=network_calls,
            execute=False,
            exit_code=0,
            summary=summary,
            diagnostics=list(manifest_diff.get("diagnostics") or []),
            extra={
                "plan_ready": True,
                "live_build_sha": manifest_diff.get("live_build_sha"),
                "previous_revision": effective_prev_rev,
                "safety_lock": safety_lock,
                "rollout_plan": rollout_plan,
                "commands": rollout_plan["commands"],
            },
        )

    # 7. Execute Step A: --no-traffic --tag deployment
    executed_commands: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    deploy_argv = list(rollout_plan["deploy_no_traffic_argv"])
    deploy_rc, deploy_out = _invoke_runner(gcloud_runner, deploy_argv, GCLOUD_DEPLOY_TIMEOUT_S)
    executed_commands.append(
        {
            "step": "deploy_no_traffic",
            "argv": deploy_argv,
            "returncode": deploy_rc,
            "output_preview": sanitize_text_for_display(deploy_out)[:240],
        }
    )
    if deploy_rc != 0:
        msg = (
            f"Échec du déploiement Cloud Run `--no-traffic` (code={deploy_rc})."
            if is_fr
            else f"Cloud Run `--no-traffic` deploy failed (rc={deploy_rc})."
        )
        diagnostics.append(msg)
        alert = _build_incident_alert(
            gateway=clean_gw,
            verdict="deploy_failed",
            failure_categories=["gcloud_deploy_failed"],
            failed_routes=[],
            diagnostics=diagnostics,
            rollback_triggered=False,
            rollback_succeeded=None,
            rollback_revision=effective_prev_rev,
            live_build_sha=str(manifest_diff.get("live_build_sha") or "unknown"),
            expected_build_sha=resolved_sha,
            summary=msg,
        )
        return _build_merge_receipt(
            lang_code=lang_code,
            gateway=clean_gw,
            verdict="deploy_failed",
            manifest_diff=manifest_diff,
            pre_promote=pre_promote,
            comp_doc=comp_doc,
            network_calls=network_calls,
            execute=True,
            exit_code=2,
            summary=msg,
            diagnostics=diagnostics,
            extra={
                "incident_alert_emitted": True,
                "incident_alert": alert,
                "gcloud_executed": True,
                "gcloud_calls": len(executed_commands),
                "previous_revision": effective_prev_rev,
                "safety_lock": safety_lock,
                "rollout_plan": rollout_plan,
                "executed_commands": executed_commands,
            },
        )

    candidate_rev = _extract_deployed_revision(deploy_out, fallback=f"{clean_service}-{clean_tag}")

    if no_traffic_only:
        summary = (
            f"Révision `{candidate_rev}` préparée avec `--no-traffic --tag {clean_tag}` (0 % trafic basculé)."
            if is_fr
            else f"Staged revision `{candidate_rev}` with `--no-traffic --tag {clean_tag}` (0% live traffic shifted)."
        )
        return _build_merge_receipt(
            lang_code=lang_code,
            gateway=clean_gw,
            verdict="staged_no_traffic_only",
            manifest_diff=manifest_diff,
            pre_promote=pre_promote,
            comp_doc=comp_doc,
            network_calls=network_calls,
            execute=True,
            exit_code=0,
            summary=summary,
            diagnostics=diagnostics,
            extra={
                "staged_no_traffic_only": True,
                "traffic_shifted": False,
                "smoke_skipped_no_traffic_only": True,
                "gcloud_executed": True,
                "gcloud_calls": len(executed_commands),
                "cloud_state_mutated": True,
                "candidate_revision": candidate_rev,
                "candidate_tag": clean_tag,
                "previous_revision": effective_prev_rev,
                "safety_lock": safety_lock,
                "rollout_plan": rollout_plan,
                "executed_commands": executed_commands,
            },
        )

    # 8. Execute Step B: Traffic cutover + Expanded Post-Deploy Smoke Suite
    final_plan = build_idempotent_rollout_plan(
        gateway=clean_gw,
        target_sha=resolved_sha,
        previous_revision=effective_prev_rev,
        candidate_revision=candidate_rev,
        candidate_tag=clean_tag,
        service_name=clean_service,
        region=clean_region,
        gcp_project=clean_project,
        no_traffic_only=False,
        env=env,
        cwd=eff_cwd,
        lang=lang_code,
    )
    promote_argv = list(final_plan["promote_traffic_argv"])
    promote_rc, promote_out = _invoke_runner(gcloud_runner, promote_argv, GCLOUD_TRAFFIC_TIMEOUT_S)
    executed_commands.append(
        {
            "step": "promote_traffic",
            "argv": promote_argv,
            "returncode": promote_rc,
            "output_preview": sanitize_text_for_display(promote_out)[:240],
        }
    )

    smoke_report: dict[str, Any] | None = None
    failure_categories: list[str] = []
    failed_routes: list[str] = []
    if promote_rc != 0:
        failure_categories.append("traffic_promotion_failed")
        diagnostics.append(
            f"Échec `update-traffic` vers `{candidate_rev}` (code={promote_rc})."
            if is_fr
            else f"`update-traffic` to `{candidate_rev}` failed (rc={promote_rc})."
        )
        smoke_passed = False
    else:
        smoke_report = verify_post_deploy_smoke_suite(
            clean_gw,
            expected_sha=resolved_sha,
            probe_feedback_post=probe_feedback_post,
            require_coder_model=require_coder_model,
            previous_revision=effective_prev_rev,
            active_volunteer_gpus=active_volunteer_gpus,
            http_fn=http_fn,
            env=env,
            cwd=eff_cwd,
            lang=lang_code,
        )
        network_calls += int(smoke_report.get("network_calls", 0))
        smoke_passed = bool(smoke_report["smoke_ok"])
        diagnostics.extend(smoke_report.get("diagnostics") or [])
        failure_categories.extend(smoke_report.get("failure_categories") or [])
        failed_routes.extend(smoke_report.get("failed_routes") or [])

    if smoke_passed and smoke_report is not None:
        summary = (
            f"Révision `{candidate_rev}` promue à 100 % et validée par la suite smoke sur {clean_gw} "
            f"(readiness_gates={smoke_report['readiness_gates_passed']}/5)."
            if is_fr
            else f"Promoted revision `{candidate_rev}` to 100% traffic and passed expanded smoke suite on {clean_gw} "
            f"(readiness_gates={smoke_report['readiness_gates_passed']}/5)."
        )
        return _build_merge_receipt(
            lang_code=lang_code,
            gateway=clean_gw,
            verdict="promoted",
            manifest_diff=manifest_diff,
            pre_promote=pre_promote,
            comp_doc=comp_doc,
            network_calls=network_calls,
            execute=True,
            exit_code=0,
            summary=summary,
            diagnostics=diagnostics,
            extra={
                "traffic_shifted": True,
                "promoted": True,
                "canary_ok": True,
                "smoke_ok": True,
                "readiness_projection": smoke_report["readiness_projection"],
                "readiness_gates_passed": smoke_report["readiness_gates_passed"],
                "readiness_gates_total": smoke_report["readiness_gates_total"],
                "gcloud_executed": True,
                "gcloud_calls": len(executed_commands),
                "cloud_state_mutated": True,
                "candidate_revision": candidate_rev,
                "candidate_tag": clean_tag,
                "previous_revision": effective_prev_rev,
                "safety_lock": safety_lock,
                "rollout_plan": final_plan,
                "smoke_suite": smoke_report,
                "canary": smoke_report,
                "executed_commands": executed_commands,
            },
        )

    # 9. Automatic sanitized rollback + incident_alert on smoke/traffic failure
    rollback_succeeded = False
    rollback_argv: list[str] | None = None
    if effective_prev_rev is not None:
        rollback_argv = list(final_plan["rollback_traffic_argv"])
        rb_rc, rb_out = _invoke_runner(gcloud_runner, rollback_argv, GCLOUD_TRAFFIC_TIMEOUT_S)
        rollback_succeeded = rb_rc == 0
        executed_commands.append(
            {
                "step": "rollback_traffic",
                "argv": rollback_argv,
                "returncode": rb_rc,
                "output_preview": sanitize_text_for_display(rb_out)[:240],
            }
        )
    else:
        diagnostics.append(
            "Rollback automatique demandé mais `--previous-revision` absent et aucune révision antérieure connue."
            if is_fr
            else "Automatic rollback requested, but `--previous-revision` was not provided and no prior live revision was found."
        )

    summary = (
        f"ROLLBACK DÉCLENCHÉ vers `{effective_prev_rev or 'unknown'}` après échec de la suite smoke "
        f"(succeeded={rollback_succeeded})."
        if is_fr
        else f"ROLLBACK TRIGGERED to `{effective_prev_rev or 'unknown'}` after post-deploy smoke suite failure "
        f"(succeeded={rollback_succeeded})."
    )
    alert = _build_incident_alert(
        gateway=clean_gw,
        verdict="rolled_back_after_smoke_failure",
        failure_categories=failure_categories,
        failed_routes=failed_routes,
        diagnostics=diagnostics,
        rollback_triggered=True,
        rollback_succeeded=rollback_succeeded,
        rollback_revision=effective_prev_rev,
        live_build_sha=str(
            (smoke_report or {}).get("live_build_sha") or manifest_diff.get("live_build_sha") or "unknown"
        ),
        expected_build_sha=resolved_sha,
        summary=summary,
    )
    readiness_proj = (
        smoke_report["readiness_projection"]
        if smoke_report is not None
        else project_adoption_readiness_gates(gateway_and_models_ok=False, holiday_routes_ok=False, lang=lang_code)
    )
    return _build_merge_receipt(
        lang_code=lang_code,
        gateway=clean_gw,
        verdict="rolled_back_after_smoke_failure",
        manifest_diff=manifest_diff,
        pre_promote=pre_promote,
        comp_doc=comp_doc,
        network_calls=network_calls,
        execute=True,
        exit_code=2,
        summary=summary,
        diagnostics=diagnostics,
        extra={
            "traffic_shifted": promote_rc == 0,
            "rollback_triggered": True,
            "rollback_succeeded": rollback_succeeded,
            "rollback_revision": effective_prev_rev,
            "rollback_command_argv": rollback_argv,
            "incident_alert_emitted": True,
            "incident_alert": alert,
            "readiness_projection": readiness_proj,
            "readiness_gates_passed": readiness_proj["readiness_gates_passed"],
            "readiness_gates_total": readiness_proj["readiness_gates_total"],
            "gcloud_executed": True,
            "gcloud_calls": len(executed_commands),
            "cloud_state_mutated": True,
            "candidate_revision": candidate_rev,
            "candidate_tag": clean_tag,
            "previous_revision": effective_prev_rev,
            "safety_lock": safety_lock,
            "rollout_plan": final_plan,
            "smoke_suite": smoke_report,
            "canary": smoke_report,
            "executed_commands": executed_commands,
        },
    )


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render an FR/EN GitHub-flavored Markdown report with masked secrets, `~` paths, and SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    header = (
        "# Promoteur Idempotent Cloud Run Noël & Verrou d'Entraînement HiveMind"
        if is_fr
        else "# HiveMind Holiday Cloud Run Revision Promoter & Training Safety Lock"
    )
    lines: list[str] = [
        header,
        "",
        f"> **{'Conformité' if is_fr else 'Compliance'}**: {COMPLIANCE_NOTICE_BILINGUAL}",
        "",
        f"- **Gateway**: `{report.get('gateway', DEFAULT_GATEWAY)}`",
        f"- **Mode / Verdict**: `{report.get('mode')}` / `{report.get('verdict', report.get('mode'))}`",
        f"- **Compliance OK**: `{report.get('compliance_ok')}`",
        f"- **Pre-Promote & FinOps OK**: `{report.get('pre_promote_ok')}`",
        f"- **Drift Detected**: `{report.get('drift_detected')}`",
        f"- **Idempotent No-Op**: `{report.get('idempotent_noop')}` (`already_at_target_revision={report.get('already_at_target_revision')}`)",
        f"- **Deferred (Training Active)**: `{report.get('promotion_deferred_training_active')}`",
        f"- **Promoted**: `{report.get('promoted')}` | **Rollback Triggered**: `{report.get('rollback_triggered')}`",
        f"- **Dry-Run Only**: `{report.get('dry_run_only')}` (`gcloud_calls={report.get('gcloud_calls', 0)}`)",
    ]

    if "manifest_sha256" in report:
        lines.append(
            f"- **Deploy Needed**: `{report.get('deploy_needed')}` | **Manifest SHA-256**: `{str(report.get('manifest_sha256'))[:16]}`"
        )
    if "readiness_gates_passed" in report:
        lines.append(
            f"- **Readiness Gates**: `{report.get('readiness_gates_passed')}/{report.get('readiness_gates_total', 5)}`"
        )

    if "canary" in report and isinstance(report["canary"], Mapping):
        c_doc = report["canary"]
        lines.append(
            f"- **Canary OK**: `{c_doc.get('canary_ok')}` (`unique_csp_nonces={c_doc.get('unique_csp_nonces_count')}`)"
        )
    elif "smoke_ok" in report:
        lines.append(
            f"- **Smoke OK**: `{report.get('smoke_ok')}` (`models_ok={report.get('models_ok')}`, `unique_csp_nonces={report.get('unique_csp_nonces_count')}`)"
        )

    alert = report.get("incident_alert")
    if isinstance(alert, Mapping):
        lines.extend(
            [
                "",
                "## Alerte Incident Cloud Run" if is_fr else "## Cloud Run Incident Alert",
                "",
                f"- **Severity**: `{alert.get('severity')}` | **Categories**: `{', '.join(alert.get('failure_categories') or ())}`",
                f"- **Rollback Revision**: `{alert.get('rollback_revision')}` (`succeeded={alert.get('rollback_succeeded')}`)",
            ]
        )

    commands = report.get("commands")
    if isinstance(commands, Sequence) and commands:
        lines.extend(["", "## Planned `gcloud` Commands (Dry-Run)" if not is_fr else "## Commandes `gcloud` Planifiées (Dry-Run)", "", "```bash"])
        for cmd in commands:
            if isinstance(cmd, Sequence) and not isinstance(cmd, str):
                lines.append(_format_cmd_display([str(x) for x in cmd]))
        lines.append("```")

    executed = report.get("executed_commands")
    if isinstance(executed, Sequence) and executed:
        lines.extend(["", "## Executed `gcloud` Steps" if not is_fr else "## Étapes `gcloud` Exécutées", ""])
        for row in executed:
            if isinstance(row, Mapping):
                lines.append(
                    f"- `{row.get('step')}` (rc=`{row.get('returncode')}`): `{_format_cmd_display([str(x) for x in row.get('argv') or ()])}`"
                )

    diags = report.get("diagnostics")
    if isinstance(diags, Sequence) and diags:
        lines.extend(["", "## Diagnostics", ""])
        for item in diags:
            lines.append(f"- {sanitize_text_for_display(str(item))}")

    lines.extend(["", f"- **Summary**: {sanitize_text_for_display(str(report.get('summary') or ''))}"])
    lines.extend(["", f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`"])
    return _assert_safe_output(sanitize_text_for_display("\n".join(lines)))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_cloud_run_revision_promoter.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "Idempotent Cloud Run holiday revision promoter, active-training safety lock, "
            "and /noel+/feedback canary verifier (100% personal PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument("--gateway", default=default_gw, help="HiveMind gateway URL (https://*.run.app or loopback).")
    parser.add_argument("--expected-sha", "--target-sha", dest="expected_sha", default=None, help="Target HIVEMIND_BUILD_SHA.")
    parser.add_argument("--check-git-head", action="store_true", help="Resolve target SHA from local `git rev-parse HEAD`.")
    parser.add_argument("--previous-revision", default=None, help="Previous known-good K_REVISION for automatic rollback.")
    parser.add_argument("--candidate-tag", default=DEFAULT_CANDIDATE_TAG, help="Cloud Run canary traffic tag.")
    parser.add_argument("--service", default=DEFAULT_CLOUD_RUN_SERVICE, help="Cloud Run service name.")
    parser.add_argument("--region", default=DEFAULT_CLOUD_RUN_REGION, help="Cloud Run region.")
    parser.add_argument("--project", default=None, help="Optional GCP project id.")
    parser.add_argument("--repo-root", default="", help="Optional repository root path for manifest diff and git inspection.")
    parser.add_argument("--deploy-script", default="", help="Optional path to deploy/gcp_openrouter/deploy.sh.")
    parser.add_argument("--server-py", default="", help="Optional path to deploy/gcp_openrouter/server.py.")
    parser.add_argument("--lock-file", default="", help="Optional local training lockfile path.")
    parser.add_argument("--deploy-on-merge", action="store_true", help="Run end-to-end deploy-on-merge orchestrator.")
    parser.add_argument("--smoke-suite", action="store_true", help="Run expanded post-deploy smoke suite (/v1/models, /noel, /quick, /raid-noel, /arena-noel, feedback).")
    parser.add_argument("--force-deploy", action="store_true", help="Force deploy even when git diff only touched non-Cloud-Run paths.")
    parser.add_argument("--require-coder-model", action="store_true", help="Require hivemind/coder presence in GET /v1/models.")
    parser.add_argument("--execute", action="store_true", help="Execute gcloud rollout commands (defaults to safe dry-run).")
    parser.add_argument("--no-traffic-only", action="store_true", help="Stage tagged revision with --no-traffic without shifting live traffic.")
    parser.add_argument("--allow-during-training", action="store_true", help="Allow live traffic shift even when active training signals are detected.")
    parser.add_argument("--defer-on-active-training", action="store_true", help="Defer safely with exit code 3 if active training would be disturbed.")
    parser.add_argument("--probe-feedback-post", action="store_true", help="Include idempotent zero-PII POST /v1/hivemind/onboarding/feedback in canary.")
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output with deterministic receipt_sha256.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output (default).")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    gcloud_runner: RunnerFn = default_gcloud_runner,
    http_fn: HttpFn = default_http_request,
    git_runner: RunnerFn = default_git_runner,
    app_factory: Callable[..., Any] | None = None,
    process_probe_fn: Callable[[], Iterable[str]] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """CLI entrypoint returning `0` (ready/promoted/noop), `2` (error/compliance/rollback), or `3` (deferred)."""
    out_stream = sys.stdout if stdout is None else stdout
    err_stream = sys.stderr if stderr is None else stderr
    args = build_parser().parse_args(argv)

    try:
        lang_code = normalize_lang(args.lang)
        repo_root_arg = Path(args.repo_root) if args.repo_root else None
        deploy_path_arg = Path(args.deploy_script) if args.deploy_script else None
        server_path_arg = Path(args.server_py) if args.server_py else None
        lock_path_arg = Path(args.lock_file) if args.lock_file else None
        if args.deploy_on_merge:
            report = orchestrate_deploy_on_merge(
                args.gateway,
                repo_root=repo_root_arg,
                target_sha=args.expected_sha,
                previous_revision=args.previous_revision,
                candidate_tag=args.candidate_tag,
                service_name=args.service,
                region=args.region,
                gcp_project=args.project,
                deploy_script_path=deploy_path_arg,
                server_py_path=server_path_arg,
                app_factory=app_factory,
                lock_file_path=lock_path_arg,
                process_probe_fn=process_probe_fn,
                check_git_head=bool(args.check_git_head),
                force_deploy=bool(args.force_deploy),
                execute=bool(args.execute),
                no_traffic_only=bool(args.no_traffic_only),
                allow_during_training=bool(args.allow_during_training),
                defer_on_active_training=bool(args.defer_on_active_training),
                probe_feedback_post=bool(args.probe_feedback_post),
                require_coder_model=bool(args.require_coder_model),
                gcloud_runner=gcloud_runner,
                http_fn=http_fn,
                git_runner=git_runner,
                env=env,
                cwd=cwd,
                lang=lang_code,
            )
        elif args.smoke_suite:
            report = verify_post_deploy_smoke_suite(
                args.gateway,
                expected_sha=args.expected_sha,
                probe_feedback_post=bool(args.probe_feedback_post),
                require_coder_model=bool(args.require_coder_model),
                previous_revision=args.previous_revision,
                http_fn=http_fn,
                env=env,
                cwd=cwd,
                lang=lang_code,
            )
        else:
            report = promote_holiday_revision(
                args.gateway,
                target_sha=args.expected_sha,
                previous_revision=args.previous_revision,
                candidate_tag=args.candidate_tag,
                service_name=args.service,
                region=args.region,
                gcp_project=args.project,
                deploy_script_path=deploy_path_arg,
                server_py_path=server_path_arg,
                app_factory=app_factory,
                lock_file_path=lock_path_arg,
                process_probe_fn=process_probe_fn,
                check_git_head=bool(args.check_git_head),
                execute=bool(args.execute),
                no_traffic_only=bool(args.no_traffic_only),
                allow_during_training=bool(args.allow_during_training),
                defer_on_active_training=bool(args.defer_on_active_training),
                probe_feedback_post=bool(args.probe_feedback_post),
                gcloud_runner=gcloud_runner,
                http_fn=http_fn,
                git_runner=git_runner,
                env=env,
                cwd=cwd,
                lang=lang_code,
            )
        rendered = (
            _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False))
            if args.json
            else render_markdown_report(report)
        )
        print(rendered, file=out_stream)
        return int(report.get("exit_code", 0))
    except RevisionPromoterError as exc:
        safe_msg = sanitize_text_for_display(str(exc))
        err_report = _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "campaign_id": CAMPAIGN_ID,
                "mode": "error",
                "compliance_ok": True,
                "exit_code": 2,
                "error": safe_msg,
                "zero_inference_verified": True,
            }
        )
        if args.json:
            print(_assert_safe_output(json.dumps(err_report, indent=2, ensure_ascii=False)), file=out_stream)
        else:
            print(f"Error: {safe_msg}", file=err_stream)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
