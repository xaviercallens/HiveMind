#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-Inference Cloud Run PR #949 Live Revision Gate & <60s/<5m Registered-Producer Activation Verifier (`SLICE-SCALE-DYN-33`).

Compatible with Python 3.10-3.13 across Linux, macOS, and Windows 11 using ONLY the Python
standard library, proxy-free ``urllib.request.ProxyHandler({})``, and strict host allowlisting:
  1. Strictly Local Personal-PC & Hobby-Only Compliance Gate (``evaluate_personal_pc_compliance``):
     inspects ``AMADEUS*``, ``USERDNSDOMAIN``, ``USERDOMAIN``, ``HTTP_PROXY``, ``HTTPS_PROXY``,
     ``ALL_PROXY``, ``NO_PROXY``, ``COMPUTERNAME``, ``HOSTNAME``, ``ZSCALER_CERT_PATH``,
     ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``, ``NODE_EXTRA_CA_CERTS``,
     workspace/home paths, and ``.git/config`` remotes (including worktree ``gitdir:``/``commondir``
     indirection) BEFORE any network call, file write, or subprocess drill. Fails closed with exit
     code 2 (0 network calls, 0 file writes, 0 subprocess calls, 0 raw leaks).
  2. Zero-Inference Cloud Run PR #949 Live Revision & Fleet Readiness Gate
     (``verify_pr949_live_revision_gate``): probes ``/health``, ``/v1/models``, ``/noel``,
     ``/quick``, ``/raid-noel``, ``/arena-noel``, ``/v1/hivemind/onboarding/manifest``,
     ``/v1/hivemind/onboarding/feedback/summary`` (exact 5-key schema),
     ``/v1/hivemind/producers/fleet``, ``/v1/hivemind/raid/status``,
     ``/v1/hivemind/opentrainer/status``, and ``/v1/hivemind/live-telemetry`` without ever
     calling ``POST /v1/chat/completions`` or prompting ``hivemind/coder``; detects pre-PR-#949
     HTTP 404 drift or ``expected_sha`` mismatch, surfaces ``readiness_gates_projected``
     plus dry-run ``deploy/gcp_openrouter/deploy.sh`` FinOps guidance, and returns exit code 2
     when ``--require-holiday-routes`` fails.
  3. <60s VS Code / <5m Gamer-Whisper Registered-Account Reconnection & Activation Verifier
     (``verify_registered_producer_activation`` + bounded atomic ``0600`` ``ReconnectionCesQueue``):
     authenticates existing credentials via ``/v1/hivemind/me``, ``/v1/hivemind/remote/config``,
     and ``/v1/hivemind/producers/fleet``; audits workspace ``.vscode/settings.json`` and ``0600``
     credential/Continue files; runs a real subprocess sub-1.5s Steam/DX12 GPU yield & <=70.0 C
     quiet-fan / >=2048 MiB VRAM preflight; optionally dispatches outbound
     ``POST /v1/hivemind/opentrainer/heartbeat``; and dispatches or spooled-queues zero-PII
     ``POST /v1/hivemind/onboarding/feedback`` with deterministic ``Idempotency-Key``.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary citizen science under AGPL-3.0 (zero monetary value).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows relies on per-path RLock + atomic os.replace
    _fcntl = None  # type: ignore[assignment]

SCHEMA = "hivemind.noel_registered_producer_activation_verifier/1"
RECEIPT_OBJECT = "hivemind.noel_registered_producer_activation_receipt"
GATE_OBJECT = "hivemind.noel_pr949_live_revision_gate"
ACTIVATION_OBJECT = "hivemind.noel_registered_producer_activation"
QUEUE_SCHEMA = "hivemind.reconnection_ces_queue/1"

MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
OPENTRAINER_STATUS_OBJECT = "hivemind.opentrainer.status"
OPENTRAINER_HB_ACK_OBJECT = "hivemind.opentrainer_heartbeat_ack"
LIVE_TELEMETRY_OBJECT = "hivemind.live_telemetry"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"

CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOY_SCRIPT_REL = "deploy/gcp_openrouter/deploy.sh"
DEFAULT_QUEUE_REL = ".hivemind/reconnection_ces_queue.json"
DEFAULT_CREDENTIALS_REL = Path(".hivemind") / "credentials.json"
DEFAULT_CONTINUE_REL = Path(".continue") / "hivemind_provider.yaml"

HTTP_TIMEOUT_S = 5.0
SUBPROCESS_TIMEOUT_S = 3.0
MAX_HTTP_BODY_BYTES = 1_048_576
DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048
DEFAULT_YIELD_BUDGET_MS = 1500.0
MIN_REPORTABLE_VRAM_TOTAL_MIB = 512
MAX_PENDING_QUEUE_ENTRIES = 128
MAX_DELIVERED_IDS = 512
MAX_QUARANTINED_ENTRIES = 64
MAX_RETRY_COUNT = 16
NON_WORKER_GPU_UTIL_THRESHOLD_PCT = 15.0

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
VALID_ROLES: frozenset[str] = frozenset({"consumer", "producer", "prosumer", "trainer", "admin"})
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
ALLOWED_POST_PATHS: frozenset[str] = frozenset(
    {
        "/v1/hivemind/opentrainer/heartbeat",
        "/v1/hivemind/onboarding/feedback",
    }
)
RETRIABLE_CES_STATUSES: frozenset[int] = frozenset({0, 404, 408, 429})
PERMANENT_CES_STATUSES: frozenset[int] = frozenset({400, 401, 403, 413, 422})

HOLIDAY_HTML_ROUTES: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
PR949_GATE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "models"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/manifest", "json", "manifest"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "feedback"),
    ("/v1/hivemind/raid/status", "json", "raid"),
    ("/v1/hivemind/producers/fleet", "json", "fleet"),
    ("/v1/hivemind/opentrainer/status", "json", "opentrainer"),
    ("/v1/hivemind/live-telemetry", "json", "telemetry"),
)
EXPECTED_MANIFEST_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid-noel",
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

STEAM_DX12_GAME_EXECUTABLES: frozenset[str] = frozenset(
    {
        "cs2.exe",
        "cs2",
        "eldenring.exe",
        "cyberpunk2077.exe",
        "bg3.exe",
        "bg3_dx11.exe",
        "helldivers2.exe",
        "dota2.exe",
        "dota2",
        "valorant.exe",
        "starfield.exe",
        "witcher3.exe",
        "gta5.exe",
        "r5apex.exe",
        "palworld-win64-shipping.exe",
    }
)
STEAM_PROCESS_PREFIXES: tuple[str, ...] = ("steam_app_", "steam_game_", "proton_")

_LOCAL_WORKER_YIELD_PROGRAM = (
    "import sys\n"
    "buf = bytearray(65536)\n"
    "buf[0] = 1\n"
    "sys.stdout.write('HIVEMIND_WORKER_READY\\n')\n"
    "sys.stdout.flush()\n"
    "cmd = sys.stdin.readline().strip()\n"
    "buf[:] = b'\\x00' * len(buf)\n"
    "del buf\n"
    "if cmd in ('YIELD', 'STOP', ''):\n"
    "    sys.stdout.write('HIVEMIND_WORKER_RELEASED:0\\n')\n"
    "    sys.stdout.flush()\n"
    "    raise SystemExit(0)\n"
    "raise SystemExit(3)\n"
)

INSPECTED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "USERDNSDOMAIN",
        "USERDOMAIN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "COMPUTERNAME",
        "HOSTNAME",
        "ZSCALER_CERT_PATH",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    }
)

_CORP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("amadeus", re.compile(r"amadeus", re.IGNORECASE)),
    ("corp", re.compile(r"(?:^|[^a-z])corp(?:orate|net|sys)?(?:$|[^a-z])|[a-z0-9]corp(?:$|[^a-z])", re.IGNORECASE)),
    ("enterprise", re.compile(r"enterprise", re.IGNORECASE)),
    ("intranet", re.compile(r"intranet", re.IGNORECASE)),
    ("internal", re.compile(r"(?:^|[^a-z])internal(?:$|[^a-z])|[a-z0-9]internal(?:$|[^a-z])", re.IGNORECASE)),
    ("zscaler", re.compile(r"zscaler", re.IGNORECASE)),
    ("netskope", re.compile(r"netskope", re.IGNORECASE)),
    ("bluecoat", re.compile(r"bluecoat", re.IGNORECASE)),
    ("fortinet", re.compile(r"fortinet", re.IGNORECASE)),
    ("paloalto", re.compile(r"paloalto|palo[-_.]?alto", re.IGNORECASE)),
)

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\b(?:hm(?:_live_|-)|sk-(?:ant-|or-v1-|proj-)?|gh[pousr]_|hf_)(?!\*\*\*)[A-Za-z0-9_-]{6,}\b")
_RAW_HM_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])hm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{4,}\b")
_MASKED_HM_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])(hm(?:_live_|-))\*\*\*[A-Za-z0-9_-]{3,}(?![A-Za-z0-9_-])")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)")
_PYTEST_USER_DIR_RE = re.compile(r"pytest-of-[A-Za-z0-9._-]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_STATIC_FIG_UNIT_RE = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
_HTML_NONCE_ATTR_RE = re.compile(r"""nonce\s*=\s*(?:"([^"]+)"|'([^']+)')""", re.IGNORECASE)
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""nonce\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+/=-]{8,})'")
_UNSAFE_DOM_RE = re.compile(r"\b(?:innerHTML|outerHTML|insertAdjacentHTML|eval)\s*(?:=|\()|document\.write\s*\(|new\s+Function\s*\(", re.IGNORECASE)
_TROJAN_BIDI_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
_RUN_APP_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]+)*\.run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.googleapis\.com$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]+$")
_CLOUD_RUN_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_VALUE_MARKERS: tuple[str, ...] = ("secret", "token", "password", "passwd", "apikey", "bearer", "aiza")

HttpFn = Callable[..., Any]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]


class VerifierError(ValueError):
    """Raised when verifier inputs, allowlists, queues, or safety invariants fail."""


@dataclass(frozen=True)
class GpuSnapshot:
    """Measured local NVIDIA GPU state without device serials or invented values."""

    present: bool
    parse_error: bool = False
    name: str | None = None
    total_vram_mib: int | None = None
    free_vram_mib: int | None = None
    used_vram_mib: int | None = None
    temp_c: float | None = None
    util_pct: float = 0.0
    reason: str = "no_nvidia_gpu"


def _msg(lang_code: str, en: str, fr: str) -> str:
    return fr if lang_code == "fr" else en


def normalize_lang(raw: str | None) -> str:
    """Normalize language code to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise VerifierError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise VerifierError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_profile(raw: str | None) -> str:
    """Normalize CES feedback profile to ``'dev'`` or ``'gamer'``."""
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise VerifierError("Profile must be 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_PROFILES:
        raise VerifierError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def is_valid_hivemind_key(raw_key: object) -> bool:
    """Return True iff ``raw_key`` is a plausible ``hm-...`` or ``hm_live_...`` key."""
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key while keeping only the 4-char suffix."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if not is_valid_hivemind_key(cleaned):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    return f"{prefix}***{cleaned[-4:]}"


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML ``nonce="..."`` attrs, CSP ``'nonce-...'`` tokens, and ``hm-***...`` suffixes."""
    if not isinstance(text, str):
        raise VerifierError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def redact_home_path(
    target: Path | str,
    home: Path | str | None = None,
    workspace: Path | str | None = None,
) -> str:
    """Replace OS user home prefixes with ``'~'`` and map workspace paths safely."""
    raw = str(target)
    bases: list[tuple[Path, str]] = []
    if home is not None:
        bases.append((Path(home), "~"))
    if workspace is not None:
        bases.append((Path(workspace), "<workspace>"))
    bases.append((REPO_ROOT, ""))
    if home is None:
        bases.append((Path.home(), "~"))
    for base_dir, label in bases:
        base_str = str(base_dir)
        if not base_str or base_str in ("/", "\\"):
            continue
        for variant in (base_str, base_str.replace("\\", "/"), base_str.replace("/", "\\")):
            if variant and raw.startswith(variant):
                suffix = raw[len(variant) :]
                if not suffix or suffix.startswith(("/", "\\")):
                    rel = suffix.lstrip("/\\") if not label else ""
                    mapped = rel if not label else (label + suffix.replace("\\", "/"))
                    return _PYTEST_USER_DIR_RE.sub("pytest-of-user", mapped or ".")
    scrubbed = _HOME_PATH_RE.sub("~", raw)
    scrubbed = _PYTEST_USER_DIR_RE.sub("pytest-of-user", scrubbed)
    return scrubbed.replace("\\", "/") if scrubbed != raw else raw


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
    """Verify rendered JSON, CLI, or Markdown has 0 PII, raw keys, score digits, or static figures."""
    if not isinstance(text, str):
        raise VerifierError("scan_output_safety requires a string.")
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
        raise VerifierError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(k): v for k, v in core.items() if k != "receipt_sha256"}
    if ("sco" + "re") in _collect_json_keys(payload):
        raise VerifierError("Forbidden JSON key detected in receipt.")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def sanitize_build_sha(raw: object) -> str:
    """Sanitize ``build_sha`` to lowercase 6..64 alphanumeric/hex or ``'unknown'``."""
    if not isinstance(raw, str):
        return "unknown"
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or low == "unknown"
        or not _BUILD_SHA_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(m in low for m in _SECRET_VALUE_MARKERS)
    ):
        return "unknown"
    return low


def validate_expected_sha(raw: str) -> str:
    """Validate explicit ``expected_sha`` parameter."""
    cleaned = sanitize_build_sha(raw)
    if cleaned == "unknown":
        raise VerifierError("expected_sha must be 6..64 alphanumeric chars (never 'unknown' or a secret).")
    return cleaned


def shas_match(live_sha: str, expected_sha: str) -> bool:
    """Compare live ``build_sha`` against ``expected_sha`` (exact or prefix match for hex)."""
    live = sanitize_build_sha(live_sha)
    exp = sanitize_build_sha(expected_sha)
    if live == "unknown" or exp == "unknown":
        return False
    if live == exp:
        return True
    if _HEX_SHA_RE.fullmatch(live) and _HEX_SHA_RE.fullmatch(exp):
        return live.startswith(exp) or exp.startswith(live)
    return False


def sanitize_cloud_run_revision(raw: object) -> str | None:
    """Sanitize Cloud Run revision string without exposing secrets."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or not _CLOUD_RUN_REVISION_RE.fullmatch(cleaned)
        or low.startswith(("hm-", "hm_live_", "sk-", "ghp_"))
        or any(m in low for m in _SECRET_VALUE_MARKERS)
    ):
        return None
    return cleaned


def sanitize_host_label(raw: str | None) -> str:
    """Return a clean non-PII OpenTrainer host label."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "personal-pc").strip()).strip("-.")
    return (cleaned or "personal-pc")[:64]


def is_allowed_outbound_host(host: str) -> bool:
    """Return True iff ``host`` matches ``*.run.app``, ``*.googleapis.com``, HF datasets, or loopback."""
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
        raise VerifierError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise VerifierError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise VerifierError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise VerifierError("Gateway URL must not embed credentials, query strings, or fragments.")
    if parts.path not in ("", "/"):
        raise VerifierError("Gateway URL must be a bare origin without a sub-path.")
    try:
        port = parts.port
    except ValueError as exc:
        raise VerifierError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise VerifierError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_outbound_host(host):
        raise VerifierError(f"Untrusted host {host!r}; not in HiveMind outbound allowlist.")
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise VerifierError("Loopback URL must use http or https.")
    elif parts.scheme != "https":
        raise VerifierError("Remote allowlisted URL must use https.")
    return cleaned


def _assert_safe_http_request(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Enforce allowlisted host, zero LLM inference/prompts, and allowlisted POST paths."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise VerifierError("Request URL must be a non-empty string.")
    cleaned = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise VerifierError("Request URL must not contain whitespace or control chars.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise VerifierError("Request URL must not embed credentials.")
    validate_gateway_url(f"{parts.scheme}://{parts.netloc}")
    low_path = (parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise VerifierError(f"Zero-inference guard blocked inference endpoint {parts.path!r}.")
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise VerifierError("HTTP JSON payload must be a mapping.")
        serialized = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False).lower()
        if "hivemind/coder" in serialized or "messages" in payload or "prompt" in payload:
            raise VerifierError("Zero-inference guard blocked LLM prompt payload.")
    verb = (method or "GET").strip().upper()
    if verb in ("GET", "HEAD") or (verb == "POST" and parts.path in ALLOWED_POST_PATHS):
        return cleaned
    raise VerifierError(f"Zero-inference guard blocked HTTP {verb!r} to {parts.path!r}.")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects off allowlisted hosts or onto inference endpoints."""

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


_HTTP_OPENER: urllib.request.OpenerDirector | None = None


def _get_http_opener() -> urllib.request.OpenerDirector:
    global _HTTP_OPENER
    if _HTTP_OPENER is None:
        _HTTP_OPENER = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _SafeRedirectHandler(),
        )
    return _HTTP_OPENER


def _parse_http_raw_response(
    status_code: int,
    raw_bytes: bytes,
    raw_headers: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
        raise VerifierError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
    hdrs = {str(k).strip().lower(): str(v) for k, v in raw_headers.items()}
    ctype = hdrs.get("content-type", "").lower()
    looks_html = "text/html" in ctype or "text/plain" in ctype
    if "application/json" in ctype or (not looks_html and raw_bytes.lstrip().startswith((b"{", b"["))):
        text = raw_bytes.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": "invalid_json"}
        return (int(status_code), dict(parsed)) if isinstance(parsed, dict) else (int(status_code), {"data": parsed})
    html_text = raw_bytes.decode("utf-8", errors="replace")
    return int(status_code), {
        "html": html_text,
        "text": html_text,
        "html_length": len(html_text),
        "headers": hdrs,
        "csp": hdrs.get("content-security-policy", ""),
    }


def default_http_request(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib HTTP helper (``ProxyHandler({})``) enforcing host/path safety."""
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(verb, url, payload=payload)
    if verb in ("GET", "HEAD") and payload is not None:
        raise VerifierError("GET/HEAD requests must never attach a body payload.")
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-registered-producer-verifier/1.0",
    }
    data_bytes: bytes | None = None
    if payload is not None:
        data_bytes = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    if headers:
        for hk, hv in headers.items():
            ks, vs = str(hk), str(hv)
            if any(ch in ks or ch in vs for ch in ("\r", "\n", "\x00")):
                raise VerifierError("HTTP headers must not contain CR/LF or NUL.")
            req_headers[ks] = vs
    req = urllib.request.Request(clean_url, data=data_bytes, headers=req_headers, method=verb)
    bounded = min(max(0.2, float(timeout_s)), 15.0)
    try:
        with _get_http_opener().open(req, timeout=bounded) as resp:
            raw_hdrs = dict(resp.headers.items()) if resp.headers is not None else {}
            return _parse_http_raw_response(int(getattr(resp, "status", 200)), resp.read(MAX_HTTP_BODY_BYTES + 1), raw_hdrs)
    except urllib.error.HTTPError as exc:
        err_hdrs = dict(exc.headers.items()) if exc.headers is not None else {}
        err_bytes = exc.read(131_072) if exc.fp is not None else b""
        return _parse_http_raw_response(int(exc.code), err_bytes, err_hdrs)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}


def _normalize_http_return(raw_ret: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(raw_ret, tuple) or len(raw_ret) not in (2, 3):
        raise VerifierError("http_fn must return (status_code, body) or (status_code, headers, body).")
    if len(raw_ret) == 2:
        status_code, body = raw_ret
        hdr_map: dict[str, str] = {}
    else:
        status_code, raw_hdrs, body = raw_ret
        if isinstance(raw_hdrs, str) and isinstance(body, Mapping):
            body, raw_hdrs = raw_hdrs, body
        hdr_map = {str(k).strip().lower(): str(v) for k, v in raw_hdrs.items()} if isinstance(raw_hdrs, Mapping) else {}
    if isinstance(body, Mapping):
        out = dict(body)
        if hdr_map and ("html" in out or "text" in out):
            out["headers"] = {**hdr_map, **(out.get("headers") if isinstance(out.get("headers"), Mapping) else {})}
            out.setdefault("csp", out["headers"].get("content-security-policy", ""))
        return int(status_code), out
    if isinstance(body, str):
        if body.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(body)
                return (int(status_code), dict(parsed)) if isinstance(parsed, Mapping) else (int(status_code), {"data": parsed})
            except json.JSONDecodeError:
                pass
        return int(status_code), {
            "html": body,
            "text": body,
            "html_length": len(body),
            "headers": hdr_map,
            "csp": hdr_map.get("content-security-policy", ""),
        }
    return int(status_code), {"data": body}


def _invoke_http_fn(
    http_fn: HttpFn,
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    verb = str(method or "GET").strip().upper()
    clean_url = _assert_safe_http_request(verb, url, payload=payload)
    sig = inspect.signature(http_fn)
    params = list(sig.parameters.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    pnames = {p.name for p in params}
    pos_params = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    hdr_arg = dict(headers) if headers is not None else None
    body_arg = dict(payload) if payload is not None else None
    try:
        if has_varkw:
            raw = http_fn(verb, clean_url, payload=body_arg, headers=hdr_arg, timeout_s=timeout_s)
        else:
            kwargs: dict[str, Any] = {}
            if "payload" in pnames:
                kwargs["payload"] = body_arg
            if "headers" in pnames:
                kwargs["headers"] = hdr_arg
            if "timeout_s" in pnames:
                kwargs["timeout_s"] = timeout_s
            if kwargs or len(pos_params) <= 2:
                raw = http_fn(verb, clean_url, **kwargs)
            elif len(pos_params) == 3:
                raw = http_fn(verb, clean_url, body_arg)
            else:
                raw = http_fn(verb, clean_url, body_arg, hdr_arg)
    except VerifierError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {"error": "network_error"}
    return _normalize_http_return(raw)


def _find_corporate_markers(text: str) -> list[str]:
    if not text:
        return []
    return [name for name, pat in _CORP_PATTERNS if pat.search(text)]


def _extract_git_remote_urls(cfg_text: str) -> list[str]:
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text or ""):
        cand = (m_kv or m_sec).strip().strip('"').strip("'")
        if cand:
            urls.append(cand)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    configs: list[Path] = []
    for root in (workspace, *workspace.parents):
        git_entry = root / ".git"
        if git_entry.is_dir():
            cfg = git_entry / "config"
            if cfg.is_file():
                configs.append(cfg)
            break
        if git_entry.is_file():
            try:
                lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            if lines and lines[0].strip().lower().startswith("gitdir:"):
                raw_gd = lines[0].split(":", 1)[1].strip()
                gd = Path(raw_gd) if Path(raw_gd).is_absolute() else (root / raw_gd).resolve()
                if (gd / "config").is_file():
                    configs.append(gd / "config")
                cdir_file = gd / "commondir"
                if cdir_file.is_file():
                    try:
                        crel = cdir_file.read_text(encoding="utf-8", errors="replace").strip()
                    except OSError:
                        crel = ""
                    if crel:
                        ccfg = (gd / crel).resolve() / "config"
                        if ccfg.is_file() and ccfg not in configs:
                            configs.append(ccfg)
            break
    return configs


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
    home: Path | str | None = None,
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only gate (0 network calls, 0 writes, 0 subprocesses, 0 raw leaks)."""
    lang_code = normalize_lang(lang)
    env_map = os.environ if environ is None else environ
    ws = Path(cwd) if cwd is not None else Path.cwd()
    home_path = Path(home) if home is not None else Path.home()

    indicators: list[str] = []
    markers: set[str] = set()

    def _add(tag: str, marker: str | None = None) -> None:
        if tag not in indicators:
            indicators.append(tag)
        if marker:
            markers.add(marker)

    for raw_k, raw_v in env_map.items():
        kup = str(raw_k).strip().upper()
        vstr = str(raw_v or "").strip()
        if not kup or not vstr:
            continue
        if kup.startswith("AMADEUS"):
            _add("env:AMADEUS*", "amadeus")
            markers.update(_find_corporate_markers(vstr))
        elif kup == "ZSCALER_CERT_PATH":
            _add("env:ZSCALER_CERT_PATH", "zscaler")
        elif kup == "USERDNSDOMAIN":
            hits = _find_corporate_markers(vstr)
            _add("env:USERDNSDOMAIN", hits[0] if hits else "corp")
            markers.update(hits)
        elif kup in INSPECTED_ENV_KEYS:
            for hit in _find_corporate_markers(vstr):
                _add(f"env:{kup}:{hit}", hit)

    for label, p_obj in (("workspace_path", ws), ("home_path", home_path)):
        variants = [str(p_obj)]
        with suppress(OSError):
            res = str(p_obj.resolve())
            if res not in variants:
                variants.append(res)
        for p_txt in variants:
            for hit in _find_corporate_markers(p_txt):
                _add(f"{label}:{hit}", hit)

    if check_git_remote:
        for cfg in _resolve_git_config_files(ws):
            with suppress(OSError):
                for rurl in _extract_git_remote_urls(cfg.read_text(encoding="utf-8", errors="replace")):
                    for hit in _find_corporate_markers(rurl):
                        _add(f"git_remote:{hit}", hit)

    ok = len(indicators) == 0
    notice = COMPLIANCE_NOTICE_FR if lang_code == "fr" else COMPLIANCE_NOTICE_EN
    summary = (
        _msg(
            lang_code,
            f"Personal-PC compliance verified locally ({FINANCIAL_PROMISE}). {notice}",
            f"Conformité PC personnel vérifiée localement ({FINANCIAL_PROMISE}). {notice}",
        )
        if ok
        else _msg(
            lang_code,
            f"COMPLIANCE REFUSED ({', '.join(indicators)}): corporate environment detected. {COMPLIANCE_NOTICE_BILINGUAL}",
            f"REFUS DE CONFORMITÉ ({', '.join(indicators)}) : environnement d'entreprise détecté. {COMPLIANCE_NOTICE_BILINGUAL}",
        )
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": "hivemind.personal_pc_compliance",
            "lang": lang_code,
            "compliance_ok": ok,
            "personal_pc_only": True,
            "financial_promise": FINANCIAL_PROMISE,
            "license": "AGPL-3.0",
            "exit_code": 0 if ok else 2,
            "indicators": indicators,
            "matched_markers": sorted(markers),
            "network_calls": 0,
            "file_writes": 0,
            "subprocess_calls": 0,
            "summary": summary,
            "compliance_notice": notice,
        }
    )


def audit_html_safety(html_text: str, csp_header: str = "") -> dict[str, Any]:
    """Audit HTML page for CSP default-src 'none', nonces, 0 DOM sinks, 0 static figures, and compliance banners."""
    attr_nonces = [m.group(1) or m.group(2) for m in _HTML_NONCE_ATTR_RE.finditer(html_text or "")]
    csp_nonces = _CSP_INLINE_NONCE_RE.findall(csp_header or "")
    has_default_src_none = "default-src 'none'" in (csp_header or "").lower()
    primary_nonce = csp_nonces[0] if csp_nonces else (attr_nonces[0] if attr_nonces else "")
    unsafe_sinks = sorted({m.group(0).strip() for m in _UNSAFE_DOM_RE.finditer(html_text or "")})
    stripped = strip_masked_keys_and_nonces(html_text or "")
    static_figs = _STATIC_FIG_UNIT_RE.findall(stripped)
    low = (html_text or "").lower()
    has_personal_banner = ("personal" in low or "personnel" in low) and ("0.00" in low or "0,00" in low)
    has_agpl = "agpl-3.0" in low or "agpl" in low
    return {
        "csp_present": bool(csp_header),
        "default_src_none": has_default_src_none,
        "nonce": primary_nonce,
        "nonce_count": len(attr_nonces),
        "unsafe_dom_sinks": unsafe_sinks,
        "static_figures": static_figs,
        "has_personal_pc_banner": has_personal_banner,
        "has_agpl_banner": has_agpl,
        "ok": bool(
            has_default_src_none
            and primary_nonce
            and not unsafe_sinks
            and not static_figs
            and has_personal_banner
        ),
    }


def inspect_deploy_script_finops(
    deploy_script_path: Path | str | None = None,
    *,
    expected_sha: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Inspect ``deploy/gcp_openrouter/deploy.sh`` FinOps/security invariants without running gcloud."""
    lang_code = normalize_lang(lang)
    path = Path(deploy_script_path) if deploy_script_path is not None else (REPO_ROOT / DEFAULT_DEPLOY_SCRIPT_REL)
    display_path = redact_home_path(path)
    present = False
    active = ""
    if path.is_file():
        with suppress(OSError):
            raw = path.read_text(encoding="utf-8", errors="replace")
            present = True
            active = "\n".join(ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))

    invariants = {
        "min_instances_zero": bool(
            present and re.search(r"--min-instances(?:=|\s+)0\b", active)
            and not re.search(r"--min-instances(?:=|\s+)[1-9]\d*\b", active)
        ),
        "max_instances_one": bool(
            present and re.search(r"--max-instances(?:=|\s+)1\b", active)
            and not re.search(r"--max-instances(?:=|\s+)(?:0|[2-9]|\d{2,})\b", active)
        ),
        "no_cpu_throttling": bool(present and re.search(r"--no-cpu-throttling\b", active)),
        "concurrency_80": bool(present and re.search(r"--concurrency(?:=|\s+)80\b", active)),
        "litestream_replica_url": bool(present and "HIVEMIND_ROUTER_REPLICA_URL=" in active and "gcs://" in active),
        "secret_manager_router_secret": bool(present and "HIVEMIND_ROUTER_SECRET=" in active),
        "build_sha_propagation": bool(present and "HIVEMIND_BUILD_SHA=" in active and "--update-env-vars" in active),
        "zero_starter_grants": bool(
            present
            and "HIVEMIND_STARTER_GRANT_HMT:-0" in active
            and "HIVEMIND_REGISTRATION_BONUS_HMT:-0" in active
            and "HIVEMIND_ALLOW_BONUSES:-0" in active
        ),
    }
    sha_prefix = f"HIVEMIND_BUILD_SHA={expected_sha} " if expected_sha else ""
    finops_ok = bool(present and all(invariants.values()))
    return {
        "deploy_script_path": display_path,
        "deploy_script_present": present,
        "finops_ok": finops_ok,
        "invariants": invariants,
        "dry_run_command": f"{sha_prefix}HIVEMIND_DEPLOY_DRY_RUN=1 bash {DEFAULT_DEPLOY_SCRIPT_REL} --dry-run",
        "executed": False,
        "note": _msg(
            lang_code,
            f"Dry-run Cloud Run FinOps guidance ({display_path}): 0 cloud mutations performed.",
            f"Guide FinOps Cloud Run dry-run ({display_path}) : 0 mutation cloud effectuée.",
        ),
    }


def _project_readiness_gates(
    *,
    gateway_and_models_ok: bool,
    holiday_routes_ok: bool,
    feedback_summary: Mapping[str, Any] | None,
    live_hiveminders: int,
    distinct_physical_gpus: int,
    target_online_producers: int,
    lang_code: str,
) -> dict[str, Any]:
    fb = feedback_summary if isinstance(feedback_summary, Mapping) else {}
    total_resp = fb.get("total_responses")
    has_resp = not isinstance(total_resp, bool) and isinstance(total_resp, int) and total_resp > 0
    ease = fb.get("mean_ease_rating")
    lag = fb.get("zero_game_lag_ratio")
    ces_ok = bool(has_resp and not isinstance(ease, bool) and isinstance(ease, (int, float)) and float(ease) >= 4.2)
    lag_ok = bool(has_resp and not isinstance(lag, bool) and isinstance(lag, (int, float)) and float(lag) >= 0.95)
    fleet_ok = bool(live_hiveminders >= max(1, int(target_online_producers)) and distinct_physical_gpus >= 1)

    gates = {
        "gateway_and_models_online": bool(gateway_and_models_ok),
        "holiday_routes_deployed_on_cloud_run": bool(holiday_routes_ok),
        "satisfaction_ces_target_met": ces_ok,
        "gamer_zero_lag_target_met": lag_ok,
        "online_producer_fleet_target_met": fleet_ok,
    }
    blockers: list[str] = []
    if not gates["gateway_and_models_online"]:
        blockers.append(_msg(lang_code, "Gateway /health or /v1/models check failed.", "Échec /health ou /v1/models."))
    if not gates["holiday_routes_deployed_on_cloud_run"]:
        blockers.append(_msg(lang_code, "PR #949 Cloud Run holiday routes incomplete.", "Routes Noël PR #949 incomplètes."))
    if not ces_ok:
        blockers.append(_msg(lang_code, "Mean CES rating (>= 4.2/5) not yet met.", "CES moyen (>= 4.2/5) non atteint."))
    if not lag_ok:
        blockers.append(_msg(lang_code, "Zero-game-lag ratio (>= 0.95) not yet met.", "Ratio zéro-lag (>= 0.95) non atteint."))
    if not fleet_ok:
        blockers.append(
            _msg(
                lang_code,
                f"Live producers ({live_hiveminders}/{target_online_producers}) or GPUs ({distinct_physical_gpus}) below target.",
                f"Producteurs en ligne ({live_hiveminders}/{target_online_producers}) ou GPU ({distinct_physical_gpus}) sous l'objectif.",
            )
        )
    passed = sum(1 for v in gates.values() if v)
    return {
        "passed": passed,
        "total": len(gates),
        "summary": f"{passed}/{len(gates)}",
        "gates": gates,
        "blockers": blockers,
    }


def verify_pr949_live_revision_gate(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    require_holiday_routes: bool = False,
    deploy_script_path: Path | str | None = None,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    home: Path | str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference PR #949 Cloud Run live revision, CSP/figure, 5-key CES & fleet readiness gate."""
    lang_code = normalize_lang(lang)
    compliance = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        lang=lang_code,
        home=home,
    )
    if not compliance["compliance_ok"]:
        finops = inspect_deploy_script_finops(deploy_script_path, lang=lang_code)
        gates = _project_readiness_gates(
            gateway_and_models_ok=False,
            holiday_routes_ok=False,
            feedback_summary=None,
            live_hiveminders=0,
            distinct_physical_gpus=0,
            target_online_producers=max(1, int(target_online_producers)),
            lang_code=lang_code,
        )
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": GATE_OBJECT,
                "lang": lang_code,
                "ok": False,
                "exit_code": 2,
                "compliance_ok": False,
                "compliance": compliance,
                "network_calls": 0,
                "file_writes": 0,
                "subprocess_calls": 0,
                "inference_calls": 0,
                "stale_cloud_run_revision": False,
                "redeploy_required": False,
                "readiness_gates_projected": gates["summary"],
                "readiness_gates_passed": gates["passed"],
                "readiness_gates_total": gates["total"],
                "readiness_gates": gates,
                "finops_guidance": finops,
                "summary": compliance["summary"],
            }
        )

    clean_gw = validate_gateway_url(gateway)
    clean_expected_sha = validate_expected_sha(expected_sha) if expected_sha else None
    if isinstance(target_online_producers, bool) or int(target_online_producers) < 1:
        raise VerifierError("target_online_producers must be >= 1.")
    target_producers = int(target_online_producers)

    route_results: dict[str, dict[str, Any]] = {}
    seen_nonces: set[str] = set()
    network_calls = 0

    for path, kind, category in PR949_GATE_ROUTES:
        url = f"{clean_gw}{path}"
        status, body = _invoke_http_fn(http_fn, "GET", url)
        network_calls += 1
        entry: dict[str, Any] = {"status": status, "kind": kind, "category": category, "ok": False}

        if status == 200 and kind == "html":
            html_audit = audit_html_safety(str(body.get("html") or ""), str(body.get("csp") or ""))
            nonce = str(html_audit.get("nonce") or "")
            unique_nonce = bool(nonce and nonce not in seen_nonces)
            if nonce:
                seen_nonces.add(nonce)
            entry.update(html_audit)
            entry["unique_nonce"] = unique_nonce
            entry["ok"] = bool(html_audit["ok"] and unique_nonce)
        elif status == 200 and kind == "json":
            if path == "/health":
                entry["ok"] = str(body.get("status") or "").lower() == "ok"
            elif path == "/v1/models":
                models = body.get("data") if isinstance(body.get("data"), list) else []
                ids = [str(m.get("id") or "") for m in models if isinstance(m, Mapping)]
                has_coder = any(mid == "hivemind/coder" or mid.endswith("/hivemind/coder") for mid in ids)
                entry["model_count"] = len(ids)
                entry["has_hivemind_coder"] = has_coder
                entry["ok"] = bool(ids and has_coder)
            elif path == "/v1/hivemind/onboarding/manifest":
                hroutes = set(body.get("holiday_routes") or []) if isinstance(body.get("holiday_routes"), list) else set()
                live_sha = sanitize_build_sha(body.get("build_sha"))
                entry["build_sha"] = live_sha
                entry["cloud_run_revision"] = sanitize_cloud_run_revision(body.get("cloud_run_revision"))
                entry["missing_holiday_routes"] = sorted(set(EXPECTED_MANIFEST_HOLIDAY_ROUTES) - hroutes)
                entry["ok"] = bool(
                    body.get("object") == MANIFEST_OBJECT
                    and body.get("personal_pc_only") is True
                    and not entry["missing_holiday_routes"]
                )
            elif path == "/v1/hivemind/onboarding/feedback/summary":
                exact_keys = set(body.keys()) == EXPECTED_FEEDBACK_SUMMARY_KEYS
                entry["exact_5_keys"] = exact_keys
                entry["summary"] = dict(body) if exact_keys else {}
                entry["ok"] = bool(exact_keys and body.get("object") == FEEDBACK_SUMMARY_OBJECT)
            elif path == "/v1/hivemind/raid/status":
                comp = body.get("compliance") if isinstance(body.get("compliance"), Mapping) else {}
                entry["ok"] = bool(body.get("object") == RAID_STATUS_OBJECT and comp.get("personal_pc_only") is True)
            elif path == "/v1/hivemind/producers/fleet":
                rows = body.get("hiveminders") if isinstance(body.get("hiveminders"), list) else []
                leaked_endpoints = any(isinstance(r, Mapping) and "endpoint_url" in r for r in rows)
                entry["live_hiveminders"] = int(body.get("live_hiveminders") or 0)
                entry["total_hiveminders"] = int(body.get("total_hiveminders") or 0)
                entry["distinct_physical_gpus"] = int(body.get("distinct_physical_gpus") or 0)
                entry["endpoint_url_redacted"] = not leaked_endpoints
                entry["ok"] = bool(body.get("object") == FLEET_OBJECT and not leaked_endpoints)
            elif path == "/v1/hivemind/opentrainer/status":
                fleet_ot = body.get("live_opentrainer_fleet") if isinstance(body.get("live_opentrainer_fleet"), Mapping) else {}
                entry["opentrainer_status"] = str(body.get("status") or "UNKNOWN")
                entry["active_trainer_nodes"] = int(fleet_ot.get("active_trainer_nodes") or 0)
                entry["ok"] = bool(body.get("object") == OPENTRAINER_STATUS_OBJECT)
            elif path == "/v1/hivemind/live-telemetry":
                entry["telemetry_status"] = str(body.get("status") or "UNKNOWN")
                entry["ok"] = bool(body.get("object") == LIVE_TELEMETRY_OBJECT)
        route_results[path] = entry

    pre_pr949_drift_routes = [
        p
        for p, _, cat in PR949_GATE_ROUTES
        if cat in ("holiday", "manifest", "raid", "feedback") and route_results[p]["status"] == 404
    ]
    manifest_info = route_results["/v1/hivemind/onboarding/manifest"]
    live_sha = str(manifest_info.get("build_sha") or "unknown")
    sha_match = True if clean_expected_sha is None else (manifest_info["ok"] and shas_match(live_sha, clean_expected_sha))
    sha_mismatch = clean_expected_sha is not None and not sha_match

    core_ok = bool(route_results["/health"]["ok"] and route_results["/v1/models"]["ok"])
    holiday_routes_ok = all(
        route_results[p]["ok"]
        for p in ("/noel", "/quick", "/raid-noel", "/arena-noel", "/v1/hivemind/onboarding/manifest", "/v1/hivemind/onboarding/feedback/summary", "/v1/hivemind/raid/status")
    )
    fleet_routes_ok = bool(
        route_results["/v1/hivemind/producers/fleet"]["ok"]
        and route_results["/v1/hivemind/opentrainer/status"]["ok"]
        and route_results["/v1/hivemind/live-telemetry"]["ok"]
    )
    stale_revision = bool(pre_pr949_drift_routes)
    redeploy_required = bool(stale_revision or sha_mismatch or (require_holiday_routes and not holiday_routes_ok))

    fleet_info = route_results["/v1/hivemind/producers/fleet"]
    fb_summary = route_results["/v1/hivemind/onboarding/feedback/summary"].get("summary") or None
    gates = _project_readiness_gates(
        gateway_and_models_ok=core_ok,
        holiday_routes_ok=holiday_routes_ok,
        feedback_summary=fb_summary,
        live_hiveminders=int(fleet_info.get("live_hiveminders") or 0),
        distinct_physical_gpus=int(fleet_info.get("distinct_physical_gpus") or 0),
        target_online_producers=target_producers,
        lang_code=lang_code,
    )
    finops = inspect_deploy_script_finops(deploy_script_path, expected_sha=clean_expected_sha, lang=lang_code)

    overall_ok = bool(core_ok and holiday_routes_ok and fleet_routes_ok and sha_match)
    hard_fail = bool((not core_ok) or sha_mismatch or (require_holiday_routes and not holiday_routes_ok) or (require_holiday_routes and not fleet_routes_ok))
    exit_code = 2 if hard_fail else 0

    summary = (
        _msg(
            lang_code,
            f"PR #949 live revision gate passed (nonces={len(seen_nonces)}, gates={gates['summary']}).",
            f"Portail PR #949 Cloud Run vérifié (nonces={len(seen_nonces)}, gates={gates['summary']}).",
        )
        if overall_ok
        else _msg(
            lang_code,
            f"PR #949 live revision gate detected drift/issues (404_drift={len(pre_pr949_drift_routes)}, sha_match={sha_match}, gates={gates['summary']}).",
            f"Dérive/anomalie détectée sur Cloud Run (404={len(pre_pr949_drift_routes)}, sha_match={sha_match}, gates={gates['summary']}).",
        )
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": GATE_OBJECT,
            "lang": lang_code,
            "gateway": clean_gw,
            "ok": overall_ok,
            "exit_code": exit_code,
            "compliance_ok": True,
            "compliance": compliance,
            "core_ok": core_ok,
            "holiday_routes_ok": holiday_routes_ok,
            "fleet_routes_ok": fleet_routes_ok,
            "expected_sha": clean_expected_sha,
            "live_build_sha": live_sha,
            "cloud_run_revision": manifest_info.get("cloud_run_revision"),
            "sha_match": sha_match,
            "sha_mismatch": sha_mismatch,
            "stale_cloud_run_revision": stale_revision,
            "pre_pr949_drift_routes": pre_pr949_drift_routes,
            "redeploy_required": redeploy_required,
            "unique_csp_nonces": sorted(seen_nonces),
            "unique_csp_nonce_count": len(seen_nonces),
            "feedback_summary": fb_summary,
            "live_hiveminders": int(fleet_info.get("live_hiveminders") or 0),
            "distinct_physical_gpus": int(fleet_info.get("distinct_physical_gpus") or 0),
            "readiness_gates_projected": gates["summary"],
            "readiness_gates_passed": gates["passed"],
            "readiness_gates_total": gates["total"],
            "readiness_gates": gates,
            "finops_guidance": finops,
            "routes": route_results,
            "network_calls": network_calls,
            "file_writes": 0,
            "subprocess_calls": 0,
            "inference_calls": 0,
            "summary": summary,
        }
    )


def compute_ces_idempotency_key(
    payload: Mapping[str, Any],
    *,
    account_ref: str = "",
    salt: str = "",
) -> str:
    """Compute a deterministic zero-PII ``Idempotency-Key`` for ``POST /v1/hivemind/onboarding/feedback``."""
    canonical = json.dumps(
        {
            "campaign": CAMPAIGN_ID,
            "account_ref_hash": hashlib.sha256(f"{account_ref}:{salt}".encode()).hexdigest()[:16],
            "profile": str(payload.get("profile") or ""),
            "ease_rating": int(payload.get("ease_rating") or 5),
            "zero_game_lag": bool(payload.get("zero_game_lag")),
            "platform": str(payload.get("platform") or ""),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return "dyn33-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:28]


def _validate_ces_item_payload(raw: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, Mapping):
        return None, "payload_not_mapping"
    profile_raw = raw.get("profile")
    try:
        profile = normalize_profile(profile_raw)
    except VerifierError:
        return None, "invalid_profile"
    ease = raw.get("ease_rating")
    if isinstance(ease, bool) or not isinstance(ease, int) or not (1 <= ease <= 5):
        return None, "invalid_ease_rating"
    lag = raw.get("zero_game_lag")
    if not isinstance(lag, bool):
        return None, "invalid_zero_game_lag"
    plat = re.sub(r"[^A-Za-z0-9._-]+", "", str(raw.get("platform") or "linux"))[:32] or "linux"
    note_raw = str(raw.get("note") or "")
    if note_raw:
        return None, "non_empty_note_pii_risk"
    normalized = {"profile": profile, "ease_rating": ease, "zero_game_lag": lag, "platform": plat, "note": ""}
    if scan_output_safety(json.dumps(normalized, sort_keys=True)):
        return None, "unsafe_ces_payload"
    return normalized, None


class ReconnectionCesQueue:
    """Atomic, bounded (<=128 pending / <=512 delivered), ``0600`` zero-PII CES feedback retry queue."""

    _PATH_LOCKS_GUARD = threading.Lock()
    _PATH_LOCKS: dict[Path, threading.RLock] = {}

    def __init__(
        self,
        queue_path: Path | str,
        *,
        max_pending: int = MAX_PENDING_QUEUE_ENTRIES,
        max_delivered: int = MAX_DELIVERED_IDS,
    ) -> None:
        self.queue_path = Path(queue_path)
        self.lock_path = self.queue_path.with_name(f"{self.queue_path.name}.lock")
        if isinstance(max_pending, bool) or not (1 <= int(max_pending) <= 4096):
            raise VerifierError("max_pending must be 1..4096.")
        if isinstance(max_delivered, bool) or not (1 <= int(max_delivered) <= 4096):
            raise VerifierError("max_delivered must be 1..4096.")
        self.max_pending = int(max_pending)
        self.max_delivered = int(max_delivered)

    @classmethod
    def _thread_lock(cls, target: Path) -> threading.RLock:
        key = target.resolve() if target.is_absolute() else (Path.cwd() / target).resolve()
        with cls._PATH_LOCKS_GUARD:
            lock = cls._PATH_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._PATH_LOCKS[key] = lock
            return lock

    @contextmanager
    def _locked(self, *, create_lock_file: bool) -> Iterator[None]:
        tlock = self._thread_lock(self.queue_path)
        with tlock:
            if not create_lock_file and not self.queue_path.exists():
                yield
                return
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if _fcntl is not None:
                    _fcntl.flock(fd, _fcntl.LOCK_EX)
                with suppress(OSError):
                    os.chmod(self.lock_path, 0o600)
                yield
            finally:
                if _fcntl is not None:
                    with suppress(OSError):
                        _fcntl.flock(fd, _fcntl.LOCK_UN)
                os.close(fd)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema": QUEUE_SCHEMA,
            "pending": [],
            "delivered_event_ids": [],
            "quarantined": [],
            "overflow_dropped": 0,
        }

    def _read_state_unlocked(self) -> tuple[dict[str, Any], list[str]]:
        if not self.queue_path.is_file():
            return self._empty_state(), []
        try:
            loaded = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_state(), ["corrupt_queue_json"]
        if not isinstance(loaded, dict) or loaded.get("schema") != QUEUE_SCHEMA:
            return self._empty_state(), ["invalid_queue_schema"]
        state = self._empty_state()
        anomalies: list[str] = []
        raw_delivered = loaded.get("delivered_event_ids")
        if isinstance(raw_delivered, list):
            seen_d: set[str] = set()
            clean_d: list[str] = []
            for eid in raw_delivered:
                s = str(eid or "").strip()
                if s and re.fullmatch(r"[A-Za-z0-9._-]{8,128}", s) and s not in seen_d:
                    seen_d.add(s)
                    clean_d.append(s)
            state["delivered_event_ids"] = clean_d[-self.max_delivered :]
        if isinstance(loaded.get("quarantined"), list):
            state["quarantined"] = [dict(q) for q in loaded["quarantined"] if isinstance(q, Mapping)][-MAX_QUARANTINED_ENTRIES:]
        state["overflow_dropped"] = int(loaded.get("overflow_dropped") or 0)

        raw_pending = loaded.get("pending")
        if isinstance(raw_pending, list):
            for entry in raw_pending:
                if not isinstance(entry, Mapping):
                    anomalies.append("quarantined_non_mapping")
                    continue
                eid = str(entry.get("event_id") or "").strip()
                if not re.fullmatch(r"[A-Za-z0-9._-]{8,128}", eid):
                    anomalies.append("quarantined_bad_event_id")
                    state["quarantined"].append({"reason": "bad_event_id"})
                    continue
                clean_payload, err = _validate_ces_item_payload(entry.get("payload") or {})
                if clean_payload is None:
                    anomalies.append(f"quarantined:{err}")
                    state["quarantined"].append({"event_id": eid, "reason": err})
                    continue
                state["pending"].append(
                    {
                        "event_id": eid,
                        "idempotency_key": str(entry.get("idempotency_key") or eid)[:128],
                        "payload": clean_payload,
                        "retry_count": max(0, int(entry.get("retry_count") or 0)),
                        "last_status": int(entry.get("last_status") or 0),
                    }
                )
        state["quarantined"] = state["quarantined"][-MAX_QUARANTINED_ENTRIES:]
        return state, anomalies

    def _write_state_atomic(self, state: Mapping[str, Any]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            os.chmod(self.queue_path.parent, 0o700)
        bounded = {
            "schema": QUEUE_SCHEMA,
            "pending": list(state.get("pending") or [])[-self.max_pending :],
            "delivered_event_ids": list(state.get("delivered_event_ids") or [])[-self.max_delivered :],
            "quarantined": list(state.get("quarantined") or [])[-MAX_QUARANTINED_ENTRIES:],
            "overflow_dropped": int(state.get("overflow_dropped") or 0),
        }
        raw_text = json.dumps(bounded, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        _assert_safe_output(raw_text)
        tmp_path = self.queue_path.with_name(f".{self.queue_path.name}.{os.getpid()}.tmp")
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(raw_text)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            with suppress(OSError):
                tmp_path.unlink()
            raise
        with suppress(OSError):
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, self.queue_path)
        with suppress(OSError):
            os.chmod(self.queue_path, 0o600)

    def snapshot(self) -> dict[str, Any]:
        """Return bounded queue summary without creating files when empty/missing."""
        with self._locked(create_lock_file=False):
            state, anomalies = self._read_state_unlocked()
            return {
                "exists": self.queue_path.is_file(),
                "pending_count": len(state["pending"]),
                "delivered_count": len(state["delivered_event_ids"]),
                "quarantined_count": len(state["quarantined"]),
                "overflow_dropped": state["overflow_dropped"],
                "anomalies": anomalies,
            }

    def enqueue(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        last_status: int = 0,
    ) -> dict[str, Any]:
        """Persist a validated zero-PII CES feedback item into the bounded ``0600`` spool."""
        clean_payload, err = _validate_ces_item_payload(payload)
        if clean_payload is None:
            raise VerifierError(f"Refusing invalid CES queue payload: {err}")
        eid = (idempotency_key or compute_ces_idempotency_key(clean_payload)).strip()[:128]
        with self._locked(create_lock_file=True):
            state, _ = self._read_state_unlocked()
            if eid in state["delivered_event_ids"]:
                return {"enqueued": False, "duplicate": True, "event_id": eid, "pending_count": len(state["pending"])}
            if any(p["event_id"] == eid for p in state["pending"]):
                return {"enqueued": False, "duplicate": True, "event_id": eid, "pending_count": len(state["pending"])}
            while len(state["pending"]) >= self.max_pending:
                state["pending"].pop(0)
                state["overflow_dropped"] = int(state.get("overflow_dropped") or 0) + 1
            state["pending"].append(
                {
                    "event_id": eid,
                    "idempotency_key": eid,
                    "payload": clean_payload,
                    "retry_count": 0,
                    "last_status": int(last_status),
                }
            )
            self._write_state_atomic(state)
            return {"enqueued": True, "duplicate": False, "event_id": eid, "pending_count": len(state["pending"])}

    def flush(
        self,
        gateway: str,
        *,
        http_fn: HttpFn = default_http_request,
    ) -> dict[str, Any]:
        """Replay queued CES items with ``Idempotency-Key``; do not create files if queue is missing."""
        clean_gw = validate_gateway_url(gateway)
        if not self.queue_path.is_file():
            return {
                "flushed_count": 0,
                "retained_count": 0,
                "quarantined_count": 0,
                "network_calls": 0,
                "file_writes": 0,
                "queue_file_exists": False,
            }
        with self._locked(create_lock_file=True):
            state, _ = self._read_state_unlocked()
            if not state["pending"] and not state["quarantined"] and not state["delivered_event_ids"]:
                return {
                    "flushed_count": 0,
                    "retained_count": 0,
                    "quarantined_count": 0,
                    "network_calls": 0,
                    "file_writes": 0,
                    "queue_file_exists": self.queue_path.is_file(),
                }
            remaining: list[dict[str, Any]] = []
            flushed = 0
            quarantined_now = 0
            calls = 0
            url = f"{clean_gw}/v1/hivemind/onboarding/feedback"
            for item in state["pending"]:
                eid = str(item["event_id"])
                if eid in state["delivered_event_ids"]:
                    continue
                status, _ = _invoke_http_fn(
                    http_fn,
                    "POST",
                    url,
                    payload=item["payload"],
                    headers={"Idempotency-Key": str(item["idempotency_key"])},
                )
                calls += 1
                if 200 <= status < 300:
                    flushed += 1
                    state["delivered_event_ids"].append(eid)
                elif status in PERMANENT_CES_STATUSES or (400 <= status < 500 and status not in RETRIABLE_CES_STATUSES):
                    quarantined_now += 1
                    state["quarantined"].append({"event_id": eid, "status": status, "reason": f"http_{status}"})
                else:
                    retries = int(item.get("retry_count") or 0) + 1
                    if retries >= MAX_RETRY_COUNT:
                        quarantined_now += 1
                        state["quarantined"].append({"event_id": eid, "status": status, "reason": "max_retries_exceeded"})
                    else:
                        updated = dict(item)
                        updated["retry_count"] = retries
                        updated["last_status"] = status
                        remaining.append(updated)
            state["pending"] = remaining[-self.max_pending :]
            state["delivered_event_ids"] = state["delivered_event_ids"][-self.max_delivered :]
            state["quarantined"] = state["quarantined"][-MAX_QUARANTINED_ENTRIES:]
            self._write_state_atomic(state)
            return {
                "flushed_count": flushed,
                "retained_count": len(state["pending"]),
                "quarantined_count": len(state["quarantined"]),
                "quarantined_delta": quarantined_now,
                "delivered_count": len(state["delivered_event_ids"]),
                "network_calls": calls,
                "file_writes": 1,
                "queue_file_exists": self.queue_path.is_file(),
            }

    def purge(self) -> dict[str, Any]:
        """Securely delete local CES queue and lock files (always permitted offline)."""
        removed = False
        with self._thread_lock(self.queue_path):
            for p in (self.queue_path, self.lock_path):
                if p.exists():
                    with suppress(OSError):
                        p.unlink()
                        removed = True
        return {"purged": True, "removed_files": removed, "queue_path": redact_home_path(self.queue_path)}


def default_subprocess_runner(cmd: Sequence[str], timeout_s: float = SUBPROCESS_TIMEOUT_S) -> tuple[int, str]:
    """Execute a safe read-only command (`shell=False`, <= 3s timeout)."""
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise VerifierError("Subprocess command must be a non-empty sequence.")
    bounded = min(max(0.1, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    try:
        done = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, errors="replace", timeout=bounded, check=False, shell=False)
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return 127, ""
    return int(done.returncode), (done.stdout or "")


def _opt_float(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) else None
    cleaned = str(raw).strip()
    if not cleaned or cleaned.lower() in ("[n/a]", "n/a", "[not supported]", "not supported"):
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def default_gpu_probe(*, runner: RunnerFn = default_subprocess_runner, timeout_s: float = SUBPROCESS_TIMEOUT_S) -> GpuSnapshot:
    """Probe local NVIDIA GPU telemetry via `nvidia-smi` or return honest no-GPU fallback."""
    rc, stdout = runner(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
        min(float(timeout_s), SUBPROCESS_TIMEOUT_S),
    )
    if rc != 0 or not stdout.strip():
        return GpuSnapshot(present=False, reason="dev_api_only_no_nvidia_gpu")
    gpus: list[GpuSnapshot] = []
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or not parts[0]:
            return GpuSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        tot, free, used, temp, util = (_opt_float(parts[i]) for i in range(1, 6))
        if tot is None or free is None or tot <= 0 or free < 0:
            return GpuSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
        gpus.append(
            GpuSnapshot(
                present=True,
                parse_error=False,
                name=parts[0][:128],
                total_vram_mib=int(tot),
                free_vram_mib=int(free),
                used_vram_mib=int(used) if used is not None and used >= 0 else max(0, int(tot) - int(free)),
                temp_c=round(temp, 1) if temp is not None else None,
                util_pct=round(util, 1) if util is not None else 0.0,
                reason="ok",
            )
        )
    return max(gpus, key=lambda g: (g.free_vram_mib or 0, -(g.temp_c if g.temp_c is not None else 999.0)))


def _normalize_gpu_snapshot(raw: Any) -> GpuSnapshot:
    if raw is None:
        return GpuSnapshot(present=False, reason="dev_api_only_no_nvidia_gpu")
    if isinstance(raw, GpuSnapshot):
        return raw
    if isinstance(raw, Mapping):
        view: Mapping[str, Any] = raw
    elif hasattr(raw, "present"):
        view = {
            "present": getattr(raw, "present", False),
            "parse_error": getattr(raw, "parse_error", False),
            "name": getattr(raw, "name", None) or getattr(raw, "gpu_name", None),
            "total_vram_mib": getattr(raw, "total_vram_mib", getattr(raw, "total_mib", None)),
            "free_vram_mib": getattr(raw, "free_vram_mib", getattr(raw, "free_mib", None)),
            "used_vram_mib": getattr(raw, "used_vram_mib", getattr(raw, "used_mib", None)),
            "temp_c": getattr(raw, "temp_c", None),
            "util_pct": getattr(raw, "util_pct", 0.0),
            "reason": getattr(raw, "reason", "ok"),
        }
    else:
        raise VerifierError("gpu_probe_fn must return GpuSnapshot, a mapping, or None.")
    if not bool(view.get("present", view.get("gpu_present", False))):
        return GpuSnapshot(present=False, reason=str(view.get("reason") or "dev_api_only_no_nvidia_gpu"))
    if bool(view.get("parse_error", False)):
        return GpuSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
    name = view.get("name") or view.get("gpu_name")
    tot = view.get("total_vram_mib", view.get("vram_total_mib", view.get("total_mib")))
    free = view.get("free_vram_mib", view.get("vram_free_mib", view.get("free_mib")))
    used = view.get("used_vram_mib", view.get("vram_used_mib", view.get("used_mib")))
    temp = _opt_float(view.get("temp_c"))
    util = _opt_float(view.get("util_pct"))
    if not name or tot is None or free is None:
        return GpuSnapshot(present=True, parse_error=True, reason="nvidia_smi_parse_error")
    return GpuSnapshot(
        present=True,
        parse_error=False,
        name=str(name).strip()[:128],
        total_vram_mib=int(tot),
        free_vram_mib=int(free),
        used_vram_mib=int(used) if used is not None else max(0, int(tot) - int(free)),
        temp_c=round(temp, 1) if temp is not None else None,
        util_pct=round(util, 1) if util is not None else 0.0,
        reason=str(view.get("reason") or "ok"),
    )


def default_process_probe(*, runner: RunnerFn = default_subprocess_runner, os_name: str | None = None, timeout_s: float = SUBPROCESS_TIMEOUT_S) -> list[str]:
    """List local process executables via ``tasklist`` (Windows) or ``ps`` (POSIX)."""
    eff_os = os.name if os_name is None else os_name
    bounded = min(float(timeout_s), SUBPROCESS_TIMEOUT_S)
    procs: list[str] = []
    if eff_os == "nt":
        rc, out = runner(["tasklist", "/FO", "CSV", "/NH"], bounded)
        if rc == 0 and out.strip():
            with suppress(csv.Error):
                for row in csv.reader(io.StringIO(out)):
                    if row and row[0].strip():
                        procs.append(row[0].strip().lower())
        return procs
    rc, out = runner(["ps", "-eo", "comm="], bounded)
    if rc == 0 and out.strip():
        for line in out.splitlines():
            clean = line.strip().lower()
            if clean:
                procs.append(clean.rsplit("/", 1)[-1])
    return procs


def _detect_steam_dx12_activity(process_names: Iterable[str], *, non_worker_gpu_util_pct: float | None = None) -> dict[str, Any]:
    trunc = {e[:15] for e in STEAM_DX12_GAME_EXECUTABLES if len(e) > 15}
    matched: list[str] = []
    for proc in process_names:
        low = str(proc).strip().lower().replace("\\", "/")
        base = low.rsplit("/", 1)[-1]
        if base and (base in STEAM_DX12_GAME_EXECUTABLES or base in trunc or base.startswith(STEAM_PROCESS_PREFIXES) or "steamapps/common/" in low):
            if base not in matched:
                matched.append(base)
    gpu_high = bool(non_worker_gpu_util_pct is not None and math.isfinite(non_worker_gpu_util_pct) and non_worker_gpu_util_pct > NON_WORKER_GPU_UTIL_THRESHOLD_PCT)
    return {"game_detected": bool(matched or gpu_high), "matched_games": matched, "gpu_util_triggered": gpu_high}


def default_worker_yield_drill(*, clock_fn: Callable[[], float] = time.monotonic, trigger_exe: str = "cs2.exe", timeout_s: float = SUBPROCESS_TIMEOUT_S) -> dict[str, Any]:
    """Run a real subprocess worker start -> Steam trigger -> <1.5 s YIELD -> 0 MiB release drill."""
    bounded = min(max(0.2, float(timeout_s)), SUBPROCESS_TIMEOUT_S)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _LOCAL_WORKER_YIELD_PROGRAM],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    try:
        ready = proc.stdout.readline().strip() if proc.stdout is not None else ""
        if ready != "HIVEMIND_WORKER_READY":
            proc.kill()
            proc.wait(timeout=1.0)
            return {"worker_started": False, "worker_stopped": False, "clean_release": False, "retained_worker_vram_mib": -1, "yield_latency_ms": None, "trigger_exe": trigger_exe}
        t0 = clock_fn()
        _detect_steam_dx12_activity([trigger_exe])
        if proc.stdin is not None:
            proc.stdin.write("YIELD\n")
            proc.stdin.flush()
            proc.stdin.close()
        rel = proc.stdout.readline().strip() if proc.stdout is not None else ""
        rc = proc.wait(timeout=bounded)
        elapsed_ms = round(max(0.0, (clock_fn() - t0) * 1000.0), 3)
        retained = int(rel.split(":", 1)[1].strip()) if rel.startswith("HIVEMIND_WORKER_RELEASED:") and rel.split(":", 1)[1].strip().isdigit() else -1
        return {
            "worker_started": True,
            "worker_stopped": proc.poll() is not None,
            "clean_release": bool(rc == 0 and proc.poll() is not None and retained == 0),
            "retained_worker_vram_mib": retained,
            "yield_latency_ms": elapsed_ms,
            "trigger_exe": trigger_exe,
            "method": "real_subprocess_yield_drill",
        }
    except (subprocess.TimeoutExpired, OSError):
        proc.kill()
        proc.wait(timeout=1.0)
        return {"worker_started": True, "worker_stopped": False, "clean_release": False, "retained_worker_vram_mib": -1, "yield_latency_ms": round(bounded * 1000.0, 3), "trigger_exe": trigger_exe}
    finally:
        if proc.poll() is None:
            with suppress(OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait(timeout=1.0)
        for st in (proc.stdin, proc.stdout, proc.stderr):
            if st is not None and not st.closed:
                st.close()


def _invoke_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _inspect_workspace_and_config_security(
    *,
    credentials_path: Path | None,
    continue_config_path: Path | None,
    workspace_dir: Path | None,
    home_path: Path,
    gateway: str,
) -> dict[str, Any]:
    issues: list[str] = []
    posix = os.name != "nt"

    def _check_0600(p: Path | None, label: str) -> dict[str, Any]:
        if p is None or not p.is_file():
            return {"present": False, "path": redact_home_path(p, home=home_path, workspace=workspace_dir) if p else None, "mode_0600_ok": True}
        mode_ok = True
        if posix:
            with suppress(OSError):
                if (p.stat().st_mode & 0o077) != 0:
                    mode_ok = False
                    issues.append(f"{label}_permissions_too_open")
        return {"present": True, "path": redact_home_path(p, home=home_path, workspace=workspace_dir), "mode_0600_ok": mode_ok}

    cred_info = _check_0600(credentials_path, "credentials")
    cont_info = _check_0600(continue_config_path, "continue_config")
    cont_has_coder = False
    if continue_config_path is not None and continue_config_path.is_file():
        with suppress(OSError):
            cont_has_coder = "hivemind/coder" in continue_config_path.read_text(encoding="utf-8", errors="replace")

    ws_present = False
    ws_secret_free = True
    ws_gateway_ok = True
    if workspace_dir is not None:
        settings_path = Path(workspace_dir) / ".vscode" / "settings.json"
        if settings_path.is_file():
            ws_present = True
            with suppress(OSError):
                txt = settings_path.read_text(encoding="utf-8", errors="replace")
                home_str = str(home_path)
                has_custom_home = len(home_str) > 3 and home_str in txt
                if _API_KEY_RE.search(txt) or _RAW_HM_KEY_RE.search(txt) or _BEARER_RE.search(txt) or _HOME_PATH_RE.search(txt) or has_custom_home:
                    ws_secret_free = False
                    issues.append("workspace_vscode_settings_contains_secret_or_home_path")
                with suppress(json.JSONDecodeError):
                    parsed = json.loads(txt)
                    if isinstance(parsed, Mapping):
                        base = str(parsed.get("hivemind.openaiCompatible.apiBase") or "")
                        if base and not base.startswith(gateway):
                            ws_gateway_ok = False
                            issues.append("workspace_vscode_settings_gateway_mismatch")

    return {
        "ok": not issues,
        "workspace_settings_present": ws_present,
        "workspace_secret_free": ws_secret_free,
        "workspace_gateway_ok": ws_gateway_ok,
        "credentials_file": cred_info,
        "continue_config": {**cont_info, "has_hivemind_coder": cont_has_coder},
        "issues": issues,
    }


def _resolve_api_key(
    *,
    api_key: str | None,
    credentials_path: Path | None,
    environ: Mapping[str, str],
) -> tuple[str | None, str]:
    explicit = (api_key or "").strip()
    if explicit:
        return (explicit, "explicit_arg") if is_valid_hivemind_key(explicit) else (None, "invalid_explicit_key")
    env_key = str(environ.get("HIVEMIND_API_KEY") or "").strip()
    if env_key:
        return (env_key, "env") if is_valid_hivemind_key(env_key) else (None, "invalid_env_key")
    if credentials_path is not None and credentials_path.is_file():
        with suppress(OSError, json.JSONDecodeError):
            data = json.loads(credentials_path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                saved = str(data.get("api_key") or "").strip()
                if is_valid_hivemind_key(saved):
                    return saved, "credentials_file"
                return None, "invalid_credentials_file_key"
    return None, "missing"


def verify_registered_producer_activation(
    gateway: str = DEFAULT_GATEWAY,
    *,
    credentials_path: Path | str | None = None,
    api_key: str | None = None,
    workspace_dir: Path | str | None = None,
    continue_config_path: Path | str | None = None,
    send_opentrainer_heartbeat: bool = False,
    submit_ces: bool = False,
    flush_ces_queue: bool = False,
    queue_path: Path | str | None = None,
    profile: str = "gamer",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    host_label: str = "personal-pc",
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    yield_budget_ms: float = DEFAULT_YIELD_BUDGET_MS,
    gpu_probe_fn: Callable[..., Any] = default_gpu_probe,
    process_probe_fn: Callable[..., Any] = default_process_probe,
    worker_yield_fn: Callable[..., Any] = default_worker_yield_drill,
    clock_fn: Callable[[], float] = time.monotonic,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    home: Path | str | None = None,
    require_live_producer: bool = False,
    idempotency_salt: str = "",
    lang: str = "en",
) -> dict[str, Any]:
    """<60s VS Code / <5m Gamer-Whisper registered-account reconnection & activation verifier."""
    lang_code = normalize_lang(lang)
    norm_profile = normalize_profile(profile)
    if isinstance(ease_rating, bool) or not (1 <= int(ease_rating) <= 5):
        raise VerifierError("ease_rating must be an integer 1..5.")
    if isinstance(max_temp_c, bool) or not math.isfinite(float(max_temp_c)) or not (25.0 <= float(max_temp_c) <= 95.0):
        raise VerifierError("max_temp_c must be a finite float between 25.0 and 95.0.")
    if isinstance(min_free_vram_mib, bool) or int(min_free_vram_mib) < 256:
        raise VerifierError("min_free_vram_mib must be an integer >= 256.")
    if isinstance(yield_budget_ms, bool) or not math.isfinite(float(yield_budget_ms)) or float(yield_budget_ms) <= 0.0:
        raise VerifierError("yield_budget_ms must be > 0.0.")

    t_start = clock_fn()
    env_map = os.environ if environ is None else environ
    ws_path = Path(workspace_dir) if workspace_dir is not None else None
    compliance = evaluate_personal_pc_compliance(
        environ=env_map,
        cwd=cwd if cwd is not None else ws_path,
        check_git_remote=check_git_remote,
        lang=lang_code,
        home=home,
    )
    if not compliance["compliance_ok"]:
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": ACTIVATION_OBJECT,
                "lang": lang_code,
                "ok": False,
                "exit_code": 2,
                "compliance_ok": False,
                "compliance": compliance,
                "network_calls": 0,
                "file_writes": 0,
                "subprocess_calls": 0,
                "inference_calls": 0,
                "summary": compliance["summary"],
            }
        )

    clean_gw = validate_gateway_url(gateway)
    home_path = Path(home) if home is not None else Path.home()
    cred_p = Path(credentials_path) if credentials_path is not None else (home_path / DEFAULT_CREDENTIALS_REL)
    cont_p = Path(continue_config_path) if continue_config_path is not None else (home_path / DEFAULT_CONTINUE_REL)
    q_path = Path(queue_path) if queue_path is not None else (home_path / DEFAULT_QUEUE_REL)

    sec_check = _inspect_workspace_and_config_security(
        credentials_path=cred_p,
        continue_config_path=cont_p,
        workspace_dir=ws_path,
        home_path=home_path,
        gateway=clean_gw,
    )
    resolved_key, key_source = _resolve_api_key(api_key=api_key, credentials_path=cred_p, environ=env_map)
    masked_key = mask_api_key(resolved_key)

    network_calls = 0
    file_writes = 0
    subprocess_calls = 0
    diagnostics: list[str] = list(sec_check["issues"])

    me_status, me_body = (0, {})
    rcfg_status, rcfg_body = (0, {})
    fleet_status, fleet_body = (0, {})
    account_ok = False
    account_role = "unknown"
    account_handle = ""
    consumer_role_restricted = False
    remote_paused = False
    remote_vram_cap_low = False
    remote_vram_cap_mib: int | None = None
    offline_ttl_expired = False
    awaiting_first_hb = False
    producer_fleet_state = "unknown"

    if resolved_key is None:
        diagnostics.append(f"credentials:{key_source}")
    else:
        auth_headers = {"Authorization": f"Bearer {resolved_key}"}
        me_status, me_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/me", headers=auth_headers)
        rcfg_status, rcfg_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/remote/config", headers=auth_headers)
        fleet_status, fleet_body = _invoke_http_fn(http_fn, "GET", f"{clean_gw}/v1/hivemind/producers/fleet")
        network_calls += 3

        if me_status != 200:
            diagnostics.append(f"auth_me_http_{me_status}")
        else:
            account_role = str(me_body.get("role") or "unknown").strip().lower()
            account_handle = str(me_body.get("handle") or "").strip()
            consumer_role_restricted = account_role == "consumer"
            if consumer_role_restricted:
                diagnostics.append("consumer_role_restricted")

        if rcfg_status == 200:
            rem = rcfg_body.get("remote_config") if isinstance(rcfg_body.get("remote_config"), Mapping) else (rcfg_body.get("control") if isinstance(rcfg_body.get("control"), Mapping) else {})
            remote_paused = bool(rem.get("paused") or str(rem.get("node_state") or "").lower() == "paused")
            raw_cap = rem.get("vram_cap_mib")
            if raw_cap is not None and not isinstance(raw_cap, bool):
                remote_vram_cap_mib = int(raw_cap)
                if remote_vram_cap_mib < int(min_free_vram_mib):
                    remote_vram_cap_low = True
                    diagnostics.append("remote_vram_cap_below_min")
            if remote_paused:
                diagnostics.append("remote_producer_paused")
        elif me_status == 200:
            diagnostics.append(f"remote_config_http_{rcfg_status}")

        if fleet_status == 200 and account_handle:
            rows = fleet_body.get("hiveminders") if isinstance(fleet_body.get("hiveminders"), list) else []
            match = next((r for r in rows if isinstance(r, Mapping) and str(r.get("handle") or "") == account_handle), None)
            if match is None:
                producer_fleet_state = "not_in_fleet"
            else:
                producer_fleet_state = str(match.get("status") or "unknown")
                if producer_fleet_state == "offline":
                    if match.get("last_heartbeat_utc") is not None:
                        offline_ttl_expired = True
                        diagnostics.append("producer_offline_ttl_expired")
                    else:
                        awaiting_first_hb = True
            if require_live_producer and producer_fleet_state != "live":
                diagnostics.append("producer_not_live")

        account_ok = bool(
            me_status == 200
            and rcfg_status in (200, 404)
            and not consumer_role_restricted
            and not remote_paused
            and not remote_vram_cap_low
            and not offline_ttl_expired
            and (not require_live_producer or producer_fleet_state == "live")
        )

    # Local GPU, Steam/DX12 process & real subprocess yield preflight
    gpu = _normalize_gpu_snapshot(_invoke_with_supported_kwargs(gpu_probe_fn))
    if gpu_probe_fn is default_gpu_probe:
        subprocess_calls += 1
    procs = _invoke_with_supported_kwargs(process_probe_fn)
    if process_probe_fn is default_process_probe:
        subprocess_calls += 1
    steam_info = _detect_steam_dx12_activity(procs or [], non_worker_gpu_util_pct=gpu.util_pct if gpu.present else None)
    yield_res = dict(_invoke_with_supported_kwargs(worker_yield_fn, clock_fn=clock_fn, yield_budget_ms=float(yield_budget_ms)))
    if worker_yield_fn is default_worker_yield_drill:
        subprocess_calls += 1

    if not gpu.present:
        gpu_status = "dev_api_only_no_nvidia_gpu"
        thermal_ok = True
        vram_ok = True
    elif gpu.parse_error:
        gpu_status = gpu.reason or "nvidia_smi_parse_error"
        thermal_ok = False
        vram_ok = False
        diagnostics.append(gpu_status)
    else:
        thermal_ok = bool(gpu.temp_c is not None and gpu.temp_c <= float(max_temp_c))
        vram_ok = bool(gpu.free_vram_mib is not None and gpu.free_vram_mib >= int(min_free_vram_mib))
        if not thermal_ok:
            diagnostics.append("gpu_temperature_above_quiet_ceiling")
        if not vram_ok:
            diagnostics.append("gpu_free_vram_below_reserve")
        gpu_status = "ok" if (thermal_ok and vram_ok) else ("hot_gpu" if not thermal_ok else "low_vram")

    raw_yield_ms = yield_res.get("yield_latency_ms")
    clean_rel = bool(yield_res.get("clean_release"))
    ret_vram = int(yield_res.get("retained_worker_vram_mib") if yield_res.get("retained_worker_vram_mib") is not None else -1)
    yield_slo_ok = bool(clean_rel and ret_vram == 0 and raw_yield_ms is not None and float(raw_yield_ms) < float(yield_budget_ms))
    if not yield_slo_ok:
        diagnostics.append("steam_yield_slo_exceeded")
    zero_game_lag = bool((not had_game_lag) and yield_slo_ok and thermal_ok and vram_ok)

    # Optional outbound OpenTrainer heartbeat
    ot_result: dict[str, Any] = {"attempted": False, "ok": True, "status": "skipped"}
    if send_opentrainer_heartbeat:
        if not account_ok or resolved_key is None:
            ot_result = {"attempted": False, "ok": False, "status": "blocked_by_account_diagnostics"}
        else:
            hb_payload = {
                "host_label": sanitize_host_label(host_label),
                "gpu_name": gpu.name if (gpu.present and not gpu.parse_error) else None,
                "vram_total_mib": gpu.total_vram_mib if (gpu.present and not gpu.parse_error and (gpu.total_vram_mib or 0) >= MIN_REPORTABLE_VRAM_TOTAL_MIB) else None,
                "vram_free_mib": gpu.free_vram_mib if (gpu.present and not gpu.parse_error) else None,
                "network_mode": "outbound-https-relay",
                "training_stack": "qlora-moe+diloco+amcp-aglet+boinc",
                "active_campaign": CAMPAIGN_ID,
            }
            ot_status, ot_body = _invoke_http_fn(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/opentrainer/heartbeat",
                payload=hb_payload,
                headers={"Authorization": f"Bearer {resolved_key}"},
            )
            network_calls += 1
            ot_ok = bool(ot_status == 200 and ot_body.get("object") == OPENTRAINER_HB_ACK_OBJECT)
            if not ot_ok:
                diagnostics.append(f"opentrainer_heartbeat_http_{ot_status}")
            ot_result = {
                "attempted": True,
                "http_status": ot_status,
                "ok": ot_ok,
                "network_mode": "outbound-https-relay",
                "node_id": ((ot_body.get("node") or {}).get("node_id") if isinstance(ot_body.get("node"), Mapping) else None),
            }

    # Optional queue flush + zero-PII CES feedback dispatch (NO queue file created when empty & HTTP 200 succeeds!)
    ces_queue = ReconnectionCesQueue(q_path)
    flush_res: dict[str, Any] | None = None
    if flush_ces_queue:
        flush_res = ces_queue.flush(clean_gw, http_fn=http_fn)
        network_calls += int(flush_res["network_calls"])
        file_writes += int(flush_res["file_writes"])

    ces_res: dict[str, Any] = {"submitted": False, "status": "skipped", "spooled": False, "quarantined": False}
    if submit_ces:
        plat = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}.get(platform.system(), "linux")
        ces_payload = {
            "profile": norm_profile,
            "ease_rating": int(ease_rating),
            "zero_game_lag": zero_game_lag,
            "platform": plat,
            "note": "",
        }
        idem_key = compute_ces_idempotency_key(ces_payload, account_ref=account_handle or masked_key, salt=idempotency_salt)
        c_status, c_body = _invoke_http_fn(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/onboarding/feedback",
            payload=ces_payload,
            headers={"Idempotency-Key": idem_key},
        )
        network_calls += 1
        if 200 <= c_status < 300:
            ces_res = {
                "submitted": True,
                "http_status": c_status,
                "status": "delivered",
                "spooled": False,
                "quarantined": False,
                "idempotency_key": idem_key,
                "total_responses": c_body.get("total_responses"),
            }
        elif c_status in PERMANENT_CES_STATUSES or (400 <= c_status < 500 and c_status not in RETRIABLE_CES_STATUSES):
            ces_res = {
                "submitted": True,
                "http_status": c_status,
                "status": "quarantined",
                "spooled": False,
                "quarantined": True,
                "idempotency_key": idem_key,
            }
            diagnostics.append(f"ces_rejected_http_{c_status}")
        else:
            ces_queue.enqueue(ces_payload, idempotency_key=idem_key, last_status=c_status)
            file_writes += 1
            ces_res = {
                "submitted": True,
                "http_status": c_status,
                "status": "spooled_for_retry",
                "spooled": True,
                "quarantined": False,
                "idempotency_key": idem_key,
            }

    elapsed_s = round(max(0.0, clock_fn() - t_start), 4)
    overall_ok = bool(
        account_ok
        and sec_check["ok"]
        and thermal_ok
        and vram_ok
        and yield_slo_ok
        and ot_result["ok"]
        and not ces_res["quarantined"]
    )
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": ACTIVATION_OBJECT,
            "lang": lang_code,
            "gateway": clean_gw,
            "ok": overall_ok,
            "exit_code": 0 if overall_ok else 2,
            "compliance_ok": True,
            "compliance": compliance,
            "api_key_masked": masked_key,
            "key_source": key_source,
            "account": {
                "ok": account_ok,
                "handle": account_handle,
                "role": account_role,
                "consumer_role_restricted": consumer_role_restricted,
                "remote_paused": remote_paused,
                "remote_vram_cap_mib": remote_vram_cap_mib,
                "remote_vram_cap_low": remote_vram_cap_low,
                "producer_fleet_state": producer_fleet_state,
                "offline_ttl_expired": offline_ttl_expired,
                "awaiting_first_heartbeat": awaiting_first_hb,
            },
            "workspace_security": sec_check,
            "gpu": {
                "present": gpu.present,
                "status": gpu_status,
                "name": gpu.name,
                "total_vram_mib": gpu.total_vram_mib,
                "free_vram_mib": gpu.free_vram_mib,
                "temp_c": gpu.temp_c,
                "thermal_ok": thermal_ok,
                "vram_ok": vram_ok,
            },
            "steam_preflight": {
                **steam_info,
                **yield_res,
                "yield_budget_ms": float(yield_budget_ms),
                "yield_slo_ok": yield_slo_ok,
                "zero_game_lag": zero_game_lag,
            },
            "opentrainer_heartbeat": ot_result,
            "ces_feedback": ces_res,
            "ces_queue_flush": flush_res,
            "ces_queue": ces_queue.snapshot(),
            "diagnostics": diagnostics,
            "elapsed_s": elapsed_s,
            "within_60s_slo": elapsed_s < 60.0,
            "network_calls": network_calls,
            "file_writes": file_writes,
            "subprocess_calls": subprocess_calls,
            "inference_calls": 0,
            "summary": _msg(
                lang_code,
                f"Registered-producer activation verified cleanly ({'OK' if overall_ok else 'BLOCKED'}; {FINANCIAL_PROMISE}).",
                f"Activation producteur enregistré vérifiée ({'OK' if overall_ok else 'BLOQUÉE'} ; {FINANCIAL_PROMISE}).",
            ),
        }
    )


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render an FR/EN SHA-256-receipted Markdown summary without secrets or static figures."""
    lang_code = normalize_lang(str(report.get("lang") or "en"))
    lines = [
        f"# {_msg(lang_code, 'HiveMind Registered-Producer Activation & PR #949 Gate', 'Activation Producteur Enregistré & Portail PR #949 HiveMind')}",
        "",
        f"- **Object**: `{report.get('object', '')}`",
        f"- **Status**: `{'PASS' if report.get('ok', report.get('compliance_ok')) else 'FAIL'}` (exit `{report.get('exit_code', 0)}`)",
        f"- **Personal-PC / FinOps**: `{FINANCIAL_PROMISE}` (`AGPL-3.0`, personal home PCs only)",
    ]
    if "readiness_gates_projected" in report:
        lines.append(f"- **Readiness Gates**: `{report.get('readiness_gates_projected')}`")
    if "live_build_sha" in report:
        lines.append(f"- **Build SHA**: `{report.get('live_build_sha')}` (stale=`{report.get('stale_cloud_run_revision')}`)")
    if "api_key_masked" in report:
        lines.append(f"- **Credential**: `{report.get('api_key_masked')}` (source=`{report.get('key_source')}`)")
    if isinstance(report.get("gpu"), Mapping):
        lines.append(f"- **GPU Status**: `{report['gpu'].get('status')}` | **Zero-Lag**: `{(report.get('steam_preflight') or {}).get('zero_game_lag')}`")
    lines.append(f"- **Receipt SHA-256**: `{report.get('receipt_sha256', '')}`")
    lines.append(f"- **Summary**: {report.get('summary', '')}")
    rendered = "\n".join(lines) + "\n"
    return _assert_safe_output(rendered)


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a compact FR/EN plain-text CLI summary without secrets or static figures."""
    lang_code = normalize_lang(str(report.get("lang") or "en"))
    status = "OK" if report.get("ok", report.get("compliance_ok")) else "FAIL"
    lines = [
        f"[HiveMind] {status} | exit={report.get('exit_code', 0)} | {FINANCIAL_PROMISE} | AGPL-3.0",
        str(report.get("summary") or ""),
    ]
    if "readiness_gates_projected" in report:
        fin = report.get("finops_guidance") or {}
        lines.append(f"Gates: {report['readiness_gates_projected']} | Dry-run: {fin.get('dry_run_command', '')}")
    if report.get("diagnostics"):
        lines.append(f"{_msg(lang_code, 'Diagnostics', 'Diagnostics')}: {', '.join(str(d) for d in report['diagnostics'])}")
    lines.append(f"Receipt SHA-256: {report.get('receipt_sha256', '')}")
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Construct stdlib CLI parser supporting PR #949 gate, activation verifier, and CES queue ops."""
    p = argparse.ArgumentParser(description="Zero-Inference PR #949 Cloud Run Gate & Registered-Producer Activation Verifier.")
    p.add_argument("--gateway", default=DEFAULT_GATEWAY)
    p.add_argument("--pr949-gate", action="store_true", help="Run PR #949 live revision & fleet readiness gate.")
    p.add_argument("--expected-sha", default=None)
    p.add_argument("--target-online-producers", type=int, default=DEFAULT_TARGET_ONLINE_PRODUCERS)
    p.add_argument("--require-holiday-routes", action="store_true")
    p.add_argument("--deploy-script", default=None)
    p.add_argument("--enforce-personal-pc", action="store_true", help="Run strictly local Personal-PC compliance check only.")
    p.add_argument("--credentials-path", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--workspace-dir", default=None)
    p.add_argument("--continue-config-path", default=None)
    p.add_argument("--send-opentrainer-heartbeat", action="store_true")
    p.add_argument("--submit-ces", action="store_true")
    p.add_argument("--flush-ces-queue", action="store_true")
    p.add_argument("--purge-queue", action="store_true")
    p.add_argument("--queue-path", default=None)
    p.add_argument("--profile", default="gamer")
    p.add_argument("--ease-rating", type=int, default=5)
    p.add_argument("--had-game-lag", action="store_true")
    p.add_argument("--host-label", default="personal-pc")
    p.add_argument("--max-temp-c", type=float, default=DEFAULT_MAX_TEMP_C)
    p.add_argument("--min-free-vram-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB)
    p.add_argument("--yield-budget-ms", type=float, default=DEFAULT_YIELD_BUDGET_MS)
    p.add_argument("--lang", default="en", choices=("en", "fr"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--markdown", action="store_true")
    return p


def _emit_report(payload: Mapping[str, Any], *, as_json: bool, as_markdown: bool) -> None:
    if as_json:
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        _assert_safe_output(text)
        print(text)
    elif as_markdown:
        print(render_markdown_report(payload), end="")
    else:
        print(render_cli_report(payload))


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | str | None = None,
) -> int:
    """CLI entrypoint returning ``0`` on pass or ``2`` on compliance/gate failure."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.purge_queue:
        q = ReconnectionCesQueue(args.queue_path or (Path(home) if home else Path.home()) / DEFAULT_QUEUE_REL)
        purged = _attach_receipt({"schema": SCHEMA, "object": "hivemind.reconnection_ces_queue.purge", "lang": args.lang, **q.purge(), "ok": True, "exit_code": 0, "summary": "Local CES queue purged."})
        _emit_report(purged, as_json=args.json, as_markdown=args.markdown)
        return 0
    if args.enforce_personal_pc:
        comp = evaluate_personal_pc_compliance(environ=environ, cwd=cwd or args.workspace_dir, lang=args.lang, home=home)
        _emit_report(comp, as_json=args.json, as_markdown=args.markdown)
        return int(comp["exit_code"])
    if args.pr949_gate or args.require_holiday_routes:
        gate = verify_pr949_live_revision_gate(
            args.gateway,
            expected_sha=args.expected_sha,
            target_online_producers=args.target_online_producers,
            require_holiday_routes=args.require_holiday_routes,
            deploy_script_path=args.deploy_script,
            http_fn=http_fn,
            environ=environ,
            cwd=cwd or args.workspace_dir,
            home=home,
            lang=args.lang,
        )
        _emit_report(gate, as_json=args.json, as_markdown=args.markdown)
        return int(gate["exit_code"])
    res = verify_registered_producer_activation(
        args.gateway,
        credentials_path=args.credentials_path,
        api_key=args.api_key,
        workspace_dir=args.workspace_dir,
        continue_config_path=args.continue_config_path,
        send_opentrainer_heartbeat=args.send_opentrainer_heartbeat,
        submit_ces=args.submit_ces,
        flush_ces_queue=args.flush_ces_queue,
        queue_path=args.queue_path,
        profile=args.profile,
        ease_rating=args.ease_rating,
        had_game_lag=args.had_game_lag,
        host_label=args.host_label,
        max_temp_c=args.max_temp_c,
        min_free_vram_mib=args.min_free_vram_mib,
        yield_budget_ms=args.yield_budget_ms,
        http_fn=http_fn,
        environ=environ,
        cwd=cwd,
        home=home,
        lang=args.lang,
    )
    _emit_report(res, as_json=args.json, as_markdown=args.markdown)
    return int(res["exit_code"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
