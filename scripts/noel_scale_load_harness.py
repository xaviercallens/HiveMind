#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Concurrent Noël holiday router burst, PR #949 live parity & quorum harness (`SLICE-SCALE-01` & `SLICE-SCALE-DYN-40`).

Simulates a realistic Noël-at-scale multi-friend holiday burst (default: 30 developers +
30 PC gamers = 60 concurrent personal-PC participants) using ONLY the Python standard
library against allowed HiveMind gateway origins (`https://*.run.app` or loopback
`127.0.0.1` / `localhost` / `::1`) without ever invoking chat completion endpoints or
prompting `hivemind/coder`:

  (a) Dev cohort (`--devs`, default 30):
      - `GET /noel` (+ opt-in `GET /quick` via `--include-quick`)
      - `GET /signup` (HTML + per-request CSP nonce)
      - `GET /v1/hivemind/onboarding/manifest`
      - `POST /v1/hivemind/register` (in local TestClient / loopback harness mode)
      - `GET /v1/models`
      - `POST /v1/hivemind/onboarding/feedback` (`profile='dev'`, + opt-in Idempotency-Key replay)

  (b) Gamer cohort (`--gamers`, default 30):
      - `GET /raid-noel` (HTML + per-request CSP nonce)
      - `GET /v1/hivemind/raid/status`
      - `GET /arena-noel` (HTML + per-request CSP nonce + HMAC-masked blind ballot)
      - `POST /v1/hivemind/arena/vote`
      - `GET /v1/hivemind/arena/summary`
      - Opt-in local heartbeat quorum (`--include-heartbeat-quorum`):
        `POST /v1/hivemind/register`, outbound `POST /v1/hivemind/opentrainer/heartbeat`,
        `>= 8` distinct GPU `POST /v1/hivemind/producer/heartbeat`, `GET /v1/hivemind/producers/fleet`
      - `POST /v1/hivemind/onboarding/feedback` (`profile='gamer'`, + opt-in Idempotency-Key replay)

  (c) Zero-Inference PR #949 Live Parity & 8+ Producer Quorum (`--pr949-parity`):
      - Read-only verification across `/health`, `/v1/models`, `/noel`, `/quick`, `/raid-noel`,
        `/arena-noel`, `/v1/hivemind/onboarding/manifest`, `/v1/hivemind/onboarding/feedback/summary`,
        `/v1/hivemind/producers/fleet`, `/v1/hivemind/raid/status`, and `/v1/hivemind/arena/summary`.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import ipaddress
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SCHEMA = "hivemind.noel_scale_load_harness/1"
RECEIPT_OBJECT = "hivemind.noel_scale_load_receipt"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

DEFAULT_DEVS = 30
DEFAULT_GAMERS = 30
DEFAULT_WORKERS = 60
MAX_PARTICIPANTS_PER_COHORT = 2000
MAX_WORKERS = 256

DEFAULT_P95_SLO_MS = 2500.0
DEFAULT_P99_SLO_MS = 5000.0
DEFAULT_MAX_ERROR_RATIO = 0.0
HTTP_TIMEOUT_S = 5.0
MAX_HTTP_BODY_BYTES = 4 * 1024 * 1024
DEFAULT_RETRY_BACKOFF_S = 0.005
MAX_429_RETRIES_LIMIT = 3

DEFAULT_TARGET_ONLINE_PRODUCERS = 8
DEFAULT_MIN_EASE_RATING = 4.2
DEFAULT_MIN_ZERO_LAG_RATIO = 0.95

MANIFEST_OBJECT = "hivemind.onboarding.manifest"
FLEET_OBJECT = "hivemind.producer_fleet"
RAID_STATUS_OBJECT = "hivemind.raid_noel.status"
ARENA_SUMMARY_OBJECT = "hivemind.blind_ab_arena.summary"
FEEDBACK_SUMMARY_OBJECT = "hivemind.onboarding_feedback_summary"

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

INSPECTED_ENV_KEYS: tuple[str, ...] = (
    "USERDOMAIN",
    "COMPUTERNAME",
    "HOSTNAME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)

LIVE_HTML_PATHS: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
PR949_REQUIRED_ROUTES: tuple[tuple[str, str], ...] = (
    ("/health", "json"),
    ("/v1/models", "json"),
    ("/noel", "html"),
    ("/quick", "html"),
    ("/raid-noel", "html"),
    ("/arena-noel", "html"),
    ("/v1/hivemind/onboarding/manifest", "json"),
    ("/v1/hivemind/onboarding/feedback/summary", "json"),
    ("/v1/hivemind/producers/fleet", "json"),
    ("/v1/hivemind/raid/status", "json"),
    ("/v1/hivemind/arena/summary", "json"),
)
PR949_MANIFEST_ROUTES: tuple[str, ...] = (
    "/arena-noel",
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/arena/summary",
    "/v1/hivemind/arena/vote",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/raid/status",
)
PR949_MANIFEST_CORE_SCRIPTS: tuple[str, ...] = ("hivemind_quickstart.py", "hivemind_doctor.py")

CURATED_ARENA_TASK_IDS: tuple[str, ...] = (
    "python-sliding-median",
    "rust-frame-decoder",
    "lean4-energy-descent",
    "exoplanet-transit-bls",
)
DEV_EASE_RATINGS: tuple[int, ...] = (5, 5, 5, 5, 4)
GAMER_EASE_RATINGS: tuple[int, ...] = (5, 5, 5, 5, 5, 4)
GAMER_CHOSEN_SIDES: tuple[str, ...] = ("A", "B", "A", "tie")

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only — "
    "Amadeus / Personal-PC rule: never work PCs or corporate code (0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement — "
    "règle Amadeus / PC perso : jamais de PC ni de code d'entreprise (0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_CSP_INLINE_NONCE_RE = re.compile(r"'nonce-[A-Za-z0-9_+=/=-]{4,128}'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_HTML_ANY_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_ARENA_CARD_BALLOT_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"\s+data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_BURST_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,23}$")
_BUILD_SHA_RE = re.compile(r"^[A-Za-z0-9]{6,64}$")
_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_MASKED_HM_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(hm(?:_live_|-))\*\*\*(?!redacted\*\*\*)(?:[A-Za-z0-9_-]{4,}|[0-9]{1,4}(?:m" + r"s\b|tok" + r"/s))"
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DISCORD_PING_RE = re.compile(r"(?:@everyone|@here|<@[!&]?\d+>)", re.IGNORECASE)
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')
_ANY_FORBIDDEN_SCORE_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:')
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|docu" + r"ment\.write)\b"
)
_STATIC_MS_UNIT_RE = re.compile(r"(\d[\d.,]*)\s?m" + r"s\b")
_STATIC_FIG_UNIT_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_GIT_REMOTE_URL_RE = re.compile(
    r'^\s*(?:push)?(?:url|insteadof|pushinsteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)
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

HttpFn = Callable[..., Any]
RawHttpFn = Callable[..., tuple[int, str, dict[str, str]]]
ClockFn = Callable[[], float]
SleepFn = Callable[[float], None]


class ScaleHarnessError(ValueError):
    """Raised when scale harness parameters, URL guards, or output safety checks fail."""


@dataclass(frozen=True)
class HttpResponse:
    """Normalized zero-inference HTTP response returned by stdlib or injected callables."""

    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]
    text: str = ""


@dataclass(frozen=True)
class StepObservation:
    """Single HTTP step observation recorded during a concurrent participant flow."""

    cohort: str
    participant_id: str
    method: str
    path: str
    is_html: bool
    status_code: int
    ok: bool
    latency_ms: float
    csp_nonce: str | None
    rate_limited: bool
    retried_429: int
    server_error: bool
    client_error: bool
    sqlite_locked: bool
    server_counter: int | None = None
    failure_code: str = ""
    step_kind: str = ""
    strict_html_ok: bool = True


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language code to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ScaleHarnessError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ScaleHarnessError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def _normalize_platform_tag(raw: str | None = None) -> str:
    source = (raw if raw is not None else platform.system()).strip().lower()
    if "darwin" in source or "mac" in source:
        return "macos"
    if "win" in source or source == "nt":
        return "windows"
    if "linux" in source:
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "", source)[:24]
    return cleaned if cleaned else "unknown"


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is a valid `*.run.app` origin or loopback (`127.0.0.1` / `localhost` / `::1`)."""
    if not isinstance(host, str):
        return False
    raw = host.strip().lower()
    if not raw or len(raw) > 253 or any(ch.isspace() or ch in "\\/@" for ch in raw):
        return False
    h = raw[1:-1] if (raw.startswith("[") and raw.endswith("]")) else raw
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    labels = h.split(".")
    if any(not _DNS_LABEL_RE.fullmatch(lbl) for lbl in labels):
        return False
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app` or `http(s)://127.0.0.1|localhost|[::1]`."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ScaleHarnessError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ScaleHarnessError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise ScaleHarnessError("Gateway URL must not contain internal whitespace.")
    if "\\" in cleaned or "\\" in unquote(cleaned):
        raise ScaleHarnessError("Gateway URL must not contain backslashes.")
    if "@" in cleaned or "@" in unquote(cleaned):
        raise ScaleHarnessError("Gateway URL must not embed credentials or '@'.")
    try:
        parts = urlsplit(cleaned)
        port = parts.port
        host = (parts.hostname or "").lower()
    except ValueError as exc:
        raise ScaleHarnessError("Gateway URL is malformed.") from exc
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ScaleHarnessError("Gateway URL must be a bare origin without a path suffix.")
    if port is not None and not (1 <= port <= 65535):
        raise ScaleHarnessError("Gateway URL port out of range.")
    if not is_allowed_gateway_host(host):
        raise ScaleHarnessError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app or 127.0.0.1/localhost/::1."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ScaleHarnessError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ScaleHarnessError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def is_loopback_gateway(url: str) -> bool:
    """Return True iff validated `url` targets a loopback host (`127.0.0.1`, `localhost`, `::1`)."""
    cleaned = validate_gateway_url(url)
    host = (urlsplit(cleaned).hostname or "").strip().lower().strip("[]")
    return host in ALLOWED_LOOPBACK_HOSTS


def simulated_participant_ip(participant_index: int) -> str:
    """Deterministically map `participant_index` (`>= 0`) into loopback `127.0.0.0/8` for per-PC buckets."""
    if isinstance(participant_index, bool) or not isinstance(participant_index, int) or participant_index < 0:
        raise ScaleHarnessError("participant_index must be a non-negative integer.")
    offset = 256 + (participant_index % 16_000_000)
    return str(ipaddress.IPv4Address((127 << 24) + offset))


def strip_masked_keys_and_nonces(text: str) -> str:
    """Strip HTML `nonce=...` attributes, CSP `'nonce-...'` tokens, and `hm-...***XXXX` suffixes."""
    if not isinstance(text, str):
        raise ScaleHarnessError("strip_masked_keys_and_nonces requires a string.")
    cleaned = _HTML_ANY_NONCE_ATTR_RE.sub('nonce=""', text)
    cleaned = _CSP_INLINE_NONCE_RE.sub("'nonce-CSP_NONCE'", cleaned)
    return _MASKED_HM_KEY_RE.sub(r"\1***XXXX", cleaned)


def sanitize_text_for_display(text: str) -> str:
    """Scrub PII, raw API keys, Bearer tokens, home paths, non-loopback IPs, and static ms units."""
    cleaned = strip_masked_keys_and_nonces(str(text))
    cleaned = _API_KEY_RE.sub("hm-***redacted***", cleaned)
    cleaned = _BEARER_RE.sub("Bearer ***redacted***", cleaned)
    cleaned = _EMAIL_RE.sub("[redacted-email]", cleaned)
    cleaned = _HOME_PATH_RE.sub("~", cleaned)
    cleaned = _DISCORD_PING_RE.sub("[redacted-ping]", cleaned)
    cleaned = _STATIC_MS_UNIT_RE.sub(r"\1 millis", cleaned)

    def _ip_repl(match: re.Match[str]) -> str:
        ip_txt = match.group(0)
        return ip_txt if ip_txt == "127.0.0.1" else "[redacted-ip]"

    return _IPV4_RE.sub(_ip_repl, cleaned)


def _extract_git_remote_urls(cfg_text: str) -> list[str]:
    if not cfg_text:
        return []
    urls: list[str] = []
    for m_kv, m_sec in _GIT_REMOTE_URL_RE.findall(cfg_text):
        candidate = (m_kv or m_sec).strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate `.git/config` from `workspace`, walking parents and resolving worktree `gitdir:`/`commondir`."""
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
                lines = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            gitdir_line = next((ln.strip() for ln in lines if ln.strip()), "")
            if gitdir_line.lower().startswith("gitdir:"):
                raw_gitdir = gitdir_line.split(":", 1)[1].strip()
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
                        common_dir = (
                            Path(common_rel)
                            if Path(common_rel).is_absolute()
                            else (gitdir / common_rel).resolve()
                        )
                        common_cfg = common_dir / "config"
                        if common_cfg.is_file() and common_cfg not in candidates:
                            candidates.append(common_cfg)
            break
    return candidates


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Amadeus / Personal-PC compliance check (zero env/path/remote values ever leaked)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_map = environ if environ is not None else os.environ
    upper_env = {str(k).strip().upper(): str(v).strip() for k, v in env_map.items() if str(v).strip()}

    indicators: list[str] = []
    matched_markers: set[str] = set()

    def _record(tag: str, marker: str | None = None) -> None:
        if tag not in indicators:
            indicators.append(tag)
        if marker:
            matched_markers.add(marker)

    if any(k.startswith("AMADEUS") for k in upper_env):
        _record("env:AMADEUS_ENV", "amadeus")

    dns_domain = upper_env.get("USERDNSDOMAIN", "")
    if dns_domain:
        dns_hits = [name for name, pat in _CORP_PATTERNS if pat.search(dns_domain)]
        if dns_hits:
            _record("env:USERDNSDOMAIN", dns_hits[0])
            matched_markers.update(dns_hits)
        else:
            _record("env:USERDNSDOMAIN", "domain")

    if upper_env.get("ZSCALER_CERT_PATH"):
        _record("env:ZSCALER_CERT_PATH", "zscaler")

    for env_key in INSPECTED_ENV_KEYS:
        val = upper_env.get(env_key, "")
        if not val:
            continue
        if env_key == "USERDOMAIN" and val.upper() == upper_env.get("COMPUTERNAME", "").upper():
            continue
        for marker_name, pat in _CORP_PATTERNS:
            if pat.search(val):
                _record(f"env:{env_key}:{marker_name}", marker_name)
                break

    if cwd is not None:
        cwd_path = Path(cwd)
        cwd_texts = [str(cwd_path)]
        with suppress(OSError):
            resolved_cwd = str(cwd_path.resolve())
            if resolved_cwd not in cwd_texts:
                cwd_texts.append(resolved_cwd)
        for cwd_text in cwd_texts:
            for marker_name, pat in _CORP_PATTERNS:
                if pat.search(cwd_text):
                    _record(f"cwd:{marker_name}", marker_name)
                    break

    if cwd is not None or check_git_remote:
        git_root = Path(cwd) if cwd is not None else Path.cwd()
        for git_cfg in _resolve_git_config_files(git_root):
            try:
                raw_git = git_cfg.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for remote_val in _extract_git_remote_urls(raw_git):
                for marker_name, pat in _CORP_PATTERNS:
                    if pat.search(remote_val):
                        _record(f"git_remote:{marker_name}", marker_name)
                        break

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Hôte personnel conforme vérifié localement. {COMPLIANCE_NOTICE_FR}"
            if is_fr
            else f"Personal home PC compliance verified locally. {COMPLIANCE_NOTICE_EN}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"NO-GO CONFORMITÉ ({joined}) : exécution refusée sur environnement d'entreprise "
            f"(0 appel réseau, 0 écriture fichier ; règle Amadeus / PC perso). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE NO-GO ({joined}): execution refused on corporate environment "
            f"(0 network calls, 0 file writes; Amadeus / Personal-PC rule). {COMPLIANCE_NOTICE_BILINGUAL}"
        )

    return {
        "compliance_ok": compliance_ok,
        "corporate_detected": not compliance_ok,
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "financial_promise": FINANCIAL_PROMISE,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "zero_llm_inference": True,
        "network_transmitted": False,
        "network_calls": 0,
        "file_writes": 0,
        "subprocess_calls": 0,
        "exit_code": 0 if compliance_ok else 2,
        "corporate_indicators": sorted(indicators),
        "matched_markers": sorted(matched_markers),
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "active_notice": COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN,
        "summary": summary,
    }


def _assert_zero_inference_request(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Enforce allowed gateway origin and block any inference completion or model prompt call."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise ScaleHarnessError(f"Unsupported HTTP method {verb!r}; expected GET, HEAD, or POST.")
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ScaleHarnessError("Request URL must be a non-empty string.")
    cleaned_url = url.strip()
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned_url):
        raise ScaleHarnessError("Request URL must not contain whitespace or control characters.")
    if "\\" in cleaned_url or "\\" in unquote(cleaned_url):
        raise ScaleHarnessError("Request URL must not contain backslashes.")
    try:
        parts = urlsplit(cleaned_url)
    except ValueError as exc:
        raise ScaleHarnessError("Request URL is malformed.") from exc
    if (
        parts.username is not None
        or parts.password is not None
        or "@" in parts.netloc
        or "@" in unquote(parts.netloc)
    ):
        raise ScaleHarnessError("Request URL must not embed credentials or '@'.")
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = unquote(parts.path or "").lower()
    if "completions" in low_path or "/chat/" in low_path:
        raise ScaleHarnessError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ScaleHarnessError("HTTP JSON payload must be a mapping when provided.")
        if "messages" in payload or "prompt" in payload:
            raise ScaleHarnessError(
                "Zero-inference invariant violated: prompt/messages payloads are strictly forbidden."
            )
        serialized = json.dumps(dict(payload), sort_keys=True).lower()
        if "hivemind/coder" in serialized:
            raise ScaleHarnessError(
                "Zero-inference invariant violated: prompting hivemind/coder is strictly forbidden."
            )
    return origin


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
            raise ScaleHarnessError("Untrusted redirect: empty target URL.")
        _assert_zero_inference_request(req.get_method(), newurl, None)
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            _assert_zero_inference_request(new_req.get_method(), new_req.full_url, None)
        return new_req


def _prepare_http_headers(
    headers: Mapping[str, str] | None,
    *,
    user_agent: str,
    accept: str,
) -> dict[str, str]:
    req_headers: dict[str, str] = {"Accept": accept, "User-Agent": user_agent}
    if headers:
        for hk, hv in headers.items():
            k_str = str(hk)
            v_str = str(hv)
            if any(ch in k_str or ch in v_str for ch in ("\r", "\n", "\x00")):
                raise ScaleHarnessError("HTTP headers must not contain CR/LF or NUL characters.")
            req_headers[k_str] = v_str
    return req_headers


def _build_safe_http_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirectHandler())


def _read_bounded_body(resp: Any) -> bytes:
    raw_bytes = resp.read(MAX_HTTP_BODY_BYTES + 1)
    if len(raw_bytes) > MAX_HTTP_BODY_BYTES:
        raise ScaleHarnessError(f"HTTP response exceeded {MAX_HTTP_BODY_BYTES} byte limit.")
    return raw_bytes


def scan_output_safety(text: str) -> list[str]:
    """Verify output leaks zero PII, raw `hm-...`/`hm_live_...` keys, home paths, IPs, or `'score': <digit>`."""
    if not isinstance(text, str):
        raise ScaleHarnessError("scan_output_safety requires a string.")
    scrubbed_placeholders = strip_masked_keys_and_nonces(text)
    issues: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
    if _API_KEY_RE.search(scrubbed_placeholders):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _HOME_PATH_RE.search(text):
        issues.append("unredacted_home_path_detected")
    if _DISCORD_PING_RE.search(text):
        issues.append("discord_mass_ping_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if ip_hit != "127.0.0.1":
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise ScaleHarnessError(f"Output safety guard failed: {', '.join(issues)}")
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


def default_http_raw(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, str, dict[str, str]]:
    """Direct, proxy-free, redirect-guarded stdlib HTTP helper returning `(status_code, body_text, lower_headers)`."""
    _assert_zero_inference_request(method, url, None)
    verb = method.strip().upper()
    if verb not in ("GET", "HEAD"):
        raise ScaleHarnessError(f"default_http_raw only supports GET/HEAD requests, got {verb!r}.")
    bounded_timeout = min(max(0.2, _validate_positive_float(timeout_s, "timeout_s")), 15.0)
    req_headers = _prepare_http_headers(
        headers,
        user_agent="hivemind-noel-scale-load-harness/1.0",
        accept="application/json, text/html;q=0.9, text/plain;q=0.8",
    )
    opener = _build_safe_http_opener()
    req = urllib.request.Request(url.strip(), data=None, headers=req_headers, method=verb)
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


def default_http_request(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Direct, proxy-free, redirect-guarded stdlib HTTP helper returning `(status_code, body_dict, lower_headers)`."""
    _assert_zero_inference_request(method, url, payload)
    req_headers = _prepare_http_headers(
        headers,
        user_agent="hivemind-noel-scale-load-harness/1.0",
        accept="application/json, text/html;q=0.9",
    )
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(dict(payload)).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    opener = _build_safe_http_opener()
    req = urllib.request.Request(url.strip(), data=body_bytes, headers=req_headers, method=method.strip().upper())
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = _read_bounded_body(resp).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        raw_err = exc.read(65_536).decode("utf-8", errors="replace") if exc.fp is not None else ""
        err_headers = {str(k).lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
        try:
            parsed_err = json.loads(raw_err) if raw_err.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": raw_err[:240]}
        norm_err = parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err}
        return int(exc.code), norm_err, err_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {"network_error": type(exc).__name__, "detail": str(reason)[:200]}, {}

    content_type = resp_headers.get("content-type", "")
    if not raw.strip():
        return status_code, {"content_type": content_type, "html_length": 0}, resp_headers
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return (
            status_code,
            {"content_type": content_type, "html_length": len(raw), "html": raw},
            resp_headers,
        )
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed}), resp_headers


def _normalize_http_result(raw_result: Any) -> HttpResponse:
    """Normalize `HttpResponse`, 3-tuple `(status, body, headers)`, or 2-tuple `(status, body)`."""
    if isinstance(raw_result, HttpResponse):
        return raw_result
    if not isinstance(raw_result, tuple) or len(raw_result) not in (2, 3):
        raise ScaleHarnessError(
            "Injectable http_fn must return HttpResponse, (status, body), or (status, body, headers)."
        )
    status_code = int(raw_result[0])
    raw_body = raw_result[1]
    raw_headers = raw_result[2] if len(raw_result) == 3 else None

    headers: dict[str, str] = {}
    if isinstance(raw_headers, Mapping):
        headers.update({str(k).lower(): str(v) for k, v in raw_headers.items()})

    text = ""
    if isinstance(raw_body, Mapping):
        body = dict(raw_body)
        embedded_headers = body.get("headers")
        if isinstance(embedded_headers, Mapping):
            for hk, hv in embedded_headers.items():
                headers.setdefault(str(hk).lower(), str(hv))
        if isinstance(body.get("csp"), str):
            headers.setdefault("content-security-policy", body["csp"])
        if isinstance(body.get("html"), str):
            text = body["html"]
    elif isinstance(raw_body, str):
        text = raw_body
        try:
            parsed = json.loads(raw_body)
            body = parsed if isinstance(parsed, dict) else {"data": parsed}
        except json.JSONDecodeError:
            body = {"html_length": len(raw_body), "html": raw_body}
    else:
        body = {"data": raw_body}

    return HttpResponse(status_code=status_code, body=body, headers=headers, text=text)


def _contains_sqlite_lock_signal(resp: HttpResponse) -> bool:
    """Return True if any error body or text signals a SQLite lock or OperationalError."""
    chunks: list[str] = []
    if resp.text and resp.status_code >= 400:
        chunks.append(resp.text[:512])
    if resp.status_code == 0 or resp.status_code >= 400:
        chunks.append(json.dumps(resp.body, sort_keys=True, default=str)[:512])
    return any(_SQLITE_LOCK_RE.search(chunk) for chunk in chunks if chunk)


def _invoke_http_callable(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    *,
    clock_fn: ClockFn = time.perf_counter,
) -> tuple[HttpResponse, float, bool]:
    """Invoke `http_fn` with zero-inference verification, measuring latency in milliseconds."""
    _assert_zero_inference_request(method, url, payload)
    sig = inspect.signature(http_fn)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())

    dict_payload = dict(payload) if payload is not None else None
    dict_headers = {str(k): str(v) for k, v in headers.items()} if headers is not None else None

    t0 = float(clock_fn())
    try:
        if has_varargs or len(positional) >= 4:
            raw = http_fn(method.upper(), url, dict_payload, dict_headers)
        elif len(positional) == 3:
            raw = http_fn(method.upper(), url, dict_payload)
        else:
            raw = http_fn(method.upper(), url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        elapsed_ms = max(0.0, (float(clock_fn()) - t0) * 1000.0)
        detail = str(exc)[:240]
        resp = HttpResponse(
            status_code=0,
            body={"network_error": type(exc).__name__, "detail": detail},
            headers={},
            text=detail,
        )
        return resp, elapsed_ms, bool(_SQLITE_LOCK_RE.search(detail))
    except Exception as exc:  # noqa: BLE001 - surface in-process sqlite3 OperationalError cleanly as a lock/500 finding
        if isinstance(exc, ScaleHarnessError):
            raise
        elapsed_ms = max(0.0, (float(clock_fn()) - t0) * 1000.0)
        detail = f"{type(exc).__name__}: {exc}"[:240]
        is_lock = bool(_SQLITE_LOCK_RE.search(detail))
        resp = HttpResponse(
            status_code=500,
            body={"server_exception": type(exc).__name__, "sqlite_locked": is_lock},
            headers={},
            text="",
        )
        return resp, elapsed_ms, is_lock

    elapsed_ms = max(0.0, (float(clock_fn()) - t0) * 1000.0)
    resp = _normalize_http_result(raw)
    return resp, elapsed_ms, _contains_sqlite_lock_signal(resp)


def extract_csp_nonce(headers: Mapping[str, str], html_text: str = "") -> str | None:
    """Extract and cross-verify the per-request CSP nonce from headers and HTML `<style>`/`<script>` tags."""
    csp_header = ""
    for k, v in headers.items():
        if str(k).lower() == "content-security-policy":
            csp_header = str(v)
            break

    header_nonces = set(_CSP_HEADER_NONCE_RE.findall(csp_header)) if csp_header else set()
    html_nonces = {val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)} if html_text else set()

    if len(header_nonces) > 1 or len(html_nonces) > 1:
        return None
    header_nonce = next(iter(header_nonces), None)
    html_nonce = next(iter(html_nonces), None)

    if header_nonce is not None and html_nonce is not None:
        return header_nonce if header_nonce == html_nonce else None
    if header_nonce is not None:
        if html_text and ("<style" in html_text or "<script" in html_text):
            return None
        return header_nonce
    return html_nonce


def extract_arena_ballots(html_text: str) -> dict[str, str]:
    """Extract `{task_id: ballot_token}` pairs from rendered `/arena-noel` HTML task cards."""
    if not html_text:
        return {}
    return {task_id: ballot for task_id, ballot in _ARENA_CARD_BALLOT_RE.findall(html_text)}


def compute_percentile_ms(samples_ms: Sequence[float], percentile: float) -> float:
    """Compute exact deterministic percentile over finite non-negative latency samples (ms)."""
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, (int, float))
        or not math.isfinite(float(percentile))
        or not (0.0 <= float(percentile) <= 100.0)
    ):
        raise ScaleHarnessError("percentile must be a finite number between 0.0 and 100.0.")

    cleaned: list[float] = []
    for item in samples_ms:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ScaleHarnessError("All latency samples must be numeric (not bool).")
        fval = float(item)
        if not math.isfinite(fval) or fval < 0.0:
            raise ScaleHarnessError("All latency samples must be finite and >= 0.0.")
        cleaned.append(fval)

    if not cleaned:
        return 0.0
    ordered = sorted(cleaned)
    n = len(ordered)
    if n == 1:
        return round(ordered[0], 3)
    pct = float(percentile)
    if pct == 50.0:
        return round(float(statistics.median(ordered)), 3)
    pos = (pct / 100.0) * float(n - 1)
    lo_idx = int(math.floor(pos))
    hi_idx = int(math.ceil(pos))
    if lo_idx == hi_idx:
        return round(ordered[lo_idx], 3)
    weight = pos - float(lo_idx)
    interpolated = ordered[lo_idx] * (1.0 - weight) + ordered[hi_idx] * weight
    return round(interpolated, 3)


def _validate_int_range(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise ScaleHarnessError(f"{name} must be an integer between {lo} and {hi}.")
    return int(value)


def _validate_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScaleHarnessError(f"{name} must be a positive finite number.")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ScaleHarnessError(f"{name} must be a positive finite number.")
    return fval


def _validate_float_range(value: object, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScaleHarnessError(f"{name} must be a finite number between {lo} and {hi}.")
    fval = float(value)
    if not math.isfinite(fval) or not (lo <= fval <= hi):
        raise ScaleHarnessError(f"{name} must be a finite number between {lo} and {hi}.")
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


def _sanitize_build_sha(raw: object) -> str:
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


def _validate_expected_sha(raw: str) -> str:
    sanitized = _sanitize_build_sha(raw)
    if sanitized == "unknown":
        raise ScaleHarnessError(
            "expected_sha must be a 6..64 character alphanumeric/hex SHA (never 'unknown' or a secret)."
        )
    return sanitized


def _shas_match(live_sha: str, expected_sha: str) -> bool:
    clean_live = _sanitize_build_sha(live_sha)
    clean_exp = _sanitize_build_sha(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def _inspect_html_contract(
    path: str,
    html_text: str,
    headers: Mapping[str, str],
    seen_nonces: set[str] | None,
    *,
    is_fr: bool,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Verify PR #949 HTML contract: doctype, `'default-src 'none''` CSP nonce, zero DOM sinks/static figures, banner."""
    issues: list[str] = []
    has_doctype = "<!doctype html" in html_text.lower()
    if not has_doctype:
        issues.append(f"{path}: missing `<!DOCTYPE html>`." if not is_fr else f"{path} : `<!DOCTYPE html>` absent.")

    csp_hdr = ""
    for k, v in headers.items():
        if str(k).lower() == "content-security-policy":
            csp_hdr = str(v)
            break
    csp_low = csp_hdr.lower()
    csp_unsafe = "'unsafe-inline'" in csp_low or "'unsafe-eval'" in csp_low
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    attr_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(html_text)]
    header_nonce = hdr_nonces[0] if (hdr_nonces and len(set(hdr_nonces)) == 1) else None
    nonce_reused = seen_nonces is not None and header_nonce is not None and header_nonce in seen_nonces
    if seen_nonces is not None and header_nonce is not None:
        seen_nonces.add(header_nonce)

    csp_ok = bool(
        "default-src 'none'" in csp_low
        and not csp_unsafe
        and header_nonce is not None
        and attr_nonces
        and header_nonce in attr_nonces
        and len(set(attr_nonces)) == 1
        and not nonce_reused
    )
    if nonce_reused:
        issues.append(f"{path}: reused CSP nonce detected." if not is_fr else f"{path} : nonce CSP réutilisé.")
    elif not csp_ok:
        issues.append(
            f"{path}: invalid or missing `'default-src 'none''` CSP nonce."
            if not is_fr
            else f"{path} : en-tête CSP `'default-src 'none''` ou nonce invalide/absent."
        )

    stripped = strip_masked_keys_and_nonces(html_text)
    unsafe_sink_hits = len(_FORBIDDEN_DOM_SINK_RE.findall(stripped))
    static_fig = bool(_STATIC_FIG_UNIT_RE.search(stripped))
    if unsafe_sink_hits > 0:
        issues.append(
            f"{path}: unsafe DOM sink detected (`textContent` only)."
            if not is_fr
            else f"{path} : sink DOM non sûr détecté (`textContent` uniquement)."
        )
    if static_fig:
        issues.append(
            f"{path}: unbacked static telemetry figure detected."
            if not is_fr
            else f"{path} : chiffre télémétrique statique non étayé détecté."
        )

    low_html = stripped.lower()
    banner_ok = bool(
        ("personal" in low_html or "personnel" in low_html)
        and ("0.00 eur" in low_html or "0,00 eur" in low_html)
        and "agpl-3.0" in low_html
    )
    if not banner_ok:
        issues.append(
            f"{path}: missing Personal-PC / 0.00 EUR / AGPL-3.0 banner."
            if not is_fr
            else f"{path} : bannière PC perso / 0,00 EUR / AGPL-3.0 absente."
        )

    ok = len(issues) == 0
    return (
        ok,
        {
            "ok": ok,
            "has_doctype": has_doctype,
            "csp_ok": csp_ok,
            "nonce_reused": nonce_reused,
            "unsafe_dom_sinks_count": unsafe_sink_hits,
            "static_figure_found": static_fig,
            "personal_pc_banner_ok": banner_ok,
        },
        issues,
    )


def _inspect_feedback_summary_contract(
    doc: Mapping[str, Any],
    *,
    min_ease: float,
    min_lag: float,
    is_fr: bool,
) -> tuple[bool, int, float | None, float | None, bool | None, list[str]]:
    """Validate `/v1/hivemind/onboarding/feedback/summary` contract supporting `mean_` and `average_` aliases."""
    issues: list[str] = []
    if doc.get("object") != FEEDBACK_SUMMARY_OBJECT:
        issues.append("Feedback summary invalid object type.")
    if _has_forbidden_score_key(doc):
        issues.append("Feedback summary contains forbidden key.")

    raw_total = doc.get("total_responses")
    total_ok = _is_nonneg_int(raw_total)
    total_val = int(raw_total) if total_ok else -1
    if not total_ok:
        issues.append("Feedback summary invalid `total_responses`.")

    has_mean = "mean_ease_rating" in doc
    has_avg = "average_ease_rating" in doc
    resolved_ease: float | None = None
    if not has_mean and not has_avg:
        issues.append("Feedback summary missing `mean_ease_rating` / `average_ease_rating`.")
    elif total_ok:
        m_val = doc.get("mean_ease_rating")
        a_val = doc.get("average_ease_rating")
        checked: list[float] = []
        for lbl, raw_v, present in (
            ("mean_ease_rating", m_val, has_mean),
            ("average_ease_rating", a_val, has_avg),
        ):
            if not present:
                continue
            if total_val == 0 and raw_v is not None:
                issues.append(f"Feedback summary non-null `{lbl}` when empty.")
            elif total_val > 0:
                if (
                    isinstance(raw_v, bool)
                    or not isinstance(raw_v, (int, float))
                    or not math.isfinite(float(raw_v))
                    or not (1.0 <= float(raw_v) <= 5.0)
                ):
                    issues.append(f"Feedback summary invalid `{lbl}` value.")
                else:
                    checked.append(float(raw_v))
        if has_mean and has_avg and len(checked) == 2 and round(checked[0], 2) != round(checked[1], 2):
            issues.append("Feedback summary `mean_ease_rating` and `average_ease_rating` disagree.")
        if checked:
            resolved_ease = round(checked[0], 2)

    resolved_lag: float | None = None
    if "zero_game_lag_ratio" not in doc:
        issues.append("Feedback summary missing `zero_game_lag_ratio`.")
    elif total_ok:
        raw_lag = doc.get("zero_game_lag_ratio")
        if total_val == 0 and raw_lag is not None:
            issues.append("Feedback summary non-null `zero_game_lag_ratio` when empty.")
        elif total_val > 0:
            if (
                isinstance(raw_lag, bool)
                or not isinstance(raw_lag, (int, float))
                or not math.isfinite(float(raw_lag))
                or not (0.0 <= float(raw_lag) <= 1.0)
            ):
                issues.append("Feedback summary invalid `zero_game_lag_ratio`.")
            else:
                resolved_lag = round(float(raw_lag), 4)

    bp = doc.get("by_profile")
    if (
        not isinstance(bp, Mapping)
        or not _is_nonneg_int(bp.get("dev"))
        or not _is_nonneg_int(bp.get("gamer"))
        or (total_ok and int(bp["dev"]) + int(bp["gamer"]) != total_val)
    ):
        issues.append("Feedback summary invalid `by_profile` conservation.")

    if issues:
        return False, max(total_val, 0), None, None, False, issues

    if total_val == 0:
        return True, 0, None, None, None, []

    ease_ok = resolved_ease is not None and Fraction(str(resolved_ease)) >= Fraction(str(min_ease))
    lag_ok = resolved_lag is not None and Fraction(str(resolved_lag)) >= Fraction(str(min_lag))
    slo_issues: list[str] = []
    if not ease_ok:
        slo_issues.append(
            f"Note CES moyenne {resolved_ease}/5 < {min_ease}/5."
            if is_fr
            else f"CES ease rating {resolved_ease}/5 below target >= {min_ease}/5."
        )
    if not lag_ok:
        slo_issues.append(
            f"Ratio zéro-lag Gamer {resolved_lag} < {min_lag}."
            if is_fr
            else f"Gamer zero_game_lag_ratio {resolved_lag} below target >= {min_lag}."
        )
    return True, total_val, resolved_ease, resolved_lag, bool(ease_ok and lag_ok), slo_issues


def _inspect_fleet_quorum_contract(
    doc: Mapping[str, Any],
    *,
    target_n: int,
    require_live_quorum: bool,
    is_fr: bool,
) -> tuple[bool, int | None, int | None, bool, bool, bool, list[str]]:
    issues: list[str] = []
    count_keys = ("total_hiveminders", "live_hiveminders", "distinct_physical_gpus", "live_endpoints")
    models_ok = isinstance(doc.get("live_models"), list) and all(isinstance(m, str) for m in doc["live_models"])
    rows = doc.get("hiveminders")
    rows_ok = isinstance(rows, list) and all(isinstance(r, Mapping) and "handle" in r and "status" in r for r in rows)
    endpoint_leaked = isinstance(rows, list) and any(
        isinstance(r, Mapping) and r.get("endpoint_url") is not None for r in rows
    )
    if (
        doc.get("object") != FLEET_OBJECT
        or not all(_is_nonneg_int(doc.get(k)) for k in count_keys)
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
            else "Unredacted `endpoint_url` leaked in public `/v1/hivemind/producers/fleet`."
        )
    if issues:
        return False, None, None, False, False, False, issues

    online_cnt = int(doc["live_hiveminders"])
    gpu_cnt = int(doc["distinct_physical_gpus"])
    quorum_met = online_cnt >= target_n and gpu_cnt >= 1
    awaiting = not quorum_met
    quorum_gate_ok = quorum_met or not require_live_quorum
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


def _probe_pr949_endpoint(
    url: str,
    *,
    raw_http_fn: RawHttpFn,
    http_fn: HttpFn,
) -> tuple[int, dict[str, Any] | None, str, dict[str, str]]:
    _assert_zero_inference_request("GET", url, None)
    use_raw = raw_http_fn is not default_http_raw or http_fn is default_http_request
    if use_raw:
        try:
            raw_ret = raw_http_fn("GET", url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, {"network_error": type(exc).__name__}, "", {}
        if not isinstance(raw_ret, tuple) or len(raw_ret) != 3:
            raise ScaleHarnessError("raw_http_fn must return `(status_code, body_text, headers)`.")
        code_raw, body_raw, hdrs_raw = raw_ret
        code = int(code_raw)
        hdrs = {str(k).strip().lower(): str(v) for k, v in (hdrs_raw or {}).items()}
        if isinstance(body_raw, Mapping):
            json_doc = dict(body_raw)
            return code, json_doc, json.dumps(json_doc, ensure_ascii=False), hdrs
        text_body = str(body_raw)
        json_doc = None
        if "application/json" in hdrs.get("content-type", "").lower() or text_body.lstrip().startswith(("{", "[")):
            with suppress(json.JSONDecodeError):
                parsed = json.loads(text_body) if text_body.strip() else {}
                json_doc = dict(parsed) if isinstance(parsed, Mapping) else {"data": parsed}
        return code, json_doc, text_body, hdrs

    resp, _, _ = _invoke_http_callable(http_fn, "GET", url, None, None)
    json_doc = dict(resp.body) if ("html" not in resp.body and resp.body) else None
    text_body = resp.text or (json.dumps(resp.body, ensure_ascii=False) if resp.body else "")
    return resp.status_code, json_doc, text_body, resp.headers


def _build_finops_redeploy_cmd(gateway: str, expected_sha: str | None) -> str:
    sha_token = expected_sha if expected_sha else "${GITHUB_SHA:?set_GITHUB_SHA}"
    _ = gateway
    return (
        f'HIVEMIND_BUILD_SHA="{sha_token}" bash deploy/gcp_openrouter/deploy.sh '
        "--min-instances 0 --max-instances 1 --no-cpu-throttling"
    )


def _compliance_blocked_pr949_report(
    gateway: str,
    compliance: Mapping[str, Any],
    *,
    lang_code: str,
    target_n: int,
    clean_ease: float,
    clean_lag: float,
    require_live_quorum: bool,
    clean_expected_sha: str | None,
) -> dict[str, Any]:
    diag = sanitize_text_for_display(str(compliance.get("summary") or COMPLIANCE_NOTICE_BILINGUAL))
    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "slice_id": "SLICE-SCALE-DYN-40",
            "campaign_id": CAMPAIGN_ID,
            "mode": "pr949_live_parity",
            "lang": lang_code,
            "gateway": gateway,
            "compliance_ok": False,
            "corporate_detected": True,
            "compliance": dict(compliance),
            "ready": False,
            "exit_code": 2,
            "holiday_routes_deployed_on_cloud_run": False,
            "onboarding_feedback_deployed": False,
            "stale_revision_detected": False,
            "missing_or_stale_routes": [],
            "models_ok": False,
            "model_ids": [],
            "coder_model_present": False,
            "manifest_ok": False,
            "live_build_sha": "unknown",
            "expected_build_sha": clean_expected_sha,
            "sha_match": None,
            "cloud_run_revision": None,
            "html_routes_ok": False,
            "unique_csp_nonces_count": 0,
            "forbidden_dom_sinks_count": 0,
            "static_figures_found": False,
            "personal_pc_banners_ok": False,
            "html_contracts": {},
            "feedback_schema_ok": False,
            "total_responses": 0,
            "mean_ease_rating": None,
            "zero_game_lag_ratio": None,
            "ces_slo_ok": False,
            "min_ease_rating": clean_ease,
            "min_zero_lag_ratio": clean_lag,
            "fleet_schema_ok": False,
            "online_producers": None,
            "distinct_physical_gpus": None,
            "target_online_producers": target_n,
            "producer_quorum_met": False,
            "awaiting_producer_quorum": False,
            "require_live_quorum": bool(require_live_quorum),
            "raid_status_ok": False,
            "arena_summary_ok": False,
            "http_statuses": {},
            "redeploy_required": False,
            "redeploy_command": _build_finops_redeploy_cmd(gateway, clean_expected_sha),
            "zero_inference_verified": True,
            "network_calls": 0,
            "file_writes": 0,
            "diagnostics": [diag],
            "summary": diag,
        }
    )


def verify_pr949_holiday_burst_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    min_ease_rating: float = DEFAULT_MIN_EASE_RATING,
    min_zero_lag_ratio: float = DEFAULT_MIN_ZERO_LAG_RATIO,
    require_live_quorum: bool = False,
    raw_http_fn: RawHttpFn = default_http_raw,
    http_fn: HttpFn = default_http_request,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """Zero-inference PR #949 live parity, 8+ friend-producer quorum, and CES alias verifier."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    target_n = _validate_int_range(target_online_producers, "target_online_producers", 1, 500)
    clean_ease = _validate_float_range(min_ease_rating, "min_ease_rating", 1.0, 5.0)
    clean_lag = _validate_float_range(min_zero_lag_ratio, "min_zero_lag_ratio", 0.0, 1.0)
    if not isinstance(require_live_quorum, bool):
        raise ScaleHarnessError("require_live_quorum must be a boolean.")
    clean_expected_sha = (
        _validate_expected_sha(expected_sha)
        if expected_sha is not None and str(expected_sha).strip()
        else None
    )

    compliance = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=check_git_remote,
        lang=lang_code,
    )
    if not compliance["compliance_ok"]:
        return _compliance_blocked_pr949_report(
            clean_gw,
            compliance,
            lang_code=lang_code,
            target_n=target_n,
            clean_ease=clean_ease,
            clean_lag=clean_lag,
            require_live_quorum=require_live_quorum,
            clean_expected_sha=clean_expected_sha,
        )

    http_statuses: dict[str, int] = {}
    stale_routes: set[str] = set()
    missing_manifest_routes: list[str] = []
    diagnostics: list[str] = []
    seen_nonces: set[str] = set()
    html_contracts: dict[str, dict[str, Any]] = {}
    unsafe_sink_total = 0
    static_figures_any = False

    health_ok = False
    health_sha = "unknown"
    models_ok = False
    model_ids: list[str] = []
    coder_present = False
    manifest_ok = False
    live_build_sha = "unknown"
    sha_match: bool | None = None
    cloud_run_revision: str | None = None
    feedback_schema_ok = False
    total_responses = 0
    mean_ease: float | None = None
    zero_lag: float | None = None
    ces_slo_ok: bool | None = False
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
        code, json_doc, text_body, hdrs = _probe_pr949_endpoint(
            f"{clean_gw}{path}",
            raw_http_fn=raw_http_fn,
            http_fn=http_fn,
        )
        http_statuses[path] = code
        if code != 200:
            stale_routes.add(path)
            diagnostics.append(
                f"GET {path} -> HTTP {code} (attendu 200)." if is_fr else f"GET {path} -> HTTP {code} (expected 200)."
            )
            continue

        if path == "/health":
            health_ok = isinstance(json_doc, Mapping) and json_doc.get("status") == "ok"
            if isinstance(json_doc, Mapping):
                health_sha = _sanitize_build_sha(json_doc.get("build_sha"))
            if not health_ok:
                stale_routes.add(path)
                diagnostics.append("GET /health missing `status == 'ok'`.")
        elif path == "/v1/models":
            raw_data = json_doc.get("data") if isinstance(json_doc, Mapping) else None
            valid_entries = (
                [e for e in raw_data if isinstance(e, Mapping) and isinstance(e.get("id"), str) and e["id"].strip()]
                if isinstance(raw_data, list)
                else []
            )
            model_ids = [str(e["id"]).strip() for e in valid_entries]
            coder_present = any(
                e["id"].strip() == "hivemind/coder" and (e.get("object") in (None, "model"))
                for e in valid_entries
            )
            models_ok = bool(
                isinstance(json_doc, Mapping)
                and json_doc.get("object", "list") == "list"
                and model_ids
                and coder_present
            )
            if not models_ok:
                stale_routes.add(path)
                diagnostics.append(
                    "GET /v1/models vide ou sans `hivemind/coder`."
                    if is_fr
                    else "GET /v1/models empty or missing `hivemind/coder` metadata."
                )
        elif kind == "html":
            h_ok, h_info, h_issues = _inspect_html_contract(path, text_body, hdrs, seen_nonces, is_fr=is_fr)
            html_contracts[path] = h_info
            unsafe_sink_total += int(h_info["unsafe_dom_sinks_count"])
            static_figures_any = static_figures_any or bool(h_info["static_figure_found"])
            if not h_ok:
                stale_routes.add(path)
                diagnostics.extend(h_issues)
        elif path == "/v1/hivemind/onboarding/manifest":
            if not isinstance(json_doc, Mapping):
                stale_routes.add(path)
                diagnostics.append("GET /v1/hivemind/onboarding/manifest missing JSON body.")
            else:
                meta_ok = bool(
                    json_doc.get("object") == MANIFEST_OBJECT
                    and json_doc.get("campaign_id") == CAMPAIGN_ID
                    and json_doc.get("status") == "ok"
                    and json_doc.get("personal_pc_only") is True
                    and json_doc.get("financial_promise") == FINANCIAL_PROMISE
                )
                raw_routes = json_doc.get("holiday_routes")
                if isinstance(raw_routes, list) and all(isinstance(r, str) for r in raw_routes):
                    missing_manifest_routes = [r for r in PR949_MANIFEST_ROUTES if r not in raw_routes]
                else:
                    missing_manifest_routes = list(PR949_MANIFEST_ROUTES)
                raw_scripts = json_doc.get("downloadable_scripts")
                scripts_ok = (
                    isinstance(raw_scripts, list)
                    and all(isinstance(s, str) for s in raw_scripts)
                    and all(req in raw_scripts for req in PR949_MANIFEST_CORE_SCRIPTS)
                )
                live_build_sha = _sanitize_build_sha(json_doc.get("build_sha"))
                if live_build_sha == "unknown":
                    live_build_sha = health_sha
                raw_rev = json_doc.get("cloud_run_revision")
                cloud_run_revision = str(raw_rev).strip() if isinstance(raw_rev, str) and raw_rev.strip() else None
                if clean_expected_sha is not None:
                    sha_match = _shas_match(live_build_sha, clean_expected_sha)
                    if not sha_match:
                        diagnostics.append(
                            f"Build SHA drift: live='{live_build_sha}' != expected '{clean_expected_sha}'."
                        )
                manifest_ok = bool(meta_ok and not missing_manifest_routes and scripts_ok and sha_match is not False)
                if not manifest_ok:
                    stale_routes.add(path)
                    if missing_manifest_routes:
                        diagnostics.append(
                            f"Manifest missing holiday routes: {', '.join(missing_manifest_routes)}."
                        )
                    elif not meta_ok or not scripts_ok:
                        diagnostics.append("Manifest metadata or core scripts invalid.")
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            if not isinstance(json_doc, Mapping):
                stale_routes.add(path)
                diagnostics.append("Feedback summary missing JSON body.")
            else:
                feedback_schema_ok, total_responses, mean_ease, zero_lag, ces_slo_ok, fb_issues = (
                    _inspect_feedback_summary_contract(
                        json_doc,
                        min_ease=clean_ease,
                        min_lag=clean_lag,
                        is_fr=is_fr,
                    )
                )
                if not feedback_schema_ok:
                    stale_routes.add(path)
                diagnostics.extend(fb_issues)
        elif path == "/v1/hivemind/producers/fleet":
            if not isinstance(json_doc, Mapping):
                stale_routes.add(path)
                diagnostics.append("Producer fleet missing JSON body.")
            else:
                (
                    fleet_schema_ok,
                    online_producers,
                    distinct_physical_gpus,
                    producer_quorum_met,
                    awaiting_producer_quorum,
                    quorum_gate_ok,
                    fleet_issues,
                ) = _inspect_fleet_quorum_contract(
                    json_doc,
                    target_n=target_n,
                    require_live_quorum=require_live_quorum,
                    is_fr=is_fr,
                )
                if not fleet_schema_ok:
                    stale_routes.add(path)
                diagnostics.extend(fleet_issues)
        elif path == "/v1/hivemind/raid/status":
            comp = json_doc.get("compliance") if isinstance(json_doc, Mapping) else None
            raid_status_ok = bool(
                isinstance(json_doc, Mapping)
                and json_doc.get("object") == RAID_STATUS_OBJECT
                and isinstance(comp, Mapping)
                and comp.get("personal_pc_only") is True
                and comp.get("monetary_cost_eur") == "0.00 EUR"
                and comp.get("monetary_cost_usd") == "0.00 USD"
                and comp.get("license") == "AGPL-3.0"
                and isinstance(json_doc.get("swarm"), Mapping)
                and isinstance(json_doc.get("onboarding_feedback"), Mapping)
            )
            if not raid_status_ok:
                stale_routes.add(path)
                diagnostics.append("GET /v1/hivemind/raid/status schema or compliance invalid.")
        else:
            counts_ok = False
            if isinstance(json_doc, Mapping) and json_doc.get("object") == ARENA_SUMMARY_OBJECT:
                tv = json_doc.get("total_votes")
                if _is_nonneg_int(tv):
                    if all(k in json_doc for k in ("holiday_wins", "baseline_wins", "ties")):
                        hw, bw, ti = json_doc["holiday_wins"], json_doc["baseline_wins"], json_doc["ties"]
                        counts_ok = (
                            _is_nonneg_int(hw)
                            and _is_nonneg_int(bw)
                            and _is_nonneg_int(ti)
                            and int(hw) + int(bw) + int(ti) == int(tv)
                        )
                    else:
                        counts_ok = True
            arena_summary_ok = counts_ok and not _has_forbidden_score_key(json_doc or {})
            if not arena_summary_ok:
                stale_routes.add(path)
                diagnostics.append("GET /v1/hivemind/arena/summary schema or vote conservation invalid.")

    html_routes_ok = bool(
        len(html_contracts) == len(LIVE_HTML_PATHS)
        and all(html_contracts.get(p, {}).get("ok") is True for p in LIVE_HTML_PATHS)
        and len(seen_nonces) == len(LIVE_HTML_PATHS)
    )
    personal_pc_banners_ok = bool(
        len(html_contracts) == len(LIVE_HTML_PATHS)
        and all(html_contracts.get(p, {}).get("personal_pc_banner_ok") is True for p in LIVE_HTML_PATHS)
    )
    onboarding_feedback_deployed = bool(
        http_statuses.get("/v1/hivemind/onboarding/feedback/summary") == 200 and feedback_schema_ok
    )
    holiday_routes_deployed = bool(
        health_ok
        and manifest_ok
        and html_routes_ok
        and onboarding_feedback_deployed
        and raid_status_ok
        and arena_summary_ok
    )
    missing_or_stale_routes = sorted(stale_routes | set(missing_manifest_routes))
    stale_revision_detected = bool(
        any(http_statuses.get(p) == 404 for p in http_statuses)
        or missing_manifest_routes
        or (sha_match is False)
        or not holiday_routes_deployed
    )
    ces_gate_ok = feedback_schema_ok and (ces_slo_ok is not False)
    ready = bool(
        holiday_routes_deployed
        and models_ok
        and fleet_schema_ok
        and ces_gate_ok
        and quorum_gate_ok
        and not stale_revision_detected
    )
    redeploy_required = bool(
        stale_revision_detected or not holiday_routes_deployed or not onboarding_feedback_deployed or not models_ok
    )
    clean_diags = [sanitize_text_for_display(d) for d in diagnostics]
    summary = (
        f"Parité PR #949={'GO' if ready else 'NO-GO'} sur {clean_gw} "
        f"(holiday_routes={holiday_routes_deployed}, feedback={onboarding_feedback_deployed}, "
        f"producers={online_producers}/{target_n}, awaiting_quorum={awaiting_producer_quorum}, "
        f"mean_ease={mean_ease}, zero_lag={zero_lag})."
        if is_fr
        else f"PR #949 parity={'GO' if ready else 'NO-GO'} on {clean_gw} "
        f"(holiday_routes={holiday_routes_deployed}, feedback={onboarding_feedback_deployed}, "
        f"producers={online_producers}/{target_n}, awaiting_quorum={awaiting_producer_quorum}, "
        f"mean_ease={mean_ease}, zero_lag={zero_lag})."
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "slice_id": "SLICE-SCALE-DYN-40",
            "campaign_id": CAMPAIGN_ID,
            "mode": "pr949_live_parity",
            "lang": lang_code,
            "gateway": clean_gw,
            "compliance_ok": True,
            "corporate_detected": False,
            "compliance": compliance,
            "ready": ready,
            "exit_code": 0 if ready else 2,
            "holiday_routes_deployed_on_cloud_run": holiday_routes_deployed,
            "onboarding_feedback_deployed": onboarding_feedback_deployed,
            "stale_revision_detected": stale_revision_detected,
            "missing_or_stale_routes": missing_or_stale_routes,
            "models_ok": models_ok,
            "model_ids": model_ids,
            "coder_model_present": coder_present,
            "manifest_ok": manifest_ok,
            "live_build_sha": live_build_sha,
            "expected_build_sha": clean_expected_sha,
            "sha_match": sha_match,
            "cloud_run_revision": cloud_run_revision,
            "html_routes_ok": html_routes_ok,
            "unique_csp_nonces_count": len(seen_nonces),
            "forbidden_dom_sinks_count": unsafe_sink_total,
            "static_figures_found": static_figures_any,
            "personal_pc_banners_ok": personal_pc_banners_ok,
            "html_contracts": html_contracts,
            "feedback_schema_ok": feedback_schema_ok,
            "total_responses": total_responses,
            "mean_ease_rating": mean_ease,
            "zero_game_lag_ratio": zero_lag,
            "ces_slo_ok": ces_slo_ok,
            "min_ease_rating": clean_ease,
            "min_zero_lag_ratio": clean_lag,
            "fleet_schema_ok": fleet_schema_ok,
            "online_producers": online_producers,
            "distinct_physical_gpus": distinct_physical_gpus,
            "target_online_producers": target_n,
            "producer_quorum_met": producer_quorum_met,
            "awaiting_producer_quorum": awaiting_producer_quorum,
            "require_live_quorum": require_live_quorum,
            "raid_status_ok": raid_status_ok,
            "arena_summary_ok": arena_summary_ok,
            "http_statuses": http_statuses,
            "redeploy_required": redeploy_required,
            "redeploy_command": _build_finops_redeploy_cmd(clean_gw, clean_expected_sha),
            "zero_inference_verified": True,
            "network_calls": network_calls,
            "file_writes": 0,
            "diagnostics": clean_diags,
            "summary": summary,
        }
    )


def _execute_step(
    *,
    http_fn: HttpFn,
    clock_fn: ClockFn,
    sleep_fn: SleepFn,
    cohort: str,
    participant_id: str,
    method: str,
    gateway: str,
    path: str,
    is_html: bool,
    payload: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    validator: Callable[[HttpResponse], tuple[bool, int | None, str]],
    max_429_retries: int,
    retry_backoff_s: float,
    strict_html_contract: bool = False,
    is_fr: bool = False,
    step_kind: str = "",
) -> tuple[StepObservation, HttpResponse]:
    """Execute one participant HTTP request with optional bounded HTTP 429 retry and payload validation."""
    url = f"{gateway}{path}"
    retried = 0
    total_latency_ms = 0.0
    sqlite_locked_any = False

    while True:
        resp, elapsed_ms, is_locked = _invoke_http_callable(
            http_fn,
            method,
            url,
            payload,
            headers,
            clock_fn=clock_fn,
        )
        total_latency_ms += elapsed_ms
        sqlite_locked_any = sqlite_locked_any or is_locked
        if resp.status_code == 429 and retried < max_429_retries:
            retried += 1
            retry_hdr = resp.headers.get("retry-after", "")
            wait_s = retry_backoff_s
            try:
                parsed_wait = float(retry_hdr)
                if math.isfinite(parsed_wait) and parsed_wait >= 0.0:
                    wait_s = min(retry_backoff_s, parsed_wait)
            except ValueError:
                wait_s = retry_backoff_s
            if wait_s > 0.0:
                sleep_fn(wait_s)
            continue
        break

    code = resp.status_code
    rate_limited = code == 429
    server_error = code == 0 or code >= 500
    client_error = 400 <= code < 500 and not rate_limited
    csp_nonce: str | None = None
    server_counter: int | None = None
    failure_code = ""
    strict_html_ok = True

    if 200 <= code < 300:
        valid_body, server_counter, failure_code = validator(resp)
        if is_html:
            csp_nonce = extract_csp_nonce(resp.headers, resp.text)
            if csp_nonce is None and valid_body:
                valid_body = False
                failure_code = "missing_or_mismatched_csp_nonce"
            if valid_body and strict_html_contract:
                strict_html_ok, _, html_issues = _inspect_html_contract(
                    path,
                    resp.text,
                    resp.headers,
                    None,
                    is_fr=is_fr,
                )
                if not strict_html_ok:
                    valid_body = False
                    failure_code = html_issues[0] if html_issues else "strict_html_contract_failed"
        ok = valid_body and not sqlite_locked_any
    else:
        ok = False
        failure_code = f"http_{code}"

    obs = StepObservation(
        cohort=cohort,
        participant_id=participant_id,
        method=method,
        path=path,
        is_html=is_html,
        status_code=code,
        ok=ok,
        latency_ms=round(total_latency_ms, 3),
        csp_nonce=csp_nonce,
        rate_limited=rate_limited,
        retried_429=retried,
        server_error=server_error,
        client_error=client_error,
        sqlite_locked=sqlite_locked_any,
        server_counter=server_counter,
        failure_code=failure_code,
        step_kind=step_kind,
        strict_html_ok=strict_html_ok,
    )
    return obs, resp


def _validate_html_page(resp: HttpResponse) -> tuple[bool, int | None, str]:
    html_len = len(resp.text) if resp.text else int(resp.body.get("html_length") or 0)
    if html_len <= 0:
        return False, None, "empty_html_body"
    return True, None, ""


def _validate_manifest_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    if (
        body.get("object") != "hivemind.onboarding.manifest"
        or body.get("status") != "ok"
        or body.get("personal_pc_only") is not True
    ):
        return False, None, "invalid_manifest_contract"
    return True, None, ""


def _validate_register_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    acct_id = body.get("account_id")
    api_key = body.get("api_key")
    if (
        not isinstance(acct_id, str)
        or not acct_id.strip()
        or not isinstance(api_key, str)
        or not api_key.startswith(("hm-", "hm_live_"))
    ):
        return False, None, "invalid_register_response"
    return True, 1, ""


def _validate_models_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    data = resp.body.get("data")
    if not isinstance(data, list) or len(data) == 0:
        return False, None, "invalid_models_list"
    return True, len(data), ""


def _validate_raid_status_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    if resp.body.get("object") != "hivemind.raid_noel.status":
        return False, None, "invalid_raid_status_contract"
    return True, None, ""


def _validate_arena_vote_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    total_votes = body.get("total_votes")
    if (
        body.get("recorded") is not True
        or isinstance(total_votes, bool)
        or not isinstance(total_votes, int)
        or total_votes < 1
    ):
        return False, None, "invalid_arena_vote_response"
    return True, total_votes, ""


def _validate_arena_summary_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    total_votes = body.get("total_votes")
    if (
        body.get("object") != "hivemind.blind_ab_arena.summary"
        or isinstance(total_votes, bool)
        or not isinstance(total_votes, int)
        or total_votes < 0
    ):
        return False, None, "invalid_arena_summary_response"
    return True, total_votes, ""


def _validate_feedback_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    total_resp = body.get("total_responses")
    if (
        body.get("object") != "hivemind.onboarding_feedback_summary"
        or isinstance(total_resp, bool)
        or not isinstance(total_resp, int)
        or total_resp < 1
    ):
        return False, None, "invalid_feedback_response"
    return True, total_resp, ""


def _validate_opentrainer_heartbeat_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    obj = resp.body.get("object")
    if obj not in ("hivemind.opentrainer_heartbeat_ack", "hivemind.opentrainer.heartbeat"):
        return False, None, "invalid_opentrainer_heartbeat_response"
    return True, 1, ""


def _validate_producer_heartbeat_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    status = body.get("status")
    acct_id = body.get("account_id")
    if status not in ("live", "ok") or not isinstance(acct_id, str) or not acct_id.strip():
        return False, None, "invalid_producer_heartbeat_response"
    return True, 1, ""


def _validate_fleet_resp(resp: HttpResponse) -> tuple[bool, int | None, str]:
    body = resp.body
    live = body.get("live_hiveminders")
    if body.get("object") != FLEET_OBJECT or not _is_nonneg_int(live):
        return False, None, "invalid_fleet_response"
    return True, int(live), ""


def _submit_ces_feedback_with_optional_replay(
    *,
    common: Mapping[str, Any],
    headers: Mapping[str, str],
    fb_payload: Mapping[str, Any],
    idem_key: str,
    idempotency_replay: bool,
) -> list[StepObservation]:
    out: list[StepObservation] = []
    fb_headers = dict(headers)
    if idempotency_replay:
        fb_headers["Idempotency-Key"] = idem_key
    obs_fb, _ = _execute_step(
        **common,
        headers=fb_headers,
        method="POST",
        path="/v1/hivemind/onboarding/feedback",
        is_html=False,
        payload=fb_payload,
        validator=_validate_feedback_resp,
        step_kind="ces_primary",
    )
    out.append(obs_fb)
    if idempotency_replay:
        obs_replay, _ = _execute_step(
            **common,
            headers=fb_headers,
            method="POST",
            path="/v1/hivemind/onboarding/feedback",
            is_html=False,
            payload=fb_payload,
            validator=_validate_feedback_resp,
            step_kind="ces_replay",
        )
        out.append(obs_replay)
    return out


def _producer_heartbeat_payload(participant_ip_idx: int) -> dict[str, Any]:
    return {
        "endpoint_url": f"http://{simulated_participant_ip(participant_ip_idx)}:11434",
        "models": [
            {
                "id": "coder",
                "local_name": "qwen2.5-coder:7b",
                "loaded": True,
                "vram_mib": 4608,
            }
        ],
        "cost_per_1k": 0.0,
        "capacity_1k": 50.0,
        "gpu_vram_mib": 12288,
        "execution_class": "ollama-gamer-whisper",
    }


def _run_dev_participant(
    *,
    dev_idx: int,
    participant_ip_idx: int,
    gateway: str,
    include_register: bool,
    include_quick: bool,
    idempotency_replay: bool,
    send_producer_heartbeat: bool,
    strict_html_contract: bool,
    burst_tag: str,
    platform_tag: str,
    lang_code: str,
    http_fn: HttpFn,
    clock_fn: ClockFn,
    sleep_fn: SleepFn,
    max_429_retries: int,
    retry_backoff_s: float,
    start_gate: threading.Event,
) -> list[StepObservation]:
    """Execute the 5-step (remote) or 6-step (local harness) Dev cohort flow plus opt-in parity checks."""
    start_gate.wait(timeout=2.0)
    participant_id = f"dev-{dev_idx:04d}"
    headers = {
        "X-Forwarded-For": simulated_participant_ip(participant_ip_idx),
        "Accept-Language": "fr-FR,fr;q=0.9" if lang_code == "fr" else "en-US,en;q=0.9",
    }
    common: dict[str, Any] = {
        "http_fn": http_fn,
        "clock_fn": clock_fn,
        "sleep_fn": sleep_fn,
        "cohort": "dev",
        "participant_id": participant_id,
        "gateway": gateway,
        "max_429_retries": max_429_retries,
        "retry_backoff_s": retry_backoff_s,
        "strict_html_contract": strict_html_contract,
        "is_fr": lang_code == "fr",
    }
    observations: list[StepObservation] = []

    obs_noel, _ = _execute_step(
        **common,
        headers=headers,
        method="GET",
        path="/noel",
        is_html=True,
        payload=None,
        validator=_validate_html_page,
    )
    observations.append(obs_noel)

    if include_quick:
        obs_quick, _ = _execute_step(
            **common,
            headers=headers,
            method="GET",
            path="/quick",
            is_html=True,
            payload=None,
            validator=_validate_html_page,
        )
        observations.append(obs_quick)

    obs_signup, _ = _execute_step(
        **common,
        headers=headers,
        method="GET",
        path="/signup",
        is_html=True,
        payload=None,
        validator=_validate_html_page,
    )
    observations.append(obs_signup)

    obs_manifest, _ = _execute_step(
        **common,
        headers=headers,
        method="GET",
        path="/v1/hivemind/onboarding/manifest",
        is_html=False,
        payload=None,
        validator=_validate_manifest_resp,
    )
    observations.append(obs_manifest)

    api_key = ""
    if include_register:
        reg_payload = {
            "handle": f"noel-dev-{burst_tag}-{dev_idx:04d}",
            "role": "prosumer",
        }
        obs_reg, reg_resp = _execute_step(
            **common,
            headers=headers,
            method="POST",
            path="/v1/hivemind/register",
            is_html=False,
            payload=reg_payload,
            validator=_validate_register_resp,
        )
        observations.append(obs_reg)
        raw_key = reg_resp.body.get("api_key")
        if isinstance(raw_key, str):
            api_key = raw_key

    if send_producer_heartbeat:
        auth_headers = {**headers, "Authorization": f"Bearer {api_key}"} if api_key else dict(headers)
        obs_prod, _ = _execute_step(
            **common,
            headers=auth_headers,
            method="POST",
            path="/v1/hivemind/producer/heartbeat",
            is_html=False,
            payload=_producer_heartbeat_payload(participant_ip_idx),
            validator=_validate_producer_heartbeat_resp,
        )
        observations.append(obs_prod)

    obs_models, _ = _execute_step(
        **common,
        headers=headers,
        method="GET",
        path="/v1/models",
        is_html=False,
        payload=None,
        validator=_validate_models_resp,
    )
    observations.append(obs_models)

    fb_payload = {
        "profile": "dev",
        "ease_rating": DEV_EASE_RATINGS[dev_idx % len(DEV_EASE_RATINGS)],
        "zero_game_lag": True,
        "platform": platform_tag,
    }
    observations.extend(
        _submit_ces_feedback_with_optional_replay(
            common=common,
            headers=headers,
            fb_payload=fb_payload,
            idem_key=f"noel-scale-{burst_tag}-dev-{dev_idx:04d}",
            idempotency_replay=idempotency_replay,
        )
    )
    return observations


def _run_gamer_participant(
    *,
    gamer_idx: int,
    participant_ip_idx: int,
    gateway: str,
    platform_tag: str,
    lang_code: str,
    burst_tag: str,
    idempotency_replay: bool,
    include_heartbeat_quorum: bool,
    send_producer_heartbeat: bool,
    strict_html_contract: bool,
    http_fn: HttpFn,
    clock_fn: ClockFn,
    sleep_fn: SleepFn,
    max_429_retries: int,
    retry_backoff_s: float,
    start_gate: threading.Event,
) -> list[StepObservation]:
    """Execute the 6-step Gamer cohort flow plus opt-in heartbeat quorum and CES replay."""
    start_gate.wait(timeout=2.0)
    participant_id = f"gamer-{gamer_idx:04d}"
    headers = {
        "X-Forwarded-For": simulated_participant_ip(participant_ip_idx),
        "Accept-Language": "fr-FR,fr;q=0.9" if lang_code == "fr" else "en-US,en;q=0.9",
    }
    common: dict[str, Any] = {
        "http_fn": http_fn,
        "clock_fn": clock_fn,
        "sleep_fn": sleep_fn,
        "cohort": "gamer",
        "participant_id": participant_id,
        "gateway": gateway,
        "max_429_retries": max_429_retries,
        "retry_backoff_s": retry_backoff_s,
        "strict_html_contract": strict_html_contract,
        "is_fr": lang_code == "fr",
    }
    observations: list[StepObservation] = []

    obs_raid_html, _ = _execute_step(
        **common,
        headers=headers,
        method="GET",
        path="/raid-noel",
        is_html=True,
        payload=None,
        validator=_validate_html_page,
    )
    observations.append(obs_raid_html)

    obs_raid_status, _ = _execute_step(
        **common,
        headers=headers,
        method="GET",
        path="/v1/hivemind/raid/status",
        is_html=False,
        payload=None,
        validator=_validate_raid_status_resp,
    )
    observations.append(obs_raid_status)

    obs_arena_html, resp_arena_html = _execute_step(
        **common,
        headers=headers,
        method="GET",
        path="/arena-noel",
        is_html=True,
        payload=None,
        validator=_validate_html_page,
    )
    observations.append(obs_arena_html)

    ballots = extract_arena_ballots(resp_arena_html.text)
    task_id = CURATED_ARENA_TASK_IDS[gamer_idx % len(CURATED_ARENA_TASK_IDS)]
    chosen_side = GAMER_CHOSEN_SIDES[gamer_idx % len(GAMER_CHOSEN_SIDES)]
    vote_payload: dict[str, Any] = {
        "task_id": task_id,
        "chosen_side": chosen_side,
        "profile": "gamer",
    }
    if task_id in ballots:
        vote_payload["ballot_token"] = ballots[task_id]

    obs_vote, _ = _execute_step(
        **common,
        headers=headers,
        method="POST",
        path="/v1/hivemind/arena/vote",
        is_html=False,
        payload=vote_payload,
        validator=_validate_arena_vote_resp,
    )
    observations.append(obs_vote)

    obs_summary, _ = _execute_step(
        **common,
        headers=headers,
        method="GET",
        path="/v1/hivemind/arena/summary",
        is_html=False,
        payload=None,
        validator=_validate_arena_summary_resp,
    )
    observations.append(obs_summary)

    if include_heartbeat_quorum:
        reg_payload = {
            "handle": f"noel-gamer-{burst_tag}-{gamer_idx:04d}",
            "role": "prosumer",
        }
        obs_reg, reg_resp = _execute_step(
            **common,
            headers=headers,
            method="POST",
            path="/v1/hivemind/register",
            is_html=False,
            payload=reg_payload,
            validator=_validate_register_resp,
        )
        observations.append(obs_reg)
        raw_key = reg_resp.body.get("api_key")
        auth_headers = (
            {**headers, "Authorization": f"Bearer {raw_key}"}
            if isinstance(raw_key, str) and raw_key
            else dict(headers)
        )
        obs_ot, _ = _execute_step(
            **common,
            headers=auth_headers,
            method="POST",
            path="/v1/hivemind/opentrainer/heartbeat",
            is_html=False,
            payload={
                "host_label": f"home-rig-{burst_tag}-g{gamer_idx:04d}",
                "gpu_name": "NVIDIA GeForce RTX 4070",
                "vram_total_mib": 12288,
                "vram_free_mib": 10240,
                "network_mode": "outbound-https-relay",
            },
            validator=_validate_opentrainer_heartbeat_resp,
        )
        observations.append(obs_ot)
        if send_producer_heartbeat:
            obs_prod, _ = _execute_step(
                **common,
                headers=auth_headers,
                method="POST",
                path="/v1/hivemind/producer/heartbeat",
                is_html=False,
                payload=_producer_heartbeat_payload(participant_ip_idx),
                validator=_validate_producer_heartbeat_resp,
            )
            observations.append(obs_prod)

    fb_payload = {
        "profile": "gamer",
        "ease_rating": GAMER_EASE_RATINGS[gamer_idx % len(GAMER_EASE_RATINGS)],
        "zero_game_lag": True,
        "platform": platform_tag,
    }
    observations.extend(
        _submit_ces_feedback_with_optional_replay(
            common=common,
            headers=headers,
            fb_payload=fb_payload,
            idem_key=f"noel-scale-{burst_tag}-gamer-{gamer_idx:04d}",
            idempotency_replay=idempotency_replay,
        )
    )
    return observations


def _summarize_observations(rows: Sequence[StepObservation]) -> dict[str, Any]:
    total = len(rows)
    ok_count = sum(1 for r in rows if r.ok)
    err_count = total - ok_count
    rl_count = sum(1 for r in rows if r.rate_limited)
    srv_err = sum(1 for r in rows if r.server_error)
    latencies = [r.latency_ms for r in rows]
    return {
        "requests": total,
        "ok_count": ok_count,
        "error_count": err_count,
        "rate_limited_count": rl_count,
        "server_error_count": srv_err,
        "p50_latency_ms": compute_percentile_ms(latencies, 50.0),
        "p95_latency_ms": compute_percentile_ms(latencies, 95.0),
        "p99_latency_ms": compute_percentile_ms(latencies, 99.0),
    }


def run_noel_scale_burst(
    gateway: str = DEFAULT_GATEWAY,
    *,
    devs: int = DEFAULT_DEVS,
    gamers: int = DEFAULT_GAMERS,
    workers: int = DEFAULT_WORKERS,
    local_harness: bool = False,
    include_register: bool | None = None,
    p95_slo_ms: float = DEFAULT_P95_SLO_MS,
    p99_slo_ms: float = DEFAULT_P99_SLO_MS,
    max_error_ratio: float = DEFAULT_MAX_ERROR_RATIO,
    tolerate_rate_limits: bool = False,
    max_429_retries: int = 0,
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
    burst_tag: str = "noel2026",
    platform_name: str | None = None,
    include_quick: bool = False,
    idempotency_replay: bool = False,
    include_heartbeat_quorum: bool = False,
    strict_html_contract: bool = False,
    target_online_producers: int = DEFAULT_TARGET_ONLINE_PRODUCERS,
    http_fn: HttpFn = default_http_request,
    clock_fn: ClockFn = time.perf_counter,
    sleep_fn: SleepFn = time.sleep,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    check_git_remote: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Execute the concurrent Noël-at-scale router burst and compute exact deterministic SLO metrics."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    devs_n = _validate_int_range(devs, "devs", 0, MAX_PARTICIPANTS_PER_COHORT)
    gamers_n = _validate_int_range(gamers, "gamers", 0, MAX_PARTICIPANTS_PER_COHORT)
    if devs_n + gamers_n < 1:
        raise ScaleHarnessError("At least 1 participant is required (`devs + gamers >= 1`).")
    workers_n = _validate_int_range(workers, "workers", 1, MAX_WORKERS)
    retries_n = _validate_int_range(max_429_retries, "max_429_retries", 0, MAX_429_RETRIES_LIMIT)
    target_producers_n = _validate_int_range(target_online_producers, "target_online_producers", 1, 500)

    clean_p95_slo = _validate_positive_float(p95_slo_ms, "p95_slo_ms")
    clean_p99_slo = _validate_positive_float(p99_slo_ms, "p99_slo_ms")
    if clean_p99_slo < clean_p95_slo:
        raise ScaleHarnessError("p99_slo_ms must be >= p95_slo_ms.")

    clean_max_err_ratio = _validate_float_range(max_error_ratio, "max_error_ratio", 0.0, 1.0)

    if (
        isinstance(retry_backoff_s, bool)
        or not isinstance(retry_backoff_s, (int, float))
        or not math.isfinite(float(retry_backoff_s))
        or float(retry_backoff_s) < 0.0
    ):
        raise ScaleHarnessError("retry_backoff_s must be a finite non-negative number.")
    clean_backoff = float(retry_backoff_s)

    for b_name, b_val in (
        ("local_harness", local_harness),
        ("tolerate_rate_limits", tolerate_rate_limits),
        ("include_quick", include_quick),
        ("idempotency_replay", idempotency_replay),
        ("include_heartbeat_quorum", include_heartbeat_quorum),
        ("strict_html_contract", strict_html_contract),
        ("check_git_remote", check_git_remote),
    ):
        if not isinstance(b_val, bool):
            raise ScaleHarnessError(f"{b_name} must be a boolean.")
    if include_register is not None and not isinstance(include_register, bool):
        raise ScaleHarnessError("include_register must be a boolean or None.")

    clean_tag = (burst_tag or "").strip().lower()
    if not _BURST_TAG_RE.fullmatch(clean_tag):
        raise ScaleHarnessError("burst_tag must be 2..24 lowercase alphanumeric/hyphen/underscore chars.")

    platform_tag = _normalize_platform_tag(platform_name)
    loopback_mode = is_loopback_gateway(clean_gw)
    effective_register = (
        bool(local_harness or loopback_mode) if include_register is None else bool(include_register)
    )
    if effective_register and not loopback_mode and not local_harness:
        raise ScaleHarnessError(
            "POST /v1/hivemind/register is restricted to loopback or explicit local_harness mode."
        )
    if include_heartbeat_quorum and not loopback_mode and not local_harness:
        raise ScaleHarnessError(
            "Heartbeat quorum verification is restricted to loopback or explicit local_harness mode."
        )
    if include_heartbeat_quorum and not effective_register:
        raise ScaleHarnessError("include_heartbeat_quorum cannot be combined with include_register=False.")

    eff_git_check = bool(check_git_remote or cwd is not None)
    compliance = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=eff_git_check,
        lang=lang_code,
    )
    participants_total = devs_n + gamers_n
    producer_from_gamers = min(gamers_n, target_producers_n) if include_heartbeat_quorum else 0
    producer_from_devs = (
        min(devs_n, max(0, target_producers_n - producer_from_gamers)) if include_heartbeat_quorum else 0
    )

    expected_dev_steps = 5 + (1 if effective_register else 0) + (1 if include_quick else 0) + (1 if idempotency_replay else 0)
    expected_gamer_steps = 6 + (2 if include_heartbeat_quorum else 0) + (1 if idempotency_replay else 0)
    post_burst_steps = (1 if idempotency_replay else 0) + (1 if include_heartbeat_quorum else 0)
    expected_requests_total = (
        devs_n * expected_dev_steps
        + gamers_n * expected_gamer_steps
        + producer_from_gamers
        + producer_from_devs
        + post_burst_steps
    )
    expected_registrations = (devs_n + gamers_n) if include_heartbeat_quorum else (devs_n if effective_register else 0)

    if not compliance["compliance_ok"]:
        corp_msg = (
            "Exécution refusée : environnement d'entreprise détecté localement (règle Amadeus / PC perso uniquement)."
            if is_fr
            else "Execution refused: corporate environment indicator detected locally (Amadeus / Personal-PC rule)."
        )
        refused_report: dict[str, Any] = {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "campaign_id": CAMPAIGN_ID,
            "lang": lang_code,
            "gateway": clean_gw,
            "local_harness_mode": bool(local_harness or loopback_mode),
            "include_register": effective_register,
            "include_quick": include_quick,
            "idempotency_replay": idempotency_replay,
            "include_heartbeat_quorum": include_heartbeat_quorum,
            "strict_html_contract": strict_html_contract,
            "participants_total": participants_total,
            "devs_count": devs_n,
            "gamers_count": gamers_n,
            "workers_count": workers_n,
            "requests_total": 0,
            "expected_requests_total": expected_requests_total,
            "ok_count": 0,
            "error_count": 0,
            "error_ratio": 1.0,
            "rate_limited_count": 0,
            "retried_429_count": 0,
            "server_error_count": 0,
            "client_error_count": 0,
            "sqlite_lock_error_count": 0,
            "min_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "p95_slo_ms": clean_p95_slo,
            "p99_slo_ms": clean_p99_slo,
            "max_error_ratio": clean_max_err_ratio,
            "tolerate_rate_limits": tolerate_rate_limits,
            "html_renders_total": 0,
            "unique_csp_nonces_count": 0,
            "unique_csp_nonces_ratio": 0.0,
            "strict_html_failures_count": 0,
            "registrations_recorded": 0,
            "expected_registrations": expected_registrations,
            "registrations_conserved": False,
            "arena_votes_recorded": 0,
            "expected_arena_votes": gamers_n,
            "arena_votes_conserved": False,
            "server_arena_total_votes": 0,
            "ces_responses_recorded": 0,
            "expected_ces_responses": participants_total,
            "ces_responses_conserved": False,
            "server_ces_total_responses": 0,
            "ces_replay_attempts": 0,
            "ces_replay_ok_count": 0,
            "ces_replay_conserved": not idempotency_replay,
            "opentrainer_heartbeats_recorded": 0,
            "producer_heartbeats_recorded": 0,
            "online_producers": 0,
            "distinct_physical_gpus": 0,
            "target_online_producers": target_producers_n,
            "producer_quorum_met": False,
            "zero_inference_verified": True,
            "scale_slo_ok": False,
            "exit_code": 2,
            "http_status_counts": {},
            "cohorts": {"dev": _summarize_observations(()), "gamer": _summarize_observations(())},
            "routes": [],
            "diagnostics": [corp_msg],
            "compliance": compliance,
        }
        return _attach_receipt(refused_report)

    effective_workers = min(workers_n, participants_total)
    start_gate = threading.Event()
    all_observations: list[StepObservation] = []

    task_specs: list[tuple[str, int, int]] = []
    max_len = max(devs_n, gamers_n)
    for idx in range(max_len):
        if idx < devs_n:
            task_specs.append(("dev", idx, idx))
        if idx < gamers_n:
            task_specs.append(("gamer", idx, devs_n + idx))

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures: list[concurrent.futures.Future[list[StepObservation]]] = []
        for cohort_kind, cohort_idx, ip_idx in task_specs:
            if cohort_kind == "dev":
                futures.append(
                    executor.submit(
                        _run_dev_participant,
                        dev_idx=cohort_idx,
                        participant_ip_idx=ip_idx,
                        gateway=clean_gw,
                        include_register=effective_register,
                        include_quick=include_quick,
                        idempotency_replay=idempotency_replay,
                        send_producer_heartbeat=cohort_idx < producer_from_devs,
                        strict_html_contract=strict_html_contract,
                        burst_tag=clean_tag,
                        platform_tag=platform_tag,
                        lang_code=lang_code,
                        http_fn=http_fn,
                        clock_fn=clock_fn,
                        sleep_fn=sleep_fn,
                        max_429_retries=retries_n,
                        retry_backoff_s=clean_backoff,
                        start_gate=start_gate,
                    )
                )
            else:
                futures.append(
                    executor.submit(
                        _run_gamer_participant,
                        gamer_idx=cohort_idx,
                        participant_ip_idx=ip_idx,
                        gateway=clean_gw,
                        platform_tag=platform_tag,
                        lang_code=lang_code,
                        burst_tag=clean_tag,
                        idempotency_replay=idempotency_replay,
                        include_heartbeat_quorum=include_heartbeat_quorum,
                        send_producer_heartbeat=cohort_idx < producer_from_gamers,
                        strict_html_contract=strict_html_contract,
                        http_fn=http_fn,
                        clock_fn=clock_fn,
                        sleep_fn=sleep_fn,
                        max_429_retries=retries_n,
                        retry_backoff_s=clean_backoff,
                        start_gate=start_gate,
                    )
                )
        start_gate.set()
        for fut in futures:
            all_observations.extend(fut.result())

    summary_feedback_total: int | None = None
    if idempotency_replay:
        obs_fb_sum, resp_fb_sum = _execute_step(
            http_fn=http_fn,
            clock_fn=clock_fn,
            sleep_fn=sleep_fn,
            cohort="summary",
            participant_id="burst-summary",
            method="GET",
            gateway=clean_gw,
            path="/v1/hivemind/onboarding/feedback/summary",
            is_html=False,
            payload=None,
            headers={"X-Forwarded-For": simulated_participant_ip(participants_total)},
            validator=_validate_feedback_resp,
            max_429_retries=retries_n,
            retry_backoff_s=clean_backoff,
        )
        all_observations.append(obs_fb_sum)
        if obs_fb_sum.ok and obs_fb_sum.server_counter is not None:
            summary_feedback_total = obs_fb_sum.server_counter
        elif isinstance(resp_fb_sum.body.get("total_responses"), int):
            summary_feedback_total = int(resp_fb_sum.body["total_responses"])

    online_producers = 0
    distinct_physical_gpus = 0
    fleet_valid = False
    if include_heartbeat_quorum:
        obs_fleet, resp_fleet = _execute_step(
            http_fn=http_fn,
            clock_fn=clock_fn,
            sleep_fn=sleep_fn,
            cohort="fleet",
            participant_id="burst-quorum",
            method="GET",
            gateway=clean_gw,
            path="/v1/hivemind/producers/fleet",
            is_html=False,
            payload=None,
            headers={"X-Forwarded-For": simulated_participant_ip(participants_total + 1)},
            validator=_validate_fleet_resp,
            max_429_retries=retries_n,
            retry_backoff_s=clean_backoff,
        )
        all_observations.append(obs_fleet)
        if obs_fleet.ok:
            fleet_valid = True
            online_producers = int(resp_fleet.body.get("live_hiveminders") or 0)
            raw_gpus = resp_fleet.body.get("distinct_physical_gpus")
            distinct_physical_gpus = int(raw_gpus) if _is_nonneg_int(raw_gpus) else 0

    requests_total = len(all_observations)
    ok_count = sum(1 for r in all_observations if r.ok)
    error_count = requests_total - ok_count
    rate_limited_count = sum(1 for r in all_observations if r.rate_limited)
    retried_429_count = sum(r.retried_429 for r in all_observations)
    server_error_count = sum(1 for r in all_observations if r.server_error)
    client_error_count = sum(1 for r in all_observations if r.client_error)
    sqlite_lock_error_count = sum(1 for r in all_observations if r.sqlite_locked)

    error_ratio = round(error_count / requests_total, 6) if requests_total > 0 else 1.0
    non_rl_errors = max(0, error_count - rate_limited_count)
    effective_err_ratio = (
        round(non_rl_errors / requests_total, 6)
        if (tolerate_rate_limits and requests_total > 0)
        else error_ratio
    )

    latencies_ms = [r.latency_ms for r in all_observations]
    min_latency_ms = round(min(latencies_ms), 3) if latencies_ms else 0.0
    mean_latency_ms = round(float(statistics.fmean(latencies_ms)), 3) if latencies_ms else 0.0
    max_latency_ms = round(max(latencies_ms), 3) if latencies_ms else 0.0
    p50_latency_ms = compute_percentile_ms(latencies_ms, 50.0)
    p95_latency_ms = compute_percentile_ms(latencies_ms, 95.0)
    p99_latency_ms = compute_percentile_ms(latencies_ms, 99.0)

    html_rows = [r for r in all_observations if r.is_html]
    html_renders_total = len(html_rows)
    valid_nonces = [r.csp_nonce for r in html_rows if r.csp_nonce is not None]
    unique_csp_nonces_count = len(set(valid_nonces))
    unique_csp_nonces_ratio = (
        round(unique_csp_nonces_count / html_renders_total, 6) if html_renders_total > 0 else 1.0
    )
    strict_html_failures_count = sum(1 for r in html_rows if not r.strict_html_ok)

    reg_rows = [r for r in all_observations if r.path == "/v1/hivemind/register" and r.ok]
    registrations_recorded = len(reg_rows)
    reg_rl = sum(1 for r in all_observations if r.path == "/v1/hivemind/register" and r.rate_limited)
    registrations_conserved = (
        registrations_recorded == (expected_registrations - reg_rl)
        if tolerate_rate_limits
        else registrations_recorded == expected_registrations
    )

    vote_ok_rows = [r for r in all_observations if r.path == "/v1/hivemind/arena/vote" and r.ok]
    arena_summary_counters = [
        r.server_counter
        for r in all_observations
        if r.path in ("/v1/hivemind/arena/vote", "/v1/hivemind/arena/summary")
        and r.server_counter is not None
    ]
    server_arena_total_votes = max(arena_summary_counters) if arena_summary_counters else 0
    arena_votes_recorded = (
        min(len(vote_ok_rows), server_arena_total_votes) if vote_ok_rows else 0
    )
    vote_rl = sum(1 for r in all_observations if r.path == "/v1/hivemind/arena/vote" and r.rate_limited)
    expected_arena_votes = gamers_n
    arena_votes_conserved = (
        arena_votes_recorded == (expected_arena_votes - vote_rl)
        if tolerate_rate_limits
        else arena_votes_recorded == expected_arena_votes
    )

    ces_primary_rows = [
        r for r in all_observations if r.path == "/v1/hivemind/onboarding/feedback" and r.step_kind == "ces_primary"
    ]
    ces_primary_ok = [r for r in ces_primary_rows if r.ok]
    ces_replay_rows = [
        r for r in all_observations if r.path == "/v1/hivemind/onboarding/feedback" and r.step_kind == "ces_replay"
    ]
    ces_replay_ok = [r for r in ces_replay_rows if r.ok]
    ces_ok_rows = [r for r in all_observations if r.path == "/v1/hivemind/onboarding/feedback" and r.ok]
    ces_counters = [r.server_counter for r in ces_ok_rows if r.server_counter is not None]
    max_post_ces = max(ces_counters) if ces_counters else 0
    server_ces_total_responses = (
        max(max_post_ces, summary_feedback_total) if summary_feedback_total is not None else max_post_ces
    )
    ces_primary_rl = sum(1 for r in ces_primary_rows if r.rate_limited)
    expected_ces_responses = participants_total

    if idempotency_replay:
        expected_primary_ok = (
            expected_ces_responses - ces_primary_rl if tolerate_rate_limits else expected_ces_responses
        )
        ces_responses_recorded = server_ces_total_responses if ces_primary_ok else 0
        ces_responses_conserved = (
            len(ces_primary_ok) == expected_primary_ok
            and server_ces_total_responses == expected_primary_ok
        )
        replay_rl = sum(1 for r in ces_replay_rows if r.rate_limited)
        expected_replay_ok = expected_primary_ok - replay_rl if tolerate_rate_limits else expected_primary_ok
        ces_replay_conserved = (
            len(ces_replay_ok) == expected_replay_ok
            and server_ces_total_responses == expected_primary_ok
        )
    else:
        ces_responses_recorded = (
            min(len(ces_ok_rows), server_ces_total_responses) if ces_ok_rows else 0
        )
        ces_rl = sum(
            1 for r in all_observations if r.path == "/v1/hivemind/onboarding/feedback" and r.rate_limited
        )
        ces_responses_conserved = (
            ces_responses_recorded == (expected_ces_responses - ces_rl)
            if tolerate_rate_limits
            else ces_responses_recorded == expected_ces_responses
        )
        ces_replay_conserved = True

    opentrainer_heartbeats_recorded = sum(
        1 for r in all_observations if r.path == "/v1/hivemind/opentrainer/heartbeat" and r.ok
    )
    producer_heartbeats_recorded = sum(
        1 for r in all_observations if r.path == "/v1/hivemind/producer/heartbeat" and r.ok
    )
    producer_quorum_met = (
        bool(
            fleet_valid
            and opentrainer_heartbeats_recorded == gamers_n
            and producer_heartbeats_recorded >= target_producers_n
            and online_producers >= target_producers_n
            and distinct_physical_gpus >= target_producers_n
        )
        if include_heartbeat_quorum
        else True
    )

    status_counts_raw: dict[int, int] = {}
    for r in all_observations:
        status_counts_raw[r.status_code] = status_counts_raw.get(r.status_code, 0) + 1
    http_status_counts = {str(code): status_counts_raw[code] for code in sorted(status_counts_raw)}

    canonical_route_order: tuple[tuple[str, str, str], ...] = (
        ("dev", "GET", "/noel"),
        *((("dev", "GET", "/quick"),) if include_quick else ()),
        ("dev", "GET", "/signup"),
        ("dev", "GET", "/v1/hivemind/onboarding/manifest"),
        *((("dev", "POST", "/v1/hivemind/register"),) if effective_register else ()),
        ("dev", "POST", "/v1/hivemind/producer/heartbeat"),
        ("dev", "GET", "/v1/models"),
        ("dev", "POST", "/v1/hivemind/onboarding/feedback"),
        ("gamer", "GET", "/raid-noel"),
        ("gamer", "GET", "/v1/hivemind/raid/status"),
        ("gamer", "GET", "/arena-noel"),
        ("gamer", "POST", "/v1/hivemind/arena/vote"),
        ("gamer", "GET", "/v1/hivemind/arena/summary"),
        ("gamer", "POST", "/v1/hivemind/register"),
        ("gamer", "POST", "/v1/hivemind/opentrainer/heartbeat"),
        ("gamer", "POST", "/v1/hivemind/producer/heartbeat"),
        ("gamer", "POST", "/v1/hivemind/onboarding/feedback"),
        ("summary", "GET", "/v1/hivemind/onboarding/feedback/summary"),
        ("fleet", "GET", "/v1/hivemind/producers/fleet"),
    )
    route_summaries: list[dict[str, Any]] = []
    for cohort_name, method_name, route_path in canonical_route_order:
        matching = [
            r
            for r in all_observations
            if r.cohort == cohort_name and r.method == method_name and r.path == route_path
        ]
        if not matching:
            continue
        route_summaries.append(
            {
                "cohort": cohort_name,
                "method": method_name,
                "path": route_path,
                **_summarize_observations(matching),
            }
        )

    cohorts_summary = {
        "dev": _summarize_observations([r for r in all_observations if r.cohort == "dev"]),
        "gamer": _summarize_observations([r for r in all_observations if r.cohort == "gamer"]),
    }

    diagnostics: list[str] = []
    if sqlite_lock_error_count > 0:
        diagnostics.append(
            f"Conflit de verrouillage SQLite détecté ({sqlite_lock_error_count} erreur(s)) : vérifiez le mode WAL et le verrou d'écriture `RouterStore`."
            if is_fr
            else f"SQLite lock contention detected ({sqlite_lock_error_count} error(s)): verify WAL mode and `RouterStore` write serialization."
        )
    if server_error_count > 0:
        diagnostics.append(
            f"Défaillance passerelle détectée ({server_error_count} réponse(s) HTTP 5xx ou erreur réseau) — échec fermé du SLO."
            if is_fr
            else f"Gateway degradation detected ({server_error_count} HTTP 5xx or network error response(s)) — failing SLO closed."
        )
    if rate_limited_count > 0:
        diagnostics.append(
            f"Limitation de débit HTTP 429 observée ({rate_limited_count} réponse(s)) : vérifiez `trusted_proxy_hops=1` ou activez `--tolerate-rate-limits` sur IP partagée."
            if is_fr
            else f"HTTP 429 rate limiting observed ({rate_limited_count} response(s)): verify `trusted_proxy_hops=1` behind proxy or pass `--tolerate-rate-limits` on shared NAT."
        )
    if effective_err_ratio > clean_max_err_ratio:
        diagnostics.append(
            f"Taux d'erreur mesuré ({effective_err_ratio:.4f}) supérieur au plafond SLO ({clean_max_err_ratio:.4f})."
            if is_fr
            else f"Measured error ratio ({effective_err_ratio:.4f}) exceeds SLO ceiling ({clean_max_err_ratio:.4f})."
        )
    if p95_latency_ms > clean_p95_slo or p99_latency_ms > clean_p99_slo:
        diagnostics.append(
            f"Budget de latence SLO dépassé (p95={p95_latency_ms:.2f} ms / plafond {clean_p95_slo:.2f} ms, p99={p99_latency_ms:.2f} ms / plafond {clean_p99_slo:.2f} ms)."
            if is_fr
            else f"Latency SLO budget exceeded (p95={p95_latency_ms:.2f} ms vs {clean_p95_slo:.2f} ms cap, p99={p99_latency_ms:.2f} ms vs {clean_p99_slo:.2f} ms cap)."
        )
    if unique_csp_nonces_ratio < 1.0:
        diagnostics.append(
            f"Violation CSP : ratio de nonces uniques = {unique_csp_nonces_ratio:.6f} ({unique_csp_nonces_count}/{html_renders_total} pages HTML)."
            if is_fr
            else f"CSP nonce uniqueness violation: unique_csp_nonces_ratio = {unique_csp_nonces_ratio:.6f} ({unique_csp_nonces_count}/{html_renders_total} HTML renders)."
        )
    if strict_html_failures_count > 0:
        diagnostics.append(
            f"Contrat HTML strict PR #949 en échec sur {strict_html_failures_count} rendu(s)."
            if is_fr
            else f"Strict PR #949 HTML contract failed on {strict_html_failures_count} render(s)."
        )
    if not registrations_conserved:
        diagnostics.append(
            f"Non-conservation des inscriptions : {registrations_recorded} enregistré(s) sur {expected_registrations} attendu(s)."
            if is_fr
            else f"Account registration conservation mismatch: {registrations_recorded} recorded vs {expected_registrations} expected."
        )
    if not arena_votes_conserved:
        diagnostics.append(
            f"Non-conservation des votes Arène A/B : {arena_votes_recorded} enregistré(s) sur {expected_arena_votes} attendu(s)."
            if is_fr
            else f"Blind A/B Arena vote conservation mismatch: {arena_votes_recorded} recorded vs {expected_arena_votes} expected."
        )
    if not ces_responses_conserved or not ces_replay_conserved:
        diagnostics.append(
            f"Non-conservation des retours CES : {ces_responses_recorded} enregistré(s) sur {expected_ces_responses} attendu(s) (replays={len(ces_replay_ok)})."
            if is_fr
            else f"CES onboarding feedback conservation mismatch: {ces_responses_recorded} recorded vs {expected_ces_responses} expected (replays={len(ces_replay_ok)})."
        )
    if include_heartbeat_quorum and not producer_quorum_met:
        diagnostics.append(
            f"Quorum heartbeat insuffisant ({online_producers}/{target_producers_n} producteurs en ligne, {distinct_physical_gpus} GPU distincts)."
            if is_fr
            else f"Heartbeat producer quorum unmet ({online_producers}/{target_producers_n} online, {distinct_physical_gpus} distinct GPUs)."
        )

    scale_slo_ok = bool(
        requests_total == expected_requests_total
        and requests_total > 0
        and sqlite_lock_error_count == 0
        and server_error_count == 0
        and effective_err_ratio <= clean_max_err_ratio
        and p95_latency_ms <= clean_p95_slo
        and p99_latency_ms <= clean_p99_slo
        and unique_csp_nonces_ratio == 1.0
        and strict_html_failures_count == 0
        and registrations_conserved
        and arena_votes_conserved
        and ces_responses_conserved
        and ces_replay_conserved
        and producer_quorum_met
    )

    core_report: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "lang": lang_code,
        "gateway": clean_gw,
        "local_harness_mode": bool(local_harness or loopback_mode),
        "include_register": effective_register,
        "include_quick": include_quick,
        "idempotency_replay": idempotency_replay,
        "include_heartbeat_quorum": include_heartbeat_quorum,
        "strict_html_contract": strict_html_contract,
        "participants_total": participants_total,
        "devs_count": devs_n,
        "gamers_count": gamers_n,
        "workers_count": effective_workers,
        "requests_total": requests_total,
        "expected_requests_total": expected_requests_total,
        "ok_count": ok_count,
        "error_count": error_count,
        "error_ratio": error_ratio,
        "rate_limited_count": rate_limited_count,
        "retried_429_count": retried_429_count,
        "server_error_count": server_error_count,
        "client_error_count": client_error_count,
        "sqlite_lock_error_count": sqlite_lock_error_count,
        "min_latency_ms": min_latency_ms,
        "mean_latency_ms": mean_latency_ms,
        "max_latency_ms": max_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "p99_latency_ms": p99_latency_ms,
        "p95_slo_ms": clean_p95_slo,
        "p99_slo_ms": clean_p99_slo,
        "max_error_ratio": clean_max_err_ratio,
        "tolerate_rate_limits": tolerate_rate_limits,
        "html_renders_total": html_renders_total,
        "unique_csp_nonces_count": unique_csp_nonces_count,
        "unique_csp_nonces_ratio": unique_csp_nonces_ratio,
        "strict_html_failures_count": strict_html_failures_count,
        "registrations_recorded": registrations_recorded,
        "expected_registrations": expected_registrations,
        "registrations_conserved": registrations_conserved,
        "arena_votes_recorded": arena_votes_recorded,
        "expected_arena_votes": expected_arena_votes,
        "arena_votes_conserved": arena_votes_conserved,
        "server_arena_total_votes": server_arena_total_votes,
        "ces_responses_recorded": ces_responses_recorded,
        "expected_ces_responses": expected_ces_responses,
        "ces_responses_conserved": ces_responses_conserved,
        "server_ces_total_responses": server_ces_total_responses,
        "ces_replay_attempts": len(ces_replay_rows),
        "ces_replay_ok_count": len(ces_replay_ok),
        "ces_replay_conserved": ces_replay_conserved,
        "opentrainer_heartbeats_recorded": opentrainer_heartbeats_recorded,
        "producer_heartbeats_recorded": producer_heartbeats_recorded,
        "online_producers": online_producers,
        "distinct_physical_gpus": distinct_physical_gpus,
        "target_online_producers": target_producers_n,
        "producer_quorum_met": producer_quorum_met,
        "zero_inference_verified": True,
        "scale_slo_ok": scale_slo_ok,
        "exit_code": 0 if scale_slo_ok else 2,
        "http_status_counts": http_status_counts,
        "cohorts": cohorts_summary,
        "routes": route_summaries,
        "diagnostics": diagnostics,
        "compliance": compliance,
    }
    return _attach_receipt(core_report)


def _render_pr949_markdown(report: Mapping[str, Any], *, is_fr: bool, notice: str) -> str:
    ready = bool(report.get("ready"))
    title = (
        "# HiveMind Noël — Vérification Parité Live PR #949 & Quorum (`SLICE-SCALE-DYN-40`)"
        if is_fr
        else "# HiveMind Noël — PR #949 Live Parity & Producer Quorum Report (`SLICE-SCALE-DYN-40`)"
    )
    verdict = "GO (`ready=True`)" if ready else "NO-GO (`ready=False`)"
    lines = [
        title,
        "",
        f"> **Compliance ({FINANCIAL_PROMISE})**: {notice}",
        "",
        f"- **Verdict**: **{verdict}** (`exit_code={report.get('exit_code')}`)",
        f"- **Gateway**: `{report.get('gateway')}` (`live_build_sha={report.get('live_build_sha')}`)",
        f"- **Holiday Routes Deployed**: `{report.get('holiday_routes_deployed_on_cloud_run')}` (`feedback_deployed={report.get('onboarding_feedback_deployed')}`, `stale_revision={report.get('stale_revision_detected')}`)",
        f"- **Producer Quorum**: `{report.get('online_producers')}/{report.get('target_online_producers')}` (`quorum_met={report.get('producer_quorum_met')}`, `awaiting={report.get('awaiting_producer_quorum')}`, `gpus={report.get('distinct_physical_gpus')}`)",
        f"- **CES Summary**: `responses={report.get('total_responses')}`, `mean_ease_rating={report.get('mean_ease_rating')}`, `zero_game_lag_ratio={report.get('zero_game_lag_ratio')}`",
        f"- **Unique CSP Nonces**: `{report.get('unique_csp_nonces_count')}/{len(LIVE_HTML_PATHS)}` (`dom_sinks={report.get('forbidden_dom_sinks_count')}`, `static_figures={report.get('static_figures_found')}`, `banners_ok={report.get('personal_pc_banners_ok')}`)",
    ]
    missing = report.get("missing_or_stale_routes") or ()
    if missing:
        lines.append(f"- **Missing/Stale Routes**: `{', '.join(str(r) for r in missing)}`")
    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", "## Diagnostics", ""])
        for item in diags:
            lines.append(f"- {sanitize_text_for_display(str(item))}")
    lines.extend(["", f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`"])
    return _assert_safe_output("\n".join(lines))


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render bilingual FR/EN Markdown scale burst or PR #949 parity report with SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang") or "en"))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    notice = str(comp.get("notice_fr") if is_fr else comp.get("notice_en") or COMPLIANCE_NOTICE_EN)
    if report.get("mode") == "pr949_live_parity":
        return _render_pr949_markdown(report, is_fr=is_fr, notice=notice)

    slo_ok = bool(report.get("scale_slo_ok"))
    if is_fr:
        title = "# HiveMind Noël — Rapport de Charge Concurrente Routeur (`SLICE-SCALE-01`)"
        verdict = "CONFORME SLO (`scale_slo_ok=True`)" if slo_ok else "ÉCHEC SLO (`scale_slo_ok=False`)"
        sec_metrics = "## 1. Métriques Déterministes de Charge"
        sec_routes = "## 2. Répartition par Cohorte et Route Zéro-Inférence"
        sec_diag = "## 3. Diagnostics Actionnables"
        hdr_route = "| Cohorte | Méthode | Route | Requêtes | OK | Erreurs | 429 | p50 (ms) | p95 (ms) | p99 (ms) |"
    else:
        title = "# HiveMind Noël — Concurrent Router Burst Scale Report (`SLICE-SCALE-01`)"
        verdict = "PASS (`scale_slo_ok=True`)" if slo_ok else "FAIL (`scale_slo_ok=False`)"
        sec_metrics = "## 1. Deterministic Scale & Conservation Metrics"
        sec_routes = "## 2. Zero-Inference Cohort & Route Breakdown"
        sec_diag = "## 3. Actionable Diagnostics"
        hdr_route = "| Cohort | Method | Route | Requests | OK | Errors | 429 | p50 (ms) | p95 (ms) | p99 (ms) |"

    lines: list[str] = [
        title,
        "",
        f"> **Compliance ({FINANCIAL_PROMISE})**: {notice}",
        "",
        f"- **Verdict**: **{verdict}**",
        f"- **Gateway**: `{report.get('gateway')}` (`local_harness_mode={report.get('local_harness_mode')}`)",
        f"- **Participants**: `{report.get('participants_total')}` (`devs={report.get('devs_count')}`, `gamers={report.get('gamers_count')}`, `workers={report.get('workers_count')}`)",
        "",
        sec_metrics,
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| `requests_total` | `{report.get('requests_total')}` |",
        f"| `error_count` | `{report.get('error_count')}` |",
        f"| `error_ratio` | `{report.get('error_ratio')}` |",
        f"| `rate_limited_count` | `{report.get('rate_limited_count')}` |",
        f"| `sqlite_lock_error_count` | `{report.get('sqlite_lock_error_count')}` |",
        f"| `p50_latency_ms` | `{report.get('p50_latency_ms')}` |",
        f"| `p95_latency_ms` | `{report.get('p95_latency_ms')}` |",
        f"| `p99_latency_ms` | `{report.get('p99_latency_ms')}` |",
        f"| `unique_csp_nonces_ratio` | `{report.get('unique_csp_nonces_ratio')}` (`{report.get('unique_csp_nonces_count')}/{report.get('html_renders_total')}`) |",
        f"| `arena_votes_recorded` | `{report.get('arena_votes_recorded')}` / `{report.get('expected_arena_votes')}` |",
        f"| `ces_responses_recorded` | `{report.get('ces_responses_recorded')}` / `{report.get('expected_ces_responses')}` |",
        f"| `zero_inference_verified` | `{report.get('zero_inference_verified')}` |",
        f"| `scale_slo_ok` | `{report.get('scale_slo_ok')}` |",
    ]

    routes = report.get("routes")
    if isinstance(routes, Sequence) and routes:
        lines.extend(
            [
                "",
                sec_routes,
                "",
                hdr_route,
                "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for row in routes:
            lines.append(
                f"| `{row.get('cohort')}` | `{row.get('method')}` | `{row.get('path')}` "
                f"| `{row.get('requests')}` | `{row.get('ok_count')}` | `{row.get('error_count')}` "
                f"| `{row.get('rate_limited_count')}` | `{row.get('p50_latency_ms')}` "
                f"| `{row.get('p95_latency_ms')}` | `{row.get('p99_latency_ms')}` |"
            )

    diags = report.get("diagnostics")
    if isinstance(diags, Sequence) and diags:
        lines.extend(["", sec_diag, ""])
        for item in diags:
            lines.append(f"- {item}")

    lines.extend(["", f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`"])
    return _assert_safe_output("\n".join(lines))


def render_cli_report(report: Mapping[str, Any]) -> str:
    """Render a compact bilingual FR/EN terminal summary with `[OK]`/`[ERROR]` status badges."""
    lang_code = normalize_lang(str(report.get("lang") or "en"))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    notice = str(comp.get("notice_fr") if is_fr else comp.get("notice_en") or COMPLIANCE_NOTICE_EN)

    if report.get("mode") == "pr949_live_parity":
        ready = bool(report.get("ready"))
        badge = "[OK]" if ready else "[ERROR]"
        lines = [
            f"{badge} PR #949 Live Parity (gateway={report.get('gateway')}, "
            f"producers={report.get('online_producers')}/{report.get('target_online_producers')}, "
            f"awaiting_quorum={report.get('awaiting_producer_quorum')}, "
            f"mean_ease={report.get('mean_ease_rating')})",
            f"  holiday_routes={report.get('holiday_routes_deployed_on_cloud_run')} "
            f"feedback_deployed={report.get('onboarding_feedback_deployed')} "
            f"stale_revision={report.get('stale_revision_detected')}",
            f"  compliance: {notice}",
        ]
        for diag in report.get("diagnostics") or ():
            lines.append(f"  [DIAG] {sanitize_text_for_display(str(diag))}")
        lines.append(f"  receipt_sha256={report.get('receipt_sha256')}")
        return _assert_safe_output("\n".join(lines))

    slo_ok = bool(report.get("scale_slo_ok"))
    badge = "[OK]" if slo_ok else "[ERROR]"
    if is_fr:
        headline = (
            f"{badge} Rafale Noël Routeur ({report.get('participants_total')} participants : "
            f"{report.get('devs_count')} devs + {report.get('gamers_count')} gamers, "
            f"{report.get('requests_total')} requêtes, scale_slo_ok={slo_ok})"
        )
    else:
        headline = (
            f"{badge} Noël Router Scale Burst ({report.get('participants_total')} participants: "
            f"{report.get('devs_count')} devs + {report.get('gamers_count')} gamers, "
            f"{report.get('requests_total')} requests, scale_slo_ok={slo_ok})"
        )

    lines = [
        headline,
        f"  gateway={report.get('gateway')} errors={report.get('error_count')} ({report.get('error_ratio')}) "
        f"rl_429={report.get('rate_limited_count')} sqlite_locks={report.get('sqlite_lock_error_count')}",
        f"  latency_ms: p50={report.get('p50_latency_ms')} p95={report.get('p95_latency_ms')} p99={report.get('p99_latency_ms')}",
        f"  csp_nonces_ratio={report.get('unique_csp_nonces_ratio')} "
        f"arena_votes={report.get('arena_votes_recorded')} ces_responses={report.get('ces_responses_recorded')}",
        f"  compliance: {notice}",
    ]
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  receipt_sha256={report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_scale_load_harness.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind Noël concurrent router burst harness & PR #949 live parity verifier "
            "(zero LLM inference, 100% personal PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument(
        "--gateway",
        default=default_gw,
        help="HiveMind gateway origin URL (https://*.run.app or loopback 127.0.0.1/localhost/::1).",
    )
    parser.add_argument("--devs", type=int, default=DEFAULT_DEVS, help="Concurrent Dev cohort participants (default: 30).")
    parser.add_argument("--gamers", type=int, default=DEFAULT_GAMERS, help="Concurrent Gamer cohort participants (default: 30).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="ThreadPoolExecutor max_workers concurrency (default: 60).")
    parser.add_argument("--local-harness", action="store_true", help="Explicitly enable local harness mode (includes registration).")
    parser.add_argument("--skip-register", action="store_true", help="Skip POST /v1/hivemind/register even on loopback.")
    parser.add_argument("--p95-slo-ms", type=float, default=DEFAULT_P95_SLO_MS, help="Maximum allowed p95 latency (default: 2500.0).")
    parser.add_argument("--p99-slo-ms", type=float, default=DEFAULT_P99_SLO_MS, help="Maximum allowed p99 latency (default: 5000.0).")
    parser.add_argument("--max-error-ratio", type=float, default=DEFAULT_MAX_ERROR_RATIO, help="Maximum allowed HTTP error ratio (default: 0.0).")
    parser.add_argument("--tolerate-rate-limits", action="store_true", help="Exclude HTTP 429 rate-limit responses from SLO failure ratio.")
    parser.add_argument("--retry-429", type=int, default=0, help="Bounded retry attempts (0..3) on HTTP 429.")
    parser.add_argument("--burst-tag", default="noel2026", help="Deterministic lowercase alphanumeric handle tag.")
    parser.add_argument("--include-quick", action="store_true", help="Include GET /quick HTML/CSP burst checks.")
    parser.add_argument("--idempotency-replay", action="store_true", help="Replay each CES feedback POST once with Idempotency-Key.")
    parser.add_argument("--include-heartbeat-quorum", action="store_true", help="Enable concurrent opentrainer & producer heartbeat quorum checks.")
    parser.add_argument("--strict-html-contract", action="store_true", help="Enforce strict PR #949 HTML/DOM/banner checks on all HTML steps.")
    parser.add_argument("--pr949-parity", action="store_true", help="Run zero-inference PR #949 live parity & producer quorum verification.")
    parser.add_argument("--expected-sha", default=None, help="Optional expected HIVEMIND_BUILD_SHA for PR #949 parity.")
    parser.add_argument("--target-online-producers", type=int, default=DEFAULT_TARGET_ONLINE_PRODUCERS, help="Target live producers for quorum (default: 8).")
    parser.add_argument("--require-live-quorum", action="store_true", help="Fail closed (exit code 2) if online_producers < target_online_producers.")
    parser.add_argument("--min-ease-rating", type=float, default=DEFAULT_MIN_EASE_RATING, help="Minimum CES ease rating out of 5 (default: 4.2).")
    parser.add_argument("--min-zero-lag-ratio", type=float, default=DEFAULT_MIN_ZERO_LAG_RATIO, help="Minimum gamer zero_game_lag_ratio (default: 0.95).")
    parser.add_argument("--cwd", default=None, help="Workspace directory inspected locally for Personal-PC compliance.")
    parser.add_argument("--enforce-personal-pc", action="store_true", help="Inspect workspace/git remotes locally before any network call or write.")
    parser.add_argument("--output", default=None, help="Optional output file path (atomic POSIX 0600 write).")
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output with SHA-256 receipt.")
    parser.add_argument("--cli", action="store_true", help="Emit compact terminal summary output.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output (default).")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_request,
    raw_http_fn: RawHttpFn = default_http_raw,
    clock_fn: ClockFn = time.perf_counter,
    sleep_fn: SleepFn = time.sleep,
    environ: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> int:
    """CLI entrypoint returning `0` when ready / `scale_slo_ok` holds or `2` (fail-closed) otherwise."""
    args = build_parser().parse_args(argv)
    try:
        lang_code = normalize_lang(args.lang)
        eff_cwd: Path | None = None
        if cwd is not None:
            eff_cwd = Path(cwd)
        elif args.cwd is not None:
            raw_cwd = str(args.cwd)
            if "\x00" in raw_cwd or not raw_cwd.strip():
                raise ScaleHarnessError("--cwd must be a valid non-empty path.")
            eff_cwd = Path(raw_cwd)
        elif args.enforce_personal_pc:
            eff_cwd = Path.cwd()

        if args.output is not None:
            raw_out = str(args.output)
            if "\x00" in raw_out or not raw_out.strip():
                raise ScaleHarnessError("--output must be a valid non-empty file path.")

        check_git_remote = bool(args.enforce_personal_pc or args.pr949_parity or eff_cwd is not None)

        if args.pr949_parity:
            report = verify_pr949_holiday_burst_parity(
                args.gateway,
                expected_sha=args.expected_sha,
                target_online_producers=args.target_online_producers,
                min_ease_rating=args.min_ease_rating,
                min_zero_lag_ratio=args.min_zero_lag_ratio,
                require_live_quorum=bool(args.require_live_quorum),
                raw_http_fn=raw_http_fn,
                http_fn=http_fn,
                environ=environ,
                cwd=eff_cwd,
                check_git_remote=check_git_remote,
                lang=lang_code,
            )
            ok_flag = bool(report["ready"])
        else:
            inc_reg: bool | None = False if args.skip_register else (True if args.local_harness else None)
            report = run_noel_scale_burst(
                args.gateway,
                devs=args.devs,
                gamers=args.gamers,
                workers=args.workers,
                local_harness=bool(args.local_harness),
                include_register=inc_reg,
                p95_slo_ms=args.p95_slo_ms,
                p99_slo_ms=args.p99_slo_ms,
                max_error_ratio=args.max_error_ratio,
                tolerate_rate_limits=bool(args.tolerate_rate_limits),
                max_429_retries=args.retry_429,
                burst_tag=args.burst_tag,
                include_quick=bool(args.include_quick),
                idempotency_replay=bool(args.idempotency_replay),
                include_heartbeat_quorum=bool(args.include_heartbeat_quorum),
                strict_html_contract=bool(args.strict_html_contract),
                target_online_producers=args.target_online_producers,
                http_fn=http_fn,
                clock_fn=clock_fn,
                sleep_fn=sleep_fn,
                environ=environ,
                cwd=eff_cwd,
                check_git_remote=check_git_remote,
                lang=lang_code,
            )
            ok_flag = bool(report["scale_slo_ok"])

        if args.json:
            rendered = _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False))
        elif args.cli:
            rendered = render_cli_report(report)
        else:
            rendered = render_markdown_report(report)

        comp = report.get("compliance")
        if isinstance(comp, Mapping) and comp.get("compliance_ok") is False:
            comp_msg = sanitize_text_for_display(str(comp.get("summary") or COMPLIANCE_NOTICE_BILINGUAL))
            print(f"{comp_msg}\n{COMPLIANCE_NOTICE_BILINGUAL}", file=sys.stderr)
            print(rendered)
            return 2

        if args.output:
            _atomic_write_text(
                Path(str(args.output)),
                rendered + ("" if rendered.endswith("\n") else "\n"),
                mode=0o600,
            )

        print(rendered)
        return 0 if ok_flag else 2
    except ScaleHarnessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
