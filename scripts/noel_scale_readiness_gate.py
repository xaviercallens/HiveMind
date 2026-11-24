#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unified 5-gate Noël-at-Scale Go/No-Go readiness runner and scorecard (`SLICE-SCALE-05`).

Aggregates all 5 objective Noël-at-Scale launch readiness gates using ONLY the Python 3.10-3.13
standard library and allowlisted HiveMind gateway hosts (`*.run.app`, `*.googleapis.com`,
`datasets-server.huggingface.co`, or loopback `127.0.0.1` / `localhost` / `::1`) without ever
invoking `POST /v1/chat/completions` or touching live GPU inference:

  - (G1) Cloud Run route & script parity:
         verifies all 8 holiday/core routes, `/v1/hivemind/onboarding/manifest`, all 17
         downloadable `/scripts/<name>` artifacts (AST & SHA-256 parity on disk), and all 5
         `deploy/gcp_openrouter/deploy.sh` FinOps invariants (`--min-instances 0`,
         `--max-instances 1`, `--no-cpu-throttling`, Litestream GCS, Secret Manager).
  - (G2) 60-user router burst SLO:
         executes a concurrent 60-participant burst (`30` devs + `30` gamers across `60` worker
         threads = `360` zero-inference requests) verifying `0` errors, `100%` unique per-request
         CSP nonces across HTML renders, latency within SLO (`p95_ms <= 2500.0`, `p99_ms <= 5000.0`),
         and exact conservation of registrations, blind A/B arena votes, and CES responses.
  - (G3) 50-workspace multi-OS cohort & 100% Amadeus/corporate-PC block rate:
         rehearses `50` heterogeneous personal home workspaces across 5 OS/hardware archetypes
         plus `10` hostile Corporate / Amadeus negative-control workspaces, verifying `100%`
         personal readiness, `100%` fail-closed corporate blocking (`0` writes, `0` network calls),
         and `0` raw `hm-...` key or PII leaks.
  - (G4) Gamer-Whisper `< 1.5 s` (`yield_ms < 1500.0`) GPU yield & `<= 70.0 C` quiet-fan self-test:
         verifies sub-1.5s Steam/DX12 worker stop (`0 MiB` retained VRAM), `<= 70.0 C` quiet-fan
         governor, `>= 2048 MiB` free VRAM reserve, `300 s` post-game cooldown, `< 50 MB/h` LoRA
         delta upload cap, honest no-GPU reporting, and fail-closed thermal/yield negative controls.
  - (G5) Dev `< 60 s` VS Code autoconfig & zero-PII CES funnel readiness:
         verifies 1-command VS Code + Continue autoconfig, POSIX `0600` secret hygiene, chained
         post-setup probe, clean `--uninstall` round-trip, and zero-PII `onboarding_funnel_report`
         health (`mean_ease_rating >= 4.2 / 5`, `zero_game_lag_ratio == 1.0`, PII scrubbing).

Supports deterministic offline rehearsal mode (`< 10.0 s` wall-clock budget, zero external network
egress) and optional `--live-url <origin>` read-only zero-inference gateway drift verification,
emitting FR/EN/bilingual Markdown, compact CLI, or `--json` with cryptographic `receipt_sha256`
(`gates_passed_count`, `gates_total_count`, `go_for_launch`).

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import importlib.util
import inspect
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_scale_readiness_gate/1"
RECEIPT_OBJECT = "hivemind.noel_scale_readiness_receipt"
SLICE_ID = "SLICE-SCALE-05"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_OFFLINE_BUILD_SHA = "noel2026scale05go"
DEFAULT_OFFLINE_REVISION = "hivemind-openrouter-00099-noel"

DEFAULT_SLO_BUDGET_S = 10.0
AUTOCONFIG_SLO_S = 60.0
CES_EASE_MIN = Fraction(42, 10)
ZERO_LAG_TARGET = Fraction(1, 1)

DEFAULT_BURST_DEVS = 30
DEFAULT_BURST_GAMERS = 30
DEFAULT_BURST_WORKERS = 60
MAX_BURST_COHORT = 500
MAX_BURST_WORKERS = 256
DEFAULT_P95_SLO_MS = 2500.0
DEFAULT_P99_SLO_MS = 5000.0

DEFAULT_COHORT_SIZE = 50
MIN_COHORT_SIZE = 5
MAX_COHORT_SIZE = 500
DEFAULT_CORPORATE_CONTROLS = 10
MIN_CORPORATE_CONTROLS = 1
MAX_CORPORATE_CONTROLS = 200
DEFAULT_COHORT_WORKERS = 16
MAX_COHORT_WORKERS = 128

DEFAULT_YIELD_SLO_MS = 1500.0
DEFAULT_MAX_TEMP_C = 70.0
DEFAULT_MIN_FREE_VRAM_MIB = 2048

HTTP_TIMEOUT_S = 5.0
SUBPROCESS_TIMEOUT_S = 3.0

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOY_SCRIPT = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

GATE_IDS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5")
SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr", "bilingual")

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})
HOBBY_DISPLAY_GIT_URL = "https://github.com/Vxtzq/CrowdGPT"

LIVE_HTML_PATHS: tuple[str, ...] = ("/noel", "/quick", "/raid-noel", "/arena-noel")
LIVE_EXTRA_SUMMARY_PATHS: tuple[tuple[str, str], ...] = (
    ("/v1/hivemind/raid/status", "hivemind.raid_noel.status"),
    ("/v1/hivemind/arena/summary", "hivemind.blind_ab_arena.summary"),
)
CURATED_ARENA_TASK_IDS: tuple[str, ...] = (
    "python-sliding-median",
    "rust-frame-decoder",
    "lean4-energy-descent",
    "exoplanet-transit-bls",
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

_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_GOOGLEAPIS_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+googleapis\.com$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_DISCORD_PING_RE = re.compile(r"(?:@everyone|@here|<@[!&]?\d+>)", re.IGNORECASE)
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:')
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\.write)\b"
)
_STATIC_MS_UNIT_RE = re.compile(r"(\d[\d.,]*)\s?m" + r"s\b")

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
    ("internal", re.compile(r"(?:^|[^a-z])internal(?:$|[^a-z])", re.IGNORECASE)),
    ("zscaler", re.compile(r"zscaler", re.IGNORECASE)),
    ("netskope", re.compile(r"netskope", re.IGNORECASE)),
    ("bluecoat", re.compile(r"bluecoat", re.IGNORECASE)),
    ("fortinet", re.compile(r"fortinet", re.IGNORECASE)),
    ("paloalto", re.compile(r"paloalto", re.IGNORECASE)),
)

LiveHttpFn = Callable[..., Any]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
ClockFn = Callable[[], float]
GateCallable = Callable[[], dict[str, Any]]


class ReadinessGateError(ValueError):
    """Raised when readiness gate arguments, URL allowlists, or output safety invariants fail."""


@dataclass(frozen=True)
class ScaleReadinessModules:
    """Loaded sibling Python stdlib modules powering the 5 Noël-at-Scale Go/No-Go gates."""

    parity_gate: Any
    load_harness: Any
    cohort_rehearsal: Any
    gamer_selftest: Any
    vscode_autoconfig: Any
    post_setup_probe: Any
    funnel_report: Any
    raid_bulletin: Any


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to `'en'`, `'fr'`, or `'bilingual'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ReadinessGateError("Language must be 'en', 'fr', or 'bilingual'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    if cleaned in ("bilingual", "both", "fr-en", "en-fr", "all"):
        return "bilingual"
    raise ReadinessGateError(f"Unsupported language {raw!r}; expected 'en', 'fr', or 'bilingual'.")


def _single_script_lang(lang_code: str) -> str:
    return "fr" if lang_code == "fr" else "en"


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` matches the strict HiveMind lateral-movement allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if h == "storage.googleapis.com" or _GOOGLEAPIS_HOST_RE.fullmatch(h):
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate and normalize gateway origin URL to `https://*.run.app`, `https://*.googleapis.com`, or loopback."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ReadinessGateError("Gateway URL must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ReadinessGateError("Gateway URL must not contain control characters.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() for ch in cleaned):
        raise ReadinessGateError("Gateway URL must not contain whitespace.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ReadinessGateError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ReadinessGateError("Gateway URL must be a bare origin without path, query, or fragment.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ReadinessGateError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ReadinessGateError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise ReadinessGateError(
            f"Untrusted gateway host {host!r}; allowed hosts are *.run.app, *.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ReadinessGateError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ReadinessGateError("Remote gateway URL must use https.")
    return cleaned


def validate_report_date(raw: str | None) -> str:
    """Validate an ISO `YYYY-MM-DD` date string or return today's UTC date."""
    if raw is None:
        return time.strftime("%Y-%m-%d", time.gmtime())
    if isinstance(raw, bool) or not isinstance(raw, str) or not raw.strip():
        raise ReadinessGateError("Report date must be a 'YYYY-MM-DD' string.")
    cleaned = raw.strip()
    if not _ISO_DATE_RE.fullmatch(cleaned):
        raise ReadinessGateError(f"Invalid date {raw!r}; expected 'YYYY-MM-DD'.")
    try:
        date.fromisoformat(cleaned)
    except ValueError as exc:
        raise ReadinessGateError(f"Invalid calendar date {cleaned!r}.") from exc
    return cleaned


def _validate_int_range(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise ReadinessGateError(f"{name} must be an integer between {lo} and {hi}.")
    return int(value)


def _validate_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReadinessGateError(f"{name} must be a positive finite number.")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ReadinessGateError(f"{name} must be a positive finite number.")
    return fval


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


def sanitize_text_for_display(text: str, *, home: Path | None = None) -> str:
    """Scrub emails, raw `hm-...` keys, Bearer tokens, home paths, non-loopback IPs, and unit figures."""
    cleaned = redact_home_path(str(text), home=home)
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


def scan_output_safety(text: str) -> list[str]:
    """Scan JSON, Markdown, or CLI output for PII, raw secrets, home paths, non-loopback IPs, or `'score':`."""
    if not isinstance(text, str):
        raise ReadinessGateError("scan_output_safety requires a string.")
    issues: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    if _API_KEY_RE.search(text):
        issues.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        issues.append("bearer_token_detected")
    if _EMAIL_RE.search(text):
        issues.append("email_detected")
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
        raise ReadinessGateError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in core.items() if k != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    payload["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def verify_receipt_sha256(report: Mapping[str, Any]) -> bool:
    """Verify the cryptographic `receipt_sha256` digest of a gate or sub-pipeline report."""
    digest = report.get("receipt_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    body = {k: v for k, v in report.items() if k != "receipt_sha256"}
    std_json = json.dumps(body, sort_keys=True, ensure_ascii=False)
    if hashlib.sha256(std_json.encode("utf-8")).hexdigest() == digest:
        return True
    compact_json = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(compact_json.encode("utf-8")).hexdigest() == digest


def evaluate_personal_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    check_git_remote: bool = False,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Amadeus / Personal-PC compliance check executed BEFORE gates or network I/O."""
    norm_lang = normalize_lang(lang)
    is_fr = norm_lang == "fr"
    env_map = environ if environ is not None else os.environ
    upper_env = {str(k).upper(): str(v).strip() for k, v in env_map.items() if str(v).strip()}

    indicators: list[str] = []
    matched_markers: set[str] = set()

    if any(k.startswith("AMADEUS") for k in upper_env):
        indicators.append("env:AMADEUS_ENV")
        matched_markers.add("amadeus")
    if upper_env.get("USERDNSDOMAIN"):
        indicators.append("env:USERDNSDOMAIN")
        for marker_name, pat in _CORP_PATTERNS:
            if pat.search(upper_env["USERDNSDOMAIN"]):
                matched_markers.add(marker_name)
        if not matched_markers:
            matched_markers.add("domain")
    if upper_env.get("ZSCALER_CERT_PATH"):
        indicators.append("env:ZSCALER_CERT_PATH")
        matched_markers.add("zscaler")

    for env_key in ("USERDOMAIN", "COMPUTERNAME", "HOSTNAME", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        val = upper_env.get(env_key, "")
        if not val:
            continue
        if env_key == "USERDOMAIN" and val.upper() == upper_env.get("COMPUTERNAME", "").upper():
            continue
        for marker_name, pat in _CORP_PATTERNS:
            if pat.search(val):
                indicators.append(f"env:{env_key}:{marker_name}")
                matched_markers.add(marker_name)
                break

    if cwd is not None:
        cwd_text = str(cwd)
        for marker_name, pat in _CORP_PATTERNS:
            if pat.search(cwd_text):
                indicators.append(f"cwd:{marker_name}")
                matched_markers.add(marker_name)
                break

        if check_git_remote:
            git_cfg = cwd / ".git" / "config"
            if git_cfg.is_file():
                try:
                    raw_git = git_cfg.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    raw_git = ""
                for line in raw_git.splitlines():
                    stripped = line.strip()
                    if stripped.lower().startswith("url") and "=" in stripped:
                        remote_val = stripped.split("=", 1)[1].strip()
                        for marker_name, pat in _CORP_PATTERNS:
                            if pat.search(remote_val):
                                indicators.append(f"git_remote:{marker_name}")
                                matched_markers.add(marker_name)
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
            f"(règle Amadeus / PC perso uniquement). {COMPLIANCE_NOTICE_FR}"
            if is_fr
            else f"COMPLIANCE NO-GO ({joined}): execution refused on corporate environment "
            f"(Amadeus / Personal-PC rule). {COMPLIANCE_NOTICE_EN}"
        )

    return {
        "compliance_ok": compliance_ok,
        "corporate_detected": not compliance_ok,
        "corporate_indicators": indicators,
        "matched_markers": sorted(matched_markers),
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "network_transmitted": False,
        "financial_promise": FINANCIAL_PROMISE,
        "monetary_cost": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "summary": summary,
    }


def _load_or_reuse_script_module(module_name: str, script_path: Path) -> Any:
    if not script_path.is_file():
        raise ReadinessGateError(f"Required Noël readiness script missing: {script_path.name}")
    resolved_str = str(script_path.resolve())
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and str(Path(existing_file).resolve()) == resolved_str:
            return existing
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ReadinessGateError(f"Unable to construct module spec for {script_path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_scale_readiness_modules(scripts_dir: Path | None = None) -> ScaleReadinessModules:
    """Load all sibling Python stdlib Noël onboarding and scale modules."""
    base = scripts_dir if scripts_dir is not None else DEFAULT_SCRIPTS_DIR
    return ScaleReadinessModules(
        parity_gate=_load_or_reuse_script_module(
            "noel_cloud_run_parity_gate",
            base / "noel_cloud_run_parity_gate.py",
        ),
        load_harness=_load_or_reuse_script_module(
            "noel_scale_load_harness",
            base / "noel_scale_load_harness.py",
        ),
        cohort_rehearsal=_load_or_reuse_script_module(
            "noel_canary_cohort_rehearsal",
            base / "noel_canary_cohort_rehearsal.py",
        ),
        gamer_selftest=_load_or_reuse_script_module(
            "noel_gamer_whisper_selftest",
            base / "noel_gamer_whisper_selftest.py",
        ),
        vscode_autoconfig=_load_or_reuse_script_module(
            "noel_vscode_autoconfig",
            base / "noel_vscode_autoconfig.py",
        ),
        post_setup_probe=_load_or_reuse_script_module(
            "noel_post_setup_probe",
            base / "noel_post_setup_probe.py",
        ),
        funnel_report=_load_or_reuse_script_module(
            "onboarding_funnel_report",
            base / "onboarding_funnel_report.py",
        ),
        raid_bulletin=_load_or_reuse_script_module(
            "daily_raid_bulletin",
            base / "daily_raid_bulletin.py",
        ),
    )


class LocalScaleGatewaySimulator:
    """Thread-safe zero-inference HiveMind router simulator for deterministic `< 10 s` rehearsal."""

    def __init__(
        self,
        gateway_url: str = DEFAULT_GATEWAY,
        *,
        scripts_dir: Path = DEFAULT_SCRIPTS_DIR,
        allowed_scripts: Sequence[str] = (),
        holiday_routes: Sequence[str] = (),
        build_sha: str = DEFAULT_OFFLINE_BUILD_SHA,
        revision: str = DEFAULT_OFFLINE_REVISION,
        static_csp_nonce: str | None = None,
        fail_route: str | None = None,
        fail_status: int = 503,
    ) -> None:
        self.gateway_url = validate_gateway_url(gateway_url)
        self.scripts_dir = scripts_dir
        self.allowed_scripts = tuple(sorted(set(allowed_scripts)))
        self.holiday_routes = tuple(sorted(set(holiday_routes)))
        self.build_sha = build_sha
        self.revision = revision
        self.static_csp_nonce = static_csp_nonce
        self.fail_route = fail_route
        self.fail_status = fail_status

        self._ballot_secret = b"noel-scale-05-deterministic-ballot-key!"
        self._lock = threading.Lock()
        self._acct_seq = 0
        self._nonce_seq = 0
        self.requests_total = 0
        self.inference_calls_count = 0
        self.arena_votes_count = 0
        self.accounts_by_key: dict[str, dict[str, str]] = {}
        self.activated_account_ids: set[str] = set()
        self.feedback_items: list[dict[str, Any]] = []

    def _sign_ballot(self, task_id: str, swap_bit: int) -> str:
        prefix = hashlib.sha256(f"scale05:{task_id}".encode()).hexdigest()[:12]
        mac = hmac.new(
            self._ballot_secret,
            f"{task_id}:{prefix}:{swap_bit}".encode(),
            hashlib.sha256,
        ).hexdigest()[:24]
        return f"{prefix}.{swap_bit}.{mac}"

    def _verify_ballot(self, task_id: str, token: str) -> bool:
        parts = token.split(".")
        if len(parts) != 3 or parts[1] not in ("0", "1"):
            return False
        expected_mac = hmac.new(
            self._ballot_secret,
            f"{task_id}:{parts[0]}:{parts[1]}".encode(),
            hashlib.sha256,
        ).hexdigest()[:24]
        return hmac.compare_digest(parts[2], expected_mac)

    def _feedback_summary_unlocked(self) -> dict[str, Any]:
        total = len(self.feedback_items)
        by_prof = {"dev": 0, "gamer": 0}
        if total == 0:
            return {
                "object": "hivemind.onboarding_feedback_summary",
                "campaign_id": CAMPAIGN_ID,
                "total_responses": 0,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": by_prof,
            }
        ease_sum = Fraction(0, 1)
        lag_ok = 0
        for item in self.feedback_items:
            ease_sum += Fraction(int(item["ease_rating"]), 1)
            if item["zero_game_lag"]:
                lag_ok += 1
            prof = str(item["profile"])
            if prof in by_prof:
                by_prof[prof] += 1
        return {
            "object": "hivemind.onboarding_feedback_summary",
            "campaign_id": CAMPAIGN_ID,
            "total_responses": total,
            "mean_ease_rating": round(float(ease_sum / Fraction(total, 1)), 2),
            "zero_game_lag_ratio": round(float(Fraction(lag_ok, total)), 4),
            "by_profile": by_prof,
        }

    def request_full(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
        """Handle a zero-inference request returning `(status_code, body_or_text, headers)`."""
        verb = (method or "GET").strip().upper()
        parts = urlsplit(url)
        validate_gateway_url(f"{parts.scheme}://{parts.netloc}")
        path = parts.path or "/"

        if "completions" in path.lower() or (
            isinstance(payload, Mapping) and ("messages" in payload or "prompt" in payload)
        ):
            with self._lock:
                self.inference_calls_count += 1
            raise ReadinessGateError(
                f"Zero-inference invariant violated: {verb} {path} is strictly forbidden."
            )

        hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}

        with self._lock:
            self.requests_total += 1
            if self.fail_route is not None and path == self.fail_route:
                return self.fail_status, {"error": f"simulated_{self.fail_status}"}, {}

            if verb == "GET" and path == "/health":
                return (
                    200,
                    {
                        "status": "ok",
                        "service": "hivemind-openrouter",
                        "campaign_id": CAMPAIGN_ID,
                        "build_sha": self.build_sha,
                    },
                    {"content-type": "application/json"},
                )

            if verb == "GET" and path == "/v1/models":
                return (
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "hivemind/coder"},
                            {"id": "hivemind/bee-stars"},
                            {"id": "hivemind/auto"},
                        ],
                    },
                    {"content-type": "application/json"},
                )

            if verb == "GET" and path == "/v1/hivemind/onboarding/manifest":
                return (
                    200,
                    {
                        "object": "hivemind.onboarding.manifest",
                        "campaign_id": CAMPAIGN_ID,
                        "status": "ok",
                        "build_sha": self.build_sha,
                        "cloud_run_revision": self.revision,
                        "holiday_routes": list(self.holiday_routes),
                        "downloadable_scripts": list(self.allowed_scripts),
                        "personal_pc_only": True,
                        "financial_promise": FINANCIAL_PROMISE,
                    },
                    {"content-type": "application/json"},
                )

            if verb == "GET" and path in ("/noel", "/quick", "/signup", "/raid-noel", "/arena-noel"):
                self._nonce_seq += 1
                nonce = (
                    self.static_csp_nonce
                    if self.static_csp_nonce is not None
                    else f"NoelScale05N{self._nonce_seq:06d}{secrets.token_hex(4)}"
                )
                extra_cards = ""
                if path == "/arena-noel":
                    cards = []
                    for idx, tid in enumerate(CURATED_ARENA_TASK_IDS):
                        ballot = self._sign_ballot(tid, (self._nonce_seq + idx) % 2)
                        cards.append(f'<article data-task-id="{tid}" data-ballot="{ballot}"></article>')
                    extra_cards = "".join(cards)
                html_doc = (
                    f'<!DOCTYPE html><html lang="en"><head>'
                    f'<style nonce="{nonce}">body{{font-family:sans-serif}}</style></head>'
                    f'<body><main data-campaign="{CAMPAIGN_ID}">{extra_cards}</main>'
                    f'<script nonce="{nonce}">document.documentElement.dataset.ready="1";</script>'
                    f"</body></html>"
                )
                csp = (
                    f"default-src 'none'; style-src 'nonce-{nonce}'; "
                    f"script-src 'nonce-{nonce}'; connect-src 'self'; base-uri 'none'"
                )
                return (
                    200,
                    html_doc,
                    {
                        "content-type": "text/html; charset=utf-8",
                        "content-security-policy": csp,
                    },
                )

            if verb == "GET" and path.startswith("/scripts/"):
                script_name = path.removeprefix("/scripts/")
                if script_name not in self.allowed_scripts:
                    return 404, {"error": "not_found"}, {"content-type": "application/json"}
                disk_file = self.scripts_dir / script_name
                if not disk_file.is_file():
                    return 404, {"error": "missing_on_disk"}, {"content-type": "application/json"}
                raw_bytes = disk_file.read_bytes()
                digest = hashlib.sha256(raw_bytes).hexdigest()
                return (
                    200,
                    raw_bytes.decode("utf-8", errors="replace"),
                    {
                        "content-type": "text/x-python; charset=utf-8",
                        "x-content-sha256": digest,
                    },
                )

            if verb == "POST" and path == "/v1/hivemind/register":
                body_in = dict(payload or {})
                handle = str(body_in.get("handle") or "noel-friend")
                role = str(body_in.get("role") or "prosumer")
                self._acct_seq += 1
                acct_id = f"acct_scale05_{self._acct_seq:04d}"
                api_key = f"hm-scale05_{self._acct_seq:04d}_{secrets.token_hex(6)}"
                record = {
                    "account_id": acct_id,
                    "api_key": api_key,
                    "handle": handle,
                    "role": role,
                }
                self.accounts_by_key[api_key] = record
                return (
                    200,
                    {"object": "hivemind.account.registered", **record},
                    {"content-type": "application/json"},
                )

            if verb == "GET" and path == "/v1/hivemind/me":
                auth = hdrs.get("authorization", "")
                token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
                acct = self.accounts_by_key.get(token)
                if acct is None:
                    return 401, {"error": "invalid_api_key"}, {"content-type": "application/json"}
                self.activated_account_ids.add(acct["account_id"])
                return (
                    200,
                    {
                        "object": "hivemind.account",
                        "account_id": acct["account_id"],
                        "handle": acct["handle"],
                        "role": acct["role"],
                    },
                    {"content-type": "application/json"},
                )

            if verb == "POST" and path == "/v1/hivemind/arena/vote":
                vote_in = dict(payload or {})
                task_id = str(vote_in.get("task_id") or "")
                ballot = str(vote_in.get("ballot_token") or "")
                chosen = str(vote_in.get("chosen_side") or "")
                if (
                    task_id not in CURATED_ARENA_TASK_IDS
                    or chosen not in ("A", "B", "tie")
                    or not self._verify_ballot(task_id, ballot)
                ):
                    return 422, {"error": "invalid_ballot"}, {"content-type": "application/json"}
                self.arena_votes_count += 1
                return (
                    200,
                    {
                        "object": "hivemind.blind_ab_arena.summary",
                        "recorded": True,
                        "total_votes": self.arena_votes_count,
                    },
                    {"content-type": "application/json"},
                )

            if verb == "GET" and path == "/v1/hivemind/arena/summary":
                return (
                    200,
                    {
                        "object": "hivemind.blind_ab_arena.summary",
                        "campaign_id": CAMPAIGN_ID,
                        "total_votes": self.arena_votes_count,
                    },
                    {"content-type": "application/json"},
                )

            if verb == "POST" and path == "/v1/hivemind/onboarding/feedback":
                fb_in = dict(payload or {})
                allowed_fb_keys = {"profile", "ease_rating", "zero_game_lag", "platform"}
                if set(fb_in.keys()) - allowed_fb_keys:
                    return 422, {"error": "unexpected_pii_field"}, {"content-type": "application/json"}
                prof = str(fb_in.get("profile") or "dev").strip().lower()
                raw_ease = fb_in.get("ease_rating")
                if (
                    prof not in ("dev", "gamer")
                    or isinstance(raw_ease, bool)
                    or not isinstance(raw_ease, int)
                    or not (1 <= raw_ease <= 5)
                ):
                    return 422, {"error": "invalid_feedback"}, {"content-type": "application/json"}
                self.feedback_items.append(
                    {
                        "profile": prof,
                        "ease_rating": int(raw_ease),
                        "zero_game_lag": bool(fb_in.get("zero_game_lag", True)),
                        "platform": str(fb_in.get("platform") or "linux"),
                    }
                )
                return 200, self._feedback_summary_unlocked(), {"content-type": "application/json"}

            if verb == "GET" and path == "/v1/hivemind/onboarding/feedback/summary":
                return 200, self._feedback_summary_unlocked(), {"content-type": "application/json"}

            if verb == "GET" and path == "/v1/hivemind/raid/status":
                return (
                    200,
                    {
                        "object": "hivemind.raid_noel.status",
                        "campaign_id": CAMPAIGN_ID,
                        "swarm": {
                            "registered_accounts": len(self.accounts_by_key),
                            "active_accounts_24h": len(self.activated_account_ids),
                            "registered_trainer_accounts": 0,
                            "registered_trainer_nodes": 0,
                            "active_trainers": 0,
                            "accepted_opentrainer_rounds": 0,
                        },
                        "onboarding_feedback": self._feedback_summary_unlocked(),
                    },
                    {"content-type": "application/json"},
                )

            if verb == "GET" and path == "/v1/hivemind/live-telemetry":
                return (
                    200,
                    {
                        "object": "hivemind.live_telemetry",
                        "accounts": {
                            "total_registered_accounts": len(self.accounts_by_key),
                            "active_accounts_24h": len(self.activated_account_ids),
                        },
                        "opentrainer": {
                            "registered_trainer_accounts_count": 0,
                            "total_registered_nodes": 0,
                        },
                    },
                    {"content-type": "application/json"},
                )

            return 404, {"error": "not_found"}, {"content-type": "application/json"}

    def request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """2-tuple `(status_code, dict_body)` adapter for parity gate, autoconfig, and post-setup probe."""
        code, raw_body, resp_headers = self.request_full(method, url, payload, headers)
        if isinstance(raw_body, dict):
            return code, raw_body
        text = str(raw_body)
        return code, {
            "_html": text,
            "html_length": len(text),
            "text_length": len(text),
            "content_type": resp_headers.get("content-type", ""),
        }


def _assert_read_only_zero_inference(method: str, url: str, payload: Any = None) -> str:
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD"):
        raise ReadinessGateError(
            f"Live gateway check is strictly read-only: HTTP method {verb!r} is forbidden."
        )
    if payload is not None:
        raise ReadinessGateError("Live gateway check never sends request body payloads.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    if "completions" in (parts.path or "").lower():
        raise ReadinessGateError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
        )
    return origin


def default_live_http_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
    """Perform a proxy-free, read-only GET request to an allowlisted live HiveMind gateway."""
    _assert_read_only_zero_inference(method, url, payload)
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-scale-readiness-gate/1.0",
    }
    if headers:
        req_headers.update(headers)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=None, headers=req_headers, method=method.upper())
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw_text = resp.read(524_288).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            lower_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        raw_err = exc.read(65_536).decode("utf-8", errors="replace")
        err_headers = {str(k).lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
        try:
            parsed_err = json.loads(raw_err) if raw_err.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": raw_err[:240]}
        body_out = parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err}
        return int(exc.code), body_out, err_headers
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {"network_error": type(exc).__name__, "detail": str(reason)[:200]}, {}

    ctype = lower_headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            parsed = json.loads(raw_text) if raw_text.strip() else {}
            return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed}), lower_headers
        except json.JSONDecodeError:
            return status_code, raw_text, lower_headers
    try:
        parsed_any = json.loads(raw_text)
        if isinstance(parsed_any, dict):
            return status_code, parsed_any, lower_headers
    except json.JSONDecodeError:
        pass
    return status_code, raw_text, lower_headers


def _invoke_live_http(
    live_http_fn: LiveHttpFn,
    method: str,
    url: str,
) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
    """Invoke a 2-arg or 4-arg live HTTP callable returning a 2-tuple or 3-tuple safely."""
    _assert_read_only_zero_inference(method, url, None)
    sig = inspect.signature(live_http_fn)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    try:
        if has_varargs or len(positional) >= 4:
            raw = live_http_fn(method.upper(), url, None, None)
        else:
            raw = live_http_fn(method.upper(), url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": type(exc).__name__, "detail": str(exc)[:200]}, {}

    if isinstance(raw, tuple) and len(raw) == 3:
        code, body_or_text, hdrs = raw
        norm_hdrs = {str(k).lower(): str(v) for k, v in (hdrs or {}).items()}
        return int(code), body_or_text, norm_hdrs
    if isinstance(raw, tuple) and len(raw) == 2:
        code, body_or_text = raw
        norm_hdrs = {}
        if isinstance(body_or_text, Mapping):
            if isinstance(body_or_text.get("headers"), Mapping):
                norm_hdrs = {str(k).lower(): str(v) for k, v in body_or_text["headers"].items()}
            if isinstance(body_or_text.get("csp"), str):
                norm_hdrs.setdefault("content-security-policy", body_or_text["csp"])
        return int(code), body_or_text, norm_hdrs
    raise ReadinessGateError("live_http_fn must return `(status, body)` or `(status, body, headers)`.")


def default_subprocess_runner(
    argv: Sequence[str],
    timeout_s: float = SUBPROCESS_TIMEOUT_S,
) -> tuple[int, str]:
    """Run a local read-only command (`git rev-parse HEAD`) with `shell=False` and `gcloud` blocked."""
    if not argv:
        raise ReadinessGateError("Subprocess argv must not be empty.")
    exe_name = Path(str(argv[0])).name.lower()
    if exe_name in ("gcloud", "gcloud.cmd", "gcloud.exe"):
        raise ReadinessGateError("Cloud mutation guard: executing gcloud is strictly forbidden.")
    proc = subprocess.run(
        [str(x) for x in argv],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout_s,
        check=False,
        shell=False,
    )
    return int(proc.returncode), proc.stdout


def run_live_gateway_check(
    live_url: str,
    *,
    modules: ScaleReadinessModules,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    deploy_script_path: Path | None = None,
    live_http_fn: LiveHttpFn = default_live_http_request,
    runner: RunnerFn = default_subprocess_runner,
    lang: str = "en",
) -> dict[str, Any]:
    """Perform read-only zero-inference parity, 404 drift, and CSP nonce verification on `live_url`."""
    script_lang = _single_script_lang(normalize_lang(lang))
    is_fr = script_lang == "fr"
    clean_live = validate_gateway_url(live_url)

    captured_html: dict[str, tuple[int, str, dict[str, str]]] = {}

    def _parity_http_adapter(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (payload, headers)
        code, body_or_text, resp_headers = _invoke_live_http(live_http_fn, method, url)
        path = urlsplit(url).path or "/"
        if path in LIVE_HTML_PATHS:
            html_str = (
                body_or_text
                if isinstance(body_or_text, str)
                else str(body_or_text.get("_html") or body_or_text.get("html") or "")
                if isinstance(body_or_text, Mapping)
                else ""
            )
            captured_html[path] = (code, html_str, resp_headers)
        if isinstance(body_or_text, Mapping):
            return code, dict(body_or_text)
        text = str(body_or_text)
        return code, {
            "content_type": resp_headers.get("content-type", ""),
            "text_length": len(text),
            "html_length": len(text),
        }

    parity_report = modules.parity_gate.check_cloud_run_holiday_parity(
        clean_live,
        expected_sha=expected_sha,
        check_git_head=check_git_head,
        verify_all_scripts=True,
        http_fn=_parity_http_adapter,
        runner=runner,
        lang=script_lang,
    )
    parity_receipt_ok = verify_receipt_sha256(parity_report)

    extra_missing_routes: list[str] = []
    extra_failed_routes: list[str] = []
    extra_statuses: dict[str, int] = {}
    diagnostics: list[str] = [
        sanitize_text_for_display(str(d)) for d in (parity_report.get("diagnostics") or ())
    ]

    for extra_path, expected_obj in LIVE_EXTRA_SUMMARY_PATHS:
        code, body_or_text, _ = _invoke_live_http(live_http_fn, "GET", f"{clean_live}{extra_path}")
        extra_statuses[extra_path] = code
        valid_json = isinstance(body_or_text, Mapping) and body_or_text.get("object") == expected_obj
        if code != 200 or not valid_json:
            extra_failed_routes.append(extra_path)
            if code == 404:
                extra_missing_routes.append(extra_path)
            diag = (
                f"GET {extra_path} en échec sur la passerelle live (HTTP {code})."
                if is_fr
                else f"GET {extra_path} failed on live gateway (HTTP {code})."
            )
            diagnostics.append(diag)

    html_200_count = 0
    valid_nonces: list[str] = []
    dom_sink_hits = 0
    csp_missing_paths: list[str] = []

    for html_path in LIVE_HTML_PATHS:
        entry = captured_html.get(html_path)
        if entry is None:
            continue
        code, html_text, hdrs = entry
        if code != 200:
            continue
        html_200_count += 1
        if _FORBIDDEN_DOM_SINK_RE.search(html_text):
            dom_sink_hits += 1
        nonce = modules.load_harness.extract_csp_nonce(hdrs, html_text)
        if nonce is None:
            csp_missing_paths.append(html_path)
        else:
            valid_nonces.append(nonce)

    unique_nonces_count = len(set(valid_nonces))
    unique_nonces_ratio = (
        round(unique_nonces_count / html_200_count, 6) if html_200_count > 0 else 1.0
    )
    live_csp_ok = bool(
        html_200_count == len(LIVE_HTML_PATHS)
        and not csp_missing_paths
        and unique_nonces_count == html_200_count
        and dom_sink_hits == 0
    )
    if html_200_count > 0 and (csp_missing_paths or unique_nonces_count < html_200_count):
        csp_diag = (
            f"Violation CSP sur pages HTML live : nonces uniques={unique_nonces_count}/{html_200_count}."
            if is_fr
            else f"Live HTML CSP nonce violation: unique_nonces={unique_nonces_count}/{html_200_count}."
        )
        diagnostics.append(csp_diag)
    if dom_sink_hits > 0:
        sink_diag = (
            "Puits DOM dangereux détecté dans le HTML live (textContent uniquement autorisé)."
            if is_fr
            else "Unsafe DOM sink detected in live HTML response (textContent only allowed)."
        )
        diagnostics.append(sink_diag)

    missing_routes = sorted(set(parity_report["missing_routes"]) | set(extra_missing_routes))
    failed_routes = sorted(set(parity_report["failed_routes"]) | set(extra_failed_routes))
    missing_scripts = list(parity_report["missing_scripts"])
    failed_scripts = list(parity_report["failed_scripts"])
    missing_manifest_routes = list(parity_report["missing_manifest_routes"])
    missing_manifest_scripts = list(parity_report["missing_manifest_scripts"])

    drift_404_detected = bool(missing_routes or missing_scripts)
    sha_drift = bool(parity_report["sha_drift"])
    cloud_run_lags_main = bool(
        parity_report["cloud_run_lags_main"] or drift_404_detected or sha_drift
    )
    live_ok = bool(
        parity_report["parity_ok"]
        and parity_receipt_ok
        and not extra_failed_routes
        and live_csp_ok
        and not drift_404_detected
        and not sha_drift
    )
    redeploy_required = not live_ok

    redeploy_plan = modules.parity_gate.build_cloud_run_redeploy_plan(
        deploy_script_path=deploy_script_path,
        target_sha=parity_report.get("expected_build_sha"),
        check_git_head=False,
        gateway=clean_live,
        runner=runner,
        lang=script_lang,
    )

    merged_http_statuses = {
        **dict(parity_report["http_statuses"]),
        **extra_statuses,
    }

    return _attach_receipt(
        {
            "live_url": clean_live,
            "live_ok": live_ok,
            "parity_ok": bool(parity_report["parity_ok"] and not extra_failed_routes),
            "live_csp_ok": live_csp_ok,
            "drift_404_detected": drift_404_detected,
            "cloud_run_lags_main": cloud_run_lags_main,
            "redeploy_required": redeploy_required,
            "routes_verified_count": int(parity_report["routes_verified_count"])
            + (len(LIVE_EXTRA_SUMMARY_PATHS) - len(extra_failed_routes)),
            "routes_total_count": int(parity_report["routes_total_count"])
            + len(LIVE_EXTRA_SUMMARY_PATHS),
            "scripts_verified_count": int(parity_report["scripts_verified_count"]),
            "scripts_total_count": int(parity_report["scripts_total_count"]),
            "html_pages_checked_count": html_200_count,
            "unique_csp_nonces_count": unique_nonces_count,
            "unique_csp_nonces_ratio": unique_nonces_ratio,
            "forbidden_dom_sinks_count": dom_sink_hits,
            "live_build_sha": str(parity_report["live_build_sha"]),
            "expected_build_sha": parity_report["expected_build_sha"],
            "sha_drift": sha_drift,
            "cloud_run_revision": parity_report["cloud_run_revision"],
            "http_statuses": merged_http_statuses,
            "script_http_statuses": dict(parity_report["script_http_statuses"]),
            "missing_routes": missing_routes,
            "failed_routes": failed_routes,
            "missing_scripts": missing_scripts,
            "failed_scripts": failed_scripts,
            "missing_manifest_routes": missing_manifest_routes,
            "missing_manifest_scripts": missing_manifest_scripts,
            "redeploy_command": str(redeploy_plan["redeploy_command"]),
            "zero_inference_verified": True,
            "parity_receipt_sha256": str(parity_report["receipt_sha256"]),
            "parity_receipt_valid": parity_receipt_ok,
            "diagnostics": diagnostics,
        }
    )


def run_gate_g1_parity(
    *,
    modules: ScaleReadinessModules,
    simulator: LocalScaleGatewaySimulator,
    scripts_dir: Path = DEFAULT_SCRIPTS_DIR,
    deploy_script_path: Path | None = None,
    live_check: Mapping[str, Any] | None = None,
    runner: RunnerFn = default_subprocess_runner,
    lang: str = "en",
) -> dict[str, Any]:
    """(G1) Cloud Run holiday route, manifest, 17-script allowlist, and FinOps parity gate."""
    t0 = time.perf_counter()
    script_lang = _single_script_lang(normalize_lang(lang))
    is_fr = script_lang == "fr"
    diagnostics: list[str] = []

    required_scripts = tuple(modules.parity_gate.REQUIRED_SCRIPTS)
    disk_verified = 0
    missing_disk_scripts: list[str] = []
    syntax_error_scripts: list[str] = []

    for s_name in required_scripts:
        s_path = scripts_dir / s_name
        if not s_path.is_file() or s_path.stat().st_size == 0:
            missing_disk_scripts.append(s_name)
            continue
        if s_name.endswith(".py"):
            try:
                ast.parse(s_path.read_text(encoding="utf-8"), filename=f"scripts/{s_name}")
            except SyntaxError:
                syntax_error_scripts.append(s_name)
                continue
        disk_verified += 1

    if missing_disk_scripts:
        joined_m = ", ".join(missing_disk_scripts)
        diagnostics.append(
            f"Scripts allowlistés introuvables ou vides sur disque : {joined_m}."
            if is_fr
            else f"Allowlisted scripts missing or empty on disk: {joined_m}."
        )
    if syntax_error_scripts:
        joined_syn = ", ".join(syntax_error_scripts)
        diagnostics.append(
            f"Erreur de syntaxe AST dans les scripts : {joined_syn}."
            if is_fr
            else f"Python AST syntax error in allowlisted scripts: {joined_syn}."
        )

    eff_deploy = deploy_script_path if deploy_script_path is not None else DEFAULT_DEPLOY_SCRIPT
    finops_plan = modules.parity_gate.build_cloud_run_redeploy_plan(
        deploy_script_path=eff_deploy,
        target_sha=DEFAULT_OFFLINE_BUILD_SHA,
        check_git_head=False,
        gateway=simulator.gateway_url,
        runner=runner,
        lang=script_lang,
    )
    finops_receipt_ok = verify_receipt_sha256(finops_plan)
    for d_item in finops_plan.get("diagnostics") or ():
        diagnostics.append(sanitize_text_for_display(str(d_item)))

    local_parity = modules.parity_gate.check_cloud_run_holiday_parity(
        simulator.gateway_url,
        expected_sha=simulator.build_sha,
        check_git_head=False,
        verify_all_scripts=True,
        http_fn=simulator.request_json,
        runner=runner,
        lang=script_lang,
    )
    local_parity_receipt_ok = verify_receipt_sha256(local_parity)
    for d_item in local_parity.get("diagnostics") or ():
        diagnostics.append(sanitize_text_for_display(str(d_item)))

    live_checked = live_check is not None
    live_parity_ok = True
    drift_404 = False
    redeploy_required = bool(local_parity["redeploy_required"] or not finops_plan["finops_ok"])
    routes_verified = int(local_parity["routes_verified_count"])
    routes_total = int(local_parity["routes_total_count"])
    scripts_verified = int(local_parity["scripts_verified_count"])
    scripts_total = int(local_parity["scripts_total_count"])
    missing_routes: list[str] = list(local_parity["missing_routes"])
    missing_scripts: list[str] = list(local_parity["missing_scripts"])
    sha_drift = bool(local_parity["sha_drift"])

    if live_check is not None:
        live_parity_ok = bool(
            live_check.get("parity_ok")
            and not live_check.get("drift_404_detected")
            and not live_check.get("sha_drift")
            and live_check.get("parity_receipt_valid")
        )
        drift_404 = bool(live_check.get("drift_404_detected"))
        redeploy_required = redeploy_required or bool(live_check.get("redeploy_required"))
        routes_verified = int(live_check["routes_verified_count"])
        routes_total = int(live_check["routes_total_count"])
        scripts_verified = int(live_check["scripts_verified_count"])
        scripts_total = int(live_check["scripts_total_count"])
        missing_routes = list(live_check.get("missing_routes") or ())
        missing_scripts = list(live_check.get("missing_scripts") or ())
        sha_drift = bool(live_check.get("sha_drift"))
        for d_item in live_check.get("diagnostics") or ():
            clean_d = sanitize_text_for_display(str(d_item))
            if clean_d not in diagnostics:
                diagnostics.append(clean_d)

    sub_receipt_valid = bool(
        finops_receipt_ok
        and local_parity_receipt_ok
        and (verify_receipt_sha256(live_check) if live_check is not None else True)
    )
    zero_inf = bool(
        local_parity["zero_inference_verified"]
        and finops_plan["zero_inference_verified"]
        and simulator.inference_calls_count == 0
    )

    passed = bool(
        disk_verified == len(required_scripts)
        and not missing_disk_scripts
        and not syntax_error_scripts
        and finops_plan["finops_ok"]
        and local_parity["parity_ok"]
        and live_parity_ok
        and sub_receipt_valid
        and zero_inf
    )
    elapsed_s = round(max(0.0, time.perf_counter() - t0), 4)

    summary_en = (
        f"Routes {routes_verified}/{routes_total}, scripts {scripts_verified}/{scripts_total}, "
        f"FinOps {finops_plan['invariants_verified_count']}/{finops_plan['invariants_total_count']} "
        f"(live_checked={live_checked}, drift_404={drift_404})."
    )
    summary_fr = (
        f"Routes {routes_verified}/{routes_total}, scripts {scripts_verified}/{scripts_total}, "
        f"FinOps {finops_plan['invariants_verified_count']}/{finops_plan['invariants_total_count']} "
        f"(live_vérifié={live_checked}, dérive_404={drift_404})."
    )

    return {
        "gate_id": "G1",
        "slug": "cloud_run_route_and_script_parity",
        "title_en": "Cloud Run Holiday Route, Manifest & 17-Script Parity",
        "title_fr": "Parité Cloud Run Routes Noël, Manifeste & 17 Scripts",
        "passed": passed,
        "status": "GO" if passed else "NO_GO",
        "elapsed_s": elapsed_s,
        "zero_inference_verified": zero_inf,
        "sub_receipt_sha256": str(
            live_check["receipt_sha256"] if live_check is not None else local_parity["receipt_sha256"]
        ),
        "sub_receipt_valid": sub_receipt_valid,
        "summary_en": summary_en,
        "summary_fr": summary_fr,
        "metrics": {
            "routes_verified_count": routes_verified,
            "routes_total_count": routes_total,
            "scripts_verified_count": scripts_verified,
            "scripts_total_count": scripts_total,
            "disk_scripts_verified_count": disk_verified,
            "disk_scripts_total_count": len(required_scripts),
            "finops_ok": bool(finops_plan["finops_ok"]),
            "invariants_verified_count": int(finops_plan["invariants_verified_count"]),
            "invariants_total_count": int(finops_plan["invariants_total_count"]),
            "live_checked": live_checked,
            "drift_404_detected": drift_404,
            "sha_drift": sha_drift,
            "redeploy_required": redeploy_required,
            "missing_routes": missing_routes,
            "missing_scripts": missing_scripts,
        },
        "diagnostics": diagnostics,
    }


def run_gate_g2_router_burst(
    *,
    modules: ScaleReadinessModules,
    simulator: LocalScaleGatewaySimulator,
    devs: int = DEFAULT_BURST_DEVS,
    gamers: int = DEFAULT_BURST_GAMERS,
    workers: int = DEFAULT_BURST_WORKERS,
    p95_slo_ms: float = DEFAULT_P95_SLO_MS,
    p99_slo_ms: float = DEFAULT_P99_SLO_MS,
    burst_http_fn: LiveHttpFn | None = None,
    live_check: Mapping[str, Any] | None = None,
    clean_environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """(G2) 60-user concurrent router burst SLO (0 errors, unique per-request CSP nonces)."""
    t0 = time.perf_counter()
    script_lang = _single_script_lang(normalize_lang(lang))
    is_fr = script_lang == "fr"

    eff_http = burst_http_fn if burst_http_fn is not None else simulator.request_full
    eff_env = dict(clean_environ) if clean_environ is not None else {"PATH": "/usr/bin:/bin"}

    burst_report = modules.load_harness.run_noel_scale_burst(
        simulator.gateway_url,
        devs=devs,
        gamers=gamers,
        workers=workers,
        local_harness=True,
        include_register=True,
        p95_slo_ms=p95_slo_ms,
        p99_slo_ms=p99_slo_ms,
        max_error_ratio=0.0,
        http_fn=eff_http,
        environ=eff_env,
        lang=script_lang,
    )
    burst_receipt_ok = verify_receipt_sha256(burst_report)
    diagnostics = [sanitize_text_for_display(str(d)) for d in (burst_report.get("diagnostics") or ())]

    live_csp_ok = True
    if live_check is not None and int(live_check.get("html_pages_checked_count") or 0) > 0:
        if (
            float(live_check.get("unique_csp_nonces_ratio") or 0.0) < 1.0
            or int(live_check.get("forbidden_dom_sinks_count") or 0) > 0
        ):
            live_csp_ok = False
            msg = (
                "Échec CSP ou puits DOM sur les pages HTML de la passerelle live."
                if is_fr
                else "Live gateway HTML pages failed unique CSP nonce or DOM sink check."
            )
            diagnostics.append(msg)

    zero_inf = bool(burst_report["zero_inference_verified"] and simulator.inference_calls_count == 0)
    passed = bool(
        burst_report["scale_slo_ok"]
        and int(burst_report["error_count"]) == 0
        and float(burst_report["unique_csp_nonces_ratio"]) == 1.0
        and bool(burst_report["registrations_conserved"])
        and bool(burst_report["arena_votes_conserved"])
        and bool(burst_report["ces_responses_conserved"])
        and live_csp_ok
        and burst_receipt_ok
        and zero_inf
    )
    elapsed_s = round(max(0.0, time.perf_counter() - t0), 4)

    summary_en = (
        f"Participants={burst_report['participants_total']}, requests={burst_report['requests_total']}, "
        f"errors={burst_report['error_count']}, csp_nonce_ratio={burst_report['unique_csp_nonces_ratio']}, "
        f"p95_ms={burst_report['p95_latency_ms']}, p99_ms={burst_report['p99_latency_ms']}."
    )
    summary_fr = (
        f"Participants={burst_report['participants_total']}, requêtes={burst_report['requests_total']}, "
        f"erreurs={burst_report['error_count']}, ratio_nonces_csp={burst_report['unique_csp_nonces_ratio']}, "
        f"p95_ms={burst_report['p95_latency_ms']}, p99_ms={burst_report['p99_latency_ms']}."
    )

    return {
        "gate_id": "G2",
        "slug": "router_60_user_burst_slo",
        "title_en": "60-User Concurrent Router Burst SLO & Unique CSP Nonces",
        "title_fr": "SLO Rafale Concurrente 60 Utilisateurs & Nonces CSP Uniques",
        "passed": passed,
        "status": "GO" if passed else "NO_GO",
        "elapsed_s": elapsed_s,
        "zero_inference_verified": zero_inf,
        "sub_receipt_sha256": str(burst_report["receipt_sha256"]),
        "sub_receipt_valid": burst_receipt_ok,
        "summary_en": summary_en,
        "summary_fr": summary_fr,
        "metrics": {
            "participants_total": int(burst_report["participants_total"]),
            "devs_count": int(burst_report["devs_count"]),
            "gamers_count": int(burst_report["gamers_count"]),
            "requests_total": int(burst_report["requests_total"]),
            "error_count": int(burst_report["error_count"]),
            "error_ratio": float(burst_report["error_ratio"]),
            "sqlite_lock_error_count": int(burst_report["sqlite_lock_error_count"]),
            "html_renders_total": int(burst_report["html_renders_total"]),
            "unique_csp_nonces_count": int(burst_report["unique_csp_nonces_count"]),
            "unique_csp_nonces_ratio": float(burst_report["unique_csp_nonces_ratio"]),
            "p95_latency_ms": float(burst_report["p95_latency_ms"]),
            "p99_latency_ms": float(burst_report["p99_latency_ms"]),
            "registrations_conserved": bool(burst_report["registrations_conserved"]),
            "arena_votes_conserved": bool(burst_report["arena_votes_conserved"]),
            "ces_responses_conserved": bool(burst_report["ces_responses_conserved"]),
            "live_csp_ok": live_csp_ok,
        },
        "diagnostics": diagnostics,
    }


def run_gate_g3_cohort_rehearsal(
    *,
    modules: ScaleReadinessModules,
    cohort_size: int = DEFAULT_COHORT_SIZE,
    corporate_controls: int = DEFAULT_CORPORATE_CONTROLS,
    workers: int = DEFAULT_COHORT_WORKERS,
    yield_slo_ms: float = DEFAULT_YIELD_SLO_MS,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    inject_thermal_failures: int = 0,
    inject_yield_failures: int = 0,
    workspace_post_hook: Callable[[str, str, Path, Path], None] | None = None,
    report_date: str = "2026-12-25",
    clean_environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """(G3) 50-workspace multi-OS cohort & 100% Amadeus/corporate-PC block rate."""
    t0 = time.perf_counter()
    norm_lang = normalize_lang(lang)
    eff_env = dict(clean_environ) if clean_environ is not None else {"PATH": "/usr/bin:/bin"}

    pipeline_mods = modules.cohort_rehearsal.OnboardingPipelineModules(
        autoconfig=modules.vscode_autoconfig,
        gamer_selftest=modules.gamer_selftest,
        post_setup_probe=modules.post_setup_probe,
        funnel_report=modules.funnel_report,
        raid_bulletin=modules.raid_bulletin,
    )
    cohort_report = modules.cohort_rehearsal.run_canary_cohort_rehearsal(
        DEFAULT_GATEWAY,
        cohort_size=cohort_size,
        corporate_controls=corporate_controls,
        workers=workers,
        yield_slo_ms=yield_slo_ms,
        max_temp_c=max_temp_c,
        min_free_vram_mib=min_free_vram_mib,
        inject_gamer_thermal_failures=inject_thermal_failures,
        inject_gamer_yield_failures=inject_yield_failures,
        report_date=report_date,
        lang=norm_lang,
        environ=eff_env,
        pipeline_modules=pipeline_mods,
        workspace_post_hook=workspace_post_hook,
    )
    cohort_receipt_ok = verify_receipt_sha256(cohort_report)
    diagnostics = [sanitize_text_for_display(str(d)) for d in (cohort_report.get("diagnostics") or ())]

    zero_inf = bool(
        cohort_report["zero_inference_verified"] and int(cohort_report["inference_calls_count"]) == 0
    )
    passed = bool(
        cohort_report["rehearsal_ok"]
        and float(cohort_report["personal_ready_ratio"]) == 1.0
        and float(cohort_report["corporate_blocked_ratio"]) == 1.0
        and int(cohort_report["corporate_file_writes_count"]) == 0
        and int(cohort_report["corporate_network_calls_count"]) == 0
        and bool(cohort_report["zero_key_leaks_verified"])
        and cohort_receipt_ok
        and zero_inf
    )
    elapsed_s = round(max(0.0, time.perf_counter() - t0), 4)

    summary_en = (
        f"Personal ready {cohort_report['personal_ready_count']}/{cohort_report['cohort_size']} "
        f"(ratio={cohort_report['personal_ready_ratio']:.4f}), corporate/Amadeus blocked "
        f"{cohort_report['corporate_blocked_count']}/{cohort_report['corporate_cohort_size']} "
        f"(ratio={cohort_report['corporate_blocked_ratio']:.4f}, key_leaks=0)."
    )
    summary_fr = (
        f"Espaces perso prêts {cohort_report['personal_ready_count']}/{cohort_report['cohort_size']} "
        f"(ratio={cohort_report['personal_ready_ratio']:.4f}), contrôles Amadeus/entreprise bloqués "
        f"{cohort_report['corporate_blocked_count']}/{cohort_report['corporate_cohort_size']} "
        f"(ratio={cohort_report['corporate_blocked_ratio']:.4f}, fuites_clé=0)."
    )

    return {
        "gate_id": "G3",
        "slug": "multi_os_cohort_and_corporate_block",
        "title_en": "50-Workspace Multi-OS Cohort & 100% Amadeus/Corporate Block",
        "title_fr": "Cohorte 50 Espaces Multi-OS & Blocage 100 % Amadeus/Entreprise",
        "passed": passed,
        "status": "GO" if passed else "NO_GO",
        "elapsed_s": elapsed_s,
        "zero_inference_verified": zero_inf,
        "sub_receipt_sha256": str(cohort_report["receipt_sha256"]),
        "sub_receipt_valid": cohort_receipt_ok,
        "summary_en": summary_en,
        "summary_fr": summary_fr,
        "metrics": {
            "cohort_size": int(cohort_report["cohort_size"]),
            "personal_ready_count": int(cohort_report["personal_ready_count"]),
            "personal_ready_ratio": float(cohort_report["personal_ready_ratio"]),
            "corporate_cohort_size": int(cohort_report["corporate_cohort_size"]),
            "corporate_blocked_count": int(cohort_report["corporate_blocked_count"]),
            "corporate_blocked_ratio": float(cohort_report["corporate_blocked_ratio"]),
            "corporate_file_writes_count": int(cohort_report["corporate_file_writes_count"]),
            "corporate_network_calls_count": int(cohort_report["corporate_network_calls_count"]),
            "zero_key_leaks_verified": bool(cohort_report["zero_key_leaks_verified"]),
            "archetypes_verified_count": len(cohort_report.get("archetype_breakdown") or ()),
        },
        "diagnostics": diagnostics,
    }


def run_gate_g4_gamer_whisper(
    *,
    modules: ScaleReadinessModules,
    simulator: LocalScaleGatewaySimulator,
    yield_slo_ms: float = DEFAULT_YIELD_SLO_MS,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    gpu_probe_fn: Callable[..., Any] | None = None,
    worker_lifecycle_fn: Callable[..., Mapping[str, Any]] | None = None,
    clean_environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """(G4) Gamer-Whisper `< 1.5 s` (`yield_ms < 1500.0`) GPU yield & `<= 70.0 C` quiet-fan self-test."""
    t0 = time.perf_counter()
    script_lang = _single_script_lang(normalize_lang(lang))
    is_fr = script_lang == "fr"
    eff_env = dict(clean_environ) if clean_environ is not None else {"PATH": "/usr/bin:/bin"}

    observed_temp = 58.0

    def _default_rtx_gpu_probe(**_kwargs: Any) -> Any:
        return modules.gamer_selftest.GpuTelemetry(
            present=True,
            name="NVIDIA GeForce RTX 4070",
            total_vram_mib=12288,
            free_vram_mib=9216,
            used_vram_mib=3072,
            temp_c=observed_temp,
            util_pct=4.0,
            reason="ok",
        )

    def _idle_proc_probe(**_kwargs: Any) -> list[str]:
        return ["explorer.exe", "steam.exe"]

    eff_gpu_probe = gpu_probe_fn if gpu_probe_fn is not None else _default_rtx_gpu_probe
    eff_lifecycle = (
        worker_lifecycle_fn
        if worker_lifecycle_fn is not None
        else modules.gamer_selftest.default_worker_lifecycle_drill
    )

    with tempfile.TemporaryDirectory(prefix="hm_scale05_g4_") as raw_tmp:
        g4_home = Path(raw_tmp) / "gamer_home"
        g4_cwd = g4_home / "personal_gaming"
        g4_cwd.mkdir(parents=True, exist_ok=True)

        primary = modules.gamer_selftest.run_gamer_whisper_selftest(
            gateway=simulator.gateway_url,
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            yield_slo_ms=yield_slo_ms,
            check_gateway=True,
            submit_feedback=True,
            ease_rating=5,
            lang=script_lang,
            os_name="nt",
            system_name="Windows",
            environ=eff_env,
            home=g4_home,
            cwd=g4_cwd,
            gpu_probe_fn=eff_gpu_probe,
            process_probe_fn=_idle_proc_probe,
            worker_lifecycle_fn=eff_lifecycle,
            http_fn=simulator.request_json,
        )

        nogpu_report = modules.gamer_selftest.run_gamer_whisper_selftest(
            gateway=simulator.gateway_url,
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            yield_slo_ms=yield_slo_ms,
            check_gateway=False,
            submit_feedback=False,
            lang=script_lang,
            os_name="posix",
            system_name="Linux",
            environ=eff_env,
            home=g4_home,
            cwd=g4_cwd,
            gpu_probe_fn=lambda **_kw: modules.gamer_selftest.GpuTelemetry(
                present=False,
                reason="nvidia_smi_not_found",
            ),
            process_probe_fn=lambda **_kw: ["bash", "python3"],
            worker_lifecycle_fn=lambda **_kw: {
                "worker_started": True,
                "worker_stopped": True,
                "clean_release": True,
                "retained_worker_vram_mib": 0,
                "yield_latency_ms": 140.0,
                "method": "deterministic_nogpu_drill",
            },
            http_fn=simulator.request_json,
        )

        hot_neg = modules.gamer_selftest.run_gamer_whisper_selftest(
            gateway=simulator.gateway_url,
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            yield_slo_ms=yield_slo_ms,
            check_gateway=False,
            submit_feedback=False,
            lang=script_lang,
            os_name="nt",
            system_name="Windows",
            environ=eff_env,
            home=g4_home,
            cwd=g4_cwd,
            gpu_probe_fn=lambda **_kw: modules.gamer_selftest.GpuTelemetry(
                present=True,
                name="NVIDIA GeForce RTX 4070",
                total_vram_mib=12288,
                free_vram_mib=9216,
                used_vram_mib=3072,
                temp_c=float(max_temp_c) + 8.5,
                util_pct=45.0,
                reason="ok",
            ),
            process_probe_fn=_idle_proc_probe,
            worker_lifecycle_fn=lambda **_kw: {
                "worker_started": True,
                "worker_stopped": True,
                "clean_release": True,
                "retained_worker_vram_mib": 0,
                "yield_latency_ms": 210.0,
                "method": "deterministic_hot_drill",
            },
            http_fn=simulator.request_json,
        )

        slow_neg = modules.gamer_selftest.run_gamer_whisper_selftest(
            gateway=simulator.gateway_url,
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            yield_slo_ms=yield_slo_ms,
            check_gateway=False,
            submit_feedback=False,
            lang=script_lang,
            os_name="nt",
            system_name="Windows",
            environ=eff_env,
            home=g4_home,
            cwd=g4_cwd,
            gpu_probe_fn=_default_rtx_gpu_probe,
            process_probe_fn=_idle_proc_probe,
            worker_lifecycle_fn=lambda **_kw: {
                "worker_started": True,
                "worker_stopped": False,
                "clean_release": False,
                "retained_worker_vram_mib": 256,
                "yield_latency_ms": float(yield_slo_ms) + 280.0,
                "method": "deterministic_slow_drill",
            },
            http_fn=simulator.request_json,
        )

    primary_receipt_ok = verify_receipt_sha256(primary)
    nogpu_ok = bool(
        nogpu_report["ready"]
        and nogpu_report["verdict"] == "cpu_or_dev_only_no_nvidia_gpu"
        and verify_receipt_sha256(nogpu_report)
    )
    negative_controls_ok = bool(
        not hot_neg["thermal_ok"]
        and not hot_neg["ready"]
        and not slow_neg["yield_slo_ok"]
        and not slow_neg["ready"]
        and slow_neg["exit_code"] == 2
    )

    diagnostics: list[str] = []
    for item in (*(primary.get("errors") or ()), *(primary.get("warnings") or ())):
        diagnostics.append(sanitize_text_for_display(str(item)))
    if not nogpu_ok:
        diagnostics.append(
            "Échec du contrôle honnête sans GPU NVIDIA."
            if is_fr
            else "Honest no-GPU reporting check failed."
        )
    if not negative_controls_ok:
        diagnostics.append(
            "Échec des contrôles négatifs Gamer-Whisper (surchauffe ou dépassement budget d'éviction)."
            if is_fr
            else "Gamer-Whisper fail-closed negative controls did not reject injected faults."
        )

    gov = primary.get("thermal_vram_governor") if isinstance(primary.get("thermal_vram_governor"), Mapping) else {}
    reported_temp = gov.get("temp_c")
    zero_inf = bool(primary["zero_inference_verified"] and simulator.inference_calls_count == 0)

    passed = bool(
        primary["ready"]
        and primary["exit_code"] == 0
        and primary["yield_slo_ok"]
        and primary["retained_worker_vram_mib"] == 0
        and primary["thermal_ok"]
        and primary["vram_reserve_ok"]
        and primary["cooldown_policy_ok"]
        and primary["bandwidth_cap_ok"]
        and primary["within_10s_slo"]
        and nogpu_ok
        and negative_controls_ok
        and primary_receipt_ok
        and zero_inf
    )
    elapsed_s = round(max(0.0, time.perf_counter() - t0), 4)

    yield_val = primary.get("yield_latency_ms")
    summary_en = (
        f"yield_ms={yield_val} (budget_ms < {yield_slo_ms:.1f}), "
        f"retained_vram_mib={primary.get('retained_worker_vram_mib')}, "
        f"temp_c={reported_temp} (max_temp_c <= {max_temp_c:.1f}), "
        f"negative_controls={negative_controls_ok}."
    )
    summary_fr = (
        f"éviction_ms={yield_val} (budget_ms < {yield_slo_ms:.1f}), "
        f"vram_retenue_mib={primary.get('retained_worker_vram_mib')}, "
        f"temp_c={reported_temp} (plafond_c <= {max_temp_c:.1f}), "
        f"contrôles_négatifs={negative_controls_ok}."
    )

    return {
        "gate_id": "G4",
        "slug": "gamer_whisper_yield_and_thermal_selftest",
        "title_en": "Gamer-Whisper Sub-1.5s GPU Yield & <=70 C Quiet-Fan Self-Test",
        "title_fr": "Autotest Gamer-Whisper Éviction <1,5 s & Silence <=70 C",
        "passed": passed,
        "status": "GO" if passed else "NO_GO",
        "elapsed_s": elapsed_s,
        "zero_inference_verified": zero_inf,
        "sub_receipt_sha256": str(primary["receipt_sha256"]),
        "sub_receipt_valid": primary_receipt_ok,
        "summary_en": summary_en,
        "summary_fr": summary_fr,
        "metrics": {
            "ready": bool(primary["ready"]),
            "yield_slo_ok": bool(primary["yield_slo_ok"]),
            "yield_latency_ms": yield_val,
            "yield_slo_ms": float(primary["yield_slo_ms"]),
            "retained_worker_vram_mib": primary["retained_worker_vram_mib"],
            "thermal_ok": bool(primary["thermal_ok"]),
            "observed_temp_c": reported_temp,
            "max_temp_c": float(primary["max_temp_c"]),
            "vram_reserve_ok": bool(primary["vram_reserve_ok"]),
            "cooldown_policy_ok": bool(primary["cooldown_policy_ok"]),
            "bandwidth_cap_ok": bool(primary["bandwidth_cap_ok"]),
            "within_10s_slo": bool(primary["within_10s_slo"]),
            "honest_nogpu_verified": nogpu_ok,
            "negative_controls_verified": negative_controls_ok,
        },
        "diagnostics": diagnostics,
    }


def run_gate_g5_dev_autoconfig_and_funnel(
    *,
    modules: ScaleReadinessModules,
    simulator: LocalScaleGatewaySimulator,
    dev_ease_rating: int = 5,
    autoconfig_clock_fn: ClockFn | None = None,
    clean_environ: Mapping[str, str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """(G5) Dev `< 60 s` VS Code autoconfig, clean uninstall & zero-PII CES funnel readiness."""
    t0 = time.perf_counter()
    norm_lang = normalize_lang(lang)
    script_lang = _single_script_lang(norm_lang)
    is_fr = script_lang == "fr"
    eff_env = dict(clean_environ) if clean_environ is not None else {"PATH": "/usr/bin:/bin"}

    diagnostics: list[str] = []

    with tempfile.TemporaryDirectory(prefix="hm_scale05_g5_") as raw_tmp:
        g5_root = Path(raw_tmp)
        g5_home = g5_root / "dev_home"
        g5_ws = g5_home / "projects" / "advent_of_code_2026"
        g5_ws.mkdir(parents=True, exist_ok=True)
        (g5_ws / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")
        git_dir = g5_ws / ".git"
        git_dir.mkdir(parents=True, exist_ok=True)
        (git_dir / "config").write_text(
            f'[core]\n\trepositoryformatversion = 0\n[remote "origin"]\n\turl = {HOBBY_DISPLAY_GIT_URL}\n',
            encoding="utf-8",
        )
        vs_dir = g5_ws / ".vscode"
        vs_dir.mkdir(parents=True, exist_ok=True)
        (vs_dir / "settings.json").write_text(
            '// Personal editor preferences\n{\n  "editor.tabSize": 4,\n}\n',
            encoding="utf-8",
        )

        ws_env = {**eff_env, "HOME": str(g5_home), "HOSTNAME": "home-dev-box"}

        def _deterministic_probe_runner(argv: Sequence[str], timeout_s: float) -> tuple[int, str]:
            _ = timeout_s
            cmd0 = Path(str(argv[0])).name.lower() if argv else ""
            if cmd0 == "nvidia-smi":
                return 1, ""
            return 0, "bash\ncode\npython3\n"

        captured_probe: list[dict[str, Any]] = []

        def _probe_hook(**probe_kwargs: Any) -> dict[str, Any]:
            res = modules.post_setup_probe.run_post_setup_probe(
                gateway=probe_kwargs["gateway"],
                smoke_only=False,
                verify_account=True,
                credentials_path=probe_kwargs["credentials_path"],
                continue_path=probe_kwargs["continue_path"],
                feedback_result=probe_kwargs.get("feedback_result"),
                lang=script_lang,
                os_name="posix",
                environ=ws_env,
                home=g5_home,
                http_fn=simulator.request_json,
                runner=_deterministic_probe_runner,
            )
            captured_probe.append(res)
            return res

        eff_clock = autoconfig_clock_fn if autoconfig_clock_fn is not None else time.monotonic

        auto_rc, auto_report = modules.vscode_autoconfig.run_vscode_autoconfig(
            gateway=simulator.gateway_url,
            handle="noel-scale05-dev",
            role="prosumer",
            workspace=g5_ws,
            write_vscode_settings=True,
            skip_probe=False,
            ease_rating=dev_ease_rating,
            system_name="Linux",
            lang=script_lang,
            os_name="posix",
            environ=ws_env,
            home=g5_home,
            cwd=g5_ws,
            http_fn=simulator.request_json,
            probe_fn=_probe_hook,
            clock_fn=eff_clock,
        )

        cred_file = g5_home / ".hivemind" / "credentials.json"
        cont_file = g5_home / ".continue" / "hivemind_provider.yaml"
        vs_file = g5_ws / ".vscode" / "settings.json"

        posix_0600_ok = True
        if os.name != "nt":
            for sec_path in (cred_file, cont_file, vs_file):
                if not sec_path.is_file() or (sec_path.stat().st_mode & 0o777) != 0o600:
                    posix_0600_ok = False

        vs_text = vs_file.read_text(encoding="utf-8") if vs_file.is_file() else ""
        vs_parsed = json.loads(vs_text) if vs_text.strip() else {}
        zero_ws_secret_leaks = (
            not bool(_API_KEY_RE.search(vs_text))
            and vs_parsed.get("editor.tabSize") == 4
        )

        un_rc, un_report = modules.vscode_autoconfig.run_vscode_autoconfig(
            gateway=simulator.gateway_url,
            workspace=g5_ws,
            uninstall=True,
            lang=script_lang,
            os_name="posix",
            environ=ws_env,
            home=g5_home,
            cwd=g5_ws,
            http_fn=simulator.request_json,
        )
        vs_after_un = json.loads(vs_file.read_text(encoding="utf-8")) if vs_file.is_file() else {}
        uninstall_clean = bool(
            un_rc == 0
            and not cred_file.exists()
            and not cont_file.exists()
            and vs_after_un == {"editor.tabSize": 4}
            and verify_receipt_sha256(un_report)
        )

    auto_receipt_ok = verify_receipt_sha256(auto_report)
    probe_receipt_ok = bool(captured_probe and verify_receipt_sha256(captured_probe[0]))
    probe_ready = bool(auto_report.get("probe") and auto_report["probe"].get("ready"))
    within_60s = bool(auto_report.get("within_60s_slo")) and float(auto_report.get("elapsed_s", 999.0)) < AUTOCONFIG_SLO_S

    for err_item in auto_report.get("errors") or ():
        diagnostics.append(sanitize_text_for_display(str(err_item)))

    if not within_60s:
        diagnostics.append(
            f"Budget SLO autoconfig VS Code dépassé ({auto_report.get('elapsed_s')} s >= {AUTOCONFIG_SLO_S:.0f} s)."
            if is_fr
            else f"VS Code autoconfig exceeded 60s SLO ({auto_report.get('elapsed_s')} s >= {AUTOCONFIG_SLO_S:.0f} s)."
        )
    if not uninstall_clean or not zero_ws_secret_leaks or not posix_0600_ok:
        diagnostics.append(
            "Échec d'hygiène locale `.vscode/settings.json`, permissions 0600 ou désinstallation propre."
            if is_fr
            else "Workspace `.vscode/settings.json` secret hygiene, 0600 mode, or `--uninstall` round-trip failed."
        )

    # Zero-PII CES funnel readiness check with planted PII fields that MUST be dropped
    planted_funnel_snapshot: dict[str, Any] = {
        "funnel": {
            "landing_visits": 60,
            "signups": 54,
            "first_requests": 52,
            "preflight_runs": 24,
            "install_completed": 22,
            "first_rounds": 21,
        },
        "feedback_items": [
            {"profile": "dev", "ease_rating": dev_ease_rating, "zero_game_lag": True},
            {"profile": "dev", "ease_rating": dev_ease_rating, "zero_game_lag": True},
            {"profile": "dev", "ease_rating": dev_ease_rating, "zero_game_lag": True},
            {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True},
            {"profile": "gamer", "ease_rating": dev_ease_rating, "zero_game_lag": True},
        ],
        "email": "planted-pii@example.org",
        "handle": "planted-handle-secret",
        "api_key": "hm-planted_secret_key_should_drop",
        "ip_address": "127.0.0.1",
        "hostname": "planted-home-rig",
    }
    funnel_receipt = modules.funnel_report.analyze_onboarding_funnel(
        planted_funnel_snapshot,
        generated_at_utc="2026-12-25T20:00:00Z",
    )
    funnel_md = modules.funnel_report.render_markdown_report(funnel_receipt, lang=norm_lang)
    funnel_receipt_ok = verify_receipt_sha256(funnel_receipt)
    funnel_leaks = modules.funnel_report.scan_for_pii_leaks(
        json.dumps(funnel_receipt, sort_keys=True, ensure_ascii=False) + "\n" + funnel_md
    )
    privacy_block = (
        funnel_receipt.get("privacy")
        if isinstance(funnel_receipt.get("privacy"), Mapping)
        else {}
    )
    pii_dropped = int(privacy_block.get("redacted_fields_dropped") or 0)
    zero_pii_verified = bool(
        privacy_block.get("zero_pii") is True
        and pii_dropped >= 5
        and funnel_leaks == []
        and "planted-pii" not in funnel_md
        and "planted-handle-secret" not in funnel_md
    )

    mean_ease = funnel_receipt.get("mean_ease_rating")
    zero_lag_ratio = funnel_receipt.get("zero_game_lag_ratio")
    ease_target_ok = bool(
        isinstance(mean_ease, (int, float))
        and Fraction(str(mean_ease)) >= CES_EASE_MIN
        and isinstance(zero_lag_ratio, (int, float))
        and Fraction(str(zero_lag_ratio)) == ZERO_LAG_TARGET
    )
    funnel_healthy = bool(funnel_receipt.get("status") == "healthy" and not funnel_receipt.get("alerts"))
    if not funnel_healthy or not ease_target_ok:
        diagnostics.append(
            f"Alerte entonnoir CES Noël : statut={funnel_receipt.get('status')}, "
            f"mean_ease_rating={mean_ease} (cible >= 4.2)."
            if is_fr
            else f"CES onboarding funnel readiness alert: status={funnel_receipt.get('status')}, "
            f"mean_ease_rating={mean_ease} (target >= 4.2)."
        )

    sub_receipt_valid = bool(auto_receipt_ok and probe_receipt_ok and funnel_receipt_ok)
    zero_inf = bool(auto_report["zero_inference_verified"] and simulator.inference_calls_count == 0)

    passed = bool(
        auto_rc == 0
        and auto_report["ready"]
        and within_60s
        and probe_ready
        and posix_0600_ok
        and zero_ws_secret_leaks
        and uninstall_clean
        and zero_pii_verified
        and funnel_healthy
        and ease_target_ok
        and sub_receipt_valid
        and zero_inf
    )
    elapsed_s = round(max(0.0, time.perf_counter() - t0), 4)

    summary_en = (
        f"Autoconfig elapsed={auto_report.get('elapsed_s')}s (<60s={within_60s}), "
        f"probe_ready={probe_ready}, uninstall_clean={uninstall_clean}, "
        f"funnel={funnel_receipt.get('status')} (mean_ease={mean_ease}/5, pii_dropped={pii_dropped})."
    )
    summary_fr = (
        f"Autoconfig durée={auto_report.get('elapsed_s')}s (<60s={within_60s}), "
        f"sonde_prête={probe_ready}, désinstall_propre={uninstall_clean}, "
        f"entonnoir={funnel_receipt.get('status')} (facilité={mean_ease}/5, pii_supprimés={pii_dropped})."
    )

    return {
        "gate_id": "G5",
        "slug": "dev_vscode_autoconfig_and_ces_funnel",
        "title_en": "Dev <60s VS Code Autoconfig & Zero-PII CES Funnel Readiness",
        "title_fr": "Autoconfig VS Code Dev <60 s & Entonnoir CES Zéro-PII",
        "passed": passed,
        "status": "GO" if passed else "NO_GO",
        "elapsed_s": elapsed_s,
        "zero_inference_verified": zero_inf,
        "sub_receipt_sha256": str(funnel_receipt["receipt_sha256"]),
        "sub_receipt_valid": sub_receipt_valid,
        "summary_en": summary_en,
        "summary_fr": summary_fr,
        "metrics": {
            "autoconfig_ready": bool(auto_report["ready"]),
            "autoconfig_elapsed_s": float(auto_report.get("elapsed_s") or 0.0),
            "within_60s_slo": within_60s,
            "probe_ready": probe_ready,
            "posix_0600_ok": posix_0600_ok,
            "zero_workspace_secret_leaks": zero_ws_secret_leaks,
            "uninstall_clean": uninstall_clean,
            "funnel_status": str(funnel_receipt.get("status")),
            "funnel_alerts_count": len(funnel_receipt.get("alerts") or ()),
            "mean_ease_rating": mean_ease,
            "zero_game_lag_ratio": zero_lag_ratio,
            "pii_fields_dropped_count": pii_dropped,
            "zero_pii_verified": zero_pii_verified,
            "autoconfig_receipt_sha256": str(auto_report["receipt_sha256"]),
        },
        "diagnostics": diagnostics,
    }


def _skipped_compliance_gates() -> list[dict[str, Any]]:
    specs = (
        (
            "G1",
            "cloud_run_route_and_script_parity",
            "Cloud Run Holiday Route, Manifest & 17-Script Parity",
            "Parité Cloud Run Routes Noël, Manifeste & 17 Scripts",
        ),
        (
            "G2",
            "router_60_user_burst_slo",
            "60-User Concurrent Router Burst SLO & Unique CSP Nonces",
            "SLO Rafale Concurrente 60 Utilisateurs & Nonces CSP Uniques",
        ),
        (
            "G3",
            "multi_os_cohort_and_corporate_block",
            "50-Workspace Multi-OS Cohort & 100% Amadeus/Corporate Block",
            "Cohorte 50 Espaces Multi-OS & Blocage 100 % Amadeus/Entreprise",
        ),
        (
            "G4",
            "gamer_whisper_yield_and_thermal_selftest",
            "Gamer-Whisper Sub-1.5s GPU Yield & <=70 C Quiet-Fan Self-Test",
            "Autotest Gamer-Whisper Éviction <1,5 s & Silence <=70 C",
        ),
        (
            "G5",
            "dev_vscode_autoconfig_and_ces_funnel",
            "Dev <60s VS Code Autoconfig & Zero-PII CES Funnel Readiness",
            "Autoconfig VS Code Dev <60 s & Entonnoir CES Zéro-PII",
        ),
    )
    rows: list[dict[str, Any]] = []
    for gid, slug, title_en, title_fr in specs:
        rows.append(
            {
                "gate_id": gid,
                "slug": slug,
                "title_en": title_en,
                "title_fr": title_fr,
                "passed": False,
                "status": "SKIPPED_COMPLIANCE",
                "elapsed_s": 0.0,
                "zero_inference_verified": True,
                "sub_receipt_sha256": "",
                "sub_receipt_valid": False,
                "summary_en": "Skipped: host refused by Amadeus / Personal-PC compliance rule.",
                "summary_fr": "Ignoré : hôte refusé par la règle Amadeus / PC personnel.",
                "metrics": {},
                "diagnostics": [],
            }
        )
    return rows


def _execute_gate_safely(
    gate_id: str,
    slug: str,
    title_en: str,
    title_fr: str,
    fn: GateCallable,
    *,
    is_fr: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = fn()
        if not isinstance(result, Mapping) or result.get("gate_id") != gate_id:
            raise ReadinessGateError(f"Gate {gate_id} returned malformed gate payload.")
        return dict(result)
    except Exception as exc:  # noqa: BLE001 - fail closed into structured NO_GO gate entry
        elapsed_s = round(max(0.0, time.perf_counter() - t0), 4)
        safe_err = sanitize_text_for_display(f"{type(exc).__name__}: {exc}")
        diag = (
            f"Exception interne dans {gate_id} (échec fermé) : {safe_err}"
            if is_fr
            else f"Unhandled {gate_id} error (failing closed): {safe_err}"
        )
        return {
            "gate_id": gate_id,
            "slug": slug,
            "title_en": title_en,
            "title_fr": title_fr,
            "passed": False,
            "status": "NO_GO",
            "elapsed_s": elapsed_s,
            "zero_inference_verified": True,
            "sub_receipt_sha256": "",
            "sub_receipt_valid": False,
            "summary_en": f"Failed closed on runtime error: {safe_err}",
            "summary_fr": f"Échec fermé sur erreur d'exécution : {safe_err}",
            "metrics": {"exception_type": type(exc).__name__},
            "diagnostics": [diag],
        }


def run_noel_scale_readiness_gate(
    *,
    live_url: str | None = None,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    burst_devs: int = DEFAULT_BURST_DEVS,
    burst_gamers: int = DEFAULT_BURST_GAMERS,
    burst_workers: int = DEFAULT_BURST_WORKERS,
    p95_slo_ms: float = DEFAULT_P95_SLO_MS,
    p99_slo_ms: float = DEFAULT_P99_SLO_MS,
    cohort_size: int = DEFAULT_COHORT_SIZE,
    corporate_controls: int = DEFAULT_CORPORATE_CONTROLS,
    cohort_workers: int = DEFAULT_COHORT_WORKERS,
    yield_slo_ms: float = DEFAULT_YIELD_SLO_MS,
    max_temp_c: float = DEFAULT_MAX_TEMP_C,
    min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    slo_budget_s: float = DEFAULT_SLO_BUDGET_S,
    report_date: str | None = None,
    generated_at_utc: str | None = None,
    lang: str = "en",
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    check_cwd_git_remote: bool = False,
    scripts_dir: Path | None = None,
    deploy_script_path: Path | None = None,
    modules: ScaleReadinessModules | None = None,
    live_http_fn: LiveHttpFn = default_live_http_request,
    runner: RunnerFn = default_subprocess_runner,
    clock_fn: ClockFn = time.perf_counter,
    burst_http_fn: LiveHttpFn | None = None,
    static_csp_nonce: str | None = None,
    simulator_fail_route: str | None = None,
    inject_g3_thermal_failures: int = 0,
    inject_g3_yield_failures: int = 0,
    g3_workspace_post_hook: Callable[[str, str, Path, Path], None] | None = None,
    g4_gpu_probe_fn: Callable[..., Any] | None = None,
    g4_worker_lifecycle_fn: Callable[..., Mapping[str, Any]] | None = None,
    g5_dev_ease_rating: int = 5,
    g5_autoconfig_clock_fn: ClockFn | None = None,
    gate_overrides: Mapping[str, GateCallable] | None = None,
) -> dict[str, Any]:
    """Execute all 5 objective Noël-at-Scale Go/No-Go gates and return a SHA-256-receipted scorecard."""
    t_wall0 = float(clock_fn())
    norm_lang = normalize_lang(lang)
    is_fr = norm_lang == "fr"
    clean_date = validate_report_date(report_date)
    stamp = (
        str(generated_at_utc).strip()
        if (generated_at_utc is not None and str(generated_at_utc).strip())
        else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    clean_live_url: str | None = None
    if live_url is not None and str(live_url).strip():
        clean_live_url = validate_gateway_url(live_url)

    eff_gateway = clean_live_url if clean_live_url is not None else DEFAULT_GATEWAY
    mode_name = (
        "offline_rehearsal_plus_live_gateway"
        if clean_live_url is not None
        else "offline_rehearsal"
    )

    devs_n = _validate_int_range(burst_devs, "burst_devs", 1, MAX_BURST_COHORT)
    gamers_n = _validate_int_range(burst_gamers, "burst_gamers", 1, MAX_BURST_COHORT)
    b_workers_n = _validate_int_range(burst_workers, "burst_workers", 1, MAX_BURST_WORKERS)
    clean_p95 = _validate_positive_float(p95_slo_ms, "p95_slo_ms")
    clean_p99 = _validate_positive_float(p99_slo_ms, "p99_slo_ms")
    if clean_p99 < clean_p95:
        raise ReadinessGateError("p99_slo_ms must be >= p95_slo_ms.")

    cohort_n = _validate_int_range(cohort_size, "cohort_size", MIN_COHORT_SIZE, MAX_COHORT_SIZE)
    corp_n = _validate_int_range(
        corporate_controls,
        "corporate_controls",
        MIN_CORPORATE_CONTROLS,
        MAX_CORPORATE_CONTROLS,
    )
    c_workers_n = _validate_int_range(cohort_workers, "cohort_workers", 1, MAX_COHORT_WORKERS)
    clean_yield_ms = _validate_positive_float(yield_slo_ms, "yield_slo_ms")
    clean_temp_c = _validate_positive_float(max_temp_c, "max_temp_c")
    clean_vram_mib = _validate_int_range(min_free_vram_mib, "min_free_vram_mib", 512, 131072)
    clean_slo_s = _validate_positive_float(slo_budget_s, "slo_budget_s")
    clean_g5_ease = _validate_int_range(g5_dev_ease_rating, "g5_dev_ease_rating", 1, 5)

    # Step 0: Amadeus / Personal-PC Compliance Gate BEFORE executing any gate or live network I/O
    compliance = evaluate_personal_pc_compliance(
        environ=environ,
        cwd=cwd,
        check_git_remote=check_cwd_git_remote,
        lang=norm_lang,
    )
    if not compliance["compliance_ok"]:
        elapsed_refused = round(max(0.0, float(clock_fn()) - t_wall0), 4)
        skipped_gates = _skipped_compliance_gates()
        return _attach_receipt(
            {
                "schema": SCHEMA,
                "object": RECEIPT_OBJECT,
                "slice_id": SLICE_ID,
                "campaign_id": CAMPAIGN_ID,
                "generated_at_utc": stamp,
                "report_date": clean_date,
                "mode": mode_name,
                "lang": norm_lang,
                "gateway": eff_gateway,
                "live_url": clean_live_url,
                "live_check_enabled": clean_live_url is not None,
                "live_check_ok": False if clean_live_url is not None else None,
                "drift_404_detected": False,
                "compliance_ok": False,
                "go_for_launch": False,
                "verdict": "NO_GO",
                "exit_code": 2,
                "gates_passed_count": 0,
                "gates_total_count": len(GATE_IDS),
                "gates_failed_count": len(GATE_IDS),
                "passed_gate_ids": [],
                "failed_gate_ids": list(GATE_IDS),
                "gate_verdicts": {gid: False for gid in GATE_IDS},
                "elapsed_s": elapsed_refused,
                "slo_budget_s": clean_slo_s,
                "within_10s_slo": elapsed_refused < clean_slo_s,
                "zero_inference_verified": True,
                "inference_calls_count": 0,
                "all_sub_receipts_valid": False,
                "sub_receipts_verified_count": 0,
                "gates": skipped_gates,
                "live_gateway_check": None,
                "compliance": compliance,
                "diagnostics": [sanitize_text_for_display(compliance["summary"])],
            }
        )

    eff_scripts_dir = scripts_dir if scripts_dir is not None else DEFAULT_SCRIPTS_DIR
    loaded_mods = modules if modules is not None else load_scale_readiness_modules(eff_scripts_dir)

    if expected_sha is not None and str(expected_sha).strip():
        _ = loaded_mods.parity_gate.validate_expected_sha(expected_sha)

    simulator = LocalScaleGatewaySimulator(
        DEFAULT_GATEWAY,
        scripts_dir=eff_scripts_dir,
        allowed_scripts=loaded_mods.parity_gate.REQUIRED_SCRIPTS,
        holiday_routes=loaded_mods.parity_gate.EXPECTED_MANIFEST_HOLIDAY_ROUTES,
        build_sha=DEFAULT_OFFLINE_BUILD_SHA,
        revision=DEFAULT_OFFLINE_REVISION,
        static_csp_nonce=static_csp_nonce,
        fail_route=simulator_fail_route,
    )

    live_check_report: dict[str, Any] | None = None
    if clean_live_url is not None:
        live_check_report = run_live_gateway_check(
            clean_live_url,
            modules=loaded_mods,
            expected_sha=expected_sha,
            check_git_head=check_git_head,
            deploy_script_path=deploy_script_path,
            live_http_fn=live_http_fn,
            runner=runner,
            lang=norm_lang,
        )

    clean_sub_env = {"PATH": str((environ or os.environ).get("PATH") or "/usr/bin:/bin")}
    overrides = dict(gate_overrides or {})

    gate_builders: tuple[tuple[str, str, str, str, GateCallable], ...] = (
        (
            "G1",
            "cloud_run_route_and_script_parity",
            "Cloud Run Holiday Route, Manifest & 17-Script Parity",
            "Parité Cloud Run Routes Noël, Manifeste & 17 Scripts",
            overrides.get("G1")
            or (
                lambda: run_gate_g1_parity(
                    modules=loaded_mods,
                    simulator=simulator,
                    scripts_dir=eff_scripts_dir,
                    deploy_script_path=deploy_script_path,
                    live_check=live_check_report,
                    runner=runner,
                    lang=norm_lang,
                )
            ),
        ),
        (
            "G2",
            "router_60_user_burst_slo",
            "60-User Concurrent Router Burst SLO & Unique CSP Nonces",
            "SLO Rafale Concurrente 60 Utilisateurs & Nonces CSP Uniques",
            overrides.get("G2")
            or (
                lambda: run_gate_g2_router_burst(
                    modules=loaded_mods,
                    simulator=simulator,
                    devs=devs_n,
                    gamers=gamers_n,
                    workers=b_workers_n,
                    p95_slo_ms=clean_p95,
                    p99_slo_ms=clean_p99,
                    burst_http_fn=burst_http_fn,
                    live_check=live_check_report,
                    clean_environ=clean_sub_env,
                    lang=norm_lang,
                )
            ),
        ),
        (
            "G3",
            "multi_os_cohort_and_corporate_block",
            "50-Workspace Multi-OS Cohort & 100% Amadeus/Corporate Block",
            "Cohorte 50 Espaces Multi-OS & Blocage 100 % Amadeus/Entreprise",
            overrides.get("G3")
            or (
                lambda: run_gate_g3_cohort_rehearsal(
                    modules=loaded_mods,
                    cohort_size=cohort_n,
                    corporate_controls=corp_n,
                    workers=c_workers_n,
                    yield_slo_ms=clean_yield_ms,
                    max_temp_c=clean_temp_c,
                    min_free_vram_mib=clean_vram_mib,
                    inject_thermal_failures=inject_g3_thermal_failures,
                    inject_yield_failures=inject_g3_yield_failures,
                    workspace_post_hook=g3_workspace_post_hook,
                    report_date=clean_date,
                    clean_environ=clean_sub_env,
                    lang=norm_lang,
                )
            ),
        ),
        (
            "G4",
            "gamer_whisper_yield_and_thermal_selftest",
            "Gamer-Whisper Sub-1.5s GPU Yield & <=70 C Quiet-Fan Self-Test",
            "Autotest Gamer-Whisper Éviction <1,5 s & Silence <=70 C",
            overrides.get("G4")
            or (
                lambda: run_gate_g4_gamer_whisper(
                    modules=loaded_mods,
                    simulator=simulator,
                    yield_slo_ms=clean_yield_ms,
                    max_temp_c=clean_temp_c,
                    min_free_vram_mib=clean_vram_mib,
                    gpu_probe_fn=g4_gpu_probe_fn,
                    worker_lifecycle_fn=g4_worker_lifecycle_fn,
                    clean_environ=clean_sub_env,
                    lang=norm_lang,
                )
            ),
        ),
        (
            "G5",
            "dev_vscode_autoconfig_and_ces_funnel",
            "Dev <60s VS Code Autoconfig & Zero-PII CES Funnel Readiness",
            "Autoconfig VS Code Dev <60 s & Entonnoir CES Zéro-PII",
            overrides.get("G5")
            or (
                lambda: run_gate_g5_dev_autoconfig_and_funnel(
                    modules=loaded_mods,
                    simulator=simulator,
                    dev_ease_rating=clean_g5_ease,
                    autoconfig_clock_fn=g5_autoconfig_clock_fn,
                    clean_environ=clean_sub_env,
                    lang=norm_lang,
                )
            ),
        ),
    )

    gate_rows: list[dict[str, Any]] = []
    for gid, slug, title_en, title_fr, builder in gate_builders:
        gate_rows.append(
            _execute_gate_safely(gid, slug, title_en, title_fr, builder, is_fr=is_fr)
        )

    elapsed_s = round(max(0.0, float(clock_fn()) - t_wall0), 4)
    within_10s_slo = elapsed_s < clean_slo_s

    passed_gate_ids = [g["gate_id"] for g in gate_rows if g.get("passed") is True]
    failed_gate_ids = [g["gate_id"] for g in gate_rows if g.get("passed") is not True]
    gates_passed_count = len(passed_gate_ids)
    gates_total_count = len(gate_rows)
    gates_failed_count = gates_total_count - gates_passed_count
    gate_verdicts = {str(g["gate_id"]): bool(g.get("passed")) for g in gate_rows}

    sub_receipts_verified_count = sum(1 for g in gate_rows if g.get("sub_receipt_valid") is True)
    all_sub_receipts_valid = sub_receipts_verified_count == gates_total_count

    zero_inference_verified = bool(
        simulator.inference_calls_count == 0
        and all(bool(g.get("zero_inference_verified")) for g in gate_rows)
        and (bool(live_check_report["zero_inference_verified"]) if live_check_report is not None else True)
    )

    live_check_ok: bool | None = (
        bool(live_check_report["live_ok"]) if live_check_report is not None else None
    )
    drift_404_detected = bool(
        live_check_report is not None and live_check_report.get("drift_404_detected")
    )

    diagnostics: list[str] = []
    for g in gate_rows:
        if not g.get("passed"):
            for d_msg in g.get("diagnostics") or ():
                prefixed = f"[{g['gate_id']}] {sanitize_text_for_display(str(d_msg))}"
                if prefixed not in diagnostics:
                    diagnostics.append(prefixed)

    if not within_10s_slo:
        slo_msg = (
            f"[SLO] Durée totale d'exécution ({elapsed_s:.3f} s) supérieure au budget ({clean_slo_s:.1f} s)."
            if is_fr
            else f"[SLO] Total readiness gate runtime ({elapsed_s:.3f} s) exceeded budget ({clean_slo_s:.1f} s)."
        )
        diagnostics.append(slo_msg)

    go_for_launch = bool(
        compliance["compliance_ok"]
        and gates_passed_count == gates_total_count
        and gates_total_count == len(GATE_IDS)
        and within_10s_slo
        and zero_inference_verified
        and all_sub_receipts_valid
        and (live_check_ok is not False)
    )

    return _attach_receipt(
        {
            "schema": SCHEMA,
            "object": RECEIPT_OBJECT,
            "slice_id": SLICE_ID,
            "campaign_id": CAMPAIGN_ID,
            "generated_at_utc": stamp,
            "report_date": clean_date,
            "mode": mode_name,
            "lang": norm_lang,
            "gateway": eff_gateway,
            "live_url": clean_live_url,
            "live_check_enabled": clean_live_url is not None,
            "live_check_ok": live_check_ok,
            "drift_404_detected": drift_404_detected,
            "compliance_ok": True,
            "go_for_launch": go_for_launch,
            "verdict": "GO" if go_for_launch else "NO_GO",
            "exit_code": 0 if go_for_launch else 2,
            "gates_passed_count": gates_passed_count,
            "gates_total_count": gates_total_count,
            "gates_failed_count": gates_failed_count,
            "passed_gate_ids": passed_gate_ids,
            "failed_gate_ids": failed_gate_ids,
            "gate_verdicts": gate_verdicts,
            "elapsed_s": elapsed_s,
            "slo_budget_s": clean_slo_s,
            "within_10s_slo": within_10s_slo,
            "zero_inference_verified": zero_inference_verified,
            "inference_calls_count": simulator.inference_calls_count,
            "all_sub_receipts_valid": all_sub_receipts_valid,
            "sub_receipts_verified_count": sub_receipts_verified_count,
            "gates": gate_rows,
            "live_gateway_check": live_check_report,
            "compliance": compliance,
            "diagnostics": diagnostics,
        }
    )


def _render_single_lang_markdown(report: Mapping[str, Any], *, fr: bool) -> str:
    go = bool(report.get("go_for_launch"))
    passed_n = int(report.get("gates_passed_count") or 0)
    total_n = int(report.get("gates_total_count") or len(GATE_IDS))

    if fr:
        heading = "# Tableau de Bord Go/No-Go Noël à l'Échelle (`SLICE-SCALE-05`)"
        decision_txt = "GO POUR LANCEMENT NOËL" if go else "NO-GO — BLOCAGE LANCEMENT"
        compliance_txt = COMPLIANCE_NOTICE_FR
        sec_summary = "## 1. Décision Globale & Invariants"
        sec_gates = "## 2. Les 5 Gates Objectives Go/No-Go (`G1`..`G5`)"
        tbl_hdr = "| Gate | Objectif | Statut | Durée (s) | Synthèse Mesurée |"
        sec_live = "## 3. Vérification Passerelle Live Zéro-Inférence (`--live-url`)"
        sec_diag = "## Diagnostics Actionnables"
    else:
        heading = "# Noël-at-Scale Unified Go/No-Go Readiness Scorecard (`SLICE-SCALE-05`)"
        decision_txt = "GO FOR LAUNCH" if go else "NO-GO — HOLD LAUNCH"
        compliance_txt = COMPLIANCE_NOTICE_EN
        sec_summary = "## 1. Launch Decision & Safety Invariants"
        sec_gates = "## 2. Objective 5-Gate Go/No-Go Scorecard (`G1`..`G5`)"
        tbl_hdr = "| Gate | Objective | Status | Elapsed (s) | Measured Summary |"
        sec_live = "## 3. Zero-Inference Live Gateway Check (`--live-url`)"
        sec_diag = "## Actionable Diagnostics"

    lines: list[str] = [
        heading,
        "",
        f"> **Compliance ({FINANCIAL_PROMISE})**: {compliance_txt}",
        "",
        sec_summary,
        "",
        f"- **Decision**: **{decision_txt}** (`go_for_launch={go}`, `verdict={report.get('verdict')}`)",
        f"- **Gates Passed**: `{passed_n}/{total_n}` (`gates_passed_count={passed_n}`, `gates_total_count={total_n}`)",
        f"- **Mode**: `{report.get('mode')}` (`gateway={report.get('gateway')}`)",
        f"- **Runtime SLO**: `{report.get('elapsed_s')} s` / `{report.get('slo_budget_s')} s` (`within_10s_slo={report.get('within_10s_slo')}`)",
        f"- **Zero GPU Inference**: `{report.get('zero_inference_verified')}` (`inference_calls_count={report.get('inference_calls_count')}`)",
        f"- **Sub-Receipts Verified**: `{report.get('sub_receipts_verified_count')}/{total_n}` (`all_sub_receipts_valid={report.get('all_sub_receipts_valid')}`)",
        "",
        sec_gates,
        "",
        tbl_hdr,
        "| :---: | :--- | :---: | :---: | :--- |",
    ]

    for gate in report.get("gates") or ():
        title = gate.get("title_fr") if fr else gate.get("title_en")
        summary = sanitize_text_for_display(
            str(gate.get("summary_fr") if fr else gate.get("summary_en") or "")
        )
        badge = "GO" if gate.get("passed") else str(gate.get("status") or "NO_GO")
        lines.append(
            f"| **{gate.get('gate_id')}** | {title} | **{badge}** | `{gate.get('elapsed_s')}` | {summary} |"
        )

    live_doc = report.get("live_gateway_check")
    if isinstance(live_doc, Mapping):
        lines.extend(
            [
                "",
                sec_live,
                "",
                f"- **Live URL**: `{live_doc.get('live_url')}` (`live_ok={live_doc.get('live_ok')}`)",
                f"- **404 Drift Detected**: `{live_doc.get('drift_404_detected')}` (`cloud_run_lags_main={live_doc.get('cloud_run_lags_main')}`)",
                f"- **Routes / Scripts**: `{live_doc.get('routes_verified_count')}/{live_doc.get('routes_total_count')}` routes, `{live_doc.get('scripts_verified_count')}/{live_doc.get('scripts_total_count')}` scripts",
                f"- **Live CSP Nonces**: `{live_doc.get('unique_csp_nonces_count')}/{live_doc.get('html_pages_checked_count')}` (`ratio={live_doc.get('unique_csp_nonces_ratio')}`)",
            ]
        )
        if live_doc.get("redeploy_required") and live_doc.get("redeploy_command"):
            lines.extend(
                [
                    "",
                    "```bash",
                    sanitize_text_for_display(str(live_doc["redeploy_command"])),
                    "```",
                ]
            )

    diags = report.get("diagnostics") or ()
    if diags:
        lines.extend(["", sec_diag, ""])
        for item in diags:
            lines.append(f"- {sanitize_text_for_display(str(item))}")

    lines.extend(["", f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`", ""])
    return "\n".join(lines)


def render_markdown_report(report: Mapping[str, Any], *, lang: str | None = None) -> str:
    """Render an English, French, or bilingual Markdown Go/No-Go scorecard audited for zero leaks."""
    eff_lang = normalize_lang(lang if lang is not None else str(report.get("lang") or "en"))
    if eff_lang == "fr":
        rendered = _render_single_lang_markdown(report, fr=True)
    elif eff_lang == "en":
        rendered = _render_single_lang_markdown(report, fr=False)
    else:
        fr_part = _render_single_lang_markdown(report, fr=True).rstrip()
        en_part = _render_single_lang_markdown(report, fr=False)
        rendered = f"{fr_part}\n\n---\n\n{en_part}"
    return _assert_safe_output(rendered)


def render_cli_report(report: Mapping[str, Any], *, lang: str | None = None) -> str:
    """Render a compact single-screen FR/EN CLI Go/No-Go scorecard audited for zero leaks."""
    eff_lang = normalize_lang(lang if lang is not None else str(report.get("lang") or "en"))
    is_fr = eff_lang == "fr"
    go = bool(report.get("go_for_launch"))
    badge = "[GO]" if go else "[NO-GO]"
    passed_n = int(report.get("gates_passed_count") or 0)
    total_n = int(report.get("gates_total_count") or len(GATE_IDS))
    notice = COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN

    headline = (
        f"{badge} Noël-at-Scale Readiness ({passed_n}/{total_n} gates, "
        f"go_for_launch={go}, durée={report.get('elapsed_s')}s)"
        if is_fr
        else f"{badge} Noël-at-Scale Readiness ({passed_n}/{total_n} gates, "
        f"go_for_launch={go}, elapsed={report.get('elapsed_s')}s)"
    )
    lines = [headline]
    for gate in report.get("gates") or ():
        g_tag = "PASS" if gate.get("passed") else "FAIL"
        g_title = gate.get("title_fr") if is_fr else gate.get("title_en")
        lines.append(f"  [{g_tag}] {gate.get('gate_id')} — {g_title} ({gate.get('elapsed_s')}s)")
    if report.get("live_check_enabled"):
        lines.append(
            f"  live_url={report.get('live_url')} live_ok={report.get('live_check_ok')} "
            f"drift_404={report.get('drift_404_detected')}"
        )
    for diag in report.get("diagnostics") or ():
        lines.append(f"  [DIAG] {sanitize_text_for_display(str(diag))}")
    lines.append(f"  compliance: {notice}")
    lines.append(f"  receipt_sha256={report.get('receipt_sha256')}")
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_scale_readiness_gate.py`."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified 5-gate Noël-at-Scale Go/No-Go readiness runner and scorecard "
            "(zero GPU inference, 100% personal home PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run deterministic offline 5-gate rehearsal without live gateway calls (default).",
    )
    parser.add_argument(
        "--live-url",
        default=None,
        help="Optional allowlisted gateway origin (https://*.run.app or loopback) for read-only parity check.",
    )
    parser.add_argument(
        "--expected-sha",
        default=None,
        help="Optional expected HIVEMIND_BUILD_SHA when --live-url is checked.",
    )
    parser.add_argument(
        "--check-git-head",
        action="store_true",
        help="Compare live Cloud Run build_sha against local `git rev-parse HEAD` when --live-url is set.",
    )
    parser.add_argument("--burst-devs", type=int, default=DEFAULT_BURST_DEVS)
    parser.add_argument("--burst-gamers", type=int, default=DEFAULT_BURST_GAMERS)
    parser.add_argument("--burst-workers", type=int, default=DEFAULT_BURST_WORKERS)
    parser.add_argument("--p95-slo-ms", type=float, default=DEFAULT_P95_SLO_MS)
    parser.add_argument("--p99-slo-ms", type=float, default=DEFAULT_P99_SLO_MS)
    parser.add_argument("--cohort-size", type=int, default=DEFAULT_COHORT_SIZE)
    parser.add_argument("--corporate-controls", type=int, default=DEFAULT_CORPORATE_CONTROLS)
    parser.add_argument("--cohort-workers", type=int, default=DEFAULT_COHORT_WORKERS)
    parser.add_argument("--yield-slo-ms", type=float, default=DEFAULT_YIELD_SLO_MS)
    parser.add_argument("--max-temp-c", type=float, default=DEFAULT_MAX_TEMP_C)
    parser.add_argument("--min-free-vram-mib", type=int, default=DEFAULT_MIN_FREE_VRAM_MIB)
    parser.add_argument("--slo-budget-s", type=float, default=DEFAULT_SLO_BUDGET_S)
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD report date.")
    parser.add_argument(
        "--lang",
        default="en",
        help="Output language: 'en', 'fr', or 'bilingual' (default: 'en').",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON scorecard with SHA-256 receipt.")
    parser.add_argument("--cli", action="store_true", help="Emit compact CLI scorecard summary.")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit FR/EN Markdown scorecard report (default when --json/--cli omitted).",
    )
    parser.add_argument("--output", default=None, help="Optional output file path.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    modules: ScaleReadinessModules | None = None,
    live_http_fn: LiveHttpFn = default_live_http_request,
    runner: RunnerFn = default_subprocess_runner,
) -> int:
    """CLI entrypoint returning `0` when `go_for_launch` is True or `2` (fail-closed) otherwise."""
    args = build_parser().parse_args(argv)
    try:
        if args.offline and args.live_url:
            raise ReadinessGateError(
                "Cannot combine --offline with --live-url; omit --offline to check a live gateway."
            )
        lang_code = normalize_lang(args.lang)
        report = run_noel_scale_readiness_gate(
            live_url=args.live_url,
            expected_sha=args.expected_sha,
            check_git_head=bool(args.check_git_head),
            burst_devs=args.burst_devs,
            burst_gamers=args.burst_gamers,
            burst_workers=args.burst_workers,
            p95_slo_ms=args.p95_slo_ms,
            p99_slo_ms=args.p99_slo_ms,
            cohort_size=args.cohort_size,
            corporate_controls=args.corporate_controls,
            cohort_workers=args.cohort_workers,
            yield_slo_ms=args.yield_slo_ms,
            max_temp_c=args.max_temp_c,
            min_free_vram_mib=args.min_free_vram_mib,
            slo_budget_s=args.slo_budget_s,
            report_date=args.date,
            lang=lang_code,
            environ=environ,
            cwd=cwd,
            modules=modules,
            live_http_fn=live_http_fn,
            runner=runner,
        )
        if args.json:
            rendered = _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        elif args.cli:
            rendered = render_cli_report(report, lang=lang_code) + "\n"
        else:
            rendered = render_markdown_report(report, lang=lang_code)

        if args.output:
            out_path = Path(str(args.output))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered, encoding="utf-8")

        sys.stdout.write(rendered)
        return 0 if report["go_for_launch"] else 2
    except ReadinessGateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
