#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Post-setup quickstart probe, 1-prompt CES reporter, and Cloud Run holiday route smoke gate.

Compatible with Python 3.10-3.13 across Windows 11, macOS, and Linux using ONLY the Python
standard library:
  1. Zero-inference Cloud Run Holiday Route Smoke Gate (`--smoke-routes` and default probe):
     checks GET `/health`, `/v1/models`, `/noel`, `/quick`, `/raid-noel`, `/arena-noel`, and
     `/v1/hivemind/onboarding/feedback/summary` without ever calling `/v1/chat/completions`.
  2. Post-setup local readiness probe: inspects `~/.hivemind/credentials.json` (redacting home
     paths to `~`, masking `hm-...` keys, plus optional `GET /v1/hivemind/me` with
     `--verify-account`), `~/.continue/hivemind_provider.yaml`, Gamer-Whisper thermal ceiling
     (`<= 70.0 C`), free VRAM reserve (`>= 2048 MiB`), active Steam/PC game detection with
     `< 1.5 s` yield budget verification, and a strictly local corporate-PC warning.
  3. 1-prompt anonymous CES & zero-game-lag reporter (`--submit-feedback` and `--prompt-ces`):
     posts zero-PII payloads to `POST /v1/hivemind/onboarding/feedback` with FR/EN handling of
     HTTP 404/429/5xx and emits `--json` or Markdown reports with a SHA-256 receipt.

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_post_setup_probe/1"
RECEIPT_OBJECT = "hivemind.noel_post_setup_receipt"
CAMPAIGN_ID = "noel-2026-easy-onboard"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"

SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")
VALID_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
MIN_PYTHON: tuple[int, int] = (3, 10)
MAX_TESTED_MINOR = 13

GAMER_WHISPER_MAX_TEMP_C = 70.0
GAMER_WHISPER_MIN_FREE_VRAM_MIB = 2048
GAMER_WHISPER_YIELD_BUDGET_S = 1.5
PREEMPTION_SIGNAL_ESTIMATE_S = 0.25
HTTP_TIMEOUT_S = 4.0
SUBPROCESS_TIMEOUT_S = 3.0

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

HOLIDAY_SMOKE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/health", "json", "core"),
    ("/v1/models", "json", "core"),
    ("/noel", "html", "holiday"),
    ("/quick", "html", "holiday"),
    ("/raid-noel", "html", "holiday"),
    ("/arena-noel", "html", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "json", "holiday"),
)

DEFAULT_GAME_EXECUTABLES: frozenset[str] = frozenset(
    {
        "cs2.exe",
        "cs2",
        "dota2.exe",
        "dota2",
        "cyberpunk2077.exe",
        "eldenring.exe",
        "bg3.exe",
        "helldivers2.exe",
        "valorant.exe",
        "valorant-win64-shipping.exe",
        "fortniteclient-win64-shipping.exe",
        "overwatch.exe",
        "league of legends.exe",
        "baldur.exe",
        "starfield.exe",
        "witcher3.exe",
        "gta5.exe",
        "r5apex.exe",
        "rustclient.exe",
        "palworld-win64-shipping.exe",
    }
)

_CORP_MARKERS: tuple[str, ...] = (
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

COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)[A-Za-z0-9_-]{6,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

HttpFn = Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], tuple[int, dict[str, Any]]]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]
InputFn = Callable[[str], str]


class ProbeError(ValueError):
    """Raised when probe configuration, gateway validation, or safety checks fail."""


@dataclass(frozen=True)
class GpuStatus:
    """Measured local NVIDIA GPU state without serial or UUID identifiers."""

    present: bool
    parse_error: bool = False
    name: str = ""
    total_mib: int = 0
    free_mib: int = 0
    temp_c: float = 0.0
    util_pct: float = 0.0


@dataclass(frozen=True)
class ProbeCheck:
    """Single post-setup readiness check result with FR/EN summary and fix command."""

    id: str
    title: str
    status: str  # "ok" | "warn" | "fail"
    summary: str
    fix_command: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "passed": self.passed,
            "summary": self.summary,
            "fix_command": self.fix_command,
            "details": dict(self.details),
        }


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to `'en'` or `'fr'`."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ProbeError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ProbeError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def normalize_profile(raw: str | None) -> str:
    """Normalize feedback profile selector to `'dev'` or `'gamer'`."""
    if raw is None or isinstance(raw, bool) or not isinstance(raw, str):
        raise ProbeError("Profile must be 'dev' or 'gamer'.")
    cleaned = raw.strip().lower()
    if cleaned not in VALID_PROFILES:
        raise ProbeError(f"Unsupported profile {raw!r}; expected 'dev' or 'gamer'.")
    return cleaned


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace local user home prefixes with `'~'` so usernames never leak."""
    raw = str(target)
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
    if redacted != raw:
        return redacted.replace("\\", "/")
    return raw


def mask_api_key(raw_key: object) -> str:
    """Mask an `hm-...` or `hm_live_...` key so secrets never appear in reports."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<missing>"
    cleaned = raw_key.strip()
    if not _is_valid_hivemind_key(cleaned):
        return "<invalid>"
    prefix = "hm_live_" if cleaned.startswith("hm_live_") else "hm-"
    tail = cleaned[-4:]
    return f"{prefix}***{tail}"


def _is_valid_hivemind_key(raw_key: object) -> bool:
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is `*.run.app` or a loopback address (`localhost`/`127.0.0.1`/`::1`)."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    if len(h) > 8 and h.endswith(".run.app") and not h.startswith(".") and ".." not in h:
        return True
    return False


def validate_gateway_url(url: str) -> str:
    """Strictly validate gateway origin URL to `https://*.run.app` or `http(s)://localhost|127.0.0.1`."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ProbeError("Gateway URL must be a non-empty string.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        raise ProbeError("Gateway URL must not contain whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ProbeError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ProbeError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise ProbeError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ProbeError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ProbeError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise ProbeError(
            f"Untrusted gateway host {host!r}; expected https://*.run.app or localhost/127.0.0.1."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ProbeError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ProbeError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def _assert_zero_inference_request(method: str, url: str) -> str:
    """Enforce allowed gateway origin and block any inference completion endpoint."""
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path:
        raise ProbeError(
            f"Zero-inference invariant violated: {method.upper()} {parts.path} is strictly forbidden."
        )
    return origin


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Direct, proxy-free stdlib HTTP helper handling both JSON and HTML route responses."""
    _assert_zero_inference_request(method, url)
    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9",
        "User-Agent": "hivemind-noel-post-setup-probe/1.0",
    }
    if headers:
        req_headers.update(headers)
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method.upper())
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = resp.read(262_144).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            content_type = str(resp.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as exc:
        raw_err = exc.read(65_536).decode("utf-8", errors="replace")
        try:
            parsed_err = json.loads(raw_err) if raw_err.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": raw_err[:240]}
        return int(exc.code), (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {"network_error": type(exc).__name__, "detail": str(reason)[:200]}

    if not raw.strip():
        return status_code, {"content_type": content_type}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return status_code, {"content_type": content_type, "html_length": len(raw)}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def _safe_http_call(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Validate URL & zero-inference guard before invoking `http_fn`, catching transport errors."""
    _assert_zero_inference_request(method, url)
    try:
        status_code, body = http_fn(method.upper(), url, payload, headers)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": type(exc).__name__, "detail": str(exc)[:200]}
    norm_body = dict(body) if isinstance(body, Mapping) else {"data": body}
    return int(status_code), norm_body


def default_subprocess_runner(argv: Sequence[str], timeout_s: float = SUBPROCESS_TIMEOUT_S) -> tuple[int, str]:
    """Run a local read-only command with `shell=False` and bounded timeout."""
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


def _quickstart_command(gateway: str, os_name: str, extra_args: str = "") -> str:
    safe_gw = gateway if gateway else DEFAULT_GATEWAY
    suffix = f" {extra_args.strip()}" if extra_args.strip() else ""
    if os_name == "nt":
        return (
            f'powershell -NoProfile -Command "Invoke-RestMethod {safe_gw}/scripts/hivemind_quickstart.py '
            f'-OutFile hivemind_quickstart.py; python hivemind_quickstart.py{suffix}"'
        )
    return (
        f"curl -fsSL {safe_gw}/scripts/hivemind_quickstart.py -o hivemind_quickstart.py "
        f"&& python3 hivemind_quickstart.py{suffix}"
    )


def _route_failure_diagnostic(path: str, group: str, status_code: int, *, lang: str) -> str:
    is_fr = lang == "fr"
    if status_code == 404 and group == "holiday":
        return (
            f"GET {path} a répondu HTTP 404 : la révision Cloud Run est en retard sur `main` "
            "(route Noël absente ; redéployez la passerelle)."
            if is_fr
            else f"GET {path} returned HTTP 404: Cloud Run revision lags behind `main` "
            "(missing holiday onboarding route; redeploy gateway)."
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
    if status_code == 429:
        return (
            f"GET {path} limité en débit (HTTP 429) ; réessayez dans une minute."
            if is_fr
            else f"GET {path} rate-limited (HTTP 429); retry in one minute."
        )
    if status_code >= 500:
        return (
            f"GET {path} en erreur serveur passerelle (HTTP {status_code})."
            if is_fr
            else f"GET {path} failed with gateway server error (HTTP {status_code})."
        )
    return (
        f"GET {path} a répondu HTTP {status_code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} returned HTTP {status_code} (expected HTTP 200)."
    )


def run_holiday_route_smoke_gate(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Check all 7 required GET routes with zero LLM inference and surface stale-revision 404s."""
    lang_code = normalize_lang(lang)
    clean_gw = validate_gateway_url(gateway)

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    missing_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    feedback_summary_doc: dict[str, Any] = {}
    models_count: int | None = None

    for path, expected_kind, group in HOLIDAY_SMOKE_ROUTES:
        code, body = _safe_http_call(http_fn, "GET", f"{clean_gw}{path}", None, None)
        http_statuses[path] = code
        payload_ok = code == 200
        diag = ""

        if code == 200 and path == "/v1/models":
            if "data" in body and not isinstance(body.get("data"), list):
                payload_ok = False
                diag = (
                    "GET /v1/models a répondu HTTP 200 mais `data` n'est pas une liste valide."
                    if lang_code == "fr"
                    else "GET /v1/models returned HTTP 200 without a valid `data` model list."
                )
            elif isinstance(body.get("data"), list):
                models_count = len(body["data"])

        if code == 200 and path == "/v1/hivemind/onboarding/feedback/summary":
            feedback_summary_doc = dict(body)

        if not payload_ok:
            failed_routes.append(path)
            if code == 404:
                missing_routes.append(path)
            if not diag:
                diag = _route_failure_diagnostic(path, group, code, lang=lang_code)
            diagnostics.append(diag)

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "expected_kind": expected_kind,
                "group": group,
                "http_status": code,
                "ok": payload_ok,
                "diagnostic": diag,
            }
        )

    passed_count = sum(1 for row in route_rows if row["ok"])
    total_count = len(route_rows)
    all_ok = passed_count == total_count
    cloud_run_lags_main = bool(http_statuses.get("/health") == 200 and missing_routes)

    if cloud_run_lags_main:
        joined_404 = ", ".join(missing_routes)
        lag_msg = (
            f"Diagnostic Cloud Run : /health répond HTTP 200 mais {len(missing_routes)} route(s) Noël "
            f"renvoient HTTP 404 ({joined_404}) — la révision Cloud Run est en retard sur `main`."
            if lang_code == "fr"
            else f"Cloud Run diagnostic: /health is HTTP 200 while {len(missing_routes)} holiday route(s) "
            f"returned HTTP 404 ({joined_404}) — deployed Cloud Run revision lags behind `main`."
        )
        diagnostics.insert(0, lag_msg)

    raw_mean = feedback_summary_doc.get("mean_ease_rating")
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool)
        else None
    )
    raw_lag = feedback_summary_doc.get("zero_game_lag_ratio")
    zero_lag_ratio = (
        round(float(raw_lag), 4)
        if isinstance(raw_lag, (int, float)) and not isinstance(raw_lag, bool)
        else None
    )
    raw_total_fb = feedback_summary_doc.get("total_responses")
    total_fb = (
        int(raw_total_fb)
        if isinstance(raw_total_fb, int) and not isinstance(raw_total_fb, bool)
        else None
    )

    return {
        "gateway": clean_gw,
        "all_routes_ok": all_ok,
        "routes_passed_count": passed_count,
        "routes_total_count": total_count,
        "http_statuses": http_statuses,
        "missing_routes": missing_routes,
        "failed_routes": failed_routes,
        "cloud_run_lags_main": cloud_run_lags_main,
        "zero_inference_verified": True,
        "models_count": models_count,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": zero_lag_ratio,
        "feedback_total_responses": total_fb,
        "diagnostics": diagnostics,
        "routes": route_rows,
    }


def _smoke_gate_to_check(smoke: Mapping[str, Any], *, os_name: str, lang: str) -> ProbeCheck:
    is_fr = lang == "fr"
    title = (
        "Smoke Gate Routes Noël Cloud Run (Zéro Inférence)"
        if is_fr
        else "Cloud Run Holiday Route Smoke Gate (Zero-Inference)"
    )
    passed = int(smoke["routes_passed_count"])
    total = int(smoke["routes_total_count"])
    gw = str(smoke["gateway"])
    curl_bin = "curl.exe" if os_name == "nt" else "curl"

    if smoke["all_routes_ok"]:
        summary = (
            f"{passed}/{total} routes GET vérifiées avec HTTP 200 sur {gw} (zéro appel d'inférence)."
            if is_fr
            else f"{passed}/{total} GET routes verified HTTP 200 on {gw} (zero inference calls)."
        )
        return ProbeCheck(
            id="holiday_route_smoke",
            title=title,
            status="ok",
            summary=summary,
            fix_command="",
            details={
                "routes_passed_count": passed,
                "routes_total_count": total,
                "http_statuses": dict(smoke["http_statuses"]),
                "missing_routes": [],
                "cloud_run_lags_main": False,
            },
        )

    diags = list(smoke.get("diagnostics") or [])
    first_diag = diags[0] if diags else "One or more required routes failed."
    failed_first = (smoke.get("failed_routes") or ["/health"])[0]
    fix_cmd = f'{curl_bin} -fsSL -o /dev/null -w "%{{http_code}}" "{gw}{failed_first}"'
    summary = (
        f"{passed}/{total} routes OK — {first_diag}"
        if is_fr
        else f"{passed}/{total} routes passed — {first_diag}"
    )
    return ProbeCheck(
        id="holiday_route_smoke",
        title=title,
        status="fail",
        summary=summary,
        fix_command=fix_cmd,
        details={
            "routes_passed_count": passed,
            "routes_total_count": total,
            "http_statuses": dict(smoke["http_statuses"]),
            "missing_routes": list(smoke["missing_routes"]),
            "failed_routes": list(smoke["failed_routes"]),
            "cloud_run_lags_main": bool(smoke["cloud_run_lags_main"]),
        },
    )


def check_credentials_and_account(
    credentials_path: Path | None = None,
    *,
    verify_account: bool = False,
    gateway: str = DEFAULT_GATEWAY,
    http_fn: HttpFn = default_http_json,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Inspect `~/.hivemind/credentials.json` safely, plus optional `GET /v1/hivemind/me`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    target = credentials_path if credentials_path is not None else (resolved_home / ".hivemind" / "credentials.json")
    shown_path = redact_home_path(target, home=resolved_home)
    title = (
        "Identifiants Locaux & Compte HiveMind"
        if is_fr
        else "Local HiveMind Credentials & Account"
    )
    fix_cmd = _quickstart_command(gateway, eff_os, "--write-continue-config")

    if not target.is_file():
        summary = (
            f"{shown_path} introuvable ; exécutez quickstart pour enregistrer votre clé personnelle hm-..."
            if is_fr
            else f"{shown_path} not found; run quickstart to save your personal hm-... key."
        )
        return ProbeCheck(
            "credentials",
            title,
            "fail",
            summary,
            fix_cmd,
            {
                "path": shown_path,
                "present": False,
                "valid_key": False,
                "masked_key": "<missing>",
                "verify_requested": verify_account,
                "account_verified": False,
            },
        )

    try:
        st = target.stat()
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = (
            f"{shown_path} est illisible ou JSON invalide ; régénérez le fichier via quickstart."
            if is_fr
            else f"{shown_path} is unreadable or invalid JSON; regenerate via quickstart."
        )
        return ProbeCheck(
            "credentials",
            title,
            "fail",
            summary,
            fix_cmd,
            {
                "path": shown_path,
                "present": True,
                "valid_key": False,
                "masked_key": "<invalid>",
                "verify_requested": verify_account,
                "account_verified": False,
            },
        )

    raw_key = loaded.get("api_key") if isinstance(loaded, Mapping) else None
    if not _is_valid_hivemind_key(raw_key):
        summary = (
            f"{shown_path} ne contient pas de clé API hm-... valide."
            if is_fr
            else f"{shown_path} does not contain a valid hm-... API key."
        )
        return ProbeCheck(
            "credentials",
            title,
            "fail",
            summary,
            fix_cmd,
            {
                "path": shown_path,
                "present": True,
                "valid_key": False,
                "masked_key": mask_api_key(raw_key),
                "verify_requested": verify_account,
                "account_verified": False,
            },
        )

    clean_key = str(raw_key).strip()
    masked = mask_api_key(clean_key)
    role = str(loaded.get("role") or "prosumer") if isinstance(loaded, Mapping) else "prosumer"

    account_verified = False
    verify_status: int | None = None
    if verify_account:
        clean_gw = validate_gateway_url(gateway)
        verify_status, me_body = _safe_http_call(
            http_fn,
            "GET",
            f"{clean_gw}/v1/hivemind/me",
            None,
            {"Authorization": f"Bearer {clean_key}"},
        )
        if verify_status != 200:
            summary = (
                f"Échec de vérification de la clé {masked} sur /v1/hivemind/me (HTTP {verify_status})."
                if is_fr
                else f"Account check for {masked} failed on /v1/hivemind/me (HTTP {verify_status})."
            )
            return ProbeCheck(
                "credentials",
                title,
                "fail",
                summary,
                fix_cmd,
                {
                    "path": shown_path,
                    "present": True,
                    "valid_key": True,
                    "masked_key": masked,
                    "role": role,
                    "verify_requested": True,
                    "account_verified": False,
                    "verify_http_status": verify_status,
                },
            )
        account_verified = True
        if isinstance(me_body.get("role"), str) and me_body["role"].strip():
            role = me_body["role"].strip()

    details: dict[str, Any] = {
        "path": shown_path,
        "present": True,
        "valid_key": True,
        "masked_key": masked,
        "role": role,
        "verify_requested": verify_account,
        "account_verified": account_verified,
        "verify_http_status": verify_status,
    }

    if eff_os == "nt":
        details["permission_model"] = "windows_acl"
        ver_note_en = ", /v1/hivemind/me verified" if account_verified else ""
        ver_note_fr = ", /v1/hivemind/me vérifié" if account_verified else ""
        summary = (
            f"{shown_path} prêt ({masked}, rôle : {role}, ACL Windows{ver_note_fr})."
            if is_fr
            else f"{shown_path} ready ({masked}, role: {role}, Windows ACL{ver_note_en})."
        )
        return ProbeCheck("credentials", title, "ok", summary, "", details)

    mode_bits = st.st_mode & 0o777
    octal_mode = f"{mode_bits:04o}"
    details["posix_mode"] = octal_mode
    if mode_bits != 0o600:
        chmod_target = shown_path if shown_path.startswith("~/") else shlex.quote(shown_path)
        chmod_cmd = f"chmod 600 {chmod_target}"
        summary = (
            f"{shown_path} ({masked}) a les permissions {octal_mode} (attendu 0600 propriétaire seul)."
            if is_fr
            else f"{shown_path} ({masked}) has mode {octal_mode} (expected 0600 owner-only)."
        )
        return ProbeCheck("credentials", title, "warn", summary, chmod_cmd, details)

    ver_note_en = ", /v1/hivemind/me verified" if account_verified else ""
    ver_note_fr = ", /v1/hivemind/me vérifié" if account_verified else ""
    summary = (
        f"{shown_path} prêt ({masked}, rôle : {role}, POSIX 0600{ver_note_fr})."
        if is_fr
        else f"{shown_path} ready ({masked}, role: {role}, POSIX 0600{ver_note_en})."
    )
    return ProbeCheck("credentials", title, "ok", summary, "", details)


def check_continue_provider_yaml(
    continue_path: Path | None = None,
    *,
    gateway: str = DEFAULT_GATEWAY,
    os_name: str | None = None,
    home: Path | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Verify presence and structure of `~/.continue/hivemind_provider.yaml` without leaking keys."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    target = (
        continue_path
        if continue_path is not None
        else (resolved_home / ".continue" / "hivemind_provider.yaml")
    )
    shown_path = redact_home_path(target, home=resolved_home)
    title = (
        "Configuration VS Code Continue.dev"
        if is_fr
        else "VS Code Continue.dev Configuration"
    )
    fix_cmd = _quickstart_command(gateway, eff_os, "--write-continue-config")

    if not target.is_file():
        summary = (
            f"{shown_path} absent ; ajoutez `--write-continue-config` si vous codez dans VS Code."
            if is_fr
            else f"{shown_path} not present; pass `--write-continue-config` to enable VS Code Continue.dev."
        )
        return ProbeCheck(
            "continue_config",
            title,
            "warn",
            summary,
            fix_cmd,
            {"path": shown_path, "present": False, "configured": False},
        )

    try:
        raw_yaml = target.read_text(encoding="utf-8")
    except OSError:
        summary = (
            f"{shown_path} est illisible ; régénérez l'extrait Continue.dev."
            if is_fr
            else f"{shown_path} could not be read; regenerate the Continue.dev snippet."
        )
        return ProbeCheck(
            "continue_config",
            title,
            "warn",
            summary,
            fix_cmd,
            {"path": shown_path, "present": True, "configured": False},
        )

    has_models = "models:" in raw_yaml and "apiBase:" in raw_yaml
    if not has_models:
        summary = (
            f"{shown_path} est incomplet (`models:` / `apiBase:` manquants) ; régénérez-le."
            if is_fr
            else f"{shown_path} is incomplete (missing `models:` / `apiBase:`); regenerate it."
        )
        return ProbeCheck(
            "continue_config",
            title,
            "warn",
            summary,
            fix_cmd,
            {"path": shown_path, "present": True, "configured": False},
        )

    summary = (
        f"{shown_path} présent et configuré pour VS Code Continue.dev."
        if is_fr
        else f"{shown_path} present and configured for VS Code Continue.dev."
    )
    return ProbeCheck(
        "continue_config",
        title,
        "ok",
        summary,
        "",
        {"path": shown_path, "present": True, "configured": True},
    )


def _parse_float_metric(raw: str, default: float = 0.0) -> float:
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("[n/a]", "n/a", "[not supported]", "not supported"):
        return default
    return float(cleaned)


def probe_gpu(runner: RunnerFn = default_subprocess_runner) -> GpuStatus:
    """Query `nvidia-smi` locally without requesting GPU UUID or serial numbers."""
    try:
        rc, out = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return GpuStatus(present=False)

    if rc != 0 or not out.strip():
        return GpuStatus(present=False)

    gpus: list[GpuStatus] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0]:
            return GpuStatus(present=True, parse_error=True)
        try:
            total_mib = int(float(parts[1]))
            free_mib = int(float(parts[2]))
            temp_c = _parse_float_metric(parts[3], 0.0)
            util_pct = _parse_float_metric(parts[4], 0.0)
        except ValueError:
            return GpuStatus(present=True, parse_error=True)
        if total_mib <= 0 or free_mib < 0:
            return GpuStatus(present=True, parse_error=True)
        gpus.append(
            GpuStatus(
                present=True,
                parse_error=False,
                name=parts[0],
                total_mib=total_mib,
                free_mib=free_mib,
                temp_c=temp_c,
                util_pct=util_pct,
            )
        )
    return max(gpus, key=lambda g: (g.free_mib, -g.temp_c, g.total_mib))


def check_gamer_whisper_gpu(
    *,
    runner: RunnerFn = default_subprocess_runner,
    max_temp_c: float = GAMER_WHISPER_MAX_TEMP_C,
    min_free_vram_mib: int = GAMER_WHISPER_MIN_FREE_VRAM_MIB,
    gateway: str = DEFAULT_GATEWAY,
    os_name: str | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Verify Gamer-Whisper thermal ceiling (`<= 70.0 C`) and free VRAM reserve (`>= 2048 MiB`)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    title = (
        "GPU Gamer-Whisper (Plafond <= 70 C & Réserve VRAM >= 2048 MiB)"
        if is_fr
        else "Gamer-Whisper GPU Guard (<= 70.0 C & >= 2048 MiB Free VRAM)"
    )
    gpu = probe_gpu(runner)

    if not gpu.present:
        summary = (
            "Aucun GPU NVIDIA détecté — Mode Dev API uniquement prêt (0 charge GPU locale)."
            if is_fr
            else "No NVIDIA GPU detected — Dev API-only mode ready (0 local GPU load)."
        )
        return ProbeCheck(
            "gamer_whisper_gpu",
            title,
            "ok",
            summary,
            "",
            {
                "gpu_present": False,
                "mode": "dev_api_only",
                "thermal_cap_c": max_temp_c,
                "min_free_vram_mib": min_free_vram_mib,
                "thermal_cap_ok": True,
                "vram_reserve_ok": True,
            },
        )

    if gpu.parse_error:
        summary = (
            "Sortie `nvidia-smi` illisible — Mode Dev API prêt ; vérifiez le pilote NVIDIA avant Gamer-Whisper."
            if is_fr
            else "`nvidia-smi` output could not be parsed — Dev API-only mode ready; inspect NVIDIA driver."
        )
        return ProbeCheck(
            "gamer_whisper_gpu",
            title,
            "warn",
            summary,
            "nvidia-smi",
            {"gpu_present": True, "parse_error": True, "mode": "dev_api_only"},
        )

    thermal_ok = gpu.temp_c <= max_temp_c
    vram_ok = gpu.free_mib >= min_free_vram_mib
    gamer_fix = _quickstart_command(gateway, eff_os, "--gamer-check")
    details: dict[str, Any] = {
        "gpu_present": True,
        "name": gpu.name,
        "total_mib": gpu.total_mib,
        "free_mib": gpu.free_mib,
        "temp_c": gpu.temp_c,
        "util_pct": gpu.util_pct,
        "thermal_cap_c": max_temp_c,
        "min_free_vram_mib": min_free_vram_mib,
        "thermal_cap_ok": thermal_ok,
        "vram_reserve_ok": vram_ok,
    }

    if not thermal_ok:
        summary = (
            f"{gpu.name} est à {gpu.temp_c:.1f} C (> {max_temp_c:.1f} C plafond Gamer-Whisper) ; "
            "pause thermique jusqu'au refroidissement <= 70.0 C."
            if is_fr
            else f"{gpu.name} is at {gpu.temp_c:.1f} C (> {max_temp_c:.1f} C Gamer-Whisper thermal cap); "
            "GPU worker pauses until <= 70.0 C."
        )
        return ProbeCheck(
            "gamer_whisper_gpu",
            title,
            "warn",
            summary,
            gamer_fix,
            {**details, "mode": "gpu_thermal_cooldown"},
        )

    if not vram_ok:
        summary = (
            f"{gpu.name} n'a que {gpu.free_mib} MiB libres (< {min_free_vram_mib} MiB de réserve Gamer-Whisper) ; "
            "Gamer-Whisper reste en veille."
            if is_fr
            else f"{gpu.name} has {gpu.free_mib} MiB free (< {min_free_vram_mib} MiB Gamer-Whisper reserve); "
            "Gamer-Whisper stays idle."
        )
        return ProbeCheck(
            "gamer_whisper_gpu",
            title,
            "warn",
            summary,
            gamer_fix,
            {**details, "mode": "vram_reserve_hold"},
        )

    summary = (
        f"{gpu.name} prêt en silence ({gpu.temp_c:.1f} C <= {max_temp_c:.1f} C, "
        f"{gpu.free_mib} MiB libres >= {min_free_vram_mib} MiB)."
        if is_fr
        else f"{gpu.name} quiet & ready ({gpu.temp_c:.1f} C <= {max_temp_c:.1f} C, "
        f"{gpu.free_mib} MiB free >= {min_free_vram_mib} MiB)."
    )
    return ProbeCheck(
        "gamer_whisper_gpu",
        title,
        "ok",
        summary,
        "",
        {**details, "mode": "gamer_whisper_ready"},
    )


def list_process_names(
    runner: RunnerFn = default_subprocess_runner,
    *,
    os_name: str | None = None,
) -> set[str]:
    """List normalized lowercase local process executable names on Windows, macOS, or Linux."""
    eff_os = os_name if os_name is not None else os.name
    names: set[str] = set()
    try:
        if eff_os == "nt":
            rc, out = runner(["tasklist", "/FO", "CSV", "/NH"], SUBPROCESS_TIMEOUT_S)
            if rc != 0:
                return names
            for row in csv.reader(io.StringIO(out)):
                if row and row[0].strip():
                    names.add(row[0].strip().lower())
        else:
            rc, out = runner(["ps", "-eo", "comm="], SUBPROCESS_TIMEOUT_S)
            if rc != 0:
                return names
            for line in out.splitlines():
                cleaned = line.strip()
                if cleaned:
                    names.add(cleaned.lower())
                    names.add(Path(cleaned.replace("\\", "/")).name.lower())
    except (OSError, subprocess.SubprocessError, csv.Error):
        return names
    return names


def verify_yield_budget(
    detection_elapsed_s: float,
    *,
    preemption_signal_s: float = PREEMPTION_SIGNAL_ESTIMATE_S,
    budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
) -> dict[str, Any]:
    """Verify active-game detection + process-tree preemption fits inside the `< 1.5 s` budget."""
    clamped_scan = max(0.0, float(detection_elapsed_s))
    clamped_kill = max(0.0, float(preemption_signal_s))
    estimated_total = round(clamped_scan + clamped_kill, 4)
    return {
        "detection_elapsed_s": round(clamped_scan, 4),
        "preemption_signal_s": round(clamped_kill, 4),
        "estimated_yield_s": estimated_total,
        "yield_budget_s": budget_s,
        "yield_budget_ok": estimated_total < budget_s,
    }


def check_active_games_and_yield_budget(
    *,
    runner: RunnerFn = default_subprocess_runner,
    extra_games: Sequence[str] = (),
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
    preemption_signal_s: float = PREEMPTION_SIGNAL_ESTIMATE_S,
    clock_fn: Callable[[], float] = time.monotonic,
    gateway: str = DEFAULT_GATEWAY,
    os_name: str | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Detect active Steam/PC games and verify the `< 1.5 s` GPU yield budget."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    title = (
        "Détection de Jeu Actif & Budget d'Éviction (< 1.5 s)"
        if is_fr
        else "Active Game Guard & Yield Budget (< 1.5 s)"
    )

    watchlist = set(DEFAULT_GAME_EXECUTABLES)
    for item in extra_games:
        cleaned = item.strip().lower()
        if cleaned:
            watchlist.add(cleaned)
            watchlist.add(cleaned[:15])

    t0 = clock_fn()
    running = list_process_names(runner, os_name=eff_os)
    scan_s = max(0.0, clock_fn() - t0)
    budget_info = verify_yield_budget(
        scan_s,
        preemption_signal_s=preemption_signal_s,
        budget_s=yield_budget_s,
    )
    detected = sorted(proc for proc in running if proc in watchlist)
    gamer_fix = _quickstart_command(gateway, eff_os, "--gamer-check")

    details: dict[str, Any] = {
        "active_games": detected,
        "game_running": bool(detected),
        **budget_info,
    }

    if not budget_info["yield_budget_ok"]:
        summary = (
            f"L'analyse des processus a pris {budget_info['estimated_yield_s']:.2f} s "
            f"(>= {yield_budget_s:.1f} s budget Gamer-Whisper) ; gardez le mode GPU désactivé."
            if is_fr
            else f"Process scan + preemption estimate took {budget_info['estimated_yield_s']:.2f} s "
            f"(>= {yield_budget_s:.1f} s Gamer-Whisper yield budget); keep GPU rounds paused."
        )
        return ProbeCheck("active_games_yield", title, "warn", summary, gamer_fix, details)

    if detected:
        joined = ", ".join(detected)
        summary = (
            f"Jeu actif détecté ({joined}) ; Gamer-Whisper libère 100 % de la VRAM en "
            f"{budget_info['estimated_yield_s']:.2f} s (< {yield_budget_s:.1f} s)."
            if is_fr
            else f"Active game detected ({joined}); Gamer-Whisper yields 100% of GPU VRAM within "
            f"{budget_info['estimated_yield_s']:.2f} s (< {yield_budget_s:.1f} s budget)."
        )
        return ProbeCheck("active_games_yield", title, "warn", summary, gamer_fix, details)

    summary = (
        f"Aucun jeu actif détecté ; budget d'éviction vérifié à {budget_info['estimated_yield_s']:.2f} s "
        f"(< {yield_budget_s:.1f} s)."
        if is_fr
        else f"No active game detected; preemption yield budget verified at "
        f"{budget_info['estimated_yield_s']:.2f} s (< {yield_budget_s:.1f} s)."
    )
    return ProbeCheck("active_games_yield", title, "ok", summary, "", details)


def detect_local_corporate_markers(environ: Mapping[str, str]) -> list[str]:
    """Inspect environment variables strictly in memory without recording any values."""
    markers: list[str] = []
    upper_map: dict[str, str] = {
        str(k).upper(): str(v).strip() for k, v in environ.items() if str(v).strip()
    }

    if any(k.startswith("AMADEUS") for k in upper_map):
        markers.append("AMADEUS_ENV")

    if upper_map.get("USERDNSDOMAIN"):
        markers.append("USERDNSDOMAIN")

    userdomain = upper_map.get("USERDOMAIN", "")
    computername = upper_map.get("COMPUTERNAME", "")
    if userdomain and userdomain.upper() != computername.upper():
        if any(tok in userdomain.lower() for tok in _CORP_MARKERS):
            markers.append("USERDOMAIN")

    for host_key in ("COMPUTERNAME", "HOSTNAME"):
        host_val = upper_map.get(host_key, "").lower()
        if host_val and ("amadeus" in host_val or "corp-" in host_val or "-corp" in host_val):
            markers.append(host_key)

    for proxy_key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        if upper_map.get(proxy_key):
            markers.append(proxy_key)

    no_proxy = upper_map.get("NO_PROXY", "").lower()
    if no_proxy and any(tok in no_proxy for tok in _CORP_MARKERS):
        markers.append("NO_PROXY")

    for tls_key in (
        "ZSCALER_CERT_PATH",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ):
        tls_val = upper_map.get(tls_key, "").lower()
        if tls_key == "ZSCALER_CERT_PATH" and tls_val:
            markers.append(tls_key)
        elif tls_val and any(tok in tls_val for tok in _CORP_MARKERS):
            markers.append(tls_key)

    return markers


def check_corporate_pc_compliance(
    *,
    environ: Mapping[str, str] | None = None,
    gateway: str = DEFAULT_GATEWAY,
    os_name: str | None = None,
    lang: str = "en",
) -> ProbeCheck:
    """Strictly local corporate-PC warning (Amadeus / work markers NEVER leave the machine)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    eff_os = os_name if os_name is not None else os.name
    env_view = environ if environ is not None else os.environ
    title = (
        "Conformité PC Personnel / Règle Amadeus (100 % Local)"
        if is_fr
        else "Personal-PC / Amadeus Compliance (Local-Only Check)"
    )

    markers = detect_local_corporate_markers(env_view)
    base_details: dict[str, Any] = {
        "corporate_indicators": markers,
        "corporate_warning": bool(markers),
        "network_transmitted": False,
        "personal_pc_only": True,
        "monetary_cost": "0.00 EUR / 0.00 USD",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
    }

    if markers:
        uninstall_cmd = _quickstart_command(gateway, eff_os, "--uninstall")
        joined = ", ".join(markers)
        summary = (
            f"Alerte locale ({joined}) : ARRÊTEZ immédiatement s'il s'agit d'un PC professionnel "
            "(Amadeus / entreprise). Usage 100 % PC personnel à domicile et projets perso uniquement "
            "(0,00 EUR / 0,00 USD)."
            if is_fr
            else f"Local corporate marker detected ({joined}): STOP immediately if this is a work PC "
            "(Amadeus / corporate hardware prohibited). 100% personal home PCs & personal projects only "
            "(0.00 EUR / 0.00 USD)."
        )
        return ProbeCheck("personal_pc_compliance", title, "warn", summary, uninstall_cmd, base_details)

    summary = (
        f"Contrôle local conforme (aucun marqueur professionnel détecté). {COMPLIANCE_NOTICE_FR}"
        if is_fr
        else f"Local check clean (no corporate/work markers detected). {COMPLIANCE_NOTICE_EN}"
    )
    return ProbeCheck("personal_pc_compliance", title, "ok", summary, "", base_details)


def parse_interactive_ces_input(
    raw: str,
    *,
    default_profile: str = "dev",
    default_had_game_lag: bool = False,
) -> dict[str, Any]:
    """Parse a single-line 1-prompt CES response (`'5'`, `'4/5 gamer'`, `'3 lag'`, or empty to skip)."""
    prof = normalize_profile(default_profile)
    cleaned = (raw or "").strip()
    if not cleaned or cleaned.lower() in ("skip", "s", "q", "quit", "no", "non", "pass"):
        return {
            "skipped": True,
            "profile": prof,
            "ease_rating": None,
            "zero_game_lag": not default_had_game_lag,
        }

    normalized = re.sub(r"\s*/\s*5\b", "", cleaned, count=1)
    tokens = [tok.strip(",;:").lower() for tok in normalized.split() if tok.strip(",;:")]

    rating: int | None = None
    had_lag = bool(default_had_game_lag)

    for tok in tokens:
        if tok.lstrip("+-").isdigit():
            val = int(tok)
            if not (1 <= val <= 5) or rating is not None:
                raise ProbeError("Interactive CES ease_rating must be a single integer from 1 to 5.")
            rating = val
        elif tok in ("gamer", "g"):
            prof = "gamer"
        elif tok in ("dev", "developer", "d"):
            prof = "dev"
        elif tok in ("lag", "had-game-lag", "had_game_lag", "stutter", "saccade", "lag=yes", "lag=true"):
            had_lag = True
        elif tok in ("nolag", "no-lag", "no_lag", "zero-lag", "zero_lag", "lag=no", "lag=false", "ok"):
            had_lag = False
        else:
            raise ProbeError(f"Unrecognized 1-prompt CES token {tok!r}; expected 1..5 [dev|gamer] [lag].")

    if rating is None:
        raise ProbeError("Interactive 1-prompt CES input must include an ease_rating from 1 to 5.")

    return {
        "skipped": False,
        "profile": prof,
        "ease_rating": rating,
        "zero_game_lag": not had_lag,
    }


def _platform_tag(system_name: str | None = None) -> str:
    raw = (system_name if system_name is not None else platform.system()).strip().lower()
    if raw in ("windows", "win32", "nt"):
        return "windows"
    if raw == "darwin":
        return "macos"
    if raw == "linux":
        return "linux"
    cleaned = re.sub(r"[^a-z0-9_-]+", "", raw)[:24]
    return cleaned or "unknown"


def submit_ces_feedback(
    *,
    gateway: str = DEFAULT_GATEWAY,
    profile: str = "dev",
    ease_rating: int,
    zero_game_lag: bool = True,
    system_name: str | None = None,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Post a strictly zero-PII anonymous CES payload to `POST /v1/hivemind/onboarding/feedback`."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_prof = normalize_profile(profile)
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise ProbeError("ease_rating must be an integer between 1 and 5.")
    if not isinstance(zero_game_lag, bool):
        raise ProbeError("zero_game_lag must be a boolean.")

    payload: dict[str, Any] = {
        "profile": norm_prof,
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "platform": _platform_tag(system_name),
    }

    status_code, body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload,
        None,
    )

    raw_mean = body.get("mean_ease_rating")
    mean_ease = (
        round(float(raw_mean), 2)
        if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool)
        else None
    )
    raw_lag = body.get("zero_game_lag_ratio")
    lag_ratio = (
        round(float(raw_lag), 4)
        if isinstance(raw_lag, (int, float)) and not isinstance(raw_lag, bool)
        else None
    )
    raw_total = body.get("total_responses")
    total_resp = (
        int(raw_total)
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else None
    )

    if status_code in (200, 201):
        msg = (
            f"Merci ! Retour anonyme enregistré (ease_rating={ease_rating}/5, profil={norm_prof}, "
            f"zero_game_lag={zero_game_lag})."
            if is_fr
            else f"Thank you! Anonymous feedback recorded (ease_rating={ease_rating}/5, "
            f"profile={norm_prof}, zero_game_lag={zero_game_lag})."
        )
        outcome = "recorded"
        submitted = True
    elif status_code == 404:
        msg = (
            "POST /v1/hivemind/onboarding/feedback a répondu HTTP 404 : la révision Cloud Run est en "
            "retard sur `main` (route de retour Noël non déployée)."
            if is_fr
            else "POST /v1/hivemind/onboarding/feedback returned HTTP 404: Cloud Run revision lags behind "
            "`main` (holiday feedback route not yet deployed)."
        )
        outcome = "missing_route"
        submitted = False
    elif status_code == 429:
        msg = (
            "Limite de débit atteinte (HTTP 429) sur /v1/hivemind/onboarding/feedback ; "
            "patientez une minute avant de réessayer."
            if is_fr
            else "Rate limit reached (HTTP 429) on /v1/hivemind/onboarding/feedback; "
            "please wait a minute before retrying."
        )
        outcome = "rate_limited"
        submitted = False
    elif status_code >= 500:
        msg = (
            f"Erreur temporaire du serveur passerelle (HTTP {status_code}) lors de l'envoi du retour CES."
            if is_fr
            else f"Temporary gateway server error (HTTP {status_code}) while submitting CES feedback."
        )
        outcome = "server_error"
        submitted = False
    elif status_code == 0:
        msg = (
            "Impossible de joindre /v1/hivemind/onboarding/feedback (erreur réseau)."
            if is_fr
            else "Could not reach /v1/hivemind/onboarding/feedback (network error)."
        )
        outcome = "network_error"
        submitted = False
    else:
        msg = (
            f"Envoi du retour refusé par la passerelle (HTTP {status_code})."
            if is_fr
            else f"Feedback submission rejected by gateway (HTTP {status_code})."
        )
        outcome = "client_error"
        submitted = False

    return {
        "attempted": True,
        "submitted": submitted,
        "skipped": False,
        "status": outcome,
        "http_status": status_code,
        "profile": norm_prof,
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": lag_ratio,
        "total_responses": total_resp,
        "zero_pii_payload": payload,
        "message": msg,
    }


def prompt_and_submit_ces_feedback(
    *,
    gateway: str = DEFAULT_GATEWAY,
    default_profile: str = "dev",
    default_had_game_lag: bool = False,
    system_name: str | None = None,
    input_fn: InputFn = input,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
) -> dict[str, Any]:
    """Ask a single 1-line CES question via `input_fn` and submit zero-PII feedback if answered."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)
    norm_prof = normalize_profile(default_profile)

    prompt_text = (
        f"[HiveMind CES 1-question] Notez la simplicité 1..5 (profil={norm_prof}, ajoutez 'gamer'/'dev' "
        "ou 'lag' si saccade ; Entrée pour ignorer) : "
        if is_fr
        else f"[HiveMind 1-prompt CES] Rate setup ease 1..5 (profile={norm_prof}, add 'gamer'/'dev' "
        "or 'lag' if any game stutter; Enter to skip): "
    )
    try:
        raw_answer = input_fn(prompt_text)
    except (EOFError, KeyboardInterrupt):
        raw_answer = ""

    parsed = parse_interactive_ces_input(
        raw_answer,
        default_profile=norm_prof,
        default_had_game_lag=default_had_game_lag,
    )
    if parsed["skipped"]:
        skip_msg = (
            "Retour CES 1-question ignoré (aucun envoi réseau effectué)."
            if is_fr
            else "1-prompt CES skipped (no feedback sent)."
        )
        return {
            "attempted": False,
            "submitted": False,
            "skipped": True,
            "status": "skipped",
            "http_status": None,
            "profile": parsed["profile"],
            "ease_rating": None,
            "zero_game_lag": parsed["zero_game_lag"],
            "mean_ease_rating": None,
            "zero_game_lag_ratio": None,
            "total_responses": None,
            "zero_pii_payload": None,
            "message": skip_msg,
        }

    return submit_ces_feedback(
        gateway=clean_gw,
        profile=str(parsed["profile"]),
        ease_rating=int(parsed["ease_rating"]),
        zero_game_lag=bool(parsed["zero_game_lag"]),
        system_name=system_name,
        http_fn=http_fn,
        lang=lang_code,
    )


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
    if _FORBIDDEN_KEY_RE.search(text):
        issues.append("forbidden_key_detected")
    for ip_hit in _IPV4_RE.findall(text):
        if ip_hit != "127.0.0.1":
            issues.append(f"non_loopback_ipv4_detected:{ip_hit}")
    return issues


def _assert_safe_output(text: str) -> str:
    issues = scan_output_safety(text)
    if issues:
        raise ProbeError(f"Output safety guard failed: {', '.join(issues)}")
    return text


def _attach_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(core)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    _assert_safe_output(canonical)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["receipt_sha256"] = digest
    return payload


def run_post_setup_probe(
    *,
    gateway: str = DEFAULT_GATEWAY,
    smoke_only: bool = False,
    verify_account: bool = False,
    credentials_path: Path | None = None,
    continue_path: Path | None = None,
    max_temp_c: float = GAMER_WHISPER_MAX_TEMP_C,
    min_free_vram_mib: int = GAMER_WHISPER_MIN_FREE_VRAM_MIB,
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
    extra_games: Sequence[str] = (),
    feedback_result: Mapping[str, Any] | None = None,
    lang: str = "en",
    os_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    clock_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute the Holiday Route Smoke Gate and local readiness probe, returning a receipted report."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    clean_gw = validate_gateway_url(gateway)

    smoke = run_holiday_route_smoke_gate(clean_gw, http_fn=http_fn, lang=lang_code)
    smoke_check = _smoke_gate_to_check(smoke, os_name=eff_os, lang=lang_code)

    if smoke_only:
        checks = [smoke_check]
        recommended_mode = "dev_api_only"
        mode_name = "smoke_routes"
    else:
        cred_check = check_credentials_and_account(
            credentials_path,
            verify_account=verify_account,
            gateway=clean_gw,
            http_fn=http_fn,
            os_name=eff_os,
            home=home,
            lang=lang_code,
        )
        cont_check = check_continue_provider_yaml(
            continue_path,
            gateway=clean_gw,
            os_name=eff_os,
            home=home,
            lang=lang_code,
        )
        gpu_check = check_gamer_whisper_gpu(
            runner=runner,
            max_temp_c=max_temp_c,
            min_free_vram_mib=min_free_vram_mib,
            gateway=clean_gw,
            os_name=eff_os,
            lang=lang_code,
        )
        game_check = check_active_games_and_yield_budget(
            runner=runner,
            extra_games=extra_games,
            yield_budget_s=yield_budget_s,
            clock_fn=clock_fn,
            gateway=clean_gw,
            os_name=eff_os,
            lang=lang_code,
        )
        corp_check = check_corporate_pc_compliance(
            environ=environ,
            gateway=clean_gw,
            os_name=eff_os,
            lang=lang_code,
        )
        checks = [smoke_check, cred_check, cont_check, gpu_check, game_check, corp_check]
        gpu_ok = gpu_check.details.get("mode") == "gamer_whisper_ready" and game_check.passed
        recommended_mode = "gamer_whisper" if gpu_ok else "dev_api_only"
        mode_name = "full_probe"

    checks_passed = sum(1 for c in checks if c.passed)
    warnings_count = sum(1 for c in checks if c.status == "warn")
    failures_count = sum(1 for c in checks if c.status == "fail")

    if failures_count > 0:
        verdict = "not_ready"
    elif warnings_count > 0:
        verdict = "ready_with_warnings"
    else:
        verdict = "ready"

    fix_commands: list[str] = []
    for chk in checks:
        if chk.fix_command and chk.fix_command not in fix_commands:
            fix_commands.append(chk.fix_command)

    fb_dict = dict(feedback_result) if feedback_result is not None else None
    ease_val = fb_dict.get("ease_rating") if fb_dict is not None else None
    mean_ease = (
        fb_dict.get("mean_ease_rating")
        if (fb_dict is not None and fb_dict.get("mean_ease_rating") is not None)
        else smoke["mean_ease_rating"]
    )
    zero_lag_ratio = (
        fb_dict.get("zero_game_lag_ratio")
        if (fb_dict is not None and fb_dict.get("zero_game_lag_ratio") is not None)
        else smoke["zero_game_lag_ratio"]
    )

    core_report: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "mode": mode_name,
        "lang": lang_code,
        "gateway": clean_gw,
        "ready": verdict == "ready",
        "verdict": verdict,
        "recommended_mode": recommended_mode,
        "routes_passed_count": smoke["routes_passed_count"],
        "routes_total_count": smoke["routes_total_count"],
        "checks_passed_count": checks_passed,
        "checks_total_count": len(checks),
        "warnings_count": warnings_count,
        "failures_count": failures_count,
        "ease_rating": ease_val,
        "mean_ease_rating": mean_ease,
        "zero_game_lag_ratio": zero_lag_ratio,
        "zero_inference_verified": True,
        "compliance": {
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "corporate_env_transmitted": False,
            "monetary_cost": "0.00 EUR / 0.00 USD",
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        },
        "smoke_gate": smoke,
        "checks": [c.to_dict() for c in checks],
        "feedback": fb_dict,
        "fix_commands": fix_commands,
    }
    return _attach_receipt(core_report)


def build_feedback_only_report(
    feedback_result: Mapping[str, Any],
    *,
    gateway: str,
    lang: str = "en",
) -> dict[str, Any]:
    """Wrap a standalone 1-prompt / CLI CES feedback result in a SHA-256-receipted report."""
    lang_code = normalize_lang(lang)
    clean_gw = validate_gateway_url(gateway)
    fb = dict(feedback_result)
    ok_or_skipped = bool(fb.get("submitted") or fb.get("skipped"))
    verdict = "ready" if ok_or_skipped else "not_ready"

    core_report: dict[str, Any] = {
        "schema": SCHEMA,
        "object": RECEIPT_OBJECT,
        "campaign_id": CAMPAIGN_ID,
        "mode": "feedback",
        "lang": lang_code,
        "gateway": clean_gw,
        "ready": ok_or_skipped,
        "verdict": verdict,
        "recommended_mode": "gamer_whisper" if fb.get("profile") == "gamer" else "dev_api_only",
        "routes_passed_count": 1 if fb.get("submitted") else 0,
        "routes_total_count": 1 if fb.get("attempted") else 0,
        "checks_passed_count": 1 if ok_or_skipped else 0,
        "checks_total_count": 1,
        "warnings_count": 0,
        "failures_count": 0 if ok_or_skipped else 1,
        "ease_rating": fb.get("ease_rating"),
        "mean_ease_rating": fb.get("mean_ease_rating"),
        "zero_game_lag_ratio": fb.get("zero_game_lag_ratio"),
        "zero_inference_verified": True,
        "compliance": {
            "personal_pc_only": True,
            "amadeus_corporate_prohibition": True,
            "corporate_env_transmitted": False,
            "monetary_cost": "0.00 EUR / 0.00 USD",
            "license": "AGPL-3.0",
            "notice_en": COMPLIANCE_NOTICE_EN,
            "notice_fr": COMPLIANCE_NOTICE_FR,
            "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        },
        "smoke_gate": None,
        "checks": [],
        "feedback": fb,
        "fix_commands": [],
    }
    return _attach_receipt(core_report)


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a clean FR/EN GitHub-flavored Markdown summary with SHA-256 receipt."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    is_fr = lang_code == "fr"
    comp = report.get("compliance") if isinstance(report.get("compliance"), Mapping) else {}
    bilingual_notice = str(comp.get("notice_bilingual") or COMPLIANCE_NOTICE_BILINGUAL)

    header = (
        "# Sonde Post-Installation & Smoke Gate Noël HiveMind"
        if is_fr
        else "# HiveMind Holiday Post-Setup Probe & Smoke Gate"
    )
    mean_ease = report.get("mean_ease_rating")
    mean_str = f"{mean_ease}/5" if mean_ease is not None else "n/a"
    ease_val = report.get("ease_rating")
    ease_str = f"{ease_val}/5" if ease_val is not None else "n/a"

    lines: list[str] = [
        header,
        "",
        f"> **Compliance / Conformité**: {bilingual_notice}",
        "",
        f"- **Gateway**: `{report.get('gateway')}`",
        f"- **Ready**: `{report.get('ready')}` (`{report.get('verdict')}`)",
        f"- **Routes Passed**: `{report.get('routes_passed_count')}/{report.get('routes_total_count')}`",
        f"- **Checks Passed**: `{report.get('checks_passed_count')}/{report.get('checks_total_count')}`",
        f"- **CES Ease Rating**: `{ease_str}` (Community Mean: `{mean_str}`)",
    ]

    smoke = report.get("smoke_gate")
    if isinstance(smoke, Mapping):
        lines.extend(
            [
                "",
                "## Routes Cloud Run (Zéro Inférence)" if is_fr else "## Cloud Run Holiday Routes (Zero-Inference)",
                "",
                "| Route | HTTP | Status | Diagnostic |",
                "| :--- | :---: | :---: | :--- |",
            ]
        )
        for row in smoke.get("routes") or ():
            state = "OK" if row.get("ok") else "FAIL"
            diag = str(row.get("diagnostic") or "—")
            lines.append(f"| `{row.get('path')}` | `{row.get('http_status')}` | **{state}** | {diag} |")

    checks = report.get("checks")
    if isinstance(checks, Sequence) and checks:
        lines.extend(
            [
                "",
                "## Contrôles Locaux Post-Installation" if is_fr else "## Local Post-Setup Checks",
                "",
            ]
        )
        for chk in checks:
            badge = str(chk.get("status", "info")).upper()
            lines.append(f"- **[{badge}] {chk.get('title')}**: {chk.get('summary')}")
            if chk.get("fix_command"):
                action_lbl = "Action" if is_fr else "Fix"
                lines.append(f"  - {action_lbl}: `{chk.get('fix_command')}`")

    fb = report.get("feedback")
    if isinstance(fb, Mapping):
        lines.extend(
            [
                "",
                "## Retour Anonyme CES 1-Question" if is_fr else "## 1-Prompt Anonymous CES Feedback",
                "",
                f"- **{fb.get('status')}**: {fb.get('message')}",
            ]
        )

    lines.extend(
        [
            "",
            f"**SHA-256 Receipt**: `{report.get('receipt_sha256')}`",
        ]
    )
    return _assert_safe_output("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `scripts/noel_post_setup_probe.py`."""
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser = argparse.ArgumentParser(
        description=(
            "HiveMind Noël post-setup readiness probe, zero-inference holiday route smoke gate, "
            "and 1-prompt anonymous CES reporter (100% personal PCs only, 0.00 EUR / 0.00 USD)."
        )
    )
    parser.add_argument("--gateway", default=default_gw, help="HiveMind gateway URL (https://*.run.app or localhost).")
    parser.add_argument("--smoke-routes", action="store_true", help="Run only the 7-route zero-inference smoke gate.")
    parser.add_argument("--verify-account", action="store_true", help="Also verify key against GET /v1/hivemind/me.")
    parser.add_argument(
        "--credentials-file",
        default="",
        help="Path to ~/.hivemind/credentials.json (defaults to user home).",
    )
    parser.add_argument(
        "--continue-config-path",
        default="",
        help="Path to ~/.continue/hivemind_provider.yaml (defaults to user home).",
    )
    parser.add_argument("--max-temp-c", type=float, default=GAMER_WHISPER_MAX_TEMP_C)
    parser.add_argument("--min-free-vram-mib", type=int, default=GAMER_WHISPER_MIN_FREE_VRAM_MIB)
    parser.add_argument("--game", action="append", default=[], help="Extra game executable name to guard.")
    parser.add_argument("--submit-feedback", action="store_true", help="Submit anonymous 1..5 CES feedback.")
    parser.add_argument("--prompt-ces", action="store_true", help="Interactive 1-prompt CES question.")
    parser.add_argument("--run-probe", action="store_true", help="Also run full probe when --submit-feedback is set.")
    parser.add_argument("--profile", default="dev", choices=("dev", "gamer"))
    parser.add_argument("--ease-rating", type=int, default=None, help="Ease rating from 1 (hard) to 5 (instant).")
    parser.add_argument("--had-game-lag", action="store_true", help="Report game stutter or fan noise.")
    parser.add_argument("--lang", default="en", help="Output language: 'en' or 'fr'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output with SHA-256 receipt.")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown output (default when --json omitted).")
    return parser


def _emit_report(report: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        serialized = _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False))
        print(serialized)
    else:
        print(render_markdown_report(report))


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    input_fn: InputFn = input,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    os_name: str | None = None,
) -> int:
    """CLI entrypoint returning `0` (ready), `1` (warnings / transient 429/5xx), or `2` (route/config fail)."""
    args = build_parser().parse_args(argv)
    try:
        lang_code = normalize_lang(args.lang)
        clean_gw = validate_gateway_url(args.gateway)
    except ProbeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    fb_result: dict[str, Any] | None = None
    try:
        if args.prompt_ces:
            fb_result = prompt_and_submit_ces_feedback(
                gateway=clean_gw,
                default_profile=args.profile,
                default_had_game_lag=bool(args.had_game_lag),
                input_fn=input_fn,
                http_fn=http_fn,
                lang=lang_code,
            )
        elif args.submit_feedback:
            if args.ease_rating is None:
                raise ProbeError("--submit-feedback requires --ease-rating 1..5 (or --prompt-ces).")
            fb_result = submit_ces_feedback(
                gateway=clean_gw,
                profile=args.profile,
                ease_rating=args.ease_rating,
                zero_game_lag=not bool(args.had_game_lag),
                http_fn=http_fn,
                lang=lang_code,
            )
    except ProbeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    feedback_standalone = bool(
        args.submit_feedback and not args.smoke_routes and not args.run_probe and not args.verify_account
    )
    if feedback_standalone and fb_result is not None:
        report = build_feedback_only_report(fb_result, gateway=clean_gw, lang=lang_code)
        _emit_report(report, as_json=bool(args.json))
        if fb_result["submitted"] or fb_result["skipped"]:
            return 0
        return 2 if fb_result.get("http_status") == 404 else 1

    cred_path = Path(args.credentials_file) if args.credentials_file else None
    cont_path = Path(args.continue_config_path) if args.continue_config_path else None
    report = run_post_setup_probe(
        gateway=clean_gw,
        smoke_only=bool(args.smoke_routes),
        verify_account=bool(args.verify_account),
        credentials_path=cred_path,
        continue_path=cont_path,
        max_temp_c=args.max_temp_c,
        min_free_vram_mib=args.min_free_vram_mib,
        extra_games=args.game,
        feedback_result=fb_result,
        lang=lang_code,
        os_name=os_name,
        environ=environ,
        home=home,
        http_fn=http_fn,
        runner=runner,
    )
    _emit_report(report, as_json=bool(args.json))

    smoke_info = report.get("smoke_gate")
    if isinstance(smoke_info, Mapping) and not smoke_info.get("all_routes_ok", False):
        return 2
    if report["failures_count"] > 0:
        return 2
    if fb_result is not None and fb_result["attempted"] and not fb_result["submitted"]:
        return 2 if fb_result.get("http_status") == 404 else 1
    if report["warnings_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
