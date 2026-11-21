#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-dependency HiveMind Canary Alpha preflight & troubleshooting assistant.

Compatible with Python 3.10-3.13 on Windows 11, macOS, and Linux using ONLY the
Python standard library (completes in < 10 seconds):
  1. Python runtime compatibility (3.10-3.13).
  2. ~/.hivemind/credentials.json presence, key format & POSIX 0600 permissions.
  3. Outbound HTTPS reachability to gateway /health and /v1/models (allowed hosts only).
  4. GPU & nvidia-smi readiness (VRAM >= 4096 MiB & temp <= 70 C for Gamer-Whisper,
     or 'Dev API-only mode ready' when no NVIDIA GPU is present).
  5. Active game process detection (Gamer-Whisper zero-lag guard).
  6. Local-only corporate environment warning if USERDNSDOMAIN / corporate proxy
     environment variables suggest a work PC (strictly local; NEVER transmitted).

Personal Home Use & Non-Monetary Compliance Rule:
  - 100% personal home PCs & personal hobby/open-source projects only (never work PCs).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal compute; symbolic HMT has zero monetary value.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
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

SCHEMA = "hivemind.doctor/1"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
SUPPORTED_LANGS = ("en", "fr")
MIN_PYTHON = (3, 10)
MAX_TESTED_MINOR = 13
GAMER_WHISPER_MIN_VRAM_MIB = 4096
GAMER_WHISPER_MAX_TEMP_C = 70.0
HTTP_TIMEOUT_S = 4.0
SUBPROCESS_TIMEOUT_S = 3.0

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_READ_ONLY_DATA_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})

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

EXIT_CODES: dict[str, int] = {
    "ready": 0,
    "ready_with_warnings": 1,
    "not_ready": 2,
}

COMPLIANCE_NOTICE: dict[str, str] = {
    "en": "100% personal home PCs & personal hobby/open-source projects only (0.00 EUR / 0.00 USD).",
    "fr": "100 % PC personnels à domicile et projets hobby/open-source uniquement (0,00 EUR / 0,00 USD).",
}

HttpGetFn = Callable[[str, float], tuple[int, dict[str, Any]]]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]


@dataclass(frozen=True)
class GpuInfo:
    """Measured local GPU state from nvidia-smi without serial/UUID."""

    present: bool
    parse_error: bool = False
    name: str = ""
    total_mib: int = 0
    free_mib: int = 0
    temp_c: float = 0.0
    util_pct: float = 0.0


@dataclass(frozen=True)
class DoctorCheck:
    """Single preflight check outcome with bilingual summary and one-line fix command."""

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
    """Normalize CLI language tag to 'en' or 'fr'."""
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ValueError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Replace user home directory prefix with '~' so local usernames never leak."""
    raw = str(target)
    resolved_home = str(home if home is not None else Path.home())
    if resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (resolved_home, resolved_home.replace("\\", "/"), resolved_home.replace("/", "\\")):
            if variant and raw.startswith(variant):
                suffix = raw[len(variant) :]
                if not suffix or suffix.startswith(("/", "\\")):
                    return "~" + suffix.replace("\\", "/")
    return raw


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff host satisfies the repository lateral-movement allowlist."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS or h in ALLOWED_READ_ONLY_DATA_HOSTS:
        return True
    if (len(h) > 8 and h.endswith(".run.app")) or h == "storage.googleapis.com":
        return True
    if len(h) > 15 and h.endswith(".googleapis.com"):
        return True
    return False


def validate_gateway_url(url: str) -> str:
    """Validate and normalize gateway base URL against the strict host allowlist."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Gateway URL must be a non-empty string.")
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        raise ValueError("Gateway URL must not contain whitespace or control characters.")
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ValueError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ValueError("Gateway URL must not contain query parameters or fragments.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise ValueError(
            f"Untrusted gateway host {host!r}; allowed hosts are *.run.app, *.googleapis.com, or loopback."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ValueError("Loopback gateway URL must use the http or https scheme.")
    elif parts.scheme != "https":
        raise ValueError("Remote gateway URL must use the https scheme.")
    return cleaned


def default_http_get_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, dict[str, Any]]:
    """Perform a direct, proxy-free GET request to an allowlisted HiveMind endpoint.

    Uses an empty ProxyHandler so corporate/ambient HTTP_PROXY env vars are never
    contacted and zero environment or hostname metadata is transmitted.
    """
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hivemind-doctor/1.0",
        },
        method="GET",
    )
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read(262_144).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = exc.read(65_536).decode("utf-8", errors="replace")
        try:
            parsed_err = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed_err = {"error": raw[:240]}
        return int(exc.code), (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return status_code, {"raw": raw[:240]}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def default_subprocess_runner(argv: Sequence[str], timeout_s: float = SUBPROCESS_TIMEOUT_S) -> tuple[int, str]:
    """Run a local read-only command with shell=False and a strict timeout."""
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


def _friendly_os_label(os_name: str, system_name: str) -> str:
    sys_low = (system_name or "").strip().lower()
    if os_name == "nt" or sys_low == "windows":
        return "Windows"
    if sys_low == "darwin":
        return "macOS"
    if sys_low == "linux":
        return "Linux"
    return system_name or os_name


def check_python_runtime(
    version_info: tuple[int, int, int] | None = None,
    *,
    os_name: str | None = None,
    system_name: str | None = None,
    lang: str = "en",
) -> DoctorCheck:
    """(1) Verify Python 3.10-3.13 runtime compatibility across Windows 11, macOS, and Linux."""
    lang_code = normalize_lang(lang)
    ver = tuple(version_info[:3]) if version_info is not None else tuple(sys.version_info[:3])
    eff_os = os_name if os_name is not None else os.name
    eff_sys = system_name if system_name is not None else platform.system()
    os_label = _friendly_os_label(eff_os, eff_sys)
    ver_text = f"{ver[0]}.{ver[1]}.{ver[2]}"
    title = "Python Runtime Compatibility" if lang_code == "en" else "Compatibilité Python"

    if eff_os == "nt" or os_label == "Windows":
        install_cmd = "winget install -e --id Python.Python.3.12"
    elif os_label == "macOS":
        install_cmd = "brew install python@3.12"
    else:
        install_cmd = "sudo apt-get update && sudo apt-get install -y python3"

    details = {"python_version": ver_text, "os": os_label, "target_range": "3.10-3.13"}

    if ver[:2] < MIN_PYTHON:
        summary = (
            f"Python {ver_text} on {os_label} is too old; HiveMind requires Python 3.10-3.13."
            if lang_code == "en"
            else f"Python {ver_text} sur {os_label} est trop ancien ; HiveMind requiert Python 3.10 à 3.13."
        )
        return DoctorCheck("python_runtime", title, "fail", summary, install_cmd, details)

    if ver[0] > 3 or ver[1] > MAX_TESTED_MINOR:
        summary = (
            f"Python {ver_text} on {os_label} runs stdlib API tools, while Python 3.10-3.13 is recommended for GPU packages."
            if lang_code == "en"
            else f"Python {ver_text} sur {os_label} exécute les outils standard, mais Python 3.10 à 3.13 est recommandé pour le GPU."
        )
        return DoctorCheck("python_runtime", title, "warn", summary, install_cmd, details)

    summary = (
        f"Python {ver_text} on {os_label} is compatible (Python 3.10-3.13 stdlib ready)."
        if lang_code == "en"
        else f"Python {ver_text} sur {os_label} est compatible (Python 3.10 à 3.13 prêt, zéro dépendance)."
    )
    return DoctorCheck("python_runtime", title, "ok", summary, "", details)


def _is_valid_hivemind_key(raw_key: object) -> bool:
    if not isinstance(raw_key, str):
        return False
    cleaned = raw_key.strip()
    if len(cleaned) < 12 or any(ch.isspace() for ch in cleaned):
        return False
    return cleaned.startswith(("hm-", "hm_live_"))


def check_credentials_file(
    credentials_path: Path | None = None,
    *,
    os_name: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
    home: Path | None = None,
) -> DoctorCheck:
    """(2) Verify ~/.hivemind/credentials.json presence, key prefix, and POSIX 0600 permissions."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    resolved_home = home if home is not None else Path.home()
    target = credentials_path if credentials_path is not None else (resolved_home / ".hivemind" / "credentials.json")
    shown_path = redact_home_path(target, home=resolved_home)
    title = "HiveMind Credentials & Permissions" if lang_code == "en" else "Identifiants HiveMind & Permissions"
    connect_fix = _quickstart_command(gateway, eff_os, "--write-continue-config")

    if not target.is_file():
        summary = (
            f"{shown_path} not found; run quickstart to register or save your personal hm-... key."
            if lang_code == "en"
            else f"{shown_path} introuvable ; lancez quickstart pour créer ou enregistrer votre clé personnelle hm-..."
        )
        return DoctorCheck(
            "credentials",
            title,
            "fail",
            summary,
            connect_fix,
            {"path": shown_path, "present": False, "valid_key": False},
        )

    try:
        st = target.stat()
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = (
            f"{shown_path} could not be parsed as valid JSON; regenerate your credentials file."
            if lang_code == "en"
            else f"{shown_path} est illisible ou JSON invalide ; régénérez votre fichier d'identifiants."
        )
        return DoctorCheck(
            "credentials",
            title,
            "fail",
            summary,
            connect_fix,
            {"path": shown_path, "present": True, "valid_key": False},
        )

    api_key = payload.get("api_key") if isinstance(payload, dict) else None
    if not _is_valid_hivemind_key(api_key):
        summary = (
            f"{shown_path} does not contain a valid hm-... API key; re-run quickstart."
            if lang_code == "en"
            else f"{shown_path} ne contient pas de clé API hm-... valide ; relancez quickstart."
        )
        return DoctorCheck(
            "credentials",
            title,
            "fail",
            summary,
            connect_fix,
            {"path": shown_path, "present": True, "valid_key": False},
        )

    role = str(payload.get("role") or "prosumer") if isinstance(payload, dict) else "prosumer"

    if eff_os == "nt":
        summary = (
            f"{shown_path} present with valid hm-... key (role: {role}; Windows user-profile ACL)."
            if lang_code == "en"
            else f"{shown_path} présent avec clé hm-... valide (rôle : {role} ; ACL profil utilisateur Windows)."
        )
        return DoctorCheck(
            "credentials",
            title,
            "ok",
            summary,
            "",
            {
                "path": shown_path,
                "present": True,
                "valid_key": True,
                "role": role,
                "posix_permissions_checked": False,
                "permission_model": "windows_acl",
            },
        )

    mode_bits = st.st_mode & 0o777
    octal_mode = f"{mode_bits:04o}"
    if mode_bits != 0o600:
        chmod_target = shown_path if shown_path.startswith("~/") else shlex.quote(shown_path)
        chmod_cmd = f"chmod 600 {chmod_target}"
        summary = (
            f"{shown_path} has POSIX mode {octal_mode} (expected 0600 owner-only); tighten file permissions."
            if lang_code == "en"
            else f"{shown_path} a les permissions POSIX {octal_mode} (attendu 0600 propriétaire seul) ; restreignez l'accès."
        )
        return DoctorCheck(
            "credentials",
            title,
            "warn",
            summary,
            chmod_cmd,
            {
                "path": shown_path,
                "present": True,
                "valid_key": True,
                "role": role,
                "posix_permissions_checked": True,
                "posix_mode": octal_mode,
            },
        )

    summary = (
        f"{shown_path} present with valid hm-... key and POSIX 0600 permissions (role: {role})."
        if lang_code == "en"
        else f"{shown_path} présent avec clé hm-... valide et permissions POSIX 0600 (rôle : {role})."
    )
    return DoctorCheck(
        "credentials",
        title,
        "ok",
        summary,
        "",
        {
            "path": shown_path,
            "present": True,
            "valid_key": True,
            "role": role,
            "posix_permissions_checked": True,
            "posix_mode": octal_mode,
        },
    )


def check_gateway_reachability(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_get: HttpGetFn = default_http_get_json,
    timeout_s: float = HTTP_TIMEOUT_S,
    os_name: str | None = None,
    lang: str = "en",
) -> DoctorCheck:
    """(3) Verify outbound HTTPS reachability to gateway /health and /v1/models."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    title = "Gateway HTTPS Reachability (/health & /v1/models)" if lang_code == "en" else "Accès Passerelle HTTPS (/health & /v1/models)"

    try:
        clean_gw = validate_gateway_url(gateway)
    except ValueError as exc:
        fix_cmd = (
            f'python scripts\\hivemind_doctor.py --gateway "{DEFAULT_GATEWAY}"'
            if eff_os == "nt"
            else f'python3 scripts/hivemind_doctor.py --gateway "{DEFAULT_GATEWAY}"'
        )
        summary = (
            f"Refused gateway URL ({exc})"
            if lang_code == "en"
            else f"URL de passerelle refusée ({exc})"
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            fix_cmd,
            {"allowed_host": False, "health_ok": False, "models_ok": False},
        )

    curl_bin = "curl.exe" if eff_os == "nt" else "curl"
    health_fix = f'{curl_bin} -fsSL "{clean_gw}/health"'
    models_fix = f'{curl_bin} -fsSL "{clean_gw}/v1/models"'

    started = time.monotonic()
    try:
        health_status, health_body = http_get(f"{clean_gw}/health", timeout_s)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        summary = (
            f"Cannot reach {clean_gw}/health ({type(exc).__name__}); verify internet/TLS 443 access."
            if lang_code == "en"
            else f"Impossible de joindre {clean_gw}/health ({type(exc).__name__}) ; vérifiez la connexion HTTPS 443."
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            health_fix,
            {"gateway": clean_gw, "allowed_host": True, "health_ok": False, "models_ok": False},
        )

    if health_status != 200:
        summary = (
            f"Gateway {clean_gw}/health returned HTTP {health_status}."
            if lang_code == "en"
            else f"La passerelle {clean_gw}/health a répondu HTTP {health_status}."
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            health_fix,
            {
                "gateway": clean_gw,
                "allowed_host": True,
                "health_status": health_status,
                "health_ok": False,
                "models_ok": False,
            },
        )

    try:
        models_status, models_body = http_get(f"{clean_gw}/v1/models", timeout_s)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        summary = (
            f"{clean_gw}/health is OK, while /v1/models failed ({type(exc).__name__})."
            if lang_code == "en"
            else f"{clean_gw}/health répond OK, mais /v1/models a échoué ({type(exc).__name__})."
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            models_fix,
            {"gateway": clean_gw, "allowed_host": True, "health_ok": True, "models_ok": False},
        )

    raw_models = models_body.get("data") if isinstance(models_body, dict) else None
    if models_status != 200 or not isinstance(raw_models, list):
        summary = (
            f"{clean_gw}/v1/models returned HTTP {models_status} without a valid model list."
            if lang_code == "en"
            else f"{clean_gw}/v1/models a répondu HTTP {models_status} sans liste de modèles valide."
        )
        return DoctorCheck(
            "gateway_reachability",
            title,
            "fail",
            summary,
            models_fix,
            {
                "gateway": clean_gw,
                "allowed_host": True,
                "health_ok": True,
                "models_status": models_status,
                "models_ok": False,
            },
        )

    model_ids = [
        str(item["id"])
        for item in raw_models
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
    ]
    probe_elapsed_s = round(time.monotonic() - started, 3)
    health_state = str(health_body.get("status", "ok")) if isinstance(health_body, dict) else "ok"

    summary = (
        f"Gateway {clean_gw} reachable (/health={health_state}, /v1/models={len(model_ids)} model(s))."
        if lang_code == "en"
        else f"Passerelle {clean_gw} joignable (/health={health_state}, /v1/models={len(model_ids)} modèle(s))."
    )
    return DoctorCheck(
        "gateway_reachability",
        title,
        "ok",
        summary,
        "",
        {
            "gateway": clean_gw,
            "allowed_host": True,
            "health_ok": True,
            "models_ok": True,
            "live_models_count": len(model_ids),
            "sample_models": model_ids[:5],
            "probe_elapsed_s": probe_elapsed_s,
        },
    )


def _parse_float_field(raw: str, default: float = 0.0) -> float:
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in ("[n/a]", "n/a", "[not supported]", "not supported"):
        return default
    return float(cleaned)


def probe_gpu(runner: RunnerFn = default_subprocess_runner) -> GpuInfo:
    """Query nvidia-smi locally without requesting GPU serial or UUID."""
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
        return GpuInfo(present=False)

    if rc != 0 or not out.strip():
        return GpuInfo(present=False)

    candidates: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0]:
            return GpuInfo(present=True, parse_error=True)
        try:
            total_mib = int(float(parts[1]))
            free_mib = int(float(parts[2]))
            temp_c = _parse_float_field(parts[3], default=0.0)
            util_pct = _parse_float_field(parts[4], default=0.0)
        except ValueError:
            return GpuInfo(present=True, parse_error=True)
        if total_mib <= 0 or free_mib < 0:
            return GpuInfo(present=True, parse_error=True)
        candidates.append(
            GpuInfo(
                present=True,
                parse_error=False,
                name=parts[0],
                total_mib=total_mib,
                free_mib=free_mib,
                temp_c=temp_c,
                util_pct=util_pct,
            )
        )

    return max(candidates, key=lambda g: (g.total_mib, -g.temp_c, g.free_mib))


def check_gpu_readiness(
    *,
    runner: RunnerFn = default_subprocess_runner,
    os_name: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
) -> DoctorCheck:
    """(4) Check GPU & nvidia-smi readiness for Gamer-Whisper or Dev API-only mode."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    title = "GPU & Gamer-Whisper / Dev API Readiness" if lang_code == "en" else "Disponibilité GPU Gamer-Whisper / Mode Dev API"
    gpu = probe_gpu(runner)

    if not gpu.present:
        summary = (
            "No NVIDIA GPU detected — Dev API-only mode ready (VS Code Continue.dev & OpenAI-compatible API, 0 local GPU load)."
            if lang_code == "en"
            else "Aucun GPU NVIDIA détecté — Mode Dev API uniquement prêt (Dev API-only mode ready : VS Code Continue.dev sans GPU local)."
        )
        return DoctorCheck(
            "gpu_readiness",
            title,
            "ok",
            summary,
            "",
            {"gpu_present": False, "mode": "dev_api_only"},
        )

    if gpu.parse_error:
        summary = (
            "nvidia-smi returned unparseable output — Dev API-only mode ready; inspect your NVIDIA driver before Gamer-Whisper."
            if lang_code == "en"
            else "Sortie nvidia-smi illisible — Mode Dev API uniquement prêt (Dev API-only mode ready) ; vérifiez le pilote NVIDIA."
        )
        return DoctorCheck(
            "gpu_readiness",
            title,
            "warn",
            summary,
            "nvidia-smi",
            {"gpu_present": True, "parse_error": True, "mode": "dev_api_only"},
        )

    gpu_details = {
        "gpu_present": True,
        "name": gpu.name,
        "total_mib": gpu.total_mib,
        "free_mib": gpu.free_mib,
        "temp_c": gpu.temp_c,
        "util_pct": gpu.util_pct,
        "min_vram_mib": GAMER_WHISPER_MIN_VRAM_MIB,
        "max_temp_c": GAMER_WHISPER_MAX_TEMP_C,
    }

    if gpu.total_mib < GAMER_WHISPER_MIN_VRAM_MIB:
        consumer_cmd = _quickstart_command(gateway, eff_os, "--role consumer --write-continue-config")
        summary = (
            f"{gpu.name} has {gpu.total_mib} MiB VRAM (< {GAMER_WHISPER_MIN_VRAM_MIB} MiB for Gamer-Whisper) — Dev API-only mode ready."
            if lang_code == "en"
            else (
                f"{gpu.name} a {gpu.total_mib} MiB de VRAM (< {GAMER_WHISPER_MIN_VRAM_MIB} MiB pour Gamer-Whisper) "
                "— Mode Dev API uniquement prêt (Dev API-only mode ready)."
            )
        )
        return DoctorCheck(
            "gpu_readiness",
            title,
            "warn",
            summary,
            consumer_cmd,
            {**gpu_details, "mode": "dev_api_only"},
        )

    if gpu.temp_c > GAMER_WHISPER_MAX_TEMP_C:
        gamer_check_cmd = _quickstart_command(gateway, eff_os, "--gamer-check")
        summary = (
            f"{gpu.name} ({gpu.total_mib} MiB VRAM) is at {gpu.temp_c:.0f} C (> {GAMER_WHISPER_MAX_TEMP_C:.0f} C Gamer-Whisper ceiling); "
            "let it cool <= 70 C before GPU rounds (Dev API-only mode ready now)."
            if lang_code == "en"
            else (
                f"{gpu.name} ({gpu.total_mib} MiB VRAM) est à {gpu.temp_c:.0f} C (> {GAMER_WHISPER_MAX_TEMP_C:.0f} C plafond Gamer-Whisper) ; "
                "laissez refroidir <= 70 C avant un round GPU."
            )
        )
        return DoctorCheck(
            "gpu_readiness",
            title,
            "warn",
            summary,
            gamer_check_cmd,
            {**gpu_details, "mode": "gpu_thermal_cooldown"},
        )

    summary = (
        f"{gpu.name} ready for Gamer-Whisper ({gpu.total_mib} MiB VRAM >= {GAMER_WHISPER_MIN_VRAM_MIB} MiB, "
        f"{gpu.free_mib} MiB free, {gpu.temp_c:.0f} C <= {GAMER_WHISPER_MAX_TEMP_C:.0f} C)."
        if lang_code == "en"
        else (
            f"{gpu.name} prêt pour Gamer-Whisper ({gpu.total_mib} MiB VRAM >= {GAMER_WHISPER_MIN_VRAM_MIB} MiB, "
            f"{gpu.free_mib} MiB libres, {gpu.temp_c:.0f} C <= {GAMER_WHISPER_MAX_TEMP_C:.0f} C)."
        )
    )
    return DoctorCheck(
        "gpu_readiness",
        title,
        "ok",
        summary,
        "",
        {**gpu_details, "mode": "gamer_whisper_ready"},
    )


def list_process_names(
    runner: RunnerFn = default_subprocess_runner,
    *,
    os_name: str | None = None,
) -> set[str]:
    """Return normalized lowercase process names running locally on Windows, macOS, or Linux."""
    eff_os = os_name if os_name is not None else os.name
    names: set[str] = set()
    try:
        if eff_os == "nt":
            rc, out = runner(["tasklist", "/FO", "CSV", "/NH"], SUBPROCESS_TIMEOUT_S)
            if rc != 0:
                return names
            for row in csv.reader(io.StringIO(out)):
                if row and row[0].strip():
                    exe_name = row[0].strip().lower()
                    names.add(exe_name)
        else:
            rc, out = runner(["ps", "-eo", "comm="], SUBPROCESS_TIMEOUT_S)
            if rc != 0:
                return names
            for line in out.splitlines():
                cleaned = line.strip()
                if cleaned:
                    low = cleaned.lower()
                    names.add(low)
                    names.add(Path(cleaned.replace("\\", "/")).name.lower())
    except (OSError, subprocess.SubprocessError, csv.Error):
        return names
    return names


def check_active_games(
    *,
    runner: RunnerFn = default_subprocess_runner,
    os_name: str | None = None,
    extra_games: Sequence[str] = (),
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
) -> DoctorCheck:
    """(5) Inspect local process list for active games so Gamer-Whisper never disturbs play."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    title = "Active Game Process Guard" if lang_code == "en" else "Détection de Jeu Actif"

    watchlist = set(DEFAULT_GAME_EXECUTABLES)
    for item in extra_games:
        cleaned = item.strip().lower()
        if cleaned:
            watchlist.add(cleaned)
            watchlist.add(cleaned[:15])

    running = list_process_names(runner, os_name=eff_os)
    detected = sorted(proc for proc in running if proc in watchlist)

    if detected:
        joined = ", ".join(detected)
        gamer_check_cmd = _quickstart_command(gateway, eff_os, "--gamer-check")
        summary = (
            f"Active game detected ({joined}); Gamer-Whisper yields 100% of GPU VRAM until you close the game."
            if lang_code == "en"
            else f"Jeu actif détecté ({joined}) ; Gamer-Whisper cède 100 % de la VRAM tant que le jeu est ouvert."
        )
        return DoctorCheck(
            "active_games",
            title,
            "warn",
            summary,
            gamer_check_cmd,
            {"active_games": detected, "game_running": True},
        )

    summary = (
        "No active game process detected; Gamer-Whisper idle guard is clear."
        if lang_code == "en"
        else "Aucun jeu actif détecté ; créneau silencieux Gamer-Whisper disponible."
    )
    return DoctorCheck(
        "active_games",
        title,
        "ok",
        summary,
        "",
        {"active_games": [], "game_running": False},
    )


def _detect_corporate_indicators(environ: Mapping[str, str]) -> list[str]:
    """Inspect environment variables strictly in memory without recording values."""
    indicators: list[str] = []
    upper_map: dict[str, str] = {str(k).upper(): str(v).strip() for k, v in environ.items() if str(v).strip()}

    if upper_map.get("USERDNSDOMAIN"):
        indicators.append("USERDNSDOMAIN")

    userdomain = upper_map.get("USERDOMAIN", "")
    computername = upper_map.get("COMPUTERNAME", "")
    if userdomain and userdomain.upper() != computername.upper():
        low_dom = userdomain.lower()
        if any(marker in low_dom for marker in _CORP_MARKERS):
            indicators.append("USERDOMAIN")

    for proxy_key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        if upper_map.get(proxy_key):
            indicators.append(proxy_key)

    no_proxy_val = upper_map.get("NO_PROXY", "").lower()
    if no_proxy_val and any(marker in no_proxy_val for marker in _CORP_MARKERS):
        indicators.append("NO_PROXY")

    for tls_key in ("ZSCALER_CERT_PATH", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"):
        tls_val = upper_map.get(tls_key, "").lower()
        if tls_key == "ZSCALER_CERT_PATH" and tls_val:
            indicators.append(tls_key)
        elif tls_val and any(marker in tls_val for marker in _CORP_MARKERS):
            indicators.append(tls_key)

    return indicators


def check_corporate_environment(
    *,
    environ: Mapping[str, str] | None = None,
    os_name: str | None = None,
    gateway: str = DEFAULT_GATEWAY,
    lang: str = "en",
) -> DoctorCheck:
    """(6) Strictly local corporate/work-PC warning (NEVER transmits env or hostname)."""
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    env_view = environ if environ is not None else os.environ
    title = (
        "Personal-PC Compliance (Local-Only Check)"
        if lang_code == "en"
        else "Conformité PC Personnel (Contrôle 100 % Local)"
    )

    indicators = _detect_corporate_indicators(env_view)
    if indicators:
        uninstall_cmd = _quickstart_command(gateway, eff_os, "--uninstall")
        joined = ", ".join(indicators)
        summary = (
            f"Corporate/work environment indicator detected locally ({joined}). "
            "STOP if this is a work machine: HiveMind is strictly 100% personal home PCs & personal hobby/open-source projects only (0.00 EUR / 0.00 USD)."
            if lang_code == "en"
            else (
                f"Indicateur d'environnement professionnel détecté localement ({joined}). "
                "ARRÊTEZ s'il s'agit d'un PC professionnel : HiveMind est réservé à 100 % aux PC personnels à domicile et projets hobby/open-source (0,00 EUR / 0,00 USD)."
            )
        )
        return DoctorCheck(
            "personal_pc_compliance",
            title,
            "warn",
            summary,
            uninstall_cmd,
            {
                "corporate_indicators": indicators,
                "network_transmitted": False,
                "personal_pc_only": True,
                "monetary_cost": "0.00 EUR / 0.00 USD",
            },
        )

    summary = (
        "Local environment check clean (no corporate domain/proxy detected). "
        "Reminder: 100% personal home PCs & personal hobby/open-source projects only (0.00 EUR / 0.00 USD)."
        if lang_code == "en"
        else (
            "Contrôle local conforme (aucun domaine/proxy professionnel détecté). "
            "Rappel : 100 % PC personnels à domicile et projets hobby/open-source uniquement (0,00 EUR / 0,00 USD)."
        )
    )
    return DoctorCheck(
        "personal_pc_compliance",
        title,
        "ok",
        summary,
        "",
        {
            "corporate_indicators": [],
            "network_transmitted": False,
            "personal_pc_only": True,
            "monetary_cost": "0.00 EUR / 0.00 USD",
        },
    )


def run_doctor(
    *,
    gateway: str = DEFAULT_GATEWAY,
    credentials_path: Path | None = None,
    lang: str = "en",
    extra_games: Sequence[str] = (),
    timeout_s: float = HTTP_TIMEOUT_S,
    version_info: tuple[int, int, int] | None = None,
    os_name: str | None = None,
    system_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    http_get: HttpGetFn = default_http_get_json,
    runner: RunnerFn = default_subprocess_runner,
) -> dict[str, Any]:
    """Run all 6 Canary Alpha preflight checks in < 10 seconds and build a structured report."""
    started = time.monotonic()
    lang_code = normalize_lang(lang)
    eff_os = os_name if os_name is not None else os.name
    bounded_timeout = min(max(0.5, float(timeout_s)), 4.5)

    try:
        safe_gateway = validate_gateway_url(gateway)
    except ValueError:
        safe_gateway = DEFAULT_GATEWAY

    checks: list[DoctorCheck] = [
        check_python_runtime(version_info, os_name=eff_os, system_name=system_name, lang=lang_code),
        check_credentials_file(
            credentials_path,
            os_name=eff_os,
            gateway=safe_gateway,
            lang=lang_code,
            home=home,
        ),
        check_gateway_reachability(
            gateway,
            http_get=http_get,
            timeout_s=bounded_timeout,
            os_name=eff_os,
            lang=lang_code,
        ),
        check_gpu_readiness(runner=runner, os_name=eff_os, gateway=safe_gateway, lang=lang_code),
        check_active_games(
            runner=runner,
            os_name=eff_os,
            extra_games=extra_games,
            gateway=safe_gateway,
            lang=lang_code,
        ),
        check_corporate_environment(
            environ=environ,
            os_name=eff_os,
            gateway=safe_gateway,
            lang=lang_code,
        ),
    ]

    passed_count = sum(1 for c in checks if c.passed)
    warnings_count = sum(1 for c in checks if c.status == "warn")
    failures_count = sum(1 for c in checks if c.status == "fail")

    if failures_count > 0:
        verdict = "not_ready"
    elif warnings_count > 0:
        verdict = "ready_with_warnings"
    else:
        verdict = "ready"

    gpu_mode = str(checks[3].details.get("mode", "dev_api_only"))
    recommended_mode = "gamer_whisper" if gpu_mode == "gamer_whisper_ready" else "dev_api_only"

    fix_commands: list[str] = []
    for chk in checks:
        if chk.fix_command and chk.fix_command not in fix_commands:
            fix_commands.append(chk.fix_command)

    reported_gateway = safe_gateway if checks[2].details.get("allowed_host", False) else "rejected_untrusted_gateway"

    return {
        "schema": SCHEMA,
        "ready": verdict == "ready",
        "verdict": verdict,
        "lang": lang_code,
        "recommended_mode": recommended_mode,
        "readiness_checks_passed": passed_count,
        "readiness_checks_total": len(checks),
        "warnings_count": warnings_count,
        "failures_count": failures_count,
        "elapsed_s": round(time.monotonic() - started, 4),
        "gateway": reported_gateway,
        "compliance_notice": COMPLIANCE_NOTICE[lang_code],
        "checks": [c.to_dict() for c in checks],
        "fix_commands": fix_commands,
    }


def render_report(report: dict[str, Any], *, color: bool = False) -> str:
    """Format a human-readable bilingual terminal summary with actionable one-line fixes."""
    lang_code = normalize_lang(str(report.get("lang", "en")))
    badges = {"ok": " OK ", "warn": "WARN", "fail": "FAIL"}
    ansi = {"ok": "32", "warn": "33", "fail": "31"}

    if lang_code == "fr":
        banner_title = "  HIVEMIND DOCTOR — PRÉ-CONTRÔLE CANARY ALPHA (PC PERSO • 0,00 EUR)"
        rule_prefix = "  Règle"
        fix_prefix = "Action"
        summary_tpl = "Contrôles réussis : {passed}/{total} | Mode conseillé : {mode} | Verdict : {verdict}"
    else:
        banner_title = "  HIVEMIND DOCTOR — CANARY ALPHA PREFLIGHT (PERSONAL PC ONLY • 0.00 EUR)"
        rule_prefix = "  Rule"
        fix_prefix = "Fix"
        summary_tpl = "Checks passed: {passed}/{total} | Recommended mode: {mode} | Verdict: {verdict}"

    lines = [
        "=" * 72,
        banner_title,
        "=" * 72,
        f"{rule_prefix}: {report['compliance_notice']}",
        "",
    ]

    for chk in report["checks"]:
        status = str(chk["status"])
        tag = f"[{badges.get(status, 'INFO')}]"
        if color and status in ansi:
            tag = f"\x1b[{ansi[status]}m{tag}\x1b[0m"
        lines.append(f"{tag} {chk['title']}")
        lines.append(f"       {chk['summary']}")
        if chk.get("fix_command"):
            lines.append(f"       {fix_prefix} -> {chk['fix_command']}")

    lines.append("")
    lines.append(
        summary_tpl.format(
            passed=report["readiness_checks_passed"],
            total=report["readiness_checks_total"],
            mode=report["recommended_mode"],
            verdict=str(report["verdict"]).upper(),
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-dependency HiveMind Canary Alpha preflight & troubleshooting assistant (0.00 EUR, personal PCs only)."
    )
    default_gw = (
        os.environ.get("HIVEMIND_GATEWAY_URL")
        or os.environ.get("HIVEMIND_GATEWAY")
        or DEFAULT_GATEWAY
    )
    parser.add_argument("--gateway", default=default_gw, help="HiveMind HTTPS gateway URL (*.run.app or localhost).")
    parser.add_argument(
        "--credentials-file",
        default=str(Path.home() / ".hivemind" / "credentials.json"),
        help="Path to local HiveMind credentials JSON file.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        type=normalize_lang,
        choices=SUPPORTED_LANGS,
        help="Output language: 'en' or 'fr'.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report.")
    parser.add_argument("--game", action="append", default=[], help="Extra game executable name to check.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=HTTP_TIMEOUT_S,
        help="Per-request HTTP timeout in seconds (bounded to <= 4.5s for < 10s total runtime).",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in terminal output.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_get: HttpGetFn = default_http_get_json,
    runner: RunnerFn = default_subprocess_runner,
    environ: Mapping[str, str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    report = run_doctor(
        gateway=args.gateway,
        credentials_path=Path(args.credentials_file),
        lang=args.lang,
        extra_games=args.game,
        timeout_s=args.timeout,
        environ=environ,
        http_get=http_get,
        runner=runner,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        use_color = bool(sys.stdout.isatty() and not args.no_color and os.name != "nt")
        print(render_report(report, color=use_color))
    return EXIT_CODES[str(report["verdict"])]


if __name__ == "__main__":
    raise SystemExit(main())
