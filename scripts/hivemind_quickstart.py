#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-dependency cross-platform HiveMind quick-connect, smoke gate & Gamer-Whisper assistant.

Designed for personal home machines (Windows 11, macOS, Linux) with standard-library Python 3.10-3.13 only:
  1. ``connect`` (default): enforce local Personal-PC compliance, register or load
     ``~/.hivemind/credentials.json``, verify API access against ``/v1/hivemind/me`` and
     ``/v1/models``, and output a ready-to-use VS Code (Continue.dev) snippet without PyTorch.
  2. ``--smoke-routes`` / ``--require-holiday-routes``: zero-inference Cloud Run holiday route
     smoke gate checking ``GET /health``, ``GET /noel``, ``GET /quick``, and
     ``GET /v1/hivemind/onboarding/feedback/summary`` with masked ``hm-...`` keys, ``~`` home-path
     redaction, and a deterministic SHA-256 receipt.
  3. ``--gamer-check`` / ``--gamer-whisper``: quiet gaming-PC guard (``<= 70 C`` default ceiling,
     ``2048 MiB`` free-VRAM reserve, active Steam/PC game detection, ``< 1.5 s`` process-tree stop).
  4. ``--post-setup-telemetry`` / ``--feedback``: timeout-safe anonymous CES & Steam zero-game-lag
     telemetry to ``POST /v1/hivemind/onboarding/feedback`` (non-blocking on HTTP 404/429/5xx/timeout).
  5. ``--uninstall``: remove saved local HiveMind credentials and snippet cleanly in one step.

Personal Home Use & Non-Monetary Notice (Amadeus / Personal-PC Compliance Rule):
  - 100% personal home PCs and personal hobby/open-source side-projects ONLY (never work laptops or corporate code/data).
  - 0.00 EUR / 0.00 USD: symbolic points have zero monetary value.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import ipaddress
import json
import os
import platform
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
COMPLIANCE_NOTICE_EN = (
    "100% personal home PCs & personal hobby/open-source projects only "
    "(Amadeus / Personal-PC rule: never work PCs or corporate code; 0.00 EUR / 0.00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_FR = (
    "100 % PC personnels à domicile et projets hobby/open-source uniquement "
    "(règle Amadeus / PC perso : jamais de PC ni de code d'entreprise ; 0,00 EUR / 0,00 USD, AGPL-3.0)."
)
COMPLIANCE_NOTICE_BILINGUAL = f"{COMPLIANCE_NOTICE_EN} | {COMPLIANCE_NOTICE_FR}"

HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,47}$")
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
_RUN_APP_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+run\.app$")
_PLATFORM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")

GAMER_WHISPER_MAX_TEMP_C = 70.0
GAMER_WHISPER_YIELD_BUDGET_S = 1.5
HTTP_TIMEOUT_S = 10.0
TELEMETRY_TIMEOUT_S = 3.0

HOLIDAY_SMOKE_ROUTES: tuple[tuple[str, str], ...] = (
    ("/health", "core"),
    ("/noel", "holiday"),
    ("/quick", "holiday"),
    ("/v1/hivemind/onboarding/feedback/summary", "holiday"),
)

CORPORATE_INDICATORS: tuple[str, ...] = (
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

_CORP_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
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
    ("paloalto", re.compile(r"palo[-_]?alto", re.IGNORECASE)),
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RAW_API_KEY_RE = re.compile(r"\bhm(?:_live_|-)(?!\*\*\*)[A-Za-z0-9_-]{4,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+(?!<redacted>)[A-Za-z0-9._~+/=*-]{4,}\b", re.IGNORECASE)
_HOME_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+)"
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_FORBIDDEN_KEY_RE = re.compile(r'"' + "sco" + r're"\s*:\s*\d')

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
    }
)


@dataclass(frozen=True)
class GpuSnapshot:
    """Measured local GPU state from nvidia-smi."""

    present: bool
    name: str
    total_mib: int
    free_mib: int
    temp_c: float
    util_pct: float


HttpFn = Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], tuple[int, dict[str, Any]]]
RunnerFn = Callable[[Sequence[str], float], tuple[int, str]]


def _default_credentials_path() -> Path:
    return Path.home() / ".hivemind" / "credentials.json"


def _default_continue_snippet_path() -> Path:
    return Path.home() / ".continue" / "hivemind_provider.yaml"


def normalize_lang(raw: str | None) -> str:
    """Normalize CLI language selector to ``'en'`` or ``'fr'``."""
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, str)):
        raise ValueError("Language must be 'en' or 'fr'.")
    cleaned = (raw or "en").strip().lower().replace("_", "-")
    if cleaned in ("en", "en-us", "en-gb"):
        return "en"
    if cleaned in ("fr", "fr-fr", "fr-be", "fr-ca", "fr-ch"):
        return "fr"
    raise ValueError(f"Unsupported language {raw!r}; expected 'en' or 'fr'.")


def mask_api_key(raw_key: object) -> str:
    """Mask an ``hm-...`` or ``hm_live_...`` key so raw secrets never appear in output."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        return "<none>"
    cleaned = raw_key.strip()
    if any(ch.isspace() for ch in cleaned):
        return "<redacted>"
    if cleaned.startswith("hm_live_") and len(cleaned) >= 12:
        return f"hm_live_***{cleaned[-4:]}"
    if cleaned.startswith("hm-") and len(cleaned) >= 8:
        return f"hm-***{cleaned[-4:]}"
    return "<redacted>"


def redact_home_path(target: Path | str, home: Path | None = None) -> str:
    """Redact local user home directory prefixes to ``'~'`` so OS usernames never leak."""
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
    return redacted.replace("\\", "/") if redacted != raw else raw


def redact_sensitive_text(text: str, home: Path | None = None) -> str:
    """Scrub API keys, Bearer tokens, emails, OS home paths, and non-loopback IPs from free text."""
    if not text:
        return ""
    out = redact_home_path(text, home=home)
    resolved_home = str(home if home is not None else Path.home())
    if resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (
            resolved_home,
            resolved_home.replace("\\", "/"),
            resolved_home.replace("/", "\\"),
        ):
            if variant and variant in out:
                out = out.replace(variant, "~")
    out = _HOME_PATH_RE.sub("~", out)
    out = _BEARER_RE.sub("Bearer <redacted>", out)
    out = _RAW_API_KEY_RE.sub(lambda m: mask_api_key(m.group(0)), out)
    out = _EMAIL_RE.sub("<redacted-email>", out)

    def _scrub_ip(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            ip_obj = ipaddress.ip_address(candidate)
        except ValueError:
            return candidate
        return candidate if ip_obj.is_loopback else "<redacted-ip>"

    return _IPV4_RE.sub(_scrub_ip, out)


def scan_output_safety(text: str, home: Path | None = None) -> list[str]:
    """Audit rendered text/JSON for secret leaks, home paths, non-loopback IPs, or forbidden keys."""
    if not isinstance(text, str):
        raise ValueError("scan_output_safety requires a string.")
    violations: list[str] = []
    if _FORBIDDEN_KEY_RE.search(text):
        violations.append("forbidden_key_detected")
    if _RAW_API_KEY_RE.search(text):
        violations.append("unmasked_api_key_detected")
    if _BEARER_RE.search(text):
        violations.append("bearer_token_detected")
    if _EMAIL_RE.search(text):
        violations.append("email_detected")
    resolved_home = str(home if home is not None else Path.home())
    has_home_leak = bool(_HOME_PATH_RE.search(text))
    if not has_home_leak and resolved_home and resolved_home not in ("/", "\\", ""):
        for variant in (resolved_home, resolved_home.replace("\\", "/")):
            if variant and variant in text:
                has_home_leak = True
                break
    if has_home_leak:
        violations.append("unredacted_home_path_detected")
    for ip_str in _IPV4_RE.findall(text):
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not ip_obj.is_loopback:
            violations.append(f"non_loopback_ipv4_detected:{ip_str}")
    return violations


def compute_receipt_sha256(payload: Mapping[str, Any]) -> str:
    """Compute a deterministic canonical JSON SHA-256 hex receipt (excluding ``receipt_sha256``)."""
    canonical = json.dumps(
        {k: v for k, v in payload.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff ``host`` is ``*.run.app`` or loopback (``127.0.0.1`` / ``localhost`` / ``::1``)."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
    return bool(_RUN_APP_HOST_RE.fullmatch(h))


def validate_gateway_url(url: str) -> str:
    """Validate gateway origin URL against ``https://*.run.app`` or loopback ``http(s)://127.0.0.1|localhost|[::1]``."""
    if isinstance(url, bool) or not isinstance(url, str) or not url.strip():
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: expected a *.run.app HTTPS gateway or localhost."
        )
    cleaned = url.strip().rstrip("/")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: whitespace and control characters are forbidden."
        )
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ValueError("Refusing gateway URL with embedded credentials.")
    if parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("Refusing gateway URL with path, query, or fragment; provide bare origin only.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Refusing gateway URL with invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Refusing gateway URL with out-of-range port.")
    host = (parts.hostname or "").lower()
    if not is_allowed_gateway_host(host):
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: expected a *.run.app HTTPS gateway or localhost."
        )
    if host in ALLOWED_LOOPBACK_HOSTS:
        if parts.scheme not in ("http", "https"):
            raise ValueError("Loopback gateway URL must use http or https.")
    elif parts.scheme != "https":
        raise ValueError("Remote Cloud Run gateway URL must use https.")
    return cleaned


def _assert_zero_inference_request(method: str, url: str) -> str:
    """Enforce allowed gateway host and strictly forbid any ``/v1/chat/completions`` inference call."""
    verb = (method or "").strip().upper()
    if verb not in ("GET", "HEAD", "POST"):
        raise ValueError(f"Unsupported HTTP method {verb!r}.")
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    low_path = (parts.path or "").lower()
    if "completions" in low_path:
        raise ValueError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden in quickstart."
        )
    return origin


def sanitize_handle(raw: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._+-]+", "-", (raw or "").strip()).strip("-.")
    if len(candidate) < 2:
        candidate = f"friend-{secrets.token_hex(3)}"
    candidate = candidate[:48]
    if not HANDLE_RE.match(candidate) or candidate.lower().startswith(("google.", "github.", "oidc.")):
        raise ValueError(
            "Invalid handle: use 2-48 ASCII letters, digits, '.', '_', '+', or '-' (not starting with a provider prefix)."
        )
    return candidate


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """Proxy-free stdlib HTTP helper enforcing host allowlist and zero-inference invariants."""
    _assert_zero_inference_request(method, url)
    req_headers = {
        "Accept": "application/json, text/html;q=0.9",
        "User-Agent": "hivemind-quickstart/1.0",
    }
    if headers:
        req_headers.update(headers)
    data_bytes: bytes | None = None
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method.upper())
    try:
        with opener.open(req, timeout=max(0.5, float(timeout_s))) as resp:
            body = resp.read(524_288).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            resp_headers = getattr(resp, "headers", None)
            content_type = str(resp_headers.get("Content-Type", "")) if resp_headers else ""
    except urllib.error.HTTPError as exc:
        err_body = exc.read(65_536).decode("utf-8", errors="replace")
        try:
            parsed_err = json.loads(err_body) if err_body.strip() else {"error": err_body}
        except json.JSONDecodeError:
            parsed_err = {"error": redact_sensitive_text(err_body)[:240]}
        return int(exc.code), (parsed_err if isinstance(parsed_err, dict) else {"data": parsed_err})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return 0, {
            "network_error": type(exc).__name__,
            "error": redact_sensitive_text(str(reason))[:200],
        }

    if not body.strip():
        return status_code, {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return status_code, {"content_type": content_type, "html_length": len(body)}
    return status_code, (parsed if isinstance(parsed, dict) else {"data": parsed})


def _safe_http_call(
    http_fn: HttpFn,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout_s: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Invoke ``http_fn`` behind URL/zero-inference validation and transport-error containment."""
    _assert_zero_inference_request(method, url)
    try:
        if http_fn is default_http_json and timeout_s is not None:
            status_code, body = default_http_json(
                method.upper(),
                url,
                payload,
                headers,
                timeout_s=timeout_s,
            )
        else:
            status_code, body = http_fn(method.upper(), url, payload, headers)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {
            "network_error": type(exc).__name__,
            "error": redact_sensitive_text(str(exc))[:200],
        }
    norm_body = dict(body) if isinstance(body, Mapping) else {"data": body}
    return int(status_code), norm_body


def _find_corporate_markers(raw_text: str) -> list[str]:
    """Return corporate indicator tokens matched in ``raw_text`` without retaining raw values."""
    if not raw_text:
        return []
    matched: list[str] = []
    for marker_name, pattern in _CORP_TOKEN_PATTERNS:
        if pattern.search(raw_text) and marker_name not in matched:
            matched.append(marker_name)
    return matched


def _resolve_git_config_files(workspace: Path) -> list[Path]:
    """Locate ``.git/config`` (and git worktree commondir ``config`` if present) strictly on disk."""
    dot_git = workspace / ".git"
    candidates: list[Path] = []
    try:
        if dot_git.is_dir():
            cfg = dot_git / "config"
            if cfg.is_file():
                candidates.append(cfg)
        elif dot_git.is_file():
            lines = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines and lines[0].lower().startswith("gitdir:"):
                raw_gitdir = lines[0].split(":", 1)[1].strip()
                gitdir = Path(raw_gitdir)
                if not gitdir.is_absolute():
                    gitdir = (workspace / gitdir).resolve()
                wt_cfg = gitdir / "config"
                if wt_cfg.is_file():
                    candidates.append(wt_cfg)
                commondir_file = gitdir / "commondir"
                if commondir_file.is_file():
                    rel_common = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                    if rel_common:
                        common_cfg = (gitdir / rel_common).resolve() / "config"
                        if common_cfg.is_file() and common_cfg not in candidates:
                            candidates.append(common_cfg)
    except OSError:
        return candidates
    return candidates


def extract_git_config_remote_urls(config_text: str) -> list[str]:
    """Parse ``[remote ...]`` and ``[url ...]`` remote URLs from ``.git/config`` text in memory."""
    urls: list[str] = []
    in_remote_section = False
    for raw_line in config_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped[1:-1].strip().lower()
            in_remote_section = header.startswith(("remote ", "remote\t", 'remote"', "url "))
            if header.startswith("url ") and '"' in stripped:
                quoted = stripped.split('"', 2)
                if len(quoted) >= 2 and quoted[1].strip():
                    urls.append(quoted[1].strip())
            continue
        if in_remote_section and "=" in stripped:
            key, val = stripped.split("=", 1)
            if key.strip().lower() in ("url", "pushurl", "insteadof", "pushinsteadof"):
                cleaned_val = val.strip().strip('"')
                if cleaned_val:
                    urls.append(cleaned_val)
    return urls


def check_personal_pc_compliance(
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    *,
    lang: str = "en",
) -> dict[str, Any]:
    """Strictly local Personal-PC & Hobby-Only Compliance Gate (0 network calls, 0 file writes)."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    env_view = env if env is not None else os.environ
    workspace = Path(cwd) if cwd is not None else Path.cwd()

    indicators: list[str] = []
    matched_markers: set[str] = set()
    upper_env: dict[str, str] = {
        str(k).upper(): str(v).strip() for k, v in env_view.items() if str(v).strip()
    }

    if any(k.startswith("AMADEUS") for k in upper_env):
        indicators.append("env:AMADEUS_ENV")
        matched_markers.add("amadeus")

    dns_domain = upper_env.get("USERDNSDOMAIN", "")
    if dns_domain:
        indicators.append("env:USERDNSDOMAIN")
        dns_hits = _find_corporate_markers(dns_domain)
        matched_markers.update(dns_hits or ["corp"])

    userdomain = upper_env.get("USERDOMAIN", "")
    computername = upper_env.get("COMPUTERNAME", "")
    if userdomain and userdomain.upper() != computername.upper():
        dom_hits = _find_corporate_markers(userdomain)
        if dom_hits:
            indicators.append("env:USERDOMAIN")
            matched_markers.update(dom_hits)

    for env_key in (
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
    ):
        val = upper_env.get(env_key, "")
        hits = _find_corporate_markers(val)
        if hits:
            indicators.append(f"env:{env_key}")
            matched_markers.update(hits)

    if upper_env.get("ZSCALER_CERT_PATH"):
        indicators.append("env:ZSCALER_CERT_PATH")
        matched_markers.add("zscaler")

    for cfg_path in _resolve_git_config_files(workspace):
        try:
            cfg_text = cfg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for remote_url in extract_git_config_remote_urls(cfg_text):
            remote_hits = _find_corporate_markers(remote_url)
            for hit in remote_hits:
                tag = f"git_remote:{hit}"
                if tag not in indicators:
                    indicators.append(tag)
                matched_markers.add(hit)

    compliance_ok = len(indicators) == 0
    if compliance_ok:
        summary = (
            f"Contrôle local conforme (PC personnel & dépôt hobby vérifiés). {COMPLIANCE_NOTICE_FR}"
            if is_fr
            else f"Local compliance gate passed (personal PC & hobby repository verified). {COMPLIANCE_NOTICE_EN}"
        )
    else:
        joined = ", ".join(indicators)
        summary = (
            f"REFUS DE CONFORMITÉ ({joined}) : environnement professionnel / entreprise détecté "
            f"(0 appel réseau, 0 écriture disque). {COMPLIANCE_NOTICE_BILINGUAL}"
            if is_fr
            else f"COMPLIANCE REFUSED ({joined}): corporate machine or repository indicator detected "
            f"(0 network calls, 0 file writes). {COMPLIANCE_NOTICE_BILINGUAL}"
        )

    report: dict[str, Any] = {
        "compliance_ok": compliance_ok,
        "status": "compliant" if compliance_ok else "refused_compliance",
        "corporate_detected": not compliance_ok,
        "corporate_indicators": indicators,
        "matched_markers": sorted(matched_markers),
        "personal_pc_only": True,
        "amadeus_corporate_prohibition": True,
        "network_calls": 0,
        "file_writes": 0,
        "financial_promise": FINANCIAL_PROMISE,
        "license": "AGPL-3.0",
        "notice_en": COMPLIANCE_NOTICE_EN,
        "notice_fr": COMPLIANCE_NOTICE_FR,
        "notice_bilingual": COMPLIANCE_NOTICE_BILINGUAL,
        "summary": summary,
    }
    report["receipt_sha256"] = compute_receipt_sha256(report)
    return report


def _sanitize_feedback_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only known zero-PII onboarding feedback summary fields (never ``score``)."""
    by_prof_raw = raw.get("by_profile") if isinstance(raw.get("by_profile"), Mapping) else {}
    total_raw = raw.get("total_responses")
    total_val = int(total_raw) if isinstance(total_raw, int) and not isinstance(total_raw, bool) else 0
    mean_raw = raw.get("mean_ease_rating")
    mean_val = (
        round(float(mean_raw), 2)
        if isinstance(mean_raw, (int, float)) and not isinstance(mean_raw, bool)
        else None
    )
    lag_raw = raw.get("zero_game_lag_ratio")
    lag_val = (
        round(float(lag_raw), 4)
        if isinstance(lag_raw, (int, float)) and not isinstance(lag_raw, bool)
        else None
    )
    return {
        "object": "hivemind.onboarding_feedback_summary",
        "total_responses": total_val,
        "mean_ease_rating": mean_val,
        "zero_game_lag_ratio": lag_val,
        "by_profile": {
            "dev": int(by_prof_raw.get("dev", 0)) if isinstance(by_prof_raw.get("dev", 0), int) else 0,
            "gamer": int(by_prof_raw.get("gamer", 0)) if isinstance(by_prof_raw.get("gamer", 0), int) else 0,
        },
    }


def _route_smoke_diagnostic(path: str, group: str, status_code: int, *, lang: str) -> str:
    is_fr = lang == "fr"
    if status_code == 404 and group == "holiday":
        return (
            f"GET {path} a répondu HTTP 404 : la révision Cloud Run déployée est en retard sur `main` "
            "(route Noël manquante ; redéployez `hivemind-openrouter` ou vérifiez `noel_cloud_run_parity_gate.py`)."
            if is_fr
            else f"GET {path} returned HTTP 404: deployed Cloud Run revision lags behind `main` "
            "(missing holiday route; redeploy `hivemind-openrouter` or run `noel_cloud_run_parity_gate.py`)."
        )
    if status_code == 404:
        return (
            f"GET {path} a répondu HTTP 404 (route principale absente sur la passerelle)."
            if is_fr
            else f"GET {path} returned HTTP 404 (core gateway route missing)."
        )
    if status_code == 0:
        return (
            f"GET {path} injoignable (délai dépassé ou erreur réseau/TLS vers la passerelle)."
            if is_fr
            else f"GET {path} unreachable (timeout or network/TLS error contacting gateway)."
        )
    if status_code >= 500:
        return (
            f"GET {path} en échec serveur passerelle (HTTP {status_code}) ; vérifiez l'état Cloud Run."
            if is_fr
            else f"GET {path} failed with gateway server error (HTTP {status_code}); check Cloud Run health."
        )
    return (
        f"GET {path} a répondu HTTP {status_code} (attendu HTTP 200)."
        if is_fr
        else f"GET {path} returned HTTP {status_code} (expected HTTP 200)."
    )


def verify_holiday_routes(
    gateway: str = DEFAULT_GATEWAY,
    *,
    http_fn: HttpFn = default_http_json,
    lang: str = "en",
    api_key: str | None = None,
    credentials_path: Path | str | None = None,
    headers: dict[str, str] | None = None,
    home: Path | None = None,
) -> dict[str, Any]:
    """Zero-inference Cloud Run Holiday Route Smoke Gate with masked keys, ``~`` paths, and SHA-256 receipt."""
    lang_code = normalize_lang(lang)
    clean_gw = validate_gateway_url(gateway)

    cred_file = Path(credentials_path) if credentials_path is not None else _default_credentials_path()
    saved_creds = load_credentials(cred_file)
    resolved_key = (api_key or (str(saved_creds.get("api_key", "")) if saved_creds else "")).strip()
    masked_key = mask_api_key(resolved_key)
    redacted_cred_path = redact_home_path(cred_file, home=home)

    route_rows: list[dict[str, Any]] = []
    http_statuses: dict[str, int] = {}
    missing_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    feedback_summary: dict[str, Any] | None = None

    for path, group in HOLIDAY_SMOKE_ROUTES:
        code, body = _safe_http_call(http_fn, "GET", f"{clean_gw}{path}", None, headers)
        http_statuses[path] = code
        ok = code == 200
        diag = ""

        if ok and path == "/health":
            if "status" in body and body.get("status") != "ok":
                ok = False
                diag = (
                    "GET /health a répondu HTTP 200 mais `status` != 'ok'."
                    if lang_code == "fr"
                    else "GET /health returned HTTP 200 with non-ok status payload."
                )
        elif ok and path == "/v1/hivemind/onboarding/feedback/summary":
            feedback_summary = _sanitize_feedback_summary(body)

        if not ok:
            failed_routes.append(path)
            if code == 404:
                missing_routes.append(path)
            if not diag:
                diag = _route_smoke_diagnostic(path, group, code, lang=lang_code)
            diagnostics.append(redact_sensitive_text(diag, home=home))

        route_rows.append(
            {
                "path": path,
                "method": "GET",
                "group": group,
                "http_status": code,
                "ok": ok,
                "diagnostic": redact_sensitive_text(diag, home=home),
            }
        )

    passed_count = sum(1 for row in route_rows if row["ok"])
    total_count = len(route_rows)
    smoke_ok = passed_count == total_count
    holiday_404s = [p for p in missing_routes if p != "/health"]
    cloud_run_lags_main = bool(http_statuses.get("/health") == 200 and holiday_404s)

    if cloud_run_lags_main:
        joined_404 = ", ".join(holiday_404s)
        stale_banner = (
            f"Diagnostic Cloud Run obsolète : GET /health répond HTTP 200 mais {len(holiday_404s)} route(s) "
            f"Noël renvoient HTTP 404 ({joined_404}) — redéployez Cloud Run depuis `main`."
            if lang_code == "fr"
            else f"Stale Cloud Run diagnostic: GET /health is HTTP 200 while {len(holiday_404s)} holiday "
            f"route(s) returned HTTP 404 ({joined_404}) — redeploy Cloud Run from `main`."
        )
        diagnostics.insert(0, stale_banner)

    report: dict[str, Any] = {
        "object": "hivemind.quickstart.holiday_smoke_receipt",
        "gateway": clean_gw,
        "lang": lang_code,
        "smoke_ok": smoke_ok,
        "all_routes_ok": smoke_ok,
        "routes_passed_count": passed_count,
        "routes_total_count": total_count,
        "http_statuses": http_statuses,
        "missing_routes": missing_routes,
        "failed_routes": failed_routes,
        "cloud_run_lags_main": cloud_run_lags_main,
        "redeploy_recommended": bool(cloud_run_lags_main or any(c >= 500 for c in http_statuses.values())),
        "zero_inference_verified": True,
        "masked_api_key": masked_key,
        "credentials_file": redacted_cred_path,
        "feedback_summary": feedback_summary,
        "diagnostics": diagnostics,
        "routes": route_rows,
        "compliance_notice": COMPLIANCE_NOTICE_FR if lang_code == "fr" else COMPLIANCE_NOTICE_EN,
        "financial_promise": FINANCIAL_PROMISE,
    }
    report["receipt_sha256"] = compute_receipt_sha256(report)
    return report


def derive_zero_game_lag(
    *,
    had_game_lag: bool = False,
    yield_elapsed_s: float | None = None,
    gpu_temp_c: float | None = None,
    quiet_temp_ceiling_c: float = GAMER_WHISPER_MAX_TEMP_C,
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
) -> tuple[bool, bool, bool]:
    """Derive ``(zero_game_lag, yield_slo_ok, quiet_fan_ok)`` honestly from ``<1.5s`` yield & ``<=70C`` status."""
    yield_slo_ok = (yield_elapsed_s is None) or (float(yield_elapsed_s) < float(yield_budget_s))
    quiet_fan_ok = (gpu_temp_c is None) or (float(gpu_temp_c) <= float(quiet_temp_ceiling_c))
    zero_game_lag = (not bool(had_game_lag)) and yield_slo_ok and quiet_fan_ok
    return zero_game_lag, yield_slo_ok, quiet_fan_ok


def _normalize_platform_token(raw_platform: str | None) -> str:
    candidate = (raw_platform if raw_platform is not None else platform.system()).strip().lower()
    candidate = re.sub(r"[^a-z0-9_-]+", "", candidate)[:24].rstrip("_-")
    if candidate and _PLATFORM_RE.fullmatch(candidate):
        return candidate
    return "unknown"


def submit_post_setup_telemetry(
    gateway: str = DEFAULT_GATEWAY,
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    had_game_lag: bool = False,
    yield_elapsed_s: float | None = None,
    gpu_temp_c: float | None = None,
    quiet_temp_ceiling_c: float = GAMER_WHISPER_MAX_TEMP_C,
    yield_budget_s: float = GAMER_WHISPER_YIELD_BUDGET_S,
    platform_name: str | None = None,
    note: str = "",
    headers: dict[str, str] | None = None,
    http_fn: HttpFn = default_http_json,
    timeout_s: float = TELEMETRY_TIMEOUT_S,
    lang: str = "en",
    home: Path | None = None,
) -> dict[str, Any]:
    """Submit timeout-safe, non-blocking zero-PII CES & Steam zero-game-lag telemetry."""
    lang_code = normalize_lang(lang)
    is_fr = lang_code == "fr"
    clean_gw = validate_gateway_url(gateway)

    norm_profile = (profile or "").strip().lower()
    if norm_profile not in ("dev", "gamer"):
        raise ValueError(f"Unsupported profile {profile!r}; expected 'dev' or 'gamer'.")
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise ValueError("ease_rating must be an integer between 1 and 5.")

    zero_game_lag, yield_slo_ok, quiet_fan_ok = derive_zero_game_lag(
        had_game_lag=had_game_lag,
        yield_elapsed_s=yield_elapsed_s,
        gpu_temp_c=gpu_temp_c,
        quiet_temp_ceiling_c=quiet_temp_ceiling_c,
        yield_budget_s=yield_budget_s,
    )
    clean_platform = _normalize_platform_token(platform_name)
    clean_note = redact_sensitive_text(str(note or ""), home=home)[:240]

    payload: dict[str, Any] = {
        "profile": norm_profile,
        "ease_rating": int(ease_rating),
        "zero_game_lag": bool(zero_game_lag),
        "platform": clean_platform,
        "note": clean_note,
    }

    status_code, body = _safe_http_call(
        http_fn,
        "POST",
        f"{clean_gw}/v1/hivemind/onboarding/feedback",
        payload,
        headers,
        timeout_s=timeout_s,
    )

    telemetry_recorded = status_code in (200, 201)
    stale_cloud_run = status_code == 404
    degraded_reason: str | None = None
    diagnostic = ""
    sanitized_summary: dict[str, Any] | None = None

    if telemetry_recorded:
        sanitized_summary = _sanitize_feedback_summary(body)
        diagnostic = (
            "Télémétrie CES post-installation anonyme enregistrée."
            if is_fr
            else "Anonymous post-setup CES telemetry recorded."
        )
    else:
        if status_code == 404:
            degraded_reason = "stale_cloud_run_404"
            diagnostic = (
                "Télémétrie ignorée sans bloquer : Cloud Run en retard sur `main` (HTTP 404)."
                if is_fr
                else "Telemetry skipped without blocking setup: Cloud Run revision lags behind `main` (HTTP 404)."
            )
        elif status_code == 429:
            degraded_reason = "rate_limited_429"
            diagnostic = (
                "Télémétrie ignorée sans bloquer : limite de débit atteinte (HTTP 429)."
                if is_fr
                else "Telemetry skipped without blocking setup: gateway rate limit reached (HTTP 429)."
            )
        elif status_code >= 500:
            degraded_reason = "gateway_5xx"
            diagnostic = (
                f"Télémétrie ignorée sans bloquer : erreur passerelle (HTTP {status_code})."
                if is_fr
                else f"Telemetry skipped without blocking setup: gateway error (HTTP {status_code})."
            )
        elif status_code == 0:
            degraded_reason = "network_timeout_or_unreachable"
            diagnostic = (
                "Télémétrie ignorée sans bloquer : délai réseau dépassé ou passerelle injoignable."
                if is_fr
                else "Telemetry skipped without blocking setup: network timeout or gateway unreachable."
            )
        else:
            degraded_reason = f"http_{status_code}"
            diagnostic = (
                f"Télémétrie ignorée sans bloquer (HTTP {status_code})."
                if is_fr
                else f"Telemetry skipped without blocking setup (HTTP {status_code})."
            )

    result: dict[str, Any] = {
        "object": "hivemind.quickstart.post_setup_telemetry",
        "telemetry_recorded": telemetry_recorded,
        "http_status": status_code,
        "profile": norm_profile,
        "ease_rating": int(ease_rating),
        "zero_game_lag": bool(zero_game_lag),
        "yield_slo_ok": bool(yield_slo_ok),
        "quiet_fan_ok": bool(quiet_fan_ok),
        "platform": clean_platform,
        "stale_cloud_run": stale_cloud_run,
        "degraded_reason": degraded_reason,
        "diagnostic": redact_sensitive_text(diagnostic, home=home),
        "summary": sanitized_summary,
    }
    result["receipt_sha256"] = compute_receipt_sha256(result)
    return result


def save_credentials(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    tmp.replace(path)


def load_credentials(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and str(data.get("api_key", "")).startswith("hm-"):
        return data
    return None


def probe_local_gpu(*, runner_fn: RunnerFn | None = None) -> GpuSnapshot:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        if runner_fn is not None:
            rc, stdout = runner_fn(cmd, 5.0)
        else:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                shell=False,
            )
            rc, stdout = proc.returncode, proc.stdout
        if rc == 0 and stdout.strip():
            parts = [p.strip() for p in stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 5:
                return GpuSnapshot(
                    present=True,
                    name=parts[0],
                    total_mib=max(0, int(float(parts[1]))),
                    free_mib=max(0, int(float(parts[2]))),
                    temp_c=float(parts[3]),
                    util_pct=float(parts[4]),
                )
    except (OSError, ValueError, subprocess.SubprocessError):
        return GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0)
    return GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0)


def list_running_process_names(
    *,
    os_name: str | None = None,
    runner_fn: RunnerFn | None = None,
) -> set[str]:
    names: set[str] = set()
    eff_os = os_name if os_name is not None else os.name
    try:
        if eff_os == "nt":
            cmd = ["tasklist", "/FO", "CSV", "/NH"]
            if runner_fn is not None:
                _, stdout = runner_fn(cmd, 5.0)
            else:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                    shell=False,
                )
                stdout = proc.stdout
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith('"') and '","' in line:
                    names.add(line.split('","', 1)[0].strip('"').lower())
        else:
            cmd = ["ps", "-eo", "comm="]
            if runner_fn is not None:
                _, stdout = runner_fn(cmd, 5.0)
            else:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                    shell=False,
                )
                stdout = proc.stdout
            for line in stdout.splitlines():
                cleaned = line.strip().lower()
                if cleaned:
                    names.add(cleaned)
    except (OSError, subprocess.SubprocessError):
        return names
    return names


def get_os_idle_seconds() -> float:
    if sys.platform != "win32":
        return 999.0
    try:
        class _LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]

        lii = _LastInputInfo()
        lii.cbSize = ctypes.sizeof(_LastInputInfo)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):  # type: ignore[attr-defined]
            tick = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
            elapsed_ms = (tick - lii.dwTime) & 0xFFFFFFFF
            return round(elapsed_ms / 1000.0, 3)
    except Exception:
        return 999.0
    return 999.0


def detect_active_game(running: set[str], extra_games: list[str] | None = None) -> str | None:
    watch = set(DEFAULT_GAME_EXECUTABLES)
    for g in extra_games or []:
        cleaned = g.strip().lower()
        if cleaned:
            watch.add(cleaned)
            watch.add(cleaned[:15])
    for proc_name in sorted(running):
        if proc_name in watch:
            return proc_name
    return None


def evaluate_gamer_whisper_guard(
    gpu: GpuSnapshot,
    *,
    running_procs: set[str],
    extra_games: list[str] | None = None,
    reserve_mib: int = 2048,
    hot_c: float = 70.0,
    max_external_util_pct: float = 15.0,
    worker_running: bool = False,
) -> dict[str, Any]:
    """Decide whether a gaming PC is quiet and free enough to run or keep running a lightweight round."""
    active_game = detect_active_game(running_procs, extra_games)
    if active_game is not None:
        return {"allow": False, "reason": f"game_active:{active_game}", "active_game": active_game}
    if not gpu.present or gpu.total_mib <= 0:
        return {"allow": False, "reason": "no_nvidia_gpu_detected", "active_game": None}
    if gpu.temp_c >= hot_c:
        return {"allow": False, "reason": f"thermal_whisper_cap:{gpu.temp_c:.0f}C>={hot_c:.0f}C", "active_game": None}
    if gpu.free_mib < reserve_mib:
        return {"allow": False, "reason": f"vram_reserve_guard:{gpu.free_mib}MiB<{reserve_mib}MiB", "active_game": None}
    if not worker_running and gpu.util_pct > max_external_util_pct:
        return {
            "allow": False,
            "reason": f"external_gpu_busy:{gpu.util_pct:.0f}%>{max_external_util_pct:.0f}%",
            "active_game": None,
        }
    return {"allow": True, "reason": "ready_gamer_whisper", "active_game": None}


def build_continue_yaml(gateway: str, api_key: str) -> str:
    return (
        "# HiveMind personal hobby/open-source coding configuration for VS Code (Continue.dev)\n"
        "# Personal projects ONLY — never send corporate or confidential source code.\n"
        "name: HiveMind Holiday Lab\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: HiveMind Coder\n"
        "    provider: openai\n"
        "    model: hivemind/coder\n"
        f"    apiBase: {gateway}/v1\n"
        f"    apiKey: {api_key}\n"
        "    roles:\n"
        "      - chat\n"
        "      - edit\n"
        "  - name: HiveMind Bee-Stars (Science)\n"
        "    provider: openai\n"
        "    model: hivemind/bee-stars\n"
        f"    apiBase: {gateway}/v1\n"
        f"    apiKey: {api_key}\n"
        "    roles:\n"
        "      - chat\n"
    )


def terminate_process_tree_fast(proc: subprocess.Popen[Any], timeout_s: float = 1.2) -> float:
    """Stop a worker process tree immediately so 100% of GPU VRAM returns to the owner within < 1.5 s."""
    started = time.monotonic()
    if proc.poll() is not None:
        return round(time.monotonic() - started, 4)
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=timeout_s,
                check=False,
                shell=False,
            )
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        try:
            proc.wait(timeout=max(0.2, timeout_s * 0.6))
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.wait(timeout=0.5)
    except (OSError, subprocess.SubprocessError):
        return round(time.monotonic() - started, 4)
    return round(time.monotonic() - started, 4)


def ensure_connected_account(
    *,
    gateway: str,
    credentials_path: Path,
    handle: str | None,
    role: str,
    explicit_api_key: str | None,
    http_fn: HttpFn = default_http_json,
    env: Mapping[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    clean_gw = validate_gateway_url(gateway)
    env_view = env if env is not None else os.environ
    api_key = (explicit_api_key or env_view.get("HIVEMIND_API_KEY") or "").strip()
    saved = load_credentials(credentials_path)
    if not api_key and saved and saved.get("gateway") == clean_gw:
        api_key = str(saved["api_key"])

    created = False
    resolved_handle = handle or (saved.get("handle") if saved else None) or ""
    if not api_key:
        base_handle = sanitize_handle(handle or f"hm-{platform.node() or 'home'}")
        status, body = _safe_http_call(
            http_fn,
            "POST",
            f"{clean_gw}/v1/hivemind/register",
            {"handle": base_handle, "role": role},
            headers,
        )
        if status == 409 and not handle:
            retry_handle = sanitize_handle(f"{base_handle[:38]}-{secrets.token_hex(2)}")
            status, body = _safe_http_call(
                http_fn,
                "POST",
                f"{clean_gw}/v1/hivemind/register",
                {"handle": retry_handle, "role": role},
                headers,
            )
            base_handle = retry_handle
        if status != 200 or not str(body.get("api_key", "")).startswith("hm-"):
            msg = body.get("error") or body
            raise RuntimeError(f"Registration refused (HTTP {status}): {msg}")
        api_key = str(body["api_key"])
        resolved_handle = base_handle
        created = True

    me_headers = dict(headers or {})
    me_headers["Authorization"] = f"Bearer {api_key}"
    me_status, me_body = _safe_http_call(
        http_fn,
        "GET",
        f"{clean_gw}/v1/hivemind/me",
        None,
        me_headers,
    )
    if me_status != 200:
        raise RuntimeError(f"API key check failed on /v1/hivemind/me (HTTP {me_status}): {me_body}")

    models_status, models_body = _safe_http_call(http_fn, "GET", f"{clean_gw}/v1/models", None, headers)
    models = [m.get("id") for m in (models_body.get("data") or []) if isinstance(m, dict) and m.get("id")]
    if models_status != 200:
        models = []

    record = {
        "account_id": me_body.get("account_id", ""),
        "api_key": api_key,
        "created_here": created,
        "gateway": clean_gw,
        "handle": me_body.get("handle") or resolved_handle,
        "role": me_body.get("role") or role,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    save_credentials(credentials_path, record)
    return {"account": record, "balance": me_body.get("balance", 0.0), "models": models}


def run_gamer_whisper_session(
    *,
    gateway: str,
    api_key: str,
    extra_games: list[str],
    reserve_mib: int,
    hot_c: float,
    cool_c: float,
    once: bool,
    gpu_probe_fn: Callable[[], GpuSnapshot] = probe_local_gpu,
    proc_list_fn: Callable[[], set[str]] = list_running_process_names,
    popen_fn: Callable[..., subprocess.Popen[Any]] | None = None,
    poll_interval_s: float = 0.5,
) -> dict[str, Any]:
    clean_gw = validate_gateway_url(gateway)
    gpu = gpu_probe_fn()
    max_temp_c: float | None = round(float(gpu.temp_c), 2) if gpu.present else None
    preflight_decision = evaluate_gamer_whisper_guard(
        gpu,
        running_procs=proc_list_fn(),
        extra_games=extra_games,
        reserve_mib=reserve_mib,
        hot_c=hot_c,
        worker_running=False,
    )
    if not preflight_decision["allow"]:
        return {
            "status": "yielded_before_start",
            **preflight_decision,
            "yield_elapsed_s": 0.0,
            "gpu_temp_c": max_temp_c,
        }

    agent_script = Path(__file__).resolve().with_name("hivemind_opentrainer_agent.py")
    cmd = [
        sys.executable,
        str(agent_script),
        "--gateway",
        clean_gw,
        "--max-vram-gb",
        "4.5",
        "--train",
    ]
    if once:
        cmd.append("--once")
    env = dict(os.environ)
    env["HIVEMIND_API_KEY"] = api_key
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = repo_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    launcher = popen_fn or subprocess.Popen
    popen_kwargs: dict[str, Any] = {"env": env}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = launcher(cmd, **popen_kwargs)

    while proc.poll() is None:
        time.sleep(max(0.05, poll_interval_s))
        live_gpu = gpu_probe_fn()
        if live_gpu.present:
            max_temp_c = (
                round(float(live_gpu.temp_c), 2)
                if max_temp_c is None
                else round(max(max_temp_c, float(live_gpu.temp_c)), 2)
            )
        live_decision = evaluate_gamer_whisper_guard(
            live_gpu,
            running_procs=proc_list_fn(),
            extra_games=extra_games,
            reserve_mib=reserve_mib,
            hot_c=hot_c,
            worker_running=True,
        )
        if not live_decision["allow"]:
            elapsed_s = terminate_process_tree_fast(proc)
            return {
                "status": "yielded_to_owner",
                **live_decision,
                "cool_resume_c": cool_c,
                "yield_elapsed_s": elapsed_s,
                "gpu_temp_c": max_temp_c,
            }

    return {
        "status": "round_completed" if proc.returncode == 0 else "worker_exited",
        "returncode": proc.returncode,
        "gpu_temp_c": max_temp_c,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HiveMind 60-second Personal Quick-Connect & Gamer-Whisper Assistant (0.00 EUR, personal PCs only)."
    )
    p.add_argument("--gateway", default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY))
    p.add_argument("--handle", default="", help="Desired handle (2-48 chars, e.g. alex-rtx4070).")
    p.add_argument("--role", default="prosumer", choices=("consumer", "producer", "prosumer", "trainer"))
    p.add_argument("--api-key", default="", help="Existing hm-... API key (optional).")
    p.add_argument("--credentials-file", default=str(_default_credentials_path()))
    p.add_argument("--write-continue-config", action="store_true", help="Write ~/.continue/hivemind_provider.yaml.")
    p.add_argument("--continue-config-path", default=str(_default_continue_snippet_path()))
    p.add_argument("--lang", default="en", help="Output and diagnostic language ('en' or 'fr').")
    p.add_argument(
        "--smoke-routes",
        action="store_true",
        help="Run zero-inference Cloud Run holiday route smoke gate and exit.",
    )
    p.add_argument(
        "--require-holiday-routes",
        action="store_true",
        help="Fail closed (exit 2) before setup when required Cloud Run holiday routes fail or return 404/5xx.",
    )
    p.add_argument("--gamer-check", action="store_true", help="Inspect GPU, temperature, and running games only.")
    p.add_argument("--gamer-whisper", action="store_true", help="Run quiet GPU contributor (<70C, <1.5s game stop).")
    p.add_argument("--game", action="append", default=[], help="Extra game executable name to auto-stop on.")
    p.add_argument("--reserve-mib", type=int, default=2048, help="Free VRAM MiB reserved for desktop/games.")
    p.add_argument("--hot-c", type=float, default=70.0, help="Quiet fan temperature cap in Celsius (default: 70).")
    p.add_argument("--cool-c", type=float, default=64.0, help="Cool-down temperature in Celsius (default: 64).")
    p.add_argument("--once", action="store_true", help="Run a single quiet round and exit.")
    p.add_argument(
        "--post-setup-telemetry",
        action="store_true",
        help="Submit non-blocking anonymous CES & Steam zero-game-lag telemetry after connect or --gamer-whisper.",
    )
    p.add_argument("--feedback", action="store_true", help="Submit anonymous usability feedback (1..5).")
    p.add_argument("--ease-rating", type=int, default=5, help="Ease of setup rating from 1 (hard) to 5 (instant).")
    p.add_argument("--had-game-lag", action="store_true", help="Report that a game stuttered or fans were noisy.")
    p.add_argument("--profile", default="dev", choices=("dev", "gamer"))
    p.add_argument("--note", default="", help="Optional short feedback note (no personal data).")
    p.add_argument("--uninstall", action="store_true", help="Delete local saved HiveMind credentials and snippet.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return p


def _render_smoke_cli_output(smoke: Mapping[str, Any]) -> str:
    is_fr = smoke.get("lang") == "fr"
    header = (
        "HIVEMIND NOËL — SMOKE GATE CLOUD RUN (ZÉRO INFÉRENCE • 0,00 EUR / 0,00 USD)"
        if is_fr
        else "HIVEMIND HOLIDAY — CLOUD RUN ROUTE SMOKE GATE (ZERO INFERENCE • 0.00 EUR / 0.00 USD)"
    )
    lines = [
        "====================================================================",
        f"  {header}",
        "====================================================================",
        f"  [Rule] {smoke.get('compliance_notice')}",
        f"  Gateway          : {smoke.get('gateway')}",
        f"  Credentials      : {smoke.get('credentials_file')} (key {smoke.get('masked_api_key')})",
        f"  Routes verified  : {smoke.get('routes_passed_count')}/{smoke.get('routes_total_count')} "
        f"(smoke_ok={smoke.get('smoke_ok')})",
    ]
    for row in smoke.get("routes") or ():
        mark = "OK" if row.get("ok") else "FAIL"
        lines.append(f"  [{mark:<4}] GET {row.get('path'):<42} -> HTTP {row.get('http_status')}")
    for diag in smoke.get("diagnostics") or ():
        lines.append(f"  [DIAG] {diag}")
    lines.append(f"  SHA-256 Receipt  : {smoke.get('receipt_sha256')}")
    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    home: Path | None = None,
    headers: dict[str, str] | None = None,
    gpu_probe_fn: Callable[[], GpuSnapshot] = probe_local_gpu,
    proc_list_fn: Callable[[], set[str]] = list_running_process_names,
    popen_fn: Callable[..., subprocess.Popen[Any]] | None = None,
    poll_interval_s: float = 0.5,
    out_stream: Any = None,
    err_stream: Any = None,
) -> int:
    eff_out = out_stream if out_stream is not None else sys.stdout
    eff_err = err_stream if err_stream is not None else sys.stderr
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    cred_path = Path(args.credentials_file)
    cont_path = Path(args.continue_config_path)

    if args.uninstall:
        removed: list[str] = []
        for target in (cred_path, cont_path):
            if target.is_file():
                target.unlink()
                removed.append(redact_home_path(target, home=home))
        print(json.dumps({"status": "uninstalled", "removed_files": removed}, indent=2), file=eff_out)
        return 0

    try:
        lang_code = normalize_lang(args.lang)
    except ValueError as exc:
        print(f"Error: {exc}", file=eff_err)
        return 2

    eff_env = env if env is not None else os.environ
    eff_cwd = Path(cwd) if cwd is not None else Path.cwd()
    compliance = check_personal_pc_compliance(env=eff_env, cwd=eff_cwd, lang=lang_code)
    if not compliance["compliance_ok"]:
        print(compliance["summary"], file=eff_err)
        if args.json:
            print(json.dumps(compliance, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(f"compliance_ok=False | {compliance['summary']}", file=eff_out)
        return 2

    raw_gateway = (
        eff_env.get("HIVEMIND_GATEWAY_URL", args.gateway)
        if (env is not None and not any(tok == "--gateway" or tok.startswith("--gateway=") for tok in raw_argv))
        else args.gateway
    )
    try:
        gateway = validate_gateway_url(raw_gateway)
    except ValueError as exc:
        print(f"Error: {redact_sensitive_text(str(exc), home=home)}", file=eff_err)
        return 2

    if (args.feedback or args.post_setup_telemetry) and not (1 <= int(args.ease_rating) <= 5):
        print("Error: --ease-rating must be between 1 and 5.", file=eff_err)
        return 2

    explicit_or_env_key = (args.api_key or eff_env.get("HIVEMIND_API_KEY") or "").strip() or None

    if args.smoke_routes:
        smoke = verify_holiday_routes(
            gateway,
            http_fn=http_fn,
            lang=lang_code,
            api_key=explicit_or_env_key,
            credentials_path=cred_path,
            headers=headers,
            home=home,
        )
        if args.json:
            print(json.dumps(smoke, indent=2, ensure_ascii=False), file=eff_out)
        else:
            print(_render_smoke_cli_output(smoke), file=eff_out)
        if not smoke["smoke_ok"]:
            for diag in smoke["diagnostics"]:
                print(diag, file=eff_err)
            return 2
        return 0

    if args.gamer_check:
        gpu = gpu_probe_fn()
        decision = evaluate_gamer_whisper_guard(
            gpu,
            running_procs=proc_list_fn(),
            extra_games=args.game,
            reserve_mib=args.reserve_mib,
            hot_c=args.hot_c,
        )
        out = {
            "gpu": {
                "present": gpu.present,
                "name": gpu.name,
                "total_mib": gpu.total_mib,
                "free_mib": gpu.free_mib,
                "temp_c": gpu.temp_c,
                "util_pct": gpu.util_pct,
            },
            "os_idle_seconds": get_os_idle_seconds(),
            "gamer_whisper": decision,
        }
        print(json.dumps(out, indent=2), file=eff_out)
        return 0 if decision["allow"] else 2

    required_smoke: dict[str, Any] | None = None
    if args.require_holiday_routes:
        required_smoke = verify_holiday_routes(
            gateway,
            http_fn=http_fn,
            lang=lang_code,
            api_key=explicit_or_env_key,
            credentials_path=cred_path,
            headers=headers,
            home=home,
        )
        if not required_smoke["smoke_ok"]:
            for diag in required_smoke["diagnostics"]:
                print(diag, file=eff_err)
            if args.json:
                fail_doc: dict[str, Any] = {
                    "status": "holiday_routes_failed",
                    "smoke_ok": False,
                    "compliance_ok": True,
                    "holiday_routes": required_smoke,
                }
                fail_doc["receipt_sha256"] = compute_receipt_sha256(fail_doc)
                print(json.dumps(fail_doc, indent=2, ensure_ascii=False), file=eff_out)
            else:
                print(_render_smoke_cli_output(required_smoke), file=eff_out)
            return 2

    if args.feedback:
        status, body = _safe_http_call(
            http_fn,
            "POST",
            f"{gateway}/v1/hivemind/onboarding/feedback",
            {
                "profile": args.profile,
                "ease_rating": int(args.ease_rating),
                "zero_game_lag": not bool(args.had_game_lag),
                "platform": _normalize_platform_token(None),
                "note": redact_sensitive_text(str(args.note or ""), home=home)[:240],
            },
            headers,
        )
        if status not in (200, 201):
            print(
                f"Feedback submission failed (HTTP {status}): {redact_sensitive_text(str(body), home=home)}",
                file=eff_err,
            )
            return 1
        print(
            json.dumps({"status": "feedback_recorded", "summary": _sanitize_feedback_summary(body)}, indent=2),
            file=eff_out,
        )
        return 0

    try:
        connected = ensure_connected_account(
            gateway=gateway,
            credentials_path=cred_path,
            handle=args.handle or None,
            role=args.role,
            explicit_api_key=args.api_key or None,
            http_fn=http_fn,
            env=eff_env,
            headers=headers,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Connection error: {redact_sensitive_text(str(exc), home=home)}", file=eff_err)
        return 1

    acct = connected["account"]
    yaml_snippet = build_continue_yaml(gateway, acct["api_key"])
    if args.write_continue_config:
        cont_path.parent.mkdir(parents=True, exist_ok=True)
        cont_path.write_text(yaml_snippet, encoding="utf-8")
        if os.name != "nt":
            os.chmod(cont_path, 0o600)

    explicit_profile_flag = any(tok == "--profile" or tok.startswith("--profile=") for tok in raw_argv)
    eff_profile = args.profile if explicit_profile_flag else ("gamer" if args.gamer_whisper else args.profile)

    if args.gamer_whisper:
        result = run_gamer_whisper_session(
            gateway=gateway,
            api_key=acct["api_key"],
            extra_games=args.game,
            reserve_mib=args.reserve_mib,
            hot_c=args.hot_c,
            cool_c=args.cool_c,
            once=args.once,
            gpu_probe_fn=gpu_probe_fn,
            proc_list_fn=proc_list_fn,
            popen_fn=popen_fn,
            poll_interval_s=poll_interval_s,
        )
        if required_smoke is not None:
            result["holiday_routes"] = required_smoke
        if args.post_setup_telemetry:
            telemetry = submit_post_setup_telemetry(
                gateway,
                profile=eff_profile,
                ease_rating=int(args.ease_rating),
                had_game_lag=bool(args.had_game_lag),
                yield_elapsed_s=result.get("yield_elapsed_s"),
                gpu_temp_c=result.get("gpu_temp_c"),
                note=args.note,
                headers=headers,
                http_fn=http_fn,
                lang=lang_code,
                home=home,
            )
            result["telemetry_recorded"] = telemetry["telemetry_recorded"]
            result["post_setup_telemetry"] = telemetry
        result["receipt_sha256"] = compute_receipt_sha256(result)
        print(json.dumps(result, indent=2, ensure_ascii=False), file=eff_out)
        return 0 if result.get("status") in ("round_completed", "yielded_to_owner", "yielded_before_start") else 1

    telemetry_res: dict[str, Any] | None = None
    if args.post_setup_telemetry:
        telemetry_res = submit_post_setup_telemetry(
            gateway,
            profile=eff_profile,
            ease_rating=int(args.ease_rating),
            had_game_lag=bool(args.had_game_lag),
            note=args.note,
            headers=headers,
            http_fn=http_fn,
            lang=lang_code,
            home=home,
        )

    redacted_cred = redact_home_path(cred_path, home=home)
    redacted_cont = redact_home_path(cont_path, home=home) if args.write_continue_config else None
    masked_key = mask_api_key(acct["api_key"])

    if args.json:
        json_out: dict[str, Any] = {
            "status": "connected",
            "compliance_ok": True,
            "handle": acct["handle"],
            "role": acct["role"],
            "masked_api_key": masked_key,
            "credentials_file": redacted_cred,
            "models": connected["models"],
            "continue_config_written": redacted_cont,
        }
        if required_smoke is not None:
            json_out["holiday_routes"] = required_smoke
        if telemetry_res is not None:
            json_out["telemetry_recorded"] = telemetry_res["telemetry_recorded"]
            json_out["post_setup_telemetry"] = telemetry_res
        json_out["receipt_sha256"] = compute_receipt_sha256(json_out)
        print(json.dumps(json_out, indent=2, ensure_ascii=False), file=eff_out)
        return 0

    is_fr = lang_code == "fr"
    masked_yaml = build_continue_yaml(gateway, masked_key)
    cli_receipt = compute_receipt_sha256(
        {
            "status": "connected",
            "gateway": gateway,
            "handle": acct["handle"],
            "role": acct["role"],
            "masked_api_key": masked_key,
            "credentials_file": redacted_cred,
            "telemetry_recorded": telemetry_res["telemetry_recorded"] if telemetry_res else None,
        }
    )
    print("====================================================================", file=eff_out)
    if is_fr:
        print("  HIVEMIND LABO DE NOËL — CONNECTÉ (PC PERSONNEL • 0,00 EUR)", file=eff_out)
    else:
        print("  HIVEMIND HOLIDAY LAB — CONNECTED (PERSONAL HOME PC ONLY • 0.00 EUR)", file=eff_out)
    print("====================================================================", file=eff_out)
    print(f"  [Rule] {COMPLIANCE_NOTICE_FR if is_fr else COMPLIANCE_NOTICE_EN}", file=eff_out)
    print(f"  Handle           : {acct['handle']} ({acct['role']})", file=eff_out)
    print(f"  Gateway          : {gateway}", file=eff_out)
    print(f"  API key (masked) : {masked_key}", file=eff_out)
    print(f"  Credentials saved: {redacted_cred}", file=eff_out)
    print(f"  Models online    : {', '.join(connected['models']) or 'verified'}", file=eff_out)
    if redacted_cont:
        print(f"  VS Code snippet  : {redacted_cont}", file=eff_out)
    if telemetry_res is not None:
        print(
            f"  CES telemetry    : recorded={telemetry_res['telemetry_recorded']} ({telemetry_res['diagnostic']})",
            file=eff_out,
        )
    print(f"  SHA-256 Receipt  : {cli_receipt}", file=eff_out)
    print(
        "\n--- VS Code (Continue.dev) snippet (key masked in terminal; use --write-continue-config) ---",
        file=eff_out,
    )
    print(masked_yaml.rstrip(), file=eff_out)
    print("\n--- Clean uninstall anytime ---", file=eff_out)
    print("  python3 scripts/hivemind_quickstart.py --uninstall", file=eff_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
