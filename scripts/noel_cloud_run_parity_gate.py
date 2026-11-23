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
     `noel_canary_cohort_rehearsal.py`) without ever calling `/v1/chat/completions`
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
  3. Dry-Run Cloud Run Redeploy & Rollback Verifiers (`build_cloud_run_redeploy_plan`,
     `build_cloud_run_rollback_plan`, `--print-redeploy-plan`): inspects `deploy/gcp_openrouter/deploy.sh`
     invariants (`--min-instances 0`, `--max-instances 1`, `--no-cpu-throttling`, Litestream
     `HIVEMIND_ROUTER_REPLICA_URL`, Secret Manager `HIVEMIND_ROUTER_SECRET`), emits a sanitized
     `deploy_manifest`, and renders FinOps-safe redeploy and `gcloud run services update-traffic`
     rollback commands without ever executing `gcloud` or mutating cloud state.
  4. Strictly Local Personal-PC & Amadeus Compliance Guard (`check_personal_pc_compliance`,
     `--enforce-personal-pc`): blocks corporate environments and corporate git remotes locally
     with exit code 2 (`0` network calls, `0` file writes, `0` env/path value leaks).

Personal Home Use & Non-Monetary Compliance Rule (Amadeus / Personal-PC Rule):
  - 100% personal home PCs & personal hobby/open-source projects only (never corporate hardware).
  - 0.00 EUR / 0.00 USD: voluntary reciprocal citizen science with symbolic non-monetary points under AGPL-3.0.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "hivemind.noel_cloud_run_parity_gate/1"
RECEIPT_OBJECT = "hivemind.noel_cloud_run_parity_receipt"
MANIFEST_OBJECT = "hivemind.onboarding.manifest"
CAMPAIGN_ID = "noel-2026-easy-onboard"
FINANCIAL_PROMISE = "0.00 EUR / 0.00 USD"
DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_CLOUD_RUN_SERVICE = "hivemind-openrouter"
DEFAULT_CLOUD_RUN_REGION = "us-central1"

HTTP_TIMEOUT_S = 5.0
SUBPROCESS_TIMEOUT_S = 3.0

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
    r'^\s*(?:push)?(?:url|insteadof)\s*=\s*(\S.*?)\s*$|^\s*\[url\s+"([^"]+)"\]',
    re.IGNORECASE | re.MULTILINE,
)
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r'\bnonce="([A-Za-z0-9_+=/=-]{8,128})"')
_FORBIDDEN_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + r"HTML|document\.write)\b"
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


def is_allowed_gateway_host(host: str) -> bool:
    """Return True iff `host` is `*.run.app` or loopback (`127.0.0.1` / `localhost` / `::1`)."""
    h = (host or "").strip().lower().strip("[]")
    if not h:
        return False
    if h in ALLOWED_LOOPBACK_HOSTS:
        return True
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
    parts = urlsplit(cleaned)
    if parts.username is not None or parts.password is not None:
        raise ParityGateError("Gateway URL must not embed credentials.")
    if parts.query or parts.fragment:
        raise ParityGateError("Gateway URL must not contain query parameters or fragments.")
    if parts.path not in ("", "/"):
        raise ParityGateError("Gateway URL must be a bare origin without a path suffix.")
    try:
        port = parts.port
    except ValueError as exc:
        raise ParityGateError("Gateway URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ParityGateError("Gateway URL port out of range.")
    host = (parts.hostname or "").lower()
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


def _assert_zero_inference_request(
    method: str,
    url: str,
    *,
    allow_feedback_post: bool = False,
) -> str:
    """Enforce read-only GET/HEAD (or zero-PII POST /v1/hivemind/onboarding/feedback), allowed gateway, and zero LLM inference."""
    verb = (method or "").strip().upper()
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    validate_gateway_url(origin)
    if "completions" in (parts.path or "").lower():
        raise ParityGateError(
            f"Zero-inference invariant violated: {verb} {parts.path} is strictly forbidden."
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
    return json.loads(
        _assert_safe_output(
            json.dumps(
                {
                    "profile": payload["profile"],
                    "ease_rating": int(payload["ease_rating"]),
                    "zero_game_lag": bool(payload["zero_game_lag"]),
                    "platform": str(payload.get("platform") or "linux")[:24],
                },
                sort_keys=True,
            )
        )
    )


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Direct, proxy-free stdlib HTTP helper handling GET parity probes and optional zero-PII feedback canary POST."""
    verb = (method or "").strip().upper()
    is_feedback_post = verb == "POST" and urlsplit(url).path == "/v1/hivemind/onboarding/feedback"
    _assert_zero_inference_request(verb, url, allow_feedback_post=is_feedback_post)
    if payload is not None and not is_feedback_post:
        raise ParityGateError("Zero-inference parity gate never sends request body payloads.")
    body_bytes = (
        json.dumps(_validate_zero_pii_feedback_payload(payload)).encode("utf-8")
        if is_feedback_post
        else None
    )

    req_headers: dict[str, str] = {
        "Accept": "application/json, text/html;q=0.9, text/plain;q=0.8",
        "User-Agent": "hivemind-noel-cloud-run-parity-gate/1.0",
        **({"Content-Type": "application/json"} if body_bytes is not None else {}),
        **(headers or {}),
    }

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=verb)
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            raw = resp.read(524_288).decode("utf-8", errors="replace")
            status_code = int(getattr(resp, "status", 200))
            resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            content_type = resp_headers.get("content-type", "")
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
    allow_feedback_post: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Invoke a 2-arg `(method, url)` or 4-arg `(method, url, payload, headers)` HTTP callable safely."""
    verb = (method or "").strip().upper()
    _assert_zero_inference_request(verb, url, allow_feedback_post=allow_feedback_post)
    clean_payload = (
        _validate_zero_pii_feedback_payload(payload)
        if (allow_feedback_post and verb == "POST")
        else None
    )
    sig = inspect.signature(http_fn)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    try:
        raw_res = (
            http_fn(verb, url, clean_payload, None)
            if (has_varargs or len(positional) >= 4)
            else http_fn(verb, url)
        )
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


def _extract_git_remote_urls(workspace: Path) -> list[str]:
    """Read `.git/config` (including worktree `gitdir:` indirection) and extract remote URLs locally."""
    git_entry = workspace / ".git"
    cfg_paths: list[Path] = [git_entry / "config"]
    if git_entry.is_file():
        first_line = git_entry.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        if first_line and first_line[0].strip().lower().startswith("gitdir:"):
            raw_gitdir = first_line[0].split(":", 1)[1].strip()
            gitdir = Path(raw_gitdir) if Path(raw_gitdir).is_absolute() else (workspace / raw_gitdir).resolve()
            commondir_file = gitdir / "commondir"
            common_rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip() if commondir_file.is_file() else "."
            cfg_paths = [gitdir / "config", (gitdir / common_rel).resolve() / "config"]

    raw_texts = [p.read_text(encoding="utf-8", errors="replace") for p in cfg_paths if p.is_file()]
    return [
        (m1 or m2).strip().strip('"')
        for txt in raw_texts
        for m1, m2 in _GIT_REMOTE_URL_RE.findall(txt)
        if (m1 or m2).strip().strip('"')
    ]


def check_personal_pc_compliance(
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    *,
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

    indicators = (
        amadeus_ind
        + zscaler_ind
        + (["env:USERDNSDOMAIN"] if dns_val else [])
        + [tag for tag, hits in env_hits if hits]
        + [f"git_remote:{name}" for name in git_hits]
    )
    matched_markers = sorted(
        set(["amadeus"] if amadeus_ind else [])
        | set(["zscaler"] if zscaler_ind else [])
        | set(dns_hits or (["corp"] if dns_val else []))
        | {name for _, hits in env_hits for name in hits}
        | set(git_hits)
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
    lang: str = "en",
) -> dict[str, Any]:
    local_check = check_personal_pc_compliance(env=env, cwd=cwd, lang=lang)
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


def scan_output_safety(text: str) -> list[str]:
    """Verify JSON or Markdown output contains zero PII, raw keys, home paths, IPs, or `'score': <digit>`."""
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
        raise ParityGateError(f"Output safety guard failed: {', '.join(issues)}")
    return text


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
) -> tuple[bool, dict[str, Any], list[str]]:
    """Inspect `/noel` or `/quick` HTML, per-request CSP nonce, DOM safety, and personal-PC compliance."""
    is_fr = lang == "fr"
    html_text = str(body.get("html") or "")
    hdrs = body.get("headers") if isinstance(body.get("headers"), Mapping) else {}
    csp_hdr = str(hdrs.get("content-security-policy") or body.get("csp") or "")

    has_doctype = "<!doctype html" in html_text.lower()
    attr_nonces = _HTML_ATTR_NONCE_RE.findall(html_text)
    hdr_nonces = _CSP_HEADER_NONCE_RE.findall(csp_hdr) if csp_hdr else []
    nonce = hdr_nonces[0] if hdr_nonces else (attr_nonces[0] if attr_nonces else None)

    csp_ok = bool(
        attr_nonces
        and (not csp_hdr or (hdr_nonces and hdr_nonces[0] in attr_nonces and "default-src 'none'" in csp_hdr.lower()))
        and nonce is not None
        and nonce not in seen_nonces
    )
    if nonce is not None:
        seen_nonces.add(nonce)

    dom_clean = not (_FORBIDDEN_DOM_SINK_RE.search(html_text) or _STATIC_FIG_UNIT_RE.search(html_text))
    low_html = html_text.lower()
    compliance_banner_ok = (
        ("100% personal" in low_html or "100 % personnel" in low_html)
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
            "personal_pc_banner_present": compliance_banner_ok,
        },
        issues,
    )


def _inspect_feedback_summary_schema(
    body: Mapping[str, Any],
    *,
    lang: str,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate `GET /v1/hivemind/onboarding/feedback/summary` schema and zero-PII bounds."""
    is_fr = lang == "fr"
    raw_total = body.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    total_val = int(raw_total) if total_ok else -1

    has_ease_key = "average_ease_rating" in body or "mean_ease_rating" in body
    raw_ease = (
        body.get("average_ease_rating")
        if "average_ease_rating" in body
        else body.get("mean_ease_rating")
    )
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

    schema_ok = bool(total_ok and ease_ok and lag_ok and prof_ok)
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
        }
    )


def check_cloud_run_holiday_parity(
    gateway: str = DEFAULT_GATEWAY,
    *,
    expected_sha: str | None = None,
    check_git_head: bool = False,
    verify_all_scripts: bool = False,
    enforce_personal_pc: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    lang: str = "en",
) -> dict[str, Any]:
    """Run the zero-inference Cloud Run Holiday Parity & Manifest Gate, failing closed on any drift."""
    lang_code = normalize_lang(lang)
    comp_doc = check_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
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
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    lang: str = "en",
) -> dict[str, Any]:
    """Verify post-deploy holiday manifest, `/noel` & `/quick` HTML/CSP, CES summary/canary, and rollback readiness."""
    lang_code = normalize_lang(lang)
    comp_doc = check_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
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
        "--previous-revision",
        default=None,
        help="Previous known-good Cloud Run K_REVISION for dry-run traffic rollback planning.",
    )
    parser.add_argument(
        "--probe-feedback-post",
        action="store_true",
        help="Send a zero-PII canary POST /v1/hivemind/onboarding/feedback round-trip during --post-deploy-gate.",
    )
    parser.add_argument(
        "--enforce-personal-pc",
        action="store_true",
        help="Enforce local Personal-PC & Amadeus compliance guard before any network or subprocess call.",
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


def _emit_report(report: Mapping[str, Any], *, as_json: bool, ok: bool = True) -> int:
    print(
        _assert_safe_output(json.dumps(report, indent=2, ensure_ascii=False))
        if as_json
        else render_markdown_report(report)
    )
    return 0 if ok else 2


def main(
    argv: Sequence[str] | None = None,
    *,
    http_fn: HttpFn = default_http_json,
    runner: RunnerFn = default_subprocess_runner,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
) -> int:
    """CLI entrypoint returning `0` on verified parity/plan or `2` (fail-closed) on any drift/error."""
    args = build_parser().parse_args(argv)
    deploy_path = Path(args.deploy_script) if args.deploy_script else None

    try:
        lang_code = normalize_lang(args.lang)
        comp = check_personal_pc_compliance(env=env, cwd=cwd, lang=lang_code)
        if args.enforce_personal_pc and not comp["compliance_ok"]:
            print(comp["summary"], file=sys.stderr)
            return _emit_report(
                _compliance_blocked_report(
                    "post_deploy_gate" if args.post_deploy_gate else ("redeploy_plan" if args.print_redeploy_plan else "parity_gate"),
                    DEFAULT_GATEWAY,
                    comp,
                    lang_code=lang_code,
                ),
                as_json=bool(args.json),
                ok=False,
            )

        clean_gw = validate_gateway_url(args.gateway)
        if args.post_deploy_gate:
            return _emit_report(
                post_rep := verify_post_deploy_holiday_readiness(
                    clean_gw,
                    expected_sha=args.expected_sha,
                    check_git_head=bool(args.check_git_head),
                    previous_revision=args.previous_revision,
                    probe_feedback_post=bool(args.probe_feedback_post),
                    env=env,
                    cwd=cwd,
                    http_fn=http_fn,
                    runner=runner,
                    lang=lang_code,
                ),
                as_json=bool(args.json),
                ok=bool(post_rep["post_deploy_ready"]),
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
            return _emit_report(plan, as_json=bool(args.json), ok=bool(plan["plan_ready"]))

        report = check_cloud_run_holiday_parity(
            clean_gw,
            expected_sha=args.expected_sha,
            check_git_head=bool(args.check_git_head),
            verify_all_scripts=bool(args.verify_all_scripts),
            env=env,
            cwd=cwd,
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
            return _emit_report(report, as_json=bool(args.json), ok=bool(report["parity_ok"] and plan["plan_ready"]))

        return _emit_report(report, as_json=bool(args.json), ok=bool(report["parity_ok"]))
    except ParityGateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
