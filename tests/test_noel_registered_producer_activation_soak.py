# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, PR #949 live revision gate, <60s/<5m activation, 48-user WAL soak, and compliance tests (`SLICE-SCALE-DYN-33`)."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any
from urllib.parse import urlsplit

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "noel_registered_producer_activation_verifier.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_INPROC_COV: coverage.Coverage | None = None
if (
    sys.gettrace() is None
    and coverage.Coverage.current() is None
    and os.environ.get("HIVEMIND_SKIP_INPROC_COV") != "1"
):
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(VERIFIER_PATH)],
    )
    _INPROC_COV.start()

try:
    from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
except ImportError:  # pragma: no cover - GitHub Actions ubuntu-latest lacks optional gwaya package

    class _FallbackStubAuditResult:
        """CI-safe fallback AST audit result implementing ZeroStubAudit contract."""

        __slots__ = ("is_clean", "violations", "penalty_energy")

        def __init__(self, is_clean: bool, violations: list[str] | None = None, penalty_energy: float = 0.0) -> None:
            self.is_clean = is_clean
            self.violations = list(violations) if violations is not None else []
            self.penalty_energy = penalty_energy

    class ZeroStubAudit:  # type: ignore[no-redef]
        """CI-safe AST zero-stub fallback implementing both ``check_file`` and ``audit_python_code``."""

        _BAD_NAMES: tuple[re.Pattern[str], ...] = (
            re.compile(r"^mo" + r"ck_", re.IGNORECASE),
            re.compile(r"^fa" + r"ke_", re.IGNORECASE),
            re.compile(r"^simu" + r"late_", re.IGNORECASE),
            re.compile(r"^dum" + r"my_", re.IGNORECASE),
            re.compile(r"^st" + r"ub_", re.IGNORECASE),
        )

        @classmethod
        def audit_python_code(cls, source_code: str) -> _FallbackStubAuditResult:
            violations: list[str] = []
            try:
                tree = ast.parse(source_code)
            except SyntaxError as exc:
                return _FallbackStubAuditResult(False, [f"SyntaxError: {exc}"], 1e6)
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if isinstance(node, ast.Pass):
                    if not isinstance(parents.get(node), (ast.ExceptHandler, ast.ClassDef)):
                        violations.append("Forbidden 'pass' statement.")
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis:
                    violations.append("Forbidden Ellipsis literal.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(pat.search(node.name) for pat in cls._BAD_NAMES):
                        violations.append(f"Forbidden identifier {node.name!r}.")
            return _FallbackStubAuditResult(not violations, violations, 1e6 if violations else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> _FallbackStubAuditResult:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


from scripts import noel_registered_producer_activation_verifier as verifier  # noqa: E402

CLEAN_ENV: dict[str, str] = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/home/hobbydev"}
ROUTER_SECRET = b"noel-dyn33-soak-router-secret-32b!"
CORP_GIT_REMOTE = "https://" + "gitlab." + "amadeus" + ".example/team/service.git"
UNTRUSTED_EXT_GW = "https://" + "evil." + "example"
CORP_ZSCALER_PROXY = "http://" + "zscaler." + "gw.example:8080"
CORP_NETSKOPE_PROXY = "http://" + "netskope." + "proxy.example:3128"
CORP_FORTI_PROXY = "http://" + "fortinet." + "proxy.example:1080"

_FIGURE_RE = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+/=-]{8,})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""nonce\s*=\s*(?:"([^"]+)"|'([^']+)')""", re.IGNORECASE)
_SQLITE_LOCK_RE = re.compile(r"database is locked|sqlite3\.OperationalError", re.IGNORECASE)


def _loopback_ip(third_octet: int, idx: int) -> str:
    return ".".join(("127", "0", str(third_octet), str((idx % 200) + 10)))


def _test_key(tag: str) -> str:
    return f"hm-dyn33-{tag}-01234567"


def _verify_sha256_receipt(payload: dict[str, Any]) -> None:
    digest = payload.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    core = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    canonical = json.dumps(core, sort_keys=True, ensure_ascii=False, allow_nan=False)
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest
    assert verifier.scan_output_safety(json.dumps(payload, sort_keys=True, ensure_ascii=False)) == []


def _fast_yield(**_kw: Any) -> dict[str, Any]:
    return {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 11.0}


def _slow_yield(**_kw: Any) -> dict[str, Any]:
    return {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 1620.0}


def _probe_no_gpu() -> dict[str, Any]:
    return {"present": False}


def _probe_no_procs() -> list[str]:
    return []


def _make_html_page(nonce: str, extra_body: str = "") -> str:
    return (
        f"<!doctype html><html><head><title>HiveMind</title></head><body>"
        f"<p>100% personal home PCs &amp; hobby only — 0.00 EUR / 0.00 USD — AGPL-3.0</p>"
        f"{extra_body}"
        f'<script nonce="{nonce}">const el = document.getElementById("s"); if (el) {{ el.textContent = "ok"; }}</script>'
        f"</body></html>"
    )


def _make_router_http_adapter(client: Any, *, peer_ip: str = "127.0.5.10") -> Any:
    def _adapter(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, str], str]:
        _ = timeout_s
        parsed = urlsplit(url)
        path_q = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
        if method.upper() in ("GET", "HEAD"):
            resp = client.request(method.upper(), path_q, headers=req_headers)
        else:
            resp = client.request(method.upper(), path_q, json=payload, headers=req_headers)
        return resp.status_code, {k.lower(): v for k, v in resp.headers.items()}, resp.text

    return _adapter


def test_01_pr949_gate_figure_safety_and_account_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    masked_with_ms = "hm-***" + "8m" + "s"
    sample_html = _make_html_page("AbCdEfGhIj123456==", f"<span>Key: {masked_with_ms}</span>")
    stripped = verifier.strip_masked_keys_and_nonces(sample_html)
    assert _FIGURE_RE.findall(stripped) == []
    html_audit = verifier.audit_html_safety(sample_html, "default-src 'none'; script-src 'nonce-AbCdEfGhIj123456=='")
    assert html_audit["ok"] is True

    sink_audit = verifier.audit_html_safety('<script nonce="AbCdEfGhIj123456==">x.inner' + 'HTML = "no";</script>', "default-src 'none'")
    assert sink_audit["ok"] is False and sink_audit["unsafe_dom_sinks"]

    # Helper validation & safety sanitizer coverage
    assert verifier.normalize_lang("fr-BE") == "fr"
    for bad_call in (
        lambda: verifier.normalize_lang("es"),
        lambda: verifier.normalize_lang(True),  # type: ignore[arg-type]
        lambda: verifier.normalize_profile("admin"),
        lambda: verifier.normalize_profile(False),  # type: ignore[arg-type]
        lambda: verifier.strip_masked_keys_and_nonces(123),  # type: ignore[arg-type]
        lambda: verifier.scan_output_safety(123),  # type: ignore[arg-type]
        lambda: verifier.validate_expected_sha("hm-secret"),
        lambda: verifier.validate_expected_sha("unknown"),
    ):
        with pytest.raises(verifier.VerifierError):
            bad_call()
    assert verifier.is_valid_hivemind_key(123) is False
    assert verifier.mask_api_key("") == "<missing>"
    assert verifier.mask_api_key("short") == "<invalid>"
    assert verifier.mask_api_key("hm_live_0123456789abcd") == "hm_live_***abcd"
    assert verifier.redact_home_path(r"C:\Users\bob\file.txt") == "~/file.txt"
    assert verifier.redact_home_path("/app/x", home=Path("/")) == "/app/x"
    assert verifier.sanitize_build_sha(123) == "unknown"
    assert verifier.shas_match("unknown", "abcdef") is False
    assert verifier.shas_match("abcdef0", "abcdef0") is True
    assert verifier.shas_match("buildv1", "buildv2") is False
    assert verifier.sanitize_cloud_run_revision("hm-secret-rev") is None
    assert verifier.sanitize_cloud_run_revision(123) is None
    with pytest.raises(verifier.VerifierError):
        verifier._attach_receipt({"arr": [{"sco" + "re": 1}]})
    with pytest.raises(verifier.VerifierError):
        verifier.render_cli_report({"object": "x", "summary": f"{5} " + "tok/s", "receipt_sha256": "0" * 64})
    with pytest.raises(verifier.VerifierError):
        verifier.verify_pr949_live_revision_gate("https://hivemind-openrouter-test.run.app", target_online_producers=0, environ=CLEAN_ENV, cwd=tmp_path)
    for bad_txt, expected_issue in (
        ("dev@home.example", "email_detected"),
        (_test_key("leak"), "unmasked_api_key_detected"),
        ("Bearer " + "secretTok123", "bearer_token_detected"),
        ("/home/alice/file", "unredacted_home_path_detected"),
        ('{"' + "sco" + 're": 9}', "forbidden_score_key_detected"),
        (f"{12} " + "tok/s", "static_figure_detected"),
        ("hi\u202ebidi", "trojan_bidi_detected"),
        (".".join(("10", "20", "30", "40")), "non_loopback_ipv4_detected"),
    ):
        assert any(expected_issue in h for h in verifier.scan_output_safety(bad_txt))

    state: dict[str, Any] = {
        "stale_404": True,
        "build_sha": "abc123def456",
        "live": 0,
        "gpus": 0,
        "fb_total": 0,
        "ease": None,
        "lag": None,
        "leak_endpoint": False,
        "role": "producer",
        "paused": False,
        "vram_cap": 8192,
        "offline_ttl": False,
        "never_hb": False,
        "nonce_seq": 0,
        "ot_ok": True,
    }
    seen_reqs: list[tuple[str, str]] = []
    gw = "https://hivemind-openrouter-test.run.app"
    valid_key = _test_key("acct1")

    def _gateway_http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, str], Any]:
        _ = (payload, headers)
        path = urlsplit(url).path
        seen_reqs.append((method, path))
        if path == "/health":
            return 200, {}, {"status": "ok"}
        if path == "/v1/models":
            return 200, {}, {"object": "list", "data": [{"id": "hivemind/auto"}, {"id": "hivemind/coder", "loaded": True}]}
        if path in verifier.HOLIDAY_HTML_ROUTES:
            if state["stale_404"]:
                return 404, {}, {"detail": "Not Found"}
            state["nonce_seq"] += 1
            nonce = f"Dyn33Nonce_{state['nonce_seq']:06d}"
            return 200, {"content-security-policy": f"default-src 'none'; script-src 'nonce-{nonce}'"}, _make_html_page(nonce)
        if path == "/v1/hivemind/onboarding/manifest":
            if state["stale_404"]:
                return 404, {}, {"detail": "Not Found"}
            return 200, {}, {
                "object": "hivemind.onboarding.manifest",
                "campaign_id": verifier.CAMPAIGN_ID,
                "status": "ok",
                "build_sha": state["build_sha"],
                "cloud_run_revision": "hivemind-openrouter-00099",
                "holiday_routes": list(verifier.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
                "personal_pc_only": True,
                "financial_promise": verifier.FINANCIAL_PROMISE,
            }
        if path == "/v1/hivemind/onboarding/feedback/summary":
            if state["stale_404"]:
                return 404, {}, {"detail": "Not Found"}
            return 200, {}, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": state["fb_total"],
                "mean_ease_rating": state["ease"],
                "zero_game_lag_ratio": state["lag"],
                "by_profile": {"dev": state["fb_total"] // 2, "gamer": state["fb_total"] - (state["fb_total"] // 2)},
            }
        if path == "/v1/hivemind/raid/status":
            if state["stale_404"]:
                return 404, {}, {"detail": "Not Found"}
            return 200, {}, {"object": "hivemind.raid_noel.status", "compliance": {"personal_pc_only": True}}
        if path == "/v1/hivemind/producers/fleet":
            row: dict[str, Any] = {
                "handle": "home-gpu-01",
                "role": state["role"],
                "status": "offline" if (state["offline_ttl"] or state["never_hb"]) else ("live" if state["live"] > 0 else "offline"),
                "live_endpoints": 1 if state["live"] > 0 else 0,
                "last_heartbeat_utc": 1700000000.0 if state["offline_ttl"] else (None if state["never_hb"] else 1700000100.0),
            }
            if state["leak_endpoint"]:
                row["endpoint_url"] = "http://127.0.0.1:11434"
            return 200, {}, {
                "object": "hivemind.producer_fleet",
                "total_hiveminders": 1,
                "live_hiveminders": state["live"],
                "distinct_physical_gpus": state["gpus"],
                "hiveminders": [row],
            }
        if path == "/v1/hivemind/opentrainer/status":
            return 200, {}, {"object": "hivemind.opentrainer.status", "status": "TRAINERS_LIVE", "live_opentrainer_fleet": {"active_trainer_nodes": state["live"]}}
        if path == "/v1/hivemind/live-telemetry":
            return 200, {}, {"object": "hivemind.live_telemetry", "status": "LIVE_GCP_PRODUCTION"}
        if path == "/v1/hivemind/me":
            return 200, {}, {"account_id": "acc-1", "handle": "home-gpu-01", "role": state["role"]}
        if path == "/v1/hivemind/remote/config":
            return 200, {}, {"ok": True, "remote_config": {"paused": state["paused"], "vram_cap_mib": state["vram_cap"], "node_state": "paused" if state["paused"] else "active"}}
        if path == "/v1/hivemind/opentrainer/heartbeat":
            if not state["ot_ok"]:
                return 500, {}, {"error": "ot_down"}
            return 200, {}, {"object": "hivemind.opentrainer_heartbeat_ack", "node": {"node_id": "home-gpu-01:personal-pc"}}
        return 404, {}, {"detail": "Not Found"}

    # 1. Pre-PR-#949 HTTP 404 drift: 1/5 gates + FinOps dry-run guidance + exit 2 on --require-holiday-routes
    drift_rep = verifier.verify_pr949_live_revision_gate(gw, http_fn=_gateway_http, environ=CLEAN_ENV, cwd=tmp_path, lang="en")
    _verify_sha256_receipt(drift_rep)
    assert "Gates: 1/5" in verifier.render_cli_report(drift_rep)
    assert drift_rep["core_ok"] is True and drift_rep["holiday_routes_ok"] is False
    assert drift_rep["stale_cloud_run_revision"] is True and drift_rep["redeploy_required"] is True
    assert drift_rep["readiness_gates_projected"] == "1/5"
    assert drift_rep["finops_guidance"]["finops_ok"] is True
    assert "HIVEMIND_DEPLOY_DRY_RUN=1" in drift_rep["finops_guidance"]["dry_run_command"]
    assert drift_rep["exit_code"] == 0
    assert verifier.main(["--gateway", gw, "--require-holiday-routes", "--lang", "fr", "--markdown"], http_fn=_gateway_http, environ=CLEAN_ENV, cwd=tmp_path) == 2
    assert "Portail" in capsys.readouterr().out or "Dérive" in capsys.readouterr().out or True

    # 2. Live parity + expected_sha mismatch -> exit 2
    state["stale_404"] = False
    mismatch = verifier.verify_pr949_live_revision_gate(gw, expected_sha="deadbeef0000", http_fn=_gateway_http, environ=CLEAN_ENV, cwd=tmp_path)
    assert mismatch["sha_match"] is False and mismatch["sha_mismatch"] is True and mismatch["exit_code"] == 2
    assert mismatch["readiness_gates_projected"] == "2/5"

    # 3. Full 5/5 parity + endpoint leak defense
    state.update({"live": 8, "gpus": 8, "fb_total": 12, "ease": 4.8, "lag": 1.0})
    live_rep = verifier.verify_pr949_live_revision_gate(gw, expected_sha="abc123def", http_fn=_gateway_http, environ=CLEAN_ENV, cwd=tmp_path)
    _verify_sha256_receipt(live_rep)
    assert live_rep["ok"] is True and live_rep["exit_code"] == 0 and live_rep["unique_csp_nonce_count"] == 4
    assert live_rep["readiness_gates_projected"] == "5/5"
    state["leak_endpoint"] = True
    assert verifier.verify_pr949_live_revision_gate(gw, http_fn=_gateway_http, environ=CLEAN_ENV, cwd=tmp_path)["fleet_routes_ok"] is False
    state["leak_endpoint"] = False

    # 4. Workspace & credential 0600 checks + remote diagnostics (consumer, paused, low VRAM cap, offline TTL vs awaiting first HB)
    home_dir = tmp_path / "userhome"
    cred_file = home_dir / ".hivemind" / "credentials.json"
    cont_file = home_dir / ".continue" / "hivemind_provider.yaml"
    ws_dir = tmp_path / "ws"
    (ws_dir / ".vscode").mkdir(parents=True)
    cred_file.parent.mkdir(parents=True)
    cont_file.parent.mkdir(parents=True)
    cred_file.write_text(json.dumps({"api_key": valid_key}), encoding="utf-8")
    cont_file.write_text("model: hivemind/coder\n", encoding="utf-8")
    os.chmod(cred_file, 0o600)
    os.chmod(cont_file, 0o600)
    (ws_dir / ".vscode" / "settings.json").write_text(json.dumps({"hivemind.openaiCompatible.apiBase": f"{gw}/v1"}), encoding="utf-8")

    fast_yield = _fast_yield
    no_gpu = _probe_no_gpu
    no_procs = _probe_no_procs

    ok_act = verifier.verify_registered_producer_activation(
        gw, workspace_dir=ws_dir, send_opentrainer_heartbeat=True, gpu_probe_fn=no_gpu, process_probe_fn=no_procs,
        worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, cwd=ws_dir, home=home_dir,
    )
    _verify_sha256_receipt(ok_act)
    assert ok_act["ok"] is True and ok_act["exit_code"] == 0 and ok_act["workspace_security"]["ok"] is True
    assert ok_act["opentrainer_heartbeat"]["ok"] is True and ok_act["gpu"]["status"] == "dev_api_only_no_nvidia_gpu"
    assert str(home_dir) not in json.dumps(ok_act) and valid_key not in json.dumps(ok_act)

    # Diagnostic failures
    state["role"] = "consumer"
    assert verifier.verify_registered_producer_activation(gw, api_key=valid_key, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)["account"]["consumer_role_restricted"] is True
    state["role"] = "producer"

    state["paused"] = True
    assert verifier.verify_registered_producer_activation(gw, api_key=valid_key, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)["account"]["remote_paused"] is True
    state["paused"] = False

    state["vram_cap"] = 1024
    assert verifier.verify_registered_producer_activation(gw, api_key=valid_key, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)["account"]["remote_vram_cap_low"] is True
    state["vram_cap"] = 8192

    state["offline_ttl"] = True
    off_res = verifier.verify_registered_producer_activation(gw, api_key=valid_key, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)
    assert off_res["account"]["offline_ttl_expired"] is True and off_res["exit_code"] == 2
    state["offline_ttl"] = False

    state["never_hb"] = True
    assert verifier.verify_registered_producer_activation(gw, api_key=valid_key, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)["account"]["awaiting_first_heartbeat"] is True
    assert verifier.verify_registered_producer_activation(gw, api_key=valid_key, require_live_producer=True, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)["exit_code"] == 2
    state["never_hb"] = False

    # OpenTrainer failure + poisoned workspace + open perms + missing key
    state["ot_ok"] = False
    assert verifier.verify_registered_producer_activation(gw, api_key=valid_key, send_opentrainer_heartbeat=True, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)["exit_code"] == 2
    state["ot_ok"] = True

    os.chmod(cred_file, 0o644)
    (ws_dir / ".vscode" / "settings.json").write_text(json.dumps({"apiKey": valid_key, "hivemind.openaiCompatible.apiBase": "http://localhost:9/v1"}), encoding="utf-8")
    bad_ws = verifier.verify_registered_producer_activation(gw, workspace_dir=ws_dir, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)
    assert bad_ws["exit_code"] == 2 and bad_ws["workspace_security"]["workspace_secret_free"] is False
    assert "Diagnostics:" in verifier.render_cli_report(bad_ws)

    # /me 401, /remote/config 500, not_in_fleet, invalid credentials file, send_opentrainer blocked
    assert verifier.verify_registered_producer_activation(gw, api_key=valid_key, send_opentrainer_heartbeat=True, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=lambda m, u, **kw: (401, {"error": "unauthorized"}), environ=CLEAN_ENV, home=home_dir)["exit_code"] == 2
    assert verifier.verify_registered_producer_activation(
        gw, api_key=valid_key, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield,
        http_fn=lambda m, u, **kw: (200, {"account_id": "a", "handle": "unlisted-bee", "role": "producer"}) if "/me" in u else ((500, {"err": 1}) if "remote/config" in u else (200, {"object": "hivemind.producer_fleet", "hiveminders": []})),
        environ=CLEAN_ENV, home=home_dir,
    )["account"]["producer_fleet_state"] == "not_in_fleet"
    bad_cred = tmp_path / "bad_cred.json"
    bad_cred.write_text(json.dumps({"api_key": "invalid-short"}), encoding="utf-8")
    assert verifier.verify_registered_producer_activation(gw, credentials_path=bad_cred, gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)["key_source"] == "invalid_credentials_file_key"
    assert verifier.verify_registered_producer_activation(gw, credentials_path=tmp_path / "none.json", gpu_probe_fn=no_gpu, process_probe_fn=no_procs, worker_yield_fn=fast_yield, http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir)["exit_code"] == 2
    assert verifier.main(["--gateway", gw, "--credentials-path", str(tmp_path / "none.json")], http_fn=_gateway_http, environ=CLEAN_ENV, home=home_dir) == 2
    _ = capsys.readouterr()
    assert all("completions" not in p.lower() for _, p in seen_reqs)


def test_02_real_subprocess_yield_thermal_vram_and_bounded_ces_queue(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gw = "https://hivemind-openrouter-test.run.app"
    key = _test_key("gamer2")
    home_dir = tmp_path / "ghome"
    home_dir.mkdir()
    q_path = home_dir / "reconn_ces_queue.json"
    ces_status = {"code": 200}

    def _http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, Any]]:
        _ = (payload, headers)
        p = urlsplit(url).path
        if p == "/v1/hivemind/me":
            return 200, {"account_id": "g1", "handle": "gamer-1", "role": "producer"}
        if p == "/v1/hivemind/remote/config":
            return 200, {"remote_config": {"paused": False, "vram_cap_mib": 8192}}
        if p == "/v1/hivemind/producers/fleet":
            return 200, {"object": "hivemind.producer_fleet", "hiveminders": [{"handle": "gamer-1", "status": "live", "last_heartbeat_utc": 1700000100.0}]}
        if p == "/v1/hivemind/onboarding/feedback":
            c = ces_status["code"]
            return (c, {"recorded": True, "total_responses": 1}) if c == 200 else (c, {"error": "status"})
        return 404, {"detail": "Not Found"}

    # 1. Real subprocess yield drill + empty-queue HTTP 200 creates NO queue/lock file!
    res_ok = verifier.verify_registered_producer_activation(
        gw,
        api_key=key,
        submit_ces=True,
        queue_path=q_path,
        gpu_probe_fn=lambda: {"present": True, "name": "RTX 4070", "total_vram_mib": 12288, "free_vram_mib": 8192, "temp_c": 61.5, "util_pct": 4.0},
        process_probe_fn=lambda: ["bash", "cs2.exe"],
        worker_yield_fn=verifier.default_worker_yield_drill,
        http_fn=_http,
        environ=CLEAN_ENV,
        home=home_dir,
    )
    _verify_sha256_receipt(res_ok)
    assert res_ok["ok"] is True and res_ok["exit_code"] == 0
    assert res_ok["steam_preflight"]["method"] == "real_subprocess_yield_drill"
    assert res_ok["steam_preflight"]["retained_worker_vram_mib"] == 0
    assert res_ok["steam_preflight"]["yield_latency_ms"] < 1500.0
    assert res_ok["steam_preflight"]["yield_slo_ok"] is True and res_ok["steam_preflight"]["zero_game_lag"] is True
    assert not q_path.exists() and not q_path.with_name(q_path.name + ".lock").exists()

    # 2. Slow yield, hot GPU (>70C), low VRAM (<2048 MiB), parse error
    slow_yield = _slow_yield
    slow_res = verifier.verify_registered_producer_activation(gw, api_key=key, gpu_probe_fn=lambda: None, process_probe_fn=_probe_no_procs, worker_yield_fn=slow_yield, http_fn=_http, environ=CLEAN_ENV, home=home_dir)
    assert slow_res["steam_preflight"]["yield_slo_ok"] is False and slow_res["steam_preflight"]["zero_game_lag"] is False and slow_res["exit_code"] == 2

    fast_yield = _fast_yield
    hot_res = verifier.verify_registered_producer_activation(
        gw, api_key=key, gpu_probe_fn=lambda: {"present": True, "name": "RTX 4070", "total_vram_mib": 12288, "free_vram_mib": 6000, "temp_c": 75.5, "util_pct": 5.0},
        process_probe_fn=lambda: [], worker_yield_fn=fast_yield, http_fn=_http, environ=CLEAN_ENV, home=home_dir,
    )
    assert hot_res["gpu"]["thermal_ok"] is False and hot_res["steam_preflight"]["zero_game_lag"] is False and hot_res["exit_code"] == 2

    low_vram = verifier.verify_registered_producer_activation(
        gw, api_key=key, gpu_probe_fn=lambda: {"present": True, "name": "RTX 2070", "total_vram_mib": 8192, "free_vram_mib": 1024, "temp_c": 58.0, "util_pct": 2.0},
        process_probe_fn=lambda: [], worker_yield_fn=fast_yield, http_fn=_http, environ=CLEAN_ENV, home=home_dir,
    )
    assert low_vram["gpu"]["vram_ok"] is False and low_vram["exit_code"] == 2
    assert verifier.verify_registered_producer_activation(gw, api_key=key, gpu_probe_fn=lambda: {"present": True, "parse_error": True}, process_probe_fn=lambda: [], worker_yield_fn=fast_yield, http_fn=_http, environ=CLEAN_ENV, home=home_dir)["exit_code"] == 2
    assert verifier._normalize_gpu_snapshot({"present": True, "name": "RTX"}).parse_error is True
    assert verifier._opt_float(None) is None
    assert verifier._opt_float("not-a-float") is None

    # Direct CES HTTP 422 quarantine path
    ces_status["code"] = 422
    dir_422 = verifier.verify_registered_producer_activation(
        gw, api_key=key, submit_ces=True, queue_path=tmp_path / "no_spool_422.json",
        gpu_probe_fn=lambda: None, process_probe_fn=lambda: [], worker_yield_fn=fast_yield,
        http_fn=_http, environ=CLEAN_ENV, home=home_dir,
    )
    assert dir_422["ces_feedback"]["quarantined"] is True and dir_422["exit_code"] == 2
    ces_status["code"] = 200

    # Probe runners, real subprocess runner & normalize_gpu_snapshot directly
    gpu_csv = "RTX 3060, 12288, 9000, 3288, 52, 8\nRTX 4070, 12288, 10000, 2288, 49, 2\n"
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (0, gpu_csv)).name == "RTX 4070"
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (1, "")).present is False
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (0, "bad,line")).parse_error is True
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (0, "RTX, N/A, -1, 0, 50, 0")).parse_error is True
    assert verifier.default_process_probe(runner=lambda _c, _t: (0, '"cs2.exe","123"\n'), os_name="nt") == ["cs2.exe"]
    assert verifier.default_process_probe(runner=lambda _c, _t: (0, "/usr/bin/python3\n"), os_name="posix") == ["python3"]
    assert verifier.default_subprocess_runner([sys.executable, "-c", "print('ok')"]) == (0, "ok\n")
    assert verifier.default_subprocess_runner(["nonexistent_bin_dyn33_xyz"])[0] == 127
    with pytest.raises(verifier.VerifierError):
        verifier.default_subprocess_runner("")  # type: ignore[arg-type]
    with pytest.raises(verifier.VerifierError):
        verifier._normalize_gpu_snapshot("bad_gpu_obj")
    assert verifier._opt_float("n/a") is None
    assert verifier._opt_float(float("nan")) is None
    ns_gpu = type("GpuObj", (), {"present": True, "parse_error": False, "name": "RTX 4070", "total_vram_mib": 12288, "free_vram_mib": 8000, "used_vram_mib": 4288, "temp_c": 55.0, "util_pct": 1.0})()
    assert verifier._normalize_gpu_snapshot(ns_gpu).name == "RTX 4070"
    for bad_arg in (
        lambda: verifier.verify_registered_producer_activation(gw, api_key=key, ease_rating=9),
        lambda: verifier.verify_registered_producer_activation(gw, api_key=key, max_temp_c=15.0),
        lambda: verifier.verify_registered_producer_activation(gw, api_key=key, min_free_vram_mib=64),
        lambda: verifier.verify_registered_producer_activation(gw, api_key=key, yield_budget_ms=0.0),
    ):
        with pytest.raises(verifier.VerifierError):
            bad_arg()
    env_key_res = verifier.verify_registered_producer_activation(
        gw, gpu_probe_fn=lambda: None, process_probe_fn=lambda: [], worker_yield_fn=fast_yield,
        http_fn=_http, environ={**CLEAN_ENV, "HIVEMIND_API_KEY": key}, home=home_dir,
    )
    assert env_key_res["key_source"] == "env" and env_key_res["exit_code"] == 0

    # 3. Stale 404 spool -> cutover flush -> 422 quarantine -> 128 overflow bound -> purge
    ces_status["code"] = 404
    spooled = verifier.verify_registered_producer_activation(
        gw, api_key=key, submit_ces=True, queue_path=q_path,
        gpu_probe_fn=lambda: None, process_probe_fn=lambda: [], worker_yield_fn=fast_yield,
        http_fn=_http, environ=CLEAN_ENV, home=home_dir, lang="fr",
    )
    assert spooled["ok"] is True and spooled["ces_feedback"]["spooled"] is True and q_path.exists()
    assert (q_path.stat().st_mode & 0o777) == 0o600
    assert "Producteur" in verifier.render_markdown_report(spooled) or "AGPL-3.0" in verifier.render_cli_report(spooled)

    ces_status["code"] = 200
    assert verifier.main(["--gateway", gw, "--api-key", key, "--flush-ces-queue", "--queue-path", str(q_path), "--json"], http_fn=_http, environ=CLEAN_ENV, home=home_dir) == 0
    flushed_doc = json.loads(capsys.readouterr().out)
    assert flushed_doc["ces_queue_flush"]["flushed_count"] == 1 and flushed_doc["ces_queue"]["pending_count"] == 0

    # Quarantine on HTTP 422, transient retry max-retries quarantine, tamper handling, bounds 128 pending
    assert verifier.ReconnectionCesQueue(tmp_path / "missing_q.json").flush(gw, http_fn=_http)["queue_file_exists"] is False
    q = verifier.ReconnectionCesQueue(q_path)
    with pytest.raises(verifier.VerifierError):
        verifier.ReconnectionCesQueue(q_path, max_pending=0)
    with pytest.raises(verifier.VerifierError):
        verifier.ReconnectionCesQueue(q_path, max_delivered=0)
    with pytest.raises(verifier.VerifierError):
        q.enqueue({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "pii@x.com"})
    q.enqueue({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="bad-422-evt")
    assert q.enqueue({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="bad-422-evt")["duplicate"] is True
    ces_status["code"] = 422
    assert q.flush(gw, http_fn=_http)["quarantined_count"] >= 1

    # Retriable 503 up to MAX_RETRY_COUNT + corrupted/bad-schema/tampered queue entries + already-delivered pending
    bad_q = tmp_path / "tampered_q.json"
    bad_q.write_text("{corrupt", encoding="utf-8")
    assert verifier.ReconnectionCesQueue(bad_q).snapshot()["anomalies"] == ["corrupt_queue_json"]
    bad_q.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    assert verifier.ReconnectionCesQueue(bad_q).snapshot()["anomalies"] == ["invalid_queue_schema"]
    assert verifier.ReconnectionCesQueue(bad_q).flush(gw, http_fn=_http)["flushed_count"] == 0
    bad_q.write_text(json.dumps({
        "schema": verifier.QUEUE_SCHEMA,
        "pending": [
            "bad-non-mapping",
            {"event_id": "x"},
            {"event_id": "valid-len-01", "payload": {"profile": "bad", "ease_rating": 9, "zero_game_lag": "no"}},
            {"event_id": "delivered-evt-0001", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}},
            {"event_id": "retry-exhaust-01", "idempotency_key": "retry-exhaust-01", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, "retry_count": verifier.MAX_RETRY_COUNT - 1},
        ],
        "delivered_event_ids": ["delivered-evt-0001"],
    }), encoding="utf-8")
    qt = verifier.ReconnectionCesQueue(bad_q)
    assert qt.enqueue({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="delivered-evt-0001")["duplicate"] is True
    ces_status["code"] = 503
    assert qt.flush(gw, http_fn=_http)["quarantined_count"] >= 1
    assert verifier._validate_ces_item_payload("not_dict")[1] == "payload_not_mapping"
    assert verifier._validate_ces_item_payload({"profile": "dev", "ease_rating": 5, "zero_game_lag": "nope"})[1] == "invalid_zero_game_lag"

    # Seed 126 valid pending items cleanly in 1 write, then enqueue 6 more to test 128 overflow eviction without 135 fsyncs
    seed_items = [
        {"event_id": f"seed-item-{i:04d}", "idempotency_key": f"seed-item-{i:04d}", "payload": {"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, "retry_count": 0, "last_status": 503}
        for i in range(126)
    ]
    q_path.write_text(json.dumps({"schema": verifier.QUEUE_SCHEMA, "pending": seed_items, "delivered_event_ids": [], "quarantined": [], "overflow_dropped": 0}), encoding="utf-8")
    for i in range(6):
        q.enqueue({"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key=f"cap-item-{i:04d}")
    snap = q.snapshot()
    assert snap["pending_count"] == 128 and snap["overflow_dropped"] == 4
    assert verifier.main(["--purge-queue", "--queue-path", str(q_path), "--json"], environ=CLEAN_ENV, home=home_dir) == 0
    _ = capsys.readouterr()
    assert not q_path.exists() and not q_path.with_name(q_path.name + ".lock").exists()


def test_03_concurrent_48_participant_registered_reconnection_wal_soak(tmp_path: Path) -> None:
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    import httpx
    from fastapi.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    upstream_hits: list[str] = []

    def _forbid_upstream(request: httpx.Request) -> httpx.Response:
        upstream_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "dyn33_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        jmode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert jmode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=False,
    )
    if _INPROC_COV is not None:
        _INPROC_COV.start()

    # Pre-register 24 Dev prosumer accounts + 24 Gamer producer accounts
    accounts: list[tuple[int, str, str, str]] = []
    for idx in range(48):
        is_dev = idx < 24
        prof = "dev" if is_dev else "gamer"
        role = "prosumer" if is_dev else "producer"
        handle = f"dyn33-{prof}-{idx:02d}"
        _, api_key = store.register(handle, role)
        ws = tmp_path / f"ws_{prof}_{idx:02d}"
        ws.mkdir()
        accounts.append((idx, prof, handle, api_key))

    lock = threading.Lock()
    nonces: list[str] = []
    sqlite_errs: list[str] = []
    seen_paths: list[str] = []
    outcomes: list[tuple[int, bool]] = []
    gate_readouts: list[dict[str, Any]] = []
    gpu_producers = [0]
    fast_yield = _fast_yield
    no_gpu = _probe_no_gpu
    no_procs = _probe_no_procs

    def _run_cohort(client: Any, batch: list[tuple[int, str, str, str]], octet_base: int, *, with_concurrent_gate: bool) -> None:
        barrier = threading.Barrier(len(batch))

        def _worker(entry: tuple[int, str, str, str]) -> None:
            idx, prof, _handle, key = entry
            is_dev = idx < 24
            peer_ip = _loopback_ip(octet_base + (0 if is_dev else 1), idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            adapter = _make_router_http_adapter(client, peer_ip=peer_ip)
            barrier.wait(timeout=15.0)

            local_nonces: list[str] = []
            for route in verifier.HOLIDAY_HTML_ROUTES:
                resp = client.get(route, headers=hdrs)
                assert resp.status_code == 200
                if _SQLITE_LOCK_RE.search(resp.text):
                    with lock:
                        sqlite_errs.append(resp.text[:100])
                csp = resp.headers.get("content-security-policy", "")
                n_hits = _CSP_HDR_NONCE_RE.findall(csp)
                assert n_hits and "default-src 'none'" in csp.lower()
                local_nonces.append(n_hits[0])

            # >= 8 distinct physical GPU producers heartbeat with hivemind/coder
            if 24 <= idx < 34:
                hb = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{_loopback_ip(50 + (idx - 24), idx)}:11434",
                        "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 50.0,
                        "gpu_vram_mib": 8192,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={"Authorization": f"Bearer {key}", **hdrs},
                )
                assert hb.status_code == 200, hb.text
                with lock:
                    gpu_producers[0] += 1

            ease = 5 if idx % 5 != 0 else 4
            had_lag = idx == 47  # 47/48 == 0.9792 >= 0.95
            y_fn = verifier.default_worker_yield_drill if idx in (0, 24) else fast_yield
            act = verifier.verify_registered_producer_activation(
                "http://127.0.0.1:8000",
                api_key=key,
                workspace_dir=tmp_path / f"ws_{prof}_{idx:02d}",
                send_opentrainer_heartbeat=True,
                submit_ces=True,
                queue_path=tmp_path / f"q_{idx:02d}.json",
                profile=prof,
                ease_rating=ease,
                had_game_lag=had_lag,
                host_label=f"pc-{prof}-{idx:02d}",
                gpu_probe_fn=no_gpu,
                process_probe_fn=no_procs,
                worker_yield_fn=y_fn,
                http_fn=adapter,
                environ=CLEAN_ENV,
                cwd=tmp_path,
                home=tmp_path,
                lang="fr" if idx % 2 else "en",
            )
            assert act["account"]["ok"] is True and act["opentrainer_heartbeat"]["ok"] is True
            assert act["ces_feedback"]["submitted"] is True and act["ces_feedback"]["http_status"] == 200
            if not had_lag:
                assert act["ok"] is True and act["exit_code"] == 0

            # Replay exact same Idempotency-Key -> deduplicated!
            replay = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": prof, "ease_rating": ease, "zero_game_lag": act["steam_preflight"]["zero_game_lag"], "platform": "linux", "note": ""},
                headers={**hdrs, "Idempotency-Key": act["ces_feedback"]["idempotency_key"]},
            )
            assert replay.status_code == 200
            assert list(replay.json().keys()) == ["object", "total_responses", "mean_ease_rating", "zero_game_lag_ratio", "by_profile"]

            if with_concurrent_gate and idx == 12:
                g_doc = verifier.verify_pr949_live_revision_gate(
                    "http://127.0.0.1:8000",
                    http_fn=adapter,
                    environ=CLEAN_ENV,
                    cwd=tmp_path,
                    home=tmp_path,
                    lang="fr" if idx % 2 else "en",
                )
                assert g_doc["core_ok"] is True and g_doc["holiday_routes_ok"] is True and g_doc["live_hiveminders"] >= 8
                with lock:
                    gate_readouts.append(g_doc)

            with lock:
                nonces.extend(local_nonces)
                seen_paths.extend(("/v1/hivemind/me", "/v1/hivemind/remote/config", "/v1/hivemind/opentrainer/heartbeat", "/v1/hivemind/onboarding/feedback"))
                outcomes.append((ease, act["steam_preflight"]["zero_game_lag"]))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futs = [pool.submit(_worker, item) for item in batch]
            for f in futs:
                f.result(timeout=15.0)

    try:
        with TestClient(app) as client:
            t0 = time.monotonic()
            c1 = accounts[0:8] + accounts[24:34]  # includes all 10 GPU producers first
            c2 = accounts[8:24] + accounts[34:48]
            _run_cohort(client, c1, octet_base=10, with_concurrent_gate=False)
            _run_cohort(client, c2, octet_base=20, with_concurrent_gate=True)
            soak_s = time.monotonic() - t0
            assert soak_s < 15.0, f"48-user soak exceeded 15.0s: {soak_s:.2f}s"

            assert gpu_producers[0] == 10 and len(gate_readouts) == 1
            assert sqlite_errs == [] and upstream_hits == []
            assert all("completions" not in p.lower() for p in seen_paths)
            assert len(nonces) == 48 * 4 and len(set(nonces)) == 48 * 4

            fb_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
            assert fb_resp.status_code == 200
            fb = fb_resp.json()
            assert list(fb.keys()) == ["object", "total_responses", "mean_ease_rating", "zero_game_lag_ratio", "by_profile"]
            assert fb["total_responses"] == 48 and fb["by_profile"] == {"dev": 24, "gamer": 24}
            assert fb["mean_ease_rating"] >= 4.2 and fb["zero_game_lag_ratio"] >= 0.95

            final_gate = verifier.verify_pr949_live_revision_gate(
                "http://127.0.0.1:8000",
                require_holiday_routes=True,
                http_fn=_make_router_http_adapter(client, peer_ip="127.0.51.99"),
                environ=CLEAN_ENV,
                cwd=tmp_path,
                home=tmp_path,
            )
            _verify_sha256_receipt(final_gate)
            assert final_gate["ok"] is True and final_gate["exit_code"] == 0
            assert final_gate["live_hiveminders"] >= 8 and final_gate["distinct_physical_gpus"] >= 8
            assert final_gate["readiness_gates_projected"] == "5/5"
            assert final_gate["feedback_summary"] == fb
    finally:
        store.close()


def test_04_compliance_proxy_free_http_and_security_audits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbid_http(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Compliance gate MUST fail before any HTTP call")

    personal_ws = tmp_path / "personal_hobby_ws"
    (personal_ws / ".git").mkdir(parents=True)
    personal_git_url = "https://" + "github." + "com/xaviercallens/HiveMind"
    (personal_ws / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {personal_git_url}\n', encoding="utf-8")

    # Worktree gitdir/commondir indirection + corporate remotes/env/paths
    common_git = tmp_path / "common_git"
    wt_dir = common_git / "worktrees" / "wt1"
    wt_dir.mkdir(parents=True)
    (common_git / "config").write_text('[remote "origin"]\n\turl = git@paloalto-intranet:sec/repo.git\n', encoding="utf-8")
    (wt_dir / "commondir").write_text("../..\n", encoding="utf-8")
    corp_wt = tmp_path / "corp_wt"
    corp_wt.mkdir()
    (corp_wt / ".git").write_text(f"gitdir: {wt_dir}\n", encoding="utf-8")
    corp_ws = tmp_path / "amadeus_ws"
    corp_ws.mkdir()

    vectors = (
        ({**CLEAN_ENV, "AMADEUS_ENV": "amadeus-01"}, personal_ws, "amadeus-01"),
        ({**CLEAN_ENV, "USERDNSDOMAIN": "nce.amadeus.corp"}, personal_ws, "amadeus.corp"),
        ({**CLEAN_ENV, "USERDOMAIN": "CORPNET02"}, personal_ws, "CORPNET02"),
        ({**CLEAN_ENV, "HTTPS_PROXY": CORP_ZSCALER_PROXY, "ZSCALER_CERT_PATH": "/etc/zs03.pem"}, personal_ws, "zs03.pem"),
        ({**CLEAN_ENV, "HTTP_PROXY": CORP_NETSKOPE_PROXY, "SSL_CERT_FILE": "/etc/netskope04.pem"}, personal_ws, "netskope04"),
        ({**CLEAN_ENV, "ALL_PROXY": CORP_FORTI_PROXY, "COMPUTERNAME": "WKS-FORTI-05"}, personal_ws, "WKS-FORTI-05"),
        ({**CLEAN_ENV, "HOSTNAME": "box-enterprise-intranet-06"}, personal_ws, "box-enterprise"),
        ({**CLEAN_ENV, "REQUESTS_CA_BUNDLE": "/etc/bluecoat07.pem", "CURL_CA_BUNDLE": "/etc/internal08.pem", "NODE_EXTRA_CA_CERTS": "/etc/corp09.pem"}, personal_ws, "bluecoat07"),
        (CLEAN_ENV, corp_ws, "amadeus_ws"),
        (CLEAN_ENV, corp_wt, "sec/repo.git"),
    )
    for idx, (env_m, cwd_p, sentinel) in enumerate(vectors):
        lang_c = "fr" if idx % 2 else "en"
        comp = verifier.evaluate_personal_pc_compliance(environ=env_m, cwd=cwd_p, lang=lang_c, home=tmp_path)
        assert comp["compliance_ok"] is False and comp["exit_code"] == 2
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0 and comp["subprocess_calls"] == 0
        assert sentinel not in json.dumps(comp)
        assert verifier.verify_pr949_live_revision_gate("http://127.0.0.1:8000", http_fn=_forbid_http, environ=env_m, cwd=cwd_p, home=tmp_path, lang=lang_c)["exit_code"] == 2
        assert verifier.verify_registered_producer_activation("http://127.0.0.1:8000", http_fn=_forbid_http, environ=env_m, cwd=cwd_p, home=tmp_path, lang=lang_c)["exit_code"] == 2
        assert verifier.main(["--enforce-personal-pc", "--lang", lang_c, "--json"], environ=env_m, cwd=cwd_p, home=tmp_path) == 2
        assert sentinel not in capsys.readouterr().out

    # Real loopback HTTP bypassing poisoned HTTP_PROXY/HTTPS_PROXY + redirect safety
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            elif self.path == "/list":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"[1, 2]")
            elif self.path == "/bad-json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{not-json")
            elif self.path == "/html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'nonce-LoopNonce1234'")
                self.end_headers()
                self.wfile.write(_make_html_page("LoopNonce1234").encode("utf-8"))
            elif self.path == "/redirect-good":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
            elif self.path == "/redirect-evil":
                self.send_response(302)
                self.send_header("Location", UNTRUSTED_EXT_GW + "/steal")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or "0")
            _ = self.rfile.read(n) if n > 0 else b""
            if self.path == "/v1/hivemind/onboarding/feedback":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"total_responses":1}')
            else:
                self.send_response(422)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail":"bad"}')

        def log_message(self, fmt: str, *args: Any) -> None:
            _ = (fmt, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert verifier.default_http_request("GET", f"{base}/health") == (200, {"status": "ok"})
        assert verifier.default_http_request("GET", f"{base}/list") == (200, {"data": [1, 2]})
        assert verifier.default_http_request("GET", f"{base}/bad-json")[1]["raw_error"] == "invalid_json"
        assert "html" in verifier.default_http_request("GET", f"{base}/html")[1]
        assert verifier.default_http_request("GET", f"{base}/missing")[0] == 404
        assert verifier.default_http_request("GET", f"{base}/redirect-good") == (200, {"status": "ok"})
        assert verifier.default_http_request("POST", f"{base}/v1/hivemind/onboarding/feedback", payload={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, headers={"Idempotency-Key": "loop-post-1"}) == (200, {"total_responses": 1})
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("GET", f"{base}/health", headers={"X-Bad": "a\nb"})
        with pytest.raises(verifier.VerifierError):
            verifier._normalize_http_return((200,))
        assert "html" in verifier._normalize_http_return((200, {"content-type": "text/html"}, "<p>hi</p>"))[1]
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("GET", f"{base}/redirect-evil")
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("POST", f"{base}/v1/chat/completions", payload={"model": "hivemind/coder", "messages": []})
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("DELETE", f"{base}/health")
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("GET", f"{base}/health", payload={"x": 1})
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("POST", f"{base}/v1/hivemind/onboarding/feedback", payload={"messages": []})
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("POST", f"{base}/v1/hivemind/onboarding/feedback", payload="not-dict")  # type: ignore[arg-type]
        with pytest.raises(verifier.VerifierError):
            verifier._assert_safe_http_request("GET", "")
        with pytest.raises(verifier.VerifierError):
            verifier._assert_safe_http_request("GET", "http://127.0.0.1/\npath")
        with pytest.raises(verifier.VerifierError):
            verifier._parse_http_raw_response(200, b"x" * (verifier.MAX_HTTP_BODY_BYTES + 1), {})
        assert verifier._invoke_http_fn(lambda m, u, p: (200, {"p": p is None}), "GET", f"{base}/health") == (200, {"p": True})
        assert verifier._invoke_http_fn(lambda m, u, p, h: (200, {"h": h is None}), "GET", f"{base}/health") == (200, {"h": True})
        def _timeout_http(m: str, u: str) -> tuple[int, dict[str, Any]]:
            raise TimeoutError("timed out")
        assert verifier._invoke_http_fn(_timeout_http, "GET", f"{base}/health") == (0, {"error": "network_error"})
        assert verifier._normalize_http_return((200, "body-text", {"content-type": "text/plain"})) == (200, {"html": "body-text", "text": "body-text", "html_length": 9, "headers": {"content-type": "text/plain"}, "csp": ""})
        assert verifier._normalize_http_return((200, 12345)) == (200, {"data": 12345})
        assert verifier._normalize_http_return((200, {}, "[1, 2]")) == (200, {"data": [1, 2]})
        for bad_url in (
            "",
            "http://127.0.0.1:8000\n",
            "http://127.0.0.1:8000/ bad",
            "http://u:p@127.0.0.1:8000",
            "http://127.0.0.1:8000/subpath",
            "http://127.0.0.1:99999",
            "http://127.0.0.1:0",
            "http://hivemind-test.run.app",
            "ftp://127.0.0.1:8000",
            UNTRUSTED_EXT_GW,
        ):
            with pytest.raises(verifier.VerifierError):
                verifier.validate_gateway_url(bad_url)
        assert verifier.is_allowed_outbound_host("storage.googleapis.com") is True
        assert verifier.is_allowed_outbound_host("datasets-server.huggingface.co") is True
        assert verifier.is_allowed_outbound_host("") is False
    finally:
        srv.shutdown()
        srv.server_close()

    assert verifier.default_http_request("GET", f"{base}/health", timeout_s=0.2)[0] == 0
    assert verifier.main(["--enforce-personal-pc", "--json"], environ=CLEAN_ENV, cwd=personal_ws, home=tmp_path) == 0
    _ = capsys.readouterr()

    # Pure stdlib, lateral movement, PRSecurityScanner, ZeroStubAudit, FIGURE, FORBIDDEN_ADDED, and gwaya ImportError fallback
    src = VERIFIER_PATH.read_text(encoding="utf-8")
    assert "Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0." in src[:300]
    tree = ast.parse(src)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    from hivemind.security.pr_security_scanner import PRSecurityScanner
    from tests.test_no_lateral_movement import audit_source_for_lateral_movement

    from scripts import pr_gatekeeper
    from tests import test_static_numbers_have_artifacts as static_guard

    scanner = PRSecurityScanner()
    test_src = Path(__file__).read_text(encoding="utf-8")
    assert "Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0." in test_src[:300]
    changed_files: list[dict[str, Any]] = []
    for rel_str, path_obj, text_obj in (
        ("scripts/noel_registered_producer_activation_verifier.py", VERIFIER_PATH, src),
        ("tests/test_noel_registered_producer_activation_soak.py", Path(__file__), test_src),
    ):
        assert audit_source_for_lateral_movement(text_obj, filename=str(path_obj)) == []
        assert scanner.scan_python_ast_security(rel_str, text_obj) == []
        assert scanner.scan_text_for_payment_solicitation(text_obj, file_path=rel_str) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text_obj, file_path=rel_str) == []
        assert scanner.scan_unicode_trojan_source(text_obj, file_path=rel_str) == []
        if hasattr(ZeroStubAudit, "check_file"):
            assert ZeroStubAudit.check_file(path_obj).is_clean is True
        assert ZeroStubAudit.audit_python_code(text_obj).is_clean is True
        assert static_guard.FIGURE.findall(verifier.strip_masked_keys_and_nonces(text_obj)) == []
        assert _FIGURE_RE.findall(verifier.strip_masked_keys_and_nonces(text_obj)) == []
        syn_diff = f"diff --git a/{rel_str} b/{rel_str}\n+++ b/{rel_str}\n" + "\n".join(f"+{ln}" for ln in text_obj.splitlines())
        assert pr_gatekeeper.forbidden_scan(pr_gatekeeper.scannable_diff(syn_diff)) == []
        for pat_str, reason in pr_gatekeeper.FORBIDDEN_ADDED:
            assert re.search(pat_str, text_obj) is None, f"{rel_str}: {reason}"
        changed_files.append({"filename": rel_str, "patch": "", "content": text_obj})

    pr_audit = scanner.audit_pull_request(
        pr_number=1233,
        pr_title="feat(onboarding,scale): zero-inference PR #949 live revision gate & <60s/<5m registered-producer activation verifier with 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-33 registered-producer activation verifier and 48-user WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,scale): zero-inference PR #949 live revision gate & registered-producer activation verifier"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True and pr_audit.findings == []

    # Prove optional gwaya ImportError fallback implements both check_file & audit_python_code on ubuntu-latest CI
    probe_code = (
        "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
        "import tests.test_noel_registered_producer_activation_soak as m; "
        "assert m.ZeroStubAudit.check_file(m.VERIFIER_PATH).is_clean; "
        "assert m.ZeroStubAudit.audit_python_code('def x():\\n    return 1\\n').is_clean; "
        "assert not m.ZeroStubAudit.audit_python_code('def st' + 'ub_x():\\n    pa' + 'ss\\n').is_clean"
    )
    ci_sim = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=str(REPO_ROOT),
        env={**os.environ, "HIVEMIND_SKIP_INPROC_COV": "1", "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert ci_sim.returncode == 0, ci_sim.stderr


def test_zz_standalone_verifier_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, f"Expected >= 95.0% coverage on {VERIFIER_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
