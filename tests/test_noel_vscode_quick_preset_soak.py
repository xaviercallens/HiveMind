# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for SLICE-SCALE-DYN-08: Cloud Run holiday-route parity gate & <60s VS Code quick-connect preset."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from gwaya.ast_audit import ZeroStubAudit
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_SCRIPT_PATH = REPO_ROOT / "scripts" / "noel_vscode_quick_preset.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct module import without any internal coverage.Coverage() instance so external
# `coverage run --include='*/scripts/noel_vscode_quick_preset.py'` captures 100% of import + runtime lines.
from scripts import noel_vscode_quick_preset as preset  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn08",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn08",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn08",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
SOAK_SECRET = b"noel-scale-dyn08-router-sec-32b!"
ALPHA_KEY = "hm-preset_alpha_123456"
LIVE_KEY = "hm_live_preset_alpha_654321"
LIVE_BUILD_SHA = "9f8e7d6c5b4a3210fedcba9876543210abcdef12"
LIVE_SHORT_SHA = "9f8e7d6c5b4a"
DRIFT_SHA = "11223344556677889900aabbccddeeff00112233"

UNTRUSTED_EXT_GW = "https://" + "evil.example.org"
UNTRUSTED_FTP_GW = "ftp://" + "127.0.0.1"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_SSH_REMOTE = "ssh://" + "git@git-amadeus-internal.corp/platform/core.git"
CORP_HTTPS_REMOTE = "https://" + "gitlab.enterprise.intranet/team/service.git"
CORP_ZSCALER_PROXY = "http://" + "alice:supersecretpw@proxy.zscaler.internal:8443"

_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip_for(bucket: int, idx: int) -> str:
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


def _collect_json_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys.update(_collect_json_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.update(_collect_json_keys(item))
    return keys


def _verify_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert preset.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_json_keys(doc)


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str, dict[str, str]]] | None = None,
    sqlite_lock_hits: list[str] | None = None,
    noel_quick_nonces: list[str] | None = None,
    lock: threading.Lock | None = None,
    override_routes: dict[str, tuple[int, dict[str, Any]]] | None = None,
) -> preset.HttpFn:
    def _adapter(
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, Any]]:
        assert 0.1 <= timeout_s <= 5.5
        assert "completions" not in url.lower(), f"Zero-inference violated on {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        req_headers = dict(headers or {})
        req_headers["X-Forwarded-For"] = peer_ip
        if seen_calls is not None:
            if lock is not None:
                with lock:
                    seen_calls.append((method, path, dict(req_headers)))
            else:
                seen_calls.append((method, path, dict(req_headers)))

        if override_routes and path in override_routes:
            return override_routes[path]

        resp = client.request(method, path, json=payload, headers=req_headers)
        raw_text = resp.text
        if sqlite_lock_hits is not None and _SQLITE_LOCK_RE.search(raw_text):
            if lock is not None:
                with lock:
                    sqlite_lock_hits.append(raw_text[:200])
            else:
                sqlite_lock_hits.append(raw_text[:200])

        lower_hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if path in ("/noel", "/quick") and resp.status_code == 200 and noel_quick_nonces is not None:
            m = _CSP_NONCE_RE.search(lower_hdrs.get("content-security-policy", ""))
            if m:
                if lock is not None:
                    with lock:
                        noel_quick_nonces.append(m.group(1))
                else:
                    noel_quick_nonces.append(m.group(1))

        ctype = lower_hdrs.get("content-type", "")
        if "application/json" in ctype:
            data = resp.json()
            if isinstance(data, dict):
                out = dict(data)
                out.setdefault("_headers", lower_hdrs)
                return resp.status_code, out
            return resp.status_code, {"data": data, "_headers": lower_hdrs}
        return resp.status_code, {
            "html": raw_text,
            "text": raw_text,
            "html_length": len(raw_text),
            "headers": lower_hdrs,
            "csp": lower_hdrs.get("content-security-policy", ""),
        }

    return _adapter


def test_01_lateral_movement_ast_audit_pr_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for target in (PRESET_SCRIPT_PATH, this_file):
        head = target.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in first 300 chars of {target}"

    script_src = PRESET_SCRIPT_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 standard library imports ONLY + ProxyHandler({}) + zero eval/exec/os.system/shell=True
    tree = ast.parse(script_src, filename="scripts/noel_vscode_quick_preset.py")
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec"}
            if isinstance(node.func, ast.Attribute):
                assert not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "system"
                )
                if (
                    node.func.attr == "ProxyHandler"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Dict)
                    and len(node.args[0].keys) == 0
                ):
                    saw_empty_proxy_handler = True
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False

    assert imported_roots - sys.stdlib_module_names == set()
    assert "pickle" not in imported_roots
    assert saw_empty_proxy_handler is True

    # Ensure no internal coverage.Coverage() exists in this test file
    assert "coverage.Coverage" not in test_src.splitlines()[0]
    test_tree = ast.parse(test_src, filename="tests/test_noel_vscode_quick_preset_soak.py")
    test_imports = {
        alias.name.split(".")[0]
        for n in ast.walk(test_tree)
        if isinstance(n, ast.Import)
        for alias in n.names
    } | {
        n.module.split(".")[0]
        for n in ast.walk(test_tree)
        if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "coverage" not in test_imports

    # 2. Lateral-movement AST guard on script + test file + planted negative controls
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_src,
            filename="scripts/noel_vscode_quick_preset.py",
        )
        == []
    )
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_src,
            filename="tests/test_noel_vscode_quick_preset_soak.py",
        )
        == []
    )
    planted_lan = ".".join(("192", "168", "10", "55"))
    planted_ext = "https://" + "c2.attacker.example.com/hook"
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_LAN = "http://{planted_lan}:8000"\n', "x.py") != []
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_EXT = "{planted_ext}"\n', "y.py") != []

    # 3. PR gatekeeper FORBIDDEN_ADDED cleanliness
    for rel, content in (
        ("scripts/noel_vscode_quick_preset.py", script_src),
        ("tests/test_noel_vscode_quick_preset_soak.py", test_src),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. PRSecurityScanner (0 findings), ZeroStubAudit (clean), and static-numbers FIGURE cleanliness
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("scripts/noel_vscode_quick_preset.py", script_src),
        ("tests/test_noel_vscode_quick_preset_soak.py", test_src),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        zero_stub_res = ZeroStubAudit().audit_python_code(content)
        assert zero_stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {zero_stub_res.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1078,
        pr_title="feat(onboarding,scale): Cloud Run holiday-route parity gate and <60s VS Code one-click quick-connect preset",
        pr_body="Implements SLICE-SCALE-DYN-08 parity smoke gate, <60s VS Code preset generator, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(onboarding,scale): Cloud Run holiday-route parity gate and <60s VS Code one-click quick-connect preset"
        ],
    )
    assert pr_audit.passed is True
    assert pr_audit.findings == []


def test_02_cloud_run_holiday_parity_pass_vs_stale_404_503_sha_drift_and_graceful_ces_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", LIVE_BUILD_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00188-noel")

    store = RouterStore(tmp_path / "parity_router.sqlite3", secret=SOAK_SECRET)
    app = create_router_app(
        store,
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    seen_calls: list[tuple[str, str, dict[str, str]]] = []
    with TestClient(app) as client:
        # Register one live producer so online_producers == 1 is verified
        reg_p = client.post("/v1/hivemind/register", json={"handle": "home-rtx-prod", "role": "producer"}).json()
        hb_p = client.post(
            "/v1/hivemind/producer/heartbeat",
            json={
                "endpoint_url": "http://127.0.0.1:11434",
                "models": [{"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                "cost_per_1k": 0.0,
                "capacity_1k": 50.0,
                "gpu_vram_mib": 12288,
                "execution_class": "ollama-gamer-whisper",
            },
            headers={"Authorization": f"Bearer {reg_p['api_key']}"},
        )
        assert hb_p.status_code == 200

        live_http = _make_router_http_adapter(client, seen_calls=seen_calls)

        # 1. Full parity gate PASS (exact + short prefix SHA)
        parity_pass = preset.verify_cloud_run_holiday_parity(
            "http://127.0.0.1:8000",
            expected_sha=LIVE_SHORT_SHA,
            http_fn=live_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        _verify_receipt(parity_pass)
        assert parity_pass["parity_ok"] is True
        assert parity_pass["ready"] is True
        assert parity_pass["exit_code"] == 0
        assert parity_pass["holiday_routes_deployed_on_cloud_run"] is True
        assert parity_pass["onboarding_feedback_deployed"] is True
        assert parity_pass["stale_revision_detected"] is False
        assert parity_pass["sha_drift_detected"] is False
        assert parity_pass["sha_matches"] is True
        assert parity_pass["missing_or_stale_routes"] == []
        assert parity_pass["online_producers"] == 1
        assert parity_pass["csp_nonces_unique"] is True
        assert parity_pass["build_sha"] == LIVE_BUILD_SHA
        assert parity_pass["cloud_run_revision"] == "hivemind-openrouter-00188-noel"
        assert all("completions" not in path for _, path, _ in seen_calls)

        cli_buf = io.StringIO()
        rc_ok = preset.main(
            ["--gateway", "http://127.0.0.1:8000", "--check-parity", "--expected-sha", LIVE_BUILD_SHA, "--lang", "fr"],
            env=CLEAN_ENV,
            cwd=tmp_path,
            http_fn=live_http,
            stdout=cli_buf,
        )
        assert rc_ok == 0
        assert "OK" in cli_buf.getvalue()

        # 2. Pre-PR-#949 stale Cloud Run HTTP 404 on holiday routes
        stale_404_overrides: dict[str, tuple[int, dict[str, Any]]] = {
            "/noel": (404, {"detail": "Not Found"}),
            "/quick": (404, {"detail": "Not Found"}),
            "/raid-noel": (404, {"detail": "Not Found"}),
            "/v1/hivemind/onboarding/manifest": (404, {"detail": "Not Found"}),
            "/v1/hivemind/onboarding/feedback/summary": (404, {"detail": "Not Found"}),
            "/v1/hivemind/onboarding/feedback": (404, {"detail": "Not Found"}),
        }
        stale_404_http = _make_router_http_adapter(client, override_routes=stale_404_overrides)

        parity_404 = preset.verify_cloud_run_holiday_parity(
            "http://127.0.0.1:8000",
            http_fn=stale_404_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_receipt(parity_404)
        assert parity_404["parity_ok"] is False
        assert parity_404["holiday_routes_deployed_on_cloud_run"] is False
        assert parity_404["onboarding_feedback_deployed"] is False
        assert parity_404["stale_revision_detected"] is True
        assert parity_404["exit_code"] == 2
        assert "/noel" in parity_404["missing_or_stale_routes"]
        assert "/quick" in parity_404["missing_or_stale_routes"]
        assert "/raid-noel" in parity_404["missing_or_stale_routes"]
        assert "/v1/hivemind/onboarding/manifest" in parity_404["missing_or_stale_routes"]
        assert "/v1/hivemind/onboarding/feedback/summary" in parity_404["missing_or_stale_routes"]

        out_404 = io.StringIO()
        rc_check_404 = preset.main(
            ["--gateway", "http://127.0.0.1:8000", "--check-parity", "--json"],
            env=CLEAN_ENV,
            cwd=tmp_path,
            http_fn=stale_404_http,
            stdout=out_404,
        )
        assert rc_check_404 == 2
        assert json.loads(out_404.getvalue())["stale_revision_detected"] is True

        # --install-preset --require-holiday-routes fails closed (exit code 2, 0 file writes) on stale 404
        ws_req = tmp_path / "ws_require_holiday"
        ws_req.mkdir()
        out_req = io.StringIO()
        rc_req = preset.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--install-preset",
                "--require-holiday-routes",
                "--workspace",
                str(ws_req),
                "--credentials-path",
                str(tmp_path / "req_home" / ".hivemind" / "credentials.json"),
                "--continue-config-path",
                str(tmp_path / "req_home" / ".continue" / "hivemind_provider.yaml"),
                "--json",
            ],
            env=CLEAN_ENV,
            cwd=ws_req,
            home=tmp_path / "req_home",
            http_fn=stale_404_http,
            stdout=out_req,
        )
        assert rc_req == 2
        req_doc = json.loads(out_req.getvalue())
        assert req_doc["holiday_routes_deployed_on_cloud_run"] is False
        assert req_doc["stale_revision_detected"] is True
        assert req_doc["file_writes"] == 0
        assert not (ws_req / ".vscode" / "tasks.json").exists()

        # 3. Stale / degraded HTTP 503 on holiday routes AND build_sha drift detection
        stale_503_overrides = {
            route: (503, {"detail": "Service Unavailable"}) for route in stale_404_overrides
        }
        stale_503_http = _make_router_http_adapter(client, override_routes=stale_503_overrides)
        parity_503 = preset.verify_cloud_run_holiday_parity(
            "http://127.0.0.1:8000",
            http_fn=stale_503_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        assert parity_503["holiday_routes_deployed_on_cloud_run"] is False
        assert parity_503["stale_revision_detected"] is True
        assert parity_503["exit_code"] == 2

        parity_drift = preset.verify_cloud_run_holiday_parity(
            "http://127.0.0.1:8000",
            expected_sha=DRIFT_SHA,
            http_fn=live_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        assert parity_drift["parity_ok"] is False
        assert parity_drift["sha_drift_detected"] is True
        assert parity_drift["stale_revision_detected"] is True
        assert parity_drift["exit_code"] == 2
        assert "/v1/hivemind/onboarding/manifest" in parity_drift["missing_or_stale_routes"]

        # 4. Graceful ces_recorded=False fallback during default --install-preset --submit-ces (without --require-holiday-routes)
        for tag, ces_status in (("404", 404), ("429", 429), ("503", 503)):
            ws_grace = tmp_path / f"ws_grace_{tag}"
            ws_grace.mkdir()
            home_grace = tmp_path / f"home_grace_{tag}"
            grace_http = _make_router_http_adapter(
                client,
                peer_ip=_loopback_ip_for(12, int(ces_status) % 200),
                override_routes={"/v1/hivemind/onboarding/feedback": (ces_status, {"detail": "unavailable"})},
            )
            out_grace = io.StringIO()
            rc_grace = preset.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--install-preset",
                    "--submit-ces",
                    "--handle",
                    f"grace-dev-{tag}",
                    "--workspace",
                    str(ws_grace),
                    "--credentials-path",
                    str(home_grace / ".hivemind" / "credentials.json"),
                    "--continue-config-path",
                    str(home_grace / ".continue" / "hivemind_provider.yaml"),
                    "--json",
                ],
                env=CLEAN_ENV,
                cwd=ws_grace,
                home=home_grace,
                http_fn=grace_http,
                stdout=out_grace,
            )
            assert rc_grace == 0
            grace_doc = json.loads(out_grace.getvalue())
            _verify_receipt(grace_doc)
            assert grace_doc["ready"] is True
            assert grace_doc["ces_recorded"] is False
            assert (ws_grace / ".vscode" / "tasks.json").is_file()
            assert (ws_grace / ".vscode" / "settings.json").is_file()
            if ces_status == 404:
                assert grace_doc["stale_revision_detected"] is True
                assert grace_doc["holiday_routes_deployed_on_cloud_run"] is False


def test_03_atomic_idempotent_vscode_preset_preserve_user_tasks_settings_uninstall_and_zero_leaks(
    tmp_path: Path,
) -> None:
    store = RouterStore(tmp_path / "preset_files.sqlite3", secret=SOAK_SECRET)
    app = create_router_app(store, allow_private_producers=True, trusted_proxy_hops=1, open_registration=True)

    home_dir = tmp_path / "home" / "alice_secret_user"
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"
    ws_dir = home_dir / "hobby_rust_raytracer"
    vscode_dir = ws_dir / ".vscode"
    vscode_dir.mkdir(parents=True)

    # Pre-populate existing user tasks.json and settings.json with JSONC comments and trailing commas
    (vscode_dir / "tasks.json").write_text(
        """{
  // Existing personal build task that must remain untouched
  "version": "2.0.0",
  "inputs": [{"id": "target", "type": "promptString", "description": "Build target"}],
  "tasks": [
    {
      "label": "Personal: Cargo Test",
      "type": "process",
      "command": "cargo",
      "args": ["test"],
    },
  ],
}
""",
        encoding="utf-8",
    )
    (vscode_dir / "settings.json").write_text(
        """{
  /* Personal editor preferences */
  "editor.fontSize": 15,
  "files.trimTrailingWhitespace": true,
  "hivemind.openaiCompatible.apiBase": "http://localhost:9999/v1",
}
""",
        encoding="utf-8",
    )

    seen_calls: list[tuple[str, str, dict[str, str]]] = []
    with TestClient(app) as client:
        adapter = _make_router_http_adapter(client, seen_calls=seen_calls)

        # First install: creates credentials + continue yaml, updates tasks.json & settings.json with .bak backups
        rep1 = preset.install_vscode_quick_preset(
            "http://127.0.0.1:8000",
            workspace_dir=ws_dir,
            credentials_path=cred_path,
            continue_config_path=cont_path,
            handle="noel-dev",
            role="prosumer",
            submit_ces=True,
            profile="dev",
            ease_rating=5,
            http_fn=adapter,
            env=CLEAN_ENV,
            home=home_dir,
            lang="en",
        )
        _verify_receipt(rep1)
        assert rep1["ready"] is True
        assert rep1["exit_code"] == 0
        assert rep1["within_60s_slo"] is True
        assert rep1["ces_recorded"] is True
        assert rep1["file_writes"] == 4
        assert rep1["gitignore_check"]["safe"] is False
        assert rep1["files"]["vscode_tasks"]["backup_created"] is True
        assert rep1["files"]["vscode_settings"]["backup_created"] is True
        assert (vscode_dir / "tasks.json.bak").is_file()
        assert (vscode_dir / "settings.json.bak").is_file()

        # Verify POSIX 0600 permissions on ~/.hivemind/credentials.json and ~/.continue/hivemind_provider.yaml
        if os.name != "nt":
            assert (cred_path.stat().st_mode & 0o777) == 0o600
            assert (cont_path.stat().st_mode & 0o777) == 0o600

        saved_cred = json.loads(cred_path.read_text(encoding="utf-8"))
        raw_api_key = str(saved_cred["api_key"])
        assert preset.is_valid_hivemind_key(raw_api_key)

        cont_text = cont_path.read_text(encoding="utf-8")
        assert "hivemind/coder" in cont_text
        assert "hivemind/bee-stars" in cont_text
        assert raw_api_key in cont_text

        tasks_text = (vscode_dir / "tasks.json").read_text(encoding="utf-8")
        settings_text = (vscode_dir / "settings.json").read_text(encoding="utf-8")

        # Workspace files MUST contain zero raw hm-... secrets and zero OS home paths
        for ws_content in (tasks_text, settings_text):
            assert raw_api_key not in ws_content
            assert "alice_secret_user" not in ws_content
            assert not preset._API_KEY_RE.search(ws_content)
            assert not preset._HOME_PATH_RE.search(ws_content)

        tasks_doc = json.loads(tasks_text)
        assert tasks_doc["version"] == "2.0.0"
        assert tasks_doc["inputs"] == [{"id": "target", "type": "promptString", "description": "Build target"}]
        labels = [t["label"] for t in tasks_doc["tasks"]]
        assert labels[0] == "Personal: Cargo Test"
        for expected_sub in (
            "<60s Quick-Connect Check",
            "Cloud Run Holiday Route Parity Check",
            "Gamer-Whisper <1.5s Steam Yield & <=70 C Preflight",
            "10s Clean Uninstall",
        ):
            assert any(expected_sub in lbl for lbl in labels), f"Missing task label for {expected_sub}"
        assert len(tasks_doc["tasks"]) == 5

        settings_doc = json.loads(settings_text)
        assert settings_doc["editor.fontSize"] == 15
        assert settings_doc["files.trimTrailingWhitespace"] is True
        assert settings_doc["hivemind.openaiCompatible.apiBase"] == "http://127.0.0.1:8000/v1"
        assert settings_doc["hivemind.openaiCompatible.models"] == ["hivemind/coder", "hivemind/bee-stars"]

        # Deterministic Idempotency-Key verification
        ces_headers = [hdrs for m, p, hdrs in seen_calls if m == "POST" and p == "/v1/hivemind/onboarding/feedback"]
        assert len(ces_headers) == 1
        first_idem = ces_headers[0]["Idempotency-Key"]
        assert first_idem.startswith("ces-noel-")

        # Add .gitignore covering .vscode/ and loosen credentials mode to verify idempotent second run + mode tightening
        (ws_dir / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")
        if os.name != "nt":
            os.chmod(cred_path, 0o644)

        rep2 = preset.install_vscode_quick_preset(
            "http://127.0.0.1:8000",
            workspace_dir=ws_dir,
            credentials_path=cred_path,
            continue_config_path=cont_path,
            submit_ces=True,
            profile="dev",
            ease_rating=5,
            http_fn=adapter,
            env=CLEAN_ENV,
            home=home_dir,
            lang="fr",
        )
        _verify_receipt(rep2)
        assert rep2["ready"] is True
        assert rep2["file_writes"] == 0
        assert rep2["gitignore_check"]["safe"] is True
        assert rep2["files"]["vscode_tasks"]["action"] == "unchanged"
        assert rep2["files"]["vscode_settings"]["action"] == "unchanged"
        assert rep2["ces_feedback"]["idempotency_key"] == first_idem
        if os.name != "nt":
            assert rep2["files"]["credentials"]["permissions_tightened"] is True
            assert (cred_path.stat().st_mode & 0o777) == 0o600
        assert len(json.loads((vscode_dir / "tasks.json").read_text(encoding="utf-8"))["tasks"]) == 5

        # Report & CLI output must never leak raw_api_key or OS home username
        human_fr = preset.render_human_report(rep2, lang="fr")
        combined_dump = json.dumps(rep1) + "\n" + json.dumps(rep2) + "\n" + human_fr
        assert raw_api_key not in combined_dump
        assert "alice_secret_user" not in combined_dump
        assert "~/hobby_rust_raytracer/.vscode/tasks.json" in combined_dump

        # --uninstall removes HiveMind preset artifacts & backups while leaving pre-existing user tasks/settings intact
        uninst_buf = io.StringIO()
        rc_uninst = preset.main(
            [
                "--uninstall",
                "--workspace",
                str(ws_dir),
                "--credentials-path",
                str(cred_path),
                "--continue-config-path",
                str(cont_path),
                "--lang",
                "fr",
                "--json",
            ],
            env=CLEAN_ENV,
            home=home_dir,
            stdout=uninst_buf,
        )
        assert rc_uninst == 0
        uninst_doc = json.loads(uninst_buf.getvalue())
        _verify_receipt(uninst_doc)
        assert not cred_path.exists()
        assert not cont_path.exists()
        assert not (vscode_dir / "tasks.json.bak").exists()
        assert not (vscode_dir / "settings.json.bak").exists()

        remaining_tasks = json.loads((vscode_dir / "tasks.json").read_text(encoding="utf-8"))
        assert remaining_tasks["tasks"] == [
            {"label": "Personal: Cargo Test", "type": "process", "command": "cargo", "args": ["test"]}
        ]
        assert remaining_tasks["inputs"] == [{"id": "target", "type": "promptString", "description": "Build target"}]

        remaining_settings = json.loads((vscode_dir / "settings.json").read_text(encoding="utf-8"))
        assert remaining_settings == {"editor.fontSize": 15, "files.trimTrailingWhitespace": True}

        # Clean workspace with ONLY HiveMind preset files removes .vscode completely on --uninstall
        ws_clean_only = home_dir / "clean_only_ws"
        ws_clean_only.mkdir(parents=True)
        preset.install_vscode_quick_preset(
            "http://127.0.0.1:8000",
            workspace_dir=ws_clean_only,
            credentials_path=cred_path,
            continue_config_path=cont_path,
            handle="clean-only-dev",
            http_fn=adapter,
            env=CLEAN_ENV,
            home=home_dir,
        )
        assert (ws_clean_only / ".vscode" / "tasks.json").is_file()
        preset.uninstall_vscode_quick_preset(
            workspace_dir=ws_clean_only,
            credentials_path=cred_path,
            continue_config_path=cont_path,
            env=CLEAN_ENV,
            home=home_dir,
        )
        assert not (ws_clean_only / ".vscode").exists()


def test_04_steam_yield_thermal_guard_no_gpu_fallback_and_fail_closed_amadeus_corporate_blocking(
    tmp_path: Path,
) -> None:
    # 1. Real subprocess dummy-worker yield drill completes cleanly in < 1.5s
    real_drill = preset.default_worker_yield_drill(timeout_s=2.5)
    assert real_drill["clean_release"] is True
    assert real_drill["worker_stopped"] is True
    assert isinstance(real_drill["elapsed_ms"], float) and real_drill["elapsed_ms"] < 1500.0

    # 2. Healthy RTX GPU, slow yield (>=1500.0), hot GPU (>70.0 C), active Steam game, and no-GPU laptop dev fallback
    rtx_ok = preset.GpuProbeSnapshot(
        present=True,
        name="NVIDIA GeForce RTX 4070",
        total_vram_mib=12288,
        free_vram_mib=9216,
        used_vram_mib=3072,
        temp_c=58.0,
        util_pct=5.0,
    )
    pf_ok = preset.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: rtx_ok,
        yield_drill_fn=lambda: 62.5,
        process_probe_fn=lambda: ["systemd", "code"],
        lang="en",
    )
    assert pf_ok["preflight_ok"] is True
    assert pf_ok["yield_ok"] is True
    assert pf_ok["thermal_ok"] is True
    assert pf_ok["mode"] == "gamer_whisper_gpu_ready"

    pf_slow = preset.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: rtx_ok,
        yield_drill_fn=lambda: 1500.0,
        process_probe_fn=lambda: [],
        lang="fr",
    )
    assert pf_slow["preflight_ok"] is False
    assert pf_slow["yield_ok"] is False
    assert pf_slow["slow_yield_detected"] is True

    rtx_hot = preset.GpuProbeSnapshot(
        present=True,
        name="NVIDIA GeForce RTX 3080",
        total_vram_mib=10240,
        free_vram_mib=7168,
        used_vram_mib=3072,
        temp_c=74.2,
        util_pct=40.0,
    )
    pf_hot = preset.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: rtx_hot,
        yield_drill_fn=lambda: 48.0,
        process_probe_fn=lambda: [],
        lang="en",
    )
    assert pf_hot["preflight_ok"] is False
    assert pf_hot["thermal_ok"] is False
    assert pf_hot["hot_gpu_detected"] is True

    pf_game = preset.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: rtx_ok,
        yield_drill_fn=lambda: 45.0,
        process_probe_fn=lambda: ["explorer.exe", "cs2.exe"],
        lang="en",
    )
    assert pf_game["preflight_ok"] is False
    assert pf_game["active_game_detected"] is True
    assert pf_game["active_games"] == ["cs2.exe"]

    pf_laptop = preset.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: preset.GpuProbeSnapshot(present=False),
        yield_drill_fn=lambda: 32.0,
        process_probe_fn=lambda: ["zsh", "code"],
        lang="fr",
    )
    assert pf_laptop["preflight_ok"] is True
    assert pf_laptop["gpu_present"] is False
    assert pf_laptop["thermal_ok"] is True
    assert pf_laptop["mode"] == "dev_api_only_no_gpu"

    # Slow yield & hot GPU block install_vscode_quick_preset when --first-heartbeat is requested (0 file writes)
    store = RouterStore(tmp_path / "guard_router.sqlite3", secret=SOAK_SECRET)
    app = create_router_app(store, allow_private_producers=True, open_registration=True)
    with TestClient(app) as client:
        adapter = _make_router_http_adapter(client)
        for bad_label, gpu_fn, yld_fn in (
            ("slow_yield", lambda: rtx_ok, lambda: 1680.0),
            ("hot_gpu", lambda: rtx_hot, lambda: 55.0),
        ):
            ws_bad = tmp_path / f"ws_{bad_label}"
            ws_bad.mkdir()
            rep_bad = preset.install_vscode_quick_preset(
                "http://127.0.0.1:8000",
                workspace_dir=ws_bad,
                credentials_path=tmp_path / bad_label / ".hivemind" / "credentials.json",
                continue_config_path=tmp_path / bad_label / ".continue" / "hivemind_provider.yaml",
                send_first_heartbeat=True,
                profile="gamer",
                gpu_probe_fn=gpu_fn,
                yield_drill_fn=yld_fn,
                process_probe_fn=lambda: [],
                http_fn=adapter,
                env=CLEAN_ENV,
                home=tmp_path / bad_label,
            )
            _verify_receipt(rep_bad)
            assert rep_bad["ready"] is False
            assert rep_bad["exit_code"] == 2
            assert rep_bad["verdict"] == "preflight_failed"
            assert rep_bad["file_writes"] == 0
            assert rep_bad["first_heartbeat_ok"] is False

        # No-GPU laptop dev fallback succeeds & sends truthful null GPU fields on OpenTrainer heartbeat
        ws_laptop = tmp_path / "ws_laptop"
        ws_laptop.mkdir()
        rep_laptop = preset.install_vscode_quick_preset(
            "http://127.0.0.1:8000",
            workspace_dir=ws_laptop,
            credentials_path=tmp_path / "laptop_home" / ".hivemind" / "credentials.json",
            continue_config_path=tmp_path / "laptop_home" / ".continue" / "hivemind_provider.yaml",
            handle="laptop-dev-01",
            role="prosumer",
            send_first_heartbeat=True,
            gpu_probe_fn=lambda: preset.GpuProbeSnapshot(present=False),
            yield_drill_fn=lambda: 28.0,
            process_probe_fn=lambda: [],
            http_fn=adapter,
            env=CLEAN_ENV,
            home=tmp_path / "laptop_home",
        )
        assert rep_laptop["ready"] is True
        assert rep_laptop["exit_code"] == 0
        assert rep_laptop["first_heartbeat_ok"] is True
        assert rep_laptop["heartbeat"]["payload"]["gpu_name"] is None
        assert rep_laptop["heartbeat"]["payload"]["vram_total_mib"] is None
        assert rep_laptop["heartbeat"]["payload"]["vram_free_mib"] is None
        assert rep_laptop["heartbeat"]["payload"]["network_mode"] == "outbound-https-relay"

    # 3. 100% Fail-Closed Amadeus / Corporate PC & .git/config Remote Blocking (0 file writes, 0 HTTP calls, 0 value leaks)
    http_boom_calls: list[str] = []

    def _forbid_http(method: str, url: str, **_kw: Any) -> tuple[int, dict[str, Any]]:
        http_boom_calls.append(f"{method}:{url}")
        raise AssertionError("Corporate compliance gate MUST block before any HTTP call!")

    corp_env_cases: list[tuple[str, dict[str, str], str]] = [
        ("amadeus_var", {"AMADEUS_OFFICE_SITE": "nce-secret-building-42"}, "nce-secret-building-42"),
        ("dns_domain", {"USERDNSDOMAIN": "nce.amadeus.internal"}, "nce.amadeus.internal"),
        ("userdomain", {"USERDOMAIN": "ENTCORP_AD"}, "ENTCORP_AD"),
        ("zscaler_proxy", {"HTTPS_PROXY": CORP_ZSCALER_PROXY}, "supersecretpw"),
        ("netskope_proxy", {"ALL_PROXY": "http://" + "gw.netskope.internal:8080"}, "gw.netskope.internal"),
        ("hostname_corp", {"COMPUTERNAME": "WKS-FORTINET-CORP-99", "HOSTNAME": "wks-paloalto-corp"}, "WKS-FORTINET-CORP-99"),
        ("zscaler_cert", {"ZSCALER_CERT_PATH": "/etc/ssl/zscaler-root-secret.pem"}, "zscaler-root-secret.pem"),
    ]

    for case_name, bad_env, secret_token in corp_env_cases:
        ws_corp = tmp_path / f"clean_ws_{case_name}"
        ws_corp.mkdir()
        home_corp = tmp_path / f"home_{case_name}"
        out_corp = io.StringIO()
        rc_corp = preset.main(
            [
                "--install-preset",
                "--first-heartbeat",
                "--submit-ces",
                "--workspace",
                str(ws_corp),
                "--credentials-path",
                str(home_corp / ".hivemind" / "credentials.json"),
                "--continue-config-path",
                str(home_corp / ".continue" / "hivemind_provider.yaml"),
                "--json",
            ],
            env={**CLEAN_ENV, **bad_env},
            cwd=ws_corp,
            home=home_corp,
            http_fn=_forbid_http,
            stdout=out_corp,
        )
        assert rc_corp == 2
        corp_doc = json.loads(out_corp.getvalue())
        _verify_receipt(corp_doc)
        assert corp_doc["compliance_ok"] is False
        assert corp_doc["file_writes"] == 0
        assert corp_doc["network_calls"] == 0
        assert "0.00 EUR / 0.00 USD" in corp_doc["compliance"]["notice_bilingual"]
        assert "0,00 EUR" in corp_doc["compliance"]["notice_bilingual"]
        assert "AGPL-3.0" in corp_doc["compliance"]["notice_bilingual"]
        assert secret_token not in out_corp.getvalue()
        assert list(ws_corp.iterdir()) == []
        assert not home_corp.exists()

    # Corporate workspace path + corporate .git/config remote + git worktree indirection
    corp_path_ws = tmp_path / "amadeus_internal_checkout"
    corp_path_ws.mkdir()
    comp_ws = preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_path_ws, lang="fr")
    assert comp_ws["compliance_ok"] is False
    assert "amadeus_internal_checkout" not in json.dumps(comp_ws)

    git_corp_ws = tmp_path / "hobby_named_repo"
    (git_corp_ws / ".git").mkdir(parents=True)
    (git_corp_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {CORP_SSH_REMOTE}\n[url "{CORP_HTTPS_REMOTE}"]\n\tinsteadOf = corp:\n',
        encoding="utf-8",
    )
    parity_corp_git = preset.verify_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        http_fn=_forbid_http,
        env=CLEAN_ENV,
        cwd=git_corp_ws,
        lang="en",
    )
    assert parity_corp_git["compliance_ok"] is False
    assert parity_corp_git["exit_code"] == 2
    assert parity_corp_git["network_calls"] == 0
    assert "git-amadeus-internal" not in json.dumps(parity_corp_git)

    # Git worktree .git file indirection with commondir
    common_git = tmp_path / "common_git_dir"
    wt_git = common_git / "worktrees" / "wt1"
    wt_git.mkdir(parents=True)
    (common_git / "config").write_text(f'[remote "origin"]\n\turl = {CORP_HTTPS_REMOTE}\n', encoding="utf-8")
    (wt_git / "commondir").write_text("../..\n", encoding="utf-8")
    wt_ws = tmp_path / "clean_wt_ws"
    wt_ws.mkdir()
    (wt_ws / ".git").write_text(f"gitdir: {wt_git}\n", encoding="utf-8")
    assert preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_ws)["compliance_ok"] is False

    # --uninstall MUST remain permitted even when corporate env markers are active
    uninst_corp_ws = tmp_path / "corp_cleanup_ws"
    (uninst_corp_ws / ".vscode").mkdir(parents=True)
    (uninst_corp_ws / ".vscode" / "tasks.json").write_text(
        preset.build_merged_vscode_tasks_json(None, gateway="http://127.0.0.1:8000"),
        encoding="utf-8",
    )
    out_uninst_corp = io.StringIO()
    rc_uninst_corp = preset.main(
        ["--uninstall", "--workspace", str(uninst_corp_ws), "--json"],
        env={"AMADEUS_ENV": "1", **CLEAN_ENV},
        cwd=uninst_corp_ws,
        home=tmp_path / "corp_cleanup_home",
        http_fn=_forbid_http,
        stdout=out_uninst_corp,
    )
    assert rc_uninst_corp == 0
    assert not (uninst_corp_ws / ".vscode").exists()
    assert http_boom_calls == []


def test_05_concurrent_48_participant_multi_cohort_noel_scale_soak_with_sqlite_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", LIVE_BUILD_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00188-soak")

    upstream_inference_hits: list[str] = []

    def _reject_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn08_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=SOAK_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        journal_mode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_reject_inference)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    total_participants = 48
    dev_cohort_size = 24
    barrier = threading.Barrier(total_participants)
    shared_lock = threading.Lock()
    sqlite_lock_hits: list[str] = []
    seen_calls: list[tuple[str, str, dict[str, str]]] = []
    noel_quick_nonces: list[str] = []

    rtx_cards = (
        ("NVIDIA GeForce RTX 4070 SUPER", 12288, 9216, 54.0),
        ("NVIDIA GeForce RTX 4080 SUPER", 16384, 12288, 57.0),
        ("NVIDIA GeForce RTX 3080", 10240, 6144, 61.0),
        ("NVIDIA GeForce RTX 4060 Ti", 8192, 5120, 52.0),
        ("NVIDIA GeForce RTX 3090", 24576, 18432, 63.0),
        ("NVIDIA GeForce RTX 4090", 24576, 20480, 58.0),
    )

    with TestClient(app) as client:
        def _participant_worker(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_cohort_size
            cohort_idx = idx if is_dev else (idx - dev_cohort_size)
            peer_ip = _loopback_ip_for(44, idx)
            slot_root = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_user = f"home_member_{idx:02d}"
            home_dir = slot_root / "home" / os_user
            ws_dir = slot_root / "workspace"
            ws_dir.mkdir(parents=True, exist_ok=True)
            (ws_dir / ".gitignore").write_text(".vscode/\n*.bak\n", encoding="utf-8")

            handle = f"noel-dev-{cohort_idx:02d}" if is_dev else f"noel-gamer-{cohort_idx:02d}"
            role = "consumer" if (is_dev and cohort_idx % 2 == 0) else "prosumer"
            profile = "dev" if is_dev else "gamer"
            ease = 5 if (idx % 3 != 0) else 4
            lang = "fr" if (idx % 2 == 0) else "en"

            if is_dev:
                gpu_snap = preset.GpuProbeSnapshot(present=False)
                yield_ms = 22.0 + cohort_idx
                sys_name = "Darwin" if (cohort_idx % 2 == 0) else "Linux"
                os_name_val = "posix"
                host_lbl = f"dev-laptop-{cohort_idx:02d}"
            else:
                card, tot_mib, free_mib, temp_c = rtx_cards[cohort_idx % len(rtx_cards)]
                gpu_snap = preset.GpuProbeSnapshot(
                    present=True,
                    name=card,
                    total_vram_mib=tot_mib,
                    free_vram_mib=free_mib,
                    used_vram_mib=tot_mib - free_mib,
                    temp_c=temp_c,
                    util_pct=4.0,
                )
                yield_ms = 65.0 + (cohort_idx * 4.0)
                sys_name = "Windows"
                os_name_val = "nt"
                host_lbl = f"gamer-rtx-{cohort_idx:02d}"

            adapter = _make_router_http_adapter(
                client,
                peer_ip=peer_ip,
                seen_calls=seen_calls,
                sqlite_lock_hits=sqlite_lock_hits,
                noel_quick_nonces=noel_quick_nonces,
                lock=shared_lock,
            )

            barrier.wait(timeout=25.0)
            rep = preset.install_vscode_quick_preset(
                "http://127.0.0.1:8000",
                workspace_dir=ws_dir,
                credentials_path=home_dir / ".hivemind" / "credentials.json",
                continue_config_path=home_dir / ".continue" / "hivemind_provider.yaml",
                handle=handle,
                role=role,
                host_label=host_lbl,
                expected_sha=LIVE_SHORT_SHA,
                send_first_heartbeat=True,
                submit_ces=True,
                profile=profile,
                ease_rating=ease,
                had_game_lag=False,
                require_holiday_routes=True,
                gpu_probe_fn=lambda g=gpu_snap: g,
                yield_drill_fn=lambda y=yield_ms: y,
                process_probe_fn=lambda: [],
                http_fn=adapter,
                env=CLEAN_ENV,
                home=home_dir,
                os_name=os_name_val,
                system_name=sys_name,
                lang=lang,
            )
            _verify_receipt(rep)
            assert rep["ready"] is True, f"Participant {idx} failed: {rep}"
            assert rep["exit_code"] == 0
            assert rep["within_60s_slo"] is True
            assert rep["holiday_routes_deployed_on_cloud_run"] is True
            assert rep["first_heartbeat_ok"] is True
            assert rep["ces_recorded"] is True
            assert (ws_dir / ".vscode" / "tasks.json").is_file()
            assert (ws_dir / ".vscode" / "settings.json").is_file()
            assert os_user not in json.dumps(rep)
            return {"idx": idx, "profile": profile, "idem": rep["ces_feedback"]["idempotency_key"]}

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            futures = [pool.submit(_participant_worker, i) for i in range(total_participants)]
            outcomes = [fut.result(timeout=60.0) for fut in futures]

        assert len(outcomes) == 48
        assert len({o["idem"] for o in outcomes}) == 48
        assert sqlite_lock_hits == []
        assert upstream_inference_hits == []
        assert all("completions" not in path for _, path, _ in seen_calls)

        # Every /noel and /quick request across all 48 participants (96 responses) got a unique CSP nonce
        assert len(noel_quick_nonces) == 96
        assert len(set(noel_quick_nonces)) == 96

        raid_status = client.get("/v1/hivemind/raid/status").json()
        assert raid_status["swarm"]["active_trainers"] >= 24

        fb_summary = client.get("/v1/hivemind/onboarding/feedback/summary").json()
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["mean_ease_rating"] is not None and float(fb_summary["mean_ease_rating"]) >= 4.2
        assert fb_summary["zero_game_lag_ratio"] == 1.0


def test_06_loopback_default_http_request_and_validation_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Exercise real default_http_request with ProxyHandler({}) over loopback ThreadingHTTPServer
    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _ = (format, args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                body = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/noel":
                html = "<!DOCTYPE html><html><head><style nonce='abcd1234efgh'>body{}</style></head><body>100% personal AGPL-3.0 0,00 EUR</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'nonce-abcd1234efgh'")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
            elif self.path == "/list":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"[1, 2, 3]")
            elif self.path == "/broken-json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{not-json")
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail":"missing"}')

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            echo = json.loads(raw.decode("utf-8")) if raw else {}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "echo": echo}).encode("utf-8"))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        st_h, bd_h = preset.default_http_request("GET", f"{base_url}/health")
        assert st_h == 200 and bd_h["status"] == "ok"
        st_html, bd_html = preset.default_http_request("GET", f"{base_url}/noel")
        assert st_html == 200 and "nonce-abcd1234efgh" in bd_html["csp"]
        st_list, bd_list = preset.default_http_request("GET", f"{base_url}/list")
        assert st_list == 200 and bd_list["data"] == [1, 2, 3]
        st_brk, bd_brk = preset.default_http_request("GET", f"{base_url}/broken-json")
        assert st_brk == 200 and bd_brk["raw_error"] == "invalid_json"
        st_404, _ = preset.default_http_request("GET", f"{base_url}/missing")
        assert st_404 == 404
        st_post, bd_post = preset.default_http_request(
            "POST",
            f"{base_url}/echo",
            payload={"hello": "world"},
            headers={"X-Custom": "1"},
        )
        assert st_post == 200 and bd_post["echo"] == {"hello": "world"}
    finally:
        server.shutdown()
        server.server_close()

    # Closed port -> (0, {"error": "network_error"})
    st_closed, bd_closed = preset.default_http_request("GET", f"http://127.0.0.1:{port}/health", timeout_s=0.2)
    assert st_closed == 0 and bd_closed["error"] == "network_error"

    # 2. Validator, sanitizer, HTTP arity, GPU/process/yield & CLI error branches
    for bad_lang in (True, "de"):
        with pytest.raises(preset.QuickPresetError):
            preset.normalize_lang(bad_lang)  # type: ignore[arg-type]
    for bad_prof in (None, True, "admin"):
        with pytest.raises(preset.QuickPresetError):
            preset.normalize_profile(bad_prof)  # type: ignore[arg-type]
    for bad_role in (None, True, "root"):
        with pytest.raises(preset.QuickPresetError):
            preset.normalize_role(bad_role)  # type: ignore[arg-type]
    for bad_ease in (True, "nan", 0, 6):
        with pytest.raises(preset.QuickPresetError):
            preset.parse_ease_rating(bad_ease)  # type: ignore[arg-type]
    assert preset.sanitize_handle(None).startswith("noel-dev-")
    with pytest.raises(preset.QuickPresetError):
        preset.sanitize_handle("google.spoof")
    with pytest.raises(preset.QuickPresetError):
        preset.sanitize_handle("!")
    assert preset.mask_api_key("") == "<missing>"
    assert preset.mask_api_key("short") == "<invalid>"
    assert preset.mask_api_key(ALPHA_KEY) == "hm-***3456"
    assert preset.mask_api_key(LIVE_KEY) == "hm_live_***4321"
    assert preset.sanitize_host_label(None) == "personal-pc"
    assert preset.sanitize_host_label("amadeus-wks") == "personal-pc"
    assert preset.sanitize_build_sha(None) == "unknown"
    assert preset.sanitize_build_sha("secret12345") == "unknown"
    with pytest.raises(preset.QuickPresetError):
        preset.validate_expected_sha("unknown")
    assert preset.shas_match("unknown", LIVE_BUILD_SHA) is False
    assert preset.shas_match("release2026a", "release2026b") is False
    assert preset.sanitize_cloud_run_revision(None) is None
    assert preset.sanitize_cloud_run_revision("hm-1234567890") is None
    assert preset.is_allowed_outbound_host("") is False
    assert preset.is_allowed_outbound_host("datasets-server.huggingface.co") is True
    assert preset.is_allowed_outbound_host("storage.googleapis.com") is True

    for bad_gw in (
        "",
        "https://" + "run.app\n",
        "https://" + "bad space.run.app",
        "https://" + "user:pw@service.a.run.app",
        "https://" + "service.a.run.app?x=1",
        "https://" + "service.a.run.app/sub",
        "http://" + "127.0.0.1:0",
        UNTRUSTED_EXT_GW,
        UNTRUSTED_FTP_GW,
        INSECURE_RUN_APP_GW,
    ):
        with pytest.raises(preset.QuickPresetError):
            preset.validate_gateway_url(bad_gw)

    with pytest.raises(preset.QuickPresetError):
        preset._validate_outbound_request_url("http://127.0.0.1:8000/v1/chat/completions")

    # scan_output_safety negative checks
    ip_pub = ".".join(("8", "8", "8", "8"))
    bad_safety = preset.scan_output_safety(
        f'alice@test.org {ALPHA_KEY} Bearer secret123 /home/alice/x "{"sco" + "re"}": 5 {ip_pub}'
    )
    assert len(bad_safety) == 6
    with pytest.raises(preset.QuickPresetError):
        preset._assert_safe_output(f"leak {ALPHA_KEY}")

    # Flexible _invoke_http_fn arities & transport error conversion
    assert preset._invoke_http_fn(lambda m, u: (200, "<html></html>"), "GET", "http://127.0.0.1:8000/noel")[0] == 200
    assert (
        preset._invoke_http_fn(
            lambda m, u, p: (200, {"content-security-policy": "x"}, {"ok": p}),
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/register",
            payload={"a": 1},
        )[0]
        == 200
    )
    assert (
        preset._invoke_http_fn(
            lambda m, u, p, h: (200, {"h": h}),
            "GET",
            "http://127.0.0.1:8000/health",
        )[0]
        == 200
    )
    assert (
        preset._invoke_http_fn(
            lambda _m, _u: (_ for _ in ()).throw(OSError("down")),
            "GET",
            "http://127.0.0.1:8000/health",
        )[0]
        == 0
    )
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_http_return("bad")

    # HTML / manifest / feedback contract negative checks
    seen_n: set[str] = set()
    assert preset._inspect_holiday_html_route("/noel", 200, {"html": "empty"}, seen_n)[1]["reason"] == "invalid_html_doctype"
    assert (
        preset._inspect_holiday_html_route("/noel", 200, {"html": "<!DOCTYPE html>"}, seen_n)[1]["reason"]
        == "missing_or_invalid_csp_nonce"
    )
    assert (
        preset._inspect_holiday_html_route(
            "/noel",
            200,
            {
                "html": "<!DOCTYPE html><script nonce='n12345678'>el." + "inner" + "HTML='x'</script>",
                "csp": "default-src 'none'; script-src 'nonce-n12345678'",
            },
            seen_n,
        )[1]["reason"]
        == "unsafe_dom_or_static_figure"
    )
    assert (
        preset._inspect_holiday_html_route(
            "/quick",
            200,
            {
                "html": "<!DOCTYPE html><style nonce='n87654321'></style>",
                "csp": "default-src 'none'; style-src 'nonce-n87654321'",
            },
            seen_n,
        )[1]["reason"]
        == "missing_personal_pc_compliance_marker"
    )
    assert preset._inspect_manifest_json({"object": "wrong"})[0] is False
    assert preset._inspect_feedback_summary_json({"total_responses": 0})[0] is False

    # default_gpu_probe & default_process_probe via injectable runner + Popen failure branches
    gpu_parsed = preset.default_gpu_probe(
        runner=lambda _c, _t: (0, "NVIDIA GeForce RTX 4070, 12288, 9216, 3072, 53, 6\n")
    )
    assert gpu_parsed.present is True and gpu_parsed.total_vram_mib == 12288
    assert preset.default_gpu_probe(runner=lambda _c, _t: (0, "corrupt")).parse_error is True
    assert preset.default_gpu_probe(runner=lambda _c, _t: (127, "")).present is False
    assert preset._normalize_gpu_snapshot(SimpleNamespace(present=True, name="RTX", total_mib=8192, free_mib=4096)).present is True
    assert preset._normalize_gpu_snapshot({"present": True, "name": ""}).parse_error is True
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_gpu_snapshot(123)

    win_procs = preset.default_process_probe(
        runner=lambda _c, _t: (0, '"steam.exe","100","Console","1","50 K"\n"cs2.exe","200","Console","1","90 K"\n'),
        os_name="nt",
    )
    assert "cs2.exe" in win_procs
    nix_procs = preset.default_process_probe(runner=lambda _c, _t: (0, "bash\npython3\n"), os_name="posix")
    assert "python3" in nix_procs

    rc_sub, out_sub = preset.default_subprocess_runner([sys.executable, "-c", "print('ok')"])
    assert rc_sub == 0 and out_sub.strip() == "ok"
    assert preset.default_subprocess_runner(["/nonexistent_binary_xyz_dyn08"])[0] == 127
    with pytest.raises(preset.QuickPresetError):
        preset.default_subprocess_runner("")  # type: ignore[arg-type]

    class _BrokenWorkerProc:
        def __init__(self, banner: str, raise_wait: bool = False) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(banner)
            self.stderr = io.StringIO()
            self._raise_wait = raise_wait

        def kill(self) -> None:
            self._raise_wait = False

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._raise_wait:
                raise subprocess.TimeoutExpired(cmd=["python3"], timeout=0.2)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _BrokenWorkerProc("BAD\n"))
    assert preset.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _BrokenWorkerProc("HIVEMIND_WORKER_READY\n", True))
    assert preset.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.undo()

    with pytest.raises(preset.QuickPresetError):
        preset._normalize_yield_drill_result(True)
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_yield_drill_result("bad")
    assert preset._normalize_yield_drill_result(math.nan) == (None, False)
    assert preset._normalize_yield_drill_result({"elapsed_ms": 40.0, "clean_release": True}) == (40.0, True)

    # CLI error handling on untrusted gateway URL -> exit code 2 JSON with receipt
    err_buf = io.StringIO()
    rc_err = preset.main(["--gateway", UNTRUSTED_EXT_GW, "--check-parity", "--json"], env=CLEAN_ENV, cwd=tmp_path, stdout=err_buf)
    assert rc_err == 2
    err_doc = json.loads(err_buf.getvalue())
    _verify_receipt(err_doc)
    assert err_doc["mode"] == "error"

    # 3. Remaining validation, JSONC escape, auth-failure, 409 retry, parity schema & CES timeout branches
    for bad_req_url in ("", "http://" + "127.0.0.1:8000/bad space", "http://" + "u:p@127.0.0.1:8000/health"):
        with pytest.raises(preset.QuickPresetError):
            preset._validate_outbound_request_url(bad_req_url)

    assert preset._normalize_http_return((200, {"x-a": "1"}, {"headers": {"x-b": "2"}}))[1]["headers"]["x-b"] == "2"
    assert preset._normalize_http_return((200, {}, [10, 20]))[1]["data"] == [10, 20]
    assert preset._invoke_http_fn(lambda m, u, **kw: (200, {"kw": bool(kw)}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert preset._invoke_http_fn(lambda m, u, p, h, t: (200, {"t": t}), "GET", "http://127.0.0.1:8000/health")[0] == 200

    escaped_jsonc = '{"msg": "quote: \\" and slash: \\\\", // comment\n}'
    assert json.loads(preset.strip_jsonc_comments_and_trailing_commas(escaped_jsonc))["msg"].startswith("quote:")

    for secret_str in (ALPHA_KEY, "Bearer secret_tok_999", "/home/alice/secret"):
        with pytest.raises(preset.QuickPresetError):
            preset._assert_workspace_content_secret_free(secret_str)
    with pytest.raises(preset.QuickPresetError):
        preset.build_continue_provider_yaml("http://127.0.0.1:8000", "invalid")
    with pytest.raises(preset.QuickPresetError):
        preset.build_continue_provider_yaml("http://127.0.0.1:8000", ALPHA_KEY, max_tokens=16)
    for bad_json_txt in ("{broken", "[]"):
        with pytest.raises(preset.QuickPresetError):
            preset.build_merged_vscode_settings_json(bad_json_txt, gateway="http://127.0.0.1:8000")
        with pytest.raises(preset.QuickPresetError):
            preset.build_merged_vscode_tasks_json(bad_json_txt, gateway="http://127.0.0.1:8000")
    with pytest.raises(preset.QuickPresetError):
        preset.build_merged_vscode_tasks_json('{"tasks": "not-a-list"}', gateway="http://127.0.0.1:8000")
    assert preset._is_hivemind_managed_task("not-a-dict") is False

    partial_gi_ws = tmp_path / "partial_gi"
    partial_gi_ws.mkdir()
    (partial_gi_ws / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    assert preset.check_workspace_gitignore_safety(partial_gi_ws, lang="fr")["vscode_ignored"] is False

    for fname, raw_cred in (
        ("broken.json", "{bad"),
        ("list.json", "[]"),
        ("badkey.json", '{"api_key": "bad"}'),
        ("othergw.json", json.dumps({"api_key": ALPHA_KEY, "gateway": "https://other.a.run.app"})),
    ):
        p = tmp_path / fname
        p.write_text(raw_cred, encoding="utf-8")
        assert preset.load_saved_credentials(p, expected_gateway="http://127.0.0.1:8000") is None

    with pytest.raises(preset.QuickPresetError):
        preset.install_vscode_quick_preset(
            "http://127.0.0.1:8000",
            workspace_dir=partial_gi_ws,
            explicit_api_key="invalid-key",
            env=CLEAN_ENV,
            home=tmp_path / "h_badkey",
        )

    # 409 handle collision retry + register/me/models failure branches
    reg_attempts = 0

    def _retry_409_http(method: str, url: str, **_kw: Any) -> tuple[int, dict[str, Any]]:
        nonlocal reg_attempts
        if method == "POST" and url.endswith("/v1/hivemind/register"):
            reg_attempts += 1
            if reg_attempts == 1:
                return 409, {"detail": "handle_taken"}
            return 200, {"account_id": "acct_409_ok", "api_key": ALPHA_KEY, "role": "prosumer"}
        if url.endswith("/v1/hivemind/me"):
            return 200, {"account_id": "acct_409_ok", "handle": "noel-dev-retried", "role": "prosumer"}
        return 200, {"object": "list", "data": [{"id": "hivemind/coder"}]}

    ws_409 = tmp_path / "ws_409"
    ws_409.mkdir()
    rep_409 = preset.install_vscode_quick_preset(
        "http://127.0.0.1:8000",
        workspace_dir=ws_409,
        credentials_path=tmp_path / "h409" / ".hivemind" / "credentials.json",
        continue_config_path=tmp_path / "h409" / ".continue" / "hivemind_provider.yaml",
        http_fn=_retry_409_http,
        env=CLEAN_ENV,
        home=tmp_path / "h409",
    )
    assert rep_409["ready"] is True and reg_attempts == 2

    for stage_fail_http in (
        lambda m, u, **_k: (500, {}) if u.endswith("/register") else (200, {}),
        lambda m, u, **_k: (200, {"api_key": ALPHA_KEY}) if u.endswith("/register") else (401, {}),
        lambda m, u, **_k: (
            (200, {"api_key": ALPHA_KEY})
            if u.endswith("/register")
            else ((200, {"role": "prosumer"}) if u.endswith("/me") else (503, {}))
        ),
    ):
        rep_f = preset.install_vscode_quick_preset(
            "http://127.0.0.1:8000",
            workspace_dir=ws_409,
            credentials_path=tmp_path / "h_fail" / ".hivemind" / "credentials.json",
            continue_config_path=tmp_path / "h_fail" / ".continue" / "hivemind_provider.yaml",
            http_fn=stage_fail_http,
            env=CLEAN_ENV,
            home=tmp_path / "h_fail",
            lang="fr",
        )
        assert rep_f["exit_code"] == 2 and rep_f["verdict"] == "auth_or_gateway_failed"

    # Parity 503 on core/fleet + invalid 200 contracts
    parity_broken_200 = preset.verify_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        http_fn=lambda _m, u, **_k: (
            (503, {})
            if any(u.endswith(s) for s in ("/health", "/v1/models", "/fleet"))
            else (200, {"html": "bad"} if any(u.endswith(s) for s in ("/noel", "/quick", "/raid-noel")) else {"bad": 1})
        ),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert parity_broken_200["parity_ok"] is False and parity_broken_200["stale_revision_detected"] is True

    # GPU parse_error preflight + _optional_finite_float + CES timeout/400 + malformed uninstall files
    assert preset._optional_finite_float("n/a") is None
    assert preset._optional_finite_float("not-a-float") is None
    assert preset.default_gpu_probe(runner=lambda _c, _t: (0, "RTX, -5, 0, 0, n/a, bad\n")).parse_error is True
    pf_perr = preset.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda **_kw: {"present": True, "parse_error": True},
        yield_drill_fn=lambda: 30.0,
        process_probe_fn=lambda: ["", "steam_app_730"],
    )
    assert pf_perr["preflight_ok"] is False and pf_perr["mode"] == "gpu_telemetry_parse_error"

    assert preset._platform_tag("FreeBSD_14") == "freebsd_14"
    assert preset._platform_tag("!!!") == "unknown"
    with pytest.raises(preset.QuickPresetError):
        preset.submit_preset_ces_feedback(gateway="http://127.0.0.1:8000", zero_game_lag="yes")  # type: ignore[arg-type]
    ces_to, _ = preset.submit_preset_ces_feedback(
        gateway="http://127.0.0.1:8000",
        http_fn=lambda _m, _u, **_k: (0, {}),
        lang="fr",
    )
    assert ces_to["ces_recorded"] is False and ces_to["status"] == "network_timeout"
    ces_400, _ = preset.submit_preset_ces_feedback(
        gateway="http://127.0.0.1:8000",
        http_fn=lambda _m, _u, **_k: (400, {}),
        lang="en",
    )
    assert ces_400["ces_recorded"] is False and ces_400["status"] == "http_400"

    mal_ws = tmp_path / "malformed_ws"
    (mal_ws / ".vscode").mkdir(parents=True)
    (mal_ws / ".vscode" / "settings.json").write_text("{corrupt", encoding="utf-8")
    (mal_ws / ".vscode" / "tasks.json").write_text("{corrupt", encoding="utf-8")
    assert preset.uninstall_vscode_quick_preset(workspace_dir=mal_ws, env=CLEAN_ENV, home=tmp_path)["exit_code"] == 0
