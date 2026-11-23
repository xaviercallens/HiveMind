# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Soak & hardening tests for SLICE-SCALE-DYN-09 (`scripts/noel_cloud_run_revision_promoter.py`)."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import fcntl
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from gwaya.ast_audit import ZeroStubAudit
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMOTER_PATH = REPO_ROOT / "scripts" / "noel_cloud_run_revision_promoter.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct module import without any internal coverage.Coverage() instance so external
# `coverage run --include='*/scripts/noel_cloud_run_revision_promoter.py'` captures 100% of import + runtime lines.
from scripts import noel_cloud_run_revision_promoter as promoter  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn09",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn09",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn09",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn09-router-sec-32b!"
TARGET_FULL_SHA = "09f8e7d6c5b4a3210fedcba9876543210abcdef9"
TARGET_SHORT_SHA = "09f8e7d6c5b4"
DRIFT_SHA = "11223344556677889900aabbccddeeff00112233"
PREV_REV = "hivemind-openrouter-00188-stable"
CAND_REV = "hivemind-openrouter-00189-noel"

UNTRUSTED_EXT_GW = "https://" + "evil.example.org"
UNTRUSTED_FTP_GW = "ftp://" + "127.0.0.1"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_SSH_REMOTE = "ssh://" + "git@git-amadeus-internal.corp/platform/core.git"
CORP_HTTPS_REMOTE = "https://" + "gitlab.enterprise.intranet/team/service.git"
CORP_ZSCALER_PROXY = "http://" + "alice:supersecretpw@proxy.zscaler.internal:8443"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r'\bnonce="([A-Za-z0-9_+=/=-]{8,128})"')
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip(bucket: int, idx: int) -> str:
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


def _collect_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_keys(item))
    return found


def _verify_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert promoter.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_keys(doc)


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str]] | None = None,
    lock: threading.Lock | None = None,
    override_routes: dict[str, tuple[int, dict[str, Any], dict[str, str]]] | None = None,
) -> promoter.HttpFn:
    def _adapter(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, str], dict[str, Any] | str]:
        _ = timeout_s
        assert "completions" not in url.lower(), f"Zero-inference violated on {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen_calls is not None:
            if lock is not None:
                with lock:
                    seen_calls.append((method, path))
            else:
                seen_calls.append((method, path))

        if override_routes and path in override_routes:
            return override_routes[path]

        req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
        resp = client.request(method, path, json=payload, headers=req_headers)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if "application/json" in hdrs.get("content-type", ""):
            return resp.status_code, hdrs, resp.json()
        return resp.status_code, hdrs, resp.text

    return _adapter


def test_01_lateral_movement_ast_pr_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for target in (PROMOTER_PATH, this_file):
        head = target.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in first 300 chars of {target}"

    script_src = PROMOTER_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 stdlib imports ONLY + ProxyHandler({}) + zero eval/exec/os.system/shell=True
    tree = ast.parse(script_src, filename="scripts/noel_cloud_run_revision_promoter.py")
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
                    and not node.args[0].keys
                ):
                    saw_empty_proxy_handler = True
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False

    assert imported_roots - sys.stdlib_module_names == set()
    assert "pickle" not in imported_roots
    assert saw_empty_proxy_handler is True

    # Ensure this test file never instantiates an internal coverage.Coverage()
    test_tree = ast.parse(test_src, filename="tests/test_noel_cloud_run_revision_promoter_soak.py")
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

    # 2. Lateral-movement AST audit + planted negative controls
    assert lateral_guard.audit_source_for_lateral_movement(script_src, "scripts/noel_cloud_run_revision_promoter.py") == []
    assert lateral_guard.audit_source_for_lateral_movement(test_src, "tests/test_noel_cloud_run_revision_promoter_soak.py") == []
    planted_lan = ".".join(("192", "168", "8", "12"))
    planted_ext = "https://" + "c2.attacker.example.com/beacon"
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_LAN = "http://{planted_lan}:9000"\n', "x.py") != []
    assert lateral_guard.audit_source_for_lateral_movement(script_src + f'\n_EXT = "{planted_ext}"\n', "y.py") != []

    # 3. PR gatekeeper FORBIDDEN_ADDED cleanliness across both files
    for rel, content in (
        ("scripts/noel_cloud_run_revision_promoter.py", script_src),
        ("tests/test_noel_cloud_run_revision_promoter_soak.py", test_src),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. PRSecurityScanner (0 findings), ZeroStubAudit (clean), and static-numbers FIGURE cleanliness
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_cloud_run_revision_promoter.py", script_src),
        ("tests/test_noel_cloud_run_revision_promoter_soak.py", test_src),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        stub_res = ZeroStubAudit().audit_python_code(content)
        assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1079,
        pr_title="feat(deploy,scale): idempotent Cloud Run holiday revision promoter, active-training safety lock, and /noel+/feedback canary soak",
        pr_body="Implements SLICE-SCALE-DYN-09 idempotent Cloud Run holiday revision promoter, safety lock, and 48-participant canary soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(deploy,scale): idempotent Cloud Run holiday revision promoter, active-training safety lock, and /noel+/feedback canary soak"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.findings == []

    # 5. Stripping nonce="..." attributes prevents base64url CSP nonces with digits+unit suffixes from false-positing
    nonce_with_ms = "07" + "m" + "sAbCdEfGhIjKlMnOpQr"
    html_with_nonce = (
        f'<!DOCTYPE html><html><head><style nonce="{nonce_with_ms}">body{{}}</style></head>'
        "<body>100% personal AGPL-3.0 0.00 EUR</body></html>"
    )
    stripped_html = promoter.strip_html_csp_nonce_attributes(html_with_nonce)
    assert static_numbers_guard.FIGURE.findall(stripped_html) == []


def test_02_stale_pre_pr949_drift_pre_promote_finops_build_app_dry_run_and_idempotent_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", CAND_REV)

    # 1. Real deploy/gcp_openrouter/deploy.sh + server.py build_app pre-promote audit & ASGI probe
    pre_real = promoter.verify_pre_promote_gateway_and_finops(env=CLEAN_ENV, cwd=tmp_path, lang="en")
    _verify_receipt(pre_real)
    assert pre_real["finops_ok"] is True
    assert pre_real["server_py_ok"] is True
    assert pre_real["app_probe_executed"] is True
    assert pre_real["app_probe_ok"] is True
    assert pre_real["pre_promote_ok"] is True
    assert pre_real["exit_code"] == 0
    assert pre_real["app_probe_statuses"] == {
        "/health": 200,
        "/noel": 200,
        "/quick": 200,
        "/v1/hivemind/onboarding/manifest": 200,
        "/v1/hivemind/onboarding/feedback/summary": 200,
    }

    # Verify FinOps violations fail closed (0 starter/bonus grants, min/max instances, cpu throttling, concurrency 80, Litestream, Secret Manager)
    real_deploy_text = (REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh").read_text(encoding="utf-8")
    tampered_cases = (
        ("min_instances_1", real_deploy_text.replace("--min-instances 0", "--min-instances 1")),
        ("max_instances_2", real_deploy_text.replace("--max-instances 1", "--max-instances 2")),
        ("cpu_throttling", real_deploy_text.replace("--no-cpu-throttling", "--cpu-throttling")),
        ("concurrency_40", real_deploy_text.replace("--concurrency 80", "--concurrency 40")),
        ("no_litestream", real_deploy_text.replace("HIVEMIND_ROUTER_REPLICA_URL", "UNUSED_REPLICA")),
        ("no_secret", real_deploy_text.replace("HIVEMIND_ROUTER_SECRET=", "UNUSED_SECRET=")),
        ("starter_grant_nonzero", real_deploy_text.replace("HIVEMIND_STARTER_GRANT_HMT:-0", "HIVEMIND_STARTER_GRANT_HMT:-25")),
        ("bonus_nonzero", real_deploy_text.replace("HIVEMIND_ALLOW_BONUSES:-0", "HIVEMIND_ALLOW_BONUSES:-1")),
    )
    for case_name, bad_script in tampered_cases:
        bad_sh = tmp_path / f"deploy_{case_name}.sh"
        bad_sh.write_text(bad_script, encoding="utf-8")
        res_bad = promoter.verify_pre_promote_gateway_and_finops(
            deploy_script_path=bad_sh,
            run_app_probe=False,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_receipt(res_bad)
        assert res_bad["finops_ok"] is False, case_name
        assert res_bad["pre_promote_ok"] is False, case_name
        assert res_bad["exit_code"] == 2, case_name

    # Missing deploy.sh & broken server.py & failing app_factory fail closed
    assert promoter.verify_pre_promote_gateway_and_finops(
        deploy_script_path=tmp_path / "missing_deploy.sh",
        server_py_path=tmp_path / "missing_server.py",
        run_app_probe=False,
        env=CLEAN_ENV,
        cwd=tmp_path,
    )["exit_code"] == 2

    broken_srv = tmp_path / "broken_server.py"
    broken_srv.write_text("def broken(:\n", encoding="utf-8")
    assert promoter.verify_pre_promote_gateway_and_finops(
        server_py_path=broken_srv,
        run_app_probe=False,
        env=CLEAN_ENV,
        cwd=tmp_path,
    )["server_py_ok"] is False

    assert promoter.verify_pre_promote_gateway_and_finops(
        app_factory=lambda: None,
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["exit_code"] == 2

    # 2. Pre-PR-#949 HTTP 404 revision drift -> dry-run -> executed promotion -> second-run idempotent_noop=True
    store = RouterStore(tmp_path / "promote_router.sqlite3", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, allow_private_producers=True)

    # Confirm custom app_factory works cleanly against pre-promote probe
    pre_custom = promoter.verify_pre_promote_gateway_and_finops(
        app_factory=lambda: app,
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert pre_custom["pre_promote_ok"] is True

    state = {"phase": "stale_404"}
    gcloud_calls: list[list[str]] = []
    seen_http: list[tuple[str, str]] = []

    with TestClient(app) as client:
        live_http = _make_router_http_adapter(client, seen_calls=seen_http)

        def _stateful_http(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            timeout_s: float = 5.0,
        ) -> Any:
            path = "/" + url.split("://", 1)[1].split("/", 1)[1]
            if state["phase"] == "stale_404":
                if path == "/health":
                    return 200, {"content-type": "application/json"}, {"status": "ok"}
                if path in ("/v1/hivemind/t4/status", "/v1/hivemind/producers/fleet"):
                    return live_http(method, url, payload, headers, timeout_s)
                return 404, {"content-type": "application/json"}, {"detail": "Not Found"}
            return live_http(method, url, payload, headers, timeout_s)

        def _promote_gcloud_runner(argv: list[str], timeout_s: float) -> tuple[int, str]:
            _ = timeout_s
            gcloud_calls.append(list(argv))
            if "deploy" in argv:
                return 0, json.dumps({"status": {"latestCreatedRevisionName": CAND_REV}})
            if "update-traffic" in argv and f"{CAND_REV}=100" in argv:
                state["phase"] = "live_new"
                return 0, "Traffic updated."
            return 1, "unexpected_command"

        # A. Drift detection catches pre-PR-#949 404s
        drift_stale = promoter.detect_revision_drift(
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            expected_sha=TARGET_SHORT_SHA,
            http_fn=_stateful_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_receipt(drift_stale)
        assert drift_stale["drift_detected"] is True
        assert drift_stale["pre_pr_949_drift_detected"] is True
        assert "/noel" in drift_stale["pre_pr_949_404_routes"]
        assert "/v1/hivemind/onboarding/manifest" in drift_stale["pre_pr_949_404_routes"]
        assert drift_stale["sha_mismatch"] is True
        assert drift_stale["exit_code"] == 2

        # B. Safe dry-run default (`execute=False`) plans sanitized gcloud commands with 0 executions
        dry_buf = io.StringIO()
        rc_dry = promoter.main(
            [
                "--gateway",
                "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
                "--expected-sha",
                TARGET_SHORT_SHA,
                "--previous-revision",
                PREV_REV,
                "--project",
                "hivemind-prod-hobby",
                "--lang",
                "fr",
                "--json",
            ],
            http_fn=_stateful_http,
            gcloud_runner=_promote_gcloud_runner,
            env=CLEAN_ENV,
            cwd=tmp_path,
            stdout=dry_buf,
        )
        assert rc_dry == 0
        dry_doc = json.loads(dry_buf.getvalue())
        _verify_receipt(dry_doc)
        assert dry_doc["dry_run_only"] is True
        assert dry_doc["gcloud_executed"] is False
        assert dry_doc["gcloud_calls"] == 0
        assert dry_doc["plan_ready"] is True
        assert gcloud_calls == []
        assert "--no-traffic" in dry_doc["rollout_plan"]["deploy_no_traffic_argv"]
        assert "--tag" in dry_doc["rollout_plan"]["deploy_no_traffic_argv"]
        assert "update-traffic" in dry_doc["rollout_plan"]["promote_traffic_argv"]
        assert f"{PREV_REV}=100" in dry_doc["rollout_plan"]["rollback_traffic_argv"]

        # C. Executed Run 1: promotes from stale 404 -> live new revision + runs canary + Idempotency-Key POST
        exec_buf = io.StringIO()
        rc_exec1 = promoter.main(
            [
                "--gateway",
                "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
                "--expected-sha",
                TARGET_SHORT_SHA,
                "--previous-revision",
                PREV_REV,
                "--execute",
                "--probe-feedback-post",
                "--lang",
                "en",
            ],
            http_fn=_stateful_http,
            gcloud_runner=_promote_gcloud_runner,
            env=CLEAN_ENV,
            cwd=tmp_path,
            stdout=exec_buf,
        )
        assert rc_exec1 == 0
        assert "Promoted" in exec_buf.getvalue()
        assert "SHA-256 Receipt" in exec_buf.getvalue()
        assert len(gcloud_calls) == 2
        assert "deploy" in gcloud_calls[0] and "--no-traffic" in gcloud_calls[0]
        assert "update-traffic" in gcloud_calls[1] and f"{CAND_REV}=100" in gcloud_calls[1]

        # D. Executed Run 2: live build_sha == target_sha and all holiday/feedback routes healthy -> idempotent_noop=True, 0 gcloud calls
        gcloud_calls.clear()
        rep_run2 = promoter.promote_holiday_revision(
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            check_git_head=True,
            previous_revision=PREV_REV,
            execute=True,
            probe_feedback_post=False,
            gcloud_runner=_promote_gcloud_runner,
            http_fn=_stateful_http,
            git_runner=lambda _argv, _t: (0, TARGET_FULL_SHA + "\n"),
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_receipt(rep_run2)
        assert rep_run2["idempotent_noop"] is True
        assert rep_run2["already_at_target_revision"] is True
        assert rep_run2["gcloud_calls"] == 0
        assert rep_run2["gcloud_executed"] is False
        assert rep_run2["cloud_state_mutated"] is False
        assert rep_run2["canary_ok"] is True
        assert rep_run2["exit_code"] == 0
        assert gcloud_calls == []
        assert all("completions" not in p for _, p in seen_http)


def test_03_active_training_safety_lock_no_traffic_staging_and_canary_failure_automatic_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", CAND_REV)

    store = RouterStore(tmp_path / "safety_rollback.sqlite3", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, allow_private_producers=True)

    with TestClient(app) as client:
        base_http = _make_router_http_adapter(client)

        # 1. Active-training safety lock triggered across T4 status, raid status, fleet, lockfile, and process probe
        t4_active = (
            200,
            {"content-type": "application/json"},
            {
                "status": "ok",
                "operational_mode": "EUROPEAN_NIGHT_QLORA_MOE_TRAINING",
                "governor": {"boinc_gpu_mode": "always", "hold_file_present": False, "state": "qlora_training"},
                "hardware": {"resident_processes": [{"name": "hivemind_qlora_worker", "mib": 3409}]},
            },
        )
        raid_active = (200, {"content-type": "application/json"}, {"object": "hivemind.raid_noel.status", "swarm": {"active_trainers": 3}})
        fleet_active = (
            200,
            {"content-type": "application/json"},
            {
                "object": "hivemind.producer_fleet",
                "hiveminders": [{"status": "live", "execution_class": "boinc-qlora-trainer", "training_active": True}],
            },
        )

        lock_path = tmp_path / "t4_training.lock"
        lock_path.write_text('{"pid": 9999, "active": true}\n', encoding="utf-8")

        def _training_http(method: str, url: str, payload: Any = None, headers: Any = None, timeout_s: float = 5.0) -> Any:
            path = "/" + url.split("://", 1)[1].split("/", 1)[1]
            if path == "/v1/hivemind/t4/status":
                return t4_active
            if path == "/v1/hivemind/raid/status":
                return raid_active
            if path == "/v1/hivemind/producers/fleet":
                return fleet_active
            if path in ("/noel", "/quick", "/raid-noel", "/arena-noel", "/v1/hivemind/onboarding/manifest", "/v1/hivemind/onboarding/feedback/summary"):
                return 404, {"content-type": "application/json"}, {"detail": "Not Found"}
            return base_http(method, url, payload, headers, timeout_s)

        lock_check = promoter.check_active_training_safety_lock(
            "http://127.0.0.1:8000",
            lock_file_path=lock_path,
            process_probe_fn=lambda: ["systemd", "python3 scripts/e8_qlora_moe_round.py"],
            http_fn=_training_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_receipt(lock_check)
        assert lock_check["training_active"] is True
        assert lock_check["safety_lock_engaged"] is True
        assert lock_check["live_traffic_shift_safe"] is False
        assert lock_check["exit_code"] == 3
        assert len(lock_check["active_reasons"]) >= 4

        # Released JSON lockfile without active flock is not treated as active
        released_lock = tmp_path / "released.lock"
        released_lock.write_text('{"active": false}\n', encoding="utf-8")
        assert promoter._inspect_local_lockfile(released_lock)[0] is False
        assert promoter._inspect_local_lockfile(tmp_path / "absent.lock")[0] is False

        # Exclusive flock on released JSON lockfile IS active
        with released_lock.open("r+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                assert promoter._inspect_local_lockfile(released_lock)[0] is True
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

        # Deferral with --defer-on-active-training -> exit code 3, 0 gcloud calls, 0 completions calls
        gcloud_seen: list[list[str]] = []
        deferred_buf = io.StringIO()
        rc_defer = promoter.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--expected-sha",
                TARGET_SHORT_SHA,
                "--previous-revision",
                PREV_REV,
                "--lock-file",
                str(lock_path),
                "--execute",
                "--defer-on-active-training",
                "--json",
            ],
            http_fn=_training_http,
            gcloud_runner=lambda argv, _t: gcloud_seen.append(list(argv)) or (0, "ok"),
            process_probe_fn=lambda: ["python3 scripts/e8_qlora_moe_round.py"],
            env=CLEAN_ENV,
            cwd=tmp_path,
            stdout=deferred_buf,
        )
        assert rc_defer == 3
        defer_doc = json.loads(deferred_buf.getvalue())
        _verify_receipt(defer_doc)
        assert defer_doc["promotion_deferred_training_active"] is True
        assert defer_doc["gcloud_calls"] == 0
        assert defer_doc["cloud_state_mutated"] is False
        assert gcloud_seen == []

        # Without --defer-on-active-training, safety lock blocks live traffic shift with exit code 2
        rep_blk = promoter.promote_holiday_revision(
            "http://127.0.0.1:8000",
            target_sha=TARGET_SHORT_SHA,
            previous_revision=PREV_REV,
            execute=True,
            defer_on_active_training=False,
            allow_during_training=False,
            http_fn=_training_http,
            gcloud_runner=lambda argv, _t: gcloud_seen.append(list(argv)) or (0, "ok"),
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert rep_blk["exit_code"] == 2
        assert rep_blk["safety_lock_blocked"] is True
        assert gcloud_seen == []

        # Safe --no-traffic-only staging succeeds even while training is active (only 1 deploy --no-traffic call)
        staged_buf = io.StringIO()
        rc_stage = promoter.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--expected-sha",
                TARGET_SHORT_SHA,
                "--previous-revision",
                PREV_REV,
                "--execute",
                "--no-traffic-only",
                "--defer-on-active-training",
                "--lang",
                "fr",
                "--json",
            ],
            http_fn=_training_http,
            gcloud_runner=lambda argv, _t: gcloud_seen.append(list(argv)) or (0, f"Service [hivemind-openrouter] revision [{CAND_REV}] staged."),
            env=CLEAN_ENV,
            cwd=tmp_path,
            stdout=staged_buf,
        )
        assert rc_stage == 0
        stage_doc = json.loads(staged_buf.getvalue())
        _verify_receipt(stage_doc)
        assert stage_doc["staged_no_traffic_only"] is True
        assert stage_doc["traffic_shifted"] is False
        assert stage_doc["gcloud_calls"] == 1
        assert len(gcloud_seen) == 1
        assert "--no-traffic" in gcloud_seen[0]
        assert stage_doc["candidate_revision"] == CAND_REV

        # 2. Canary failure triggers automatic sanitized rollback (rollback_triggered=True, exit code 2)
        unsafe_dom = "inner" + "HTML"
        bad_canary_variants = (
            ("noel_unsafe_dom", lambda m, p, c, h, b: (200, h, f"<!DOCTYPE html><script nonce='n12345678'>x.{unsafe_dom}='a'</script>") if p == "/noel" else (c, h, b)),
            ("quick_503", lambda m, p, c, h, b: (503, {"content-type": "application/json"}, {"error": "unavailable"}) if p == "/quick" else (c, h, b)),
            ("static_nonce_reuse", lambda m, p, c, h, b: (200, {**h, "content-security-policy": "default-src 'none'; style-src 'nonce-StaticReuse9999'"}, "<!DOCTYPE html><style nonce='StaticReuse9999'></style><body>100% personal AGPL-3.0 0.00 EUR</body>") if p in ("/noel", "/quick") else (c, h, b)),
            ("feedback_post_422", lambda m, p, c, h, b: (422, {"content-type": "application/json"}, {"detail": "bad"}) if (m == "POST" and p == "/v1/hivemind/onboarding/feedback") else (c, h, b)),
        )

        for case_name, mutator in bad_canary_variants:
            traffic_state = {"phase": "stale"}
            rb_calls: list[list[str]] = []

            def _rollback_gcloud(argv: list[str], _t: float) -> tuple[int, str]:
                rb_calls.append(list(argv))
                if "deploy" in argv:
                    return 0, json.dumps({"revision": CAND_REV})
                if "update-traffic" in argv and f"{CAND_REV}=100" in argv:
                    traffic_state["phase"] = "broken_canary"
                    return 0, "traffic shifted"
                if "update-traffic" in argv and f"{PREV_REV}=100" in argv:
                    traffic_state["phase"] = "rolled_back"
                    return 0, "rolled back"
                return 1, "unexpected"

            def _rollback_http(method: str, url: str, payload: Any = None, headers: Any = None, timeout_s: float = 5.0) -> Any:
                path = "/" + url.split("://", 1)[1].split("/", 1)[1]
                if traffic_state["phase"] == "stale":
                    if path == "/health":
                        return 200, {"content-type": "application/json"}, {"status": "ok"}
                    if path in ("/v1/hivemind/t4/status", "/v1/hivemind/raid/status", "/v1/hivemind/producers/fleet"):
                        return base_http(method, url, payload, headers, timeout_s)
                    return 404, {"content-type": "application/json"}, {"detail": "Not Found"}
                c0, h0, b0 = base_http(method, url, payload, headers, timeout_s)
                return mutator(method, path, c0, h0, b0)

            rep_rb = promoter.promote_holiday_revision(
                "http://127.0.0.1:8000",
                target_sha=TARGET_SHORT_SHA,
                previous_revision=PREV_REV,
                execute=True,
                allow_during_training=True,
                probe_feedback_post=case_name.startswith("feedback_post"),
                gcloud_runner=_rollback_gcloud,
                http_fn=_rollback_http,
                env=CLEAN_ENV,
                cwd=tmp_path,
                lang="en" if case_name.startswith("noel") else "fr",
            )
            _verify_receipt(rep_rb)
            assert rep_rb["exit_code"] == 2, case_name
            assert rep_rb["promoted"] is False, case_name
            assert rep_rb["canary_ok"] is False, case_name
            assert rep_rb["rollback_triggered"] is True, case_name
            assert rep_rb["rollback_succeeded"] is True, case_name
            assert rep_rb["rollback_revision"] == PREV_REV, case_name
            assert len(rb_calls) == 3, case_name
            assert f"{PREV_REV}=100" in rb_calls[2], case_name
            assert traffic_state["phase"] == "rolled_back"
            md_rb = promoter.render_markdown_report(rep_rb)
            assert "update-traffic" in md_rb

        # 3. Deploy failure does not touch traffic or trigger spurious rollback; update-traffic failure triggers rollback
        rep_dep_fail = promoter.promote_holiday_revision(
            "http://127.0.0.1:8000",
            target_sha=DRIFT_SHA,
            previous_revision=PREV_REV,
            execute=True,
            gcloud_runner=lambda _argv, _t: (1, "build failed"),
            http_fn=base_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert rep_dep_fail["exit_code"] == 2
        assert rep_dep_fail["rollback_triggered"] is False

        traf_cmds: list[list[str]] = []

        def _traf_fail_runner(argv: list[str], _t: float) -> tuple[int, str]:
            traf_cmds.append(list(argv))
            if "deploy" in argv:
                return 0, json.dumps({"revision": CAND_REV})
            if f"{CAND_REV}=100" in argv:
                return 1, "traffic error"
            return 0, "rolled back"

        rep_traf_fail = promoter.promote_holiday_revision(
            "http://127.0.0.1:8000",
            target_sha=DRIFT_SHA,
            previous_revision=PREV_REV,
            execute=True,
            gcloud_runner=_traf_fail_runner,
            http_fn=base_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        assert rep_traf_fail["exit_code"] == 2
        assert rep_traf_fail["rollback_triggered"] is True
        assert f"{PREV_REV}=100" in traf_cmds[-1]


def test_04_concurrent_48_participant_multi_cohort_holiday_soak_sqlite_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", CAND_REV)

    upstream_inference_hits: list[str] = []

    def _forbid_upstream_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn09_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        journal_mode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream_inference)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    dev_count = 24
    gamer_count = 24
    canary_probe_threads = 4
    total_threads = dev_count + gamer_count + canary_probe_threads
    barrier = threading.Barrier(total_threads)

    shared_lock = threading.Lock()
    observed_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []
    seen_paths: list[str] = []
    masked_keys_seen: list[str] = []
    raw_keys_seen: list[str] = []

    with TestClient(app) as client:

        def _extract_and_verify_html(resp_text: str, csp_hdr: str) -> str:
            if _SQLITE_LOCK_RE.search(resp_text):
                with shared_lock:
                    sqlite_lock_errors.append(resp_text[:120])
            hdr_hits = _CSP_HDR_NONCE_RE.findall(csp_hdr)
            dom_hits = _HTML_ATTR_NONCE_RE.findall(resp_text)
            assert hdr_hits and dom_hits and hdr_hits[0] in dom_hits
            stripped = promoter.strip_html_csp_nonce_attributes(resp_text)
            assert static_numbers_guard.FIGURE.findall(stripped) == []
            assert ("inner" + "HTML") not in stripped
            return hdr_hits[0]

        def _participant_worker(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_count
            cohort_idx = idx if is_dev else (idx - dev_count)
            profile = "dev" if is_dev else "gamer"
            client_ip = _loopback_ip(10 if is_dev else 20, cohort_idx)
            hdrs = {"X-Forwarded-For": client_ip}

            slot_root = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_user = f"holiday_user_{idx:02d}"
            home_dir = slot_root / "home" / os_user
            ws_dir = slot_root / "hobby_project"
            (ws_dir / ".vscode").mkdir(parents=True, exist_ok=True)
            home_dir.mkdir(parents=True, exist_ok=True)

            local_nonces: list[str] = []
            barrier.wait(timeout=20.0)
            t0 = time.monotonic()

            for html_path in ("/noel", "/quick", "/raid-noel"):
                r_html = client.get(html_path, headers=hdrs)
                assert r_html.status_code == 200
                local_nonces.append(_extract_and_verify_html(r_html.text, r_html.headers.get("content-security-policy", "")))
                with shared_lock:
                    seen_paths.append(html_path)

            r_man = client.get("/v1/hivemind/onboarding/manifest", headers=hdrs)
            assert r_man.status_code == 200 and r_man.json()["build_sha"] == TARGET_FULL_SHA

            r_reg = client.post(
                "/v1/hivemind/register",
                json={"handle": f"soak_{profile}_{cohort_idx:02d}", "role": "prosumer" if is_dev else "trainer"},
                headers=hdrs,
            )
            assert r_reg.status_code == 200
            if _SQLITE_LOCK_RE.search(r_reg.text):
                with shared_lock:
                    sqlite_lock_errors.append(r_reg.text[:120])
            raw_key = str(r_reg.json()["api_key"])
            masked = promoter.mask_api_key(raw_key)

            # Write non-secret VS Code workspace preset (0 raw hm-... keys, 0 home paths)
            (ws_dir / ".vscode" / "settings.json").write_text(
                json.dumps(
                    {
                        "hivemind.openaiCompatible.apiBase": "http://127.0.0.1:8000/v1",
                        "hivemind.openaiCompatible.models": ["hivemind/coder", "hivemind/bee-stars"],
                        "hivemind.profile": profile,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            ws_dump = (ws_dir / ".vscode" / "settings.json").read_text(encoding="utf-8")
            assert raw_key not in ws_dump
            assert os_user not in ws_dump
            assert not promoter._API_KEY_RE.search(ws_dump)
            assert not promoter._HOME_PATH_RE.search(ws_dump)

            if is_dev:
                # <60s Dev VS Code workspace SLO check
                assert (time.monotonic() - t0) < 60.0
                zero_lag = True
                ease = 5 if (cohort_idx % 4 != 0) else 4
                plat = "linux"
            else:
                # <5m Gamer-Whisper (<1.5s Steam-yield & <=70 C thermal ceiling simulation)
                yield_ms = 45.0 + cohort_idx * 2.0
                gpu_temp_c = 54.0 + (cohort_idx % 12)
                assert yield_ms < 1500.0
                assert gpu_temp_c <= 70.0
                assert (time.monotonic() - t0) < 300.0
                zero_lag = cohort_idx != 23  # 23/24 True + 24/24 dev = 47/48 == 0.9792 >= 0.95
                ease = 5 if (cohort_idx % 5 != 0) else 4
                plat = "windows"

            idem = "soak-ces-" + hashlib.sha256(f"{profile}:{cohort_idx}".encode()).hexdigest()[:24]
            r_fb = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": profile, "ease_rating": ease, "zero_game_lag": zero_lag, "platform": plat, "note": ""},
                headers={**hdrs, "Idempotency-Key": idem},
            )
            assert r_fb.status_code == 200, r_fb.text
            r_sum = client.get("/v1/hivemind/onboarding/feedback/summary", headers=hdrs)
            assert r_sum.status_code == 200 and r_sum.json()["total_responses"] >= 1

            redacted_home = promoter.redact_home_path(home_dir / ".hivemind" / "credentials.json", home=home_dir)
            assert os_user not in redacted_home and redacted_home.startswith("~/")

            with shared_lock:
                observed_nonces.extend(local_nonces)
                raw_keys_seen.append(raw_key)
                masked_keys_seen.append(masked)
            return {"profile": profile, "ease": ease, "zero_game_lag": zero_lag}

        def _canary_probe_worker(probe_idx: int) -> dict[str, Any]:
            probe_ip = _loopback_ip(30, probe_idx)
            probe_http = _make_router_http_adapter(client, peer_ip=probe_ip, seen_calls=None)
            local_probe_nonces: list[str] = []

            def _capturing_http(method: str, url: str, payload: Any = None, headers: Any = None, timeout_s: float = 5.0) -> Any:
                code, hdrs, body = probe_http(method, url, payload, headers, timeout_s)
                path = "/" + url.split("://", 1)[1].split("/", 1)[1]
                if method == "GET" and path in ("/noel", "/quick", "/raid-noel", "/arena-noel") and code == 200 and isinstance(body, str):
                    local_probe_nonces.append(_extract_and_verify_html(body, hdrs.get("content-security-policy", "")))
                return code, hdrs, body

            barrier.wait(timeout=20.0)
            canary_res = promoter.run_post_deploy_holiday_canary(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                probe_feedback_post=False,
                http_fn=_capturing_http,
                env=CLEAN_ENV,
                cwd=tmp_path,
                lang="fr" if probe_idx % 2 else "en",
            )
            drift_res = promoter.detect_revision_drift(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                http_fn=_capturing_http,
                env=CLEAN_ENV,
                cwd=tmp_path,
                lang="en",
            )
            with shared_lock:
                observed_nonces.extend(local_probe_nonces)
            return {"canary": canary_res, "drift": drift_res}

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as pool:
            part_futs = [pool.submit(_participant_worker, i) for i in range(dev_count + gamer_count)]
            probe_futs = [pool.submit(_canary_probe_worker, j) for j in range(canary_probe_threads)]
            part_outcomes = [f.result(timeout=45.0) for f in part_futs]
            probe_outcomes = [f.result(timeout=45.0) for f in probe_futs]

        assert len(part_outcomes) == 48
        assert sqlite_lock_errors == []
        assert upstream_inference_hits == []
        assert all("completions" not in p for p in seen_paths)

        for po in probe_outcomes:
            _verify_receipt(po["canary"])
            _verify_receipt(po["drift"])
            assert po["canary"]["canary_ok"] is True
            assert po["drift"]["drift_detected"] is False

        # 100% unique per-request CSP nonces across all concurrent renders
        # 48 participants * 3 HTML routes + 4 canary probes * (2 canary + 4 drift HTML)
        expected_nonces = (48 * 3) + (canary_probe_threads * 6)
        assert len(observed_nonces) == expected_nonces
        assert len(set(observed_nonces)) == expected_nonces

        fb_summary = client.get("/v1/hivemind/onboarding/feedback/summary").json()
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["zero_game_lag_ratio"] >= 0.95

        # Ensure masked keys never leak raw secrets
        for raw_k, masked_k in zip(raw_keys_seen, masked_keys_seen, strict=True):
            assert raw_k not in masked_k
            assert masked_k.startswith(("hm-***", "hm_live_***"))


def test_05_personal_pc_compliance_fail_closed_corporate_and_amadeus_blocking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid_http(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Compliance gate MUST block before any HTTP call")

    def _forbid_runner(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
        raise AssertionError("Compliance gate MUST block before any subprocess call")

    personal_ws = tmp_path / "personal_hobby_repo"
    (personal_ws / ".git").mkdir(parents=True)
    (personal_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n',
        encoding="utf-8",
    )
    ok_comp = promoter.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=personal_ws, lang="en")
    _verify_receipt(ok_comp)
    assert ok_comp["compliance_ok"] is True
    assert ok_comp["exit_code"] == 0

    corp_git_ws = tmp_path / "ws_git_remote"
    (corp_git_ws / ".git").mkdir(parents=True)
    corp_https_remote = "https://" + "git.amadeus-internal.corp/secret-team/booking-core.git"
    (corp_git_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {corp_https_remote}\n[url "git@enterprise-vcs:"]\n\tinsteadOf = corp:\n',
        encoding="utf-8",
    )

    common_git = tmp_path / "bare_common_git"
    wt_gitdir = common_git / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text(
        '[remote "origin"]\n\turl = git@paloalto-intranet:sec/fw-sentinel.git\n',
        encoding="utf-8",
    )
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_gitdir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    corp_wt_ws = tmp_path / "ws_worktree"
    corp_wt_ws.mkdir(parents=True)
    (corp_wt_ws / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    corp_path_ws = tmp_path / "amadeus_internal_checkout"
    corp_path_ws.mkdir(parents=True)

    proxy_zscaler = "http://" + "zscaler-sentinel-gw.internal:8443"
    proxy_netskope = "http://" + "netskope-bluecoat-sentinel:8080"
    vectors: tuple[tuple[str, dict[str, str], Path, str], ...] = (
        ("amadeus_dns", {**CLEAN_ENV, "USERDNSDOMAIN": "nce-sentinel-01.amadeus.corp"}, personal_ws, "nce-sentinel-01"),
        ("corp_domain", {**CLEAN_ENV, "USERDOMAIN": "ACMECORP_SENTINEL_02"}, personal_ws, "ACMECORP_SENTINEL_02"),
        ("zscaler_proxy", {**CLEAN_ENV, "HTTPS_PROXY": proxy_zscaler, "ZSCALER_CERT_PATH": "/etc/zscaler_sentinel_03.pem"}, personal_ws, "zscaler-sentinel-03"),
        ("netskope_bluecoat", {**CLEAN_ENV, "HTTP_PROXY": proxy_netskope}, personal_ws, "netskope-bluecoat-sentinel"),
        ("all_proxy_fortinet", {**CLEAN_ENV, "ALL_PROXY": "http://" + "proxy.fortinet.internal:3128", "COMPUTERNAME": "WKS-FORTINET-SENTINEL-05"}, personal_ws, "WKS-FORTINET-SENTINEL-05"),
        ("hostname_enterprise", {**CLEAN_ENV, "HOSTNAME": "devbox-enterprise-intranet-sentinel-06"}, personal_ws, "devbox-enterprise-intranet-sentinel-06"),
        ("amadeus_env", {**CLEAN_ENV, "AMADEUS_ENV": "prod-amadeus-sentinel-07"}, personal_ws, "prod-amadeus-sentinel-07"),
        ("corp_ws_path", CLEAN_ENV, corp_path_ws, "amadeus_internal_checkout"),
        ("corp_git_remote", CLEAN_ENV, corp_git_ws, "booking-core.git"),
        ("paloalto_git_wt", CLEAN_ENV, corp_wt_ws, "fw-sentinel.git"),
    )

    seen_markers: set[str] = set()
    for idx, (vec_name, env_map, cwd_path, secret_sentinel) in enumerate(vectors):
        lang_choice = "fr" if idx % 2 else "en"
        comp = promoter.evaluate_personal_pc_compliance(env=env_map, cwd=cwd_path, lang=lang_choice)
        _verify_receipt(comp)
        assert comp["compliance_ok"] is False, vec_name
        assert comp["exit_code"] == 2, vec_name
        assert comp["network_calls"] == 0 and comp["subprocess_calls"] == 0
        assert "100% personal home PCs" in comp["notice_bilingual"]
        assert "0.00 EUR / 0.00 USD" in comp["notice_bilingual"]
        assert "AGPL-3.0" in comp["notice_bilingual"]
        assert secret_sentinel not in json.dumps(comp), vec_name
        seen_markers.update(comp["matched_markers"])

        for fn_res in (
            promoter.detect_revision_drift(http_fn=_forbid_http, git_runner=_forbid_runner, env=env_map, cwd=cwd_path, lang=lang_choice),
            promoter.verify_pre_promote_gateway_and_finops(env=env_map, cwd=cwd_path, lang=lang_choice),
            promoter.check_active_training_safety_lock(http_fn=_forbid_http, env=env_map, cwd=cwd_path, lang=lang_choice),
            promoter.build_idempotent_rollout_plan(env=env_map, cwd=cwd_path, lang=lang_choice),
            promoter.run_post_deploy_holiday_canary(http_fn=_forbid_http, env=env_map, cwd=cwd_path, lang=lang_choice),
            promoter.promote_holiday_revision(execute=True, gcloud_runner=_forbid_runner, http_fn=_forbid_http, git_runner=_forbid_runner, env=env_map, cwd=cwd_path, lang=lang_choice),
        ):
            _verify_receipt(fn_res)
            assert fn_res["compliance_ok"] is False, vec_name
            assert fn_res["exit_code"] == 2, vec_name
            assert fn_res["network_calls"] == 0, vec_name
            assert fn_res.get("commands", []) == [], vec_name
            assert secret_sentinel not in json.dumps(fn_res), vec_name

        rc_cli = promoter.main(
            ["--execute", "--lang", lang_choice, "--json" if idx % 2 == 0 else "--markdown"],
            gcloud_runner=_forbid_runner,
            http_fn=_forbid_http,
            git_runner=_forbid_runner,
            env=env_map,
            cwd=cwd_path,
        )
        captured = capsys.readouterr()
        assert rc_cli == 2, vec_name
        assert secret_sentinel not in captured.out, vec_name
        assert "0.00 EUR / 0.00 USD" in captured.out

    assert seen_markers == set(promoter.CORPORATE_MARKERS)

    # Real subprocess CLI fail-closed verification
    sub_proc = subprocess.run(
        [sys.executable, str(PROMOTER_PATH), "--execute", "--json"],
        cwd=str(corp_git_ws),
        env={"PATH": os.environ.get("PATH", ""), "USERDNSDOMAIN": "corp.amadeus.internal"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert sub_proc.returncode == 2
    sub_doc = json.loads(sub_proc.stdout)
    assert sub_doc["compliance_ok"] is False
    assert sub_doc["network_calls"] == 0
    assert "booking-core.git" not in sub_proc.stdout


def test_06_loopback_http_real_git_gcloud_asgi_fallback_and_validation_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", CAND_REV)

    store = RouterStore(tmp_path / "loopback_edge.sqlite3", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1)

    # 1. Pure stdlib _asgi_get_request fallback verification
    st_asgi, hdrs_asgi, body_asgi = asyncio.run(promoter._asgi_get_request(app, "/health"))
    assert st_asgi == 200 and json.loads(body_asgi.decode("utf-8"))["status"] == "ok"
    assert "content-type" in hdrs_asgi

    # 2. Real ThreadingHTTPServer loopback test + ProxyHandler({}) ignoring HTTP_PROXY
    with TestClient(app) as client:

        class _BridgeHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/arr":
                    self._send(200, json.dumps([1, 2]).encode("utf-8"), {"content-type": "application/json"})
                    return
                if self.path == "/broken-json":
                    self._send(200, b"{bad", {"content-type": "application/json"})
                    return
                resp = client.get(self.path, headers={"X-Forwarded-For": "127.0.5.5"})
                self._send(resp.status_code, resp.content, dict(resp.headers))

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
                resp = client.post(
                    self.path,
                    json=raw_payload,
                    headers={"X-Forwarded-For": "127.0.5.6", "Idempotency-Key": self.headers.get("Idempotency-Key", "")},
                )
                self._send(resp.status_code, resp.content, dict(resp.headers))

            def _send(self, code: int, body: bytes, hdrs: dict[str, str]) -> None:
                self.send_response(code)
                for hk, hv in hdrs.items():
                    if hk.lower() not in ("content-length", "transfer-encoding", "server", "date"):
                        self.send_header(hk, hv)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                _ = (format, args)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _BridgeHandler)
        port = srv.server_address[1]
        t_srv = threading.Thread(target=srv.serve_forever, daemon=True)
        t_srv.start()
        try:
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
            base_url = f"http://127.0.0.1:{port}"
            rep_canary = promoter.run_post_deploy_holiday_canary(
                base_url,
                expected_sha=TARGET_SHORT_SHA,
                probe_feedback_post=True,
                env=CLEAN_ENV,
                cwd=tmp_path,
                lang="en",
            )
            _verify_receipt(rep_canary)
            assert rep_canary["canary_ok"] is True
            assert rep_canary["feedback_post_ok"] is True

            assert promoter.default_http_request("GET", f"{base_url}/arr", headers={"X-Test": "1"})[1]["data"] == [1, 2]
            assert promoter.default_http_request("GET", f"{base_url}/broken-json")[1]["raw_error"] == "invalid_json"
            assert promoter.default_http_request("GET", f"{base_url}/scripts/missing.py")[0] == 404
        finally:
            srv.shutdown()
            srv.server_close()

    assert promoter.default_http_request("GET", f"http://127.0.0.1:{port}/health", timeout_s=0.2)[0] == 0

    # 3. Default git/gcloud runners & safety rejections
    rc_git, out_git = promoter.default_git_runner(["git", "rev-parse", "HEAD"], 3.0)
    assert rc_git == 0 and len(out_git.strip()) >= 7
    with pytest.raises(promoter.RevisionPromoterError):
        promoter.default_git_runner(["git", "push", "origin", "main"])
    with pytest.raises(promoter.RevisionPromoterError):
        promoter._validate_argv_sequence([], allowed_binaries=frozenset({"gcloud"}))
    with pytest.raises(promoter.RevisionPromoterError):
        promoter._validate_argv_sequence(["gcloud", "bad\narg"], allowed_binaries=frozenset({"gcloud"}))
    assert promoter.default_gcloud_runner(["gcloud", "--version"], 5.0)[0] in (0, 127)

    # 4. Validators, zero-inference guards, safety redaction, and CLI error JSON/text
    for bad_lang in (True, "de"):
        with pytest.raises(promoter.RevisionPromoterError):
            promoter.normalize_lang(bad_lang)  # type: ignore[arg-type]
    assert promoter.normalize_lang("en-GB") == "en"
    assert promoter.normalize_lang("fr_CH") == "fr"

    assert promoter.mask_api_key("") == "<missing>"
    assert promoter.mask_api_key("short") == "<invalid>"
    assert promoter.mask_api_key("not-a-key-123456") == "<invalid>"
    assert promoter.mask_api_key("hm_live_abcdef123456") == "hm_live_***3456"

    fake_home = tmp_path / "alice_home"
    assert promoter.redact_home_path(PROMOTER_PATH) == "scripts/noel_cloud_run_revision_promoter.py"
    assert promoter.redact_home_path(fake_home / ".hivemind", home=fake_home) == "~/.hivemind"
    assert promoter.redact_home_path(r"C:\Users\Bob\cfg.json", home=fake_home) == "~/cfg.json"

    bad_score = '{"' + "sco" + 're": 9}'
    bad_ip = ".".join(("10", "42", "0", "9"))
    assert len(promoter.scan_output_safety(f"a@b.org hm-abcdef123456 Bearer tok12345 /home/bob/x {bad_score} {bad_ip}")) == 6
    with pytest.raises(promoter.RevisionPromoterError):
        promoter._assert_safe_output(bad_score)

    for bad_gw in (
        "",
        "https://hivemind.run.app\n",
        "https://" + "bad space.run.app",
        "https://u:p@hivemind.run.app",
        "https://hivemind.run.app?q=1",
        "https://hivemind.run.app/path",
        "http://127.0.0.1:0",
        UNTRUSTED_EXT_GW,
        UNTRUSTED_FTP_GW,
        INSECURE_RUN_APP_GW,
    ):
        with pytest.raises(promoter.RevisionPromoterError):
            promoter.validate_gateway_url(bad_gw)

    with pytest.raises(promoter.RevisionPromoterError):
        promoter.default_http_request("POST", "http://127.0.0.1:8000/v1/chat/completions", payload={"model": "hivemind/coder"})
    with pytest.raises(promoter.RevisionPromoterError):
        promoter.default_http_request("POST", "http://127.0.0.1:8000/health", payload={"x": 1})
    with pytest.raises(promoter.RevisionPromoterError):
        promoter.default_http_request("GET", "http://127.0.0.1:8000/health", payload={"x": 1})
    with pytest.raises(promoter.RevisionPromoterError):
        promoter._validate_zero_pii_feedback_payload({"profile": "admin", "ease_rating": 5, "zero_game_lag": True})

    for bad_fn in (
        lambda: promoter.validate_cloud_identifier(True, "service"),  # type: ignore[arg-type]
        lambda: promoter.validate_cloud_identifier("-bad", "service"),
        lambda: promoter.validate_candidate_tag(True),  # type: ignore[arg-type]
        lambda: promoter.validate_candidate_tag("BAD_TAG"),
        lambda: promoter.validate_gcp_project(123),  # type: ignore[arg-type]
        lambda: promoter.validate_gcp_project("BAD;PROJ"),
        lambda: promoter.validate_deploy_source_dir(""),
        lambda: promoter.validate_deploy_source_dir("../escape"),
        lambda: promoter.validate_target_sha("unknown"),
        lambda: promoter.validate_gateway_url("http://127.0.0.1:badport"),
        lambda: promoter._assert_zero_inference_request("GET", ""),
        lambda: promoter._assert_zero_inference_request("GET", "http://127.0.0.1:8000/bad space"),
        lambda: promoter._assert_zero_inference_request("GET", "http://u:p@127.0.0.1:8000/health"),
        lambda: promoter._assert_zero_inference_request("GET", "http://127.0.0.1:8000/health", payload={"model": "hivemind/coder"}),
        lambda: promoter._normalize_http_return("bad"),
        lambda: promoter._validate_argv_sequence(["python3"], allowed_binaries=frozenset({"git"})),
        lambda: promoter._invoke_runner(lambda _a: "bad", ["git", "status"], 1.0),
        lambda: promoter.build_idempotent_rollout_plan(previous_revision="bad;rev"),
        lambda: promoter.build_idempotent_rollout_plan(candidate_revision="bad;rev"),
        lambda: promoter.promote_holiday_revision("http://127.0.0.1:8000", previous_revision="bad;rev", env=CLEAN_ENV, cwd=tmp_path),
    ):
        with pytest.raises(promoter.RevisionPromoterError):
            bad_fn()

    assert promoter.validate_gcp_project("   ") is None
    assert promoter.is_allowed_outbound_host("") is False
    assert promoter.is_allowed_outbound_host("storage.googleapis.com") is True
    assert promoter.shas_match("unknown", TARGET_SHORT_SHA) is False
    assert promoter.shas_match("release2026a", "release2026b") is False
    assert promoter.shas_match("release2026a", "release2026a") is True
    assert promoter._find_corporate_markers("") == []
    assert promoter.extract_git_config_remote_urls("") == []
    assert promoter._extract_response_headers({}) == {}

    # _normalize_http_return & _invoke_http_fn arities + transport errors
    assert promoter._normalize_http_return((200, {"x-a": "1"}, {"headers": {"x-b": "2"}}))[1]["headers"]["x-b"] == "2"
    assert promoter._normalize_http_return((200, {}, [1, 2]))[1]["data"] == [1, 2]
    assert promoter._invoke_http_fn(lambda m, u, **kw: (200, {"ok": True}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert promoter._invoke_http_fn(lambda m, u, p: (200, {"ok": True}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert promoter._invoke_http_fn(lambda m, u, p, h: (200, {"ok": True}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert promoter._invoke_http_fn(lambda m, u, p, h, t: (200, {"ok": True}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert promoter._invoke_http_fn(
        lambda _m, _u: (_ for _ in ()).throw(urllib.error.URLError("down")),
        "GET",
        "http://127.0.0.1:8000/health",
    )[0] == 0
    assert promoter._invoke_runner(lambda _a: (0, "out", "err"), ["git", "status"], 1.0) == (0, "out\nerr")

    # default_git_runner / default_gcloud_runner error handling & _extract_deployed_revision fallbacks
    orig_sub_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no git")))
    assert promoter.default_git_runner(["git", "rev-parse", "HEAD"])[0] == 127
    assert promoter.default_gcloud_runner(["gcloud", "--version"])[0] == 127
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(["gcloud"], 1.0)))
    assert promoter.default_gcloud_runner(["gcloud", "--version"]) == (124, "gcloud_timeout")
    monkeypatch.setattr(subprocess, "run", orig_sub_run)

    assert promoter._extract_deployed_revision("{not-json", "fallback-rev") == "fallback-rev"
    assert promoter._extract_deployed_revision("plain ok", "fallback-rev") == "fallback-rev"

    # Drift detection & canary broken-response / git-error / schema-mismatch branches
    assert promoter._inspect_manifest_contract({"object": "bad", "holiday_routes": None, "downloadable_scripts": None}, lang_code="en")[0] is False
    for lang_c in ("en", "fr"):
        drift_broken = promoter.detect_revision_drift(
            "http://127.0.0.1:8000",
            check_git_head=True,
            git_runner=lambda _a, _t: (128, ""),
            http_fn=lambda _m, u: (
                (200, {"status": "bad"})
                if u.endswith("/health")
                else (
                    (200, {"object": "bad", "holiday_routes": [], "downloadable_scripts": []})
                    if u.endswith("/manifest")
                    else (
                        (200, {"html": ""})
                        if u.endswith("/noel")
                        else (
                            (0, {"error": "down"})
                            if u.endswith("/quick")
                            else (
                                (503, {"error": "down"})
                                if u.endswith("/raid-noel")
                                else (
                                    (200, {"total_responses": "bad"})
                                    if u.endswith("/summary")
                                    else (200, "<!DOCTYPE html><html></html>")
                                )
                            )
                        )
                    )
                )
            ),
            env=CLEAN_ENV,
            cwd=tmp_path,
            lang=lang_c,
        )
        assert drift_broken["drift_detected"] is True
        assert drift_broken["exit_code"] == 2

    drift_git_exc = promoter.detect_revision_drift(
        "http://127.0.0.1:8000",
        check_git_head=True,
        git_runner=lambda _a, _t: (_ for _ in ()).throw(OSError("no git")),
        http_fn=lambda _m, _u: (404, {}),
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert drift_git_exc["sha_mismatch"] is True

    # Canary: /health status!=ok, SHA mismatch, POST feedback ok followed by GET summary 500
    bad_canary_h = promoter.run_post_deploy_holiday_canary(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        probe_feedback_post=True,
        http_fn=lambda m, u, _p=None, _h=None: (
            (200, {"status": "bad"})
            if u.endswith("/health")
            else (
                (200, {"object": "hivemind.onboarding.manifest", "campaign_id": "noel-2026-easy-onboard", "status": "ok", "build_sha": DRIFT_SHA, "personal_pc_only": True, "financial_promise": "0.00 EUR / 0.00 USD", "holiday_routes": list(promoter.EXPECTED_MANIFEST_HOLIDAY_ROUTES), "downloadable_scripts": list(promoter.EXPECTED_MANIFEST_SCRIPTS)})
                if u.endswith("/manifest")
                else (
                    (200, {"object": "hivemind.onboarding_feedback_summary", "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}})
                    if (m == "POST" and u.endswith("/feedback"))
                    else (500, {})
                )
            )
        ),
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert bad_canary_h["canary_ok"] is False and bad_canary_h["sha_ok"] is False and bad_canary_h["feedback_post_ok"] is False

    # Pre-promote server.py edge failures & pure-stdlib ASGI _probe_asgi_pre_promote_routes path
    srv_no_build = tmp_path / "srv_no_build.py"
    srv_no_build.write_text("X = 1\n", encoding="utf-8")
    assert promoter._load_build_app_from_server_py(srv_no_build)[0] is False

    srv_boom = tmp_path / "srv_boom.py"
    srv_boom.write_text("def build_app():\n    return None\nHIVEMIND_ROUTER_SECRET='x'\ncreate_router_app = None\nraise RuntimeError('boom')\n", encoding="utf-8")
    assert promoter._load_build_app_from_server_py(srv_boom)[0] is False

    async def _mini_broken_asgi(scope: dict[str, Any], receive: Any, send: Any) -> None:
        _ = await receive()
        path = scope["path"]
        if path == "/health":
            body, ctype, st = b'{"status":"bad"}', b"application/json", 200
        elif path == "/v1/hivemind/onboarding/manifest":
            body, ctype, st = b'{"object":"bad"}', b"application/json", 200
        elif path == "/v1/hivemind/onboarding/feedback/summary":
            body, ctype, st = b'{"bad":1}', b"application/json", 200
        elif path == "/noel":
            body, ctype, st = b"<html>bad</html>", b"text/html", 200
        else:
            body, ctype, st = b"{}", b"application/json", 500
        await send({"type": "http.response.start", "status": st, "headers": [(b"content-type", ctype)]})
        await send({"type": "http.response.body", "body": body})

    orig_import_mod = promoter.importlib.import_module
    monkeypatch.setattr(
        promoter.importlib,
        "import_module",
        lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no testclient")) if name == "fastapi.testclient" else orig_import_mod(name, *a, **kw),
    )
    ok_mini, st_mini, issues_mini, tr_mini = promoter._probe_asgi_pre_promote_routes(_mini_broken_asgi, lang_code="fr")
    monkeypatch.setattr(promoter.importlib, "import_module", orig_import_mod)
    assert ok_mini is False and tr_mini == "asgi_stdlib" and len(issues_mini) >= 5 and st_mini["/quick"] == 500

    # Lockfile corrupt JSON & safety lock extra branches + promote_holiday_revision pre-promote failure & unpinned rollback + dry-run Markdown
    corrupt_lock = tmp_path / "corrupt.lock"
    corrupt_lock.write_text("{corrupt", encoding="utf-8")
    assert promoter._inspect_local_lockfile(corrupt_lock)[0] is True

    extra_lock_chk = promoter.check_active_training_safety_lock(
        "http://127.0.0.1:8000",
        http_fn=lambda _m, u: (
            (200, {"training_active": True})
            if u.endswith("/t4/status")
            else (
                (200, {"training_active": True, "swarm": {"active_trainers": 0}})
                if u.endswith("/raid/status")
                else (200, {"training_active": True, "hiveminders": [{"status": "offline"}]})
            )
        ),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert extra_lock_chk["training_active"] is True and len(extra_lock_chk["active_reasons"]) == 3

    rep_pre_fail = promoter.promote_holiday_revision(
        "http://127.0.0.1:8000",
        deploy_script_path=tmp_path / "absent_deploy.sh",
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert rep_pre_fail["verdict"] == "pre_promote_failed" and rep_pre_fail["exit_code"] == 2

    rep_unpinned_rb = promoter.promote_holiday_revision(
        "http://127.0.0.1:8000",
        target_sha=DRIFT_SHA,
        previous_revision=None,
        execute=True,
        gcloud_runner=lambda _a, _t: (0, "staged"),
        http_fn=lambda _m, _u: (404, {}),
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert rep_unpinned_rb["rollback_triggered"] is True and rep_unpinned_rb["rollback_succeeded"] is False

    dry_plan_rep = promoter.promote_holiday_revision(
        "http://127.0.0.1:8000",
        target_sha=DRIFT_SHA,
        execute=False,
        http_fn=lambda _m, u: (404, {}),
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    dry_md = promoter.render_markdown_report(dry_plan_rep)
    assert "gcloud" in dry_md and "Dry-Run" in dry_md

    err_json_buf = io.StringIO()
    assert promoter.main(["--gateway", UNTRUSTED_EXT_GW, "--json"], env=CLEAN_ENV, cwd=tmp_path, stdout=err_json_buf) == 2
    _verify_receipt(json.loads(err_json_buf.getvalue()))

    err_txt_buf = io.StringIO()
    assert promoter.main(["--gateway", UNTRUSTED_EXT_GW], env=CLEAN_ENV, cwd=tmp_path, stderr=err_txt_buf) == 2
    assert "Error:" in err_txt_buf.getvalue()
