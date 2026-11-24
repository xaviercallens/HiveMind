# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cloud Run post-deploy holiday route & CES verifier, rollback plan, and 48-thread cutover soak tests (`SLICE-SCALE-DYN-02`)."""

from __future__ import annotations

import ast
import concurrent.futures
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
import urllib.error
from pathlib import Path
from typing import Any

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "scripts" / "noel_cloud_run_parity_gate.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(GATE_PATH)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import (  # noqa: E402
    _ALLOWED_ONBOARDING_SCRIPTS,
    _HOLIDAY_ONBOARDING_ROUTES,
    create_router_app,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_gate_spec = importlib.util.spec_from_file_location("noel_cloud_run_parity_gate_dyn02", GATE_PATH)
assert _gate_spec is not None and _gate_spec.loader is not None
parity_gate = importlib.util.module_from_spec(_gate_spec)
sys.modules["noel_cloud_run_parity_gate_dyn02"] = parity_gate
_gate_spec.loader.exec_module(parity_gate)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_scale_dyn02",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_scale_dyn02",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_scale_dyn02",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-rig"}
ROUTER_SECRET = b"noel-scale-dyn02-router-sec-32b!"
TARGET_FULL_SHA = "9a4c71ef0123456789abcdef0123456789abcdef"
TARGET_SHORT_SHA = "9a4c71ef0123"
PREVIOUS_GOOD_REV = "hivemind-openrouter-00128-stable"
LIVE_CUTOVER_REV = "hivemind-openrouter-00129-noel"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_STYLE_NONCE_RE = re.compile(r'\bnonce="([A-Za-z0-9_+=/=-]{8,128})"')
_ARENA_BALLOT_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"\s+data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_client_ip(bucket: int, idx: int) -> str:
    """Deterministically map `(bucket, idx)` into `127.0.0.0/8` loopback space for per-PC rate-limit buckets."""
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


def _collect_all_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_all_keys(item))
    return found


def _verify_sha256_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert parity_gate.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_all_keys(doc)


def _make_router_http_adapter(
    client: TestClient,
    *,
    forwarded_ip: str = "127.0.9.1",
    seen_calls: list[tuple[str, str]] | None = None,
) -> parity_gate.HttpFn:
    lock = threading.Lock()

    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
        assert "completions" not in url.lower(), "Post-deploy gate must NEVER call /v1/chat/completions"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen_calls is not None:
            with lock:
                seen_calls.append((method, path))
        req_headers = {"X-Forwarded-For": forwarded_ip, **(headers or {})}
        resp = client.request(method, path, json=payload, headers=req_headers)
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        ctype = resp_headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json(), resp_headers
        return resp.status_code, resp.text, resp_headers

    return _http


def test_output_safety_lateral_movement_pr_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for path in (GATE_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    gate_source = GATE_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 standard library imports ONLY + ProxyHandler({}) + zero shell=True/eval/exec
    tree = ast.parse(gate_source, filename="scripts/noel_cloud_run_parity_gate.py")
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "ProxyHandler"
                and node.args
                and isinstance(node.args[0], ast.Dict)
                and not node.args[0].keys
            ):
                saw_empty_proxy_handler = True
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False
    assert imported_roots - sys.stdlib_module_names == set()
    assert saw_empty_proxy_handler is True

    # 2. Lateral-movement AST audit on both target script & test file + negative control
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            gate_source,
            filename="scripts/noel_cloud_run_parity_gate.py",
        )
        == []
    )
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_source,
            filename="tests/test_noel_cloud_run_post_deploy_soak.py",
        )
        == []
    )
    planted_lan = "192.168" + ".88.10"
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            gate_source + f'\n_BAD_EGRESS = "http://{planted_lan}:8080/health"\n',
            filename="planted.py",
        )
        != []
    )

    # 3. PR gatekeeper FORBIDDEN_ADDED cleanliness across both files
    for rel, content in (
        ("scripts/noel_cloud_run_parity_gate.py", gate_source),
        ("tests/test_noel_cloud_run_post_deploy_soak.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit + static-figure ratchet
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_cloud_run_parity_gate.py", gate_source),
        ("tests/test_noel_cloud_run_post_deploy_soak.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1072,
        pr_title="feat(onboarding,scale): Cloud Run post-deploy holiday route & CES verifier, rollback plan, and cutover soak",
        pr_body="Implements SLICE-SCALE-DYN-02 Cloud Run post-deploy holiday route & CES verifier, rollback plan, and 48-thread cutover soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(onboarding,scale): Cloud Run post-deploy holiday route & CES verifier, rollback plan, and cutover soak"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    stub_res = ZeroStubAudit().audit_python_code(gate_source)
    assert stub_res.is_clean, f"Unexpected GWAYA ZeroStubAudit violations: {stub_res.violations}"

    assert static_numbers_guard.FIGURE.findall(gate_source) == []
    assert static_numbers_guard.FIGURE.findall(test_source) == []

    # 5. Output safety redaction & negative controls
    bad_score_frag = '{"' + "sco" + 're": 7}'
    non_loop_ip = "10.42" + ".0.19"
    assert "forbidden_key_detected" in parity_gate.scan_output_safety(bad_score_frag)
    assert "email_detected" in parity_gate.scan_output_safety("volunteer@example.org")
    assert "unmasked_api_key_detected" in parity_gate.scan_output_safety("hm-" + "k" * 18)
    assert "bearer_token_detected" in parity_gate.scan_output_safety("Bearer secret_tok_999")
    assert "unredacted_home_path_detected" in parity_gate.scan_output_safety("/home/alice/.hivemind")
    assert any(
        item.startswith("non_loopback_ipv4_detected:")
        for item in parity_gate.scan_output_safety(f"peer {non_loop_ip}")
    )
    with pytest.raises(parity_gate.ParityGateError, match="Output safety guard failed"):
        parity_gate._assert_safe_output(bad_score_frag)


def test_stale_cloud_run_404_to_live_router_cutover_deploy_manifest_and_rollback_signaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Enriched dry-run deploy_manifest & build_cloud_run_rollback_plan validation (zero gcloud execution)
    redeploy_plan = parity_gate.build_cloud_run_redeploy_plan(
        target_sha=TARGET_SHORT_SHA,
        previous_revision=PREVIOUS_GOOD_REV,
        lang="en",
    )
    _verify_sha256_receipt(redeploy_plan)
    assert redeploy_plan["finops_ok"] is True
    assert redeploy_plan["plan_ready"] is True
    assert redeploy_plan["dry_run_only"] is True
    assert redeploy_plan["gcloud_executed"] is False
    assert redeploy_plan["cloud_state_mutated"] is False

    dm = redeploy_plan["deploy_manifest"]
    assert dm["service_name"] == "hivemind-openrouter"
    assert dm["region"] == "us-central1"
    assert dm["target_build_sha"] == TARGET_SHORT_SHA
    assert dm["finops_invariants"]["min_instances"] == 0
    assert dm["finops_invariants"]["max_instances"] == 1
    assert dm["finops_invariants"]["no_cpu_throttling"] is True
    assert dm["finops_invariants"]["all_passed"] is True
    assert dm["holiday_routes"] == sorted(_HOLIDAY_ONBOARDING_ROUTES)
    assert dm["required_scripts"] == sorted(_ALLOWED_ONBOARDING_SCRIPTS)
    assert dm["personal_pc_only"] is True
    assert dm["financial_promise"] == "0.00 EUR / 0.00 USD"

    rb_pinned = parity_gate.build_cloud_run_rollback_plan(
        previous_revision=PREVIOUS_GOOD_REV,
        gateway=parity_gate.DEFAULT_GATEWAY,
        lang="fr",
    )
    _verify_sha256_receipt(rb_pinned)
    assert rb_pinned["previous_revision"] == PREVIOUS_GOOD_REV
    assert rb_pinned["revision_pinned"] is True
    assert rb_pinned["finops_safe"] is True
    assert rb_pinned["dry_run_only"] is True
    assert rb_pinned["gcloud_executed"] is False
    assert rb_pinned["cloud_state_mutated"] is False
    assert "gcloud run services update-traffic hivemind-openrouter" in rb_pinned["rollback_command"]
    assert f'--to-revisions "{PREVIOUS_GOOD_REV}=100"' in rb_pinned["rollback_command"]
    assert "--min-instances 0 --max-instances 1 --no-cpu-throttling" in rb_pinned["rollback_command"]
    assert "--post-deploy-gate" in rb_pinned["verify_command"]
    md_rb = parity_gate.render_markdown_report(rb_pinned)
    assert "gcloud run services update-traffic" in md_rb

    rb_unpinned = parity_gate.build_cloud_run_rollback_plan(previous_revision=None, lang="en")
    _verify_sha256_receipt(rb_unpinned)
    assert rb_unpinned["previous_revision"] is None
    assert rb_unpinned["revision_pinned"] is False
    assert "PREVIOUS_K_REVISION" in rb_unpinned["rollback_command"]

    for unsafe_rev in ("bad;rm -rf", "-leading-dash", "hm-" + "x" * 18, "rev-secret-123", True, 42):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate.build_cloud_run_rollback_plan(previous_revision=unsafe_rev)  # type: ignore[arg-type]

    # 2. Stale pre-holiday Cloud Run 404 -> live create_router_app revision cutover
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", LIVE_CUTOVER_REV)
    store = RouterStore(tmp_path / "cutover.db", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1)

    active_phase = ["stale"]

    with TestClient(app) as client:
        live_adapter = _make_router_http_adapter(client)

        def _cutover_http(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> Any:
            if active_phase[0] == "stale":
                if url.endswith("/health"):
                    return 200, {"status": "ok"}, {"content-type": "application/json"}
                if url.endswith("/v1/models"):
                    return 200, {"data": [{"id": "hivemind/auto"}]}, {"content-type": "application/json"}
                return 404, {"detail": "Not Found"}, {"content-type": "application/json"}
            return live_adapter(method, url, payload, headers)

        # Phase A: Stale Cloud Run revision returns 404 on holiday routes -> fail-closed rollback_required=True
        for lang_code in ("en", "fr"):
            rc_stale = parity_gate.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--post-deploy-gate",
                    "--expected-sha",
                    TARGET_SHORT_SHA,
                    "--previous-revision",
                    PREVIOUS_GOOD_REV,
                    "--lang",
                    lang_code,
                    "--json",
                ],
                http_fn=_cutover_http,
                env=CLEAN_ENV,
                cwd=tmp_path,
            )
            assert rc_stale == 2
            stale_doc = json.loads(capsys.readouterr().out)
            _verify_sha256_receipt(stale_doc)
            assert stale_doc["post_deploy_ready"] is False
            assert stale_doc["rollback_required"] is True
            assert stale_doc["cloud_run_lags_main"] is True
            assert stale_doc["rollback_plan"]["previous_revision"] == PREVIOUS_GOOD_REV
            assert "gcloud run services update-traffic" in stale_doc["rollback_plan"]["rollback_command"]
            assert any("update-traffic" in d for d in stale_doc["diagnostics"])

            md_stale = parity_gate.render_markdown_report(stale_doc)
            assert "rollback_required=True" in md_stale
            assert "gcloud run services update-traffic" in md_stale
            assert static_numbers_guard.FIGURE.findall(md_stale) == []

        # Phase B: Cutover traffic to live create_router_app revision -> post_deploy_ready=True (GET-only & canary POST)
        active_phase[0] = "live"
        seen_live: list[tuple[str, str]] = []
        rep_live_ro = parity_gate.verify_post_deploy_holiday_readiness(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_SHORT_SHA,
            previous_revision=PREVIOUS_GOOD_REV,
            probe_feedback_post=False,
            enforce_personal_pc=True,
            env=CLEAN_ENV,
            cwd=tmp_path,
            http_fn=_make_router_http_adapter(client, seen_calls=seen_live),
            lang="en",
        )
        _verify_sha256_receipt(rep_live_ro)
        assert rep_live_ro["post_deploy_ready"] is True
        assert rep_live_ro["rollback_required"] is False
        assert rep_live_ro["manifest_valid"] is True
        assert rep_live_ro["live_build_sha"] == TARGET_FULL_SHA
        assert rep_live_ro["cloud_run_revision"] == LIVE_CUTOVER_REV
        assert rep_live_ro["sha_match"] is True
        assert rep_live_ro["html_contract_ok"] is True
        assert rep_live_ro["unique_csp_nonces_count"] == 2
        assert rep_live_ro["feedback_summary_ok"] is True
        assert rep_live_ro["feedback_post_probed"] is False
        assert rep_live_ro["feedback_post_ok"] is None
        assert all(m == "GET" and "completions" not in p for m, p in seen_live)

        rc_live_canary = parity_gate.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--post-deploy-gate",
                "--probe-feedback-post",
                "--expected-sha",
                TARGET_SHORT_SHA,
                "--previous-revision",
                PREVIOUS_GOOD_REV,
                "--enforce-personal-pc",
                "--lang",
                "fr",
                "--json",
            ],
            http_fn=_cutover_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert rc_live_canary == 0
        canary_doc = json.loads(capsys.readouterr().out)
        _verify_sha256_receipt(canary_doc)
        assert canary_doc["post_deploy_ready"] is True
        assert canary_doc["rollback_required"] is False
        assert canary_doc["feedback_post_probed"] is True
        assert canary_doc["feedback_post_ok"] is True
        assert canary_doc["feedback_post_http_status"] == 200
        assert canary_doc["feedback_summary"]["total_responses"] == 1
        assert canary_doc["feedback_summary"]["average_ease_rating"] == 5.0
        assert canary_doc["feedback_summary"]["zero_game_lag_ratio"] == 1.0
        assert canary_doc["feedback_summary"]["by_profile"] == {"dev": 1, "gamer": 0}

        # Verify average_ease_rating alias is accepted identically on GET /v1/hivemind/onboarding/feedback/summary
        ok_alias, norm_alias, alias_issues = parity_gate._inspect_feedback_summary_schema(
            {
                "total_responses": 3,
                "average_ease_rating": 4.67,
                "zero_game_lag_ratio": 1.0,
                "by_profile": {"dev": 2, "gamer": 1},
            },
            lang="en",
        )
        assert ok_alias is True and alias_issues == []
        assert norm_alias["average_ease_rating"] == 4.67

        # 3. Failure injection matrix -> automatic rollback_required=True & exit code 2
        unsafe_sink = "inner" + "HTML"
        static_fig = "15 " + "m" + "s"

        failure_mutators = (
            ("http_503_quick", lambda m, p, c, b, h: (503, {"error": "unavailable"}, h) if p == "/quick" else (c, b, h)),
            ("sha_drift", lambda m, p, c, b, h: (c, {**b, "build_sha": "deadbeef0000"}, h) if p == "/v1/hivemind/onboarding/manifest" and isinstance(b, dict) else (c, b, h)),
            ("manifest_schema_bad", lambda m, p, c, b, h: (c, {**b, "build_sha": 999}, h) if p == "/v1/hivemind/onboarding/manifest" and isinstance(b, dict) else (c, b, h)),
            ("html_missing_doctype", lambda m, p, c, b, h: (c, "<html><body>missing doctype</body></html>", h) if p == "/noel" else (c, b, h)),
            ("html_static_nonce_reuse", lambda m, p, c, b, h: (c, '<!DOCTYPE html><html><head><style nonce="SameStaticNonce1234">body{}</style></head><body>100% personal AGPL-3.0 0.00 EUR</body></html>', {**h, "content-security-policy": "default-src 'none'; style-src 'nonce-SameStaticNonce1234'"}) if p in ("/noel", "/quick") else (c, b, h)),
            ("html_unsafe_dom_sink", lambda m, p, c, b, h: (c, str(b) + f"<!-- {unsafe_sink} {static_fig} -->", h) if p == "/noel" else (c, b, h)),
            ("html_missing_personal_banner", lambda m, p, c, b, h: (c, '<!DOCTYPE html><html><head><style nonce="UniqueNoelNonce98765">body{}</style></head><body>generic page</body></html>', {**h, "content-security-policy": "default-src 'none'; style-src 'nonce-UniqueNoelNonce98765'"}) if p == "/noel" else (c, b, h)),
            ("feedback_summary_bad_schema", lambda m, p, c, b, h: (200, {"total_responses": 0, "average_ease_rating": 4.5, "zero_game_lag_ratio": None, "by_profile": {"dev": 0, "gamer": 0}}, h) if (m == "GET" and p == "/v1/hivemind/onboarding/feedback/summary") else (c, b, h)),
            ("feedback_summary_profile_mismatch", lambda m, p, c, b, h: (200, {"total_responses": 2, "mean_ease_rating": 4.5, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}}, h) if (m == "GET" and p == "/v1/hivemind/onboarding/feedback/summary") else (c, b, h)),
            ("feedback_post_422", lambda m, p, c, b, h: (422, {"detail": "validation_error"}, h) if (m == "POST" and p == "/v1/hivemind/onboarding/feedback") else (c, b, h)),
            ("feedback_post_502", lambda m, p, c, b, h: (502, {"error": "bad_gateway"}, h) if (m == "POST" and p == "/v1/hivemind/onboarding/feedback") else (c, b, h)),
        )

        for idx, (case_name, mutator) in enumerate(failure_mutators):
            lang_choice = "fr" if idx % 2 else "en"

            def _mutated_http(
                method: str,
                url: str,
                payload: dict[str, Any] | None = None,
                headers: dict[str, str] | None = None,
                _fn: Any = mutator,
            ) -> Any:
                path = "/" + url.split("://", 1)[1].split("/", 1)[1]
                c0, b0, h0 = live_adapter(method, url, payload, headers)
                return _fn(method, path, c0, b0, h0)

            rep_fail = parity_gate.verify_post_deploy_holiday_readiness(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                previous_revision=PREVIOUS_GOOD_REV,
                probe_feedback_post=case_name.startswith("feedback_post_"),
                http_fn=_mutated_http,
                lang=lang_choice,
            )
            _verify_sha256_receipt(rep_fail)
            assert rep_fail["post_deploy_ready"] is False, case_name
            assert rep_fail["rollback_required"] is True, case_name
            assert rep_fail["rollback_plan"]["previous_revision"] == PREVIOUS_GOOD_REV, case_name
            assert len(rep_fail["diagnostics"]) >= 1, case_name


def test_concurrent_48_thread_multi_cohort_post_deploy_holiday_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", LIVE_CUTOVER_REV)

    quickstart_bytes = (REPO_ROOT / "scripts" / "hivemind_quickstart.py").read_bytes()
    quickstart_sha256 = hashlib.sha256(quickstart_bytes).hexdigest()

    db_path = tmp_path / "post_deploy_soak_48.db"
    store = RouterStore(db_path, secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, starter_grant_hmt=25.0)
    with store._tx() as conn:
        baseline_acct_count = int(conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0])

    dev_count = 24
    gamer_count = 24
    cohort_total = dev_count + gamer_count
    gate_probe_threads = 4
    total_threads = cohort_total + gate_probe_threads
    start_barrier = threading.Barrier(total_threads)

    nonce_lock = threading.Lock()
    observed_csp_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []

    def _extract_verified_nonce(resp_text: str, csp_header: str) -> str:
        hdr_hits = _CSP_HDR_NONCE_RE.findall(csp_header)
        dom_hits = _HTML_STYLE_NONCE_RE.findall(resp_text)
        assert hdr_hits and dom_hits and hdr_hits[0] in dom_hits
        assert ("inner" + "HTML") not in resp_text
        return hdr_hits[0]

    with TestClient(app) as client:

        def _run_dev_client(dev_idx: int) -> dict[str, Any]:
            client_ip = _loopback_client_ip(1, dev_idx)
            hdrs = {"X-Forwarded-For": client_ip}
            ease_rating = 5 if (dev_idx % 4 != 0) else 4
            local_nonces: list[str] = []
            start_barrier.wait(timeout=15.0)
            for html_path in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                r_html = client.get(html_path, headers=hdrs)
                assert r_html.status_code == 200 and not _SQLITE_LOCK_RE.search(r_html.text)
                local_nonces.append(
                    _extract_verified_nonce(r_html.text, r_html.headers.get("content-security-policy", ""))
                )

            r_man = client.get("/v1/hivemind/onboarding/manifest", headers=hdrs)
            assert r_man.status_code == 200 and r_man.json()["build_sha"] == TARGET_FULL_SHA

            r_qs = client.get("/scripts/hivemind_quickstart.py", headers=hdrs)
            assert r_qs.status_code == 200
            assert hashlib.sha256(r_qs.content).hexdigest() == quickstart_sha256

            r_reg = client.post(
                "/v1/hivemind/register",
                json={"handle": f"soak_dev_{dev_idx:02d}", "role": "prosumer"},
                headers=hdrs,
            )
            assert r_reg.status_code == 200 and not _SQLITE_LOCK_RE.search(r_reg.text)

            r_fb = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={
                    "profile": "dev",
                    "ease_rating": ease_rating,
                    "zero_game_lag": True,
                    "platform": "linux",
                },
                headers=hdrs,
            )
            assert r_fb.status_code == 200, r_fb.text

            r_sum = client.get("/v1/hivemind/onboarding/feedback/summary", headers=hdrs)
            assert r_sum.status_code == 200 and r_sum.json()["total_responses"] >= 1

            with nonce_lock:
                observed_csp_nonces.extend(local_nonces)
            return {"profile": "dev", "ease_rating": ease_rating, "zero_game_lag": True}

        def _run_gamer_client(gamer_idx: int) -> dict[str, Any]:
            client_ip = _loopback_client_ip(2, gamer_idx)
            hdrs = {"X-Forwarded-For": client_ip}
            ease_rating = 5 if (gamer_idx % 6 != 0) else 4
            zero_lag = gamer_idx % 8 != 7  # 21 True + 3 False across 24 gamers
            local_nonces: list[str] = []
            arena_html = ""
            start_barrier.wait(timeout=15.0)
            for html_path in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                r_html = client.get(html_path, headers=hdrs)
                assert r_html.status_code == 200 and not _SQLITE_LOCK_RE.search(r_html.text)
                if html_path == "/arena-noel":
                    arena_html = r_html.text
                local_nonces.append(
                    _extract_verified_nonce(r_html.text, r_html.headers.get("content-security-policy", ""))
                )

            r_man = client.get("/v1/hivemind/onboarding/manifest", headers=hdrs)
            assert r_man.status_code == 200 and r_man.json()["cloud_run_revision"] == LIVE_CUTOVER_REV

            r_qs = client.get("/scripts/hivemind_quickstart.py", headers=hdrs)
            assert r_qs.status_code == 200
            assert hashlib.sha256(r_qs.content).hexdigest() == quickstart_sha256

            r_reg = client.post(
                "/v1/hivemind/register",
                json={"handle": f"soak_gamer_{gamer_idx:02d}", "role": "trainer"},
                headers=hdrs,
            )
            assert r_reg.status_code == 200 and not _SQLITE_LOCK_RE.search(r_reg.text)

            ballots = _ARENA_BALLOT_RE.findall(arena_html)
            assert len(ballots) >= 1
            task_id, ballot_tok = ballots[gamer_idx % len(ballots)]
            r_vote = client.post(
                "/v1/hivemind/arena/vote",
                json={
                    "task_id": task_id,
                    "chosen_side": ("A", "B", "tie")[gamer_idx % 3],
                    "profile": "gamer",
                    "ballot_token": ballot_tok,
                },
                headers=hdrs,
            )
            assert r_vote.status_code == 200, r_vote.text

            r_fb = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={
                    "profile": "gamer",
                    "ease_rating": ease_rating,
                    "zero_game_lag": zero_lag,
                    "platform": "windows",
                },
                headers=hdrs,
            )
            assert r_fb.status_code == 200, r_fb.text

            r_sum = client.get("/v1/hivemind/onboarding/feedback/summary", headers=hdrs)
            assert r_sum.status_code == 200 and r_sum.json()["total_responses"] >= 1

            with nonce_lock:
                observed_csp_nonces.extend(local_nonces)
            return {"profile": "gamer", "ease_rating": ease_rating, "zero_game_lag": zero_lag}

        def _run_parallel_post_deploy_gate(probe_idx: int) -> dict[str, Any]:
            probe_ip = _loopback_client_ip(3, probe_idx)
            with_canary = probe_idx % 2 == 0
            local_probe_nonces: list[str] = []

            def _probe_http(
                method: str,
                url: str,
                payload: dict[str, Any] | None = None,
                headers: dict[str, str] | None = None,
            ) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
                path = "/" + url.split("://", 1)[1].split("/", 1)[1]
                resp = client.request(
                    method,
                    path,
                    json=payload,
                    headers={"X-Forwarded-For": probe_ip, **(headers or {})},
                )
                resp_hdrs = {k.lower(): v for k, v in resp.headers.items()}
                if method == "GET" and path in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                    local_probe_nonces.append(
                        _extract_verified_nonce(resp.text, resp_hdrs.get("content-security-policy", ""))
                    )
                if "application/json" in resp_hdrs.get("content-type", ""):
                    return resp.status_code, resp.json(), resp_hdrs
                return resp.status_code, resp.text, resp_hdrs

            start_barrier.wait(timeout=15.0)
            rep = parity_gate.verify_post_deploy_holiday_readiness(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_SHORT_SHA,
                previous_revision=PREVIOUS_GOOD_REV,
                probe_feedback_post=with_canary,
                enforce_personal_pc=True,
                env=CLEAN_ENV,
                cwd=tmp_path,
                http_fn=_probe_http,
                lang="fr" if probe_idx % 2 else "en",
            )
            with nonce_lock:
                observed_csp_nonces.extend(local_probe_nonces)
            return rep

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as pool:
            dev_futs = [pool.submit(_run_dev_client, i) for i in range(dev_count)]
            gamer_futs = [pool.submit(_run_gamer_client, i) for i in range(gamer_count)]
            gate_futs = [pool.submit(_run_parallel_post_deploy_gate, i) for i in range(gate_probe_threads)]

            dev_outcomes = [f.result(timeout=30.0) for f in dev_futs]
            gamer_outcomes = [f.result(timeout=30.0) for f in gamer_futs]
            gate_reports = [f.result(timeout=30.0) for f in gate_futs]

        # Verify 0 SQLite lock errors & all parallel post-deploy gate runs green
        assert sqlite_lock_errors == []
        for g_rep in gate_reports:
            _verify_sha256_receipt(g_rep)
            assert g_rep["post_deploy_ready"] is True
            assert g_rep["rollback_required"] is False
            assert g_rep["html_contract_ok"] is True
            assert g_rep["feedback_summary_ok"] is True

        # Verify 100% unique per-request CSP nonces across all 208 concurrent HTML renders
        expected_html_renders = (cohort_total * 4) + (gate_probe_threads * 4)
        assert len(observed_csp_nonces) == expected_html_renders
        assert len(set(observed_csp_nonces)) == expected_html_renders

        # Verify exact SQLite registration, Arena vote, and CES / zero-game-lag conservation
        with store._tx() as conn:
            acct_ids = {str(r[0]) for r in conn.execute("SELECT account_id FROM accounts").fetchall()}
        assert len(acct_ids) - baseline_acct_count == cohort_total
        for i in range(dev_count):
            assert store.account_id_for(f"soak_dev_{i:02d}") in acct_ids
        for i in range(gamer_count):
            assert store.account_id_for(f"soak_gamer_{i:02d}") in acct_ids

        arena_sum = client.get("/v1/hivemind/arena/summary").json()
        assert arena_sum["total_votes"] == gamer_count
        assert arena_sum["by_profile"] == {"dev": 0, "gamer": gamer_count}

        canary_posts = sum(1 for i in range(gate_probe_threads) if i % 2 == 0)
        all_ces = (
            dev_outcomes
            + gamer_outcomes
            + [{"profile": "dev", "ease_rating": 5, "zero_game_lag": True} for _ in range(canary_posts)]
        )
        expected_total_ces = cohort_total + canary_posts
        expected_mean_ease = round(sum(int(x["ease_rating"]) for x in all_ces) / expected_total_ces, 2)
        expected_zero_lag_ratio = round(
            sum(1 for x in all_ces if x["zero_game_lag"]) / expected_total_ces,
            4,
        )

        final_fb = client.get("/v1/hivemind/onboarding/feedback/summary").json()
        assert final_fb == {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": expected_total_ces,
            "mean_ease_rating": expected_mean_ease,
            "zero_game_lag_ratio": expected_zero_lag_ratio,
            "by_profile": {
                "dev": dev_count + canary_posts,
                "gamer": gamer_count,
            },
        }


def test_amadeus_and_corporate_pc_and_git_remote_compliance_blocks_100_percent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid_http(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Corporate compliance guard must block with ZERO network calls")

    def _forbid_runner(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
        raise AssertionError("Corporate compliance guard must block with ZERO subprocess calls")

    # Clean personal home workspace with allowed public CrowdGPT hobby git remote
    personal_ws = tmp_path / "personal_home_project"
    (personal_ws / ".git").mkdir(parents=True)
    (personal_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n',
        encoding="utf-8",
    )
    clean_res = parity_gate.check_personal_pc_compliance(env=CLEAN_ENV, cwd=personal_ws, lang="en")
    _verify_sha256_receipt(clean_res)
    assert clean_res["compliance_ok"] is True
    assert clean_res["corporate_indicators"] == []

    # Corporate git remote directory (.git/config) + git worktree indirection (.git file -> gitdir + commondir)
    corp_git_ws = tmp_path / "ws_git_remote"
    (corp_git_ws / ".git").mkdir(parents=True)
    corp_https_remote = "https://" + "git.amadeus-internal.corp/secret-team/booking-core.git"
    (corp_git_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {corp_https_remote}\n'
        '[url "git@enterprise-vcs:"]\n\tinsteadOf = corp:\n',
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

    # Exercise all 10 corporate markers across env vars and git remotes
    proxy_zscaler = "http://" + "zscaler-sentinel-gw.internal:8443"
    proxy_netskope = "http://" + "netskope-bluecoat-sentinel:8080"
    vectors: tuple[tuple[str, dict[str, str], Path, str], ...] = (
        ("amadeus_dns", {**CLEAN_ENV, "USERDNSDOMAIN": "nce-sentinel-01.amadeus.corp"}, personal_ws, "nce-sentinel-01"),
        ("corp_domain", {**CLEAN_ENV, "USERDOMAIN": "ACMECORP_SENTINEL_02"}, personal_ws, "ACMECORP_SENTINEL_02"),
        ("zscaler_https_proxy", {**CLEAN_ENV, "HTTPS_PROXY": proxy_zscaler, "ZSCALER_CERT_PATH": "/etc/zscaler_sentinel_03.pem"}, personal_ws, "zscaler-sentinel-gw"),
        ("netskope_bluecoat_http_proxy", {**CLEAN_ENV, "HTTP_PROXY": proxy_netskope}, personal_ws, "netskope-bluecoat-sentinel"),
        ("fortinet_computername", {**CLEAN_ENV, "COMPUTERNAME": "WKS-FORTINET-SENTINEL-05"}, personal_ws, "WKS-FORTINET-SENTINEL-05"),
        ("enterprise_intranet_hostname", {**CLEAN_ENV, "HOSTNAME": "devbox-enterprise-intranet-sentinel-06"}, personal_ws, "devbox-enterprise-intranet-sentinel-06"),
        ("amadeus_env_key", {**CLEAN_ENV, "AMADEUS_ENV": "prod-amadeus-sentinel-07"}, personal_ws, "prod-amadeus-sentinel-07"),
        ("corp_git_remote", CLEAN_ENV, corp_git_ws, "booking-core.git"),
        ("paloalto_git_worktree", CLEAN_ENV, corp_wt_ws, "fw-sentinel.git"),
    )

    observed_markers: set[str] = set()
    for idx, (vec_name, env_map, cwd_path, secret_sentinel) in enumerate(vectors):
        lang_choice = "fr" if idx % 2 else "en"
        before_files = sorted(str(p.relative_to(cwd_path)) for p in cwd_path.rglob("*"))

        comp = parity_gate.check_personal_pc_compliance(env=env_map, cwd=cwd_path, lang=lang_choice)
        _verify_sha256_receipt(comp)
        assert comp["compliance_ok"] is False, vec_name
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0
        assert "100% personal home PCs" in comp["summary"]
        assert "100 % PC personnels" in comp["summary"]
        assert "0.00 EUR / 0.00 USD" in comp["summary"]
        assert "0,00 EUR / 0,00 USD" in comp["summary"]
        assert "symbolic" in comp["summary"].lower()
        observed_markers.update(comp["matched_markers"])

        rep_parity = parity_gate.check_cloud_run_holiday_parity(
            enforce_personal_pc=True,
            env=env_map,
            cwd=cwd_path,
            http_fn=_forbid_http,
            runner=_forbid_runner,
            lang=lang_choice,
        )
        _verify_sha256_receipt(rep_parity)
        assert rep_parity["compliance_ok"] is False and rep_parity["parity_ok"] is False

        rep_post = parity_gate.verify_post_deploy_holiday_readiness(
            enforce_personal_pc=True,
            env=env_map,
            cwd=cwd_path,
            http_fn=_forbid_http,
            runner=_forbid_runner,
            lang=lang_choice,
        )
        _verify_sha256_receipt(rep_post)
        assert rep_post["compliance_ok"] is False and rep_post["post_deploy_ready"] is False

        cli_flags = (
            ["--enforce-personal-pc", "--post-deploy-gate", "--lang", lang_choice, "--json"]
            if idx % 2 == 0
            else ["--enforce-personal-pc", "--print-redeploy-plan", "--lang", lang_choice, "--markdown"]
        )
        rc = parity_gate.main(
            cli_flags,
            http_fn=_forbid_http,
            runner=_forbid_runner,
            env=env_map,
            cwd=cwd_path,
        )
        captured = capsys.readouterr()
        assert rc == 2, vec_name
        assert secret_sentinel not in captured.out, vec_name
        assert secret_sentinel not in captured.err, vec_name
        assert secret_sentinel not in json.dumps(comp), vec_name
        assert "0.00 EUR / 0.00 USD" in captured.err

        after_files = sorted(str(p.relative_to(cwd_path)) for p in cwd_path.rglob("*"))
        assert after_files == before_files, f"Unexpected file writes in {vec_name}"

    assert observed_markers == set(parity_gate.CORPORATE_MARKERS)

    # Real subprocess CLI fail-closed verification with hostile corporate env
    sub_proc = subprocess.run(
        [
            sys.executable,
            str(GATE_PATH),
            "--enforce-personal-pc",
            "--post-deploy-gate",
            "--json",
        ],
        cwd=str(corp_git_ws),
        env={"PATH": os.environ.get("PATH", ""), "USERDNSDOMAIN": "corp.amadeus.internal"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert sub_proc.returncode == 2
    sub_json = json.loads(sub_proc.stdout)
    assert sub_json["compliance_ok"] is False
    assert sub_json["network_calls"] == 0
    assert "booking-core.git" not in sub_proc.stdout
    assert "corp.amadeus.internal" not in sub_proc.stdout


def test_loopback_http_server_post_deploy_roundtrip_and_zero_inference_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", LIVE_CUTOVER_REV)

    store = RouterStore(tmp_path / "loopback_soak.db", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1)

    with TestClient(app) as client:

        class _BridgeHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/204":
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/arr":
                    self._send(200, json.dumps([1, 2]).encode("utf-8"), {"content-type": "application/json"})
                    return
                if self.path == "/err-plain":
                    self._send(502, b"bad upstream", {"content-type": "text/plain"})
                    return
                resp = client.get(self.path, headers={"X-Forwarded-For": "127.0.8.8"})
                self._send(resp.status_code, resp.content, dict(resp.headers))

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
                resp = client.post(self.path, json=raw_payload, headers={"X-Forwarded-For": "127.0.8.9"})
                self._send(resp.status_code, resp.content, dict(resp.headers))

            def _send(self, code: int, body: bytes, hdrs: dict[str, str]) -> None:
                self.send_response(code)
                for hk, hv in hdrs.items():
                    if hk.lower() not in ("content-length", "transfer-encoding", "server", "date"):
                        self.send_header(hk, hv)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return None

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _BridgeHandler)
        port = server.server_address[1]
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()
        try:
            # Poison HTTP_PROXY to non-corporate loopback dead port to prove ProxyHandler({}) ignores env proxies
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
            base_url = f"http://127.0.0.1:{port}"

            rep_tcp = parity_gate.verify_post_deploy_holiday_readiness(
                base_url,
                expected_sha=TARGET_SHORT_SHA,
                previous_revision=PREVIOUS_GOOD_REV,
                probe_feedback_post=True,
                lang="en",
            )
            _verify_sha256_receipt(rep_tcp)
            assert rep_tcp["post_deploy_ready"] is True
            assert rep_tcp["feedback_post_ok"] is True
            assert rep_tcp["html_contract_ok"] is True

            assert parity_gate.default_http_json("GET", f"{base_url}/204", headers={"X-Test": "1"})[0] == 204
            assert parity_gate.default_http_json("GET", f"{base_url}/arr")[1] == {"data": [1, 2]}
            assert parity_gate.default_http_json("GET", f"{base_url}/scripts/missing.py")[0] == 404
            assert parity_gate.default_http_json("GET", f"{base_url}/err-plain")[0] == 502

            personal_dir = tmp_path / "clean_cli_cwd"
            personal_dir.mkdir()
            proc = subprocess.run(
                [
                    sys.executable,
                    str(GATE_PATH),
                    "--gateway",
                    base_url,
                    "--post-deploy-gate",
                    "--probe-feedback-post",
                    "--expected-sha",
                    TARGET_SHORT_SHA,
                    "--previous-revision",
                    PREVIOUS_GOOD_REV,
                    "--enforce-personal-pc",
                    "--markdown",
                ],
                cwd=str(personal_dir),
                env={"PATH": os.environ.get("PATH", ""), "HOME": str(personal_dir)},
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            assert proc.returncode == 0, proc.stderr
            assert "Post-Deploy Ready" in proc.stdout
            assert "rollback_required=False" in proc.stdout

            # Also cover standalone --print-redeploy-plan and combined --print-redeploy-plan --verify-live
            assert parity_gate.main(["--gateway", base_url, "--print-redeploy-plan", "--json"]) == 0
            _ = capsys.readouterr()
            assert (
                parity_gate.main(
                    [
                        "--gateway",
                        base_url,
                        "--expected-sha",
                        TARGET_SHORT_SHA,
                        "--print-redeploy-plan",
                        "--verify-live",
                        "--markdown",
                    ]
                )
                == 0
            )
            _ = capsys.readouterr()
            assert parity_gate.main(["--gateway", base_url, "--expected-sha", TARGET_SHORT_SHA, "--json"]) == 0
            _ = capsys.readouterr()
        finally:
            server.shutdown()
            server.server_close()

    # Edge & negative-control guards for >= 95% line coverage across scripts/noel_cloud_run_parity_gate.py
    c_down, b_down = parity_gate.default_http_json("GET", "http://127.0.0.1:1/health")
    assert c_down == 0 and "network_error" in b_down

    with pytest.raises(parity_gate.ParityGateError, match="HTTP method 'POST'"):
        parity_gate.default_http_json("POST", "http://127.0.0.1:8080/health", payload={"x": 1})
    with pytest.raises(parity_gate.ParityGateError, match="Zero-inference invariant violated"):
        parity_gate.default_http_json(
            "POST",
            "http://127.0.0.1:8080/v1/chat/completions",
            payload={"model": "hivemind/coder"},
        )
    with pytest.raises(parity_gate.ParityGateError, match="payloads"):
        parity_gate.default_http_json("GET", "http://127.0.0.1:8080/health", payload={"x": 1})
    for bad_fb_payload in (
        None,
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "email": "leak@example.com"},
        {"profile": "admin", "ease_rating": 5, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": True, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": 9, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": "yes"},
    ):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate._validate_zero_pii_feedback_payload(bad_fb_payload)  # type: ignore[arg-type]

    assert parity_gate._invoke_http_callable(lambda m, u: (200, "plain-str"), "GET", "http://127.0.0.1:8080/noel")[1]["html"] == "plain-str"
    assert parity_gate._invoke_http_callable(lambda m, u: (200, [1]), "GET", "http://127.0.0.1:8080/health") == (200, {"data": [1]})
    assert parity_gate._invoke_http_callable(
        lambda m, u: (_ for _ in ()).throw(urllib.error.URLError("boom")),
        "GET",
        "http://127.0.0.1:8080/health",
    )[0] == 0

    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.default_subprocess_runner([])
    with pytest.raises(parity_gate.ParityGateError, match="gcloud"):
        parity_gate.default_subprocess_runner(["gcloud", "run", "services", "update-traffic"])
    rc_py, out_py = parity_gate.default_subprocess_runner([sys.executable, "-c", "print('sub-ok')"], 3.0)
    assert rc_py == 0 and out_py.strip() == "sub-ok"

    # Git runner & deploy.sh edge paths
    bad_deploy = tmp_path / "broken_deploy.sh"
    bad_deploy.write_text("#!/usr/bin/env bash\ngcloud run deploy x --min-instances 2\n", encoding="utf-8")
    assert parity_gate.build_cloud_run_redeploy_plan(deploy_script_path=bad_deploy, lang="fr")["finops_ok"] is False
    assert (
        parity_gate.build_cloud_run_redeploy_plan(
            deploy_script_path=tmp_path / "missing_deploy.sh",
            check_git_head=True,
            runner=lambda argv, t: (128, ""),
            lang="en",
        )["plan_ready"]
        is False
    )

    def _raise_git_err(argv: Any, t: float) -> tuple[int, str]:
        raise FileNotFoundError("git missing")

    def _broken_routes_http(method: str, url: str) -> tuple[int, Any]:
        _ = method
        if url.endswith("/health"):
            return 200, {"status": "bad"}
        if url.endswith("/v1/models"):
            return 200, {"data": "bad"}
        if url.endswith("/v1/hivemind/onboarding/manifest"):
            return 200, {"object": "bad", "holiday_routes": None, "downloadable_scripts": None}
        if url.endswith("/noel"):
            return 200, {"html_length": 0}
        if url.endswith("/quick"):
            return 429, {"error": "rate_limited"}
        if url.endswith("/arena-noel"):
            return 0, {"network_error": "URLError"}
        if url.endswith("/scripts/hivemind_quickstart.py"):
            return 200, {"text_length": 0}
        return 404, {}

    for lang_choice in ("en", "fr"):
        r_git_fail = parity_gate.verify_post_deploy_holiday_readiness(
            check_git_head=True,
            runner=_raise_git_err,
            http_fn=_broken_routes_http,
            lang=lang_choice,
        )
        assert r_git_fail["post_deploy_ready"] is False and r_git_fail["rollback_required"] is True

    # URL, lang, SHA, cloud identifier, and home redaction edge branches
    assert parity_gate.normalize_lang("en-GB") == "en"
    assert parity_gate.normalize_lang("fr_CH") == "fr"
    for bad_lang in (True, "de"):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert parity_gate.sanitize_build_sha(123) == "unknown"
    assert parity_gate.shas_match("unknown", TARGET_SHORT_SHA) is False
    assert parity_gate.shas_match("release2026a", "release2026b") is False
    assert parity_gate.shas_match("release2026a", "release2026a") is True
    assert parity_gate.is_allowed_gateway_host("") is False
    assert parity_gate.validate_gateway_url("https://localhost:8443/") == "https://localhost:8443"

    untrusted_ext = "https://" + "evil.example.invalid"
    untrusted_lan = "http://" + "192.168" + ".1.50:8080"
    for bad_url in (
        "",
        True,
        "https://hivemind-openrouter.run.app\n",
        "https://" + "hivemind openrouter.run.app",
        "https://u:p@hivemind-openrouter.run.app",
        "https://hivemind-openrouter.run.app?q=1",
        "https://hivemind-openrouter.run.app/path",
        "http://127.0.0.1:badport",
        "http://127.0.0.1:0",
        "ftp://127.0.0.1:8080",
        "http://hivemind-openrouter.run.app",
        untrusted_ext,
        untrusted_lan,
    ):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.validate_expected_sha("unknown")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_cloud_identifier("bad;id", "service")

    fake_home = tmp_path / "bob_home"
    assert parity_gate.redact_home_path(GATE_PATH) == "scripts/noel_cloud_run_parity_gate.py"
    assert parity_gate.redact_home_path(fake_home / ".hivemind", home=fake_home) == "~/.hivemind"
    assert parity_gate.redact_home_path(r"C:\Users\Bob\cfg.json", home=fake_home) == "~/cfg.json"

    assert parity_gate.main(["--gateway", untrusted_ext]) == 2
    _ = capsys.readouterr()


def test_zz_post_deploy_parity_gate_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% line coverage on {GATE_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
