# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, security, FinOps, and gateway integration tests for SLICE-DYN-02 (`scripts/noel_cloud_run_parity_gate.py`)."""

from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import (
    _ALLOWED_ONBOARDING_SCRIPTS,
    _HOLIDAY_ONBOARDING_ROUTES,
    _sanitize_build_sha,
    _sanitize_cloud_run_revision,
    build_onboarding_manifest,
    create_router_app,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "scripts" / "noel_cloud_run_parity_gate.py"
GATEWAY_PY_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "gateway.py"

_spec = importlib.util.spec_from_file_location("noel_cloud_run_parity_gate", GATE_PATH)
assert _spec is not None and _spec.loader is not None
parity_gate = importlib.util.module_from_spec(_spec)
sys.modules["noel_cloud_run_parity_gate"] = parity_gate
_spec.loader.exec_module(parity_gate)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn02",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn02",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

SAMPLE_FULL_SHA = "3bb869116563a9b44da817047a8b17b8dddf5e5b"
SAMPLE_SHORT_SHA = "3bb86911"


def _make_asgi_http(client: TestClient, seen: list[tuple[str, str]] | None = None) -> parity_gate.HttpFn:
    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert method == "GET", "Parity gate must strictly perform GET requests only"
        assert payload is None, "Parity gate must never send inference payloads"
        assert "completions" not in url.lower(), "Parity gate must NEVER call /v1/chat/completions"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen is not None:
            seen.append((method, path))
        resp = client.request(method, path, headers=headers)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json()
        return resp.status_code, {
            "content_type": ctype,
            "text_length": len(resp.text),
            "html_length": len(resp.text),
        }

    return _http


def test_compliance_headers_stdlib_lateral_movement_and_security_scanners() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    this_file = Path(__file__).resolve()
    for path in (GATE_PATH, GATEWAY_PY_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    gate_source = GATE_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")
    gateway_source = GATEWAY_PY_PATH.read_text(encoding="utf-8")

    # 1. Lateral-movement AST audit (zero raw sockets, zero unauthorized URLs, zero SSH/LAN probes)
    assert lateral_guard.audit_source_for_lateral_movement(
        gate_source,
        filename="scripts/noel_cloud_run_parity_gate.py",
    ) == []

    # 2. Pure Python standard library only in scripts/noel_cloud_run_parity_gate.py
    tree = ast.parse(gate_source, filename="scripts/noel_cloud_run_parity_gate.py")
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 3. PR gatekeeper FORBIDDEN_ADDED cleanliness across all new/modified files
    for rel, content in (
        ("scripts/noel_cloud_run_parity_gate.py", gate_source),
        ("tests/test_noel_cloud_run_parity_gate.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit
    scanner = PRSecurityScanner()
    for rel, content in (
        ("scripts/noel_cloud_run_parity_gate.py", gate_source),
        ("src/hivemind/router/gateway.py", gateway_source),
        ("tests/test_noel_cloud_run_parity_gate.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []

    from gwaya.ast_audit import ZeroStubAudit

    audit_res = ZeroStubAudit().audit_python_code(gate_source)
    assert audit_res.is_clean, f"Unexpected GWAYA stub violations: {audit_res.violations}"

    # 5. Output safety guard negative controls
    bad_key_fragment = '{"' + "sco" + 're": 9}'
    fake_hm_key = "hm-" + "k" * 16
    assert "forbidden_key_detected" in parity_gate.scan_output_safety(bad_key_fragment)
    assert "email_detected" in parity_gate.scan_output_safety("ops@example.org")
    assert "unmasked_api_key_detected" in parity_gate.scan_output_safety(fake_hm_key)
    assert "bearer_token_detected" in parity_gate.scan_output_safety("Bearer tok_abcdef123")
    assert "unredacted_home_path_detected" in parity_gate.scan_output_safety("/home/bob/.hivemind")
    assert any(
        item.startswith("non_loopback_ipv4_detected:")
        for item in parity_gate.scan_output_safety("peer 192.168.0.15")
    )
    with pytest.raises(parity_gate.ParityGateError, match="Output safety guard failed"):
        parity_gate._assert_safe_output(bad_key_fragment)


def test_gateway_manifest_sanitization_health_and_script_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unit coverage of gateway.py sanitization helpers
    assert _sanitize_build_sha(None) == "unknown"
    assert _sanitize_build_sha("   ") == "unknown"
    assert _sanitize_build_sha("UNKNOWN") == "unknown"
    assert _sanitize_build_sha("short") == "unknown"
    assert _sanitize_build_sha("3bb86911;id") == "unknown"
    assert _sanitize_build_sha("mysecret1234") == "unknown"
    assert _sanitize_build_sha("AIza" + "B" * 20) == "unknown"
    assert _sanitize_build_sha("  3BB86911ABCDEF  ") == "3bb86911abcdef"
    assert _sanitize_build_sha("noel2026rc1") == "noel2026rc1"

    assert _sanitize_cloud_run_revision(None) is None
    assert _sanitize_cloud_run_revision("  ") is None
    assert _sanitize_cloud_run_revision("-bad-start") is None
    assert _sanitize_cloud_run_revision("rev with spaces") is None
    assert _sanitize_cloud_run_revision("hm-" + "z" * 18) is None
    assert _sanitize_cloud_run_revision("rev-token-leak") is None
    assert _sanitize_cloud_run_revision("hivemind-openrouter-00099-noel") == "hivemind-openrouter-00099-noel"

    direct_manifest = build_onboarding_manifest(
        {"HIVEMIND_BUILD_SHA": "3BB86911", "K_REVISION": "hivemind-openrouter-00099-noel"}
    )
    assert direct_manifest["build_sha"] == "3bb86911"
    assert direct_manifest["cloud_run_revision"] == "hivemind-openrouter-00099-noel"

    # Live FastAPI TestClient verification
    monkeypatch.delenv("HIVEMIND_BUILD_SHA", raising=False)
    monkeypatch.delenv("K_REVISION", raising=False)
    store = RouterStore(tmp_path / "router.db", secret=b"slice-dyn-02-router-secret-32b!!")
    client = TestClient(create_router_app(store))

    # 1. /health and /healthz remain strictly {'status': 'ok'}
    for health_path in ("/health", "/healthz"):
        r_health = client.get(health_path)
        assert r_health.status_code == 200
        assert r_health.json() == {"status": "ok"}

    # 2. /v1/hivemind/onboarding/manifest default (no env vars)
    r_man_default = client.get("/v1/hivemind/onboarding/manifest")
    assert r_man_default.status_code == 200
    assert r_man_default.headers.get("cache-control") == "no-store"
    man = r_man_default.json()
    assert man == {
        "object": "hivemind.onboarding.manifest",
        "campaign_id": "noel-2026-easy-onboard",
        "status": "ok",
        "build_sha": "unknown",
        "cloud_run_revision": None,
        "holiday_routes": sorted(_HOLIDAY_ONBOARDING_ROUTES),
        "downloadable_scripts": sorted(_ALLOWED_ONBOARDING_SCRIPTS),
        "personal_pc_only": True,
        "financial_promise": "0.00 EUR / 0.00 USD",
    }
    assert "noel_post_setup_probe.py" in man["downloadable_scripts"]
    assert "noel_cloud_run_parity_gate.py" in man["downloadable_scripts"]

    # 3. Direct GET /scripts/<name> downloads for both holiday scripts (no git clone needed)
    for script_name in ("noel_post_setup_probe.py", "noel_cloud_run_parity_gate.py"):
        r_dl = client.get(f"/scripts/{script_name}")
        assert r_dl.status_code == 200
        assert "text/x-python" in r_dl.headers.get("content-type", "")
        assert f'filename="{script_name}"' in r_dl.headers.get("content-disposition", "")
        assert "Licensed under the AGPL-3.0" in r_dl.text[:300]

    # 4. Live TestClient full-parity run with HIVEMIND_BUILD_SHA and K_REVISION set
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", SAMPLE_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00108-abc")
    seen_paths: list[tuple[str, str]] = []
    asgi_http = _make_asgi_http(client, seen_paths)

    report = parity_gate.check_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        expected_sha=SAMPLE_SHORT_SHA,
        http_fn=asgi_http,
        lang="en",
    )
    assert report["parity_ok"] is True
    assert report["redeploy_required"] is False
    assert report["cloud_run_lags_main"] is False
    assert report["routes_verified_count"] == 8
    assert report["routes_total_count"] == 8
    assert report["scripts_verified_count"] == 5
    assert report["scripts_total_count"] == 5
    assert report["live_build_sha"] == SAMPLE_FULL_SHA
    assert report["expected_build_sha"] == SAMPLE_SHORT_SHA
    assert report["sha_match"] is True
    assert report["sha_drift"] is False
    assert report["cloud_run_revision"] == "hivemind-openrouter-00108-abc"
    assert report["zero_inference_verified"] is True
    assert report["diagnostics"] == []
    assert len(report["receipt_sha256"]) == 64

    # Verify deterministic SHA-256 receipt
    expected_digest = hashlib.sha256(
        json.dumps(
            {k: v for k, v in report.items() if k != "receipt_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert report["receipt_sha256"] == expected_digest

    # Confirm exact 13 zero-inference GET endpoints hit (8 routes + 5 scripts, zero completions)
    assert [p for _, p in seen_paths] == [
        "/health",
        "/v1/models",
        "/v1/hivemind/onboarding/manifest",
        "/noel",
        "/quick",
        "/raid-noel",
        "/arena-noel",
        "/v1/hivemind/onboarding/feedback/summary",
        "/scripts/hivemind_quickstart.py",
        "/scripts/hivemind_doctor.py",
        "/scripts/noel_post_setup_probe.py",
        "/scripts/noel_cloud_run_parity_gate.py",
        "/scripts/join_hivemind.ps1",
    ]


def test_stale_cloud_run_404_drift_manifest_gaps_and_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Stale Cloud Run revision (only /health, /v1/models, and older scripts exist; holiday routes 404)
    def _stale_http(method: str, url: str) -> tuple[int, Any]:
        assert method == "GET"
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/auto"}]}
        if any(
            url.endswith(s)
            for s in (
                "/scripts/hivemind_quickstart.py",
                "/scripts/hivemind_doctor.py",
                "/scripts/join_hivemind.ps1",
            )
        ):
            return 200, "#!/usr/bin/env python3\n"
        return 404, {"detail": "Not Found"}

    rc_stale_en = parity_gate.main(["--lang", "en", "--json"], http_fn=_stale_http)
    assert rc_stale_en == 2
    stale_en = json.loads(capsys.readouterr().out)
    assert stale_en["parity_ok"] is False
    assert stale_en["redeploy_required"] is True
    assert stale_en["cloud_run_lags_main"] is True
    assert stale_en["routes_verified_count"] == 2
    assert stale_en["scripts_verified_count"] == 3
    assert "/v1/hivemind/onboarding/manifest" in stale_en["missing_routes"]
    assert "/noel" in stale_en["missing_routes"]
    assert stale_en["missing_scripts"] == [
        "noel_post_setup_probe.py",
        "noel_cloud_run_parity_gate.py",
    ]
    assert any("lags behind `main`" in d for d in stale_en["diagnostics"])

    # Same stale Cloud Run in French Markdown mode
    rc_stale_fr = parity_gate.main(["--lang", "fr", "--markdown"], http_fn=_stale_http)
    assert rc_stale_fr == 2
    md_fr = capsys.readouterr().out
    assert "# Gate de Parité Cloud Run Noël & Plan FinOps HiveMind" in md_fr
    assert "en retard sur `main`" in md_fr
    assert "| `noel_cloud_run_parity_gate.py` | `404` | **FAIL** |" in md_fr

    # 2. Build SHA mismatch & --check-git-head drift against live TestClient
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", "11111111deadbeef")
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00050-old")
    store = RouterStore(tmp_path / "drift.db", secret=b"slice-dyn-02-drift-secret-32b!!!")
    client = TestClient(create_router_app(store))
    asgi_http = _make_asgi_http(client)

    rc_drift = parity_gate.main(
        ["--gateway", "http://127.0.0.1:8000", "--expected-sha", SAMPLE_SHORT_SHA, "--json"],
        http_fn=asgi_http,
    )
    assert rc_drift == 2
    drift_doc = json.loads(capsys.readouterr().out)
    assert drift_doc["parity_ok"] is False
    assert drift_doc["redeploy_required"] is True
    assert drift_doc["sha_match"] is False
    assert drift_doc["sha_drift"] is True
    assert drift_doc["routes_verified_count"] == 8
    assert drift_doc["scripts_verified_count"] == 5
    assert any("SHA drift detected" in d for d in drift_doc["diagnostics"])

    # --check-git-head matching vs mismatching vs failing git runner (FR & EN)
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", SAMPLE_SHORT_SHA)
    rc_git_ok = parity_gate.main(
        ["--gateway", "http://127.0.0.1:8000", "--check-git-head", "--json"],
        http_fn=asgi_http,
        runner=lambda argv, timeout_s: (0, SAMPLE_FULL_SHA + "\n"),
    )
    assert rc_git_ok == 0
    git_ok_doc = json.loads(capsys.readouterr().out)
    assert git_ok_doc["parity_ok"] is True
    assert git_ok_doc["expected_sha_source"] == "git_head"
    assert git_ok_doc["sha_match"] is True

    rc_git_drift_fr = parity_gate.main(
        ["--gateway", "http://127.0.0.1:8000", "--check-git-head", "--lang", "fr", "--json"],
        http_fn=asgi_http,
        runner=lambda argv, timeout_s: (0, "abcdef1234567890\n"),
    )
    assert rc_git_drift_fr == 2
    git_drift_fr = json.loads(capsys.readouterr().out)
    assert git_drift_fr["parity_ok"] is False
    assert any("Dérive de SHA Cloud Run" in d for d in git_drift_fr["diagnostics"])

    rc_git_err = parity_gate.main(
        ["--gateway", "http://127.0.0.1:8000", "--check-git-head", "--json"],
        http_fn=asgi_http,
        runner=lambda argv, timeout_s: (128, ""),
    )
    assert rc_git_err == 2
    git_err_doc = json.loads(capsys.readouterr().out)
    assert git_err_doc["parity_ok"] is False
    assert git_err_doc["sha_drift"] is True

    def _raise_oserror(argv: Any, timeout_s: float) -> tuple[int, str]:
        raise FileNotFoundError("git missing")

    rep_git_exc_fr = parity_gate.check_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        check_git_head=True,
        http_fn=asgi_http,
        runner=_raise_oserror,
        lang="fr",
    )
    assert rep_git_exc_fr["parity_ok"] is False
    assert any("FileNotFoundError" in d for d in rep_git_exc_fr["diagnostics"])

    # 3. Malformed manifest, HTTP 5xx, network error, empty HTML/script, core 404, HTTP 429
    def _broken_payloads_http(method: str, url: str) -> tuple[int, Any]:
        if url.endswith("/health"):
            return 200, {"status": "degraded"}
        if url.endswith("/v1/models"):
            return 200, {"data": "not-a-list"}
        if url.endswith("/v1/hivemind/onboarding/manifest"):
            return 200, {
                "object": "wrong.object",
                "campaign_id": "noel-2026-easy-onboard",
                "status": "ok",
                "build_sha": "unknown",
                "holiday_routes": ["/noel"],
                "downloadable_scripts": ["hivemind_quickstart.py"],
                "personal_pc_only": True,
                "financial_promise": "0.00 EUR / 0.00 USD",
            }
        if url.endswith("/noel"):
            return 200, {"html_length": 0}
        if url.endswith("/quick"):
            return 503, {"error": "unavailable"}
        if url.endswith("/raid-noel"):
            return 429, {"error": "rate_limited"}
        if url.endswith("/arena-noel"):
            raise urllib.error.URLError("tls reset")
        if url.endswith("/scripts/hivemind_quickstart.py"):
            return 200, {"text_length": 0}
        return 404, {}

    for lang_choice in ("en", "fr"):
        broken_rep = parity_gate.check_cloud_run_holiday_parity(
            http_fn=_broken_payloads_http,
            lang=lang_choice,
        )
        assert broken_rep["parity_ok"] is False
        assert broken_rep["redeploy_required"] is True
        assert broken_rep["routes_verified_count"] == 0
        assert broken_rep["scripts_verified_count"] == 0
        assert "/arena-noel" in broken_rep["missing_manifest_routes"]
        assert "noel_cloud_run_parity_gate.py" in broken_rep["missing_manifest_scripts"]

    # Non-list holiday_routes & downloadable_scripts + core 404 branch
    def _nonlist_manifest_http(method: str, url: str) -> tuple[int, Any]:
        if url.endswith("/v1/hivemind/onboarding/manifest"):
            return 200, {
                "object": "hivemind.onboarding.manifest",
                "campaign_id": "noel-2026-easy-onboard",
                "status": "ok",
                "holiday_routes": None,
                "downloadable_scripts": None,
                "personal_pc_only": True,
                "financial_promise": "0.00 EUR / 0.00 USD",
            }
        return 404, "missing"

    for lang_choice in ("en", "fr"):
        nonlist_rep = parity_gate.check_cloud_run_holiday_parity(
            http_fn=_nonlist_manifest_http,
            lang=lang_choice,
        )
        assert nonlist_rep["parity_ok"] is False
        assert len(nonlist_rep["missing_manifest_routes"]) == len(
            parity_gate.EXPECTED_MANIFEST_HOLIDAY_ROUTES
        )
        assert len(nonlist_rep["missing_manifest_scripts"]) == len(
            parity_gate.EXPECTED_MANIFEST_SCRIPTS
        )


def test_dry_run_finops_redeploy_plan_and_combined_cli_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Real deploy/gcp_openrouter/deploy.sh passes all 5 FinOps invariants with zero gcloud calls
    plan = parity_gate.build_cloud_run_redeploy_plan(
        target_sha=SAMPLE_SHORT_SHA,
        lang="en",
    )
    assert plan["finops_ok"] is True
    assert plan["plan_ready"] is True
    assert plan["dry_run_only"] is True
    assert plan["gcloud_executed"] is False
    assert plan["cloud_state_mutated"] is False
    assert plan["deploy_script"] == "deploy/gcp_openrouter/deploy.sh"
    assert plan["invariants_verified_count"] == 5
    assert plan["invariants_total_count"] == 5
    assert plan["failed_invariants"] == []
    assert f'HIVEMIND_BUILD_SHA="{SAMPLE_SHORT_SHA}"' in plan["redeploy_command"]
    assert f"--update-env-vars \"HIVEMIND_BUILD_SHA={SAMPLE_SHORT_SHA}\"" in plan["redeploy_command"]
    assert "--min-instances 0 --max-instances 1" in plan["redeploy_command"]
    assert f"--expected-sha {SAMPLE_SHORT_SHA}" in plan["verify_command"]
    assert "0.00 EUR / 0.00 USD" in plan["compliance"]["notice_bilingual"]
    assert "0,00 EUR / 0,00 USD" in plan["compliance"]["notice_bilingual"]

    # CLI standalone --print-redeploy-plan (JSON & FR Markdown) makes ZERO HTTP calls
    def _forbid_http(*_args: Any) -> tuple[int, Any]:
        raise AssertionError("--print-redeploy-plan without --verify-live must not perform HTTP calls")

    rc_plan_json = parity_gate.main(
        ["--print-redeploy-plan", "--json"],
        http_fn=_forbid_http,
    )
    assert rc_plan_json == 0
    plan_cli = json.loads(capsys.readouterr().out)
    assert plan_cli["finops_ok"] is True
    assert "--check-git-head" in plan_cli["verify_command"]

    rc_plan_md_fr = parity_gate.main(
        ["--print-redeploy-plan", "--check-git-head", "--lang", "fr", "--markdown"],
        http_fn=_forbid_http,
        runner=lambda argv, timeout_s: (0, SAMPLE_SHORT_SHA + "\n"),
    )
    assert rc_plan_md_fr == 0
    md_plan_fr = capsys.readouterr().out
    assert "## Vérification FinOps & Commande Dry-Run" in md_plan_fr
    assert f'HIVEMIND_BUILD_SHA="{SAMPLE_SHORT_SHA}"' in md_plan_fr

    # 2. Broken deploy.sh FinOps invariant failures + missing file + git failure
    bad_deploy = tmp_path / "broken_deploy.sh"
    bad_deploy.write_text(
        "#!/usr/bin/env bash\n"
        "# --min-instances 0 --max-instances 1 --no-cpu-throttling\n"
        "gcloud run deploy hivemind-openrouter --min-instances 2 --max-instances 4 --cpu-throttling\n",
        encoding="utf-8",
    )
    rc_bad = parity_gate.main(
        ["--print-redeploy-plan", "--deploy-script", str(bad_deploy), "--lang", "fr", "--json"],
        http_fn=_forbid_http,
    )
    assert rc_bad == 2
    bad_plan = json.loads(capsys.readouterr().out)
    assert bad_plan["finops_ok"] is False
    assert bad_plan["plan_ready"] is False
    assert set(bad_plan["failed_invariants"]) == {
        "min_instances_zero",
        "max_instances_one",
        "no_cpu_throttling",
        "litestream_replica_url",
        "secret_manager_router_secret",
    }

    missing_plan = parity_gate.build_cloud_run_redeploy_plan(
        deploy_script_path=tmp_path / "does_not_exist.sh",
        check_git_head=True,
        runner=lambda argv, timeout_s: (1, ""),
        lang="en",
    )
    assert missing_plan["deploy_script_present"] is False
    assert missing_plan["finops_ok"] is False
    assert missing_plan["plan_ready"] is False

    # 3. Combined --print-redeploy-plan --verify-live against live TestClient
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", SAMPLE_SHORT_SHA)
    store = RouterStore(tmp_path / "combo.db", secret=b"slice-dyn-02-combo-secret-32b!!!")
    client = TestClient(create_router_app(store))
    rc_combo = parity_gate.main(
        [
            "--gateway",
            "http://127.0.0.1:8000",
            "--expected-sha",
            SAMPLE_SHORT_SHA,
            "--print-redeploy-plan",
            "--verify-live",
            "--markdown",
        ],
        http_fn=_make_asgi_http(client),
    )
    assert rc_combo == 0
    combo_md = capsys.readouterr().out
    assert "# HiveMind Holiday Cloud Run Parity Gate & FinOps Plan" in combo_md
    assert "## FinOps Invariants & Dry-Run Redeploy Command" in combo_md
    assert "**SHA-256 Receipt**:" in combo_md


def test_url_rejection_zero_inference_guards_and_loopback_http_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Accepted gateway URLs (Cloud Run https + loopback IPv4, localhost, IPv6 ::1)
    assert (
        parity_gate.validate_gateway_url("https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/")
        == "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
    )
    assert parity_gate.validate_gateway_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
    assert parity_gate.validate_gateway_url("https://localhost:8443") == "https://localhost:8443"
    assert parity_gate.validate_gateway_url("http://[::1]:9000") == "http://[::1]:9000"

    # 2. Rejected gateway URLs, languages, SHAs, cloud identifiers, and zero-inference violations
    bad_urls: list[Any] = [
        "",
        "   ",
        True,
        None,
        "https://hivemind-openrouter.run.app/v1",
        "https://hivemind-openrouter.run.app?x=1",
        "https://hivemind-openrouter.run.app#frag",
        "https://user:pass@hivemind-openrouter.run.app",
        "https://hivemind-openrouter.run.app\n",
        "https://hivemind openrouter.run.app",
        "http://hivemind-openrouter.run.app",
        "ftp://127.0.0.1:8080",
        "http://127.0.0.1:0",
        "http://127.0.0.1:notaport",
        "http://192.168.1.42:8080",
        "https://evil.example.com",
        "https://run.app",
        "https://.run.app",
        "https://bad..run.app",
        "https://://",
    ]
    for candidate in bad_urls:
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate.validate_gateway_url(candidate)  # type: ignore[arg-type]

    assert parity_gate.is_allowed_gateway_host("") is False
    assert parity_gate.normalize_lang("en-GB") == "en"
    assert parity_gate.normalize_lang("fr_CA") == "fr"
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.normalize_lang("de")

    assert parity_gate.sanitize_build_sha(12345) == "unknown"
    assert parity_gate.sanitize_cloud_run_revision(12345) is None
    assert parity_gate.sanitize_cloud_run_revision("hm-" + "z" * 16) is None
    assert parity_gate._invoke_http_callable(lambda m, u: (200, [1, 2]), "GET", "http://127.0.0.1:8080/health") == (
        200,
        {"data": [1, 2]},
    )
    assert parity_gate.shas_match("noel2026a", "noel2026b") is False
    assert parity_gate.shas_match("unknown", SAMPLE_SHORT_SHA) is False
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.validate_expected_sha("unknown")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_cloud_identifier("bad;rm", "service")

    # Home path redaction variants
    custom_home = tmp_path / "custom_user"
    assert parity_gate.redact_home_path(custom_home / ".hivemind" / "x", home=custom_home) == "~/.hivemind/x"
    assert (
        parity_gate.redact_home_path(r"C:\Users\GamerAlice\Desktop\deploy.sh", home=custom_home)
        == "~/Desktop/deploy.sh"
    )

    # Zero-inference & gcloud execution guards
    with pytest.raises(parity_gate.ParityGateError, match="HTTP method 'POST'"):
        parity_gate._assert_zero_inference_request("POST", f"{parity_gate.DEFAULT_GATEWAY}/health")
    with pytest.raises(parity_gate.ParityGateError, match="Zero-inference invariant violated"):
        parity_gate._assert_zero_inference_request(
            "GET",
            f"{parity_gate.DEFAULT_GATEWAY}/v1/chat/completions",
        )
    with pytest.raises(parity_gate.ParityGateError, match="payloads"):
        parity_gate.default_http_json("GET", "http://127.0.0.1:8080/health", payload={"model": "hivemind/coder"})
    with pytest.raises(parity_gate.ParityGateError, match="empty"):
        parity_gate.default_subprocess_runner([])
    with pytest.raises(parity_gate.ParityGateError, match="gcloud"):
        parity_gate.default_subprocess_runner(["gcloud", "run", "deploy"])

    # CLI error handling returns exit code 2
    assert parity_gate.main(["--gateway", "https://evil.example.com"]) == 2
    assert "Untrusted gateway host" in capsys.readouterr().err

    # 3. Live loopback HTTP server round-trip testing default_http_json & subprocess CLI
    manifest_payload = build_onboarding_manifest(
        {"HIVEMIND_BUILD_SHA": SAMPLE_SHORT_SHA, "K_REVISION": "hivemind-openrouter-00110-loop"}
    )

    class _ParityLoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send_json(200, {"data": [{"id": "hivemind/auto"}]})
            elif self.path == "/v1/hivemind/onboarding/manifest":
                self._send_json(200, manifest_payload)
            elif self.path == "/v1/hivemind/onboarding/feedback/summary":
                self._send_json(200, {"total_responses": 4, "mean_ease_rating": 4.9})
            elif self.path in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<!DOCTYPE html><title>HiveMind Noel</title>")
            elif self.path.startswith("/scripts/"):
                self.send_response(200)
                self.send_header("Content-Type", "text/x-python; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind.\n")
            elif self.path == "/empty":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/json-list":
                self._send_json(200, [1, 2, 3])
            elif self.path == "/err-json":
                self._send_json(404, {"detail": "not_found"})
            else:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"upstream error")

        def _send_json(self, status: int, data: Any) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return None

    server = http.server.HTTPServer(("127.0.0.1", 0), _ParityLoopbackHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Poison proxy env vars to prove ProxyHandler({}) bypasses corporate HTTP proxies
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        base = f"http://127.0.0.1:{port}"

        assert parity_gate.default_http_json("GET", f"{base}/noel")[1]["html_length"] > 0
        assert parity_gate.default_http_json("GET", f"{base}/empty", headers={"X-Test": "1"})[0] == 204
        assert parity_gate.default_http_json("GET", f"{base}/json-list")[1] == {"data": [1, 2, 3]}
        assert parity_gate.default_http_json("GET", f"{base}/err-json")[0] == 404
        assert parity_gate.default_http_json("GET", f"{base}/err-plain")[0] == 502

        proc = subprocess.run(
            [
                sys.executable,
                str(GATE_PATH),
                "--gateway",
                base,
                "--expected-sha",
                SAMPLE_SHORT_SHA,
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
        )
        assert proc.returncode == 0, proc.stderr
        cli_out = json.loads(proc.stdout)
        assert cli_out["parity_ok"] is True
        assert cli_out["routes_verified_count"] == 8
        assert cli_out["scripts_verified_count"] == 5
    finally:
        server.shutdown()
        server.server_close()

    c_down, b_down = parity_gate.default_http_json("GET", "http://127.0.0.1:1/health")
    assert c_down == 0 and "network_error" in b_down

    rc_sub, out_sub = parity_gate.default_subprocess_runner(
        [sys.executable, "-c", "print('parity-sub-ok')"],
        3.0,
    )
    assert rc_sub == 0 and out_sub.strip() == "parity-sub-ok"

    # Confirm static figure ratchet regex matches zero tokens in rendered Markdown
    fig_re = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
    sample_md = parity_gate.render_markdown_report(
        parity_gate.build_cloud_run_redeploy_plan(target_sha=SAMPLE_SHORT_SHA)
    )
    assert fig_re.findall(sample_md) == []
