# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unified 5-gate Noël-at-Scale Go/No-Go readiness runner & scorecard tests (`SLICE-SCALE-05`)."""

from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import urllib.error
from pathlib import Path
from typing import Any

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT_PATH = REPO_ROOT / "scripts" / "noel_scale_readiness_gate.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(GATE_SCRIPT_PATH)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import create_router_app  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_spec = importlib.util.spec_from_file_location("noel_scale_readiness_gate", GATE_SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
readiness_gate = importlib.util.module_from_spec(_spec)
sys.modules["noel_scale_readiness_gate"] = readiness_gate
_spec.loader.exec_module(readiness_gate)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_scale05",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_scale05",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_scale05",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin"}
SAMPLE_FULL_SHA = "715ab6830123456789abcdef0123456789abcdef"
SAMPLE_SHORT_SHA = "715ab6830123"
ROUTER_SECRET = b"noel-scale-05-router-secret-32b!"
MODULES = readiness_gate.load_scale_readiness_modules()


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


def _verify_sha256_receipt(report: dict[str, Any]) -> None:
    expected = hashlib.sha256(
        json.dumps(
            {k: v for k, v in report.items() if k != "receipt_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert len(report["receipt_sha256"]) == 64
    assert report["receipt_sha256"] == expected
    assert readiness_gate.verify_receipt_sha256(report) is True
    assert ("sco" + "re") not in _collect_all_keys(report)


def _make_live_asgi_http(
    client: TestClient,
    *,
    seen_calls: list[tuple[str, str]] | None = None,
    drop_paths_404: set[str] | None = None,
    static_nonce_override: str | None = None,
    inject_dom_sink: bool = False,
) -> readiness_gate.LiveHttpFn:
    lock = threading.Lock()
    dropped = drop_paths_404 or set()

    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
        assert method == "GET", "Live readiness gate must strictly perform read-only GET requests"
        assert payload is None, "Live readiness gate must never send request payloads"
        assert "completions" not in url.lower(), "Live readiness gate must never call /v1/chat/completions"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen_calls is not None:
            with lock:
                seen_calls.append((method, path))
        if path in dropped:
            return 404, {"error": "simulated_404_drift"}, {"content-type": "application/json"}
        resp = client.request(method, path, headers=headers)
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        ctype = resp_headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json(), resp_headers
        text = resp.text
        if static_nonce_override is not None and path in readiness_gate.LIVE_HTML_PATHS:
            resp_headers["content-security-policy"] = f"default-src 'none'; style-src 'nonce-{static_nonce_override}'"
            text = (
                f'<!DOCTYPE html><html><head><style nonce="{static_nonce_override}">body{{}}</style>'
                f"</head><body>ok</body></html>"
            )
        if inject_dom_sink and path == "/noel":
            sink_word = "inner" + "HTML"
            text = text + f"<!-- {sink_word} -->"
        return resp.status_code, text, resp_headers

    return _http


def test_compliance_headers_stdlib_lateral_movement_gatekeeper_and_security_scanners() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    this_file = Path(__file__).resolve()
    for path in (GATE_SCRIPT_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    script_source = GATE_SCRIPT_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Lateral-movement AST audit across script & test file + negative control
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source,
            filename="scripts/noel_scale_readiness_gate.py",
        )
        == []
    )
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_source,
            filename="tests/test_noel_scale_readiness_gate.py",
        )
        == []
    )
    bad_lan_ip = "192.168" + ".1.77"
    poisoned = script_source + f'\n_BAD_EGRESS = "http://{bad_lan_ip}:8000/noel"\n'
    assert lateral_guard.audit_source_for_lateral_movement(poisoned, "poisoned.py") != []

    # 2. Pure Python 3.10-3.13 standard library only
    tree = ast.parse(script_source, filename="scripts/noel_scale_readiness_gate.py")
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 3. PR gatekeeper FORBIDDEN_ADDED cleanliness across both files
    for rel, content in (
        ("scripts/noel_scale_readiness_gate.py", script_source),
        ("tests/test_noel_scale_readiness_gate.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit + static-figure ratchet
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_scale_readiness_gate.py", script_source),
        ("tests/test_noel_scale_readiness_gate.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1065,
        pr_title="feat(scale): unified 5-gate Noel-at-Scale Go/No-Go readiness runner and scorecard",
        pr_body="Implements SLICE-SCALE-05 unified 5-gate Noel-at-Scale Go/No-Go readiness runner and scorecard.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(scale): unified 5-gate Noel-at-Scale Go/No-Go readiness runner and scorecard"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    stub_audit = ZeroStubAudit().audit_python_code(script_source)
    assert stub_audit.is_clean, f"Unexpected GWAYA stub violations: {stub_audit.violations}"

    assert static_numbers_guard.FIGURE.findall(script_source) == []
    assert static_numbers_guard.FIGURE.findall(test_source) == []


def test_offline_5_gate_all_green_go_scorecard_and_sha256_receipts() -> None:
    report = readiness_gate.run_noel_scale_readiness_gate(
        report_date="2026-12-19",
        generated_at_utc="2026-12-19T08:00:00Z",
        lang="bilingual",
        environ=CLEAN_ENV,
        modules=MODULES,
    )
    _verify_sha256_receipt(report)

    assert report["schema"] == "hivemind.noel_scale_readiness_gate/1"
    assert report["slice_id"] == "SLICE-SCALE-05"
    assert report["mode"] == "offline_rehearsal"
    assert report["live_url"] is None
    assert report["live_check_enabled"] is False
    assert report["live_check_ok"] is None
    assert report["drift_404_detected"] is False
    assert report["compliance_ok"] is True
    assert report["go_for_launch"] is True
    assert report["verdict"] == "GO"
    assert report["exit_code"] == 0
    assert report["gates_passed_count"] == 5
    assert report["gates_total_count"] == 5
    assert report["gates_failed_count"] == 0
    assert report["passed_gate_ids"] == ["G1", "G2", "G3", "G4", "G5"]
    assert report["failed_gate_ids"] == []
    assert report["gate_verdicts"] == {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True}
    assert report["within_10s_slo"] is True
    assert 0.0 < report["elapsed_s"] < 10.0
    assert report["zero_inference_verified"] is True
    assert report["inference_calls_count"] == 0
    assert report["all_sub_receipts_valid"] is True
    assert report["sub_receipts_verified_count"] == 5
    assert report["diagnostics"] == []

    by_id = {g["gate_id"]: g for g in report["gates"]}
    assert tuple(by_id.keys()) == ("G1", "G2", "G3", "G4", "G5")

    # G1 assertions
    g1_m = by_id["G1"]["metrics"]
    assert g1_m["routes_verified_count"] == 8 and g1_m["routes_total_count"] == 8
    assert g1_m["scripts_verified_count"] == 17 and g1_m["scripts_total_count"] == 17
    assert g1_m["disk_scripts_verified_count"] == 17
    assert g1_m["finops_ok"] is True
    assert g1_m["invariants_verified_count"] == 5

    # G2 assertions
    g2_m = by_id["G2"]["metrics"]
    assert g2_m["participants_total"] == 60
    assert g2_m["requests_total"] == 360
    assert g2_m["error_count"] == 0
    assert g2_m["unique_csp_nonces_count"] == 120
    assert g2_m["unique_csp_nonces_ratio"] == 1.0
    assert g2_m["registrations_conserved"] is True
    assert g2_m["arena_votes_conserved"] is True
    assert g2_m["ces_responses_conserved"] is True

    # G3 assertions
    g3_m = by_id["G3"]["metrics"]
    assert g3_m["cohort_size"] == 50
    assert g3_m["personal_ready_count"] == 50
    assert g3_m["personal_ready_ratio"] == 1.0
    assert g3_m["corporate_cohort_size"] == 10
    assert g3_m["corporate_blocked_count"] == 10
    assert g3_m["corporate_blocked_ratio"] == 1.0
    assert g3_m["corporate_file_writes_count"] == 0
    assert g3_m["corporate_network_calls_count"] == 0
    assert g3_m["zero_key_leaks_verified"] is True
    assert g3_m["archetypes_verified_count"] == 5

    # G4 assertions
    g4_m = by_id["G4"]["metrics"]
    assert g4_m["yield_slo_ok"] is True
    assert 0.0 <= g4_m["yield_latency_ms"] < 1500.0
    assert g4_m["retained_worker_vram_mib"] == 0
    assert g4_m["thermal_ok"] is True
    assert g4_m["observed_temp_c"] <= 70.0
    assert g4_m["honest_nogpu_verified"] is True
    assert g4_m["negative_controls_verified"] is True

    # G5 assertions
    g5_m = by_id["G5"]["metrics"]
    assert g5_m["autoconfig_ready"] is True
    assert g5_m["within_60s_slo"] is True
    assert g5_m["probe_ready"] is True
    assert g5_m["uninstall_clean"] is True
    assert g5_m["funnel_status"] == "healthy"
    assert g5_m["mean_ease_rating"] >= 4.2
    assert g5_m["zero_game_lag_ratio"] == 1.0
    assert g5_m["pii_fields_dropped_count"] >= 5
    assert g5_m["zero_pii_verified"] is True

    # Verify bilingual, EN, FR Markdown & CLI reports contain zero static-figure violations or leaks
    md_bi = readiness_gate.render_markdown_report(report, lang="bilingual")
    md_en = readiness_gate.render_markdown_report(report, lang="en")
    md_fr = readiness_gate.render_markdown_report(report, lang="fr")
    cli_en = readiness_gate.render_cli_report(report, lang="en")
    cli_fr = readiness_gate.render_cli_report(report, lang="fr")
    assert "---" in md_bi and "GO POUR LANCEMENT NOËL" in md_bi and "GO FOR LAUNCH" in md_bi
    assert "GO FOR LAUNCH" in md_en and report["receipt_sha256"] in md_en
    assert "GO POUR LANCEMENT NOËL" in md_fr and "0,00 EUR / 0,00 USD" in md_fr
    assert "[GO]" in cli_en and "5/5 gates" in cli_en
    assert "[GO]" in cli_fr and "durée=" in cli_fr
    for rendered in (md_bi, md_en, md_fr, cli_en, cli_fr):
        assert readiness_gate.scan_output_safety(rendered) == []
        assert static_numbers_guard.FIGURE.findall(rendered) == []


def test_single_gate_nogo_failure_modes_g1_through_g5_and_slo_timeout(tmp_path: Path) -> None:
    fast_kwargs: dict[str, Any] = {
        "burst_devs": 4,
        "burst_gamers": 4,
        "burst_workers": 8,
        "cohort_size": 5,
        "corporate_controls": 5,
        "cohort_workers": 5,
        "environ": CLEAN_ENV,
        "modules": MODULES,
    }

    # 1. Single-gate G1 No-Go (broken deploy.sh FinOps invariant --max-instances 3)
    bad_deploy = tmp_path / "bad_deploy.sh"
    bad_deploy.write_text(
        "#!/usr/bin/env bash\ngcloud run deploy hivemind-openrouter --min-instances 0 --max-instances 3\n",
        encoding="utf-8",
    )
    r_g1 = readiness_gate.run_noel_scale_readiness_gate(
        **fast_kwargs,
        deploy_script_path=bad_deploy,
        lang="fr",
    )
    _verify_sha256_receipt(r_g1)
    assert r_g1["go_for_launch"] is False
    assert r_g1["verdict"] == "NO_GO"
    assert r_g1["exit_code"] == 2
    assert r_g1["gates_passed_count"] == 4
    assert r_g1["gates_total_count"] == 5
    assert r_g1["failed_gate_ids"] == ["G1"]
    assert any("[G1]" in d for d in r_g1["diagnostics"])

    # Also test G1 disk script missing + AST syntax error in isolated scripts_dir
    broken_scripts = tmp_path / "broken_scripts"
    broken_scripts.mkdir(parents=True, exist_ok=True)
    for idx, s_name in enumerate(MODULES.parity_gate.REQUIRED_SCRIPTS):
        if idx == 0:
            continue  # missing script
        target = broken_scripts / s_name
        if s_name == "hivemind_doctor.py":
            target.write_text("def broken_syntax(:\n", encoding="utf-8")
        else:
            target.write_bytes((REPO_ROOT / "scripts" / s_name).read_bytes())
    sim_g1 = readiness_gate.LocalScaleGatewaySimulator(
        scripts_dir=broken_scripts,
        allowed_scripts=MODULES.parity_gate.REQUIRED_SCRIPTS,
        holiday_routes=MODULES.parity_gate.EXPECTED_MANIFEST_HOLIDAY_ROUTES,
    )
    g1_direct = readiness_gate.run_gate_g1_parity(
        modules=MODULES,
        simulator=sim_g1,
        scripts_dir=broken_scripts,
        lang="en",
    )
    assert g1_direct["passed"] is False
    assert any("missing or empty" in d for d in g1_direct["diagnostics"])
    assert any("AST syntax error" in d for d in g1_direct["diagnostics"])

    # 2. Single-gate G2 No-Go (static CSP nonce reuse across concurrent HTML renders)
    r_g2 = readiness_gate.run_noel_scale_readiness_gate(
        **fast_kwargs,
        static_csp_nonce="StaticNonceViolation9999",
    )
    _verify_sha256_receipt(r_g2)
    assert r_g2["go_for_launch"] is False
    assert r_g2["gates_passed_count"] == 4
    assert r_g2["failed_gate_ids"] == ["G2"]
    assert any("[G2]" in d for d in r_g2["diagnostics"])

    # 3. Single-gate G3 No-Go (injected gamer thermal spike in multi-OS cohort)
    r_g3 = readiness_gate.run_noel_scale_readiness_gate(
        **fast_kwargs,
        inject_g3_thermal_failures=1,
    )
    _verify_sha256_receipt(r_g3)
    assert r_g3["go_for_launch"] is False
    assert r_g3["gates_passed_count"] == 4
    assert r_g3["failed_gate_ids"] == ["G3"]
    assert any("[G3]" in d for d in r_g3["diagnostics"])

    # 4. Single-gate G4 No-Go (GPU yield_ms >= 1500.0 budget breach)
    r_g4 = readiness_gate.run_noel_scale_readiness_gate(
        **fast_kwargs,
        g4_worker_lifecycle_fn=lambda **_kw: {
            "worker_started": True,
            "worker_stopped": False,
            "clean_release": False,
            "retained_worker_vram_mib": 512,
            "yield_latency_ms": 1820.0,
            "method": "injected_slow_yield",
        },
    )
    _verify_sha256_receipt(r_g4)
    assert r_g4["go_for_launch"] is False
    assert r_g4["gates_passed_count"] == 4
    assert r_g4["failed_gate_ids"] == ["G4"]
    assert any("[G4]" in d for d in r_g4["diagnostics"])

    # 5. Single-gate G5 No-Go (low Dev CES rating < 4.2 + slow >60s autoconfig clock)
    slow_auto_ticks = iter([10.0, 85.0, 86.0, 87.0, 88.0, 89.0])
    r_g5 = readiness_gate.run_noel_scale_readiness_gate(
        **fast_kwargs,
        g5_dev_ease_rating=2,
        g5_autoconfig_clock_fn=lambda: next(slow_auto_ticks, 90.0),
    )
    _verify_sha256_receipt(r_g5)
    assert r_g5["go_for_launch"] is False
    assert r_g5["gates_passed_count"] == 4
    assert r_g5["failed_gate_ids"] == ["G5"]
    assert any("[G5]" in d for d in r_g5["diagnostics"])

    # 6. Wall-clock 10s SLO budget breach & fail-closed gate exception handling
    wall_ticks = iter([100.0, 114.5])

    def _make_noop_gate(gid: str) -> readiness_gate.GateCallable:
        def _runner() -> dict[str, Any]:
            return {
                "gate_id": gid,
                "slug": f"stub_{gid.lower()}",
                "title_en": gid,
                "title_fr": gid,
                "passed": True,
                "status": "GO",
                "elapsed_s": 0.01,
                "zero_inference_verified": True,
                "sub_receipt_sha256": "a" * 64,
                "sub_receipt_valid": True,
                "summary_en": "ok",
                "summary_fr": "ok",
                "metrics": {},
                "diagnostics": [],
            }

        return _runner

    r_slo = readiness_gate.run_noel_scale_readiness_gate(
        environ=CLEAN_ENV,
        modules=MODULES,
        clock_fn=lambda: next(wall_ticks, 115.0),
        gate_overrides={gid: _make_noop_gate(gid) for gid in readiness_gate.GATE_IDS},
    )
    assert r_slo["gates_passed_count"] == 5
    assert r_slo["within_10s_slo"] is False
    assert r_slo["go_for_launch"] is False
    assert any("[SLO]" in d for d in r_slo["diagnostics"])

    def _boom_gate() -> dict[str, Any]:
        raise RuntimeError("simulated unexpected crash for hm-secretkey999999")

    r_crash = readiness_gate.run_noel_scale_readiness_gate(
        environ=CLEAN_ENV,
        modules=MODULES,
        lang="fr",
        gate_overrides={
            **{gid: _make_noop_gate(gid) for gid in ("G1", "G2", "G3", "G5")},
            "G4": _boom_gate,
        },
    )
    _verify_sha256_receipt(r_crash)
    assert r_crash["go_for_launch"] is False
    assert r_crash["failed_gate_ids"] == ["G4"]
    assert "hm-***redacted***" in r_crash["diagnostics"][0]


def test_live_url_all_green_and_404_drift_detection_against_real_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", SAMPLE_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00105-scale05")

    store = RouterStore(tmp_path / "scale05_live_router.db", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1)

    fast_overrides = {
        gid: (
            lambda g=gid: {
                "gate_id": g,
                "slug": f"fast_{g.lower()}",
                "title_en": g,
                "title_fr": g,
                "passed": True,
                "status": "GO",
                "elapsed_s": 0.01,
                "zero_inference_verified": True,
                "sub_receipt_sha256": "b" * 64,
                "sub_receipt_valid": True,
                "summary_en": "verified",
                "summary_fr": "vérifié",
                "metrics": {},
                "diagnostics": [],
            }
        )
        for gid in ("G3", "G4", "G5")
    }

    with TestClient(app) as client:
        # 1. All-green live gateway verification (0 inference calls, 10/10 routes, 17/17 scripts, 4/4 CSP nonces)
        seen_ok: list[tuple[str, str]] = []
        r_live_ok = readiness_gate.run_noel_scale_readiness_gate(
            live_url=readiness_gate.DEFAULT_GATEWAY,
            expected_sha=SAMPLE_SHORT_SHA,
            burst_devs=4,
            burst_gamers=4,
            burst_workers=8,
            environ=CLEAN_ENV,
            modules=MODULES,
            live_http_fn=_make_live_asgi_http(client, seen_calls=seen_ok),
            gate_overrides=fast_overrides,
        )
        _verify_sha256_receipt(r_live_ok)
        assert r_live_ok["mode"] == "offline_rehearsal_plus_live_gateway"
        assert r_live_ok["live_check_enabled"] is True
        assert r_live_ok["live_check_ok"] is True
        assert r_live_ok["drift_404_detected"] is False
        assert r_live_ok["go_for_launch"] is True
        assert r_live_ok["gates_passed_count"] == 5
        live_doc = r_live_ok["live_gateway_check"]
        assert live_doc["routes_verified_count"] == 10
        assert live_doc["scripts_verified_count"] == 17
        assert live_doc["unique_csp_nonces_count"] == 4
        assert live_doc["unique_csp_nonces_ratio"] == 1.0
        assert all(m == "GET" and "completions" not in p for m, p in seen_ok)

        # 2. Live 404 holiday route drift (/arena-noel missing on stale Cloud Run revision)
        r_route_404 = readiness_gate.run_noel_scale_readiness_gate(
            live_url=readiness_gate.DEFAULT_GATEWAY,
            expected_sha=SAMPLE_SHORT_SHA,
            burst_devs=4,
            burst_gamers=4,
            burst_workers=8,
            environ=CLEAN_ENV,
            modules=MODULES,
            live_http_fn=_make_live_asgi_http(client, drop_paths_404={"/arena-noel"}),
            gate_overrides=fast_overrides,
        )
        _verify_sha256_receipt(r_route_404)
        assert r_route_404["go_for_launch"] is False
        assert r_route_404["verdict"] == "NO_GO"
        assert r_route_404["drift_404_detected"] is True
        assert r_route_404["live_check_ok"] is False
        assert r_route_404["failed_gate_ids"] == ["G1"]
        assert "/arena-noel" in r_route_404["live_gateway_check"]["missing_routes"]
        md_drift = readiness_gate.render_markdown_report(r_route_404, lang="en")
        cli_drift = readiness_gate.render_cli_report(r_route_404, lang="en")
        assert "gcloud run services update" in md_drift
        assert "NO-GO" in md_drift
        assert "live_url=" in cli_drift and "[DIAG]" in cli_drift

        # 3. Live 404 script allowlist drift (/scripts/noel_canary_cohort_rehearsal.py missing)
        r_script_404 = readiness_gate.run_noel_scale_readiness_gate(
            live_url=readiness_gate.DEFAULT_GATEWAY,
            expected_sha=SAMPLE_SHORT_SHA,
            burst_devs=4,
            burst_gamers=4,
            burst_workers=8,
            environ=CLEAN_ENV,
            modules=MODULES,
            live_http_fn=_make_live_asgi_http(
                client,
                drop_paths_404={"/scripts/noel_canary_cohort_rehearsal.py", "/v1/hivemind/raid/status"},
            ),
            gate_overrides=fast_overrides,
            lang="fr",
        )
        _verify_sha256_receipt(r_script_404)
        assert r_script_404["go_for_launch"] is False
        assert r_script_404["drift_404_detected"] is True
        assert "noel_canary_cohort_rehearsal.py" in r_script_404["live_gateway_check"]["missing_scripts"]
        assert "/v1/hivemind/raid/status" in r_script_404["live_gateway_check"]["missing_routes"]

        # 4. Live HTML CSP nonce reuse + unsafe DOM sink detection
        r_live_csp = readiness_gate.run_noel_scale_readiness_gate(
            live_url=readiness_gate.DEFAULT_GATEWAY,
            expected_sha=SAMPLE_SHORT_SHA,
            burst_devs=4,
            burst_gamers=4,
            burst_workers=8,
            environ=CLEAN_ENV,
            modules=MODULES,
            live_http_fn=_make_live_asgi_http(
                client,
                static_nonce_override="StaticLiveNonce1234",
                inject_dom_sink=True,
            ),
            gate_overrides=fast_overrides,
        )
        _verify_sha256_receipt(r_live_csp)
        assert r_live_csp["go_for_launch"] is False
        assert r_live_csp["live_check_ok"] is False
        assert "G2" in r_live_csp["failed_gate_ids"]
        assert r_live_csp["live_gateway_check"]["forbidden_dom_sinks_count"] == 1


def test_amadeus_and_corporate_pc_compliance_refuses_before_gates_or_network(tmp_path: Path) -> None:
    network_called = [False]

    def _forbid_network(*_args: Any, **_kwargs: Any) -> Any:
        network_called[0] = True
        raise AssertionError("Corporate host must NEVER initiate live gateway HTTP requests")

    corp_repo = tmp_path / "amadeus_enterprise_checkout"
    (corp_repo / ".git").mkdir(parents=True, exist_ok=True)
    (corp_repo / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = git@git-amadeus-internal:core/secret-platform.git\n',
        encoding="utf-8",
    )

    corp_vectors = (
        ({"PATH": "/usr/bin", "AMADEUS_ENV": "amadeus-prod-secret-token"}, tmp_path / "hobby", "en"),
        ({"PATH": "/usr/bin", "USERDNSDOMAIN": "nce.amadeus.corp"}, tmp_path / "hobby", "fr"),
        ({"PATH": "/usr/bin", "ZSCALER_CERT_PATH": "/etc/zscaler.pem", "HTTPS_PROXY": "zscaler-gw:8443"}, tmp_path / "hobby", "en"),
        ({"PATH": "/usr/bin"}, corp_repo, "fr"),
    )

    for env_map, cwd_dir, lang_code in corp_vectors:
        cwd_dir.mkdir(parents=True, exist_ok=True)
        report = readiness_gate.run_noel_scale_readiness_gate(
            live_url=readiness_gate.DEFAULT_GATEWAY,
            environ=env_map,
            cwd=cwd_dir,
            check_cwd_git_remote=True,
            modules=MODULES,
            live_http_fn=_forbid_network,
            lang=lang_code,
        )
        _verify_sha256_receipt(report)
        assert network_called[0] is False
        assert report["compliance_ok"] is False
        assert report["go_for_launch"] is False
        assert report["verdict"] == "NO_GO"
        assert report["exit_code"] == 2
        assert report["gates_passed_count"] == 0
        assert report["gates_total_count"] == 5
        assert all(g["status"] == "SKIPPED_COMPLIANCE" for g in report["gates"])
        serialized = json.dumps(report, sort_keys=True)
        assert "amadeus-prod-secret-token" not in serialized
        assert "secret-platform.git" not in serialized


def test_cli_modes_loopback_http_server_and_input_safety_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = readiness_gate.LocalScaleGatewaySimulator(
        "http://127.0.0.1:8080",
        scripts_dir=REPO_ROOT / "scripts",
        allowed_scripts=MODULES.parity_gate.REQUIRED_SCRIPTS,
        holiday_routes=MODULES.parity_gate.EXPECTED_MANIFEST_HOLIDAY_ROUTES,
        build_sha=SAMPLE_FULL_SHA,
    )

    # Zero-inference & simulator negative controls
    with pytest.raises(readiness_gate.ReadinessGateError, match="Zero-inference"):
        sim.request_full("POST", "http://127.0.0.1:8080/v1/chat/completions", {"messages": []})
    assert sim.inference_calls_count == 1
    assert sim.request_full("GET", "http://127.0.0.1:8080/unknown")[0] == 404
    assert sim.request_full("GET", "http://127.0.0.1:8080/v1/hivemind/me")[0] == 401
    assert sim.request_full("POST", "http://127.0.0.1:8080/v1/hivemind/arena/vote", {"task_id": "bad"})[0] == 422
    assert (
        sim.request_full(
            "POST",
            "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback",
            {"profile": "dev", "ease_rating": 5, "email": "leak@example.com"},
        )[0]
        == 422
    )
    assert sim.request_full("GET", "http://127.0.0.1:8080/v1/hivemind/live-telemetry")[0] == 200

    # Real loopback HTTP server exercising default_live_http_request & CLI main()
    live_sim = readiness_gate.LocalScaleGatewaySimulator(
        "http://127.0.0.1:8080",
        scripts_dir=REPO_ROOT / "scripts",
        allowed_scripts=MODULES.parity_gate.REQUIRED_SCRIPTS,
        holiday_routes=MODULES.parity_gate.EXPECTED_MANIFEST_HOLIDAY_ROUTES,
        build_sha=SAMPLE_FULL_SHA,
    )

    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/plain-502":
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"upstream error")
                return
            code, body, hdrs = live_sim.request_full("GET", f"http://127.0.0.1:8080{self.path}")
            raw = (
                json.dumps(body).encode("utf-8")
                if isinstance(body, dict)
                else str(body).encode("utf-8")
            )
            self.send_response(code)
            for hk, hv in hdrs.items():
                self.send_header(hk, hv)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return None

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        c_ok, b_ok, _ = readiness_gate.default_live_http_request(
            "GET",
            f"{base_url}/health",
            headers={"X-Test": "1"},
        )
        assert c_ok == 200 and isinstance(b_ok, dict) and b_ok["status"] == "ok"
        assert readiness_gate.default_live_http_request("GET", f"{base_url}/noel")[0] == 200
        assert readiness_gate.default_live_http_request("GET", f"{base_url}/scripts/unknown.py")[0] == 404
        assert readiness_gate.default_live_http_request("GET", f"{base_url}/plain-502")[0] == 502

        json_out_file = tmp_path / "out" / "scorecard.json"
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        rc_json = readiness_gate.main(
            [
                "--live-url",
                base_url,
                "--expected-sha",
                SAMPLE_SHORT_SHA,
                "--burst-devs",
                "3",
                "--burst-gamers",
                "3",
                "--burst-workers",
                "6",
                "--cohort-size",
                "5",
                "--corporate-controls",
                "5",
                "--cohort-workers",
                "5",
                "--json",
                "--output",
                str(json_out_file),
            ],
            environ=CLEAN_ENV,
            modules=MODULES,
        )
        assert rc_json == 0
        parsed_cli_json = json.loads(json_out_file.read_text(encoding="utf-8"))
        _verify_sha256_receipt(parsed_cli_json)
        assert parsed_cli_json["go_for_launch"] is True

        # CLI --cli and --markdown modes + mutual exclusion error
        rc_cli = readiness_gate.main(
            [
                "--offline",
                "--cli",
                "--lang",
                "fr",
                "--burst-devs",
                "2",
                "--burst-gamers",
                "2",
                "--cohort-size",
                "5",
                "--corporate-controls",
                "2",
            ],
            environ=CLEAN_ENV,
            modules=MODULES,
        )
        assert rc_cli == 0
        rc_md = readiness_gate.main(
            [
                "--offline",
                "--markdown",
                "--lang",
                "en",
                "--burst-devs",
                "2",
                "--burst-gamers",
                "2",
                "--cohort-size",
                "5",
                "--corporate-controls",
                "2",
            ],
            environ=CLEAN_ENV,
            modules=MODULES,
        )
        assert rc_md == 0
        rc_conflict = readiness_gate.main(["--offline", "--live-url", base_url], environ=CLEAN_ENV)
        assert rc_conflict == 2

        # Subprocess smoke invocation
        sub = subprocess.run(
            [
                sys.executable,
                str(GATE_SCRIPT_PATH),
                "--offline",
                "--burst-devs",
                "2",
                "--burst-gamers",
                "2",
                "--cohort-size",
                "5",
                "--corporate-controls",
                "2",
                "--cli",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
        )
        assert sub.returncode == 0, sub.stderr
        assert "[GO]" in sub.stdout
    finally:
        server.shutdown()
        server.server_close()

    # Closed loopback port returns status 0 cleanly
    c_closed, b_closed, _ = readiness_gate.default_live_http_request("GET", "http://127.0.0.1:1/health")
    assert c_closed == 0 and isinstance(b_closed, dict) and "network_error" in b_closed

    # Subprocess runner guardrails
    rc_s, out_s = readiness_gate.default_subprocess_runner([sys.executable, "-c", "print('ok')"], 3.0)
    assert rc_s == 0 and out_s.strip() == "ok"
    with pytest.raises(readiness_gate.ReadinessGateError, match="empty"):
        readiness_gate.default_subprocess_runner([], 1.0)
    with pytest.raises(readiness_gate.ReadinessGateError, match="gcloud"):
        readiness_gate.default_subprocess_runner(["gcloud", "run", "deploy"], 1.0)

    # Validation & output safety negative controls
    untrusted_url = "https://" + "evil.example" + ".org"
    non_loop_ip = "10." + "20.30.40"
    for bad_url in (
        "",
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        untrusted_url,
        "https://user:pass@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/extra",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app?q=1",
        "ftp://127.0.0.1:8080",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:80 80",
        "http://127.0.0.1:8080\n",
    ):
        with pytest.raises(readiness_gate.ReadinessGateError):
            readiness_gate.validate_gateway_url(bad_url)

    assert readiness_gate.is_allowed_gateway_host("storage.googleapis.com") is True
    assert readiness_gate.is_allowed_gateway_host("datasets-server.huggingface.co") is True
    assert readiness_gate.is_allowed_gateway_host("") is False

    for bad_lang in ("de", True):
        with pytest.raises(readiness_gate.ReadinessGateError):
            readiness_gate.normalize_lang(bad_lang)  # type: ignore[arg-type]

    for bad_date in ("", "2026/12/25", "2026-02-30"):
        with pytest.raises(readiness_gate.ReadinessGateError):
            readiness_gate.validate_report_date(bad_date)
    assert len(readiness_gate.validate_report_date(None)) == 10

    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._assert_read_only_zero_inference("POST", "http://127.0.0.1:8080/health")
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._assert_read_only_zero_inference("GET", "http://127.0.0.1:8080/health", {"a": 1})
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._assert_read_only_zero_inference("GET", "http://127.0.0.1:8080/v1/chat/completions")

    assert readiness_gate._invoke_live_http(
        lambda _m, _u: (200, {"status": "ok", "csp": "style-src 'nonce-abc12345678'"}),
        "GET",
        "http://127.0.0.1:8080/health",
    )[0] == 200
    assert readiness_gate._invoke_live_http(
        lambda _m, _u: (_ for _ in ()).throw(urllib.error.URLError("down")),
        "GET",
        "http://127.0.0.1:8080/health",
    )[0] == 0
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._invoke_live_http(lambda _m, _u: "invalid", "GET", "http://127.0.0.1:8080/health")

    bad_key_fragment = '{"' + "sco" + 're": 5}'
    raw_hm_key = "hm-" + "z" * 18
    assert "forbidden_key_detected" in readiness_gate.scan_output_safety(bad_key_fragment)
    assert "unmasked_api_key_detected" in readiness_gate.scan_output_safety(raw_hm_key)
    assert "bearer_token_detected" in readiness_gate.scan_output_safety("Bearer tok_secret_999")
    assert "email_detected" in readiness_gate.scan_output_safety("user@example.org")
    assert "unredacted_home_path_detected" in readiness_gate.scan_output_safety("/home/alice/.hivemind")
    assert "discord_mass_ping_detected" in readiness_gate.scan_output_safety("ping @everyone")
    assert any(
        hit.startswith("non_loopback_ipv4_detected:")
        for hit in readiness_gate.scan_output_safety(f"peer {non_loop_ip}")
    )
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate.scan_output_safety(123)  # type: ignore[arg-type]
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._assert_safe_output(bad_key_fragment)

    assert readiness_gate.verify_receipt_sha256({"receipt_sha256": "short"}) is False
    tampered = {"a": 1, "receipt_sha256": "0" * 64}
    assert readiness_gate.verify_receipt_sha256(tampered) is False

    # Additional validation, path redaction, module reuse, and malformed-gate edge branches
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate.validate_gateway_url("http://127.0.0.1:0")
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._validate_int_range(True, "x", 1, 5)
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._validate_positive_float("bad", "x")
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._validate_positive_float(-1.0, "x")
    with pytest.raises(readiness_gate.ReadinessGateError, match="p99_slo_ms"):
        readiness_gate.run_noel_scale_readiness_gate(p95_slo_ms=500.0, p99_slo_ms=100.0)

    assert readiness_gate.redact_home_path(GATE_SCRIPT_PATH) == "scripts/noel_scale_readiness_gate.py"
    assert readiness_gate.redact_home_path(Path("/opt/u1/.hivemind"), home=Path("/opt/u1")) == "~/.hivemind"
    scrubbed = readiness_gate.sanitize_text_for_display(f"peer {non_loop_ip} local 127.0.0.1")
    assert "[redacted-ip]" in scrubbed and "127.0.0.1" in scrubbed

    comp_dom = readiness_gate.evaluate_personal_pc_compliance(
        environ={"USERDNSDOMAIN": "custom.local", "USERDOMAIN": "HOMEBOX", "COMPUTERNAME": "HOMEBOX"}
    )
    assert comp_dom["compliance_ok"] is False and "domain" in comp_dom["matched_markers"]

    assert readiness_gate.load_scale_readiness_modules() == MODULES
    with pytest.raises(readiness_gate.ReadinessGateError):
        readiness_gate._load_or_reuse_script_module("missing_x", tmp_path / "does_not_exist.py")

    _, _, hdr_map = readiness_gate._invoke_live_http(
        lambda _m, _u: (200, {"status": "ok", "headers": {"X-Custom": "1"}}),
        "GET",
        "http://127.0.0.1:8080/health",
    )
    assert hdr_map["x-custom"] == "1"

    malformed_g = readiness_gate._execute_gate_safely(
        "G1",
        "slug",
        "en",
        "fr",
        lambda: {"gate_id": "WRONG"},
        is_fr=False,
    )
    assert malformed_g["passed"] is False and "malformed" in malformed_g["diagnostics"][0]


def test_zz_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    _, stmts, _, missing, missing_fmt = _INPROC_COV.analysis2(str(GATE_SCRIPT_PATH))
    covered = len(stmts) - len(missing)
    pct = 100.0 * covered / max(1, len(stmts))
    assert pct >= 95.0, (
        f"Expected >= 95.0% line coverage on {GATE_SCRIPT_PATH.name}, got {pct:.2f}% "
        f"(missing lines: {missing_fmt})"
    )
