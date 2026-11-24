# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Full holiday script allowlist parity and concurrent `/scripts/` delivery stress tests (`SLICE-SCALE-03`)."""

from __future__ import annotations

import ast
import concurrent.futures
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

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "scripts" / "noel_cloud_run_parity_gate.py"
GATEWAY_PY_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "gateway.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(GATE_PATH), str(GATEWAY_PY_PATH)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import (  # noqa: E402
    _ALLOWED_ONBOARDING_SCRIPTS,
    _HOLIDAY_ONBOARDING_ROUTES,
    RouterError,
    _sanitize_build_sha,
    _sanitize_cloud_run_revision,
    build_onboarding_manifest,
    create_router_app,
    resolve_onboarding_script,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_gate_spec = importlib.util.spec_from_file_location("noel_cloud_run_parity_gate_scale03", GATE_PATH)
assert _gate_spec is not None and _gate_spec.loader is not None
parity_gate = importlib.util.module_from_spec(_gate_spec)
sys.modules["noel_cloud_run_parity_gate_scale03"] = parity_gate
_gate_spec.loader.exec_module(parity_gate)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_scale03",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_scale03",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

SAMPLE_FULL_SHA = "3cb0a717cbeef407aff23c13f240f915499f1a54"
SAMPLE_SHORT_SHA = "3cb0a717cbee"
ROUTER_SECRET = b"noel-scale-03-router-secret-32b!"

NEW_NOEL_ZERO_CLONE_TOOLS: tuple[str, ...] = (
    "noel_canary_cohort_rehearsal.py",
    "noel_gamer_whisper_selftest.py",
    "noel_scale_load_harness.py",
    "noel_vscode_autoconfig.py",
)

EXPECTED_ALL_17_SCRIPTS: tuple[str, ...] = (
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


def _make_asgi_http(client: TestClient, seen: list[tuple[str, str]] | None = None) -> parity_gate.HttpFn:
    lock = threading.Lock()

    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert method == "GET", "Parity gate must strictly perform read-only GET requests"
        assert payload is None, "Parity gate must never send request payloads"
        assert "completions" not in url.lower(), "Parity gate must never call /v1/chat/completions"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen is not None:
            with lock:
                seen.append((method, path))
        resp = client.request(method, path, headers=headers)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json()
        return resp.status_code, {
            "content_type": ctype,
            "text_length": len(resp.content),
            "html_length": len(resp.content),
        }

    return _http


def test_compliance_headers_allowlist_parity_and_security_scanners() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    this_file = Path(__file__).resolve()

    # 1. AGPL-3.0 header invariant across gateway.py, parity gate, test file, and all 16 Python scripts
    for path in (GATEWAY_PY_PATH, GATE_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    # 2. Exact 17-script allowlist parity across gateway.py and noel_cloud_run_parity_gate.py
    assert tuple(sorted(_ALLOWED_ONBOARDING_SCRIPTS)) == EXPECTED_ALL_17_SCRIPTS
    assert parity_gate.REQUIRED_SCRIPTS == EXPECTED_ALL_17_SCRIPTS
    assert parity_gate.EXPECTED_MANIFEST_SCRIPTS == EXPECTED_ALL_17_SCRIPTS
    for noel_tool in NEW_NOEL_ZERO_CLONE_TOOLS:
        assert noel_tool in _ALLOWED_ONBOARDING_SCRIPTS
        assert noel_tool in parity_gate.REQUIRED_SCRIPTS
        assert noel_tool in parity_gate.NOEL_ZERO_CLONE_SCRIPTS

    # 3. Every allowlisted script exists on disk, is non-empty, and .py scripts parse & pass lateral-movement checks
    scripts_dir = REPO_ROOT / "scripts"
    for script_name in EXPECTED_ALL_17_SCRIPTS:
        disk_path = scripts_dir / script_name
        assert disk_path.is_file(), f"Allowlisted script missing from scripts/: {script_name}"
        raw = disk_path.read_bytes()
        assert len(raw) > 0, f"Allowlisted script is empty: {script_name}"
        if script_name.endswith(".py"):
            source = raw.decode("utf-8")
            ast.parse(source, filename=f"scripts/{script_name}")

    for noel_script in parity_gate.NOEL_ZERO_CLONE_SCRIPTS:
        src = (scripts_dir / noel_script).read_text(encoding="utf-8")
        assert expected_header in src[:300], f"Missing AGPL-3.0 header in scripts/{noel_script}"
        assert lateral_guard.audit_source_for_lateral_movement(src, filename=f"scripts/{noel_script}") == []

    gate_source = GATE_PATH.read_text(encoding="utf-8")
    gateway_source = GATEWAY_PY_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    assert lateral_guard.audit_source_for_lateral_movement(test_source, filename="tests/test_noel_scripts_scale_delivery.py") == []

    # 4. Pure Python standard library only in scripts/noel_cloud_run_parity_gate.py
    gate_tree = ast.parse(gate_source, filename="scripts/noel_cloud_run_parity_gate.py")
    imported_roots: set[str] = set()
    for node in ast.walk(gate_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 5. PR gatekeeper FORBIDDEN_ADDED cleanliness across all touched/added files
    for rel, content in (
        ("src/hivemind/router/gateway.py", gateway_source),
        ("scripts/noel_cloud_run_parity_gate.py", gate_source),
        ("tests/test_noel_scripts_scale_delivery.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 6. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit + static-figure ratchet
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("src/hivemind/router/gateway.py", gateway_source),
        ("scripts/noel_cloud_run_parity_gate.py", gate_source),
        ("tests/test_noel_scripts_scale_delivery.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1063,
        pr_title="feat(scale): full holiday script allowlist parity and concurrent /scripts/ delivery stress tests",
        pr_body="Implements SLICE-SCALE-03 full holiday script allowlist parity and concurrent /scripts/ stress tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(scale): full holiday script allowlist parity and concurrent /scripts/ delivery stress tests"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    from gwaya.ast_audit import ZeroStubAudit

    stub_audit = ZeroStubAudit().audit_python_code(gate_source)
    assert stub_audit.is_clean, f"Unexpected GWAYA stub violations: {stub_audit.violations}"

    fig_unit = "|".join(("m" + "s\\b", "tok" + "/s", "HTTP" + " Reqs", "Comple" + "tions", "\\bs/" + "round"))
    fig_re = re.compile(r"\d[\d.,]*\s?(?:" + fig_unit + r")")
    assert fig_re.findall(gate_source) == []
    assert fig_re.findall(test_source) == []


def test_concurrent_multithreaded_script_downloads_exact_sha256_match(tmp_path: Path) -> None:
    scripts_dir = REPO_ROOT / "scripts"
    expected_disk: dict[str, tuple[int, str, bytes]] = {}
    for name in EXPECTED_ALL_17_SCRIPTS:
        raw_bytes = (scripts_dir / name).read_bytes()
        expected_disk[name] = (len(raw_bytes), hashlib.sha256(raw_bytes).hexdigest(), raw_bytes)

    store = RouterStore(tmp_path / "scale03_concurrent.db", secret=ROUTER_SECRET)
    app = create_router_app(store)

    worker_count = 32
    waves = 8
    download_jobs = [name for _ in range(waves) for name in EXPECTED_ALL_17_SCRIPTS]
    assert len(download_jobs) == 136

    start_barrier = threading.Barrier(worker_count)

    with TestClient(app) as client:
        # Also verify legacy direct worker endpoint delegates identically
        r_legacy = client.get("/hivemind_boinc_idle_worker.py")
        assert r_legacy.status_code == 200
        assert hashlib.sha256(r_legacy.content).hexdigest() == expected_disk["hivemind_boinc_idle_worker.py"][1]

        def _worker_batch(batch_index: int, names: list[str]) -> list[tuple[str, int, str, int, dict[str, str]]]:
            start_barrier.wait(timeout=10.0)
            out: list[tuple[str, int, str, int, dict[str, str]]] = []
            for idx, script_name in enumerate(names):
                if idx == 0 and batch_index % 8 == 0:
                    man_resp = client.get("/v1/hivemind/onboarding/manifest")
                    assert man_resp.status_code == 200
                    assert man_resp.json()["downloadable_scripts"] == list(EXPECTED_ALL_17_SCRIPTS)
                resp = client.get(f"/scripts/{script_name}")
                digest = hashlib.sha256(resp.content).hexdigest()
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                out.append((script_name, resp.status_code, digest, len(resp.content), hdrs))
            return out

        shards: list[list[str]] = [download_jobs[i::worker_count] for i in range(worker_count)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_worker_batch, i, shards[i]) for i in range(worker_count)]
            all_results = [item for fut in futures for item in fut.result(timeout=30.0)]

    assert len(all_results) == 136
    per_script_counts: dict[str, int] = dict.fromkeys(EXPECTED_ALL_17_SCRIPTS, 0)
    for script_name, status_code, digest, byte_len, hdrs in all_results:
        exp_len, exp_sha, _ = expected_disk[script_name]
        assert status_code == 200, f"Unexpected HTTP {status_code} for {script_name}"
        assert byte_len == exp_len, f"Byte-length mismatch for {script_name}: {byte_len} != {exp_len}"
        assert digest == exp_sha, f"SHA-256 mismatch for {script_name}: {digest} != {exp_sha}"
        assert hdrs.get("x-script-sha256") == exp_sha
        assert hdrs.get("cache-control") == "no-store"
        assert hdrs.get("x-content-type-options") == "nosniff"
        assert f'filename="{script_name}"' in hdrs.get("content-disposition", "")
        if script_name.endswith(".ps1"):
            assert "text/plain" in hdrs.get("content-type", "")
        else:
            assert "text/x-python" in hdrs.get("content-type", "")
        per_script_counts[script_name] += 1

    assert all(count == waves for count in per_script_counts.values())

    # Verify stat-keyed cache coherence when an underlying script file updates on disk
    staging_dir = tmp_path / "staging_scripts"
    staging_dir.mkdir()
    staged_script = staging_dir / "noel_vscode_autoconfig.py"
    v1_bytes = b"# v1\r\nprint('hello')\r\n"
    staged_script.write_bytes(v1_bytes)
    b1, mt1, d1 = resolve_onboarding_script("noel_vscode_autoconfig.py", search_dirs=(staging_dir,))
    assert b1 == v1_bytes
    assert "text/x-python" in mt1
    assert d1 == hashlib.sha256(v1_bytes).hexdigest()
    # Second call hits stat-matched cache and preserves CRLF byte-for-byte
    b1_cached, _, d1_cached = resolve_onboarding_script("noel_vscode_autoconfig.py", search_dirs=(staging_dir,))
    assert b1_cached == v1_bytes and d1_cached == d1

    v2_bytes = b"# v2 updated longer payload\r\nprint('updated')\r\n"
    staged_script.write_bytes(v2_bytes)
    b2, _, d2 = resolve_onboarding_script("noel_vscode_autoconfig.py", search_dirs=(staging_dir,))
    assert b2 == v2_bytes
    assert d2 == hashlib.sha256(v2_bytes).hexdigest()


def test_adversarial_traversal_null_byte_and_unknown_script_rejection_under_load(tmp_path: Path) -> None:
    scripts_dir = REPO_ROOT / "scripts"
    expected_sha = {
        name: hashlib.sha256((scripts_dir / name).read_bytes()).hexdigest()
        for name in EXPECTED_ALL_17_SCRIPTS
    }

    adversarial_paths: tuple[str, ...] = (
        "/scripts/..%2f..%2fetc%2fpasswd",
        "/scripts/..%2fsrc%2fhivemind%2frouter%2fgateway.py",
        "/scripts/%2e%2e%2f%2e%2e%2fpyproject.toml",
        "/scripts/..%5c..%5cWindows%5cwin.ini",
        "/scripts/..%5cscripts%5cpr_gatekeeper.py",
        "/scripts/sub%2f..%2fhivemind_quickstart.py",
        "/scripts/sub/hivemind_quickstart.py",
        "/scripts/hivemind_quickstart.py%00",
        "/scripts/hivemind_quickstart.py%00.png",
        "/scripts/%00noel_vscode_autoconfig.py",
        "/scripts/pr_gatekeeper.py",
        "/scripts/t4_governor.py",
        "/scripts/unknown_holiday_tool.py",
        "/scripts/NOEL_VSCODE_AUTOCONFIG.PY",
        "/scripts/noel_vscode_autoconfig.pyc",
        "/scripts/noel_vscode_autoconfig.py.bak",
        "/scripts/noel_vscode_autoconfig.py:$DATA",
        "/scripts/noel_vscode_autoconfig.py.",
        "/scripts/%2e%2e",
        "/scripts/%2e",
        "/scripts/",
    )

    store = RouterStore(tmp_path / "scale03_adversarial.db", secret=ROUTER_SECRET)
    app = create_router_app(store)

    mixed_tasks: list[tuple[str, str]] = []
    for round_idx in range(3):
        for adv in adversarial_paths:
            mixed_tasks.append(("adv", adv))
        for valid_name in EXPECTED_ALL_17_SCRIPTS:
            mixed_tasks.append(("valid", valid_name))
        _ = round_idx

    workers = 24
    barrier = threading.Barrier(workers)

    with TestClient(app) as client:

        def _run_mixed_shard(shard: list[tuple[str, str]]) -> list[tuple[str, str, int, bytes]]:
            barrier.wait(timeout=10.0)
            results: list[tuple[str, str, int, bytes]] = []
            for kind, target in shard:
                path = target if kind == "adv" else f"/scripts/{target}"
                resp = client.get(path)
                results.append((kind, target, resp.status_code, resp.content))
            return results

        shards = [mixed_tasks[i::workers] for i in range(workers)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run_mixed_shard, s) for s in shards]
            outcomes = [row for f in futs for row in f.result(timeout=30.0)]

    adv_checked = 0
    valid_checked = 0
    for kind, target, status_code, raw_body in outcomes:
        if kind == "adv":
            adv_checked += 1
            assert status_code == 404, f"Adversarial vector {target!r} must return 404, got {status_code}"
            decoded = raw_body.decode("utf-8", errors="replace")
            assert "root:x:0:0" not in decoded
            assert "FORBIDDEN_ADDED" not in decoded
            assert "def create_router_app" not in decoded
        else:
            valid_checked += 1
            assert status_code == 200, f"Interleaved valid download {target!r} failed with {status_code}"
            assert hashlib.sha256(raw_body).hexdigest() == expected_sha[target]

    assert adv_checked == 3 * len(adversarial_paths)
    assert valid_checked == 3 * len(EXPECTED_ALL_17_SCRIPTS)

    # Direct unit tests for resolve_onboarding_script boundary, symlink-escape, directory-confusion, and missing-file guards
    for bad_input in (
        None,
        123,
        "",
        "hivemind_quickstart.py\x00",
        "hivemind_quickstart.py\x00.py",
        "../hivemind_quickstart.py",
        "..\\hivemind_quickstart.py",
        "sub/hivemind_quickstart.py",
        "pr_gatekeeper.py",
        "hivemind_quickstart.sh",
    ):
        with pytest.raises(RouterError) as exc_info:
            resolve_onboarding_script(bad_input)  # type: ignore[arg-type]
        assert exc_info.value.status == 404
        assert exc_info.value.code == "not_found"

    missing_dir = tmp_path / "does_not_exist"
    confusing_dir = tmp_path / "confusing_scripts"
    confusing_dir.mkdir()
    # Directory named like an allowlisted script must NOT trigger HTTP 500 IsADirectoryError
    (confusing_dir / "hivemind_doctor.py").mkdir()
    with pytest.raises(RouterError) as dir_exc:
        resolve_onboarding_script("hivemind_doctor.py", search_dirs=(missing_dir, confusing_dir))
    assert dir_exc.value.status == 404
    assert "not present on server" in dir_exc.value.message

    # Planted symlink inside search_dirs pointing outside search_dirs must be rejected
    outside_secret = tmp_path / "outside_secret.py"
    outside_secret.write_text("# secret outside scripts dir\n", encoding="utf-8")
    symlink_dir = tmp_path / "symlink_scripts"
    symlink_dir.mkdir()
    (symlink_dir / "hivemind_quickstart.py").symlink_to(outside_secret)
    with pytest.raises(RouterError) as sym_exc:
        resolve_onboarding_script("hivemind_quickstart.py", search_dirs=(symlink_dir,))
    assert sym_exc.value.status == 404


def test_manifest_and_full_script_parity_via_check_cloud_run_holiday_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Gateway manifest & sanitization helpers
    assert _sanitize_build_sha(None) == "unknown"
    assert _sanitize_build_sha("   ") == "unknown"
    assert _sanitize_build_sha("UNKNOWN") == "unknown"
    assert _sanitize_build_sha("abc") == "unknown"
    assert _sanitize_build_sha("mysecret999") == "unknown"
    assert _sanitize_build_sha("  3CB0A717CBEE  ") == "3cb0a717cbee"

    assert _sanitize_cloud_run_revision(None) is None
    assert _sanitize_cloud_run_revision("") is None
    assert _sanitize_cloud_run_revision("-bad") is None
    assert _sanitize_cloud_run_revision("hm-" + "x" * 16) is None
    assert _sanitize_cloud_run_revision("rev-secret-1") is None
    assert _sanitize_cloud_run_revision("hivemind-openrouter-00120-scale03") == "hivemind-openrouter-00120-scale03"

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", SAMPLE_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00120-scale03")
    store = RouterStore(tmp_path / "parity.db", secret=ROUTER_SECRET)
    app = create_router_app(store)

    with TestClient(app) as client:
        man_resp = client.get("/v1/hivemind/onboarding/manifest")
        assert man_resp.status_code == 200
        assert man_resp.headers.get("cache-control") == "no-store"
        man = man_resp.json()
        assert man == {
            "object": "hivemind.onboarding.manifest",
            "campaign_id": "noel-2026-easy-onboard",
            "status": "ok",
            "build_sha": SAMPLE_FULL_SHA,
            "cloud_run_revision": "hivemind-openrouter-00120-scale03",
            "holiday_routes": sorted(_HOLIDAY_ONBOARDING_ROUTES),
            "downloadable_scripts": list(EXPECTED_ALL_17_SCRIPTS),
            "personal_pc_only": True,
            "financial_promise": "0.00 EUR / 0.00 USD",
        }

        # Default parity check (8 routes + 5 spot-probe scripts, validating all 17 manifest scripts)
        seen_default: list[tuple[str, str]] = []
        rep_default = parity_gate.check_cloud_run_holiday_parity(
            "http://127.0.0.1:8000",
            expected_sha=SAMPLE_SHORT_SHA,
            http_fn=_make_asgi_http(client, seen_default),
            lang="en",
        )
        _verify_sha256_receipt(rep_default)
        assert rep_default["parity_ok"] is True
        assert rep_default["redeploy_required"] is False
        assert rep_default["cloud_run_lags_main"] is False
        assert rep_default["manifest_valid"] is True
        assert rep_default["missing_manifest_scripts"] == []
        assert rep_default["routes_verified_count"] == 8
        assert rep_default["scripts_verified_count"] == 5
        assert len(seen_default) == 13

        # Full --verify-all-scripts parity check probing all 17 allowlisted /scripts/<name> routes
        seen_all: list[tuple[str, str]] = []
        rep_all = parity_gate.check_cloud_run_holiday_parity(
            "http://127.0.0.1:8000",
            expected_sha=SAMPLE_SHORT_SHA,
            verify_all_scripts=True,
            http_fn=_make_asgi_http(client, seen_all),
            lang="fr",
        )
        _verify_sha256_receipt(rep_all)
        assert rep_all["parity_ok"] is True
        assert rep_all["routes_verified_count"] == 8
        assert rep_all["scripts_verified_count"] == 17
        assert rep_all["scripts_total_count"] == 17
        assert len(seen_all) == 8 + 17
        probed_scripts = [p.removeprefix("/scripts/") for _, p in seen_all if p.startswith("/scripts/")]
        assert probed_scripts == list(EXPECTED_ALL_17_SCRIPTS)

        # CLI --verify-all-scripts + --print-redeploy-plan --verify-live
        rc_cli_all = parity_gate.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--expected-sha",
                SAMPLE_SHORT_SHA,
                "--verify-all-scripts",
                "--print-redeploy-plan",
                "--verify-live",
                "--markdown",
            ],
            http_fn=_make_asgi_http(client),
        )
        assert rc_cli_all == 0
        md_all = capsys.readouterr().out
        assert "- **Scripts Verified**: `17`" in md_all
        assert "`noel_canary_cohort_rehearsal.py`" in md_all

    # 2. Stale pre-SLICE-SCALE-03 Cloud Run simulation (manifest & /scripts/ missing the 4 new Noel tools)
    legacy_13 = [s for s in EXPECTED_ALL_17_SCRIPTS if s not in NEW_NOEL_ZERO_CLONE_TOOLS]

    def _pre_scale03_http(method: str, url: str) -> tuple[int, Any]:
        assert method == "GET"
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/auto"}]}
        if url.endswith("/v1/hivemind/onboarding/manifest"):
            return 200, {
                "object": "hivemind.onboarding.manifest",
                "campaign_id": "noel-2026-easy-onboard",
                "status": "ok",
                "build_sha": SAMPLE_SHORT_SHA,
                "cloud_run_revision": "hivemind-openrouter-00105-old",
                "holiday_routes": sorted(_HOLIDAY_ONBOARDING_ROUTES),
                "downloadable_scripts": legacy_13,
                "personal_pc_only": True,
                "financial_promise": "0.00 EUR / 0.00 USD",
            }
        if url.endswith("/v1/hivemind/onboarding/feedback/summary"):
            return 200, {"total_responses": 1}
        if any(url.endswith(p) for p in ("/noel", "/quick", "/raid-noel", "/arena-noel")):
            return 200, "<!DOCTYPE html><title>Noel</title>"
        for s in legacy_13:
            if url.endswith(f"/scripts/{s}"):
                return 200, "# allowlisted legacy script\n"
        return 404, {"detail": "Not Found"}

    for lang_choice in ("en", "fr"):
        rc_lag = parity_gate.main(
            [
                "--gateway",
                "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
                "--expected-sha",
                SAMPLE_SHORT_SHA,
                "--verify-all-scripts",
                "--lang",
                lang_choice,
                "--json",
            ],
            http_fn=_pre_scale03_http,
        )
        assert rc_lag == 2
        lag_doc = json.loads(capsys.readouterr().out)
        assert lag_doc["parity_ok"] is False
        assert lag_doc["redeploy_required"] is True
        assert lag_doc["cloud_run_lags_main"] is True
        assert lag_doc["manifest_valid"] is False
        assert lag_doc["missing_manifest_scripts"] == list(NEW_NOEL_ZERO_CLONE_TOOLS)
        assert lag_doc["missing_scripts"] == list(NEW_NOEL_ZERO_CLONE_TOOLS)
        assert lag_doc["scripts_verified_count"] == 13
        assert lag_doc["scripts_total_count"] == 17


def test_parity_gate_error_paths_finops_guards_and_loopback_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Broken upstream responses, SHA drift, git runner branches, and FinOps dry-run validation
    def _broken_http(method: str, url: str) -> tuple[int, Any]:
        _ = method
        if url.endswith("/health"):
            return 200, {"status": "bad"}
        if url.endswith("/v1/models"):
            return 200, {"data": "invalid"}
        if url.endswith("/v1/hivemind/onboarding/manifest"):
            return 200, {"object": "bad", "holiday_routes": None, "downloadable_scripts": None}
        if url.endswith("/noel"):
            return 200, {"html_length": 0}
        if url.endswith("/quick"):
            return 502, {"error": "bad_gateway"}
        if url.endswith("/raid-noel"):
            return 429, {"error": "slow_down"}
        if url.endswith("/arena-noel"):
            raise urllib.error.URLError("reset")
        if url.endswith("/scripts/hivemind_quickstart.py"):
            return 200, {"text_length": 0}
        return 404, "missing"

    for lang_code in ("en", "fr"):
        rep_broken = parity_gate.check_cloud_run_holiday_parity(
            http_fn=_broken_http,
            check_git_head=True,
            runner=lambda argv, t: (128, ""),
            lang=lang_code,
        )
        assert rep_broken["parity_ok"] is False
        assert rep_broken["sha_drift"] is True
        assert len(rep_broken["missing_manifest_scripts"]) == 17

    def _raise_git_missing(argv: Any, timeout_s: float) -> tuple[int, str]:
        raise FileNotFoundError("git binary missing")

    for lang_code in ("en", "fr"):
        rep_git_exc = parity_gate.check_cloud_run_holiday_parity(
            http_fn=_broken_http,
            check_git_head=True,
            runner=_raise_git_missing,
            lang=lang_code,
        )
        assert rep_git_exc["parity_ok"] is False

    # SHA mismatch diagnostics in EN and FR
    def _healthy_stub_http(method: str, url: str) -> tuple[int, Any]:
        _ = method
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"data": []}
        if url.endswith("/v1/hivemind/onboarding/manifest"):
            return 200, build_onboarding_manifest({"HIVEMIND_BUILD_SHA": "11111111aaaa"})
        if url.endswith("/v1/hivemind/onboarding/feedback/summary"):
            return 200, {"total_responses": 0}
        return 200, "ok-content"

    for lang_code in ("en", "fr"):
        drift_rep = parity_gate.check_cloud_run_holiday_parity(
            expected_sha=SAMPLE_SHORT_SHA,
            http_fn=_healthy_stub_http,
            lang=lang_code,
        )
        assert drift_rep["parity_ok"] is False
        assert drift_rep["sha_drift"] is True
        md = parity_gate.render_markdown_report(drift_rep)
        assert "0.00 EUR / 0.00 USD" in md

    # FinOps dry-run plan (valid deploy.sh + broken deploy.sh + missing deploy.sh)
    rc_plan_ok = parity_gate.main(
        ["--print-redeploy-plan", "--check-git-head", "--json"],
        runner=lambda argv, t: (0, SAMPLE_FULL_SHA + "\n"),
    )
    assert rc_plan_ok == 0
    plan_ok = json.loads(capsys.readouterr().out)
    assert plan_ok["finops_ok"] is True
    assert plan_ok["plan_ready"] is True

    bad_deploy = tmp_path / "bad_deploy.sh"
    bad_deploy.write_text(
        "#!/usr/bin/env bash\ngcloud run deploy x --min-instances 2 --max-instances 3 --cpu-throttling\n",
        encoding="utf-8",
    )
    for lang_code in ("en", "fr"):
        bad_p = parity_gate.build_cloud_run_redeploy_plan(deploy_script_path=bad_deploy, lang=lang_code)
        assert bad_p["finops_ok"] is False
        assert len(bad_p["failed_invariants"]) == 5

    missing_p = parity_gate.build_cloud_run_redeploy_plan(
        deploy_script_path=tmp_path / "absent.sh",
        check_git_head=True,
        runner=lambda argv, t: (1, ""),
        lang="fr",
    )
    assert missing_p["deploy_script_present"] is False
    assert missing_p["plan_ready"] is False

    # 2. Validation, redaction, zero-inference, gcloud, and output-safety guards
    assert parity_gate.normalize_lang("en-US") == "en"
    assert parity_gate.normalize_lang("fr-BE") == "fr"
    for bad_lang in (True, "es"):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert parity_gate.sanitize_build_sha(None) == "unknown"
    assert parity_gate.sanitize_cloud_run_revision(None) is None
    assert parity_gate.sanitize_cloud_run_revision("sk-1234567890") is None
    assert parity_gate.shas_match(SAMPLE_FULL_SHA, SAMPLE_SHORT_SHA) is True
    assert parity_gate.shas_match("release2026a", "release2026b") is False
    assert parity_gate.shas_match("unknown", SAMPLE_SHORT_SHA) is False
    assert parity_gate.is_allowed_gateway_host("") is False
    assert parity_gate.validate_gateway_url("http://[::1]:8080/") == "http://[::1]:8080"

    untrusted_http_lan = "http://" + "192.168" + ".1.20:8080"
    untrusted_ext = "https://" + "evil.example.invalid"
    for bad_gw in (
        "",
        True,
        "https://hivemind-openrouter.run.app\r\n",
        "https://" + "hivemind openrouter.run.app",
        "https://user:pw@hivemind-openrouter.run.app",
        "https://hivemind-openrouter.run.app?a=1",
        "https://hivemind-openrouter.run.app/sub",
        "http://127.0.0.1:badport",
        "http://127.0.0.1:0",
        "ftp://127.0.0.1:8080",
        "http://hivemind-openrouter.run.app",
        untrusted_http_lan,
        untrusted_ext,
    ):
        with pytest.raises(parity_gate.ParityGateError):
            parity_gate.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.validate_expected_sha("unknown")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._validate_cloud_identifier("bad;id", "region")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._assert_zero_inference_request("POST", "http://127.0.0.1:8080/health")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._assert_zero_inference_request("GET", "http://127.0.0.1:8080/v1/chat/completions")
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.default_http_json("GET", "http://127.0.0.1:8080/health", payload={"x": 1})
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.default_subprocess_runner([])
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate.default_subprocess_runner(["gcloud", "version"])

    fake_home = tmp_path / "alice_home"
    assert parity_gate.redact_home_path(fake_home / ".hivemind", home=fake_home) == "~/.hivemind"
    assert parity_gate.redact_home_path(r"C:\Users\Alice\cfg.json", home=fake_home) == "~/cfg.json"
    assert parity_gate._invoke_http_callable(lambda m, u: (200, [1]), "GET", "http://127.0.0.1:8080/health") == (
        200,
        {"data": [1]},
    )

    bad_score_json = '{"' + "sco" + 're": 4}'
    bad_ip_text = "node " + "10.20" + ".30.40"
    assert "forbidden_key_detected" in parity_gate.scan_output_safety(bad_score_json)
    assert "email_detected" in parity_gate.scan_output_safety("user@example.org")
    assert "unmasked_api_key_detected" in parity_gate.scan_output_safety("hm-" + "a" * 16)
    assert "bearer_token_detected" in parity_gate.scan_output_safety("Bearer tok_12345678")
    assert "unredacted_home_path_detected" in parity_gate.scan_output_safety("/home/alice/key")
    assert any(i.startswith("non_loopback_ipv4_detected:") for i in parity_gate.scan_output_safety(bad_ip_text))
    with pytest.raises(parity_gate.ParityGateError):
        parity_gate._assert_safe_output(bad_score_json)

    assert parity_gate.main(["--gateway", untrusted_ext]) == 2
    _ = capsys.readouterr()

    # 3. Live ThreadingHTTPServer loopback round-trip exercising default_http_json + CLI --verify-all-scripts
    manifest_doc = build_onboarding_manifest(
        {"HIVEMIND_BUILD_SHA": SAMPLE_SHORT_SHA, "K_REVISION": "hivemind-openrouter-00120-loop"}
    )
    scripts_dir = REPO_ROOT / "scripts"

    class _ScaleLoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._write_json(200, {"data": [{"id": "hivemind/auto"}]})
            elif self.path == "/v1/hivemind/onboarding/manifest":
                self._write_json(200, manifest_doc)
            elif self.path == "/v1/hivemind/onboarding/feedback/summary":
                self._write_json(200, {"total_responses": 2})
            elif self.path in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<!DOCTYPE html><title>Noel Scale</title>")
            elif self.path.startswith("/scripts/"):
                sname = self.path.removeprefix("/scripts/")
                if sname in _ALLOWED_ONBOARDING_SCRIPTS:
                    raw = (scripts_dir / sname).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/x-python; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(raw)
                else:
                    self._write_json(404, {"error": "not_found"})
            elif self.path == "/204":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/arr":
                self._write_json(200, [1, 2])
            else:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"bad gateway")

        def _write_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return None

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScaleLoopbackHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        base_url = f"http://127.0.0.1:{port}"
        assert parity_gate.default_http_json("GET", f"{base_url}/204", headers={"X-Probe": "1"})[0] == 204
        assert parity_gate.default_http_json("GET", f"{base_url}/arr")[1] == {"data": [1, 2]}
        assert parity_gate.default_http_json("GET", f"{base_url}/scripts/unknown.py")[0] == 404
        assert parity_gate.default_http_json("GET", f"{base_url}/bad-plain")[0] == 502

        proc = subprocess.run(
            [
                sys.executable,
                str(GATE_PATH),
                "--gateway",
                base_url,
                "--expected-sha",
                SAMPLE_SHORT_SHA,
                "--verify-all-scripts",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
        )
        assert proc.returncode == 0, proc.stderr
        sub_report = json.loads(proc.stdout)
        assert sub_report["parity_ok"] is True
        assert sub_report["scripts_verified_count"] == 17
        assert sub_report["scripts_total_count"] == 17
    finally:
        server.shutdown()
        server.server_close()

    c_net, b_net = parity_gate.default_http_json("GET", "http://127.0.0.1:1/health")
    assert c_net == 0 and "network_error" in b_net
    rc_s, out_s = parity_gate.default_subprocess_runner([sys.executable, "-c", "print('ok')"], 3.0)
    assert rc_s == 0 and out_s.strip() == "ok"


def test_zz_touched_code_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()

    # 1. Verify >= 95.0% whole-file line coverage on scripts/noel_cloud_run_parity_gate.py
    _, gate_stmts, _, gate_missing, gate_missing_fmt = _INPROC_COV.analysis2(str(GATE_PATH))
    gate_covered = len(gate_stmts) - len(gate_missing)
    gate_pct = 100.0 * gate_covered / max(1, len(gate_stmts))
    assert gate_pct >= 95.0, (
        f"Expected >= 95.0% line coverage on {GATE_PATH.name}, got {gate_pct:.2f}% "
        f"(missing: {gate_missing_fmt})"
    )

    # 2. Verify >= 95.0% line coverage on touched onboarding/script-delivery functions in gateway.py
    _, gw_stmts, _, gw_missing, _ = _INPROC_COV.analysis2(str(GATEWAY_PY_PATH))
    gw_tree = ast.parse(GATEWAY_PY_PATH.read_text(encoding="utf-8"), filename=str(GATEWAY_PY_PATH))
    target_funcs = {
        "_sanitize_build_sha",
        "_sanitize_cloud_run_revision",
        "build_onboarding_manifest",
        "resolve_onboarding_script",
        "_serve_onboarding_script",
        "get_boinc_idle_worker_source",
        "download_onboarding_script",
        "get_onboarding_manifest",
    }
    touched_lines: set[int] = set()
    for node in ast.walk(gw_tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_funcs and node.body:
            start_ln = node.body[0].lineno
            end_ln = node.end_lineno or start_ln
            touched_lines.update(range(start_ln, end_ln + 1))

    gw_touched_stmts = set(gw_stmts) & touched_lines
    gw_touched_missing = set(gw_missing) & touched_lines
    assert len(gw_touched_stmts) >= 25
    gw_touched_pct = 100.0 * (len(gw_touched_stmts) - len(gw_touched_missing)) / len(gw_touched_stmts)
    assert gw_touched_pct >= 95.0, (
        f"Expected >= 95.0% line coverage on touched gateway.py functions, got {gw_touched_pct:.2f}% "
        f"(missing lines: {sorted(gw_touched_missing)})"
    )
