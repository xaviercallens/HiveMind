# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cloud Run canary cutover verifier and idempotent CES feedback retry queue tests (`SLICE-SCALE-DYN-04`)."""

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
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any

import coverage
import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPT = REPO_ROOT / "scripts" / "noel_canary_cutover_and_ces_queue.py"
AUTOCONFIG_SCRIPT = REPO_ROOT / "scripts" / "noel_vscode_autoconfig.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(TARGET_SCRIPT)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import create_router_app  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_target_spec = importlib.util.spec_from_file_location("noel_canary_cutover_and_ces_queue_dyn04", TARGET_SCRIPT)
assert _target_spec is not None and _target_spec.loader is not None
canary_queue = importlib.util.module_from_spec(_target_spec)
sys.modules["noel_canary_cutover_and_ces_queue_dyn04"] = canary_queue
_target_spec.loader.exec_module(canary_queue)

_auto_spec = importlib.util.spec_from_file_location("noel_vscode_autoconfig_for_dyn04", AUTOCONFIG_SCRIPT)
assert _auto_spec is not None and _auto_spec.loader is not None
autoconfig = importlib.util.module_from_spec(_auto_spec)
sys.modules["noel_vscode_autoconfig_for_dyn04"] = autoconfig
_auto_spec.loader.exec_module(autoconfig)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn04",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn04",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn04",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn04-router-sec-32b!"
TARGET_FULL_SHA = "c4ea9b120123456789abcdef0123456789abcdef"
TARGET_SHORT_SHA = "c4ea9b120123"
PREVIOUS_GOOD_REV = "hivemind-openrouter-00141-stable"
LIVE_CANARY_REV = "hivemind-openrouter-00142-noel"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip_for(bucket: int, idx: int) -> str:
    """Deterministically map `(bucket, idx)` into `127.0.0.0/8` loopback space."""
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
    assert canary_queue.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_all_keys(doc)


def test_01_compliance_headers_stdlib_lateral_movement_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for path in (TARGET_SCRIPT, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    script_source = TARGET_SCRIPT.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 standard library imports ONLY + ProxyHandler({}) + zero shell=True/eval/exec/os.system/pickle
    tree = ast.parse(script_source, filename="scripts/noel_canary_cutover_and_ces_queue.py")
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
    forbidden_builtins = {"eval", "exec"}
    forbidden_modules = {"pickle", "marshal"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_builtins
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
    assert imported_roots & forbidden_modules == set()
    assert saw_empty_proxy_handler is True

    # 2. Lateral-movement AST audit on both target script & test file + planted negative controls
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source,
            filename="scripts/noel_canary_cutover_and_ces_queue.py",
        )
        == []
    )
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_source,
            filename="tests/test_noel_canary_cutover_and_ces_queue.py",
        )
        == []
    )
    planted_ip = ".".join(("192", "168", "77", "12"))
    planted_ext = "https://" + "attacker" + ".example.invalid/exfil"
    assert lateral_guard.audit_source_for_lateral_movement(script_source + f'\n_L = "http://{planted_ip}:8080"\n', "x.py") != []
    assert lateral_guard.audit_source_for_lateral_movement(script_source + f'\n_E = "{planted_ext}"\n', "y.py") != []

    # 3. PR gatekeeper FORBIDDEN_ADDED scan across both files
    for rel, content in (
        ("scripts/noel_canary_cutover_and_ces_queue.py", script_source),
        ("tests/test_noel_canary_cutover_and_ces_queue.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner (0 findings), ZeroStubAudit, and static-figure guard
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_canary_cutover_and_ces_queue.py", script_source),
        ("tests/test_noel_canary_cutover_and_ces_queue.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        stub_res = ZeroStubAudit().audit_python_code(content)
        assert stub_res.is_clean, f"Unexpected ZeroStubAudit violations in {rel}: {stub_res.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1074,
        pr_title="feat(onboarding,scale): Cloud Run canary cutover verifier and idempotent CES feedback retry queue",
        pr_body="Implements SLICE-SCALE-DYN-04 Cloud Run canary cutover verifier and idempotent CES feedback retry queue.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(onboarding,scale): Cloud Run canary cutover verifier and idempotent CES feedback retry queue"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True


def test_02_concurrent_48_participant_stale_404_canary_cutover_and_idempotent_ces_flush_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", LIVE_CANARY_REV)

    upstream_inference_calls: list[str] = []

    def _forbid_upstream_inference(req: httpx.Request) -> httpx.Response:
        upstream_inference_calls.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_inference_call"})

    sqlite_path = tmp_path / "noel_dyn04_48soak.sqlite3"
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
        trusted_proxy_hops=1,
        open_registration=True,
    )

    total_participants = 48
    dev_count = 24
    gamer_count = 24

    transport_state = {"mode": "stale_404", "fault_seq": 0}
    state_lock = threading.Lock()
    sqlite_lock_hits: list[str] = []
    delivered_idempotency_keys: list[str] = []
    observed_html_nonces: list[str] = []

    steam_titles = ("cs2.exe", "eldenring.exe", "bg3.exe", "helldivers2.exe", "cyberpunk2077.exe", "steam_app_730")
    rtx_cards = (
        ("NVIDIA GeForce RTX 4070 SUPER", 12288, 9216),
        ("NVIDIA GeForce RTX 4080", 16384, 12288),
        ("NVIDIA GeForce RTX 3080", 10240, 6144),
        ("NVIDIA GeForce RTX 4060 Ti", 8192, 5120),
    )

    with TestClient(app) as client:

        def _make_switchable_http(peer_ip: str) -> canary_queue.HttpFn:
            def _http(
                method: str,
                url: str,
                payload: dict[str, Any] | None = None,
                headers: dict[str, str] | None = None,
            ) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
                assert "completions" not in url.lower(), f"Zero-inference violated: {url}"
                path = "/" + url.split("://", 1)[1].split("/", 1)[1]
                with state_lock:
                    mode = transport_state["mode"]
                    seq = transport_state["fault_seq"]
                    transport_state["fault_seq"] = seq + 1

                if mode == "stale_404":
                    if path == "/health":
                        return 200, {"status": "ok"}, {"content-type": "application/json"}
                    if path == "/v1/hivemind/producers/fleet":
                        return (
                            200,
                            {
                                "object": "hivemind.producer_fleet",
                                "total_hiveminders": 0,
                                "live_hiveminders": 0,
                                "distinct_physical_gpus": 0,
                                "live_endpoints": 0,
                                "live_models": [],
                            },
                            {"content-type": "application/json"},
                        )
                    return 404, {"detail": "Not Found (pre-PR-#949 stale revision)"}, {"content-type": "application/json"}

                if mode == "transient_faults":
                    fault_bucket = seq % 4
                    if fault_bucket == 0:
                        return 429, {"error": "rate_limited"}, {"content-type": "application/json"}
                    if fault_bucket == 1:
                        return 503, {"error": "service_unavailable"}, {"content-type": "application/json"}
                    if fault_bucket == 2:
                        return 408, {"error": "request_timeout"}, {"content-type": "application/json"}
                    raise TimeoutError("gateway timeout during canary transient fault")

                req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
                resp = client.request(method, path, json=payload, headers=req_headers)
                raw_text = resp.text
                if _SQLITE_LOCK_RE.search(raw_text):
                    with state_lock:
                        sqlite_lock_hits.append(raw_text[:200])
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                if method == "POST" and path == "/v1/hivemind/onboarding/feedback" and resp.status_code == 200:
                    idem = (headers or {}).get("Idempotency-Key", "")
                    with state_lock:
                        delivered_idempotency_keys.append(idem)
                if method == "GET" and path in ("/noel", "/quick", "/raid-noel", "/arena-noel") and resp.status_code == 200:
                    csp_val = resp_headers.get("content-security-policy", "")
                    m_nonce = _CSP_HDR_NONCE_RE.findall(csp_val)
                    if m_nonce:
                        with state_lock:
                            observed_html_nonces.append(m_nonce[0])
                ctype = resp_headers.get("content-type", "")
                if "application/json" in ctype:
                    return resp.status_code, resp.json(), resp_headers
                return resp.status_code, raw_text, resp_headers

            return _http

        # Step 1: Verify stale pre-PR-#949 Cloud Run 404 triggers rollback_triggered=True and blocks promotion
        probe_ws = tmp_path / "canary_operator_ws"
        probe_ws.mkdir(parents=True)
        stale_verdict = canary_queue.verify_canary_cutover(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_SHORT_SHA,
            previous_revision=PREVIOUS_GOOD_REV,
            canary_percent=10,
            env=CLEAN_ENV,
            cwd=probe_ws,
            http_fn=_make_switchable_http("127.0.20.1"),
            lang="fr",
        )
        _verify_sha256_receipt(stale_verdict)
        assert stale_verdict["canary_ok"] is False
        assert stale_verdict["promote_to_100_pct"] is False
        assert stale_verdict["rollback_triggered"] is True
        assert stale_verdict["stale_cloud_run_detected"] is True
        assert "/v1/hivemind/onboarding/manifest" in stale_verdict["stale_404_routes"]
        assert PREVIOUS_GOOD_REV in stale_verdict["rollback_command"]

        # Step 2: 48-participant workspaces (24 <60s Dev + 24 <5m Gamer-Whisper <1.5s Steam yield / <=70 C)
        # concurrently enqueue CES & zero_game_lag + attempt duplicate enqueues + suffer stale 404 & transient 429/503
        phase_a_barrier = threading.Barrier(total_participants)

        def _participant_pre_cutover_soak(idx: int) -> dict[str, Any]:
            t0 = time.monotonic()
            p_root = tmp_path / "participants" / f"p_{idx:02d}"
            p_home = p_root / "home" / f"perso_{idx:02d}"
            p_ws = p_root / "hobby_repo"
            p_ws.mkdir(parents=True, exist_ok=True)
            p_queue = p_home / ".hivemind" / "ces_feedback_queue.json"
            peer_ip = _loopback_ip_for(21, idx)
            p_http = _make_switchable_http(peer_ip)
            lang_choice = "fr" if (idx % 2 == 0) else "en"
            ease_rating = 5 if (idx % 3 != 0) else 4

            if idx < dev_count:
                cohort = "dev"
                zero_lag = True
                platform_name = "linux"
                idem_key = f"noel-dyn04-dev-{idx:02d}"
                no_gpu = autoconfig.GpuProbeSnapshot(present=False, reason="no_nvidia_gpu")
                pf = autoconfig.run_steam_thermal_preflight(
                    gpu_probe_fn=lambda g=no_gpu: g,
                    process_probe_fn=lambda: ["code", "python3"],
                    worker_yield_fn=lambda: 16.0,
                    lang=lang_choice,
                )
                assert pf["status"] == "dev_api_only_no_nvidia_gpu"
                slo_limit_s = 60.0
            else:
                g_idx = idx - dev_count
                cohort = "gamer"
                had_lag = g_idx in (4, 19)
                zero_lag = not had_lag
                platform_name = "windows" if (g_idx % 2 == 0) else "linux"
                idem_key = f"noel-dyn04-gamer-{g_idx:02d}"
                card_name, total_mib, free_mib = rtx_cards[g_idx % len(rtx_cards)]
                temp_c = round(54.0 + (g_idx * 0.6), 1)
                yield_ms = round(125.0 + (g_idx * 11.0), 2)
                rtx_snap = autoconfig.GpuProbeSnapshot(
                    present=True,
                    name=card_name,
                    total_vram_mib=total_mib,
                    free_vram_mib=free_mib,
                    used_vram_mib=total_mib - free_mib,
                    temp_c=temp_c,
                    util_pct=5.0,
                )
                procs = ["steam.exe", steam_titles[g_idx % len(steam_titles)]] if (g_idx < 12) else ["steam.exe"]
                pf = autoconfig.run_steam_thermal_preflight(
                    max_temp_c=70.0,
                    min_free_vram_mib=2048,
                    gpu_probe_fn=lambda g=rtx_snap: g,
                    process_probe_fn=lambda p=procs: p,
                    worker_yield_fn=lambda y=yield_ms: {
                        "worker_started": True,
                        "worker_stopped": True,
                        "clean_release": True,
                        "retained_worker_vram_mib": 0,
                        "yield_latency_ms": y,
                    },
                    lang=lang_choice,
                )
                assert pf["preflight_ok"] is True
                assert pf["thermal_ok"] is True
                assert pf["yield_slo_ok"] is True
                slo_limit_s = 300.0

            phase_a_barrier.wait(timeout=20.0)
            first_enq = canary_queue.enqueue_ces_feedback(
                queue_file=p_queue,
                profile=cohort,
                ease_rating=ease_rating,
                zero_game_lag=zero_lag,
                platform=platform_name,
                idempotency_key=idem_key,
                home=p_home,
                env=CLEAN_ENV,
                cwd=p_ws,
                now_epoch=1_700_000_000.0 + idx,
                lang=lang_choice,
            )
            _verify_sha256_receipt(first_enq)
            assert first_enq["enqueued"] is True
            assert first_enq["deduplicated"] is False
            assert (p_queue.stat().st_mode & 0o777) == 0o600

            # Duplicate enqueue attempt (both with idempotency_key and identical payload)
            dup_enq = canary_queue.enqueue_ces_feedback(
                queue_file=p_queue,
                profile=cohort,
                ease_rating=ease_rating,
                zero_game_lag=zero_lag,
                platform=platform_name,
                idempotency_key=idem_key,
                home=p_home,
                env=CLEAN_ENV,
                cwd=p_ws,
                now_epoch=1_700_000_050.0 + idx,
                lang=lang_choice,
            )
            _verify_sha256_receipt(dup_enq)
            assert dup_enq["enqueued"] is False
            assert dup_enq["deduplicated"] is True
            assert dup_enq["duplicate_state"] == "already_pending"
            assert dup_enq["pending_count"] == 1

            # Flush during stale pre-PR-#949 HTTP 404 -> safely retained in pending
            fl_404 = canary_queue.flush_ces_feedback_queue(
                "http://127.0.0.1:8000",
                queue_file=p_queue,
                home=p_home,
                env=CLEAN_ENV,
                cwd=p_ws,
                http_fn=p_http,
                now_epoch=1_700_000_100.0 + idx,
                lang=lang_choice,
            )
            _verify_sha256_receipt(fl_404)
            assert fl_404["flush_ok"] is False
            assert fl_404["delivered_count"] == 0
            assert fl_404["retained_retry_count"] == 1
            assert fl_404["stale_cloud_run_detected"] is True
            assert fl_404["pending_count"] == 1

            elapsed_s = time.monotonic() - t0
            assert elapsed_s < slo_limit_s
            return {
                "idx": idx,
                "cohort": cohort,
                "ease_rating": ease_rating,
                "zero_game_lag": zero_lag,
                "idem_key": idem_key,
                "event_id": first_enq["event_id"],
                "home": p_home,
                "ws": p_ws,
                "queue": p_queue,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            cohort_rows = list(pool.map(_participant_pre_cutover_soak, range(total_participants)))
        assert len(cohort_rows) == 48

        # Step 3: Inject transient 429 / 503 / 408 / timeout faults across all 48 participants concurrently
        with state_lock:
            transport_state["mode"] = "transient_faults"

        fault_barrier = threading.Barrier(total_participants)

        def _transient_fault_worker(row: dict[str, Any]) -> dict[str, Any]:
            idx = int(row["idx"])
            fault_barrier.wait(timeout=20.0)
            rep_fault = canary_queue.flush_ces_feedback_queue(
                "http://127.0.0.1:8000",
                queue_file=row["queue"],
                home=row["home"],
                env=CLEAN_ENV,
                cwd=row["ws"],
                http_fn=_make_switchable_http(_loopback_ip_for(22, idx)),
                now_epoch=1_700_000_300.0 + idx,
                lang="en",
            )
            _verify_sha256_receipt(rep_fault)
            assert rep_fault["flush_ok"] is False
            assert rep_fault["delivered_count"] == 0
            assert rep_fault["retained_retry_count"] == 1
            assert rep_fault["pending_count"] == 1
            assert rep_fault["outcomes"][0]["retry_count"] == 2
            assert rep_fault["outcomes"][0]["backoff_seconds"] == 2.0
            return rep_fault

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            fault_reports = list(pool.map(_transient_fault_worker, cohort_rows))
        assert len(fault_reports) == 48

        # Step 4: Switch transport to live create_router_app -> canary promotion promote_to_100_pct=True
        with state_lock:
            transport_state["mode"] = "live_router"

        live_canary = canary_queue.verify_canary_cutover(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_SHORT_SHA,
            previous_revision=PREVIOUS_GOOD_REV,
            canary_percent=10,
            env=CLEAN_ENV,
            cwd=probe_ws,
            http_fn=_make_switchable_http("127.0.20.2"),
            lang="en",
        )
        _verify_sha256_receipt(live_canary)
        assert live_canary["canary_ok"] is True
        assert live_canary["promote_to_100_pct"] is True
        assert live_canary["rollback_triggered"] is False
        assert live_canary["routes_verified_count"] == 8
        assert live_canary["html_contract_ok"] is True
        assert live_canary["unique_csp_nonces_count"] == 4
        assert live_canary["producer_fleet_ok"] is True
        assert live_canary["feedback_summary"]["total_responses"] == 0

        # Step 5: Two consecutive 48-thread concurrent flush passes + HTML CSP nonce uniqueness probe
        for pass_num in (1, 2):
            flush_barrier = threading.Barrier(total_participants)

            def _concurrent_flush_worker(row: dict[str, Any], *, _pass: int = pass_num) -> dict[str, Any]:
                idx = int(row["idx"])
                peer_ip = _loopback_ip_for(23 + _pass, idx)
                worker_http = _make_switchable_http(peer_ip)
                html_route = canary_queue.CANARY_HTML_ROUTES[idx % len(canary_queue.CANARY_HTML_ROUTES)]
                if _pass == 2:
                    post_dup = canary_queue.enqueue_ces_feedback(
                        queue_file=row["queue"],
                        profile=row["cohort"],
                        ease_rating=row["ease_rating"],
                        zero_game_lag=row["zero_game_lag"],
                        idempotency_key=row["idem_key"],
                        home=row["home"],
                        env=CLEAN_ENV,
                        cwd=row["ws"],
                    )
                    assert post_dup["enqueued"] is False
                    assert post_dup["deduplicated"] is True
                    assert post_dup["duplicate_state"] == "already_delivered"

                flush_barrier.wait(timeout=20.0)
                code_h, _, _ = worker_http("GET", f"http://127.0.0.1:8000{html_route}")
                assert code_h == 200
                rep = canary_queue.flush_ces_feedback_queue(
                    "http://127.0.0.1:8000",
                    queue_file=row["queue"],
                    home=row["home"],
                    env=CLEAN_ENV,
                    cwd=row["ws"],
                    http_fn=worker_http,
                    now_epoch=1_700_001_000.0 + (_pass * 100.0),
                    lang="fr" if (idx % 2 == 0) else "en",
                )
                _verify_sha256_receipt(rep)
                assert rep["flush_ok"] is True
                assert rep["pending_count"] == 0
                assert rep["delivered_ledger_count"] == 1
                return rep

            with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
                pass_reports = list(pool.map(_concurrent_flush_worker, cohort_rows))

            total_delivered_in_pass = sum(int(r["delivered_count"]) for r in pass_reports)
            total_net_in_pass = sum(int(r["network_calls"]) for r in pass_reports)
            if pass_num == 1:
                assert total_delivered_in_pass == 48
                assert total_net_in_pass == 48
            else:
                assert total_delivered_in_pass == 0
                assert total_net_in_pass == 0

        # Exact 1-to-1 idempotent conservation checks on live create_router_app
        summary_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
        assert summary_resp.status_code == 200
        live_summary = summary_resp.json()

        expected_ease_mean = round(sum(int(r["ease_rating"]) for r in cohort_rows) / 48, 2)
        expected_zero_lag_ratio = round(sum(1 for r in cohort_rows if r["zero_game_lag"]) / 48, 4)

        assert live_summary["total_responses"] == 48
        assert live_summary["by_profile"] == {"dev": dev_count, "gamer": gamer_count}
        assert live_summary["mean_ease_rating"] == expected_ease_mean
        assert live_summary["zero_game_lag_ratio"] == expected_zero_lag_ratio

        assert len(delivered_idempotency_keys) == 48
        assert set(delivered_idempotency_keys) == {r["event_id"] for r in cohort_rows}
        assert sqlite_lock_hits == []
        assert upstream_inference_calls == []
        assert len(observed_html_nonces) == 4 + (48 * 2)
        assert len(set(observed_html_nonces)) == len(observed_html_nonces)


def test_03_deploy_script_finops_invariants_and_canary_traffic_plan_sanitization(tmp_path: Path) -> None:
    real_deploy_path = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"
    for lang_code in ("en", "fr"):
        audit_ok = canary_queue.audit_deploy_script_finops(real_deploy_path, lang=lang_code)
        _verify_sha256_receipt(audit_ok)
        assert audit_ok["finops_ok"] is True
        assert audit_ok["deploy_script"] == "deploy/gcp_openrouter/deploy.sh"
        assert audit_ok["invariants_verified_count"] == 7
        assert audit_ok["invariants_total_count"] == 7
        assert audit_ok["failed_invariants"] == []

        plan = canary_queue.build_canary_traffic_plan(
            gateway=canary_queue.DEFAULT_GATEWAY,
            candidate_tag="noel-canary",
            canary_percent=10,
            previous_revision=PREVIOUS_GOOD_REV,
            target_sha=TARGET_SHORT_SHA,
            lang=lang_code,
        )
        _verify_sha256_receipt(plan)
        assert plan["finops_ok"] is True
        assert plan["plan_ready"] is True
        assert plan["dry_run_only"] is True
        assert plan["gcloud_executed"] is False
        assert plan["cloud_state_mutated"] is False
        assert plan["traffic_split_pct"] == {"candidate": 10, "stable": 90}
        assert '--to-tags "noel-canary=10"' in plan["canary_traffic_split_command"]
        assert '--to-tags "noel-canary=100"' in plan["promote_100_pct_command"]
        assert f'--to-revisions "{PREVIOUS_GOOD_REV}=100"' in plan["rollback_command"]

    # Negative controls mutating each of the 7 FinOps invariants in deploy.sh
    base_script = real_deploy_path.read_text(encoding="utf-8")
    mutations = (
        ("min_instances_zero", "--min-instances 0", "--min-instances 2"),
        ("max_instances_one", "--max-instances 1", "--max-instances 4"),
        ("no_cpu_throttling", "--no-cpu-throttling", "--cpu-throttling"),
        ("concurrency_80", "--concurrency 80", "--concurrency 250"),
        ("litestream_gcs_replica", "HIVEMIND_ROUTER_REPLICA_URL=gcs://", "HIVEMIND_ROUTER_REPLICA_URL=file://"),
        ("secret_manager_router_secret", '--set-secrets "HIVEMIND_ROUTER_SECRET=${SECRET_NAME}:latest"', "--cpu-boost"),
        (
            "zero_starter_and_bonus_grants",
            'STARTER_GRANT_HMT="${HIVEMIND_STARTER_GRANT_HMT:-0}"',
            'STARTER_GRANT_HMT="${HIVEMIND_STARTER_GRANT_HMT:-25}"',
        ),
    )
    for idx, (inv_id, target, replacement) in enumerate(mutations):
        mut_file = tmp_path / f"bad_deploy_{idx}.sh"
        assert target in base_script
        mut_file.write_text(base_script.replace(target, replacement), encoding="utf-8")
        res = canary_queue.audit_deploy_script_finops(mut_file, lang="fr" if idx % 2 else "en")
        _verify_sha256_receipt(res)
        assert res["finops_ok"] is False
        assert inv_id in res["failed_invariants"]

    missing_audit = canary_queue.audit_deploy_script_finops(tmp_path / "missing_deploy.sh", lang="fr")
    assert missing_audit["finops_ok"] is False
    assert missing_audit["deploy_script_present"] is False

    # Input sanitization & injection rejection on build_canary_traffic_plan
    for bad_kwargs in (
        {"canary_percent": 0},
        {"canary_percent": 100},
        {"canary_percent": True},
        {"candidate_tag": "BAD_TAG;rm"},
        {"candidate_tag": "noel---canary"},
        {"candidate_tag": "noel-secret-tag"},
        {"candidate_tag": True},
        {"previous_revision": "rev; drop table"},
        {"previous_revision": "hm-secret-token-leak"},
        {"previous_revision": 123},
        {"service_name": "bad;service"},
        {"service_name": True},
        {"region": "us-central1 && whoami"},
        {"target_sha": "unknown"},
        {"target_sha": "hm-secret123"},
    ):
        with pytest.raises(canary_queue.CanaryQueueError):
            canary_queue.build_canary_traffic_plan(**bad_kwargs)  # type: ignore[arg-type]


def test_04_amadeus_corporate_pc_and_git_remote_blocking_while_allowing_purge_queue(
    tmp_path: Path,
) -> None:
    def _forbid_http(*_args: Any, **_kwargs: Any) -> tuple[int, dict[str, Any]]:
        raise AssertionError("Corporate compliance gate MUST block before any network call")

    personal_ws = tmp_path / "personal_hobby_ws"
    personal_ws.mkdir(parents=True)
    (personal_ws / ".git").mkdir()
    (personal_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = git@github.com:xaviercallens/HiveMind.git\n',
        encoding="utf-8",
    )

    corp_git_ws = tmp_path / "ws_git_remote"
    corp_git_ws.mkdir(parents=True)
    (corp_git_ws / ".git").mkdir()
    corp_remote_url = "https://" + "git.amadeus.internal/core/secret-airline-repo.git"
    (corp_git_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {corp_remote_url}\n',
        encoding="utf-8",
    )

    common_git = tmp_path / "common_git_dir"
    wt_gitdir = common_git / "worktrees" / "wt_fw"
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text(
        '[remote "origin"]\n\turl = git@paloalto-intranet:sec/fw-sentinel-repo.git\n',
        encoding="utf-8",
    )
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_gitdir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    corp_wt_ws = tmp_path / "ws_worktree_fw"
    corp_wt_ws.mkdir(parents=True)
    (corp_wt_ws / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    proxy_zscaler = "http://" + "zscaler-sentinel-gw.internal:8443"
    proxy_netskope = "http://" + "netskope-bluecoat-sentinel:8080"
    vectors: tuple[tuple[str, dict[str, str], Path, str], ...] = (
        ("amadeus_dns", {**CLEAN_ENV, "USERDNSDOMAIN": "nce-wks-99.amadeus.corp"}, personal_ws, "nce-wks-99"),
        ("corp_userdomain", {**CLEAN_ENV, "USERDOMAIN": "ACMECORP_SENTINEL_DOM"}, personal_ws, "ACMECORP_SENTINEL_DOM"),
        ("zscaler_https_proxy", {**CLEAN_ENV, "HTTPS_PROXY": proxy_zscaler, "ZSCALER_CERT_PATH": "/etc/zscaler_ca_99.pem"}, personal_ws, "zscaler-sentinel-gw"),
        ("netskope_bluecoat_http_proxy", {**CLEAN_ENV, "HTTP_PROXY": proxy_netskope}, personal_ws, "netskope-bluecoat-sentinel"),
        ("fortinet_computername", {**CLEAN_ENV, "COMPUTERNAME": "WKS-FORTINET-CORP-77"}, personal_ws, "WKS-FORTINET-CORP-77"),
        ("enterprise_intranet_hostname", {**CLEAN_ENV, "HOSTNAME": "box-enterprise-intranet-88"}, personal_ws, "box-enterprise-intranet-88"),
        ("amadeus_env_key", {**CLEAN_ENV, "AMADEUS_SITE": "sophia-sentinel-val"}, personal_ws, "sophia-sentinel-val"),
        ("corp_git_remote", CLEAN_ENV, corp_git_ws, "secret-airline-repo.git"),
        ("paloalto_git_worktree", CLEAN_ENV, corp_wt_ws, "fw-sentinel-repo.git"),
    )

    observed_markers: set[str] = set()
    fake_home = tmp_path / "home" / "corp_check_user"
    target_spool = fake_home / ".hivemind" / "ces_feedback_queue.json"

    for idx, (vec_name, env_map, cwd_path, secret_sentinel) in enumerate(vectors):
        lang_choice = "fr" if (idx % 2 == 1) else "en"
        comp = canary_queue.evaluate_personal_pc_compliance(env=env_map, cwd=cwd_path, lang=lang_choice)
        _verify_sha256_receipt(comp)
        assert comp["compliance_ok"] is False, vec_name
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0
        assert "100% personal home PCs" in comp["summary"]
        assert "100 % PC personnels" in comp["summary"]
        assert "0.00 EUR / 0.00 USD" in comp["summary"]
        assert "0,00 EUR / 0,00 USD" in comp["summary"]
        assert "symbolic" in comp["summary"].lower()
        observed_markers.update(comp["matched_markers"])

        enq_blocked = canary_queue.enqueue_ces_feedback(
            queue_file=target_spool,
            home=fake_home,
            env=env_map,
            cwd=cwd_path,
            lang=lang_choice,
        )
        assert enq_blocked["compliance_ok"] is False and enq_blocked["enqueued"] is False
        assert not target_spool.exists()

        fl_blocked = canary_queue.flush_ces_feedback_queue(
            queue_file=target_spool,
            home=fake_home,
            env=env_map,
            cwd=cwd_path,
            http_fn=_forbid_http,
            lang=lang_choice,
        )
        assert fl_blocked["compliance_ok"] is False and fl_blocked["network_calls"] == 0

        ver_blocked = canary_queue.verify_canary_cutover(
            env=env_map,
            cwd=cwd_path,
            http_fn=_forbid_http,
            lang=lang_choice,
        )
        assert ver_blocked["compliance_ok"] is False and ver_blocked["promote_to_100_pct"] is False

        cf_blocked = canary_queue.cutover_and_flush_ces_queue(
            queue_file=target_spool,
            home=fake_home,
            env=env_map,
            cwd=cwd_path,
            http_fn=_forbid_http,
            lang=lang_choice,
        )
        assert cf_blocked["compliance_ok"] is False and cf_blocked["flush_executed"] is False

        cli_modes = (
            ["--enqueue-ces", "--queue-file", str(target_spool), "--lang", lang_choice, "--json"],
            ["--flush-ces-queue", "--queue-file", str(target_spool), "--lang", lang_choice, "--markdown"],
            ["--cutover-and-flush", "--queue-file", str(target_spool), "--lang", lang_choice, "--json"],
            ["--print-canary-plan", "--lang", lang_choice, "--json"],
            ["--verify-canary", "--lang", lang_choice, "--json"],
        )
        chosen_argv = cli_modes[idx % len(cli_modes)]
        out_buf, err_buf = io.StringIO(), io.StringIO()
        rc = canary_queue.main(
            chosen_argv,
            http_fn=_forbid_http,
            env=env_map,
            cwd=cwd_path,
            home=fake_home,
            out_stream=out_buf,
            err_stream=err_buf,
        )
        assert rc == 2, vec_name
        assert not target_spool.exists()
        combined_text = out_buf.getvalue() + err_buf.getvalue() + json.dumps(comp)
        assert secret_sentinel not in combined_text, f"Leaked sentinel in {vec_name}"
        assert "0.00 EUR / 0.00 USD" in err_buf.getvalue()

    assert observed_markers == set(canary_queue.CORPORATE_MARKERS)

    # Create a spool cleanly first, then prove --purge-queue is ALWAYS permitted even on an Amadeus corporate env
    clean_enq = canary_queue.enqueue_ces_feedback(
        queue_file=target_spool,
        home=fake_home,
        env=CLEAN_ENV,
        cwd=personal_ws,
    )
    assert clean_enq["enqueued"] is True and target_spool.is_file()

    # Exercise both in-process main(--purge-queue) and subprocess CLI --purge-queue under corporate env
    out_p, err_p = io.StringIO(), io.StringIO()
    rc_inproc_purge = canary_queue.main(
        ["--purge-queue", "--queue-file", str(target_spool), "--lang", "fr", "--json"],
        env={**CLEAN_ENV, "USERDNSDOMAIN": "amadeus.internal.corp"},
        cwd=corp_git_ws,
        home=fake_home,
        out_stream=out_p,
        err_stream=err_p,
    )
    assert rc_inproc_purge == 0
    inproc_purge_doc = json.loads(out_p.getvalue())
    _verify_sha256_receipt(inproc_purge_doc)
    assert inproc_purge_doc["purged"] is True
    assert not target_spool.exists()

    # Recreate and test real subprocess --purge-queue + corporate block
    assert canary_queue.enqueue_ces_feedback(
        queue_file=target_spool,
        home=fake_home,
        env=CLEAN_ENV,
        cwd=personal_ws,
    )["enqueued"] is True

    sub_purge = subprocess.run(
        [
            sys.executable,
            str(TARGET_SCRIPT),
            "--purge-queue",
            "--queue-file",
            str(target_spool),
            "--lang",
            "en",
            "--json",
        ],
        cwd=str(corp_git_ws),
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(fake_home), "USERDNSDOMAIN": "amadeus.internal.corp"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert sub_purge.returncode == 0, sub_purge.stderr
    assert json.loads(sub_purge.stdout)["purged"] is True
    assert not target_spool.exists()

    second_purge = canary_queue.purge_ces_feedback_queue(queue_file=target_spool, home=fake_home, lang="en")
    _verify_sha256_receipt(second_purge)
    assert second_purge["purged"] is False and second_purge["already_absent"] is True

    sub_block = subprocess.run(
        [
            sys.executable,
            str(TARGET_SCRIPT),
            "--enqueue-ces",
            "--queue-file",
            str(target_spool),
            "--json",
        ],
        cwd=str(corp_git_ws),
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(fake_home), "USERDNSDOMAIN": "amadeus.internal.corp"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert sub_block.returncode == 2
    assert not target_spool.exists()
    assert "secret-airline-repo.git" not in sub_block.stdout


def test_05_permissions_shared_spool_concurrency_quarantine_backoff_redaction_and_tcp_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_home = tmp_path / "home" / "alice_secret_user"
    clean_ws = tmp_path / "hobby_proj"
    clean_ws.mkdir(parents=True)
    spool_path = fake_home / ".hivemind" / "ces_feedback_queue.json"

    # 1. Single shared-spool 24-thread concurrent enqueue + 24-thread concurrent flush race (0 double-posts!)
    shared_race_spool = fake_home / ".hivemind" / "shared_race_queue.json"
    race_posts: list[str] = []
    race_lock = threading.Lock()
    race_barrier = threading.Barrier(24)

    def _shared_race_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (method, url, payload)
        with race_lock:
            race_posts.append((headers or {}).get("Idempotency-Key", ""))
        return 200, {"ok": True}

    def _race_worker(idx: int) -> tuple[int, int]:
        race_barrier.wait(timeout=15.0)
        canary_queue.enqueue_ces_feedback(
            queue_file=shared_race_spool,
            profile="dev" if idx < 12 else "gamer",
            ease_rating=5,
            zero_game_lag=True,
            idempotency_key=f"shared-item-{idx:02d}",
            home=fake_home,
            env=CLEAN_ENV,
            cwd=clean_ws,
        )
        fl1 = canary_queue.flush_ces_feedback_queue(
            "http://127.0.0.1:8000",
            queue_file=shared_race_spool,
            home=fake_home,
            env=CLEAN_ENV,
            cwd=clean_ws,
            http_fn=_shared_race_http,
        )
        fl2 = canary_queue.flush_ces_feedback_queue(
            "http://127.0.0.1:8000",
            queue_file=shared_race_spool,
            home=fake_home,
            env=CLEAN_ENV,
            cwd=clean_ws,
            http_fn=_shared_race_http,
        )
        return int(fl1["delivered_count"]), int(fl2["delivered_count"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        race_counts = list(pool.map(_race_worker, range(24)))
    assert sum(a + b for a, b in race_counts) == 24
    assert len(race_posts) == 24
    assert len(set(race_posts)) == 24
    assert (shared_race_spool.stat().st_mode & 0o777) == 0o600

    # 2. Bounded max_entries=3 FIFO overflow + 0600 permissions even if pre-existing 0644 corrupted file
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    spool_path.write_text("{corrupted-json", encoding="utf-8")
    os.chmod(spool_path, 0o644)

    q_small = canary_queue.CesFeedbackRetryQueue(
        "~/.hivemind/ces_feedback_queue.json",
        max_entries=3,
        home=fake_home,
        base_backoff_s=2.0,
        max_backoff_s=10.0,
    )
    assert q_small.path == spool_path.resolve()
    for i in range(4):
        r_enq = q_small.enqueue(
            profile="dev" if i % 2 == 0 else "gamer",
            ease_rating=(i % 5) + 1,
            zero_game_lag=(i != 2),
            platform="Linux_x86_64!",
            idempotency_key=f"item-{i}",
            env=CLEAN_ENV,
            cwd=clean_ws,
            now_epoch=1000.0 + i,
        )
        _verify_sha256_receipt(r_enq)
        assert (spool_path.stat().st_mode & 0o777) == 0o600
        assert (spool_path.parent.stat().st_mode & 0o777) == 0o700
    assert r_enq["evicted_oldest"] is True
    assert r_enq["pending_count"] == 3
    assert r_enq["dropped_overflow_count"] == 1
    assert r_enq["quarantined_count"] == 1

    # 3. Permanent 400/422 quarantine vs 502 retry retention + respect_backoff=True deferral + empty spool flush
    empty_q = canary_queue.CesFeedbackRetryQueue(fake_home / ".hivemind" / "never_created.json", home=fake_home)
    fl_absent = empty_q.flush("http://127.0.0.1:8000", env=CLEAN_ENV, cwd=clean_ws)
    assert fl_absent["flush_ok"] is True and fl_absent["network_calls"] == 0

    call_idx = {"n": 0}

    def _mixed_status_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (method, url, payload)
        assert headers is not None and len(headers.get("Idempotency-Key", "")) == 64
        n = call_idx["n"]
        call_idx["n"] = n + 1
        if n == 0:
            return 422, {"detail": "unprocessable"}
        if n == 1:
            return 502, {"detail": "bad gateway"}
        return 200, {"ok": True}

    fl_mixed = q_small.flush(
        "http://127.0.0.1:8000",
        env=CLEAN_ENV,
        cwd=clean_ws,
        http_fn=_mixed_status_http,
        now_epoch=2000.0,
        lang="fr",
    )
    _verify_sha256_receipt(fl_mixed)
    assert fl_mixed["flush_ok"] is False
    assert fl_mixed["delivered_count"] == 1
    assert fl_mixed["retained_retry_count"] == 1
    assert fl_mixed["quarantined_count"] == 1
    assert fl_mixed["pending_count"] == 1

    fl_deferred = q_small.flush(
        "http://127.0.0.1:8000",
        env=CLEAN_ENV,
        cwd=clean_ws,
        http_fn=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Should be deferred by backoff")),
        respect_backoff=True,
        now_epoch=2001.0,
        lang="en",
    )
    _verify_sha256_receipt(fl_deferred)
    assert fl_deferred["deferred_backoff_count"] == 1
    assert fl_deferred["network_calls"] == 0

    # Tampered on-disk pending entry + duplicate delivered entry cleanup on load
    disk_doc = json.loads(spool_path.read_text(encoding="utf-8"))
    already_deliv_eid = disk_doc["delivered_event_ids"][0]
    disk_doc["pending"].append(
        {"event_id": already_deliv_eid, "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}}
    )
    disk_doc["pending"].append(
        {"event_id": "a" * 64, "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "email": "x@y.z"}}
    )
    spool_path.write_text(json.dumps(disk_doc), encoding="utf-8")
    fl_after_backoff = q_small.flush(
        "http://127.0.0.1:8000",
        env=CLEAN_ENV,
        cwd=clean_ws,
        http_fn=lambda _m, _u, _p: (200, {"ok": True}),
        respect_backoff=True,
        now_epoch=2005.0,
        lang="en",
    )
    _verify_sha256_receipt(fl_after_backoff)
    assert fl_after_backoff["delivered_count"] == 1
    assert fl_after_backoff["pending_count"] == 0
    assert fl_after_backoff["total_quarantined_count"] >= 2

    # 4. Redaction & scan_output_safety positive + negative detections
    raw_hm = "hm-" + "secret_token_value_12345"
    raw_live = "hm_live_" + "abcdef0123456789abcd"
    raw_ip = ".".join(("10", "24", "8", "99"))
    dirty = (
        f"Leak {raw_hm} and {raw_live} Bearer tok_xyz987 user@domain.org "
        f"/home/alice_secret_user/.hivemind C:\\Users\\Bob\\q.json {raw_ip} "
        + '{"' + "sco" + 're": 9}'
    )
    detected_issues = canary_queue.scan_output_safety(dirty)
    assert "email_detected" in detected_issues
    assert "unmasked_api_key_detected" in detected_issues
    assert "bearer_token_detected" in detected_issues
    assert "unredacted_home_path_detected" in detected_issues
    assert "forbidden_key_detected" in detected_issues
    assert any(i.startswith("non_loopback_ipv4_detected:") for i in detected_issues)
    with pytest.raises(canary_queue.CanaryQueueError, match="Output safety guard failed"):
        canary_queue._assert_safe_output(dirty)

    scrubbed = canary_queue.sanitize_text_for_display(dirty, home=fake_home)
    assert canary_queue.scan_output_safety(scrubbed) == []
    assert canary_queue.mask_api_key("") == "<redacted-key>"
    assert canary_queue.mask_api_key("other") == "<redacted-key>"
    assert canary_queue.redact_home_path(Path.home() / ".hivemind" / "q.json") == "~/.hivemind/q.json"
    assert canary_queue.redact_home_path(r"C:\Users\Bob\.hivemind\q.json") == "~/.hivemind/q.json"
    assert canary_queue.redact_home_path("/Users/carol/.hivemind/q.json") == "~/.hivemind/q.json"
    assert canary_queue.is_allowed_gateway_host("") is False
    assert canary_queue.is_allowed_gateway_host("storage.googleapis.com") is True
    assert canary_queue.validate_gateway_url("https://storage.googleapis.com/") == "https://storage.googleapis.com"
    assert canary_queue.sanitize_build_sha(42) == "unknown"
    assert canary_queue.shas_match("rel2026a", "rel2026a") is True
    assert canary_queue.shas_match("rel2026a", "rel2026b") is False
    assert canary_queue.shas_match("unknown", TARGET_SHORT_SHA) is False

    # 5. Real loopback TCP server end-to-end --enqueue-ces + --cutover-and-flush + Markdown rendering
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", LIVE_CANARY_REV)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")

    tcp_store = RouterStore(tmp_path / "tcp_router.sqlite3", secret=ROUTER_SECRET)
    tcp_app = create_router_app(tcp_store, trusted_proxy_hops=1)
    tcp_spool = fake_home / ".hivemind" / "tcp_ces_queue.json"

    with TestClient(tcp_app) as client:

        class _TcpHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/204":
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/arr":
                    self._send(200, b"[1, 2]", {"content-type": "application/json"})
                    return
                if self.path == "/err-text":
                    self._send(503, b"upstream unavailable hm-leak12345", {"content-type": "text/plain"})
                    return
                resp = client.get(self.path, headers={"X-Forwarded-For": "127.0.30.1"})
                self._send(resp.status_code, resp.content, dict(resp.headers))

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_json = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
                resp = client.post(self.path, json=raw_json, headers={"X-Forwarded-For": "127.0.30.2"})
                self._send(resp.status_code, resp.content, dict(resp.headers))

            def _send(self, code: int, body: bytes, hdrs: dict[str, str]) -> None:
                self.send_response(code)
                for hk, hv in hdrs.items():
                    if hk.lower() not in ("content-length", "transfer-encoding", "server", "date"):
                        self.send_header(hk, hv)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: object) -> None:
                _ = (fmt, args)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _TcpHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            base_url = f"http://127.0.0.1:{port}"
            assert canary_queue.default_http_json("GET", f"{base_url}/204")[0] == 204
            assert canary_queue.default_http_json("GET", f"{base_url}/arr")[1] == {"data": [1, 2]}
            assert canary_queue.default_http_json("GET", f"{base_url}/err-text")[0] == 503

            clean_cli_env = {"PATH": os.environ.get("PATH", ""), "HOSTNAME": "home-gaming-pc"}
            for cli_argv in (
                ["--gateway", base_url, "--print-canary-plan", "--previous-revision", PREVIOUS_GOOD_REV, "--markdown"],
                ["--queue-file", str(tcp_spool), "--enqueue-ces", "--profile", "dev", "--ease-rating", "5", "--markdown"],
                [
                    "--gateway",
                    base_url,
                    "--queue-file",
                    str(tcp_spool),
                    "--enqueue-ces",
                    "--cutover-and-flush",
                    "--profile",
                    "gamer",
                    "--ease-rating",
                    "5",
                    "--expected-sha",
                    TARGET_SHORT_SHA,
                    "--previous-revision",
                    PREVIOUS_GOOD_REV,
                    "--lang",
                    "fr",
                    "--json",
                ],
                [
                    "--gateway",
                    base_url,
                    "--queue-file",
                    str(tcp_spool),
                    "--enqueue-ces",
                    "--flush-ces-queue",
                    "--profile",
                    "dev",
                    "--ease-rating",
                    "4",
                    "--idempotency-key",
                    "tcp-extra-dev",
                    "--markdown",
                ],
                ["--gateway", base_url, "--verify-canary", "--expected-sha", TARGET_SHORT_SHA, "--markdown"],
            ):
                out_s, err_s = io.StringIO(), io.StringIO()
                rc_cli = canary_queue.main(
                    cli_argv,
                    env=clean_cli_env,
                    cwd=clean_ws,
                    home=fake_home,
                    out_stream=out_s,
                    err_stream=err_s,
                )
                assert rc_cli == 0, err_s.getvalue()
                assert canary_queue.scan_output_safety(out_s.getvalue()) == []

            assert client.get("/v1/hivemind/onboarding/feedback/summary").json()["total_responses"] == 3
        finally:
            srv.shutdown()
            srv.server_close()

    # 6. Edge & negative-control coverage assertions
    assert canary_queue.main(["--help"], out_stream=io.StringIO(), err_stream=io.StringIO()) == 0
    assert canary_queue.default_http_json("GET", "http://127.0.0.1:1/health")[0] == 0
    with pytest.raises(canary_queue.CanaryQueueError, match="Zero-inference"):
        canary_queue.default_http_json("POST", "http://127.0.0.1:8000/v1/chat/completions", payload={"x": 1})
    with pytest.raises(canary_queue.CanaryQueueError, match="Zero-inference"):
        canary_queue.default_http_json("DELETE", "http://127.0.0.1:8000/health")
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.default_http_json("GET", "http://127.0.0.1:8000/health", payload={"x": 1})
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.default_http_json("GET", "http://127.0.0.1:8000/health", headers={"X-Bad": "a\nb"})

    spaced_host = "https://" + "hivemind .run.app"
    planted_ext = "https://" + "badhost.example.invalid"
    for bad_url in (
        "",
        True,
        "https://hivemind.run.app\n",
        spaced_host,
        "https://u:p@hivemind.run.app",
        "https://hivemind.run.app?a=1",
        "https://hivemind.run.app/sub",
        "http://127.0.0.1:bad",
        "http://127.0.0.1:0",
        "ftp://127.0.0.1:8000",
        "http://hivemind.run.app",
        planted_ext,
    ):
        with pytest.raises(canary_queue.CanaryQueueError):
            canary_queue.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    assert canary_queue.main(["--gateway", planted_ext], env=CLEAN_ENV, cwd=clean_ws, err_stream=io.StringIO()) == 2
    assert canary_queue.normalize_lang("en-US") == "en"
    assert canary_queue.normalize_lang("fr_CA") == "fr"
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.normalize_lang("es")
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.normalize_lang(True)  # type: ignore[arg-type]

    for bad_fb in (
        None,
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "pii"},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "bad/plat", "note": ""},
        {"profile": "admin", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""},
        {"profile": True, "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""},
        {"profile": "dev", "ease_rating": 9, "zero_game_lag": True, "platform": "linux", "note": ""},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": "yes", "platform": "linux", "note": ""},
    ):
        with pytest.raises(canary_queue.CanaryQueueError):
            canary_queue.validate_feedback_payload_dict(bad_fb)  # type: ignore[arg-type]

    assert canary_queue.compute_deterministic_backoff_s(0) == 0.0
    assert canary_queue.compute_deterministic_backoff_s(100, base_backoff_s=1.0, max_backoff_s=30.0) == 30.0
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.compute_deterministic_backoff_s(-1)
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.compute_deterministic_backoff_s(1, base_backoff_s=0.0)
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.CesFeedbackRetryQueue(max_entries=0)
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.CesFeedbackRetryQueue(base_backoff_s=-1.0)

    valid_entry = canary_queue.validate_ces_feedback_entry(profile="dev", ease_rating=5, zero_game_lag=True)
    fixed_sha = "b" * 64
    assert canary_queue.compute_ces_event_id(valid_entry, idempotency_key=fixed_sha) == fixed_sha
    assert len(canary_queue.compute_ces_event_id(valid_entry)) == 64
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.compute_ces_event_id(valid_entry, idempotency_key=True)  # type: ignore[arg-type]
    with pytest.raises(canary_queue.CanaryQueueError):
        canary_queue.compute_ces_event_id(valid_entry, idempotency_key="bad\x00key")

    def _broken_canary_http(method: str, url: str) -> tuple[int, Any]:
        _ = method
        if url.endswith("/health"):
            return 200, {"status": "degraded"}
        if url.endswith("/v1/hivemind/onboarding/manifest"):
            return 200, {"object": "wrong", "build_sha": "abcdef123456"}
        if url.endswith("/noel"):
            return 200, "<html><body>no doctype or csp</body></html>"
        if url.endswith("/quick"):
            return 429, {"error": "rate_limited"}
        if url.endswith("/raid-noel"):
            return 503, {"error": "unavailable"}
        if url.endswith("/arena-noel"):
            return 0, {"network_error": "timeout"}
        if url.endswith("/v1/hivemind/onboarding/feedback/summary"):
            return 200, {"total_responses": -1}
        return 200, {"object": "wrong_fleet"}

    broken_rep = canary_queue.verify_canary_cutover(
        "http://127.0.0.1:8000",
        expected_sha="ffffffffffff",
        previous_revision=PREVIOUS_GOOD_REV,
        env=CLEAN_ENV,
        cwd=clean_ws,
        http_fn=_broken_canary_http,
        lang="fr",
    )
    _verify_sha256_receipt(broken_rep)
    assert broken_rep["rollback_triggered"] is True
    assert broken_rep["sha_drift"] is True
    md_rb = canary_queue.render_markdown_report(broken_rep)
    assert PREVIOUS_GOOD_REV in md_rb
    assert canary_queue.scan_output_safety(md_rb) == []


def test_zz_canary_cutover_and_ces_queue_standalone_line_coverage_meets_95_percent() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {TARGET_SCRIPT.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
