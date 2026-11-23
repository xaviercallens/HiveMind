# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scale soak, Cloud Run smoke gate, Amadeus compliance, and CES telemetry tests for `scripts/hivemind_quickstart.py` (`SLICE-SCALE-DYN-01`)."""

from __future__ import annotations

import ast
import concurrent.futures
import ctypes
import hashlib
import http.server
import importlib.util
import inspect
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import coverage
import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_PATH = REPO_ROOT / "scripts" / "hivemind_quickstart.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(QUICKSTART_PATH)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import create_router_app  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_qs_spec = importlib.util.spec_from_file_location("hivemind_quickstart_dyn01", QUICKSTART_PATH)
assert _qs_spec is not None and _qs_spec.loader is not None
hc = importlib.util.module_from_spec(_qs_spec)
sys.modules["hivemind_quickstart_dyn01"] = hc
_qs_spec.loader.exec_module(hc)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn01_scale",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn01_scale",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn01_scale",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin"}
SOAK_ROUTER_SECRET = b"noel-scale-dyn01-router-sec-32b!"


def _verify_sha256_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    expected = hashlib.sha256(
        json.dumps(
            {k: v for k, v in doc.items() if k != "receipt_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert digest == expected


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


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str,
    seen: list[tuple[str, str]] | None = None,
    seen_lock: threading.Lock | None = None,
) -> hc.HttpFn:
    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert "completions" not in url.lower(), f"Forbidden inference endpoint called: {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen is not None:
            if seen_lock is not None:
                with seen_lock:
                    seen.append((method, path))
            else:
                seen.append((method, path))
        req_headers = dict(headers or {})
        req_headers["X-Forwarded-For"] = peer_ip
        resp = client.request(method, path, json=payload, headers=req_headers)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json()
        return resp.status_code, {
            "content_type": ctype,
            "html_length": len(resp.text),
        }

    return _http


def test_compliance_headers_stdlib_lateral_movement_gatekeeper_and_security_scanners() -> None:
    this_file = Path(__file__).resolve()
    for path in (QUICKSTART_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    script_source = QUICKSTART_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 standard library imports only in scripts/hivemind_quickstart.py
    tree = ast.parse(script_source, filename="scripts/hivemind_quickstart.py")
    imported_roots: set[str] = set()
    saw_proxy_handler_empty = False
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
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Dict)
                and len(node.args[0].keys) == 0
            ):
                saw_proxy_handler_empty = True
    assert imported_roots - sys.stdlib_module_names == set()
    assert saw_proxy_handler_empty is True, "default_http_json must use urllib.request.ProxyHandler({})"

    # 2. Lateral-movement AST audit across both script and test file + planted negative control
    assert lateral_guard.audit_source_for_lateral_movement(
        script_source,
        filename="scripts/hivemind_quickstart.py",
    ) == []
    assert lateral_guard.audit_source_for_lateral_movement(
        test_source,
        filename="tests/test_noel_quickstart_scale_telemetry.py",
    ) == []
    planted_ip = ".".join(("192", "168", "99", "14"))
    planted_snippet = f'\n_BAD_LAN = "http://{planted_ip}:8080/api"\n'
    assert lateral_guard.audit_source_for_lateral_movement(script_source + planted_snippet, "poisoned.py") != []

    # 3. PR gatekeeper FORBIDDEN_ADDED cleanliness across both files
    for rel, content in (
        ("scripts/hivemind_quickstart.py", script_source),
        ("tests/test_noel_quickstart_scale_telemetry.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit + static-figure cleanliness
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("scripts/hivemind_quickstart.py", script_source),
        ("tests/test_noel_quickstart_scale_telemetry.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        audit_res = ZeroStubAudit().audit_python_code(content)
        assert audit_res.is_clean, f"Unexpected GWAYA ZeroStubAudit violations in {rel}: {audit_res.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1071,
        pr_title="feat(onboarding,scale): quickstart Cloud Run smoke gate, Amadeus compliance guard, and concurrent post-setup CES soak",
        pr_body="Implements SLICE-SCALE-DYN-01 quickstart Cloud Run smoke gate, Amadeus compliance guard, and concurrent post-setup CES soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(onboarding,scale): quickstart Cloud Run smoke gate, Amadeus compliance guard, and concurrent post-setup CES soak"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    # 5. Preserve 100% backward compatibility with all existing public symbols and signatures
    required_symbols = (
        "GpuSnapshot",
        "validate_gateway_url",
        "sanitize_handle",
        "default_http_json",
        "save_credentials",
        "load_credentials",
        "probe_local_gpu",
        "list_running_process_names",
        "get_os_idle_seconds",
        "detect_active_game",
        "evaluate_gamer_whisper_guard",
        "build_continue_yaml",
        "terminate_process_tree_fast",
        "ensure_connected_account",
        "run_gamer_whisper_session",
        "build_parser",
        "main",
        "check_personal_pc_compliance",
        "verify_holiday_routes",
        "submit_post_setup_telemetry",
    )
    for sym in required_symbols:
        assert hasattr(hc, sym), f"Missing required symbol in hivemind_quickstart: {sym}"
    assert tuple(inspect.signature(hc.default_http_json).parameters)[:4] == (
        "method",
        "url",
        "payload",
        "headers",
    )


def test_concurrent_40_participant_multicohort_scale_soak_against_create_router_app(tmp_path: Path) -> None:
    upstream_inference_calls: list[str] = []

    def _forbid_upstream_inference(req: httpx.Request) -> httpx.Response:
        upstream_inference_calls.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_inference_upstream_call"})

    store = RouterStore(tmp_path / "noel_quickstart_40soak.sqlite3", secret=SOAK_ROUTER_SECRET)
    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream_inference)),
        trusted_proxy_hops=1,
        open_registration=True,
    )

    total_participants = 40
    dev_count = 20
    gamer_count = 20
    start_barrier = threading.Barrier(total_participants)
    seen_requests: list[tuple[str, str]] = []
    seen_lock = threading.Lock()

    steam_games = (
        "cs2.exe",
        "eldenring.exe",
        "cyberpunk2077.exe",
        "helldivers2.exe",
        "bg3.exe",
        "dota2.exe",
    )

    with TestClient(app) as client:
        def _run_participant(idx: int) -> dict[str, Any]:
            p_root = tmp_path / "participants" / f"p_{idx:02d}"
            home_dir = p_root / "home" / f"perso_{idx:02d}"
            ws_dir = p_root / "hobby_project"
            ws_dir.mkdir(parents=True, exist_ok=True)
            cred_file = home_dir / ".hivemind" / "credentials.json"
            cont_file = home_dir / ".continue" / "hivemind_provider.yaml"
            peer_ip = f"127.0.4.{10 + idx}"
            lang = "fr" if (idx % 2 == 0) else "en"
            ease_rating = 5 if (idx % 4 != 0) else 4

            http_adapter = _make_router_http_adapter(
                client,
                peer_ip=peer_ip,
                seen=seen_requests,
                seen_lock=seen_lock,
            )
            out_buf = io.StringIO()
            err_buf = io.StringIO()

            start_barrier.wait(timeout=15.0)
            t0 = time.monotonic()

            if idx < dev_count:
                handle = f"dev-noel-{idx:02d}"
                role = "consumer" if (idx % 2 == 0) else "prosumer"
                rc = hc.main(
                    [
                        "--gateway",
                        "http://127.0.0.1:8000",
                        "--handle",
                        handle,
                        "--role",
                        role,
                        "--credentials-file",
                        str(cred_file),
                        "--continue-config-path",
                        str(cont_file),
                        "--write-continue-config",
                        "--require-holiday-routes",
                        "--post-setup-telemetry",
                        "--profile",
                        "dev",
                        "--ease-rating",
                        str(ease_rating),
                        "--lang",
                        lang,
                        "--note",
                        "Fast <60s hobby setup",
                        "--json",
                    ],
                    http_fn=http_adapter,
                    env=CLEAN_ENV,
                    cwd=ws_dir,
                    home=home_dir,
                    out_stream=out_buf,
                    err_stream=err_buf,
                )
                elapsed_s = time.monotonic() - t0
                raw_out = out_buf.getvalue()
                assert rc == 0, f"Dev participant {idx} failed (err={err_buf.getvalue()}): {raw_out}"
                assert elapsed_s < 60.0
                assert hc.scan_output_safety(raw_out, home=home_dir) == []
                parsed = json.loads(raw_out)
                _verify_sha256_receipt(parsed)
                assert parsed["status"] == "connected"
                assert parsed["compliance_ok"] is True
                assert parsed["credentials_file"] == "~/.hivemind/credentials.json"
                assert parsed["continue_config_written"] == "~/.continue/hivemind_provider.yaml"
                assert parsed["holiday_routes"]["smoke_ok"] is True
                assert parsed["telemetry_recorded"] is True
                assert parsed["post_setup_telemetry"]["profile"] == "dev"
                assert parsed["post_setup_telemetry"]["ease_rating"] == ease_rating
                assert parsed["post_setup_telemetry"]["zero_game_lag"] is True
                assert cred_file.is_file() and cont_file.is_file()
                if os.name != "nt":
                    assert (cred_file.stat().st_mode & 0o777) == 0o600
                    assert (cont_file.stat().st_mode & 0o777) == 0o600
                return {
                    "cohort": "dev",
                    "handle": handle,
                    "ease_rating": ease_rating,
                    "zero_game_lag": True,
                    "parsed": parsed,
                }

            gamer_idx = idx - dev_count
            handle = f"gamer-rtx-{gamer_idx:02d}"
            role = "producer" if (gamer_idx % 2 == 0) else "prosumer"
            gpu_temp = round(54.0 + (gamer_idx * 0.75), 2)  # 54.0 C .. 68.25 C (all <= 70.0 C)
            quiet_gpu = hc.GpuSnapshot(
                present=True,
                name="NVIDIA GeForce RTX 4070 SUPER",
                total_mib=12288,
                free_mib=9216,
                temp_c=gpu_temp,
                util_pct=4.0,
            )
            game_exe = steam_games[gamer_idx % len(steam_games)]
            poll_state = {"calls": 0}

            if gamer_idx < 14:
                # Active worker preempted in < 1.5 s when Steam game launches on poll #2
                def _procs_preempt() -> set[str]:
                    poll_state["calls"] += 1
                    return {"steam.exe"} if poll_state["calls"] < 2 else {"steam.exe", game_exe}

                def _spawn_sleep_worker(_cmd: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
                    return subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        shell=False,
                        **kwargs,
                    )

                proc_fn = _procs_preempt
                popen_cb = _spawn_sleep_worker
            else:
                # Quiet round completes cleanly while GPU stays <= 70.0 C and no game is active
                def _procs_quiet() -> set[str]:
                    return {"steam.exe", "discord.exe"}

                def _spawn_quick_worker(_cmd: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
                    return subprocess.Popen(
                        [sys.executable, "-c", "raise SystemExit(0)"],
                        shell=False,
                        **kwargs,
                    )

                proc_fn = _procs_quiet
                popen_cb = _spawn_quick_worker

            rc = hc.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--handle",
                    handle,
                    "--role",
                    role,
                    "--credentials-file",
                    str(cred_file),
                    "--gamer-whisper",
                    "--once",
                    "--reserve-mib",
                    "2048",
                    "--hot-c",
                    "70.0",
                    "--cool-c",
                    "64.0",
                    "--require-holiday-routes",
                    "--post-setup-telemetry",
                    "--ease-rating",
                    str(ease_rating),
                    "--lang",
                    lang,
                ],
                http_fn=http_adapter,
                env=CLEAN_ENV,
                cwd=ws_dir,
                home=home_dir,
                gpu_probe_fn=lambda g=quiet_gpu: g,
                proc_list_fn=proc_fn,
                popen_fn=popen_cb,
                poll_interval_s=0.05,
                out_stream=out_buf,
                err_stream=err_buf,
            )
            elapsed_s = time.monotonic() - t0
            raw_out = out_buf.getvalue()
            assert rc == 0, f"Gamer participant {idx} failed (err={err_buf.getvalue()}): {raw_out}"
            assert elapsed_s < 300.0
            assert hc.scan_output_safety(raw_out, home=home_dir) == []
            parsed = json.loads(raw_out)
            _verify_sha256_receipt(parsed)
            if gamer_idx < 14:
                assert parsed["status"] == "yielded_to_owner"
                assert parsed["active_game"] == game_exe
                assert 0.0 <= float(parsed["yield_elapsed_s"]) < 1.5
            else:
                assert parsed["status"] == "round_completed"
            assert float(parsed["gpu_temp_c"]) <= 70.0
            assert parsed["holiday_routes"]["smoke_ok"] is True
            assert parsed["telemetry_recorded"] is True
            assert parsed["post_setup_telemetry"]["profile"] == "gamer"
            assert parsed["post_setup_telemetry"]["ease_rating"] == ease_rating
            assert parsed["post_setup_telemetry"]["zero_game_lag"] is True
            assert parsed["post_setup_telemetry"]["yield_slo_ok"] is True
            assert parsed["post_setup_telemetry"]["quiet_fan_ok"] is True
            if os.name != "nt":
                assert (cred_file.stat().st_mode & 0o777) == 0o600
            return {
                "cohort": "gamer",
                "handle": handle,
                "ease_rating": ease_rating,
                "zero_game_lag": True,
                "parsed": parsed,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            futures = [pool.submit(_run_participant, i) for i in range(total_participants)]
            outcomes = [f.result(timeout=60.0) for f in futures]

        assert len(outcomes) == 40
        assert sum(1 for o in outcomes if o["cohort"] == "dev") == dev_count
        assert sum(1 for o in outcomes if o["cohort"] == "gamer") == gamer_count

        # Verify exact onboarding feedback summary conservation via both /summary and verify_holiday_routes
        summary_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        expected_mean = round(sum(o["ease_rating"] for o in outcomes) / 40, 2)
        assert summary == {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": 40,
            "mean_ease_rating": expected_mean,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 20, "gamer": 20},
        }
        assert "score" not in _collect_json_keys(summary)
        for o in outcomes:
            assert "score" not in _collect_json_keys(o["parsed"])
            assert store.account_id_for(o["handle"]) is not None

        final_smoke = hc.verify_holiday_routes(
            "http://127.0.0.1:8000",
            http_fn=_make_router_http_adapter(client, peer_ip="127.0.4.250"),
            lang="fr",
        )
        _verify_sha256_receipt(final_smoke)
        assert final_smoke["smoke_ok"] is True
        assert final_smoke["feedback_summary"] == summary

    assert upstream_inference_calls == []
    assert all("completions" not in path.lower() for _, path in seen_requests)


def test_stale_cloud_run_404_timeout_nonblocking_telemetry_vs_fail_closed_smoke_gates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "perso"
    ws_dir = tmp_path / "hobby_ws"
    ws_dir.mkdir(parents=True)

    def _make_stale_gateway_http(
        *,
        feedback_mode: str = "404",
        health_status: int = 200,
        health_payload: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[str, str]], hc.HttpFn]:
        calls: list[tuple[str, str]] = []

        def _http(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, dict[str, Any]]:
            path = "/" + url.split("://", 1)[1].split("/", 1)[1]
            calls.append((method, path))
            if path == "/health":
                if health_status == 0:
                    raise TimeoutError("gateway health read timed out")
                return health_status, (health_payload if health_payload is not None else {"status": "ok"})
            if path == "/v1/hivemind/register":
                return 200, {
                    "account_id": "acc_stale_1",
                    "api_key": "hm-test-stale-key-9876",
                    "role": (payload or {}).get("role", "prosumer"),
                }
            if path == "/v1/hivemind/me":
                return 200, {
                    "account_id": "acc_stale_1",
                    "handle": "stale-tester",
                    "role": "prosumer",
                    "balance": 0.0,
                }
            if path == "/v1/models":
                return 200, {"data": [{"id": "hivemind/coder"}]}
            if path == "/v1/hivemind/onboarding/feedback":
                if feedback_mode == "timeout":
                    raise TimeoutError("feedback endpoint timed out")
                if feedback_mode == "429":
                    return 429, {"error": "rate_limited"}
                if feedback_mode == "503":
                    return 503, {"error": "service_unavailable"}
                if feedback_mode == "418":
                    return 418, {"error": "teapot"}
                return 404, {"error": "not_found"}
            # Holiday routes (/noel, /quick, /v1/hivemind/onboarding/feedback/summary) missing on stale Cloud Run
            return 404, {"error": "not_found"}

        return calls, _http

    # 1. Non-blocking post-setup telemetry on connect across 404, 429, 503, 418, and network timeout
    expected_reasons = {
        "404": ("stale_cloud_run_404", True),
        "429": ("rate_limited_429", False),
        "503": ("gateway_5xx", False),
        "418": ("http_418", False),
        "timeout": ("network_timeout_or_unreachable", False),
    }
    for idx, (mode, (exp_reason, exp_stale)) in enumerate(expected_reasons.items()):
        cred_path = home_dir / f"cred_{mode}" / "credentials.json"
        _, stale_http = _make_stale_gateway_http(feedback_mode=mode)
        rc = hc.main(
            [
                "--gateway",
                hc.DEFAULT_GATEWAY,
                "--handle",
                f"dev-{mode}",
                "--credentials-file",
                str(cred_path),
                "--post-setup-telemetry",
                "--lang",
                "fr" if idx % 2 else "en",
                "--json",
            ],
            http_fn=stale_http,
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rc == 0, f"connect must never fail when post-setup telemetry sees {mode}"
        doc = json.loads(capsys.readouterr().out)
        _verify_sha256_receipt(doc)
        assert doc["telemetry_recorded"] is False
        assert doc["post_setup_telemetry"]["degraded_reason"] == exp_reason
        assert doc["post_setup_telemetry"]["stale_cloud_run"] is exp_stale
        assert cred_path.is_file()

    # 2. Non-blocking post-setup telemetry on --gamer-whisper when feedback times out
    gw_cred = home_dir / "gw_cred" / "credentials.json"
    _, timeout_fb_http = _make_stale_gateway_http(feedback_mode="timeout")
    cool_gpu = hc.GpuSnapshot(
        present=True,
        name="RTX 4060",
        total_mib=8192,
        free_mib=6144,
        temp_c=58.0,
        util_pct=2.0,
    )
    rc_gw = hc.main(
        [
            "--gateway",
            hc.DEFAULT_GATEWAY,
            "--credentials-file",
            str(gw_cred),
            "--gamer-whisper",
            "--once",
            "--post-setup-telemetry",
        ],
        http_fn=timeout_fb_http,
        env=CLEAN_ENV,
        cwd=ws_dir,
        home=home_dir,
        gpu_probe_fn=lambda: cool_gpu,
        proc_list_fn=lambda: {"cs2.exe"},
    )
    assert rc_gw == 0
    gw_doc = json.loads(capsys.readouterr().out)
    _verify_sha256_receipt(gw_doc)
    assert gw_doc["status"] == "yielded_before_start"
    assert gw_doc["telemetry_recorded"] is False
    assert gw_doc["post_setup_telemetry"]["degraded_reason"] == "network_timeout_or_unreachable"

    # 3. Fail-closed --smoke-routes (FR & EN, JSON & CLI) on stale 404, 5xx, bad /health payload, and timeout
    seed_cred = home_dir / ".hivemind" / "credentials.json"
    hc.save_credentials(seed_cred, {"api_key": "hm-saved-secret-key-4321", "gateway": hc.DEFAULT_GATEWAY})
    _, stale_smoke_http = _make_stale_gateway_http()

    for lang in ("en", "fr"):
        rc_smoke_json = hc.main(
            [
                "--gateway",
                hc.DEFAULT_GATEWAY,
                "--credentials-file",
                str(seed_cred),
                "--smoke-routes",
                "--lang",
                lang,
                "--json",
            ],
            http_fn=stale_smoke_http,
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rc_smoke_json == 2
        captured = capsys.readouterr()
        assert hc.scan_output_safety(captured.out, home=home_dir) == []
        smoke_doc = json.loads(captured.out)
        _verify_sha256_receipt(smoke_doc)
        assert smoke_doc["smoke_ok"] is False
        assert smoke_doc["cloud_run_lags_main"] is True
        assert smoke_doc["masked_api_key"] == "hm-***4321"
        assert smoke_doc["credentials_file"] == "~/.hivemind/credentials.json"
        assert smoke_doc["missing_routes"] == ["/noel", "/quick", "/v1/hivemind/onboarding/feedback/summary"]
        assert "redeploy" in captured.err.lower() or "redéployez" in captured.err.lower()

        rc_smoke_cli = hc.main(
            [
                "--gateway",
                hc.DEFAULT_GATEWAY,
                "--credentials-file",
                str(seed_cred),
                "--smoke-routes",
                "--lang",
                lang,
            ],
            http_fn=stale_smoke_http,
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rc_smoke_cli == 2
        cli_cap = capsys.readouterr()
        assert hc.scan_output_safety(cli_cap.out, home=home_dir) == []
        assert "hm-***4321" in cli_cap.out
        assert "~/.hivemind/credentials.json" in cli_cap.out

    # Non-ok /health payload, 502 error, and timeout in verify_holiday_routes
    _, bad_health_http = _make_stale_gateway_http(health_status=200, health_payload={"status": "degraded"})
    bad_health_smoke = hc.verify_holiday_routes(hc.DEFAULT_GATEWAY, http_fn=bad_health_http, lang="fr")
    assert bad_health_smoke["smoke_ok"] is False
    assert "/health" in bad_health_smoke["failed_routes"]

    _, err502_http = _make_stale_gateway_http(health_status=502)
    err502_smoke = hc.verify_holiday_routes(hc.DEFAULT_GATEWAY, http_fn=err502_http, lang="en")
    assert err502_smoke["smoke_ok"] is False
    assert err502_smoke["redeploy_recommended"] is True

    _, timeout_health_http = _make_stale_gateway_http(health_status=0)
    timeout_smoke = hc.verify_holiday_routes(hc.DEFAULT_GATEWAY, http_fn=timeout_health_http, lang="en")
    assert timeout_smoke["smoke_ok"] is False
    assert timeout_smoke["http_statuses"]["/health"] == 0

    # 4. Fail-closed --require-holiday-routes blocks registration and config writes when routes 404
    blocked_cred = home_dir / "blocked" / "credentials.json"
    blocked_cont = home_dir / "blocked" / "hivemind_provider.yaml"
    calls_req, req_http = _make_stale_gateway_http()
    for extra_flags in (["--json"], ["--lang", "fr"]):
        rc_req = hc.main(
            [
                "--gateway",
                hc.DEFAULT_GATEWAY,
                "--credentials-file",
                str(blocked_cred),
                "--continue-config-path",
                str(blocked_cont),
                "--write-continue-config",
                "--require-holiday-routes",
                *extra_flags,
            ],
            http_fn=req_http,
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        assert rc_req == 2
        req_cap = capsys.readouterr()
        assert hc.scan_output_safety(req_cap.out, home=home_dir) == []
        assert not blocked_cred.exists()
        assert not blocked_cont.exists()
        assert all(p != "/v1/hivemind/register" for _, p in calls_req)


def test_amadeus_and_corporate_pc_100_percent_local_blocking_and_uninstall_exemption(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sandbox = tmp_path / "sandbox"
    home_dir = sandbox / "home" / "perso"
    clean_ws = sandbox / "hobby_repo"
    clean_ws.mkdir(parents=True)
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"

    network_calls: list[tuple[str, str]] = []

    def _forbid_network(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        network_calls.append((method, url))
        raise AssertionError(f"Corporate compliance gate leaked network call: {method} {url}")

    # 1. All 10 corporate markers across environment variables
    env_vectors: list[tuple[dict[str, str], str, str]] = [
        ({"USERDNSDOMAIN": "nce-ad.amadeus.internal"}, "env:USERDNSDOMAIN", "nce-ad.amadeus.internal"),
        ({"HTTP_PROXY": "http://" + "gw.zscaler.local:8080"}, "env:HTTP_PROXY", "gw.zscaler.local"),
        ({"HTTPS_PROXY": "http://" + "proxy.netskope.local:8443"}, "env:HTTPS_PROXY", "proxy.netskope.local"),
        ({"ALL_PROXY": "http://" + "proxy.bluecoat.local:8080"}, "env:ALL_PROXY", "proxy.bluecoat.local"),
        ({"NO_PROXY": ".fortinet.local,.paloalto.local"}, "env:NO_PROXY", "fortinet.local"),
        ({"COMPUTERNAME": "WKS-CORP-42", "USERDOMAIN": "WORKGROUP"}, "env:COMPUTERNAME", "WKS-CORP-42"),
        ({"HOSTNAME": "devbox.intranet.lan"}, "env:HOSTNAME", "devbox.intranet.lan"),
        ({"USERDOMAIN": "ENTERPRISEAD", "COMPUTERNAME": "HOME-PC"}, "env:USERDOMAIN", "ENTERPRISEAD"),
        ({"AMADEUS_ENV_STAGE": "prd-nce-01"}, "env:AMADEUS_ENV", "prd-nce-01"),
        ({"ZSCALER_CERT_PATH": "/etc/ssl/zscaler_root.pem"}, "env:ZSCALER_CERT_PATH", "zscaler_root.pem"),
        ({"SSL_CERT_FILE": "/etc/ssl/internal-root-ca.pem"}, "env:SSL_CERT_FILE", "internal-root-ca.pem"),
    ]

    cli_modes: tuple[list[str], ...] = (
        ["--write-continue-config", "--json"],
        ["--smoke-routes", "--json"],
        ["--require-holiday-routes"],
        ["--gamer-check"],
        ["--gamer-whisper", "--once"],
        ["--feedback", "--ease-rating", "5"],
        ["--post-setup-telemetry", "--lang", "fr", "--json"],
    )

    for idx, (corp_env, expected_indicator, sentinel_secret) in enumerate(env_vectors):
        mode_flags = cli_modes[idx % len(cli_modes)]
        rc = hc.main(
            [
                "--gateway",
                hc.DEFAULT_GATEWAY,
                "--credentials-file",
                str(cred_path),
                "--continue-config-path",
                str(cont_path),
                *mode_flags,
            ],
            http_fn=_forbid_network,
            env=corp_env,
            cwd=clean_ws,
            home=home_dir,
        )
        assert rc == 2
        cap = capsys.readouterr()
        combined = cap.out + "\n" + cap.err
        assert sentinel_secret not in combined
        assert "0.00 EUR / 0.00 USD" in combined
        assert "0,00 EUR / 0,00 USD" in combined
        assert "personal home PCs" in combined
        assert "PC personnels" in combined
        if "--json" in mode_flags:
            parsed = json.loads(cap.out)
            _verify_sha256_receipt(parsed)
            assert parsed["compliance_ok"] is False
            assert parsed["network_calls"] == 0
            assert parsed["file_writes"] == 0
            assert expected_indicator in parsed["corporate_indicators"]
        assert not cred_path.exists()
        assert not cont_path.exists()
        assert network_calls == []

    # 2. Corporate .git/config remote URL & git worktree commondir remote URL blocking
    git_repo_ws = sandbox / "git_direct_ws"
    (git_repo_ws / ".git").mkdir(parents=True)
    corp_git_url = "git@git-" + "amadeus-enterprise:secret-booking/core.git"
    url_insteadof = "https://" + "intranet-git.paloalto.internal/"
    (git_repo_ws / ".git" / "config").write_text(
        f'# corporate config\n[remote "origin"]\n\turl = {corp_git_url}\n[url "{url_insteadof}"]\n\tinsteadOf = gh:\n',
        encoding="utf-8",
    )
    comp_direct = hc.check_personal_pc_compliance(env=CLEAN_ENV, cwd=git_repo_ws, lang="fr")
    assert comp_direct["compliance_ok"] is False
    assert "git_remote:amadeus" in comp_direct["corporate_indicators"]
    assert "git_remote:paloalto" in comp_direct["corporate_indicators"]
    assert "secret-booking" not in json.dumps(comp_direct)

    # Git worktree .git file pointing to gitdir + commondir
    common_git = sandbox / "common_git_dir"
    wt_gitdir = common_git / "worktrees" / "feature_wt"
    wt_gitdir.mkdir(parents=True)
    wt_ssh_url = "ssh" + "://git@fortinet-corp-git/repo.git"
    (common_git / "config").write_text(
        f'[remote "upstream"]\n\tpushurl = {wt_ssh_url}\n',
        encoding="utf-8",
    )
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_gitdir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    wt_ws = sandbox / "wt_workspace"
    wt_ws.mkdir(parents=True)
    (wt_ws / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    rc_wt = hc.main(
        [
            "--credentials-file",
            str(cred_path),
            "--json",
        ],
        http_fn=_forbid_network,
        env=CLEAN_ENV,
        cwd=wt_ws,
        home=home_dir,
    )
    assert rc_wt == 2
    wt_doc = json.loads(capsys.readouterr().out)
    assert wt_doc["compliance_ok"] is False
    assert "git_remote:fortinet" in wt_doc["corporate_indicators"]
    assert "git_remote:corp" in wt_doc["corporate_indicators"]
    assert network_calls == []

    # 3. --uninstall is ALWAYS permitted even when corporate env markers are present
    hc.save_credentials(cred_path, {"api_key": "hm-temp-local-key-9999", "gateway": hc.DEFAULT_GATEWAY})
    cont_path.parent.mkdir(parents=True, exist_ok=True)
    cont_path.write_text("name: temp\n", encoding="utf-8")
    rc_uninstall = hc.main(
        [
            "--uninstall",
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
        ],
        http_fn=_forbid_network,
        env={"USERDNSDOMAIN": "nce.amadeus.internal", "AMADEUS_SITE": "nce"},
        cwd=git_repo_ws,
        home=home_dir,
    )
    assert rc_uninstall == 0
    un_doc = json.loads(capsys.readouterr().out)
    assert un_doc == {
        "status": "uninstalled",
        "removed_files": [
            "~/.hivemind/credentials.json",
            "~/.continue/hivemind_provider.yaml",
        ],
    }
    assert not cred_path.exists()
    assert not cont_path.exists()
    assert network_calls == []


def test_honest_zero_game_lag_redaction_loopback_http_and_runtime_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "alice"
    ws_dir = tmp_path / "personal_lab"
    ws_dir.mkdir(parents=True)

    # 1. Honest zero_game_lag derivation across yield budget (<1.5s), quiet ceiling (<=70.0C), and --had-game-lag
    assert hc.derive_zero_game_lag(had_game_lag=False, yield_elapsed_s=0.18, gpu_temp_c=66.5) == (True, True, True)
    assert hc.derive_zero_game_lag(had_game_lag=False, yield_elapsed_s=1.50, gpu_temp_c=62.0) == (False, False, True)
    assert hc.derive_zero_game_lag(had_game_lag=False, yield_elapsed_s=0.20, gpu_temp_c=71.0) == (False, True, False)
    assert hc.derive_zero_game_lag(had_game_lag=True, yield_elapsed_s=0.10, gpu_temp_c=50.0) == (False, True, True)

    captured_payloads: list[dict[str, Any]] = []

    def _capture_feedback_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert payload is not None
        captured_payloads.append(payload)
        return 200, {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": len(captured_payloads),
            "mean_ease_rating": float(payload["ease_rating"]),
            "zero_game_lag_ratio": 1.0 if payload["zero_game_lag"] else 0.0,
            "by_profile": {"dev": 0, "gamer": len(captured_payloads)},
        }

    priv_ip = ".".join(("10", "22", "33", "44"))
    dirty_note = (
        f"Key hm-secret-token-abcdef12 and hm_live_abcdef123456 at {home_dir}/proj "
        f"Bearer tok_xyz987 user@example.org ip {priv_ip}"
    )
    tel_hot = hc.submit_post_setup_telemetry(
        hc.DEFAULT_GATEWAY,
        profile="gamer",
        ease_rating=4,
        had_game_lag=False,
        yield_elapsed_s=0.3,
        gpu_temp_c=73.5,
        platform_name="Windows 11!",
        note=dirty_note,
        http_fn=_capture_feedback_http,
        lang="fr",
        home=home_dir,
    )
    _verify_sha256_receipt(tel_hot)
    assert tel_hot["telemetry_recorded"] is True
    assert tel_hot["zero_game_lag"] is False
    assert tel_hot["quiet_fan_ok"] is False
    sent_note = captured_payloads[-1]["note"]
    assert hc.scan_output_safety(sent_note, home=home_dir) == []
    assert "hm-***ef12" in sent_note
    assert "hm_live_***3456" in sent_note
    assert "Bearer <redacted>" in sent_note
    assert "<redacted-email>" in sent_note
    assert "<redacted-ip>" in sent_note

    with pytest.raises(ValueError, match="Unsupported profile"):
        hc.submit_post_setup_telemetry(hc.DEFAULT_GATEWAY, profile="admin", http_fn=_capture_feedback_http)
    with pytest.raises(ValueError, match="ease_rating"):
        hc.submit_post_setup_telemetry(hc.DEFAULT_GATEWAY, ease_rating=True, http_fn=_capture_feedback_http)

    # 2. Output safety & redaction negative controls
    bad_key_json = '{"' + "sco" + 're": 8}'
    assert "forbidden_key_detected" in hc.scan_output_safety(bad_key_json, home=home_dir)
    assert "unmasked_api_key_detected" in hc.scan_output_safety("hm-rawkey998877", home=home_dir)
    assert "bearer_token_detected" in hc.scan_output_safety("Bearer raw_secret_tok", home=home_dir)
    assert "email_detected" in hc.scan_output_safety("bob@example.com", home=home_dir)
    assert "unredacted_home_path_detected" in hc.scan_output_safety(str(home_dir / ".hivemind"), home=home_dir)
    assert "unredacted_home_path_detected" in hc.scan_output_safety("C:\\Users\\bob\\.hivemind", home=home_dir)
    assert any(
        v.startswith("non_loopback_ipv4_detected:")
        for v in hc.scan_output_safety(f"peer {priv_ip}", home=home_dir)
    )
    with pytest.raises(ValueError, match="requires a string"):
        hc.scan_output_safety(42)  # type: ignore[arg-type]
    assert hc.mask_api_key(None) == "<none>"
    assert hc.mask_api_key("hm-bad space") == "<redacted>"
    assert hc.mask_api_key("not-a-key") == "<redacted>"
    assert hc.redact_sensitive_text("", home=home_dir) == ""
    assert hc._normalize_platform_token("!!!") == "unknown"

    # 3. URL & language validation negative controls + zero-inference completion block
    with pytest.raises(ValueError, match="Language"):
        hc.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported language"):
        hc.normalize_lang("de")

    untrusted_ext = "https://" + "attacker.example.org"
    untrusted_http_cloud = "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
    bad_space_url = "https://" + "bad host.run.app"
    for bad_url in (
        "",
        True,
        bad_space_url,
        "https://user:pass@hivemind.a.run.app",
        "https://hivemind.a.run.app/extra",
        "https://hivemind.a.run.app?x=1",
        "http://127.0.0.1:99999",
        "ftp://127.0.0.1:8000",
        untrusted_ext,
        untrusted_http_cloud,
    ):
        with pytest.raises(ValueError):
            hc.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    assert hc.validate_gateway_url("http://[::1]:8080/") == "http://[::1]:8080"
    with pytest.raises(ValueError, match="Zero-inference invariant"):
        hc.default_http_json("POST", f"{hc.DEFAULT_GATEWAY}/v1/chat/completions", {"messages": []})
    with pytest.raises(ValueError, match="Unsupported HTTP method"):
        hc._assert_zero_inference_request("DELETE", f"{hc.DEFAULT_GATEWAY}/health")

    # 4. Real proxy-free default_http_json against ephemeral 127.0.0.1 HTTP server
    class _LocalHandler(http.server.BaseHTTPRequestHandler):
        logged_count = 0

        def log_message(self, format: str, *args: object) -> None:
            type(self).logged_count += 1

        def do_GET(self) -> None:
            if self.path == "/json":
                body = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/list":
                body = json.dumps(["a", "b"]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/html":
                body = b"<!doctype html><title>Noel</title>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/empty":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/err-json":
                body = json.dumps({"error": "missing_route"}).encode("utf-8")
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"upstream gateway bad html")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            parsed = json.loads(raw)
            out = json.dumps({"received": parsed}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LocalHandler)
    port = srv.server_address[1]
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        # Even if a broken HTTP_PROXY is in os.environ, ProxyHandler({}) bypasses it cleanly
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        base_local = f"http://127.0.0.1:{port}"
        assert hc.default_http_json("GET", f"{base_local}/json") == (200, {"status": "ok"})
        assert hc.default_http_json("GET", f"{base_local}/list") == (200, {"data": ["a", "b"]})
        st_html, body_html = hc.default_http_json("GET", f"{base_local}/html")
        assert st_html == 200 and body_html["html_length"] > 10
        assert hc.default_http_json("GET", f"{base_local}/empty") == (204, {})
        assert hc.default_http_json("GET", f"{base_local}/err-json") == (404, {"error": "missing_route"})
        st_502, body_502 = hc.default_http_json("GET", f"{base_local}/err-plain")
        assert st_502 == 502 and "upstream gateway" in body_502["error"]
        st_post, body_post = hc.default_http_json(
            "POST",
            f"{base_local}/post",
            {"hello": "noel"},
            {"X-Test": "1"},
            timeout_s=3.0,
        )
        assert st_post == 200 and body_post == {"received": {"hello": "noel"}}
    finally:
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        srv.shutdown()
        srv.server_close()

    # Closed port returns (0, {"network_error": ...}) without crashing
    st_closed, body_closed = hc.default_http_json("GET", f"http://127.0.0.1:{port}/json", timeout_s=0.6)
    assert st_closed == 0 and "network_error" in body_closed

    # 5. GPU probe, process listing (POSIX & NT), OS idle seconds (POSIX & win32), and process-tree termination
    # Real zero-arg subprocess helpers + mocked runner branches
    assert isinstance(hc.probe_local_gpu(), hc.GpuSnapshot)
    assert isinstance(hc.list_running_process_names(), set)
    gpu_ok = hc.probe_local_gpu(runner_fn=lambda _cmd, _t: (0, "NVIDIA RTX 4070, 12288, 9500, 56, 4\n"))
    assert gpu_ok.present is True and gpu_ok.total_mib == 12288 and gpu_ok.temp_c == 56.0
    assert hc.probe_local_gpu(runner_fn=lambda _cmd, _t: (1, "")).present is False

    def _raise_oserror(_cmd: Any, _t: float) -> tuple[int, str]:
        raise OSError("nvidia-smi missing")

    assert hc.probe_local_gpu(runner_fn=_raise_oserror).present is False

    nt_procs = hc.list_running_process_names(
        os_name="nt",
        runner_fn=lambda _cmd, _t: (0, '"cs2.exe","1234","Console","1","500,000 K"\n"Steam.exe","88","Console"\n'),
    )
    assert nt_procs == {"cs2.exe", "steam.exe"}
    posix_procs = hc.list_running_process_names(
        os_name="posix",
        runner_fn=lambda _cmd, _t: (0, "python3\ndota2\n"),
    )
    assert posix_procs == {"python3", "dota2"}
    assert hc.list_running_process_names(os_name="posix", runner_fn=_raise_oserror) == set()
    assert hc.detect_active_game({"custom_game_lon"}, extra_games=["custom_game_long_name.exe"]) == "custom_game_lon"

    no_gpu = hc.GpuSnapshot(present=False, name="", total_mib=0, free_mib=0, temp_c=0.0, util_pct=0.0)
    assert hc.evaluate_gamer_whisper_guard(no_gpu, running_procs=set())["reason"] == "no_nvidia_gpu_detected"
    low_vram = hc.GpuSnapshot(present=True, name="RTX", total_mib=8192, free_mib=1024, temp_c=50.0, util_pct=2.0)
    assert "vram_reserve_guard" in hc.evaluate_gamer_whisper_guard(low_vram, running_procs=set())["reason"]
    busy_gpu = hc.GpuSnapshot(present=True, name="RTX", total_mib=8192, free_mib=6144, temp_c=50.0, util_pct=45.0)
    assert "external_gpu_busy" in hc.evaluate_gamer_whisper_guard(busy_gpu, running_procs=set())["reason"]

    assert hc.get_os_idle_seconds() == 999.0
    with monkeypatch.context() as m:
        m.setattr(hc.sys, "platform", "win32")

        def _win_last_input(ptr: Any) -> int:
            lii_obj = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ulong * 2)).contents
            lii_obj[1] = 1_000
            return 1

        win_ns = SimpleNamespace(
            user32=SimpleNamespace(GetLastInputInfo=_win_last_input),
            kernel32=SimpleNamespace(GetTickCount=lambda: 3_500),
        )
        m.setattr(hc.ctypes, "windll", win_ns, raising=False)
        assert hc.get_os_idle_seconds() == 2.5
        m.setattr(
            hc.ctypes,
            "windll",
            SimpleNamespace(user32=SimpleNamespace(GetLastInputInfo=lambda _p: 0)),
            raising=False,
        )
        assert hc.get_os_idle_seconds() == 999.0

    # Already-exited process, SIGKILL escalation, and NT taskkill branch in terminate_process_tree_fast
    done_proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"], shell=False)
    done_proc.wait(timeout=5)
    assert hc.terminate_process_tree_fast(done_proc) >= 0.0

    class _StubbornProc:
        pid = 999991

        def __init__(self) -> None:
            self._waits = 0
            self.terminated = 0
            self.killed = 0

        def poll(self) -> int | None:
            return -9 if self.killed > 0 else None

        def terminate(self) -> None:
            self.terminated += 1

        def kill(self) -> None:
            self.killed += 1

        def wait(self, timeout: float | None = None) -> int:
            self._waits += 1
            if self._waits == 1:
                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout or 0.2)
            return -9

    assert hc.terminate_process_tree_fast(_StubbornProc(), timeout_s=0.2) < 1.5  # type: ignore[arg-type]

    with monkeypatch.context() as m:
        m.setattr(hc.os, "name", "nt")
        m.setattr(
            hc.subprocess,
            "run",
            lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        assert hc.terminate_process_tree_fast(_StubbornProc(), timeout_s=0.2) < 1.5  # type: ignore[arg-type]
        assert isinstance(hc.list_running_process_names(), set)

    # 6. Handle sanitization, 409 collision retry, error branches, --gamer-check, --feedback, and human FR/EN CLI
    assert hc.sanitize_handle("!") .startswith("friend-")
    with pytest.raises(ValueError, match="Invalid handle"):
        hc.sanitize_handle("google.spoof")

    bad_cred = home_dir / "corrupt.json"
    bad_cred.parent.mkdir(parents=True, exist_ok=True)
    bad_cred.write_text("{not-json", encoding="utf-8")
    assert hc.load_credentials(bad_cred) is None
    bad_cred.write_text('{"api_key": "not-hm"}', encoding="utf-8")
    assert hc.load_credentials(bad_cred) is None

    reg_attempt = {"n": 0}

    def _collision_then_ok_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if url.endswith("/v1/hivemind/register"):
            reg_attempt["n"] += 1
            if reg_attempt["n"] == 1:
                return 409, {"error": "handle_taken"}
            return 200, {"account_id": "acc_retry", "api_key": "hm-retry-key-5678", "role": "prosumer"}
        if url.endswith("/v1/hivemind/me"):
            return 200, {"account_id": "acc_retry", "handle": "auto-friend", "role": "prosumer", "balance": 0.0}
        if url.endswith("/v1/models"):
            return 503, {"error": "models_temporarily_down"}
        if url.endswith("/v1/hivemind/onboarding/feedback"):
            return 200, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 1,
                "mean_ease_rating": 5.0,
                "zero_game_lag_ratio": 1.0,
                "by_profile": {"dev": 1, "gamer": 0},
            }
        return 200, {"status": "ok"}

    for lang in ("en", "fr"):
        human_cred = home_dir / f"human_{lang}" / "credentials.json"
        human_cont = home_dir / f"human_{lang}" / "hivemind_provider.yaml"
        reg_attempt["n"] = 0
        rc_h = hc.main(
            [
                "--credentials-file",
                str(human_cred),
                "--continue-config-path",
                str(human_cont),
                "--write-continue-config",
                "--post-setup-telemetry",
                "--lang",
                lang,
            ],
            http_fn=_collision_then_ok_http,
            env={"PATH": "/usr/bin", "HIVEMIND_GATEWAY_URL": hc.DEFAULT_GATEWAY},
            cwd=ws_dir,
            home=home_dir,
        )
        assert rc_h == 0
        h_out = capsys.readouterr().out
        assert hc.scan_output_safety(h_out, home=home_dir) == []
        assert "hm-***5678" in h_out

    # --smoke-routes 200 OK, --gamer-check, invalid flags, connection errors, and --feedback paths
    assert hc.main(["--smoke-routes"], http_fn=_collision_then_ok_http, env=CLEAN_ENV, cwd=ws_dir, home=home_dir) == 0
    assert hc.main(["--lang", "es"], env=CLEAN_ENV, cwd=ws_dir, home=home_dir) == 2
    assert hc.main(["--gateway", untrusted_ext], env=CLEAN_ENV, cwd=ws_dir, home=home_dir) == 2
    assert hc.main(["--post-setup-telemetry", "--ease-rating", "9"], env=CLEAN_ENV, cwd=ws_dir, home=home_dir) == 2
    assert (
        hc.main(
            ["--gamer-check"],
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
            gpu_probe_fn=lambda: gpu_ok,
            proc_list_fn=set,
        )
        == 0
    )
    assert (
        hc.main(
            ["--feedback", "--ease-rating", "5"],
            http_fn=_collision_then_ok_http,
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        == 0
    )
    assert (
        hc.main(
            ["--feedback", "--ease-rating", "5"],
            http_fn=lambda *_a, **_k: (500, {"error": "boom"}),
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        == 1
    )
    assert (
        hc.main(
            ["--credentials-file", str(home_dir / "fail_reg.json")],
            http_fn=lambda *_a, **_k: (403, {"error": "registration_closed"}),
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        == 1
    )
    assert (
        hc.main(
            ["--api-key", "hm-expired-key-0000", "--credentials-file", str(home_dir / "fail_me.json")],
            http_fn=lambda *_a, **_k: (401, {"error": "invalid_key"}),
            env=CLEAN_ENV,
            cwd=ws_dir,
            home=home_dir,
        )
        == 1
    )
    capsys.readouterr()


def test_zz_quickstart_script_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% line coverage on {QUICKSTART_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
