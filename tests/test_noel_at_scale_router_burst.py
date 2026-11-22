# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Concurrent 60-participant router burst, conservation, security, and SLO tests (`SLICE-SCALE-01`)."""

from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import io
import itertools
import json
import sqlite3
import sys
import threading
import urllib.error
from pathlib import Path
from typing import Any

import coverage
import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "noel_scale_load_harness.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(HARNESS_PATH)],
    )
    _INPROC_COV.start()

_spec = importlib.util.spec_from_file_location("noel_scale_load_harness", HARNESS_PATH)
assert _spec is not None and _spec.loader is not None
scale_harness = importlib.util.module_from_spec(_spec)
sys.modules["noel_scale_load_harness"] = scale_harness
_spec.loader.exec_module(scale_harness)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_scale01",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_scale01",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin"}
ROUTER_SECRET = b"noel-scale-01-router-secret-32b!"


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


def _make_testclient_http(
    client: TestClient,
    recorded_paths: list[tuple[str, str]] | None = None,
) -> scale_harness.HttpFn:
    path_lock = threading.Lock()

    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any] | str, dict[str, str]]:
        assert "completions" not in url.lower(), "Scale harness must NEVER call /v1/chat/completions"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if recorded_paths is not None:
            with path_lock:
                recorded_paths.append((method, path))
        resp = client.request(method, path, json=payload, headers=headers)
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        ctype = resp_headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json(), resp_headers
        return resp.status_code, resp.text, resp_headers

    return _http


def _make_deterministic_mock_http(
    *,
    static_nonce: str | None = None,
    mismatched_html_nonce: bool = False,
    empty_html: bool = False,
    fail_route: str | None = None,
    fail_status: int = 503,
    rate_limit_first_n_votes: int = 0,
    retry_after_header: str = "0.001",
    drop_vote_conservation: bool = False,
    drop_ces_conservation: bool = False,
    raise_sqlite_lock: bool = False,
    raise_network_error: bool = False,
) -> scale_harness.HttpFn:
    nonce_seq = itertools.count(1)
    vote_seq = itertools.count(1)
    ces_seq = itertools.count(1)
    rl_remaining = [rate_limit_first_n_votes]
    lock = threading.Lock()

    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        _ = (payload, headers)
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if raise_sqlite_lock and path == "/v1/hivemind/register":
            raise sqlite3.OperationalError("database is locked")
        if raise_network_error and path == "/v1/models":
            raise urllib.error.URLError("connection reset by peer")
        if fail_route is not None and path == fail_route:
            return fail_status, {"error": f"simulated_{fail_status}"}, {}

        if method == "GET" and path in ("/noel", "/signup", "/raid-noel", "/arena-noel"):
            idx = next(nonce_seq)
            hdr_nonce = static_nonce if static_nonce is not None else f"NoelNonce{idx:06d}AbCdEf"
            dom_nonce = f"Mismatch{idx:06d}XyZ123" if mismatched_html_nonce else hdr_nonce
            if empty_html:
                return 200, "", {"content-security-policy": f"style-src 'nonce-{hdr_nonce}'"}
            ballot_html = ""
            if path == "/arena-noel":
                cards = "".join(
                    f'<article data-task-id="{tid}" data-ballot="abcdef012345.1.0123456789abcdef01234567"></article>'
                    for tid in scale_harness.CURATED_ARENA_TASK_IDS
                )
                ballot_html = cards
            html_doc = f'<!DOCTYPE html><html><head><style nonce="{dom_nonce}">body{{}}</style></head><body>{ballot_html}</body></html>'
            return (
                200,
                html_doc,
                {"content-security-policy": f"default-src 'none'; style-src 'nonce-{hdr_nonce}'"},
            )

        if method == "GET" and path == "/v1/hivemind/onboarding/manifest":
            return (
                200,
                {
                    "object": "hivemind.onboarding.manifest",
                    "campaign_id": "noel-2026-easy-onboard",
                    "status": "ok",
                    "personal_pc_only": True,
                },
                {},
            )

        if method == "POST" and path == "/v1/hivemind/register":
            return 200, {"account_id": "acct_local", "api_key": "hm-local_test_key_123456"}, {}

        if method == "GET" and path == "/v1/models":
            return 200, {"object": "list", "data": [{"id": "hivemind/auto"}]}, {}

        if method == "GET" and path == "/v1/hivemind/raid/status":
            return 200, {"object": "hivemind.raid_noel.status"}, {}

        if method == "POST" and path == "/v1/hivemind/arena/vote":
            with lock:
                if rl_remaining[0] > 0:
                    rl_remaining[0] -= 1
                    return 429, {"error": "rate_limited"}, {"retry-after": retry_after_header}
                v_num = next(vote_seq)
            reported = max(1, v_num - 1) if (drop_vote_conservation and v_num > 1) else v_num
            return (
                200,
                {
                    "object": "hivemind.blind_ab_arena.summary",
                    "recorded": True,
                    "total_votes": reported,
                },
                {},
            )

        if method == "GET" and path == "/v1/hivemind/arena/summary":
            return (
                200,
                {"object": "hivemind.blind_ab_arena.summary", "total_votes": 1},
                {},
            )

        if method == "POST" and path == "/v1/hivemind/onboarding/feedback":
            c_num = next(ces_seq)
            reported_ces = max(1, c_num - 1) if (drop_ces_conservation and c_num > 1) else c_num
            return (
                200,
                {
                    "object": "hivemind.onboarding_feedback_summary",
                    "total_responses": reported_ces,
                },
                {},
            )

        return 404, {"error": "not_found"}, {}

    return _http


def test_compliance_headers_stdlib_lateral_movement_gatekeeper_and_security_scanners() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    this_file = Path(__file__).resolve()
    for path in (HARNESS_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    harness_source = HARNESS_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Lateral-movement AST audit (zero raw sockets, zero SSH/LAN probes, zero untrusted URLs)
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            harness_source,
            filename="scripts/noel_scale_load_harness.py",
        )
        == []
    )
    poisoned = harness_source + '\n_BAD_TARGET = "http://192.168.1.99:8000/noel"\n'
    assert lateral_guard.audit_source_for_lateral_movement(poisoned, "poisoned.py") != []

    # 2. Pure Python standard library only in scripts/noel_scale_load_harness.py
    tree = ast.parse(harness_source, filename="scripts/noel_scale_load_harness.py")
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
        ("scripts/noel_scale_load_harness.py", harness_source),
        ("tests/test_noel_at_scale_router_burst.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("scripts/noel_scale_load_harness.py", harness_source),
        ("tests/test_noel_at_scale_router_burst.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1060,
        pr_title="feat(scale): concurrent 60-user Noel holiday router burst harness and scale test suite",
        pr_body="Implements SLICE-SCALE-01 concurrent 60-user router burst harness and scale SLO tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(scale): add noel_scale_load_harness and burst test suite"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    from gwaya.ast_audit import ZeroStubAudit

    stub_audit = ZeroStubAudit().audit_python_code(harness_source)
    assert stub_audit.is_clean, f"Unexpected GWAYA stub violations: {stub_audit.violations}"

    # 5. Output safety negative controls (including forbidden JSON 'score' key with digit)
    bad_key_fragment = '{"' + "sco" + 're": 5}'
    raw_hm_key = "hm-" + "z" * 18
    assert "forbidden_key_detected" in scale_harness.scan_output_safety(bad_key_fragment)
    assert "email_detected" in scale_harness.scan_output_safety("friend@example.com")
    assert "unmasked_api_key_detected" in scale_harness.scan_output_safety(raw_hm_key)
    assert "bearer_token_detected" in scale_harness.scan_output_safety("Bearer tok_secret999")
    assert "unredacted_home_path_detected" in scale_harness.scan_output_safety("/home/alice/.hivemind")
    assert any(
        hit.startswith("non_loopback_ipv4_detected:")
        for hit in scale_harness.scan_output_safety("peer 10.20.30.40")
    )
    with pytest.raises(scale_harness.ScaleHarnessError, match="Output safety guard failed"):
        scale_harness._assert_safe_output(bad_key_fragment)


def test_real_60_user_concurrent_burst_against_router_app_sqlite_conservation_and_csp_nonces(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "noel_scale_router.db"
    store = RouterStore(db_path, secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, starter_grant_hmt=25.0)
    seen_calls: list[tuple[str, str]] = []

    try:
        with TestClient(app) as client:
            http_fn = _make_testclient_http(client, seen_calls)
            report = scale_harness.run_noel_scale_burst(
                "http://127.0.0.1:8000",
                devs=30,
                gamers=30,
                workers=60,
                local_harness=True,
                http_fn=http_fn,
                environ=CLEAN_ENV,
                lang="fr",
            )

            # Exact deterministic scale metrics
            assert report["scale_slo_ok"] is True
            assert report["participants_total"] == 60
            assert report["devs_count"] == 30
            assert report["gamers_count"] == 30
            assert report["workers_count"] == 60
            assert report["requests_total"] == 360
            assert report["expected_requests_total"] == 360
            assert report["ok_count"] == 360
            assert report["error_count"] == 0
            assert report["error_ratio"] == 0.0
            assert report["rate_limited_count"] == 0
            assert report["server_error_count"] == 0
            assert report["sqlite_lock_error_count"] == 0
            assert report["zero_inference_verified"] is True
            assert report["diagnostics"] == []

            # Monotonic latency percentiles
            assert 0.0 <= report["min_latency_ms"] <= report["p50_latency_ms"]
            assert report["p50_latency_ms"] <= report["p95_latency_ms"] <= report["p99_latency_ms"]
            assert report["p99_latency_ms"] <= report["max_latency_ms"]

            # 100% unique CSP nonces across all 120 concurrent HTML renders (60 dev + 60 gamer)
            assert report["html_renders_total"] == 120
            assert report["unique_csp_nonces_count"] == 120
            assert report["unique_csp_nonces_ratio"] == 1.0

            # Exact registration, Blind A/B vote, and CES feedback conservation
            assert report["registrations_recorded"] == 30
            assert report["registrations_conserved"] is True
            assert report["arena_votes_recorded"] == 30
            assert report["arena_votes_conserved"] is True
            assert report["ces_responses_recorded"] == 60
            assert report["ces_responses_conserved"] is True

            # Direct server-side verification of post-burst Arena & CES state
            arena_after = client.get("/v1/hivemind/arena/summary").json()
            assert arena_after["total_votes"] == 30
            assert arena_after["by_profile"] == {"dev": 0, "gamer": 30}
            assert sum(arena_after["chosen_side_counts"].values()) == 30
            for tid in scale_harness.CURATED_ARENA_TASK_IDS:
                assert arena_after["by_task"][tid]["total_votes"] >= 7

            ces_after = client.get("/v1/hivemind/onboarding/feedback/summary").json()
            assert ces_after["total_responses"] == 60
            assert ces_after["by_profile"] == {"dev": 30, "gamer": 30}
            assert ces_after["zero_game_lag_ratio"] == 1.0

            # Direct SQLite verification: all 30 registered Dev accounts persisted cleanly
            accounts_summary = store.public_accounts_summary()
            assert accounts_summary["total_registered_accounts"] == 30
            for idx in range(30):
                acct_id = store.account_id_for(f"noel-dev-noel2026-{idx:04d}")
                assert store.get_account(acct_id) is not None

            # Confirm zero LLM inference endpoints were touched across all 360 calls
            assert len(seen_calls) == 360
            assert all("completions" not in path.lower() for _, path in seen_calls)
            _verify_sha256_receipt(report)
    finally:
        store.close()


def test_rate_limit_handling_shared_nat_and_bounded_429_retries(tmp_path: Path) -> None:
    # 1. Shared single-IP NAT burst (trusted_proxy_hops=0): register_limiter caps at 10/min
    store = RouterStore(tmp_path / "shared_nat.db", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=0)
    try:
        with TestClient(app) as client:
            nat_http = _make_testclient_http(client)

            strict_rep = scale_harness.run_noel_scale_burst(
                "http://127.0.0.1:8000",
                devs=8,
                gamers=8,
                workers=16,
                http_fn=nat_http,
                environ=CLEAN_ENV,
                lang="en",
            )
            assert strict_rep["rate_limited_count"] > 0
            assert strict_rep["sqlite_lock_error_count"] == 0
            assert strict_rep["server_error_count"] == 0
            assert strict_rep["scale_slo_ok"] is False
            assert any("HTTP 429" in d for d in strict_rep["diagnostics"])

        # Fresh router instance with --tolerate-rate-limits enabled
        with TestClient(create_router_app(store, trusted_proxy_hops=0)) as client2:
            tol_rep = scale_harness.run_noel_scale_burst(
                "http://127.0.0.1:8000",
                devs=8,
                gamers=8,
                workers=16,
                tolerate_rate_limits=True,
                burst_tag="natrun2",
                http_fn=_make_testclient_http(client2),
                environ=CLEAN_ENV,
                lang="fr",
            )
            assert tol_rep["rate_limited_count"] > 0
            assert tol_rep["server_error_count"] == 0
            assert tol_rep["sqlite_lock_error_count"] == 0
            assert tol_rep["scale_slo_ok"] is True
            assert any("429" in d for d in tol_rep["diagnostics"])
    finally:
        store.close()

    # 2. Transient HTTP 429 recovery with numeric and non-numeric Retry-After headers
    slept: list[float] = []
    retry_http = _make_deterministic_mock_http(
        rate_limit_first_n_votes=2,
        retry_after_header="0.05",
    )
    recovered = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=4,
        gamers=4,
        workers=8,
        max_429_retries=2,
        retry_backoff_s=0.002,
        http_fn=retry_http,
        sleep_fn=slept.append,
        environ=CLEAN_ENV,
    )
    assert recovered["scale_slo_ok"] is True
    assert recovered["include_register"] is False
    assert recovered["requests_total"] == 4 * 5 + 4 * 6
    assert recovered["error_count"] == 0
    assert recovered["retried_429_count"] == 2
    assert len(slept) == 2

    non_num_retry_http = _make_deterministic_mock_http(
        rate_limit_first_n_votes=1,
        retry_after_header="soon",
    )
    recovered_non_num = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=1,
        gamers=2,
        workers=3,
        max_429_retries=1,
        retry_backoff_s=0.001,
        http_fn=non_num_retry_http,
        sleep_fn=lambda _: None,
        environ=CLEAN_ENV,
    )
    assert recovered_non_num["scale_slo_ok"] is True
    assert recovered_non_num["retried_429_count"] == 1


def test_degraded_5xx_sqlite_lock_csp_collision_latency_and_conservation_failures() -> None:
    # 1. Degraded 503 gateway response fails SLO closed
    rep_503 = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=3,
        gamers=3,
        workers=6,
        http_fn=_make_deterministic_mock_http(fail_route="/v1/hivemind/raid/status", fail_status=503),
        environ=CLEAN_ENV,
        lang="fr",
    )
    assert rep_503["scale_slo_ok"] is False
    assert rep_503["server_error_count"] == 3
    assert any("HTTP 5xx" in d for d in rep_503["diagnostics"])

    # 2. Network URLError (status 0) fails SLO closed
    rep_net = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=2,
        gamers=2,
        workers=4,
        http_fn=_make_deterministic_mock_http(raise_network_error=True),
        environ=CLEAN_ENV,
        lang="en",
    )
    assert rep_net["scale_slo_ok"] is False
    assert rep_net["server_error_count"] == 2
    assert rep_net["http_status_counts"]["0"] == 2

    # 3. SQLite lock contention surfaces explicitly and fails SLO closed
    rep_lock = scale_harness.run_noel_scale_burst(
        "http://127.0.0.1:8000",
        devs=3,
        gamers=2,
        workers=5,
        http_fn=_make_deterministic_mock_http(raise_sqlite_lock=True),
        environ=CLEAN_ENV,
        lang="en",
    )
    assert rep_lock["scale_slo_ok"] is False
    assert rep_lock["sqlite_lock_error_count"] == 3
    assert any("SQLite lock" in d for d in rep_lock["diagnostics"])

    # 4. Duplicate / static CSP nonce across concurrent HTML renders fails SLO
    rep_static_nonce = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=3,
        gamers=3,
        workers=6,
        http_fn=_make_deterministic_mock_http(static_nonce="StaticReusedNonce12345"),
        environ=CLEAN_ENV,
        lang="fr",
    )
    assert rep_static_nonce["scale_slo_ok"] is False
    assert rep_static_nonce["unique_csp_nonces_count"] == 1
    assert rep_static_nonce["unique_csp_nonces_ratio"] < 1.0
    assert any("CSP" in d for d in rep_static_nonce["diagnostics"])

    # 5. Mismatched header vs HTML CSP nonce and empty HTML body
    rep_mismatch = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=1,
        gamers=1,
        workers=2,
        http_fn=_make_deterministic_mock_http(mismatched_html_nonce=True),
        environ=CLEAN_ENV,
    )
    assert rep_mismatch["scale_slo_ok"] is False
    assert rep_mismatch["unique_csp_nonces_ratio"] == 0.0

    rep_empty_html = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=1,
        gamers=1,
        workers=2,
        http_fn=_make_deterministic_mock_http(empty_html=True),
        environ=CLEAN_ENV,
    )
    assert rep_empty_html["scale_slo_ok"] is False
    assert rep_empty_html["error_count"] == 4

    # 6. Vote & CES lost-update conservation detection
    rep_lost_vote = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=2,
        gamers=3,
        workers=5,
        http_fn=_make_deterministic_mock_http(drop_vote_conservation=True, drop_ces_conservation=True),
        environ=CLEAN_ENV,
        lang="en",
    )
    assert rep_lost_vote["scale_slo_ok"] is False
    assert rep_lost_vote["arena_votes_conserved"] is False
    assert rep_lost_vote["ces_responses_conserved"] is False

    # 7. Latency SLO breach detection with deterministic stepping clock
    tick = itertools.count(0)
    rep_slow = scale_harness.run_noel_scale_burst(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        devs=2,
        gamers=2,
        workers=1,
        p95_slo_ms=10.0,
        p99_slo_ms=20.0,
        http_fn=_make_deterministic_mock_http(),
        clock_fn=lambda: next(tick) * 0.025,  # 25.0 ms per request
        environ=CLEAN_ENV,
        lang="fr",
    )
    assert rep_slow["scale_slo_ok"] is False
    assert rep_slow["p95_latency_ms"] == 25.0
    assert any("SLO" in d for d in rep_slow["diagnostics"])


def test_corporate_compliance_refusal_and_zero_inference_guards() -> None:
    calls: list[str] = []

    def _never_called(method: str, url: str) -> tuple[int, dict[str, Any]]:
        calls.append(f"{method} {url}")
        return 200, {}

    for corp_env in (
        {"AMADEUS_ENV": "production"},
        {"USERDNSDOMAIN": "corp.example.internal"},
        {"HTTPS_PROXY": "http://gateway.zscaler.net:8080"},
        {"USERDOMAIN": "ACMECORP", "COMPUTERNAME": "LAPTOP01"},
    ):
        refused = scale_harness.run_noel_scale_burst(
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            devs=5,
            gamers=5,
            http_fn=_never_called,
            environ=corp_env,
            lang="fr",
        )
        assert refused["compliance"]["compliance_ok"] is False
        assert refused["requests_total"] == 0
        assert refused["scale_slo_ok"] is False
        _verify_sha256_receipt(refused)

    assert calls == []

    # Matching USERDOMAIN == COMPUTERNAME on a personal Windows PC is allowed
    win_personal = scale_harness.evaluate_personal_pc_compliance(
        environ={"USERDOMAIN": "HOME-PC", "COMPUTERNAME": "HOME-PC"},
        lang="en",
    )
    assert win_personal["compliance_ok"] is True

    # Zero-inference guards reject /v1/chat/completions, /chat/, prompt/messages, and hivemind/coder
    with pytest.raises(scale_harness.ScaleHarnessError, match="Zero-inference"):
        scale_harness._assert_zero_inference_request(
            "POST", "http://127.0.0.1:8000/v1/chat/completions", {}
        )
    with pytest.raises(scale_harness.ScaleHarnessError, match="Zero-inference"):
        scale_harness._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/onboarding/feedback",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
    with pytest.raises(scale_harness.ScaleHarnessError, match="Zero-inference"):
        scale_harness._assert_zero_inference_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/onboarding/feedback",
            {"note": "please query hivemind/coder"},
        )
    with pytest.raises(scale_harness.ScaleHarnessError, match="Unsupported HTTP method"):
        scale_harness._assert_zero_inference_request("DELETE", "http://127.0.0.1:8000/noel")


def test_cli_json_markdown_fr_en_determinism_and_stdlib_http_transport(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Exact deterministic receipt across two independent runs with deterministic clock
    def _run_once() -> dict[str, Any]:
        clk = itertools.count(0)
        return scale_harness.run_noel_scale_burst(
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            devs=4,
            gamers=4,
            workers=1,
            http_fn=_make_deterministic_mock_http(),
            clock_fn=lambda: next(clk) * 0.004,
            environ=CLEAN_ENV,
            lang="en",
        )

    r1 = _run_once()
    r2 = _run_once()
    assert r1 == r2
    assert r1["receipt_sha256"] == r2["receipt_sha256"]

    # 2. CLI main() in JSON, Markdown (FR & EN), CLI summary (FR & EN), and error exit codes
    rc_json = scale_harness.main(
        ["--gateway", "http://127.0.0.1:8000", "--devs", "2", "--gamers", "2", "--workers", "4", "--json"],
        http_fn=_make_deterministic_mock_http(),
        environ=CLEAN_ENV,
    )
    assert rc_json == 0
    parsed_json = json.loads(capsys.readouterr().out)
    assert parsed_json["scale_slo_ok"] is True
    assert parsed_json["compliance"]["financial_promise"] == "0.00 EUR / 0.00 USD"
    _verify_sha256_receipt(parsed_json)

    for lang_flag, expected_phrase in (
        ("fr", "Rapport de Charge Concurrente Routeur"),
        ("en", "Concurrent Router Burst Scale Report"),
    ):
        rc_md = scale_harness.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--devs",
                "2",
                "--gamers",
                "2",
                "--skip-register",
                "--markdown",
                "--lang",
                lang_flag,
            ],
            http_fn=_make_deterministic_mock_http(),
            environ=CLEAN_ENV,
        )
        assert rc_md == 0
        md_out = capsys.readouterr().out
        assert expected_phrase in md_out
        assert "0.00 USD" in md_out
        assert "SHA-256 Receipt" in md_out

    for lang_flag, expected_cli in (("fr", "Rafale Noël Routeur"), ("en", "Noël Router Scale Burst")):
        rc_cli = scale_harness.main(
            ["--gateway", "http://127.0.0.1:8000", "--devs", "1", "--gamers", "1", "--cli", "--lang", lang_flag],
            http_fn=_make_deterministic_mock_http(fail_route="/v1/models", fail_status=502),
            environ=CLEAN_ENV,
        )
        assert rc_cli == 2
        cli_out = capsys.readouterr().out
        assert expected_cli in cli_out
        assert "[ERROR]" in cli_out
        assert "[DIAG]" in cli_out

    rc_bad_arg = scale_harness.main(
        ["--gateway", "https://evil.example.org"],
        http_fn=_make_deterministic_mock_http(),
        environ=CLEAN_ENV,
    )
    assert rc_bad_arg == 2
    assert "Error:" in capsys.readouterr().err

    # 3. Live loopback ThreadingHTTPServer exercising default_http_request over stdlib urllib
    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/json-list":
                raw = b"[1, 2, 3]"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/empty":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/html":
                raw = b'<html><style nonce="AbCdEfGh123456">body{}</style></html>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Security-Policy", "style-src 'nonce-AbCdEfGh123456'")
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/err-json":
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "1")
                self.end_headers()
                self.wfile.write(b'{"error":"rate_limited"}')
            else:
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"upstream unavailable")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"echo": body}).encode("utf-8"))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        c_list, b_list, _ = scale_harness.default_http_request("GET", f"{base}/json-list")
        assert c_list == 200 and b_list == {"data": [1, 2, 3]}

        c_emp, b_emp, _ = scale_harness.default_http_request("GET", f"{base}/empty")
        assert c_emp == 204 and b_emp["html_length"] == 0

        c_html, b_html, h_html = scale_harness.default_http_request("GET", f"{base}/html")
        assert c_html == 200 and "html" in b_html
        assert scale_harness.extract_csp_nonce(h_html, b_html["html"]) == "AbCdEfGh123456"

        c_post, b_post, _ = scale_harness.default_http_request(
            "POST", f"{base}/v1/hivemind/onboarding/feedback", payload={"profile": "dev"}, headers={"X-Test": "1"}
        )
        assert c_post == 200 and b_post == {"echo": {"profile": "dev"}}

        c_429, b_429, h_429 = scale_harness.default_http_request("GET", f"{base}/err-json")
        assert c_429 == 429 and b_429["error"] == "rate_limited" and h_429["retry-after"] == "1"

        c_503, b_503, _ = scale_harness.default_http_request("GET", f"{base}/err-text")
        assert c_503 == 503 and "upstream unavailable" in b_503["error"]
    finally:
        server.shutdown()
        server.server_close()

    # Unreachable closed loopback port returns status 0 without raising
    c_closed, b_closed, _ = scale_harness.default_http_request("GET", f"http://127.0.0.1:{port}/health")
    assert c_closed == 0 and "network_error" in b_closed


def test_validation_edge_cases_and_helper_branches() -> None:
    # Language, platform, gateway URL, and participant IP guards
    assert scale_harness.normalize_lang("fr_CA") == "fr"
    assert scale_harness.normalize_lang("en-GB") == "en"
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.normalize_lang("es")

    assert scale_harness._normalize_platform_tag("Windows-11") == "windows"
    assert scale_harness._normalize_platform_tag("Darwin") == "macos"
    assert scale_harness._normalize_platform_tag("Linux") == "linux"
    assert scale_harness._normalize_platform_tag("$$$") == "unknown"

    assert scale_harness.is_allowed_gateway_host("") is False
    assert scale_harness.validate_gateway_url("http://[::1]:8080/") == "http://[::1]:8080"
    for bad_gw in (
        "",
        "http://127.0.0.1:8000\n",
        "http://127.0.0.1 :8000",
        "http://user:pw@127.0.0.1:8000",
        "http://127.0.0.1:8000?a=1",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:abc",
        "ftp://127.0.0.1:8000",
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
    ):
        with pytest.raises(scale_harness.ScaleHarnessError):
            scale_harness.validate_gateway_url(bad_gw)

    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.simulated_participant_ip(-1)

    # Percentile & HTTP result normalization branches
    assert scale_harness.compute_percentile_ms([], 95.0) == 0.0
    assert scale_harness.compute_percentile_ms([12.3456], 95.0) == 12.346
    assert scale_harness.compute_percentile_ms([10.0, 20.0, 30.0], 50.0) == 20.0
    assert scale_harness.compute_percentile_ms([10.0, 20.0, 30.0], 100.0) == 30.0
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.compute_percentile_ms([1.0], 101.0)
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.compute_percentile_ms([True], 50.0)  # type: ignore[list-item]
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.compute_percentile_ms([-1.0], 50.0)

    direct_resp = scale_harness.HttpResponse(status_code=200, body={"ok": True}, headers={})
    assert scale_harness._normalize_http_result(direct_resp) is direct_resp
    norm_dict = scale_harness._normalize_http_result(
        (200, {"csp": "style-src 'nonce-AbCdEf1234'", "headers": {"X-Custom": "1"}, "html": "<p>ok</p>"})
    )
    assert norm_dict.headers["content-security-policy"] == "style-src 'nonce-AbCdEf1234'"
    assert norm_dict.headers["x-custom"] == "1"
    norm_str_json = scale_harness._normalize_http_result((200, '{"status":"ok"}'))
    assert norm_str_json.body == {"status": "ok"}
    norm_other = scale_harness._normalize_http_result((200, 42))
    assert norm_other.body == {"data": 42}
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness._normalize_http_result("invalid")

    # 2-arg and 3-arg http_fn signatures + CSP nonce edge cases
    r2, _, _ = scale_harness._invoke_http_callable(
        lambda m, u: (200, {"m": m, "u": u}),
        "GET",
        "http://127.0.0.1:8000/noel",
    )
    assert r2.status_code == 200
    r3, _, _ = scale_harness._invoke_http_callable(
        lambda m, u, p: (200, {"m": m, "u": u, "p": p}),
        "POST",
        "http://127.0.0.1:8000/v1/hivemind/onboarding/feedback",
        {"profile": "dev"},
    )
    assert r3.status_code == 200
    r_exc, _, locked = scale_harness._invoke_http_callable(
        lambda m, u: (_ for _ in ()).throw(RuntimeError("boom")),
        "GET",
        "http://127.0.0.1:8000/noel",
    )
    assert r_exc.status_code == 500 and locked is False

    assert (
        scale_harness.extract_csp_nonce(
            {"content-security-policy": "style-src 'nonce-NonceOne1111' 'nonce-NonceTwo2222'"},
            "",
        )
        is None
    )
    assert (
        scale_harness.extract_csp_nonce(
            {"content-security-policy": "style-src 'nonce-HeaderOnlyNonce99'"},
            "<html><style>body{}</style></html>",
        )
        is None
    )
    assert (
        scale_harness.extract_csp_nonce(
            {"content-security-policy": "style-src 'nonce-HeaderOnlyNonce99'"},
            "",
        )
        == "HeaderOnlyNonce99"
    )

    # Validator & run_noel_scale_burst parameter validation guards
    empty_resp = scale_harness.HttpResponse(status_code=200, body={}, headers={})
    assert scale_harness._validate_manifest_resp(empty_resp)[0] is False
    assert scale_harness._validate_register_resp(empty_resp)[0] is False
    assert scale_harness._validate_models_resp(empty_resp)[0] is False
    assert scale_harness._validate_raid_status_resp(empty_resp)[0] is False
    assert scale_harness._validate_arena_vote_resp(empty_resp)[0] is False
    assert scale_harness._validate_arena_summary_resp(empty_resp)[0] is False
    assert scale_harness._validate_feedback_resp(empty_resp)[0] is False

    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.run_noel_scale_burst("http://127.0.0.1:8000", devs=0, gamers=0, environ=CLEAN_ENV)
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.run_noel_scale_burst(
            "http://127.0.0.1:8000", p95_slo_ms=100.0, p99_slo_ms=50.0, environ=CLEAN_ENV
        )
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.run_noel_scale_burst("http://127.0.0.1:8000", p95_slo_ms=-1.0, environ=CLEAN_ENV)
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.run_noel_scale_burst("http://127.0.0.1:8000", max_error_ratio=1.5, environ=CLEAN_ENV)
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.run_noel_scale_burst("http://127.0.0.1:8000", retry_backoff_s=-0.1, environ=CLEAN_ENV)
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.run_noel_scale_burst("http://127.0.0.1:8000", local_harness="yes", environ=CLEAN_ENV)  # type: ignore[arg-type]
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.run_noel_scale_burst("http://127.0.0.1:8000", include_register="yes", environ=CLEAN_ENV)  # type: ignore[arg-type]
    with pytest.raises(scale_harness.ScaleHarnessError):
        scale_harness.run_noel_scale_burst("http://127.0.0.1:8000", burst_tag="BAD TAG!", environ=CLEAN_ENV)
    with pytest.raises(scale_harness.ScaleHarnessError, match="restricted to loopback"):
        scale_harness.run_noel_scale_burst(
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            include_register=True,
            local_harness=False,
            environ=CLEAN_ENV,
        )


def test_zz_harness_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, f"Expected >= 95.0% line coverage on {HARNESS_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
