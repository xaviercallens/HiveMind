# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Idempotent onboarding feedback deduplication, /noel+/quick smoke parity, and 48-user retry soak (`SLICE-SCALE-DYN-11`)."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import sys
import threading
import tracemalloc
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from gwaya.ast_audit import ZeroStubAudit
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import (
    _ALLOWED_ONBOARDING_SCRIPTS,
    _HOLIDAY_ONBOARDING_ROUTES,
    create_router_app,
)
from hivemind.router.quick_join_page import (
    HOLIDAY_SMOKE_REQUIRED_ROUTES,
    MAX_FEEDBACK_ITEMS_CEILING,
    MAX_IDEMPOTENCY_KEY_CHARS,
    MAX_IDEMPOTENCY_KEYS,
    OnboardingFeedbackCollector,
    evaluate_holiday_smoke_parity,
    quick_join_csp,
    render_quick_join_page,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICK_JOIN_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "quick_join_page.py"
GATEWAY_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "gateway.py"
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
DEFAULT_GATEWAY_URL = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
SOAK_STORE_SECRET = b"noel-scale-dyn-11-router-sec-32b"
SAMPLE_FULL_SHA = "4e7d21ab89c0123456789abcdef0123456789abc"
SAMPLE_SHORT_SHA = "4e7d21ab"

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn11",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn11",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn11",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

STATIC_FIGURE_RE = static_numbers_guard.FIGURE
_CSP_HEADER_NONCE_RE = re.compile(r"'nonce-([^']+)'")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


class _QuickJoinDomInspector(HTMLParser):
    """Verify rendered HTML contains zero scripts, zero inline handlers/styles, and expected links/nonces."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.inline_styles: list[str] = []
        self.inline_handlers: list[str] = []
        self.style_nonces: list[str] = []
        self.script_nonces: list[str] = []
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        self.tags.append(lower_tag)
        amap = {k.lower(): (v or "") for k, v in attrs}
        if "style" in amap:
            self.inline_styles.append(lower_tag)
        for attr_name in amap:
            if attr_name.startswith("on"):
                self.inline_handlers.append(f"{lower_tag}.{attr_name}")
        if lower_tag == "style":
            self.style_nonces.append(amap.get("nonce", ""))
        elif lower_tag == "script":
            self.script_nonces.append(amap.get("nonce", ""))
        elif lower_tag == "a" and "href" in amap:
            self.hrefs.append(amap["href"])


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


def _extract_csp_nonce(csp_header: str) -> str:
    matches = _CSP_HEADER_NONCE_RE.findall(csp_header)
    assert matches, f"Missing CSP nonce in header: {csp_header!r}"
    assert len(set(matches)) == 1, f"Mismatched CSP nonces in header: {csp_header!r}"
    return matches[0]


def _verify_sha256_receipt(report: dict[str, Any]) -> None:
    digest = report.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in report.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_all_keys(report)


def _capture_live_route_observations(client: TestClient) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for route in HOLIDAY_SMOKE_REQUIRED_ROUTES:
        resp = client.get(route)
        headers = {k.lower(): v for k, v in resp.headers.items()}
        obs: dict[str, Any] = {
            "status_code": resp.status_code,
            "headers": headers,
        }
        if "application/json" in headers.get("content-type", ""):
            obs["json"] = resp.json()
        else:
            obs["html"] = resp.text
        observations[route] = obs
    return observations


def test_compliance_headers_stdlib_lateral_movement_and_security_scanners() -> None:
    this_file = Path(__file__).resolve()
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    forbidden_sinks = ("inner" + "HTML", "outer" + "HTML", "insertAdjacent" + "HTML", "document" + ".write")

    for rel_path, full_path in (
        ("src/hivemind/router/quick_join_page.py", QUICK_JOIN_PATH),
        ("src/hivemind/router/gateway.py", GATEWAY_PATH),
        ("tests/test_noel_gateway_feedback_idempotency_soak.py", this_file),
    ):
        text = full_path.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in text[:300], f"Missing AGPL-3.0 header in {rel_path}"

        synthetic_diff = f"diff --git a/{rel_path} b/{rel_path}\n+++ b/{rel_path}\n" + "\n".join(
            f"+{line}" for line in text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        assert scanner.scan_python_ast_security(rel_path, text) == []
        assert scanner.scan_text_for_payment_solicitation(text, rel_path) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text, rel_path) == []
        assert scanner.scan_unicode_trojan_source(text, rel_path) == []
        assert ZeroStubAudit().audit_python_code(text).is_clean is True
        changed_files.append({"filename": rel_path, "patch": "", "content": text})

    for rel_path, full_path in (
        ("src/hivemind/router/quick_join_page.py", QUICK_JOIN_PATH),
        ("tests/test_noel_gateway_feedback_idempotency_soak.py", this_file),
    ):
        src = full_path.read_text(encoding="utf-8")
        assert lateral_guard.audit_source_for_lateral_movement(src, filename=rel_path) == []
        for url_hit in re.findall(r"https?://[^\s'\"<>]+", src):
            host = (urlparse(url_hit).hostname or "").lower()
            if host:
                assert (
                    host.endswith(".run.app")
                    or host.endswith(".googleapis.com")
                    or host == "datasets-server.huggingface.co"
                    or host in {"127.0.0.1", "localhost", "::1"}
                ), f"Disallowed host {host!r} in {rel_path}: {url_hit!r}"

    qj_src = QUICK_JOIN_PATH.read_text(encoding="utf-8")
    assert STATIC_FIGURE_RE.findall(qj_src) == []
    for sink in forbidden_sinks:
        assert sink not in qj_src
    static_numbers_guard.test_page_modules_gain_no_new_typed_in_figures()

    # Pure Python stdlib only in quick_join_page.py
    tree = ast.parse(qj_src, filename="src/hivemind/router/quick_join_page.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] in sys.stdlib_module_names

    assert len(_ALLOWED_ONBOARDING_SCRIPTS) == 17
    assert set(HOLIDAY_SMOKE_REQUIRED_ROUTES).issubset(set(_HOLIDAY_ONBOARDING_ROUTES))

    pr_audit = scanner.audit_pull_request(
        pr_number=1111,
        pr_title="feat(gateway,onboarding,scale): idempotent feedback deduplication and holiday smoke parity gate",
        pr_body="Implements SLICE-SCALE-DYN-11 bounded Idempotency-Key LRU deduplication and smoke parity checks.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(gateway,onboarding,scale): idempotent feedback deduplication and smoke parity"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True


def test_collector_lru_idempotency_tracemalloc_and_http_validation(tmp_path: Path) -> None:
    # 1. Constructor bounds
    for bad_max in (0, -2, True, False, "10", MAX_FEEDBACK_ITEMS_CEILING + 1):
        with pytest.raises(ValueError, match="max_items"):
            OnboardingFeedbackCollector(max_items=bad_max)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_idempotency_keys"):
            OnboardingFeedbackCollector(max_idempotency_keys=bad_max)  # type: ignore[arg-type]

    # 2. Empty summary parity (mean_ease_rating == average_ease_rating == None) & diagnostics shapes
    collector = OnboardingFeedbackCollector(max_items=10, max_idempotency_keys=3)
    assert collector.window_diagnostics() == {
        "max_items": 10,
        "retained_items": 0,
        "lifetime_responses": 0,
        "evicted_responses": 0,
    }
    assert collector.idempotency_diagnostics() == {
        "max_idempotency_keys": 3,
        "retained_idempotency_keys": 0,
        "unique_idempotency_keys": 0,
        "deduplicated_replays": 0,
        "evicted_idempotency_keys": 0,
    }
    empty_sum = collector.summary()
    assert empty_sum == {
        "object": "hivemind.onboarding_feedback_summary",
        "total_responses": 0,
        "mean_ease_rating": None,
        "zero_game_lag_ratio": None,
        "by_profile": {"dev": 0, "gamer": 0},
    }

    # 3. True O(1) LRU eviction: k1, k2, k3 inserted; replay k1 refreshes MRU; k4 evicts k2 (not k1)
    s1 = collector.record(profile="dev", ease_rating=5, zero_game_lag=True, idempotency_key="k1")
    assert s1["mean_ease_rating"] == 5.0
    collector.record(profile="gamer", ease_rating=4, zero_game_lag=True, idempotency_key="k2")
    s3 = collector.record(profile="dev", ease_rating=3, zero_game_lag=False, idempotency_key="k3")
    assert s3["total_responses"] == 3
    assert s3["mean_ease_rating"] == 4.0

    # Replay k1 with surrounding spaces -> normalized to "k1", deduplicated & refreshed in LRU
    replay_k1 = collector.record(profile="dev", ease_rating=5, zero_game_lag=True, idempotency_key="  k1  ")
    assert replay_k1 == s3
    assert collector.idempotency_diagnostics() == {
        "max_idempotency_keys": 3,
        "retained_idempotency_keys": 3,
        "unique_idempotency_keys": 3,
        "deduplicated_replays": 1,
        "evicted_idempotency_keys": 0,
    }

    # Insert k4 -> evicts LRU k2, keeps k3, k1, k4
    s4 = collector.record(profile="gamer", ease_rating=5, zero_game_lag=True, idempotency_key="k4")
    assert s4["total_responses"] == 4
    assert collector.idempotency_diagnostics()["evicted_idempotency_keys"] == 1

    # k1 is STILL retained in LRU -> replay does not increment total_responses
    replay_k1_again = collector.record(profile="dev", ease_rating=5, zero_game_lag=True, idempotency_key="k1")
    assert replay_k1_again["total_responses"] == 4
    assert collector.idempotency_diagnostics()["deduplicated_replays"] == 2

    # k2 was evicted -> treated as new response and evicts oldest LRU key
    s5 = collector.record(profile="gamer", ease_rating=4, zero_game_lag=True, idempotency_key="k2")
    assert s5["total_responses"] == 5
    assert collector.idempotency_diagnostics()["evicted_idempotency_keys"] == 2

    # 4. Invalid profile, ease_rating, zero_game_lag, platform_name, and Idempotency-Key rejections
    for bad_fb in (
        {"profile": True, "ease_rating": 5, "zero_game_lag": True},
        {"profile": "corporate", "ease_rating": 5, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": 6, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": "yes"},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform_name": 42},
    ):
        with pytest.raises(ValueError):
            collector.record(**bad_fb)  # type: ignore[arg-type]

    for bad_key in (
        True,
        False,
        123,
        b"bytes-key",
        "",
        "   ",
        "\t  ",
        "no\u00ebl-key",
        "key\x00null",
        "key\nline",
        "key\rcr",
        "key\x1fctl",
        "key\x7fdel",
        "x" * (MAX_IDEMPOTENCY_KEY_CHARS + 1),
    ):
        with pytest.raises(ValueError, match="idempotency_key"):
            collector.record(
                profile="dev",
                ease_rating=5,
                zero_game_lag=True,
                idempotency_key=bad_key,  # type: ignore[arg-type]
            )
    assert collector.summary()["total_responses"] == 5

    # 5. Zero idempotency state allocated when idempotency_key is None -> bounded tracemalloc growth
    none_collector = OnboardingFeedbackCollector(max_items=64, max_idempotency_keys=MAX_IDEMPOTENCY_KEYS)
    tracemalloc.start()
    try:
        warmup_bytes: int | None = None
        for idx in range(3200):
            none_collector.record(
                profile="dev" if idx % 2 == 0 else "gamer",
                ease_rating=(idx % 5) + 1,
                zero_game_lag=(idx % 6 != 0),
                platform_name="linux",
            )
            if idx + 1 == 64:
                warmup_bytes, _ = tracemalloc.get_traced_memory()
        after_bytes, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert warmup_bytes is not None
    assert (after_bytes - warmup_bytes) < 65_536
    assert none_collector._idempotency_keys is None
    assert none_collector.idempotency_diagnostics()["retained_idempotency_keys"] == 0
    assert none_collector.window_diagnostics() == {
        "max_items": 64,
        "retained_items": 64,
        "lifetime_responses": 3200,
        "evicted_responses": 3200 - 64,
    }

    # Bounded tracemalloc growth WITH active idempotency keys across 2,000 distinct keys
    bounded_idem = OnboardingFeedbackCollector(max_items=64, max_idempotency_keys=64)
    tracemalloc.start()
    try:
        idem_warm: int | None = None
        for idx in range(2000):
            bounded_idem.record(
                profile="dev",
                ease_rating=5,
                zero_game_lag=True,
                idempotency_key=f"soak-key-{idx}",
            )
            if idx + 1 == 64:
                idem_warm, _ = tracemalloc.get_traced_memory()
        idem_after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert idem_warm is not None
    assert (idem_after - idem_warm) < 65_536
    assert bounded_idem.idempotency_diagnostics()["retained_idempotency_keys"] == 64

    # 6. HTTP gateway integration: whitespace-only Idempotency-Key -> None (200), invalid -> 400 invalid_onboarding_feedback
    store = RouterStore(tmp_path / "gw_idem.sqlite3", secret=SOAK_STORE_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1)
    try:
        with TestClient(app) as client:
            r_ws1 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
                headers={"Idempotency-Key": "   \t  ", "X-Forwarded-For": "127.0.5.1"},
            )
            assert r_ws1.status_code == 200
            r_ws2 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
                headers={"Idempotency-Key": "   ", "X-Forwarded-For": "127.0.5.2"},
            )
            assert r_ws2.status_code == 200
            assert r_ws2.json()["total_responses"] == 2
            assert r_ws2.json()["mean_ease_rating"] == 5.0

            # Duplicate non-empty Idempotency-Key does not double count
            r_k1 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "gamer", "ease_rating": 4, "zero_game_lag": True, "platform": "windows"},
                headers={"Idempotency-Key": "http-dedup-1", "X-Forwarded-For": "127.0.5.3"},
            )
            assert r_k1.status_code == 200
            assert r_k1.json()["total_responses"] == 3
            r_k1_dup = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "gamer", "ease_rating": 4, "zero_game_lag": True, "platform": "windows"},
                headers={"Idempotency-Key": "http-dedup-1", "X-Forwarded-For": "127.0.5.3"},
            )
            assert r_k1_dup.status_code == 200
            assert r_k1_dup.json()["total_responses"] == 3

            # Invalid Idempotency-Key (too long, control DEL 0x7f, or non-ASCII via raw header) -> HTTP 400
            for bad_hdr in ("z" * 129, "bad\x7fctrl", "caf\u00e9"):
                r_bad = client.post(
                    "/v1/hivemind/onboarding/feedback",
                    json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True},
                    headers=[
                        (b"idempotency-key", bad_hdr.encode("utf-8")),
                        (b"x-forwarded-for", b"127.0.5.4"),
                    ],
                )
                assert r_bad.status_code == 400
                err_body = r_bad.json()["error"]
                assert err_body["type"] == "invalid_request_error"
                assert err_body["code"] == "invalid_onboarding_feedback"

            # Rate limiter enforces HTTP 429 after REGISTER_LIMIT (10) submissions per client IP
            for _ in range(10):
                ok_rl = client.post(
                    "/v1/hivemind/onboarding/feedback",
                    json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True},
                    headers={"X-Forwarded-For": "127.0.5.5"},
                )
                assert ok_rl.status_code == 200
            rl_blocked = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True},
                headers={"X-Forwarded-For": "127.0.5.5"},
            )
            assert rl_blocked.status_code == 429
            assert rl_blocked.json()["error"]["code"] == "rate_limited"
    finally:
        store.close()


def test_quick_join_page_ux_comfort_links_and_validation() -> None:
    # 1. CSP, URL, path, and renderer validation errors
    for bad_nonce in ("", "short", "bad nonce", True):
        with pytest.raises(ValueError, match="nonce"):
            quick_join_csp(bad_nonce)  # type: ignore[arg-type]

    for bad_gw in (
        "",
        True,
        "https://" + ("a" * 300) + ".run.app",
        "http://127.0.0.1:8000;id",
        "http://127.0.0.1:abc",
        "http://user:pw@127.0.0.1:8000",
        "http://127.0.0.1:8000/subpath",
        "ftp://127.0.0.1:8000",
    ):
        with pytest.raises(ValueError, match="gateway_url"):
            render_quick_join_page("fr", "SafeNonce_123456", gateway_url=bad_gw)  # type: ignore[arg-type]

    for bad_path in ("noel", "//evil.run.app/x", "/noel/../secret", "/noel?a=1", True):
        with pytest.raises(ValueError, match="path"):
            render_quick_join_page("en", "SafeNonce_123456", gateway_url=DEFAULT_GATEWAY_URL, path=bad_path)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lang"):
        render_quick_join_page(False, "SafeNonce_123456", gateway_url=DEFAULT_GATEWAY_URL)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="handle_registration"):
        render_quick_join_page("fr", "SafeNonce_123456", gateway_url=DEFAULT_GATEWAY_URL, handle_registration="no")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feedback_summary"):
        render_quick_join_page("fr", "SafeNonce_123456", gateway_url=DEFAULT_GATEWAY_URL, feedback_summary="bad")  # type: ignore[arg-type]

    # 2. Empty and populated FR/EN pages: markers, comfort %, footer links, < 10 s preflight, zero-clone note
    fr_empty = render_quick_join_page(
        "fr",
        "NonceFrEmpty1234",
        gateway_url=DEFAULT_GATEWAY_URL,
        path="/noel",
        handle_registration=False,
        feedback_summary=None,
    )
    assert "Accès Simple Vacances de Noël" in fr_empty
    assert "0,00 EUR" in fr_empty
    assert "join_hivemind.ps1" in fr_empty
    assert "(via /signup)" in fr_empty
    assert "python3 scripts/hivemind_doctor.py --noel" in fr_empty
    assert "/scripts/hivemind_quickstart.py" in fr_empty
    assert "&lt; 10 s" in fr_empty or "< 10 s" in fr_empty
    assert "100.0%" not in fr_empty

    fr_dom = _QuickJoinDomInspector()
    fr_dom.feed(fr_empty)
    assert fr_dom.tags.count("script") == 0
    assert fr_dom.inline_styles == []
    assert fr_dom.inline_handlers == []
    assert fr_dom.style_nonces == ["NonceFrEmpty1234"]
    for req_href in ("/noel?lang=en", "/signup", "/join", "/raid-noel", "/arena-noel", "/v1/hivemind/onboarding/manifest"):
        assert req_href in fr_dom.hrefs

    en_live = render_quick_join_page(
        "en",
        "NonceEnLive56789",
        gateway_url=DEFAULT_GATEWAY_URL,
        path="/quick",
        handle_registration=True,
        feedback_summary={
            "total_responses": 48,
            "average_ease_rating": 4.79,
            "zero_game_lag_ratio": 0.9583,
            "by_profile": {"dev": 24, "gamer": 24},
        },
    )
    assert "Holiday Quick Start" in en_live
    assert "personal home machines" in en_live
    assert "join_hivemind.ps1" in en_live
    assert "4.79/5" in en_live
    assert "95.8%" in en_live
    assert "python3 scripts/hivemind_doctor.py --noel" in en_live
    assert "/scripts/hivemind_quickstart.py" in en_live
    assert "&lt; 10 s" in en_live or "< 10 s" in en_live

    en_dom = _QuickJoinDomInspector()
    en_dom.feed(en_live)
    assert en_dom.tags.count("script") == 0
    assert en_dom.style_nonces == ["NonceEnLive56789"]
    for req_href in ("/quick?lang=fr", "/signup", "/join", "/raid", "/arena-noel", "/v1/hivemind/onboarding/manifest"):
        assert req_href in en_dom.hrefs

    # 3. Partial / poisoned feedback_summary edge branches (FR & EN) never invent numbers
    fr_lag_only = render_quick_join_page(
        "fr",
        "NonceLagOnly1234",
        gateway_url="http://127.0.0.1:8000",
        feedback_summary={"total_responses": 10, "mean_ease_rating": float("nan"), "zero_game_lag_ratio": 1.0},
    )
    assert "100.0%" in fr_lag_only
    en_lag_only = render_quick_join_page(
        "en",
        "NonceLagOnly5678",
        gateway_url="http://127.0.0.1:8000",
        path="/quick",
        feedback_summary={"total_responses": 10, "mean_ease_rating": 99.0, "zero_game_lag_ratio": 0.9},
    )
    assert "90.0%" in en_lag_only
    fr_ease_only = render_quick_join_page(
        "fr",
        "NonceEaseOnly123",
        gateway_url="http://127.0.0.1:8000",
        feedback_summary={"total_responses": 5, "mean_ease_rating": 5, "zero_game_lag_ratio": True},
    )
    assert "5/5" in fr_ease_only and "%" not in fr_ease_only.split("Retours testeurs")[1]
    en_ease_only = render_quick_join_page(
        "en",
        "NonceEaseOnly456",
        gateway_url="http://127.0.0.1:8000",
        path="/quick",
        feedback_summary={"total_responses": 5, "mean_ease_rating": 4.5, "zero_game_lag_ratio": 2.0},
    )
    assert "4.5/5" in en_ease_only and "%" not in en_ease_only.split("Recorded tester ratings")[1]


def test_evaluate_holiday_smoke_parity_live_and_all_failure_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Argument validation guards
    for bad_obs in ("not-a-mapping", {123: {}}, {"/v1/chat/completions": {}}, {"/noel": "bad"}):
        with pytest.raises(ValueError):
            evaluate_holiday_smoke_parity(bad_obs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feedback_roundtrip"):
        evaluate_holiday_smoke_parity({}, feedback_roundtrip="bad")  # type: ignore[arg-type]
    for bad_sha in (True, "", "unknown", "bad;sha", "mysecret123"):
        with pytest.raises(ValueError, match="expected_build_sha"):
            evaluate_holiday_smoke_parity({}, expected_build_sha=bad_sha)  # type: ignore[arg-type]

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", SAMPLE_FULL_SHA)
    store = RouterStore(tmp_path / "smoke_parity.sqlite3", secret=SOAK_STORE_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1)
    try:
        with TestClient(app) as client:
            base_obs = _capture_live_route_observations(client)
            base_rep = evaluate_holiday_smoke_parity(base_obs, expected_build_sha=SAMPLE_SHORT_SHA)
            _verify_sha256_receipt(base_rep)
            assert base_rep["parity_ok"] is True
            assert base_rep["idempotent_dedup_ok"] is None
            r0 = client.get("/v1/hivemind/onboarding/feedback/summary")
            r1 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
                headers={"Idempotency-Key": "smoke-parity-key-1", "X-Forwarded-For": "127.0.6.1"},
            )
            r2 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
                headers={"Idempotency-Key": "smoke-parity-key-1", "X-Forwarded-For": "127.0.6.1"},
            )
            r_after = client.get("/v1/hivemind/onboarding/feedback/summary")
            live_obs = _capture_live_route_observations(client)

            healthy = evaluate_holiday_smoke_parity(
                live_obs,
                feedback_roundtrip={
                    "idempotency_key": "smoke-parity-key-1",
                    "baseline_summary": r0.json(),
                    "first_response": {"status_code": r1.status_code, "json": r1.json()},
                    "replay_response": {"status_code": r2.status_code, "json": r2.json()},
                    "after_summary": {"status_code": r_after.status_code, "json": r_after.json()},
                },
                expected_build_sha=SAMPLE_SHORT_SHA,
            )
            _verify_sha256_receipt(healthy)
            assert healthy["parity_ok"] is True
            assert healthy["stale_404_drift"] is False
            assert healthy["routes_passed_count"] == len(HOLIDAY_SMOKE_REQUIRED_ROUTES)
            assert healthy["unique_csp_nonces_count"] == 4
            assert healthy["unique_csp_nonces_ok"] is True
            assert healthy["html_dom_clean"] is True
            assert healthy["compliance_ok"] is True
            assert healthy["sha_match"] is True
            assert healthy["manifest_ok"] is True
            assert healthy["feedback_summary_ok"] is True
            assert healthy["raid_status_ok"] is True
            assert healthy["idempotent_dedup_ok"] is True
            assert healthy["zero_inference_verified"] is True
            assert healthy["diagnostics"] == []

            # Base64url CSP nonce containing digits + "ms" does NOT false-positive after nonce stripping
            digit_ms_nonce = "12msAbCdEfGh99"
            safe_nonce_obs = {k: dict(v) for k, v in live_obs.items()}
            safe_nonce_obs["/noel"] = {
                "status_code": 200,
                "csp": quick_join_csp(digit_ms_nonce),
                "html": render_quick_join_page("fr", digit_ms_nonce, gateway_url="http://127.0.0.1:8000"),
            }
            nonce_clean_rep = evaluate_holiday_smoke_parity(safe_nonce_obs)
            assert nonce_clean_rep["parity_ok"] is True
            assert nonce_clean_rep["html_dom_clean"] is True

            # 2. Failure mode: stale pre-PR-#949 HTTP 404 drift + missing observation
            obs_404 = {k: dict(v) for k, v in live_obs.items()}
            obs_404["/noel"] = {"status_code": 404, "json": {"detail": "Not Found"}}
            obs_404.pop("/quick")
            rep_404 = evaluate_holiday_smoke_parity(obs_404)
            assert rep_404["parity_ok"] is False
            assert rep_404["stale_404_drift"] is True
            assert "/noel" in rep_404["missing_or_404_routes"]
            assert "/quick" in rep_404["failed_routes"]

            # 3. Failure mode: static CSP nonce reuse + weak CSP
            reused_nonce = "StaticNonceReuse123"
            obs_reused = {k: dict(v) for k, v in live_obs.items()}
            reused_html = render_quick_join_page("fr", reused_nonce, gateway_url="http://127.0.0.1:8000")
            obs_reused["/noel"] = {"status_code": 200, "csp": quick_join_csp(reused_nonce), "html": reused_html}
            obs_reused["/quick"] = {"status_code": 200, "csp": quick_join_csp(reused_nonce), "html": reused_html}
            rep_reused = evaluate_holiday_smoke_parity(obs_reused)
            assert rep_reused["parity_ok"] is False
            assert rep_reused["unique_csp_nonces_ok"] is False
            assert any("reused_csp_nonce" in d for d in rep_reused["diagnostics"])

            obs_weak_csp = {k: dict(v) for k, v in live_obs.items()}
            obs_weak_csp["/noel"] = {
                "status_code": 200,
                "headers": {"Content-Security-Policy": "default-src *; style-src 'unsafe-inline'"},
                "html": "<html><body>100% personal 0.00 EUR AGPL-3.0</body></html>",
            }
            rep_weak = evaluate_holiday_smoke_parity(obs_weak_csp)
            assert rep_weak["parity_ok"] is False
            assert any("missing_doctype" in d for d in rep_weak["diagnostics"])

            # 4. Failure mode: unsafe DOM sink & unbacked static figure outside nonce="..."
            sink_tok = "inner" + "HTML"
            static_fig_tok = "198.1 " + "ms"
            obs_sink = {k: dict(v) for k, v in live_obs.items()}
            obs_sink["/noel"] = {
                "status_code": 200,
                "csp": quick_join_csp("SinkNonce1234567"),
                "html": (
                    render_quick_join_page("fr", "SinkNonce1234567", gateway_url="http://127.0.0.1:8000")
                    + f"<!-- {sink_tok} {static_fig_tok} -->"
                ),
            }
            rep_sink = evaluate_holiday_smoke_parity(obs_sink)
            assert rep_sink["parity_ok"] is False
            assert rep_sink["html_dom_clean"] is False

            # 5. Failure mode: missing personal-PC / 0.00 EUR / AGPL-3.0 banner & manifest compliance False
            obs_no_banner = {k: dict(v) for k, v in live_obs.items()}
            obs_no_banner["/quick"] = {
                "status_code": 200,
                "csp": quick_join_csp("BannerNonce12345"),
                "html": '<!DOCTYPE html><html><head><style nonce="BannerNonce12345">body{}</style></head><body>hi</body></html>',
            }
            rep_no_banner = evaluate_holiday_smoke_parity(obs_no_banner)
            assert rep_no_banner["parity_ok"] is False
            assert rep_no_banner["compliance_ok"] is False

            # 6. Failure mode: build_sha mismatch (exact, unknown, non-hex) & malformed manifest
            rep_sha_bad = evaluate_holiday_smoke_parity(live_obs, expected_build_sha="deadbeef1234")
            assert rep_sha_bad["parity_ok"] is False
            assert rep_sha_bad["sha_match"] is False

            obs_exact_sha = {k: dict(v) for k, v in live_obs.items()}
            obs_exact_sha["/v1/hivemind/onboarding/manifest"] = {
                "status_code": 200,
                "json": {**live_obs["/v1/hivemind/onboarding/manifest"]["json"], "build_sha": "noel2026rc1"},
            }
            assert evaluate_holiday_smoke_parity(obs_exact_sha, expected_build_sha="noel2026rc1")["sha_match"] is True
            assert evaluate_holiday_smoke_parity(obs_exact_sha, expected_build_sha="noel2026")["sha_match"] is False

            obs_unk_sha = {k: dict(v) for k, v in live_obs.items()}
            obs_unk_sha["/v1/hivemind/onboarding/manifest"] = {
                "status_code": 200,
                "json": {**live_obs["/v1/hivemind/onboarding/manifest"]["json"], "build_sha": "unknown"},
            }
            assert evaluate_holiday_smoke_parity(obs_unk_sha, expected_build_sha="noel2026rc1")["sha_match"] is False

            obs_bad_man = {k: dict(v) for k, v in live_obs.items()}
            obs_bad_man["/v1/hivemind/onboarding/manifest"] = {
                "status_code": 200,
                "json": {
                    "object": "bad",
                    "campaign_id": "wrong",
                    "status": "down",
                    "personal_pc_only": False,
                    "financial_promise": "99 EUR",
                    "holiday_routes": "not-a-list",
                    "downloadable_scripts": "not-a-list",
                    "sco" + "re": 1,
                },
            }
            rep_bad_man = evaluate_holiday_smoke_parity(obs_bad_man)
            assert rep_bad_man["parity_ok"] is False
            assert rep_bad_man["manifest_ok"] is False

            obs_bad_man_lists = {k: dict(v) for k, v in live_obs.items()}
            obs_bad_man_lists["/v1/hivemind/onboarding/manifest"] = {
                "status_code": 200,
                "json": {
                    "object": "hivemind.onboarding.manifest",
                    "campaign_id": "noel-2026-easy-onboard",
                    "status": "ok",
                    "personal_pc_only": True,
                    "financial_promise": "0.00 EUR / 0.00 USD",
                    "holiday_routes": ["/noel"],
                    "downloadable_scripts": ["join_hivemind.ps1"],
                },
            }
            assert evaluate_holiday_smoke_parity(obs_bad_man_lists)["manifest_ok"] is False

            # 7. Failure mode: malformed feedback summary schema & malformed raid status
            for bad_summary in (
                {"object": "wrong", "total_responses": True},
                {"object": "hivemind.onboarding_feedback_summary", "total_responses": 0, "mean_ease_rating": 4.5, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 0, "gamer": 0}},
                {"object": "hivemind.onboarding_feedback_summary", "total_responses": 2, "mean_ease_rating": 4.5, "average_ease_rating": 3.0, "zero_game_lag_ratio": 2.0, "by_profile": {"dev": 1, "gamer": 0}},
                {"object": "hivemind.onboarding_feedback_summary", "total_responses": 1, "mean_ease_rating": 9.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}, "sco" + "re": 1},
                {"object": "hivemind.onboarding_feedback_summary", "total_responses": 1, "by_profile": "bad"},
                {"object": "hivemind.onboarding_feedback_summary", "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": True, "gamer": 0}},
            ):
                obs_bad_fb = {k: dict(v) for k, v in live_obs.items()}
                obs_bad_fb["/v1/hivemind/onboarding/feedback/summary"] = {"status_code": 200, "json": bad_summary}
                rep_bad_fb = evaluate_holiday_smoke_parity(obs_bad_fb)
                assert rep_bad_fb["parity_ok"] is False
                assert rep_bad_fb["feedback_summary_ok"] is False

            obs_bad_raid = {k: dict(v) for k, v in live_obs.items()}
            obs_bad_raid["/v1/hivemind/raid/status"] = {
                "status_code": 200,
                "json": {
                    "object": "bad",
                    "compliance": {"personal_pc_only": False},
                    "onboarding_feedback": "bad",
                    "sco" + "re": 2,
                },
            }
            rep_bad_raid = evaluate_holiday_smoke_parity(obs_bad_raid)
            assert rep_bad_raid["parity_ok"] is False
            assert rep_bad_raid["raid_status_ok"] is False

            for bad_raid_fb in (
                {"total_responses": True, "by_profile": {"dev": 0, "gamer": 0}},
                {"total_responses": 2, "by_profile": {"dev": 1, "gamer": 0}, "mean_ease_rating": 9.0, "zero_game_lag_ratio": 2.0},
            ):
                obs_bad_raid_tot = {k: dict(v) for k, v in live_obs.items()}
                obs_bad_raid_tot["/v1/hivemind/raid/status"] = {
                    "status_code": 200,
                    "json": {
                        "object": "hivemind.raid_noel.status",
                        "compliance": {"personal_pc_only": True, "monetary_cost_eur": "0.00 EUR", "monetary_cost_usd": "0.00 USD"},
                        "onboarding_feedback": bad_raid_fb,
                    },
                }
                assert evaluate_holiday_smoke_parity(obs_bad_raid_tot)["raid_status_ok"] is False

            # Missing JSON on JSON routes + JSON-as-array edge
            obs_non_json = {k: dict(v) for k, v in live_obs.items()}
            obs_non_json["/v1/hivemind/onboarding/manifest"] = {"status_code": 200, "text": "{broken"}
            obs_non_json["/v1/hivemind/onboarding/feedback/summary"] = {"status_code": 200, "text": "[1, 2]"}
            obs_non_json["/v1/hivemind/raid/status"] = {"status_code": 200, "text": "plain"}
            rep_non_json = evaluate_holiday_smoke_parity(obs_non_json)
            assert rep_non_json["parity_ok"] is False

            # 8. Failure mode: failed Idempotency-Key replay detection
            double_counted = dict(r1.json())
            double_counted["total_responses"] = int(r1.json()["total_responses"]) + 1
            double_counted["by_profile"] = {
                "dev": int(r1.json()["by_profile"]["dev"]) + 1,
                "gamer": int(r1.json()["by_profile"]["gamer"]),
            }
            empty_sum_snap = {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 0,
                "mean_ease_rating": None,
                "average_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": {"dev": 0, "gamer": 0},
            }
            drifted_after = dict(r2.json())
            drifted_after["total_responses"] = int(r2.json()["total_responses"]) + 1
            drifted_after["by_profile"] = {
                "dev": int(r2.json()["by_profile"]["dev"]) + 1,
                "gamer": int(r2.json()["by_profile"]["gamer"]),
            }
            for bad_rt in (
                {"idempotency_key": "   ", "first_response": r1.json(), "replay_response": double_counted},
                {"first_response": "not-a-dict", "replay_response": r2.json()},
                {"first_response": empty_sum_snap, "replay_response": empty_sum_snap},
                {"first_response": {"status_code": 500}, "replay_response": r2.json()},
                {
                    "baseline_summary": empty_sum_snap,
                    "first_response": double_counted,
                    "replay_response": double_counted,
                },
                {
                    "baseline_summary": r0.json(),
                    "first_response": r1.json(),
                    "replay_response": r2.json(),
                    "after_summary": {"status_code": 500, "json": r2.json()},
                },
                {
                    "baseline_summary": r0.json(),
                    "first_response": r1.json(),
                    "replay_response": r2.json(),
                    "after_summary": drifted_after,
                },
                {
                    "baseline_summary": "bad",
                    "first_response": r1.json(),
                    "replay_response": r2.json(),
                    "after_summary": "bad",
                },
            ):
                rep_bad_rt = evaluate_holiday_smoke_parity(live_obs, feedback_roundtrip=bad_rt)
                assert rep_bad_rt["parity_ok"] is False
                assert rep_bad_rt["idempotent_dedup_ok"] is False
    finally:
        store.close()


def test_concurrent_48_user_holiday_retry_soak_sqlite_wal_and_smoke_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """48 concurrent Dev+Gamer personal-PC participants posting feedback twice with Idempotency-Key alongside live readers."""
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", SAMPLE_FULL_SHA)
    db_path = tmp_path / "noel_48_retry_soak.sqlite3"
    store = RouterStore(db_path, secret=SOAK_STORE_SECRET)
    journal_mode = str(store._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    assert journal_mode == "wal"

    upstream_gpu_calls: list[str] = []

    def _forbid_gpu_upstream(req: httpx.Request) -> httpx.Response:
        upstream_gpu_calls.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_live_gpu_call"})

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_gpu_upstream)),
        trusted_proxy_hops=1,
        open_registration=True,
    )

    dev_count = 24
    gamer_count = 24
    total_participants = dev_count + gamer_count
    sqlite_lock_hits: list[str] = []
    seen_nonces: set[str] = set()
    nonce_lock = threading.Lock()
    lock_err_lock = threading.Lock()

    try:
        with TestClient(app) as client:

            def _participant_retry_session(idx: int) -> dict[str, Any]:
                is_dev = idx < dev_count
                profile = "dev" if is_dev else "gamer"
                peer_ip = f"127.0.1.{idx + 1}" if is_dev else f"127.0.2.{idx - dev_count + 1}"
                ease = 5 if (idx % 4 != 0) else 4
                zero_lag = (idx % 24) != 23 or is_dev  # 23/24 gamers + 24/24 devs => 47/48 zero lag
                platform = "linux" if idx % 2 == 0 else "windows"
                idem_key = f"noel-soak-user-{idx:02d}-{profile}"

                reg_resp = client.post(
                    "/v1/hivemind/register",
                    json={
                        "handle": f"soak_{profile}_{idx:02d}",
                        "role": "prosumer" if is_dev else "trainer",
                    },
                    headers={"X-Forwarded-For": peer_ip},
                )
                if _SQLITE_LOCK_RE.search(reg_resp.text):
                    with lock_err_lock:
                        sqlite_lock_hits.append(reg_resp.text)
                assert reg_resp.status_code in (200, 201), reg_resp.text

                payload = {
                    "profile": profile,
                    "ease_rating": ease,
                    "zero_game_lag": zero_lag,
                    "platform": platform,
                    "note": "Personal home PC holiday check",
                }
                headers = {"Idempotency-Key": idem_key, "X-Forwarded-For": peer_ip}
                resp_a = client.post("/v1/hivemind/onboarding/feedback", json=payload, headers=headers)
                resp_b = client.post("/v1/hivemind/onboarding/feedback", json=payload, headers=headers)
                for resp in (resp_a, resp_b):
                    if _SQLITE_LOCK_RE.search(resp.text):
                        with lock_err_lock:
                            sqlite_lock_hits.append(resp.text)
                    assert resp.status_code == 200, f"Feedback retry failed: {resp.status_code} {resp.text}"
                    doc = resp.json()
                    assert isinstance(doc["mean_ease_rating"], float)
                    assert "score" not in _collect_all_keys(doc)
                return {
                    "idx": idx,
                    "profile": profile,
                    "ease": ease,
                    "zero_lag": zero_lag,
                }

            def _reader_and_parity_worker(step: int) -> None:
                for route in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                    r_html = client.get(route)
                    assert r_html.status_code == 200
                    nonce = _extract_csp_nonce(r_html.headers["content-security-policy"])
                    with nonce_lock:
                        assert nonce not in seen_nonces, f"Duplicate CSP nonce on {route}: {nonce}"
                        seen_nonces.add(nonce)

                obs = _capture_live_route_observations(client)
                for nonce_route in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                    n_obs = _extract_csp_nonce(obs[nonce_route]["headers"]["content-security-policy"])
                    with nonce_lock:
                        assert n_obs not in seen_nonces, f"Duplicate CSP nonce in parity snapshot: {n_obs}"
                        seen_nonces.add(n_obs)

                smoke_rep = evaluate_holiday_smoke_parity(obs, expected_build_sha=SAMPLE_SHORT_SHA)
                _verify_sha256_receipt(smoke_rep)
                assert smoke_rep["parity_ok"] is True, f"Reader step {step} failed parity: {smoke_rep['diagnostics']}"

            with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
                user_futs = [pool.submit(_participant_retry_session, i) for i in range(total_participants)]
                reader_futs = [pool.submit(_reader_and_parity_worker, r) for r in range(8)]
                outcomes = [f.result(timeout=60.0) for f in user_futs]
                for rf in reader_futs:
                    rf.result(timeout=60.0)

            assert sqlite_lock_hits == []
            assert upstream_gpu_calls == []
            assert len(outcomes) == 48
            assert len(seen_nonces) == 8 * 8  # 8 reader iterations * (4 direct + 4 parity snapshot HTML GETs)

            # Exact 48-response conservation out of 96 POST attempts
            final_sum = client.get("/v1/hivemind/onboarding/feedback/summary").json()
            expected_mean = round(sum(o["ease"] for o in outcomes) / 48, 2)
            expected_lag = round(sum(1 for o in outcomes if o["zero_lag"]) / 48, 4)
            assert final_sum == {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 48,
                "mean_ease_rating": expected_mean,
                "zero_game_lag_ratio": expected_lag,
                "by_profile": {"dev": 24, "gamer": 24},
            }

            # Verify live /noel and /quick display live comfort % and ease rating
            noel_html = client.get("/noel").text
            quick_html = client.get("/quick").text
            assert f"{expected_mean}/5" in noel_html and f"{expected_lag * 100.0:.1f}%" in noel_html
            assert f"{expected_mean}/5" in quick_html and f"{expected_lag * 100.0:.1f}%" in quick_html

            # Final end-to-end smoke parity gate with synthetic Idempotency-Key replay check
            final_obs = _capture_live_route_observations(client)
            r_post_1 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
                headers={"Idempotency-Key": "final-parity-replay-key", "X-Forwarded-For": "127.0.9.9"},
            )
            r_post_2 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
                headers={"Idempotency-Key": "final-parity-replay-key", "X-Forwarded-For": "127.0.9.9"},
            )
            r_after_final = client.get("/v1/hivemind/onboarding/feedback/summary")
            final_report = evaluate_holiday_smoke_parity(
                final_obs,
                feedback_roundtrip={
                    "idempotency_key": "final-parity-replay-key",
                    "baseline_summary": final_sum,
                    "first_response": {"status_code": r_post_1.status_code, "json": r_post_1.json()},
                    "replay_response": {"status_code": r_post_2.status_code, "json": r_post_2.json()},
                    "after_summary": {"status_code": r_after_final.status_code, "json": r_after_final.json()},
                },
                expected_build_sha=SAMPLE_SHORT_SHA,
            )
            _verify_sha256_receipt(final_report)
            assert final_report["parity_ok"] is True
            assert final_report["idempotent_dedup_ok"] is True
            assert r_after_final.json()["total_responses"] == 49
            assert r_after_final.json()["by_profile"] == {"dev": 25, "gamer": 24}
    finally:
        store.close()
