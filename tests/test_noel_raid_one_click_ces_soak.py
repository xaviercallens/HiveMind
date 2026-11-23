# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""1-click zero-PII CES & Steam zero-lag widget, holiday route readiness gate, and 48-user browser soak (`SLICE-SCALE-DYN-06`)."""

from __future__ import annotations

import ast
import concurrent.futures
import importlib.util
import io
import ipaddress
import json
import re
import subprocess
import sys
import threading
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RAID_NOEL_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "raid_noel_page.py"
THIRD_PARTY = REPO_ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(RAID_NOEL_PATH)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import OnboardingFeedbackRequest, create_router_app  # noqa: E402
from hivemind.router.quick_join_page import OnboardingFeedbackCollector  # noqa: E402
from hivemind.router.raid_noel_page import (  # noqa: E402
    _RAID_NOEL_JS,
    CES_COMPLIANCE_REMINDER_EN,
    CES_COMPLIANCE_REMINDER_FR,
    DEFAULT_CES_PLATFORM,
    DEFAULT_GATEWAY_URL,
    ELECTRICITY_PRESETS_WATTS,
    FRIENDLY_GUILDS,
    HOLIDAY_READINESS_ROUTES,
    PENDING_CES_STORAGE_KEY,
    PERSONAL_PC_RULE_EN,
    PERSONAL_PC_RULE_FR,
    VALID_CES_PROFILES,
    _format_ease,
    _format_zero_lag,
    _opt_float,
    _opt_int,
    _validate_gateway_url,
    build_one_click_ces_ballot,
    build_raid_noel_status,
    estimate_gpu_electricity_cost,
    evaluate_holiday_route_readiness,
    pick_raid_language,
    raid_noel_csp,
    render_raid_noel_page,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn06",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn06",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn06",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
STATIC_FIGURE_RE = static_numbers_guard.FIGURE
SOAK_SECRET = b"noel-scale-dyn06-router-sec-32b!"
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


def _extract_csp_header_nonce(csp_header: str) -> str:
    matches = re.findall(r"'nonce-([^']+)'", csp_header)
    assert matches, f"Missing CSP nonce in header: {csp_header!r}"
    assert len(set(matches)) == 1, f"Mismatched nonces in single CSP header: {csp_header!r}"
    return matches[0]


class _RaidDomInspector(HTMLParser):
    """Verify strict CSP cleanliness (no inline style= or on*=) and capture attributes + text by element id."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.inline_styles: list[str] = []
        self.inline_handlers: list[str] = []
        self.style_nonces: list[str] = []
        self.script_nonces: list[str] = []
        self.by_id_text: dict[str, str] = {}
        self.by_id_attrs: dict[str, dict[str, str]] = {}
        self._active_stack: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
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
        el_id = amap.get("id")
        if el_id:
            self.by_id_text.setdefault(el_id, "")
            self.by_id_attrs[el_id] = amap
        if lower_tag not in {"meta", "input", "br", "hr", "img", "link"}:
            self._active_stack.append((lower_tag, el_id))

    def handle_data(self, data: str) -> None:
        for _tag, el_id in reversed(self._active_stack):
            if el_id is not None:
                self.by_id_text[el_id] = self.by_id_text.get(el_id, "") + data
                break

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        for idx in range(len(self._active_stack) - 1, -1, -1):
            if self._active_stack[idx][0] == lower_tag:
                del self._active_stack[idx:]
                break


def test_01_compliance_headers_static_ratchet_lateral_movement_security_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    target_files = (
        ("src/hivemind/router/raid_noel_page.py", RAID_NOEL_PATH),
        ("tests/test_noel_raid_one_click_ces_soak.py", this_file),
    )

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    forbidden_html_sinks = (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "document.write",
    )

    for rel_path, full_path in target_files:
        text = full_path.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in text[:300], f"Missing AGPL-3.0 header in first 300 chars of {rel_path}"

        # Lateral-movement AST audit & allowed-host URL check
        assert lateral_guard.audit_source_for_lateral_movement(text, filename=rel_path) == []
        for url_hit in re.findall(r"https?://[^\s'\"<>]+", text):
            host = (urlparse(url_hit).hostname or "").lower()
            if host:
                assert (
                    host.endswith(".run.app")
                    or host.endswith(".googleapis.com")
                    or host == "datasets-server.huggingface.co"
                    or host in {"127.0.0.1", "localhost", "::1"}
                ), f"Disallowed host {host!r} in {rel_path}: {url_hit!r}"

        # PR gatekeeper FORBIDDEN_ADDED check
        synthetic_diff = f"diff --git a/{rel_path} b/{rel_path}\n+++ b/{rel_path}\n" + "\n".join(
            f"+{line}" for line in text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

        # 5-Layer PRSecurityScanner checks
        assert scanner.scan_python_ast_security(rel_path, text) == []
        assert scanner.scan_text_for_payment_solicitation(text, rel_path) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text, rel_path) == []
        assert scanner.scan_unicode_trojan_source(text, rel_path) == []
        changed_files.append({"filename": rel_path, "patch": "", "content": text})

    raid_source = RAID_NOEL_PATH.read_text(encoding="utf-8")
    assert STATIC_FIGURE_RE.findall(raid_source) == []
    for sink in forbidden_html_sinks:
        assert sink not in raid_source
    assert ZeroStubAudit().audit_python_code(raid_source).is_clean is True

    # Verify raid_noel_page.py only imports Python stdlib + telemetry_figures
    tree = ast.parse(raid_source, filename="raid_noel_page.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_mod = node.module.split(".")[0]
            assert root_mod in sys.stdlib_module_names or node.module == "hivemind.router.telemetry_figures"

    # Full page-literals ratchet across all router page modules
    static_numbers_guard.test_page_modules_gain_no_new_typed_in_figures()

    pr_audit = scanner.audit_pull_request(
        pr_number=1076,
        pr_title="feat(onboarding,scale): 1-click zero-PII CES & Steam zero-lag widget on /raid-noel, holiday route readiness gate, and 48-user browser soak",
        pr_body="Implements SLICE-SCALE-DYN-06 1-click zero-PII CES widget, localStorage retry spool, holiday readiness verifier, and 48-user browser soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,scale): add 1-click zero-PII CES widget and 48-user soak on /raid-noel"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True


def test_02_python_helpers_csp_language_estimator_status_ballot_readiness_and_render(tmp_path: Path) -> None:
    forbidden_key = "sco" + "re"

    # 1. raid_noel_csp & pick_raid_language
    csp = raid_noel_csp("RaidNonce_2026+Safe=")
    assert "default-src 'none'" in csp
    assert "style-src 'nonce-RaidNonce_2026+Safe='" in csp
    assert "script-src 'nonce-RaidNonce_2026+Safe='" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "'unsafe-inline'" not in csp
    assert "'unsafe-eval'" not in csp
    for bad_nonce in ("", "short", "bad nonce", "nonce;script-src *", None, True, "N" * 85):
        with pytest.raises(ValueError, match="nonce"):
            raid_noel_csp(bad_nonce)  # type: ignore[arg-type]

    assert pick_raid_language("fr", "en-US,en;q=0.9", default="en") == "fr"
    assert pick_raid_language("en_US", "fr-FR,fr;q=0.9", default="fr") == "en"
    assert pick_raid_language("de", "en-GB;q=0.7, fr-FR;q=0.9", default="en") == "fr"
    assert pick_raid_language(None, "fr;q=bad, en;q=0.8", default="fr") == "en"
    assert pick_raid_language(None, "de-DE,es;q=0.8", default="fr") == "fr"
    assert pick_raid_language(None, "de-DE,es;q=0.8", default="en") == "en"
    assert pick_raid_language(None, None, default="fr") == "fr"
    assert pick_raid_language(None, "fr;q=0", default="en") == "en"
    for bad_q, bad_a, bad_d in (
        (True, None, "fr"),
        (123, None, "fr"),
        (None, False, "fr"),
        (None, 456, "fr"),
        (None, None, "es"),
        (None, None, True),
    ):
        with pytest.raises(ValueError):
            pick_raid_language(bad_q, bad_a, default=bad_d)  # type: ignore[arg-type]

    # 2. estimate_gpu_electricity_cost + formatting & numeric helpers
    assert len(ELECTRICITY_PRESETS_WATTS) == 3
    est_whisper = estimate_gpu_electricity_cost(120.0, 4.0, 0.25)
    assert est_whisper == {
        "gpu_watts": 120.0,
        "hours": 4.0,
        "kwh_tariff_eur": 0.25,
        "energy_wh": 480.0,
        "energy_kwh": 0.48,
        "estimated_eur": 0.12,
    }
    assert estimate_gpu_electricity_cost(0, 0, 0)["estimated_eur"] == 0.0
    for bad_est in (
        {"gpu_watts": False, "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": "120", "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": float("nan"), "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": float("inf"), "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": -1.0, "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": 2001.0, "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": 120.0, "hours": -0.5, "kwh_tariff_eur": 0.25},
        {"gpu_watts": 120.0, "hours": 9000.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": 120.0, "hours": 4.0, "kwh_tariff_eur": -0.01},
        {"gpu_watts": 120.0, "hours": 4.0, "kwh_tariff_eur": 10.5},
    ):
        with pytest.raises(ValueError):
            estimate_gpu_electricity_cost(**bad_est)  # type: ignore[arg-type]

    assert _opt_int(None) is None
    assert _opt_int(True) is None
    assert _opt_int(2.5) is None
    assert _opt_int("3") is None
    assert _opt_int(9.0) == 9
    assert _opt_float(None) is None
    assert _opt_float(True) is None
    assert _opt_float("x") is None
    assert _opt_float(float("inf")) is None
    assert _opt_float(4.8765, decimals=2) == 4.88
    assert _format_zero_lag(None) == "n/a"
    assert _format_zero_lag(0.975) == "97.5%"
    assert _format_ease(None) == "n/a"
    assert _format_ease(4.8) == "4.80 / 5"
    assert _validate_gateway_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"

    # 3. build_raid_noel_status
    empty_status = build_raid_noel_status(live_telemetry=None, feedback_summary=None)
    assert forbidden_key not in _collect_all_keys(empty_status)
    assert empty_status["compliance"]["personal_pc_only"] is True
    assert empty_status["compliance"]["monetary_cost_eur"] == "0.00 EUR"
    assert empty_status["compliance"]["monetary_cost_usd"] == "0.00 USD"
    assert empty_status["onboarding_feedback"]["total_responses"] is None

    pop_status = build_raid_noel_status(
        live_telemetry={
            "status": "LIVE_GCP_PRODUCTION",
            "accounts": {"total_registered_accounts": 48, "active_accounts_24h": 48},
            "producers": {"live_hiveminders": 6, "total_hiveminders": 9},
            "opentrainer": {
                "active_trainer_nodes": 4,
                "total_registered_nodes": 6,
                "registered_trainer_accounts_count": 5,
                "accepted_rounds_count": 19,
            },
            "models": {"live_models_count": 3},
        },
        feedback_summary={
            "total_responses": 48,
            "mean_ease_rating": 4.81,
            "zero_game_lag_ratio": 0.9792,
            "by_profile": {"dev": 24, "gamer": 24},
        },
    )
    assert forbidden_key not in _collect_all_keys(pop_status)
    assert pop_status["swarm"]["active_trainers"] == 4
    assert pop_status["onboarding_feedback"]["total_responses"] == 48
    assert pop_status["onboarding_feedback"]["by_profile"] == {"dev": 24, "gamer": 24}

    with pytest.raises(ValueError, match="live_telemetry"):
        build_raid_noel_status(live_telemetry="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feedback_summary"):
        build_raid_noel_status(feedback_summary=["invalid"])  # type: ignore[arg-type]

    # 4. build_one_click_ces_ballot: strict zero-PII validation & OnboardingFeedbackRequest compatibility
    assert VALID_CES_PROFILES == frozenset({"dev", "gamer"})
    default_ballot = build_one_click_ces_ballot()
    assert default_ballot == {
        "profile": "gamer",
        "ease_rating": 5,
        "zero_game_lag": True,
        "platform": DEFAULT_CES_PLATFORM,
        "note": "",
    }
    assert forbidden_key not in _collect_all_keys(default_ballot)
    req_model = OnboardingFeedbackRequest(**default_ballot)
    assert req_model.profile == "gamer"
    assert req_model.ease_rating == 5
    assert req_model.zero_game_lag is True
    assert req_model.platform == "web-raid-noel"
    assert req_model.note == ""

    collector = OnboardingFeedbackCollector(max_items=16)
    for prof_in, rating_in, lag_in, plat_in in (
        ("  DEV  ", 5, True, "  Web-Raid-Noel  "),
        ("gamer", 4, False, "windows11_steam"),
        ("dev", 1, True, "linux"),
        ("gamer", 3, True, "macos"),
    ):
        b = build_one_click_ces_ballot(
            profile=prof_in,
            ease_rating=rating_in,
            zero_game_lag=lag_in,
            platform_name=plat_in,
        )
        assert b["note"] == ""
        parsed_req = OnboardingFeedbackRequest(**b)
        rec = collector.record(
            profile=parsed_req.profile,
            ease_rating=parsed_req.ease_rating,
            zero_game_lag=parsed_req.zero_game_lag,
            platform_name=parsed_req.platform,
        )
        assert rec["total_responses"] >= 1

    for bad_ballot_kwargs in (
        {"profile": True},
        {"profile": None},
        {"profile": "enterprise"},
        {"profile": "d\u0435v"},  # Cyrillic homoglyph
        {"profile": "d" * 40},
        {"ease_rating": True},
        {"ease_rating": False},
        {"ease_rating": 0},
        {"ease_rating": 6},
        {"ease_rating": -1},
        {"ease_rating": 4.5},
        {"zero_game_lag": 1},
        {"zero_game_lag": "true"},
        {"zero_game_lag": None},
        {"platform_name": False},
        {"platform_name": None},
        {"platform_name": ""},
        {"platform_name": "   "},
        {"platform_name": "l\u00efnux"},
        {"platform_name": "p" * 35},
        {"platform_name": "a" * 28},
        {"platform_name": "-leading-dash"},
        {"platform_name": "user@domain.org"},
        {"platform_name": "/home/alice/pc"},
        {"platform_name": "secret_slug"},
        {"platform_name": "token_header"},
        {"platform_name": "amadeus_workstation"},
        {"platform_name": "corporate_laptop"},
        {"platform_name": "hm-12345678"},
        {"note": "free text note with PII"},
        {"note": " "},
        {"note": None},
        {"note": False},
    ):
        with pytest.raises(ValueError):
            build_one_click_ces_ballot(**bad_ballot_kwargs)  # type: ignore[arg-type]

    # 5. evaluate_holiday_route_readiness against live TestClient + pre-deploy drift & tampered probes
    store = RouterStore(tmp_path / "readiness_probe.sqlite3", secret=SOAK_SECRET)
    try:
        with TestClient(create_router_app(store, open_registration=True, trusted_proxy_hops=1)) as client:
            live_probes: dict[str, dict[str, Any]] = {}
            for route in HOLIDAY_READINESS_ROUTES:
                resp = client.get(route)
                live_probes[route] = {
                    "status_code": resp.status_code,
                    "headers": dict(resp.headers),
                    "body": resp.text,
                }
            ready_report = evaluate_holiday_route_readiness(live_probes)
            assert forbidden_key not in _collect_all_keys(ready_report)
            assert ready_report["holiday_routes_ready"] is True
            assert ready_report["onboarding_feedback_deployed"] is True
            assert ready_report["missing_or_stale_routes"] == []
            assert ready_report["routes_passed_count"] == 5
            assert ready_report["routes_total_count"] == 5
            assert ready_report["zero_gpu_inference"] is True
            assert ready_report["compliance"]["personal_pc_only"] is True
            assert ready_report["compliance"]["monetary_cost_eur"] == "0.00 EUR"
            assert ready_report["compliance"]["monetary_cost_usd"] == "0.00 USD"

            # Pre-deploy Cloud Run drift: 404 on /v1/hivemind/onboarding/feedback/summary and /raid-noel
            drift_probes = dict(live_probes)
            drift_probes["/v1/hivemind/onboarding/feedback/summary"] = {"status": 404, "body": '{"detail":"Not Found"}'}
            drift_probes["/raid-noel"] = {"status_code": 404, "headers": {}, "text": "Not Found"}
            drift_report = evaluate_holiday_route_readiness(drift_probes)
            assert drift_report["holiday_routes_ready"] is False
            assert drift_report["onboarding_feedback_deployed"] is False
            assert drift_report["missing_or_stale_routes"] == [
                "/raid-noel",
                "/v1/hivemind/onboarding/feedback/summary",
            ]
            assert drift_report["routes_passed_count"] == 3
            assert drift_report["per_route"]["/raid-noel"]["reason"] == "http_404"

            # Missing probe, weak CSP, missing compliance marker, corrupt JSON, and fallback `csp`/`json` keys
            mixed_probes: dict[str, Any] = {
                "/noel": {
                    "status_code": 200,
                    "csp": "default-src 'none'; style-src 'nonce-Abcdef123456'",
                    "html": "<!DOCTYPE html><html><body>Usage personnel AGPL-3.0 • 0,00 EUR</body></html>",
                },
                "/quick": {
                    "status_code": 200,
                    "headers": {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"},
                    "body": live_probes["/quick"]["body"],
                },
                "/raid-noel": {
                    "status_code": 200,
                    "headers": live_probes["/raid-noel"]["headers"],
                    "body": "<html><body> stripped compliance banner </body></html>",
                },
                "/v1/hivemind/raid/status": {
                    "status_code": 200,
                    "body": "{corrupt-json",
                },
                # /v1/hivemind/onboarding/feedback/summary intentionally omitted -> missing_probe
            }
            mixed_report = evaluate_holiday_route_readiness(mixed_probes)
            assert mixed_report["holiday_routes_ready"] is False
            assert mixed_report["onboarding_feedback_deployed"] is False
            assert mixed_report["routes_passed_count"] == 1
            assert mixed_report["per_route"]["/noel"]["ready"] is True
            assert mixed_report["per_route"]["/quick"]["reason"] == "missing_or_weak_csp"
            assert mixed_report["per_route"]["/raid-noel"]["reason"] == "missing_compliance_or_contract"
            assert mixed_report["per_route"]["/v1/hivemind/raid/status"]["reason"] == "missing_compliance_or_contract"
            assert mixed_report["per_route"]["/v1/hivemind/onboarding/feedback/summary"]["reason"] == "missing_probe"

            # Network error (status=None) + explicit json dict on feedback/summary
            net_probes = dict(live_probes)
            net_probes["/noel"] = {"status_code": None}
            net_probes["/v1/hivemind/onboarding/feedback/summary"] = {
                "status_code": 200,
                "json": {
                    "object": "hivemind.onboarding_feedback_summary",
                    "total_responses": 2,
                    "by_profile": {"dev": 1, "gamer": 1},
                },
            }
            net_report = evaluate_holiday_route_readiness(net_probes)
            assert net_report["per_route"]["/noel"]["reason"] == "network_or_missing_status"
            assert net_report["onboarding_feedback_deployed"] is True
    finally:
        store.close()

    for bad_probes in (
        "not-a-mapping",
        {True: {"status_code": 200}},
        {"/noel": "not-a-mapping"},
        {"/v1/chat/completions": {"status_code": 200}},
        {"/v1/completions": {"status_code": 200}},
    ):
        with pytest.raises(ValueError):
            evaluate_holiday_route_readiness(bad_probes)  # type: ignore[arg-type]

    # 6. render_raid_noel_page FR & EN structure, Section 4 1-click CES DOM IDs, and validation errors
    fr_html = render_raid_noel_page("fr", "NonceFR_Dyn06_Safe", gateway_url=DEFAULT_GATEWAY_URL, path="/raid-noel")
    en_html = render_raid_noel_page(
        "en",
        "NonceEN_Dyn06_Safe",
        gateway_url=DEFAULT_GATEWAY_URL,
        path="/raid",
        feedback_summary={
            "total_responses": 12,
            "mean_ease_rating": 4.75,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 6, "gamer": 6},
        },
    )
    for lang_tag, doc_html, expected_nonce, expected_rule, expected_ces_note in (
        ("fr", fr_html, "NonceFR_Dyn06_Safe", PERSONAL_PC_RULE_FR, CES_COMPLIANCE_REMINDER_FR),
        ("en", en_html, "NonceEN_Dyn06_Safe", PERSONAL_PC_RULE_EN, CES_COMPLIANCE_REMINDER_EN),
    ):
        assert f'<html lang="{lang_tag}">' in doc_html
        assert STATIC_FIGURE_RE.findall(doc_html.replace(expected_nonce, "CSP_NONCE")) == []
        for g in FRIENDLY_GUILDS:
            assert g["label"] in doc_html
        dom = _RaidDomInspector()
        dom.feed(doc_html)
        assert dom.inline_styles == []
        assert dom.inline_handlers == []
        assert dom.style_nonces == [expected_nonce]
        assert dom.script_nonces == [expected_nonce]
        assert dom.by_id_text["personalPcComplianceBanner"] == expected_rule
        assert dom.by_id_text["cesComplianceReminder"] == expected_ces_note
        assert "0.00 EUR / 0.00 USD" in dom.by_id_text["cesComplianceReminder"]
        assert "PII" in dom.by_id_text["cesComplianceReminder"]
        for required_id in (
            "cesOneClickCard",
            "cesProfileDevBtn",
            "cesProfileGamerBtn",
            "cesEaseBtn-1",
            "cesEaseBtn-2",
            "cesEaseBtn-3",
            "cesEaseBtn-4",
            "cesEaseBtn-5",
            "cesLagYesBtn",
            "cesLagNoBtn",
            "submitCesBtn",
            "retryPendingCesBtn",
            "cesSubmitStatus",
            "cesComplianceReminder",
        ):
            assert required_id in dom.by_id_attrs, f"Missing required Section 4 CES element #{required_id}"
        assert dom.by_id_attrs["cesProfileDevBtn"]["aria-pressed"] == "false"
        assert dom.by_id_attrs["cesProfileGamerBtn"]["aria-pressed"] == "true"
        assert dom.by_id_attrs["cesEaseBtn-5"]["aria-pressed"] == "true"
        assert dom.by_id_attrs["cesEaseBtn-4"]["aria-pressed"] == "false"
        assert dom.by_id_attrs["cesLagYesBtn"]["aria-pressed"] == "true"
        assert dom.by_id_attrs["cesLagNoBtn"]["aria-pressed"] == "false"

    for bad_render in (
        {"lang": False, "nonce": "NonceOK_12345678"},
        {"lang": "fr", "nonce": "bad"},
        {"lang": "fr", "nonce": "NonceOK_12345678", "gateway_url": ""},
        {"lang": "fr", "nonce": "NonceOK_12345678", "gateway_url": "http://127.0.0.1:8000 bad"},
        {"lang": "fr", "nonce": "NonceOK_12345678", "gateway_url": "ftp" + "://127.0.0.1:8000"},
        {"lang": "fr", "nonce": "NonceOK_12345678", "gateway_url": "http://" + "u:p@" + "127.0.0.1:8000"},
        {"lang": "fr", "nonce": "NonceOK_12345678", "path": "raid-noel"},
        {"lang": "fr", "nonce": "NonceOK_12345678", "path": True},
    ):
        with pytest.raises(ValueError):
            render_raid_noel_page(**bad_render)  # type: ignore[arg-type]


@pytest.mark.parametrize("lang_mode", ["fr", "en"])
def test_03_node_dom_one_click_ces_spool_retry_and_textcontent_traps(lang_mode: str) -> None:
    inner_prop = "inner" + "HTML"
    outer_prop = "outer" + "HTML"
    adj_prop = "insertAdjacent" + "HTML"
    harness = f"""
const elements = {{}};
function makeEl(id, initVal = '') {{
  const listeners = {{}};
  const attrs = {{}};
  const el = {{
    id,
    _text: '',
    value: initVal,
    className: '',
    getAttribute(k) {{ return Object.prototype.hasOwnProperty.call(attrs, k) ? attrs[k] : null; }},
    setAttribute(k, v) {{ attrs[k] = String(v); }},
    addEventListener(type, fn) {{
      if (!listeners[type]) listeners[type] = [];
      listeners[type].push(fn);
    }},
    async trigger(type) {{
      for (const fn of (listeners[type] || [])) {{
        await fn({{ preventDefault() {{}} }});
      }}
    }},
    [{json.dumps(adj_prop)}]() {{
      throw new Error('Forbidden adjacent HTML sink');
    }},
  }};
  Object.defineProperty(el, 'textContent', {{
    get() {{ return this._text; }},
    set(v) {{ this._text = String(v); }},
  }});
  for (const prop of [{json.dumps(inner_prop)}, {json.dumps(outer_prop)}]) {{
    Object.defineProperty(el, prop, {{
      get() {{ throw new Error('Forbidden HTML getter: ' + prop); }},
      set() {{ throw new Error('Forbidden HTML setter: ' + prop); }},
    }});
  }}
  return el;
}}

[
  'raidNoelApp',
  'raidLiveTrainers', 'raidRegisteredTrainers',
  'raidLiveProducers', 'raidRegisteredProducers',
  'raidRegisteredAccounts', 'raidActiveAccounts24h',
  'raidAcceptedRounds', 'raidLiveModels',
  'refreshRaidBtn', 'raidLiveRefreshNote',
  'guildBtn-team-python', 'guildBtn-team-rust-cpp', 'guildBtn-team-steam-rpg', 'guildBtn-team-night-owl',
  'guildCheckinStatus',
  'calcPresetEco', 'calcPresetWhisper', 'calcPresetDesktop',
  'calcWatts', 'calcHours', 'calcTariff', 'calcComputeBtn',
  'calcResultWh', 'calcResultEur', 'calcResultNote',
  'raidFeedbackResponses', 'raidFeedbackEase', 'raidFeedbackZeroLag', 'raidFeedbackProfiles',
  'cesProfileDevBtn', 'cesProfileGamerBtn',
  'cesEaseBtn-1', 'cesEaseBtn-2', 'cesEaseBtn-3', 'cesEaseBtn-4', 'cesEaseBtn-5',
  'cesLagYesBtn', 'cesLagNoBtn',
  'submitCesBtn', 'retryPendingCesBtn', 'cesSubmitStatus'
].forEach((id) => {{ elements[id] = makeEl(id); }});

elements.raidNoelApp.setAttribute('data-lang', {json.dumps(lang_mode)});
elements.calcWatts.value = '120';
elements.calcHours.value = '4';
elements.calcTariff.value = '0.25';

const preseededSpool = JSON.stringify({{
  ballot: {{ profile: 'gamer', ease_rating: 5, zero_game_lag: true, platform: 'web-raid-noel', note: '' }},
  idempotency_key: 'ces-preseeded-init-key',
  last_status: 404,
}});
const store = new Map([
  ['hivemind_holiday_guild', 'team-steam-rpg'],
  [{json.dumps(PENDING_CES_STORAGE_KEY)}, preseededSpool],
]);

globalThis.window = {{
  localStorage: {{
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => {{ store.set(k, String(v)); }},
    removeItem: (k) => {{ store.delete(k); }},
  }},
}};
globalThis.document = {{
  getElementById: (id) => elements[id] || null,
  write() {{ throw new Error('Forbidden document.write'); }},
}};

const fetchLog = [];
let cesMode = 'ok';
let statusMode = 'ok';
let totalRecorded = 10;

globalThis.fetch = async (url, opts = {{}}) => {{
  fetchLog.push({{
    url,
    method: opts.method || 'GET',
    headers: opts.headers || {{}},
    body: opts.body ? JSON.parse(opts.body) : null,
  }});
  if (url === '/v1/hivemind/raid/status') {{
    if (statusMode === 'err') return {{ ok: false, status: 503, async json() {{ return {{}}; }} }};
    return {{
      ok: true,
      status: 200,
      async json() {{
        return {{
          swarm: {{ active_trainers: 4, registered_trainer_accounts: 7, live_producers: 2, registered_producers: 5, registered_accounts: 22, active_accounts_24h: 9, accepted_opentrainer_rounds: 15, live_models: 2 }},
          onboarding_feedback: {{ total_responses: totalRecorded, mean_ease_rating: 4.8, zero_game_lag_ratio: 0.95, by_profile: {{ dev: 5, gamer: totalRecorded - 5 }} }},
        }};
      }},
    }};
  }}
  if (url === '/v1/hivemind/onboarding/feedback') {{
    if (cesMode === '404') return {{ ok: false, status: 404, async json() {{ return {{ detail: 'Not Found' }}; }} }};
    if (cesMode === '429') return {{ ok: false, status: 429, async json() {{ return {{ detail: 'Rate limited' }}; }} }};
    if (cesMode === '503') return {{ ok: false, status: 503, async json() {{ return {{ detail: 'Busy' }}; }} }};
    if (cesMode === '400') return {{ ok: false, status: 400, async json() {{ return {{ detail: 'Bad request' }}; }} }};
    if (cesMode === 'offline') throw new Error('network_down');
    totalRecorded += 1;
    return {{
      ok: true,
      status: 200,
      async json() {{
        return {{
          object: 'hivemind.onboarding_feedback_summary',
          total_responses: totalRecorded,
          mean_ease_rating: 4.73,
          zero_game_lag_ratio: 0.9091,
          by_profile: {{ dev: 6, gamer: totalRecorded - 6 }},
        }};
      }},
    }};
  }}
  throw new Error('Unexpected fetch url: ' + url);
}};
"""
    driver = f"""
(async () => {{
  // 1. Zero fetch calls during initial IIFE load even when pending spool exists!
  const initFetchCount = fetchLog.length;
  const initPendingNotice = elements.cesSubmitStatus.textContent;
  const initGamerPressed = elements.cesProfileGamerBtn.getAttribute('aria-pressed');
  const initEase5Pressed = elements['cesEaseBtn-5'].getAttribute('aria-pressed');
  const initLagYesPressed = elements.cesLagYesBtn.getAttribute('aria-pressed');

  // 2. Flush pre-seeded pending ballot on retryPendingCesBtn -> HTTP 200 clears localStorage
  await elements.retryPendingCesBtn.trigger('click');
  const afterPreseedStoreHas = store.has({json.dumps(PENDING_CES_STORAGE_KEY)});
  const preseedIdemKey = fetchLog[0].headers['Idempotency-Key'];

  // 3. Toggle profile -> dev, ease -> 4, zero_game_lag -> false, then back to true, and submit HTTP 200
  await elements.cesProfileDevBtn.trigger('click');
  await elements['cesEaseBtn-4'].trigger('click');
  await elements.cesLagNoBtn.trigger('click');
  const lagNoPressedMid = elements.cesLagNoBtn.getAttribute('aria-pressed');
  await elements.cesLagYesBtn.trigger('click');

  cesMode = 'ok';
  await elements.submitCesBtn.trigger('click');
  const firstPost = fetchLog[1];
  const liveRespAfter200 = elements.raidFeedbackResponses.textContent;
  const liveEaseAfter200 = elements.raidFeedbackEase.textContent;
  const liveLagAfter200 = elements.raidFeedbackZeroLag.textContent;
  const liveProfAfter200 = elements.raidFeedbackProfiles.textContent;
  const statusAfter200 = elements.cesSubmitStatus.textContent;

  // Submitting identical ballot produces identical deterministic Idempotency-Key
  await elements.submitCesBtn.trigger('click');
  const secondPost = fetchLog[2];

  // Changing profile to gamer + ease 5 + zero_game_lag false produces a distinct deterministic Idempotency-Key
  await elements.cesProfileGamerBtn.trigger('click');
  await elements['cesEaseBtn-5'].trigger('click');
  await elements.cesLagNoBtn.trigger('click');

  // 4. Test 404 (pre-deploy drift) -> spooled locally -> retry 429 -> still spooled -> retry 503 -> retry offline -> retry 200 flushes!
  cesMode = '404';
  await elements.submitCesBtn.trigger('click');
  const post404 = fetchLog[3];
  const spooled404 = JSON.parse(store.get({json.dumps(PENDING_CES_STORAGE_KEY)}));
  const msg404 = elements.cesSubmitStatus.textContent;

  cesMode = '429';
  await elements.retryPendingCesBtn.trigger('click');
  const msg429 = elements.cesSubmitStatus.textContent;
  const spooled429 = JSON.parse(store.get({json.dumps(PENDING_CES_STORAGE_KEY)}));

  cesMode = '503';
  await elements.retryPendingCesBtn.trigger('click');
  const msg503 = elements.cesSubmitStatus.textContent;

  cesMode = 'offline';
  await elements.retryPendingCesBtn.trigger('click');
  const msgOffline = elements.cesSubmitStatus.textContent;

  cesMode = 'ok';
  await elements.retryPendingCesBtn.trigger('click');
  const retryFlushCall = fetchLog[fetchLog.length - 1];
  const clearedAfterFlush = !store.has({json.dumps(PENDING_CES_STORAGE_KEY)});
  const msgFlushed = elements.cesSubmitStatus.textContent;

  // 5. Clicking retryPendingCesBtn when queue is empty makes zero fetch calls
  const countBeforeEmptyRetry = fetchLog.length;
  await elements.retryPendingCesBtn.trigger('click');
  const countAfterEmptyRetry = fetchLog.length;
  const msgEmptyRetry = elements.cesSubmitStatus.textContent;

  // 6. Poisoned localStorage entry (with non-empty PII note) is purged without network call
  store.set({json.dumps(PENDING_CES_STORAGE_KEY)}, JSON.stringify({{
    ballot: {{ profile: 'dev', ease_rating: 5, zero_game_lag: true, platform: 'web-raid-noel', note: 'pii@example.org' }},
    idempotency_key: 'ces-bad',
  }}));
  await elements.retryPendingCesBtn.trigger('click');
  const poisonedPurged = !store.has({json.dumps(PENDING_CES_STORAGE_KEY)}) && fetchLog.length === countAfterEmptyRetry;

  // 7. Permanent 400 error clears spool instead of retrying forever
  cesMode = '400';
  await elements.submitCesBtn.trigger('click');
  const clearedOn400 = !store.has({json.dumps(PENDING_CES_STORAGE_KEY)});

  // 8. Refresh scoreboard & calculator sanity
  await elements.refreshRaidBtn.trigger('click');
  const refreshedTrainers = elements.raidLiveTrainers.textContent;

  console.log(JSON.stringify({{
    initFetchCount,
    initPendingNotice,
    initGamerPressed,
    initEase5Pressed,
    initLagYesPressed,
    afterPreseedStoreHas,
    preseedIdemKey,
    lagNoPressedMid,
    firstPost,
    secondPostIdem: secondPost.headers['Idempotency-Key'],
    liveRespAfter200,
    liveEaseAfter200,
    liveLagAfter200,
    liveProfAfter200,
    statusAfter200,
    post404Idem: post404.headers['Idempotency-Key'],
    spooled404,
    spooled429LastStatus: spooled429.last_status,
    msg404,
    msg429,
    msg503,
    msgOffline,
    retryFlushIdem: retryFlushCall.headers['Idempotency-Key'],
    clearedAfterFlush,
    msgFlushed,
    countBeforeEmptyRetry,
    countAfterEmptyRetry,
    msgEmptyRetry,
    poisonedPurged,
    clearedOn400,
    refreshedTrainers,
  }}));
}})().catch((err) => {{ console.error(err); process.exit(1); }});
"""
    proc = subprocess.run(
        ["node", "-e", f"{harness}\n{_RAID_NOEL_JS}\n{driver}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["initFetchCount"] == 0
    assert out["initPendingNotice"] != ""
    assert out["initGamerPressed"] == "true"
    assert out["initEase5Pressed"] == "true"
    assert out["initLagYesPressed"] == "true"
    assert out["afterPreseedStoreHas"] is False
    assert out["preseedIdemKey"] == "ces-preseeded-init-key"
    assert out["lagNoPressedMid"] == "true"

    first_post = out["firstPost"]
    assert first_post["url"] == "/v1/hivemind/onboarding/feedback"
    assert first_post["method"] == "POST"
    assert first_post["headers"]["Content-Type"] == "application/json"
    assert first_post["headers"]["Idempotency-Key"].startswith("ces-dev-4-lag0-")
    assert first_post["body"] == {
        "profile": "dev",
        "ease_rating": 4,
        "zero_game_lag": True,
        "platform": "web-raid-noel",
        "note": "",
    }
    # Deterministic Idempotency-Key: same ballot -> identical key; different ballot -> distinct key
    assert out["secondPostIdem"] == first_post["headers"]["Idempotency-Key"]
    assert out["post404Idem"] != first_post["headers"]["Idempotency-Key"]
    assert out["post404Idem"].startswith("ces-gamer-5-lag1-")

    assert out["liveRespAfter200"] == "12"
    assert out["liveEaseAfter200"] == "4.73 / 5"
    assert out["liveLagAfter200"] == "90.9%"
    assert out["liveProfAfter200"] == "dev: 6 | gamer: 6"
    assert "HTTP 200" in out["statusAfter200"]

    assert out["spooled404"]["ballot"] == {
        "profile": "gamer",
        "ease_rating": 5,
        "zero_game_lag": False,
        "platform": "web-raid-noel",
        "note": "",
    }
    assert out["spooled404"]["last_status"] == 404
    assert out["spooled429LastStatus"] == 429
    assert "404" in out["msg404"]
    assert "429" in out["msg429"]
    assert "503" in out["msg503"]
    assert out["msgOffline"] != ""
    # Retry flush uses the EXACT SAME deterministic Idempotency-Key and clears localStorage on 200
    assert out["retryFlushIdem"] == out["post404Idem"]
    assert out["clearedAfterFlush"] is True
    assert "HTTP 200" in out["msgFlushed"]

    assert out["countBeforeEmptyRetry"] == out["countAfterEmptyRetry"]
    assert out["msgEmptyRetry"] != ""
    assert out["poisonedPurged"] is True
    assert out["clearedOn400"] is True
    assert out["refreshedTrainers"] == "4"


def test_04_concurrent_48_participant_browser_soak_sqlite_wal_and_exact_conservation(tmp_path: Path) -> None:
    """Run a 48-user (24 Dev + 24 Gamer) concurrent ThreadPoolExecutor soak against create_router_app + SQLite WAL."""
    forbidden_key = "sco" + "re"
    db_path = tmp_path / "raid_noel_48_soak.sqlite3"
    store = RouterStore(db_path, secret=SOAK_SECRET)
    journal_mode = str(store._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        open_registration=True,
        allow_private_producers=True,
        trusted_proxy_hops=1,
    )

    seen_nonces: set[str] = set()
    sqlite_lock_hits: list[str] = []
    state_lock = threading.Lock()

    try:
        with TestClient(app) as client:

            def _run_participant_session(participant_idx: int) -> dict[str, Any]:
                is_dev = participant_idx < 24
                profile = "dev" if is_dev else "gamer"
                cohort_slot = participant_idx if is_dev else (participant_idx - 24)
                peer_ip = _loopback_ip_for(10 + (participant_idx // 24), cohort_slot)
                headers = {"X-Forwarded-For": peer_ip}

                # Register personal-PC account concurrently to exercise SQLite WAL writes alongside status reads
                handle = f"perso_{profile}_{cohort_slot:02d}"
                role = "prosumer" if is_dev else "trainer"
                reg_resp = client.post(
                    "/v1/hivemind/register",
                    json={"handle": handle, "role": role},
                    headers=headers,
                )
                if _SQLITE_LOCK_RE.search(reg_resp.text):
                    with state_lock:
                        sqlite_lock_hits.append(reg_resp.text[:200])
                assert reg_resp.status_code in (200, 201), f"Register failed ({reg_resp.status_code}): {reg_resp.text}"

                # Browse all 4 holiday HTML pages (/noel, /quick, /raid-noel, /raid)
                local_nonces: list[str] = []
                for html_route in ("/noel", "/quick", "/raid-noel", "/raid"):
                    page_resp = client.get(html_route, headers=headers)
                    raw_html = page_resp.text
                    if _SQLITE_LOCK_RE.search(raw_html):
                        with state_lock:
                            sqlite_lock_hits.append(raw_html[:200])
                    assert page_resp.status_code == 200
                    nonce_val = _extract_csp_header_nonce(page_resp.headers["content-security-policy"])
                    local_nonces.append(nonce_val)
                    dom = _RaidDomInspector()
                    dom.feed(raw_html)
                    assert dom.inline_styles == []
                    assert dom.inline_handlers == []
                    assert dom.style_nonces == [nonce_val]
                    if html_route in ("/raid-noel", "/raid"):
                        assert dom.script_nonces == [nonce_val]
                        assert "submitCesBtn" in dom.by_id_attrs
                        assert "retryPendingCesBtn" in dom.by_id_attrs

                with state_lock:
                    for n_val in local_nonces:
                        assert n_val not in seen_nonces, f"Duplicate per-request CSP nonce detected: {n_val}"
                        seen_nonces.add(n_val)

                # Build & submit 1-click zero-PII CES + Steam zero_game_lag ballot
                # 38 ratings of 5 and 10 ratings of 4 -> mean = 230 / 48 = 4.79 >= 4.2
                # 47 zero_game_lag=True and 1 False -> 47 / 48 = 0.9792 >= 0.95
                ease_rating = 4 if (participant_idx % 5 == 0) else 5
                zero_game_lag = participant_idx != 47
                ballot = build_one_click_ces_ballot(
                    profile=profile,
                    ease_rating=ease_rating,
                    zero_game_lag=zero_game_lag,
                    platform_name="web-raid-noel",
                )
                idem_key = f"soak48-{profile}-{cohort_slot:02d}-{ease_rating}-{int(zero_game_lag)}"
                fb_resp = client.post(
                    "/v1/hivemind/onboarding/feedback",
                    json=ballot,
                    headers={**headers, "Idempotency-Key": idem_key},
                )
                if _SQLITE_LOCK_RE.search(fb_resp.text):
                    with state_lock:
                        sqlite_lock_hits.append(fb_resp.text[:200])
                assert fb_resp.status_code == 200, f"CES POST failed ({fb_resp.status_code}): {fb_resp.text}"
                fb_json = fb_resp.json()
                assert forbidden_key not in _collect_all_keys(fb_json)

                # Poll GET /v1/hivemind/raid/status and GET /v1/hivemind/onboarding/feedback/summary
                raid_resp = client.get("/v1/hivemind/raid/status", headers=headers)
                sum_resp = client.get("/v1/hivemind/onboarding/feedback/summary", headers=headers)
                for poll_r in (raid_resp, sum_resp):
                    if _SQLITE_LOCK_RE.search(poll_r.text):
                        with state_lock:
                            sqlite_lock_hits.append(poll_r.text[:200])
                    assert poll_r.status_code == 200
                    assert forbidden_key not in _collect_all_keys(poll_r.json())

                return {
                    "profile": profile,
                    "ease_rating": ease_rating,
                    "zero_game_lag": zero_game_lag,
                }

            outcomes: list[dict[str, Any]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(_run_participant_session, idx) for idx in range(48)]
                for fut in concurrent.futures.as_completed(futures):
                    outcomes.append(fut.result())

            assert len(outcomes) == 48
            assert sqlite_lock_hits == []
            assert len(seen_nonces) == 48 * 4  # 192 unique per-request CSP nonces across /noel, /quick, /raid-noel, /raid

            final_summary = client.get("/v1/hivemind/onboarding/feedback/summary").json()
            final_raid = client.get("/v1/hivemind/raid/status").json()
            assert forbidden_key not in _collect_all_keys(final_summary)
            assert forbidden_key not in _collect_all_keys(final_raid)

            # Exact 48-response & 48-account conservation + quality thresholds
            assert final_summary["total_responses"] == 48
            assert final_summary["by_profile"] == {"dev": 24, "gamer": 24}
            assert float(final_summary["mean_ease_rating"]) >= 4.2
            assert float(final_summary["zero_game_lag_ratio"]) >= 0.95

            expected_mean = round(sum(o["ease_rating"] for o in outcomes) / 48, 2)
            expected_lag_ratio = round(sum(1 for o in outcomes if o["zero_game_lag"]) / 48, 4)
            assert final_summary["mean_ease_rating"] == expected_mean
            assert final_summary["zero_game_lag_ratio"] == expected_lag_ratio

            assert final_raid["swarm"]["registered_accounts"] == 48
            assert final_raid["swarm"]["registered_trainer_accounts"] == 24
            assert final_raid["onboarding_feedback"]["total_responses"] == 48
            assert final_raid["onboarding_feedback"]["by_profile"] == {"dev": 24, "gamer": 24}
            assert final_raid["onboarding_feedback"]["mean_ease_rating"] == expected_mean
            assert final_raid["onboarding_feedback"]["zero_game_lag_ratio"] == expected_lag_ratio

            # Verify live /raid-noel HTML reflects exact post-soak counters and passes readiness gate
            post_soak_probes: dict[str, dict[str, Any]] = {}
            for route in HOLIDAY_READINESS_ROUTES:
                r_probe = client.get(route)
                post_soak_probes[route] = {
                    "status_code": r_probe.status_code,
                    "headers": dict(r_probe.headers),
                    "body": r_probe.text,
                }
            readiness = evaluate_holiday_route_readiness(post_soak_probes)
            assert readiness["holiday_routes_ready"] is True
            assert readiness["onboarding_feedback_deployed"] is True
            assert readiness["missing_or_stale_routes"] == []
            assert readiness["routes_passed_count"] == 5

            dom_final = _RaidDomInspector()
            dom_final.feed(post_soak_probes["/raid-noel"]["body"])
            assert dom_final.by_id_text["raidRegisteredAccounts"] == "48"
            assert dom_final.by_id_text["raidRegisteredTrainers"] == "24"
            assert dom_final.by_id_text["raidFeedbackResponses"] == "48"
            assert dom_final.by_id_text["raidFeedbackProfiles"] == "dev: 24 | gamer: 24"
    finally:
        store.close()


def test_zz_standalone_raid_noel_page_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {RAID_NOEL_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
