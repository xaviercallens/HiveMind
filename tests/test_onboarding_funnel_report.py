# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Comprehensive unit, privacy, gateway-integration, and security tests for SLICE-06
(`scripts/onboarding_funnel_report.py`).
"""

from __future__ import annotations

import ast
import http.server
import importlib.util
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "onboarding_funnel_report.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import onboarding_funnel_report as funnel_mod  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_mod",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gate_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_mod",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gate_spec is not None and _gate_spec.loader is not None
gatekeeper_mod = importlib.util.module_from_spec(_gate_spec)
_gate_spec.loader.exec_module(gatekeeper_mod)


def _collect_all_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys.update(_collect_all_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_all_keys(item))
    return keys


def test_compliance_headers_stdlib_lateral_movement_and_gatekeeper() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    for target in (SCRIPT_PATH, Path(__file__).resolve()):
        head = target.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in {target}"

    script_source = SCRIPT_PATH.read_text(encoding="utf-8")
    test_source = Path(__file__).resolve().read_text(encoding="utf-8")

    # 1. Zero lateral movement / unauthorized egress on scripts/onboarding_funnel_report.py
    violations = lateral_guard.audit_source_for_lateral_movement(
        script_source,
        filename="scripts/onboarding_funnel_report.py",
    )
    assert violations == []

    # 2. Only Python standard library imports in scripts/onboarding_funnel_report.py
    tree = ast.parse(script_source, filename="scripts/onboarding_funnel_report.py")
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    non_stdlib = imported_roots - sys.stdlib_module_names
    assert non_stdlib == set(), f"Non-stdlib imports found: {non_stdlib}"

    # 3. 5-Layer PRSecurityScanner passes cleanly
    scanner = PRSecurityScanner()
    for rel, text in (
        ("scripts/onboarding_funnel_report.py", script_source),
        ("tests/test_onboarding_funnel_report.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, text) == []
        assert scanner.scan_text_for_payment_solicitation(text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text, rel) == []
        assert scanner.scan_unicode_trojan_source(text, rel) == []

    # 4. PR gatekeeper FORBIDDEN_ADDED check passes cleanly on added lines
    simulated_diff = "\n".join("+" + line for line in (script_source + "\n" + test_source).splitlines())
    assert gatekeeper_mod.forbidden_scan(simulated_diff) == []

    # 5. GWAYA zero-stub audit passes cleanly
    from gwaya.ast_audit import ZeroStubAudit

    audit_res = ZeroStubAudit().audit_python_code(script_source)
    assert audit_res.is_clean, f"ZeroStubAudit violations: {audit_res.violations}"


def test_healthy_snapshot_metrics_boundaries_and_receipt_determinism() -> None:
    healthy_snapshot = {
        "landing_visits": 100,
        "signups": 40,
        "first_requests": 30,
        "preflight_runs": 20,
        "install_completed": 18,
        "first_rounds": 16,
        "onboarding_feedback": {
            "total_responses": 10,
            "mean_ease_rating": 4.6,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 6, "gamer": 4},
        },
        "arena_summary": {"total_votes": 14},
    }
    r1 = funnel_mod.analyze_onboarding_funnel(
        healthy_snapshot,
        generated_at_utc="2026-12-24T12:00:00Z",
    )
    r2 = funnel_mod.analyze_onboarding_funnel(
        healthy_snapshot,
        generated_at_utc="2026-12-24T12:00:00Z",
    )
    assert r1["receipt_sha256"] == r2["receipt_sha256"]
    assert len(r1["receipt_sha256"]) == 64

    assert r1["status"] == "healthy"
    assert r1["activation_ratio"] == 0.75
    assert r1["visitor_to_signup_ratio"] == 0.4
    assert r1["signup_conversion_ratio"] == 0.4
    assert r1["preflight_to_first_round_ratio"] == 0.8
    assert r1["install_dropoff_ratio"] == 0.2
    assert r1["mean_ease_rating"] == 4.6
    assert r1["zero_game_lag_ratio"] == 1.0
    assert r1["alerts"] == []
    assert r1["alert_flags"] == {
        "low_signup_conversion": False,
        "high_install_dropoff": False,
        "low_ease_rating": False,
        "game_lag_detected": False,
    }
    assert r1["target_checks"] == {
        "signup_conversion_ok": True,
        "install_dropoff_ok": True,
        "ease_rating_ok": True,
        "zero_game_lag_ok": True,
    }
    assert r1["compliance"]["personal_pc_only"] is True
    assert r1["compliance"]["monetary_cost_eur"] == "0.00 EUR"
    assert r1["compliance"]["monetary_cost_usd"] == "0.00 USD"

    forbidden_word = "sco" + "re"
    assert forbidden_word not in _collect_all_keys(r1)
    json_out = funnel_mod.serialize_receipt_json(r1)
    assert f'"{forbidden_word}"' not in json_out

    # Exact rational threshold boundary check (20% conversion, 30% dropoff = 7/10, 4.2 ease, 100% lag-free)
    boundary_snapshot = {
        "funnel": {
            "signups": 100,
            "first_requests": 20,
            "preflight_runs": 10,
            "first_rounds": 7,
        },
        "mean_ease_rating": 4.2,
        "zero_game_lag_ratio": 1.0,
        "total_responses": 5,
    }
    rb = funnel_mod.analyze_onboarding_funnel(boundary_snapshot, generated_at_utc="2026-12-24T12:00:00Z")
    assert rb["status"] == "healthy"
    assert rb["activation_ratio"] == 0.2
    assert rb["preflight_to_first_round_ratio"] == 0.7
    assert rb["install_dropoff_ratio"] == 0.3
    assert rb["mean_ease_rating"] == 4.2
    assert rb["alerts"] == []


def test_feedback_items_aggregation_and_privacy_redaction() -> None:
    fake_ip = ".".join(("198", "51", "100", "24"))
    fake_email = "alice.secret" + "@" + "example.org"
    fake_key = "hm-" + "secretkey987654"
    raw = {
        "counts": {
            "signups": 25,
            "activated_accounts": 18,
            "preflight_passed": 10,
            "completed_first_round": 9,
        },
        "newest_registered": [{"handle": "alice-home-pc", "email": fake_email}],
        "live_handles": ["bob-gamer-rig"],
        "client_ip": fake_ip,
        "api_key": fake_key,
        "feedback_items": [
            {
                "profile": "dev",
                "ease_rating": 5,
                "zero_game_lag": True,
                "handle": "alice-home-pc",
                "comment": f"Great setup from {fake_email}",
            },
            {
                "profile": "gamer",
                "ease_rating": 4,
                "zero_game_lag": True,
                "ip": fake_ip,
            },
            {
                "profile": "GAMER",
                "ease_rating": 5,
                "zero_game_lag": True,
            },
        ],
    }
    receipt = funnel_mod.analyze_onboarding_funnel(raw, generated_at_utc="2026-12-25T08:00:00Z")
    assert receipt["status"] == "healthy"
    assert receipt["counts"]["feedback_responses"] == 3
    assert receipt["counts"]["feedback_by_profile"] == {"dev": 1, "gamer": 2}
    assert receipt["mean_ease_rating"] == 4.67
    assert receipt["zero_game_lag_ratio"] == 1.0
    assert receipt["privacy"]["zero_pii"] is True
    assert receipt["privacy"]["redacted_fields_dropped"] >= 7

    serialized_json = funnel_mod.serialize_receipt_json(receipt)
    rendered_md = funnel_mod.render_markdown_report(receipt, lang="bilingual")
    for sensitive in ("alice-home-pc", "bob-gamer-rig", fake_email, fake_ip, fake_key):
        assert sensitive not in serialized_json
        assert sensitive not in rendered_md


def test_ux_alert_flags_individual_and_combined() -> None:
    # 1. All 4 UX alerts active simultaneously
    degraded = {
        "signups": 100,
        "first_requests": 19,  # 19% < 20%
        "preflight_runs": 100,
        "first_rounds": 69,  # 31% dropoff > 30%
        "onboarding_feedback": {
            "total_responses": 20,
            "mean_ease_rating": 4.15,  # < 4.2
            "zero_game_lag_ratio": 0.9,  # < 1.0
            "by_profile": {"dev": 10, "gamer": 10},
        },
    }
    rec = funnel_mod.analyze_onboarding_funnel(degraded, generated_at_utc="2026-12-26T00:00:00Z")
    assert rec["status"] == "alert"
    assert rec["alert_flags"] == {
        "low_signup_conversion": True,
        "high_install_dropoff": True,
        "low_ease_rating": True,
        "game_lag_detected": True,
    }
    codes = [a["code"] for a in rec["alerts"]]
    assert codes == [
        "low_signup_conversion",
        "high_install_dropoff",
        "low_ease_rating",
        "game_lag_detected",
    ]

    md_fr = funnel_mod.render_markdown_report(rec, lang="fr")
    assert "ALERTE UX" in md_fr
    assert "low_signup_conversion" in md_fr
    assert "high_install_dropoff" in md_fr
    assert "low_ease_rating" in md_fr
    assert "game_lag_detected" in md_fr
    assert "## Onboarding Usability Funnel" not in md_fr

    md_en = funnel_mod.render_markdown_report(rec, lang="en")
    assert "UX ALERT" in md_en
    assert "below the 20.0% minimum UX threshold" in md_en
    assert "above the 30.0% maximum UX dropoff threshold" in md_en
    assert "below the >= 4.20 / 5 target" in md_en
    assert "below the 100.0% Gamer-Whisper target" in md_en
    assert "## Bilan Entonnoir" not in md_en

    # 2. Visitor-to-signup < 20% alert even when signup->first_request is high, plus install_completed-only dropoff
    vis_alert = funnel_mod.analyze_onboarding_funnel(
        {
            "landing_visits": 200,
            "signups": 30,  # 15% < 20%
            "first_requests": 27,  # 90%
            "preflight_runs": 20,
            "install_completed": 12,  # 40% install dropoff without first_rounds key
        },
        generated_at_utc="2026-12-26T01:00:00Z",
    )
    assert vis_alert["status"] == "alert"
    assert vis_alert["alert_flags"]["low_signup_conversion"] is True
    assert vis_alert["alerts"][0]["metric"] == "signup_conversion_ratio"
    assert vis_alert["alert_flags"]["high_install_dropoff"] is True
    assert vis_alert["install_dropoff_ratio"] == 0.4


def test_empty_and_zero_observation_inputs() -> None:
    empty_rec = funnel_mod.analyze_onboarding_funnel({}, generated_at_utc="2026-12-20T00:00:00Z")
    assert empty_rec["status"] == "insufficient_data"
    assert empty_rec["activation_ratio"] is None
    assert empty_rec["preflight_to_first_round_ratio"] is None
    assert empty_rec["install_dropoff_ratio"] is None
    assert empty_rec["mean_ease_rating"] is None
    assert empty_rec["zero_game_lag_ratio"] is None
    assert empty_rec["alerts"] == []
    assert all(v is None for v in empty_rec["target_checks"].values())

    zero_rec = funnel_mod.analyze_onboarding_funnel(
        {
            "signups": 0,
            "first_requests": 0,
            "preflight_runs": 0,
            "first_rounds": 0,
            "feedback_items": [],
        },
        generated_at_utc="2026-12-20T00:00:00Z",
    )
    assert zero_rec["status"] == "insufficient_data"
    assert zero_rec["alerts"] == []
    assert zero_rec["counts"]["feedback_responses"] == 0

    bilingual_md = funnel_mod.render_markdown_report(zero_rec, lang="both")
    assert "DONNÉES INSUFFISANTES" in bilingual_md
    assert "INSUFFICIENT DATA" in bilingual_md
    assert "`n/a`" in bilingual_md


@pytest.mark.parametrize(
    "bad_payload",
    [
        [],
        "not-a-dict",
        {"funnel": "bad"},
        {"signups": True, "first_requests": 0},
        {"signups": -1, "first_requests": 0},
        {"signups": 3.5, "first_requests": 1},
        {"signups": float("nan")},
        {"landing_visits": 10, "signups": 12},
        {"signups": 5, "first_requests": 6},
        {"first_requests": 2},
        {"preflight_runs": 5, "install_completed": 6},
        {"preflight_runs": 5, "first_rounds": 6},
        {"preflight_runs": 10, "install_completed": 6, "first_rounds": 7},
        {"first_rounds": 3},
        {"signups": 2, "swarm": {"registered_accounts": 10, "active_accounts_24h": 5}},
        {"preflight_runs": 1, "swarm": {"registered_trainer_nodes": 4, "active_trainers": 3, "accepted_opentrainer_rounds": 5}},
        {"onboarding_feedback": "not-an-object"},
        {"raid_status": {"onboarding_feedback": "bad"}},
        {"raid_status": {"swarm": "bad"}},
        {"swarm": "bad"},
        {"object": "hivemind.live_telemetry", "accounts": "bad"},
        {"object": "hivemind.live_telemetry", "accounts": {}, "opentrainer": "bad"},
        {"arena_summary": "bad"},
        {"mean_ease_rating": True},
        {"mean_ease_rating": "4.5"},
        {"mean_ease_rating": float("inf")},
        {"mean_ease_rating": 0.5},
        {"mean_ease_rating": 5.2},
        {"zero_game_lag_ratio": -0.05},
        {"zero_game_lag_ratio": 1.05},
        {"total_responses": 0, "mean_ease_rating": 4.5},
        {"by_profile": "bad"},
        {"by_profile": {"admin": 1}},
        {"total_responses": 1, "by_profile": {"dev": 1, "gamer": 1}},
        {"feedback_items": "not-a-list"},
        {"feedback_items": ["not-a-dict"]},
        {"feedback_items": [{"zero_game_lag": True}]},
        {"feedback_items": [{"ease_rating": 5, "zero_game_lag": "yes"}]},
        {"feedback_items": [{"ease_rating": 5, "profile": 123}]},
        {"feedback_items": [{"ease_rating": 5, "profile": "enterprise"}]},
    ],
)
def test_malformed_snapshots_rejected_cleanly(bad_payload: Any) -> None:
    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod.analyze_onboarding_funnel(bad_payload)


def test_feedback_override_and_by_profile_sum_inference() -> None:
    rec = funnel_mod.analyze_onboarding_funnel(
        {"signups": 10, "first_requests": 8},
        feedback_override={
            "mean_ease_rating": 4.4,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 3, "gamer": 2},
        },
        generated_at_utc="2026-12-27T00:00:00Z",
    )
    assert rec["counts"]["feedback_responses"] == 5
    assert rec["mean_ease_rating"] == 4.4
    assert rec["status"] == "healthy"

    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod.analyze_onboarding_funnel({"signups": 5}, feedback_override="bad")  # type: ignore[arg-type]


def test_pii_leak_scanner_positive_and_negative_controls() -> None:
    assert funnel_mod.scan_for_pii_leaks("Clean report: 80.0% activation, 0.00 EUR.") == []
    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod.scan_for_pii_leaks(123)  # type: ignore[arg-type]

    planted_email = "user" + "@" + "domain.fr"
    planted_key = "hm_" + "live_secret_token_99"
    planted_bearer = "Bearer " + "eyJhbGciOiJIUzI1NiJ9"
    planted_ip = ".".join(("10", "20", "30", "40"))
    planted_home = "/" + "home/xavsecret/config"
    planted_forbidden = '{"' + "sco" + 're": 5}'

    assert "email_address_detected" in funnel_mod.scan_for_pii_leaks(planted_email)
    assert "hivemind_api_key_detected" in funnel_mod.scan_for_pii_leaks(planted_key)
    assert "bearer_token_detected" in funnel_mod.scan_for_pii_leaks(planted_bearer)
    assert "ipv4_literal_detected" in funnel_mod.scan_for_pii_leaks(planted_ip)
    assert "local_home_path_detected" in funnel_mod.scan_for_pii_leaks(planted_home)
    assert "forbidden_key_detected" in funnel_mod.scan_for_pii_leaks(planted_forbidden)

    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod._assert_clean_output(planted_email)


def test_gateway_url_allowlist_and_validation() -> None:
    assert funnel_mod.is_allowed_gateway_host("") is False
    assert funnel_mod.is_allowed_gateway_host("localhost") is True
    assert funnel_mod.is_allowed_gateway_host("storage.googleapis.com") is True
    assert funnel_mod.is_allowed_gateway_host("run.googleapis.com") is True
    assert funnel_mod.is_allowed_gateway_host("datasets-server.huggingface.co") is True
    assert (
        funnel_mod.validate_gateway_url("https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/")
        == "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
    )
    assert funnel_mod.validate_gateway_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"

    bad_urls = [
        "",
        True,
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/ has space",
        "https://user:pass@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app?foo=1",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app#frag",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1/models",
        "http://127.0.0.1:abc",
        "http://127.0.0.1:0",
        "https://evil.example.com",
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "ftp://127.0.0.1",
    ]
    for bad in bad_urls:
        with pytest.raises(funnel_mod.FunnelInputError):
            funnel_mod.validate_gateway_url(bad)  # type: ignore[arg-type]


def test_live_router_fastapi_integration_and_telemetry_variants() -> None:
    store = RouterStore(":memory:", b"test-secret-0123456789abcdef0123456789abcdef")
    app = create_router_app(store)
    client = TestClient(app)

    # Register 2 accounts, post 2 feedback items, and cast 1 blind A/B vote
    r_reg1 = client.post("/v1/hivemind/register", json={"handle": "noel-dev-one", "role": "prosumer"})
    assert r_reg1.status_code == 200
    r_reg2 = client.post("/v1/hivemind/register", json={"handle": "noel-gamer-two", "role": "trainer"})
    assert r_reg2.status_code == 200

    assert (
        client.post(
            "/v1/hivemind/onboarding/feedback",
            json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/hivemind/onboarding/feedback",
            json={"profile": "gamer", "ease_rating": 4, "zero_game_lag": True, "platform": "windows"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/hivemind/arena/vote",
            json={"task_id": "python-sliding-median", "chosen_side": "A", "profile": "dev"},
        ).status_code
        == 200
    )

    def _bridge_http(url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
        assert 0.5 <= timeout_s <= 15.0
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        resp = client.get(path)
        return resp.status_code, resp.json()

    bundle = funnel_mod.fetch_live_gateway_snapshot(
        funnel_mod.DEFAULT_GATEWAY,
        http_get=_bridge_http,
        timeout_s=4.0,
    )
    receipt = funnel_mod.analyze_onboarding_funnel(bundle, generated_at_utc="2026-12-25T18:00:00Z")
    assert receipt["source_kind"] == "live_gateway"
    assert receipt["counts"]["signups"] == 2
    assert receipt["counts"]["feedback_responses"] == 2
    assert receipt["mean_ease_rating"] == 4.5
    assert receipt["zero_game_lag_ratio"] == 1.0
    assert receipt["counts"]["arena_total_votes"] == 1

    # Ensure zero handles leaked from live_telemetry.accounts.newest_registered
    out_json = funnel_mod.serialize_receipt_json(receipt)
    assert "noel-dev-one" not in out_json
    assert "noel-gamer-two" not in out_json

    # Standalone live_telemetry payload with multi-round opentrainer activity (>0 and ==0)
    lt_active = funnel_mod.analyze_onboarding_funnel(
        {
            "object": "hivemind.live_telemetry",
            "accounts": {"total_registered_accounts": 10, "active_accounts_24h": 7},
            "opentrainer": {
                "registered_trainer_accounts_count": 5,
                "total_registered_nodes": 4,
                "active_trainer_nodes": 4,
                "accepted_rounds_count": 42,
            },
        }
    )
    assert lt_active["activation_ratio"] == 0.7
    assert lt_active["preflight_to_first_round_ratio"] == 0.8
    assert lt_active["install_dropoff_ratio"] == 0.2

    lt_zero_rounds = funnel_mod.analyze_onboarding_funnel(
        {
            "object": "hivemind.live_telemetry",
            "accounts": {"total_registered_accounts": 10, "active_accounts_24h": 5},
            "opentrainer": {
                "registered_trainer_accounts_count": 4,
                "total_registered_nodes": 4,
                "active_trainer_nodes": 0,
                "accepted_rounds_count": 0,
            },
        }
    )
    assert lt_zero_rounds["preflight_to_first_round_ratio"] == 0.0
    assert lt_zero_rounds["alert_flags"]["high_install_dropoff"] is True

    # Standalone raid_status payload with cumulative rounds > 0
    raid_only = funnel_mod.analyze_onboarding_funnel(
        {
            "object": "hivemind.raid_noel.status",
            "swarm": {
                "registered_accounts": 8,
                "active_accounts_24h": 6,
                "registered_trainer_accounts": 4,
                "registered_trainer_nodes": 4,
                "active_trainers": 4,
                "accepted_opentrainer_rounds": 15,
            },
            "onboarding_feedback": {
                "total_responses": 4,
                "mean_ease_rating": 4.75,
                "zero_game_lag_ratio": 1.0,
                "by_profile": {"dev": 2, "gamer": 2},
            },
        }
    )
    assert raid_only["status"] == "healthy"
    assert raid_only["preflight_to_first_round_ratio"] == 1.0


def test_default_http_get_json_and_live_fetch_errors() -> None:
    mode = {"behavior": "ok"}

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            beh = mode["behavior"]
            if beh == "ok":
                body = json.dumps(
                    {
                        "object": "hivemind.onboarding_feedback_summary",
                        "total_responses": 2,
                        "mean_ease_rating": 4.5,
                        "zero_game_lag_ratio": 1.0,
                        "by_profile": {"dev": 1, "gamer": 1},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif beh == "err_non_json":
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"gateway warming up")
            elif beh == "not_found_json":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error":"not_found"}')
            elif beh == "bad_json_200":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{not-json")
            elif beh == "array_200":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"[1, 2, 3]")

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        code, doc = funnel_mod.default_http_get_json(
            f"{base_url}/v1/hivemind/onboarding/feedback/summary",
            timeout_s=2.0,
        )
        assert code == 200
        assert doc["mean_ease_rating"] == 4.5

        mode["behavior"] = "not_found_json"
        with pytest.raises(funnel_mod.FunnelInputError):
            funnel_mod.fetch_live_gateway_snapshot(base_url, timeout_s=2.0)

        mode["behavior"] = "err_non_json"
        with pytest.raises(funnel_mod.FunnelInputError):
            funnel_mod.fetch_live_gateway_snapshot(base_url, timeout_s=2.0)

        mode["behavior"] = "bad_json_200"
        with pytest.raises(funnel_mod.FunnelInputError):
            funnel_mod.default_http_get_json(f"{base_url}/v1/hivemind/raid/status", timeout_s=2.0)

        mode["behavior"] = "array_200"
        with pytest.raises(funnel_mod.FunnelInputError):
            funnel_mod.default_http_get_json(f"{base_url}/v1/hivemind/raid/status", timeout_s=2.0)
    finally:
        server.shutdown()
        server.server_close()

    # Closed loopback port -> URLError converted cleanly to FunnelInputError
    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod.default_http_get_json(f"http://127.0.0.1:{port}/v1/hivemind/raid/status", timeout_s=0.5)

    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod.fetch_live_gateway_snapshot("http://127.0.0.1:8080", timeout_s=True)  # type: ignore[arg-type]


def test_cli_main_modes_files_stdin_and_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snap_path = tmp_path / "snap.json"
    fb_path = tmp_path / "fb.json"
    out_json = tmp_path / "out" / "receipt.json"
    out_md = tmp_path / "out" / "report.md"

    snap_path.write_text(
        json.dumps(
            {
                "signups": 50,
                "first_requests": 35,
                "preflight_runs": 20,
                "first_rounds": 16,
            }
        ),
        encoding="utf-8",
    )
    fb_path.write_text(
        json.dumps(
            {
                "total_responses": 8,
                "mean_ease_rating": 4.5,
                "zero_game_lag_ratio": 1.0,
                "by_profile": {"dev": 5, "gamer": 3},
            }
        ),
        encoding="utf-8",
    )

    rc = funnel_mod.main(
        [
            "--snapshot",
            str(snap_path),
            "--feedback-snapshot",
            str(fb_path),
            "--json-out",
            str(out_json),
            "--markdown-out",
            str(out_md),
            "--generated-at",
            "2026-12-28T10:00:00Z",
            "--fail-on-alert",
        ]
    )
    assert rc == 0
    stdout_md = capsys.readouterr().out
    assert "SAIN" in stdout_md and "HEALTHY" in stdout_md
    assert out_json.is_file() and out_md.is_file()
    saved_receipt = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved_receipt["activation_ratio"] == 0.7
    assert saved_receipt["mean_ease_rating"] == 4.5

    # Alert snapshot via stdin ('-') with --json and --fail-on-alert -> returns exit code 1
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"signups": 20, "first_requests": 2, "mean_ease_rating": 3.5})),
    )
    rc_alert = funnel_mod.main(["--snapshot", "-", "--json", "--fail-on-alert"])
    assert rc_alert == 1
    alert_doc = json.loads(capsys.readouterr().out)
    assert alert_doc["status"] == "alert"

    # Without --fail-on-alert, report generation still exits 0
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"signups": 20, "first_requests": 2})),
    )
    assert funnel_mod.main(["--snapshot", "-", "--lang", "en"]) == 0
    capsys.readouterr()

    # Live CLI flag with mock http_get
    def _mock_live(url: str, _timeout: float) -> tuple[int, dict[str, Any]]:
        if url.endswith("/onboarding/feedback/summary"):
            return 200, {
                "total_responses": 4,
                "mean_ease_rating": 4.8,
                "zero_game_lag_ratio": 1.0,
                "by_profile": {"dev": 2, "gamer": 2},
            }
        return 404, {}

    assert funnel_mod.main(["--live", "--format", "json"], http_get=_mock_live) == 0
    live_doc = json.loads(capsys.readouterr().out)
    assert live_doc["source_kind"] == "live_gateway"
    assert live_doc["mean_ease_rating"] == 4.8

    # CLI validation error cases -> exit code 2 (and PII in error message scrubbed)
    assert funnel_mod.main([]) == 2
    assert funnel_mod.main(["--snapshot", str(snap_path), "--live"]) == 2
    assert funnel_mod.main(["--snapshot", str(tmp_path / "missing.json")]) == 2
    assert funnel_mod.main(["--snapshot", str(snap_path), "--lang", "de"]) == 2

    empty_file = tmp_path / "empty.json"
    empty_file.write_text("   ", encoding="utf-8")
    assert funnel_mod.main(["--snapshot", str(empty_file)]) == 2

    bad_syntax = tmp_path / "bad.json"
    bad_syntax.write_text("{broken", encoding="utf-8")
    assert funnel_mod.main(["--snapshot", str(bad_syntax)]) == 2

    arr_file = tmp_path / "arr.json"
    arr_file.write_text("[1, 2]", encoding="utf-8")
    assert funnel_mod.main(["--snapshot", str(arr_file)]) == 2

    leaked_lang = "bad-" + "user@secret.org"
    assert funnel_mod.main(["--snapshot", str(snap_path), "--lang", leaked_lang]) == 2
    err_out = capsys.readouterr().err
    assert "user@secret.org" not in err_out

    # Direct subprocess CLI smoke check
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--snapshot", str(snap_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    sub_doc = json.loads(proc.stdout)
    assert sub_doc["status"] == "healthy"


def test_helper_edge_cases_for_lang_and_markdown() -> None:
    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod.render_markdown_report("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(funnel_mod.FunnelInputError):
        funnel_mod.load_json_file("   ")
    assert funnel_mod._fmt_ratio_cell(True, fr=False) == "n/a"
    assert funnel_mod._fmt_ease_cell(True, fr=True) == "n/a"
    assert funnel_mod._fmt_int_cell(True) == "n/a"
