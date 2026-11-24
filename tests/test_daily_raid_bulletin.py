# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Comprehensive unit, honesty, security, gateway-integration, and coverage tests for SLICE-07:
  - `docs/FICHE_TRANSPARENCE_NOEL_FR_EN.md`
  - `scripts/daily_raid_bulletin.py`
"""

from __future__ import annotations

import ast
import http.server
import importlib.util
import io
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hivemind.router import t4_visibility as t4_mod
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "FICHE_TRANSPARENCE_NOEL_FR_EN.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "daily_raid_bulletin.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import daily_raid_bulletin as bulletin_mod  # noqa: E402

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


def _sample_measured_t4() -> dict[str, Any]:
    return {
        "status": "ok",
        "telemetry_status": "FRESH",
        "telemetry_source": "LIVE_UPDATE",
        "operational_mode": "EUROPEAN_NIGHT_QLORA_MOE_TRAINING",
        "active_producers_count": 9,
        "distinct_physical_gpus": 7,
        "hardware": {
            "primary_gpu": "NVIDIA Tesla T4 (16 GB GDDR6)",
            "vram_used_mib": 4841,
            "vram_total_mib": 15360,
            "vram_free_mib": 10519,
            "temp_c": 58.5,
            "power_w": 30.2,
        },
        "european_schedule": {
            "current_utc_iso": "2026-12-21T21:30:00+00:00",
        },
        "qlora_progress": {
            "qlora_rounds_assimilated": 145,
            "ratchet_accepted_rounds": 20,
            "held_out_eval_loss_before": 2.2705,
            "held_out_eval_loss_after": 2.1380,
        },
    }


def _sample_measured_feedback() -> dict[str, Any]:
    return {
        "object": "hivemind.onboarding_feedback_summary",
        "total_responses": 12,
        "mean_ease_rating": 4.5,
        "zero_game_lag_ratio": 1.0,
        "by_profile": {"dev": 5, "gamer": 7},
    }


def test_compliance_headers_stdlib_lateral_movement_and_security_gates() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    for py_target in (SCRIPT_PATH, Path(__file__).resolve()):
        head = py_target.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in {py_target}"

    doc_text = DOC_PATH.read_text(encoding="utf-8")
    script_source = SCRIPT_PATH.read_text(encoding="utf-8")
    test_source = Path(__file__).resolve().read_text(encoding="utf-8")

    # 1. Zero lateral movement / unauthorized egress on scripts/daily_raid_bulletin.py
    violations = lateral_guard.audit_source_for_lateral_movement(
        script_source,
        filename="scripts/daily_raid_bulletin.py",
    )
    assert violations == []

    # 2. Strictly Python standard library imports in scripts/daily_raid_bulletin.py
    tree = ast.parse(script_source, filename="scripts/daily_raid_bulletin.py")
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 3. 5-Layer PRSecurityScanner passes on doc, script, and test
    scanner = PRSecurityScanner()
    for rel, content in (
        ("docs/FICHE_TRANSPARENCE_NOEL_FR_EN.md", doc_text),
        ("scripts/daily_raid_bulletin.py", script_source),
        ("tests/test_daily_raid_bulletin.py", test_source),
    ):
        if rel.endswith(".py"):
            assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []

    # 4. PR gatekeeper FORBIDDEN_ADDED check passes across all added lines
    combined_added = "\n".join(
        "+" + ln for ln in (doc_text + "\n" + script_source + "\n" + test_source).splitlines()
    )
    assert gatekeeper_mod.forbidden_scan(combined_added) == []

    # 5. GWAYA zero-stub audit passes cleanly
    from gwaya.ast_audit import ZeroStubAudit

    audit_res = ZeroStubAudit().audit_python_code(script_source)
    assert audit_res.is_clean, f"ZeroStubAudit violations: {audit_res.violations}"


def test_transparency_sheet_fr_en_answers_all_four_friend_gamer_questions() -> None:
    assert DOC_PATH.is_file()
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "## Partie FR" in text
    assert "## EN Section" in text

    fr_part, en_part = text.split("## EN Section", 1)

    # Question 1: Personal home PC only / Amadeus compliance rule & 0.00 EUR / 0.00 USD
    assert "Amadeus" in fr_part and "Amadeus" in en_part
    assert "0,00 EUR" in fr_part and "0.00 EUR" in en_part and "0.00 USD" in en_part
    assert "PC personnel" in fr_part and "personal home PCs" in en_part
    assert "AGPL-3.0" in fr_part and "AGPL-3.0" in en_part

    # Question 2: Real electricity cost in Wh/cents (~60W * 2h = 120 Wh = 0.12 kWh)
    assert "60 W" in fr_part and "2 h" in fr_part and "120 Wh" in fr_part and "0,12 kWh" in fr_part
    assert "60 W" in en_part and "2 h" in en_part and "120 Wh" in en_part and "0.12 kWh" in en_part
    assert "centimes" in fr_part and "cents" in en_part

    # Question 3: What leaves my PC (<50 MB/h LoRA math weight deltas over HTTPS 443, zero personal files/prompts)
    assert "50 MB/h" in fr_part and "50 MB/h" in en_part
    assert "LoRA" in fr_part and "LoRA" in en_part
    assert "HTTPS" in fr_part and "443" in fr_part
    assert "HTTPS" in en_part and "443" in en_part
    assert "*.run.app" in fr_part and "*.run.app" in en_part
    assert "datasets-server.huggingface.co" in fr_part and "datasets-server.huggingface.co" in en_part

    # Question 4: Pause in <1.5s for Steam or uninstall in 10s (--uninstall)
    assert "1,5 s" in fr_part and "Steam" in fr_part and "--uninstall" in fr_part and "10 s" in fr_part
    assert "1.5 s" in en_part and "Steam" in en_part and "--uninstall" in en_part and "10-second" in en_part
    assert "hivemind_quickstart.py --uninstall" in fr_part
    assert "hivemind_quickstart.py --uninstall" in en_part
    assert "join_hivemind.ps1 -Mode Uninstall" in fr_part
    assert "join_hivemind.ps1 -Mode Uninstall" in en_part


def test_five_line_bulletin_measured_rendering_across_fr_en_and_bilingual() -> None:
    t4 = _sample_measured_t4()
    fb = _sample_measured_feedback()

    for lang in ("bilingual", "fr", "en"):
        text = bulletin_mod.render_daily_raid_bulletin(t4, fb, lang=lang)
        lines = text.splitlines()
        assert len(lines) == 5, f"Expected 5 lines for {lang}, got {len(lines)}"
        assert all(ln.strip() for ln in lines)
        assert len(text) < bulletin_mod.DISCORD_MAX_CHARS
        assert "2026-12-21" in lines[0]
        assert "FRESH" in lines[0]
        assert "9" in lines[0] and "7" in lines[0]
        assert "10519 MiB" in lines[1] and "15360 MiB" in lines[1]
        assert "58.5 C" in lines[1] and "30.2 W" in lines[1]
        assert "145" in lines[2] and "20" in lines[2]
        assert "2.2705 -> 2.1380" in lines[2]
        assert "12" in lines[3] and "dev 5 / gamer 7" in lines[3]
        assert "4.50/5" in lines[3] and "100.0%" in lines[3]
        assert "Amadeus" in lines[4] and "--uninstall" in lines[4]
        assert ("0,00 USD" if lang == "fr" else "0.00 USD") in lines[4]
        assert "n/a" not in text

    receipt = bulletin_mod.build_bulletin_receipt(t4, fb, lang="fr-BE", date_override="2026-12-24")
    assert receipt["language"] == "fr"
    assert receipt["bulletin_line_count"] == 5
    assert len(receipt["bulletin_lines"]) == 5
    assert "2026-12-24" in receipt["bulletin_fr"]
    assert "2026-12-24" in receipt["bulletin_en"]
    assert "2026-12-24" in receipt["bulletin_bilingual"]
    assert len(receipt["receipt_sha256"]) == 64

    # Deterministic SHA-256 receipt check
    receipt_again = bulletin_mod.build_bulletin_receipt(t4, fb, lang="fr", date_override="2026-12-24")
    assert receipt["receipt_sha256"] == receipt_again["receipt_sha256"]


def test_absent_and_unreceived_metrics_strictly_render_na_without_inventing_numbers() -> None:
    # 1. Completely empty inputs -> every metric slot in lines 1..4 is 'n/a', zero invented metrics
    for lang in ("bilingual", "fr", "en"):
        empty_text = bulletin_mod.render_daily_raid_bulletin({}, {}, lang=lang)
        lines = empty_text.splitlines()
        assert len(lines) == 5
        assert empty_text.count("n/a") >= 15
        # Aside from fixed label 'T4' on line 2 and 'QLoRA' on line 3, lines 1..4 must carry no invented digits
        scrubbed = "\n".join(lines[:4]).replace("T4", "")
        assert re.search(r"\d", scrubbed) is None, f"Invented digit found in empty bulletin ({lang}): {scrubbed!r}"

    # 2. Zero feedback responses -> counts are 0, but mean_ease_rating and zero_game_lag_ratio are 'n/a'
    fb_zero = {
        "total_responses": 0,
        "mean_ease_rating": None,
        "zero_game_lag_ratio": None,
        "by_profile": {"dev": 0, "gamer": 0},
    }
    zero_text = bulletin_mod.render_daily_raid_bulletin(None, fb_zero, lang="en")
    z_lines = zero_text.splitlines()
    assert len(z_lines) == 5
    assert "(0 replies, dev 0 / gamer 0): ease n/a | zero Steam lag n/a" in z_lines[3]

    # 3. Unreceived T4 telemetry -> static fallback 4841 MiB / 145 rounds MUST NOT appear; rendered as 'n/a'
    unreceived_t4 = {
        "telemetry_status": "NO_TELEMETRY_RECEIVED",
        "telemetry_source": "DEFAULT_SNAPSHOT_NO_TELEMETRY_RECEIVED",
        "operational_mode": "SOLE_T4_INFERENCE_PRIORITY_ACTIVE",
        "active_producers_count": 2,
        "distinct_physical_gpus": 2,
        "hardware": {"vram_free_mib": 10519, "vram_total_mib": 15360, "temp_c": 66, "power_w": 30.2},
        "qlora_progress": {
            "qlora_rounds_assimilated": 145,
            "ratchet_accepted_rounds": 20,
            "held_out_eval_loss_before": 2.2705,
            "held_out_eval_loss_after": 2.1380,
        },
    }
    unrec_text = bulletin_mod.render_daily_raid_bulletin(unreceived_t4, {}, lang="bilingual")
    u_lines = unrec_text.splitlines()
    assert len(u_lines) == 5
    assert "NO_TELEMETRY_RECEIVED" in u_lines[0]
    assert "Producers: 2 | GPUs: 2" in u_lines[0]
    assert "10519" not in unrec_text and "145" not in unrec_text and "2.2705" not in unrec_text
    assert "VRAM libre/free n/a / n/a | Temp n/a | Power n/a" in u_lines[1]
    assert "rondes/rounds n/a (ratchet: n/a) | Loss n/a -> n/a" in u_lines[2]

    # 4. Raw GPU block & fallback field aliases (`gpu.free_mib`, `assimilated_workunits`, `average_ease_rating`)
    alt_t4 = {
        "captured_utc": "2026-12-25T08:00:00Z",
        "gpu": {"free_mib": 8192, "total_mib": 12288, "temp_c": 54, "power_w": 60},
        "qlora_progress": {"assimilated_workunits": 33, "latest_eval_loss_after": 1.9876},
    }
    alt_fb = {"feedback_responses": 4, "average_ease_rating": 4.75}
    alt_text = bulletin_mod.render_daily_raid_bulletin(alt_t4, alt_fb, lang="en-US")
    assert "2026-12-25" in alt_text
    assert "8192 MiB / 12288 MiB | Temp 54 C | Power 60 W" in alt_text
    assert "assimilated rounds 33 (ratchet-kept: n/a) | Eval loss n/a -> 1.9876" in alt_text
    assert "4.75/5" in alt_text


def test_live_router_gateway_integration_before_and_after_telemetry() -> None:
    admin_key = "hm-slice07-admin-sync-secret"
    saved_snap = (
        t4_mod._LATEST_T4_SNAPSHOT["updated_epoch"],
        t4_mod._LATEST_T4_SNAPSHOT["telemetry_source"],
        dict(t4_mod._LATEST_T4_SNAPSHOT["gpu"]),
        dict(t4_mod._LATEST_T4_SNAPSHOT["qlora_progress"]),
    )
    try:
        t4_mod._LATEST_T4_SNAPSHOT["updated_epoch"] = None
        t4_mod._LATEST_T4_SNAPSHOT["telemetry_source"] = "DEFAULT_SNAPSHOT_NO_TELEMETRY_RECEIVED"

        store = RouterStore(":memory:", b"s" * 32)
        app = create_router_app(store, admin_key_hash=store.key_digest(admin_key))
        client = TestClient(app)

        def _client_http_get(url: str, _timeout_s: float) -> tuple[int, dict[str, Any]]:
            path = "/" + url.split("://", 1)[1].split("/", 1)[1]
            resp = client.get(path)
            return resp.status_code, resp.json()

        # Before T4 telemetry push or feedback: T4 hardware/QLoRA & ease rating must all be 'n/a'
        t4_pre, fb_pre, codes_pre = bulletin_mod.fetch_live_bulletin_inputs(
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            http_get=_client_http_get,
        )
        assert codes_pre == {"t4_status_http": 200, "feedback_summary_http": 200}
        pre_bulletin = bulletin_mod.render_daily_raid_bulletin(t4_pre, fb_pre, lang="fr")
        assert len(pre_bulletin.splitlines()) == 5
        assert "NO_TELEMETRY_RECEIVED" in pre_bulletin
        assert "VRAM libre n/a / n/a" in pre_bulletin
        assert "facilité n/a | zéro saccade Steam n/a" in pre_bulletin

        # Push live T4 telemetry + 2 anonymous friend feedback submissions
        r_push = client.post(
            "/v1/hivemind/t4/telemetry",
            headers={"Authorization": f"Bearer {admin_key}"},
            json={
                "gpu": {"used_mib": 5000, "total_mib": 15360, "temp_c": 57, "power_w": 42.5},
                "qlora_progress": {"qlora_rounds_assimilated": 160, "ratchet_accepted_rounds": 24},
            },
        )
        assert r_push.status_code == 200
        assert (
            client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "gamer", "ease_rating": 5, "zero_game_lag": True, "platform": "windows"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": "dev", "ease_rating": 4, "zero_game_lag": True, "platform": "linux"},
            ).status_code
            == 200
        )

        t4_live, fb_live, _ = bulletin_mod.fetch_live_bulletin_inputs(
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            http_get=_client_http_get,
        )
        post_bulletin = bulletin_mod.render_daily_raid_bulletin(t4_live, fb_live, lang="bilingual")
        assert len(post_bulletin.splitlines()) == 5
        assert "FRESH" in post_bulletin
        assert "10360 MiB / 15360 MiB" in post_bulletin
        assert "Temp 57 C | Power 42.5 W" in post_bulletin
        assert "rondes/rounds 160 (ratchet: 24)" in post_bulletin
        assert "4.50/5" in post_bulletin and "100.0%" in post_bulletin
    finally:
        t4_mod._LATEST_T4_SNAPSHOT["updated_epoch"] = saved_snap[0]
        t4_mod._LATEST_T4_SNAPSHOT["telemetry_source"] = saved_snap[1]
        t4_mod._LATEST_T4_SNAPSHOT["gpu"] = saved_snap[2]
        t4_mod._LATEST_T4_SNAPSHOT["qlora_progress"] = saved_snap[3]


def test_loopback_http_server_and_error_branches() -> None:
    mode = {"kind": "ok"}

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            k = mode["kind"]
            if k == "ok":
                if self.path == "/v1/hivemind/t4/status":
                    body = json.dumps(_sample_measured_t4()).encode("utf-8")
                else:
                    body = json.dumps(_sample_measured_feedback()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif k == "http_404_text":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not deployed yet")
            elif k == "http_503_json":
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'["list_error"]')
            elif k == "bad_json":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{broken-json")
            elif k == "list_json":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"[1, 2, 3]")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        rc = bulletin_mod.main(["--live", "--gateway", base_url, "--lang", "en"])
        assert rc == 0

        # Non-200 HTTP responses degrade missing metrics safely to 'n/a' instead of inventing figures
        mode["kind"] = "http_404_text"
        t4_doc, fb_doc, codes = bulletin_mod.fetch_live_bulletin_inputs(base_url, timeout_s=2.0)
        assert t4_doc == {} and fb_doc == {} and codes["t4_status_http"] == 404

        mode["kind"] = "http_503_json"
        code_503, doc_503 = bulletin_mod.default_http_get_json(f"{base_url}/v1/hivemind/t4/status", 2.0)
        assert code_503 == 503 and doc_503 == {"data": ["list_error"]}

        mode["kind"] = "bad_json"
        with pytest.raises(bulletin_mod.BulletinInputError, match="Malformed JSON"):
            bulletin_mod.default_http_get_json(f"{base_url}/v1/hivemind/t4/status", 2.0)

        mode["kind"] = "list_json"
        with pytest.raises(bulletin_mod.BulletinInputError, match="Expected JSON object"):
            bulletin_mod.default_http_get_json(f"{base_url}/v1/hivemind/t4/status", 2.0)
    finally:
        server.shutdown()
        server.server_close()

    # Unreachable closed loopback port -> BulletinInputError
    with pytest.raises(bulletin_mod.BulletinInputError, match="Unable to reach gateway"):
        bulletin_mod.default_http_get_json(f"http://127.0.0.1:{port}/v1/hivemind/t4/status", 0.5)


def test_gateway_allowlist_safety_scanner_and_validation_negative_controls() -> None:
    # Allowed gateway URLs
    assert bulletin_mod.validate_gateway_url("https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/") == (
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
    )
    assert bulletin_mod.validate_gateway_url("https://storage.googleapis.com") == "https://storage.googleapis.com"
    assert bulletin_mod.validate_gateway_url("https://compute.googleapis.com") == "https://compute.googleapis.com"
    assert bulletin_mod.validate_gateway_url("https://datasets-server.huggingface.co") == (
        "https://datasets-server.huggingface.co"
    )
    assert bulletin_mod.validate_gateway_url("http://localhost:8080") == "http://localhost:8080"
    assert bulletin_mod.validate_gateway_url("http://[::1]:8000") == "http://[::1]:8000"
    assert not bulletin_mod.is_allowed_gateway_host("")

    # Rejected gateway URLs
    for bad_url in (
        "",
        True,
        "https://evil.example.com",
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "ftp://127.0.0.1:8080",
        "https://user:pass@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/extra",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app?x=1",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app#frag",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app:99999",
        "http://127.0.0.1:0",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app\n",
    ):
        with pytest.raises(bulletin_mod.BulletinInputError):
            bulletin_mod.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    # Timeout validation
    with pytest.raises(bulletin_mod.BulletinInputError):
        bulletin_mod.fetch_live_bulletin_inputs("http://127.0.0.1:8000", timeout_s=True)  # type: ignore[arg-type]
    with pytest.raises(bulletin_mod.BulletinInputError):
        bulletin_mod.fetch_live_bulletin_inputs("http://127.0.0.1:8000", timeout_s=-1.0)

    # Discord mass-ping & PII scanner negative controls
    unsafe_token_t4 = {"telemetry_status": "@everyone PWNED", "operational_mode": "bad\nline"}
    safe_rendered = bulletin_mod.render_daily_raid_bulletin(unsafe_token_t4, {})
    assert "@everyone" not in safe_rendered
    assert len(safe_rendered.splitlines()) == 5

    assert "email_detected" in bulletin_mod.scan_output_safety("contact alice@example.com")
    assert "api_key_detected" in bulletin_mod.scan_output_safety("hm-secretkey123456")
    assert "bearer_token_detected" in bulletin_mod.scan_output_safety("Bearer secretToken123")
    assert "home_path_detected" in bulletin_mod.scan_output_safety("/home/alice/.hivemind")
    assert "discord_mass_ping_detected" in bulletin_mod.scan_output_safety("hey @here wake up")
    assert "forbidden_key_detected" in bulletin_mod.scan_output_safety('{"sc' + 'ore": 5}')
    with pytest.raises(bulletin_mod.BulletinInputError, match="Output safety guard failed"):
        bulletin_mod._assert_safe_output("leak @everyone")

    # Input validation negative controls
    for bad_lang in (123, True, "de"):
        with pytest.raises(bulletin_mod.BulletinInputError):
            bulletin_mod.normalize_lang(bad_lang)  # type: ignore[arg-type]

    with pytest.raises(bulletin_mod.BulletinInputError):
        bulletin_mod.render_bulletin_from_metrics("not_metrics")  # type: ignore[arg-type]
    with pytest.raises(bulletin_mod.BulletinInputError):
        bulletin_mod.extract_bulletin_metrics("bad")  # type: ignore[arg-type]
    with pytest.raises(bulletin_mod.BulletinInputError):
        bulletin_mod.extract_bulletin_metrics({}, "bad")  # type: ignore[arg-type]

    for bad_date in (True, "", "21-12-2026", "2026-02-30"):
        with pytest.raises(bulletin_mod.BulletinInputError):
            bulletin_mod.extract_bulletin_metrics({}, {}, date_override=bad_date)  # type: ignore[arg-type]

    # Non-strict invalid date fields inside t4_status safely fall back to None ('n/a')
    m_bad_inferred = bulletin_mod.extract_bulletin_metrics(
        {"european_schedule": {"current_utc_iso": "2026-99-99"}, "date": 123},
        {},
    )
    assert m_bad_inferred.report_date is None

    for bad_t4, bad_fb in (
        ({"european_schedule": "bad"}, {}),
        ({"hardware": []}, {}),
        ({"gpu": "bad"}, {}),
        ({"qlora_progress": "bad"}, {}),
        ({"active_producers_count": True}, {}),
        ({"active_producers_count": 1.5}, {}),
        ({"active_producers_count": -1}, {}),
        ({"hardware": {"temp_c": True}}, {}),
        ({"hardware": {"temp_c": float("nan")}}, {}),
        ({"hardware": {"temp_c": 999.0}}, {}),
        ({}, {"by_profile": []}),
        ({}, {"total_responses": 2, "by_profile": {"dev": 2, "gamer": 1}}),
        ({}, {"total_responses": 0, "mean_ease_rating": 4.5}),
        ({}, {"mean_ease_rating": 6.0}),
        ({}, {"zero_game_lag_ratio": -0.1}),
    ):
        with pytest.raises(bulletin_mod.BulletinInputError):
            bulletin_mod.extract_bulletin_metrics(bad_t4, bad_fb)


def test_cli_files_bundles_stdin_and_subprocess_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t4_file = tmp_path / "t4.json"
    fb_file = tmp_path / "fb.json"
    bundle_file = tmp_path / "bundle.json"
    fb_only_file = tmp_path / "fb_only.json"
    out_txt = tmp_path / "out" / "bulletin.txt"
    out_json = tmp_path / "out" / "receipt.json"

    t4_file.write_text(json.dumps(_sample_measured_t4()), encoding="utf-8")
    fb_file.write_text(json.dumps(_sample_measured_feedback()), encoding="utf-8")
    bundle_file.write_text(
        json.dumps({"t4_status": _sample_measured_t4(), "onboarding_feedback": _sample_measured_feedback()}),
        encoding="utf-8",
    )
    fb_only_file.write_text(json.dumps(_sample_measured_feedback()), encoding="utf-8")

    # 1. Separate --t4-status and --feedback-summary with --output and --json-out
    rc = bulletin_mod.main(
        [
            "--t4-status",
            str(t4_file),
            "--feedback-summary",
            str(fb_file),
            "--lang",
            "both",
            "--output",
            str(out_txt),
            "--json-out",
            str(out_json),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert len(captured.splitlines()) == 5
    assert out_txt.read_text(encoding="utf-8") == captured
    saved_receipt = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved_receipt["bulletin_line_count"] == 5

    # 2. Combined --snapshot with --json output
    rc_bundle = bulletin_mod.main(["--snapshot", str(bundle_file), "--json"])
    assert rc_bundle == 0
    parsed_stdout = json.loads(capsys.readouterr().out)
    assert parsed_stdout["metrics"]["qlora_rounds_assimilated"] == 145
    assert parsed_stdout["metrics"]["mean_ease_rating"] == 4.5

    # 3. Single-endpoint snapshots via --snapshot (feedback-only vs t4-only)
    assert bulletin_mod.main(["--snapshot", str(fb_only_file)]) == 0
    fb_snap_out = capsys.readouterr().out
    assert "4.50/5" in fb_snap_out and "VRAM libre/free n/a / n/a" in fb_snap_out

    assert bulletin_mod.main(["--snapshot", str(t4_file)]) == 0
    t4_snap_out = capsys.readouterr().out
    assert "10519 MiB" in t4_snap_out and "facilité/ease n/a" in t4_snap_out

    # 4. Stdin '-' support
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_sample_measured_t4())))
    assert bulletin_mod.main(["--t4-status", "-", "--lang", "fr"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 5

    # 5. CLI error paths & unsafe error message redaction
    assert bulletin_mod.main([]) == 2
    assert bulletin_mod.main(["--t4-status", str(t4_file), "--live"]) == 2
    assert bulletin_mod.main(["--t4-status", "-", "--feedback-summary", "-"]) == 2
    assert bulletin_mod.main(["--t4-status", str(tmp_path / "missing.json")]) == 2

    empty_file = tmp_path / "empty.json"
    empty_file.write_text("   ", encoding="utf-8")
    assert bulletin_mod.main(["--t4-status", str(empty_file)]) == 2

    syntax_file = tmp_path / "syntax.json"
    syntax_file.write_text("{bad", encoding="utf-8")
    assert bulletin_mod.main(["--t4-status", str(syntax_file)]) == 2

    arr_file = tmp_path / "arr.json"
    arr_file.write_text("[1, 2]", encoding="utf-8")
    assert bulletin_mod.main(["--t4-status", str(arr_file)]) == 2

    bad_bundle_t4 = tmp_path / "bad_bundle_t4.json"
    bad_bundle_t4.write_text('{"t4_status": 123}', encoding="utf-8")
    assert bulletin_mod.main(["--snapshot", str(bad_bundle_t4)]) == 2

    bad_bundle_fb = tmp_path / "bad_bundle_fb.json"
    bad_bundle_fb.write_text('{"feedback_summary": []}', encoding="utf-8")
    assert bulletin_mod.main(["--snapshot", str(bad_bundle_fb)]) == 2

    with pytest.raises(bulletin_mod.BulletinInputError):
        bulletin_mod.load_json_file("   ")

    # Error message carrying PII/token is redacted on stderr
    assert bulletin_mod.main(["--lang", "hm-leakedsecret123456"]) == 2
    err_out = capsys.readouterr().err
    assert "hm-leakedsecret123456" not in err_out
    assert "redacted unsafe token" in err_out

    # 6. Direct subprocess CLI execution (`__main__` path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--t4-status", str(t4_file), "--feedback-summary", str(fb_file)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert proc.returncode == 0
    assert len(proc.stdout.splitlines()) == 5
