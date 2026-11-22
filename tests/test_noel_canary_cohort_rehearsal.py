# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""50-workspace multi-OS Noël canary & scale cohort rehearsal test suite (`SLICE-SCALE-02`)."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import coverage
import pytest
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
REHEARSAL_SCRIPT_PATH = REPO_ROOT / "scripts" / "noel_canary_cohort_rehearsal.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(REHEARSAL_SCRIPT_PATH)],
    )
    _INPROC_COV.start()

_spec = importlib.util.spec_from_file_location("noel_canary_cohort_rehearsal", REHEARSAL_SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
rehearsal = importlib.util.module_from_spec(_spec)
sys.modules["noel_canary_cohort_rehearsal"] = rehearsal
_spec.loader.exec_module(rehearsal)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_scale02",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_scale02",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin"}
PIPELINE_MODULES = rehearsal.load_onboarding_pipeline_modules()


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
    assert rehearsal.verify_sub_receipt_sha256(report) is True


def test_compliance_headers_stdlib_lateral_movement_gatekeeper_and_security_scanners() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    this_file = Path(__file__).resolve()
    for path in (REHEARSAL_SCRIPT_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    script_source = REHEARSAL_SCRIPT_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Lateral-movement AST audit (allowed hosts only, zero raw sockets, zero SSH/LAN probes)
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source,
            filename="scripts/noel_canary_cohort_rehearsal.py",
        )
        == []
    )
    poisoned = script_source + '\n_BAD_EGRESS = "http://192.168.10.25:8080/noel"\n'
    assert lateral_guard.audit_source_for_lateral_movement(poisoned, "poisoned.py") != []

    # 2. Pure Python 3.10-3.13 stdlib only in scripts/noel_canary_cohort_rehearsal.py
    tree = ast.parse(script_source, filename="scripts/noel_canary_cohort_rehearsal.py")
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
        ("scripts/noel_canary_cohort_rehearsal.py", script_source),
        ("tests/test_noel_canary_cohort_rehearsal.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("scripts/noel_canary_cohort_rehearsal.py", script_source),
        ("tests/test_noel_canary_cohort_rehearsal.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1062,
        pr_title="feat(scale): 50-workspace multi-OS Noel canary & scale cohort rehearsal suite",
        pr_body="Implements SLICE-SCALE-02 50-workspace multi-OS canary & scale cohort rehearsal suite.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(scale): 50-workspace multi-OS Noel canary & scale cohort rehearsal suite"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    from gwaya.ast_audit import ZeroStubAudit

    stub_audit = ZeroStubAudit().audit_python_code(script_source)
    assert stub_audit.is_clean, f"Unexpected GWAYA stub violations: {stub_audit.violations}"

    # 5. Output safety negative controls
    bad_key_fragment = '{"' + "sco" + 're": 5}'
    raw_hm_key = "hm-" + "k" * 18
    assert "forbidden_key_detected" in rehearsal.scan_output_safety(bad_key_fragment)
    assert "unmasked_api_key_detected" in rehearsal.scan_output_safety(raw_hm_key)
    assert "bearer_token_detected" in rehearsal.scan_output_safety("Bearer secret_tok_987654")
    assert "email_detected" in rehearsal.scan_output_safety("alice@example.org")
    assert "unredacted_home_path_detected" in rehearsal.scan_output_safety("/home/alice/.hivemind")
    assert "discord_mass_ping_detected" in rehearsal.scan_output_safety("hello @everyone")
    assert any(
        hit.startswith("non_loopback_ipv4_detected:")
        for hit in rehearsal.scan_output_safety("peer 10.11.12.13")
    )
    assert rehearsal.scan_output_safety("loopback 127.0.0.1 masked hm-***1234") == []
    with pytest.raises(rehearsal.RehearsalError, match="requires a string"):
        rehearsal.scan_output_safety(123)  # type: ignore[arg-type]
    with pytest.raises(rehearsal.RehearsalError, match="Output safety guard failed"):
        rehearsal._assert_safe_output(bad_key_fragment)


def test_50_workspace_parallel_rehearsal_across_5_archetypes_and_negative_controls() -> None:
    with tempfile.TemporaryDirectory(prefix="hm_test_canary50_") as raw_tmp:
        root_dir = Path(raw_tmp) / "rehearsal_tree"
        report = rehearsal.run_canary_cohort_rehearsal(
            rehearsal.DEFAULT_GATEWAY,
            cohort_size=50,
            corporate_controls=10,
            workers=16,
            report_date="2026-12-25",
            lang="bilingual",
            environ=CLEAN_ENV,
            rehearsal_root=root_dir,
            keep_workspaces=True,
            pipeline_modules=PIPELINE_MODULES,
            generated_at_utc="2026-12-25T20:00:00Z",
        )
        _verify_sha256_receipt(report)

        # Required top-level keys & exact readiness invariants
        assert report["rehearsal_ok"] is True
        assert report["verdict"] == "ready"
        assert report["exit_code"] == 0
        assert report["cohort_size"] == 50
        assert report["personal_ready_count"] == 50
        assert report["personal_failed_count"] == 0
        assert report["personal_ready_ratio"] == 1.0
        assert report["corporate_cohort_size"] == 10
        assert report["corporate_blocked_count"] == 10
        assert report["corporate_leaked_count"] == 0
        assert report["corporate_blocked_ratio"] == 1.0
        assert report["corporate_file_writes_count"] == 0
        assert report["corporate_network_calls_count"] == 0
        assert report["zero_inference_verified"] is True
        assert report["inference_calls_count"] == 0
        assert report["zero_key_leaks_verified"] is True
        assert report["workspace_key_leak_count"] == 0
        assert report["report_key_leak_count"] == 0
        assert report["diagnostics"] == []

        if rehearsal.os.name != "nt":
            assert report["posix_0600_verified_count"] == 60

        # Gamer-Whisper <1500 ms yield & <=70 C thermal enforcement
        gw_enf = report["gamer_whisper_enforcement"]
        assert gw_enf["gpu_gamer_workspaces_count"] == 20
        assert gw_enf["nogpu_selftest_workspaces_count"] == 10
        assert gw_enf["gamer_yield_slo_ok_count"] == 20
        assert gw_enf["gamer_thermal_ok_count"] == 20
        assert gw_enf["gamer_vram_reserve_ok_count"] == 20
        assert gw_enf["gamer_yield_slo_all_ok"] is True
        assert gw_enf["gamer_thermal_all_ok"] is True
        assert 0.0 < gw_enf["max_gamer_yield_latency_ms"] < 1500.0
        assert 0.0 < gw_enf["p95_gamer_yield_latency_ms"] < 1500.0
        assert 0.0 < gw_enf["max_observed_gpu_temp_c"] <= 70.0

        # All 5 personal-PC archetypes exercised at N=10 each
        arch_by_id = {row["archetype_id"]: row for row in report["archetype_breakdown"]}
        assert set(arch_by_id.keys()) == {
            "linux_dev",
            "macos_dev",
            "win11_rtx_8gb_gamer",
            "win11_rtx_16gb_gamer",
            "nogpu_laptop_dev",
        }
        for arch_id, row in arch_by_id.items():
            assert row["total_count"] == 10, arch_id
            assert row["ready_count"] == 10, arch_id
            assert row["ready_ratio"] == 1.0, arch_id
            assert row["mean_ease_rating"] == 4.8, arch_id
            assert row["zero_game_lag_ratio"] == 1.0, arch_id

        # All 5 corporate / Amadeus negative-control vectors blocked 100% locally
        corp_rows = report["corporate_control_breakdown"]
        assert len(corp_rows) == 5
        for c_row in corp_rows:
            assert c_row["total_count"] == 2
            assert c_row["blocked_count"] == 2
            assert c_row["file_writes_count"] == 0
            assert c_row["network_calls_count"] == 0
            assert len(c_row["matched_markers"]) >= 1

        # All 5 pipeline scripts exercised with cryptographic receipts
        pipe = report["pipeline_verification"]
        assert pipe["scripts_exercised"] == [
            "noel_vscode_autoconfig",
            "noel_gamer_whisper_selftest",
            "noel_post_setup_probe",
            "onboarding_funnel_report",
            "daily_raid_bulletin",
        ]
        assert pipe["autoconfig_runs_count"] == 60
        assert pipe["gamer_selftest_runs_count"] == 40
        assert pipe["post_setup_probe_runs_count"] == 50
        assert pipe["sub_receipts_verified_count"] == 152
        assert pipe["all_sub_receipts_valid"] is True

        # Exact end-to-end funnel conversion & CES aggregation
        funnel = report["funnel_aggregation"]
        assert funnel["status"] == "healthy"
        assert funnel["conservation_ok"] is True
        assert funnel["landing_visits"] == 50
        assert funnel["signups"] == 50
        assert funnel["first_requests"] == 50
        assert funnel["preflight_runs"] == 20
        assert funnel["install_completed"] == 20
        assert funnel["first_rounds"] == 20
        assert funnel["activation_ratio"] == 1.0
        assert funnel["signup_conversion_ratio"] == 1.0
        assert funnel["preflight_to_first_round_ratio"] == 1.0
        assert funnel["install_dropoff_ratio"] == 0.0
        assert funnel["feedback_responses"] == 50
        assert funnel["feedback_by_profile"] == {"dev": 30, "gamer": 20}
        assert funnel["mean_ease_rating"] == 4.8
        assert funnel["zero_game_lag_ratio"] == 1.0
        assert funnel["alerts"] == []

        # Daily raid bulletin parity
        bull = report["daily_raid_bulletin"]
        assert bull["bulletin_line_count"] == 5
        assert bull["active_producers_count"] == 20
        assert bull["feedback_total_responses"] == 50
        assert bull["mean_ease_rating"] == 4.8
        assert bull["zero_game_lag_ratio"] == 1.0
        assert len(bull["bulletin_en"].splitlines()) == 5
        assert len(bull["bulletin_fr"].splitlines()) == 5

        # Direct filesystem verification on isolated temporary directories
        sample_vs = (
            root_dir
            / "personal"
            / "ws_000_linux_dev"
            / "home_box"
            / "projects"
            / "hobby_open_source"
            / ".vscode"
            / "settings.json"
        )
        assert sample_vs.is_file()
        assert rehearsal._API_KEY_RE.search(sample_vs.read_text(encoding="utf-8")) is None

        corp_home = root_dir / "negative_ctrl" / "ctrl_000" / "home_box"
        assert not (corp_home / ".hivemind" / "credentials.json").exists()
        assert not (corp_home / ".continue" / "hivemind_provider.yaml").exists()

        # Verify FR, EN, bilingual Markdown and CLI renderers leak zero secrets or sentinels
        md_en = rehearsal.render_markdown_report(report, lang="en")
        md_fr = rehearsal.render_markdown_report(report, lang="fr")
        md_bi = rehearsal.render_markdown_report(report, lang="bilingual")
        cli_txt = rehearsal.render_cli_report(report)
        json_txt = json.dumps(report, indent=2, ensure_ascii=False)

        assert "Multi-OS Noël Canary" in md_en
        assert "Répétition Générale Cohorte Noël" in md_fr
        assert "---" in md_bi
        assert "rehearsal_ok=True" in cli_txt
        for c_spec in rehearsal.CORPORATE_NEGATIVE_CONTROLS:
            for out_blob in (md_en, md_fr, md_bi, cli_txt, json_txt):
                assert c_spec.secret_sentinel not in out_blob


def test_partial_gamer_thermal_and_yield_failure_injection_and_total_gpu_pause() -> None:
    # 1. Partial failure injection across 50 workspaces: 2 thermal (>70 C) + 5 yield (>=1500 ms) = 7/20 failed
    report = rehearsal.run_canary_cohort_rehearsal(
        "http://127.0.0.1:8000",
        cohort_size=50,
        corporate_controls=5,
        workers=12,
        inject_gamer_thermal_failures=2,
        inject_gamer_yield_failures=5,
        report_date="2026-12-26",
        lang="fr",
        environ=CLEAN_ENV,
        pipeline_modules=PIPELINE_MODULES,
    )
    _verify_sha256_receipt(report)

    assert report["rehearsal_ok"] is False
    assert report["verdict"] == "failed_rehearsal_gate"
    assert report["exit_code"] == 2
    assert report["cohort_size"] == 50
    assert report["personal_ready_count"] == 43
    assert report["personal_failed_count"] == 7
    assert report["personal_ready_ratio"] == 0.86
    assert report["corporate_blocked_ratio"] == 1.0

    gw_enf = report["gamer_whisper_enforcement"]
    assert gw_enf["gamer_thermal_ok_count"] == 18
    assert gw_enf["gamer_yield_slo_ok_count"] == 15
    assert gw_enf["gamer_thermal_all_ok"] is False
    assert gw_enf["gamer_yield_slo_all_ok"] is False
    assert gw_enf["max_observed_gpu_temp_c"] == rehearsal.INJECTED_THERMAL_SPIKE_C
    assert gw_enf["max_gamer_yield_latency_ms"] >= 1500.0

    funnel = report["funnel_aggregation"]
    assert funnel["conservation_ok"] is True
    assert funnel["status"] == "alert"
    assert funnel["preflight_runs"] == 20
    assert funnel["install_completed"] == 13
    assert funnel["first_rounds"] == 13
    assert funnel["install_dropoff_ratio"] == 0.35
    assert funnel["zero_game_lag_ratio"] == 0.86
    alert_codes = {a["code"] for a in funnel["alerts"]}
    assert "high_install_dropoff" in alert_codes
    assert "game_lag_detected" in alert_codes
    assert len(report["diagnostics"]) >= 3

    md_fr = rehearsal.render_markdown_report(report, lang="fr")
    assert "## Diagnostics" in md_fr
    assert "ÉCHEC" in md_fr

    # 2. All GPU gamer workspaces paused (cohort_size=5 has 2 GPU gamers -> inject 1 thermal + 1 yield)
    # Exercises NO_TELEMETRY_RECEIVED branch on /v1/hivemind/t4/status in daily_raid_bulletin
    all_paused = rehearsal.run_canary_cohort_rehearsal(
        "http://127.0.0.1:8000",
        cohort_size=5,
        corporate_controls=0,
        workers=4,
        inject_gamer_thermal_failures=1,
        inject_gamer_yield_failures=1,
        report_date="2026-12-26",
        lang="en",
        environ=CLEAN_ENV,
        pipeline_modules=PIPELINE_MODULES,
    )
    _verify_sha256_receipt(all_paused)
    assert all_paused["rehearsal_ok"] is False
    assert all_paused["personal_ready_count"] == 3
    assert all_paused["personal_ready_ratio"] == 0.6
    assert all_paused["corporate_blocked_ratio"] == 1.0
    assert all_paused["daily_raid_bulletin"]["active_producers_count"] == 0
    assert "NO_TELEMETRY_RECEIVED" in all_paused["daily_raid_bulletin"]["bulletin_en"]
    md_en_fail = rehearsal.render_markdown_report(all_paused, lang="en")
    assert "FAILED" in md_en_fail
    assert "## Diagnostics" in md_en_fail


def test_negative_controls_for_key_leaks_workspace_breaches_and_host_compliance_refusal() -> None:
    # 1. Host machine compliance refusal (both EN and FR, before creating any workspaces)
    host_refused_en = rehearsal.run_canary_cohort_rehearsal(
        "http://127.0.0.1:8000",
        cohort_size=5,
        corporate_controls=2,
        environ={"PATH": "/usr/bin", "AMADEUS_ENV": "1", "USERDNSDOMAIN": "internal.example"},
        lang="en",
        pipeline_modules=PIPELINE_MODULES,
    )
    _verify_sha256_receipt(host_refused_en)
    assert host_refused_en["rehearsal_ok"] is False
    assert host_refused_en["verdict"] == "refused_host_compliance"
    assert host_refused_en["exit_code"] == 2
    assert host_refused_en["personal_ready_ratio"] == 0.0

    host_refused_fr = rehearsal.run_canary_cohort_rehearsal(
        "http://127.0.0.1:8000",
        cohort_size=5,
        corporate_controls=2,
        environ={
            "PATH": "/usr/bin",
            "ZSCALER_CERT_PATH": "/etc/zscaler.pem",
            "HTTPS_PROXY": "netskope-gw:8443",
        },
        lang="fr",
        pipeline_modules=PIPELINE_MODULES,
    )
    assert host_refused_fr["verdict"] == "refused_host_compliance"
    assert "REFUS HÔTE" in host_refused_fr["diagnostics"][0]

    # Local Windows personal account where USERDOMAIN == COMPUTERNAME must pass host compliance
    win_personal = rehearsal.evaluate_host_compliance(
        environ={"COMPUTERNAME": "HOME-RIG", "USERDOMAIN": "HOME-RIG"},
        lang="en",
    )
    assert win_personal["compliance_ok"] is True

    # 2. Negative control: planted raw hm-... key & Bearer token in a personal workspace file
    def _plant_workspace_key_leak(
        cohort_kind: str,
        item_id: str,
        _ws_home: Path,
        ws_dir: Path,
    ) -> None:
        if cohort_kind == "personal" and item_id == "linux_dev":
            leaked_key = "hm-" + "leakedsecret123456"
            (ws_dir / ".vscode" / "leaked.txt").write_text(
                f"Authorization: Bearer tok_abc12345\nkey={leaked_key}\n",
                encoding="utf-8",
            )

    leaked_report = rehearsal.run_canary_cohort_rehearsal(
        "http://127.0.0.1:8000",
        cohort_size=5,
        corporate_controls=2,
        workers=4,
        lang="en",
        environ=CLEAN_ENV,
        pipeline_modules=PIPELINE_MODULES,
        workspace_post_hook=_plant_workspace_key_leak,
    )
    _verify_sha256_receipt(leaked_report)
    assert leaked_report["rehearsal_ok"] is False
    assert leaked_report["zero_key_leaks_verified"] is False
    assert leaked_report["workspace_key_leak_count"] >= 1
    assert any("leak detected" in d for d in leaked_report["diagnostics"])

    # Also verify French diagnostic message for key leak + corporate file write breach
    def _plant_control_file_write_and_leak(
        cohort_kind: str,
        _item_id: str,
        ws_home: Path,
        ws_dir: Path,
    ) -> None:
        if cohort_kind == "corporate":
            (ws_home / "unwanted_write.json").write_text("{}", encoding="utf-8")
        elif cohort_kind == "personal":
            (ws_dir / "leak.env").write_text("KEY=hm-" + "badleak99999999\n", encoding="utf-8")

    breach_fr = rehearsal.run_canary_cohort_rehearsal(
        "http://127.0.0.1:8000",
        cohort_size=5,
        corporate_controls=2,
        workers=4,
        lang="fr",
        environ=CLEAN_ENV,
        pipeline_modules=PIPELINE_MODULES,
        workspace_post_hook=_plant_control_file_write_and_leak,
    )
    _verify_sha256_receipt(breach_fr)
    assert breach_fr["rehearsal_ok"] is False
    assert breach_fr["corporate_blocked_ratio"] == 0.0
    assert breach_fr["corporate_file_writes_count"] == 2
    assert any("Amadeus/entreprise" in d for d in breach_fr["diagnostics"])
    assert any("Fuite de clé" in d for d in breach_fr["diagnostics"])


def test_cli_entrypoint_modes_mock_gateway_guards_and_input_validation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="hm_test_cli_") as raw_tmp:
        tmp_dir = Path(raw_tmp)
        json_out_file = tmp_dir / "out" / "rehearsal_receipt.json"

        # 1. CLI --json + --output
        rc_json = rehearsal.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--cohort-size",
                "5",
                "--corporate-controls",
                "5",
                "--workers",
                "5",
                "--date",
                "2026-12-25",
                "--lang",
                "en",
                "--json",
                "--output",
                str(json_out_file),
            ],
            environ=CLEAN_ENV,
            pipeline_modules=PIPELINE_MODULES,
        )
        assert rc_json == 0
        captured_json = json.loads(capsys.readouterr().out)
        assert json_out_file.is_file()
        assert json.loads(json_out_file.read_text(encoding="utf-8"))["rehearsal_ok"] is True
        _verify_sha256_receipt(captured_json)

        # 2. CLI --cli mode
        rc_cli = rehearsal.main(
            [
                "--gateway",
                "http://localhost:8000",
                "--cohort-size",
                "5",
                "--corporate-controls",
                "2",
                "--cli",
            ],
            environ=CLEAN_ENV,
            pipeline_modules=PIPELINE_MODULES,
        )
        assert rc_cli == 0
        assert "[Noel Canary Cohort Rehearsal: OK]" in capsys.readouterr().out

        # 3. CLI default --markdown mode + failure exit code 2 when thermal failure injected
        rc_md_fail = rehearsal.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--cohort-size",
                "5",
                "--corporate-controls",
                "2",
                "--inject-gamer-thermal-failures",
                "1",
                "--lang",
                "fr",
                "--markdown",
            ],
            environ=CLEAN_ENV,
            pipeline_modules=PIPELINE_MODULES,
        )
        assert rc_md_fail == 2
        assert "Répétition Générale Cohorte Noël" in capsys.readouterr().out

        # 4. CLI error exit code 2 on invalid arguments
        rc_bad = rehearsal.main(["--lang", "es"], environ=CLEAN_ENV, pipeline_modules=PIPELINE_MODULES)
        assert rc_bad == 2
        assert "Error:" in capsys.readouterr().err

        # 5. MockZeroInferenceGateway edge cases (zero-inference guard, empty summary, 401, 404)
        gw = rehearsal.MockZeroInferenceGateway("http://127.0.0.1:8000", report_date="2026-12-25")
        empty_code, empty_fb = gw.request_json("GET", "http://127.0.0.1:8000/v1/hivemind/onboarding/feedback/summary")
        assert empty_code == 200 and empty_fb["total_responses"] == 0 and empty_fb["mean_ease_rating"] is None

        unauth_code, _ = gw.request_json(
            "GET",
            "http://127.0.0.1:8000/v1/hivemind/me",
            None,
            {"Authorization": "Bearer hm-unknown_token_0000"},
        )
        assert unauth_code == 401

        not_found_code, _ = gw.request_json("GET", "http://127.0.0.1:8000/v1/unknown/route")
        assert not_found_code == 404

        with pytest.raises(rehearsal.RehearsalError, match="Zero-inference invariant violated"):
            gw.request_json("POST", "http://127.0.0.1:8000/v1/chat/completions", {"messages": []})
        assert gw.snapshot_state()["inference_calls_count"] == 1

        # 6. Helper & validator edge cases
        assert rehearsal.normalize_lang("en-US") == "en"
        assert rehearsal.normalize_lang("fr_FR") == "fr"
        assert rehearsal.normalize_lang("both") == "bilingual"
        with pytest.raises(rehearsal.RehearsalError):
            rehearsal.normalize_lang(True)  # type: ignore[arg-type]

        assert rehearsal.is_allowed_gateway_host("") is False
        assert rehearsal.is_allowed_gateway_host("storage.googleapis.com") is True
        assert rehearsal.is_allowed_gateway_host("datasets-server.huggingface.co") is True
        assert rehearsal.validate_gateway_url("https://storage.googleapis.com") == "https://storage.googleapis.com"
        assert len(rehearsal.validate_report_date(None)) == 10

        for bad_url in (
            "",
            True,
            "http://127.0.0.1:8000\nbad",
            "http://127.0.0.1:8000 /space",
            "http://user:pw@127.0.0.1:8000",
            "http://127.0.0.1:8000/extra",
            "http://127.0.0.1:8000?q=1",
            "http://127.0.0.1:abc",
            "http://127.0.0.1:0",
            "https://untrusted-domain.invalid",
            "ftp://127.0.0.1:8000",
            "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        ):
            with pytest.raises(rehearsal.RehearsalError):
                rehearsal.validate_gateway_url(bad_url)  # type: ignore[arg-type]

        for bad_date in ("", True, "2026/12/25", "2026-02-30"):
            with pytest.raises(rehearsal.RehearsalError):
                rehearsal.validate_report_date(bad_date)  # type: ignore[arg-type]

        assert rehearsal.verify_sub_receipt_sha256({}) is False
        assert rehearsal.scan_directory_for_key_leaks(tmp_dir / "does_not_exist") == (0, [])
        assert rehearsal._percentile_nearest_rank([], 95.0) is None

        with pytest.raises(rehearsal.RehearsalError, match="not found"):
            rehearsal._load_script_module("missing_mod", tmp_dir / "missing.py")

        for bad_kwargs in (
            {"cohort_size": 4},
            {"corporate_controls": -1},
            {"workers": 0},
            {"yield_slo_ms": 0.0},
            {"max_temp_c": 120.0},
            {"min_free_vram_mib": 64},
            {"inject_gamer_thermal_failures": -1},
            {"inject_gamer_thermal_failures": 3, "cohort_size": 5},
        ):
            with pytest.raises(rehearsal.RehearsalError):
                rehearsal.run_canary_cohort_rehearsal(
                    "http://127.0.0.1:8000",
                    environ=CLEAN_ENV,
                    pipeline_modules=PIPELINE_MODULES,
                    **bad_kwargs,  # type: ignore[arg-type]
                )


def test_zz_rehearsal_script_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% line coverage on {REHEARSAL_SCRIPT_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
