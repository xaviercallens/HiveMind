# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, security, compliance, and gateway integration tests for SLICE-DYN-01 (`scripts/noel_post_setup_probe.py`)."""

from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import json
import os
import re
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
PROBE_PATH = REPO_ROOT / "scripts" / "noel_post_setup_probe.py"

_spec = importlib.util.spec_from_file_location("noel_post_setup_probe", PROBE_PATH)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
sys.modules["noel_post_setup_probe"] = probe
_spec.loader.exec_module(probe)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_mod",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_mod",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

TEST_KEY = "hm-test_noel_probe_key_987654"
TEST_LIVE_KEY = "hm_live_noel_probe_key_123456"


def _write_creds(path: Path, api_key: str = TEST_KEY, mode: int = 0o600, role: str = "prosumer") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "account_id": "acct_noel_secret_99",
                "api_key": api_key,
                "gateway": probe.DEFAULT_GATEWAY,
                "handle": "secret-alice-handle",
                "role": role,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        os.chmod(path, mode)
    return path


def _write_continue_yaml(path: Path, api_key: str = TEST_KEY) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "name: HiveMind Holiday Lab\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: HiveMind Coder\n"
        "    provider: openai\n"
        "    model: hivemind/coder\n"
        f"    apiBase: {probe.DEFAULT_GATEWAY}/v1\n"
        f"    apiKey: {api_key}\n",
        encoding="utf-8",
    )
    return path


def _mock_http_healthy(
    seen: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] | None = None,
    *,
    mean_ease: float | None = 4.8,
    zero_lag: float | None = 1.0,
    total_fb: int = 12,
    me_status: int = 200,
) -> probe.HttpFn:
    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert "completions" not in url.lower(), "Probe must NEVER call /v1/chat/completions"
        if seen is not None:
            seen.append((method, url, payload, headers))
        if method == "GET" and url.endswith("/health"):
            return 200, {"status": "ok"}
        if method == "GET" and url.endswith("/v1/models"):
            return 200, {"object": "list", "data": [{"id": "hivemind/auto"}, {"id": "hivemind/coder"}]}
        if method == "GET" and any(url.endswith(p) for p in ("/noel", "/quick", "/raid-noel", "/arena-noel")):
            return 200, {"content_type": "text/html; charset=utf-8", "html_length": 4096}
        if method == "GET" and url.endswith("/v1/hivemind/onboarding/feedback/summary"):
            return 200, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": total_fb,
                "mean_ease_rating": mean_ease,
                "zero_game_lag_ratio": zero_lag,
                "by_profile": {"dev": 7, "gamer": 5},
            }
        if method == "GET" and url.endswith("/v1/hivemind/me"):
            if me_status != 200:
                return me_status, {"error": "unauthorized"}
            return 200, {"account_id": "acct_1", "handle": "friend", "role": "trainer", "balance": 500.0}
        if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
            return 200, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": total_fb + 1,
                "mean_ease_rating": 4.85,
                "zero_game_lag_ratio": 1.0,
                "by_profile": {"dev": 7, "gamer": 6},
            }
        return 404, {"error": "not_found"}

    return _http


def _mock_runner(
    *,
    smi_rc: int = 0,
    smi_out: str = "NVIDIA GeForce RTX 4070, 12288, 9216, 54, 4\n",
    smi_exc: Exception | None = None,
    ps_rc: int = 0,
    ps_out: str = "systemd\nbash\npython3\n",
    tasklist_rc: int = 0,
    tasklist_out: str = '"explorer.exe","1200","Console","1","85,000 K"\n"steam.exe","2100","Console","1","95,000 K"\n',
    proc_exc: Exception | None = None,
) -> probe.RunnerFn:
    def _run(argv: Any, timeout_s: float) -> tuple[int, str]:
        assert timeout_s <= 5.0
        cmd0 = str(argv[0])
        if cmd0 == "nvidia-smi":
            joined = " ".join(str(x) for x in argv).lower()
            assert "uuid" not in joined and "serial" not in joined
            if smi_exc is not None:
                raise smi_exc
            return smi_rc, smi_out
        if proc_exc is not None:
            raise proc_exc
        if cmd0 == "tasklist":
            return tasklist_rc, tasklist_out
        if cmd0 == "ps":
            return ps_rc, ps_out
        return 1, ""

    return _run


def _check_by_id(report: dict[str, Any], check_id: str) -> dict[str, Any]:
    return next(item for item in report["checks"] if item["id"] == check_id)


def test_compliance_headers_stdlib_lateral_movement_and_forbidden_patterns() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    this_file = Path(__file__).resolve()
    for path in (PROBE_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    probe_source = PROBE_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Lateral-movement AST audit
    violations = lateral_guard.audit_source_for_lateral_movement(
        probe_source,
        filename="scripts/noel_post_setup_probe.py",
    )
    assert violations == []

    # 2. Pure Python stdlib-only check
    tree = ast.parse(probe_source, filename="scripts/noel_post_setup_probe.py")
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 3. PR gatekeeper FORBIDDEN_ADDED scan across both new files
    for rel, content in (
        ("scripts/noel_post_setup_probe.py", probe_source),
        ("tests/test_noel_post_setup_probe.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner + GWAYA ZeroStubAudit
    scanner = PRSecurityScanner()
    for rel, content in (
        ("scripts/noel_post_setup_probe.py", probe_source),
        ("tests/test_noel_post_setup_probe.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []

    from gwaya.ast_audit import ZeroStubAudit

    audit_res = ZeroStubAudit().audit_python_code(probe_source)
    assert audit_res.is_clean, f"Unexpected GWAYA stub violations: {audit_res.violations}"

    # 5. Output safety negative controls (assembled from fragments to avoid diff false positives)
    bad_key_fragment = '{"' + "sco" + 're": 5}'
    assert "forbidden_key_detected" in probe.scan_output_safety(bad_key_fragment)
    assert "email_detected" in probe.scan_output_safety("contact user@example.org")
    assert "unmasked_api_key_detected" in probe.scan_output_safety(f"token={TEST_KEY}")
    assert "bearer_token_detected" in probe.scan_output_safety("Authorization: Bearer secret_token_123")
    assert "unredacted_home_path_detected" in probe.scan_output_safety("/home/alice/.hivemind/credentials.json")
    assert any(
        issue.startswith("non_loopback_ipv4_detected:")
        for issue in probe.scan_output_safety("connected to 10.20.30.40")
    )
    with pytest.raises(probe.ProbeError, match="Output safety guard failed"):
        probe._assert_safe_output(bad_key_fragment)


def test_healthy_smoke_and_full_probe_runs_with_sha256_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "xavier_home_pc"
    cred_path = _write_creds(home_dir / ".hivemind" / "credentials.json", mode=0o600)
    cont_path = _write_continue_yaml(home_dir / ".continue" / "hivemind_provider.yaml")
    seen_calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] = []

    # 1. Smoke-only CLI run (--smoke-routes --json)
    rc_smoke = probe.main(
        ["--smoke-routes", "--json"],
        http_fn=_mock_http_healthy(seen_calls),
        runner=_mock_runner(),
        environ={"PATH": "/usr/bin"},
        home=home_dir,
    )
    assert rc_smoke == 0
    smoke_json = json.loads(capsys.readouterr().out)
    assert smoke_json["mode"] == "smoke_routes"
    assert smoke_json["ready"] is True
    assert smoke_json["routes_passed_count"] == 7
    assert smoke_json["routes_total_count"] == 7
    assert smoke_json["checks_passed_count"] == 1
    assert smoke_json["mean_ease_rating"] == 4.8
    assert smoke_json["zero_inference_verified"] is True
    assert len(smoke_json["receipt_sha256"]) == 64
    assert [u.split(".run.app", 1)[1] for _, u, _, _ in seen_calls] == [
        "/health",
        "/v1/models",
        "/noel",
        "/quick",
        "/raid-noel",
        "/arena-noel",
        "/v1/hivemind/onboarding/feedback/summary",
    ]

    # Verify deterministic SHA-256 receipt calculation
    core_copy = dict(smoke_json)
    expected_digest = hashlib.sha256(
        json.dumps(
            {k: v for k, v in core_copy.items() if k != "receipt_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert smoke_json["receipt_sha256"] == expected_digest

    # 2. Full post-setup probe with --verify-account on Linux (JSON + Markdown)
    seen_full: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] = []
    rc_full = probe.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--verify-account",
            "--lang",
            "fr",
            "--json",
        ],
        http_fn=_mock_http_healthy(seen_full),
        runner=_mock_runner(),
        environ={"PATH": "/usr/bin"},
        home=home_dir,
        os_name="posix",
    )
    assert rc_full == 0
    full_json = json.loads(capsys.readouterr().out)
    assert full_json["ready"] is True
    assert full_json["verdict"] == "ready"
    assert full_json["recommended_mode"] == "gamer_whisper"
    assert full_json["routes_passed_count"] == 7
    assert full_json["checks_passed_count"] == 6
    assert full_json["checks_total_count"] == 6
    cred_chk = _check_by_id(full_json, "credentials")
    assert cred_chk["details"]["path"] == "~/.hivemind/credentials.json"
    assert cred_chk["details"]["masked_key"] == "hm-***7654"
    assert cred_chk["details"]["account_verified"] is True
    assert cred_chk["details"]["role"] == "trainer"

    raw_text = json.dumps(full_json)
    assert "xavier_home_pc" not in raw_text
    assert TEST_KEY not in raw_text
    assert "0.00 EUR / 0.00 USD" in full_json["compliance"]["notice_bilingual"]
    assert "0,00 EUR / 0,00 USD" in full_json["compliance"]["notice_bilingual"]

    # Markdown output rendering (EN & FR)
    rc_md = probe.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--markdown",
            "--lang",
            "en",
        ],
        http_fn=_mock_http_healthy(),
        runner=_mock_runner(),
        environ={"PATH": "/usr/bin"},
        home=home_dir,
        os_name="nt",
    )
    assert rc_md == 0
    md_out = capsys.readouterr().out
    assert "# HiveMind Holiday Post-Setup Probe & Smoke Gate" in md_out
    assert "| `/noel` | `200` | **OK** |" in md_out
    assert "**SHA-256 Receipt**:" in md_out


def test_stale_cloud_run_404_routes_and_route_error_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _stale_cloud_run_http(
        method: str,
        url: str,
        _payload: dict[str, Any] | None = None,
        _headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if method == "GET" and url.endswith("/health"):
            return 200, {"status": "ok"}
        if method == "GET" and url.endswith("/v1/models"):
            return 200, {"data": [{"id": "hivemind/coder"}]}
        return 404, {"detail": "Not Found"}

    # Stale Cloud Run in EN (--smoke-routes returns exit code 2!)
    rc_en = probe.main(["--smoke-routes", "--lang", "en", "--json"], http_fn=_stale_cloud_run_http)
    assert rc_en == 2
    rep_en = json.loads(capsys.readouterr().out)
    assert rep_en["ready"] is False
    assert rep_en["routes_passed_count"] == 2
    assert rep_en["routes_total_count"] == 7
    smoke_en = rep_en["smoke_gate"]
    assert smoke_en["cloud_run_lags_main"] is True
    assert smoke_en["missing_routes"] == [
        "/noel",
        "/quick",
        "/raid-noel",
        "/arena-noel",
        "/v1/hivemind/onboarding/feedback/summary",
    ]
    assert smoke_en["http_statuses"]["/noel"] == 404
    assert smoke_en["http_statuses"]["/health"] == 200
    assert any("lags behind `main`" in d for d in smoke_en["diagnostics"])

    # Stale Cloud Run in FR (full probe with valid local setup STILL exits 2 when required routes fail!)
    home_dir = tmp_path / "home" / "fr_tester"
    cred_path = _write_creds(home_dir / ".hivemind" / "credentials.json")
    cont_path = _write_continue_yaml(home_dir / ".continue" / "hivemind_provider.yaml")
    rc_fr = probe.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--lang",
            "fr",
        ],
        http_fn=_stale_cloud_run_http,
        runner=_mock_runner(),
        environ={"PATH": "/usr/bin"},
        home=home_dir,
        os_name="posix",
    )
    assert rc_fr == 2
    md_fr = capsys.readouterr().out
    assert "en retard sur `main`" in md_fr
    assert "| `/raid-noel` | `404` | **FAIL** |" in md_fr

    # Exercise remaining route failure branches: core 404, malformed /v1/models, 429, 503, 0, 403
    def _mixed_failures_http(_m: str, url: str, _p: Any = None, _h: Any = None) -> tuple[int, dict[str, Any]]:
        if url.endswith("/health"):
            return 404, {}
        if url.endswith("/v1/models"):
            return 200, {"data": "not-a-list"}
        if url.endswith("/noel"):
            return 429, {}
        if url.endswith("/quick"):
            return 503, {}
        if url.endswith("/raid-noel"):
            raise OSError("connection reset")
        if url.endswith("/arena-noel"):
            return 403, {}
        return 200, {"total_responses": 0, "mean_ease_rating": None, "zero_game_lag_ratio": None}

    mixed_en = probe.run_holiday_route_smoke_gate(http_fn=_mixed_failures_http, lang="en")
    assert mixed_en["all_routes_ok"] is False
    assert mixed_en["cloud_run_lags_main"] is False
    assert mixed_en["routes_passed_count"] == 1
    assert mixed_en["http_statuses"]["/raid-noel"] == 0

    mixed_fr = probe.run_holiday_route_smoke_gate(http_fn=_mixed_failures_http, lang="fr")
    assert mixed_fr["routes_passed_count"] == 1
    assert len(mixed_fr["diagnostics"]) == 6


def test_ces_feedback_reporter_200_404_429_500_and_interactive_1_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    posted: list[tuple[str, dict[str, Any] | None, dict[str, str] | None]] = []

    def _capture_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if method == "POST":
            posted.append((url, payload, headers))
        return _mock_http_healthy()(method, url, payload, headers)

    # 1. Healthy non-interactive --submit-feedback (--json)
    rc_ok = probe.main(
        [
            "--submit-feedback",
            "--profile",
            "gamer",
            "--ease-rating",
            "5",
            "--json",
        ],
        http_fn=_capture_http,
    )
    assert rc_ok == 0
    fb_json = json.loads(capsys.readouterr().out)
    assert fb_json["ready"] is True
    assert fb_json["ease_rating"] == 5
    assert fb_json["mean_ease_rating"] == 4.85
    assert fb_json["feedback"]["zero_game_lag"] is True
    assert len(posted) == 1
    _, sent_payload, sent_headers = posted[0]
    assert sent_headers is None
    assert set(sent_payload or {}) == {"profile", "ease_rating", "zero_game_lag", "platform"}
    assert sent_payload["profile"] == "gamer"
    assert sent_payload["ease_rating"] == 5

    # 2. HTTP 404 on feedback (stale Cloud Run -> exit code 2 in FR & EN)
    rc_404 = probe.main(
        ["--submit-feedback", "--profile", "dev", "--ease-rating", "4", "--lang", "fr"],
        http_fn=lambda *_: (404, {"detail": "Not Found"}),
    )
    assert rc_404 == 2
    out_404 = capsys.readouterr().out
    assert "HTTP 404" in out_404 and "en retard sur `main`" in out_404

    res_404_en = probe.submit_ces_feedback(ease_rating=4, profile="dev", http_fn=lambda *_: (404, {}), lang="en")
    assert res_404_en["status"] == "missing_route" and res_404_en["submitted"] is False

    # 3. HTTP 429 rate-limited & HTTP 500 server error & network 0 & HTTP 400 in EN/FR
    rc_429 = probe.main(
        ["--submit-feedback", "--profile", "gamer", "--ease-rating", "3", "--had-game-lag", "--lang", "en", "--json"],
        http_fn=lambda *_: (429, {"error": "rate_limited"}),
    )
    assert rc_429 == 1
    out_429 = json.loads(capsys.readouterr().out)
    assert out_429["feedback"]["status"] == "rate_limited"
    assert out_429["feedback"]["zero_game_lag"] is False

    res_429_fr = probe.submit_ces_feedback(ease_rating=5, http_fn=lambda *_: (429, {}), lang="fr")
    assert res_429_fr["status"] == "rate_limited"

    rc_500 = probe.main(
        ["--submit-feedback", "--profile", "dev", "--ease-rating", "5", "--lang", "fr", "--json"],
        http_fn=lambda *_: (500, {"error": "internal"}),
    )
    assert rc_500 == 1
    assert json.loads(capsys.readouterr().out)["feedback"]["status"] == "server_error"

    res_502_en = probe.submit_ces_feedback(ease_rating=5, http_fn=lambda *_: (502, {}), lang="en")
    assert res_502_en["status"] == "server_error"

    res_net_en = probe.submit_ces_feedback(
        ease_rating=4,
        system_name="Darwin",
        http_fn=lambda *_: (_ for _ in ()).throw(OSError("offline")),
        lang="en",
    )
    assert res_net_en["status"] == "network_error" and res_net_en["zero_pii_payload"]["platform"] == "macos"

    res_net_fr = probe.submit_ces_feedback(
        ease_rating=4,
        system_name="Windows",
        http_fn=lambda *_: (0, {"network_error": "URLError"}),
        lang="fr",
    )
    assert res_net_fr["status"] == "network_error" and res_net_fr["zero_pii_payload"]["platform"] == "windows"

    res_400_en = probe.submit_ces_feedback(
        ease_rating=2,
        system_name="FreeBSD!",
        http_fn=lambda *_: (400, {"error": "bad"}),
        lang="en",
    )
    assert res_400_en["status"] == "client_error" and res_400_en["zero_pii_payload"]["platform"] == "freebsd"
    res_400_fr = probe.submit_ces_feedback(ease_rating=2, system_name="!!!", http_fn=lambda *_: (422, {}), lang="fr")
    assert res_400_fr["status"] == "client_error" and res_400_fr["zero_pii_payload"]["platform"] == "unknown"

    # 4. Interactive 1-prompt CES parsing & injectable input_fn (verified single prompt call!)
    prompts_seen: list[str] = []

    def _one_prompt_input(prompt_str: str) -> str:
        prompts_seen.append(prompt_str)
        return "4/5 gamer lag"

    rc_interactive = probe.main(
        ["--submit-feedback", "--prompt-ces", "--lang", "en", "--json"],
        input_fn=_one_prompt_input,
        http_fn=_mock_http_healthy(),
    )
    assert rc_interactive == 0
    assert len(prompts_seen) == 1
    inter_json = json.loads(capsys.readouterr().out)
    assert inter_json["ease_rating"] == 4
    assert inter_json["feedback"]["profile"] == "gamer"
    assert inter_json["feedback"]["zero_game_lag"] is False

    # Interactive skip via empty Enter and EOFError
    skip_en = probe.prompt_and_submit_ces_feedback(input_fn=lambda _p: "   ", http_fn=_mock_http_healthy(), lang="en")
    assert skip_en["skipped"] is True and skip_en["attempted"] is False
    skip_eof = probe.prompt_and_submit_ces_feedback(
        input_fn=lambda _p: (_ for _ in ()).throw(EOFError()),
        http_fn=_mock_http_healthy(),
        lang="fr",
    )
    assert skip_eof["skipped"] is True

    # Parser unit variations & invalid inputs
    p_dev = probe.parse_interactive_ces_input("5 dev nolag", default_profile="gamer", default_had_game_lag=True)
    assert p_dev == {"skipped": False, "profile": "dev", "ease_rating": 5, "zero_game_lag": True}

    for bad_ces in ("0", "6", "5 4", "banana", "gamer lag"):
        with pytest.raises(probe.ProbeError):
            probe.parse_interactive_ces_input(bad_ces)

    # Missing --ease-rating without --prompt-ces -> exit 2
    assert probe.main(["--submit-feedback"], http_fn=_mock_http_healthy()) == 2
    assert probe.main(["--submit-feedback", "--ease-rating", "9"], http_fn=_mock_http_healthy()) == 2

    # Combined full probe + failing feedback returns feedback error code
    home_dir = tmp_path / "home" / "combo"
    cred = _write_creds(home_dir / ".hivemind" / "credentials.json")
    cont = _write_continue_yaml(home_dir / ".continue" / "hivemind_provider.yaml")

    def _probe_ok_fb_429(m: str, u: str, p: Any = None, h: Any = None) -> tuple[int, dict[str, Any]]:
        if m == "POST":
            return 429, {"error": "busy"}
        return _mock_http_healthy()(m, u, p, h)

    rc_combo = probe.main(
        [
            "--credentials-file",
            str(cred),
            "--continue-config-path",
            str(cont),
            "--submit-feedback",
            "--run-probe",
            "--ease-rating",
            "5",
            "--json",
        ],
        http_fn=_probe_ok_fb_429,
        runner=_mock_runner(),
        environ={"PATH": "/usr/bin"},
        home=home_dir,
        os_name="posix",
    )
    assert rc_combo == 1
    capsys.readouterr()


def test_gpu_scenarios_active_steam_game_yield_budget_and_corporate_warning(tmp_path: Path) -> None:
    home_dir = tmp_path / "Users" / "GamerHome"
    cred_path = _write_creds(home_dir / ".hivemind" / "credentials.json", api_key=TEST_LIVE_KEY, mode=0o644)
    cont_path = _write_continue_yaml(home_dir / ".continue" / "hivemind_provider.yaml", api_key=TEST_LIVE_KEY)

    # 1. Quiet GPU (<= 70.0 C, >= 2048 MiB free) + idle Steam tray on Windows -> ready!
    quiet_rep = probe.run_post_setup_probe(
        credentials_path=cred_path,
        continue_path=cont_path,
        os_name="nt",
        environ={"COMPUTERNAME": "HOME-RIG", "USERDOMAIN": "HOME-RIG"},
        home=home_dir,
        http_fn=_mock_http_healthy(),
        runner=_mock_runner(smi_out="NVIDIA GeForce RTX 4070, 12288, 6144, 70.0, 5\n"),
        lang="en",
    )
    assert quiet_rep["ready"] is True
    assert quiet_rep["recommended_mode"] == "gamer_whisper"
    assert _check_by_id(quiet_rep, "gamer_whisper_gpu")["status"] == "ok"

    # 2. Hot GPU (> 70.0 C) in EN & FR -> warn (gpu_thermal_cooldown)
    hot_en = probe.check_gamer_whisper_gpu(
        runner=_mock_runner(smi_out="NVIDIA GeForce RTX 4080, 16384, 12000, 74.5, 80\n"),
        lang="en",
    )
    assert hot_en.status == "warn"
    assert hot_en.details["mode"] == "gpu_thermal_cooldown"
    assert hot_en.details["thermal_cap_ok"] is False

    hot_fr = probe.check_gamer_whisper_gpu(
        runner=_mock_runner(smi_out="NVIDIA GeForce RTX 4080, 16384, 12000, 72.0, [N/A]\n"),
        lang="fr",
    )
    assert hot_fr.status == "warn" and "72.0 C" in hot_fr.summary

    # 3. Low free VRAM (< 2048 MiB) -> warn (vram_reserve_hold)
    low_vram = probe.check_gamer_whisper_gpu(
        runner=_mock_runner(smi_out="NVIDIA GeForce RTX 3060, 8192, 1536, 58.0, 20\n"),
        lang="en",
    )
    assert low_vram.status == "warn"
    assert low_vram.details["mode"] == "vram_reserve_hold"
    assert low_vram.details["vram_reserve_ok"] is False
    low_vram_fr = probe.check_gamer_whisper_gpu(
        runner=_mock_runner(smi_out="NVIDIA GeForce RTX 3060, 8192, 1024, 58.0, 20\n"),
        lang="fr",
    )
    assert low_vram_fr.status == "warn"

    # 4. Dev no-GPU scenario & nvidia-smi parse error scenarios
    no_gpu = probe.check_gamer_whisper_gpu(
        runner=_mock_runner(smi_exc=FileNotFoundError("no nvidia-smi")),
        lang="en",
    )
    assert no_gpu.status == "ok" and no_gpu.details["mode"] == "dev_api_only"
    no_gpu_fr = probe.check_gamer_whisper_gpu(runner=_mock_runner(smi_rc=1, smi_out=""), lang="fr")
    assert no_gpu_fr.status == "ok" and no_gpu_fr.details["gpu_present"] is False

    for bad_smi in ("corrupt_single_col\n", "GPU, not_a_num, 2048, 50, 0\n", "GPU, 0, 0, 50, 0\n"):
        bad_gpu = probe.check_gamer_whisper_gpu(runner=_mock_runner(smi_out=bad_smi), lang="en")
        assert bad_gpu.status == "warn" and bad_gpu.details["parse_error"] is True
    bad_gpu_fr = probe.check_gamer_whisper_gpu(runner=_mock_runner(smi_out="bad\n"), lang="fr")
    assert bad_gpu_fr.status == "warn"

    # 5. Active Steam game (cs2.exe on Windows & eldenring.exe / custom game on Linux) + < 1.5 s yield budget
    ticks = iter([10.0, 10.04])
    win_game = probe.check_active_games_and_yield_budget(
        runner=_mock_runner(tasklist_out='"cs2.exe","4400","Console","1","4,200,000 K"\n'),
        clock_fn=lambda: next(ticks),
        os_name="nt",
        lang="en",
    )
    assert win_game.status == "warn"
    assert win_game.details["active_games"] == ["cs2.exe"]
    assert win_game.details["yield_budget_ok"] is True
    assert win_game.details["estimated_yield_s"] < 1.5

    nix_game = probe.check_active_games_and_yield_budget(
        runner=_mock_runner(ps_out="systemd\n/opt/games/custom_raid_game\n"),
        extra_games=["custom_raid_game"],
        os_name="posix",
        lang="fr",
    )
    assert nix_game.status == "warn" and "custom_raid_game" in nix_game.details["active_games"]

    # Exceeded >= 1.5 s yield budget simulation + ps/tasklist error fallbacks
    slow_ticks = iter([1.0, 2.5])
    slow_yield = probe.check_active_games_and_yield_budget(
        runner=_mock_runner(),
        clock_fn=lambda: next(slow_ticks),
        os_name="posix",
        lang="en",
    )
    assert slow_yield.status == "warn" and slow_yield.details["yield_budget_ok"] is False

    slow_ticks_fr = iter([1.0, 2.6])
    slow_fr = probe.check_active_games_and_yield_budget(
        runner=_mock_runner(ps_rc=1),
        clock_fn=lambda: next(slow_ticks_fr),
        os_name="posix",
        lang="fr",
    )
    assert slow_fr.status == "warn"
    assert probe.list_process_names(_mock_runner(tasklist_rc=1), os_name="nt") == set()
    assert probe.list_process_names(_mock_runner(proc_exc=OSError("ps failed")), os_name="posix") == set()

    # 6. Corporate warning (Amadeus / work env markers strictly local, never leaked or transmitted)
    seen_corp_http: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] = []
    corp_env = {
        "AMADEUS_ENV_STAGE": "secret-internal-prod",
        "USERDNSDOMAIN": "CORP.AMADEUS.INTERNAL",
        "USERDOMAIN": "AMADEUSCORP",
        "COMPUTERNAME": "CORP-LAPTOP-99",
        "HTTPS_PROXY": "http://proxy.corp.internal:8080",
        "NO_PROXY": ".amadeus.internal",
        "ZSCALER_CERT_PATH": "/etc/zscaler/root.pem",
        "SSL_CERT_FILE": "/etc/certs/amadeus-ca.pem",
    }
    corp_rep = probe.run_post_setup_probe(
        credentials_path=cred_path,
        continue_path=cont_path,
        os_name="posix",
        environ=corp_env,
        home=home_dir,
        http_fn=_mock_http_healthy(seen_corp_http),
        runner=_mock_runner(),
        lang="fr",
    )
    assert corp_rep["verdict"] == "ready_with_warnings"
    corp_chk = _check_by_id(corp_rep, "personal_pc_compliance")
    assert corp_chk["status"] == "warn"
    assert corp_chk["details"]["corporate_warning"] is True
    assert corp_chk["details"]["network_transmitted"] is False
    assert "AMADEUS_ENV" in corp_chk["details"]["corporate_indicators"]
    assert "USERDNSDOMAIN" in corp_chk["details"]["corporate_indicators"]

    # Ensure raw secret corporate values NEVER appear in JSON or Markdown or outbound HTTP calls
    corp_json_str = json.dumps(corp_rep)
    corp_md_str = probe.render_markdown_report(corp_rep)
    for secret_val in ("secret-internal-prod", "CORP.AMADEUS.INTERNAL", "CORP-LAPTOP-99", "proxy.corp.internal"):
        assert secret_val not in corp_json_str
        assert secret_val not in corp_md_str
    assert all(payload is None and headers is None for _, _, payload, headers in seen_corp_http)


def test_credentials_continue_yaml_and_url_rejection_edge_cases(tmp_path: Path) -> None:
    # 1. URL validation & zero-inference enforcement
    assert probe.validate_gateway_url("https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/") == probe.DEFAULT_GATEWAY
    assert probe.validate_gateway_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert probe.validate_gateway_url("http://localhost:5000/") == "http://localhost:5000"
    assert probe.validate_gateway_url("https://[::1]:8443") == "https://[::1]:8443"
    assert probe.is_allowed_gateway_host("") is False

    invalid_urls: list[Any] = [
        "",
        "   ",
        True,
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://attacker.example.com",
        "http://192.168.1.25:8080",
        "ftp://localhost:21",
        "https://user:pass@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app?x=1",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app#frag",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/extra/path",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app:abc",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app:0",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app\ninjected",
        "https://.run.app",
        "https://bad..run.app",
        "https://storage.googleapis.com",
    ]
    for bad in invalid_urls:
        with pytest.raises(probe.ProbeError):
            probe.validate_gateway_url(bad)

    # Rejects /v1/chat/completions zero-inference violation before http_fn is ever invoked
    with pytest.raises(probe.ProbeError, match="Zero-inference invariant"):
        probe._safe_http_call(
            lambda *_: (200, {}),
            "POST",
            f"{probe.DEFAULT_GATEWAY}/v1/chat/completions",
            {"model": "hivemind/coder"},
            None,
        )

    # CLI exits 2 on invalid URL or invalid --lang
    assert probe.main(["--gateway", "https://evil.example.org"]) == 2
    assert probe.main(["--lang", "es"]) == 2
    with pytest.raises(probe.ProbeError):
        probe.normalize_lang(True)  # type: ignore[arg-type]
    with pytest.raises(probe.ProbeError):
        probe.normalize_profile("admin")
    with pytest.raises(probe.ProbeError):
        probe.normalize_profile(None)
    with pytest.raises(probe.ProbeError):
        probe.submit_ces_feedback(ease_rating=True)  # type: ignore[arg-type]
    with pytest.raises(probe.ProbeError):
        probe.submit_ces_feedback(ease_rating=5, zero_game_lag="yes")  # type: ignore[arg-type]

    # 2. Credentials file edge cases (missing, corrupt JSON, invalid key, /v1/hivemind/me 401, redact fallback)
    missing_c = probe.check_credentials_and_account(tmp_path / "missing.json", lang="en")
    assert missing_c.status == "fail" and missing_c.details["masked_key"] == "<missing>"
    missing_c_fr = probe.check_credentials_and_account(tmp_path / "missing.json", lang="fr")
    assert missing_c_fr.status == "fail"

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not-json", encoding="utf-8")
    assert probe.check_credentials_and_account(corrupt_path, lang="en").status == "fail"
    assert probe.check_credentials_and_account(corrupt_path, lang="fr").status == "fail"

    bad_key_path = tmp_path / "bad_key.json"
    bad_key_path.write_text('{"api_key": "sk-wrongprefix123456"}', encoding="utf-8")
    bad_k = probe.check_credentials_and_account(bad_key_path, lang="en")
    assert bad_k.status == "fail" and bad_k.details["masked_key"] == "<invalid>"
    assert probe.check_credentials_and_account(bad_key_path, lang="fr").status == "fail"
    assert probe.mask_api_key(123) == "<missing>"

    valid_cred = _write_creds(tmp_path / "valid_creds.json")
    me_fail_en = probe.check_credentials_and_account(
        valid_cred,
        verify_account=True,
        http_fn=_mock_http_healthy(me_status=401),
        lang="en",
    )
    assert me_fail_en.status == "fail" and me_fail_en.details["verify_http_status"] == 401
    me_fail_fr = probe.check_credentials_and_account(
        valid_cred,
        verify_account=True,
        http_fn=_mock_http_healthy(me_status=403),
        lang="fr",
    )
    assert me_fail_fr.status == "fail"

    # Cross-platform home redaction regex fallback
    win_raw = r"C:\Users\SecretUser\.hivemind\credentials.json"
    assert probe.redact_home_path(win_raw, home=tmp_path / "other") == "~/.hivemind/credentials.json"

    # 3. Continue.dev YAML edge cases (missing, incomplete, unreadable)
    miss_yaml = probe.check_continue_provider_yaml(tmp_path / "no_continue.yaml", lang="en")
    assert miss_yaml.status == "warn" and miss_yaml.details["present"] is False
    assert probe.check_continue_provider_yaml(tmp_path / "no_continue.yaml", lang="fr").status == "warn"

    inc_yaml = tmp_path / "incomplete.yaml"
    inc_yaml.write_text("name: empty\n", encoding="utf-8")
    assert probe.check_continue_provider_yaml(inc_yaml, lang="en").status == "warn"
    assert probe.check_continue_provider_yaml(inc_yaml, lang="fr").status == "warn"

    # Full probe with failed credentials returns exit code 2; warnings return exit code 1
    assert (
        probe.main(
            ["--credentials-file", str(tmp_path / "missing.json"), "--json"],
            http_fn=_mock_http_healthy(),
            runner=_mock_runner(),
            environ={"PATH": "/usr/bin"},
            home=tmp_path,
        )
        == 2
    )
    assert (
        probe.main(
            [
                "--credentials-file",
                str(valid_cred),
                "--continue-config-path",
                str(tmp_path / "no_continue.yaml"),
                "--json",
            ],
            http_fn=_mock_http_healthy(),
            runner=_mock_runner(),
            environ={"PATH": "/usr/bin"},
            home=tmp_path,
            os_name="nt",
        )
        == 1
    )


def test_loopback_stdlib_http_subprocess_and_live_router_app_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Verify against real FastAPI create_router_app via TestClient adapter (proves schema parity!)
    store = RouterStore(tmp_path / "router.db", secret=b"slice-dyn-01-router-secret-32b!!")
    client = TestClient(create_router_app(store))
    _, issued_key = store.register("noel-tester", "prosumer")

    def _asgi_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        resp = client.request(method, path, json=payload, headers=headers)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json()
        return resp.status_code, {"content_type": ctype, "html_length": len(resp.text)}

    cred_file = _write_creds(tmp_path / "home" / "u" / ".hivemind" / "credentials.json", api_key=issued_key)
    cont_file = _write_continue_yaml(tmp_path / "home" / "u" / ".continue" / "hivemind_provider.yaml", api_key=issued_key)

    rc_asgi_fb = probe.main(
        [
            "--gateway",
            "http://127.0.0.1:8000",
            "--submit-feedback",
            "--profile",
            "gamer",
            "--ease-rating",
            "5",
            "--json",
        ],
        http_fn=_asgi_http,
    )
    assert rc_asgi_fb == 0

    rc_asgi_probe = probe.main(
        [
            "--gateway",
            "http://127.0.0.1:8000",
            "--credentials-file",
            str(cred_file),
            "--continue-config-path",
            str(cont_file),
            "--verify-account",
            "--json",
        ],
        http_fn=_asgi_http,
        runner=_mock_runner(smi_exc=FileNotFoundError("dev laptop")),
        environ={"PATH": "/usr/bin"},
        home=tmp_path / "home" / "u",
        os_name="posix",
    )
    assert rc_asgi_probe == 0

    # 2. Exercise default_http_json, ProxyHandler({}), default_subprocess_runner, and CLI __main__
    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/health", "/v1/models", "/v1/hivemind/onboarding/feedback/summary"):
                doc: dict[str, Any] = {"status": "ok"}
                if self.path == "/v1/models":
                    doc = {"data": [{"id": "hivemind/coder"}]}
                elif self.path.endswith("/summary"):
                    doc = {"total_responses": 2, "mean_ease_rating": 4.5, "zero_game_lag_ratio": 1.0}
                raw = json.dumps(doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
            elif self.path in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<!DOCTYPE html><title>Noel</title>")
            elif self.path == "/empty":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/err-json":
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail":"missing"}')
            else:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"bad gateway")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"total_responses":3,"mean_ease_rating":4.7,"zero_game_lag_ratio":1.0}')

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return None

    server = http.server.HTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Poison ambient proxy env vars to prove ProxyHandler({}) ignores corporate proxies
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        base = f"http://127.0.0.1:{port}"

        c_html, b_html = probe.default_http_json("GET", f"{base}/noel")
        assert c_html == 200 and b_html["html_length"] > 0
        c_empty, _ = probe.default_http_json("GET", f"{base}/empty")
        assert c_empty == 204
        c_404, b_404 = probe.default_http_json("GET", f"{base}/err-json")
        assert c_404 == 404 and b_404["detail"] == "missing"
        c_502, b_502 = probe.default_http_json("GET", f"{base}/err-plain")
        assert c_502 == 502 and "bad gateway" in b_502["error"]
        c_post, b_post = probe.default_http_json(
            "POST",
            f"{base}/v1/hivemind/onboarding/feedback",
            {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
            {"X-Probe-Test": "1"},
        )
        assert c_post == 200 and b_post["mean_ease_rating"] == 4.7

        proc = subprocess.run(
            [sys.executable, str(PROBE_PATH), "--gateway", base, "--smoke-routes", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
        )
        assert proc.returncode == 0
        assert json.loads(proc.stdout)["routes_passed_count"] == 7
    finally:
        server.shutdown()
        server.server_close()

    c_down, b_down = probe.default_http_json("GET", "http://127.0.0.1:1/health")
    assert c_down == 0 and "network_error" in b_down

    rc_sub, out_sub = probe.default_subprocess_runner([sys.executable, "-c", "print('probe-sub-ok')"], 3.0)
    assert rc_sub == 0 and out_sub.strip() == "probe-sub-ok"

    # Double-check regex figure guard compatibility even on our Markdown output
    fig_re = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
    sample_md = probe.render_markdown_report(
        probe.run_post_setup_probe(smoke_only=True, http_fn=_mock_http_healthy())
    )
    assert fig_re.findall(sample_md) == []
