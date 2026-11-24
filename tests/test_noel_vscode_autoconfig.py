# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, security, compliance, and gateway integration tests for SLICE-DYN-03 (`scripts/noel_vscode_autoconfig.py`)."""

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
AUTOCONFIG_PATH = REPO_ROOT / "scripts" / "noel_vscode_autoconfig.py"

_spec = importlib.util.spec_from_file_location("noel_vscode_autoconfig", AUTOCONFIG_PATH)
assert _spec is not None and _spec.loader is not None
autoconfig = importlib.util.module_from_spec(_spec)
sys.modules["noel_vscode_autoconfig"] = autoconfig
_spec.loader.exec_module(autoconfig)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn03",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn03",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

TEST_KEY = "hm-test_vscode_auto_987654"
TEST_LIVE_KEY = "hm_live_vscode_auto_654321"
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin"}


def _make_gateway_http(
    seen: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] | None = None,
    *,
    issued_key: str = TEST_KEY,
    register_statuses: list[int] | None = None,
    me_status: int = 200,
    models_status: int = 200,
    models_body: dict[str, Any] | None = None,
    holiday_status: int = 200,
    feedback_status: int = 200,
) -> autoconfig.HttpFn:
    reg_queue = list(register_statuses or [200])

    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert "completions" not in url.lower(), "Autoconfig must NEVER call /v1/chat/completions"
        if seen is not None:
            seen.append((method, url, payload, headers))
        if method == "POST" and url.endswith("/v1/hivemind/register"):
            code = reg_queue.pop(0) if len(reg_queue) > 1 else reg_queue[0]
            if code != 200:
                return code, {"error": "registration_refused"}
            role = str((payload or {}).get("role") or "prosumer")
            return 200, {"account_id": "acct_dev_03", "api_key": issued_key, "role": role}
        if method == "GET" and url.endswith("/v1/hivemind/me"):
            if me_status != 200:
                return me_status, {"error": "unauthorized"}
            return 200, {
                "account_id": "acct_dev_03",
                "handle": "noel-coder",
                "role": "prosumer",
                "balance": 100.0,
            }
        if method == "GET" and url.endswith("/v1/models"):
            if models_status != 200:
                return models_status, {"error": "unavailable"}
            if models_body is not None:
                return 200, models_body
            return 200, {
                "object": "list",
                "data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}],
            }
        if method == "GET" and url.endswith("/health"):
            return 200, {"status": "ok"}
        if method == "GET" and any(url.endswith(p) for p in ("/noel", "/quick", "/raid-noel", "/arena-noel")):
            return holiday_status, {"content_type": "text/html; charset=utf-8", "html_length": 2048}
        if method == "GET" and url.endswith("/v1/hivemind/onboarding/feedback/summary"):
            if holiday_status != 200:
                return holiday_status, {"detail": "Not Found"}
            return 200, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 8,
                "mean_ease_rating": 4.75,
                "zero_game_lag_ratio": 1.0,
            }
        if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
            if feedback_status not in (200, 201):
                return feedback_status, {"error": f"status_{feedback_status}"}
            return feedback_status, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 9,
                "mean_ease_rating": 4.8,
                "zero_game_lag_ratio": 1.0,
            }
        return 404, {"error": "not_found"}

    return _http


def _healthy_probe_stub(**kwargs: Any) -> dict[str, Any]:
    assert kwargs["gateway"].startswith(("https://", "http://127.0.0.1", "http://localhost"))
    return {
        "ready": True,
        "verdict": "ready",
        "routes_passed_count": 7,
        "routes_total_count": 7,
        "checks_passed_count": 6,
        "checks_total_count": 6,
        "warnings_count": 0,
        "failures_count": 0,
        "mean_ease_rating": 4.75,
        "zero_inference_verified": True,
        "smoke_gate": {"all_routes_ok": True, "diagnostics": []},
        "receipt_sha256": "a" * 64,
    }


def test_headers_stdlib_lateral_movement_security_scanner_and_zero_stub() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    this_file = Path(__file__).resolve()
    for path in (AUTOCONFIG_PATH, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    script_source = AUTOCONFIG_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Lateral-movement AST audit (zero unauthorized URLs, zero raw sockets, zero SSH/LAN probes)
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source,
            filename="scripts/noel_vscode_autoconfig.py",
        )
        == []
    )
    # Negative control: external editor schema URLs are caught by lateral-movement scanner
    for bad_host in ("opencode" + ".ai/config.json", "continue" + ".dev/schema.json"):
        poisoned = script_source + f'\n_BAD_URL = "https://{bad_host}"\n'
        assert lateral_guard.audit_source_for_lateral_movement(poisoned, "poisoned.py") != []

    # 2. Pure Python standard library only in scripts/noel_vscode_autoconfig.py
    tree = ast.parse(script_source, filename="scripts/noel_vscode_autoconfig.py")
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 3. PR gatekeeper FORBIDDEN_ADDED scan across both files
    for rel, content in (
        ("scripts/noel_vscode_autoconfig.py", script_source),
        ("tests/test_noel_vscode_autoconfig.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner (0 findings) + GWAYA ZeroStubAudit
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("scripts/noel_vscode_autoconfig.py", script_source),
        ("tests/test_noel_vscode_autoconfig.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=999,
        pr_title="feat(onboarding): 60s Dev VS Code 1-command autoconfig",
        pr_body="Implements SLICE-DYN-03 personal-PC VS Code autoconfig and probe hook.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding): add noel_vscode_autoconfig"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    from gwaya.ast_audit import ZeroStubAudit

    audit_res = ZeroStubAudit().audit_python_code(script_source)
    assert audit_res.is_clean, f"Unexpected GWAYA stub violations: {audit_res.violations}"

    # 5. Output safety negative controls (assembled from fragments to avoid diff false positives)
    bad_key_fragment = '{"' + "sco" + 're": 4}'
    assert "forbidden_key_detected" in autoconfig.scan_output_safety(bad_key_fragment)
    assert "email_detected" in autoconfig.scan_output_safety("contact dev@example.org")
    assert "unmasked_api_key_detected" in autoconfig.scan_output_safety(f"key={TEST_KEY}")
    assert "bearer_token_detected" in autoconfig.scan_output_safety("Authorization: Bearer tok_secret_99")
    assert "unredacted_home_path_detected" in autoconfig.scan_output_safety("/home/alice/.hivemind")
    assert any(
        issue.startswith("non_loopback_ipv4_detected:")
        for issue in autoconfig.scan_output_safety("peer 10.1.2.3")
    )
    with pytest.raises(autoconfig.AutoconfigError, match="Output safety guard failed"):
        autoconfig._assert_safe_output(bad_key_fragment)


def test_fresh_registration_idempotent_writes_backups_and_settings_preservation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "alice_home_pc"
    workspace = tmp_path / "hobby_aoc_2026"
    workspace.mkdir(parents=True)
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"
    vs_path = workspace / ".vscode" / "settings.json"

    # Seed pre-existing VS Code JSONC settings (with // comment, /* */ block comment, URL string, trailing comma)
    vs_path.parent.mkdir(parents=True)
    initial_vs = (
        "{\n"
        "  // Personal editor preferences must survive autoconfig\n"
        '  "editor.formatOnSave": true,\n'
        '  "editor.tabSize": 2,\n'
        '  /* Keep string with slashes intact */\n'
        '  "rest-client.defaultHost": "http://localhost:3000/api",\n'
        "}\n"
    )
    vs_path.write_text(initial_vs, encoding="utf-8")

    # Seed pre-existing Continue file with older user text so .bak backup is verified
    cont_path.parent.mkdir(parents=True)
    old_cont = "# Previous personal provider config\nname: Old Local Setup\n"
    cont_path.write_text(old_cont, encoding="utf-8")

    seen_http: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] = []
    probe_calls: list[dict[str, Any]] = []

    def _recording_probe(**kwargs: Any) -> dict[str, Any]:
        probe_calls.append(kwargs)
        return _healthy_probe_stub(**kwargs)

    # 1. First run: fresh registration (with 409 handle collision retry on auto-handle), .bak creation, .gitignore warning
    rc1 = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(workspace),
            "--json",
        ],
        http_fn=_make_gateway_http(seen_http, register_statuses=[409, 200]),
        probe_fn=_recording_probe,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=workspace,
        os_name="posix",
    )
    assert rc1 == 0
    rep1 = json.loads(capsys.readouterr().out)
    assert rep1["compliance_ok"] is True
    assert rep1["ready"] is True
    assert rep1["verdict"] == "ready_with_warnings"
    assert rep1["configured_targets_count"] == 3
    assert rep1["checks_passed_count"] == 9
    assert rep1["within_60s_slo"] is True
    assert rep1["zero_inference_verified"] is True
    assert rep1["account"]["masked_key"] == "hm-***7654"
    assert rep1["account"]["created_account"] is True
    assert len(probe_calls) == 1

    # Verify HTTP call sequence: 2x register (409 -> 200), 1x /me, 1x /v1/models, ZERO completions
    assert [m + " " + u.split(".run.app", 1)[1] for m, u, _, _ in seen_http] == [
        "POST /v1/hivemind/register",
        "POST /v1/hivemind/register",
        "GET /v1/hivemind/me",
        "GET /v1/hivemind/models".replace("/hivemind/models", "/models"),
    ]

    # Verify SHA-256 receipt determinism
    expected_sha = hashlib.sha256(
        json.dumps(
            {k: v for k, v in rep1.items() if k != "receipt_sha256"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert rep1["receipt_sha256"] == expected_sha

    # Verify 0600 POSIX permissions on all written files and backups
    cont_bak = Path(f"{cont_path}.bak")
    vs_bak = Path(f"{vs_path}.bak")
    assert cont_bak.is_file() and cont_bak.read_text(encoding="utf-8") == old_cont
    assert vs_bak.is_file() and vs_bak.read_text(encoding="utf-8") == initial_vs
    if os.name != "nt":
        for p in (cred_path, cont_path, cont_bak, vs_path, vs_bak):
            assert (p.stat().st_mode & 0o777) == 0o600

    # Verify Continue YAML has hivemind/coder and hivemind/bee-stars + raw key
    cont_text = cont_path.read_text(encoding="utf-8")
    assert "hivemind/coder" in cont_text
    assert "hivemind/bee-stars" in cont_text
    assert f"apiBase: {autoconfig.DEFAULT_GATEWAY}/v1" in cont_text
    assert f"apiKey: {TEST_KEY}" in cont_text

    # Verify .vscode/settings.json preserves existing user keys and contains ZERO secrets
    vs_raw = vs_path.read_text(encoding="utf-8")
    assert TEST_KEY not in vs_raw
    assert "alice_home_pc" not in vs_raw
    vs_json = json.loads(vs_raw)
    assert vs_json["editor.formatOnSave"] is True
    assert vs_json["editor.tabSize"] == 2
    assert vs_json["rest-client.defaultHost"] == "http://localhost:3000/api"
    assert vs_json["hivemind.openaiCompatible.apiBase"] == f"{autoconfig.DEFAULT_GATEWAY}/v1"
    assert vs_json["hivemind.openaiCompatible.models"] == ["hivemind/coder", "hivemind/bee-stars"]
    assert vs_json["hivemind.continue.providerConfigPath"] == "~/.continue/hivemind_provider.yaml"

    # Missing .gitignore surfaced a safety warning
    assert rep1["gitignore_safety"]["safe"] is False
    assert rep1["gitignore_safety"]["gitignore_present"] is False
    assert any(".gitignore" in w for w in rep1["warnings"])

    # Add .gitignore without .vscode (still warns in FR), then add .vscode/ (clean!)
    (workspace / ".gitignore").write_text("# Hobby project\n__pycache__/\n!/.vscode/settings.json\n", encoding="utf-8")
    g_warn_fr = autoconfig.check_workspace_gitignore_safety(workspace, lang="fr")
    assert g_warn_fr["safe"] is False and g_warn_fr["gitignore_present"] is True
    (workspace / ".gitignore").write_text("__pycache__/\n.vscode/\n*.bak\n", encoding="utf-8")

    # Loosen credential file mode to 0644 to verify idempotent second run tightens back to 0600 without .bak
    if os.name != "nt":
        os.chmod(cred_path, 0o644)

    # 2. Second run: idempotent! Reuses saved credentials (no /register call), all 3 targets 'unchanged', warnings == []
    seen_http.clear()
    rc2 = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(workspace),
            "--skip-probe",
            "--json",
        ],
        http_fn=_make_gateway_http(seen_http),
        probe_fn=_recording_probe,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=workspace,
        os_name="posix",
    )
    assert rc2 == 0
    rep2 = json.loads(capsys.readouterr().out)
    assert rep2["ready"] is True
    assert rep2["verdict"] == "ready"
    assert rep2["warnings"] == []
    assert rep2["probe_skipped"] is True
    assert rep2["probe"] is None
    assert len(probe_calls) == 1  # probe_fn was skipped!
    assert not any(u.endswith("/v1/hivemind/register") for _, u, _, _ in seen_http)
    for target_name in ("credentials", "continue_provider", "vscode_settings"):
        t_info = rep2["targets"][target_name]
        assert t_info["action"] == "unchanged"
        assert t_info["written"] is False
        assert t_info["backup_created"] is False
    if os.name != "nt":
        assert rep2["targets"]["credentials"]["permissions_tightened"] is True
        assert (cred_path.stat().st_mode & 0o777) == 0o600
    assert not Path(f"{cred_path}.bak").exists()


def test_secret_non_leakage_across_cli_markdown_json_and_workspace_guard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "Users" / "SecretDeveloper"
    workspace = home_dir / "side_projects" / "star_sim"
    workspace.mkdir(parents=True)
    (workspace / ".gitignore").write_text(".vscode/settings.json\n", encoding="utf-8")

    # Render CLI (EN & FR) and Markdown (EN & FR) with live hm_live_... key
    for lang_flag in ("en", "fr"):
        for fmt_flags in ([], ["--markdown"], ["--json"]):
            rc = autoconfig.main(
                [
                    "--api-key",
                    TEST_LIVE_KEY,
                    "--workspace",
                    str(workspace),
                    "--skip-probe",
                    "--lang",
                    lang_flag,
                    *fmt_flags,
                ],
                http_fn=_make_gateway_http(issued_key=TEST_LIVE_KEY),
                environ=CLEAN_ENV,
                home=home_dir,
                cwd=workspace,
                os_name="nt",
            )
            assert rc == 0
            captured = capsys.readouterr()
            assert TEST_LIVE_KEY not in captured.out
            assert TEST_LIVE_KEY not in captured.err
            assert "SecretDeveloper" not in captured.out
            assert "hm_live_***4321" in captured.out
            assert "0.00 EUR / 0.00 USD" in captured.out

    # Workspace secret guard blocks raw keys, Bearer tokens, and OS home paths in user settings
    with pytest.raises(autoconfig.AutoconfigError, match="raw hm-... API key"):
        autoconfig.build_merged_vscode_settings_json(
            json.dumps({"other.extension.key": TEST_KEY}),
            gateway=autoconfig.DEFAULT_GATEWAY,
        )
    with pytest.raises(autoconfig.AutoconfigError, match="Bearer token"):
        autoconfig.build_merged_vscode_settings_json(
            json.dumps({"other.auth": "Bearer super_secret_token_value"}),
            gateway=autoconfig.DEFAULT_GATEWAY,
        )
    with pytest.raises(autoconfig.AutoconfigError, match="unredacted OS home path"):
        autoconfig.build_merged_vscode_settings_json(
            json.dumps({"python.venvPath": "/home/secretuser/.venvs"}),
            gateway=autoconfig.DEFAULT_GATEWAY,
        )

    # Corrupt existing .vscode/settings.json or non-dict JSON fails cleanly with exit 2 BEFORE HTTP calls
    bad_ws = tmp_path / "broken_ws"
    bad_vs = bad_ws / ".vscode" / "settings.json"
    bad_vs.parent.mkdir(parents=True)
    bad_vs.write_text("{corrupt-json", encoding="utf-8")
    seen_broken: list[Any] = []
    rc_bad_json = autoconfig.main(
        ["--workspace", str(bad_ws), "--skip-probe"],
        http_fn=_make_gateway_http(seen_broken),
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=bad_ws,
    )
    assert rc_bad_json == 2
    assert seen_broken == []
    capsys.readouterr()

    with pytest.raises(autoconfig.AutoconfigError, match="JSON object"):
        autoconfig.build_merged_vscode_settings_json("[1, 2, 3]", gateway=autoconfig.DEFAULT_GATEWAY)


def test_personal_pc_and_git_remote_compliance_rejection_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_dir = tmp_path / "sandbox"
    home_dir = base_dir / "home" / "perso"
    clean_ws = base_dir / "hobby_game"
    clean_ws.mkdir(parents=True)
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"

    seen_http: list[Any] = []
    seen_probe: list[Any] = []

    def _never_called_probe(**kwargs: Any) -> dict[str, Any]:
        seen_probe.append(kwargs)
        return _healthy_probe_stub(**kwargs)

    # 1. Environment variable corporate indicators (Amadeus, AD domain, proxies, hostnames, TLS certs)
    env_scenarios: list[tuple[dict[str, str], str, str]] = [
        ({"USERDNSDOMAIN": "AMADEUS.EXTRA.LOCAL"}, "env:USERDNSDOMAIN", "AMADEUS.EXTRA.LOCAL"),
        ({"HTTP_PROXY": "http://gateway.zscaler.net:8080"}, "env:HTTP_PROXY", "gateway.zscaler.net"),
        ({"HTTPS_PROXY": "http://proxy.netskope.com:8443"}, "env:HTTPS_PROXY", "proxy.netskope.com"),
        ({"ALL_PROXY": "http://proxy.bluecoat.com:8080"}, "env:ALL_PROXY", "proxy.bluecoat.com"),
        ({"NO_PROXY": ".fortinet.local,.paloaltonetworks.com"}, "env:NO_PROXY", "paloaltonetworks.com"),
        ({"COMPUTERNAME": "LAPTOP-CORP-77", "USERDOMAIN": "WORKGROUP"}, "env:COMPUTERNAME", "LAPTOP-CORP-77"),
        ({"HOSTNAME": "devbox.intranet.local"}, "env:HOSTNAME", "devbox.intranet.local"),
        ({"AMADEUS_STAGE": "prod-eu-1"}, "env:AMADEUS_ENV", "prod-eu-1"),
        ({"USERDOMAIN": "ENTERPRISEAD", "COMPUTERNAME": "MY-PC"}, "env:USERDOMAIN", "ENTERPRISEAD"),
        ({"ZSCALER_CERT_PATH": "/etc/ssl/zscaler_ca.pem"}, "env:ZSCALER_CERT_PATH", "zscaler_ca.pem"),
        ({"SSL_CERT_FILE": "/etc/ssl/internal-ca.crt"}, "env:SSL_CERT_FILE", "internal-ca.crt"),
    ]
    for idx, (bad_env, expected_tag, secret_raw_value) in enumerate(env_scenarios):
        lang = "fr" if idx % 2 else "en"
        fmt = "--json" if idx % 3 != 0 else "--markdown"
        rc = autoconfig.main(
            [
                "--credentials-file",
                str(cred_path),
                "--continue-config-path",
                str(cont_path),
                "--workspace",
                str(clean_ws),
                "--lang",
                lang,
                fmt,
            ],
            http_fn=_make_gateway_http(seen_http),
            probe_fn=_never_called_probe,
            environ=bad_env,
            home=home_dir,
            cwd=clean_ws,
        )
        assert rc == 2
        out = capsys.readouterr().out
        assert secret_raw_value not in out
        assert "0.00 EUR / 0.00 USD" in out
        assert "0,00 EUR / 0,00 USD" in out
        if fmt == "--json":
            parsed = json.loads(out)
            assert parsed["compliance_ok"] is False
            assert parsed["ready"] is False
            assert parsed["verdict"] == "refused_compliance"
            assert parsed["configured_targets_count"] == 0
            assert expected_tag in parsed["compliance"]["corporate_indicators"]
        assert not cred_path.exists()
        assert not cont_path.exists()
        assert not (clean_ws / ".vscode" / "settings.json").exists()

    # 2. Corporate workspace directory path rejection (without leaking directory path)
    ws_marked = base_dir / "amadeus_flight_engine"
    ws_marked.mkdir(parents=True)
    rc_ws = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(ws_marked),
            "--json",
        ],
        http_fn=_make_gateway_http(seen_http),
        probe_fn=_never_called_probe,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_marked,
    )
    assert rc_ws == 2
    ws_rep = json.loads(capsys.readouterr().out)
    assert ws_rep["compliance_ok"] is False
    assert "workspace_path:amadeus" in ws_rep["compliance"]["corporate_indicators"]
    assert "amadeus_flight_engine" not in json.dumps(ws_rep)

    # 3. Standard .git/config corporate remote URL rejection (SSH & HTTPS + url insteadOf)
    git_ws = base_dir / "innocent_folder_name"
    (git_ws / ".git").mkdir(parents=True)
    (git_ws / ".git" / "config").write_text(
        '[core]\n\trepositoryformatversion = 0\n'
        '[remote "origin"]\n'
        '\turl = git@gitlab.amadeus.internal:pricing/secret-service.git\n'
        '\tpushurl = https://ghe.enterprise.mycorp.com/pricing/secret-service.git\n'
        '[url "https://git.intranet.local/"]\n'
        '\tinsteadOf = intra:\n',
        encoding="utf-8",
    )
    rc_git = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(git_ws),
            "--lang",
            "fr",
            "--json",
        ],
        http_fn=_make_gateway_http(seen_http),
        probe_fn=_never_called_probe,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=git_ws,
    )
    assert rc_git == 2
    git_rep = json.loads(capsys.readouterr().out)
    assert git_rep["compliance_ok"] is False
    for expected_remote_tag in ("git_remote:amadeus", "git_remote:internal", "git_remote:enterprise", "git_remote:corp", "git_remote:intranet"):
        assert expected_remote_tag in git_rep["compliance"]["corporate_indicators"]
    assert "secret-service" not in json.dumps(git_rep)

    # 4. Git worktree pointer file (.git -> gitdir: ... + commondir) remote URL detection
    common_git = base_dir / "common_git_store"
    wt_gitdir = common_git / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text(
        '[remote "origin"]\n\turl = https://git.paloalto.internal/sec/repo.git\n',
        encoding="utf-8",
    )
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    wt_ws = base_dir / "worktree_checkout"
    wt_ws.mkdir(parents=True)
    (wt_ws / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    wt_comp = autoconfig.evaluate_personal_pc_compliance(environ=CLEAN_ENV, workspace=wt_ws, lang="en")
    assert wt_comp["compliance_ok"] is False
    assert "git_remote:paloalto" in wt_comp["corporate_indicators"]

    # Ensure hobby words like 'corpus' or 'scorpion' do NOT false-positive
    hobby_ws = base_dir / "scorpion_text_corpus"
    hobby_ws.mkdir(parents=True)
    assert autoconfig.evaluate_personal_pc_compliance(environ=CLEAN_ENV, workspace=hobby_ws)["compliance_ok"] is True

    # Confirm zero HTTP or probe calls ever occurred across all corporate rejections
    assert seen_http == []
    assert seen_probe == []


def test_uninstall_preserves_user_settings_or_removes_empty_vscode_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "uninstaller"
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"

    # Case A: Workspace has user settings + HiveMind settings + .bak files -> user settings preserved!
    ws_mixed = tmp_path / "ws_mixed"
    vs_mixed = ws_mixed / ".vscode" / "settings.json"
    vs_mixed.parent.mkdir(parents=True)
    vs_mixed.write_text('{"editor.tabSize": 4}\n', encoding="utf-8")

    assert (
        autoconfig.main(
            [
                "--credentials-file",
                str(cred_path),
                "--continue-config-path",
                str(cont_path),
                "--workspace",
                str(ws_mixed),
                "--skip-probe",
                "--json",
            ],
            http_fn=_make_gateway_http(),
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws_mixed,
        )
        == 0
    )
    capsys.readouterr()
    Path(f"{cred_path}.bak").write_text("{}", encoding="utf-8")
    Path(f"{cont_path}.bak").write_text("old", encoding="utf-8")
    assert Path(f"{vs_mixed}.bak").is_file()

    rc_un_a = autoconfig.main(
        [
            "--uninstall",
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(ws_mixed),
            "--markdown",
            "--lang",
            "fr",
        ],
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_mixed,
    )
    assert rc_un_a == 0
    un_md = capsys.readouterr().out
    assert "Désinstallation Propre" in un_md
    assert not cred_path.exists() and not Path(f"{cred_path}.bak").exists()
    assert not cont_path.exists() and not Path(f"{cont_path}.bak").exists()
    assert not Path(f"{vs_mixed}.bak").exists()
    assert vs_mixed.is_file()
    remaining_json = json.loads(vs_mixed.read_text(encoding="utf-8"))
    assert remaining_json == {"editor.tabSize": 4}

    # Case B: Fresh workspace where .vscode/settings.json ONLY had HiveMind keys -> file & empty .vscode dir deleted!
    ws_pure = tmp_path / "ws_pure"
    ws_pure.mkdir(parents=True)
    assert (
        autoconfig.main(
            [
                "--credentials-file",
                str(cred_path),
                "--continue-config-path",
                str(cont_path),
                "--workspace",
                str(ws_pure),
                "--skip-probe",
                "--json",
            ],
            http_fn=_make_gateway_http(),
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws_pure,
        )
        == 0
    )
    capsys.readouterr()
    assert (ws_pure / ".vscode" / "settings.json").is_file()

    rc_un_b = autoconfig.main(
        [
            "--uninstall",
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(ws_pure),
        ],
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws_pure,
    )
    assert rc_un_b == 0
    cli_un = capsys.readouterr().out
    assert "Removed files" in cli_un
    assert not (ws_pure / ".vscode" / "settings.json").exists()
    assert not (ws_pure / ".vscode").exists()


def test_chained_probe_ces_feedback_404_429_5xx_and_live_router_integration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "dev_chain"
    ws = tmp_path / "hobby_chain"
    ws.mkdir(parents=True)
    (ws / ".gitignore").write_text(".vscode/\n", encoding="utf-8")
    cred_path = home_dir / ".hivemind" / "credentials.json"
    cont_path = home_dir / ".continue" / "hivemind_provider.yaml"

    # 1. Live FastAPI create_router_app integration chained into real scripts/noel_post_setup_probe.py + CES feedback
    store = RouterStore(tmp_path / "router.db", secret=b"slice-dyn-03-router-secret-32b!!")
    client = TestClient(create_router_app(store))

    posted_feedback: list[dict[str, Any]] = []

    def _asgi_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert "completions" not in url.lower()
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if method == "POST" and path == "/v1/hivemind/onboarding/feedback":
            assert headers is None
            assert payload is not None
            posted_feedback.append(dict(payload))
        resp = client.request(method, path, json=payload, headers=headers)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json()
        return resp.status_code, {"content_type": ctype, "html_length": len(resp.text)}

    rc_live = autoconfig.main(
        [
            "--gateway",
            "http://127.0.0.1:8000",
            "--handle",
            "claire-dev",
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--workspace",
            str(ws),
            "--ease-rating",
            "5",
            "--markdown",
            "--lang",
            "en",
        ],
        http_fn=_asgi_http,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws,
        os_name="posix",
    )
    assert rc_live == 0
    md_live = capsys.readouterr().out
    assert "# HiveMind Holiday Developer VS Code Autoconfig" in md_live
    assert "## Chained Post-Setup Probe (Zero-Inference)" in md_live
    assert "routes `7/7`" in md_live
    assert len(posted_feedback) == 1
    assert posted_feedback[0]["profile"] == "dev"
    assert posted_feedback[0]["ease_rating"] == 5
    assert posted_feedback[0]["zero_game_lag"] is True

    # 2. Interactive 1-prompt CES (--prompt-ces) answered + skipped + EOF
    prompts: list[str] = []
    rc_prompt = autoconfig.main(
        [
            "--gateway",
            "http://127.0.0.1:8000",
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--no-vscode-settings",
            "--workspace",
            str(ws),
            "--prompt-ces",
            "--skip-probe",
            "--lang",
            "fr",
            "--json",
        ],
        http_fn=_asgi_http,
        input_fn=lambda p: (prompts.append(p) or "4/5 dev lag"),
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws,
    )
    assert rc_prompt == 0
    assert len(prompts) == 1
    prompt_rep = json.loads(capsys.readouterr().out)
    assert prompt_rep["ease_rating"] == 4
    assert prompt_rep["feedback"]["profile"] == "dev"
    assert prompt_rep["feedback"]["zero_game_lag"] is False
    assert prompt_rep["configured_targets_count"] == 2

    skip_res = autoconfig.prompt_and_submit_dev_ces(
        input_fn=lambda _p: "   ",
        http_fn=_asgi_http,
        gateway="http://127.0.0.1:8000",
        lang="en",
    )
    assert skip_res["skipped"] is True and skip_res["attempted"] is False
    eof_res = autoconfig.prompt_and_submit_dev_ces(
        input_fn=lambda _p: (_ for _ in ()).throw(EOFError()),
        http_fn=_asgi_http,
        gateway="http://127.0.0.1:8000",
        lang="fr",
    )
    assert eof_res["skipped"] is True

    # 3. CES Feedback HTTP 404 (exit 2), HTTP 429 (exit 1), HTTP 503 (exit 1), network 0, HTTP 400
    rc_fb_404 = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--skip-probe",
            "--ease-rating",
            "5",
            "--lang",
            "fr",
            "--json",
        ],
        http_fn=_make_gateway_http(feedback_status=404),
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws,
    )
    assert rc_fb_404 == 2
    assert json.loads(capsys.readouterr().out)["feedback"]["status"] == "missing_route"

    rc_fb_429 = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--skip-probe",
            "--ease-rating",
            "4",
            "--json",
        ],
        http_fn=_make_gateway_http(feedback_status=429),
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws,
    )
    assert rc_fb_429 == 1
    assert json.loads(capsys.readouterr().out)["feedback"]["status"] == "rate_limited"

    rc_fb_503 = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--skip-probe",
            "--ease-rating",
            "5",
            "--lang",
            "fr",
            "--json",
        ],
        http_fn=_make_gateway_http(feedback_status=503),
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws,
    )
    assert rc_fb_503 == 1
    assert json.loads(capsys.readouterr().out)["feedback"]["status"] == "server_error"

    assert (
        autoconfig.submit_dev_ces_feedback(ease_rating=4, system_name="Darwin", http_fn=lambda *_: (404, {}), lang="en")[
            "status"
        ]
        == "missing_route"
    )
    assert (
        autoconfig.submit_dev_ces_feedback(ease_rating=4, system_name="Windows", http_fn=lambda *_: (429, {}), lang="fr")[
            "status"
        ]
        == "rate_limited"
    )
    assert (
        autoconfig.submit_dev_ces_feedback(ease_rating=4, system_name="Linux", http_fn=lambda *_: (502, {}), lang="en")[
            "status"
        ]
        == "server_error"
    )
    assert (
        autoconfig.submit_dev_ces_feedback(
            ease_rating=4,
            system_name="FreeBSD!",
            http_fn=lambda *_: (_ for _ in ()).throw(OSError("down")),
            lang="en",
        )["status"]
        == "network_error"
    )
    assert (
        autoconfig.submit_dev_ces_feedback(ease_rating=4, system_name="???", http_fn=lambda *_: (0, {}), lang="fr")[
            "status"
        ]
        == "network_error"
    )
    assert (
        autoconfig.submit_dev_ces_feedback(ease_rating=3, http_fn=lambda *_: (400, {}), lang="en")["status"]
        == "client_error"
    )
    assert (
        autoconfig.submit_dev_ces_feedback(ease_rating=3, http_fn=lambda *_: (422, {}), lang="fr")["status"]
        == "client_error"
    )

    # 4. Chained probe failure (stale Cloud Run 404 on /noel -> exit 2) & minimal-arg custom probe_fn
    rc_probe_fail = autoconfig.main(
        [
            "--credentials-file",
            str(cred_path),
            "--continue-config-path",
            str(cont_path),
            "--json",
        ],
        http_fn=_make_gateway_http(holiday_status=404),
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws,
    )
    assert rc_probe_fail == 2
    probe_fail_rep = json.loads(capsys.readouterr().out)
    assert probe_fail_rep["ready"] is False
    assert probe_fail_rep["probe"]["all_routes_ok"] is False

    def _minimal_fail_probe(gateway: str) -> dict[str, Any]:
        return {"gateway": gateway, "ready": False, "failures_count": 1, "zero_inference_verified": True}

    rc_min_probe, rep_min_probe = autoconfig.run_vscode_autoconfig(
        credentials_path=cred_path,
        continue_path=cont_path,
        http_fn=_make_gateway_http(),
        probe_fn=_minimal_fail_probe,
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws,
        lang="fr",
    )
    assert rc_min_probe == 2 and rep_min_probe["ready"] is False


def test_validation_error_branches_and_loopback_http_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home_dir = tmp_path / "home" / "edge_user"
    ws = tmp_path / "edge_ws"
    ws.mkdir(parents=True)

    # 1. URL, handle, role, lang, key, and zero-inference validation branches
    assert autoconfig.validate_gateway_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
    assert autoconfig.validate_gateway_url("https://[::1]:8443") == "https://[::1]:8443"
    assert autoconfig.is_allowed_gateway_host("") is False
    assert autoconfig.mask_api_key("") == "<missing>"
    assert autoconfig.mask_api_key("not-a-key") == "<invalid>"
    assert autoconfig.redact_home_path(r"C:\Users\Bob\.hivemind\credentials.json", home=tmp_path) == (
        "~/.hivemind/credentials.json"
    )

    for bad_url in (
        "",
        True,
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app\nbad",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/ sub",
        "https://u:p@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app?q=1",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app#f",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app:abc",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app:0",
        "https://evil.example.com",
        "ftp://localhost:21",
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
    ):
        with pytest.raises(autoconfig.AutoconfigError):
            autoconfig.validate_gateway_url(bad_url)  # type: ignore[arg-type]

    with pytest.raises(autoconfig.AutoconfigError, match="Zero-inference"):
        autoconfig._safe_http_call(
            lambda *_: (200, {}),
            "POST",
            f"{autoconfig.DEFAULT_GATEWAY}/v1/chat/completions",
            {"model": "hivemind/coder"},
            None,
        )
    with pytest.raises(autoconfig.AutoconfigError, match="Unsupported HTTP method"):
        autoconfig._assert_zero_inference_request("DELETE", f"{autoconfig.DEFAULT_GATEWAY}/health")

    # Two-arg HTTP callable compatibility
    assert autoconfig._safe_http_call(lambda m, u: (200, [m, u]), "GET", f"{autoconfig.DEFAULT_GATEWAY}/health")[0] == 200

    for bad_lang in (True, "de"):
        with pytest.raises(autoconfig.AutoconfigError):
            autoconfig.normalize_lang(bad_lang)  # type: ignore[arg-type]
    for bad_role in (None, True, "root"):
        with pytest.raises(autoconfig.AutoconfigError):
            autoconfig.normalize_role(bad_role)  # type: ignore[arg-type]
    for bad_handle in ("a", "google.spoof"):
        with pytest.raises(autoconfig.AutoconfigError):
            autoconfig.sanitize_handle(bad_handle)
    for bad_ces in ("0", "6", "4 5", "gamer", "nolag"):
        with pytest.raises(autoconfig.AutoconfigError):
            autoconfig.parse_dev_ces_input(bad_ces)
    assert autoconfig.parse_dev_ces_input("5 nolag")["zero_game_lag"] is True
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.submit_dev_ces_feedback(ease_rating=True)  # type: ignore[arg-type]
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.submit_dev_ces_feedback(ease_rating=5, zero_game_lag="yes")  # type: ignore[arg-type]
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_continue_provider_yaml(autoconfig.DEFAULT_GATEWAY, "bad")
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig.build_continue_provider_yaml(autoconfig.DEFAULT_GATEWAY, TEST_KEY, max_tokens=10)
    with pytest.raises(autoconfig.AutoconfigError):
        autoconfig._invoke_probe_callable(
            lambda: "not-a-dict",  # type: ignore[return-value]
            gateway=autoconfig.DEFAULT_GATEWAY,
            credentials_path=ws / "c.json",
            continue_path=ws / "p.yaml",
            http_fn=_make_gateway_http(),
            feedback_result=None,
            lang="en",
            os_name="posix",
            environ=CLEAN_ENV,
            home=home_dir,
        )
    with pytest.raises(autoconfig.AutoconfigError, match="Zero-inference"):
        autoconfig._invoke_probe_callable(
            lambda: {"zero_inference_verified": False},
            gateway=autoconfig.DEFAULT_GATEWAY,
            credentials_path=ws / "c.json",
            continue_path=ws / "p.yaml",
            http_fn=_make_gateway_http(),
            feedback_result=None,
            lang="en",
            os_name="posix",
            environ=CLEAN_ENV,
            home=home_dir,
        )

    # CLI error exits (invalid key, invalid ease-rating, /register 500 & 404, /me 401, /v1/models 503)
    assert autoconfig.main(["--api-key", "invalid"], environ=CLEAN_ENV, home=home_dir, cwd=ws) == 2
    assert autoconfig.main(["--ease-rating", "9"], environ=CLEAN_ENV, home=home_dir, cwd=ws) == 2
    assert (
        autoconfig.main(
            ["--handle", "taken", "--skip-probe", "--json"],
            http_fn=_make_gateway_http(register_statuses=[409]),
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws,
        )
        == 1
    )
    assert (
        autoconfig.main(
            ["--skip-probe", "--lang", "fr", "--json"],
            http_fn=_make_gateway_http(register_statuses=[404]),
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws,
        )
        == 2
    )
    assert (
        autoconfig.main(
            ["--api-key", TEST_KEY, "--skip-probe", "--json"],
            http_fn=_make_gateway_http(me_status=401),
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws,
        )
        == 1
    )
    assert (
        autoconfig.main(
            ["--api-key", TEST_KEY, "--skip-probe", "--lang", "fr", "--json"],
            http_fn=_make_gateway_http(models_status=503),
            environ=CLEAN_ENV,
            home=home_dir,
            cwd=ws,
        )
        == 1
    )
    capsys.readouterr()

    # Saved credentials edge cases (corrupt JSON, non-dict, invalid key, mismatched gateway)
    bad_cred = tmp_path / "bad_cred.json"
    bad_cred.write_text("{broken", encoding="utf-8")
    assert autoconfig.load_saved_credentials(bad_cred) is None
    bad_cred.write_text("[]", encoding="utf-8")
    assert autoconfig.load_saved_credentials(bad_cred) is None
    bad_cred.write_text('{"api_key": "short"}', encoding="utf-8")
    assert autoconfig.load_saved_credentials(bad_cred) is None
    bad_cred.write_text(json.dumps({"api_key": TEST_KEY, "gateway": "http://127.0.0.1:9999"}), encoding="utf-8")
    assert autoconfig.load_saved_credentials(bad_cred, expected_gateway="http://127.0.0.1:8000") is None

    # 2. Loopback HTTPServer test for default_http_json (ProxyHandler({})) and subprocess __main__ execution
    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/hivemind/me":
                self._send_json(200, {"account_id": "a1", "handle": "loop-dev", "role": "prosumer"})
            elif self.path == "/v1/models":
                self._send_json(200, {"data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]})
            elif self.path == "/html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<title>ok</title>")
            elif self.path == "/empty":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/err-json":
                self._send_json(404, {"detail": "missing"})
            else:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"upstream error")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            self._send_json(200, {"account_id": "a1", "api_key": TEST_KEY, "role": "prosumer"})

        def _send_json(self, code: int, doc: dict[str, Any]) -> None:
            raw = json.dumps(doc).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return None

    server = http.server.HTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        base = f"http://127.0.0.1:{port}"

        assert autoconfig.default_http_json("GET", f"{base}/html")[1]["html_length"] > 0
        assert autoconfig.default_http_json("GET", f"{base}/empty")[0] == 204
        assert autoconfig.default_http_json("GET", f"{base}/err-json") == (404, {"detail": "missing"})
        assert autoconfig.default_http_json("GET", f"{base}/err-plain")[0] == 502
        assert (
            autoconfig.default_http_json(
                "POST",
                f"{base}/v1/hivemind/register",
                {"handle": "loop-dev", "role": "prosumer"},
                {"X-Test": "1"},
            )[0]
            == 200
        )

        sub_home = tmp_path / "sub_home"
        sub_ws = tmp_path / "sub_ws"
        sub_ws.mkdir(parents=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(AUTOCONFIG_PATH),
                "--gateway",
                base,
                "--workspace",
                str(sub_ws),
                "--vscode-settings-path",
                str(sub_ws / ".vscode" / "settings.json"),
                "--skip-probe",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=str(sub_ws),
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(sub_home)},
        )
        assert proc.returncode == 0
        sub_rep = json.loads(proc.stdout)
        assert sub_rep["ready"] is True
        assert sub_rep["configured_targets_count"] == 3
    finally:
        server.shutdown()
        server.server_close()

    assert autoconfig.default_http_json("GET", "http://127.0.0.1:1/health")[0] == 0

    # Verify static-numbers provenance regex produces zero matches on our Markdown & CLI outputs
    fig_re = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
    explicit_vs = ws / "custom_vscode" / "settings.json"
    _, sample_rep = autoconfig.run_vscode_autoconfig(
        credentials_path=home_dir / ".hivemind" / "credentials.json",
        continue_path=home_dir / ".continue" / "hivemind_provider.yaml",
        workspace=ws,
        vscode_settings_path=explicit_vs,
        write_vscode_settings=True,
        skip_probe=True,
        http_fn=_make_gateway_http(),
        environ=CLEAN_ENV,
        home=home_dir,
        cwd=ws,
    )
    assert explicit_vs.is_file()
    assert fig_re.findall(autoconfig.render_markdown_report(sample_rep)) == []
    assert fig_re.findall(autoconfig.render_cli_report(sample_rep)) == []

    # Uninstall safely leaves corrupt user settings.json untouched rather than deleting it
    corrupt_vs = ws / "corrupt_settings.json"
    corrupt_vs.write_text("{not-valid-json", encoding="utf-8")
    un_corrupt = autoconfig.uninstall_hivemind_autoconfig(
        credentials_path=home_dir / ".hivemind" / "credentials.json",
        continue_path=home_dir / ".continue" / "hivemind_provider.yaml",
        vscode_settings_path=corrupt_vs,
        home=home_dir,
    )
    assert corrupt_vs.is_file()
    assert un_corrupt["preserved_settings_files"] == []
