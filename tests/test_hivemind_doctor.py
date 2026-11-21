# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, security, cross-platform, and gateway-serving tests for SLICE-03 (scripts/hivemind_doctor.py)."""

from __future__ import annotations

import ast
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import _ALLOWED_ONBOARDING_SCRIPTS, create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = REPO_ROOT / "scripts" / "hivemind_doctor.py"
GATEWAY_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "gateway.py"

_spec = importlib.util.spec_from_file_location("hivemind_doctor", DOCTOR_PATH)
assert _spec is not None and _spec.loader is not None
doctor = importlib.util.module_from_spec(_spec)
sys.modules["hivemind_doctor"] = doctor
_spec.loader.exec_module(doctor)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_mod",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

TEST_KEY = "hm-test_doctor_alpha_12345"
TEST_LIVE_KEY = "hm_live_doctor_key_9876"


def _write_creds(path: Path, api_key: str = TEST_KEY, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "account_id": "acct_canary_1",
                "api_key": api_key,
                "gateway": doctor.DEFAULT_GATEWAY,
                "handle": "canary-friend",
                "role": "prosumer",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        os.chmod(path, mode)
    return path


def _mock_http_ok(seen: list[str] | None = None) -> doctor.HttpGetFn:
    def _http(url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
        assert 0.5 <= timeout_s <= 4.5
        if seen is not None:
            seen.append(url)
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {
                "object": "list",
                "data": [{"id": "hivemind/auto"}, {"id": "hivemind/coder"}],
            }
        return 404, {"error": "not_found"}

    return _http


def _mock_runner(
    *,
    smi_rc: int = 0,
    smi_out: str = "NVIDIA GeForce RTX 4070, 12288, 10240, 54, 4\n",
    smi_exc: Exception | None = None,
    ps_rc: int = 0,
    ps_out: str = "systemd\nbash\npython3\n",
    tasklist_rc: int = 0,
    tasklist_out: str = '"explorer.exe","1200","Console","1","85,000 K"\n',
    proc_exc: Exception | None = None,
) -> doctor.RunnerFn:
    def _run(argv: Any, timeout_s: float) -> tuple[int, str]:
        assert timeout_s <= 5.0
        cmd0 = str(argv[0])
        if cmd0 == "nvidia-smi":
            assert "uuid" not in " ".join(argv).lower()
            assert "serial" not in " ".join(argv).lower()
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


def _by_id(report: dict[str, Any], check_id: str) -> dict[str, Any]:
    return next(item for item in report["checks"] if item["id"] == check_id)


def test_compliance_headers_stdlib_only_and_zero_lateral_movement() -> None:
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    for target in (DOCTOR_PATH, GATEWAY_PATH, Path(__file__).resolve()):
        head = target.read_text(encoding="utf-8")[:300]
        assert expected_header in head, f"Missing AGPL-3.0 header in {target}"

    doctor_source = DOCTOR_PATH.read_text(encoding="utf-8")

    # Verify zero lateral-movement / unauthorized outbound hosts via repository AST guard
    violations = lateral_guard.audit_source_for_lateral_movement(
        doctor_source,
        filename="scripts/hivemind_doctor.py",
    )
    assert violations == []

    # Verify ONLY Python standard-library modules are imported
    tree = ast.parse(doctor_source, filename="scripts/hivemind_doctor.py")
    stdlib_names = sys.stdlib_module_names
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    non_stdlib = imported_roots - stdlib_names
    assert non_stdlib == set(), f"Unexpected third-party imports in hivemind_doctor.py: {non_stdlib}"

    # Verify 5-Layer PRSecurityScanner passes cleanly on modified/created files
    scanner = PRSecurityScanner()
    for rel, path in (
        ("scripts/hivemind_doctor.py", DOCTOR_PATH),
        ("tests/test_hivemind_doctor.py", Path(__file__).resolve()),
    ):
        text = path.read_text(encoding="utf-8")
        assert scanner.scan_python_ast_security(rel, text) == []
        assert scanner.scan_text_for_payment_solicitation(text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text, rel) == []
        assert scanner.scan_unicode_trojan_source(text, rel) == []


def test_linux_gamer_whisper_ready_and_json_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home_dir = tmp_path / "home" / "alice_secret_user"
    cred_path = _write_creds(home_dir / ".hivemind" / "credentials.json", mode=0o600)
    seen_urls: list[str] = []

    report = doctor.run_doctor(
        gateway=doctor.DEFAULT_GATEWAY,
        credentials_path=cred_path,
        lang="en",
        version_info=(3, 12, 4),
        os_name="posix",
        system_name="Linux",
        environ={"HOME": str(home_dir), "PATH": "/usr/bin"},
        home=home_dir,
        http_get=_mock_http_ok(seen_urls),
        runner=_mock_runner(),
    )

    assert report["schema"] == "hivemind.doctor/1"
    assert report["ready"] is True
    assert report["verdict"] == "ready"
    assert report["recommended_mode"] == "gamer_whisper"
    assert report["readiness_checks_passed"] == 6
    assert report["readiness_checks_total"] == 6
    assert report["warnings_count"] == 0
    assert report["failures_count"] == 0
    assert report["elapsed_s"] < 10.0
    assert report["fix_commands"] == []
    assert "0.00 EUR / 0.00 USD" in report["compliance_notice"]

    serialized = json.dumps(report)
    assert "alice_secret_user" not in serialized
    assert TEST_KEY not in serialized
    assert '"score"' not in serialized
    assert seen_urls == [
        f"{doctor.DEFAULT_GATEWAY}/health",
        f"{doctor.DEFAULT_GATEWAY}/v1/models",
    ]

    # CLI --json exit 0
    rc = doctor.main(
        ["--credentials-file", str(cred_path), "--lang", "en", "--json"],
        http_get=_mock_http_ok(),
        runner=_mock_runner(),
        environ={"PATH": "/usr/bin"},
    )
    assert rc == 0
    cli_json = json.loads(capsys.readouterr().out)
    assert cli_json["ready"] is True
    assert cli_json["readiness_checks_passed"] == 6


def test_windows_11_scenarios_acl_tasklist_and_fixes(tmp_path: Path) -> None:
    win_home = tmp_path / "Users" / "WinGamer"
    cred_path = _write_creds(win_home / ".hivemind" / "credentials.json", api_key=TEST_LIVE_KEY, mode=0o644)

    # On Windows (os_name='nt'), POSIX 0600 bits do not apply; valid credentials pass with windows_acl
    win_ok = doctor.run_doctor(
        credentials_path=cred_path,
        lang="en",
        version_info=(3, 11, 9),
        os_name="nt",
        system_name="Windows",
        environ={"COMPUTERNAME": "GAMING-RIG", "USERDOMAIN": "GAMING-RIG"},
        home=win_home,
        http_get=_mock_http_ok(),
        runner=_mock_runner(
            smi_out="NVIDIA GeForce RTX 4060, 8192, 6800, 70, 5\n",
            tasklist_out='"System Idle Process","0","Services","0","8 K"\n"steam.exe","3200","Console","1","120,000 K"\n',
        ),
    )
    assert win_ok["ready"] is True
    assert win_ok["readiness_checks_passed"] == 6
    cred_chk = _by_id(win_ok, "credentials")
    assert cred_chk["status"] == "ok"
    assert cred_chk["details"]["permission_model"] == "windows_acl"
    assert cred_chk["details"]["path"] == "~/.hivemind/credentials.json"

    # Windows with active game + old Python + missing creds -> actionable PowerShell / winget one-liners
    missing_cred = win_home / ".hivemind" / "missing.json"
    win_issues = doctor.run_doctor(
        credentials_path=missing_cred,
        lang="fr",
        version_info=(3, 9, 13),
        os_name="nt",
        system_name="Windows",
        environ={"USERDNSDOMAIN": "CORP.AMADEUS.EXAMPLE"},
        home=win_home,
        http_get=lambda _u, _t: (503, {"error": "unavailable"}),
        runner=_mock_runner(
            tasklist_out='"League of Legends.exe","7788","Console","1","940,000 K"\n',
        ),
    )
    assert win_issues["ready"] is False
    assert win_issues["verdict"] == "not_ready"
    assert "winget install -e --id Python.Python.3.12" in _by_id(win_issues, "python_runtime")["fix_command"]
    assert "Invoke-RestMethod" in _by_id(win_issues, "credentials")["fix_command"]
    assert "curl.exe -fsSL" in _by_id(win_issues, "gateway_reachability")["fix_command"]
    assert _by_id(win_issues, "active_games")["details"]["active_games"] == ["league of legends.exe"]
    assert "--uninstall" in _by_id(win_issues, "personal_pc_compliance")["fix_command"]
    assert "CORP.AMADEUS.EXAMPLE" not in json.dumps(win_issues)


def test_no_gpu_dev_api_only_mode_ready_en_and_fr(tmp_path: Path) -> None:
    cred_path = _write_creds(tmp_path / ".hivemind" / "credentials.json", mode=0o600)

    for lang_code in ("en", "fr"):
        rep = doctor.run_doctor(
            credentials_path=cred_path,
            lang=lang_code,
            version_info=(3, 13, 1),
            os_name="posix",
            system_name="Darwin",
            environ={},
            home=tmp_path,
            http_get=_mock_http_ok(),
            runner=_mock_runner(smi_exc=FileNotFoundError("nvidia-smi")),
        )
        assert rep["ready"] is True
        assert rep["verdict"] == "ready"
        assert rep["recommended_mode"] == "dev_api_only"
        gpu_chk = _by_id(rep, "gpu_readiness")
        assert gpu_chk["status"] == "ok"
        assert "Dev API-only mode ready" in gpu_chk["summary"]

    # Non-zero nvidia-smi exit code and blank stdout also map cleanly to no-GPU Dev API-only mode
    for rc_val, out_val in ((9, ""), (0, "   \n")):
        chk = doctor.check_gpu_readiness(
            runner=_mock_runner(smi_rc=rc_val, smi_out=out_val),
            os_name="posix",
            lang="en",
        )
        assert chk.status == "ok"
        assert chk.details["mode"] == "dev_api_only"
        assert "Dev API-only mode ready" in chk.summary


def test_hot_gpu_low_vram_multi_gpu_and_malformed_smi(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cred_path = _write_creds(tmp_path / ".hivemind" / "credentials.json", mode=0o600)

    # Hot GPU (> 70 C) triggers thermal cooldown warning and exit code 1
    rc_hot = doctor.main(
        ["--credentials-file", str(cred_path), "--lang", "en"],
        http_get=_mock_http_ok(),
        runner=_mock_runner(smi_out="NVIDIA GeForce RTX 4080, 16384, 14000, 76, 20\n"),
        environ={},
    )
    rendered_hot = capsys.readouterr().out
    assert rc_hot == 1
    assert "[WARN]" in rendered_hot
    assert "76 C" in rendered_hot
    assert "--gamer-check" in rendered_hot

    # French hot-GPU + ANSI color rendering check
    rep_hot_fr = doctor.run_doctor(
        credentials_path=cred_path,
        lang="fr",
        version_info=(3, 11, 5),
        os_name="posix",
        system_name="Linux",
        environ={},
        home=tmp_path,
        http_get=_mock_http_ok(),
        runner=_mock_runner(smi_out="NVIDIA GeForce RTX 4080, 16384, 14000, 74, [N/A]\n"),
    )
    colored_fr = doctor.render_report(rep_hot_fr, color=True)
    assert "\x1b[33m[WARN]\x1b[0m" in colored_fr
    assert "Action ->" in colored_fr
    assert "PRÉ-CONTRÔLE CANARY ALPHA" in colored_fr

    # Exact 4096 MiB and 70 C boundary passes Gamer-Whisper
    exact_chk = doctor.check_gpu_readiness(
        runner=_mock_runner(smi_out="NVIDIA GeForce GTX 1650, 4096, 3500, 70, [Not Supported]\n"),
        os_name="posix",
        lang="en",
    )
    assert exact_chk.status == "ok"
    assert exact_chk.details["mode"] == "gamer_whisper_ready"

    # Multi-GPU picks higher-VRAM card
    multi_chk = doctor.check_gpu_readiness(
        runner=_mock_runner(
            smi_out="NVIDIA GTX 1050, 2048, 1800, 45, 0\nNVIDIA RTX 3060, 12288, 11000, 50, 2\n"
        ),
        os_name="posix",
        lang="fr",
    )
    assert multi_chk.status == "ok"
    assert multi_chk.details["name"] == "NVIDIA RTX 3060"

    # Low VRAM (< 4096 MiB) warns with Dev API-only guidance
    for lang_code in ("en", "fr"):
        low_vram = doctor.check_gpu_readiness(
            runner=_mock_runner(smi_out="NVIDIA GT 1030, 2048, 1900, 42, 1\n"),
            os_name="posix",
            lang=lang_code,
        )
        assert low_vram.status == "warn"
        assert low_vram.details["mode"] == "dev_api_only"
        assert "Dev API-only mode ready" in low_vram.summary
        assert "--role consumer" in low_vram.fix_command

    # Malformed nvidia-smi outputs warn cleanly without crashing
    for bad_smi in ("corrupt_line", "GPU, not_a_num, 100, 40, 0", "GPU, 0, 100, 40, 0", "GPU, 4096, -5, 40, 0"):
        mal = doctor.check_gpu_readiness(
            runner=_mock_runner(smi_out=bad_smi),
            os_name="posix",
            lang="fr",
        )
        assert mal.status == "warn"
        assert mal.fix_command == "nvidia-smi"


def test_corporate_environment_local_only_never_transmits_env_or_hostname(tmp_path: Path) -> None:
    cred_path = _write_creds(tmp_path / ".hivemind" / "credentials.json", mode=0o600)
    captured_requests: list[tuple[str, float]] = []

    def _spy_http(url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
        captured_requests.append((url, timeout_s))
        assert "amadeus" not in url.lower()
        assert "zscaler" not in url.lower()
        assert "corp" not in url.lower()
        return _mock_http_ok()(url, timeout_s)

    corp_env = {
        "USERDNSDOMAIN": "INTERNAL.AMADEUS.CORP",
        "USERDOMAIN": "AMADEUS-EU",
        "COMPUTERNAME": "WORK-LAPTOP-99",
        "HTTPS_PROXY": "http://secretuser:secretpass@proxy.corp.internal:8080",
        "NO_PROXY": "localhost,.corp.internal",
        "ZSCALER_CERT_PATH": "/etc/ssl/zscaler_root.pem",
        "REQUESTS_CA_BUNDLE": "/etc/ssl/amadeus_ca.pem",
    }

    rep = doctor.run_doctor(
        credentials_path=cred_path,
        lang="en",
        version_info=(3, 12, 0),
        os_name="posix",
        system_name="Linux",
        environ=corp_env,
        home=tmp_path,
        http_get=_spy_http,
        runner=_mock_runner(),
    )
    assert rep["ready"] is False
    assert rep["verdict"] == "ready_with_warnings"
    comp = _by_id(rep, "personal_pc_compliance")
    assert comp["status"] == "warn"
    assert comp["details"]["network_transmitted"] is False
    assert set(comp["details"]["corporate_indicators"]) == {
        "USERDNSDOMAIN",
        "USERDOMAIN",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ZSCALER_CERT_PATH",
        "REQUESTS_CA_BUNDLE",
    }

    # Raw secret proxy URLs, domain values, and hostnames MUST NEVER appear in output
    dumped = json.dumps(rep) + doctor.render_report(rep)
    for sensitive in ("INTERNAL.AMADEUS.CORP", "AMADEUS-EU", "WORK-LAPTOP-99", "secretpass", "proxy.corp.internal"):
        assert sensitive not in dumped
    assert len(captured_requests) == 2

    # Ubuntu/home default NO_PROXY="localhost,127.0.0.1,::1" must NOT false-positive
    clean_chk = doctor.check_corporate_environment(
        environ={"NO_PROXY": "localhost,127.0.0.1,::1", "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt"},
        os_name="posix",
        lang="fr",
    )
    assert clean_chk.status == "ok"


def test_python_credentials_active_games_and_gateway_edge_cases(tmp_path: Path) -> None:
    # Language normalization & validation
    assert doctor.normalize_lang("FR_FR") == "fr"
    assert doctor.normalize_lang("en-US") == "en"
    with pytest.raises(ValueError, match="Unsupported language"):
        doctor.normalize_lang("de")

    # Python version matrix across macOS / Linux / custom OS
    mac_old = doctor.check_python_runtime((3, 9, 18), os_name="posix", system_name="Darwin", lang="en")
    assert mac_old.status == "fail" and mac_old.fix_command == "brew install python@3.12"
    lin_old = doctor.check_python_runtime((3, 8, 10), os_name="posix", system_name="Linux", lang="fr")
    assert lin_old.status == "fail" and "apt-get" in lin_old.fix_command
    py_future = doctor.check_python_runtime((3, 14, 0), os_name="posix", system_name="FreeBSD", lang="en")
    assert py_future.status == "warn" and py_future.details["os"] == "FreeBSD"
    py_future_fr = doctor.check_python_runtime((3, 14, 0), os_name="posix", system_name="Linux", lang="fr")
    assert py_future_fr.status == "warn"

    # Credentials edge cases: invalid JSON, invalid key format, POSIX 0644 under home & custom path
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    assert doctor.check_credentials_file(bad_json, os_name="posix", lang="fr").status == "fail"

    for invalid_key in ("", "sk-openai-key-12345", "hm-short", "hm-has space 12345", 12345):
        bad_key_file = tmp_path / "bad_key.json"
        bad_key_file.write_text(json.dumps({"api_key": invalid_key}), encoding="utf-8")
        assert doctor.check_credentials_file(bad_key_file, os_name="posix", lang="en").status == "fail"
        assert doctor.check_credentials_file(bad_key_file, os_name="posix", lang="fr").status == "fail"

    home_dir = tmp_path / "uhome"
    open_home_cred = _write_creds(home_dir / ".hivemind" / "credentials.json", mode=0o644)
    perm_warn_en = doctor.check_credentials_file(open_home_cred, os_name="posix", lang="en", home=home_dir)
    assert perm_warn_en.status == "warn"
    assert perm_warn_en.fix_command == "chmod 600 ~/.hivemind/credentials.json"

    ext_cred = _write_creds(tmp_path / "ext space" / "creds.json", mode=0o644)
    perm_warn_fr = doctor.check_credentials_file(ext_cred, os_name="posix", lang="fr", home=home_dir)
    assert perm_warn_fr.status == "warn"
    assert perm_warn_fr.fix_command.startswith("chmod 600 ")

    # Active games on Linux/macOS including custom --game and subprocess errors
    game_chk = doctor.check_active_games(
        runner=_mock_runner(ps_out="/Applications/Steam/dota2\nmycustomgame\n"),
        os_name="posix",
        extra_games=["MyCustomGame"],
        lang="en",
    )
    assert game_chk.status == "warn"
    assert game_chk.details["active_games"] == ["dota2", "mycustomgame"]

    assert doctor.list_process_names(_mock_runner(ps_rc=1), os_name="posix") == set()
    assert doctor.list_process_names(_mock_runner(tasklist_rc=1), os_name="nt") == set()
    assert doctor.list_process_names(_mock_runner(proc_exc=OSError("no ps")), os_name="posix") == set()

    # Gateway URL validation & failure branches without touching network
    for bad_gw in (
        "",
        "   ",
        "https://evil.example.com",
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "ftp://127.0.0.1:8000",
        "https://user:pw@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app?tok=1",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app#frag",
        "http://127.0.0.1:0",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app:99999",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app:notaport",
        "https://bad space.run.app",
        "https://.run.app",
    ):
        with pytest.raises(ValueError):
            doctor.validate_gateway_url(bad_gw)

    assert doctor.is_allowed_gateway_host("") is False
    assert doctor.is_allowed_gateway_host("storage.googleapis.com") is True
    assert doctor.is_allowed_gateway_host("run.googleapis.com") is True
    assert doctor.is_allowed_gateway_host("datasets-server.huggingface.co") is True

    def _never_http(_u: str, _t: float) -> tuple[int, dict[str, Any]]:
        raise AssertionError("http_get must never run on untrusted gateway URLs")

    refused_gw = doctor.check_gateway_reachability("https://attacker.example.org", http_get=_never_http, lang="en")
    assert refused_gw.status == "fail" and refused_gw.details["allowed_host"] is False

    refused_rep = doctor.run_doctor(
        gateway="https://attacker.example.org",
        credentials_path=open_home_cred,
        lang="en",
        version_info=(3, 12, 0),
        os_name="posix",
        system_name="Linux",
        environ={},
        home=home_dir,
        http_get=_never_http,
        runner=_mock_runner(),
    )
    assert refused_rep["gateway"] == "rejected_untrusted_gateway"
    assert refused_rep["verdict"] == "not_ready"

    # /health exception, /v1/models exception, and /v1/models malformed payload
    h_err = doctor.check_gateway_reachability(
        doctor.DEFAULT_GATEWAY,
        http_get=lambda _u, _t: (_ for _ in ()).throw(OSError("offline")),
        lang="fr",
    )
    assert h_err.status == "fail"

    def _models_net_fail(url: str, _t: float) -> tuple[int, dict[str, Any]]:
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        raise OSError("reset")

    m_err = doctor.check_gateway_reachability(doctor.DEFAULT_GATEWAY, http_get=_models_net_fail, lang="en")
    assert m_err.status == "fail" and m_err.details["health_ok"] is True

    def _models_bad_body(url: str, _t: float) -> tuple[int, dict[str, Any]]:
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        return 200, {"object": "invalid_not_a_list"}

    m_bad = doctor.check_gateway_reachability(doctor.DEFAULT_GATEWAY, http_get=_models_bad_body, lang="fr")
    assert m_bad.status == "fail" and m_bad.details["models_ok"] is False


def test_default_http_get_json_and_subprocess_runner_on_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers: list[dict[str, str]] = []

    class _LocalHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen_headers.append({k.lower(): v for k, v in self.headers.items()})
            if self.path == "/health":
                body = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/v1/models":
                body = json.dumps({"object": "list", "data": [{"id": "hivemind/coder"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/plain":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"plain-ok")
            elif self.path == "/err-json":
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"unauthorized"}')
            else:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"bad gateway")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return None

    server = http.server.HTTPServer(("127.0.0.1", 0), _LocalHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Even if HTTPS_PROXY / HTTP_PROXY is set in os.environ, ProxyHandler({}) ignores it!
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")

        base = f"http://127.0.0.1:{port}"
        chk = doctor.check_gateway_reachability(base, http_get=doctor.default_http_get_json, lang="en")
        assert chk.status == "ok"
        assert chk.details["sample_models"] == ["hivemind/coder"]

        st_plain, body_plain = doctor.default_http_get_json(f"{base}/plain", 2.0)
        assert st_plain == 200 and body_plain["raw"] == "plain-ok"

        st_401, body_401 = doctor.default_http_get_json(f"{base}/err-json", 2.0)
        assert st_401 == 401 and body_401["error"] == "unauthorized"

        st_502, body_502 = doctor.default_http_get_json(f"{base}/err-text", 2.0)
        assert st_502 == 502 and "bad gateway" in body_502["error"]

        for hdrs in seen_headers:
            assert hdrs.get("user-agent") == "hivemind-doctor/1.0"
            assert "authorization" not in hdrs

        # Direct __main__ CLI execution against loopback server
        cred_file = _write_creds(tmp_path / "loopback_creds.json")
        proc = subprocess.run(
            [
                sys.executable,
                str(DOCTOR_PATH),
                "--gateway",
                base,
                "--credentials-file",
                str(cred_file),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
        )
        assert proc.returncode in (0, 1)
        parsed = json.loads(proc.stdout)
        assert parsed["schema"] == "hivemind.doctor/1"
    finally:
        server.shutdown()
        server.server_close()

    # Verify default_subprocess_runner directly
    rc_sub, out_sub = doctor.default_subprocess_runner([sys.executable, "-c", "print('doctor-runner-ok')"], 3.0)
    assert rc_sub == 0 and out_sub.strip() == "doctor-runner-ok"


def test_gateway_serves_hivemind_doctor_script_without_repo_clone(tmp_path: Path) -> None:
    assert "hivemind_doctor.py" in _ALLOWED_ONBOARDING_SCRIPTS
    store = RouterStore(tmp_path / "router.db", secret=b"slice03-doctor-gateway-secret-32b!")
    client = TestClient(create_router_app(store))

    resp = client.get("/scripts/hivemind_doctor.py")
    assert resp.status_code == 200
    assert "text/x-python" in resp.headers["content-type"]
    assert 'filename="hivemind_doctor.py"' in resp.headers.get("content-disposition", "")
    assert resp.text == DOCTOR_PATH.read_text(encoding="utf-8")
    assert "def run_doctor(" in resp.text

    not_allowed = client.get("/scripts/not_allowlisted.py")
    assert not_allowed.status_code == 404
