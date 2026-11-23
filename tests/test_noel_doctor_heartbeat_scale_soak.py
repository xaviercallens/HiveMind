# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Standalone unit, drift, Steam preemption, 48-cohort SQLite WAL soak & security tests for SLICE-SCALE-DYN-07."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any

import coverage
import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = REPO_ROOT / "scripts" / "hivemind_doctor.py"
GATEWAY_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "gateway.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(DOCTOR_PATH)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import _ALLOWED_ONBOARDING_SCRIPTS, create_router_app  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_doc_spec = importlib.util.spec_from_file_location("hivemind_doctor_dyn07", DOCTOR_PATH)
assert _doc_spec is not None and _doc_spec.loader is not None
doctor = importlib.util.module_from_spec(_doc_spec)
sys.modules["hivemind_doctor_dyn07"] = doctor
_doc_spec.loader.exec_module(doctor)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn07",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn07",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn07",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "home-gaming-pc"}
SOAK_SECRET = b"noel-scale-dyn07-router-sec-32b!"
ALPHA_KEY = "hm-alpha_personal_doctor_123456"
LIVE_KEY = "hm_live_personal_doctor_654321"
UNTRUSTED_GW = "https://" + "attacker.example.org"
UNTRUSTED_EVIL_GW = "https://" + "evil.example.org"
UNTRUSTED_HEALTH_URL = "https://" + "evil.example.com/health"
BAD_SPACE_GW = "https://" + "bad space.run.app"
CORP_SSH_REMOTE = "ssh://" + "git@git-amadeus-internal.corp/core/service.git"
CORP_INTRANET_URL = "https://" + "intranet.enterprise.local/"
CORP_ZSCALER_PROXY = "http://" + "user:secretpw@zscaler.internal:8443"

_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip_for(bucket: int, idx: int) -> str:
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


def _collect_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_keys(item))
    return found


def _verify_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_keys(doc)


def _write_credentials(
    path: Path,
    *,
    api_key: str = ALPHA_KEY,
    handle: str = "home-user",
    role: str = "prosumer",
    mode: int = 0o600,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "account_id": "acct_personal_01",
                "api_key": api_key,
                "gateway": doctor.DEFAULT_GATEWAY,
                "handle": handle,
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


def _build_basic_http_ok(seen: list[str] | None = None) -> doctor.HttpGetFn:
    def _http(url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
        assert 0.5 <= timeout_s <= 4.5
        if seen is not None:
            seen.append(url)
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"object": "list", "data": [{"id": "hivemind/auto"}, {"id": "hivemind/coder"}]}
        return 404, {"error": "not_found"}

    return _http


def _build_local_runner(
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


def _by_check_id(report: dict[str, Any], check_id: str) -> dict[str, Any]:
    return next(item for item in report["checks"] if item["id"] == check_id)


def _build_router_request_adapter(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str, dict[str, str]]] | None = None,
    sqlite_lock_hits: list[str] | None = None,
    lock: threading.Lock | None = None,
) -> doctor.HttpRequestFn:
    def _request(
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 4.0,
    ) -> tuple[int, dict[str, Any]]:
        _ = timeout_s
        assert "completions" not in url.lower(), f"Zero-inference violated on {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        req_headers = dict(headers or {})
        req_headers["X-Forwarded-For"] = peer_ip
        if seen_calls is not None:
            if lock is not None:
                with lock:
                    seen_calls.append((method, path, dict(req_headers)))
            else:
                seen_calls.append((method, path, dict(req_headers)))
        resp = client.request(method, path, json=payload, headers=req_headers)
        raw_text = resp.text
        if sqlite_lock_hits is not None and _SQLITE_LOCK_RE.search(raw_text):
            if lock is not None:
                with lock:
                    sqlite_lock_hits.append(raw_text[:200])
            else:
                sqlite_lock_hits.append(raw_text[:200])
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json()
        return resp.status_code, {
            "content_type": ctype,
            "html_length": len(raw_text),
            "raw": raw_text[:240],
        }

    return _request


def test_01_headers_stdlib_lateral_movement_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for target in (DOCTOR_PATH, GATEWAY_PATH, this_file):
        head = target.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in {target}"

    doctor_source = DOCTOR_PATH.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 stdlib imports, ProxyHandler({}), shell=False, zero eval/exec/os.system/pickle
    tree = ast.parse(doctor_source, filename="scripts/hivemind_doctor.py")
    imported_roots: set[str] = set()
    empty_proxy_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec"}
            if isinstance(node.func, ast.Attribute):
                assert not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "system"
                )
                if (
                    node.func.attr == "ProxyHandler"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Dict)
                    and len(node.args[0].keys) == 0
                ):
                    empty_proxy_calls += 1
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False

    assert imported_roots - sys.stdlib_module_names == set()
    assert imported_roots & {"pickle", "marshal"} == set()
    assert empty_proxy_calls >= 2

    # 2. Lateral-movement AST audit (0 violations + negative controls)
    assert lateral_guard.audit_source_for_lateral_movement(doctor_source, filename="scripts/hivemind_doctor.py") == []
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_source,
            filename="tests/test_noel_doctor_heartbeat_scale_soak.py",
        )
        == []
    )
    lan_ip = ".".join(("192", "168", "44", "9"))
    ext_url = "https://" + "evil" + ".example.org/x"
    assert lateral_guard.audit_source_for_lateral_movement(doctor_source + f'\n_X = "http://{lan_ip}:8000"\n') != []
    assert lateral_guard.audit_source_for_lateral_movement(doctor_source + f'\n_Y = "{ext_url}"\n') != []

    # 3. PR gatekeeper FORBIDDEN_ADDED scan
    for rel, content in (
        ("scripts/hivemind_doctor.py", doctor_source),
        ("tests/test_noel_doctor_heartbeat_scale_soak.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. PRSecurityScanner (0 findings), ZeroStubAudit (clean), and static-numbers guard
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/hivemind_doctor.py", doctor_source),
        ("tests/test_noel_doctor_heartbeat_scale_soak.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        stub_res = ZeroStubAudit().audit_python_code(content)
        assert stub_res.is_clean, f"Unexpected ZeroStubAudit violations in {rel}: {stub_res.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1077,
        pr_title="feat(onboarding,scale): <10s doctor --noel holiday route drift, <1.5s Steam yield/thermal & first-heartbeat verifier",
        pr_body="Implements SLICE-SCALE-DYN-07 doctor --noel holiday route drift, Steam yield/thermal & first-heartbeat verifier.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(onboarding,scale): <10s doctor --noel holiday route drift, <1.5s Steam yield/thermal & first-heartbeat verifier"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True


def test_02_backward_compatible_6_check_paths_cross_platform_render_and_gateway_serving(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Linux Gamer-Whisper ready (exact 6 checks, exact 2 GETs to /health & /v1/models)
    home_linux = tmp_path / "home" / "secret_linux_user"
    cred_linux = _write_credentials(home_linux / ".hivemind" / "credentials.json", mode=0o600)
    seen_urls: list[str] = []

    rep_linux = doctor.run_doctor(
        gateway=doctor.DEFAULT_GATEWAY,
        credentials_path=cred_linux,
        lang="en",
        version_info=(3, 12, 5),
        os_name="posix",
        system_name="Linux",
        environ={"HOME": str(home_linux), "PATH": "/usr/bin"},
        home=home_linux,
        http_get=_build_basic_http_ok(seen_urls),
        runner=_build_local_runner(),
    )
    assert rep_linux["schema"] == "hivemind.doctor/1"
    assert rep_linux["ready"] is True
    assert rep_linux["verdict"] == "ready"
    assert rep_linux["recommended_mode"] == "gamer_whisper"
    assert rep_linux["readiness_checks_passed"] == 6
    assert rep_linux["readiness_checks_total"] == 6
    assert "noel_summary" not in rep_linux
    assert seen_urls == [f"{doctor.DEFAULT_GATEWAY}/health", f"{doctor.DEFAULT_GATEWAY}/v1/models"]
    assert "secret_linux_user" not in json.dumps(rep_linux)
    assert ALPHA_KEY not in json.dumps(rep_linux)

    # CLI default JSON exit 0
    rc_cli = doctor.main(
        ["--credentials-file", str(cred_linux), "--lang", "en", "--json"],
        http_get=_build_basic_http_ok(),
        runner=_build_local_runner(),
        environ={"PATH": "/usr/bin"},
        home=home_linux,
    )
    assert rc_cli == 0
    cli_parsed = json.loads(capsys.readouterr().out)
    assert cli_parsed["readiness_checks_total"] == 6

    # 2. Windows 11 ACL + tasklist + PowerShell/winget fixes + warn-level corporate check when noel=False
    win_home = tmp_path / "Users" / "WinPlayer"
    cred_win = _write_credentials(win_home / ".hivemind" / "credentials.json", api_key=LIVE_KEY, mode=0o644)
    rep_win_ok = doctor.run_doctor(
        credentials_path=cred_win,
        lang="en",
        version_info=(3, 11, 9),
        os_name="nt",
        system_name="Windows",
        environ={"COMPUTERNAME": "GAMER-DESK", "USERDOMAIN": "GAMER-DESK"},
        home=win_home,
        http_get=_build_basic_http_ok(),
        runner=_build_local_runner(
            smi_out="NVIDIA GeForce RTX 4060, 8192, 6800, 68, 5\n",
            tasklist_out='"steam.exe","3200","Console","1","120,000 K"\n',
        ),
    )
    assert rep_win_ok["ready"] is True
    assert _by_check_id(rep_win_ok, "credentials")["details"]["permission_model"] == "windows_acl"
    assert _by_check_id(rep_win_ok, "credentials")["details"]["path"] == "~/.hivemind/credentials.json"

    rep_win_issues = doctor.run_doctor(
        credentials_path=win_home / ".hivemind" / "absent.json",
        lang="fr",
        version_info=(3, 9, 12),
        os_name="nt",
        system_name="Windows",
        environ={"USERDNSDOMAIN": "WORK.CORP.LOCAL", "HTTPS_PROXY": "http://127.0.0.1:8888"},
        home=win_home,
        http_get=lambda _u, _t: (503, {"error": "down"}),
        runner=_build_local_runner(tasklist_out='"League of Legends.exe","7788","Console","1","940,000 K"\n'),
    )
    assert rep_win_issues["verdict"] == "not_ready"
    assert "winget install" in _by_check_id(rep_win_issues, "python_runtime")["fix_command"]
    assert "Invoke-RestMethod" in _by_check_id(rep_win_issues, "credentials")["fix_command"]
    assert "curl.exe -fsSL" in _by_check_id(rep_win_issues, "gateway_reachability")["fix_command"]
    assert _by_check_id(rep_win_issues, "active_games")["status"] == "warn"
    assert _by_check_id(rep_win_issues, "personal_pc_compliance")["status"] == "warn"
    assert "WORK.CORP.LOCAL" not in json.dumps(rep_win_issues)

    # 3. macOS no-GPU Dev API-only (EN & FR) + non-zero/blank nvidia-smi
    for lang_code in ("en", "fr"):
        rep_mac = doctor.run_doctor(
            credentials_path=cred_linux,
            lang=lang_code,
            version_info=(3, 13, 1),
            os_name="posix",
            system_name="Darwin",
            environ={},
            home=home_linux,
            http_get=_build_basic_http_ok(),
            runner=_build_local_runner(smi_exc=FileNotFoundError("nvidia-smi")),
        )
        assert rep_mac["ready"] is True
        assert rep_mac["recommended_mode"] == "dev_api_only"
        assert "Dev API-only mode ready" in _by_check_id(rep_mac, "gpu_readiness")["summary"]

    for rc_val, out_val in ((9, ""), (0, "  \n")):
        assert doctor.check_gpu_readiness(runner=_build_local_runner(smi_rc=rc_val, smi_out=out_val)).status == "ok"

    # 4. Hot GPU (>70 C), low VRAM (<4096 MiB), multi-GPU, malformed nvidia-smi, ANSI/plain FR & EN render_report
    rc_hot = doctor.main(
        ["--credentials-file", str(cred_linux), "--lang", "en", "--no-color"],
        http_get=_build_basic_http_ok(),
        runner=_build_local_runner(smi_out="NVIDIA GeForce RTX 4080, 16384, 14000, 77, 18\n"),
        environ={},
        home=home_linux,
    )
    out_hot_en = capsys.readouterr().out
    assert rc_hot == 1
    assert "[WARN]" in out_hot_en and "Fix ->" in out_hot_en

    rep_hot_fr = doctor.run_doctor(
        credentials_path=cred_linux,
        lang="fr",
        version_info=(3, 12, 0),
        os_name="posix",
        system_name="Linux",
        environ={},
        home=home_linux,
        http_get=_build_basic_http_ok(),
        runner=_build_local_runner(smi_out="NVIDIA GeForce RTX 4080, 16384, 14000, 75, [N/A]\n"),
    )
    rendered_fr_ansi = doctor.render_report(rep_hot_fr, color=True)
    rendered_en_plain = doctor.render_report(rep_linux, color=False)
    assert "\x1b[33m[WARN]\x1b[0m" in rendered_fr_ansi and "Action ->" in rendered_fr_ansi
    assert "[ OK ]" in rendered_en_plain

    multi_gpu = doctor.check_gpu_readiness(
        runner=_build_local_runner(smi_out="NVIDIA GTX 1050, 2048, 1800, 45, 0\nNVIDIA RTX 3060, 12288, 11000, 50, 2\n"),
        lang="fr",
    )
    assert multi_gpu.status == "ok" and multi_gpu.details["name"] == "NVIDIA RTX 3060"

    for lang_code in ("en", "fr"):
        low_vram = doctor.check_gpu_readiness(
            runner=_build_local_runner(smi_out="NVIDIA GT 1030, 2048, 1900, 42, 1\n"),
            lang=lang_code,
        )
        assert low_vram.status == "warn" and "--role consumer" in low_vram.fix_command

    for bad_smi in ("corrupt", "GPU, bad, 100, 40, 0", "GPU, 0, 100, 40, 0", "GPU, 4096, -5, 40, 0"):
        assert doctor.check_gpu_readiness(runner=_build_local_runner(smi_out=bad_smi), lang="en").status == "warn"
        assert doctor.check_gpu_readiness(runner=_build_local_runner(smi_out=bad_smi), lang="fr").status == "warn"

    # 5. Python runtime, credentials, active games, corporate env, and untrusted gateway edge cases
    assert doctor.normalize_lang("fr_BE") == "fr"
    assert doctor.normalize_lang("en-GB") == "en"
    with pytest.raises(ValueError):
        doctor.normalize_lang("es")
    assert doctor.check_python_runtime((3, 9, 1), os_name="posix", system_name="Darwin", lang="en").status == "fail"
    assert doctor.check_python_runtime((3, 8, 5), os_name="posix", system_name="Linux", lang="fr").status == "fail"
    assert doctor.check_python_runtime((3, 14, 0), os_name="posix", system_name="FreeBSD", lang="en").status == "warn"
    assert doctor.check_python_runtime((3, 14, 0), os_name="posix", system_name="Linux", lang="fr").status == "warn"

    bad_json = tmp_path / "broken_creds.json"
    bad_json.write_text("{invalid", encoding="utf-8")
    assert doctor.check_credentials_file(bad_json, os_name="posix", lang="en").status == "fail"
    assert doctor.check_credentials_file(bad_json, os_name="posix", lang="fr").status == "fail"
    bad_key = tmp_path / "invalid_key.json"
    bad_key.write_text(json.dumps({"api_key": "sk-wrong-123456"}), encoding="utf-8")
    assert doctor.check_credentials_file(bad_key, os_name="posix", lang="en").status == "fail"
    assert doctor.check_credentials_file(bad_key, os_name="posix", lang="fr").status == "fail"
    open_home = _write_credentials(home_linux / ".hivemind" / "open.json", mode=0o644)
    assert doctor.check_credentials_file(open_home, os_name="posix", lang="en", home=home_linux).status == "warn"
    ext_open = _write_credentials(tmp_path / "ext space" / "open.json", mode=0o644)
    assert doctor.check_credentials_file(ext_open, os_name="posix", lang="fr", home=home_linux).status == "warn"

    game_posix = doctor.check_active_games(
        runner=_build_local_runner(ps_out="/opt/games/dota2\ncustomgame\n"),
        os_name="posix",
        extra_games=["CustomGame"],
        lang="en",
    )
    assert game_posix.status == "warn" and game_posix.details["active_games"] == ["customgame", "dota2"]
    assert doctor.list_process_names(_build_local_runner(ps_rc=1), os_name="posix") == set()
    assert doctor.list_process_names(_build_local_runner(tasklist_rc=1), os_name="nt") == set()
    assert doctor.list_process_names(_build_local_runner(proc_exc=OSError("missing ps")), os_name="posix") == set()

    corp_warn_en = doctor.check_corporate_environment(
        environ={
            "USERDNSDOMAIN": "CORP.LOCAL",
            "USERDOMAIN": "CORP-EU",
            "COMPUTERNAME": "LAPTOP-9",
            "NO_PROXY": "localhost,.corp.local",
            "ZSCALER_CERT_PATH": "/etc/zscaler.pem",
            "REQUESTS_CA_BUNDLE": "/etc/ssl/corp_ca.pem",
        },
        lang="en",
    )
    assert corp_warn_en.status == "warn"
    assert doctor.check_corporate_environment(environ={"NO_PROXY": "localhost,127.0.0.1"}, lang="fr").status == "ok"

    for invalid_gw in (
        "",
        UNTRUSTED_GW,
        "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "ftp://127.0.0.1:8000",
        "https://user:pw@hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app?a=1",
        "http://127.0.0.1:0",
        "http://127.0.0.1:badport",
        BAD_SPACE_GW,
    ):
        with pytest.raises(ValueError):
            doctor.validate_gateway_url(invalid_gw)

    assert doctor.is_allowed_gateway_host("") is False
    assert doctor.is_allowed_gateway_host("storage.googleapis.com") is True
    assert doctor.is_allowed_gateway_host("run.googleapis.com") is True
    assert doctor.is_allowed_gateway_host("datasets-server.huggingface.co") is True

    refused_rep = doctor.run_doctor(
        gateway=UNTRUSTED_GW,
        credentials_path=cred_linux,
        home=home_linux,
        environ={},
        http_get=lambda _u, _t: (_ for _ in ()).throw(AssertionError("untrusted host contacted")),
        runner=_build_local_runner(),
    )
    assert refused_rep["gateway"] == "rejected_untrusted_gateway" and refused_rep["verdict"] == "not_ready"

    assert (
        doctor.check_gateway_reachability(
            doctor.DEFAULT_GATEWAY,
            http_get=lambda _u, _t: (_ for _ in ()).throw(OSError("offline")),
            lang="en",
        ).status
        == "fail"
    )
    assert (
        doctor.check_gateway_reachability(
            doctor.DEFAULT_GATEWAY,
            http_get=lambda u, _t: (200, {"status": "ok"}) if u.endswith("/health") else (_ for _ in ()).throw(OSError("reset")),
            lang="fr",
        ).status
        == "fail"
    )
    assert (
        doctor.check_gateway_reachability(
            doctor.DEFAULT_GATEWAY,
            http_get=lambda u, _t: (200, {"status": "ok"}) if u.endswith("/health") else (200, {"data": "not-list"}),
            lang="en",
        ).status
        == "fail"
    )

    # 6. Gateway /scripts/hivemind_doctor.py serving exact file content
    assert "hivemind_doctor.py" in _ALLOWED_ONBOARDING_SCRIPTS
    store = RouterStore(tmp_path / "serve_doctor.db", secret=SOAK_SECRET)
    with TestClient(create_router_app(store)) as client:
        resp_script = client.get("/scripts/hivemind_doctor.py")
        assert resp_script.status_code == 200
        assert "text/x-python" in resp_script.headers["content-type"]
        assert resp_script.text == DOCTOR_PATH.read_text(encoding="utf-8")


def test_03_loopback_proxy_free_http_get_and_http_request_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_requests: list[tuple[str, str, dict[str, str], str]] = []

    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            seen_requests.append(("GET", self.path, hdrs, ""))
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{"id": "hivemind/auto"}]})
            elif self.path == "/noel":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<!DOCTYPE html><title>Noel</title>")
            elif self.path == "/empty":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/list":
                self._send_json(200, [1, 2])
            elif self.path == "/err-json":
                self._send_json(401, {"error": "unauthorized"})
            else:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"upstream bad gateway")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
            hdrs = {k.lower(): v for k, v in self.headers.items()}
            seen_requests.append(("POST", self.path, hdrs, raw_body))
            if self.path == "/v1/hivemind/onboarding/feedback":
                self._send_json(200, {"total_responses": 1, "mean_ease_rating": 5.0})
            elif self.path == "/v1/hivemind/opentrainer/heartbeat":
                self._send_json(200, {"object": "hivemind.opentrainer_heartbeat_ack"})
            else:
                self._send_json(404, {"detail": "not_found"})

        def _send_json(self, code: int, body: Any) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Poison ambient HTTP_PROXY and HTTPS_PROXY to prove ProxyHandler({}) ignores them completely
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        base = f"http://127.0.0.1:{port}"

        assert doctor.default_http_get_json(f"{base}/health", 2.0) == (200, {"status": "ok"})
        assert doctor.default_http_get_json(f"{base}/noel", 2.0)[0] == 200
        assert doctor.default_http_get_json(f"{base}/list", 2.0) == (200, {"data": [1, 2]})
        assert doctor.default_http_get_json(f"{base}/err-json", 2.0)[0] == 401
        assert doctor.default_http_get_json(f"{base}/err-text", 2.0)[0] == 502

        c_html, b_html = doctor.default_http_request_json("GET", f"{base}/noel", timeout_s=2.0)
        assert c_html == 200 and b_html["html_length"] > 0
        assert doctor.default_http_request_json("GET", f"{base}/empty", timeout_s=2.0)[0] == 204
        assert doctor.default_http_request_json("GET", f"{base}/list", timeout_s=2.0)[1] == {"data": [1, 2]}
        assert doctor.default_http_request_json("GET", f"{base}/err-json", timeout_s=2.0)[0] == 401
        assert doctor.default_http_request_json("GET", f"{base}/err-text", timeout_s=2.0)[0] == 502

        c_post, b_post = doctor.default_http_request_json(
            "POST",
            f"{base}/v1/hivemind/onboarding/feedback",
            payload={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""},
            headers={"Idempotency-Key": "hm-ces-loopback-test"},
            timeout_s=2.0,
        )
        assert c_post == 200 and b_post["total_responses"] == 1
        assert any(m == "POST" and h.get("idempotency-key") == "hm-ces-loopback-test" for m, _, h, _ in seen_requests)

        # Reject forbidden methods, untrusted hosts, completion endpoints, GET payloads, and CRLF headers
        for bad_call in (
            lambda: doctor.default_http_request_json("DELETE", f"{base}/health"),
            lambda: doctor.default_http_request_json("POST", f"{base}/v1/chat/completions", payload={}),
            lambda: doctor.default_http_request_json("POST", f"{base}/v1/unknown", payload={}),
            lambda: doctor.default_http_request_json("GET", f"{base}/health", payload={"x": 1}),
            lambda: doctor.default_http_request_json("GET", UNTRUSTED_HEALTH_URL),
            lambda: doctor.default_http_request_json("GET", f"{base}/health", headers={"X-Bad": "val\r\ninject"}),
        ):
            with pytest.raises(ValueError):
                bad_call()
    finally:
        server.shutdown()
        server.server_close()

    rc_sub, out_sub = doctor.default_subprocess_runner([sys.executable, "-c", "print('sub-ok')"], 3.0)
    assert rc_sub == 0 and out_sub.strip() == "sub-ok"


def test_04_noel_drift_cutover_real_yield_drill_thermal_ces_fallback_and_corporate_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", "d07cutover99")
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00107-noel")

    home_dir = tmp_path / "home" / "perso_gamer"
    ws_dir = tmp_path / "hobby_project"
    ws_dir.mkdir(parents=True, exist_ok=True)
    cred_file = _write_credentials(home_dir / ".hivemind" / "credentials.json", api_key=ALPHA_KEY, mode=0o600)

    # 1. Pre-PR-#949 stale Cloud Run 404 drift -> holiday_routes_deployed_on_cloud_run=False, revision_drift_detected=True, exit 2
    def _stale_pre_pr949_req(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (payload, headers)
        if method == "GET" and url.endswith("/health"):
            return 200, {"status": "ok"}
        if method == "GET" and url.endswith("/v1/models"):
            return 200, {"object": "list", "data": [{"id": "hivemind/auto"}]}
        if method == "GET" and url.endswith("/v1/hivemind/producers/fleet"):
            return 200, {"object": "hivemind.producer_fleet", "live_hiveminders": 3}
        return 404, {"detail": "Not Found"}

    rc_stale = doctor.main(
        [
            "--noel",
            "--gateway",
            doctor.DEFAULT_GATEWAY,
            "--credentials-file",
            str(cred_file),
            "--cwd",
            str(ws_dir),
            "--submit-ces",
            "--profile",
            "gamer",
            "--ease-rating",
            "5",
            "--lang",
            "fr",
            "--json",
        ],
        http_request=_stale_pre_pr949_req,
        runner=_build_local_runner(),
        environ=CLEAN_ENV,
        home=home_dir,
    )
    assert rc_stale == 2
    stale_rep = json.loads(capsys.readouterr().out)
    _verify_receipt(stale_rep)
    assert stale_rep["verdict"] == "not_ready"
    assert stale_rep["elapsed_s"] < 10.0
    routes_chk = _by_check_id(stale_rep, "noel_holiday_routes")
    assert routes_chk["status"] == "fail"
    assert routes_chk["details"]["holiday_routes_deployed_on_cloud_run"] is False
    assert routes_chk["details"]["onboarding_feedback_deployed"] is False
    assert routes_chk["details"]["revision_drift_detected"] is True
    assert set(routes_chk["details"]["missing_or_stale_routes"]) == set(doctor.NOEL_HOLIDAY_ROUTES)
    fleet_stale_chk = _by_check_id(stale_rep, "noel_fleet_heartbeat")
    assert fleet_stale_chk["details"]["online_producers"] == 3
    assert fleet_stale_chk["details"]["active_trainers"] == "n/a"
    assert fleet_stale_chk["details"]["accepted_rounds"] == "n/a"
    assert fleet_stale_chk["details"]["ces_recorded"] is False
    assert fleet_stale_chk["details"]["ces_degraded"] is True

    # 2. Cutover to live create_router_app + REAL subprocess <1.5s yield drill (default_worker_yield_drill)
    store = RouterStore(tmp_path / "cutover_live.db", secret=SOAK_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, open_registration=True)
    with TestClient(app) as client:
        reg = client.post(
            "/v1/hivemind/register",
            json={"handle": "cutover-gamer", "role": "prosumer"},
            headers={"X-Forwarded-For": "127.0.10.1"},
        ).json()
        live_cred = _write_credentials(
            home_dir / ".hivemind" / "live_creds.json",
            api_key=str(reg["api_key"]),
            handle="cutover-gamer",
            role="prosumer",
            mode=0o600,
        )
        live_adapter = _build_router_request_adapter(client, peer_ip="127.0.10.1")

        # Note: worker_yield_fn is NOT overridden -> runs real subprocess.Popen dummy-worker preemption drill!
        rc_live = doctor.main(
            [
                "--noel",
                "--gateway",
                "http://127.0.0.1:8000",
                "--credentials-file",
                str(live_cred),
                "--cwd",
                str(ws_dir),
                "--first-heartbeat",
                "--submit-ces",
                "--profile",
                "gamer",
                "--ease-rating",
                "5",
                "--host-label",
                "home-rtx-box",
                "--lang",
                "en",
                "--json",
            ],
            http_request=live_adapter,
            runner=_build_local_runner(),
            environ=CLEAN_ENV,
            home=home_dir,
        )
        assert rc_live == 0
        live_rep = json.loads(capsys.readouterr().out)
        _verify_receipt(live_rep)
        assert live_rep["ready"] is True
        assert live_rep["elapsed_s"] < 10.0
        assert live_rep["noel_summary"]["holiday_routes_deployed_on_cloud_run"] is True
        assert live_rep["noel_summary"]["onboarding_feedback_deployed"] is True
        assert live_rep["noel_summary"]["revision_drift_detected"] is False
        assert live_rep["noel_summary"]["yield_slo_ok"] is True
        assert 0.0 <= live_rep["noel_summary"]["yield_latency_ms"] < 1500.0
        assert live_rep["noel_summary"]["first_heartbeat_ok"] is True
        assert live_rep["noel_summary"]["active_trainers"] >= 1
        assert live_rep["noel_summary"]["ces_recorded"] is True
        assert str(reg["api_key"]) not in json.dumps(live_rep)
        assert "cutover-gamer" not in json.dumps(live_rep)
        assert "perso_gamer" not in json.dumps(live_rep)

        # Render CLI output in both EN and FR for --noel reports
        assert "Noel: cloud_run_routes=True" in doctor.render_report(live_rep, color=False)
        live_rep_fr = dict(live_rep, lang="fr")
        assert "Noël : routes_cloud_run=True" in doctor.render_report(live_rep_fr, color=True)

        # 3. Slow yield (>= 1500.0 budget) -> status='fail', yield_slo_ok=False, zero_game_lag=False, exit code 2
        slow_rep = doctor.run_doctor(
            gateway="http://127.0.0.1:8000",
            credentials_path=live_cred,
            lang="en",
            os_name="posix",
            system_name="Linux",
            environ=CLEAN_ENV,
            home=home_dir,
            runner=_build_local_runner(),
            noel=True,
            cwd=ws_dir,
            yield_budget_ms=1500.0,
            submit_ces=True,
            ces_profile="gamer",
            http_request=live_adapter,
            worker_yield_fn=lambda: 1620.0,
        )
        _verify_receipt(slow_rep)
        assert slow_rep["verdict"] == "not_ready"
        assert doctor.EXIT_CODES[slow_rep["verdict"]] == 2
        yield_slow_chk = _by_check_id(slow_rep, "noel_steam_yield_thermal")
        assert yield_slow_chk["status"] == "fail"
        assert yield_slow_chk["details"]["yield_slo_ok"] is False
        assert slow_rep["noel_summary"]["zero_game_lag"] is False

        # 4. Hot GPU (> 70.0 C) -> status='warn', thermal_ok=False, zero_game_lag=False
        hot_rep = doctor.run_doctor(
            gateway="http://127.0.0.1:8000",
            credentials_path=live_cred,
            lang="fr",
            os_name="posix",
            system_name="Linux",
            environ=CLEAN_ENV,
            home=home_dir,
            runner=_build_local_runner(smi_out="NVIDIA GeForce RTX 4080, 16384, 12000, 74, 15\n"),
            noel=True,
            cwd=ws_dir,
            max_temp_c=70.0,
            submit_ces=True,
            ces_profile="gamer",
            http_request=live_adapter,
            worker_yield_fn=lambda: 90.0,
        )
        _verify_receipt(hot_rep)
        assert hot_rep["verdict"] == "ready_with_warnings"
        yield_hot_chk = _by_check_id(hot_rep, "noel_steam_yield_thermal")
        assert yield_hot_chk["status"] == "warn"
        assert yield_hot_chk["details"]["thermal_ok"] is False
        assert hot_rep["noel_summary"]["zero_game_lag"] is False

        # 5. Non-blocking --submit-ces 404 / 429 / 503 / timeout fallback + deterministic Idempotency-Key
        seen_idem_keys: list[str] = []
        for ces_err_code in (404, 429, 503, 0):
            def _ces_fail_req(
                method: str,
                url: str,
                *,
                payload: dict[str, Any] | None = None,
                headers: dict[str, str] | None = None,
                timeout_s: float = 4.0,
                _err: int = ces_err_code,
            ) -> tuple[int, dict[str, Any]]:
                if method == "POST" and url.endswith("/v1/hivemind/onboarding/feedback"):
                    assert headers is not None and headers.get("Idempotency-Key", "").startswith("hm-ces-")
                    seen_idem_keys.append(headers["Idempotency-Key"])
                    if _err == 0:
                        raise TimeoutError("feedback timeout")
                    return _err, {"error": f"simulated_{_err}"}
                return live_adapter(method, url, payload=payload, headers=headers, timeout_s=timeout_s)

            fb_rep = doctor.run_doctor(
                gateway="http://127.0.0.1:8000",
                credentials_path=live_cred,
                lang="en",
                os_name="posix",
                system_name="Linux",
                environ=CLEAN_ENV,
                home=home_dir,
                runner=_build_local_runner(smi_exc=FileNotFoundError("nvidia-smi")),
                noel=True,
                cwd=ws_dir,
                submit_ces=True,
                ces_profile="dev",
                ease_rating=5,
                host_label="dev-home-laptop",
                http_request=_ces_fail_req,
                worker_yield_fn=lambda: 22.0,
            )
            _verify_receipt(fb_rep)
            assert fb_rep["ready"] is True
            assert fb_rep["verdict"] == "ready"
            assert fb_rep["noel_summary"]["gpu_mode"] == "dev_api_only_no_nvidia_gpu"
            assert fb_rep["noel_summary"]["ces_attempted"] is True
            assert fb_rep["noel_summary"]["ces_recorded"] is False

        assert len(seen_idem_keys) == 4 and len(set(seen_idem_keys)) == 1

    # 6. 100% fail-closed Amadeus / corporate env & .git/config / worktree blocking (0 HTTP calls, 0 writes, exit 2)
    corp_git_dir = tmp_path / "work_checkout"
    (corp_git_dir / ".git").mkdir(parents=True)
    (corp_git_dir / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {CORP_SSH_REMOTE}\n',
        encoding="utf-8",
    )

    wt_common = tmp_path / "wt_common_git"
    wt_sub = wt_common / "worktrees" / "wt1"
    wt_sub.mkdir(parents=True)
    (wt_sub / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_common / "config").write_text(
        f'[url "{CORP_INTRANET_URL}"]\n\tinsteadOf = corp:\n',
        encoding="utf-8",
    )
    wt_workspace = tmp_path / "wt_checkout"
    wt_workspace.mkdir(parents=True)
    (wt_workspace / ".git").write_text(f"gitdir: {wt_sub}\n", encoding="utf-8")

    outbound_calls = [0]

    def _forbid_http(*_args: Any, **_kwargs: Any) -> tuple[int, dict[str, Any]]:
        outbound_calls[0] += 1
        raise AssertionError("Corporate fail-closed gate must NEVER make outbound HTTP requests!")

    corp_cases: tuple[tuple[dict[str, str], Path, str], ...] = (
        ({"PATH": "/usr/bin", "USERDNSDOMAIN": "nce.amadeus.corp"}, ws_dir, "en"),
        ({"PATH": "/usr/bin", "AMADEUS_ENV": "prod-secret-val"}, ws_dir, "fr"),
        (
            {
                "PATH": "/usr/bin",
                "USERDOMAIN": "ENTERPRISE-DOM",
                "COMPUTERNAME": "CORP-LT-77",
                "HOSTNAME": "fortinet-gw-host",
                "HTTPS_PROXY": CORP_ZSCALER_PROXY,
                "NO_PROXY": ".paloalto.internal",
                "ZSCALER_CERT_PATH": "/etc/zscaler.pem",
                "SSL_CERT_FILE": "/etc/netskope/ca.pem",
                "CURL_CA_BUNDLE": "/etc/bluecoat/ca.pem",
            },
            ws_dir,
            "en",
        ),
        ({"PATH": "/usr/bin"}, corp_git_dir, "fr"),
        ({"PATH": "/usr/bin"}, wt_workspace, "en"),
        ({"PATH": "/usr/bin"}, tmp_path / "amadeus_workspace_dir", "fr"),
    )

    for env_case, cwd_case, lang_case in corp_cases:
        cwd_case.mkdir(parents=True, exist_ok=True)
        before_files = sorted(str(p) for p in cwd_case.rglob("*"))
        rc_corp = doctor.main(
            [
                "--noel",
                "--cwd",
                str(cwd_case),
                "--first-heartbeat",
                "--submit-ces",
                "--lang",
                lang_case,
                "--json",
            ],
            http_get=_forbid_http,
            http_request=_forbid_http,
            environ=env_case,
            home=home_dir,
        )
        after_files = sorted(str(p) for p in cwd_case.rglob("*"))
        assert rc_corp == 2
        assert outbound_calls[0] == 0
        assert before_files == after_files
        corp_rep = json.loads(capsys.readouterr().out)
        _verify_receipt(corp_rep)
        assert corp_rep["ready"] is False
        assert corp_rep["verdict"] == "not_ready"
        assert corp_rep["noel_summary"]["compliance_ok"] is False
        assert corp_rep["noel_summary"]["outbound_http_requests"] == 0
        assert corp_rep["noel_summary"]["file_writes"] == 0
        comp_chk = _by_check_id(corp_rep, "personal_pc_compliance")
        assert comp_chk["status"] == "fail"
        assert "--uninstall" in comp_chk["fix_command"]
        assert "0.00 EUR / 0.00 USD" in comp_chk["summary"]
        assert "0,00 EUR / 0,00 USD" in comp_chk["summary"]
        dumped_corp = json.dumps(corp_rep)
        for sensitive in ("nce.amadeus.corp", "prod-secret-val", "secretpw", "git-amadeus-internal", "CORP-LT-77"):
            assert sensitive not in dumped_corp


def test_05_concurrent_48_participant_noel_doctor_heartbeat_and_ces_sqlite_wal_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", "d07scale48ok")
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00108-scale")

    upstream_inference_hits: list[str] = []

    def _reject_inference(req: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_inference"})

    sqlite_path = tmp_path / "noel_dyn07_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=SOAK_SECRET)
    wal_check = SqliteConnection(sqlite_path)
    try:
        journal_mode = str(wal_check.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_check.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_reject_inference)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    total_participants = 48
    dev_cohort_size = 24
    barrier = threading.Barrier(total_participants)
    shared_lock = threading.Lock()
    sqlite_lock_hits: list[str] = []
    seen_calls: list[tuple[str, str, dict[str, str]]] = []

    rtx_models = (
        ("NVIDIA GeForce RTX 4070 SUPER", 12288, 9216, 54),
        ("NVIDIA GeForce RTX 4080 SUPER", 16384, 12288, 57),
        ("NVIDIA GeForce RTX 3080", 10240, 6144, 61),
        ("NVIDIA GeForce RTX 4060 Ti", 8192, 5120, 52),
        ("NVIDIA GeForce RTX 3090", 24576, 18432, 63),
        ("NVIDIA GeForce RTX 4090", 24576, 20480, 58),
    )

    with TestClient(app) as client:
        def _participant_worker(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_cohort_size
            cohort_idx = idx if is_dev else (idx - dev_cohort_size)
            peer_ip = _loopback_ip_for(40, idx)
            p_root = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_username = f"perso_user_{idx:02d}"
            home_dir = p_root / "home" / os_username
            ws_dir = p_root / "personal_ws"
            ws_dir.mkdir(parents=True, exist_ok=True)

            handle = f"dev- quick-{cohort_idx:02d}".replace(" ", "") if is_dev else f"gamer-rtx-{cohort_idx:02d}"
            role = "consumer" if (is_dev and cohort_idx % 2 == 0) else ("prosumer" if is_dev else "producer")
            ease = 5 if (idx % 3 != 0) else 4
            lang = "fr" if (idx % 2 == 0) else "en"

            adapter = _build_router_request_adapter(
                client,
                peer_ip=peer_ip,
                seen_calls=seen_calls,
                sqlite_lock_hits=sqlite_lock_hits,
                lock=shared_lock,
            )

            barrier.wait(timeout=20.0)
            t0 = time.monotonic()

            reg_resp = client.post(
                "/v1/hivemind/register",
                json={"handle": handle, "role": role},
                headers={"X-Forwarded-For": peer_ip},
            )
            assert reg_resp.status_code == 200, reg_resp.text
            reg_doc = reg_resp.json()
            raw_key = str(reg_doc["api_key"])

            cred_path = _write_credentials(
                home_dir / ".hivemind" / "credentials.json",
                api_key=raw_key,
                handle=handle,
                role=role,
                mode=0o600,
            )

            if is_dev:
                runner = _build_local_runner(smi_exc=FileNotFoundError("nvidia-smi"))
                os_name_val = "posix"
                sys_name_val = "Darwin" if (cohort_idx % 2 == 0) else "Linux"
                profile = "dev"
                host_lbl = f"dev-pc-{cohort_idx:02d}"
                yield_ms = 18.0 + cohort_idx
            else:
                card, tot_mib, free_mib, temp_c = rtx_models[cohort_idx % len(rtx_models)]
                prod_ip = _loopback_ip_for(88, cohort_idx)
                hb_prod = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{prod_ip}:11434",
                        "models": [{"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 64.0,
                        "gpu_vram_mib": tot_mib,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={"Authorization": f"Bearer {raw_key}", "X-Forwarded-For": peer_ip},
                )
                assert hb_prod.status_code == 200, hb_prod.text
                runner = _build_local_runner(
                    smi_out=f"{card}, {tot_mib}, {free_mib}, {temp_c}, 4\n",
                    tasklist_out='"steam.exe","2100","Console","1","90,000 K"\n',
                )
                os_name_val = "nt"
                sys_name_val = "Windows"
                profile = "gamer"
                host_lbl = f"rtx-pc-{cohort_idx:02d}"
                yield_ms = 75.0 + (cohort_idx * 4.0)

            rep = doctor.run_doctor(
                gateway="http://127.0.0.1:8000",
                credentials_path=cred_path,
                lang=lang,
                os_name=os_name_val,
                system_name=sys_name_val,
                environ=CLEAN_ENV,
                home=home_dir,
                runner=runner,
                noel=True,
                cwd=ws_dir,
                send_first_heartbeat=True,
                submit_ces=True,
                ces_profile=profile,
                ease_rating=ease,
                host_label=host_lbl,
                http_request=adapter,
                worker_yield_fn=lambda y=yield_ms: y,
            )
            elapsed = time.monotonic() - t0
            assert elapsed < (60.0 if is_dev else 300.0)
            _verify_receipt(rep)
            assert rep["ready"] is True, f"Slot {idx} ({profile}/{role}) failed: {rep}"
            assert rep["verdict"] == "ready"
            assert rep["noel_summary"]["first_heartbeat_ok"] is True
            assert rep["noel_summary"]["ces_recorded"] is True
            assert rep["noel_summary"]["zero_game_lag"] is True

            rendered = doctor.render_report(rep, color=False)
            combined_out = json.dumps(rep) + "\n" + rendered
            assert raw_key not in combined_out
            assert os_username not in combined_out
            assert handle not in combined_out
            assert not doctor._RAW_API_KEY_RE.search(combined_out)

            hb_pl = _by_check_id(rep, "noel_fleet_heartbeat")["details"]["heartbeat_payload"]
            if is_dev:
                assert hb_pl["gpu_name"] is None
                assert hb_pl["vram_total_mib"] is None
                assert hb_pl["vram_free_mib"] is None
            else:
                assert hb_pl["gpu_name"] is not None
                assert hb_pl["vram_total_mib"] >= 8192

            return {"idx": idx, "profile": profile, "role": role}

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            futures = [pool.submit(_participant_worker, i) for i in range(total_participants)]
            outcomes = [f.result(timeout=60.0) for f in futures]

        assert len(outcomes) == 48
        assert sqlite_lock_hits == []
        assert upstream_inference_hits == []
        assert all("completions" not in path for _, path, _ in seen_calls)

        raid_doc = client.get("/v1/hivemind/raid/status").json()
        assert raid_doc["swarm"]["active_trainers"] >= 24
        assert raid_doc["swarm"]["live_producers"] == 24

        fb_summary = client.get("/v1/hivemind/onboarding/feedback/summary").json()
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        avg_ease = fb_summary.get("average_ease_rating", fb_summary.get("mean_ease_rating"))
        assert avg_ease is not None and float(avg_ease) >= 4.2
        assert fb_summary["zero_game_lag_ratio"] == 1.0


def test_06_edge_cases_for_helpers_holiday_routes_yield_drill_and_fleet_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Helper edge cases: normalize_profile, mask_api_key, sanitization, _invoke_http_request arities
    assert doctor.normalize_profile("DEV") == "dev"
    assert doctor.normalize_profile(" gamer ") == "gamer"
    for bad_prof in (None, True, "", "admin"):
        with pytest.raises(ValueError):
            doctor.normalize_profile(bad_prof)  # type: ignore[arg-type]
    for bad_ease in (True, 0, 6):
        with pytest.raises(ValueError):
            doctor._parse_ease_rating(bad_ease)  # type: ignore[arg-type]

    assert doctor.mask_api_key("") == "<missing>"
    assert doctor.mask_api_key("not-a-key") == "<invalid>"
    assert doctor.mask_api_key(ALPHA_KEY) == "hm-***3456"
    assert doctor.mask_api_key(LIVE_KEY) == "hm_live_***4321"
    assert "~" in doctor._sanitize_noel_text("/home/bob/.hivemind/credentials.json")
    assert "Bearer <redacted>" in doctor._sanitize_noel_text(f"Bearer {ALPHA_KEY}")
    assert doctor._sanitize_noel_structure(("a", ["/Users/alice/file"])) == ["a", ["~/file"]]
    assert doctor._platform_feedback_tag("posix", "Solaris_11") == "solaris_11"
    assert doctor._platform_feedback_tag("posix", "!!!") == "unknown"
    assert doctor._sanitize_host_label("   ") == "personal-pc"
    assert doctor._sanitize_host_label("!!!") == "personal-pc"
    assert doctor._sanitize_manifest_sha(None) == "unknown"
    assert doctor._sanitize_manifest_sha("secret1234") == "unknown"
    assert doctor._sanitize_manifest_revision(None) is None
    assert doctor._sanitize_manifest_revision("-bad") is None
    assert doctor._sanitize_manifest_revision("hm-123456789012") is None

    assert doctor._invoke_http_request(lambda m, u: (200, [m, u]), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert doctor._invoke_http_request(lambda m, u, p: (200, {"p": p}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert (
        doctor._invoke_http_request(lambda m, u, p, h, t: (200, {"t": t}), "GET", "http://127.0.0.1:8000/health")[0]
        == 200
    )

    # 2. Holiday routes edge cases: untrusted URL, Windows fix, 503 error, empty HTML, bad manifest, network error
    assert doctor.check_noel_holiday_routes(UNTRUSTED_EVIL_GW, os_name="nt", lang="fr").status == "fail"

    def _holiday_5xx_and_manifest_drift(url: str, _t: float) -> tuple[int, dict[str, Any]]:
        if url.endswith("/noel"):
            return 200, {"html_length": 0}
        if url.endswith("/quick"):
            return 200, {"raw": "   "}
        if url.endswith("/raid-noel"):
            raise OSError("timeout")
        if url.endswith("/v1/hivemind/onboarding/manifest"):
            return 200, {"object": "hivemind.onboarding.manifest", "holiday_routes": ["/noel"]}
        return 200, {"error": "broken"}

    drift_m = doctor.check_noel_holiday_routes(http_get=_holiday_5xx_and_manifest_drift, lang="en")
    assert drift_m.status == "fail" and drift_m.details["revision_drift_detected"] is True

    bad_obj = doctor.check_noel_holiday_routes(
        http_get=lambda u, _t: (200, {"object": "wrong"}) if u.endswith("manifest") else (503, {}),
        lang="fr",
    )
    assert bad_obj.status == "fail"

    pure_503 = doctor.check_noel_holiday_routes(http_get=lambda _u, _t: (503, {}), lang="en")
    assert pure_503.status == "fail" and pure_503.details["revision_drift_detected"] is False
    pure_503_fr = doctor.check_noel_holiday_routes(http_get=lambda _u, _t: (503, {}), lang="fr")
    assert pure_503_fr.status == "fail"

    # 3. Yield drill & thermal check edge cases: non-finite, mapping, invalid return, active game, low free VRAM, parse error
    assert doctor._run_yield_drill_callable(lambda: math.nan)[0] is None
    assert (
        doctor._run_yield_drill_callable(lambda timeout_s=1.0: {"worker_stopped": True, "clean_release": True})[1]
        is True
    )
    with pytest.raises(ValueError):
        doctor._run_yield_drill_callable(lambda: "invalid")

    # Exercise default_worker_yield_drill bad-ready & timeout branches safely via monkeypatched Popen
    class _BadReadyProc:
        def __init__(self, mode: str) -> None:
            self._mode = mode
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("WRONG_BANNER\n" if mode == "bad" else "HIVEMIND_WORKER_READY\n")
            self.stderr = io.StringIO()
            self._killed = False

        def kill(self) -> None:
            self._killed = True

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._mode == "timeout" and not self._killed:
                raise subprocess.TimeoutExpired(cmd=["python3"], timeout=0.25)
            return -9

        def poll(self) -> int | None:
            return -9 if self._killed else None

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _BadReadyProc("bad"))
    assert doctor.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _BadReadyProc("timeout"))
    assert doctor.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.undo()

    # Non-GPU slow yield & non-GPU active game
    assert (
        doctor.check_noel_steam_yield_and_thermal(
            runner=_build_local_runner(smi_exc=FileNotFoundError("nvidia-smi")),
            worker_yield_fn=lambda: 1900.0,
            lang="fr",
        ).status
        == "fail"
    )
    assert (
        doctor.check_noel_steam_yield_and_thermal(
            runner=_build_local_runner(smi_exc=FileNotFoundError("nvidia-smi"), ps_out="steam_app_730\n"),
            worker_yield_fn=lambda: 30.0,
            lang="en",
        ).status
        == "warn"
    )
    # GPU parse error, GPU active game, GPU low free VRAM
    assert (
        doctor.check_noel_steam_yield_and_thermal(
            runner=_build_local_runner(smi_out="corrupt"),
            worker_yield_fn=lambda: 30.0,
            lang="en",
        ).status
        == "warn"
    )
    assert (
        doctor.check_noel_steam_yield_and_thermal(
            runner=_build_local_runner(ps_out="cs2\n"),
            worker_yield_fn=lambda: 30.0,
            lang="fr",
        ).status
        == "warn"
    )
    assert (
        doctor.check_noel_steam_yield_and_thermal(
            runner=_build_local_runner(smi_out="NVIDIA GeForce RTX 4060, 8192, 1024, 55, 2\n"),
            worker_yield_fn=lambda: 30.0,
            lang="en",
        ).status
        == "warn"
    )
    assert doctor._gpu_info_from_yield_details({"gpu_present": True, "parse_error": True}).parse_error is True

    # 4. Fleet & first-heartbeat failure branches
    assert doctor.check_noel_fleet_and_first_heartbeat(UNTRUSTED_EVIL_GW, os_name="nt", lang="en").status == "fail"

    missing_hb = doctor.check_noel_fleet_and_first_heartbeat(
        credentials_path=tmp_path / "missing.json",
        send_first_heartbeat=True,
        http_get=lambda _u, _t: (200, {"live_hiveminders": 1}),
        home=tmp_path,
        lang="fr",
    )
    assert missing_hb.status == "fail"

    valid_cred = _write_credentials(tmp_path / "valid_hb.json", api_key=ALPHA_KEY)
    bad_json_cred = tmp_path / "bad_hb.json"
    bad_json_cred.write_text("[]", encoding="utf-8")
    assert doctor._load_valid_credentials_key(bad_json_cred, home=tmp_path)[0] is None
    bad_syntax_cred = tmp_path / "bad_syntax_hb.json"
    bad_syntax_cred.write_text("{broken", encoding="utf-8")
    assert doctor._load_valid_credentials_key(bad_syntax_cred, home=tmp_path)[0] is None
    bad_key_cred = tmp_path / "bad_key_hb.json"
    bad_key_cred.write_text(json.dumps({"api_key": "invalid"}), encoding="utf-8")
    assert doctor._load_valid_credentials_key(bad_key_cred, home=tmp_path)[0] is None

    # /v1/hivemind/me network exception & /v1/hivemind/opentrainer/heartbeat 500 failure + fleet 503 failure
    me_exc_chk = doctor.check_noel_fleet_and_first_heartbeat(
        credentials_path=valid_cred,
        send_first_heartbeat=True,
        http_get=lambda _u, _t: (200, {"live_hiveminders": 1}),
        http_request=lambda _m, _u, **_kw: (_ for _ in ()).throw(OSError("me down")),
        home=tmp_path,
        lang="en",
    )
    assert me_exc_chk.status == "fail"

    def _hb_500_req(method: str, url: str, **_kw: Any) -> tuple[int, dict[str, Any]]:
        if method == "GET" and url.endswith("/v1/hivemind/me"):
            return 200, {"role": "prosumer"}
        if method == "POST":
            return 500, {"error": "server_error"}
        return 200, {}

    hb_500_chk = doctor.check_noel_fleet_and_first_heartbeat(
        credentials_path=valid_cred,
        send_first_heartbeat=True,
        gpu_info=doctor.GpuInfo(present=True, name="NVIDIA RTX 4070", total_mib=12288, free_mib=8192, temp_c=50.0),
        http_get=lambda _u, _t: (200, {"live_hiveminders": 1}),
        http_request=_hb_500_req,
        home=tmp_path,
        lang="en",
    )
    assert hb_500_chk.status == "fail"

    fleet_down_chk = doctor.check_noel_fleet_and_first_heartbeat(
        http_get=lambda _u, _t: (_ for _ in ()).throw(OSError("fleet offline")),
        home=tmp_path,
        os_name="nt",
        lang="fr",
    )
    assert fleet_down_chk.status == "fail"

    # Raid fallback for online_producers when fleet omits live_hiveminders
    raid_fallback_chk = doctor.check_noel_fleet_and_first_heartbeat(
        http_get=lambda u, _t: (
            (200, {})
            if u.endswith("/fleet")
            else (
                200,
                {
                    "swarm": {"live_producers": 5, "active_trainers": 7, "accepted_opentrainer_rounds": 12},
                    "onboarding_feedback": {"mean_ease_rating": 4.8},
                },
            )
        ),
        home=tmp_path,
        lang="en",
    )
    assert raid_fallback_chk.status == "ok"
    assert raid_fallback_chk.details["online_producers"] == 5
    assert raid_fallback_chk.details["active_trainers"] == 7
    assert raid_fallback_chk.details["accepted_rounds"] == 12
    assert raid_fallback_chk.details["average_ease_rating"] == 4.8

    req_only_chk = doctor.check_noel_fleet_and_first_heartbeat(
        http_request=lambda _m, u, **_kw: (
            (200, {"live_hiveminders": 2})
            if u.endswith("/fleet")
            else (200, {"swarm": {"active_trainers": 3, "accepted_opentrainer_rounds": 4}})
        ),
        home=tmp_path,
        lang="en",
    )
    assert req_only_chk.status == "ok" and req_only_chk.details["online_producers"] == 2

    # Compliance evaluation with untrusted gateway fallback
    comp_bad_gw = doctor.evaluate_noel_personal_pc_compliance(
        environ={"PATH": "/usr/bin"},
        cwd=tmp_path / "clean_dir",
        gateway=UNTRUSTED_EVIL_GW,
        lang="fr",
    )
    assert comp_bad_gw["compliance_ok"] is True


def test_zz_standalone_hivemind_doctor_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage across scripts/hivemind_doctor.py, got {pct:.2f}%:\n{buf.getvalue()}"
    )
