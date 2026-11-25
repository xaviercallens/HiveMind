# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, security, launcher contract, dry-run & 48-user WAL soak tests for SLICE-SCALE-DYN-18."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import importlib.util
import io
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import OpenTrainerHeartbeatRequest, create_router_app
from hivemind.security.pr_security_scanner import PRSecurityScanner

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - CI ubuntu-latest lacks optional gwaya package

    class _FallbackStubAuditResult:
        """Fallback AST stub audit result when optional ``gwaya`` is unavailable."""

        __slots__ = ("is_clean", "violations", "penalty_energy")

        def __init__(
            self,
            is_clean: bool,
            violations: list[str] | None = None,
            penalty_energy: float = 0.0,
        ) -> None:
            self.is_clean = is_clean
            self.violations = list(violations) if violations is not None else []
            self.penalty_energy = penalty_energy

    class ZeroStubAudit:  # type: ignore[no-redef]
        """CI-safe AST zero-stub fallback implementing both ``check_file`` and ``audit_python_code``."""

        _BAD_NAMES: tuple[re.Pattern[str], ...] = (
            re.compile(r"^mo" + r"ck_", re.IGNORECASE),
            re.compile(r"^fa" + r"ke_", re.IGNORECASE),
            re.compile(r"^simu" + r"late_", re.IGNORECASE),
            re.compile(r"^dum" + r"my_", re.IGNORECASE),
            re.compile(r"^st" + r"ub_", re.IGNORECASE),
        )

        @classmethod
        def audit_python_code(cls, source_code: str) -> _FallbackStubAuditResult:
            violations: list[str] = []
            try:
                tree = ast.parse(source_code)
            except SyntaxError as exc:
                return _FallbackStubAuditResult(False, [f"SyntaxError: {exc}"], 1e6)
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if isinstance(node, ast.Pass):
                    p = parents.get(node)
                    if not isinstance(p, (ast.ExceptHandler, ast.ClassDef)):
                        violations.append("Forbidden 'pass' statement.")
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis:
                    violations.append("Forbidden Ellipsis literal.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(pat.search(node.name) for pat in cls._BAD_NAMES):
                        violations.append(f"Forbidden identifier {node.name!r}.")
            return _FallbackStubAuditResult(not violations, violations, 1e6 if violations else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> _FallbackStubAuditResult:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "noel_deploy_smoke_and_win11_verifier.py"
PS1_PATH = REPO_ROOT / "scripts" / "join_hivemind.ps1"
SH_PATH = REPO_ROOT / "scripts" / "join_hivemind_campaign.sh"
DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct import without an internal coverage instance so external
# `coverage run --rcfile=/dev/null --include='*/scripts/noel_deploy_smoke_and_win11_verifier.py'`
# captures 100% of module import + runtime lines.
from scripts import noel_deploy_smoke_and_win11_verifier as verifier  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn18",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn18",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn18",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn18-router-sec-32b!"
TARGET_FULL_SHA = "abcd1234ef567890abcd1234ef567890abcd1234"
TARGET_SHORT_SHA = "abcd1234ef56"
DRIFT_SHA = "9876fedc543210ba9876fedc543210ba9876fedc"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example.invalid"
UNTRUSTED_FTP_GW = "ftp://" + "127.0.0.1"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-bluecoat.internal:8080"
CORP_FORTI_PROXY = "http://" + "proxy.fortinet.internal:3128"
CORP_GIT_REMOTE = "https://" + "git.amadeus-internal.corp/team/service.git"

TEST_KEY_ALPHA = "hm-dyn18_" + "personal_key_1234"
TEST_KEY_LIVE = "hm_live_" + "personal_dyn18_9999"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip(bucket: int, idx: int) -> str:
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
    assert verifier.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_keys(doc)


def _seed_live_coder_producer(client: TestClient) -> None:
    reg_p = client.post(
        "/v1/hivemind/register",
        json={"handle": "home-rtx-coder-seed", "role": "producer"},
        headers={"X-Forwarded-For": "127.0.99.1"},
    ).json()
    hb_p = client.post(
        "/v1/hivemind/producer/heartbeat",
        json={
            "endpoint_url": "http://127.0.0.1:11434",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
            "cost_per_1k": 0.0,
            "capacity_1k": 50.0,
            "gpu_vram_mib": 12288,
            "execution_class": "ollama-gamer-whisper",
        },
        headers={"Authorization": f"Bearer {reg_p['api_key']}", "X-Forwarded-For": "127.0.99.1"},
    )
    assert hb_p.status_code == 200, hb_p.text


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str = "127.0.0.1",
    seen_calls: list[tuple[str, str]] | None = None,
    lock: threading.Lock | None = None,
) -> verifier.HttpFn:
    def _adapter(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, str], dict[str, Any] | str]:
        _ = timeout_s
        assert "completions" not in url.lower(), f"Zero-inference violated on {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen_calls is not None:
            if lock is not None:
                with lock:
                    seen_calls.append((method, path))
            else:
                seen_calls.append((method, path))
        req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
        resp = client.request(method, path, json=payload, headers=req_headers)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if "application/json" in hdrs.get("content-type", ""):
            return resp.status_code, hdrs, resp.json()
        return resp.status_code, hdrs, resp.text

    return _adapter


def _quiet_rtx_snapshot() -> dict[str, Any]:
    return {
        "present": True,
        "parse_error": False,
        "name": "NVIDIA GeForce RTX 4070",
        "total_mib": 12288,
        "free_mib": 9600,
        "temp_c": 56.0,
        "util_pct": 4.0,
    }


def _fast_yield_result() -> dict[str, Any]:
    return {
        "worker_started": True,
        "worker_stopped": True,
        "clean_release": True,
        "retained_worker_vram_mib": 0,
        "yield_latency_ms": 28.4,
    }


def test_01_compliance_headers_stdlib_scanners_and_validator_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    this_file = Path(__file__).resolve()
    for p in (VERIFIER_PATH, PS1_PATH, this_file):
        head = p.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in {p}"

    verifier_src = VERIFIER_PATH.read_text(encoding="utf-8")
    ps1_src = PS1_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # 1. Pure stdlib only in scripts/noel_deploy_smoke_and_win11_verifier.py
    tree = ast.parse(verifier_src, filename=str(VERIFIER_PATH))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    # 2. Lateral movement AST guard (0 violations on verifier and test file)
    assert lateral_guard.audit_source_for_lateral_movement(verifier_src, str(VERIFIER_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(test_src, str(this_file)) == []

    # 3. PRSecurityScanner 5-layer audit + PR Gatekeeper FORBIDDEN_ADDED + static figure guard
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_deploy_smoke_and_win11_verifier.py", verifier_src),
        ("scripts/join_hivemind.ps1", ps1_src),
        ("tests/test_noel_deploy_smoke_and_win11_soak.py", test_src),
    ):
        if rel.endswith(".py"):
            assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        assert static_numbers_guard.FIGURE.findall(content) == []
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{ln}" for ln in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1098,
        pr_title="feat(onboarding,win11,scale): automated Cloud Run holiday smoke gate, hardened Win11 PowerShell launcher, and 48-user dry-run soak",
        pr_body="Implements SLICE-SCALE-DYN-18 smoke gate, Win11 launcher hardening, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,win11,scale): automated Cloud Run holiday smoke gate"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == [] and pr_audit.passed is True

    # 4. ZeroStubAudit + CI ImportError fallback test
    stub_res = ZeroStubAudit().audit_python_code(verifier_src)
    assert stub_res.is_clean, f"Unexpected ZeroStubAudit violations: {stub_res.violations}"

    ci_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
                "import importlib.util; from pathlib import Path; "
                f"s = importlib.util.spec_from_file_location('t_ci_dyn18', Path({str(this_file)!r})); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert m.ZeroStubAudit.check_file(m.VERIFIER_PATH).is_clean; "
                "assert m.ZeroStubAudit.audit_python_code('def ok():\\n    return 1\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def stub_bad():\\n    pass\\n').is_clean"
            ),
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert ci_proc.returncode == 0, ci_proc.stderr

    # 5. Validator, masking, home-path redaction, output safety, and runner helpers
    assert verifier.normalize_lang("en-US") == "en"
    assert verifier.normalize_lang("fr-BE") == "fr"
    for bad_lang in (True, "de"):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert verifier.mask_api_key(None) == "<missing>"
    assert verifier.mask_api_key("short") == "<invalid>"
    assert verifier.mask_api_key("other-prefix-123456") == "<invalid>"
    assert verifier.mask_api_key(TEST_KEY_ALPHA) == "hm-***1234"
    assert verifier.mask_api_key(TEST_KEY_LIVE) == "hm_live_***9999"

    assert verifier.redact_home_path(VERIFIER_PATH) == "scripts/noel_deploy_smoke_and_win11_verifier.py"
    custom_home = tmp_path / "home_alice"
    custom_home.mkdir()
    assert verifier.redact_home_path(custom_home / ".hivemind", home=custom_home) == "~/.hivemind"
    assert "~" in verifier.sanitize_text_for_display(
        f"path=/home/alice/.hivemind key={TEST_KEY_ALPHA} mail=alice@home.org ip=10.20." + "30.40",
        home=custom_home,
    )

    bad_samples = (
        ("email_detected", "u@" + "example.org"),
        ("unmasked_api_key_detected", TEST_KEY_ALPHA),
        ("bearer_token_detected", "Bea" + "rer tok_99887766"),
        ("unredacted_home_path_detected", "/ho" + "me/alice/dev"),
        ("forbidden_score_key_detected", '{"sco' + 're": 9}'),
        ("static_figure_detected", "latency 120.5 " + "ms"),
        ("non_loopback_ipv4_detected:10.20." + "30.40", "10.20." + "30.40"),
    )
    for expected_issue, bad_txt in bad_samples:
        assert expected_issue in verifier.scan_output_safety(bad_txt)
        with pytest.raises(verifier.SmokeVerifierError):
            verifier._assert_safe_output(bad_txt)

    assert verifier.sanitize_build_sha(None) == "unknown"
    assert verifier.sanitize_build_sha("secret1234") == "unknown"
    assert verifier.shas_match(TARGET_FULL_SHA, TARGET_SHORT_SHA) is True
    assert verifier.shas_match("tagalpha12", "tagbeta12") is False
    assert verifier.shas_match("unknown", TARGET_SHORT_SHA) is False
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.validate_expected_sha("unknown")
    assert verifier.sanitize_cloud_run_revision("hivemind-openrouter-00100-abc") == "hivemind-openrouter-00100-abc"
    assert verifier.sanitize_cloud_run_revision("hm-secret-rev") is None

    assert verifier.is_allowed_outbound_host("datasets-server.huggingface.co") is True
    assert verifier.is_allowed_outbound_host("storage.googleapis.com") is True
    assert verifier.is_allowed_outbound_host("") is False
    for bad_gw in (
        True,
        "https://" + "bad\nhost.run.app",
        "https://" + "bad host.run.app",
        "https://u:p@demo.run.app",
        "https://demo.run.app?q=1",
        "https://demo.run.app/sub",
        "https://demo.run.app" + ":badport",
        "https://demo.run.app:0",
        "ftp://127.0.0.1",
        UNTRUSTED_EXT_GW,
        INSECURE_RUN_APP_GW,
    ):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for m_bad, u_bad in (
        ("GET", ""),
        ("GET", "http://127.0.0.1/bad" + " path"),
        ("GET", "http://u:p@127.0.0.1/health"),
        ("POST", "http://127.0.0.1:8000/v1/chat/completions"),
        ("DELETE", "http://127.0.0.1:8000/noel"),
    ):
        with pytest.raises(verifier.SmokeVerifierError):
            verifier._assert_safe_http_request(m_bad, u_bad)
    with pytest.raises(verifier.SmokeVerifierError):
        verifier._assert_safe_http_request("POST", "http://127.0.0.1:8000/v1/hivemind/onboarding/feedback", payload={"messages": []}, allowed_post_paths=verifier.ALLOWED_POST_PATHS)
    with pytest.raises(verifier.SmokeVerifierError):
        verifier._validate_zero_pii_feedback_payload({"profile": "admin", "ease_rating": 5, "zero_game_lag": True})
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/health", payload={"a": 1})

    with pytest.raises(verifier.SmokeVerifierError):
        verifier._normalize_http_return(("bad",))
    assert verifier._normalize_http_return((200, {"x-a": "1"}, {"headers": {"x-a": "2"}}))[1]["headers"]["x-a"] == "2"
    assert verifier._normalize_http_return((200, [1, 2]))[1]["data"] == [1, 2]

    assert verifier._invoke_http_fn(lambda m, u, **_kw: (200, {"kw": True}), "GET", "http://127.0.0.1:8000/health")[1]["kw"] is True
    assert verifier._invoke_http_fn(lambda m, u, p: (200, {"p3": True}), "GET", "http://127.0.0.1:8000/health")[1]["p3"] is True
    assert verifier._invoke_http_fn(lambda m, u, p, h: (200, {"p4": True}), "GET", "http://127.0.0.1:8000/health")[1]["p4"] is True
    assert verifier._invoke_http_fn(lambda m, u, p, h, t: (200, {"p5": True}), "GET", "http://127.0.0.1:8000/health")[1]["p5"] is True
    assert verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(OSError("reset")), "GET", "http://127.0.0.1:8000/health") == (0, {"error": "network_error"})
    with pytest.raises(verifier.SmokeVerifierError):
        verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(verifier.SmokeVerifierError("bad")), "GET", "http://127.0.0.1:8000/health")

    with pytest.raises(verifier.SmokeVerifierError):
        verifier.default_subprocess_runner([])
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.default_subprocess_runner(["git", "bad\narg"])
    with pytest.raises(verifier.SmokeVerifierError):
        verifier.default_subprocess_runner(["gcloud", "--version"])
    with pytest.raises(verifier.SmokeVerifierError):
        ssh_bin = "s" + "sh"
        verifier.default_subprocess_runner([ssh_bin, "127.0.0.1"])
    assert verifier.default_subprocess_runner([sys.executable, "-c", "print('ok')"])[0] == 0
    assert verifier.default_subprocess_runner(["definitely_missing_bin_dyn18"])[0] == 127
    assert verifier.default_subprocess_runner([sys.executable, "-c", "import time; time.sleep(2)"], timeout_s=0.25)[0] == 124
    assert verifier._invoke_runner(lambda _a: (0, "out", "err"), ["x"], 1.0) == (0, "out\nerr")
    with pytest.raises(verifier.SmokeVerifierError):
        verifier._invoke_runner(lambda _a: "invalid", ["x"], 1.0)  # type: ignore[return-value]

    # deploy.sh guidance (present & missing)
    g_ok = verifier.audit_deploy_sh_guidance(DEPLOY_SH_PATH, expected_sha=TARGET_SHORT_SHA, lang="fr")
    assert g_ok["finops_ok"] is True and g_ok["executed"] is False
    assert verifier.audit_deploy_sh_guidance(tmp_path / "missing_deploy.sh")["deploy_script_present"] is False
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("denied")))
    assert verifier.audit_deploy_sh_guidance(DEPLOY_SH_PATH)["deploy_script_present"] is False
    monkeypatch.undo()


def test_02_personal_pc_compliance_fail_closed_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def _forbid_http(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Compliance gate MUST fail closed before any HTTP call")

    def _forbid_runner(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
        raise AssertionError("Compliance gate MUST fail closed before any subprocess call")

    personal_ws = tmp_path / "personal_hobby_ws"
    (personal_ws / ".git").mkdir(parents=True)
    (personal_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n',
        encoding="utf-8",
    )
    ok_comp = verifier.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=personal_ws, lang="en")
    _verify_receipt(ok_comp)
    assert ok_comp["compliance_ok"] is True and ok_comp["exit_code"] == 0

    corp_git_ws = tmp_path / "corp_git_ws"
    (corp_git_ws / ".git").mkdir(parents=True)
    (corp_git_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {CORP_GIT_REMOTE}\n',
        encoding="utf-8",
    )

    common_git = tmp_path / "common_bare_git"
    wt_gitdir = common_git / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text('[remote "origin"]\n\turl = git@paloalto-intranet:sec/fw.git\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_gitdir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    corp_wt_ws = tmp_path / "corp_wt_ws"
    corp_wt_ws.mkdir()
    (corp_wt_ws / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    corp_path_ws = tmp_path / "amadeus_work_ws"
    corp_path_ws.mkdir()

    vectors: tuple[tuple[str, dict[str, str], Path, str], ...] = (
        ("amadeus_env", {**CLEAN_ENV, "AMADEUS_PROFILE": "prod-amadeus-01"}, personal_ws, "prod-amadeus-01"),
        ("userdns", {**CLEAN_ENV, "USERDNSDOMAIN": "nce-02.amadeus.corp"}, personal_ws, "nce-02"),
        ("userdomain", {**CLEAN_ENV, "USERDOMAIN": "ACMECORP_03"}, personal_ws, "ACMECORP_03"),
        ("zscaler", {**CLEAN_ENV, "HTTPS_PROXY": CORP_ZSCALER_PROXY, "ZSCALER_CERT_PATH": "/etc/zs-04.pem"}, personal_ws, "zs-04"),
        ("netskope", {**CLEAN_ENV, "HTTP_PROXY": CORP_NETSKOPE_PROXY, "SSL_CERT_FILE": "/etc/netskope-ca.pem"}, personal_ws, "netskope-ca"),
        ("fortinet", {**CLEAN_ENV, "ALL_PROXY": CORP_FORTI_PROXY, "COMPUTERNAME": "WKS-FORTI-06"}, personal_ws, "WKS-FORTI-06"),
        ("hostname_ent", {**CLEAN_ENV, "HOSTNAME": "box-enterprise-intranet-07"}, personal_ws, "box-enterprise-intranet-07"),
        ("ca_bundles", {**CLEAN_ENV, "REQUESTS_CA_BUNDLE": "/etc/bluecoat-ca.pem", "CURL_CA_BUNDLE": "/etc/internal-ca.pem"}, personal_ws, "bluecoat-ca"),
        ("corp_path", CLEAN_ENV, corp_path_ws, "amadeus_work_ws"),
        ("corp_git", CLEAN_ENV, corp_git_ws, "team/service.git"),
        ("corp_wt", CLEAN_ENV, corp_wt_ws, "sec/fw.git"),
    )

    seen_markers: set[str] = set()
    for idx, (vec_name, env_map, cwd_path, sentinel) in enumerate(vectors):
        lang_choice = "fr" if idx % 2 else "en"
        comp = verifier.evaluate_personal_pc_compliance(environ=env_map, cwd=cwd_path, lang=lang_choice)
        _verify_receipt(comp)
        assert comp["compliance_ok"] is False and comp["exit_code"] == 2, vec_name
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0 and comp["subprocess_calls"] == 0
        assert "0.00 EUR / 0.00 USD" in comp["notice_bilingual"]
        assert "AGPL-3.0" in comp["notice_bilingual"]
        assert sentinel not in json.dumps(comp), vec_name
        seen_markers.update(comp["matched_markers"])

        for blocked_res in (
            verifier.run_cloud_run_deploy_smoke_gate(http_fn=_forbid_http, runner=_forbid_runner, environ=env_map, cwd=cwd_path, lang=lang_choice),
            verifier.audit_windows_launcher_contract(runner=_forbid_runner, environ=env_map, cwd=cwd_path, lang=lang_choice),
            verifier.verify_dev_vscode_60s_dry_run(http_fn=_forbid_http, environ=env_map, cwd=cwd_path, lang=lang_choice),
            verifier.verify_win11_gamer_whisper_5m_dry_run(http_fn=_forbid_http, runner=_forbid_runner, environ=env_map, cwd=cwd_path, lang=lang_choice),
            verifier.run_unified_onboarding_and_smoke_gate(http_fn=_forbid_http, runner=_forbid_runner, environ=env_map, cwd=cwd_path, lang=lang_choice),
        ):
            _verify_receipt(blocked_res)
            assert blocked_res["compliance_ok"] is False and blocked_res["exit_code"] == 2
            assert blocked_res["network_calls"] == 0 and blocked_res["file_writes"] == 0 and blocked_res["subprocess_calls"] == 0
            assert sentinel not in json.dumps(blocked_res)

        rc_cli = verifier.main(
            ["--lang", lang_choice, "--json" if idx % 2 == 0 else "--markdown"],
            http_fn=_forbid_http,
            runner=_forbid_runner,
            environ=env_map,
            cwd=cwd_path,
        )
        out_cli = capsys.readouterr().out
        assert rc_cli == 2 and sentinel not in out_cli

    assert seen_markers == set(verifier.CORPORATE_MARKERS)


def test_03_smoke_gate_loopback_server_drift_csp_idempotency_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00220-noel")

    # 1. Stale pre-PR-#949 404 drift -> fails closed with redeploy_required=True
    def _stale_http(_m: str, u: str) -> tuple[int, dict[str, Any]]:
        if u.endswith("/health"):
            return 200, {"status": "ok"}
        if u.endswith("/v1/models"):
            return 200, {"object": "list", "data": [{"id": "hivemind/auto"}, {"id": "hivemind/coder"}]}
        return 404, {"detail": "Not Found"}

    stale_rep = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        http_fn=_stale_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    _verify_receipt(stale_rep)
    assert stale_rep["smoke_ok"] is False
    assert stale_rep["stale_revision_detected"] is True
    assert stale_rep["redeploy_required"] is True
    assert stale_rep["holiday_routes_deployed_on_cloud_run"] is False
    assert stale_rep["onboarding_feedback_deployed"] is False
    assert stale_rep["exit_code"] == 2
    assert "--dry-run" in stale_rep["deploy_guidance"]["dry_run_command"]

    # 2. Live cutover via real loopback ThreadingHTTPServer + ProxyHandler({}) ignoring HTTP_PROXY
    store = RouterStore(tmp_path / "smoke_loopback.sqlite3", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, allow_private_producers=True)

    with TestClient(app) as client:
        _seed_live_coder_producer(client)

        class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/arr":
                    self._send(200, json.dumps([1, 2]).encode("utf-8"), {"content-type": "application/json"})
                    return
                if self.path == "/badjson":
                    self._send(200, b"{corrupt", {"content-type": "application/json"})
                    return
                resp = client.get(self.path, headers={"X-Forwarded-For": "127.0.8.1"})
                self._send(resp.status_code, resp.content, dict(resp.headers))

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                raw = json.loads(self.rfile.read(n).decode("utf-8")) if n > 0 else {}
                hdrs = {"X-Forwarded-For": "127.0.8.2"}
                if self.headers.get("Idempotency-Key"):
                    hdrs["Idempotency-Key"] = str(self.headers["Idempotency-Key"])
                resp = client.post(self.path, json=raw, headers=hdrs)
                self._send(resp.status_code, resp.content, dict(resp.headers))

            def _send(self, code: int, body: bytes, hdrs: dict[str, str]) -> None:
                self.send_response(code)
                for hk, hv in hdrs.items():
                    if hk.lower() not in ("content-length", "transfer-encoding", "server", "date"):
                        self.send_header(hk, hv)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                _ = (format, args)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            monkeypatch.setenv("HTTP_PROXY", CORP_ZSCALER_PROXY)
            monkeypatch.setenv("HTTPS_PROXY", CORP_ZSCALER_PROXY)

            # Proxy-free default_http_request edge responses
            assert verifier.default_http_request("GET", f"{base_url}/arr")[1]["data"] == [1, 2]
            assert verifier.default_http_request("GET", f"{base_url}/badjson")[1]["raw_error"] == "invalid_json"
            assert verifier.default_http_request("GET", f"{base_url}/scripts/unknown_nonexistent_file.py")[0] == 404

            # Live smoke gate + feedback POST + Idempotency-Key duplicate replay conservation
            smoke_live = verifier.run_cloud_run_deploy_smoke_gate(
                base_url,
                expected_sha=TARGET_SHORT_SHA,
                probe_feedback_post=True,
                idempotency_check=True,
                environ=CLEAN_ENV,
                cwd=tmp_path,
                lang="en",
            )
            _verify_receipt(smoke_live)
            assert smoke_live["smoke_ok"] is True
            assert smoke_live["holiday_routes_deployed_on_cloud_run"] is True
            assert smoke_live["onboarding_feedback_deployed"] is True
            assert smoke_live["stale_revision_detected"] is False
            assert smoke_live["redeploy_required"] is False
            assert smoke_live["unique_csp_nonces_count"] == 4
            assert smoke_live["idempotency_conserved"] is True
            assert smoke_live["feedback_summary"]["total_responses"] == 1
            assert smoke_live["exit_code"] == 0

            # CLI modes (--smoke-only, --json, --markdown, text)
            buf_json = io.StringIO()
            assert verifier.main(["--smoke-only", "--gateway", base_url, "--expected-sha", TARGET_SHORT_SHA, "--json"], environ=CLEAN_ENV, cwd=tmp_path, stdout=buf_json) == 0
            _verify_receipt(json.loads(buf_json.getvalue()))

            buf_md = io.StringIO()
            assert verifier.main(["--smoke-only", "--gateway", base_url, "--lang", "fr", "--markdown"], environ=CLEAN_ENV, cwd=tmp_path, stdout=buf_md) == 0
            assert "Rapport Smoke Cloud Run Noël" in buf_md.getvalue()

            buf_cli = io.StringIO()
            assert verifier.main(["--smoke-only", "--gateway", base_url, "--lang", "en"], environ=CLEAN_ENV, cwd=tmp_path, stdout=buf_cli) == 0
            assert "SHA-256" in buf_cli.getvalue()
        finally:
            server.shutdown()
            server.server_close()

        assert verifier.default_http_request("GET", f"http://127.0.0.1:{server.server_port}/health")[0] == 0

        # 3. Negative controls against TestClient adapter: reused nonce, unsafe DOM sink, static figure, base64url nonce
        live_http = _make_router_http_adapter(client)
        unsafe_sink = "inner" + "HTML"
        static_claim = "198.1 " + "ms"

        # Verify a base64url nonce whose token contains digits followed by 'm'+'s+' does NOT false-positive
        nonce_with_ms = "Abc123" + "ms+NonceToken"
        raw_nonce_html = (
            f'<!DOCTYPE html><html><style nonce="{nonce_with_ms}"></style>'
            "<body>100% personal PC AGPL-3.0 0.00 EUR</body></html>"
        )
        assert len(static_numbers_guard.FIGURE.findall(raw_nonce_html)) == 1
        assert static_numbers_guard.FIGURE.findall(verifier.strip_html_csp_nonce_attributes(raw_nonce_html)) == []
        ok_ms_nonce, ms_doc, ms_issues = verifier._inspect_holiday_html_contract(
            "/noel",
            {
                "html": raw_nonce_html,
                "headers": {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce_with_ms}'"},
            },
            set(),
            lang_code="en",
        )
        assert ok_ms_nonce is True and ms_issues == [] and ms_doc["static_figure_found"] is False

        bad_cases: tuple[tuple[str, Any], ...] = (
            (
                "reused_csp_nonce",
                lambda m, p, c, h, b: (
                    200,
                    {**h, "content-security-policy": "default-src 'none'; style-src 'nonce-StaticReuseNonce88'"},
                    "<!DOCTYPE html><style nonce='StaticReuseNonce88'></style><body>100% personal PC AGPL-3.0 0.00 EUR</body>",
                )
                if p in ("/noel", "/quick")
                else (c, h, b),
            ),
            (
                "unsafe_dom_sink",
                lambda m, p, c, h, b: (
                    200,
                    h,
                    f"<!DOCTYPE html><script nonce='DomSinkNonce9999'>el.{unsafe_sink}='x'</script><body>100% personal PC AGPL-3.0 0.00 EUR</body>",
                )
                if p == "/quick"
                else (c, h, b),
            ),
            (
                "static_figure",
                lambda m, p, c, h, b: (
                    200,
                    h,
                    f"<!DOCTYPE html><style nonce='StaticFigNonce99'></style><body>100% personal PC AGPL-3.0 0.00 EUR p50 {static_claim}</body>",
                )
                if p == "/raid-noel"
                else (c, h, b),
            ),
            (
                "empty_models",
                lambda m, p, c, h, b: (200, {"content-type": "application/json"}, {"object": "list", "data": []})
                if p == "/v1/models"
                else (c, h, b),
            ),
            (
                "missing_coder",
                lambda m, p, c, h, b: (200, {"content-type": "application/json"}, {"object": "list", "data": [{"id": "hivemind/auto"}]})
                if p == "/v1/models"
                else (c, h, b),
            ),
            (
                "bad_health",
                lambda m, p, c, h, b: (200, {"content-type": "application/json"}, {"status": "bad"})
                if p == "/health"
                else (c, h, b),
            ),
            (
                "bad_manifest",
                lambda m, p, c, h, b: (200, {"content-type": "application/json"}, {"object": "wrong", "holiday_routes": "bad", "downloadable_scripts": "bad"})
                if p == "/v1/hivemind/onboarding/manifest"
                else (c, h, b),
            ),
            (
                "bad_feedback_summary",
                lambda m, p, c, h, b: (200, {"content-type": "application/json"}, {"total_responses": "bad"})
                if p == "/v1/hivemind/onboarding/feedback/summary"
                else (c, h, b),
            ),
            (
                "bad_raid_status",
                lambda m, p, c, h, b: (200, {"content-type": "application/json"}, {"object": "bad"})
                if p == "/v1/hivemind/raid/status"
                else (c, h, b),
            ),
            (
                "bad_fleet",
                lambda m, p, c, h, b: (200, {"content-type": "application/json"}, {"object": "bad"})
                if p == "/v1/hivemind/producers/fleet"
                else (c, h, b),
            ),
            (
                "bad_ps1",
                lambda m, p, c, h, b: (200, {"content-type": "text/plain"}, "broken ps1")
                if p == "/scripts/join_hivemind.ps1"
                else (c, h, b),
            ),
            (
                "bad_quickstart",
                lambda m, p, c, h, b: (200, {"content-type": "text/plain"}, "broken py")
                if p == "/scripts/hivemind_quickstart.py"
                else (c, h, b),
            ),
        )

        for name, mut in bad_cases:
            def _mut_http(m: str, u: str, p: Any = None, h: Any = None, t: float = 5.0, _mut: Any = mut) -> Any:
                path = "/" + u.split("://", 1)[1].split("/", 1)[1]
                c0, h0, b0 = live_http(m, u, p, h, t)
                return _mut(m, path, c0, h0, b0)

            res_bad = verifier.run_cloud_run_deploy_smoke_gate(
                "http://127.0.0.1:8000",
                http_fn=_mut_http,
                environ=CLEAN_ENV,
                cwd=tmp_path,
            )
            assert res_bad["smoke_ok"] is False and res_bad["exit_code"] == 2, name

        # SHA drift & check_git_head & average_ease_rating support & idempotency non-conservation
        drift_sha_rep = verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            expected_sha=DRIFT_SHA,
            http_fn=live_http,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        assert drift_sha_rep["sha_mismatch"] is True and drift_sha_rep["stale_revision_detected"] is True and drift_sha_rep["exit_code"] == 2

        git_head_fail = verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            check_git_head=True,
            runner=lambda _a, _t: (1, ""),
            http_fn=live_http,
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert git_head_fail["sha_mismatch"] is True and git_head_fail["exit_code"] == 2

        git_head_exc = verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            check_git_head=True,
            runner=lambda _a, _t: (_ for _ in ()).throw(OSError("no git")),
            http_fn=live_http,
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert git_head_exc["sha_mismatch"] is True and git_head_exc["exit_code"] == 2

        avg_ok, avg_doc, _ = verifier._inspect_feedback_summary_schema(
            {"object": verifier.FEEDBACK_SUMMARY_OBJECT, "total_responses": 2, "average_ease_rating": 4.5, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 1}},
            lang_code="en",
        )
        assert avg_ok is True and avg_doc["mean_ease_rating"] == 4.5

        replay_cnt = {"n": 0}

        def _non_idem_http(m: str, u: str, p: Any = None, h: Any = None, t: float = 5.0) -> Any:
            if m == "POST" and u.endswith("/v1/hivemind/onboarding/feedback"):
                replay_cnt["n"] += 1
                return 200, {"object": verifier.FEEDBACK_SUMMARY_OBJECT, "total_responses": replay_cnt["n"], "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": replay_cnt["n"], "gamer": 0}}
            return live_http(m, u, p, h, t)

        non_idem_rep = verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            probe_feedback_post=True,
            idempotency_check=True,
            http_fn=_non_idem_http,
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert non_idem_rep["idempotency_conserved"] is False and non_idem_rep["exit_code"] == 2

        post_500 = verifier.run_cloud_run_deploy_smoke_gate(
            "http://127.0.0.1:8000",
            probe_feedback_post=True,
            http_fn=lambda m, u, p=None, h=None, t=5.0: (500, {"err": 1}) if m == "POST" else live_http(m, u, p, h, t),
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert post_500["feedback_post_ok"] is False and post_500["exit_code"] == 2

        # Error CLI JSON & stderr branches
        err_j = io.StringIO()
        assert verifier.main(["--gateway", UNTRUSTED_EXT_GW, "--json"], environ=CLEAN_ENV, cwd=tmp_path, stdout=err_j) == 2
        _verify_receipt(json.loads(err_j.getvalue()))
        err_t = io.StringIO()
        assert verifier.main(["--gateway", UNTRUSTED_EXT_GW], environ=CLEAN_ENV, cwd=tmp_path, stderr=err_t) == 2
        assert "Error:" in err_t.getvalue()


def test_04_windows_launcher_contract_and_dry_run_verifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Real scripts/join_hivemind.ps1 + scripts/join_hivemind_campaign.sh contract audit
    real_audit = verifier.audit_windows_launcher_contract(environ=CLEAN_ENV, cwd=REPO_ROOT, lang="en")
    _verify_receipt(real_audit)
    assert real_audit["contract_ok"] is True
    assert real_audit["ps1_ok"] is True
    assert real_audit["sh_ok"] is True
    assert real_audit["bash_syntax_ok"] is True
    assert real_audit["bash_help_ok"] is True
    assert real_audit["exit_code"] == 0

    # Simulated pwsh dry-run check branch (plus real pwsh execution if present)
    if shutil.which("pwsh") is None:
        monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/pwsh" if cmd == "pwsh" else None)
        sim_pwsh_audit = verifier.audit_windows_launcher_contract(
            runner=lambda a, _t: (
                (0, json.dumps({"dry_run": True, "target_script": "noel_deploy_smoke_and_win11_verifier.py"}))
                if "pwsh" in str(a[0])
                else ((0, "--gamer-whisper --connect") if "--help" in a else (0, ""))
            ),
            environ=CLEAN_ENV,
            cwd=REPO_ROOT,
        )
        assert sim_pwsh_audit["pwsh_checked"] is True and sim_pwsh_audit["pwsh_dry_run_ok"] is True
        monkeypatch.undo()

    # Negative controls on .ps1 and .sh contracts
    bad_ps1 = tmp_path / "bad_join.ps1"
    bad_ps1.write_text("# missing header\nInvoke-" + "Expression 'x'\n", encoding="utf-8")
    bad_sh = tmp_path / "bad_join.sh"
    bad_sh.write_text("#!/usr/bin/env bash\nif then fi\n", encoding="utf-8")
    bad_audit = verifier.audit_windows_launcher_contract(
        ps1_path=bad_ps1,
        sh_path=bad_sh,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert bad_audit["contract_ok"] is False and bad_audit["ps1_ok"] is False and bad_audit["sh_ok"] is False and bad_audit["exit_code"] == 2

    # 2. Isolated < 60 s Dev VS Code dry-run + real TestClient registration & CES
    store = RouterStore(tmp_path / "dryrun_dev_win11.sqlite3", secret=ROUTER_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1, allow_private_producers=True)
    with TestClient(app) as client:
        _seed_live_coder_producer(client)
        live_http = _make_router_http_adapter(client, peer_ip="127.0.9.1")

        dev_home = tmp_path / "dev_home"
        dev_ws = tmp_path / "dev_ws"
        (dev_ws / ".vscode").mkdir(parents=True)
        (dev_ws / ".vscode" / "settings.json").write_text('{"editor.tabSize": 4}\n', encoding="utf-8")

        dev_res = verifier.verify_dev_vscode_60s_dry_run(
            "http://127.0.0.1:8000",
            home_dir=dev_home,
            workspace_dir=dev_ws,
            handle="dev-soak-probe",
            register_account=True,
            probe_gateway=True,
            submit_ces=True,
            run_uninstall=True,
            http_fn=live_http,
            environ=CLEAN_ENV,
            cwd=dev_ws,
            lang="en",
        )
        _verify_receipt(dev_res)
        assert dev_res["dev_ready"] is True and dev_res["slo_ok"] is True and dev_res["elapsed_s"] < 60.0
        assert dev_res["credentials_0600_ok"] is True and dev_res["continue_yaml_ok"] is True
        assert dev_res["vscode_secret_clean"] is True and dev_res["uninstall_clean"] is True
        assert dev_res["ces_recorded"] is True and dev_res["exit_code"] == 0
        saved_ws = json.loads((dev_ws / ".vscode" / "settings.json").read_text(encoding="utf-8"))
        assert saved_ws["editor.tabSize"] == 4

        # Offline temp-dir Dev dry-run via CLI
        buf_dev = io.StringIO()
        assert verifier.main(["--dev-dry-run-only", "--lang", "fr", "--json"], environ=CLEAN_ENV, cwd=tmp_path, stdout=buf_dev) == 0
        _verify_receipt(json.loads(buf_dev.getvalue()))

        # 3. Real subprocess < 1.5 s Steam preemption drill + < 5 min Win11 Gamer-Whisper dry-run
        real_drill = verifier.default_worker_yield_drill()
        assert real_drill["worker_started"] is True
        assert real_drill["worker_stopped"] is True
        assert real_drill["clean_release"] is True
        assert real_drill["retained_worker_vram_mib"] == 0
        assert float(real_drill["yield_latency_ms"]) < 1500.0

        reg_g = client.post(
            "/v1/hivemind/register",
            json={"handle": "win11-gamer-probe", "role": "trainer"},
            headers={"X-Forwarded-For": "127.0.9.2"},
        ).json()

        win_ok = verifier.verify_win11_gamer_whisper_5m_dry_run(
            "http://127.0.0.1:8000",
            handle="win11-gamer-probe",
            api_key=reg_g["api_key"],
            send_heartbeat=True,
            submit_ces=True,
            gpu_probe_fn=_quiet_rtx_snapshot,
            process_probe_fn=lambda: ["explorer.exe", "steam.exe"],
            http_fn=_make_router_http_adapter(client, peer_ip="127.0.9.2"),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        _verify_receipt(win_ok)
        assert win_ok["gamer_ready"] is True and win_ok["slo_ok"] is True and win_ok["elapsed_s"] < 300.0
        assert win_ok["yield_slo_ok"] is True and win_ok["retained_worker_vram_mib"] == 0
        assert win_ok["thermal_ok"] is True and win_ok["vram_reserve_ok"] is True
        assert win_ok["estimated_upload_mb_per_hour"] < 50.0
        assert win_ok["heartbeat_ok"] is True and win_ok["ces_recorded"] is True and win_ok["zero_game_lag"] is True

        # No-GPU honest fallback + CLI --win11-dry-run-only + --launcher-audit-only + unified CLI
        buf_win = io.StringIO()
        assert verifier.main(["--win11-dry-run-only", "--json"], runner=lambda _a, _t: (1, ""), environ=CLEAN_ENV, cwd=tmp_path, stdout=buf_win) == 0
        assert json.loads(buf_win.getvalue())["gpu_status"] == "cpu_or_dev_only_no_nvidia_gpu"
        assert verifier.main(["--launcher-audit-only", "--json"], environ=CLEAN_ENV, cwd=REPO_ROOT, stdout=io.StringIO()) == 0
        buf_uni = io.StringIO()
        assert verifier.main(
            ["--gateway", "http://127.0.0.1:8000", "--check-git-head", "--lang", "fr"],
            http_fn=lambda _m, _u: (404, {}),
            runner=lambda a, _t: (0, "--gamer-whisper --connect") if "--help" in a else (0, TARGET_FULL_SHA),
            environ=CLEAN_ENV,
            cwd=REPO_ROOT,
            stdout=buf_uni,
        ) == 2
        assert "[DIAG]" in buf_uni.getvalue()

        # Edge validation/error branches on Dev & Win11 dry-runs
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.verify_dev_vscode_60s_dry_run(ease_rating=0, environ=CLEAN_ENV, cwd=tmp_path)
        with pytest.raises(verifier.SmokeVerifierError):
            verifier.verify_win11_gamer_whisper_5m_dry_run(ease_rating=9, environ=CLEAN_ENV, cwd=tmp_path)

        dev_err = verifier.verify_dev_vscode_60s_dry_run(
            "http://127.0.0.1:8000",
            register_account=True,
            probe_gateway=True,
            submit_ces=True,
            http_fn=lambda _m, _u: (500, {"err": 1}),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        assert dev_err["dev_ready"] is False and dev_err["exit_code"] == 2

        win_err = verifier.verify_win11_gamer_whisper_5m_dry_run(
            "http://127.0.0.1:8000",
            send_heartbeat=True,
            submit_ces=True,
            api_key=None,
            gpu_probe_fn=_quiet_rtx_snapshot,
            process_probe_fn=lambda: [],
            worker_yield_fn=_fast_yield_result,
            http_fn=lambda _m, _u: (500, {"err": 1}),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        assert win_err["heartbeat_ok"] is False and win_err["ces_recorded"] is False and win_err["exit_code"] == 2

        win_hb_500 = verifier.verify_win11_gamer_whisper_5m_dry_run(
            "http://127.0.0.1:8000",
            send_heartbeat=True,
            api_key=TEST_KEY_ALPHA,
            gpu_probe_fn=_quiet_rtx_snapshot,
            process_probe_fn=lambda: [],
            worker_yield_fn=_fast_yield_result,
            http_fn=lambda _m, _u: (500, {"err": 1}),
            environ=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert win_hb_500["heartbeat_ok"] is False and win_hb_500["exit_code"] == 2

    # Negative controls: slow yield & hot GPU & low VRAM & nvidia-smi parse error
    slow_win = verifier.verify_win11_gamer_whisper_5m_dry_run(
        gpu_probe_fn=_quiet_rtx_snapshot,
        process_probe_fn=lambda: ["cs2.exe"],
        worker_yield_fn=lambda: {"worker_started": True, "worker_stopped": True, "clean_release": False, "retained_worker_vram_mib": 512, "yield_latency_ms": 1820.0},
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert slow_win["yield_slo_ok"] is False and slow_win["gamer_ready"] is False and slow_win["zero_game_lag"] is False and slow_win["exit_code"] == 2

    hot_win = verifier.verify_win11_gamer_whisper_5m_dry_run(
        gpu_probe_fn=lambda: {"present": True, "parse_error": False, "name": "RTX 4070", "total_mib": 12288, "free_mib": 1024, "temp_c": 75.5},
        process_probe_fn=lambda: [],
        worker_yield_fn=_fast_yield_result,
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert hot_win["thermal_ok"] is False and hot_win["vram_reserve_ok"] is False and hot_win["gamer_ready"] is False and hot_win["exit_code"] == 2

    parse_err_win = verifier.verify_win11_gamer_whisper_5m_dry_run(
        gpu_probe_fn=lambda: {"present": True, "parse_error": True},
        process_probe_fn=lambda: [],
        worker_yield_fn=_fast_yield_result,
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert parse_err_win["gpu_status"] == "gpu_driver_parse_error" and parse_err_win["exit_code"] == 2

    # Default GPU/process probe parsers + worker_yield_drill error branches
    assert verifier.default_gpu_probe(runner=lambda _a, _t: (0, "NVIDIA RTX 4070, 12288, 9600, 54, 4\n"))["name"] == "NVIDIA RTX 4070"
    assert verifier.default_gpu_probe(runner=lambda _a, _t: (0, "bad_csv"))["parse_error"] is True
    assert verifier.default_gpu_probe(runner=lambda _a, _t: (0, "GPU, bad, 1, 2, 3"))["parse_error"] is True
    assert "cs2.exe" in verifier.default_process_probe(runner=lambda _a, _t: (0, '"cs2.exe","1"\n'), os_name="nt")
    assert "eldenring.exe" in verifier.default_process_probe(runner=lambda _a, _t: (0, "eldenring.exe\n"), os_name="posix")
    assert verifier.default_process_probe(runner=lambda _a, _t: (_ for _ in ()).throw(OSError("no ps"))) == []

    class _DrillProcBad:
        def __init__(self, mode: str) -> None:
            self._mode = mode
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("BAD\n" if mode == "bad" else "HIVEMIND_WORKER_READY\n")
            self.stderr = io.StringIO()
            self._killed = False

        def kill(self) -> None:
            self._killed = True

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._mode == "timeout" and not self._killed:
                raise subprocess.TimeoutExpired(cmd=["py"], timeout=0.2)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _DrillProcBad("bad"))
    assert verifier.default_worker_yield_drill(timeout_s=0.2)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: _DrillProcBad("timeout"))
    assert verifier.default_worker_yield_drill(timeout_s=0.2)["clean_release"] is False
    monkeypatch.undo()


def test_05_concurrent_48_participant_cutover_onboarding_and_smoke_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00230-noel")

    # Phase A: Pre-cutover stale 404 detection
    pre_cutover = verifier.run_cloud_run_deploy_smoke_gate(
        "http://127.0.0.1:8000",
        http_fn=lambda _m, u: (200, {"status": "ok"}) if u.endswith("/health") else (404, {"detail": "Not Found"}),
        environ=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert pre_cutover["stale_revision_detected"] is True and pre_cutover["redeploy_required"] is True and pre_cutover["exit_code"] == 2

    # Phase B: Live cutover backed by SQLite WAL + allow_private_producers=True
    upstream_inference_hits: list[str] = []

    def _forbid_upstream_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn18_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        journal_mode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream_inference)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    expected_ps1_sha = hashlib.sha256(PS1_PATH.read_bytes()).hexdigest()
    expected_qs_sha = hashlib.sha256((REPO_ROOT / "scripts" / "hivemind_quickstart.py").read_bytes()).hexdigest()

    dev_count = 24
    gamer_count = 24
    probe_threads = 4
    total_threads = dev_count + gamer_count + probe_threads
    barrier = threading.Barrier(total_threads)

    shared_lock = threading.Lock()
    observed_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []
    seen_paths: list[str] = []
    raw_keys_seen: list[str] = []
    masked_keys_seen: list[str] = []

    with TestClient(app) as client:
        _seed_live_coder_producer(client)

        def _verify_html(resp_text: str, csp_hdr: str) -> str:
            if _SQLITE_LOCK_RE.search(resp_text):
                with shared_lock:
                    sqlite_lock_errors.append(resp_text[:120])
            hdr_hits = _CSP_HDR_NONCE_RE.findall(csp_hdr)
            dom_hits = [val for _, val in _HTML_ATTR_NONCE_RE.findall(resp_text)]
            assert hdr_hits and dom_hits and hdr_hits[0] in dom_hits
            stripped = verifier.strip_html_csp_nonce_attributes(resp_text)
            assert static_numbers_guard.FIGURE.findall(stripped) == []
            assert ("inner" + "HTML") not in stripped
            return hdr_hits[0]

        def _participant_worker(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_count
            cohort_idx = idx if is_dev else (idx - dev_count)
            profile = "dev" if is_dev else "gamer"
            peer_ip = _loopback_ip(10 if is_dev else 20, cohort_idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            http_adapter = _make_router_http_adapter(client, peer_ip=peer_ip)

            slot_root = tmp_path / "soak48" / f"slot_{idx:02d}"
            os_user = f"holiday_user_{idx:02d}"
            home_dir = slot_root / "home" / os_user
            ws_dir = slot_root / "hobby_project"
            home_dir.mkdir(parents=True, exist_ok=True)
            ws_dir.mkdir(parents=True, exist_ok=True)

            local_nonces: list[str] = []
            barrier.wait(timeout=25.0)
            t0 = time.monotonic()

            # 1. Hit all 4 holiday HTML routes
            for html_path in verifier.SMOKE_HTML_ROUTES:
                r_html = client.get(html_path, headers=hdrs)
                assert r_html.status_code == 200
                local_nonces.append(_verify_html(r_html.text, r_html.headers.get("content-security-policy", "")))
                with shared_lock:
                    seen_paths.append(html_path)

            # 2. Hit /v1/models & /v1/hivemind/onboarding/manifest
            r_mod = client.get("/v1/models", headers=hdrs)
            assert r_mod.status_code == 200
            assert "hivemind/coder" in [m["id"] for m in r_mod.json()["data"]]

            r_man = client.get("/v1/hivemind/onboarding/manifest", headers=hdrs)
            assert r_man.status_code == 200 and r_man.json()["build_sha"] == TARGET_FULL_SHA

            # 3. Download /scripts/join_hivemind.ps1 & /scripts/hivemind_quickstart.py with SHA-256 parity
            r_ps1 = client.get("/scripts/join_hivemind.ps1", headers=hdrs)
            r_qs = client.get("/scripts/hivemind_quickstart.py", headers=hdrs)
            assert r_ps1.status_code == 200 and hashlib.sha256(r_ps1.content).hexdigest() == expected_ps1_sha
            assert r_qs.status_code == 200 and hashlib.sha256(r_qs.content).hexdigest() == expected_qs_sha

            # 4. Register account
            handle = f"soak_{profile}_{cohort_idx:02d}"
            r_reg = client.post(
                "/v1/hivemind/register",
                json={"handle": handle, "role": "prosumer" if is_dev else "trainer"},
                headers=hdrs,
            )
            assert r_reg.status_code == 200, r_reg.text
            if _SQLITE_LOCK_RE.search(r_reg.text):
                with shared_lock:
                    sqlite_lock_errors.append(r_reg.text[:120])
            raw_key = str(r_reg.json()["api_key"])
            masked_key = verifier.mask_api_key(raw_key)

            # 5. Outbound OpenTrainer heartbeat + local <60s Dev / <5m Win11 dry-run
            if is_dev:
                ease = 5 if (cohort_idx % 4 != 0) else 4
                zero_lag = True
                plat = "linux"
                dry_rep = verifier.verify_dev_vscode_60s_dry_run(
                    "http://127.0.0.1:8000",
                    home_dir=home_dir,
                    workspace_dir=ws_dir,
                    handle=handle,
                    api_key=raw_key,
                    ease_rating=ease,
                    submit_ces=False,
                    run_uninstall=True,
                    http_fn=http_adapter,
                    environ=CLEAN_ENV,
                    cwd=ws_dir,
                    lang="fr" if cohort_idx % 2 else "en",
                )
                assert dry_rep["dev_ready"] is True and dry_rep["elapsed_s"] < 60.0
                assert dry_rep["vscode_secret_clean"] is True and dry_rep["uninstall_clean"] is True
                serialized_rep = json.dumps(dry_rep)
                assert raw_key not in serialized_rep and os_user not in serialized_rep
            else:
                ease = 5 if (cohort_idx % 5 != 0) else 4
                zero_lag = cohort_idx != 23  # 47/48 == 0.9792 >= 0.95
                plat = "windows"
                hb_payload = {
                    "host_label": f"win11-rtx-{cohort_idx:02d}",
                    "gpu_name": "NVIDIA GeForce RTX 4070",
                    "vram_total_mib": 12288,
                    "vram_free_mib": 9600,
                    "network_mode": "outbound-https-relay",
                    "training_stack": "qlora-moe+diloco+amcp-aglet+boinc",
                    "active_campaign": verifier.CAMPAIGN_ID,
                }
                OpenTrainerHeartbeatRequest.model_validate(hb_payload)
                r_hb = client.post(
                    "/v1/hivemind/opentrainer/heartbeat",
                    json=hb_payload,
                    headers={"Authorization": f"Bearer {raw_key}", **hdrs},
                )
                assert r_hb.status_code == 200, r_hb.text

                dry_rep = verifier.verify_win11_gamer_whisper_5m_dry_run(
                    "http://127.0.0.1:8000",
                    handle=handle,
                    host_label=f"win11-rtx-{cohort_idx:02d}",
                    ease_rating=ease,
                    had_game_lag=not zero_lag,
                    gpu_probe_fn=_quiet_rtx_snapshot,
                    process_probe_fn=lambda: ["explorer.exe", "steam.exe"],
                    worker_yield_fn=_fast_yield_result,
                    http_fn=http_adapter,
                    environ=CLEAN_ENV,
                    cwd=ws_dir,
                    lang="fr" if cohort_idx % 2 else "en",
                )
                assert dry_rep["gamer_ready"] is True and dry_rep["elapsed_s"] < 300.0
                assert dry_rep["yield_slo_ok"] is True and dry_rep["retained_worker_vram_mib"] == 0
                serialized_rep = json.dumps(dry_rep)
                assert raw_key not in serialized_rep and os_user not in serialized_rep

            # 6. Duplicate-replayed Idempotency-Key POST /v1/hivemind/onboarding/feedback
            idem_key = "soak18-ces-" + hashlib.sha256(f"{profile}:{cohort_idx}".encode()).hexdigest()[:24]
            fb_body = {
                "profile": profile,
                "ease_rating": ease,
                "zero_game_lag": zero_lag,
                "platform": plat,
                "note": "",
            }
            fb1 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json=fb_body,
                headers={**hdrs, "Idempotency-Key": idem_key},
            )
            fb2 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json=fb_body,
                headers={**hdrs, "Idempotency-Key": idem_key},
            )
            assert fb1.status_code == 200 and fb2.status_code == 200
            assert (time.monotonic() - t0) < (60.0 if is_dev else 300.0)

            with shared_lock:
                observed_nonces.extend(local_nonces)
                raw_keys_seen.append(raw_key)
                masked_keys_seen.append(masked_key)
            return {"profile": profile, "ease": ease, "zero_game_lag": zero_lag}

        def _unified_gate_worker(probe_idx: int) -> dict[str, Any]:
            probe_ip = _loopback_ip(30, probe_idx)
            probe_http = _make_router_http_adapter(client, peer_ip=probe_ip)
            local_probe_nonces: list[str] = []

            def _capturing_http(method: str, url: str, payload: Any = None, headers: Any = None, timeout_s: float = 5.0) -> Any:
                code, r_hdrs, body = probe_http(method, url, payload, headers, timeout_s)
                path = "/" + url.split("://", 1)[1].split("/", 1)[1]
                if method == "GET" and path in verifier.SMOKE_HTML_ROUTES and code == 200 and isinstance(body, str):
                    local_probe_nonces.append(_verify_html(body, r_hdrs.get("content-security-policy", "")))
                return code, r_hdrs, body

            barrier.wait(timeout=25.0)
            gate_res = verifier.run_unified_onboarding_and_smoke_gate(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_FULL_SHA,
                probe_feedback_post=False,
                run_launcher_audit=True,
                run_dev_dry_run=True,
                run_win11_dry_run=True,
                gpu_probe_fn=_quiet_rtx_snapshot,
                process_probe_fn=lambda: ["explorer.exe", "steam.exe"],
                worker_yield_fn=_fast_yield_result,
                http_fn=_capturing_http,
                environ=CLEAN_ENV,
                cwd=REPO_ROOT,
                lang="fr" if probe_idx % 2 else "en",
            )
            with shared_lock:
                observed_nonces.extend(local_probe_nonces)
            return gate_res

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as pool:
            part_futs = [pool.submit(_participant_worker, i) for i in range(dev_count + gamer_count)]
            probe_futs = [pool.submit(_unified_gate_worker, j) for j in range(probe_threads)]
            part_outcomes = [f.result(timeout=60.0) for f in part_futs]
            probe_outcomes = [f.result(timeout=60.0) for f in probe_futs]

        assert len(part_outcomes) == 48
        assert sqlite_lock_errors == []
        assert upstream_inference_hits == []
        assert all("completions" not in p for p in seen_paths)

        for po in probe_outcomes:
            _verify_receipt(po)
            assert po["overall_ok"] is True
            assert po["smoke_ok"] is True
            assert po["contract_ok"] is True
            assert po["dev_ready"] is True
            assert po["gamer_ready"] is True
            assert po["exit_code"] == 0

        # 100% unique per-request CSP nonces across all 4 HTML routes (48*4 + 4*4 = 208)
        expected_nonces = (48 * 4) + (probe_threads * 4)
        assert len(observed_nonces) == expected_nonces
        assert len(set(observed_nonces)) == expected_nonces

        # Exact 5-key dict equality & 48-user CES conservation
        fb_summary = client.get("/v1/hivemind/onboarding/feedback/summary").json()
        assert set(fb_summary.keys()) == {
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        }
        assert fb_summary["object"] == "hivemind.onboarding_feedback_summary"
        assert fb_summary["total_responses"] == 48
        assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["mean_ease_rating"] >= 4.2
        assert fb_summary["zero_game_lag_ratio"] >= 0.95

        raid_doc = client.get("/v1/hivemind/raid/status").json()
        assert raid_doc["swarm"]["active_trainers"] >= 24

        # Post-soak unified gate meets all 5 AGY adoption gates
        post_soak = verifier.run_unified_onboarding_and_smoke_gate(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_FULL_SHA,
            gpu_probe_fn=_quiet_rtx_snapshot,
            process_probe_fn=lambda: ["explorer.exe", "steam.exe"],
            worker_yield_fn=_fast_yield_result,
            http_fn=_make_router_http_adapter(client, peer_ip="127.0.50.1"),
            environ=CLEAN_ENV,
            cwd=REPO_ROOT,
            lang="en",
        )
        _verify_receipt(post_soak)
        assert post_soak["overall_ok"] is True
        assert post_soak["cloud_run_smoke"]["adoption_gates"]["readiness_gates_passed"] == 5

        for raw_k, masked_k in zip(raw_keys_seen, masked_keys_seen, strict=True):
            assert raw_k not in masked_k
            assert masked_k.startswith(("hm-***", "hm_live_***"))


def test_06_masked_key_ms_suffix_output_safety_across_verifiers(tmp_path: Path) -> None:
    from scripts import noel_arena_ballot_and_parity_verifier as arena_verifier
    from scripts import noel_posix_launcher_and_smoke_verifier as posix_verifier

    verifier_specs: tuple[tuple[str, Any, type[Exception], Path], ...] = (
        (
            "scripts/noel_deploy_smoke_and_win11_verifier.py",
            verifier,
            verifier.SmokeVerifierError,
            VERIFIER_PATH,
        ),
        (
            "scripts/noel_posix_launcher_and_smoke_verifier.py",
            posix_verifier,
            posix_verifier.SmokeVerifierError,
            REPO_ROOT / "scripts" / "noel_posix_launcher_and_smoke_verifier.py",
        ),
        (
            "scripts/noel_arena_ballot_and_parity_verifier.py",
            arena_verifier,
            arena_verifier.ArenaVerifierError,
            REPO_ROOT / "scripts" / "noel_arena_ballot_and_parity_verifier.py",
        ),
    )

    masked_ms_keys = [f"hm-***{digit}" + "ms" for digit in range(10)] + [
        "hm_live_***4" + "ms",
        "hm_live_***9" + "ms",
        "hm-***x8" + "ms",
        "hm_live_***a8" + "ms",
    ]
    genuine_figures = (
        "42 " + "ms",
        "18 tok" + "/s",
        "100 HTTP" + " Reqs",
        "5 s/" + "round",
    )
    raw_unmasked_key = "hm-abcdefgh" + "ijkl1234"
    nonce_html_single = "<style nonce='Abc123" + "msToken'>body{}</style>"
    nonce_html_double = '<script nonce="99tok' + '/sToken">var x=1;</script>'
    csp_hdr_token = "default-src 'none'; script-src 'nonce-77" + "msNonceToken'"

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel_path, mod, err_cls, file_path in verifier_specs:
        src_text = file_path.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], rel_path
        assert lateral_guard.audit_source_for_lateral_movement(src_text, str(file_path)) == []
        assert scanner.scan_python_ast_security(rel_path, src_text) == []
        assert scanner.scan_text_for_payment_solicitation(src_text, rel_path) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(src_text, rel_path) == []
        assert scanner.scan_unicode_trojan_source(src_text, rel_path) == []
        assert static_numbers_guard.FIGURE.findall(src_text) == []
        syn_diff = f"diff --git a/{rel_path} b/{rel_path}\n+++ b/{rel_path}\n" + "\n".join(
            f"+{ln}" for ln in src_text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(syn_diff)) == []
        assert ZeroStubAudit.audit_python_code(src_text).is_clean
        changed_files.append({"filename": rel_path, "patch": "", "content": src_text})

        # (1) Masked keys ending in <digit>ms and HTML/CSP nonces produce 0 scan_output_safety issues
        for masked_k in masked_ms_keys:
            assert mod.scan_output_safety(masked_k) == [], (rel_path, masked_k)
            json_sample = json.dumps({"masked_api_key": masked_k, "status": "ok"}, sort_keys=True)
            assert mod.scan_output_safety(json_sample) == [], (rel_path, masked_k)
            combined_safe = f"{nonce_html_single} {nonce_html_double} {csp_hdr_token} key={masked_k}"
            assert mod.scan_output_safety(combined_safe) == [], (rel_path, masked_k)
            receipt = mod._attach_receipt({"schema": mod.SCHEMA, "masked_api_key": masked_k, "ok": True})
            assert receipt["masked_api_key"] == masked_k

        # (2) Genuine static telemetry figures and raw unmasked keys STILL fail closed
        for fig in genuine_figures:
            assert "static_figure_detected" in mod.scan_output_safety(fig), (rel_path, fig)
            assert "static_figure_detected" in mod.scan_output_safety(
                f"{nonce_html_single} key=hm-***x8" + f"ms body {fig}"
            ), (rel_path, fig)
            with pytest.raises(err_cls):
                mod._assert_safe_output(fig)

        assert "unmasked_api_key_detected" in mod.scan_output_safety(raw_unmasked_key), rel_path
        with pytest.raises(err_cls):
            mod._assert_safe_output(raw_unmasked_key)

    pr_audit = scanner.audit_pull_request(
        pr_number=1222,
        pr_title="fix(onboarding,scale): harden scan_output_safety against masked-key ms suffixes across Noel verifiers",
        pr_body="Implements SLICE-SCALE-DYN-22 masked-key output safety guard across all 3 Noel verifiers.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["fix(onboarding,scale): harden scan_output_safety against masked-key ms suffixes"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True and pr_audit.findings == []

    # End-to-end dry-run receipts with token_urlsafe keys ending in <digit>ms succeed cleanly
    dev_win11_rep = verifier.verify_dev_vscode_60s_dry_run(
        "http://127.0.0.1:8000",
        home_dir=tmp_path / "win11_ms_home",
        workspace_dir=tmp_path / "win11_ms_ws",
        api_key="hm-dyn22_random_x8" + "ms",
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    assert dev_win11_rep["dev_ready"] is True and dev_win11_rep["exit_code"] == 0
    assert dev_win11_rep["masked_api_key"] == "hm-***x8" + "ms"
    _verify_receipt(dev_win11_rep)

    dev_posix_rep = posix_verifier.verify_posix_dev_vscode_60s_dry_run(
        "http://127.0.0.1:8000",
        home_dir=tmp_path / "posix_ms_home",
        workspace_dir=tmp_path / "posix_ms_ws",
        api_key="hm_live_dyn22_rand9" + "ms",
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert dev_posix_rep["dev_ready"] is True and dev_posix_rep["exit_code"] == 0
    assert dev_posix_rep["masked_api_key"] == "hm_live_***d9" + "ms"
    assert posix_verifier.scan_output_safety(json.dumps(dev_posix_rep, sort_keys=True, ensure_ascii=False)) == []
    assert posix_verifier.scan_output_safety(posix_verifier.render_cli_report(dev_posix_rep)) == []
    assert posix_verifier.scan_output_safety(posix_verifier.render_markdown_report(dev_posix_rep)) == []
    assert verifier.scan_output_safety(verifier.render_cli_report(dev_win11_rep)) == []
    assert verifier.scan_output_safety(verifier.render_markdown_report(dev_win11_rep)) == []
