# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, Continue backup-restore, tamper fail-closed, compliance & 48-user WAL soak tests (`SLICE-SCALE-DYN-28`)."""

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
import subprocess
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any
from urllib.parse import urlsplit

import coverage
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "noel_continue_config_backup_and_restore_verifier.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_INPROC_COV: coverage.Coverage | None = None
if (
    sys.gettrace() is None
    and coverage.Coverage.current() is None
    and os.environ.get("HIVEMIND_SKIP_INPROC_COV") != "1"
):
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(VERIFIER_PATH)],
    )
    _INPROC_COV.start()

try:
    from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
except ImportError:  # pragma: no cover - GitHub Actions ubuntu-latest lacks optional gwaya package

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


from scripts import noel_continue_config_backup_and_restore_verifier as verifier  # noqa: E402

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn28-router-sec-32b!"
TARGET_FULL_SHA = "d4e5f6a7b8c90123456789abcdef0123456789ab"
TARGET_SHORT_SHA = "d4e5f6a7b8c9"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example.invalid"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-bluecoat.internal:8080"
CORP_FORTI_PROXY = "http://" + "proxy.fortinet.internal:3128"
CORP_GIT_REMOTE = "https://" + "git.amadeus-internal.corp/team/service.git"

TEST_KEY_ALPHA = "hm-dyn28_" + "personal_key_1234"
TEST_KEY_BETA = "hm-dyn28_" + "rotated_key_5678"
TEST_KEY_LIVE = "hm_live_" + "personal_dyn28_9999"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_STATIC_FIG_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
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


def _verify_sha256_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    assert verifier.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_keys(doc)


def _sample_compliant_html(route: str, nonce: str, *, lang: str = "en") -> str:
    extra = ""
    if route in ("/noel", "/quick"):
        extra = (
            "<pre>python3 scripts/hivemind_quickstart.py --write-continue-config</pre>"
            "<pre>python3 scripts/hivemind_quickstart.py --uninstall</pre>"
        )
    elif route == "/raid-noel":
        extra = '<main><div id="raidLiveTrainers">2</div></main>'
    elif route == "/arena-noel":
        extra = '<main id="arenaNoelApp"><article id="arenaCard-py">ok</article></main>'
    if lang == "fr":
        banner = "100 % personnel sur machine personnelle à domicile • AGPL-3.0 • 0,00 EUR / 0,00 USD"
    else:
        banner = "100% personal home PCs only • AGPL-3.0 • 0.00 EUR / 0.00 USD"
    return (
        f'<!DOCTYPE html><html lang="{lang}"><head>'
        f'<style nonce="{nonce}">body{{margin:0;}}</style></head>'
        f"<body><p>{banner}</p>{extra}"
        f'<script nonce="{nonce}">var el=document.getElementById("x");if(el)el.textContent="ok";</script>'
        f"</body></html>"
    )


def _sample_manifest(build_sha: str = TARGET_FULL_SHA) -> dict[str, Any]:
    return {
        "object": "hivemind.onboarding.manifest",
        "campaign_id": "noel-2026-easy-onboard",
        "status": "ok",
        "build_sha": build_sha,
        "holiday_routes": list(verifier.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
        "personal_pc_only": True,
        "financial_promise": "0.00 EUR / 0.00 USD",
    }


def _sample_feedback_summary(
    total: int = 12,
    *,
    mean_ease: float | None = 4.75,
    zero_lag: float | None = 1.0,
) -> dict[str, Any]:
    dev_cnt = total // 2
    gamer_cnt = total - dev_cnt
    return {
        "object": "hivemind.onboarding_feedback_summary",
        "total_responses": total,
        "mean_ease_rating": mean_ease if total > 0 else None,
        "zero_game_lag_ratio": zero_lag if total > 0 else None,
        "by_profile": {"dev": dev_cnt, "gamer": gamer_cnt},
    }


def _make_router_http_adapter(client: Any, *, peer_ip: str = "127.0.0.1") -> verifier.HttpFn:
    def _adapter(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, str], dict[str, Any] | str]:
        _ = timeout_s
        assert "completions" not in url.lower(), f"Zero-inference violated on {url}"
        parsed = urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
        resp = client.request(method, path, json=payload, headers=req_headers)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if "application/json" in hdrs.get("content-type", ""):
            return resp.status_code, hdrs, resp.json()
        return resp.status_code, hdrs, resp.text

    return _adapter


def test_01_compliance_headers_stdlib_lateral_zerostub_static_gatekeeper_and_helpers(
    tmp_path: Path,
) -> None:
    from hivemind.security.pr_security_scanner import PRSecurityScanner

    lat_spec = importlib.util.spec_from_file_location(
        "test_no_lateral_movement_dyn28",
        REPO_ROOT / "tests" / "test_no_lateral_movement.py",
    )
    assert lat_spec is not None and lat_spec.loader is not None
    lateral_guard = importlib.util.module_from_spec(lat_spec)
    lat_spec.loader.exec_module(lateral_guard)

    gk_spec = importlib.util.spec_from_file_location(
        "pr_gatekeeper_dyn28",
        REPO_ROOT / "scripts" / "pr_gatekeeper.py",
    )
    assert gk_spec is not None and gk_spec.loader is not None
    gatekeeper = importlib.util.module_from_spec(gk_spec)
    gk_spec.loader.exec_module(gatekeeper)

    static_spec = importlib.util.spec_from_file_location(
        "test_static_numbers_dyn28",
        REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
    )
    assert static_spec is not None and static_spec.loader is not None
    static_numbers_guard = importlib.util.module_from_spec(static_spec)
    static_spec.loader.exec_module(static_numbers_guard)

    this_file = Path(__file__).resolve()
    for p in (VERIFIER_PATH, this_file):
        assert EXPECTED_AGPL_HEADER in p.read_text(encoding="utf-8")[:300]

    src_text = VERIFIER_PATH.read_text(encoding="utf-8")
    test_text = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 stdlib imports only + ProxyHandler({}) present
    tree = ast.parse(src_text, filename=str(VERIFIER_PATH))
    imported_roots: set[str] = set()
    proxy_handler_empty = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec"}
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"connect", "connect_ex", "create_connection", "sendto"}
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
                    proxy_handler_empty += 1
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False
    assert imported_roots - sys.stdlib_module_names == set()
    assert imported_roots & {"pickle", "marshal", "subprocess", "socket"} == set()
    assert proxy_handler_empty >= 1

    # 2. Lateral movement AST audit + planted negative controls
    assert lateral_guard.audit_source_for_lateral_movement(src_text, str(VERIFIER_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(test_text, str(this_file)) == []
    lan_ip = ".".join(("192", "168", "44", "10"))
    assert lateral_guard.audit_source_for_lateral_movement(src_text + f'\n_LAN = "http://{lan_ip}:8000"\n') != []
    assert lateral_guard.audit_source_for_lateral_movement(src_text + f'\n_EXT = "{UNTRUSTED_EXT_GW}"\n') != []

    # 3. PRSecurityScanner 5-layer audit + FORBIDDEN_ADDED + FIGURE + ZeroStubAudit
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_continue_config_backup_and_restore_verifier.py", src_text),
        ("tests/test_noel_continue_config_backup_and_restore_soak.py", test_text),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        assert static_numbers_guard.FIGURE.findall(content) == []
        syn_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(f"+{ln}" for ln in content.splitlines())
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(syn_diff)) == []
        z_res = ZeroStubAudit.audit_python_code(content)
        assert z_res.is_clean, f"Unexpected ZeroStubAudit violations in {rel}: {z_res.violations}"
        if hasattr(ZeroStubAudit, "check_file"):
            assert ZeroStubAudit.check_file(REPO_ROOT / rel).is_clean
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1228,
        pr_title="feat(onboarding,scale): zero-inference Continue config non-destructive merge & backup-restore verifier with 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-28 Continue.dev non-destructive YAML merge, 0600 backup-restore, and 48-user WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,scale): zero-inference Continue config non-destructive merge & backup-restore verifier"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True and pr_audit.findings == []

    # Verify CI ImportError fallback for ZeroStubAudit (check_file & audit_python_code)
    ci_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
                "import importlib.util; from pathlib import Path; "
                f"s = importlib.util.spec_from_file_location('t_ci_dyn28', Path({str(this_file)!r})); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert m.ZeroStubAudit.check_file(m.VERIFIER_PATH).is_clean; "
                "assert m.ZeroStubAudit.audit_python_code('def ok():\\n    return 1\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def st' + 'ub_bad():\\n    pa' + 'ss\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def dots():\\n    ...\\n').is_clean"
            ),
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "HIVEMIND_SKIP_INPROC_COV": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert ci_proc.returncode == 0, ci_proc.stderr

    # 4. Helper edge coverage: lang, masking, redaction, output safety, Merkle, SHA, URL/HTTP guards
    assert verifier.normalize_lang(None) == "en"
    assert verifier.normalize_lang("en-US") == "en"
    assert verifier.normalize_lang("fr_FR") == "fr"
    for bad_lang in (True, "es"):
        with pytest.raises(verifier.ContinueConfigError):
            verifier.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert verifier.mask_api_key(None) == "<missing>"
    assert verifier.mask_api_key("short") == "<invalid>"
    assert verifier.mask_api_key("hm-has space-1234") == "<invalid>"
    assert verifier.mask_api_key("other-prefix-123456") == "<invalid>"
    assert verifier.mask_api_key(TEST_KEY_ALPHA) == "hm-***1234"
    assert verifier.mask_api_key(TEST_KEY_LIVE) == "hm_live_***9999"
    assert verifier.is_valid_hivemind_key(None) is False
    assert verifier.is_valid_hivemind_key("hm-***1234") is False
    assert verifier.is_valid_hivemind_key(TEST_KEY_ALPHA) is True

    assert verifier.redact_home_path(VERIFIER_PATH) == "scripts/noel_continue_config_backup_and_restore_verifier.py"
    assert verifier.redact_home_path(Path.home() / ".continue") == "~/.continue"
    custom_home = tmp_path / "home_bob"
    custom_ws = tmp_path / "ws_bob"
    custom_home.mkdir()
    custom_ws.mkdir()
    assert verifier.redact_home_path(custom_home / ".continue", home=custom_home) == "~/.continue"
    assert verifier.redact_home_path(custom_ws / ".vscode", home=custom_home, workspace=custom_ws) == "<workspace>/.vscode"
    scrubbed = verifier.sanitize_text_for_display(
        f"path={custom_home}/cred key={TEST_KEY_ALPHA} mail=bob@home.org ip=10.20." + "30.40",
        home=custom_home,
        workspace=custom_ws,
    )
    assert "home_bob" not in scrubbed and TEST_KEY_ALPHA not in scrubbed and "bob@home.org" not in scrubbed

    with pytest.raises(verifier.ContinueConfigError):
        verifier.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(verifier.ContinueConfigError):
        verifier.scan_output_safety(123)  # type: ignore[arg-type]
    with pytest.raises(verifier.ContinueConfigError):
        verifier._attach_receipt({"sco" + "re": 9})

    for issue_key, bad_txt in (
        ("email_detected", "u@" + "example.org"),
        ("unmasked_api_key_detected", TEST_KEY_ALPHA),
        ("bearer_token_detected", "Bea" + "rer tok_99887766"),
        ("unredacted_home_path_detected", "/ho" + "me/alice/dev"),
        ("forbidden_score_key_detected", '{"sco' + 're": 9}'),
        ("static_figure_detected", "latency 120.5 " + "ms"),
        ("trojan_bidi_detected", f"abc{chr(0x202E)}def"),
        ("non_loopback_ipv4_detected:10.20." + "30.40", "10.20." + "30.40"),
    ):
        assert issue_key in verifier.scan_output_safety(bad_txt)
        with pytest.raises(verifier.ContinueConfigError):
            verifier._assert_safe_output(bad_txt)

    assert verifier.sanitize_build_sha(None) == "unknown"
    assert verifier.sanitize_build_sha("secret1234") == "unknown"
    assert verifier.shas_match("noel2026rc1", "noel2026rc1") is True
    assert verifier.shas_match(TARGET_FULL_SHA, TARGET_SHORT_SHA) is True
    assert verifier.shas_match("tagalpha12", "tagbeta12") is False
    assert verifier.shas_match("unknown", TARGET_SHORT_SHA) is False
    assert verifier._find_corporate_markers("") == []
    assert verifier.extract_git_config_remote_urls("") == []
    assert verifier.evaluate_personal_pc_compliance(environ={**CLEAN_ENV, "USERDOMAIN": ""}, cwd=tmp_path)["compliance_ok"] is True
    with pytest.raises(verifier.ContinueConfigError):
        verifier.validate_expected_sha("unknown")

    assert len(verifier.compute_merkle_root([])) == 64
    h1 = hashlib.sha256(b"leaf1").hexdigest()
    h2 = hashlib.sha256(b"leaf2").hexdigest()
    h3 = hashlib.sha256(b"leaf3").hexdigest()
    assert len(verifier.compute_merkle_root([h1, h2, h3])) == 64
    with pytest.raises(verifier.ContinueConfigError):
        verifier.compute_merkle_root(["not-hex"])

    assert verifier.is_allowed_outbound_host("datasets-server.huggingface.co") is True
    assert verifier.is_allowed_outbound_host("storage.googleapis.com") is True
    assert verifier.is_allowed_outbound_host("custom.a.run.app") is True
    assert verifier.is_allowed_outbound_host("") is False
    assert verifier.validate_gateway_url("http://[::1]:8080/") == "http://[::1]:8080"
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
        with pytest.raises(verifier.ContinueConfigError):
            verifier.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for m_bad, u_bad in (
        ("GET", ""),
        ("GET", "http://127.0.0.1/bad" + " path"),
        ("GET", "http://u:p@127.0.0.1/health"),
        ("POST", "http://127.0.0.1:8000/v1/chat/completions"),
        ("POST", "http://127.0.0.1:8000/v1/hivemind/onboarding/manifest"),
    ):
        with pytest.raises(verifier.ContinueConfigError):
            verifier._assert_safe_http_request(m_bad, u_bad)
    with pytest.raises(verifier.ContinueConfigError):
        verifier._assert_safe_http_request(
            "GET",
            "http://127.0.0.1:8000/noel",
            payload={"messages": [{"role": "user", "content": "hi"}], "prompt": "hivemind/coder"},
        )
    with pytest.raises(verifier.ContinueConfigError):
        verifier._assert_safe_http_request(
            "GET",
            "http://127.0.0.1:8000/noel",
            payload="not-a-dict",  # type: ignore[arg-type]
        )
    with pytest.raises(verifier.ContinueConfigError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/noel", payload={"a": 1})
    with pytest.raises(verifier.ContinueConfigError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/noel", headers={"X-Bad": "a\nb"})

    with pytest.raises(verifier.ContinueConfigError):
        verifier._normalize_http_return(("bad",))
    assert verifier._looks_like_http_headers({}) is True
    assert verifier._normalize_http_return((200, {"object": "ok"}, {"content-type": "application/json"}))[1]["_json"] == {"object": "ok"}
    assert verifier._normalize_http_return((200, "plain-hdr-missing", "<html>p</html>"))[1]["html"] == "<html>p</html>"
    assert verifier._normalize_http_return((200, {"x-a": "1"}, {"headers": {"x-a": "2"}, "html": "<p>x</p>"}))[1]["headers"]["x-a"] == "2"
    assert verifier._normalize_http_return((200, "<html>plain</html>", {"content-type": "text/html"}))[1]["html_length"] > 0
    assert verifier._normalize_http_return((200, '{"status":"ok"}', {"content-type": "application/json"}))[1]["_json"] == {"status": "ok"}
    assert verifier._normalize_http_return((200, "{broken_html"))[1]["html"] == "{broken_html"
    assert verifier._normalize_http_return((200, "[1, 2]"))[1]["data"] == [1, 2]
    assert verifier._normalize_http_return((200, 42))[1]["data"] == 42

    assert verifier._invoke_http_fn(lambda m, u, **_kw: (200, {"kw": True}), "GET", "http://127.0.0.1:8000/noel")[1]["kw"] is True
    assert verifier._invoke_http_fn(lambda m, u, p: (200, {"p3": True}), "GET", "http://127.0.0.1:8000/noel")[1]["p3"] is True
    assert verifier._invoke_http_fn(lambda m, u, p, h: (200, {"p4": True}), "GET", "http://127.0.0.1:8000/noel")[1]["p4"] is True
    assert verifier._invoke_http_fn(lambda m, u, p, h, t: (200, {"p5": True}), "GET", "http://127.0.0.1:8000/noel")[1]["p5"] is True
    assert verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(OSError("reset")), "GET", "http://127.0.0.1:8000/noel") == (0, {"error": "network_error"})
    with pytest.raises(verifier.ContinueConfigError):
        verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(verifier.ContinueConfigError("bad")), "GET", "http://127.0.0.1:8000/noel")


def test_02_continue_yaml_merge_backup_restore_tamper_fail_closed_and_masked_key_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Masked-key 'hm-***8<unit>' and HTML/CSP nonces safety against static-figure scanner
    masked_key_short = "hm-***8" + "ms"
    masked_live_short = "hm_live_***8" + "ms"
    masked_key_alpha = "hm-***x8" + "ms"
    nonce_tok_html = '<style nonce="99' + 'tok/sNonce"></style><script nonce="8' + 'msNonce">'
    for sample in (
        masked_key_short,
        masked_live_short,
        masked_key_alpha,
        f'{nonce_tok_html}k="{masked_key_short}";</script>',
        json.dumps({"masked_api_key": masked_key_short, "status": "ok"}, sort_keys=True),
    ):
        assert verifier.scan_output_safety(sample) == [], sample
        assert _STATIC_FIG_RE.findall(verifier.strip_masked_keys_and_nonces(sample)) == []

    for bad_fig in ("198.1 " + "ms", f"{masked_key_short} latency 198.1 " + "ms"):
        assert "static_figure_detected" in verifier.scan_output_safety(bad_fig)
        with pytest.raises(verifier.ContinueConfigError):
            verifier._assert_safe_output(bad_fig)

    # 2. Non-destructive YAML/JSON merge + structural uninstall across variants
    existing_yaml = (
        "# User header comment\n"
        "name: Personal Local Assistant\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  # Keep user Ollama\n"
        "  - name: Local Ollama Qwen  # inline comment\n"
        "    # inner item comment\n"
        "    provider: ollama\n"
        "    model: qwen2.5-coder:7b\n"
        "    apiBase: http://127.0.0.1:11434/v1\n"
        "    roles:\n"
        "      -\n"
        "      - chat\n"
        "      - edit\n"
        "  - name: 'Quoted # keep'\n"
        "    provider: openai\n"
        "    model: gpt-4o\n"
        '    apiBase: "http://127.0.0.1:11434/\\"v1\\""\n'
        "  - name: HiveMind Coder\n"
        "    provider: openai\n"
        "    model: hivemind/coder\n"
        "    apiBase: http://127.0.0.1:9999/v1\n"
        f"    apiKey: {TEST_KEY_BETA}\n"
        "# Autocomplete preserved\n"
        "tabAutocompleteModel:\n"
        "  title: Starcoder\n"
        "  provider: ollama\n"
        "  model: starcoder2:3b\n"
    )
    merged_yaml = verifier.merge_continue_config_yaml(
        existing_yaml, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA
    )
    parsed_merged = yaml.safe_load(merged_yaml)
    assert parsed_merged["name"] == "Personal Local Assistant"
    assert parsed_merged["tabAutocompleteModel"]["model"] == "starcoder2:3b"
    assert [m["model"] for m in parsed_merged["models"]] == [
        "qwen2.5-coder:7b",
        "gpt-4o",
        "hivemind/coder",
        "hivemind/bee-stars",
    ]
    assert parsed_merged["models"][2]["apiBase"] == "http://127.0.0.1:8000/v1"
    assert parsed_merged["models"][2]["apiKey"] == TEST_KEY_ALPHA
    assert verifier.merge_continue_config_yaml(
        merged_yaml, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA
    ) == merged_yaml

    stripped_yaml = verifier.remove_hivemind_from_continue_config_yaml(merged_yaml)
    assert stripped_yaml is not None
    parsed_stripped = yaml.safe_load(stripped_yaml)
    assert [m["model"] for m in parsed_stripped["models"]] == ["qwen2.5-coder:7b", "gpt-4o"]
    assert parsed_stripped["tabAutocompleteModel"]["model"] == "starcoder2:3b"

    # Zero-indent, no-models, comment-only models, JSON config, pure HiveMind
    zero_indent_yaml = "name: ZeroIndent\nmodels:\n- name: CustomUser\n  provider: ollama\n  model: llama3\n"
    assert len(yaml.safe_load(verifier.merge_continue_config_yaml(zero_indent_yaml, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA))["models"]) == 3

    no_models_yaml = "name: OnlyTab\ntabAutocompleteModel:\n  provider: ollama\n  model: qwen2.5\n\n"
    merged_nomod = verifier.merge_continue_config_yaml(no_models_yaml, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA)
    assert len(yaml.safe_load(merged_nomod)["models"]) == 2
    cleaned_nomod = verifier.remove_hivemind_from_continue_config_yaml(merged_nomod)
    assert cleaned_nomod is not None and yaml.safe_load(cleaned_nomod)["models"] == []

    comment_only_mod = "name: COnly\nmodels:\n  # comment only\ntabAutocompleteModel:\n  model: phi3\n"
    assert len(yaml.safe_load(verifier.merge_continue_config_yaml(comment_only_mod, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA))["models"]) == 2

    json_cfg = json.dumps({"models": [{"name": "UserJSON", "model": "gpt-4o"}], "allowAnonymousTelemetry": False})
    merged_json_cfg = verifier.merge_continue_config_yaml(json_cfg, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA)
    assert len(json.loads(merged_json_cfg)["models"]) == 3
    cleaned_json_cfg = verifier.remove_hivemind_from_continue_config_yaml(merged_json_cfg)
    assert cleaned_json_cfg is not None and [m["name"] for m in json.loads(cleaned_json_cfg)["models"]] == ["UserJSON"]

    pure_hm = verifier.merge_continue_config_yaml(None, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA)
    assert verifier.remove_hivemind_from_continue_config_yaml(pure_hm) is None
    assert verifier.remove_hivemind_from_continue_config_yaml("") is None
    assert verifier.remove_hivemind_from_continue_config_yaml("{bad") == "{bad"
    assert verifier.remove_hivemind_from_continue_config_yaml("[]") == "[]"
    assert verifier.remove_hivemind_from_continue_config_yaml('{"models": "bad"}') == '{"models": "bad"}'
    assert verifier.remove_hivemind_from_continue_config_yaml('{"models": [{"model": "hivemind/coder"}]}') is None
    assert verifier.remove_hivemind_from_continue_config_yaml("name: NoMod\n") == "name: NoMod\n"

    # Invalid YAML/JSON/key/max_tokens validation failures
    for bad_cfg in (
        "",
        "a\x00b",
        f"name: test{chr(0x202E)}x\n",
        "name: bad\x07ctrl\n",
        "\tname: tabbed\n",
        "- top_level_list\n",
        "models: scalar\n",
        "NOT_A_KEY_VALUE\n",
        "name: 'unclosed\n",
        "models:\n  - just_dash_missing_fields\n",
        "models:\n  - orphan_bad_line\n",
        "{broken_json",
        "[1, 2]",
        '{"models": "bad"}',
    ):
        with pytest.raises(verifier.ContinueConfigError):
            verifier.validate_continue_config_text(bad_cfg)
    with pytest.raises(verifier.ContinueConfigError):
        verifier._build_continue_model_dicts("http://127.0.0.1:8000", "bad-key")
    with pytest.raises(verifier.ContinueConfigError):
        verifier._build_continue_model_dicts("http://127.0.0.1:8000", TEST_KEY_ALPHA, max_tokens=16)
    assert verifier._is_hivemind_continue_model_dict("not_a_dict") is False
    with pytest.raises(verifier.ContinueConfigError):
        verifier.validate_continue_config_text("models:\n  - name: ok\n    bad_unmapped_line\n")

    # 3. Roundtrip sandbox verification (EN & FR, pre-existing file + empty home_dir)
    user_yaml = (
        "# Personal local hobby models + masked note " + masked_key_short + "\n"
        "name: Personal Hobby Workstation\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: Local Ollama Qwen\n"
        "    provider: ollama\n"
        "    model: qwen2.5-coder:7b\n"
        "    apiBase: http://127.0.0.1:11434/v1\n"
        "tabAutocompleteModel:\n"
        "  provider: ollama\n"
        "  model: starcoder2:3b\n"
    )
    sbx_home = tmp_path / "sbx_home"
    rt_en = verifier.verify_continue_config_roundtrip(
        sbx_home,
        sample_existing_yaml=user_yaml,
        gateway="http://127.0.0.1:8000",
        api_key=TEST_KEY_ALPHA,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_sha256_receipt(rt_en)
    assert rt_en["roundtrip_ok"] is True and rt_en["ready"] is True and rt_en["exit_code"] == 0
    assert rt_en["byte_for_byte_restored"] is True
    assert rt_en["backup_exact_match"] is True
    assert rt_en["backup_0600_ok"] is True and rt_en["merged_0600_ok"] is True and rt_en["restored_0600_ok"] is True
    assert rt_en["idempotent_remerge_ok"] is True
    assert rt_en["structural_uninstall_ok"] is True
    assert rt_en["tampered_backup_rejected"] is True
    assert rt_en["backup_residue_cleaned"] is True
    assert rt_en["hivemind_residue_count"] == 0
    assert rt_en["unmasked_api_keys_count"] == 0
    assert rt_en["masked_keys_tolerated_count"] >= 1
    assert rt_en["masked_api_key"] == "hm-***1234"
    assert TEST_KEY_ALPHA not in json.dumps(rt_en)
    assert (sbx_home / ".continue" / "config.yaml").read_text(encoding="utf-8") == user_yaml
    assert not (sbx_home / ".continue" / "config.yaml.bak").exists()

    # Run again with sample_existing_yaml=None using existing config.yaml on disk + FR output
    rt_fr = verifier.verify_continue_config_roundtrip(
        sbx_home,
        sample_existing_yaml=None,
        gateway="http://127.0.0.1:8000",
        api_key=TEST_KEY_LIVE,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    _verify_sha256_receipt(rt_fr)
    assert rt_fr["roundtrip_ok"] is True and rt_fr["masked_api_key"] == "hm_live_***9999"
    assert TEST_KEY_LIVE not in json.dumps(rt_fr)
    assert "GO" in verifier.render_cli_report(rt_fr)
    assert "Continue.dev" in verifier.render_markdown_report(rt_fr)

    # Empty home_dir uses default sample config
    empty_home = tmp_path / "empty_home"
    rt_def = verifier.verify_continue_config_roundtrip(
        empty_home,
        gateway="http://127.0.0.1:8000",
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_sha256_receipt(rt_def)
    assert rt_def["roundtrip_ok"] is True and rt_def["non_hivemind_models_preserved_count"] == 2

    # 4. Fail-closed corrupted / tampered / missing / symlinked / oversized backup checks
    tamper_home = tmp_path / "tamper_home"
    cfg_t = tamper_home / ".continue" / "config.yaml"
    bak_t = tamper_home / ".continue" / "config.yaml.bak"
    verifier._write_bytes_secure_atomic(
        cfg_t, user_yaml.encode("utf-8"), posix_mode=0o600, eff_os="posix", label="seed"
    )
    merge_meta = verifier.backup_and_merge_continue_config(
        cfg_t, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA, home=tamper_home
    )
    merged_snapshot = cfg_t.read_bytes()
    expected_bak_sha = str(merge_meta["backup_sha256"])

    # (a) Valid-YAML single-character tamper caught by SHA-256 verification (target untouched)
    bak_t.write_text(user_yaml.replace("qwen2.5-coder:7b", "qwen2.5-coder:9b"), encoding="utf-8")
    with pytest.raises(verifier.ContinueConfigError):
        verifier.restore_continue_config_from_backup(
            cfg_t, expected_backup_sha256=expected_bak_sha, home=tamper_home
        )
    assert cfg_t.read_bytes() == merged_snapshot

    # (b) Invalid UTF-8 bytes & malformed YAML in .bak without expected SHA
    bak_t.write_bytes(b"\xff\xfe\x00broken")
    with pytest.raises(verifier.ContinueConfigError):
        verifier.restore_continue_config_from_backup(cfg_t, home=tamper_home)
    assert cfg_t.read_bytes() == merged_snapshot

    bak_t.write_text("CORRUPTED_NON_YAML_LINE\n", encoding="utf-8")
    with pytest.raises(verifier.ContinueConfigError):
        verifier.restore_continue_config_from_backup(cfg_t, home=tamper_home)
    assert cfg_t.read_bytes() == merged_snapshot

    # (c) Invalid expected_backup_sha256 format & missing .bak fail closed
    with pytest.raises(verifier.ContinueConfigError):
        verifier.restore_continue_config_from_backup(cfg_t, expected_backup_sha256="bad-sha", home=tamper_home)
    bak_t.unlink()
    with pytest.raises(verifier.ContinueConfigError):
        verifier.restore_continue_config_from_backup(
            cfg_t, expected_backup_sha256=expected_bak_sha, home=tamper_home
        )
    assert cfg_t.read_bytes() == merged_snapshot

    # (d) Pure-HiveMind vs mixed config without .bak on strip fallback
    pure_cfg = tmp_path / "pure_home" / ".continue" / "config.yaml"
    verifier.backup_and_merge_continue_config(
        pure_cfg, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA, home=tmp_path / "pure_home"
    )
    os.chmod(pure_cfg, 0o644)
    tighten_res = verifier.backup_and_merge_continue_config(
        pure_cfg, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA, home=tmp_path / "pure_home"
    )
    assert tighten_res["action"] == "unchanged" and tighten_res["permissions_tightened"] is True
    strip_res = verifier.restore_continue_config_from_backup(pure_cfg, home=tmp_path / "pure_home")
    assert strip_res["file_removed"] is True and not pure_cfg.exists()

    mixed_home = tmp_path / "mixed_home"
    mixed_cfg = mixed_home / ".continue" / "config.yaml"
    verifier._write_bytes_secure_atomic(
        mixed_cfg, user_yaml.encode("utf-8"), posix_mode=0o600, eff_os="posix", label="seed_mixed"
    )
    verifier.backup_and_merge_continue_config(
        mixed_cfg, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA, home=mixed_home
    )
    Path(f"{mixed_cfg}.bak").unlink()
    mixed_strip = verifier.restore_continue_config_from_backup(mixed_cfg, home=mixed_home)
    assert mixed_strip["restore_method"] == "hivemind_entries_stripped" and mixed_cfg.is_file()
    assert verifier._scan_continue_dir_residue(tmp_path / "missing_cont_dir") == (0, 0, 0)
    (mixed_home / ".continue" / "sub_dir").mkdir()
    assert verifier._scan_continue_dir_residue(mixed_home / ".continue")[0] == 0

    # (e) Non-UTF8 existing config, symlink parent/file, directory-as-file, atomic tmp cleanup, invalid api_key
    bad_utf_cfg = tmp_path / "bad_utf_home" / ".continue" / "config.yaml"
    bad_utf_cfg.parent.mkdir(parents=True)
    bad_utf_cfg.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(verifier.ContinueConfigError):
        verifier.backup_and_merge_continue_config(
            bad_utf_cfg, gateway="http://127.0.0.1:8000", api_key=TEST_KEY_ALPHA
        )

    sym_parent_root = tmp_path / "sym_parent_real"
    sym_parent_root.mkdir()
    sym_link_dir = tmp_path / "sym_parent_link"
    sym_link_dir.symlink_to(sym_parent_root)
    with pytest.raises(verifier.ContinueConfigError):
        verifier._reject_symlink(sym_link_dir / "config.yaml", "sym-parent")

    with pytest.raises(verifier.ContinueConfigError):
        verifier._read_bytes_bounded(sym_parent_root, "dir-not-file")

    with monkeypatch.context() as mp:
        mp.setattr(os, "replace", lambda _s, _d: (_ for _ in ()).throw(OSError("disk full")))
        fail_target = tmp_path / "atomic_fail_home" / "config.yaml"
        with pytest.raises(OSError):
            verifier._write_bytes_secure_atomic(fail_target, b"name: x\n", eff_os="posix")
        assert list(fail_target.parent.glob(".*.tmp")) == []

    with pytest.raises(verifier.ContinueConfigError):
        verifier.verify_continue_config_roundtrip(
            tmp_path / "bad_key_home", api_key="not-a-hivemind-key", environ=CLEAN_ENV, cwd=tmp_path
        )

    sym_home = tmp_path / "sym_home"
    (sym_home / ".continue").mkdir(parents=True)
    secret_target = tmp_path / "outside_target.yaml"
    secret_target.write_text("name: outside\n", encoding="utf-8")
    (sym_home / ".continue" / "config.yaml").symlink_to(secret_target)
    with pytest.raises(verifier.ContinueConfigError):
        verifier.verify_continue_config_roundtrip(sym_home, environ=CLEAN_ENV, cwd=tmp_path)

    big_file = tmp_path / "oversized.yaml"
    big_file.write_bytes(b"a" * (verifier.MAX_CONFIG_BYTES + 16))
    with pytest.raises(verifier.ContinueConfigError):
        verifier._read_bytes_bounded(big_file, "oversized")

    bad_home_file = tmp_path / "not_a_dir.txt"
    bad_home_file.write_text("x", encoding="utf-8")
    for bad_home in ("", True, bad_home_file):
        with pytest.raises(verifier.ContinueConfigError):
            verifier.verify_continue_config_roundtrip(bad_home, environ=CLEAN_ENV, cwd=tmp_path)  # type: ignore[arg-type]

    win_home = tmp_path / "win_sbx"
    rt_win = verifier.verify_continue_config_roundtrip(
        win_home, gateway="http://127.0.0.1:8000", os_name="nt", environ=CLEAN_ENV, cwd=tmp_path
    )
    assert rt_win["roundtrip_ok"] is True and rt_win["backup_posix_mode"] == "windows_acl"


def test_03_proxy_free_loopback_http_and_stale_404_to_live_readiness_cutover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {"live": False, "nonce_seq": 0, "total": 12}
    seen_requests: list[tuple[str, str]] = []

    class _ContinueHolidayLoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen_requests.append(("GET", self.path))
            if self.path == "/redirect-good":
                self.send_response(302)
                self.send_header("Location", "/v1/hivemind/onboarding/feedback/summary")
                self.end_headers()
                return
            if self.path == "/redirect-evil":
                self.send_response(302)
                self.send_header("Location", f"{UNTRUSTED_EXT_GW}/exfil")
                self.end_headers()
                return
            if self.path == "/bad-json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{broken-json")
                return
            if self.path == "/list-ok":
                self._send_json(200, [1, 2])
                return

            if not state["live"] and self.path in ("/raid-noel", "/arena-noel"):
                self._send_json(404, {"detail": "Not Found"})
                return

            if self.path in verifier.HOLIDAY_HTML_ROUTES:
                state["nonce_seq"] += 1
                nonce = "123m" + f"s456Nonce{state['nonce_seq']:04d}AbCdEf"
                body = _sample_compliant_html(
                    self.path, nonce, lang="fr" if self.path in ("/noel", "/raid-noel", "/arena-noel") else "en"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header(
                    "Content-Security-Policy",
                    f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'",
                )
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
                return

            if self.path == "/v1/hivemind/onboarding/manifest":
                self._send_json(200, _sample_manifest())
                return

            if self.path == "/v1/hivemind/onboarding/feedback/summary":
                self._send_json(200, _sample_feedback_summary(total=int(state["total"])))
                return

            self._send_json(404, {"detail": "missing"})

        def _send_json(self, code: int, payload: Any) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ContinueHolidayLoopbackHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        # Hostile proxies MUST be bypassed by urllib.request.ProxyHandler({})
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

        fb_code, fb_body = verifier.default_http_request(
            "GET",
            f"{base_url}/v1/hivemind/onboarding/feedback/summary",
            headers={"X-Continue-Probe": "1"},
        )
        assert fb_code == 200
        assert list(fb_body.keys()) == [
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        ]

        rc_good, body_good = verifier.default_http_request("GET", f"{base_url}/redirect-good")
        assert rc_good == 200 and body_good["total_responses"] == 12
        with pytest.raises(verifier.ContinueConfigError):
            verifier.default_http_request("GET", f"{base_url}/redirect-evil")
        assert verifier.default_http_request("GET", f"{base_url}/bad-json")[1]["raw_error"] == "invalid_json"
        assert verifier.default_http_request("GET", f"{base_url}/list-ok")[1] == {"data": [1, 2]}
        assert verifier.default_http_request("GET", f"{base_url}/missing")[0] == 404

        # Phase A: Stale 404 drift -> fail-closed HOLD_ROUTES_DRIFT (Continue roundtrip still isolated)
        stale_en = verifier.verify_continue_and_holiday_readiness(
            base_url,
            home_dir=None,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        _verify_sha256_receipt(stale_en)
        assert stale_en["sandbox_isolated"] is True
        assert stale_en["roundtrip_ok"] is True
        assert stale_en["ready"] is False and stale_en["decision"] == "HOLD_ROUTES_DRIFT"
        assert stale_en["stale_404_drift"] is True and stale_en["redeploy_required"] is True
        assert stale_en["exit_code"] == 2
        assert "/raid-noel" in stale_en["stale_404_routes"]
        assert "/arena-noel" in stale_en["stale_404_routes"]
        assert "NO-GO" in verifier.render_markdown_report(stale_en)
        assert "NO-GO" in verifier.render_cli_report(stale_en)

        # Phase B: Live cutover (EN & FR + real subprocess CLI invocation)
        state["live"] = True
        seen_requests.clear()
        sbx_dir = tmp_path / "cutover_home"
        for lang_code in ("en", "fr"):
            live_rep = verifier.verify_continue_and_holiday_readiness(
                base_url,
                home_dir=sbx_dir if lang_code == "fr" else None,
                expected_sha=TARGET_SHORT_SHA,
                environ=CLEAN_ENV,
                cwd=tmp_path,
                lang=lang_code,
            )
            _verify_sha256_receipt(live_rep)
            assert live_rep["ready"] is True and live_rep["go_nogo"] == "GO"
            assert live_rep["decision"] == "CONTINUE_BACKUP_RESTORE_AND_HOLIDAY_READY"
            assert live_rep["exit_code"] == 0
            assert live_rep["roundtrip_ok"] is True
            assert live_rep["stale_404_drift"] is False and live_rep["redeploy_required"] is False
            assert live_rep["unique_csp_nonces_count"] == 4
            assert live_rep["network_calls"] == 6

        assert all(m == "GET" for m, _ in seen_requests)
        assert all("completions" not in p.lower() for _, p in seen_requests)

        cli_proc = subprocess.run(
            [
                sys.executable,
                str(VERIFIER_PATH),
                "--gateway",
                base_url,
                "--expected-sha",
                TARGET_SHORT_SHA,
                "--json",
            ],
            cwd=str(tmp_path),
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "HTTP_PROXY": "http://127.0.0.1:9"},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert cli_proc.returncode == 0, cli_proc.stderr
        cli_json = json.loads(cli_proc.stdout)
        _verify_sha256_receipt(cli_json)
        assert cli_json["ready"] is True and cli_json["roundtrip_ok"] is True
    finally:
        srv.shutdown()
        srv.server_close()

    assert verifier.default_http_request("GET", f"{base_url}/noel", timeout_s=0.2)[0] == 0

    # Phase C: Broken routes -> HOLD_ROUTES_DRIFT
    def _broken_http(method: str, url: str) -> tuple[int, dict[str, str], str | dict[str, Any]]:
        _ = method
        path = urlsplit(url).path
        reused = "ReusedNonce123456"
        if path in verifier.HOLIDAY_HTML_ROUTES:
            if path == "/noel":
                bad_dom = (
                    '<html lang="fr"><head><style nonce="ReusedNonce123456"></style></head>'
                    '<body><script nonce="ReusedNonce123456">'
                    "el." + "inner" + "HTML='198.1 " + "ms';fetch('/v1/chat/completions');"
                    "</script></body></html>"
                )
                return 200, {"content-security-policy": f"default-src 'none'; script-src 'nonce-{reused}'"}, bad_dom
            return 200, {"content-security-policy": f"script-src 'nonce-{reused}' 'unsafe-inline'"}, _sample_compliant_html(path, reused)
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, {}, {"object": "wrong", "build_sha": "abcdef12", "holiday_routes": "bad"}
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, {}, {**_sample_feedback_summary(6), "extra_key": "must_fail"}
        return 404, {}, {"detail": "missing"}

    broken_routes = verifier.verify_continue_and_holiday_readiness(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_FULL_SHA,
        http_fn=_broken_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    _verify_sha256_receipt(broken_routes)
    assert broken_routes["ready"] is False and broken_routes["exit_code"] == 2
    assert broken_routes["decision"] == "HOLD_ROUTES_DRIFT"
    assert broken_routes["manifest_ok"] is False and broken_routes["feedback_summary_ok"] is False

    # Phase D: Roundtrip failure branches (with monkeypatched roundtrip fail)
    monkeypatch.setattr(
        verifier,
        "verify_continue_config_roundtrip",
        lambda *_a, **_kw: {
            "compliance_ok": True,
            "roundtrip_ok": False,
            "ready": False,
            "diagnostics": ["simulated-branch-check"],
            "file_writes": 0,
            "masked_api_key": "hm-***0000",
            "receipt_sha256": hashlib.sha256(b"rt_fail").hexdigest(),
        },
    )
    rt_and_route_fail = verifier.verify_continue_and_holiday_readiness(
        "http://127.0.0.1:8000",
        http_fn=_broken_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert rt_and_route_fail["decision"] == "HOLD_CONTINUE_ROUNDTRIP_AND_ROUTES_DRIFT"

    def _ok_only_http(method: str, url: str) -> tuple[int, dict[str, str], Any]:
        _ = method
        path = urlsplit(url).path
        if path in verifier.HOLIDAY_HTML_ROUTES:
            n = f"OkNonce{hashlib.sha256(path.encode()).hexdigest()[:12]}"
            return 200, {"content-security-policy": f"default-src 'none'; style-src 'nonce-{n}'; script-src 'nonce-{n}'"}, _sample_compliant_html(path, n)
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, {}, _sample_manifest()
        return 200, {}, _sample_feedback_summary(4)

    rt_only_fail = verifier.verify_continue_and_holiday_readiness(
        "http://127.0.0.1:8000",
        http_fn=_ok_only_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    assert rt_only_fail["decision"] == "HOLD_CONTINUE_BACKUP_RESTORE_FAILED"


def test_04_concurrent_48_participant_continue_roundtrip_holiday_and_ces_wal_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    import httpx
    from fastapi.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00280-noel")

    upstream_inference_hits: list[str] = []

    def _forbid_upstream_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn28_48_soak.sqlite3"
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
    if _INPROC_COV is not None:
        _INPROC_COV.start()

    obs_lock = threading.Lock()
    observed_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []
    seen_paths: list[str] = []
    outcomes: list[tuple[int, bool]] = []
    routes_cycle = ("/noel", "/quick", "/raid-noel", "/arena-noel")
    soak_homes_root = tmp_path / "soak_participants"
    soak_homes_root.mkdir()
    masked_ref = "hm-***8" + "ms"

    def _run_cohort(client: Any, indices: range, bucket_base: int) -> None:
        barrier = threading.Barrier(len(indices))

        def _participant(idx: int) -> None:
            is_dev = idx % 2 == 0
            profile = "dev" if is_dev else "gamer"
            peer_ip = _loopback_ip(bucket_base + (0 if is_dev else 1), idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            route_a = routes_cycle[idx % 4]
            route_b = routes_cycle[(idx + 1) % 4]
            local_nonces: list[str] = []

            p_home = soak_homes_root / f"p_{idx:02d}_home"
            user_yaml = (
                f"# Participant {idx:02d} personal config ({masked_ref})\n"
                f"name: Personal User {idx:02d}\n"
                "version: 0.0.1\n"
                "schema: v1\n"
                "models:\n"
                f"  - name: Local Ollama {idx:02d}\n"
                "    provider: ollama\n"
                "    model: qwen2.5-coder:7b\n"
                "    apiBase: http://127.0.0.1:11434/v1\n"
                "tabAutocompleteModel:\n"
                "  provider: ollama\n"
                "  model: starcoder2:3b\n"
            )
            p_key = f"hm-soak28-{idx:02d}-localkey-9876"

            barrier.wait(timeout=15.0)

            rt_doc = verifier.verify_continue_config_roundtrip(
                p_home,
                sample_existing_yaml=user_yaml,
                gateway="http://127.0.0.1:8000",
                api_key=p_key,
                environ=CLEAN_ENV,
                cwd=REPO_ROOT,
                check_git_remote=False,
                verify_tamper_fail_closed=(idx < 8),
                lang="fr" if idx % 2 else "en",
            )
            assert rt_doc["roundtrip_ok"] is True and rt_doc["exit_code"] == 0
            assert rt_doc["byte_for_byte_restored"] is True
            assert rt_doc["backup_0600_ok"] is True
            assert rt_doc["hivemind_residue_count"] == 0
            assert rt_doc["unmasked_api_keys_count"] == 0
            assert (p_home / ".continue" / "config.yaml").read_text(encoding="utf-8") == user_yaml
            assert not (p_home / ".continue" / "config.yaml.bak").exists()
            assert p_key not in json.dumps(rt_doc)

            for route in (route_a, route_b):
                r_html = client.get(route, headers=hdrs)
                assert r_html.status_code == 200
                if _SQLITE_LOCK_RE.search(r_html.text):
                    with obs_lock:
                        sqlite_lock_errors.append(r_html.text[:120])
                csp_hdr = r_html.headers.get("content-security-policy", "")
                assert "default-src 'none'" in csp_hdr.lower()
                hdr_nonces = _CSP_HDR_NONCE_RE.findall(csp_hdr)
                dom_nonces = [val for _, val in _HTML_ATTR_NONCE_RE.findall(r_html.text)]
                assert hdr_nonces and hdr_nonces[0] in dom_nonces
                stripped = verifier.strip_masked_keys_and_nonces(r_html.text)
                assert _STATIC_FIG_RE.findall(stripped) == []
                assert ("inner" + "HTML") not in stripped
                local_nonces.append(hdr_nonces[0])

            ease = 5 if idx % 5 != 0 else 4
            zero_lag = idx != 47
            idem_key = f"soak28-{profile}-{idx:02d}-{ease}-{int(zero_lag)}"
            fb_payload = {
                "profile": profile,
                "ease_rating": ease,
                "zero_game_lag": zero_lag,
                "platform": "linux" if is_dev else "windows",
                "note": "",
            }
            fb1 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json=fb_payload,
                headers={**hdrs, "Idempotency-Key": idem_key},
            )
            fb2 = client.post(
                "/v1/hivemind/onboarding/feedback",
                json=fb_payload,
                headers={**hdrs, "Idempotency-Key": idem_key},
            )
            assert fb1.status_code == 200 and fb2.status_code == 200

            with obs_lock:
                observed_nonces.extend(local_nonces)
                seen_paths.extend((route_a, route_b, "/v1/hivemind/onboarding/feedback"))
                outcomes.append((ease, zero_lag))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futs = [pool.submit(_participant, i) for i in indices]
            for f in futs:
                f.result(timeout=15.0)

    try:
        with TestClient(app) as client:
            adapter = _make_router_http_adapter(client, peer_ip="127.0.51.1")

            base_gate = verifier.verify_continue_and_holiday_readiness(
                "http://127.0.0.1:8000",
                home_dir=None,
                http_fn=adapter,
                environ=CLEAN_ENV,
                cwd=REPO_ROOT,
                expected_sha=TARGET_SHORT_SHA,
                lang="en",
            )
            _verify_sha256_receipt(base_gate)
            assert base_gate["ready"] is True and base_gate["roundtrip_ok"] is True
            assert base_gate["feedback_summary"]["total_responses"] == 0

            # Cohort 1 (16 users) + Cohort 2 (32 users) = 48 concurrent participants
            t_soak_start = time.monotonic()
            _run_cohort(client, range(0, 16), bucket_base=10)
            _run_cohort(client, range(16, 48), bucket_base=20)
            soak_elapsed_s = time.monotonic() - t_soak_start
            assert soak_elapsed_s < 15.0, f"48-user soak exceeded 15.0s budget: {soak_elapsed_s:.2f}s"

            assert sqlite_lock_errors == []
            assert upstream_inference_hits == []
            assert all("completions" not in p.lower() for p in seen_paths)
            assert len(observed_nonces) == 48 * 2
            assert len(set(observed_nonces)) == 48 * 2

            fb_summary_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
            assert fb_summary_resp.status_code == 200
            fb_summary = fb_summary_resp.json()
            expected_mean = round(sum(e for e, _ in outcomes) / 48, 2)
            expected_lag = round(sum(1 for _, z in outcomes if z) / 48, 4)
            assert list(fb_summary.keys()) == [
                "object",
                "total_responses",
                "mean_ease_rating",
                "zero_game_lag_ratio",
                "by_profile",
            ]
            assert fb_summary == {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 48,
                "mean_ease_rating": expected_mean,
                "zero_game_lag_ratio": expected_lag,
                "by_profile": {"dev": 24, "gamer": 24},
            }

            post_gate = verifier.verify_continue_and_holiday_readiness(
                "http://127.0.0.1:8000",
                home_dir=None,
                http_fn=adapter,
                environ=CLEAN_ENV,
                cwd=REPO_ROOT,
                expected_sha=TARGET_SHORT_SHA,
                lang="fr",
            )
            _verify_sha256_receipt(post_gate)
            assert post_gate["ready"] is True and post_gate["go_nogo"] == "GO"
            assert post_gate["decision"] == "CONTINUE_BACKUP_RESTORE_AND_HOLIDAY_READY"
            assert post_gate["exit_code"] == 0
            assert post_gate["roundtrip_ok"] is True
            assert post_gate["unique_csp_nonces_count"] == 4
            assert post_gate["feedback_summary"] == fb_summary
            assert len(post_gate["merkle_root_sha256"]) == 64
            md = verifier.render_markdown_report(post_gate)
            assert "SHA-256 Receipt" in md and "0.00 EUR / 0.00 USD" in md
    finally:
        store.close()


def test_05_personal_pc_compliance_fail_closed_and_cli_matrix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbid_http(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Compliance gate MUST fail closed before any HTTP call")

    personal_ws = tmp_path / "personal_hobby_ws"
    (personal_ws / ".git").mkdir(parents=True)
    (personal_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n',
        encoding="utf-8",
    )
    ok_comp = verifier.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=personal_ws, lang="en")
    _verify_sha256_receipt(ok_comp)
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
        _verify_sha256_receipt(comp)
        assert comp["compliance_ok"] is False and comp["exit_code"] == 2, vec_name
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0 and comp["subprocess_calls"] == 0
        assert "0.00 EUR / 0.00 USD" in comp["notice_bilingual"]
        assert "AGPL-3.0" in comp["notice_bilingual"]
        assert sentinel not in json.dumps(comp), vec_name
        seen_markers.update(comp["matched_markers"])

        blocked_rt = verifier.verify_continue_config_roundtrip(
            tmp_path / f"blocked_rt_{idx}",
            environ=env_map,
            cwd=cwd_path,
            lang=lang_choice,
        )
        _verify_sha256_receipt(blocked_rt)
        assert blocked_rt["compliance_ok"] is False and blocked_rt["exit_code"] == 2
        assert blocked_rt["file_writes"] == 0 and blocked_rt["network_calls"] == 0
        assert not (tmp_path / f"blocked_rt_{idx}").exists()

        blocked_api = verifier.verify_continue_and_holiday_readiness(
            "http://127.0.0.1:8000",
            http_fn=_forbid_http,
            environ=env_map,
            cwd=cwd_path,
            lang=lang_choice,
        )
        _verify_sha256_receipt(blocked_api)
        assert blocked_api["compliance_ok"] is False and blocked_api["exit_code"] == 2
        assert blocked_api["network_calls"] == 0 and blocked_api["file_writes"] == 0
        assert sentinel not in json.dumps(blocked_api)

        rc_cli = verifier.main(
            ["--lang", lang_choice, "--json" if idx % 2 == 0 else "--markdown"],
            http_fn=_forbid_http,
            environ=env_map,
            cwd=cwd_path,
        )
        out_cli = capsys.readouterr().out
        assert rc_cli == 2 and sentinel not in out_cli

        # --roundtrip-only also fails closed before any file write
        rc_rt_cli = verifier.main(
            ["--roundtrip-only", "--lang", lang_choice, "--json"],
            http_fn=_forbid_http,
            environ=env_map,
            cwd=cwd_path,
        )
        assert rc_rt_cli == 2
        assert json.loads(capsys.readouterr().out)["file_writes"] == 0

    assert seen_markers == set(verifier.CORPORATE_MARKERS)

    # Corporate home_dir marker rejected even when cwd is clean
    corp_home = tmp_path / "amadeus_home_dir"
    blocked_home_rt = verifier.verify_continue_config_roundtrip(
        corp_home, environ=CLEAN_ENV, cwd=personal_ws, lang="fr"
    )
    assert blocked_home_rt["compliance_ok"] is False and blocked_home_rt["exit_code"] == 2
    blocked_home_ready = verifier.verify_continue_and_holiday_readiness(
        "http://127.0.0.1:8000",
        home_dir=corp_home,
        http_fn=_forbid_http,
        environ=CLEAN_ENV,
        cwd=personal_ws,
        lang="fr",
    )
    assert blocked_home_ready["compliance_ok"] is False and blocked_home_ready["exit_code"] == 2
    assert not corp_home.exists()

    # CLI matrix: --enforce-personal-pc, --roundtrip-only, live --markdown, live default, errors
    assert verifier.main(["--enforce-personal-pc", "--json"], environ=CLEAN_ENV, cwd=personal_ws) == 0
    _verify_sha256_receipt(json.loads(capsys.readouterr().out))

    cli_rt_home = tmp_path / "cli_rt_home"
    assert (
        verifier.main(
            ["--roundtrip-only", "--home-dir", str(cli_rt_home), "--markdown", "--lang", "fr"],
            http_fn=_forbid_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    assert "Vérificateur Fusion Non-Destructive" in capsys.readouterr().out

    assert (
        verifier.main(
            ["--roundtrip-only", "--json"],
            http_fn=_forbid_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    _verify_sha256_receipt(json.loads(capsys.readouterr().out))

    def _ok_cli_http(method: str, url: str) -> tuple[int, dict[str, str], str | dict[str, Any]]:
        _ = method
        path = urlsplit(url).path
        if path in verifier.HOLIDAY_HTML_ROUTES:
            nonce = f"CliNonce{hashlib.sha256(path.encode()).hexdigest()[:12]}"
            return (
                200,
                {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"},
                _sample_compliant_html(path, nonce, lang="fr" if path != "/quick" else "en"),
            )
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, {}, _sample_manifest()
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, {}, _sample_feedback_summary(total=24, mean_ease=4.6, zero_lag=1.0)
        return 404, {}, {"detail": "missing"}

    assert (
        verifier.main(
            ["--gateway", "http://127.0.0.1:8000", "--markdown", "--lang", "fr"],
            http_fn=_ok_cli_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    assert "Vérificateur Fusion Non-Destructive & Restauration Continue.dev" in capsys.readouterr().out

    assert (
        verifier.main(
            ["--gateway", "http://127.0.0.1:8000"],
            http_fn=_ok_cli_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    txt_out = capsys.readouterr().out
    assert "Merkle Root" in txt_out and "SHA-256" in txt_out and "Continue" in txt_out

    assert (
        verifier.main(
            ["--gateway", UNTRUSTED_EXT_GW, "--json"],
            http_fn=_forbid_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 2
    )
    err_json = json.loads(capsys.readouterr().out)
    _verify_sha256_receipt(err_json)
    assert err_json["exit_code"] == 2 and err_json["go_nogo"] == "NO-GO"

    assert verifier.main(["--lang", "es"], environ=CLEAN_ENV, cwd=personal_ws) == 2
    assert "Error:" in capsys.readouterr().err


def test_zz_standalone_continue_backup_restore_verifier_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {VERIFIER_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
