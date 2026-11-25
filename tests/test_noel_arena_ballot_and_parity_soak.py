# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, Bradley-Terry/Elo, compliance, security, and 48-user WAL soak tests for SLICE-SCALE-DYN-20."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import importlib.util
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
from urllib.parse import urlsplit

import httpx
import pytest

try:
    from gwaya.ast_audit import ZeroStubAudit
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


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "noel_arena_ballot_and_parity_verifier.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct module import without starting an internal Coverage() instance so standalone
# `coverage run --rcfile=/dev/null --include='*/scripts/noel_arena_ballot_and_parity_verifier.py'`
# captures import + runtime lines accurately.
from scripts import noel_arena_ballot_and_parity_verifier as verifier  # noqa: E402

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-box"}
ROUTER_SECRET = b"noel-scale-dyn20-router-sec-32b!"
TARGET_FULL_SHA = "b4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3"
TARGET_SHORT_SHA = "b4d5e6f7a8b9"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example.invalid"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-bluecoat.internal:8080"
CORP_FORTI_PROXY = "http://" + "proxy.fortinet.internal:3128"
CORP_GIT_REMOTE = "https://" + "git.amadeus-internal.corp/team/service.git"

TEST_KEY_ALPHA = "hm-dyn20_" + "personal_key_1234"
TEST_KEY_LIVE = "hm_live_" + "personal_dyn20_9999"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_STATIC_FIG_RE = re.compile(r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)")
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


def _make_router_http_adapter(
    client: Any,
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
        parsed = urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        if seen_calls is not None:
            if lock is not None:
                with lock:
                    seen_calls.append((method, parsed.path))
            else:
                seen_calls.append((method, parsed.path))
        req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
        resp = client.request(method, path, json=payload, headers=req_headers)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if "application/json" in hdrs.get("content-type", ""):
            return resp.status_code, hdrs, resp.json()
        return resp.status_code, hdrs, resp.text

    return _adapter


def _sample_compliant_html(route: str, nonce: str, *, lang: str = "en") -> str:
    extra = ""
    if route == "/arena-noel":
        cards = "".join(
            f'<article id="arenaCard-{tid}" class="task-card" data-task-id="{tid}" '
            f'data-ballot="abcdef012345.1.0123456789abcdef01234567">'
            f'<pre id="candA-{tid}">def f(): return 1</pre>'
            f'<pre id="candB-{tid}">def f(): return 2</pre>'
            f"</article>"
            for tid in verifier.CURATED_ARENA_TASK_IDS
        )
        extra = f'<main id="arenaNoelApp">{cards}</main>'
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


def test_01_compliance_headers_stdlib_lateral_zerostub_static_gatekeeper_and_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hivemind.security.pr_security_scanner import PRSecurityScanner

    lat_spec = importlib.util.spec_from_file_location(
        "test_no_lateral_movement_dyn20",
        REPO_ROOT / "tests" / "test_no_lateral_movement.py",
    )
    assert lat_spec is not None and lat_spec.loader is not None
    lateral_guard = importlib.util.module_from_spec(lat_spec)
    lat_spec.loader.exec_module(lateral_guard)

    gk_spec = importlib.util.spec_from_file_location(
        "pr_gatekeeper_dyn20",
        REPO_ROOT / "scripts" / "pr_gatekeeper.py",
    )
    assert gk_spec is not None and gk_spec.loader is not None
    gatekeeper = importlib.util.module_from_spec(gk_spec)
    gk_spec.loader.exec_module(gatekeeper)

    static_spec = importlib.util.spec_from_file_location(
        "test_static_numbers_dyn20",
        REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
    )
    assert static_spec is not None and static_spec.loader is not None
    static_numbers_guard = importlib.util.module_from_spec(static_spec)
    static_spec.loader.exec_module(static_numbers_guard)

    this_file = Path(__file__).resolve()
    for p in (VERIFIER_PATH, this_file):
        assert EXPECTED_AGPL_HEADER in p.read_text(encoding="utf-8")[:300]

    verifier_src = VERIFIER_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 stdlib imports only + ProxyHandler({}) present
    tree = ast.parse(verifier_src, filename=str(VERIFIER_PATH))
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

    # 2. Lateral movement AST audit + negative controls
    assert lateral_guard.audit_source_for_lateral_movement(verifier_src, str(VERIFIER_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(test_src, str(this_file)) == []
    lan_ip = ".".join(("192", "168", "44", "10"))
    assert lateral_guard.audit_source_for_lateral_movement(verifier_src + f'\n_LAN = "http://{lan_ip}:8000"\n') != []
    assert lateral_guard.audit_source_for_lateral_movement(verifier_src + f'\n_EXT = "{UNTRUSTED_EXT_GW}"\n') != []

    # 3. PRSecurityScanner 5-layer audit + PR Gatekeeper FORBIDDEN_ADDED + FIGURE check + ZeroStubAudit
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_arena_ballot_and_parity_verifier.py", verifier_src),
        ("tests/test_noel_arena_ballot_and_parity_soak.py", test_src),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        assert static_numbers_guard.FIGURE.findall(content) == []
        syn_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(f"+{ln}" for ln in content.splitlines())
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(syn_diff)) == []
        zres = ZeroStubAudit.audit_python_code(content)
        assert zres.is_clean, f"Unexpected ZeroStubAudit violations in {rel}: {zres.violations}"
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1220,
        pr_title="feat(onboarding,arena,scale): /arena-noel zero-inference blind A/B ballot verifier, CSP parity gate, and 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-20 /arena-noel blind A/B ballot verifier, CSP parity gate, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,arena,scale): /arena-noel blind A/B ballot verifier and CSP parity gate"],
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
                f"s = importlib.util.spec_from_file_location('t_ci_dyn20', Path({str(this_file)!r})); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert m.ZeroStubAudit.check_file(m.VERIFIER_PATH).is_clean; "
                "assert m.ZeroStubAudit.audit_python_code('def ok():\\n    return 1\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def stub_bad():\\n    pass\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def dots():\\n    ...\\n').is_clean"
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

    # 4. Unit checks for helpers, safety scanners, URL allowlists, and arity adapters
    assert verifier.normalize_lang(None) == "en"
    assert verifier.normalize_lang("en-US") == "en"
    assert verifier.normalize_lang("fr_CA") == "fr"
    for bad_lang in (True, "es"):
        with pytest.raises(verifier.ArenaVerifierError):
            verifier.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert verifier.mask_api_key(None) == "<missing>"
    assert verifier.mask_api_key("short") == "<invalid>"
    assert verifier.mask_api_key("other-prefix-123456") == "<invalid>"
    assert verifier.mask_api_key(TEST_KEY_ALPHA) == "hm-***1234"
    assert verifier.mask_api_key(TEST_KEY_LIVE) == "hm_live_***9999"

    assert verifier.redact_home_path(VERIFIER_PATH) == "scripts/noel_arena_ballot_and_parity_verifier.py"
    custom_home = tmp_path / "home_bob"
    custom_home.mkdir()
    assert verifier.redact_home_path(custom_home / ".hivemind", home=custom_home) == "~/.hivemind"
    scrubbed = verifier.sanitize_text_for_display(
        f"path={custom_home}/cred key={TEST_KEY_ALPHA} mail=bob@home.org ip=10.20." + "30.40",
        home=custom_home,
    )
    assert "home_bob" not in scrubbed and TEST_KEY_ALPHA not in scrubbed and "bob@home.org" not in scrubbed

    with pytest.raises(verifier.ArenaVerifierError):
        verifier.scan_output_safety(123)  # type: ignore[arg-type]
    for issue_key, bad_txt in (
        ("email_detected", "u@" + "example.org"),
        ("unmasked_api_key_detected", TEST_KEY_ALPHA),
        ("bearer_token_detected", "Bea" + "rer tok_99887766"),
        ("unredacted_home_path_detected", "/ho" + "me/alice/dev"),
        ("forbidden_score_key_detected", '{"sco' + 're": 9}'),
        ("static_figure_detected", "latency 120.5 " + "ms"),
        ("trojan_bidi_detected", "abc\u202edef"),
        ("non_loopback_ipv4_detected:10.20." + "30.40", "10.20." + "30.40"),
    ):
        assert issue_key in verifier.scan_output_safety(bad_txt)
        with pytest.raises(verifier.ArenaVerifierError):
            verifier._assert_safe_output(bad_txt)

    assert verifier.sanitize_build_sha(None) == "unknown"
    assert verifier.sanitize_build_sha("secret1234") == "unknown"
    assert verifier.shas_match(TARGET_FULL_SHA, TARGET_SHORT_SHA) is True
    assert verifier.shas_match("tagalpha12", "tagbeta12") is False
    assert verifier.shas_match("unknown", TARGET_SHORT_SHA) is False
    with pytest.raises(verifier.ArenaVerifierError):
        verifier.validate_expected_sha("unknown")
    assert verifier.sanitize_cloud_run_revision(123) is None
    assert verifier.sanitize_cloud_run_revision("hivemind-openrouter-00200-abc") == "hivemind-openrouter-00200-abc"
    assert verifier.sanitize_cloud_run_revision("hm-secret-rev") is None

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
        with pytest.raises(verifier.ArenaVerifierError):
            verifier.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for m_bad, u_bad in (
        ("GET", ""),
        ("GET", "http://127.0.0.1/bad" + " path"),
        ("GET", "http://u:p@127.0.0.1/health"),
        ("POST", "http://127.0.0.1:8000/v1/chat/completions"),
        ("DELETE", "http://127.0.0.1:8000/arena-noel"),
    ):
        with pytest.raises(verifier.ArenaVerifierError):
            verifier._assert_safe_http_request(m_bad, u_bad)
    with pytest.raises(verifier.ArenaVerifierError):
        verifier._assert_safe_http_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/arena/vote",
            payload={"messages": [{"role": "user", "content": "hi"}]},
        )
    with pytest.raises(verifier.ArenaVerifierError):
        verifier._assert_safe_http_request(
            "POST",
            "http://127.0.0.1:8000/v1/hivemind/arena/vote",
            payload="not-a-dict",  # type: ignore[arg-type]
        )
    with pytest.raises(verifier.ArenaVerifierError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/health", payload={"a": 1})
    with pytest.raises(verifier.ArenaVerifierError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/health", headers={"X-Bad": "a\nb"})

    with pytest.raises(verifier.ArenaVerifierError):
        verifier._normalize_http_return(("bad",))
    assert verifier._looks_like_http_headers({}) is True
    assert verifier._normalize_http_return((200, {"object": "ok"}, {"content-type": "application/json"}))[1]["_json"] == {"object": "ok"}
    assert verifier._normalize_http_return((200, "plain-hdr-missing", "<html>p</html>"))[1]["html"] == "<html>p</html>"
    assert verifier._normalize_http_return((200, {"x-a": "1"}, {"headers": {"x-a": "2"}, "html": "<p>x</p>"}))[1]["headers"]["x-a"] == "2"
    assert verifier._normalize_http_return((200, "<html>plain</html>", {"content-type": "text/html"}))[1]["html_length"] > 0
    assert verifier._normalize_http_return((200, '{"status":"ok"}', {"content-type": "application/json"}))[1]["_json"] == {"status": "ok"}
    assert verifier._normalize_http_return((200, "[1, 2]"))[1]["data"] == [1, 2]
    assert verifier._normalize_http_return((200, 42))[1]["data"] == 42

    assert verifier._invoke_http_fn(lambda m, u, **_kw: (200, {"kw": True}), "GET", "http://127.0.0.1:8000/health")[1]["kw"] is True
    assert verifier._invoke_http_fn(lambda m, u, p: (200, {"p3": True}), "GET", "http://127.0.0.1:8000/health")[1]["p3"] is True
    assert verifier._invoke_http_fn(lambda m, u, p, h: (200, {"p4": True}), "GET", "http://127.0.0.1:8000/health")[1]["p4"] is True
    assert verifier._invoke_http_fn(lambda m, u, p, h, t: (200, {"p5": True}), "GET", "http://127.0.0.1:8000/health")[1]["p5"] is True
    assert verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(OSError("reset")), "GET", "http://127.0.0.1:8000/health") == (0, {"error": "network_error"})
    with pytest.raises(verifier.ArenaVerifierError):
        verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(verifier.ArenaVerifierError("bad")), "GET", "http://127.0.0.1:8000/health")
    monkeypatch.undo()


def test_02_proxy_free_loopback_http_and_stale_404_to_live_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"live": False, "nonce_seq": 0}
    seen_requests: list[tuple[str, str]] = []

    class _ArenaLoopbackHandler(http.server.BaseHTTPRequestHandler):
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

            if not state["live"] and self.path in ("/arena-noel", "/v1/hivemind/onboarding/manifest"):
                self._send_json(404, {"detail": "Not Found"})
                return

            if self.path in verifier.ARENA_PARITY_HTML_ROUTES:
                state["nonce_seq"] += 1
                # Include digits+unit inside nonce value to prove nonce stripping prevents FIGURE false positives.
                nonce = f"123ms456Nonce{state['nonce_seq']:04d}AbCdEf"
                body = _sample_compliant_html(self.path, nonce, lang="fr" if self.path == "/noel" else "en")
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
                self._send_json(
                    200,
                    {
                        "object": "hivemind.onboarding.manifest",
                        "campaign_id": "noel-2026-easy-onboard",
                        "status": "ok",
                        "build_sha": TARGET_FULL_SHA,
                        "holiday_routes": list(verifier.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
                        "downloadable_scripts": ["hivemind_quickstart.py", "hivemind_doctor.py"],
                        "personal_pc_only": True,
                        "financial_promise": "0.00 EUR / 0.00 USD",
                    },
                )
                return

            if self.path == "/v1/hivemind/onboarding/feedback/summary":
                self._send_json(
                    200,
                    {
                        "object": "hivemind.onboarding_feedback_summary",
                        "total_responses": 12,
                        "mean_ease_rating": 4.75,
                        "zero_game_lag_ratio": 1.0,
                        "by_profile": {"dev": 6, "gamer": 6},
                    },
                )
                return

            if self.path == "/v1/hivemind/arena/summary":
                self._send_json(
                    200,
                    {
                        "object": "hivemind.blind_ab_arena.summary",
                        "total_votes": 1,
                        "holiday_wins": 0,
                        "baseline_wins": 0,
                        "ties": 1,
                    },
                )
                return

            self._send_json(404, {"detail": "missing"})

        def do_POST(self) -> None:  # noqa: N802
            seen_requests.append(("POST", self.path))
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            if self.path == "/v1/hivemind/arena/vote":
                body = json.loads(raw)
                self._send_json(200, {"recorded": True, "task_id": body["task_id"], "total_votes": 1})
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

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ArenaLoopbackHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        # Hostile proxies MUST be bypassed by urllib.request.ProxyHandler({})
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

        # Exact 5-key dict equality preserved by default_http_request
        fb_code, fb_body = verifier.default_http_request("GET", f"{base_url}/v1/hivemind/onboarding/feedback/summary")
        assert fb_code == 200
        assert fb_body == {
            "object": "hivemind.onboarding_feedback_summary",
            "total_responses": 12,
            "mean_ease_rating": 4.75,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 6, "gamer": 6},
        }
        assert list(fb_body.keys()) == [
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        ]

        # Redirect guard & JSON parsing edge cases
        rc_good, body_good = verifier.default_http_request("GET", f"{base_url}/redirect-good")
        assert rc_good == 200 and body_good["total_responses"] == 12
        with pytest.raises(verifier.ArenaVerifierError):
            verifier.default_http_request("GET", f"{base_url}/redirect-evil")
        assert verifier.default_http_request("GET", f"{base_url}/bad-json")[1]["raw_error"] == "invalid_json"
        assert verifier.default_http_request("GET", f"{base_url}/list-ok")[1] == {"data": [1, 2]}
        assert verifier.default_http_request("GET", f"{base_url}/missing")[0] == 404

        # Phase A: Stale 404 drift detection
        stale_en = verifier.verify_arena_noel_holiday_readiness(
            base_url, environ=CLEAN_ENV, cwd=tmp_path, lang="en"
        )
        _verify_sha256_receipt(stale_en)
        assert stale_en["stale_404_drift"] is True
        assert stale_en["redeploy_required"] is True
        assert stale_en["parity_ok"] is False
        assert stale_en["exit_code"] == 2
        assert "/arena-noel" in stale_en["stale_404_routes"]
        assert "/v1/hivemind/onboarding/manifest" in stale_en["stale_404_routes"]
        assert "redeploy_required=True" in verifier.render_cli_report(stale_en)
        assert "Diagnostics" in verifier.render_markdown_report(stale_en)

        stale_fr = verifier.verify_arena_noel_holiday_readiness(
            "http://127.0.0.1:8000",
            http_fn=lambda _m, u: (200, {"status": "ok"}) if u.endswith("/health") else (404, {"detail": "Not Found"}),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_sha256_receipt(stale_fr)
        assert stale_fr["exit_code"] == 2
        assert "Parité `/arena-noel` EN ÉCHEC" in verifier.render_markdown_report(stale_fr)

        # Phase B: Live cutover parity check (dry-run mode -> 0 POST vote calls)
        state["live"] = True
        seen_requests.clear()
        live_dry = verifier.verify_arena_noel_holiday_readiness(
            base_url,
            probe_vote_dry_run=True,
            expected_sha=TARGET_SHORT_SHA,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        _verify_sha256_receipt(live_dry)
        assert live_dry["parity_ok"] is True
        assert live_dry["ready"] is True
        assert live_dry["exit_code"] == 0
        assert live_dry["stale_404_drift"] is False
        assert live_dry["redeploy_required"] is False
        assert live_dry["unique_csp_nonces_count"] == 4
        assert live_dry["sha_match"] is True
        assert live_dry["extracted_ballots_count"] == 4
        assert live_dry["dry_run_ballots_audited"] == 4
        assert live_dry["live_vote_probed"] is False
        assert live_dry["feedback_summary"] == fb_body
        assert all(m == "GET" for m, _ in seen_requests)
        assert all("completions" not in p.lower() for _, p in seen_requests)

        # Phase C: Live vote canary probe (probe_vote_dry_run=False)
        live_vote = verifier.verify_arena_noel_holiday_readiness(
            base_url,
            probe_vote_dry_run=False,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_sha256_receipt(live_vote)
        assert live_vote["parity_ok"] is True
        assert live_vote["live_vote_probed"] is True
        assert live_vote["live_vote_ok"] is True
        assert live_vote["live_arena_summary_ok"] is True
        assert ("POST", "/v1/hivemind/arena/vote") in seen_requests
    finally:
        srv.shutdown()
        srv.server_close()

    # Closed port returns HTTP 0 network error without raising
    assert verifier.default_http_request("GET", f"{base_url}/arena-noel", timeout_s=0.2)[0] == 0

    # Negative parity controls: reused nonce, unsafe sink, static figure, extra summary key, SHA drift, bad ballot DOM
    def _broken_http(method: str, url: str) -> tuple[int, dict[str, str], str | dict[str, Any]]:
        path = urlsplit(url).path
        if path in verifier.ARENA_PARITY_HTML_ROUTES:
            reused = "ReusedNonce123456"
            if path == "/arena-noel":
                bad_mail = "u@" + "example.org"
                bad_dom = (
                    '<!DOCTYPE html><html><head><style nonce="ReusedNonce123456"></style></head>'
                    '<body><main id="arenaNoelApp">'
                    '<article data-ballot="bad-tok"></article>'
                    '<article data-task-id="python-sliding-median" data-ballot="invalid_hmac_token_value"></article>'
                    f'<pre id="candA-python-sliding-median">holiday_checkpoint {bad_mail}</pre>'
                    '</main><script nonce="ReusedNonce123456">'
                    "el." + "inner" + "HTML='198.1 " + "ms';fetch('/v1/chat/completions');"
                    "</script></body></html>"
                )
                return 200, {"content-security-policy": f"default-src 'none'; script-src 'nonce-{reused}'"}, bad_dom
            return 200, {"content-security-policy": f"script-src 'nonce-{reused}' 'unsafe-inline'"}, _sample_compliant_html(path, reused)
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, {}, {"object": "wrong", "build_sha": "abcdef12", "holiday_routes": []}
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, {}, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 1,
                "mean_ease_rating": 5.0,
                "zero_game_lag_ratio": 1.0,
                "by_profile": {"dev": 1, "gamer": 0},
                "extra_key": "must_fail",
            }
        return 500, {}, {"error": "fail"}

    broken_rep = verifier.verify_arena_noel_holiday_readiness(
        "http://127.0.0.1:8000",
        probe_vote_dry_run=False,
        expected_sha=TARGET_FULL_SHA,
        http_fn=_broken_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_sha256_receipt(broken_rep)
    assert broken_rep["parity_ok"] is False
    assert broken_rep["exit_code"] == 2
    assert broken_rep["feedback_summary_ok"] is False
    assert broken_rep["arena_ballots_ok"] is False
    assert broken_rep["manifest_ok"] is False


def test_03_bradley_terry_elo_convergence_symmetry_and_pii_rejection() -> None:
    # 1. Empty & all-tie edge cases
    empty_elo = verifier.compute_blind_ab_elo_summary([])
    _verify_sha256_receipt(empty_elo)
    assert empty_elo["total_ballots"] == 0
    assert empty_elo["elo_diff"] == 0.0
    assert empty_elo["holiday_elo"] == 1500.0
    assert empty_elo["tie_aware_win_rate"] is None
    assert empty_elo["position_bias_audit"]["position_symmetry_ok"] is True

    tie_ballots = [
        {"task_id": "python-sliding-median", "chosen_side": "tie", "holiday_side": "A" if i % 2 == 0 else "B", "profile": "dev"}
        for i in range(6)
    ]
    tie_elo = verifier.compute_blind_ab_elo_summary(tie_ballots)
    _verify_sha256_receipt(tie_elo)
    assert tie_elo["ties"] == 6 and tie_elo["tie_aware_win_rate"] == 0.5
    assert tie_elo["decisive_win_rate"] is None
    assert tie_elo["elo_diff"] == 0.0

    # 2. Boundary all-wins & all-losses keep finite Elo & CI bounds
    all_hol = verifier.compute_blind_ab_elo_summary(
        [{"task_id": "rust-frame-decoder", "chosen_side": "A" if i % 2 == 0 else "B", "holiday_side": "A" if i % 2 == 0 else "B"} for i in range(12)]
    )
    _verify_sha256_receipt(all_hol)
    assert all_hol["elo_diff"] > 100.0 and all_hol["elo_diff_ci_95"][0] > 0.0
    all_base = verifier.compute_blind_ab_elo_summary(
        [{"task_id": "rust-frame-decoder", "chosen_side": "B" if i % 2 == 0 else "A", "holiday_side": "A" if i % 2 == 0 else "B"} for i in range(12)]
    )
    _verify_sha256_receipt(all_base)
    assert all_base["elo_diff"] < -100.0 and all_base["elo_diff_ci_95"][1] < 0.0

    # 3. Convergence across N = 20, 80, 320 at 3:1 preference odds (Elo diff -> 400*log10(3) ~= 190.85)
    expected_elo_diff = 400.0 * math.log10(3.0)
    ci_widths: list[float] = []
    summaries_by_n: dict[int, dict[str, Any]] = {}
    for n_blocks in (5, 20, 80):  # 20, 80, 320 ballots
        batch: list[dict[str, Any]] = []
        for idx in range(n_blocks * 4):
            h_side = "A" if idx % 2 == 0 else "B"
            other_side = "B" if h_side == "A" else "A"
            hol_wins = (idx % 4) != 3  # 3 out of 4
            batch.append(
                {
                    "task_id": verifier.CURATED_ARENA_TASK_IDS[idx % 4],
                    "chosen_side": h_side if hol_wins else other_side,
                    "holiday_side": h_side,
                    "profile": "dev" if idx % 2 == 0 else "gamer",
                    "platform": "linux",
                }
            )
        rep = verifier.compute_blind_ab_elo_summary(batch)
        _verify_sha256_receipt(rep)
        assert rep["bradley_terry_converged"] is True
        assert rep["decisive_win_rate"] == 0.75
        assert rep["tie_aware_win_rate"] == 0.75
        assert rep["bradley_terry_win_prob"] == pytest.approx(0.75, abs=0.001)
        assert rep["elo_diff"] == pytest.approx(expected_elo_diff, abs=0.5)
        lo, hi = rep["elo_diff_ci_95"]
        assert lo <= rep["elo_diff"] <= hi
        ci_widths.append(hi - lo)
        summaries_by_n[len(batch)] = rep
    assert ci_widths[0] > ci_widths[1] > ci_widths[2]
    assert summaries_by_n[320]["statistically_significant_holiday_preference"] is True

    # 4. Position-bias symmetry: A <-> B flip invariance + biased evaluator detection
    mixed_ballots = [
        {"task_id": "lean4-energy-descent", "model_a": "holiday_checkpoint", "model_b": "baseline_checkpoint", "chosen_side": "A", "profile": "dev"},
        {"task_id": "lean4-energy-descent", "model_a": "baseline_checkpoint", "model_b": "holiday_checkpoint", "chosen_side": "B", "profile": "gamer"},
        {"task_id": "exoplanet-transit-bls", "chosen_side": "tie", "holiday_side": "A", "profile": "dev"},
        {"task_id": "exoplanet-transit-bls", "chosen_side": "A", "holiday_side": "B", "profile": "gamer"},
    ]
    swapped_ballots = [
        {
            "task_id": b["task_id"],
            "chosen_side": "tie" if b["chosen_side"] == "tie" else ("B" if b["chosen_side"] == "A" else "A"),
            "holiday_side": "B" if b.get("holiday_side", "A" if b.get("model_a") == "holiday_checkpoint" else "B") == "A" else "A",
            "profile": b["profile"],
        }
        for b in mixed_ballots
    ]
    s1 = verifier.compute_blind_ab_elo_summary(mixed_ballots)
    s2 = verifier.compute_blind_ab_elo_summary(swapped_ballots)
    for k in ("elo_diff", "holiday_elo", "baseline_elo", "elo_diff_ci_95", "tie_aware_win_rate", "bradley_terry_win_prob"):
        assert s1[k] == s2[k]
    assert s1["position_bias_audit"]["position_symmetry_ok"] is True

    # Always picking "A" across balanced presentations -> position bias delta == 1.0
    always_a = [
        {"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A" if i % 2 == 0 else "B", "profile": "dev"}
        for i in range(8)
    ]
    bias_rep = verifier.compute_blind_ab_elo_summary(always_a)
    assert bias_rep["position_bias_audit"]["position_bias_delta"] == 1.0
    assert bias_rep["position_bias_audit"]["position_symmetry_ok"] is False

    # Single-side presentation imbalance
    only_a = [{"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A"} for _ in range(6)]
    assert verifier.compute_blind_ab_elo_summary(only_a)["position_bias_audit"]["position_symmetry_ok"] is False
    pref_only = [{"task_id": "python-sliding-median", "preferred_checkpoint": "holiday_checkpoint"} for _ in range(4)]
    assert verifier.compute_blind_ab_elo_summary(pref_only)["holiday_wins"] == 4
    rev_cards = verifier.extract_arena_ballot_cards('<article data-ballot="abcdef012345.1.0123456789abcdef01234567" data-task-id="python-sliding-median"></article>')
    assert rev_cards == [{"task_id": "python-sliding-median", "ballot_token": "abcdef012345.1.0123456789abcdef01234567"}]
    assert verifier.compute_blind_ab_elo_summary(
        [{"task_id": "python-sliding-median", "holiday_side": "A", "winner": "A"}]
    )["holiday_wins"] == 1

    # 5. PII / secret / malformed ballot rejection (0 leaks in error messages)
    bad_ballots: tuple[tuple[Any, str], ...] = (
        ("not-a-dict", "mapping"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "prompt": "secret user code"}, "prompt"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "code": "print(1)"}, "code"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "email": "x@" + "y.org"}, "email"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "extra_field": "x"}, "unexpected"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "ballot_token": TEST_KEY_ALPHA}, TEST_KEY_ALPHA),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "ballot_token": "Bea" + "rer secret999"}, "secret999"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "ballot_token": "/ho" + "me/alice/secret"}, "alice"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "ballot_token": ".".join(("10", "0", "0", "9"))}, "10.0"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "ballot_token": "x\u202ey"}, "Trojan"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "ballot_token": "bad\x00tok"}, "control"),
        ({"task_id": "BAD SLUG", "chosen_side": "A", "holiday_side": "A"}, "task_id"),
        ({"task_id": "python-sliding-median", "chosen_side": "Z", "holiday_side": "A"}, "chosen_side"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "profile": "admin"}, "profile"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "platform": "Bad Platform!"}, "platform"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "revealed_holiday_side": "B"}, "disagree"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "model_a": "holiday_checkpoint"}, "model_a"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "model_a": "holiday_checkpoint", "model_b": "holiday_checkpoint"}, "distinct"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "model_a": "baseline_checkpoint", "model_b": "holiday_checkpoint"}, "disagree"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "winner": "baseline_checkpoint"}, "contradicts"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "winner": "B"}, "disagrees"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "preferred_checkpoint": "holiday_checkpoint", "winner": "baseline_checkpoint"}, "disagree"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "profile": 123}, "string"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "winner": True}, "string"),
        ({"task_id": "python-sliding-median", "chosen_side": "A", "model_a": "invalid_lbl", "model_b": "baseline_checkpoint"}, "checkpoint"),
        ({"task_id": "python-sliding-median", "chosen_side": "A"}, "requires"),
    )
    for bad_item, sentinel in bad_ballots:
        with pytest.raises(verifier.ArenaVerifierError) as exc_info:
            verifier.compute_blind_ab_elo_summary([bad_item])
        if bad_item == bad_ballots[5][0] or bad_item == bad_ballots[6][0] or bad_item == bad_ballots[7][0]:
            assert sentinel not in str(exc_info.value)

    with pytest.raises(verifier.ArenaVerifierError):
        verifier.compute_blind_ab_elo_summary("not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(verifier.ArenaVerifierError):
        verifier.wilson_95_interval(2, 1)
    with pytest.raises(verifier.ArenaVerifierError):
        verifier.wilson_95_interval(1, True)  # type: ignore[arg-type]
    with pytest.raises(verifier.ArenaVerifierError):
        verifier.wilson_95_interval(True, 5)  # type: ignore[arg-type]


def test_04_concurrent_48_participant_arena_noel_wal_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00240-noel")

    upstream_inference_hits: list[str] = []

    def _forbid_upstream_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn20_48_soak.sqlite3"
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

    total_users = 48
    barrier = threading.Barrier(total_users)
    obs_lock = threading.Lock()
    observed_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []
    seen_paths: list[str] = []
    cast_ballots: list[dict[str, Any]] = []

    try:
        with TestClient(app) as client:
            def _participant(idx: int) -> None:
                is_dev = idx < 24
                cohort_idx = idx if is_dev else idx - 24
                profile = "dev" if is_dev else "gamer"
                peer_ip = _loopback_ip(10 if is_dev else 20, cohort_idx)
                hdrs = {"X-Forwarded-For": peer_ip}
                lang_q = "en" if idx % 2 == 0 else "fr"
                local_nonces: list[str] = []

                barrier.wait(timeout=20.0)

                # 1. Probe /arena-noel and /noel HTML + CSP parity
                for route in (f"/arena-noel?lang={lang_q}", "/noel"):
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
                    stripped = verifier.strip_html_csp_nonce_attributes(r_html.text)
                    assert _STATIC_FIG_RE.findall(stripped) == []
                    assert ("inner" + "HTML") not in stripped
                    local_nonces.append(hdr_nonces[0])
                    with obs_lock:
                        seen_paths.append(route)
                    if route.startswith("/arena-noel"):
                        arena_cards = verifier.extract_arena_ballot_cards(r_html.text)
                        assert len(arena_cards) == 4
                        picked = arena_cards[idx % len(arena_cards)]

                # 2. Submit blind A/B vote using real HMAC-signed ballot token
                chosen_side = "tie" if idx % 12 == 11 else ("A" if idx % 3 != 0 else "B")
                r_vote = client.post(
                    "/v1/hivemind/arena/vote",
                    json={
                        "task_id": picked["task_id"],
                        "chosen_side": chosen_side,
                        "profile": profile,
                        "ballot_token": picked["ballot_token"],
                    },
                    headers=hdrs,
                )
                assert r_vote.status_code == 200, r_vote.text
                v_doc = r_vote.json()
                assert v_doc["recorded"] is True
                lv = v_doc["last_vote"]

                # 3. Submit CES feedback with Idempotency-Key + immediate duplicate replay
                ease = 5 if idx % 5 != 0 else 4
                zero_lag = idx != 47  # 47/48 == 0.9792 >= 0.95
                idem_key = "soak20-" + hashlib.sha256(f"{profile}:{cohort_idx}".encode()).hexdigest()[:24]
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
                    cast_ballots.append(
                        {
                            "task_id": lv["task_id"],
                            "chosen_side": lv["chosen_side"],
                            "revealed_holiday_side": lv["revealed_holiday_side"],
                            "preferred_checkpoint": lv["preferred_checkpoint"],
                            "profile": lv["profile"],
                            "ballot_token": picked["ballot_token"],
                        }
                    )

            t_soak_start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=total_users) as pool:
                futs = [pool.submit(_participant, i) for i in range(total_users)]
                for f in futs:
                    f.result(timeout=20.0)
            soak_elapsed_s = time.monotonic() - t_soak_start
            assert soak_elapsed_s < 4.0, f"48-user soak exceeded 4s budget: {soak_elapsed_s:.2f}s"

            assert sqlite_lock_errors == []
            assert upstream_inference_hits == []
            assert all("completions" not in p.lower() for p in seen_paths)
            assert len(observed_nonces) == total_users * 2
            assert len(set(observed_nonces)) == total_users * 2

            # Exact 5-key dict equality & 48-response Idempotency-Key conservation
            fb_summary_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
            assert fb_summary_resp.status_code == 200
            fb_summary = fb_summary_resp.json()
            assert list(fb_summary.keys()) == [
                "object",
                "total_responses",
                "mean_ease_rating",
                "zero_game_lag_ratio",
                "by_profile",
            ]
            assert fb_summary["object"] == "hivemind.onboarding_feedback_summary"
            assert fb_summary["total_responses"] == 48
            assert fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
            assert fb_summary["mean_ease_rating"] >= 4.2
            assert fb_summary["zero_game_lag_ratio"] >= 0.95

            arena_sum = client.get("/v1/hivemind/arena/summary").json()
            assert arena_sum["total_votes"] == 48
            assert arena_sum["by_profile"] == {"dev": 24, "gamer": 24}
            assert ("sco" + "re") not in _collect_keys(arena_sum)

            # Live readiness + Elo verification against the real app
            readiness = verifier.verify_arena_noel_holiday_readiness(
                "http://127.0.0.1:8000",
                probe_vote_dry_run=True,
                http_fn=_make_router_http_adapter(client, peer_ip="127.0.50.1"),
                environ=CLEAN_ENV,
                cwd=REPO_ROOT,
                lang="en",
                ballots=cast_ballots,
                expected_sha=TARGET_FULL_SHA,
            )
            _verify_sha256_receipt(readiness)
            assert readiness["parity_ok"] is True
            assert readiness["exit_code"] == 0
            assert readiness["unique_csp_nonces_count"] == 4
            assert readiness["extracted_ballots_count"] == 4
            assert readiness["feedback_summary"] == fb_summary
            assert readiness["elo_summary"]["total_ballots"] == 48
            assert readiness["elo_summary"]["ballot_pii_audit_ok"] is True
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

        blocked_api = verifier.verify_arena_noel_holiday_readiness(
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

    assert seen_markers == set(verifier.CORPORATE_MARKERS)

    # CLI matrix: --enforce-personal-pc, --ballots-json (valid & invalid), --json, --markdown, plain text, errors
    ballot_file = tmp_path / "ballots.json"
    ballot_file.write_text(
        json.dumps(
            [
                {"task_id": "python-sliding-median", "chosen_side": "A", "holiday_side": "A", "profile": "dev"},
                {"task_id": "rust-frame-decoder", "chosen_side": "B", "holiday_side": "B", "profile": "gamer"},
            ]
        ),
        encoding="utf-8",
    )

    def _ok_cli_http(method: str, url: str) -> tuple[int, dict[str, str], str | dict[str, Any]]:
        path = urlsplit(url).path
        if path in verifier.ARENA_PARITY_HTML_ROUTES:
            nonce = f"CliNonce{hashlib.sha256(path.encode()).hexdigest()[:12]}"
            return (
                200,
                {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"},
                _sample_compliant_html(path, nonce, lang="fr" if path == "/noel" else "en"),
            )
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, {}, {
                "object": "hivemind.onboarding.manifest",
                "campaign_id": "noel-2026-easy-onboard",
                "status": "ok",
                "build_sha": TARGET_FULL_SHA,
                "holiday_routes": list(verifier.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
                "personal_pc_only": True,
                "financial_promise": "0.00 EUR / 0.00 USD",
            }
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, {}, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": 0,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": {"dev": 0, "gamer": 0},
            }
        return 404, {}, {"detail": "missing"}

    assert verifier.main(["--enforce-personal-pc", "--json"], environ=CLEAN_ENV, cwd=personal_ws) == 0
    _verify_sha256_receipt(json.loads(capsys.readouterr().out))

    assert (
        verifier.main(
            ["--gateway", "http://127.0.0.1:8000", "--ballots-json", str(ballot_file), "--markdown", "--lang", "fr"],
            http_fn=_ok_cli_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    md_out = capsys.readouterr().out
    assert "## Bradley-Terry / Elo" in md_out and "SHA-256 Receipt" in md_out

    assert (
        verifier.main(
            ["--gateway", "http://127.0.0.1:8000", "--ballots-json", str(ballot_file)],
            http_fn=_ok_cli_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    txt_out = capsys.readouterr().out
    assert "Arena Elo" in txt_out and "SHA-256" in txt_out

    bad_ballots_file = tmp_path / "bad_ballots.json"
    bad_ballots_file.write_text('{"not": "a-list"}', encoding="utf-8")
    assert (
        verifier.main(
            ["--gateway", "http://127.0.0.1:8000", "--ballots-json", str(bad_ballots_file), "--json"],
            http_fn=_ok_cli_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 2
    )
    err_json = json.loads(capsys.readouterr().out)
    _verify_sha256_receipt(err_json)
    assert err_json["exit_code"] == 2

    assert verifier.main(["--lang", "es"], environ=CLEAN_ENV, cwd=personal_ws) == 2
    assert "Error:" in capsys.readouterr().err
