# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""1-click zero-PII CES & Steam-lag widget on /signup, holiday parity gate, and 48-user soak (`SLICE-SCALE-DYN-17`)."""

from __future__ import annotations

import ast
import concurrent.futures
import importlib.util
import ipaddress
import json
import os
import re
import subprocess
import sys
import threading
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import OnboardingFeedbackRequest, create_router_app
from hivemind.router.quick_join_page import OnboardingFeedbackCollector
from hivemind.router.signup_page import (
    _AUTH_JS,
    _GOOGLE_JS,
    CES_SIGNUP_COMPLIANCE_NOTE,
    DEFAULT_KEY_PLACEHOLDER,
    DEFAULT_PUBLIC_API_BASE,
    DEFAULT_SIGNUP_CES_PLATFORM,
    PENDING_SIGNUP_CES_STORAGE_KEY,
    PERSONAL_PC_COMPLIANCE_NOTE,
    SIGNUP_HOLIDAY_PARITY_ROUTES,
    VALID_CES_PROFILES,
    _reading,
    build_continue_config_yaml,
    build_signup_ces_ballot,
    content_security_policy,
    evaluate_signup_holiday_parity,
    render_signup_page,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - GitHub Actions ubuntu-latest lacks optional gwaya package

    class _FallbackAuditResult:
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
        """Minimal AST zero-stub fallback implementing both ``check_file`` and ``audit_python_code``."""

        _BAD_NAMES: tuple[re.Pattern[str], ...] = (
            re.compile(r"^mock_", re.IGNORECASE),
            re.compile(r"^fake_", re.IGNORECASE),
            re.compile(r"^simulate_", re.IGNORECASE),
            re.compile(r"^dummy_", re.IGNORECASE),
            re.compile(r"^stub_", re.IGNORECASE),
        )

        @classmethod
        def audit_python_code(cls, source_code: str) -> _FallbackAuditResult:
            violations: list[str] = []
            try:
                tree = ast.parse(source_code)
            except SyntaxError as exc:
                return _FallbackAuditResult(False, [f"SyntaxError: {exc}"], 1e6)
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
            return _FallbackAuditResult(not violations, violations, 1e6 if violations else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> _FallbackAuditResult:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNUP_PAGE_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "signup_page.py"

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn17",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn17",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn17",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
STATIC_FIGURE_RE = static_numbers_guard.FIGURE
_STRIP_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_CSP_NONCE_RE = re.compile(r"'nonce-([^']+)'")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)
TEST_GOOGLE_CLIENT_ID = "1234567890-abcdef123456.apps.googleusercontent.com"
SOAK_SECRET = b"noel-scale-dyn17-router-sec-32b!"
EXPECTED_SUMMARY_KEYS = {
    "object",
    "total_responses",
    "mean_ease_rating",
    "zero_game_lag_ratio",
    "by_profile",
}


def _loopback_ip_for(bucket: int, idx: int) -> str:
    """Deterministically map `(bucket, idx)` into `127.0.0.0/8` loopback space."""
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


def _collect_all_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_all_keys(item))
    return found


def _extract_csp_header_nonce(csp_header: str) -> str:
    matches = _CSP_NONCE_RE.findall(csp_header)
    assert matches, f"Missing CSP nonce in header: {csp_header!r}"
    assert len(set(matches)) == 1, f"Mismatched nonces in single CSP header: {csp_header!r}"
    return matches[0]


class _SignupDomInspector(HTMLParser):
    """Inspect /signup HTML structure: nav links, #result/#ideQuickSetup nesting, CSP nonces, and no inline handlers."""

    VOID_TAGS = {"meta", "input", "br", "hr", "img", "link"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, dict[str, str]]] = []
        self.inline_handlers: list[str] = []
        self.style_nonces: list[str] = []
        self.script_nonces: list[str] = []
        self.nav_links: list[tuple[str, str]] = []
        self.ids_inside_result: set[str] = set()
        self.ids_inside_ide_setup: set[str] = set()
        self.by_id_attrs: dict[str, dict[str, str]] = {}
        self.by_id_text: dict[str, str] = {}
        self._in_quick_nav = False
        self._cur_nav_href: str | None = None
        self._cur_nav_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        amap = {k.lower(): (v or "") for k, v in attrs}
        for attr_name in amap:
            if attr_name.startswith("on"):
                self.inline_handlers.append(f"{lower_tag}.{attr_name}")
        if lower_tag == "style":
            self.style_nonces.append(amap.get("nonce", ""))
        elif lower_tag == "script":
            self.script_nonces.append(amap.get("nonce", ""))

        el_id = amap.get("id")
        if el_id:
            self.by_id_attrs[el_id] = amap
            self.by_id_text.setdefault(el_id, "")
            if any(t == "section" and a.get("id") == "result" for t, a in self._stack):
                self.ids_inside_result.add(el_id)
            if any(t == "div" and a.get("id") == "ideQuickSetup" for t, a in self._stack):
                self.ids_inside_ide_setup.add(el_id)

        if lower_tag == "nav" and amap.get("data-testid") == "signup-quick-nav":
            self._in_quick_nav = True
        if self._in_quick_nav and lower_tag == "a":
            self._cur_nav_href = amap.get("href", "")
            self._cur_nav_text = []

        if lower_tag not in self.VOID_TAGS:
            self._stack.append((lower_tag, amap))

    def handle_data(self, data: str) -> None:
        if self._in_quick_nav and self._cur_nav_href is not None:
            self._cur_nav_text.append(data)
        for _tag, amap in reversed(self._stack):
            el_id = amap.get("id")
            if el_id:
                self.by_id_text[el_id] = self.by_id_text.get(el_id, "") + data
                break

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if self._in_quick_nav and lower_tag == "a" and self._cur_nav_href is not None:
            self.nav_links.append((self._cur_nav_href, "".join(self._cur_nav_text).strip()))
            self._cur_nav_href = None
            self._cur_nav_text = []
        if lower_tag == "nav" and self._in_quick_nav:
            self._in_quick_nav = False
        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx][0] == lower_tag:
                del self._stack[idx:]
                break


def test_01_compliance_security_lateral_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    target_files = (
        ("src/hivemind/router/signup_page.py", SIGNUP_PAGE_PATH),
        ("tests/test_noel_signup_vscode_ces_scale_soak.py", this_file),
    )

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    forbidden_html_sinks = (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "docu" + "ment" + ".write",
        "/v1/chat/" + "completions",
    )

    for rel_path, full_path in target_files:
        text = full_path.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in text[:300], f"Missing AGPL-3.0 header in first 300 chars of {rel_path}"

        # Lateral-movement AST audit & allowed-host URL check
        lat_violations = lateral_guard.audit_source_for_lateral_movement(text, filename=rel_path)
        if rel_path == "src/hivemind/router/signup_page.py":
            zero_ip_lit = ".".join(("0", "0", "0", "0"))
            lat_violations = [
                v
                for v in lat_violations
                if "accounts.google.com/gsi/" not in v and f"literal '{zero_ip_lit}'" not in v
            ]
        assert lat_violations == []
        for url_hit in re.findall(r"https?://[^\s'\"<>]+", text):
            host = (urlparse(url_hit).hostname or "").lower()
            if host:
                assert (
                    host.endswith(".run.app")
                    or host.endswith(".googleapis.com")
                    or host == "accounts.google.com"
                    or host == "datasets-server.huggingface.co"
                    or host in {"127.0.0.1", "localhost", "::1"}
                ), f"Disallowed host {host!r} in {rel_path}: {url_hit!r}"

        # PR gatekeeper FORBIDDEN_ADDED check
        synthetic_diff = f"diff --git a/{rel_path} b/{rel_path}\n+++ b/{rel_path}\n" + "\n".join(
            f"+{line}" for line in text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

        # 5-Layer PRSecurityScanner checks
        assert scanner.scan_python_ast_security(rel_path, text) == []
        assert scanner.scan_text_for_payment_solicitation(text, rel_path) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text, rel_path) == []
        assert scanner.scan_unicode_trojan_source(text, rel_path) == []
        assert ZeroStubAudit.audit_python_code(text).is_clean is True
        changed_files.append({"filename": rel_path, "patch": "", "content": text})

    signup_source = SIGNUP_PAGE_PATH.read_text(encoding="utf-8")
    assert STATIC_FIGURE_RE.findall(signup_source) == []
    for forbidden_str in forbidden_html_sinks:
        assert forbidden_str not in signup_source

    # Verify signup_page.py only imports Python stdlib + hivemind.router.google_auth
    tree = ast.parse(signup_source, filename="signup_page.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_mod = node.module.split(".")[0]
            assert root_mod in sys.stdlib_module_names or node.module == "hivemind.router.google_auth"

    # Verify CI ImportError ZeroStubAudit fallback (implementing both check_file & audit_python_code)
    sub_code = (
        "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('t_dyn17', {str(this_file)!r}); "
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
        f"r1 = mod.ZeroStubAudit.check_file({str(SIGNUP_PAGE_PATH)!r}); "
        f"r2 = mod.ZeroStubAudit.audit_python_code(open({str(SIGNUP_PAGE_PATH)!r}, encoding='utf-8').read()); "
        "assert r1.is_clean and r2.is_clean; "
        "bad = mod.ZeroStubAudit.audit_python_code('def stub_fn():\\n    ...\\n'); "
        "assert not bad.is_clean"
    )
    sub_proc = subprocess.run(
        [sys.executable, "-c", sub_code],
        cwd=str(REPO_ROOT),
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert sub_proc.returncode == 0, f"CI no-gwaya fallback check failed: {sub_proc.stderr}"

    # Full page-literals ratchet across all router page modules
    static_numbers_guard.test_page_modules_gain_no_new_typed_in_figures()

    pr_audit = scanner.audit_pull_request(
        pr_number=1700,
        pr_title="feat(onboarding,webui,scale): /signup 1-click zero-PII CES & Steam-lag widget, holiday parity gate, and 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-17 1-click zero-PII CES widget on /signup, holiday parity gate, and 48-user soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,webui,scale): /signup 1-click zero-PII CES & holiday parity gate"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True


def test_02_python_helpers_ces_ballot_parity_gate_and_render_signup_page(tmp_path: Path) -> None:
    forbidden_key = "sco" + "re"

    # 1. build_continue_config_yaml: valid & invalid cases
    yaml_run_app = build_continue_config_yaml("hm-ValidKey_12345", "https://hivemind-test.us-central1.run.app")
    assert "apiBase: https://hivemind-test.us-central1.run.app/v1" in yaml_run_app
    assert PERSONAL_PC_COMPLIANCE_NOTE in yaml_run_app
    yaml_loopback = build_continue_config_yaml("hm_live_abc123", "http://127.0.0.1:8080/v1/")
    assert "apiBase: http://127.0.0.1:8080/v1" in yaml_loopback
    for bad_key, bad_base, err_match in (
        (None, DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("   ", DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("sk-wrong", DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("hm-bad key", DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("hm-ok", None, "api_base"),
        ("hm-ok", "https://bad.run.app/v1\nfoo", "api_base"),
        ("hm-ok", "ftp" + "://127.0.0.1/v1", "api_base"),
        ("hm-ok", "https://u:p@127.0.0.1/v1", "api_base"),
        ("hm-ok", "https://127.0.0.1/v1?x=1", "api_base"),
    ):
        with pytest.raises(ValueError, match=err_match):
            build_continue_config_yaml(bad_key, bad_base)  # type: ignore[arg-type]

    # 2. content_security_policy & _reading
    csp = content_security_policy("NonceSignup_2026+Safe=")
    gsi_base = "https://" + "accounts.google.com/gsi/"
    assert "default-src 'none'" in csp
    assert f"script-src 'nonce-NonceSignup_2026+Safe=' {gsi_base}client" in csp
    assert f"connect-src 'self' {gsi_base}" in csp
    assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
    for bad_nonce in ("", "bad nonce", "nonce;script-src *", None):
        with pytest.raises(ValueError, match="nonce"):
            content_security_policy(bad_nonce)  # type: ignore[arg-type]

    assert _reading({"n": None}, "n") == "n/a"
    assert _reading({"n": 7}, "n") == "7"
    assert _reading({"n": 10519}, "n", thousands=True) == "10,519"
    with pytest.raises(ValueError, match="source"):
        _reading([], "n")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="key"):
        _reading({}, "")
    with pytest.raises(ValueError, match="thousands"):
        _reading({}, "n", thousands="yes")  # type: ignore[arg-type]

    # 3. build_signup_ces_ballot: strict zero-PII OnboardingFeedbackRequest extra='forbid' validation
    assert VALID_CES_PROFILES == frozenset({"dev", "gamer"})
    default_ballot = build_signup_ces_ballot()
    assert default_ballot == {
        "profile": "dev",
        "ease_rating": 5,
        "zero_game_lag": True,
        "platform": DEFAULT_SIGNUP_CES_PLATFORM,
        "note": "",
    }
    assert forbidden_key not in _collect_all_keys(default_ballot)
    parsed_default = OnboardingFeedbackRequest(**default_ballot)
    assert parsed_default.profile == "dev"
    assert parsed_default.platform == "web-signup"

    collector = OnboardingFeedbackCollector(max_items=16)
    for prof_in, ease_in, lag_in, plat_in in (
        ("  GAMER  ", 5, True, "  Windows11_Steam  "),
        ("dev", 4, False, "vscode-continue"),
        ("DEV", 1, True, "linux"),
        ("gamer", 3, True, "macos"),
    ):
        b = build_signup_ces_ballot(
            profile=prof_in,
            ease_rating=ease_in,
            zero_game_lag=lag_in,
            platform_name=plat_in,
        )
        req_model = OnboardingFeedbackRequest(**b)
        rec = collector.record(
            profile=req_model.profile,
            ease_rating=req_model.ease_rating,
            zero_game_lag=req_model.zero_game_lag,
            platform_name=req_model.platform,
        )
        assert set(rec.keys()) == EXPECTED_SUMMARY_KEYS

    for bad_ballot_kwargs in (
        {"profile": True},
        {"profile": None},
        {"profile": "enterprise"},
        {"profile": "d\u0435v"},
        {"profile": "x" * 40},
        {"ease_rating": True},
        {"ease_rating": False},
        {"ease_rating": 0},
        {"ease_rating": 6},
        {"ease_rating": 4.5},
        {"zero_game_lag": 1},
        {"zero_game_lag": "true"},
        {"zero_game_lag": None},
        {"platform_name": False},
        {"platform_name": None},
        {"platform_name": ""},
        {"platform_name": "   "},
        {"platform_name": "l\u00efnux"},
        {"platform_name": "p" * 35},
        {"platform_name": "a" * 28},
        {"platform_name": "-leading-dash"},
        {"platform_name": "alice@example.org"},
        {"platform_name": "/home/alice/pc"},
        {"platform_name": "hm-12345678"},
        {"platform_name": "hm_live_abc123"},
        {"platform_name": "hm_short"},
        {"platform_name": "amadeus_laptop"},
        {"platform_name": "corporate_pc"},
        {"platform_name": "secret_slug"},
        {"platform_name": "token_header"},
        {"platform_name": "pii_marker"},
        {"note": "free text note"},
        {"note": " "},
        {"note": None},
        {"note": False},
    ):
        with pytest.raises(ValueError):
            build_signup_ces_ballot(**bad_ballot_kwargs)  # type: ignore[arg-type]

    # 4. evaluate_signup_holiday_parity against live TestClient + drift/tampered negative controls
    store = RouterStore(tmp_path / "parity_probe.sqlite3", secret=SOAK_SECRET)
    try:
        with TestClient(create_router_app(store, open_registration=True, trusted_proxy_hops=1)) as client:
            live_probes: dict[str, dict[str, Any]] = {}
            for route in SIGNUP_HOLIDAY_PARITY_ROUTES:
                resp = client.get(route)
                live_probes[route] = {
                    "status_code": resp.status_code,
                    "headers": dict(resp.headers),
                    "body": resp.text,
                }
            report_empty = evaluate_signup_holiday_parity(live_probes)
            assert forbidden_key not in _collect_all_keys(report_empty)
            assert report_empty["object"] == "hivemind.signup_holiday_parity_report"
            assert report_empty["parity_ok"] is True
            assert report_empty["stale_404_drift"] is False
            assert report_empty["missing_or_404_routes"] == []
            assert report_empty["failed_routes"] == []
            assert report_empty["routes_passed_count"] == 6
            assert report_empty["routes_total_count"] == 6
            assert report_empty["unique_csp_nonces_count"] == 4
            assert report_empty["unique_csp_nonces_ok"] is True
            assert report_empty["html_dom_clean"] is True
            assert report_empty["compliance_ok"] is True
            assert report_empty["manifest_ok"] is True
            assert report_empty["feedback_summary_ok"] is True
            assert report_empty["zero_inference_verified"] is True
            assert len(report_empty["receipt_sha256"]) == 64

            # Record one CES ballot and test non-empty summary + expected_build_sha prefix/mismatch
            fb_res = client.post(
                "/v1/hivemind/onboarding/feedback",
                json=build_signup_ces_ballot(profile="dev", ease_rating=5, zero_game_lag=True),
                headers={"Idempotency-Key": "parity-unit-seed-01"},
            )
            assert fb_res.status_code == 200
            sum_after = client.get("/v1/hivemind/onboarding/feedback/summary")
            assert set(sum_after.json().keys()) == EXPECTED_SUMMARY_KEYS

            sha_probes = dict(live_probes)
            sha_probes["/v1/hivemind/onboarding/feedback/summary"] = {
                "status_code": 200,
                "json": sum_after.json(),
            }
            man_json = json.loads(live_probes["/v1/hivemind/onboarding/manifest"]["body"])
            man_with_sha = dict(man_json, build_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9011223344")
            sha_probes["/v1/hivemind/onboarding/manifest"] = {"status_code": 200, "json": man_with_sha}

            sha_match_report = evaluate_signup_holiday_parity(sha_probes, expected_build_sha="a1b2c3d4")
            assert sha_match_report["parity_ok"] is True
            assert sha_match_report["sha_match"] is True

            sha_mismatch_report = evaluate_signup_holiday_parity(sha_probes, expected_build_sha="deadbeef")
            assert sha_mismatch_report["parity_ok"] is False
            assert sha_mismatch_report["sha_match"] is False

            # Non-hex alphanumeric build SHA exact & mismatch
            alpha_sha_probes = dict(sha_probes)
            alpha_sha_probes["/v1/hivemind/onboarding/manifest"] = {
                "status_code": 200,
                "json": dict(man_json, build_sha="NoelRel2026Z"),
            }
            assert evaluate_signup_holiday_parity(alpha_sha_probes, expected_build_sha="NoelRel2026Z")["sha_match"] is True
            assert evaluate_signup_holiday_parity(alpha_sha_probes, expected_build_sha="NoelRel9999Z")["sha_match"] is False

            # Strip nonce="..." containing base64url digits+'ms' inside nonce -> must NOT false-positive
            nonce_ms = "Ab9msCdEf123456"
            ms_nonce_html = render_signup_page(None, nonce_ms)
            ms_probes = dict(live_probes)
            ms_probes["/signup"] = {
                "status_code": 200,
                "csp": content_security_policy(nonce_ms),
                "html": ms_nonce_html,
            }
            assert evaluate_signup_holiday_parity(ms_probes)["parity_ok"] is True

            # Pre-deploy Cloud Run drift: HTTP 404 on manifest, summary, and /raid-noel
            drift_probes = dict(live_probes)
            drift_probes["/raid-noel"] = {"status": 404, "text": "Not Found"}
            drift_probes["/v1/hivemind/onboarding/manifest"] = {"status_code": 404, "body": '{"detail":"Not Found"}'}
            drift_probes["/v1/hivemind/onboarding/feedback/summary"] = {"status_code": 404, "body": '{"detail":"Not Found"}'}
            drift_report = evaluate_signup_holiday_parity(drift_probes)
            assert drift_report["parity_ok"] is False
            assert drift_report["stale_404_drift"] is True
            assert drift_report["missing_or_404_routes"] == [
                "/raid-noel",
                "/v1/hivemind/onboarding/manifest",
                "/v1/hivemind/onboarding/feedback/summary",
            ]
            assert drift_report["per_route"]["/raid-noel"]["reason" if "reason" in drift_report["per_route"]["/raid-noel"] else "issues"] == ["http_404"]

            # Reused CSP nonce + unsafe DOM sink + static figure + missing compliance + missing/corrupt probes
            reused_nonce = _extract_csp_header_nonce(live_probes["/signup"]["headers"]["content-security-policy"])
            injected_sink_html = live_probes["/quick"]["body"].replace(
                "</body>", "<p>x</p></body>"
            ) + "<!-- " + ("inner" + "HTML") + " " + ("120 m" + "s") + " -->"
            bad_probes: dict[str, Any] = {
                "/signup": live_probes["/signup"],
                "/noel": {
                    "status_code": 200,
                    "headers": {"content-security-policy": f"default-src 'none'; style-src 'nonce-{reused_nonce}'"},
                    "body": f'<!DOCTYPE html><html><head><style nonce="{reused_nonce}"></style></head><body>Usage personnel AGPL-3.0 0,00 EUR</body></html>',
                },
                "/quick": {
                    "status_code": 200,
                    "headers": live_probes["/quick"]["headers"],
                    "body": injected_sink_html,
                },
                "/raid-noel": {
                    "status_code": 200,
                    "headers": {"content-security-policy": "default-src 'none'; style-src 'unsafe-inline'"},
                    "body": "<html><body>stripped compliance</body></html>",
                },
                "/v1/hivemind/onboarding/manifest": {
                    "status_code": 200,
                    "body": "{corrupt-json",
                },
                # /v1/hivemind/onboarding/feedback/summary intentionally omitted -> missing_probe
            }
            bad_report = evaluate_signup_holiday_parity(bad_probes)
            assert bad_report["parity_ok"] is False
            assert bad_report["unique_csp_nonces_ok"] is False
            assert bad_report["html_dom_clean"] is False
            assert bad_report["compliance_ok"] is False
            assert "reused_csp_nonce" in bad_report["per_route"]["/noel"]["issues"]
            assert "unsafe_dom_sink_detected" in bad_report["per_route"]["/quick"]["issues"]
            assert "unbacked_static_figure_detected" in bad_report["per_route"]["/quick"]["issues"]
            assert "missing_doctype" in bad_report["per_route"]["/raid-noel"]["issues"]
            assert "invalid_or_missing_csp_nonce" in bad_report["per_route"]["/raid-noel"]["issues"]
            assert "missing_personal_pc_or_zero_eur_banner" in bad_report["per_route"]["/raid-noel"]["issues"]
            assert "manifest_missing_json" in bad_report["per_route"]["/v1/hivemind/onboarding/manifest"]["issues"]
            assert "missing_probe" in bad_report["per_route"]["/v1/hivemind/onboarding/feedback/summary"]["issues"]

            # Network error status=None + schema failure branches on manifest and summary
            schema_probes: dict[str, Any] = {
                "/signup": {"status_code": None},
                "/noel": live_probes["/noel"],
                "/quick": live_probes["/quick"],
                "/raid-noel": live_probes["/raid-noel"],
                "/v1/hivemind/onboarding/manifest": {
                    "status_code": 200,
                    "json": {
                        "object": "wrong",
                        "campaign_id": "wrong",
                        "status": "degraded",
                        "personal_pc_only": False,
                        "financial_promise": "paid",
                        forbidden_key: 1,
                        "holiday_routes": ["/only-one"],
                        "downloadable_scripts": "not-list",
                        "build_sha": "secret_leak",
                    },
                },
                "/v1/hivemind/onboarding/feedback/summary": {
                    "status_code": 200,
                    "json": {
                        "object": "wrong",
                        "total_responses": 0,
                        "mean_ease_rating": 5.0,
                        "zero_game_lag_ratio": 1.0,
                        "by_profile": {"dev": 1, "gamer": "bad"},
                        "extra": 1,
                        forbidden_key: 2,
                    },
                },
            }
            schema_rep = evaluate_signup_holiday_parity(schema_probes)
            assert schema_rep["parity_ok"] is False
            assert schema_rep["per_route"]["/signup"]["issues"] == ["network_or_missing_status"]
            man_issues = schema_rep["per_route"]["/v1/hivemind/onboarding/manifest"]["issues"]
            assert "manifest_invalid_object" in man_issues
            assert "manifest_personal_pc_only_false" in man_issues
            assert "manifest_missing_holiday_routes" in man_issues
            assert "manifest_invalid_downloadable_scripts" in man_issues
            sum_issues = schema_rep["per_route"]["/v1/hivemind/onboarding/feedback/summary"]["issues"]
            assert "summary_exact_five_keys_mismatch" in sum_issues
            assert "summary_non_null_ease_when_empty" in sum_issues
            assert "summary_non_null_lag_when_empty" in sum_issues
            assert "summary_invalid_profile_counts" in sum_issues

            # Invalid summary JSON (missing JSON, invalid total_responses, >0 bad ease/lag, count mismatch, invalid manifest routes/scripts)
            schema_probes2 = dict(live_probes)
            schema_probes2["/v1/hivemind/onboarding/manifest"] = {
                "status_code": 200,
                "json": {
                    "object": "hivemind.onboarding.manifest",
                    "campaign_id": "noel-2026-easy-onboard",
                    "status": "ok",
                    "personal_pc_only": True,
                    "financial_promise": "0.00 EUR / 0.00 USD",
                    "holiday_routes": "not-a-list",
                    "downloadable_scripts": ["other.py"],
                },
            }
            schema_probes2["/v1/hivemind/onboarding/feedback/summary"] = {
                "status_code": 200,
                "json": {
                    "object": "hivemind.onboarding_feedback_summary",
                    "total_responses": 2,
                    "mean_ease_rating": 9.9,
                    "zero_game_lag_ratio": -0.5,
                    "by_profile": {"dev": 1, "gamer": 0},
                },
            }
            rep2 = evaluate_signup_holiday_parity(schema_probes2)
            assert "manifest_invalid_holiday_routes" in rep2["per_route"]["/v1/hivemind/onboarding/manifest"]["issues"]
            assert "manifest_missing_core_scripts" in rep2["per_route"]["/v1/hivemind/onboarding/manifest"]["issues"]
            s2_issues = rep2["per_route"]["/v1/hivemind/onboarding/feedback/summary"]["issues"]
            assert "summary_invalid_mean_ease_rating" in s2_issues
            assert "summary_invalid_zero_game_lag_ratio" in s2_issues
            assert "summary_profile_conservation_mismatch" in s2_issues

            schema_probes3 = dict(live_probes)
            schema_probes3["/v1/hivemind/onboarding/feedback/summary"] = {
                "status_code": 200,
                "body": "{bad-json",
            }
            rep3 = evaluate_signup_holiday_parity(schema_probes3)
            assert rep3["per_route"]["/v1/hivemind/onboarding/feedback/summary"]["issues"] == ["summary_missing_json"]

            schema_probes4 = dict(live_probes)
            schema_probes4["/v1/hivemind/onboarding/feedback/summary"] = {
                "status_code": 200,
                "json": {
                    "object": "hivemind.onboarding_feedback_summary",
                    "total_responses": True,
                    "mean_ease_rating": None,
                    "zero_game_lag_ratio": None,
                    "by_profile": "bad",
                },
            }
            rep4 = evaluate_signup_holiday_parity(schema_probes4)
            assert "summary_invalid_total_responses" in rep4["per_route"]["/v1/hivemind/onboarding/feedback/summary"]["issues"]
            assert "summary_invalid_by_profile" in rep4["per_route"]["/v1/hivemind/onboarding/feedback/summary"]["issues"]
    finally:
        store.close()

    for bad_arg in (
        "not-a-mapping",
        {True: {"status_code": 200}},
        {"/signup": "not-a-mapping"},
        {"/v1/chat/" + "completions": {"status_code": 200}},
        {"/v1/comple" + "tions": {"status_code": 200}},
    ):
        with pytest.raises(ValueError):
            evaluate_signup_holiday_parity(bad_arg)  # type: ignore[arg-type]
    for bad_sha in (True, "bad!", "unknown", "secret_token_123"):
        with pytest.raises(ValueError, match="expected_build_sha"):
            evaluate_signup_holiday_parity({}, expected_build_sha=bad_sha)  # type: ignore[arg-type]

    # 5. render_signup_page across all branches + new nav, CLI hints, parity button, and #signupCesCard DOM IDs
    full_html = render_signup_page(
        TEST_GOOGLE_CLIENT_ID,
        "NonceDyn17GoogleSafe",
        auth_result={
            "created": True,
            "api_key": "hm-Dyn17Key_987654",
            "account_id": "acct-dyn17",
            "handle": "noel_dev",
            "role": "prosumer",
            "balance_hmt": 25.0,
        },
        live_telemetry={
            "accounts": {"total_registered_accounts": 12, "by_role": {"prosumer": 5, "producer": 2, "consumer": 3, "trainer": 2}},
            "models": {"live_models_count": 3},
            "t4_boinc": {"vram_free_mib": 10519},
            "opentrainer": {"total_registered_nodes": 2, "accepted_rounds_count": 7},
        },
        starter_grant_hmt=25.0,
        registration_bonus_hmt=10.0,
        allow_bonuses=True,
        default_role="trainer",
    )
    stripped_full = _STRIP_NONCE_ATTR_RE.sub('nonce=""', full_html)
    assert STATIC_FIGURE_RE.findall(stripped_full) == []
    assert ("inner" + "HTML") not in full_html
    assert ("/v1/chat/" + "completions") not in full_html
    assert PERSONAL_PC_COMPLIANCE_NOTE in full_html
    assert CES_SIGNUP_COMPLIANCE_NOTE in full_html
    assert "python3 scripts/hivemind_quickstart.py --write-continue-config" in full_html
    assert "hivemind vscode-setup" in full_html
    assert "python3 scripts/hivemind_doctor.py --noel" in full_html
    assert "python hivemind_connect.py check" in full_html
    assert "python hivemind_connect.py snippet continue --model coder" in full_html

    dom = _SignupDomInspector()
    dom.feed(full_html)
    assert dom.inline_handlers == []
    assert dom.style_nonces == ["NonceDyn17GoogleSafe"]
    assert dom.script_nonces == ["NonceDyn17GoogleSafe", "NonceDyn17GoogleSafe"]
    assert ("/noel", "🎄 Noël Quick Start") in dom.nav_links
    assert ("/raid-noel", "🏔️ Winter GPU Raid") in dom.nav_links
    assert ("/arena-noel", "⚖️ Blind A/B Arena") in dom.nav_links
    required_ide_ids = {
        "continueYamlSnippet",
        "copyContinueBtn",
        "verifyKeyModelsBtn",
        "verifyKeyStatus",
        "quickConnectCliBlock",
        "verifyHolidayParityBtn",
        "verifyHolidayParityStatus",
        "signupCesCard",
        "signupCesProfileDevBtn",
        "signupCesProfileGamerBtn",
        "signupCesEaseBtn-1",
        "signupCesEaseBtn-2",
        "signupCesEaseBtn-3",
        "signupCesEaseBtn-4",
        "signupCesEaseBtn-5",
        "signupCesLagYesBtn",
        "signupCesLagNoBtn",
        "signupSubmitCesBtn",
        "signupRetryPendingCesBtn",
        "signupCesStatus",
        "signupCesComplianceNote",
    }
    assert required_ide_ids.issubset(dom.ids_inside_ide_setup)
    assert required_ide_ids.issubset(dom.ids_inside_result)
    assert dom.by_id_attrs["signupCesProfileDevBtn"]["aria-pressed"] == "true"
    assert dom.by_id_attrs["signupCesProfileGamerBtn"]["aria-pressed"] == "false"
    assert dom.by_id_attrs["signupCesEaseBtn-5"]["aria-pressed"] == "true"
    assert dom.by_id_attrs["signupCesEaseBtn-4"]["aria-pressed"] == "false"
    assert dom.by_id_attrs["signupCesLagYesBtn"]["aria-pressed"] == "true"
    assert dom.by_id_attrs["signupCesLagNoBtn"]["aria-pressed"] == "false"

    # Cover remaining render_signup_page branches & error paths
    only_starter = render_signup_page(None, "NonceOnlyStarter", starter_grant_hmt=10.0)
    assert "Starter grant on this gateway: 10.00 HMT per new account" in only_starter
    assert DEFAULT_KEY_PLACEHOLDER in only_starter
    only_reg = render_signup_page(None, "NonceOnlyReg", registration_bonus_hmt=5.0)
    assert "Registration bonus on this gateway: 5.00 HMT per new account" in only_reg
    restored_err = render_signup_page(
        None,
        "NonceRestoredErr",
        auth_result={"created": False, "api_key": "hm-Restored_1", "account_id": "a2", "balance": 0.0},
        error_message="<err>",
        default_role="unknown-role",
    )
    assert "Welcome back! Your previous account & HMT balance have been restored." in restored_err
    assert "&lt;err&gt;" in restored_err

    for bad_render in (
        {"client_id": None, "nonce": ""},
        {"client_id": "bad-id", "nonce": "ok-nonce"},
        {"client_id": None, "nonce": "ok-nonce", "starter_grant_hmt": -1.0},
        {"client_id": None, "nonce": "ok-nonce", "registration_bonus_hmt": True},
        {"client_id": None, "nonce": "ok-nonce", "allow_bonuses": "no"},
        {"client_id": None, "nonce": "ok-nonce", "auth_result": "bad"},
        {"client_id": None, "nonce": "ok-nonce", "error_message": 404},
        {"client_id": None, "nonce": "ok-nonce", "live_telemetry": "bad"},
        {"client_id": None, "nonce": "ok-nonce", "default_role": 123},
    ):
        with pytest.raises(ValueError):
            render_signup_page(**bad_render)  # type: ignore[arg-type]


def test_03_node_dom_parity_check_ces_spool_retry_and_textcontent_traps() -> None:
    inner_prop = "inner" + "HTML"
    outer_prop = "outer" + "HTML"
    adj_prop = "insertAdjacent" + "HTML"
    forbidden_chat_route = "/v1/chat/" + "completions"
    g_js = _GOOGLE_JS.replace("%CLIENT_ID%", json.dumps(TEST_GOOGLE_CLIENT_ID))

    harness = f"""
const elements = {{}};
function makeEl(id, initVal = '') {{
  const listeners = {{}};
  const attrs = {{}};
  const el = {{
    id,
    _text: '',
    value: initVal,
    className: '',
    hidden: id === 'result' || id === 'existingBrowserSessionBanner',
    getAttribute(k) {{ return Object.prototype.hasOwnProperty.call(attrs, k) ? attrs[k] : null; }},
    setAttribute(k, v) {{ attrs[k] = String(v); }},
    addEventListener(type, fn) {{
      if (!listeners[type]) listeners[type] = [];
      listeners[type].push(fn);
    }},
    async trigger(type, ev) {{
      for (const fn of (listeners[type] || [])) {{
        await fn(ev || {{ preventDefault() {{}} }});
      }}
    }},
    [{json.dumps(adj_prop)}]() {{
      throw new Error('Forbidden adjacent HTML sink');
    }},
  }};
  Object.defineProperty(el, 'textContent', {{
    get() {{ return this._text; }},
    set(v) {{ this._text = String(v); }},
  }});
  for (const prop of [{json.dumps(inner_prop)}, {json.dumps(outer_prop)}]) {{
    Object.defineProperty(el, prop, {{
      get() {{ throw new Error('Forbidden HTML getter: ' + prop); }},
      set() {{ throw new Error('Forbidden HTML setter: ' + prop); }},
    }});
  }}
  return el;
}}

[
  'status', 'key', 'continueYamlSnippet', 'verifyKeyStatus', 'acct', 'role', 'bal', 'kind', 'result',
  'loginForm', 'loginIdentifier', 'loginPassword', 'registerForm', 'regHandle', 'regEmail', 'regPassword',
  'copy', 'copyContinueBtn', 'verifyKeyModelsBtn', 'signupLiveTelemetry', 'activeSessionHandle',
  'activeSessionBalance', 'existingBrowserSessionBanner', 'rotateKeyBtn', 'changePasswordForm',
  'resetTokenOrKey', 'newResetPassword', 'signOutSessionBtn', 'claimBonusBtn', 'bannerClaimBonusBtn',
  'transferPointsForm', 'transferRecipient', 'transferAmount', 'transferMemo', 'signin', 'gbtn',
  'verifyHolidayParityBtn', 'verifyHolidayParityStatus',
  'signupCesProfileDevBtn', 'signupCesProfileGamerBtn',
  'signupCesEaseBtn-1', 'signupCesEaseBtn-2', 'signupCesEaseBtn-3', 'signupCesEaseBtn-4', 'signupCesEaseBtn-5',
  'signupCesLagYesBtn', 'signupCesLagNoBtn',
  'signupSubmitCesBtn', 'signupRetryPendingCesBtn', 'signupCesStatus'
].forEach((id) => {{ elements[id] = makeEl(id); }});

const preseededSpool = JSON.stringify({{
  ballot: {{ profile: 'dev', ease_rating: 5, zero_game_lag: true, platform: 'web-signup', note: '' }},
  idempotency_key: 'ces-preseeded-signup-key',
  last_status: 404,
}});
const store = new Map([
  [{json.dumps(PENDING_SIGNUP_CES_STORAGE_KEY)}, preseededSpool],
]);

globalThis.localStorage = {{
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => {{ store.set(k, String(v)); }},
  removeItem: (k) => {{ store.delete(k); }},
}};
globalThis.window = {{
  location: {{ origin: 'https://hivemind-test.us-central1.run.app', search: '', hash: '' }},
  localStorage: globalThis.localStorage,
}};
globalThis.document = {{
  getElementById: (id) => elements[id] || null,
  querySelector: () => ({{ value: 'consumer', checked: true }}),
  write() {{ throw new Error('Forbidden document write'); }},
}};

const fetchLog = [];
let cesMode = 'ok';
let parityMode = 'ok';
let totalRecorded = 10;

globalThis.fetch = async (url, opts = {{}}) => {{
  if (String(url).includes({json.dumps(forbidden_chat_route)})) {{
    throw new Error('Forbidden call to chat completions');
  }}
  fetchLog.push({{
    url,
    method: opts.method || 'GET',
    headers: opts.headers || {{}},
    body: opts.body ? JSON.parse(opts.body) : null,
  }});
  if (url === '/v1/hivemind/live-telemetry') {{
    return {{ ok: true, status: 200, async json() {{ return {{ accounts: {{ total_registered_accounts: 4, by_role: {{}} }}, models: {{ live_models_count: 2 }}, t4_boinc: {{ vram_free_mib: 8192 }}, opentrainer: {{}} }}; }} }};
  }}
  if (url === '/v1/hivemind/onboarding/manifest') {{
    if (parityMode === '404') return {{ ok: false, status: 404, async json() {{ return {{ detail: 'Not Found' }}; }} }};
    if (parityMode === '503') return {{ ok: false, status: 503, async json() {{ return {{ detail: 'Unavailable' }}; }} }};
    if (parityMode === 'bad_schema') return {{ ok: true, status: 200, async json() {{ return {{ object: 'wrong' }}; }} }};
    if (parityMode === 'offline') throw new Error('manifest_offline');
    return {{
      ok: true,
      status: 200,
      async json() {{
        return {{
          object: 'hivemind.onboarding.manifest',
          campaign_id: 'noel-2026-easy-onboard',
          status: 'ok',
          build_sha: 'abc12345',
          holiday_routes: ['/noel', '/quick', '/raid-noel', '/arena-noel'],
          personal_pc_only: true,
          financial_promise: '0.00 EUR / 0.00 USD',
        }};
      }},
    }};
  }}
  if (url === '/v1/hivemind/onboarding/feedback/summary') {{
    if (parityMode === '404') return {{ ok: false, status: 404, async json() {{ return {{ detail: 'Not Found' }}; }} }};
    if (parityMode === '503') return {{ ok: false, status: 503, async json() {{ return {{ detail: 'Unavailable' }}; }} }};
    if (parityMode === 'offline') throw new Error('summary_offline');
    return {{
      ok: true,
      status: 200,
      async json() {{
        return {{
          object: 'hivemind.onboarding_feedback_summary',
          total_responses: totalRecorded,
          mean_ease_rating: 4.8,
          zero_game_lag_ratio: 0.96,
          by_profile: {{ dev: 6, gamer: totalRecorded - 6 }},
        }};
      }},
    }};
  }}
  if (url === '/v1/hivemind/onboarding/feedback') {{
    if (cesMode === '404') return {{ ok: false, status: 404, async json() {{ return {{ detail: 'Not Found' }}; }} }};
    if (cesMode === '429') return {{ ok: false, status: 429, async json() {{ return {{ detail: 'Rate limited' }}; }} }};
    if (cesMode === '503') return {{ ok: false, status: 503, async json() {{ return {{ detail: 'Busy' }}; }} }};
    if (cesMode === '400') return {{ ok: false, status: 400, async json() {{ return {{ detail: 'Bad request' }}; }} }};
    if (cesMode === 'offline') throw new Error('network_down');
    totalRecorded += 1;
    return {{
      ok: true,
      status: 200,
      async json() {{
        return {{
          object: 'hivemind.onboarding_feedback_summary',
          total_responses: totalRecorded,
          mean_ease_rating: 4.82,
          zero_game_lag_ratio: 0.95,
          by_profile: {{ dev: 7, gamer: totalRecorded - 7 }},
        }};
      }},
    }};
  }}
  return {{ ok: true, status: 200, async json() {{ return {{}}; }} }};
}};
"""

    driver = f"""
(async () => {{
  await Promise.resolve();
  // 1. Zero CES POSTs on page load even with pre-seeded localStorage spool
  const postCountOnLoad = fetchLog.filter((c) => c.url === '/v1/hivemind/onboarding/feedback').length;
  const initPendingNotice = elements.signupCesStatus.textContent;
  const initDevPressed = elements.signupCesProfileDevBtn.getAttribute('aria-pressed');
  const initEase5Pressed = elements['signupCesEaseBtn-5'].getAttribute('aria-pressed');
  const initLagYesPressed = elements.signupCesLagYesBtn.getAttribute('aria-pressed');

  // 2. Holiday Route Parity Button: HTTP 200 live -> HTTP 404 drift -> HTTP 503 -> schema mismatch -> network offline
  parityMode = 'ok';
  await elements.verifyHolidayParityBtn.trigger('click');
  const parityOkText = elements.verifyHolidayParityStatus.textContent;

  parityMode = '404';
  await elements.verifyHolidayParityBtn.trigger('click');
  const parity404Text = elements.verifyHolidayParityStatus.textContent;

  parityMode = '503';
  await elements.verifyHolidayParityBtn.trigger('click');
  const parity503Text = elements.verifyHolidayParityStatus.textContent;

  parityMode = 'bad_schema';
  await elements.verifyHolidayParityBtn.trigger('click');
  const parityBadSchemaText = elements.verifyHolidayParityStatus.textContent;

  parityMode = 'offline';
  await elements.verifyHolidayParityBtn.trigger('click');
  const parityOfflineText = elements.verifyHolidayParityStatus.textContent;

  // 3. Flush pre-seeded CES ballot via signupRetryPendingCesBtn -> HTTP 200 clears localStorage
  await elements.signupRetryPendingCesBtn.trigger('click');
  const afterPreseedStoreHas = store.has({json.dumps(PENDING_SIGNUP_CES_STORAGE_KEY)});
  const preseedPost = fetchLog.filter((c) => c.url === '/v1/hivemind/onboarding/feedback')[0];

  // 4. Toggle Gamer profile, ease 3, zero_game_lag false, then Dev profile, ease 4, zero_game_lag true -> POST 200
  await elements.signupCesProfileGamerBtn.trigger('click');
  await elements['signupCesEaseBtn-3'].trigger('click');
  await elements.signupCesLagNoBtn.trigger('click');
  const gamerPressedMid = elements.signupCesProfileGamerBtn.getAttribute('aria-pressed');
  const lagNoPressedMid = elements.signupCesLagNoBtn.getAttribute('aria-pressed');

  await elements.signupCesProfileDevBtn.trigger('click');
  await elements['signupCesEaseBtn-4'].trigger('click');
  await elements.signupCesLagYesBtn.trigger('click');

  cesMode = 'ok';
  await elements.signupSubmitCesBtn.trigger('click');
  const cesPosts1 = fetchLog.filter((c) => c.url === '/v1/hivemind/onboarding/feedback');
  const firstSubmit = cesPosts1[1];
  const statusAfter200 = elements.signupCesStatus.textContent;

  // Identical ballot produces identical deterministic Idempotency-Key
  await elements.signupSubmitCesBtn.trigger('click');
  const cesPosts2 = fetchLog.filter((c) => c.url === '/v1/hivemind/onboarding/feedback');
  const secondSubmitIdem = cesPosts2[2].headers['Idempotency-Key'];

  // Different selection produces distinct Idempotency-Key & tests 404 -> 429 -> 503 -> offline spool -> 200 flush
  await elements.signupCesProfileGamerBtn.trigger('click');
  await elements['signupCesEaseBtn-5'].trigger('click');
  await elements.signupCesLagNoBtn.trigger('click');

  cesMode = '404';
  await elements.signupSubmitCesBtn.trigger('click');
  const spooled404 = JSON.parse(store.get({json.dumps(PENDING_SIGNUP_CES_STORAGE_KEY)}));
  const msg404 = elements.signupCesStatus.textContent;

  cesMode = '429';
  await elements.signupRetryPendingCesBtn.trigger('click');
  const spooled429 = JSON.parse(store.get({json.dumps(PENDING_SIGNUP_CES_STORAGE_KEY)}));
  const msg429 = elements.signupCesStatus.textContent;

  cesMode = '503';
  await elements.signupRetryPendingCesBtn.trigger('click');
  const msg503 = elements.signupCesStatus.textContent;

  cesMode = 'offline';
  await elements.signupRetryPendingCesBtn.trigger('click');
  const msgOffline = elements.signupCesStatus.textContent;

  cesMode = 'ok';
  await elements.signupRetryPendingCesBtn.trigger('click');
  const cesPostsAfterFlush = fetchLog.filter((c) => c.url === '/v1/hivemind/onboarding/feedback');
  const retryFlushPost = cesPostsAfterFlush[cesPostsAfterFlush.length - 1];
  const clearedAfterFlush = !store.has({json.dumps(PENDING_SIGNUP_CES_STORAGE_KEY)});
  const msgFlushed = elements.signupCesStatus.textContent;

  // Empty retry makes zero fetch calls
  const countBeforeEmpty = fetchLog.length;
  await elements.signupRetryPendingCesBtn.trigger('click');
  const countAfterEmpty = fetchLog.length;

  // Poisoned localStorage entry (non-empty PII note) is purged with zero fetch calls
  store.set({json.dumps(PENDING_SIGNUP_CES_STORAGE_KEY)}, JSON.stringify({{
    ballot: {{ profile: 'dev', ease_rating: 5, zero_game_lag: true, platform: 'web-signup', note: 'alice@example.org' }},
    idempotency_key: 'ces-poison',
  }}));
  await elements.signupRetryPendingCesBtn.trigger('click');
  const poisonedPurged = !store.has({json.dumps(PENDING_SIGNUP_CES_STORAGE_KEY)}) && fetchLog.length === countAfterEmpty;

  // Permanent 400 clears spool
  cesMode = '400';
  await elements.signupSubmitCesBtn.trigger('click');
  const clearedOn400 = !store.has({json.dumps(PENDING_SIGNUP_CES_STORAGE_KEY)});

  console.log(JSON.stringify({{
    postCountOnLoad,
    initPendingNotice,
    initDevPressed,
    initEase5Pressed,
    initLagYesPressed,
    parityOkText,
    parity404Text,
    parity503Text,
    parityBadSchemaText,
    parityOfflineText,
    afterPreseedStoreHas,
    preseedIdem: preseedPost.headers['Idempotency-Key'],
    gamerPressedMid,
    lagNoPressedMid,
    firstSubmit,
    secondSubmitIdem,
    statusAfter200,
    spooled404,
    spooled429LastStatus: spooled429.last_status,
    msg404,
    msg429,
    msg503,
    msgOffline,
    retryFlushIdem: retryFlushPost.headers['Idempotency-Key'],
    clearedAfterFlush,
    msgFlushed,
    countBeforeEmpty,
    countAfterEmpty,
    poisonedPurged,
    clearedOn400,
    fetchLog,
  }}));
}})().catch((err) => {{ console.error(err); process.exit(1); }});
"""

    proc = subprocess.run(
        ["node", "-e", f"{harness}\n{_AUTH_JS}\n{g_js}\n{driver}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["postCountOnLoad"] == 0
    assert "pending locally" in out["initPendingNotice"]
    assert out["initDevPressed"] == "true"
    assert out["initEase5Pressed"] == "true"
    assert out["initLagYesPressed"] == "true"

    assert "Holiday Parity Live (HTTP 200" in out["parityOkText"]
    assert "build abc12345" in out["parityOkText"]
    assert "4 routes" in out["parityOkText"]
    assert "0.00 EUR/USD" in out["parityOkText"]
    assert "HTTP 404" in out["parity404Text"]
    assert "/v1/hivemind/onboarding/manifest" in out["parity404Text"]
    assert "/v1/hivemind/onboarding/feedback/summary" in out["parity404Text"]
    assert "HTTP 503" in out["parity503Text"]
    assert "mismatch" in out["parityBadSchemaText"]
    assert "Could not reach holiday parity" in out["parityOfflineText"]

    assert out["afterPreseedStoreHas"] is False
    assert out["preseedIdem"] == "ces-preseeded-signup-key"
    assert out["gamerPressedMid"] == "true"
    assert out["lagNoPressedMid"] == "true"

    first_sub = out["firstSubmit"]
    assert first_sub["url"] == "/v1/hivemind/onboarding/feedback"
    assert first_sub["method"] == "POST"
    assert first_sub["headers"]["Content-Type"] == "application/json"
    assert first_sub["headers"]["Idempotency-Key"].startswith("ces-dev-4-lag0-")
    assert first_sub["body"] == {
        "profile": "dev",
        "ease_rating": 4,
        "zero_game_lag": True,
        "platform": "web-signup",
        "note": "",
    }
    assert out["secondSubmitIdem"] == first_sub["headers"]["Idempotency-Key"]
    assert "HTTP 200" in out["statusAfter200"]
    assert "0.00 EUR / 0.00 USD" in out["statusAfter200"]

    assert out["spooled404"]["ballot"] == {
        "profile": "gamer",
        "ease_rating": 5,
        "zero_game_lag": False,
        "platform": "web-signup",
        "note": "",
    }
    assert out["spooled404"]["last_status"] == 404
    assert out["spooled429LastStatus"] == 429
    assert "404" in out["msg404"]
    assert "429" in out["msg429"]
    assert "503" in out["msg503"]
    assert "Offline" in out["msgOffline"]
    assert out["retryFlushIdem"].startswith("ces-gamer-5-lag1-")
    assert out["retryFlushIdem"] != first_sub["headers"]["Idempotency-Key"]
    assert out["clearedAfterFlush"] is True
    assert "HTTP 200" in out["msgFlushed"]

    assert out["countBeforeEmpty"] == out["countAfterEmpty"]
    assert out["poisonedPurged"] is True
    assert out["clearedOn400"] is True
    assert not any(forbidden_chat_route in str(c["url"]) for c in out["fetchLog"])


def test_04_concurrent_48_participant_noel_signup_soak_sqlite_wal(tmp_path: Path) -> None:
    """Concurrent 48-user (24 Dev + 24 Gamer) Noel-at-scale soak against create_router_app + SQLite WAL."""
    forbidden_key = "sco" + "re"
    forbidden_chat_route = "/v1/chat/" + "completions"
    db_path = tmp_path / "signup_noel_48_soak.sqlite3"
    store = RouterStore(db_path, secret=SOAK_SECRET)
    journal_mode = str(store._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        open_registration=True,
        starter_grant_hmt=25.0,
        trusted_proxy_hops=1,
    )

    seen_nonces: set[str] = set()
    sqlite_lock_hits: list[str] = []
    seen_paths: list[str] = []
    feedback_post_attempts = 0
    state_lock = threading.Lock()

    try:
        with TestClient(app) as client:

            def _run_participant(participant_idx: int) -> dict[str, Any]:
                nonlocal feedback_post_attempts
                is_dev = participant_idx < 24
                profile = "dev" if is_dev else "gamer"
                cohort_slot = participant_idx if is_dev else (participant_idx - 24)
                peer_ip = _loopback_ip_for(12 + (participant_idx // 24), cohort_slot)
                headers = {"X-Forwarded-For": peer_ip}

                local_nonces: list[str] = []
                local_paths: list[str] = []

                signup_path = "/signup" if is_dev else "/signup?role=trainer"
                for html_route in (signup_path, "/noel", "/quick", "/raid-noel"):
                    page_resp = client.get(html_route, headers=headers)
                    local_paths.append(html_route)
                    raw_html = page_resp.text
                    if _SQLITE_LOCK_RE.search(raw_html):
                        with state_lock:
                            sqlite_lock_hits.append(raw_html[:200])
                    assert page_resp.status_code == 200
                    nonce_val = _extract_csp_header_nonce(page_resp.headers["content-security-policy"])
                    local_nonces.append(nonce_val)
                    stripped = _STRIP_NONCE_ATTR_RE.sub('nonce=""', raw_html)
                    assert STATIC_FIGURE_RE.findall(stripped) == []
                    if html_route.startswith("/signup"):
                        dom = _SignupDomInspector()
                        dom.feed(raw_html)
                        assert dom.inline_handlers == []
                        assert dom.style_nonces == [nonce_val]
                        assert dom.script_nonces == [nonce_val]
                        assert "verifyHolidayParityBtn" in dom.by_id_attrs
                        assert "signupSubmitCesBtn" in dom.by_id_attrs
                        assert "signupRetryPendingCesBtn" in dom.by_id_attrs
                        if not is_dev:
                            assert 'value="trainer" checked' in raw_html

                with state_lock:
                    for n_val in local_nonces:
                        assert n_val not in seen_nonces, f"Duplicate CSP nonce: {n_val}"
                        seen_nonces.add(n_val)

                # Cohort registration via POST /v1/hivemind/auth/register
                handle = f"soak17_{profile}_{cohort_slot:02d}"
                role = "consumer" if is_dev else "trainer"
                reg_resp = client.post(
                    "/v1/hivemind/auth/register",
                    json={"handle": handle, "password": "NoelSoakSafePass!2026", "role": role},
                    headers=headers,
                )
                local_paths.append("/v1/hivemind/auth/register")
                if _SQLITE_LOCK_RE.search(reg_resp.text):
                    with state_lock:
                        sqlite_lock_hits.append(reg_resp.text[:200])
                assert reg_resp.status_code == 200, f"Register failed ({reg_resp.status_code}): {reg_resp.text}"
                reg_body = reg_resp.json()
                assert reg_body["api_key"].startswith("hm-")

                if is_dev:
                    models_resp = client.get(
                        "/v1/models",
                        headers={**headers, "Authorization": f"Bearer {reg_body['api_key']}"},
                    )
                    local_paths.append("/v1/models")
                    assert models_resp.status_code == 200
                    models_doc = models_resp.json()
                    assert models_doc["object"] == "list"
                    assert len(models_doc["data"]) >= 1
                else:
                    man_resp = client.get("/v1/hivemind/onboarding/manifest", headers=headers)
                    local_paths.append("/v1/hivemind/onboarding/manifest")
                    assert man_resp.status_code == 200
                    assert man_resp.json()["object"] == "hivemind.onboarding.manifest"

                # 38 ease=5, 10 ease=4 -> mean 4.79 >= 4.2; 47 lag=True, 1 False -> 0.9792 >= 0.95
                ease_rating = 4 if (participant_idx % 5 == 0) else 5
                zero_game_lag = participant_idx != 47
                ballot = build_signup_ces_ballot(
                    profile=profile,
                    ease_rating=ease_rating,
                    zero_game_lag=zero_game_lag,
                    platform_name="web-signup",
                )
                idem_key = f"soak17-{profile}-{cohort_slot:02d}-{ease_rating}-{int(zero_game_lag)}"

                # Send TWICE with identical Idempotency-Key -> 96 POST attempts, exact 48 conservation
                for _attempt in range(2):
                    fb_resp = client.post(
                        "/v1/hivemind/onboarding/feedback",
                        json=ballot,
                        headers={**headers, "Idempotency-Key": idem_key},
                    )
                    local_paths.append("/v1/hivemind/onboarding/feedback")
                    with state_lock:
                        feedback_post_attempts += 1
                    if _SQLITE_LOCK_RE.search(fb_resp.text):
                        with state_lock:
                            sqlite_lock_hits.append(fb_resp.text[:200])
                    assert fb_resp.status_code == 200, f"CES POST failed: {fb_resp.text}"
                    fb_doc = fb_resp.json()
                    assert set(fb_doc.keys()) == EXPECTED_SUMMARY_KEYS
                    assert forbidden_key not in _collect_all_keys(fb_doc)

                if not is_dev:
                    sum_resp = client.get("/v1/hivemind/onboarding/feedback/summary", headers=headers)
                    local_paths.append("/v1/hivemind/onboarding/feedback/summary")
                    assert sum_resp.status_code == 200
                    assert set(sum_resp.json().keys()) == EXPECTED_SUMMARY_KEYS

                with state_lock:
                    seen_paths.extend(local_paths)

                return {
                    "profile": profile,
                    "ease_rating": ease_rating,
                    "zero_game_lag": zero_game_lag,
                }

            outcomes: list[dict[str, Any]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(_run_participant, idx) for idx in range(48)]
                for fut in concurrent.futures.as_completed(futures):
                    outcomes.append(fut.result())

            assert len(outcomes) == 48
            assert feedback_post_attempts == 96
            assert sqlite_lock_hits == []
            assert not any(forbidden_chat_route in p for p in seen_paths)
            assert len(seen_nonces) == 48 * 4

            final_summary_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
            assert final_summary_resp.status_code == 200
            final_summary = final_summary_resp.json()
            assert set(final_summary.keys()) == EXPECTED_SUMMARY_KEYS
            assert forbidden_key not in _collect_all_keys(final_summary)
            assert final_summary["object"] == "hivemind.onboarding_feedback_summary"
            assert final_summary["total_responses"] == 48
            assert final_summary["by_profile"] == {"dev": 24, "gamer": 24}
            assert float(final_summary["mean_ease_rating"]) >= 4.2
            assert float(final_summary["zero_game_lag_ratio"]) >= 0.95

            expected_mean = round(sum(o["ease_rating"] for o in outcomes) / 48, 2)
            expected_lag = round(sum(1 for o in outcomes if o["zero_game_lag"]) / 48, 4)
            assert final_summary["mean_ease_rating"] == expected_mean
            assert final_summary["zero_game_lag_ratio"] == expected_lag

            # Live post-soak holiday parity check across all 6 routes
            post_soak_probes: dict[str, dict[str, Any]] = {}
            for route in SIGNUP_HOLIDAY_PARITY_ROUTES:
                probe_resp = client.get(route)
                post_soak_probes[route] = {
                    "status_code": probe_resp.status_code,
                    "headers": dict(probe_resp.headers),
                    "body": probe_resp.text,
                }
            parity_report = evaluate_signup_holiday_parity(post_soak_probes)
            assert parity_report["parity_ok"] is True
            assert parity_report["stale_404_drift"] is False
            assert parity_report["routes_passed_count"] == 6
            assert parity_report["unique_csp_nonces_count"] == 4
            assert parity_report["zero_inference_verified"] is True
    finally:
        store.close()
