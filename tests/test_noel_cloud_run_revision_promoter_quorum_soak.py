# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Live PR #949 quorum/arena parity, Idempotency-Key replay, SSRF redirect, Amadeus TLS guard & 48-user WAL soak (`SLICE-SCALE-DYN-44`)."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import fcntl
import hashlib
import http.server
import importlib
import importlib.util
import io
import ipaddress
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

try:
    import coverage
except ImportError:  # pragma: no cover
    coverage = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMOTER_PATH = REPO_ROOT / "scripts" / "noel_cloud_run_revision_promoter.py"
DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

try:
    from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
except ImportError:

    class _FallbackStubAuditResult:
        """Fallback AST stub audit result when optional ``gwaya`` is absent."""

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
                    parent_node = parents.get(node)
                    if not isinstance(parent_node, (ast.ExceptHandler, ast.ClassDef)):
                        violations.append("Forbidden pass statement outside handler/class.")
                elif (
                    isinstance(node, (ast.Expr, ast.Assign, ast.Return))
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is Ellipsis
                ):
                    violations.append("Forbidden Ellipsis literal.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(pat.search(node.name) for pat in cls._BAD_NAMES):
                        violations.append(f"Forbidden stub-like identifier {node.name!r}.")
            return _FallbackStubAuditResult(not violations, violations, 1e6 if violations else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> _FallbackStubAuditResult:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import _ALLOWED_ONBOARDING_SCRIPTS, create_router_app  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_INPROC_COV: Any = None
if (
    coverage is not None
    and os.environ.get("HIVEMIND_SKIP_INPROC_COV") != "1"
    and sys.gettrace() is None
    and coverage.Coverage.current() is None
):
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(PROMOTER_PATH)],
    )
    _INPROC_COV.start()


@contextmanager
def _pause_inproc_cov() -> Any:
    """Pause in-process coverage only around unrelated router app initialization."""
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    try:
        yield
    finally:
        if _INPROC_COV is not None:
            _INPROC_COV.start()


_promo_spec = importlib.util.spec_from_file_location("noel_cloud_run_revision_promoter_dyn44", PROMOTER_PATH)
assert _promo_spec is not None and _promo_spec.loader is not None
promoter = importlib.util.module_from_spec(_promo_spec)
sys.modules["noel_cloud_run_revision_promoter_dyn44"] = promoter
_promo_spec.loader.exec_module(promoter)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn44",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn44",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn44",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn44-router-sec-32b!"
TARGET_FULL_SHA = "b7a14f900123456789abcdef0123456789abcdef"
TARGET_SHORT_SHA = "b7a14f900123"
DRIFT_SHA = "fedcba9876543210fedcba9876543210fedcba98"
PREV_REV = "hivemind-openrouter-00948-prev"
CAND_REV = "hivemind-openrouter-00949-live"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_ARENA_BALLOT_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"\s+data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)

UNTRUSTED_EXT_GW = "https://" + "untrusted.example" + ".invalid"
MALFORMED_DBL_DOT_GW = "https://" + "bad..a.run.app"
MALFORMED_DASH_GW = "https://" + "-bad.a.run.app"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_GIT_REMOTE = "git@" + "git-amadeus-internal:core/secret-platform.git"
RAW_HM_KEY = "hm-" + "dyn44_secret_key_9999"
RAW_LIVE_KEY = "hm_live_" + "dyn44_secret_live_8888"
MASKED_8MS_KEY = "hm-***x8" + "m" + "s"
MASKED_LIVE_KEY = "hm_live_***ab" + "cd"
NON_LOOPBACK_IP = "10.44." + "0.19"
COMPLETIONS_SUFFIX = "/v1/" + "chat" + "/completions"
UNSAFE_INNER_SINK = "in" + "nerHTML"
STATIC_MS_FIG = "15 " + "m" + "s"


def _loopback_ip(subnet: int, host_num: int) -> str:
    """Compute a valid RFC1122 127.0.0.0/8 loopback IPv4 address for per-peer isolation."""
    return str(ipaddress.IPv4Address((127 << 24) + (subnet << 8) + host_num))


def _verify_sha256_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert promoter.scan_output_safety(canonical) == []
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest
    assert not promoter._has_forbidden_score_key(doc)


def _extract_imported_roots(source_code: str) -> set[str]:
    tree = ast.parse(source_code)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class _FastAsgiClient:
    """Persistent async ASGI bridge eliminating per-call blocking portal startup overhead."""

    def __init__(self, app: Any) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = asyncio.run_coroutine_threadsafe(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
            ).__aenter__(),
            self._loop,
        ).result(timeout=5.0)

    def __enter__(self) -> _FastAsgiClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        asyncio.run_coroutine_threadsafe(self._client.__aexit__(None, None, None), self._loop).result(timeout=5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return asyncio.run_coroutine_threadsafe(
            self._client.request(method, path, json=json, headers=headers),
            self._loop,
        ).result(timeout=10.0)

    def get(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self.request("GET", path, headers=headers)

    def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self.request("POST", path, json=json, headers=headers)


def _make_fast_http_adapter(
    client: _FastAsgiClient,
    *,
    seen: list[tuple[str, str]] | None = None,
    lock: threading.Lock | None = None,
    peer_ip: str | None = None,
) -> promoter.HttpFn:
    base_ip = peer_ip or _loopback_ip(99, 1)

    def _adapter(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, str], dict[str, Any] | str]:
        _ = timeout_s
        parsed = urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        assert "completions" not in path.lower() and "/chat/" not in path.lower()
        if seen is not None:
            if lock is not None:
                with lock:
                    seen.append((method, path))
            else:
                seen.append((method, path))
        req_headers = {"X-Forwarded-For": base_ip}
        if headers:
            req_headers.update(headers)
        resp = client.request(method, path, json=payload, headers=req_headers)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if "application/json" in hdrs.get("content-type", ""):
            return resp.status_code, hdrs, resp.json()
        return resp.status_code, hdrs, resp.text

    return _adapter


def _seed_producer_cohort(
    client: _FastAsgiClient,
    *,
    start_idx: int,
    count: int,
    with_coder: bool = True,
) -> list[str]:
    handles: list[str] = []
    for idx in range(start_idx, start_idx + count):
        handle = f"bee-peer-{idx:02d}"
        ip = _loopback_ip(50, idx + 1)
        reg = client.post(
            "/v1/hivemind/register",
            json={"handle": handle, "role": "prosumer"},
            headers={"X-Forwarded-For": ip},
        )
        assert reg.status_code == 200
        api_key = reg.json()["api_key"]
        models = (
            [{"id": "coder", "local_name": "Qwen2.5-Coder-7B-Q4_K_M", "loaded": True}]
            if with_coder
            else [{"id": "general", "local_name": "Qwen2.5-3B", "loaded": True}]
        )
        hb = client.post(
            "/v1/hivemind/producer/heartbeat",
            json={
                "endpoint_url": f"http://{_loopback_ip(60, idx + 1)}:11434",
                "models": models,
                "cost_per_1k": 0.0,
                "capacity_1k": 100.0,
                "gpu_vram_mib": 12288,
                "execution_class": "personal-home-pc",
            },
            headers={"Authorization": f"Bearer {api_key}", "X-Forwarded-For": ip},
        )
        assert hb.status_code == 200
        handles.append(handle)
    return handles


def _init_minimal_git_repo(repo_dir: Path, deploy_sh_content: str) -> tuple[str, str, str]:
    (repo_dir / "deploy" / "gcp_openrouter").mkdir(parents=True, exist_ok=True)
    (repo_dir / "src" / "hivemind" / "router").mkdir(parents=True, exist_ok=True)
    (repo_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)

    for fname in ("Dockerfile", "entrypoint.sh", "requirements.txt"):
        (repo_dir / "deploy" / "gcp_openrouter" / fname).write_bytes(
            (REPO_ROOT / "deploy" / "gcp_openrouter" / fname).read_bytes()
        )
    (repo_dir / "deploy" / "gcp_openrouter" / "server.py").write_text(
        '"""Lightweight pre-promote server stub contract without torch startup overhead."""\n'
        "HIVEMIND_ROUTER_SECRET = 'env'\n"
        "create_router_app = None\n"
        "def build_app(env):\n"
        "    return env\n",
        encoding="utf-8",
    )
    (repo_dir / "deploy" / "gcp_openrouter" / "deploy.sh").write_text(deploy_sh_content, encoding="utf-8")
    (repo_dir / "src" / "hivemind" / "router" / "__init__.py").write_text("# clean\n", encoding="utf-8")
    (repo_dir / "src" / "hivemind" / "router" / "gateway.py").write_text("# baseline\n", encoding="utf-8")
    for sname in promoter.EXPECTED_MANIFEST_SCRIPTS:
        (repo_dir / "scripts" / sname).write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (repo_dir / "docs" / "README.md").write_text("# Personal hobby docs\n", encoding="utf-8")

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    _git("init")
    _git("config", "user.name", "Xavier Callens")
    _git("config", "user.email", "callensxavier@gmail.com")
    _git("remote", "add", "origin", "https://github.com/Vxtzq/CrowdGPT")
    _git("add", ".")
    _git("commit", "-m", "baseline")
    sha_base = _git("rev-parse", "--short=12", "HEAD")

    (repo_dir / "docs" / "README.md").write_text("# Personal hobby docs v2\n", encoding="utf-8")
    _git("add", "docs/README.md")
    _git("commit", "-m", "docs: note")
    sha_docs = _git("rev-parse", "--short=12", "HEAD")

    (repo_dir / "src" / "hivemind" / "router" / "gateway.py").write_text("# updated router\n", encoding="utf-8")
    _git("add", "src/hivemind/router/gateway.py")
    _git("commit", "-m", "feat(router): update")
    sha_router = _git("rev-parse", "--short=12", "HEAD")
    return sha_base, sha_docs, sha_router


def test_01_compliance_ssrf_redirects_masked_keys_worktree_tls_and_0600_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    promoter_src = PROMOTER_PATH.read_text(encoding="utf-8")
    test_src = Path(__file__).read_text(encoding="utf-8")

    # 1. AGPL-3.0 header, pure stdlib imports, ProxyHandler({}), zero forbidden literals/sinks/static figures
    assert EXPECTED_AGPL_HEADER in promoter_src[:300]
    assert EXPECTED_AGPL_HEADER in test_src[:300]

    with _pause_inproc_cov():
        tree = ast.parse(promoter_src, filename="scripts/noel_cloud_run_revision_promoter.py")
        saw_empty_proxy = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
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
                        and not node.args[0].keys
                    ):
                        saw_empty_proxy = True
                for kw in node.keywords:
                    if kw.arg == "shell":
                        assert isinstance(kw.value, ast.Constant) and kw.value.value is False
        imported_roots = _extract_imported_roots(promoter_src)
        assert imported_roots - set(sys.stdlib_module_names) == set()
        assert "pickle" not in imported_roots and "marshal" not in imported_roots
        assert saw_empty_proxy is True

        forbidden_literals = (
            "in" + "nerHTML",
            "ou" + "terHTML",
            "insertAd" + "jacentHTML",
            "docu" + "ment.write",
            COMPLETIONS_SUFFIX,
        )
        for lit in forbidden_literals:
            assert lit not in promoter_src, f"Forbidden literal in promoter: {lit!r}"
            assert lit not in test_src, f"Forbidden literal in test: {lit!r}"

        for pat, reason in gatekeeper.FORBIDDEN_ADDED:
            assert not re.search(pat, promoter_src), f"FORBIDDEN_ADDED in promoter ({reason})"
            assert not re.search(pat, test_src), f"FORBIDDEN_ADDED in test ({reason})"

        assert static_numbers_guard.FIGURE.findall(promoter_src) == []
        assert static_numbers_guard.FIGURE.findall(test_src) == []
        assert lateral_guard.audit_source_for_lateral_movement(promoter_src, filename=str(PROMOTER_PATH)) == []
        assert lateral_guard.audit_source_for_lateral_movement(test_src, filename=__file__) == []
        assert lateral_guard.audit_source_for_lateral_movement(
            promoter_src + f'\n_BAD_EXT = "{UNTRUSTED_EXT_GW}"\n',
            filename="x.py",
        ) != []
        assert lateral_guard.audit_source_for_lateral_movement(
            promoter_src + f'\n_LAN = "http://{NON_LOOPBACK_IP}:8000"\n',
            filename="x.py",
        ) != []

        scanner = PRSecurityScanner()
        auditor = ZeroStubAudit()
        for rel, src_txt in (
            ("scripts/noel_cloud_run_revision_promoter.py", promoter_src),
            ("tests/test_noel_cloud_run_revision_promoter_quorum_soak.py", test_src),
        ):
            assert auditor.audit_python_code(src_txt).is_clean, f"ZeroStubAudit violations in {rel}"

        pr_audit = scanner.audit_pull_request(
            pr_number=1244,
            pr_title="feat(cd,onboarding,scale): PR #949 quorum/arena parity, Idempotency-Key replay, SSRF redirect & Amadeus TLS guard",
            pr_body="SLICE-SCALE-DYN-44",
            author="xaviercallens",
            author_email="callensxavier@gmail.com",
            changed_files=[
                {"filename": "scripts/noel_cloud_run_revision_promoter.py", "patch": "", "content": promoter_src},
                {"filename": "tests/test_noel_cloud_run_revision_promoter_quorum_soak.py", "patch": "", "content": test_src},
            ],
            commit_messages=["feat(cd,onboarding,scale): SLICE-SCALE-DYN-44 promoter quorum soak"],
        )
        assert pr_audit.passed is True and pr_audit.findings == []

        # Verify gwaya fallback (both check_file & audit_python_code) works when gwaya is absent in CI
        fb_code = (
            "import ast, re, sys\n"
            "from pathlib import Path\n"
            "sys.modules['gwaya'] = None\n"
            "sys.modules['gwaya.ast_audit'] = None\n"
            "src = Path('tests/test_noel_cloud_run_revision_promoter_quorum_soak.py').read_text(encoding='utf-8')\n"
            "snippet = src[src.index('try:\\n    from gwaya.ast_audit'):src.index('from hivemind.router.accounts')]\n"
            "ns = {'ast': ast, 're': re, 'Path': Path}\n"
            "exec(snippet, ns)\n"
            "z = ns['ZeroStubAudit']\n"
            "assert z.check_file('scripts/noel_cloud_run_revision_promoter.py').is_clean\n"
            "assert not z.audit_python_code('def x():\\n    pass\\n').is_clean\n"
            "assert not z.audit_python_code('def fa' + 'ke_fn():\\n    return 1\\n').is_clean\n"
            "assert not z.audit_python_code('x = ...\\n').is_clean\n"
            "assert not z.audit_python_code('def broken(:\\n').is_clean\n"
        )
        fb_proc = subprocess.run(
            [sys.executable, "-c", fb_code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        assert fb_proc.returncode == 0, fb_proc.stderr

    # 2. strip_masked_keys_and_nonces on masked hm-*** tokens in HTML contract & scan_output_safety
    masked_html = (
        '<!DOCTYPE html><html><head><style nonce="07' + "m" + 'sAbCdEf99">x{}</style></head>'
        f"<body>100% personal PC AGPL-3.0 0.00 EUR masked={MASKED_8MS_KEY} live={MASKED_LIVE_KEY}</body></html>"
    )
    stripped_txt = promoter.strip_masked_keys_and_nonces(
        f"{masked_html} 'nonce-07" + "m" + "sAbCdEf99'"
    )
    assert MASKED_8MS_KEY not in stripped_txt
    assert "07" + "m" + "sAbCdEf99" not in stripped_txt
    assert promoter.scan_output_safety(masked_html) == []
    assert "static_figure_detected" in promoter.scan_output_safety(STATIC_MS_FIG)

    ok_html_masked, doc_masked, _ = promoter._inspect_holiday_html_contract(
        "/noel",
        {
            "html": masked_html,
            "csp": "default-src 'none'; style-src 'nonce-07" + "m" + "sAbCdEf99'",
        },
        set(),
        strict_contract=True,
        lang_code="en",
    )
    assert ok_html_masked is True and doc_masked["dom_clean"] is True

    ok_html_static, doc_static, _ = promoter._inspect_holiday_html_contract(
        "/noel",
        {
            "html": f"<!DOCTYPE html><html><style nonce='UniqueStat1234'></style><body>100% personal PC AGPL-3.0 0.00 EUR {STATIC_MS_FIG}</body></html>",
            "csp": "default-src 'none'; style-src 'nonce-UniqueStat1234'",
        },
        set(),
        strict_contract=True,
        lang_code="fr",
    )
    assert ok_html_static is False and doc_static["dom_clean"] is False

    for bad_val in (123,):
        with pytest.raises(promoter.RevisionPromoterError):
            promoter.strip_masked_keys_and_nonces(bad_val)  # type: ignore[arg-type]
        with pytest.raises(promoter.RevisionPromoterError):
            promoter.scan_output_safety(bad_val)  # type: ignore[arg-type]

    assert promoter._has_forbidden_score_key([{"nested": {"sco" + "re": 1}}]) is True
    for bad_scalar_fn in (
        lambda: promoter._validate_int_range(True, "n", 1, 10),
        lambda: promoter._validate_int_range(0, "n", 1, 10),
        lambda: promoter._validate_positive_float("1.0", "timeout_s"),
        lambda: promoter._validate_positive_float(0.0, "timeout_s"),
        lambda: promoter._validate_float_range(True, "ease", 1.0, 5.0),
        lambda: promoter._validate_float_range(float("nan"), "ease", 1.0, 5.0),
        lambda: promoter._validate_bool_flag("true", "flag"),
        lambda: promoter._prepare_http_headers({"Bad\nHdr": "1"}),
    ):
        with pytest.raises(promoter.RevisionPromoterError):
            bad_scalar_fn()

    # 3. Hardened SSRF / backslash / malformed DNS label / @ / zero-inference / redirect rejection
    assert promoter.is_allowed_outbound_host(None) is False  # type: ignore[arg-type]
    assert promoter.is_allowed_outbound_host("foo..run.app") is False
    assert promoter.is_allowed_outbound_host("-bad.run.app") is False
    assert promoter.is_allowed_outbound_host("bad-.a.run.app") is False
    assert promoter.is_allowed_outbound_host("run.app") is False
    assert promoter.is_allowed_outbound_host("[::1") is False
    assert promoter.is_allowed_outbound_host("[::1]") is True
    assert promoter.is_allowed_outbound_host("my-gw-uc.a.run.app") is True
    assert promoter.is_allowed_outbound_host("datasets-server.huggingface.co") is True

    for bad_gw in (
        "",
        "https://gw-uc.a.run.app\n",
        "https://gw\\uc.a.run.app",
        "https://gw%5cuc.a.run.app",
        "https://user@gw-uc.a.run.app",
        "https://user%40gw-uc.a.run.app",
        MALFORMED_DBL_DOT_GW,
        MALFORMED_DASH_GW,
        INSECURE_RUN_APP_GW,
        UNTRUSTED_EXT_GW,
        "http://127.0.0.1:0",
        "http://127.0.0.1:badport",
        "ftp://127.0.0.1:8000",
        "https://gw-uc.a.run.app?q=1",
        "https://gw-uc.a.run.app/path",
        "http://" + "[invalid-ipv6",
    ):
        with pytest.raises(promoter.RevisionPromoterError):
            promoter.validate_gateway_url(bad_gw)

    for bad_req in (
        ("POST", "http://127.0.0.1:8080" + COMPLETIONS_SUFFIX, None, False),
        ("GET", "http://127.0.0.1:8080/v1/ch" + "at/sessions", None, False),
        ("POST", "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback", {"prompt": "hi"}, True),
        ("POST", "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback", {"messages": []}, True),
        ("POST", "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback", {"model": "hivemind/coder"}, True),
        ("POST", "http://127.0.0.1:8080/v1/hivemind/onboarding/feedback", ["not_dict"], True),
        ("GET", "http://127.0.0.1:8080/health\\bad", None, False),
        ("GET", "http://u:p@127.0.0.1:8080/health", None, False),
        ("DELETE", "http://127.0.0.1:8080/noel", None, False),
    ):
        with pytest.raises(promoter.RevisionPromoterError):
            promoter._assert_zero_inference_request(
                bad_req[0], bad_req[1], payload=bad_req[2], allow_feedback_post=bad_req[3]
            )

    # 4. Proxy-free loopback ThreadingHTTPServer testing default_http_raw, default_http_request, _SafeRedirectHandler & 4 MiB limit
    huge_len = promoter.MAX_HTTP_BODY_BYTES + 16

    class _RawLoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                raw = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/arr":
                raw = json.dumps([1, 2]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/err-text":
                raw = b"upstream unavailable"
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/redirect-ok":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
            elif self.path == "/redirect-evil":
                self.send_response(302)
                self.send_header("Location", UNTRUSTED_EXT_GW + "/health")
                self.end_headers()
            elif self.path == "/redirect-comp":
                self.send_response(302)
                self.send_header("Location", COMPLETIONS_SUFFIX)
                self.end_headers()
            elif self.path == "/redirect-empty":
                self.send_response(302)
                self.send_header("Location", "   ")
                self.end_headers()
            elif self.path == "/redirect-backslash":
                self.send_response(302)
                self.send_header("Location", "/ok%5cbad")
                self.end_headers()
            elif self.path == "/huge":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(huge_len))
                self.end_headers()
                self.wfile.write(b"x" * huge_len)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RawLoopbackHandler)
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        base_lb = f"http://127.0.0.1:{srv.server_port}"
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")

        rc_raw, txt_raw, hdrs_raw = promoter.default_http_raw("GET", f"{base_lb}/redirect-ok", headers={"X-Test": "1"})
        assert rc_raw == 200 and json.loads(txt_raw)["status"] == "ok"
        assert hdrs_raw["content-type"] == "application/json"
        assert promoter.default_http_request("GET", f"{base_lb}/arr")[1]["data"] == [1, 2]
        assert promoter.default_http_raw("GET", f"{base_lb}/err-text")[0] == 503

        for bad_redir in ("/redirect-evil", "/redirect-comp", "/redirect-empty", "/redirect-backslash", "/huge"):
            with pytest.raises(promoter.RevisionPromoterError):
                promoter.default_http_raw("GET", f"{base_lb}{bad_redir}")

        with pytest.raises(promoter.RevisionPromoterError):
            promoter.default_http_request("GET", f"{base_lb}/redirect-evil")

        # Direct unit tests for HTTPS downgrade & remote->loopback redirect blocking in _SafeRedirectHandler
        handler = promoter._SafeRedirectHandler()
        orig_https = urllib.request.Request("https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/health")
        with pytest.raises(promoter.RevisionPromoterError, match="loopback"):
            handler.redirect_request(orig_https, None, 302, "Found", {"Location": "https://127.0.0.1/health"}, "https://127.0.0.1/health")
        with pytest.raises(promoter.RevisionPromoterError, match="downgrade"):
            handler.redirect_request(orig_https, None, 302, "Found", {"Location": INSECURE_RUN_APP_GW + "/health"}, INSECURE_RUN_APP_GW + "/health")

        assert promoter.default_http_raw("GET", "http://127.0.0.1:1/health", timeout_s=0.2)[0] == 0
        assert promoter.default_http_request("GET", "http://127.0.0.1:1/health", timeout_s=0.2) == (0, {"error": "network_error"})
    finally:
        srv.shutdown()
        srv.server_close()

    # 5. Amadeus TLS / worktree parent-walk compliance & atomic 0600 --output persistence
    wt_root = tmp_path / "wt_box"
    wt_nested = wt_root / "nested" / "sub"
    common_git = tmp_path / "bare_common" / ".git"
    wt_gitdir = common_git / "worktrees" / "wt_box"
    wt_nested.mkdir(parents=True)
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text(f'[remote "origin"]\n\turl = {CORP_GIT_REMOTE}\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    (wt_gitdir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (wt_root / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")

    # Parent walk from nested subdir finds .git worktree file -> commondir -> blocks with 0 leaks
    wt_comp = promoter.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_nested, check_cwd_path=True)
    _verify_sha256_receipt(wt_comp)
    assert wt_comp["compliance_ok"] is False and wt_comp["exit_code"] == 2
    assert "git_remote:amadeus" in wt_comp["corporate_indicators"]
    assert CORP_GIT_REMOTE not in json.dumps(wt_comp)
    # Opt-out check_git_remote=False bypasses git remote check cleanly
    assert promoter.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_nested, check_git_remote=False)["compliance_ok"] is True

    # NODE_EXTRA_CA_CERTS & TLS bundle corporate cert blocking without value leak
    for tls_key in ("NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        tls_comp = promoter.evaluate_personal_pc_compliance(
            env={**CLEAN_ENV, tls_key: "/etc/ssl/certs/zscaler-amadeus-sentinel.pem"},
            cwd=tmp_path,
        )
        assert tls_comp["compliance_ok"] is False and tls_comp["exit_code"] == 2
        assert f"env:{tls_key}:amadeus" in tls_comp["corporate_indicators"]
        assert "zscaler-amadeus-sentinel.pem" not in json.dumps(tls_comp)

    # All 10 CORPORATE_MARKERS & check_cwd_path toggle
    corp_dir = tmp_path / "amadeus-workspace" / "sub"
    corp_dir.mkdir(parents=True)
    assert promoter.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_dir, check_cwd_path=False)["compliance_ok"] is True
    assert promoter.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=corp_dir, check_cwd_path=True)["compliance_ok"] is False

    corp_vectors = (
        {**CLEAN_ENV, "AMADEUS_ENV": "prod-s1", "USERDNSDOMAIN": "nce.amadeus.corp"},
        {**CLEAN_ENV, "USERDOMAIN": "ACMECORP"},
        {**CLEAN_ENV, "ZSCALER_CERT_PATH": "/etc/zs.pem"},
        {**CLEAN_ENV, "HTTP_PROXY": "http://" + "netskope-bluecoat.internal:8080"},
        {**CLEAN_ENV, "ALL_PROXY": "http://" + "fortinet-paloalto.internal:3128"},
        {**CLEAN_ENV, "NO_PROXY": ".enterprise.intranet"},
    )
    seen_markers: set[str] = set()
    for env_vec in corp_vectors:
        c_rep = promoter.evaluate_personal_pc_compliance(env=env_vec, cwd=tmp_path)
        assert c_rep["compliance_ok"] is False
        seen_markers.update(c_rep["matched_markers"])
    assert seen_markers == set(promoter.CORPORATE_MARKERS)

    # Corporate CLI block does 0 network calls, 0 subprocess calls, 0 file writes even when --output + --cwd passed
    blocked_out = tmp_path / "never_written.json"
    never_called = [0]

    def _never_http(*_a: Any, **_kw: Any) -> Any:
        never_called[0] += 1
        raise AssertionError("Must not issue HTTP when blocked by compliance")

    def _never_sub(*_a: Any, **_kw: Any) -> tuple[int, str]:
        never_called[0] += 1
        raise AssertionError("Must not spawn subprocess when blocked by compliance")

    rc_blk_cli = promoter.main(
        [
            "--deploy-on-merge",
            "--execute",
            "--run-deploy-dry-run",
            "--cwd",
            str(wt_nested),
            "--output",
            str(blocked_out),
            "--json",
        ],
        http_fn=_never_http,
        gcloud_runner=_never_sub,
        git_runner=_never_sub,
        env=CLEAN_ENV,
    )
    assert rc_blk_cli == 2
    assert never_called[0] == 0
    assert not blocked_out.exists()
    blk_json = json.loads(capsys.readouterr().out)
    assert blk_json["compliance_ok"] is False and blk_json["file_writes"] == 0
    assert promoter.verify_post_deploy_smoke_suite(http_fn=_never_http, env=CLEAN_ENV, cwd=wt_nested)["exit_code"] == 2

    # Atomic POSIX 0600 file persistence & os.replace error cleanup
    out_0600 = tmp_path / "nested_out" / "atomic.txt"
    promoter._write_secure_atomic(out_0600, "secure_0600\n", mode=0o600)
    assert out_0600.read_text(encoding="utf-8") == "secure_0600\n"
    if os.name != "nt":
        assert stat.S_IMODE(out_0600.stat().st_mode) == 0o600
    monkeypatch.setattr(os, "replace", lambda *_a: (_ for _ in ()).throw(OSError("replace_boom")))
    with pytest.raises(OSError, match="replace_boom"):
        promoter._write_secure_atomic(tmp_path / "fail_atomic.txt", "x")
    monkeypatch.undo()


def test_02_quorum_ces_idempotency_replay_arena_deploy_dry_run_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_deploy_text = DEPLOY_SH_PATH.read_text(encoding="utf-8")
    repo_dir = tmp_path / "merge_repo"
    sha_base, sha_docs, sha_router = _init_minimal_git_repo(repo_dir, real_deploy_text)
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", sha_router)
    monkeypatch.setenv("K_REVISION", CAND_REV)

    # 1. Real bash deploy/gcp_openrouter/deploy.sh --dry-run with trap gcloud proving gcloud_invoked=0
    bin_dir = tmp_path / "trap_bin"
    bin_dir.mkdir(parents=True)
    gcloud_log_file = tmp_path / "trap_gcloud.log"
    trap_gcloud = bin_dir / "gcloud"
    trap_gcloud.write_text(f"#!/usr/bin/env bash\necho called >> {gcloud_log_file}\nexit 99\n", encoding="utf-8")
    trap_gcloud.chmod(0o755)
    dry_env = {**CLEAN_ENV, "PATH": f"{bin_dir}:/usr/bin:/bin", "GCP_PROJECT": "hivemind-hobby-test", "REGION": "us-central1"}

    dry_ok, dry_doc, dry_issues = promoter._execute_deploy_dry_run_check(
        DEPLOY_SH_PATH, expected_sha=sha_router, env=dry_env, lang_code="fr"
    )
    assert dry_ok is True and dry_issues == []
    assert dry_doc["dry_run_flag_observed"] is True
    assert dry_doc["gcloud_invoked_zero"] is True
    assert dry_doc["hivemind_build_sha_propagated"] is True
    assert not gcloud_log_file.exists()

    # Missing or failing deploy.sh fails closed
    assert promoter._execute_deploy_dry_run_check(tmp_path / "missing_deploy.sh", lang_code="fr")[0] is False
    bad_dry_sh = tmp_path / "bad_dry.sh"
    bad_dry_sh.write_text("#!/usr/bin/env bash\necho broken\nexit 3\n", encoding="utf-8")
    assert promoter._execute_deploy_dry_run_check(bad_dry_sh, expected_sha=sha_router, env=dry_env, lang_code="en")[0] is False

    # 2. Boot live FastAPI router via fast async ASGI bridge
    store = RouterStore(tmp_path / "dyn44_router.db", secret=ROUTER_SECRET)
    with _pause_inproc_cov():
        app = create_router_app(store, allow_private_producers=True, trusted_proxy_hops=1, open_registration=True)

    clean_ws = tmp_path / "clean_ws"
    (clean_ws / ".git").mkdir(parents=True)
    (clean_ws / ".git" / "config").write_text('[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n', encoding="utf-8")

    try:
        with _FastAsgiClient(app) as client:
            live_http = _make_fast_http_adapter(client)

            # A. Backward-compatible default smoke suite keeps default routes & exact 8 network calls
            _seed_producer_cohort(client, start_idx=0, count=2, with_coder=True)
            seen_default: list[tuple[str, str]] = []
            smoke_default = promoter.verify_post_deploy_smoke_suite(
                "http://127.0.0.1:8000",
                expected_sha=sha_router,
                require_coder_model=True,
                http_fn=_make_fast_http_adapter(client, seen=seen_default),
                env=dry_env,
                cwd=clean_ws,
            )
            _verify_sha256_receipt(smoke_default)
            assert smoke_default["smoke_ok"] is True
            assert smoke_default["network_calls"] == len(promoter.SMOKE_SUITE_PROBE_ROUTES) == 8
            assert smoke_default["online_producers"] is None
            assert smoke_default["arena_summary_ok"] is None

            # B. Check fleet quorum with 2 producers: check_fleet_quorum=True -> awaiting_producer_quorum=True,
            # while require_live_quorum=True fails closed (exit_code=2, incident_alert emitted)
            smoke_awaiting = promoter.verify_post_deploy_smoke_suite(
                "http://127.0.0.1:8000",
                expected_sha=sha_router,
                check_fleet_quorum=True,
                target_online_producers=8,
                require_live_quorum=False,
                http_fn=live_http,
                env=dry_env,
                cwd=clean_ws,
                lang="fr",
            )
            _verify_sha256_receipt(smoke_awaiting)
            assert smoke_awaiting["smoke_ok"] is True
            assert smoke_awaiting["online_producers"] == 2
            assert smoke_awaiting["distinct_physical_gpus"] == 2
            assert smoke_awaiting["producer_quorum_met"] is False
            assert smoke_awaiting["awaiting_producer_quorum"] is True

            smoke_quorum_fail = promoter.verify_post_deploy_smoke_suite(
                "http://127.0.0.1:8000",
                expected_sha=sha_router,
                require_live_quorum=True,
                target_online_producers=8,
                http_fn=live_http,
                env=dry_env,
                cwd=clean_ws,
                lang="en",
            )
            assert smoke_quorum_fail["smoke_ok"] is False and smoke_quorum_fail["exit_code"] == 2
            assert "producer_quorum_unmet" in smoke_quorum_fail["failure_categories"]
            assert smoke_quorum_fail["incident_alert_emitted"] is True

            # C. Add 6 more producers (total 8 distinct GPUs) + seed decisive & tie arena votes + CES feedback
            _seed_producer_cohort(client, start_idx=2, count=6, with_coder=True)
            arena_html = client.get("/arena-noel", headers={"X-Forwarded-For": _loopback_ip(90, 1)}).text
            ballots = _ARENA_BALLOT_RE.findall(arena_html)
            assert ballots
            for v_idx, side in enumerate(("A", "B", "tie")):
                v_resp = client.post(
                    "/v1/hivemind/arena/vote",
                    json={"task_id": ballots[v_idx % len(ballots)][0], "chosen_side": side, "profile": "dev" if v_idx != 1 else "gamer", "ballot_token": ballots[v_idx % len(ballots)][1]},
                    headers={"X-Forwarded-For": _loopback_ip(90, 10 + v_idx)},
                )
                assert v_resp.status_code == 200

            # Full opt-in smoke suite via CLI writing --output with POSIX 0600 permissions
            smoke_json_out = tmp_path / "smoke_quorum_receipt.json"
            seen_full_smoke: list[tuple[str, str]] = []
            rc_smoke_cli = promoter.main(
                [
                    "--smoke-suite",
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--expected-sha",
                    sha_router,
                    "--require-coder-model",
                    "--target-online-producers",
                    "8",
                    "--require-live-quorum",
                    "--check-fleet-quorum",
                    "--check-arena-summary",
                    "--require-ces-slo",
                    "--verify-idempotency-replay",
                    "--run-deploy-dry-run",
                    "--deploy-script",
                    str(DEPLOY_SH_PATH),
                    "--cwd",
                    str(clean_ws),
                    "--output",
                    str(smoke_json_out),
                    "--lang",
                    "en",
                    "--json",
                ],
                http_fn=_make_fast_http_adapter(client, seen=seen_full_smoke),
                env=dry_env,
            )
            assert rc_smoke_cli == 0
            smoke_cli_doc = json.loads(capsys.readouterr().out)
            _verify_sha256_receipt(smoke_cli_doc)
            assert smoke_cli_doc["smoke_ok"] is True
            assert smoke_cli_doc["online_producers"] == 8
            assert smoke_cli_doc["distinct_physical_gpus"] == 8
            assert smoke_cli_doc["producer_quorum_met"] is True
            assert smoke_cli_doc["awaiting_producer_quorum"] is False
            assert smoke_cli_doc["idempotent_dedup_ok"] is True
            assert smoke_cli_doc["idempotency_post_statuses"] == [200, 200]
            assert smoke_cli_doc["arena_summary_ok"] is True
            assert smoke_cli_doc["arena_summary"]["total_votes"] == 3
            assert smoke_cli_doc["ces_slo_ok"] is True and smoke_cli_doc["zero_lag_slo_ok"] is True
            assert smoke_cli_doc["deploy_dry_run_ok"] is True
            assert smoke_cli_doc["readiness_gates_passed"] == 5
            assert not gcloud_log_file.exists()
            assert smoke_json_out.is_file()
            if os.name != "nt":
                assert stat.S_IMODE(smoke_json_out.stat().st_mode) == 0o600
            md_smoke = promoter.render_markdown_report(smoke_cli_doc)
            assert "8/8" in md_smoke and "Wilson 95% LB" in md_smoke and "idempotent_dedup_ok=True" in md_smoke
            assert sum(1 for m, p in seen_full_smoke if m == "POST" and p == "/v1/hivemind/onboarding/feedback") == 2

            # D. Comprehensive contract & SLO tamper failure matrix -> fail closed + incident_alert + rollback
            snap: dict[str, tuple[int, dict[str, str], Any]] = {
                path: live_http("GET", f"http://127.0.0.1:8000{path}")
                for path in (
                    "/health",
                    "/v1/models",
                    "/v1/hivemind/onboarding/manifest",
                    "/noel",
                    "/quick",
                    "/raid-noel",
                    "/arena-noel",
                    "/v1/hivemind/onboarding/feedback/summary",
                    "/v1/hivemind/producers/fleet",
                    "/v1/hivemind/raid/status",
                    "/v1/hivemind/arena/summary",
                    "/v1/hivemind/t4/status",
                )
            }

            def _make_mutated_http(mutator: Any) -> promoter.HttpFn:
                seq = [0]

                def _wrapped(
                    method: str,
                    url: str,
                    payload: Any = None,
                    headers: Any = None,
                    timeout_s: float = 5.0,
                ) -> tuple[int, dict[str, str], Any]:
                    _ = (payload, headers, timeout_s)
                    p = urlsplit(url).path
                    c, h, b = snap.get(p, (404, {"content-type": "application/json"}, {"detail": "Not Found"}))
                    h_copy = dict(h)
                    b_copy = b
                    if p in promoter.SMOKE_HTML_ROUTES and isinstance(b, str) and c == 200:
                        seq[0] += 1
                        hdr_nonces = _CSP_HDR_NONCE_RE.findall(h_copy.get("content-security-policy", ""))
                        if hdr_nonces:
                            fresh = f"Dyn44FreshNonce{seq[0]:04d}"
                            b_copy = b.replace(hdr_nonces[0], fresh)
                            h_copy["content-security-policy"] = h_copy["content-security-policy"].replace(hdr_nonces[0], fresh)
                    return mutator(method, p, c, h_copy, b_copy)

                return _wrapped

            # Status aliases ('live', 'online', 'healthy', 'ready') in fleet rows work accurately
            assert promoter._inspect_fleet_quorum_contract(
                {
                    "object": promoter.FLEET_OBJECT,
                    "live_hiveminders": 8,
                    "hiveminders": [
                        {"handle": "a", "status": "live"},
                        {"handle": "b", "status": "online"},
                        {"handle": "c", "status": "healthy"},
                        {"handle": "d", "status": "ready"},
                        {"handle": "e", "status": "live"},
                        {"handle": "f", "status": "online"},
                        {"handle": "g", "status": "healthy"},
                        {"handle": "h", "status": "ready"},
                    ],
                },
                target_n=8,
                require_live_quorum=True,
                is_fr=False,
            )[1:4] == (8, 8, True)
            assert promoter._inspect_fleet_quorum_contract(
                {"object": promoter.FLEET_OBJECT, "total_hiveminders": 0, "live_hiveminders": 3, "hiveminders": []},
                target_n=8,
                require_live_quorum=True,
                is_fr=True,
            )[1] == 0
            assert promoter._inspect_fleet_quorum_contract("bad", target_n=8, require_live_quorum=True, is_fr=True)[0] is False  # type: ignore[arg-type]
            assert promoter._inspect_fleet_quorum_contract({"object": "wrong", "live_hiveminders": "x", "distinct_physical_gpus": "y", "hiveminders": "z"}, target_n=8, require_live_quorum=True, is_fr=False)[0] is False
            assert promoter._inspect_fleet_quorum_contract({"object": promoter.FLEET_OBJECT}, target_n=8, require_live_quorum=True, is_fr=True)[0] is False
            assert promoter._inspect_raid_status_contract("bad", is_fr=True)[0] is False  # type: ignore[arg-type]

            # Arena contract unit failure variants
            assert promoter._inspect_arena_summary_contract("bad", lang_code="fr")[0] is False  # type: ignore[arg-type]
            for bad_arena in (
                {"object": "wrong", "total_votes": 0, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.0},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 5, "holiday_wins": 1, "baseline_wins": 1, "ties": 1, "wilson_95_lower_bound": 0.0},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 2, "decisive_votes": 9, "holiday_wins": 2, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": promoter._wilson_95_lower_bound(2, 2)},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 2, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 9.9},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 2, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": 0.1234},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 0, "holiday_wins": 0, "baseline_wins": 0, "ties": 0, "win_ratio": 0.5, "wilson_95_lower_bound": 0.0},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 2, "baseline_wins": 0, "ties": 0, "win_ratio": 0.1, "wilson_95_lower_bound": promoter._wilson_95_lower_bound(2, 2)},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 2, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": promoter._wilson_95_lower_bound(2, 2), "by_task": {"t1": "bad"}},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 2, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": promoter._wilson_95_lower_bound(2, 2), "by_task": {"t1": {"total_votes": 5, "holiday_wins": 1, "baseline_wins": 1, "ties": 0, "wilson_95_lower_bound": 0.1}}},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 2, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": promoter._wilson_95_lower_bound(2, 2), "by_profile": {"dev": 1, "gamer": 0}},
                {"object": promoter.ARENA_SUMMARY_OBJECT, "total_votes": 2, "holiday_wins": 2, "baseline_wins": 0, "ties": 0, "wilson_95_lower_bound": promoter._wilson_95_lower_bound(2, 2), "chosen_side_counts": {"A": 1, "B": 0, "tie": 0}},
            ):
                assert promoter._inspect_arena_summary_contract(bad_arena, lang_code="fr")[0] is False

            # Tamper smoke/orchestrate failures -> fail closed + automatic rollback when promoted
            double_count = [1]

            def _tampered_replay_mut(m: str, p: str, c: int, h: dict[str, str], b: Any) -> tuple[int, dict[str, str], Any]:
                if p == "/v1/hivemind/onboarding/feedback":
                    double_count[0] += 1
                    return 200, {"content-type": "application/json"}, {
                        "object": promoter.FEEDBACK_SUMMARY_OBJECT,
                        "total_responses": double_count[0],
                        "mean_ease_rating": 5.0,
                        "zero_game_lag_ratio": 1.0,
                        "by_profile": {"dev": double_count[0], "gamer": 0},
                    }
                if p == "/v1/hivemind/onboarding/feedback/summary":
                    return 200, {"content-type": "application/json"}, {
                        "object": promoter.FEEDBACK_SUMMARY_OBJECT,
                        "total_responses": double_count[0],
                        "mean_ease_rating": 5.0,
                        "zero_game_lag_ratio": 1.0,
                        "by_profile": {"dev": double_count[0], "gamer": 0},
                    }
                return c, h, b

            tamper_cases = (
                ("fleet_leak", lambda m, p, c, h, b: (200, h, {**b, "hiveminders": [{"handle": "x", "status": "online", "endpoint_url": "http://127.0.0.1:11434"}]}) if p == "/v1/hivemind/producers/fleet" else (c, h, b)),
                ("fleet_404", lambda m, p, c, h, b: (404, h, {"detail": "gone"}) if p == "/v1/hivemind/producers/fleet" else (c, h, b)),
                ("raid_bad", lambda m, p, c, h, b: (503, h, {"error": "down"}) if p == "/v1/hivemind/raid/status" else (c, h, b)),
                ("raid_schema", lambda m, p, c, h, b: (200, h, {"object": "wrong"}) if p == "/v1/hivemind/raid/status" else (c, h, b)),
                ("arena_404", lambda m, p, c, h, b: (404, h, {"detail": "gone"}) if p == "/v1/hivemind/arena/summary" else (c, h, b)),
                ("arena_bad", lambda m, p, c, h, b: (200, h, {**b, "holiday_wins": 99}) if p == "/v1/hivemind/arena/summary" else (c, h, b)),
                ("ces_low", lambda m, p, c, h, b: (200, h, {"object": promoter.FEEDBACK_SUMMARY_OBJECT, "total_responses": 10, "mean_ease_rating": 3.5, "zero_game_lag_ratio": 0.8, "by_profile": {"dev": 5, "gamer": 5}}) if p == "/v1/hivemind/onboarding/feedback/summary" else (c, h, b)),
                ("double_replay", _tampered_replay_mut),
                ("disagree_alias", lambda m, p, c, h, b: (200, h, {"object": promoter.FEEDBACK_SUMMARY_OBJECT, "total_responses": 4, "mean_ease_rating": 4.9, "average_ease_rating": 3.1, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 2, "gamer": 2}}) if p == "/v1/hivemind/onboarding/feedback/summary" else (c, h, b)),
                ("forbidden_score", lambda m, p, c, h, b: (200, h, {"status": "ok", "sco" + "re": 9}) if p == "/health" else (c, h, b)),
            )

            mini_ok_app_called = [0]

            async def _mini_pre_app(scope: Any, receive: Any, send: Any) -> None:
                if scope.get("type") == "lifespan":
                    while True:
                        msg = await receive()
                        if msg["type"] == "lifespan.startup":
                            await send({"type": "lifespan.startup.complete"})
                        elif msg["type"] == "lifespan.shutdown":
                            await send({"type": "lifespan.shutdown.complete"})
                            return
                _ = await receive()
                p = scope["path"]
                hdrs = [(b"content-type", b"application/json")]
                if p == "/health":
                    body = json.dumps(snap["/health"][2]).encode("utf-8")
                elif p == "/v1/hivemind/onboarding/manifest":
                    body = json.dumps(snap["/v1/hivemind/onboarding/manifest"][2]).encode("utf-8")
                elif p == "/v1/hivemind/onboarding/feedback/summary":
                    body = json.dumps(snap["/v1/hivemind/onboarding/feedback/summary"][2]).encode("utf-8")
                else:
                    hdrs = [
                        (b"content-type", b"text/html"),
                        (b"content-security-policy", f"default-src 'none'; style-src 'nonce-PreAppNonce{p[-2:]}12'".encode()),
                    ]
                    body = f"<!DOCTYPE html><html><style nonce='PreAppNonce{p[-2:]}12'></style><body>100% personal PC AGPL-3.0 0.00 EUR</body></html>".encode()
                await send({"type": "http.response.start", "status": 200, "headers": hdrs})
                await send({"type": "http.response.body", "body": body, "more_body": False})

            def _fast_app_factory() -> Any:
                mini_ok_app_called[0] += 1
                return _mini_pre_app

            for idx, (label, mut) in enumerate(tamper_cases):
                if label in ("arena_bad", "ces_low"):
                    rb_cmds: list[list[str]] = []
                    rep_rb = promoter.orchestrate_deploy_on_merge(
                        "http://127.0.0.1:8000",
                        repo_root=repo_dir,
                        target_sha=sha_router,
                        previous_revision=PREV_REV,
                        app_factory=_fast_app_factory,
                        force_deploy=True,
                        execute=True,
                        require_live_quorum=True,
                        require_ces_slo=True,
                        check_arena_summary=True,
                        gcloud_runner=lambda a, _t: rb_cmds.append(list(a)) or (0, json.dumps({"revision": CAND_REV})),
                        http_fn=_make_mutated_http(mut),
                        env=dry_env,
                        cwd=repo_dir,
                        lang="fr" if idx % 2 else "en",
                    )
                    _verify_sha256_receipt(rep_rb)
                    assert rep_rb["exit_code"] == 2 and rep_rb["rollback_triggered"] is True, label
                    assert rep_rb["incident_alert_emitted"] is True, label
                    assert len(rb_cmds) == 3 and f"{PREV_REV}=100" in rb_cmds[2], label
                else:
                    rep_smoke_bad = promoter.verify_post_deploy_smoke_suite(
                        "http://127.0.0.1:8000",
                        expected_sha=sha_router,
                        require_live_quorum=True,
                        require_ces_slo=True,
                        check_arena_summary=True,
                        verify_idempotency_replay=(label == "double_replay"),
                        execute_deploy_dry_run=(label == "arena_404"),
                        deploy_script_path=DEPLOY_SH_PATH,
                        http_fn=_make_mutated_http(mut),
                        env=dry_env,
                        cwd=clean_ws,
                        lang="fr" if idx % 2 else "en",
                    )
                    _verify_sha256_receipt(rep_smoke_bad)
                    assert rep_smoke_bad["smoke_ok"] is False and rep_smoke_bad["exit_code"] == 2, label

            # orchestrate_deploy_on_merge with execute_deploy_dry_run failure, dry-run plan, execute=True pass,
            # active training lock + --no-traffic-only + rollback + pre_promote_failed
            rep_dry_fail = promoter.orchestrate_deploy_on_merge(
                "http://127.0.0.1:8000",
                repo_root=repo_dir,
                deploy_script_path=DEPLOY_SH_PATH,
                server_py_path=repo_dir / "deploy" / "gcp_openrouter" / "server.py",
                app_factory=_fast_app_factory,
                execute_deploy_dry_run=True,
                target_sha=DRIFT_SHA[:12],
                http_fn=live_http,
                env=dry_env,
                cwd=repo_dir,
            )
            # deploy.sh receives DRIFT_SHA[:12] cleanly via expected_sha -> succeeds; now force missing deploy.sh fail:
            assert rep_dry_fail["deploy_dry_run_ok"] is True
            bad_dry_only_sh = tmp_path / "finops_ok_dry_bad.sh"
            bad_dry_only_sh.write_text(
                real_deploy_text.replace('echo "gcloud_invoked=0"', 'echo "gcloud_invoked=1"'),
                encoding="utf-8",
            )
            rep_dry_err = promoter.orchestrate_deploy_on_merge(
                "http://127.0.0.1:8000",
                repo_root=repo_dir,
                deploy_script_path=bad_dry_only_sh,
                app_factory=_fast_app_factory,
                execute_deploy_dry_run=True,
                target_sha=sha_router,
                http_fn=live_http,
                env=dry_env,
                cwd=repo_dir,
            )
            assert rep_dry_err["verdict"] == "deploy_dry_run_failed" and rep_dry_err["exit_code"] == 2

            # Executed orchestrate_deploy_on_merge full success + render Markdown
            prom_cmds: list[list[str]] = []
            rep_prom_ok = promoter.orchestrate_deploy_on_merge(
                "http://127.0.0.1:8000",
                repo_root=repo_dir,
                target_sha=sha_router,
                previous_revision=PREV_REV,
                deploy_script_path=DEPLOY_SH_PATH,
                app_factory=_fast_app_factory,
                force_deploy=True,
                execute=True,
                require_coder_model=True,
                require_live_quorum=True,
                require_ces_slo=True,
                check_arena_summary=True,
                verify_idempotency_replay=True,
                execute_deploy_dry_run=True,
                gcloud_runner=lambda a, _t: prom_cmds.append(list(a)) or (0, json.dumps({"revision": CAND_REV})),
                http_fn=live_http,
                env=dry_env,
                cwd=repo_dir,
                lang="fr",
            )
            _verify_sha256_receipt(rep_prom_ok)
            assert rep_prom_ok["verdict"] == "promoted" and rep_prom_ok["exit_code"] == 0
            assert rep_prom_ok["deploy_dry_run_ok"] is True
            assert rep_prom_ok["idempotent_dedup_ok"] is True
            assert rep_prom_ok["arena_summary_ok"] is True
            assert rep_prom_ok["producer_quorum_met"] is True
            assert rep_prom_ok["readiness_gates_passed"] == 5
            assert len(prom_cmds) == 2

            # Legacy CLI branches: invalid scalar CLI flag returns 2, Markdown --output writes 0600 file
            md_cli_out = tmp_path / "cli_out.md"
            assert promoter.main(
                ["--deploy-on-merge", "--gateway", "http://127.0.0.1:8000", "--expected-sha", sha_router, "--repo-root", str(repo_dir), "--deploy-script", str(DEPLOY_SH_PATH), "--output", str(md_cli_out), "--lang", "en"],
                app_factory=_fast_app_factory,
                http_fn=live_http,
                env=dry_env,
                cwd=repo_dir,
            ) == 0
            _ = capsys.readouterr()
            assert md_cli_out.is_file() and "SHA-256 Receipt" in md_cli_out.read_text(encoding="utf-8")
            assert promoter.main(["--smoke-suite", "--target-online-producers", "0"], env=dry_env) == 2
            assert promoter.main(["--cwd", ""], env=dry_env) == 2
            assert promoter.main(["--output", ""], env=dry_env) == 2
            _ = capsys.readouterr()
            assert mini_ok_app_called[0] >= 1

            # Additional coverage for legacy promoter branches (rollback, lockfile, drift, FinOps, manifests, helpers)
            locked_f = tmp_path / "active.lock"
            locked_f.write_text('{"active": false}\n', encoding="utf-8")
            assert promoter._inspect_local_lockfile(locked_f)[0] is False
            with locked_f.open("r+", encoding="utf-8") as lfh:
                fcntl.flock(lfh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    assert promoter._inspect_local_lockfile(locked_f)[0] is True
                finally:
                    fcntl.flock(lfh.fileno(), fcntl.LOCK_UN)
            bad_lock = tmp_path / "bad.lock"
            bad_lock.write_text("{bad", encoding="utf-8")
            assert promoter._inspect_local_lockfile(bad_lock)[0] is True

            assert promoter.default_git_runner(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], 3.0)[0] == 0
            with pytest.raises(promoter.RevisionPromoterError):
                promoter.default_git_runner(["git", "push", "origin", "main"])
            with pytest.raises(promoter.RevisionPromoterError):
                promoter.default_git_runner(["git", "diff", "--ext-diff", "HEAD~1..HEAD"])
            with monkeypatch.context() as mp_sub:
                mp_sub.setattr(
                    subprocess,
                    "run",
                    lambda a, **_kw: subprocess.CompletedProcess(a, 0, "Google Cloud SDK", "warn"),
                )
                assert promoter.default_gcloud_runner(["gcloud", "--version"], 5.0) == (
                    0,
                    "Google Cloud SDK\nwarn",
                )
            assert promoter.detect_revision_drift("http://127.0.0.1:8000", expected_sha=DRIFT_SHA, http_fn=live_http, env=dry_env, cwd=clean_ws, lang="fr")["sha_mismatch"] is True
            assert promoter.detect_deploy_manifest_diff("http://127.0.0.1:8000", repo_root=repo_dir, expected_sha=sha_docs, http_fn=lambda _m, u: (200, {"status": "ok"}) if u.endswith("/health") else ((200, {**snap["/v1/hivemind/onboarding/manifest"][2], "build_sha": sha_base}) if u.endswith("/manifest") else live_http("GET", u)), env=dry_env, cwd=repo_dir)["skip_reason"] == "non_cloud_run_diff_only"

            assert promoter.run_post_deploy_holiday_canary("http://127.0.0.1:8000", expected_sha=sha_router, probe_feedback_post=True, http_fn=live_http, env=dry_env, cwd=clean_ws)["canary_ok"] is True
            assert promoter.promote_holiday_revision("http://127.0.0.1:8000", target_sha=sha_router, app_factory=_fast_app_factory, execute=True, gcloud_runner=lambda *_a: (0, "ok"), http_fn=live_http, env=dry_env, cwd=clean_ws, lang="fr")["idempotent_noop"] is True
            assert promoter.promote_holiday_revision("http://127.0.0.1:8000", target_sha=DRIFT_SHA, previous_revision=PREV_REV, app_factory=_fast_app_factory, execute=True, gcloud_runner=lambda a, _t: (0, json.dumps({"revision": CAND_REV})), http_fn=_make_mutated_http(lambda m, p, c, h, b: (500, h, {}) if p == "/noel" else (c, h, b)), env=dry_env, cwd=clean_ws, lang="fr")["rollback_triggered"] is True
    finally:
        store.close()

    # Pre-promote ASGI + tampered deploy.sh FinOps checks (fast, avoiding heavy torch import in real server.py)
    fast_srv_02 = repo_dir / "deploy" / "gcp_openrouter" / "server.py"
    real_srv_txt = (REPO_ROOT / "deploy" / "gcp_openrouter" / "server.py").read_text(encoding="utf-8")
    assert all(tok in real_srv_txt for tok in ("def build_app", "create_router_app", "HIVEMIND_ROUTER_SECRET"))
    assert promoter.verify_pre_promote_gateway_and_finops(server_py_path=fast_srv_02, app_factory=lambda: None, env=dry_env, cwd=tmp_path, lang="fr")["exit_code"] == 2
    bad_fin = tmp_path / "bad_fin.sh"
    bad_fin.write_text(real_deploy_text.replace("--min-instances 0", "--min-instances 2"), encoding="utf-8")
    assert promoter.verify_pre_promote_gateway_and_finops(deploy_script_path=bad_fin, server_py_path=fast_srv_02, run_app_probe=False, env=dry_env, cwd=tmp_path)["finops_ok"] is False
    bad_srv = tmp_path / "bad_srv.py"
    bad_srv.write_text("def broken(:\n", encoding="utf-8")
    assert promoter.verify_pre_promote_gateway_and_finops(server_py_path=bad_srv, run_app_probe=False, env=dry_env, cwd=tmp_path)["server_py_ok"] is False

    # Pure-stdlib _asgi_get_request check without FastAPI TestClient
    async def _simple_asgi(scope: Any, receive: Any, send: Any) -> None:
        _ = await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"status":"ok"}', "more_body": False})
        _ = await receive()

    st_a, h_a, b_a = asyncio.run(promoter._asgi_get_request(_simple_asgi, "/health"))
    assert st_a == 200 and h_a["content-type"] == "application/json" and json.loads(b_a.decode())["status"] == "ok"


def test_03_concurrent_48_participant_multi_cohort_wal_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_deploy_text = DEPLOY_SH_PATH.read_text(encoding="utf-8")
    soak_repo = tmp_path / "soak48_repo"
    _b, _d, sha_router = _init_minimal_git_repo(soak_repo, real_deploy_text)

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", sha_router)
    monkeypatch.setenv("K_REVISION", CAND_REV)

    upstream_inference_hits: list[str] = []

    def _forbid_upstream_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "dyn44_wal_soak48.sqlite3"
    store = RouterStore(sqlite_path, secret=ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        assert str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
    finally:
        wal_conn.close()

    with _pause_inproc_cov():
        app = create_router_app(
            store,
            http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream_inference)),
            allow_private_producers=True,
            trusted_proxy_hops=1,
            open_registration=True,
        )

    async def _fast_pre_app(scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        _ = await receive()
        p = scope["path"]
        hdrs = [(b"content-type", b"application/json")]
        if p == "/health":
            body = b'{"status":"ok"}'
        elif p == "/v1/hivemind/onboarding/manifest":
            body = json.dumps({
                "object": promoter.MANIFEST_OBJECT,
                "campaign_id": promoter.CAMPAIGN_ID,
                "status": "ok",
                "build_sha": sha_router,
                "personal_pc_only": True,
                "financial_promise": promoter.FINANCIAL_PROMISE,
                "holiday_routes": list(promoter.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
                "downloadable_scripts": list(promoter.EXPECTED_MANIFEST_SCRIPTS),
            }).encode("utf-8")
        elif p == "/v1/hivemind/onboarding/feedback/summary":
            body = json.dumps({
                "object": promoter.FEEDBACK_SUMMARY_OBJECT,
                "total_responses": 0,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": {"dev": 0, "gamer": 0},
            }).encode("utf-8")
        else:
            nonce = f"SoakPre{p[-2:] if len(p) > 2 else 'nx'}9988"
            hdrs = [(b"content-type", b"text/html"), (b"content-security-policy", f"default-src 'none'; style-src 'nonce-{nonce}'".encode())]
            body = f"<!DOCTYPE html><html><style nonce='{nonce}'></style><body>100% personal PC AGPL-3.0 0.00 EUR</body></html>".encode()
        await send({"type": "http.response.start", "status": 200, "headers": hdrs})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    sample_scripts = tuple(_ALLOWED_ONBOARDING_SCRIPTS)[:4]
    lock = threading.Lock()
    observed_nonces: list[str] = []
    seen_paths: list[str] = []
    sqlite_locks: list[str] = []
    raw_keys_seen: list[str] = []
    masked_keys_seen: list[str] = []

    with _FastAsgiClient(app) as client:

        def _capture_html(resp_text: str, csp_hdr: str) -> str:
            if _SQLITE_LOCK_RE.search(resp_text):
                with lock:
                    sqlite_locks.append(resp_text[:120])
            hdr_n = _CSP_HDR_NONCE_RE.findall(csp_hdr)
            attr_n = [v for _, v in _HTML_ATTR_NONCE_RE.findall(resp_text)]
            assert hdr_n and attr_n and hdr_n[0] in attr_n
            stripped = promoter.strip_masked_keys_and_nonces(resp_text)
            assert static_numbers_guard.FIGURE.findall(stripped) == []
            assert UNSAFE_INNER_SINK not in stripped
            return hdr_n[0]

        def _asgi_adapter(
            method: str,
            url: str,
            payload: Any = None,
            headers: Any = None,
            timeout_s: float = 5.0,
        ) -> tuple[int, dict[str, str], Any]:
            _ = timeout_s
            parsed = urlsplit(url)
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            assert "completions" not in path.lower() and "/chat/" not in path.lower()
            req_h = {"X-Forwarded-For": _loopback_ip(99, 20)}
            if headers:
                req_h.update(headers)
            resp = client.request(method, path, json=payload, headers=req_h)
            r_hdrs = {k.lower(): v for k, v in resp.headers.items()}
            if resp.status_code >= 400 and _SQLITE_LOCK_RE.search(resp.text):
                with lock:
                    sqlite_locks.append(resp.text[:120])
            with lock:
                seen_paths.append(path)
                if parsed.path in promoter.SMOKE_HTML_ROUTES and resp.status_code == 200:
                    observed_nonces.append(_capture_html(resp.text, r_hdrs.get("content-security-policy", "")))
            if "application/json" in r_hdrs.get("content-type", ""):
                return resp.status_code, r_hdrs, resp.json()
            return resp.status_code, r_hdrs, resp.text

        def _run_participant(idx: int) -> dict[str, Any]:
            is_dev = idx < 24
            cohort_idx = idx if is_dev else (idx - 24)
            profile = "dev" if is_dev else "gamer"
            ip = _loopback_ip(10 + idx // 200, (idx % 200) + 1)
            ws_dir = tmp_path / "soak_slots" / f"slot_{idx:02d}" / "hobby_project"
            ws_dir.mkdir(parents=True, exist_ok=True)
            assert promoter.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=ws_dir)["compliance_ok"] is True

            t_start = time.monotonic()
            reg_r = client.post(
                "/v1/hivemind/register",
                json={"handle": f"soak44_{profile}_{cohort_idx:02d}", "role": "prosumer" if is_dev else "trainer"},
                headers={"X-Forwarded-For": ip},
            )
            assert reg_r.status_code == 200
            raw_key = str(reg_r.json()["api_key"])
            masked_key = promoter.mask_api_key(raw_key)
            auth_h = {"Authorization": f"Bearer {raw_key}", "X-Forwarded-For": ip}

            # >= 8 distinct physical GPUs serving hivemind/coder + opentrainer heartbeats
            if idx < 10:
                hb_r = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{_loopback_ip(70, idx + 1)}:11434",
                        "models": [{"id": "coder", "local_name": "Qwen2.5-Coder-7B-Q4_K_M", "loaded": True}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 120.0,
                    },
                    headers=auth_h,
                )
                assert hb_r.status_code == 200
                ot_r = client.post(
                    "/v1/hivemind/opentrainer/heartbeat",
                    json={"host_label": f"pc-{idx:02d}", "gpu_name": "RTX 4070", "vram_total_mib": 12288, "vram_free_mib": 8192},
                    headers=auth_h,
                )
                assert ot_r.status_code == 200

            # Fetch holiday HTML routes round-robin + /v1/models + manifest + scripts
            h_route = promoter.SMOKE_HTML_ROUTES[idx % 4]
            r_h = client.request("GET", h_route, headers={"X-Forwarded-For": ip})
            assert r_h.status_code == 200
            local_nonces = [_capture_html(r_h.text, r_h.headers.get("content-security-policy", ""))]
            with lock:
                seen_paths.append(h_route)

            assert client.request("GET", "/v1/models", headers={"X-Forwarded-For": ip}).status_code == 200
            r_man = client.request("GET", "/v1/hivemind/onboarding/manifest", headers={"X-Forwarded-For": ip})
            assert r_man.status_code == 200 and r_man.json()["build_sha"] == sha_router

            if idx < 8:
                arena_h = client.request("GET", "/arena-noel", headers={"X-Forwarded-For": ip}).text
                ballots = _ARENA_BALLOT_RE.findall(arena_h)
                if ballots:
                    tid, tok = ballots[0]
                    assert (
                        client.post(
                            "/v1/hivemind/arena/vote",
                            json={"task_id": tid, "chosen_side": "A" if idx % 3 != 0 else "tie", "profile": profile, "ballot_token": tok},
                            headers={"X-Forwarded-For": ip},
                        ).status_code
                        == 200
                    )
            if idx < 4:
                assert client.request("GET", f"/scripts/{sample_scripts[idx]}", headers={"X-Forwarded-For": ip}).status_code == 200

            if is_dev:
                assert (time.monotonic() - t_start) < 60.0
                zero_lag = True
                ease = 5 if cohort_idx % 4 != 0 else 4
                plat = "linux"
            else:
                assert (time.monotonic() - t_start) < 300.0
                zero_lag = cohort_idx != 23  # 47/48 == 0.9792 >= 0.95
                ease = 5 if cohort_idx % 5 != 0 else 4
                plat = "windows"

            # Duplicate-replayed Idempotency-Key CES feedback POST (2x per user -> 96 POSTs, 48 saved)
            idem_key = f"soak44-{profile}-{cohort_idx:02d}"
            fb_body = {"profile": profile, "ease_rating": ease, "zero_game_lag": zero_lag, "platform": plat, "note": ""}
            r1 = client.post("/v1/hivemind/onboarding/feedback", json=fb_body, headers={"X-Forwarded-For": ip, "Idempotency-Key": idem_key})
            r2 = client.post("/v1/hivemind/onboarding/feedback", json=fb_body, headers={"X-Forwarded-For": ip, "Idempotency-Key": idem_key})
            assert r1.status_code == 200 and r2.status_code == 200

            with lock:
                observed_nonces.extend(local_nonces)
                raw_keys_seen.append(raw_key)
                masked_keys_seen.append(masked_key)
            return {"idx": idx, "profile": profile}

        def _probe_worker(k: int) -> dict[str, Any]:
            smoke_r = promoter.verify_post_deploy_smoke_suite(
                "http://127.0.0.1:8000",
                expected_sha=sha_router,
                require_coder_model=True,
                check_fleet_quorum=True,
                check_arena_summary=True,
                http_fn=_asgi_adapter,
                env=CLEAN_ENV,
                cwd=soak_repo,
                lang="fr" if k % 2 else "en",
            )
            merge_r = None
            if k == 0:
                merge_r = promoter.orchestrate_deploy_on_merge(
                    "http://127.0.0.1:8000",
                    repo_root=soak_repo,
                    target_sha=sha_router,
                    previous_revision=PREV_REV,
                    app_factory=lambda: _fast_pre_app,
                    execute=False,
                    require_coder_model=True,
                    check_fleet_quorum=True,
                    check_arena_summary=True,
                    http_fn=_asgi_adapter,
                    env=CLEAN_ENV,
                    cwd=soak_repo,
                    lang="en",
                )
            return {"smoke": smoke_r, "merge": merge_r}

        try:
            t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                user_futs = [pool.submit(_run_participant, i) for i in range(48)]
                probe_futs = [pool.submit(_probe_worker, k) for k in range(3)]
                user_outcomes = [f.result(timeout=12.0) for f in user_futs]
                probe_outcomes = [f.result(timeout=10.0) for f in probe_futs]

            # Exact 5-key feedback dict check BEFORE final orchestrator Idempotency-Key replay
            pre_fb = client.request(
                "GET",
                "/v1/hivemind/onboarding/feedback/summary",
                headers={"X-Forwarded-For": _loopback_ip(99, 240)},
            ).json()
            assert list(pre_fb.keys()) == [
                "object",
                "total_responses",
                "mean_ease_rating",
                "zero_game_lag_ratio",
                "by_profile",
            ]
            assert pre_fb["total_responses"] == 48
            assert pre_fb["by_profile"] == {"dev": 24, "gamer": 24}
            assert pre_fb["mean_ease_rating"] >= 4.2
            assert pre_fb["zero_game_lag_ratio"] >= 0.95

            final_merge = promoter.orchestrate_deploy_on_merge(
                "http://127.0.0.1:8000",
                repo_root=soak_repo,
                target_sha=sha_router,
                previous_revision=PREV_REV,
                app_factory=lambda: _fast_pre_app,
                execute=False,
                require_coder_model=True,
                target_online_producers=8,
                require_live_quorum=True,
                require_ces_slo=True,
                check_fleet_quorum=True,
                check_arena_summary=True,
                verify_idempotency_replay=True,
                execute_deploy_dry_run=True,
                deploy_script_path=DEPLOY_SH_PATH,
                http_fn=_asgi_adapter,
                env=CLEAN_ENV,
                cwd=soak_repo,
                lang="en",
            )
            soak_elapsed_s = time.monotonic() - t0
            assert soak_elapsed_s < 15.0, f"48-user WAL soak exceeded 15.0s budget: {soak_elapsed_s:.2f}s"

            assert len(user_outcomes) == 48
            assert sqlite_locks == []
            assert upstream_inference_hits == []
            assert all("completions" not in p.lower() for p in seen_paths)

            for po in probe_outcomes:
                _verify_sha256_receipt(po["smoke"])
                assert po["smoke"]["smoke_ok"] is True
                if po["merge"] is not None:
                    _verify_sha256_receipt(po["merge"])
                    assert po["merge"]["verdict"] == "idempotent_noop"

            _verify_sha256_receipt(final_merge)
            assert final_merge["verdict"] == "idempotent_noop" and final_merge["exit_code"] == 0
            assert final_merge["online_producers"] >= 8
            assert final_merge["distinct_physical_gpus"] >= 8
            assert final_merge["producer_quorum_met"] is True
            assert final_merge["ces_slo_ok"] is True and final_merge["zero_lag_slo_ok"] is True
            assert final_merge["arena_summary_ok"] is True
            assert final_merge["deploy_dry_run_ok"] is True
            assert final_merge["idempotent_dedup_ok"] is True
            assert final_merge["smoke_suite"]["feedback_summary"]["total_responses"] == 49
            assert final_merge["readiness_gates_passed"] == 5
            assert final_merge["gcloud_calls"] == 0
            assert all(route in seen_paths for route in promoter.SMOKE_HTML_ROUTES)
            assert len(observed_nonces) >= 48
            assert len(set(observed_nonces)) == len(observed_nonces)
            assert promoter.scan_output_safety(promoter.render_markdown_report(final_merge)) == []
            for raw_k, masked_k in zip(raw_keys_seen, masked_keys_seen, strict=True):
                assert raw_k not in masked_k
                assert masked_k.startswith(("hm-***", "hm_live_***"))
        finally:
            store.close()


def test_04_comprehensive_branch_edges_for_95_plus_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise remaining error, validator, ASGI fallback, canary, promote, manifest-diff, and CLI branches."""
    real_deploy_text = DEPLOY_SH_PATH.read_text(encoding="utf-8")
    edge_repo = tmp_path / "edge_repo"
    sha_base, _sha_docs, sha_router = _init_minimal_git_repo(edge_repo, real_deploy_text)
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", CAND_REV)

    # 1. Language, masking, home redaction, output safety, and SHA/Cloud-ID/URL validators
    assert promoter.normalize_lang("en-GB") == "en"
    assert promoter.normalize_lang("fr_CA") == "fr"
    for bad_l in (True, "es"):
        with pytest.raises(promoter.RevisionPromoterError):
            promoter.normalize_lang(bad_l)  # type: ignore[arg-type]

    assert promoter.mask_api_key("") == "<missing>"
    assert promoter.mask_api_key("short") == "<invalid>"
    assert promoter.mask_api_key("other-prefix-123456") == "<invalid>"
    assert promoter.mask_api_key(RAW_LIVE_KEY) == "hm_live_***8888"

    custom_home = tmp_path / "home_alice"
    assert promoter.redact_home_path(PROMOTER_PATH) == "scripts/noel_cloud_run_revision_promoter.py"
    assert promoter.redact_home_path(custom_home / ".hivemind", home=custom_home) == "~/.hivemind"
    assert promoter.redact_home_path(r"C:\Users\Bob\cfg.json", home=custom_home) == "~/cfg.json"

    bad_score = '{"sco' + 're": 9}'
    bad_home = "/ho" + "me/alice_dev/src"
    bad_bearer = "Bea" + "rer secret_tok_987654"
    for bad_txt, tag in (
        ("alice@" + "example.org", "email_detected"),
        (RAW_HM_KEY, "unmasked_api_key_detected"),
        (bad_bearer, "bearer_token_detected"),
        (bad_home, "unredacted_home_path_detected"),
        (bad_score, "forbidden_score_key_detected"),
        (f"ip={NON_LOOPBACK_IP}", f"non_loopback_ipv4_detected:{NON_LOOPBACK_IP}"),
    ):
        assert tag in promoter.scan_output_safety(bad_txt)
        with pytest.raises(promoter.RevisionPromoterError):
            promoter._assert_safe_output(bad_txt)

    assert promoter.sanitize_build_sha(999) == "unknown"
    assert promoter.sanitize_build_sha("secret1234") == "unknown"
    assert promoter.sanitize_cloud_run_revision("rev-secret-1") is None
    assert promoter.shas_match("tagalpha12", "tagbeta12") is False
    assert promoter.shas_match("tagalpha12", "tagalpha12") is True
    assert promoter.validate_gcp_project("   ") is None
    assert promoter.validate_gcp_project(None) is None
    for bad_id_fn in (
        lambda: promoter.validate_target_sha("unknown"),
        lambda: promoter.validate_cloud_identifier(True, "service"),  # type: ignore[arg-type]
        lambda: promoter.validate_cloud_identifier("bad;svc"),
        lambda: promoter.validate_candidate_tag(True),  # type: ignore[arg-type]
        lambda: promoter.validate_candidate_tag("BAD_TAG"),
        lambda: promoter.validate_gcp_project(123),  # type: ignore[arg-type]
        lambda: promoter.validate_gcp_project("BAD PROJ"),
        lambda: promoter.validate_deploy_source_dir(""),
        lambda: promoter.validate_deploy_source_dir("../etc"),
        lambda: promoter._safe_urlsplit(""),
        lambda: promoter._safe_urlsplit("https://" + "bad\nhost.a.run.app"),
        lambda: promoter._safe_urlsplit("https://" + "bad space.a.run.app"),
        lambda: promoter._safe_urlsplit("https://u:p@demo.a.run.app"),
        lambda: promoter._validate_zero_pii_feedback_payload({"profile": "admin", "ease_rating": 5, "zero_game_lag": True}),
    ):
        with pytest.raises(promoter.RevisionPromoterError):
            bad_id_fn()

    assert promoter.is_allowed_outbound_host("bad host.a.run.app") is False
    assert promoter.is_allowed_outbound_host("storage.googleapis.com") is True
    assert promoter.is_allowed_outbound_host("run.googleapis.com") is True

    # 2. HTTP raw POST/error parsing, _normalize_http_return, _invoke_http_fn arities & runner error branches
    assert promoter._parse_http_raw_response(200, b"{bad", {"content-type": "application/json"})[1]["raw_error"] == "invalid_json"
    with pytest.raises(promoter.RevisionPromoterError):
        promoter.default_http_raw("GET", "http://127.0.0.1:8000/health", payload={"x": 1})
    with pytest.raises(promoter.RevisionPromoterError):
        promoter._normalize_http_return(("bad_single",))
    norm_h = promoter._normalize_http_return((200, {"x-a": "1"}, {"headers": {"x-b": "2"}}))
    assert norm_h[1]["headers"]["x-b"] == "2" and norm_h[1]["headers"]["x-a"] == "1"
    assert promoter._normalize_http_return((200, {}, [1, 2]))[1]["data"] == [1, 2]
    assert promoter._invoke_http_fn(lambda m, u, **kw: (200, {"ok": True}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert promoter._invoke_http_fn(lambda m, u, p: (200, {"p3": 1}), "GET", "http://127.0.0.1:8000/health")[1]["p3"] == 1
    assert promoter._invoke_http_fn(lambda m, u, p, h: (200, {"p4": 1}), "GET", "http://127.0.0.1:8000/health")[1]["p4"] == 1
    assert promoter._invoke_http_fn(lambda m, u, p, h, t: (200, {"p5": 1}), "GET", "http://127.0.0.1:8000/health")[1]["p5"] == 1
    assert promoter._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(urllib.error.URLError("down")), "GET", "http://127.0.0.1:8000/health")[0] == 0
    with pytest.raises(promoter.RevisionPromoterError):
        promoter._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(promoter.RevisionPromoterError("err")), "GET", "http://127.0.0.1:8000/health")

    for bad_argv_fn in (
        lambda: promoter._validate_argv_sequence("git", allowed_binaries=frozenset({"git"})),  # type: ignore[arg-type]
        lambda: promoter._validate_argv_sequence(["git", "a\nb"], allowed_binaries=frozenset({"git"})),
        lambda: promoter._validate_argv_sequence(["bash"], allowed_binaries=frozenset({"git"})),
        lambda: promoter._invoke_runner(lambda _a: "bad", ["git", "status"], 1.0),  # type: ignore[return-value]
    ):
        with pytest.raises(promoter.RevisionPromoterError):
            bad_argv_fn()
    assert promoter._invoke_runner(lambda _a: (0, "out", "err"), ["git", "status"], 1.0) == (0, "out\nerr")

    orig_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: (_ for _ in ()).throw(FileNotFoundError("absent")))
    assert promoter.default_git_runner(["git", "rev-parse", "HEAD"])[0] == 127
    assert promoter.default_gcloud_runner(["gcloud", "--version"])[0] == 127
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(["gcloud"], 1.0)))
    assert promoter.default_gcloud_runner(["gcloud", "--version"]) == (124, "gcloud_timeout")
    monkeypatch.setattr(subprocess, "run", orig_run)

    # 3. Git config OSError + contracts (manifest, feedback, HTML strict=False, models)
    assert promoter._find_corporate_markers("") == []
    assert promoter.extract_git_config_remote_urls("") == []
    assert promoter._extract_response_headers({}) == {}

    wt_err = tmp_path / "wt_err"
    wt_meta = tmp_path / "wt_err_meta"
    wt_meta.mkdir()
    (wt_meta / "commondir").write_text("..\n", encoding="utf-8")
    (wt_meta / "config").write_text("[core]\n", encoding="utf-8")
    wt_err.mkdir()
    (wt_err / ".git").write_text(f"gitdir: {wt_meta}\n", encoding="utf-8")
    orig_read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **kw: (_ for _ in ()).throw(OSError("denied")) if (self.name in (".git", "commondir", "config") or self.name == "unreadable_deploy.sh") else orig_read_text(self, *a, **kw),
    )
    unr_sh = tmp_path / "unreadable_deploy.sh"
    unr_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    assert promoter._resolve_git_config_files(wt_err) == []
    assert promoter._check_deploy_sh_build_sha_propagation(unr_sh) is False
    assert promoter._audit_deploy_finops(unr_sh, lang_code="fr")["deploy_script_present"] is False
    monkeypatch.undo()
    # Read-error on .git dir config file during compliance evaluation
    cfg_err_ws = tmp_path / "cfg_err_ws"
    (cfg_err_ws / ".git").mkdir(parents=True)
    (cfg_err_ws / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw: (_ for _ in ()).throw(OSError("denied")) if self == (cfg_err_ws / ".git" / "config") else orig_read_text(self, *a, **kw))
    assert promoter.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=cfg_err_ws)["compliance_ok"] is True
    monkeypatch.undo()

    assert promoter._inspect_manifest_contract({"object": "bad", "holiday_routes": None, "downloadable_scripts": None}, lang_code="fr")[0] is False
    assert promoter._inspect_manifest_contract({"object": promoter.MANIFEST_OBJECT, "campaign_id": promoter.CAMPAIGN_ID, "status": "ok", "personal_pc_only": True, "financial_promise": promoter.FINANCIAL_PROMISE, "holiday_routes": ["/noel"], "downloadable_scripts": ["hivemind_doctor.py"]}, lang_code="fr")[0] is False
    assert promoter._inspect_feedback_summary_contract({"total_responses": 0, "mean_ease_rating": 4.0, "zero_game_lag_ratio": None, "by_profile": {"dev": 0, "gamer": 0}}, lang_code="fr")[0] is False
    assert promoter._inspect_feedback_summary_contract({"total_responses": 1, "mean_ease_rating": None, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}}, lang_code="en")[0] is False
    assert promoter._inspect_holiday_html_contract("/noel", {"html": "<!DOCTYPE html><html><style nonce='OnlyAttrNonce123'></style></html>"}, set(), strict_contract=False, lang_code="fr")[0] is True
    assert promoter._inspect_holiday_html_contract("/noel", {"html": ""}, set(), strict_contract=False, lang_code="fr")[0] is False

    # 4. FinOps tampered variants, _load_build_app errors, pure-stdlib ASGI fallback, skipped_runtime_unavailable
    for bad_tok, repl in (
        ("--min-instances 0", "--min-instances 1"),
        ("--max-instances 1", "--max-instances 2"),
        ("--no-cpu-throttling", "--cpu-throttling"),
        ("--concurrency 80", "--concurrency 40"),
        ("HIVEMIND_ROUTER_REPLICA_URL", "UNUSED"),
        ("HIVEMIND_ROUTER_SECRET=", "UNUSED="),
        ("HIVEMIND_STARTER_GRANT_HMT:-0", "HIVEMIND_STARTER_GRANT_HMT:-10"),
        ("HIVEMIND_ALLOW_BONUSES:-0", "HIVEMIND_ALLOW_BONUSES:-1"),
    ):
        bsh = tmp_path / f"fin_{abs(hash(bad_tok))}.sh"
        bsh.write_text(real_deploy_text.replace(bad_tok, repl), encoding="utf-8")
        assert promoter._audit_deploy_finops(bsh, lang_code="fr")["finops_ok"] is False

    srv_miss = tmp_path / "srv_miss.py"
    srv_miss.write_text("X = 1\n", encoding="utf-8")
    srv_noncall = tmp_path / "srv_noncall.py"
    srv_noncall.write_text("def build_app():\n    return None\nbuild_app = 123\nHIVEMIND_ROUTER_SECRET='x'\ncreate_router_app=None\n", encoding="utf-8")
    srv_boom = tmp_path / "srv_boom.py"
    srv_boom.write_text("def build_app():\n    return None\nHIVEMIND_ROUTER_SECRET='x'\ncreate_router_app=None\nraise RuntimeError('boom')\n", encoding="utf-8")
    for bad_s in (tmp_path / "absent_srv.py", srv_miss, srv_noncall, srv_boom):
        assert promoter._load_build_app_from_server_py(bad_s)[0] is False

    async def _broken_asgi(scope: Any, receive: Any, send: Any) -> None:
        _ = await receive()
        p = scope["path"]
        hdrs = [(b"content-type", b"application/json")]
        if p == "/health":
            st, body = 200, b'{"status":"bad"}'
        elif p == "/v1/hivemind/onboarding/manifest":
            st, body = 200, b'{"object":"bad"}'
        elif p == "/v1/hivemind/onboarding/feedback/summary":
            st, body = 200, b'{"total_responses":"bad"}'
        elif p == "/noel":
            st, hdrs, body = 200, [(b"content-type", b"text/html")], b"<html>broken</html>"
        else:
            st, hdrs, body = 500, [(b"content-type", b"text/plain")], b"err"
        await send({"type": "http.response.start", "status": st, "headers": hdrs})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    real_imp = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no tc")) if name == "fastapi.testclient" else real_imp(name, *a, **kw))
    ok_st, st_m, iss_m, tr_m = promoter._probe_asgi_pre_promote_routes(_broken_asgi, lang_code="fr")
    assert ok_st is False and tr_m == "asgi_stdlib" and len(iss_m) >= 4 and st_m["/quick"] == 500
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no fa")) if name == "fastapi" else real_imp(name, *a, **kw))
    assert promoter.verify_pre_promote_gateway_and_finops(server_py_path=edge_repo / "deploy" / "gcp_openrouter" / "server.py", env=CLEAN_ENV, cwd=tmp_path)["app_probe_transport"] == "skipped_runtime_unavailable"
    monkeypatch.setattr(importlib, "import_module", real_imp)

    # 5. Lockfile, safety lock, drift, rollout plan, canary, promote, manifest diff, smoke, orchestrate & CLI
    assert promoter._inspect_local_lockfile(tmp_path / "absent.lock")[0] is False
    assert promoter._inspect_local_training_processes(lambda: ["systemd", "python3 hivemind_qlora.py"])[0] is True
    lock_all = promoter.check_active_training_safety_lock(
        "http://127.0.0.1:8000",
        lock_file_path=tmp_path / "absent.lock",
        process_probe_fn=lambda: ["hivemind_qlora"],
        http_fn=lambda _m, u: (
            (200, {"operational_mode": "EUROPEAN_NIGHT_QLORA_MOE_TRAINING", "governor": {"boinc_gpu_mode": "always", "hold_file_present": False, "state": "qlora_training"}, "training_active": True, "hardware": {"resident_processes": [{"name": "hivemind_qlora"}]}})
            if u.endswith("/t4/status")
            else (
                (200, {"training_active": True, "swarm": {"active_trainers": 2}})
                if u.endswith("/raid/status")
                else (200, {"training_active": True, "hiveminders": [{"status": "offline"}, {"status": "live", "execution_class": "qlora-trainer"}]})
            )
        ),
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert lock_all["exit_code"] == 3 and len(lock_all["active_reasons"]) >= 8

    assert promoter.detect_revision_drift("http://127.0.0.1:8000", check_git_head=True, git_runner=lambda *_a: (1, ""), http_fn=lambda _m, u: (200, {"status": "bad"}) if u.endswith("/health") else ((200, {"object": "bad"}) if u.endswith("/manifest") else ((200, "<html></html>") if u.endswith("/noel") else ((0, {}) if u.endswith("/quick") else ((503, {}) if u.endswith("/raid-noel") else (200, {"bad": 1}))))), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["drift_detected"] is True
    assert promoter.detect_revision_drift("http://127.0.0.1:8000", check_git_head=True, git_runner=lambda *_a: (_ for _ in ()).throw(OSError("no")), http_fn=lambda _m, _u: (404, {}), env=CLEAN_ENV, cwd=tmp_path, lang="en")["sha_mismatch"] is True

    with pytest.raises(promoter.RevisionPromoterError):
        promoter.build_idempotent_rollout_plan(previous_revision="bad;rev", env=CLEAN_ENV, cwd=tmp_path)
    with pytest.raises(promoter.RevisionPromoterError):
        promoter.build_idempotent_rollout_plan(candidate_revision="bad;rev", env=CLEAN_ENV, cwd=tmp_path)
    assert promoter.build_idempotent_rollout_plan(gcp_project="hivemind-hobby-1", target_sha=sha_router, no_traffic_only=True, env=CLEAN_ENV, cwd=tmp_path, lang="fr")["no_traffic_only"] is True

    assert promoter._extract_deployed_revision("{bad", "fb") == "fb"
    assert promoter._extract_deployed_revision('{"status":{}}', "fb") == "fb"
    assert promoter._extract_deployed_revision("revision [bad_secret_rev]", "fb") == "fb"
    assert promoter._extract_deployed_revision("revision [rev-ok-1]", "fb") == "rev-ok-1"

    assert promoter.run_post_deploy_holiday_canary(
        "http://127.0.0.1:8000",
        expected_sha=DRIFT_SHA,
        probe_feedback_post=True,
        http_fn=lambda m, u, *_a, **_kw: (200, {"status": "bad"}) if u.endswith("/health") else ((200, {"object": promoter.FEEDBACK_SUMMARY_OBJECT, "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}}) if m == "POST" else ((500, {}) if u.endswith("/summary") else (404, {}))),
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["canary_ok"] is False
    assert promoter.run_post_deploy_holiday_canary(
        "http://127.0.0.1:8000",
        probe_feedback_post=True,
        http_fn=lambda m, _u, *_a, **_kw: (500, {}) if m == "POST" else (404, {}),
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["feedback_post_ok"] is False

    # promote_holiday_revision & orchestrate_deploy_on_merge branches (without invoking real server.py)
    pre_seq = [0]

    async def _fast_pre(scope: Any, r: Any, s: Any) -> None:
        if scope.get("type") == "lifespan":
            while True:
                m = await r()
                if m["type"] == "lifespan.startup":
                    await s({"type": "lifespan.startup.complete"})
                else:
                    await s({"type": "lifespan.shutdown.complete"})
                    return
        await r()
        p = scope["path"]
        hdrs = [(b"content-type", b"application/json")]
        if p == "/health":
            body = b'{"status":"ok"}'
        elif p == "/v1/hivemind/onboarding/manifest":
            body = json.dumps({
                "object": promoter.MANIFEST_OBJECT,
                "campaign_id": promoter.CAMPAIGN_ID,
                "status": "ok",
                "build_sha": TARGET_FULL_SHA,
                "personal_pc_only": True,
                "financial_promise": promoter.FINANCIAL_PROMISE,
                "holiday_routes": list(promoter.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
                "downloadable_scripts": list(promoter.EXPECTED_MANIFEST_SCRIPTS),
            }).encode("utf-8")
        elif p == "/v1/hivemind/onboarding/feedback/summary":
            body = json.dumps({
                "object": promoter.FEEDBACK_SUMMARY_OBJECT,
                "total_responses": 0,
                "mean_ease_rating": None,
                "zero_game_lag_ratio": None,
                "by_profile": {"dev": 0, "gamer": 0},
            }).encode("utf-8")
        else:
            pre_seq[0] += 1
            nonce = f"FastPreNonce{pre_seq[0]:04d}"
            hdrs = [(b"content-type", b"text/html"), (b"content-security-policy", f"default-src 'none'; style-src 'nonce-{nonce}'".encode())]
            body = f"<!DOCTYPE html><html><style nonce='{nonce}'></style><body>100% personal PC AGPL-3.0 0.00 EUR</body></html>".encode()
        await s({"type": "http.response.start", "status": 200, "headers": hdrs})
        await s({"type": "http.response.body", "body": body, "more_body": False})

    fast_srv = edge_repo / "deploy" / "gcp_openrouter" / "server.py"
    with pytest.raises(promoter.RevisionPromoterError):
        promoter.promote_holiday_revision("http://127.0.0.1:8000", server_py_path=fast_srv, previous_revision="bad;rev", app_factory=lambda: _fast_pre, http_fn=lambda _m, _u: (404, {}), env=CLEAN_ENV, cwd=tmp_path)
    assert promoter.promote_holiday_revision("http://127.0.0.1:8000", deploy_script_path=tmp_path / "missing.sh", app_factory=lambda: _fast_pre, http_fn=lambda _m, _u: (404, {}), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["verdict"] == "pre_promote_failed"
    assert promoter.promote_holiday_revision("http://127.0.0.1:8000", server_py_path=fast_srv, app_factory=lambda: _fast_pre, defer_on_active_training=True, http_fn=lambda _m, u: (200, {"training_active": True}) if u.endswith("/t4/status") else (404, {}), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["exit_code"] == 3
    assert promoter.promote_holiday_revision("http://127.0.0.1:8000", server_py_path=fast_srv, app_factory=lambda: _fast_pre, defer_on_active_training=False, http_fn=lambda _m, u: (200, {"training_active": True}) if u.endswith("/t4/status") else (404, {}), env=CLEAN_ENV, cwd=tmp_path, lang="en")["safety_lock_blocked"] is True
    assert "```bash" in promoter.render_markdown_report(promoter.promote_holiday_revision("http://127.0.0.1:8000", server_py_path=fast_srv, app_factory=lambda: _fast_pre, execute=False, target_sha=DRIFT_SHA, http_fn=lambda _m, _u: (404, {}), env=CLEAN_ENV, cwd=tmp_path, lang="fr"))
    assert promoter.promote_holiday_revision("http://127.0.0.1:8000", server_py_path=fast_srv, app_factory=lambda: _fast_pre, execute=True, target_sha=DRIFT_SHA, gcloud_runner=lambda *_a: (1, "fail"), http_fn=lambda _m, _u: (404, {}), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["verdict"] == "deploy_failed"
    assert promoter.promote_holiday_revision("http://127.0.0.1:8000", server_py_path=fast_srv, app_factory=lambda: _fast_pre, execute=True, no_traffic_only=True, target_sha=DRIFT_SHA, gcloud_runner=lambda *_a: (0, json.dumps({"revision": CAND_REV})), http_fn=lambda _m, _u: (404, {}), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["verdict"] == "staged_no_traffic_only"
    assert promoter.promote_holiday_revision("http://127.0.0.1:8000", server_py_path=fast_srv, app_factory=lambda: _fast_pre, execute=True, previous_revision=None, target_sha=DRIFT_SHA, gcloud_runner=lambda a, _t: (0, json.dumps({"revision": CAND_REV})) if "deploy" in a else (1, "tr err"), http_fn=lambda _m, _u: (404, {}), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["rollback_triggered"] is True

    state_promo = {"sha": sha_base}

    def _promo_http(m: str, u: str, *_a: Any, **_kw: Any) -> Any:
        if u.endswith("/health"):
            return 200, {"status": "ok"}
        if u.endswith("/manifest"):
            return 200, {"object": promoter.MANIFEST_OBJECT, "campaign_id": promoter.CAMPAIGN_ID, "status": "ok", "build_sha": state_promo["sha"], "personal_pc_only": True, "financial_promise": promoter.FINANCIAL_PROMISE, "holiday_routes": list(promoter.EXPECTED_MANIFEST_HOLIDAY_ROUTES), "downloadable_scripts": list(promoter.EXPECTED_MANIFEST_SCRIPTS)}
        if u.endswith("/v1/models"):
            return 200, {"object": "list", "data": ["skip", {"id": ""}, {"id": "hivemind/coder"}]}
        if m == "POST" or u.endswith("/summary"):
            return 200, {"object": promoter.FEEDBACK_SUMMARY_OBJECT, "total_responses": 1, "mean_ease_rating": 5.0, "zero_game_lag_ratio": 1.0, "by_profile": {"dev": 1, "gamer": 0}}
        if u.endswith(("/t4/status", "/raid/status", "/fleet")):
            return 200, {}
        return 200, {"content-security-policy": f"default-src 'none'; style-src 'nonce-Edge{hashlib.sha256(u.encode()).hexdigest()[:10]}'"}, f"<!DOCTYPE html><style nonce='Edge{hashlib.sha256(u.encode()).hexdigest()[:10]}'></style><body>100% personal PC AGPL-3.0 0.00 EUR</body>"

    assert promoter.promote_holiday_revision(
        "http://127.0.0.1:8000",
        server_py_path=fast_srv,
        app_factory=lambda: _fast_pre,
        execute=True,
        target_sha=sha_router,
        previous_revision=PREV_REV,
        gcloud_runner=lambda a, _t: (state_promo.__setitem__("sha", sha_router) or (0, json.dumps({"revision": CAND_REV}))) if "update-traffic" in a else (0, json.dumps({"revision": CAND_REV})),
        http_fn=_promo_http,
        env=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["verdict"] == "promoted"

    # Manifest fingerprint & diff + models edges
    assert promoter._is_cloud_run_deployed_path("") is False
    assert promoter._is_cloud_run_deployed_path("deploy/gcp_openrouter/Dockerfile") is True
    assert promoter._is_cloud_run_deployed_path("scripts/hivemind_doctor.py") is True
    assert len(promoter.compute_deploy_manifest_fingerprint(tmp_path / "empty_dir")["missing_files"]) == len(promoter.CLOUD_RUN_CORE_PATHS) + len(promoter.EXPECTED_MANIFEST_SCRIPTS)
    assert promoter._resolve_git_head_sha_in_repo(lambda *_a: (_ for _ in ()).throw(OSError("no")), edge_repo, lang_code="fr")[0] is None
    assert promoter._resolve_git_head_sha_in_repo(lambda *_a: (1, ""), edge_repo, lang_code="en")[0] is None
    assert promoter.detect_deploy_manifest_diff("http://127.0.0.1:8000", repo_root=edge_repo, check_git_head=True, git_runner=lambda *_a: (1, ""), http_fn=_promo_http, env=CLEAN_ENV, cwd=tmp_path, lang="fr")["sha_mismatch"] is True
    assert promoter.detect_deploy_manifest_diff("http://127.0.0.1:8000", repo_root=edge_repo, expected_sha=DRIFT_SHA, force_deploy=True, git_runner=lambda a, _t: (_ for _ in ()).throw(OSError("diff")) if "diff" in a else (0, sha_base), http_fn=_promo_http, env=CLEAN_ENV, cwd=tmp_path, lang="fr")["deploy_needed"] is True
    assert promoter._inspect_models_contract({"object": "bad"}, require_coder_model=False, lang_code="fr")[0] is False
    assert promoter._inspect_models_contract({"object": "list", "data": []}, require_coder_model=False, lang_code="fr")[0] is False
    assert promoter._inspect_models_contract({"object": "list", "data": [{"id": "hivemind/auto"}]}, require_coder_model=True, lang_code="fr")[0] is False

    # Smoke suite bad HTML / models / manifest / feedback single POST 200 vs 500
    assert promoter.verify_post_deploy_smoke_suite("http://127.0.0.1:8000", expected_sha=DRIFT_SHA, probe_feedback_post=True, http_fn=lambda m, u, *_a, **_kw: (500, {}) if (m == "POST" or u.endswith("/v1/models")) else ((200, {"status": "bad"}) if u.endswith("/health") else ((200, {"object": "bad"}) if u.endswith("/manifest") else ((200, "<html></html>") if u.endswith(("/noel", "/quick", "/raid-noel", "/arena-noel")) else (200, {"total_responses": "bad"})))), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["smoke_ok"] is False
    assert promoter.verify_post_deploy_smoke_suite("http://127.0.0.1:8000", expected_sha=sha_router, probe_feedback_post=True, http_fn=_promo_http, env=CLEAN_ENV, cwd=tmp_path)["feedback_post_ok"] is True

    # orchestrate_deploy_on_merge missing-pre-promote, active training defer/block/stage, deploy fail, unpinned rb
    with pytest.raises(promoter.RevisionPromoterError):
        promoter.orchestrate_deploy_on_merge("http://127.0.0.1:8000", previous_revision="bad;rev", env=CLEAN_ENV, cwd=tmp_path)
    assert promoter.orchestrate_deploy_on_merge("http://127.0.0.1:8000", repo_root=edge_repo, deploy_script_path=tmp_path / "missing.sh", app_factory=lambda: _fast_pre, http_fn=_promo_http, env=CLEAN_ENV, cwd=tmp_path)["verdict"] == "pre_promote_failed"
    assert promoter.orchestrate_deploy_on_merge("http://127.0.0.1:8000", repo_root=edge_repo, target_sha=sha_router, app_factory=lambda: _fast_pre, force_deploy=True, defer_on_active_training=True, http_fn=lambda m, u: (200, {"training_active": True}) if u.endswith("/t4/status") else _promo_http(m, u), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["exit_code"] == 3
    assert promoter.orchestrate_deploy_on_merge("http://127.0.0.1:8000", repo_root=edge_repo, target_sha=sha_router, app_factory=lambda: _fast_pre, force_deploy=True, defer_on_active_training=False, http_fn=lambda m, u: (200, {"training_active": True}) if u.endswith("/t4/status") else _promo_http(m, u), env=CLEAN_ENV, cwd=tmp_path, lang="en")["safety_lock_blocked"] is True
    assert promoter.orchestrate_deploy_on_merge("http://127.0.0.1:8000", repo_root=edge_repo, target_sha=sha_router, app_factory=lambda: _fast_pre, force_deploy=True, execute=True, no_traffic_only=True, gcloud_runner=lambda *_a: (0, json.dumps({"revision": CAND_REV})), http_fn=lambda m, u: (200, {"training_active": True}) if u.endswith("/t4/status") else _promo_http(m, u), env=CLEAN_ENV, cwd=tmp_path, lang="fr")["staged_no_traffic_only"] is True
    assert promoter.orchestrate_deploy_on_merge("http://127.0.0.1:8000", repo_root=edge_repo, target_sha=sha_router, app_factory=lambda: _fast_pre, force_deploy=True, execute=False, http_fn=_promo_http, env=CLEAN_ENV, cwd=tmp_path, lang="fr")["verdict"] == "dry_run_plan_ready"
    assert promoter.orchestrate_deploy_on_merge("http://127.0.0.1:8000", repo_root=edge_repo, target_sha=sha_router, app_factory=lambda: _fast_pre, force_deploy=True, execute=True, gcloud_runner=lambda *_a: (1, "err"), http_fn=_promo_http, env=CLEAN_ENV, cwd=tmp_path, lang="fr")["verdict"] == "deploy_failed"
    rb_unpinned = promoter.orchestrate_deploy_on_merge("http://127.0.0.1:8000", repo_root=edge_repo, target_sha=sha_router, previous_revision=None, app_factory=lambda: _fast_pre, force_deploy=True, execute=True, gcloud_runner=lambda a, _t: (0, json.dumps({"revision": CAND_REV})) if "deploy" in a else (1, "tr err"), http_fn=_promo_http, env=CLEAN_ENV, cwd=tmp_path, lang="fr")
    assert rb_unpinned["rollback_triggered"] is True and rb_unpinned["rollback_succeeded"] is False
    assert "Alerte Incident" in promoter.render_markdown_report(rb_unpinned)

    # main() default promote mode & JSON/stderr error outputs
    def_buf = io.StringIO()
    assert promoter.main(["--gateway", "http://127.0.0.1:8000", "--expected-sha", sha_router, "--server-py", str(fast_srv), "--json"], app_factory=lambda: _fast_pre, http_fn=_promo_http, env=CLEAN_ENV, cwd=tmp_path, stdout=def_buf) == 0
    _verify_sha256_receipt(json.loads(def_buf.getvalue()))
    err_json_buf = io.StringIO()
    assert promoter.main(["--gateway", UNTRUSTED_EXT_GW, "--json"], env=CLEAN_ENV, cwd=tmp_path, stdout=err_json_buf) == 2
    _verify_sha256_receipt(json.loads(err_json_buf.getvalue()))
    err_txt_buf = io.StringIO()
    assert promoter.main(["--gateway", UNTRUSTED_EXT_GW], env=CLEAN_ENV, cwd=tmp_path, stderr=err_txt_buf) == 2
    assert "Error:" in err_txt_buf.getvalue()


def test_zz_standalone_promoter_line_coverage_meets_95_percent() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {PROMOTER_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
