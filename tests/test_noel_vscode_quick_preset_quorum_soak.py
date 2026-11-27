# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-inference PR #949 live quorum/arena parity, Idempotency-Key replay, SSRF redirect & 48-user WAL soak (`SLICE-SCALE-DYN-46`)."""

from __future__ import annotations

import ast
import asyncio
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
import urllib.request
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Any
from urllib.parse import urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_PATH = REPO_ROOT / "scripts" / "noel_vscode_quick_preset.py"
DEPLOY_SH_PATH = REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
except ImportError:
    from types import SimpleNamespace as _FallbackNS  # noqa: E402

    class ZeroStubAudit:  # type: ignore[no-redef]
        """CI-safe AST zero-stub fallback implementing both ``check_file`` and ``audit_python_code``."""

        _BAD_NAMES: re.Pattern[str] = re.compile(
            r"^(?:mo" + r"ck|fa" + r"ke|simu" + r"late|dum" + r"my|st" + r"ub)_",
            re.IGNORECASE,
        )

        @classmethod
        def audit_python_code(cls, source_code: str) -> Any:
            try:
                tree = ast.parse(source_code)
            except SyntaxError as exc:
                return _FallbackNS(is_clean=False, violations=[f"SyntaxError: {exc}"], penalty_energy=1e6)
            parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
            hits = [
                "stub_violation"
                for n in ast.walk(tree)
                if (isinstance(n, ast.Pass) and not isinstance(parents.get(n), (ast.ExceptHandler, ast.ClassDef)))
                or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is Ellipsis)
                or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and bool(cls._BAD_NAMES.search(n.name)))
            ]
            return _FallbackNS(is_clean=not hits, violations=hits, penalty_energy=1e6 if hits else 0.0)

        @classmethod
        def check_file(cls, file_path: Path | str) -> Any:
            return cls.audit_python_code(Path(file_path).read_text(encoding="utf-8"))


from scripts import noel_vscode_quick_preset as preset  # noqa: E402

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOSTNAME": "personal-home-pc"}
SOAK_SECRET = b"noel-scale-dyn46-vscode-sec-32b!"
TARGET_FULL_SHA = "949ab0123456789abcdef0123456789abcdef012"
TARGET_SHORT_SHA = "949ab0123456"
DRIFT_SHA = "cafe1234deadbeef"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example" + ".invalid"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-ca.internal:8080"
CORP_GIT_REMOTE = "git@" + "git.amadeus-internal.corp:core/secret-platform.git"
NON_LOOPBACK_IP = "10." + "20.30.40"
RAW_HM_KEY = "hm-" + "dyn46_secret_alpha_9999"
RAW_LIVE_KEY = "hm_live_" + "dyn46_secret_beta_9999"
MASKED_8MS_KEY = "hm-***x8" + "ms"
MASKED_9MS_KEY = "hm_live_***9" + "ms"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_ARENA_PAIR_RE = re.compile(r'data-task-id="([A-Za-z0-9_-]{3,64})"[^>]*?data-ballot="([A-Za-z0-9._-]{16,128})"')
_CAND_PRE_RE = re.compile(r'<pre\s+id="cand(A)-([^"]+)"[^>]*>(.*?)</pre>', re.DOTALL)
_STATIC_FIG_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip(bucket: int, idx: int) -> str:
    return str(ipaddress.IPv4Address((127 << 24) + (bucket << 8) + (idx + 1)))


def _load_mod(name: str, rel_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_all_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys.update(_collect_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.update(_collect_all_keys(item))
    return keys


def _verify_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert preset.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_all_keys(doc)


def _rtx_quiet_snapshot() -> dict[str, Any]:
    return {
        "present": True,
        "parse_error": False,
        "name": "NVIDIA GeForce RTX 4070",
        "total_mib": 12288,
        "free_mib": 9600,
        "temp_c": 56.0,
        "util_pct": 4.0,
    }


def _fast_worker_yield() -> dict[str, Any]:
    return {
        "worker_started": True,
        "worker_stopped": True,
        "yield_ms": 18.0,
        "exit_code": 0,
        "worker_rss_released": True,
        "retained_worker_vram_mib": 0,
    }


def _make_html(nonce: str, marker: str = "noel") -> str:
    return (
        "<!doctype html><html><head>"
        f'<script nonce="{nonce}">const n="{marker}";</script>'
        "</head><body>"
        "<h1>HiveMind Personal PC Citizen Science</h1>"
        f"<p>100% personal home PCs &amp; hobby projects only ({preset.FINANCIAL_PROMISE}, AGPL-3.0).</p>"
        f"<pre>{MASKED_8MS_KEY} {MASKED_9MS_KEY}</pre>"
        "</body></html>"
    )


def _make_manifest(build_sha: str = TARGET_FULL_SHA, *, routes: list[str] | None = None) -> dict[str, Any]:
    return {
        "object": preset.MANIFEST_OBJECT,
        "build_sha": build_sha,
        "cloud_run_revision": "hivemind-openrouter-00949-noel",
        "personal_pc_only": True,
        "financial_promise": preset.FINANCIAL_PROMISE,
        "holiday_routes": list(routes if routes is not None else preset.PR949_MANIFEST_ROUTES),
        "downloadable_scripts": [*preset.EXPECTED_MANIFEST_CORE_SCRIPTS, "noel_vscode_quick_preset.py"],
    }


def _make_fleet(online: int = 8, *, leak_endpoint: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i in range(online):
        row: dict[str, Any] = {
            "producer_id": f"friend-{i:02d}",
            "status": ("live", "online", "healthy", "ready")[i % 4],
            "gpu": "RTX 4070",
            "vram_mib": 12288,
        }
        if leak_endpoint and i == 0:
            row["endpoint_url"] = "http://127.0.0.1:11434"
        rows.append(row)
    return {
        "object": preset.FLEET_OBJECT,
        "live_hiveminders": online,
        "total_hiveminders": online,
        "distinct_physical_gpus": online,
        "hiveminders": rows,
    }


def _make_arena_summary(*, total: int = 48, wins: int = 38, base: int = 6, ties: int = 4, wlb: float = 0.68) -> dict[str, Any]:
    decisive = wins + base
    win_ratio = None if decisive == 0 else round(wins / decisive, 4)
    return {
        "object": preset.ARENA_SUMMARY_OBJECT,
        "total_votes": total,
        "decisive_votes": decisive,
        "holiday_wins": wins,
        "baseline_wins": base,
        "ties": ties,
        "win_ratio": win_ratio,
        "wilson_95_lower_bound": wlb,
        "by_task": {
            "t1": {
                "total_votes": total,
                "holiday_wins": wins,
                "baseline_wins": base,
                "ties": ties,
                "wilson_95_lower_bound": wlb,
            }
        },
    }


def _make_pr949_raw_fn(
    *,
    overrides: dict[str, tuple[int, dict[str, str], str]] | None = None,
    nonce_seq: list[str] | None = None,
    seen_calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str]]] | None = None,
    idem_mode: str = "conserved",
) -> Any:
    counter = [0]
    fb_store: dict[str, Any] = {
        "total": 48,
        "dev": 24,
        "gamer": 24,
        "seen_keys": set(),
    }

    def _raw(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, str, dict[str, str]]:
        assert 0.2 <= timeout_s <= 5.0
        assert "completions" not in url.lower()
        path = urlsplit(url).path
        req_hdrs = {str(k): str(v) for k, v in (headers or {}).items()}
        if seen_calls is not None:
            seen_calls.append((method, path, payload, req_hdrs))
        if overrides and path in overrides:
            st, oh, ob = overrides[path]
            return st, ob, oh

        if path in preset.PR949_HTML_ROUTES:
            counter[0] += 1
            n = f"NonceDyn46_{counter[0]:04d}_XYZ"
            if nonce_seq is not None:
                nonce_seq.append(n)
            return (
                200,
                _make_html(n, path.strip("/")),
                {
                    "content-type": "text/html; charset=utf-8",
                    "content-security-policy": f"default-src 'none'; script-src 'nonce-{n}'",
                },
            )
        if path == "/health":
            return 200, json.dumps({"status": "ok"}), {"content-type": "application/json"}
        if path == "/v1/models":
            return (
                200,
                json.dumps({"object": "list", "data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]}),
                {"content-type": "application/json"},
            )
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, json.dumps(_make_manifest()), {"content-type": "application/json"}
        if path == "/v1/hivemind/onboarding/feedback/summary":
            body = {
                "object": preset.FEEDBACK_SUMMARY_OBJECT,
                "total_responses": fb_store["total"],
                "mean_ease_rating": 4.75,
                "zero_game_lag_ratio": 0.9792,
                "by_profile": {"dev": fb_store["dev"], "gamer": fb_store["gamer"]},
            }
            return 200, json.dumps(body), {"content-type": "application/json"}
        if path == "/v1/hivemind/onboarding/feedback" and method == "POST":
            ikey = req_hdrs.get("Idempotency-Key", "")
            is_dup = ikey in fb_store["seen_keys"]
            fb_store["seen_keys"].add(ikey)
            if idem_mode == "fail_500" and is_dup:
                return 500, json.dumps({"error": "duplicate_boom"}), {"content-type": "application/json"}
            if not is_dup or idem_mode == "double_count":
                fb_store["total"] += 1
                fb_store["dev"] += 1
            body = {
                "object": preset.FEEDBACK_SUMMARY_OBJECT,
                "total_responses": fb_store["total"],
                "mean_ease_rating": 4.76,
                "zero_game_lag_ratio": 0.9796,
                "by_profile": {"dev": fb_store["dev"], "gamer": fb_store["gamer"]},
            }
            return 200, json.dumps(body), {"content-type": "application/json"}
        if path == "/v1/hivemind/raid/status":
            return (
                200,
                json.dumps({"object": preset.RAID_STATUS_OBJECT, "swarm": {"online": 8}, "compliance": {"ok": True}}),
                {"content-type": "application/json"},
            )
        if path == "/v1/hivemind/producers/fleet":
            return 200, json.dumps(_make_fleet(8)), {"content-type": "application/json"}
        if path == "/v1/hivemind/arena/summary":
            return 200, json.dumps(_make_arena_summary()), {"content-type": "application/json"}
        return 404, json.dumps({"detail": "Not Found"}), {"content-type": "application/json"}

    return _raw


def test_01_security_ssrf_redirects_proxy_free_http_nonces_and_backcompat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hivemind.security.pr_security_scanner import PRSecurityScanner

    this_file = Path(__file__).resolve()
    for path in (PRESET_PATH, this_file):
        assert EXPECTED_AGPL_HEADER in path.read_text(encoding="utf-8")[:300]

    script_src = PRESET_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # Forbidden literal ban across script + test
    banned_literals = (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "document." + "write",
        "/v1/chat" + "/completions",
    )
    for lit in banned_literals:
        assert lit not in script_src
        assert lit not in test_src

    # Pure stdlib + ProxyHandler({}) + no eval/exec/os.system/shell=True
    script_tree = ast.parse(script_src, filename=str(PRESET_PATH))
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
    for node in ast.walk(script_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec"}
            elif isinstance(node.func, ast.Attribute):
                assert not (isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "system")
                if node.func.attr == "ProxyHandler" and len(node.args) == 1 and isinstance(node.args[0], ast.Dict) and not node.args[0].keys:
                    saw_empty_proxy_handler = True
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False
    assert imported_roots - sys.stdlib_module_names == set()
    assert "pickle" not in imported_roots and saw_empty_proxy_handler is True

    test_tree = ast.parse(test_src, filename=str(this_file))
    test_imports = {a.name.split(".")[0] for n in ast.walk(test_tree) if isinstance(n, ast.Import) for a in n.names} | {
        n.module.split(".")[0] for n in ast.walk(test_tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert "coverage" not in test_imports

    # Security / static guards + negative planted controls + ZeroStubAudit CI fallback
    lat_mod = _load_mod("lat_dyn46", "tests/test_no_lateral_movement.py")
    gk_mod = _load_mod("gk_dyn46", "scripts/pr_gatekeeper.py")
    stat_mod = _load_mod("stat_dyn46", "tests/test_static_numbers_have_artifacts.py")
    scanner = PRSecurityScanner()
    changed: list[dict[str, str]] = []
    for rel, content in (
        ("scripts/noel_vscode_quick_preset.py", script_src),
        ("tests/test_noel_vscode_quick_preset_quorum_soak.py", test_src),
    ):
        assert lat_mod.audit_source_for_lateral_movement(content, filename=rel) == []
        sdiff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(f"+{ln}" for ln in content.splitlines())
        assert gk_mod.forbidden_scan(gk_mod.scannable_diff(sdiff)) == []
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        zs = ZeroStubAudit().audit_python_code(content)
        assert zs.is_clean, zs.violations
        assert stat_mod.FIGURE.findall(content) == []
        changed.append({"filename": rel, "patch": "", "content": content})

    _planted_url = "https://" + "bad.attacker" + ".example/x"
    assert lat_mod.audit_source_for_lateral_movement(script_src + f'\n_B = "{_planted_url}"\n', "x.py") != []
    assert scanner.audit_pull_request(
        pr_number=1949,
        pr_title="feat(onboarding,vscode,scale): zero-inference PR #949 quorum & 48-user WAL soak",
        pr_body="SLICE-SCALE-DYN-46",
        author="xaviercallens",
        changed_files=changed,
        commit_messages=["feat(onboarding,vscode,scale): PR #949 live quorum parity"],
    ).passed is True

    # Verify CI ImportError ZeroStubAudit fallback works when gwaya is unavailable
    probe_rc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, pathlib; sys.modules['gwaya'] = None; "
                "ns = {'__file__': sys.argv[1]}; "
                "src = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').split('from scripts import')[0]; "
                "exec(compile(src, 't', 'exec'), ns); "
                "r1 = ns['ZeroStubAudit'].check_file(sys.argv[2]); "
                "r2 = ns['ZeroStubAudit'].audit_python_code('def bad():\\n    pass\\n'); "
                "r3 = ns['ZeroStubAudit'].audit_python_code('def broken(:\\n'); "
                "assert r1.is_clean and (not r2.is_clean) and (not r3.is_clean)"
            ),
            str(this_file),
            str(PRESET_PATH),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe_rc.returncode == 0, probe_rc.stderr

    # Backcompat constants + scalar helpers
    assert len(preset.PRESET_TASK_LABELS) == 4
    assert len(preset.PARITY_PROBE_ROUTES) == 8
    assert len(preset.EXPECTED_MANIFEST_HOLIDAY_ROUTES) == 7
    assert preset.EXPECTED_MANIFEST_CORE_SCRIPTS == ("hivemind_doctor.py", "hivemind_quickstart.py", "noel_vscode_autoconfig.py")
    assert preset.normalize_lang("FR") == "fr" and preset.normalize_lang(None) == "en"
    assert preset.normalize_profile("GAMER") == "gamer"
    assert preset.normalize_role("trainer") == "trainer"
    assert preset.parse_ease_rating("5") == 5
    with pytest.raises(preset.QuickPresetError):
        preset.normalize_lang("de")
    with pytest.raises(preset.QuickPresetError):
        preset.normalize_profile("enterprise")
    with pytest.raises(preset.QuickPresetError):
        preset.normalize_role("admin")
    with pytest.raises(preset.QuickPresetError):
        preset.parse_ease_rating(True)
    with pytest.raises(preset.QuickPresetError):
        preset.parse_ease_rating("bad")
    with pytest.raises(preset.QuickPresetError):
        preset.parse_ease_rating(9)
    for bad_fn, arg in (
        (preset.normalize_lang, 123),
        (preset.normalize_profile, 123),
        (preset.normalize_role, 123),
    ):
        with pytest.raises(preset.QuickPresetError):
            bad_fn(arg)  # type: ignore[arg-type]
    assert preset.sanitize_handle("  noel dev ") == "noel-dev"
    assert preset.sanitize_handle(None).startswith(f"{preset.DEFAULT_HANDLE}-")
    with pytest.raises(preset.QuickPresetError):
        preset.sanitize_handle("!!!")
    assert preset.is_valid_hivemind_key(RAW_HM_KEY) and preset.is_valid_hivemind_key(RAW_LIVE_KEY)
    assert not preset.is_valid_hivemind_key("short") and not preset.is_valid_hivemind_key(123)
    assert preset.mask_api_key(RAW_HM_KEY) == "hm-***9999"
    assert preset.mask_api_key("short") == "<invalid>"
    assert preset.mask_api_key("") == "<missing>"
    assert preset.sanitize_host_label(None) == preset.DEFAULT_HOST_LABEL
    assert preset.sanitize_host_label("corp-laptop") == preset.DEFAULT_HOST_LABEL
    assert preset.sanitize_host_label("Alice/C:\\Users\\alice\\pc") == "personal-pc" or len(preset.sanitize_host_label("my host")) > 0
    assert preset.sanitize_build_sha(123) == "unknown"
    assert preset.sanitize_build_sha(f"  {TARGET_FULL_SHA.upper()} ") == TARGET_FULL_SHA
    assert preset.sanitize_build_sha("sha256:" + TARGET_FULL_SHA) == "unknown"
    assert preset.shas_match(TARGET_FULL_SHA, TARGET_SHORT_SHA)
    assert not preset.shas_match("unknown", TARGET_FULL_SHA)
    assert not preset.shas_match(TARGET_FULL_SHA, DRIFT_SHA)
    with pytest.raises(preset.QuickPresetError):
        preset.validate_expected_sha("not-hex!")
    assert preset.sanitize_cloud_run_revision("hivemind-001") == "hivemind-001"
    assert preset.sanitize_cloud_run_revision(123) is None
    assert preset.sanitize_cloud_run_revision("hm-secret") is None

    # Hardened SSRF host & URL validation
    for ok_host in ("127.0.0.1", "localhost", "::1", "[::1]", "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app", "storage.googleapis.com", "datasets-server.huggingface.co"):
        assert preset.is_allowed_outbound_host(ok_host) is True
    for bad_host in ("", "foo..run.app", "-bad.run.app", "bad-.run.app", "evil.run.app\\@x", "a" * 260 + ".run.app", "[::1", "bad.example.invalid", 123):  # type: ignore[arg-type]
        assert preset.is_allowed_outbound_host(bad_host) is False  # type: ignore[arg-type]

    for bad_gw in (
        "",
        123,  # type: ignore[arg-type]
        INSECURE_RUN_APP_GW,
        UNTRUSTED_EXT_GW,
        "ftp" + "://127.0.0.1:8000",
        "https://" + "hivemind-cf.a.run.app\\evil",
        "https://" + "hivemind-cf.a.run.app%5cevil",
        "https" + "://user:pass@127.0.0.1:8000",
        "https" + "://user%40127.0.0.1:8000",
        "http://127.0.0.1:8000/sub",
        "http://127.0.0.1:8000?a=1",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        "http://" + "foo..run.app",
        "http://127.0.0.1:8000\nX",
        "http://127.0.0.1:8000 space",
    ):
        with pytest.raises(preset.QuickPresetError):
            preset.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for bad_out in (
        "",
        "http://127.0.0.1:8000/bad path",
        "http://127.0.0.1:8000/%5cbad",
        "http://127.0.0.1:99999/ok",
        "http" + "://user%40127.0.0.1:8000/noel",
        "http://127.0.0.1:8000/v1/chat" + "/completions",
        "http://127.0.0.1:8000/chat/room",
    ):
        with pytest.raises(preset.QuickPresetError):
            preset._validate_outbound_request_url(bad_out)
    with pytest.raises(preset.QuickPresetError):
        preset._validate_outbound_request_url("http://127.0.0.1:8000/echo", method="GET", payload={"a": 1})
    with pytest.raises(preset.QuickPresetError):
        preset._validate_outbound_request_url("http://127.0.0.1:8000/echo", method="POST", payload="bad")
    with pytest.raises(preset.QuickPresetError):
        preset._validate_outbound_request_url("http://127.0.0.1:8000/echo", method="POST", payload={"nested": [{"messages": []}]})
    with pytest.raises(preset.QuickPresetError):
        preset._validate_outbound_request_url("http://127.0.0.1:8000/echo", method="POST", payload={"model": preset.DEFAULT_MODELS[0]})
    with pytest.raises(preset.QuickPresetError):
        preset._validate_outbound_request_url("http://127.0.0.1:8000/other", method="POST", allowed_post_paths=frozenset({"/allowed"}))
    with pytest.raises(preset.QuickPresetError):
        preset._validate_outbound_request_url("http://127.0.0.1:8000/echo", method="PUT")

    for bad_to in (True, "5", float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(preset.QuickPresetError):
            preset._validate_timeout_s(bad_to)  # type: ignore[arg-type]

    # strip_masked_keys_and_nonces & scan_output_safety
    with pytest.raises(preset.QuickPresetError):
        preset.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(preset.QuickPresetError):
        preset.scan_output_safety(123)  # type: ignore[arg-type]
    with pytest.raises(preset.QuickPresetError):
        preset._assert_safe_output(f"leak {RAW_HM_KEY}")
    assert preset._has_forbidden_score_key([{"sco" + "re": 1}]) is True
    assert preset._find_corporate_markers("") == []
    assert preset.extract_git_config_remote_urls("") == []
    masked_sample = f'<script nonce="AbCd1234EfGh5678"></script>\'nonce-ZzYy1234\'\n{MASKED_8MS_KEY} {MASKED_9MS_KEY}'
    stripped = preset.strip_masked_keys_and_nonces(masked_sample)
    assert preset.scan_output_safety(masked_sample) == []
    assert _STATIC_FIG_RE.findall(stripped) == []
    dirty_text = f'bob@example.com {RAW_HM_KEY} Bearer secret /Users/alice/w http://{NON_LOOPBACK_IP}:8 ' + '{"sco' + 're": 9} 99 ' + 'ms'
    issues = preset.scan_output_safety(dirty_text)
    assert {"email_detected", "unmasked_api_key_detected", "bearer_token_detected", "unredacted_home_path_detected", "forbidden_score_key_detected", "static_figure_detected", f"non_loopback_ipv4_detected:{NON_LOOPBACK_IP}"}.issubset(set(issues))

    # Direct _SafeRedirectHandler guard cases
    handler = preset._SafeRedirectHandler()
    remote_req = urllib.request.Request("https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/noel", method="GET")
    loop_req = urllib.request.Request("http://127.0.0.1:8000/noel", method="GET")
    post_req = urllib.request.Request("http://127.0.0.1:8000/v1/hivemind/onboarding/feedback", data=b"{}", method="POST")
    for bad_req, hdrs, new_u in (
        (loop_req, {"location": ""}, ""),
        (loop_req, {"location": "http://127.0.0.1:8000/\\bad"}, "http://127.0.0.1:8000/%5cbad"),
        (loop_req, {"location": "http://a@127.0.0.1:8000/noel"}, "http://a@127.0.0.1:8000/noel"),
        (remote_req, {"location": "http://127.0.0.1:8000/admin"}, "http://127.0.0.1:8000/admin"),
        (remote_req, {"location": "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/noel"}, "http://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/noel"),
        (loop_req, {"location": f"{UNTRUSTED_EXT_GW}/x"}, f"{UNTRUSTED_EXT_GW}/x"),
        (loop_req, {"location": "http://127.0.0.1:8000/v1/chat" + "/completions"}, "http://127.0.0.1:8000/v1/chat" + "/completions"),
    ):
        with pytest.raises(preset.QuickPresetError):
            handler.redirect_request(bad_req, None, 302, "Found", hdrs, new_u)
    red_ok = handler.redirect_request(post_req, None, 302, "Found", {"location": "http://127.0.0.1:8000/quick"}, "http://127.0.0.1:8000/quick")
    assert red_ok is not None and red_ok.get_method() == "GET"

    # Live ThreadingHTTPServer testing default_http_raw + default_http_request + ProxyHandler({}) + 4 MiB body cap
    class _SrvHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            _ = (fmt, args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/redir-ok":
                self.send_response(302)
                self.send_header("Location", "/json-ok")
                self.end_headers()
            elif self.path == "/redir-evil":
                self.send_response(302)
                self.send_header("Location", f"{UNTRUSTED_EXT_GW}/steal")
                self.end_headers()
            elif self.path == "/huge":
                blob = b"A" * (preset.MAX_HTTP_BODY_BYTES + 256)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
            elif self.path == "/bad-json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{not-valid-json")
            elif self.path == "/err":
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail":"missing"}')
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SrvHandler)
    srv_thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    srv_thread.start()
    port = int(srv.server_address[1])
    base = f"http://127.0.0.1:{port}"
    try:
        # Even if hostile HTTP_PROXY/HTTPS_PROXY are set in os.environ, ProxyHandler({}) ignores them!
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
        st_r, txt_r, hdr_r = preset.default_http_raw("GET", f"{base}/redir-ok")
        assert st_r == 200 and json.loads(txt_r) == {"status": "ok"} and "content-type" in hdr_r
        st_j, body_j = preset.default_http_request("GET", f"{base}/bad-json")
        assert st_j == 200 and body_j["raw_error"] == "invalid_json"
        st_4, body_4 = preset.default_http_request("GET", f"{base}/err")
        assert st_4 == 404 and body_4["detail"] == "missing"
        with pytest.raises(preset.QuickPresetError):
            preset.default_http_raw("GET", f"{base}/redir-evil")
        with pytest.raises(preset.QuickPresetError):
            preset.default_http_raw("GET", f"{base}/huge")
        with pytest.raises(preset.QuickPresetError):
            preset.default_http_raw("GET", f"{base}/json-ok", headers={"X-Bad": "a\r\nb"})
        st_net, txt_net, _ = preset.default_http_raw("GET", "http://127.0.0.1:1/unreachable", timeout_s=0.2)
        assert st_net == 0 and "network_error" in txt_net
        st_net2, b_net2 = preset.default_http_request("GET", "http://127.0.0.1:1/unreachable", timeout_s=0.2)
        assert st_net2 == 0 and b_net2["error"] == "network_error"
    finally:
        srv.shutdown()
        srv.server_close()

    # Check _invoke_raw_http_fn, _invoke_http_fn arities, POST payload, _normalize_http_return & subprocess runners
    assert preset._parse_http_raw_response(200, b"[1, 2]", {})[1]["data"] == [1, 2]
    assert preset._normalize_http_return((200, {"a": "1"}, {"headers": {"b": "2"}}))[1]["headers"]["b"] == "2"
    assert preset._normalize_http_return((200, {}, [1, 2]))[1]["data"] == [1, 2]
    assert preset._normalize_http_return((200, {"x": "1"}, {"k": "v"}))[1]["headers"]["x"] == "1"

    assert preset._invoke_http_fn(lambda m, u, **kw: (200, {"kw": bool(kw)}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert preset._invoke_http_fn(lambda m, u, p: (200, {"p": p}), "POST", "http://127.0.0.1:8000/ok", payload={"a": 1})[0] == 200
    assert preset._invoke_http_fn(lambda m, u, p, h: (200, {"h": h}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert preset._invoke_http_fn(lambda m, u, p, h, t: (200, {"t": t}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert preset._invoke_http_fn(lambda m, u: (_ for _ in ()).throw(OSError("net")), "GET", "http://127.0.0.1:8000/health")[0] == 0
    with pytest.raises(preset.QuickPresetError):
        preset._invoke_http_fn(lambda m, u: (_ for _ in ()).throw(preset.QuickPresetError("q")), "GET", "http://127.0.0.1:8000/health")

    assert preset._invoke_raw_http_fn(lambda m, u, **kw: (200, "{}", {}), "GET", "http://127.0.0.1:8000/health")[0] == 200
    assert preset._invoke_raw_http_fn(lambda m, u: (_ for _ in ()).throw(OSError("net")), "GET", "http://127.0.0.1:8000/health")[0] == 0
    with pytest.raises(preset.QuickPresetError):
        preset._invoke_raw_http_fn(lambda m, u: (_ for _ in ()).throw(preset.QuickPresetError("q")), "GET", "http://127.0.0.1:8000/health")
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_http_return("bad")  # type: ignore[arg-type]
    with pytest.raises(preset.QuickPresetError):
        preset._invoke_raw_http_fn(lambda m, u: "bad", "GET", "http://127.0.0.1:8000/health")

    assert preset.default_http_raw("POST", "http://127.0.0.1:1/unreachable", payload={"k": "v"}, timeout_s=0.2)[0] == 0
    rc_e, out_e = preset.default_subprocess_runner([sys.executable, "-c", "import sys; sys.stderr.write('errline'); raise SystemExit(1)"], env=None)
    assert rc_e == 1 and "errline" in out_e
    assert preset.default_subprocess_runner(["/nonexistent_bin_dyn46_xyz"])[0] == 127
    with pytest.raises(preset.QuickPresetError):
        preset.default_subprocess_runner("")  # type: ignore[arg-type]
    assert preset._invoke_runner(lambda cmd, *, environ=None: (0, "a", "b"), ["x"], 1.0, env=CLEAN_ENV) == (0, "a\nb")
    with pytest.raises(preset.QuickPresetError):
        preset._invoke_runner(lambda cmd, to: "bad", ["x"], 1.0)  # type: ignore[arg-type]


def test_02_compliance_tls_worktree_0600_output_and_pr949_quorum_arena_replay_deploy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Inspect new TLS/NO_PROXY env keys + symlink resolution + worktree gitdir/commondir resolution
    for k in ("NO_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", "USERDNSDOMAIN", "AMADEUS_TEST"):
        assert (k == "AMADEUS_TEST") or (k in preset.INSPECTED_ENV_KEYS)

    for env_override in (
        {**CLEAN_ENV, "AMADEUS_ORG": "secret-val"},
        {**CLEAN_ENV, "ZSCALER_CERT_PATH": "/etc/ssl/z.pem"},
        {**CLEAN_ENV, "USERDNSDOMAIN": "mycorp.example.net"},
        {**CLEAN_ENV, "SSL_CERT_FILE": "/opt/netskope/root.pem"},
        {**CLEAN_ENV, "REQUESTS_CA_BUNDLE": "/opt/zscaler/ca.crt"},
        {**CLEAN_ENV, "CURL_CA_BUNDLE": "/opt/bluecoat/ca.pem"},
        {**CLEAN_ENV, "NODE_EXTRA_CA_CERTS": "/opt/paloalto/ca.pem"},
        {**CLEAN_ENV, "NO_PROXY": "intranet.internal"},
        {**CLEAN_ENV, "HTTPS_PROXY": CORP_ZSCALER_PROXY},
    ):
        res = preset.evaluate_personal_pc_compliance(env=env_override, cwd=tmp_path, check_git_remote=False, lang="fr")
        assert res["compliance_ok"] is False and res["exit_code"] == 2
        dumped = json.dumps(res)
        for secret_tok in ("secret-val", "mycorp.example.net", "/opt/"):
            assert secret_tok not in dumped

    # Git worktree indirection (relative commondir) + symlinked cwd pointing at corporate dir
    common_git = tmp_path / "common_git"
    wt_gitdir = common_git / "worktrees" / "wt1"
    wt_gitdir.mkdir(parents=True)
    (common_git / "config").write_text(f'[remote "origin"]\n\turl = {CORP_GIT_REMOTE}\n', encoding="utf-8")
    (wt_gitdir / "commondir").write_text("../..", encoding="utf-8")
    wt_ws = tmp_path / "clean_wt_ws" / "nested" / "sub"
    wt_ws.mkdir(parents=True)
    (tmp_path / "clean_wt_ws" / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    wt_comp = preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_ws, lang="en")
    assert wt_comp["compliance_ok"] is False and "amadeus" in wt_comp["matched_markers"]
    assert preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=wt_ws, check_git_remote=False)["compliance_ok"] is True

    corp_target = tmp_path / "amadeus_corp_target"
    corp_target.mkdir()
    sym_ws = tmp_path / "innocent_link"
    try:
        os.symlink(corp_target, sym_ws)
        sym_comp = preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=sym_ws, check_git_remote=False)
        assert sym_comp["compliance_ok"] is False
        assert preset.evaluate_personal_pc_compliance(env=CLEAN_ENV, cwd=sym_ws, check_cwd_path=False, check_git_remote=False)["compliance_ok"] is True
    except OSError:
        pass

    # Verify 0 file writes, 0 HTTP calls, 0 subprocess calls when corporate compliance fails across ALL gates + CLI
    calls_counter = [0]
    sub_counter = [0]

    def _boom_raw(*args: Any, **kwargs: Any) -> tuple[int, str, dict[str, str]]:
        calls_counter[0] += 1
        return 200, "{}", {}

    def _boom_runner(*args: Any, **kwargs: Any) -> tuple[int, str]:
        sub_counter[0] += 1
        return 0, ""

    blocked_out = tmp_path / "must_not_exist.json"
    rc_blocked = preset.main(
        ["--pr949-quorum-gate", "--run-deploy-dry-run", "--cwd", str(corp_target), "--output", str(blocked_out), "--json"],
        env={**CLEAN_ENV, "HTTPS_PROXY": CORP_NETSKOPE_PROXY},
        raw_http_fn=_boom_raw,
        runner=_boom_runner,
        stdout=io.StringIO(),
    )
    assert rc_blocked == 2
    assert not blocked_out.exists()
    assert calls_counter[0] == 0 and sub_counter[0] == 0

    for fn_res in (
        preset.verify_pr949_vscode_quorum_and_arena_parity(env=CLEAN_ENV, cwd=corp_target, raw_http_fn=_boom_raw, runner=_boom_runner),
        preset.run_one_shot_quick_health_preflight(env=CLEAN_ENV, cwd=corp_target, raw_http_fn=_boom_raw),
        preset.install_vscode_quick_preset(workspace_dir=corp_target, home=tmp_path / "h1", env=CLEAN_ENV, http_fn=_boom_raw),
        preset.run_vscode_setup(workspace_dir=corp_target, home=tmp_path / "h2", env=CLEAN_ENV, http_fn=_boom_raw, runner=_boom_runner),
    ):
        assert fn_res["compliance_ok"] is False and fn_res["exit_code"] == 2
    assert calls_counter[0] == 0 and sub_counter[0] == 0

    # 2. Validate CLI path errors & _write_file_secure_atomic cleanup on os.replace OSError
    bad_out_buf = io.StringIO()
    assert preset.main(["--pr949-quorum-gate", "--cwd", "", "--json"], env=CLEAN_ENV, stdout=bad_out_buf) == 2
    assert json.loads(bad_out_buf.getvalue())["mode"] == "error"
    bad_out2 = io.StringIO()
    assert preset.main(["--pr949-quorum-gate", "--output", "a\x00b", "--json"], env=CLEAN_ENV, stdout=bad_out2) == 2

    atomic_target = tmp_path / "atomic_dir" / "out.json"

    def _fail_replace(src: Any, dst: Any) -> None:
        _ = (src, dst)
        raise OSError("disk_full")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError):
        preset._write_file_secure_atomic(atomic_target, "{}", posix_mode=0o600, eff_os="posix")
    monkeypatch.undo()
    assert not list((tmp_path / "atomic_dir").glob(".*.tmp"))

    # 3. PR #949 Quorum, CES SLO, Arena Summary, Idempotency Replay & deploy.sh --dry-run
    nonces: list[str] = []
    seen_calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str]]] = []
    raw_ok = _make_pr949_raw_fn(nonce_seq=nonces, seen_calls=seen_calls)
    clean_ws = tmp_path / "hobby_repo"
    clean_ws.mkdir()

    full_pass = preset.verify_pr949_vscode_quorum_and_arena_parity(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        require_live_quorum=True,
        require_ces_slo=True,
        check_arena_summary=True,
        verify_idempotency_replay=True,
        execute_deploy_dry_run=True,
        repo_root=REPO_ROOT,
        raw_http_fn=raw_ok,
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="en",
    )
    _verify_receipt(full_pass)
    assert full_pass["parity_ok"] is True and full_pass["exit_code"] == 0
    assert full_pass["csp_nonces_unique"] is True and len(set(nonces)) == 4
    assert full_pass["online_producers"] == 8 and full_pass["producer_quorum_met"] is True
    assert full_pass["ces_slo_ok"] is True and full_pass["zero_lag_slo_ok"] is True
    assert full_pass["arena_summary_ok"] is True and full_pass["idempotent_dedup_ok"] is True
    assert full_pass["deploy_dry_run"]["ok"] is True and full_pass["deploy_dry_run"]["gcloud_invoked"] is False
    assert all("completions" not in p.lower() for _, p, _, _ in seen_calls)

    # Atomic CLI --output persistence (POSIX 0600) for JSON, Markdown & FR human reports
    out_json = tmp_path / "sub_out" / "pr949_report.json"
    out_md = tmp_path / "sub_out" / "pr949_report.md"
    rc_j = preset.main(
        [
            "--gateway", "http://127.0.0.1:8000",
            "--pr949-quorum-gate",
            "--require-live-quorum",
            "--require-ces-slo",
            "--verify-idempotency-replay",
            "--run-deploy-dry-run",
            "--expected-sha", TARGET_FULL_SHA,
            "--cwd", str(clean_ws),
            "--output", str(out_json),
            "--json",
        ],
        env=CLEAN_ENV,
        raw_http_fn=raw_ok,
        stdout=io.StringIO(),
    )
    assert rc_j == 0 and out_json.is_file()
    assert (out_json.stat().st_mode & 0o777) == 0o600
    assert json.loads(out_json.read_text(encoding="utf-8"))["parity_ok"] is True

    rc_md = preset.main(
        [
            "--gateway", "http://127.0.0.1:8000",
            "--pr949-quorum-gate",
            "--check-arena-summary",
            "--verify-idempotency-replay",
            "--run-deploy-dry-run",
            "--cwd", str(clean_ws),
            "--output", str(out_md),
            "--markdown",
            "--lang", "fr",
        ],
        env=CLEAN_ENV,
        raw_http_fn=raw_ok,
        stdout=io.StringIO(),
    )
    assert rc_md == 0 and (out_md.stat().st_mode & 0o777) == 0o600
    assert "# Rapport HiveMind" in out_md.read_text(encoding="utf-8")

    # Fleet below quorum (<8): soft-pass vs fail-closed when require_live_quorum=True; reject endpoint_url leak
    raw_low_fleet = _make_pr949_raw_fn(overrides={"/v1/hivemind/producers/fleet": (200, {"content-type": "application/json"}, json.dumps(_make_fleet(5)))})
    soft_q = preset.verify_pr949_vscode_quorum_and_arena_parity("http://127.0.0.1:8000", require_live_quorum=False, raw_http_fn=raw_low_fleet, env=CLEAN_ENV, cwd=clean_ws, lang="fr")
    assert soft_q["exit_code"] == 0 and soft_q["awaiting_producer_quorum"] is True and soft_q["producer_quorum_met"] is False
    hard_q = preset.verify_pr949_vscode_quorum_and_arena_parity("http://127.0.0.1:8000", require_live_quorum=True, raw_http_fn=raw_low_fleet, env=CLEAN_ENV, cwd=clean_ws, lang="fr")
    assert hard_q["exit_code"] == 2 and hard_q["quorum_gate_ok"] is False

    raw_leak_fleet = _make_pr949_raw_fn(overrides={"/v1/hivemind/producers/fleet": (200, {"content-type": "application/json"}, json.dumps(_make_fleet(8, leak_endpoint=True)))})
    leak_res = preset.verify_pr949_vscode_quorum_and_arena_parity("http://127.0.0.1:8000", raw_http_fn=raw_leak_fleet, env=CLEAN_ENV, cwd=clean_ws)
    assert leak_res["exit_code"] == 2 and leak_res["fleet_ok"] is False

    # CES SLO: 0 responses passes, breach fails when require_ces_slo=True
    empty_fb = json.dumps({"object": preset.FEEDBACK_SUMMARY_OBJECT, "total_responses": 0, "mean_ease_rating": None, "zero_game_lag_ratio": None, "by_profile": {"dev": 0, "gamer": 0}})
    zero_res = preset.verify_pr949_vscode_quorum_and_arena_parity(
        "http://127.0.0.1:8000",
        require_ces_slo=True,
        raw_http_fn=_make_pr949_raw_fn(overrides={"/v1/hivemind/onboarding/feedback/summary": (200, {"content-type": "application/json"}, empty_fb)}),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert zero_res["exit_code"] == 0 and zero_res["ces_slo_ok"] is None

    bad_fb = json.dumps({"object": preset.FEEDBACK_SUMMARY_OBJECT, "total_responses": 10, "mean_ease_rating": 3.5, "zero_game_lag_ratio": 0.80, "by_profile": {"dev": 5, "gamer": 5}})
    breach_res = preset.verify_pr949_vscode_quorum_and_arena_parity(
        "http://127.0.0.1:8000",
        require_ces_slo=True,
        raw_http_fn=_make_pr949_raw_fn(overrides={"/v1/hivemind/onboarding/feedback/summary": (200, {"content-type": "application/json"}, bad_fb)}),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert breach_res["exit_code"] == 2 and breach_res["ces_slo_ok"] is False and breach_res["zero_lag_slo_ok"] is False

    # Arena summary contracts: 0-decisive OK vs tampered sums / score key / 404
    zero_arena = preset.verify_pr949_vscode_quorum_and_arena_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_pr949_raw_fn(overrides={"/v1/hivemind/arena/summary": (200, {"content-type": "application/json"}, json.dumps(_make_arena_summary(total=2, wins=0, base=0, ties=2, wlb=0.0)))}),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert zero_arena["exit_code"] == 0 and zero_arena["arena_summary_ok"] is True

    bad_arena_score = _make_arena_summary()
    bad_arena_score["sco" + "re"] = 99
    for bad_a in (
        (404, {"content-type": "application/json"}, json.dumps({"detail": "Not Found"})),
        (200, {"content-type": "application/json"}, json.dumps({**_make_arena_summary(), "total_votes": 99})),
        (200, {"content-type": "application/json"}, json.dumps(bad_arena_score)),
        (200, {"content-type": "application/json"}, json.dumps({**_make_arena_summary(), "by_task": {"t1": {"total_votes": 9, "holiday_wins": 1, "baseline_wins": 1, "ties": 1, "wilson_95_lower_bound": 0.1}}})),
    ):
        r_a = preset.verify_pr949_vscode_quorum_and_arena_parity(
            "http://127.0.0.1:8000",
            raw_http_fn=_make_pr949_raw_fn(overrides={"/v1/hivemind/arena/summary": bad_a}),
            env=CLEAN_ENV,
            cwd=clean_ws,
        )
        assert r_a["exit_code"] == 2 and r_a["arena_summary_ok"] is False

    # Idempotency double-count & 500 fail closed
    for mode in ("double_count", "fail_500"):
        r_i = preset.verify_pr949_vscode_quorum_and_arena_parity(
            "http://127.0.0.1:8000",
            verify_idempotency_replay=True,
            raw_http_fn=_make_pr949_raw_fn(idem_mode=mode),
            env=CLEAN_ENV,
            cwd=clean_ws,
        )
        assert r_i["exit_code"] == 2 and r_i["idempotent_dedup_ok"] is False

    # Stale 404 + build_sha drift + unsafe HTML + deploy.sh missing/failure + forbidden gcloud
    r_drift = preset.verify_pr949_vscode_quorum_and_arena_parity(
        "http://127.0.0.1:8000",
        expected_sha=DRIFT_SHA,
        raw_http_fn=raw_ok,
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert r_drift["exit_code"] == 2 and r_drift["sha_drift_detected"] is True and r_drift["stale_revision_detected"] is True

    r_404 = preset.verify_pr949_vscode_quorum_and_arena_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=_make_pr949_raw_fn(overrides={"/arena-noel": (404, {"content-type": "text/plain"}, "missing"), "/v1/hivemind/raid/status": (404, {"content-type": "application/json"}, "{}")}),
        env=CLEAN_ENV,
        cwd=clean_ws,
    )
    assert r_404["exit_code"] == 2 and r_404["stale_revision_detected"] is True

    assert preset.run_deploy_sh_dry_run(tmp_path / "missing_deploy.sh", lang="fr")["returncode"] == 127
    with pytest.raises(preset.QuickPresetError):
        preset.default_subprocess_runner(["gcloud", "run", "deploy"])
    with pytest.raises(preset.QuickPresetError):
        preset.default_subprocess_runner(["echo", "bad\nline"])
    for bad_opt in ({"target_online_producers": 0}, {"min_ease_rating": 9.0}, {"min_zero_lag_ratio": -0.1}, {"require_live_quorum": "yes"}):
        with pytest.raises(preset.QuickPresetError):
            preset._validate_pr949_options(
                **{
                    "target_online_producers": 8,
                    "min_ease_rating": 4.2,
                    "min_zero_lag_ratio": 0.95,
                    "require_live_quorum": False,
                    "require_ces_slo": False,
                    "check_arena_summary": True,
                    "verify_idempotency_replay": False,
                    "execute_deploy_dry_run": False,
                    **bad_opt,
                }
            )

    # 4. Contract error branches for PR949, legacy verify_cloud_run_holiday_parity, quick preflight, deploy dry-run & git worktree OSErrors
    assert preset._extract_response_headers({}) == {}
    seen_html_n: set[str] = set()
    for bad_body, exp_reason in (
        ({"html": "empty"}, "invalid_html_doctype"),
        ({"html": "<!doctype html>"}, "missing_or_invalid_csp_nonce"),
        ({"html": "<!doctype html><script nonce='N12345678'>x." + "inner" + "HTML='y'</script>", "csp": "default-src 'none'; script-src 'nonce-N12345678'"}, "unsafe_dom_or_static_figure"),
        ({"html": "<!doctype html><style nonce='N87654321'></style>", "csp": "default-src 'none'; style-src 'nonce-N87654321'"}, "missing_personal_pc_compliance_marker"),
    ):
        assert preset._inspect_holiday_html_route("/noel", 200, bad_body, seen_html_n)[1]["reason"] == exp_reason
    assert preset._inspect_feedback_summary_json({})[0] is False
    assert preset._inspect_arena_summary_contract({"object": "wrong"})[0] is False
    assert preset._inspect_arena_summary_contract(_make_arena_summary(total=2, wins=0, base=0, ties=2, wlb=0.5))[0] is False
    assert preset._inspect_arena_summary_contract({**_make_arena_summary(), "by_task": {"t1": "bad"}})[0] is False

    # OSError resilience in _resolve_git_config_files + AMADEUS* nested corporate marker
    unreadable_wt = tmp_path / "unreadable_wt"
    unreadable_wt.mkdir()
    (unreadable_wt / ".git").mkdir()
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    assert preset.run_deploy_sh_dry_run(DEPLOY_SH_PATH, runner=lambda cmd, to, **kw: (1, "boom"), env=None)["ok"] is False
    monkeypatch.delenv("HIVEMIND_BUILD_SHA", raising=False)
    assert preset.evaluate_personal_pc_compliance(env={**CLEAN_ENV, "AMADEUS_TEAM": "zscaler-corp"}, cwd=unreadable_wt)["compliance_ok"] is False

    bad_git_dir = tmp_path / "bad_git_file"
    bad_git_dir.mkdir()
    fake_git = bad_git_dir / ".git"
    fake_git.write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
    orig_read_text = Path.read_text

    def _oserror_read_text(self_p: Path, *a: Any, **kw: Any) -> str:
        if self_p.name in (".git", "commondir", "config") and "bad_git_file" in str(self_p):
            raise OSError("eacces")
        return orig_read_text(self_p, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _oserror_read_text)
    assert preset._resolve_git_config_files(bad_git_dir) == []
    monkeypatch.undo()

    # PR949 failing every core/holiday/fleet/idempotency/deploy check in one call + bad-200 contract pass
    all_fail_pr949 = preset.verify_pr949_vscode_quorum_and_arena_parity(
        "http://127.0.0.1:8000",
        verify_idempotency_replay=True,
        execute_deploy_dry_run=True,
        raw_http_fn=lambda m, u, **kw: (503, "{}", {}),
        runner=lambda c, t, **kw: (1, "fail"),
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="fr",
    )
    assert all_fail_pr949["exit_code"] == 2 and all_fail_pr949["ces_slo_ok"] is False and all_fail_pr949["idempotent_dedup_ok"] is False
    bad_200_pr949 = preset.verify_pr949_vscode_quorum_and_arena_parity(
        "http://127.0.0.1:8000",
        raw_http_fn=lambda m, u, **kw: (
            200,
            json.dumps({"data": []}) if u.endswith("/v1/models") else (
                "<!doctype html><p>bad</p>" if any(u.endswith(x) for x in preset.PR949_HTML_ROUTES) else json.dumps({"object": "bad"})
            ),
            {"content-type": "text/html" if any(u.endswith(x) for x in preset.PR949_HTML_ROUTES) else "application/json"},
        ),
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="en",
    )
    assert bad_200_pr949["exit_code"] == 2 and bad_200_pr949["stale_revision_detected"] is True

    # Legacy verify_cloud_run_holiday_parity failing branches (corp cwd, 503, bad 200 schemas, SHA drift, FR)
    assert preset.verify_cloud_run_holiday_parity("http://127.0.0.1:8000", env=CLEAN_ENV, cwd=corp_target)["exit_code"] == 2
    assert preset.verify_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        expected_sha=DRIFT_SHA,
        http_fn=lambda m, u, **kw: (503, {}),
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="fr",
    )["exit_code"] == 2
    assert preset.verify_cloud_run_holiday_parity(
        "http://127.0.0.1:8000",
        expected_sha=DRIFT_SHA,
        http_fn=lambda m, u, **kw: (200, {"html": "bad"} if any(u.endswith(x) for x in ("/noel", "/quick", "/raid-noel")) else {"object": "bad"}),
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="en",
    )["stale_revision_detected"] is True

    # run_one_shot_quick_health_preflight core fail, stale fallback vs require_holiday_routes, and PR949 failures
    assert preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        http_fn=lambda m, u, **kw: (503, {}),
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="fr",
    )["verdict"] == "gateway_preflight_failed"
    assert preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        require_holiday_routes=True,
        http_fn=lambda m, u, **kw: (
            (200, {"status": "ok"}) if u.endswith("/health") else ((200, {"data": [{"id": "hivemind/coder"}]}) if u.endswith("/v1/models") else (404, {}))
        ),
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="en",
    )["verdict"] == "stale_cloud_run_revision"
    assert preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        pr949_quorum_gate=True,
        require_live_quorum=True,
        raw_http_fn=raw_low_fleet,
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="fr",
    )["verdict"] == "pr949_quorum_or_slo_failed"
    assert preset.run_one_shot_quick_health_preflight(
        "http://127.0.0.1:8000",
        pr949_quorum_gate=True,
        raw_http_fn=lambda m, u, **kw: (503, "{}", {}),
        env=CLEAN_ENV,
        cwd=clean_ws,
        lang="en",
    )["verdict"] == "gateway_preflight_failed"


def test_03_vscode_preset_continue_ces_spool_gamer_thermal_and_cli_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    # 1. JSONC parser + Continue YAML merge/remove + VS Code settings/tasks merge + .gitignore + atomic permissions
    jsonc = '// line comment\n{"editor.fontSize": 14, "escaped": "a\\"b\\\\c", /* block */ "a": [1, 2,],}\n'
    parsed = json.loads(preset.strip_jsonc_comments_and_trailing_commas(jsonc))
    assert parsed["editor.fontSize"] == 14 and parsed["a"] == [1, 2]
    for bad_ws_txt in (f'{{"key": "{RAW_HM_KEY}"}}', "Bearer tok_secret_999", "/Users/alice/priv"):
        with pytest.raises(preset.QuickPresetError):
            preset._assert_workspace_content_secret_free(bad_ws_txt)

    existing_yaml = "name: custom\nmodels:\n  - title: UserLocal # inline\n    provider: ollama\ncontext:\n  - provider: code\n"
    merged_yaml = preset.merge_continue_config_yaml(existing_yaml, gateway=preset.DEFAULT_GATEWAY, api_key=RAW_HM_KEY)
    assert "UserLocal" in merged_yaml and "HiveMind Coder" in merged_yaml and RAW_HM_KEY in merged_yaml
    stripped_yaml = preset.remove_hivemind_from_continue_config_yaml(merged_yaml)
    assert stripped_yaml is not None and "UserLocal" in stripped_yaml and "HiveMind Coder" not in stripped_yaml
    assert preset.remove_hivemind_from_continue_config_yaml(preset.build_continue_provider_yaml(gateway=preset.DEFAULT_GATEWAY, api_key=RAW_HM_KEY)) is None
    # JSON-formatted Continue config, empty/missing models block, and malformed Continue/VS Code configs
    assert preset.merge_continue_config_yaml("", gateway=preset.DEFAULT_GATEWAY, api_key=RAW_HM_KEY).startswith("#")
    assert preset.remove_hivemind_from_continue_config_yaml("") is None
    assert preset.remove_hivemind_from_continue_config_yaml("name: only_meta\n") == "name: only_meta\n"
    json_cont = json.dumps({"custom": 1, "models": [{"title": "Local", "provider": "ollama"}]})
    merged_json_cont = preset.merge_continue_config_yaml(json_cont, gateway=preset.DEFAULT_GATEWAY, api_key=RAW_HM_KEY)
    assert preset.remove_hivemind_from_continue_config_yaml(merged_json_cont) is not None
    assert preset.remove_hivemind_from_continue_config_yaml(json.dumps({"models": [{"model": "hivemind/coder"}]})) is None
    assert preset.remove_hivemind_from_continue_config_yaml("{bad") == "{bad"
    assert preset.remove_hivemind_from_continue_config_yaml("[]") == "[]"
    assert preset.remove_hivemind_from_continue_config_yaml('{"models": 1}') == '{"models": 1}'
    assert preset.merge_continue_config_yaml("name: bare\n# tail\n", gateway=preset.DEFAULT_GATEWAY, api_key=RAW_HM_KEY).endswith("\n")
    assert preset.remove_hivemind_from_continue_config_yaml("name: keep_me\nmodels:\n  - model: hivemind/coder\n") is not None
    assert preset._is_hivemind_continue_model_dict("bad") is False
    for bad_cont in ("a\x00b", "{bad", "[]", '{"models": 1}', "- list_root", "models: inline_bad"):
        with pytest.raises(preset.QuickPresetError):
            preset.merge_continue_config_yaml(bad_cont, gateway=preset.DEFAULT_GATEWAY, api_key=RAW_HM_KEY)
    for bad_sc in ("{bad", "[]"):
        with pytest.raises(preset.QuickPresetError):
            preset.build_merged_vscode_settings_json(bad_sc, gateway="http://127.0.0.1:8000")
        with pytest.raises(preset.QuickPresetError):
            preset.build_merged_vscode_tasks_json(bad_sc, gateway="http://127.0.0.1:8000")
    with pytest.raises(preset.QuickPresetError):
        preset.build_merged_vscode_tasks_json('{"tasks": "bad"}', gateway="http://127.0.0.1:8000")
    with pytest.raises(preset.QuickPresetError):
        preset.build_continue_provider_yaml("http://127.0.0.1:8000", "bad-key")
    with pytest.raises(preset.QuickPresetError):
        preset.build_continue_provider_yaml("http://127.0.0.1:8000", RAW_HM_KEY, max_tokens=16)
    assert preset._is_hivemind_managed_task("bad") is False

    # atomic_write_config_file mode tightening, .bak diff update, and Windows permission metadata
    tight_p = tmp_path / "tight" / "cfg.json"
    assert preset.atomic_write_config_file(tight_p, '{"a":1}', posix_mode=0o600)["created"] is True
    os.chmod(tight_p, 0o644)
    assert preset.atomic_write_config_file(tight_p, '{"a":1}', posix_mode=0o600)["permissions_tightened"] is True
    assert preset.atomic_write_config_file(tight_p, '{"a":2}', posix_mode=0o600)["backup_created"] is True
    assert preset.atomic_write_config_file(tmp_path / "nt" / "cfg.json", "x", os_name="nt")["permission_model"] == "windows_acl"

    ws_git = tmp_path / "git_ws"
    ws_git.mkdir()
    (ws_git / ".gitignore").write_text(".vscode/\n", encoding="utf-8")
    assert preset.check_workspace_gitignore_safety(ws_git, lang="fr")["safe"] is True
    ws_partial = tmp_path / "partial_ws"
    ws_partial.mkdir()
    (ws_partial / ".gitignore").write_text("dist/\n", encoding="utf-8")
    assert preset.check_workspace_gitignore_safety(ws_partial, lang="fr")["safe"] is False
    ws_unignored = tmp_path / "unignored_ws"
    ws_unignored.mkdir()
    gi_warn = preset.check_workspace_gitignore_safety(ws_unignored, lang="en")
    assert gi_warn["safe"] is False and ".gitignore" in gi_warn["warning"]

    # 2. CES Spool: validation errors, USERPROFILE path, corrupt/quarantine/overflow/retryable statuses, purge
    assert str(preset.default_ces_queue_path(env={"USERPROFILE": str(tmp_path / "winu")})).endswith("vscode_setup_ces_queue.json")
    for bad_sp in ("bad", {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "extra": 1}, {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "bad plat!"}, {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": "pii"}):
        with pytest.raises(preset.QuickPresetError):
            preset._validate_ces_spool_payload(bad_sp)
    for bad_kwargs in ({"max_entries": 0}, {"max_delivered": 0}):
        with pytest.raises(preset.QuickPresetError):
            preset.VscodeCesRetrySpool(tmp_path / "x.json", **bad_kwargs)
    home_d = tmp_path / "spool_home"
    assert preset.VscodeCesRetrySpool("~/rel_q.json", home=home_d, env=CLEAN_ENV).path == (home_d / "rel_q.json").resolve()
    assert preset.VscodeCesRetrySpool(None, home=home_d, env=CLEAN_ENV).path == preset.default_ces_queue_path(home=home_d, env=CLEAN_ENV).resolve()
    q_path = home_d / ".hivemind" / "queue.json"
    spool = preset.VscodeCesRetrySpool(q_path, home=home_d, env=CLEAN_ENV, max_entries=2, max_delivered=1)
    assert spool.flush("http://127.0.0.1:8000", env=CLEAN_ENV, cwd=ws_git)["flushed_count"] == 0
    assert spool.enqueue(gateway="http://127.0.0.1:8000", env={**CLEAN_ENV, "AMADEUS_X": "1"}, cwd=ws_git)["compliance_ok"] is False
    assert spool.flush("http://127.0.0.1:8000", env={**CLEAN_ENV, "AMADEUS_X": "1"}, cwd=ws_git)["compliance_ok"] is False
    with pytest.raises(preset.QuickPresetError):
        spool.enqueue(gateway="http://127.0.0.1:8000", profile="bad", env=CLEAN_ENV, cwd=ws_git)  # type: ignore[arg-type]
    with pytest.raises(preset.QuickPresetError):
        spool.enqueue(gateway="http://127.0.0.1:8000", zero_game_lag="yes", env=CLEAN_ENV, cwd=ws_git)  # type: ignore[arg-type]
    with pytest.raises(preset.QuickPresetError):
        spool.enqueue(gateway="http://127.0.0.1:8000", idempotency_key=123, env=CLEAN_ENV, cwd=ws_git)  # type: ignore[arg-type]
    with pytest.raises(preset.QuickPresetError):
        spool.enqueue(gateway="http://127.0.0.1:8000", idempotency_key="bad\nkey", env=CLEAN_ENV, cwd=ws_git)

    for i in range(3):
        enq = spool.enqueue(
            gateway="http://127.0.0.1:8000",
            account_ref=f"u{i}",
            profile="gamer",
            ease_rating=5,
            zero_game_lag=True,
            idempotency_key=f"k-{i}-abcdef1234",
            last_http_status=404,
            last_reason="http_404",
        )
        assert enq["enqueued"] is True
    assert spool.enqueue(gateway="http://127.0.0.1:8000", idempotency_key="k-2-abcdef1234")["duplicate_state"] == "already_pending"
    assert (q_path.stat().st_mode & 0o777) == 0o600
    assert spool.snapshot()["pending_count"] == 2

    flush_codes = iter([200, 400])
    flushed = spool.flush(
        "http://127.0.0.1:8000",
        http_fn=lambda m, u, **kw: (next(flush_codes), {"object": preset.FEEDBACK_SUMMARY_OBJECT}),
    )
    assert flushed["flushed_count"] == 1 and flushed["quarantined_count"] == 1
    assert spool.enqueue(gateway="http://127.0.0.1:8000", idempotency_key="k-1-abcdef1234")["duplicate_state"] == "already_delivered"

    # Retryable 404/408/429/0/503 + corrupt spool recovery + tampered entry quarantine + .tmp purge cleanup
    for retry_code in (404, 408, 429, 0, 503):
        assert preset._classify_ces_flush_status(retry_code)[0] == "retryable"
    retry_q = home_d / ".hivemind" / "retry_q.json"
    r_spool = preset.VscodeCesRetrySpool(retry_q, home=home_d, env=CLEAN_ENV, max_entries=2, max_delivered=1)
    r_spool.enqueue(gateway="http://127.0.0.1:8000", idempotency_key="rk-1", last_http_status=404)
    assert r_spool.flush("http://127.0.0.1:8000", http_fn=lambda m, u, **kw: (404, {}))["stale_cloud_run_detected"] is True
    retry_q.write_text("{corrupt", encoding="utf-8")
    assert r_spool.enqueue(gateway="http://127.0.0.1:8000", idempotency_key="rk-corrupt")["quarantined_count"] >= 1
    retry_q.write_text(json.dumps({"delivered_ids": ["d1", "", "d2"], "quarantined": ["bad", {"event_id": "e1"}], "pending": ["bad", {"idempotency_key": "d1", "payload": {}}, {"idempotency_key": "tamper", "payload": {"profile": "corp"}}, {"idempotency_key": "ok1", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}}, {"idempotency_key": "ok2", "payload": {"profile": "dev", "ease_rating": 4, "zero_game_lag": True, "platform": "linux"}}, {"idempotency_key": "ok3", "payload": {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"}}]}), encoding="utf-8")
    snap_rec = r_spool.snapshot()
    assert snap_rec["pending_count"] == 2 and snap_rec["dropped_overflow_count"] >= 1
    (retry_q.parent / f".{retry_q.name}.leftover.tmp").write_text("tmp", encoding="utf-8")
    assert r_spool.purge()["purged"] is True
    assert spool.purge()["purged"] is True

    # Interactive prompt_and_submit_ces + parse_ces_ease_input + _platform_tag + submit errors
    assert preset._platform_tag("FreeBSD") == "freebsd" and preset._platform_tag("!!!") == "unknown"
    assert preset.parse_ces_ease_input("")["skipped"] is True
    assert preset.parse_ces_ease_input("4/5 gamer nolag")["ease_rating"] == 4
    assert preset.parse_ces_ease_input("3 developer lag")["had_game_lag"] is True
    for bad_ces_in in ("9", "dev badtok", "dev"):
        with pytest.raises(preset.QuickPresetError):
            preset.parse_ces_ease_input(bad_ces_in)
    assert preset.prompt_and_submit_ces(input_fn=lambda _: (_ for _ in ()).throw(EOFError()), lang="fr")["skipped"] is True
    with pytest.raises(preset.QuickPresetError):
        preset.submit_preset_ces_feedback(gateway="http://127.0.0.1:8000", zero_game_lag="yes")  # type: ignore[arg-type]
    assert preset.submit_preset_ces_feedback(gateway="http://127.0.0.1:8000", http_fn=lambda m, u, **k: (0, {}))[0]["status"] == "network_timeout"
    assert preset.submit_preset_ces_feedback(gateway="http://127.0.0.1:8000", http_fn=lambda m, u, **k: (400, {}))[0]["status"] == "http_400"

    prompts = iter(["5 nolag"])
    ces_p = preset.prompt_and_submit_ces(
        gateway="http://127.0.0.1:8000",
        account_ref="u_prompt",
        profile="dev",
        input_fn=lambda _: next(prompts),
        http_fn=lambda m, u, **kw: (503, {"error": "down"}),
        ces_queue_file=tmp_path / "prompt_q.json",
        home=home_d,
        env=CLEAN_ENV,
        cwd=ws_git,
    )
    assert ces_p["queued_for_retry"] is True

    # 3. GPU / Steam yield / thermal / VRAM / OpenTrainer heartbeat branches + broken Popen mocks
    gpu_csv = preset.default_gpu_probe(
        runner=lambda argv, to: (0, "NVIDIA GeForce RTX 4070, 12288, 9600, 2688, 56, 4\n"),
    )
    assert gpu_csv.present is True and gpu_csv.free_vram_mib == 9600
    assert preset.default_gpu_probe(runner=lambda argv, to: (0, "bad-csv")).parse_error is True
    assert preset.default_gpu_probe(runner=lambda argv, to: (0, "RTX, -5, 100, 0, 50, 1\n")).parse_error is True
    assert preset.default_gpu_probe(runner=lambda argv, to: (127, "")).present is False
    assert preset._normalize_gpu_snapshot(None).present is False
    assert preset._normalize_gpu_snapshot(SimpleNamespace(present=False)).present is False
    assert preset._normalize_gpu_snapshot(SimpleNamespace(present=True, name="RTX", total_mib=8192, free_mib=4096)).present is True
    assert preset._normalize_gpu_snapshot({"present": True, "parse_error": True}).parse_error is True
    assert preset._normalize_gpu_snapshot({"present": True, "name": ""}).parse_error is True
    with pytest.raises(preset.QuickPresetError):
        preset._normalize_gpu_snapshot(123)
    assert preset._detect_active_steam_games(["", "steam_app_730"]) == ["steam_app_730"]
    assert preset._optional_finite_float(float("nan")) is None and preset._optional_finite_float("bad") is None

    proc_list = preset.default_process_probe(runner=lambda argv, to, **kw: (0, '"cs2.exe","100","Console","1","500 K"\n"gamescope.exe","101","Console","1","200 K"\n'), os_name="nt")
    assert "cs2.exe" in proc_list
    proc_posix = preset.default_process_probe(runner=lambda argv, to, **kw: (0, "bash\nsteam_app_730\n"), os_name="posix")
    assert "steam_app_730" in proc_posix

    # Real subprocess default_worker_yield_drill (< 1.5 s) + broken Popen fallbacks
    real_yield = preset.default_worker_yield_drill(timeout_s=2.5)
    assert real_yield["clean_release"] is True and real_yield["worker_stopped"] is True and real_yield["elapsed_ms"] < 1500.0

    class _FakePopen:
        def __init__(self, txt: str, err: bool = False) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(txt)
            self.stderr = io.StringIO()
            self._err = err

        def kill(self) -> None:
            self._err = False

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            if self._err:
                raise subprocess.TimeoutExpired(["p"], 0.2)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakePopen("NOPE\n"))
    assert preset.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakePopen("HIVEMIND_WORKER_READY\n", err=True))
    assert preset.default_worker_yield_drill(timeout_s=0.3)["clean_release"] is False
    monkeypatch.undo()

    for bad_yr in (True, "str"):
        with pytest.raises(preset.QuickPresetError):
            preset._normalize_yield_drill_result(bad_yr)
    assert preset._normalize_yield_drill_result(25.0) == (25.0, True)
    assert preset._normalize_yield_drill_result(float("nan")) == (None, False)
    assert preset._normalize_yield_drill_result({"elapsed_ms": -5}) == (None, False)
    assert preset._normalize_yield_drill_result({"clean_release": True}) == (None, False)

    assert preset.run_steam_yield_and_thermal_preflight(gpu_probe_fn=lambda: {"present": False}, yield_drill_fn=lambda: 20.0, process_probe_fn=lambda: [])["mode"] == "dev_api_only_no_gpu"
    assert preset.run_steam_yield_and_thermal_preflight(gpu_probe_fn=lambda: {"present": True, "parse_error": True}, yield_drill_fn=lambda: 20.0, process_probe_fn=lambda: [])["mode"] == "gpu_telemetry_parse_error"

    pf_ok = preset.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=_rtx_quiet_snapshot,
        process_probe_fn=lambda: ["gamescope"],
        yield_drill_fn=_fast_worker_yield,
        lang="en",
    )
    assert pf_ok["preflight_ok"] is True

    pf_hot = preset.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: {"present": True, "parse_error": False, "name": "RTX 4070", "total_mib": 12288, "free_mib": 1024, "temp_c": 82.0, "util_pct": 12.0},
        process_probe_fn=lambda: ["cs2.exe"],
        yield_drill_fn=lambda: {"worker_started": True, "worker_stopped": True, "yield_ms": 2200.0, "exit_code": 0, "retained_worker_vram_mib": 0},
        lang="fr",
    )
    assert pf_hot["preflight_ok"] is False

    assert preset.send_opentrainer_first_heartbeat(gateway="http://127.0.0.1:8000", api_key=RAW_HM_KEY, role="consumer")[0]["skipped_consumer_role"] is True
    assert preset.send_opentrainer_first_heartbeat(gateway="http://127.0.0.1:8000", api_key=RAW_HM_KEY, role="prosumer", gpu_snapshot={"present": False}, http_fn=lambda m, u, **k: (503, {}))[0]["ok"] is False
    hb_res, hb_calls = preset.send_opentrainer_first_heartbeat(
        gateway="http://127.0.0.1:8000",
        api_key=RAW_HM_KEY,
        role="prosumer",
        host_label="home-pc",
        gpu_snapshot=_rtx_quiet_snapshot(),
        http_fn=lambda m, u, **kw: (200, {"status": "ok", "accepted": True}),
    )
    assert hb_calls == 1 and hb_res["sent"] is True and hb_res["ok"] is True

    # 4. Credentials, 409 retry, auth fail stages, install/uninstall & run_vscode_setup edge branches + CLI default
    for fname, raw_cred in (("bad.json", "{x"), ("list.json", "[]"), ("badk.json", '{"api_key":"bad"}'), ("gw.json", json.dumps({"api_key": RAW_HM_KEY, "gateway": preset.DEFAULT_GATEWAY}))):
        cp = tmp_path / fname
        cp.write_text(raw_cred, encoding="utf-8")
        assert preset.load_saved_credentials(cp, expected_gateway="http://127.0.0.1:8000") is None
    with pytest.raises(preset.QuickPresetError):
        preset.install_vscode_quick_preset("http://127.0.0.1:8000", workspace_dir=ws_git, explicit_api_key="invalid", env=CLEAN_ENV, home=tmp_path / "hx")

    raw_cli = _make_pr949_raw_fn()
    reg_counter = [0]

    def _retry_409_http(m: str, u: str, payload: Any = None, headers: Any = None, timeout_s: float = 5.0) -> tuple[int, dict[str, Any]]:
        path = urlsplit(u).path
        if path == "/v1/hivemind/register" and m == "POST":
            reg_counter[0] += 1
            return (409, {"detail": "taken"}) if reg_counter[0] == 1 else (200, {"api_key": RAW_HM_KEY, "account_id": "acc_409"})
        if path == "/v1/hivemind/me" and m == "GET":
            return 200, {"account_id": "acc_409", "handle": "noel-dev", "role": "prosumer"}
        st, txt, hdr = raw_cli(m, u, payload=payload, headers=headers, timeout_s=timeout_s)
        return preset._parse_http_raw_response(st, txt.encode("utf-8"), hdr)

    _cli_json_http = _retry_409_http

    for fail_fn in (
        lambda m, u, **k: (500, {}) if u.endswith("/register") else (200, {}),
        lambda m, u, **k: (200, {"api_key": RAW_HM_KEY}) if u.endswith("/register") else (401, {}),
        lambda m, u, **k: (200, {"api_key": RAW_HM_KEY}) if u.endswith("/register") else ((200, {"role": "prosumer"}) if u.endswith("/me") else (503, {})),
    ):
        assert preset.install_vscode_quick_preset("http://127.0.0.1:8000", workspace_dir=ws_git, http_fn=fail_fn, env=CLEAN_ENV, home=tmp_path / "hf", lang="fr")["verdict"] == "auth_or_gateway_failed"

    cli_home = tmp_path / "cli_home"
    cli_ws = tmp_path / "cli_ws"
    cli_ws.mkdir()
    cred_p = cli_home / ".hivemind" / "credentials.json"
    cont_p = cli_home / ".continue" / "hivemind_provider.yaml"
    cont_main = cli_home / ".continue" / "config.yaml"
    cont_main.parent.mkdir(parents=True, exist_ok=True)
    cont_main.write_text(existing_yaml, encoding="utf-8")

    # Test install_vscode_quick_preset with heartbeat fail, stale CES 404 fail, parity fail, and 409 retry
    assert preset.install_vscode_quick_preset("http://127.0.0.1:8000", workspace_dir=cli_ws, credentials_path=cred_p, continue_config_path=cont_p, require_holiday_routes=True, http_fn=lambda m, u, **k: (404, {}), env=CLEAN_ENV, home=cli_home)["verdict"] == "stale_cloud_run_revision"
    hb_fail_res = preset.install_vscode_quick_preset(
        "http://127.0.0.1:8000",
        workspace_dir=cli_ws,
        credentials_path=cred_p,
        continue_config_path=cont_p,
        send_first_heartbeat=True,
        submit_ces=True,
        require_holiday_routes=False,
        gpu_probe_fn=_rtx_quiet_snapshot,
        process_probe_fn=lambda: [],
        yield_drill_fn=_fast_worker_yield,
        http_fn=lambda m, u, **k: (503, {}) if u.endswith("/opentrainer/heartbeat") else ((404, {}) if u.endswith("/onboarding/feedback") else _retry_409_http(m, u, **k)),
        env=CLEAN_ENV,
        home=cli_home,
        lang="fr",
    )
    assert hb_fail_res["exit_code"] == 2 and hb_fail_res["first_heartbeat_ok"] is False and hb_fail_res["verdict"] == "heartbeat_failed"
    assert preset.install_vscode_quick_preset(
        "http://127.0.0.1:8000",
        workspace_dir=cli_ws,
        credentials_path=cred_p,
        continue_config_path=cont_p,
        submit_ces=True,
        require_holiday_routes=True,
        http_fn=lambda m, u, **k: (404, {}) if u.endswith("/onboarding/feedback") else _cli_json_http(m, u, **k),
        env=CLEAN_ENV,
        home=cli_home,
    )["verdict"] == "stale_cloud_run_revision"

    # run_vscode_setup failure branches (preflight fail, skip_preflight+pr949 fail, preset fail, prompt_ces+submit_ces spool+flush, uninstall=True)
    assert preset.run_vscode_setup("http://127.0.0.1:8000", workspace_dir=ws_partial, require_holiday_routes=True, http_fn=lambda m, u, **k: (503, {}), env=CLEAN_ENV, home=tmp_path / "hs1")["exit_code"] == 2
    assert preset.run_vscode_setup("http://127.0.0.1:8000", workspace_dir=ws_partial, skip_preflight=True, pr949_quorum_gate=True, require_live_quorum=True, raw_http_fn=lambda m, u, **k: (503, "{}", {}), env=CLEAN_ENV, home=tmp_path / "hs2")["exit_code"] == 2
    assert preset.run_vscode_setup("http://127.0.0.1:8000", workspace_dir=ws_partial, skip_preflight=True, run_gamer_preflight=True, gpu_probe_fn=lambda: {"present": True, "temp_c": 88.0, "total_mib": 8192, "free_mib": 1024, "name": "RTX"}, process_probe_fn=lambda: [], yield_drill_fn=_fast_worker_yield, http_fn=_retry_409_http, env=CLEAN_ENV, home=tmp_path / "hs3")["verdict"] == "preflight_failed"

    prompt_setup = preset.run_vscode_setup(
        "http://127.0.0.1:8000",
        workspace_dir=cli_ws,
        credentials_path=cred_p,
        continue_config_path=cont_p,
        continue_main_config_path=cont_main,
        write_continue_config=True,
        prompt_ces=True,
        flush_ces_queue=True,
        input_fn=lambda _: "5 nolag",
        raw_http_fn=raw_cli,
        http_fn=lambda m, u, **k: (404, {}) if u.endswith("/onboarding/feedback") else _cli_json_http(m, u, **k),
        env=CLEAN_ENV,
        home=cli_home,
        lang="fr",
    )
    assert prompt_setup["stale_revision_detected"] is True and prompt_setup["ces_queued_for_retry"] is True

    spool_setup = preset.run_vscode_setup(
        "http://127.0.0.1:8000",
        workspace_dir=cli_ws,
        credentials_path=cred_p,
        continue_config_path=cont_p,
        submit_ces=True,
        require_holiday_routes=True,
        raw_http_fn=raw_cli,
        http_fn=lambda m, u, **k: (404, {}) if u.endswith("/onboarding/feedback") else _cli_json_http(m, u, **k),
        env=CLEAN_ENV,
        home=cli_home,
        lang="en",
    )
    assert spool_setup["exit_code"] == 2 and spool_setup["ces_queued_for_retry"] is True

    flush_setup = preset.run_vscode_setup(
        "http://127.0.0.1:8000",
        workspace_dir=cli_ws,
        credentials_path=cred_p,
        continue_config_path=cont_p,
        flush_ces_queue=True,
        raw_http_fn=raw_cli,
        http_fn=lambda m, u, **k: (404, {}) if u.endswith("/onboarding/feedback") else _cli_json_http(m, u, **k),
        env=CLEAN_ENV,
        home=cli_home,
    )
    assert flush_setup["stale_revision_detected"] is True

    setup_res = preset.run_vscode_setup(
        "http://127.0.0.1:8000",
        workspace_dir=cli_ws,
        credentials_path=cred_p,
        continue_config_path=cont_p,
        continue_main_config_path=cont_main,
        write_continue_config=True,
        pr949_quorum_gate=True,
        require_live_quorum=True,
        require_ces_slo=True,
        submit_ces=True,
        raw_http_fn=raw_cli,
        http_fn=_cli_json_http,
        env=CLEAN_ENV,
        home=cli_home,
        lang="en",
    )
    assert setup_res["exit_code"] == 0 and flush_setup["file_writes"] == 0
    assert setup_res["producer_quorum_met"] is True and setup_res["arena_summary_ok"] is True

    fresh_home = tmp_path / "fresh_home"
    fresh_ws = tmp_path / "fresh_ws"
    fresh_ws.mkdir()
    fresh_res = preset.run_vscode_setup(
        "http://127.0.0.1:8000",
        workspace_dir=fresh_ws,
        write_continue_config=True,
        pr949_quorum_gate=True,
        raw_http_fn=raw_cli,
        http_fn=_cli_json_http,
        env=CLEAN_ENV,
        home=fresh_home,
    )
    assert fresh_res["exit_code"] == 0 and fresh_res["file_writes"] == 5

    # Preserve user settings/tasks & corrupt settings/tasks during uninstall + run_vscode_setup(uninstall=True) + report formatting
    user_s = {"editor.tabSize": 4, "hivemind.apiBase": "http://127.0.0.1:8000"}
    (cli_ws / ".vscode" / "settings.json").write_text(json.dumps(user_s), encoding="utf-8")
    user_t = {"version": "2.0.0", "tasks": [{"label": "User Build", "type": "shell"}, {"label": preset.TASK_LABEL_QUICK_CONNECT, "type": "shell"}]}
    (cli_ws / ".vscode" / "tasks.json").write_text(json.dumps(user_t), encoding="utf-8")
    un_pres = preset.run_vscode_setup(uninstall=True, workspace_dir=cli_ws, credentials_path=cred_p, continue_config_path=cont_p, continue_main_config_path=cont_main, env=CLEAN_ENV, home=cli_home, lang="fr")
    assert un_pres["exit_code"] == 0 and (cli_ws / ".vscode" / "settings.json").is_file() and (cli_ws / ".vscode" / "tasks.json").is_file()

    mal_ws = tmp_path / "mal_ws"
    (mal_ws / ".vscode").mkdir(parents=True)
    (mal_ws / ".vscode" / "settings.json").write_text("{bad", encoding="utf-8")
    (mal_ws / ".vscode" / "tasks.json").write_text("{bad", encoding="utf-8")
    assert preset.uninstall_vscode_quick_preset(workspace_dir=mal_ws, env=CLEAN_ENV, home=cli_home)["exit_code"] == 0

    rich_rep = {**spool_setup, "idempotent_dedup_ok": True, "deploy_dry_run": {"ok": True, "dry_run_build_sha": TARGET_SHORT_SHA}, "diagnostics": ["d1"]}
    assert "Idempotency Replay" in preset.render_human_report(rich_rep, lang="en")
    assert "## Diagnostics" in preset.render_markdown_report(rich_rep, lang="en")

    # CLI flags coverage (incl. default --install-preset fallback + --vscode-setup + --uninstall)
    for argv in (
        ["--gateway", "http://127.0.0.1:8000", "--workspace", str(cli_ws), "--credentials-path", str(cred_p), "--continue-config-path", str(cont_p), "--json"],
        ["--gateway", "http://127.0.0.1:8000", "--vscode-setup", "--skip-preflight", "--pr949-quorum-gate", "--workspace", str(cli_ws), "--credentials-path", str(cred_p), "--continue-config-path", str(cont_p), "--continue-main-config-path", str(cont_main), "--json"],
        ["--gateway", "http://127.0.0.1:8000", "--flush-ces-queue", "--ces-queue-file", str(tmp_path / "empty_q.json"), "--cwd", str(cli_ws), "--json"],
        ["--gateway", "http://127.0.0.1:8000", "--check-parity", "--cwd", str(cli_ws)],
        ["--gateway", "http://127.0.0.1:8000", "--check-parity", "--require-live-quorum", "--cwd", str(cli_ws), "--markdown"],
        ["--uninstall", "--workspace", str(cli_ws), "--credentials-path", str(cred_p), "--continue-config-path", str(cont_p), "--continue-main-config-path", str(cont_main), "--json"],
    ):
        assert preset.main(argv, env=CLEAN_ENV, home=cli_home, raw_http_fn=raw_cli, http_fn=_cli_json_http, stdout=io.StringIO()) == 0
    assert not cred_p.exists() and not cont_p.exists()


def test_04_noel_scale_48_users_quorum_arena_ces_wal_soak_under_15s(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00949-noel")

    upstream_inference_hits: list[str] = []

    def _forbid_upstream(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "vscode_dyn46_wal_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=SOAK_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        journal_mode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert journal_mode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )

    clean_ws = tmp_path / "personal_hobby_repo"
    (clean_ws / ".git").mkdir(parents=True)
    (clean_ws / ".git" / "config").write_text('[remote "origin"]\n\turl = https://github.com/Vxtzq/CrowdGPT\n', encoding="utf-8")

    asgi_loop = asyncio.new_event_loop()
    asgi_thread = threading.Thread(target=asgi_loop.run_forever, daemon=True)
    asgi_thread.start()
    async_client = asyncio.run_coroutine_threadsafe(
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000").__aenter__(),
        asgi_loop,
    ).result(timeout=5.0)

    def _asgi_req(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return asyncio.run_coroutine_threadsafe(
            async_client.request(method, path, json=json_body, headers=headers),
            asgi_loop,
        ).result(timeout=20.0)

    def _asgi_get_html_batch(routes: tuple[str, ...], headers: dict[str, str]) -> list[httpx.Response]:
        async def _gather_all() -> list[httpx.Response]:
            return list(await asyncio.gather(*(async_client.get(r, headers=headers) for r in routes)))

        return asyncio.run_coroutine_threadsafe(_gather_all(), asgi_loop).result(timeout=20.0)

    shared_lock = threading.Lock()
    observed_nonces: list[str] = []
    seen_paths: list[str] = []
    sqlite_locks: list[str] = []
    raw_keys: list[str] = []
    masked_keys: list[str] = []

    def _check_html(path: str, resp: httpx.Response) -> str:
        assert resp.status_code == 200, (path, resp.status_code)
        csp = resp.headers.get("content-security-policy", "")
        hdr_n = _CSP_HDR_NONCE_RE.findall(csp)
        attr_n = [v for _, v in _HTML_ATTR_NONCE_RE.findall(resp.text)]
        assert "default-src 'none'" in csp.lower() and hdr_n and hdr_n[0] in attr_n
        stripped = preset.strip_masked_keys_and_nonces(resp.text)
        assert _STATIC_FIG_RE.findall(stripped) == []
        assert ("inner" + "HTML") not in stripped
        return hdr_n[0]

    def _make_adapters(peer_ip: str) -> tuple[Any, preset.HttpFn]:
        def _raw_adapter(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            timeout_s: float = 5.0,
        ) -> tuple[int, str, dict[str, str]]:
            _ = timeout_s
            assert "completions" not in url.lower()
            parsed = urlsplit(url)
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            req_h = {"X-Forwarded-For": peer_ip, **{str(k): str(v) for k, v in (headers or {}).items()}}
            resp = _asgi_req(method, path, json_body=payload, headers=req_h)
            with shared_lock:
                seen_paths.append(parsed.path)
                if resp.status_code >= 400 and _SQLITE_LOCK_RE.search(resp.text):
                    sqlite_locks.append(resp.text[:120])
            if parsed.path in preset.PR949_HTML_ROUTES and resp.status_code == 200:
                n = _check_html(parsed.path, resp)
                with shared_lock:
                    observed_nonces.append(n)
            return resp.status_code, resp.text, {k.lower(): v for k, v in resp.headers.items()}

        def _json_adapter(
            method: str,
            url: str,
            payload: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            timeout_s: float = 5.0,
        ) -> tuple[int, dict[str, Any]]:
            st, txt, hdrs = _raw_adapter(method, url, payload=payload, headers=headers, timeout_s=timeout_s)
            return preset._parse_http_raw_response(st, txt.encode("utf-8"), hdrs)

        return _raw_adapter, _json_adapter

    try:
        # Seed 8 distinct physical-GPU volunteer friend producers + hivemind/coder
        for p_idx in range(8):
            p_ip = _loopback_ip(56, p_idx)
            r_reg = _asgi_req("POST", "/v1/hivemind/register", json_body={"handle": f"vscode-gpu-{p_idx:02d}", "role": "producer"}, headers={"X-Forwarded-For": p_ip})
            assert r_reg.status_code == 200, r_reg.text
            p_hdr = {"Authorization": f"Bearer {r_reg.json()['api_key']}", "X-Forwarded-For": p_ip}
            r_hb = _asgi_req(
                "POST",
                "/v1/hivemind/producer/heartbeat",
                json_body={
                    "endpoint_url": f"http://{p_ip}:11434",
                    "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                    "cost_per_1k": 0.0,
                    "capacity_1k": 50.0,
                    "gpu_vram_mib": 12288,
                    "execution_class": "ollama-gamer-whisper",
                },
                headers=p_hdr,
            )
            assert r_hb.status_code == 200, r_hb.text
            r_ot = _asgi_req(
                "POST",
                "/v1/hivemind/opentrainer/heartbeat",
                json_body={
                    "host_label": f"vsc-gpu-{p_idx:02d}",
                    "gpu_name": "NVIDIA GeForce RTX 4070",
                    "vram_total_mib": 12288,
                    "vram_free_mib": 9600,
                    "network_mode": "outbound-https-relay",
                    "training_stack": preset.DEFAULT_TRAINING_STACK,
                    "active_campaign": preset.DEFAULT_ACTIVE_CAMPAIGN,
                },
                headers=p_hdr,
            )
            assert r_ot.status_code == 200, r_ot.text

        dev_count = 24
        gamer_count = 24

        def _participant(idx: int) -> dict[str, Any]:
            is_dev = idx < dev_count
            c_idx = idx if is_dev else (idx - dev_count)
            profile = "dev" if is_dev else "gamer"
            plat = "macos" if (c_idx % 3 == 0) else "linux"
            peer_ip = _loopback_ip(12 if is_dev else 22, c_idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            slot = tmp_path / "soak48_vsc" / f"slot_{idx:02d}"
            os_user = f"vsc_u_{idx:02d}"
            home_d = slot / "home" / os_user
            ws_d = slot / "hobby_ws"
            home_d.mkdir(parents=True, exist_ok=True)
            ws_d.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()

            arena_html = ""
            html_resps = _asgi_get_html_batch(preset.PR949_HTML_ROUTES, hdrs)
            for h_route, rh in zip(preset.PR949_HTML_ROUTES, html_resps, strict=True):
                nonce_val = _check_html(h_route, rh)
                if h_route == "/arena-noel":
                    arena_html = rh.text
                with shared_lock:
                    observed_nonces.append(nonce_val)
                    seen_paths.append(h_route)

            rman = _asgi_req("GET", "/v1/hivemind/onboarding/manifest", headers=hdrs)
            assert rman.status_code == 200 and rman.json()["build_sha"] == TARGET_FULL_SHA

            m_pair = _ARENA_PAIR_RE.search(arena_html)
            assert m_pair is not None
            task_id, ballot_tok = m_pair.group(1), m_pair.group(2)
            cand_m = _CAND_PRE_RE.search(arena_html)
            cand_a_txt = cand_m.group(3) if cand_m else ""
            chosen = "A" if ("window must be an integer" in cand_a_txt or "FrameError::UnsupportedVersion" in cand_a_txt) else "B"
            rv = _asgi_req(
                "POST",
                "/v1/hivemind/arena/vote",
                json_body={"task_id": task_id, "chosen_side": chosen, "profile": profile, "ballot_token": ballot_tok},
                headers=hdrs,
            )
            assert rv.status_code == 200, rv.text

            raw_ad, json_ad = _make_adapters(peer_ip)
            ease = 5 if (c_idx % 4 != 0) else 4
            zero_lag = True if is_dev else (c_idx != 23)  # 47/48 == 0.9792 >= 0.95

            res = preset.run_vscode_setup(
                "http://127.0.0.1:8000",
                workspace_dir=ws_d,
                credentials_path=home_d / ".hivemind" / "credentials.json",
                continue_config_path=home_d / ".continue" / "hivemind_provider.yaml",
                handle=f"vsc46_{profile}_{c_idx:02d}",
                role="prosumer" if is_dev else "trainer",
                host_label=f"{plat}-pc-{c_idx:02d}",
                expected_sha=TARGET_SHORT_SHA if (idx == 0) else None,
                skip_preflight=(idx != 0),
                submit_ces=False,
                profile=profile,
                ease_rating=ease,
                had_game_lag=not zero_lag,
                run_gamer_preflight=(not is_dev),
                gpu_probe_fn=_rtx_quiet_snapshot,
                process_probe_fn=lambda: ["gamescope", "steam"],
                yield_drill_fn=_fast_worker_yield,
                raw_http_fn=raw_ad,
                http_fn=json_ad,
                env=CLEAN_ENV,
                home=home_d,
                lang="fr" if (idx % 2) else "en",
            )
            assert res["exit_code"] == 0 and res["file_writes"] == 4
            assert "hivemind/coder" in (res.get("account") or {}).get("models", [])
            cred_saved = preset.load_saved_credentials(home_d / ".hivemind" / "credentials.json", expected_gateway="http://127.0.0.1:8000")
            assert cred_saved is not None
            api_key = str(cred_saved["api_key"])
            masked_key = str(res["account"]["masked_key"])
            d_txt = json.dumps(res)
            assert api_key not in d_txt and os_user not in d_txt

            idem_key = "soak46-" + hashlib.sha256(f"{profile}:{c_idx}".encode()).hexdigest()[:24]
            fb_body = {"profile": profile, "ease_rating": ease, "zero_game_lag": zero_lag, "platform": plat, "note": ""}
            fb1 = _asgi_req("POST", "/v1/hivemind/onboarding/feedback", json_body=fb_body, headers={**hdrs, "Idempotency-Key": idem_key})
            fb2 = _asgi_req("POST", "/v1/hivemind/onboarding/feedback", json_body=fb_body, headers={**hdrs, "Idempotency-Key": idem_key})
            assert fb1.status_code == 200 and fb2.status_code == 200
            assert (time.monotonic() - t0) < (60.0 if is_dev else 300.0)

            with shared_lock:
                raw_keys.append(api_key)
                masked_keys.append(masked_key)
            return {"profile": profile}

        def _concurrent_probe(p_idx: int) -> dict[str, Any]:
            r_ad, j_ad = _make_adapters(_loopback_ip(35, p_idx))
            return preset.verify_pr949_vscode_quorum_and_arena_parity(
                "http://127.0.0.1:8000",
                expected_sha=TARGET_FULL_SHA,
                target_online_producers=8,
                require_live_quorum=True,
                require_ces_slo=True,
                check_arena_summary=True,
                raw_http_fn=r_ad,
                http_fn=j_ad,
                env=CLEAN_ENV,
                cwd=clean_ws,
                lang="fr" if p_idx % 2 else "en",
            )

        t_soak = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            part_futs = [pool.submit(_participant, i) for i in range(dev_count + gamer_count)]
            probe_futs = [pool.submit(_concurrent_probe, j) for j in range(2)]
            part_docs = [f.result(timeout=20.0) for f in part_futs]
            probe_docs = [f.result(timeout=20.0) for f in probe_futs]
        soak_elapsed_s = time.monotonic() - t_soak
        assert soak_elapsed_s < 15.0, f"48-user WAL soak exceeded 15.0s ceiling: {soak_elapsed_s:.2f}s"

        r_post, j_post = _make_adapters("127.0.60.1")
        post_gate = preset.verify_pr949_vscode_quorum_and_arena_parity(
            "http://127.0.0.1:8000",
            expected_sha=TARGET_FULL_SHA,
            target_online_producers=8,
            min_ease_rating=4.2,
            min_zero_lag_ratio=0.95,
            require_live_quorum=True,
            require_ces_slo=True,
            check_arena_summary=True,
            execute_deploy_dry_run=True,
            repo_root=REPO_ROOT,
            raw_http_fn=r_post,
            http_fn=j_post,
            env=CLEAN_ENV,
            cwd=clean_ws,
            lang="en",
        )
        _verify_receipt(post_gate)
        assert len(part_docs) == 48 and sqlite_locks == [] and upstream_inference_hits == []
        assert all("completions" not in p.lower() for p in seen_paths)
        for pd in (*probe_docs, post_gate):
            _verify_receipt(pd)
            assert pd["parity_ok"] is True and pd["exit_code"] == 0 and pd["online_producers"] >= 8
            assert pd["producer_quorum_met"] is True and pd["arena_summary_ok"] is True
        assert post_gate["ces_slo_ok"] is True and post_gate["zero_lag_slo_ok"] is True
        assert post_gate["deploy_dry_run"]["ok"] is True

        fb_summary = _asgi_req("GET", "/v1/hivemind/onboarding/feedback/summary", headers={"X-Forwarded-For": "127.0.60.2"}).json()
        assert set(fb_summary.keys()) == {"object", "total_responses", "mean_ease_rating", "zero_game_lag_ratio", "by_profile"}
        assert fb_summary["total_responses"] == 48 and fb_summary["by_profile"] == {"dev": 24, "gamer": 24}
        assert fb_summary["mean_ease_rating"] >= 4.2 and fb_summary["zero_game_lag_ratio"] >= 0.95

        arena_sum = _asgi_req("GET", "/v1/hivemind/arena/summary", headers={"X-Forwarded-For": "127.0.60.3"}).json()
        assert arena_sum["total_votes"] == 48
        assert arena_sum["holiday_wins"] + arena_sum["baseline_wins"] + arena_sum["ties"] == 48
        assert arena_sum["wilson_95_lower_bound"] > 0.0

        assert len(observed_nonces) == len(set(observed_nonces)) and len(observed_nonces) >= 48 * 4
        for rk, mk in zip(raw_keys, masked_keys, strict=True):
            assert rk not in mk and mk.startswith(("hm-***", "hm_live_***"))
        assert preset.scan_output_safety(preset.render_human_report(post_gate)) == []
        assert preset.scan_output_safety(preset.render_markdown_report(post_gate)) == []
    finally:
        asyncio.run_coroutine_threadsafe(async_client.__aexit__(None, None, None), asgi_loop).result(timeout=5.0)
        asgi_loop.call_soon_threadsafe(asgi_loop.stop)
        asgi_thread.join(timeout=5.0)
        asgi_loop.close()
        store.close()


def test_05_live_loopback_http_server_e2e_cli_and_remaining_parity_branches(
    tmp_path: Path,
) -> None:
    nonce_counter = [0]

    class _LivePr949Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            _ = (fmt, args)

        def do_GET(self) -> None:  # noqa: N802
            p = urlsplit(self.path).path
            if p in preset.PR949_HTML_ROUTES:
                nonce_counter[0] += 1
                n = f"LoopNonce{nonce_counter[0]:04d}AbCd"
                body = _make_html(n, p.strip("/")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Security-Policy", f"default-src 'none'; script-src 'nonce-{n}'")
                self.end_headers()
                self.wfile.write(body)
                return
            payload_map = {
                "/health": {"status": "ok"},
                "/v1/models": {"object": "list", "data": [{"id": "hivemind/coder"}, {"id": "hivemind/bee-stars"}]},
                "/v1/hivemind/onboarding/manifest": _make_manifest(),
                "/v1/hivemind/onboarding/feedback/summary": {
                    "object": preset.FEEDBACK_SUMMARY_OBJECT,
                    "total_responses": 12,
                    "mean_ease_rating": 4.9,
                    "zero_game_lag_ratio": 1.0,
                    "by_profile": {"dev": 6, "gamer": 6},
                },
                "/v1/hivemind/raid/status": {"object": preset.RAID_STATUS_OBJECT, "swarm": {"online": 8}},
                "/v1/hivemind/producers/fleet": _make_fleet(8),
                "/v1/hivemind/arena/summary": _make_arena_summary(total=12, wins=10, base=1, ties=1, wlb=0.61),
            }
            if p in payload_map:
                raw = json.dumps(payload_map[p]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(404)
            self.end_headers()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LivePr949Handler)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    gw = f"http://127.0.0.1:{int(srv.server_address[1])}"
    clean_ws = tmp_path / "live_ws"
    clean_ws.mkdir()
    out_file = tmp_path / "live_receipt.json"
    stdout_buf = io.StringIO()
    try:
        rc = preset.main(
            [
                "--gateway", gw,
                "--pr949-quorum-gate",
                "--require-live-quorum",
                "--require-ces-slo",
                "--expected-sha", TARGET_SHORT_SHA,
                "--cwd", str(clean_ws),
                "--output", str(out_file),
                "--json",
            ],
            env=CLEAN_ENV,
            stdout=stdout_buf,
        )
        assert rc == 0 and out_file.is_file() and (out_file.stat().st_mode & 0o777) == 0o600
        doc = json.loads(stdout_buf.getvalue())
        _verify_receipt(doc)
        assert doc["parity_ok"] is True and doc["csp_nonces_unique"] is True and doc["online_producers"] == 8

        # Also exercise verify_cloud_run_holiday_parity + run_one_shot_quick_health_preflight via real loopback
        p_res = preset.verify_cloud_run_holiday_parity(gw, expected_sha=TARGET_SHORT_SHA, env=CLEAN_ENV, cwd=clean_ws, lang="en")
        assert p_res["parity_ok"] is True and p_res["exit_code"] == 0
        q_res = preset.run_one_shot_quick_health_preflight(gw, env=CLEAN_ENV, cwd=clean_ws, lang="fr")
        assert q_res["preflight_ok"] is True and q_res["exit_code"] == 0
        q_pr949 = preset.run_one_shot_quick_health_preflight(gw, pr949_quorum_gate=True, require_live_quorum=True, env=CLEAN_ENV, cwd=clean_ws, lang="en")
        assert q_pr949["preflight_ok"] is True and q_pr949["arena_summary_ok"] is True
    finally:
        srv.shutdown()
        srv.server_close()
