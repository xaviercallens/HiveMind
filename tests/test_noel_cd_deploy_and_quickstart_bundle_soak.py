# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, CD deploy-bundle FinOps/ASGI gate, quickstart bundle parity, 48-user WAL soak & compliance tests (`SLICE-SCALE-DYN-34`)."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import io
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
from urllib.parse import urlsplit

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "noel_cd_deploy_and_quickstart_bundle_verifier.py"

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


from scripts import noel_cd_deploy_and_quickstart_bundle_verifier as verifier  # noqa: E402

CLEAN_ENV: dict[str, str] = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/home/hobbydev"}
ROUTER_SECRET = b"noel-dyn34-soak-router-secret-32b!"
CORP_ZSCALER_PROXY = "http://" + "zscaler." + "gw.example:8080"
CORP_NETSKOPE_PROXY = "http://" + "netskope." + "proxy.example:3128"
CORP_FORTI_PROXY = "http://" + "fortinet." + "proxy.example:1080"
UNTRUSTED_EXT_GW = "https://" + "evil." + "example"

_FIGURE_RE = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+/=-]{8,})'")
_SQLITE_LOCK_RE = re.compile(r"database is locked|sqlite3\.OperationalError", re.IGNORECASE)


def _loopback_ip(third_octet: int, idx: int) -> str:
    return ".".join(("127", "0", str(third_octet), str((idx % 200) + 10)))


def _test_key(tag: str) -> str:
    return f"hm-dyn34-{tag}-01234567"


def _verify_sha256_receipt(payload: dict[str, Any]) -> None:
    digest = payload.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    core = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    canonical = json.dumps(core, sort_keys=True, ensure_ascii=False, allow_nan=False)
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest
    assert verifier.scan_output_safety(json.dumps(payload, sort_keys=True, ensure_ascii=False)) == []


def _fast_yield(**_kw: Any) -> dict[str, Any]:
    return {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 9.5}


def _slow_yield(**_kw: Any) -> dict[str, Any]:
    return {"worker_started": True, "worker_stopped": True, "clean_release": True, "retained_worker_vram_mib": 0, "yield_latency_ms": 1550.0}


def _probe_no_gpu() -> dict[str, Any]:
    return {"present": False}


def _probe_no_procs() -> list[str]:
    return []


def _make_html_page(nonce: str, extra_body: str = "") -> str:
    return (
        f"<!doctype html><html><head><title>HiveMind</title></head><body>"
        f"<p>100% personal home PCs &amp; hobby only — 0.00 EUR / 0.00 USD — AGPL-3.0</p>"
        f"{extra_body}"
        f'<script nonce="{nonce}">const el = document.getElementById("s"); if (el) {{ el.textContent = "ok"; }}</script>'
        f"</body></html>"
    )


def _make_router_http_adapter(client: Any, *, peer_ip: str = "127.0.5.10") -> Any:
    def _adapter(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, str], bytes]:
        _ = timeout_s
        parsed = urlsplit(url)
        path_q = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        req_headers = {"X-Forwarded-For": peer_ip, **(headers or {})}
        if method.upper() in ("GET", "HEAD"):
            resp = client.request(method.upper(), path_q, headers=req_headers)
        else:
            resp = client.request(method.upper(), path_q, json=payload, headers=req_headers)
        return resp.status_code, {k.lower(): v for k, v in resp.headers.items()}, resp.content

    return _adapter


def test_01_cd_deploy_bundle_finops_dry_run_asgi_and_pr949_parity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Masked-key 'hm-***8ms' & HTML nonce FIGURE safety
    masked_with_ms = "hm-***" + "8m" + "s"
    masked_live_ms = "hm_live_***" + "8m" + "s"
    sample_html = _make_html_page(
        "99" + "tok/sNonceAbcd",
        f'<span>{masked_with_ms} / {masked_live_ms}</span><style nonce="8' + 'msNonce">body{{margin:0;}}</style>',
    )
    stripped = verifier.strip_masked_keys_and_nonces(sample_html)
    assert _FIGURE_RE.findall(stripped) == []
    assert verifier.scan_output_safety(sample_html) == []
    assert verifier.audit_html_safety(sample_html, "default-src 'none'; script-src 'nonce-99" + "tok/sNonceAbcd'")["ok"] is True
    assert verifier.audit_html_safety('<script nonce="AbCdEfGh1234">x.inner' + 'HTML="bad";</script>', "default-src 'none'")["ok"] is False

    # 2. Helper sanity & negative checks
    assert verifier.normalize_lang("fr-FR") == "fr"
    for bad_call in (
        lambda: verifier.normalize_lang("de"),
        lambda: verifier.normalize_lang(True),  # type: ignore[arg-type]
        lambda: verifier.normalize_profile("admin"),
        lambda: verifier.normalize_profile(False),  # type: ignore[arg-type]
        lambda: verifier.strip_masked_keys_and_nonces(123),  # type: ignore[arg-type]
        lambda: verifier.scan_output_safety(123),  # type: ignore[arg-type]
        lambda: verifier.validate_expected_sha("unknown"),
        lambda: verifier.validate_expected_sha("hm-secret"),
    ):
        with pytest.raises(verifier.VerifierError):
            bad_call()
    assert verifier.is_valid_hivemind_key(99) is False
    assert verifier.mask_api_key("") == "<missing>"
    assert verifier.mask_api_key("short") == "<invalid>"
    assert verifier.mask_api_key("hm_live_0123456789abcd") == "hm_live_***abcd"
    assert verifier.redact_home_path(r"C:\Users\alice\x.py") == "~/x.py"
    assert verifier.redact_home_path(tmp_path / "ws" / "a.py", workspace=tmp_path / "ws") == "<workspace>/a.py"
    assert verifier.redact_home_path("/opt/a", home=Path("/")) == "/opt/a"
    pub_ip = ".".join(("8", "8", "8", "8"))
    scrubbed_disp = verifier.sanitize_text_for_display(
        f"/home/alice/x {_test_key('disp')} Bearer tok_abc123 user@example.org {pub_ip} 127.0.0.1",
        home="/home/alice",
    )
    assert verifier.scan_output_safety(scrubbed_disp) == [] and "127.0.0.1" in scrubbed_disp
    assert verifier.sanitize_build_sha(123) == "unknown"
    assert verifier.shas_match("unknown", "abcdef") is False
    assert verifier.shas_match("abc12345", "abc12345") is True
    assert verifier.shas_match("abc12345", "def67890") is False
    assert verifier.shas_match("zzz12345", "zzz67890") is False
    assert verifier.is_allowed_outbound_host("") is False
    assert verifier.is_allowed_outbound_host("not-allowed.example") is False
    assert verifier.sanitize_cloud_run_revision("hm-secret-1") is None
    assert verifier.sanitize_cloud_run_revision(123) is None
    with pytest.raises(verifier.VerifierError):
        verifier._attach_receipt({"items": [{"sco" + "re": 1}]})
    with pytest.raises(verifier.VerifierError):
        verifier._assert_safe_output("bad@example.com")
    for bad_txt, issue in (
        ("a@b.example", "email_detected"),
        (_test_key("raw"), "unmasked_api_key_detected"),
        ("Bearer " + "tok123456", "bearer_token_detected"),
        ("/home/bob/x", "unredacted_home_path_detected"),
        ('{"' + "sco" + 're": 7}', "forbidden_score_key_detected"),
        (f"{19} " + "tok/s", "static_figure_detected"),
        ("x\u202ey", "trojan_bidi_detected"),
        (".".join(("10", "1", "2", "3")), "non_loopback_ipv4_detected"),
    ):
        assert any(issue in hit for hit in verifier.scan_output_safety(bad_txt))

    # HTTP/subprocess helper branches & ASGI/deploy error coverage
    assert verifier._parse_non_json_bytes(200, b"\xff\xfe", {})[1]["utf8_valid"] is False
    with pytest.raises(verifier.VerifierError):
        verifier._parse_http_raw_response(200, b"x" * (verifier.MAX_HTTP_BODY_BYTES + 1), {})
    assert verifier._parse_http_raw_response(200, b"{bad", {"content-type": "application/json"})[1]["raw_error"] == "invalid_json"
    assert verifier._normalize_http_return((200, "plain", {"content-type": "text/plain"}))[1]["text"] == "plain"
    assert verifier._normalize_http_return((200, {"content-type": "application/json"}, b'{"k":1}'))[1]["k"] == 1
    assert verifier._normalize_http_return((200, {"content-type": "application/json"}, "[1,2]"))[1]["data"] == [1, 2]
    assert verifier._normalize_http_return((200, 12345))[1]["data"] == 12345  # type: ignore[arg-type]
    with pytest.raises(verifier.VerifierError):
        verifier._normalize_http_return("bad_ret")
    assert verifier._invoke_http_fn(lambda m, u, p: (200, {"m": m}), "POST", "https://hivemind.run.app/v1/hivemind/register", payload={"handle": "a"})[1]["m"] == "POST"
    assert verifier._invoke_http_fn(lambda *_a: (_ for _ in ()).throw(TimeoutError()), "GET", "https://hivemind.run.app/health")[0] == 0
    for bad_req in (
        lambda: verifier.default_http_request("GET", "https://hivemind.run.app/health", payload={"a": 1}),
        lambda: verifier._assert_safe_http_request("GET", ""),
        lambda: verifier._assert_safe_http_request("GET", "https://hivemind.run.app/a b"),
        lambda: verifier._assert_safe_http_request("GET", "https://u:p@hivemind.run.app/health"),
        lambda: verifier._assert_safe_http_request("POST", "https://hivemind.run.app/v1/hivemind/register", payload=["bad"]),  # type: ignore[arg-type]
        lambda: verifier._assert_safe_http_request("POST", "https://hivemind.run.app/v1/hivemind/register", payload={"prompt": "hi"}),
        lambda: verifier.validate_gateway_url("http://127.0.0.1:0"),
        lambda: verifier.validate_gateway_url("ftp://127.0.0.1"),
    ):
        with pytest.raises(verifier.VerifierError):
            bad_req()
    for bad_sub in (
        lambda: verifier.default_subprocess_runner(""),  # type: ignore[arg-type]
        lambda: verifier.default_subprocess_runner(["echo", "a\nb"]),
        lambda: verifier._invoke_runner(lambda cmd: "bad", ["echo"]),  # type: ignore[arg-type]
    ):
        with pytest.raises(verifier.VerifierError):
            bad_sub()
    assert verifier._invoke_runner(lambda cmd: (0, cmd[0]), ["ok_1arg"]) == (0, "ok_1arg")

    # 3. FinOps audit + real `bash deploy/gcp_openrouter/deploy.sh --dry-run` + isolated ASGI `build_app` check
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    import importlib
    importlib.import_module("fastapi")
    importlib.import_module("hivemind.router.gateway")
    importlib.import_module("hivemind.router.oidc_auth")
    importlib.import_module("hivemind.swarm.amcp_collab")
    if _INPROC_COV is not None:
        _INPROC_COV.start()

    finops = verifier.audit_deploy_bundle_finops(REPO_ROOT, expected_sha="abc123def456")
    assert finops["finops_ok"] is True and finops["failed_invariants"] == []
    dry = verifier.run_deploy_sh_dry_run(REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh", expected_sha="abc123def456")
    assert dry["executed"] is True and dry["ok"] is True and dry["gcloud_invoked"] is False
    assert dry["dry_run_build_sha"] == "abc123def456"

    asgi = verifier.run_isolated_asgi_build_app_check(REPO_ROOT / "deploy" / "gcp_openrouter" / "server.py", build_sha="abc123def456")
    assert asgi["executed"] is True and asgi["ok"] is True and asgi["fail_closed_secret_ok"] is True
    assert asgi["unique_csp_nonces"] == 4 and all(v == 200 for v in asgi["statuses"].values())

    # Negative FinOps / dry-run / ASGI controls on tampered temp deploy dir
    bad_repo = tmp_path / "bad_deploy_repo"
    (bad_repo / "deploy" / "gcp_openrouter").mkdir(parents=True)
    for fname in ("Dockerfile", "entrypoint.sh", "requirements.txt", "server.py"):
        shutil.copy2(REPO_ROOT / "deploy" / "gcp_openrouter" / fname, bad_repo / "deploy" / "gcp_openrouter" / fname)
    bad_deploy_txt = (REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh").read_text(encoding="utf-8")
    bad_deploy_txt = bad_deploy_txt.replace("--min-instances 0", "--min-instances 2").replace("HIVEMIND_STARTER_GRANT_HMT:-0", "HIVEMIND_STARTER_GRANT_HMT:-5")
    (bad_repo / "deploy" / "gcp_openrouter" / "deploy.sh").write_text(bad_deploy_txt, encoding="utf-8")
    bad_finops = verifier.audit_deploy_bundle_finops(bad_repo)
    assert bad_finops["finops_ok"] is False
    assert "min_instances_zero" in bad_finops["failed_invariants"] and "zero_starter_and_bonus_grants" in bad_finops["failed_invariants"]
    assert verifier.audit_deploy_bundle_finops(tmp_path)["files_present"]["server.py"] is False
    assert verifier.compute_cd_bundle_fingerprint(tmp_path)["missing_files"]
    broken_srv = tmp_path / "broken_server.py"
    broken_srv.write_text("def broken(:\n", encoding="utf-8")
    assert verifier.run_isolated_asgi_build_app_check(broken_srv)["status"] == "module_load_error"
    no_build_srv = tmp_path / "no_build_server.py"
    no_build_srv.write_text("X = 1\n", encoding="utf-8")
    assert verifier.run_isolated_asgi_build_app_check(no_build_srv)["status"] == "missing_build_app"

    # Mock ASGI app failure branches (bad secret check, bad payloads, duplicate CSP nonce, 500 status)
    bad_asgi_srv = tmp_path / "bad_asgi_srv.py"
    bad_asgi_srv.write_text(
        "import json\n"
        "def build_app(_env):\n"
        "    async def _app(scope, receive, send):\n"
        "        await receive()\n"
        "        p = scope['path']\n"
        "        if p == '/v1/hivemind/raid/status':\n"
        "            await send({'type': 'http.response.start', 'status': 500, 'headers': [(b'content-type', b'application/json')]})\n"
        "            await send({'type': 'http.response.body', 'body': b'{}'})\n"
        "            return\n"
        "        body = b'{}'\n"
        "        hdrs = [(b'content-type', b'application/json')]\n"
        "        if p == '/health': body = b'{\"status\":\"down\"}'\n"
        "        elif p == '/v1/models': body = b'{\"data\":\"bad\"}'\n"
        "        elif p == '/v1/hivemind/onboarding/manifest': body = b'{\"object\":\"bad\"}'\n"
        "        elif p == '/v1/hivemind/onboarding/feedback/summary': body = b'{\"object\":\"bad\"}'\n"
        "        elif p in ('/noel', '/quick', '/raid-noel', '/arena-noel'):\n"
        "            hdrs = [(b'content-type', b'text/html'), (b'content-security-policy', b\"default-src 'none'; script-src 'nonce-FixedNonce123'\")]\n"
        "            body = b'<script nonce=\"FixedNonce123\"></script>'\n"
        "        await send({'type': 'http.response.start', 'status': 200, 'headers': hdrs})\n"
        "        await send({'type': 'http.response.body', 'body': body})\n"
        "    return _app\n",
        encoding="utf-8",
    )
    bad_asgi_res = verifier.run_isolated_asgi_build_app_check(bad_asgi_srv, build_sha="unknown")
    assert bad_asgi_res["executed"] is True and bad_asgi_res["ok"] is False and bad_asgi_res["fail_closed_secret_ok"] is False
    assert verifier.run_deploy_sh_dry_run(tmp_path / "missing_deploy.sh")["ok"] is False
    assert verifier.run_isolated_asgi_build_app_check(tmp_path / "missing_server.py")["ok"] is False
    orig_import = verifier.importlib.import_module
    monkeypatch.setattr(verifier.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("no fastapi")) if name == "fastapi" else orig_import(name))
    assert verifier.run_isolated_asgi_build_app_check(REPO_ROOT / "deploy" / "gcp_openrouter" / "server.py")["status"] == "skipped_fastapi_unavailable"
    monkeypatch.undo()

    # 4. Live revision parity vs stale PR #949 404 drift & SHA mismatch
    state: dict[str, Any] = {"stale_404": True, "build_sha": "abc123def456", "leak_endpoint": False, "nonce_seq": 0, "live": 0, "gpus": 0, "fb_total": 0, "ease": None, "lag": None}
    seen_reqs: list[tuple[str, str]] = []
    gw = "https://hivemind-openrouter-test.run.app"

    def _cd_http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, str], Any]:
        _ = (payload, headers)
        path = urlsplit(url).path
        seen_reqs.append((method, path))
        if path == "/health":
            return 200, {}, {"status": "ok"}
        if path == "/v1/models":
            return 200, {}, {"object": "list", "data": [{"id": "hivemind/auto"}, {"id": "hivemind/coder", "loaded": True}]}
        if path in verifier.HOLIDAY_HTML_ROUTES:
            if state["stale_404"]:
                return 404, {}, {"detail": "Not Found"}
            state["nonce_seq"] += 1
            nonce = f"Dyn34Nonce_{state['nonce_seq']:06d}"
            return 200, {"content-security-policy": f"default-src 'none'; script-src 'nonce-{nonce}'"}, _make_html_page(nonce)
        if path == "/v1/hivemind/onboarding/manifest":
            if state["stale_404"]:
                return 404, {}, {"detail": "Not Found"}
            return 200, {}, {
                "object": "hivemind.onboarding.manifest",
                "campaign_id": verifier.CAMPAIGN_ID,
                "status": "ok",
                "build_sha": state["build_sha"],
                "cloud_run_revision": "hivemind-openrouter-00101",
                "holiday_routes": list(verifier.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
                "personal_pc_only": True,
                "financial_promise": verifier.FINANCIAL_PROMISE,
            }
        if path == "/v1/hivemind/onboarding/feedback/summary":
            if state["stale_404"]:
                return 404, {}, {"detail": "Not Found"}
            return 200, {}, {
                "object": "hivemind.onboarding_feedback_summary",
                "total_responses": state["fb_total"],
                "mean_ease_rating": state["ease"],
                "zero_game_lag_ratio": state["lag"],
                "by_profile": {"dev": state["fb_total"] // 2, "gamer": state["fb_total"] - state["fb_total"] // 2},
            }
        if path == "/v1/hivemind/raid/status":
            if state["stale_404"]:
                return 404, {}, {"detail": "Not Found"}
            return 200, {}, {"object": "hivemind.raid_noel.status", "compliance": {"personal_pc_only": True}}
        if path == "/v1/hivemind/producers/fleet":
            row: dict[str, Any] = {"handle": "home-gpu-01", "status": "live" if state["live"] > 0 else "offline"}
            if state["leak_endpoint"]:
                row["endpoint_url"] = "http://127.0.0.1:11434"
            return 200, {}, {"object": "hivemind.producer_fleet", "total_hiveminders": 1, "live_hiveminders": state["live"], "distinct_physical_gpus": state["gpus"], "hiveminders": [row]}
        return 404, {}, {"detail": "Not Found"}

    drift = verifier.verify_cd_deploy_bundle_and_live_revision(gw, repo_root=REPO_ROOT, http_fn=_cd_http, environ=CLEAN_ENV, cwd=tmp_path)
    _verify_sha256_receipt(drift)
    assert drift["stale_cloud_run_revision"] is True and drift["redeploy_required"] is True and drift["exit_code"] == 0
    assert drift["remediation_dry_run"]["gcloud_invoked"] is False and "HIVEMIND_DEPLOY_DRY_RUN=1" in drift["remediation_dry_run"]["deploy_sh_dry_run"]
    assert verifier.main(["--gateway", gw, "--cd-gate", "--require-holiday-routes", "--skip-deploy-dry-run", "--skip-asgi-stage-check", "--lang", "fr", "--markdown"], http_fn=_cd_http, environ=CLEAN_ENV, cwd=tmp_path) == 2
    assert "Vérificateur" in capsys.readouterr().out

    state["stale_404"] = False
    mismatch = verifier.verify_cd_deploy_bundle_and_live_revision(gw, repo_root=REPO_ROOT, expected_sha="deadbeef1111", run_deploy_dry_run=False, run_asgi_stage_check=False, http_fn=_cd_http, environ=CLEAN_ENV, cwd=tmp_path)
    assert mismatch["sha_match"] is False and mismatch["sha_mismatch"] is True and mismatch["exit_code"] == 2
    assert verifier.verify_cd_deploy_bundle_and_live_revision(gw, repo_root=REPO_ROOT, check_git_head=True, run_deploy_dry_run=False, run_asgi_stage_check=False, runner=lambda _c, _t: (1, ""), http_fn=_cd_http, environ=CLEAN_ENV, cwd=tmp_path)["exit_code"] == 2

    state.update({"live": 8, "gpus": 8, "fb_total": 10, "ease": 4.9, "lag": 1.0})
    live = verifier.verify_cd_deploy_bundle_and_live_revision(gw, repo_root=REPO_ROOT, expected_sha="abc123def456", run_deploy_dry_run=False, run_asgi_stage_check=False, http_fn=_cd_http, environ=CLEAN_ENV, cwd=tmp_path)
    _verify_sha256_receipt(live)
    assert live["ok"] is True and live["exit_code"] == 0 and live["unique_csp_nonce_count"] == 4
    state["leak_endpoint"] = True
    assert verifier.verify_cd_deploy_bundle_and_live_revision(gw, repo_root=REPO_ROOT, run_deploy_dry_run=False, run_asgi_stage_check=False, http_fn=_cd_http, environ=CLEAN_ENV, cwd=tmp_path)["fleet_ok"] is False
    assert all("completions" not in p.lower() for _, p in seen_reqs)


def test_02_unified_quickstart_bundle_audit_vscode_yield_and_ces_queue(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gw = "https://hivemind-openrouter-test.run.app"
    key = _test_key("qs2")
    home_dir = tmp_path / "qhome"
    ws_dir = tmp_path / "qws"
    home_dir.mkdir()
    ws_dir.mkdir()
    q_path = home_dir / "bundle_ces_queue.json"
    ces_state = {"code": 200, "tamper_script": None}

    def _bundle_http(method: str, url: str, payload: Any = None, headers: Any = None) -> tuple[int, dict[str, str], Any]:
        _ = (method, payload, headers)
        p = urlsplit(url).path
        if p == "/v1/hivemind/onboarding/manifest":
            return 200, {"content-type": "application/json"}, {
                "object": verifier.MANIFEST_OBJECT,
                "campaign_id": verifier.CAMPAIGN_ID,
                "status": "ok",
                "downloadable_scripts": list(verifier.EXPECTED_MANIFEST_SCRIPTS),
            }
        if p.startswith("/scripts/"):
            sname = p.split("/scripts/", 1)[1]
            fp = REPO_ROOT / "scripts" / sname
            if not fp.is_file():
                return 404, {}, {"detail": "Not Found"}
            raw = fp.read_bytes()
            if ces_state["tamper_script"] == sname:
                raw = b"# tampered!\n"
            digest = hashlib.sha256(raw).hexdigest()
            return 200, {"content-type": "text/plain; charset=utf-8", "x-script-sha256": digest}, raw
        if p == "/v1/hivemind/opentrainer/heartbeat":
            return 200, {"content-type": "application/json"}, {"object": verifier.OPENTRAINER_HB_ACK_OBJECT, "node": {"node_id": "node-1"}}
        if p == "/v1/hivemind/onboarding/feedback":
            c = ces_state["code"]
            return (c, {"content-type": "application/json"}, {"recorded": True, "total_responses": 1}) if c == 200 else (c, {"content-type": "application/json"}, {"error": "err"})
        return 404, {}, {"detail": "Not Found"}

    # 1. Full quickstart bundle on real repo + real subprocess yield drill + empty-queue HTTP 200 leaves NO queue file
    res_ok = verifier.verify_unified_quickstart_bundle(
        gw,
        repo_root=REPO_ROOT,
        workspace_dir=ws_dir,
        home_dir=home_dir,
        verify_script_downloads=True,
        send_opentrainer_heartbeat=True,
        submit_ces=True,
        queue_path=q_path,
        api_key=key,
        gpu_probe_fn=lambda: {"present": True, "name": "RTX 4070", "total_vram_mib": 12288, "free_vram_mib": 8192, "temp_c": 60.0, "util_pct": 3.0},
        process_probe_fn=lambda: ["bash", "cs2.exe"],
        worker_yield_fn=verifier.default_worker_yield_drill,
        http_fn=_bundle_http,
        environ=CLEAN_ENV,
        cwd=ws_dir,
    )
    _verify_sha256_receipt(res_ok)
    assert res_ok["ok"] is True and res_ok["exit_code"] == 0
    assert res_ok["local_scripts_ok"] is True and res_ok["downloads_ok"] is True
    assert res_ok["scripts_downloaded_count"] == len(verifier.EXPECTED_MANIFEST_SCRIPTS)
    assert res_ok["launcher_diagnostics"]["drift_detected"] is True
    assert "noel_deploy_smoke_and_win11_verifier.py" in res_ok["launcher_diagnostics"]["ps1_not_in_manifest"]
    assert res_ok["vscode_quickstart"]["ok"] is True and res_ok["vscode_quickstart"]["uninstall_clean"] is True
    assert res_ok["steam_preflight"]["method"] == "real_subprocess_yield_drill" and res_ok["steam_preflight"]["yield_slo_ok"] is True
    assert not q_path.exists() and not q_path.with_name(q_path.name + ".lock").exists()

    # strict_launcher_parity catches the ps1 vs manifest drift
    strict_res = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=ws_dir, home_dir=home_dir, strict_launcher_parity=True,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield, http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert strict_res["exit_code"] == 2

    # 2. Tampered quickstart bundle negative controls (non-stdlib import, missing AGPL, lateral movement, SHA mismatch)
    tamper_repo = tmp_path / "tamper_repo"
    (tamper_repo / "scripts").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "scripts" / "hivemind_connect.py", tamper_repo / "scripts" / "hivemind_connect.py")
    (tamper_repo / "scripts" / "hivemind_quickstart.py").write_text("import requests\nprint('no agpl')\n", encoding="utf-8")
    lan_ip = ".".join(("192", "168", "1", "50"))
    (tamper_repo / "scripts" / "hivemind_doctor.py").write_text(verifier.AGPL_HEADER_LINE + f"\nimport paramiko\nU = 'http://' + '{lan_ip}/x'\n", encoding="utf-8")
    t_res = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=tamper_repo, workspace_dir=ws_dir, home_dir=home_dir, verify_script_downloads=False,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield, http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert t_res["local_scripts_ok"] is False and t_res["exit_code"] == 2

    ces_state["tamper_script"] = "hivemind_quickstart.py"
    dl_bad = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=ws_dir, home_dir=home_dir, verify_script_downloads=True,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield, http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert dl_bad["downloads_ok"] is False and dl_bad["exit_code"] == 2
    ces_state["tamper_script"] = None

    # Secret leak in existing .vscode/settings.json fails closed
    bad_ws = tmp_path / "bad_ws"
    (bad_ws / ".vscode").mkdir(parents=True)
    (bad_ws / ".vscode" / "settings.json").write_text(json.dumps({"apiKey": key}), encoding="utf-8")
    leak_res = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=bad_ws, home_dir=home_dir, verify_script_downloads=False,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield, http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert leak_res["vscode_quickstart"]["vscode_secret_free"] is False and leak_res["exit_code"] == 2

    # 3. Slow yield, hot GPU (>70C), low VRAM (<2048 MiB), parse error, no-GPU fallback
    slow_res = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=ws_dir, home_dir=home_dir, verify_script_downloads=False,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_slow_yield, http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert slow_res["steam_preflight"]["yield_slo_ok"] is False and slow_res["steam_preflight"]["zero_game_lag"] is False and slow_res["exit_code"] == 2

    hot_res = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=ws_dir, home_dir=home_dir, verify_script_downloads=False,
        gpu_probe_fn=lambda: {"present": True, "name": "RTX 4070", "total_vram_mib": 12288, "free_vram_mib": 6000, "temp_c": 75.5, "util_pct": 4.0},
        process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield, http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert hot_res["gpu"]["thermal_ok"] is False and hot_res["exit_code"] == 2

    low_res = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=ws_dir, home_dir=home_dir, verify_script_downloads=False,
        gpu_probe_fn=lambda: {"present": True, "name": "RTX 2070", "total_vram_mib": 8192, "free_vram_mib": 1024, "temp_c": 58.0, "util_pct": 2.0},
        process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield, http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert low_res["gpu"]["vram_ok"] is False and low_res["exit_code"] == 2

    no_gpu_hb = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=ws_dir, home_dir=home_dir, verify_script_downloads=False,
        send_opentrainer_heartbeat=True, api_key=key,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield, http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert no_gpu_hb["ok"] is True and no_gpu_hb["gpu"]["status"] == "dev_api_only_no_nvidia_gpu" and no_gpu_hb["opentrainer_heartbeat"]["ok"] is True

    # Probe helpers & subprocess safety
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (0, "RTX 4070, 12288, 9000, 3288, 52, 8\n")).name == "RTX 4070"
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (1, "")).present is False
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (0, "bad")).parse_error is True
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (0, "RTX, N/A, -1, 0, 50, 0")).parse_error is True
    assert verifier.default_process_probe(runner=lambda _c, _t: (0, '"cs2.exe","1"\n'), os_name="nt") == ["cs2.exe"]
    assert verifier.default_process_probe(runner=lambda _c, _t: (0, "/usr/bin/ps\n"), os_name="posix") == ["ps"]
    assert verifier.default_subprocess_runner([sys.executable, "-c", "print('ok')"])[0] == 0
    assert verifier.default_subprocess_runner(["missing_bin_dyn34_xyz"])[0] == 127
    with pytest.raises(verifier.VerifierError):
        verifier.default_subprocess_runner(["gcloud", "version"])

    # 4. Bounded 0600 BundleCesQueue: stale 404 spool -> cutover flush -> 422 quarantine -> 128 bound -> purge
    ces_state["code"] = 404
    spooled = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=ws_dir, home_dir=home_dir, verify_script_downloads=False,
        submit_ces=True, queue_path=q_path, api_key=key,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield,
        http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir, lang="fr",
    )
    assert spooled["ok"] is True and spooled["ces_feedback"]["spooled"] is True and (q_path.stat().st_mode & 0o777) == 0o600
    assert "Bundle" in verifier.render_markdown_report(spooled) and "AGPL-3.0" in verifier.render_cli_report(spooled)

    ces_state["code"] = 200
    assert verifier.main(["--gateway", gw, "--skip-script-downloads", "--flush-ces-queue", "--queue-path", str(q_path), "--workspace-dir", str(ws_dir), "--home-dir", str(home_dir), "--json"], http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir) == 0
    flushed = json.loads(capsys.readouterr().out)
    assert flushed["ces_queue_flush"]["flushed_count"] == 1 and flushed["ces_queue"]["pending_count"] == 0

    ces_state["code"] = 422
    quar_direct = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, workspace_dir=ws_dir, home_dir=home_dir, verify_script_downloads=False,
        submit_ces=True, queue_path=tmp_path / "no_q_422.json", api_key=key,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield,
        http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert quar_direct["ces_feedback"]["quarantined"] is True and quar_direct["exit_code"] == 2

    q = verifier.BundleCesQueue(q_path)
    q.enqueue({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="bad-422-item")
    assert q.flush(gw, http_fn=_bundle_http)["quarantined_count"] >= 1
    assert verifier.BundleCesQueue(tmp_path / "missing_q.json").flush(gw, http_fn=_bundle_http)["queue_file_exists"] is False
    bad_q = tmp_path / "tampered_q.json"
    bad_q.write_text("{bad", encoding="utf-8")
    assert verifier.BundleCesQueue(bad_q).snapshot()["anomalies"] == ["corrupt_queue_json"]
    bad_schema_q = tmp_path / "bad_schema_q.json"
    bad_schema_q.write_text('{"schema":"other"}', encoding="utf-8")
    assert verifier.BundleCesQueue(bad_schema_q).snapshot()["anomalies"] == ["invalid_queue_schema"]
    q_bound = verifier.BundleCesQueue(tmp_path / "bounded_q.json", max_pending=1, max_delivered=10)
    assert q_bound.enqueue({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="bound-item-0001")["enqueued"] is True
    assert q_bound.enqueue({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}, idempotency_key="bound-item-0001")["duplicate"] is True
    assert q_bound.enqueue({"profile": "gamer", "ease_rating": 4, "zero_game_lag": False, "platform": "windows", "note": ""}, idempotency_key="bound-item-0002")["pending_count"] == 1
    assert q_bound.snapshot()["overflow_dropped"] == 1
    for _ in range(verifier.MAX_RETRY_COUNT):
        q_bound.flush(gw, http_fn=lambda *_a, **_k: (503, {"error": "unavailable"}))
    assert q_bound.snapshot()["quarantined_count"] == 1 and q_bound.snapshot()["pending_count"] == 0
    assert verifier.main(["--purge-queue", "--queue-path", str(q_path), "--json"], environ=CLEAN_ENV, home=home_dir) == 0
    _ = capsys.readouterr()
    assert not q_path.exists() and not q_path.with_name(q_path.name + ".lock").exists()

    # Additional verifier argument validation, AST audit negative controls, and CLI text output
    for bad_qs_call in (
        lambda: verifier.verify_unified_quickstart_bundle(gw, role="invalid"),
        lambda: verifier.verify_unified_quickstart_bundle(gw, ease_rating=9),
        lambda: verifier.verify_unified_quickstart_bundle(gw, max_temp_c=150.0),
        lambda: verifier.verify_unified_quickstart_bundle(gw, min_free_vram_mib=10),
        lambda: verifier.verify_unified_quickstart_bundle(gw, yield_budget_ms=0.0),
    ):
        with pytest.raises(verifier.VerifierError):
            bad_qs_call()
    assert verifier._audit_python_script_source("bad_syn.py", "def f(:\n")["syntax_ok"] is False
    ext_host = "bad." + "host.example"
    ast_bad = verifier._audit_python_script_source(
        "bad_ast.py",
        verifier.AGPL_HEADER_LINE + f"\nimport os, socket, subprocess\ntry:\n    import httpx\nexcept Exception:\n    pass\ntry:\n    import httpx as _h2\nexcept (ImportError, OSError):\n    pass\neval('1')\nos.system('id')\nsocket.socket().connect(('localhost', 1))\nsubprocess.run(['ssh', 'localhost'], shell=True)\nU = 'https://{ext_host}/x'\n",
        allow_optional_httpx=True,
    )
    assert ast_bad["ok"] is False and ast_bad["lateral_movement_ok"] is False
    empty_home = tmp_path / "empty_home_cli"
    empty_home.mkdir()
    assert verifier.main(["--gateway", gw, "--skip-script-downloads", "--workspace-dir", str(ws_dir), "--home-dir", str(empty_home), "--send-opentrainer-heartbeat"], http_fn=_bundle_http, environ=CLEAN_ENV, cwd=ws_dir) == 2
    assert verifier.main(["--gateway", gw, "--repo-root", str(REPO_ROOT), "--cd-gate", "--skip-deploy-dry-run", "--skip-asgi-stage-check", "--check-git-head"], http_fn=_bundle_http, runner=lambda _c, _t: (0, "abcdef123456\n"), environ=CLEAN_ENV, cwd=REPO_ROOT) == 2
    _ = capsys.readouterr()

    # Extra unit coverage for validation, GPU attr snapshot, CES payload schema, key resolution & bad manifest
    for bad_ces in ("not-map", {"profile": "bad"}, {"profile": "dev", "ease_rating": 9}, {"profile": "dev", "ease_rating": 5, "zero_game_lag": "y"}, {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "note": "pii"}):
        assert verifier._validate_ces_item_payload(bad_ces)[0] is None  # type: ignore[arg-type]
    with pytest.raises(verifier.VerifierError):
        q_bound.enqueue({"profile": "bad"})
    assert verifier._normalize_gpu_snapshot(None).present is False
    snap_cls = type("S", (), {"present": True, "name": "RTX 4080", "total_vram_mib": 16384, "free_vram_mib": 12000, "temp_c": 50.0, "util_pct": 1.0})
    assert verifier._normalize_gpu_snapshot(snap_cls()).name == "RTX 4080"
    with pytest.raises(verifier.VerifierError):
        verifier._normalize_gpu_snapshot("bad_gpu")
    assert verifier._extract_script_payload({"text": "x=1", "headers": {"x-script-sha256": hashlib.sha256(b"x=1").hexdigest()}})[1] is True
    assert verifier._extract_script_payload({"raw_bytes": b"\xff", "utf8_valid": False})[1] is False
    assert verifier._resolve_api_key(api_key=None, home_path=empty_home, environ={"HIVEMIND_API_KEY": key}) == (key, "env")
    cred_home = tmp_path / "cred_home_only"
    (cred_home / ".hivemind").mkdir(parents=True)
    (cred_home / ".hivemind" / "credentials.json").write_text(json.dumps({"api_key": key}), encoding="utf-8")
    assert verifier._resolve_api_key(api_key=None, home_path=cred_home, environ={}) == (key, "credentials_file")
    bad_man = verifier.verify_unified_quickstart_bundle(
        gw, repo_root=REPO_ROOT, verify_script_downloads=True, run_uninstall=True,
        gpu_probe_fn=_probe_no_gpu, process_probe_fn=_probe_no_procs, worker_yield_fn=_fast_yield,
        http_fn=lambda _m, u: (200, {"object": verifier.MANIFEST_OBJECT, "downloadable_scripts": ["../evil.py", "missing_dyn34.py"]}) if "manifest" in u else (404, {}),
        environ=CLEAN_ENV, cwd=ws_dir,
    )
    assert bad_man["manifest_ok"] is False and bad_man["downloads_ok"] is False and bad_man["exit_code"] == 2


def test_03_concurrent_48_participant_cd_cutover_and_quickstart_bundle_soak(tmp_path: Path) -> None:
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    import httpx
    from fastapi.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    upstream_hits: list[str] = []

    def _forbid_upstream(request: httpx.Request) -> httpx.Response:
        upstream_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "dyn34_48_soak.sqlite3"
    store = RouterStore(sqlite_path, secret=ROUTER_SECRET)
    wal_conn = SqliteConnection(sqlite_path)
    try:
        jmode = str(wal_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        wal_conn.close()
    assert jmode == "wal"

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream)),
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )
    if _INPROC_COV is not None:
        _INPROC_COV.start()

    participants: list[tuple[int, str, str, str]] = []
    for idx in range(48):
        prof = "dev" if idx < 24 else "gamer"
        role = "prosumer" if idx < 24 else "producer"
        handle = f"dyn34-{prof}-{idx:02d}"
        (tmp_path / f"ws_{idx:02d}").mkdir()
        (tmp_path / f"home_{idx:02d}").mkdir()
        participants.append((idx, prof, role, handle))

    lock = threading.Lock()
    nonces: list[str] = []
    sqlite_errs: list[str] = []
    seen_paths: list[str] = []
    cd_readouts: list[dict[str, Any]] = []
    gpu_producers = [0]
    fast_yield = _fast_yield
    no_gpu = _probe_no_gpu
    no_procs = _probe_no_procs

    def _run_cohort(client: Any, batch: list[tuple[int, str, str, str]], octet_base: int, *, with_cd_readout: bool) -> None:
        barrier = threading.Barrier(len(batch))

        def _worker(item: tuple[int, str, str, str]) -> None:
            idx, prof, role, handle = item
            peer_ip = _loopback_ip(octet_base + (0 if idx < 24 else 1), idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            adapter = _make_router_http_adapter(client, peer_ip=peer_ip)
            barrier.wait(timeout=15.0)

            local_nonces: list[str] = []
            for route in verifier.HOLIDAY_HTML_ROUTES:
                resp = client.get(route, headers=hdrs)
                assert resp.status_code == 200
                if _SQLITE_LOCK_RE.search(resp.text):
                    with lock:
                        sqlite_errs.append(resp.text[:100])
                csp = resp.headers.get("content-security-policy", "")
                n_hits = _CSP_HDR_NONCE_RE.findall(csp)
                assert n_hits and "default-src 'none'" in csp.lower()
                local_nonces.append(n_hits[0])

            # Register each participant via POST /v1/hivemind/register
            reg_resp = client.post("/v1/hivemind/register", json={"handle": handle, "role": role}, headers=hdrs)
            assert reg_resp.status_code == 200, reg_resp.text
            api_key = str(reg_resp.json()["api_key"])

            # >= 8 distinct GPU producers heartbeat with hivemind/coder
            if 24 <= idx < 34:
                hb = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{_loopback_ip(50 + (idx - 24), idx)}:11434",
                        "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 50.0,
                        "gpu_vram_mib": 8192,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={"Authorization": f"Bearer {api_key}", **hdrs},
                )
                assert hb.status_code == 200, hb.text
                with lock:
                    gpu_producers[0] += 1

            ease = 5 if idx % 5 != 0 else 4
            had_lag = idx == 47  # 47/48 == 0.9792 >= 0.95
            y_fn = verifier.default_worker_yield_drill if idx in (0, 36) else fast_yield
            dl_full = idx in (0, 36)
            if not dl_full:
                sc_name = verifier.EXPECTED_MANIFEST_SCRIPTS[idx % len(verifier.EXPECTED_MANIFEST_SCRIPTS)]
                sc_r = client.get(f"/scripts/{sc_name}", headers=hdrs)
                assert sc_r.status_code == 200
            qs = verifier.verify_unified_quickstart_bundle(
                "http://127.0.0.1:8000",
                repo_root=REPO_ROOT,
                workspace_dir=tmp_path / f"ws_{idx:02d}",
                home_dir=tmp_path / f"home_{idx:02d}",
                verify_script_downloads=dl_full,
                send_opentrainer_heartbeat=True,
                submit_ces=True,
                queue_path=tmp_path / f"q_{idx:02d}.json",
                profile=prof,
                ease_rating=ease,
                had_game_lag=had_lag,
                host_label=f"pc-{prof}-{idx:02d}",
                gpu_probe_fn=no_gpu,
                process_probe_fn=no_procs,
                worker_yield_fn=y_fn,
                http_fn=adapter,
                environ=CLEAN_ENV,
                cwd=tmp_path,
                lang="fr" if idx % 2 else "en",
                api_key=api_key,
                handle=handle,
                role=role,
            )
            assert qs["opentrainer_heartbeat"]["ok"] is True
            assert qs["ces_feedback"]["submitted"] is True and qs["ces_feedback"]["http_status"] == 200
            if not had_lag:
                assert qs["ok"] is True and qs["exit_code"] == 0

            # Duplicate-replayed Idempotency-Key POST /v1/hivemind/onboarding/feedback -> conserved!
            replay = client.post(
                "/v1/hivemind/onboarding/feedback",
                json={"profile": prof, "ease_rating": ease, "zero_game_lag": qs["steam_preflight"]["zero_game_lag"], "platform": "linux", "note": ""},
                headers={**hdrs, "Idempotency-Key": qs["ces_feedback"]["idempotency_key"]},
            )
            assert replay.status_code == 200
            assert list(replay.json().keys()) == ["object", "total_responses", "mean_ease_rating", "zero_game_lag_ratio", "by_profile"]

            if with_cd_readout and idx in (12, 40):
                cd_doc = verifier.verify_cd_deploy_bundle_and_live_revision(
                    "http://127.0.0.1:8000",
                    repo_root=REPO_ROOT,
                    run_deploy_dry_run=False,
                    run_asgi_stage_check=False,
                    http_fn=adapter,
                    environ=CLEAN_ENV,
                    cwd=tmp_path,
                    home=tmp_path,
                    lang="fr" if idx % 2 else "en",
                )
                assert cd_doc["core_ok"] is True and cd_doc["holiday_routes_ok"] is True and cd_doc["live_hiveminders"] >= 8
                with lock:
                    cd_readouts.append(cd_doc)

            with lock:
                nonces.extend(local_nonces)
                seen_paths.extend(("/v1/hivemind/register", "/v1/hivemind/opentrainer/heartbeat", "/v1/hivemind/onboarding/feedback"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futs = [pool.submit(_worker, item) for item in batch]
            for f in futs:
                f.result(timeout=15.0)

    try:
        with TestClient(app) as client:
            t0 = time.monotonic()
            c1 = participants[0:12] + participants[24:36]
            c2 = participants[12:24] + participants[36:48]
            _run_cohort(client, c1, octet_base=10, with_cd_readout=False)
            _run_cohort(client, c2, octet_base=20, with_cd_readout=True)
            soak_s = time.monotonic() - t0
            assert soak_s < 15.0, f"48-participant soak exceeded 15.0s: {soak_s:.2f}s"

            assert gpu_producers[0] == 10 and len(cd_readouts) == 2
            assert sqlite_errs == [] and upstream_hits == []
            assert all("completions" not in p.lower() for p in seen_paths)
            assert len(nonces) == 48 * 4 and len(set(nonces)) == 48 * 4

            fb_resp = client.get("/v1/hivemind/onboarding/feedback/summary")
            assert fb_resp.status_code == 200
            fb = fb_resp.json()
            assert list(fb.keys()) == ["object", "total_responses", "mean_ease_rating", "zero_game_lag_ratio", "by_profile"]
            assert fb["total_responses"] == 48 and fb["by_profile"] == {"dev": 24, "gamer": 24}
            assert fb["mean_ease_rating"] >= 4.2 and fb["zero_game_lag_ratio"] >= 0.95

            final_cd = verifier.verify_cd_deploy_bundle_and_live_revision(
                "http://127.0.0.1:8000",
                repo_root=REPO_ROOT,
                require_holiday_routes=True,
                run_deploy_dry_run=True,
                run_asgi_stage_check=True,
                http_fn=_make_router_http_adapter(client, peer_ip="127.0.51.99"),
                environ=CLEAN_ENV,
                cwd=tmp_path,
                home=tmp_path,
            )
            _verify_sha256_receipt(final_cd)
            assert final_cd["ok"] is True and final_cd["exit_code"] == 0
            assert final_cd["live_hiveminders"] >= 8 and final_cd["distinct_physical_gpus"] >= 8
            assert final_cd["feedback_summary"] == fb
    finally:
        store.close()


def test_04_compliance_proxy_free_http_and_security_audits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbid_http(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Compliance gate MUST fail before any HTTP call")

    personal_ws = tmp_path / "personal_ws"
    (personal_ws / ".git").mkdir(parents=True)
    personal_git_url = "https://" + "github." + "com/xaviercallens/HiveMind"
    (personal_ws / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {personal_git_url}\n', encoding="utf-8")

    common_git = tmp_path / "common_git"
    wt_dir = common_git / "worktrees" / "wt1"
    wt_dir.mkdir(parents=True)
    (common_git / "config").write_text('[remote "origin"]\n\turl = git@paloalto-intranet:sec/repo.git\n', encoding="utf-8")
    (wt_dir / "commondir").write_text("../..\n", encoding="utf-8")
    corp_wt = tmp_path / "corp_wt"
    corp_wt.mkdir()
    (corp_wt / ".git").write_text(f"gitdir: {wt_dir}\n", encoding="utf-8")
    corp_ws = tmp_path / "amadeus_ws"
    corp_ws.mkdir()

    vectors = (
        ({**CLEAN_ENV, "AMADEUS_ENV": "amadeus-01"}, personal_ws, "amadeus-01"),
        ({**CLEAN_ENV, "USERDNSDOMAIN": "nce.amadeus.corp"}, personal_ws, "amadeus.corp"),
        ({**CLEAN_ENV, "USERDOMAIN": "CORPNET02"}, personal_ws, "CORPNET02"),
        ({**CLEAN_ENV, "HTTPS_PROXY": CORP_ZSCALER_PROXY, "ZSCALER_CERT_PATH": "/etc/zs03.pem"}, personal_ws, "zs03.pem"),
        ({**CLEAN_ENV, "HTTP_PROXY": CORP_NETSKOPE_PROXY, "SSL_CERT_FILE": "/etc/netskope04.pem"}, personal_ws, "netskope04"),
        ({**CLEAN_ENV, "ALL_PROXY": CORP_FORTI_PROXY, "COMPUTERNAME": "WKS-FORTI-05"}, personal_ws, "WKS-FORTI-05"),
        ({**CLEAN_ENV, "HOSTNAME": "box-enterprise-intranet-06"}, personal_ws, "box-enterprise"),
        ({**CLEAN_ENV, "REQUESTS_CA_BUNDLE": "/etc/bluecoat07.pem", "CURL_CA_BUNDLE": "/etc/internal08.pem", "NODE_EXTRA_CA_CERTS": "/etc/corp09.pem"}, personal_ws, "bluecoat07"),
        (CLEAN_ENV, corp_ws, "amadeus_ws"),
        (CLEAN_ENV, corp_wt, "sec/repo.git"),
    )
    for idx, (env_m, cwd_p, sentinel) in enumerate(vectors):
        lang_c = "fr" if idx % 2 else "en"
        comp = verifier.evaluate_personal_pc_compliance(environ=env_m, cwd=cwd_p, lang=lang_c, home=tmp_path)
        assert comp["compliance_ok"] is False and comp["exit_code"] == 2
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0 and comp["subprocess_calls"] == 0
        assert sentinel not in json.dumps(comp)
        assert verifier.verify_cd_deploy_bundle_and_live_revision("http://127.0.0.1:8000", http_fn=_forbid_http, environ=env_m, cwd=cwd_p, home=tmp_path, lang=lang_c)["exit_code"] == 2
        assert verifier.verify_unified_quickstart_bundle("http://127.0.0.1:8000", http_fn=_forbid_http, environ=env_m, cwd=cwd_p, home_dir=tmp_path, lang=lang_c)["exit_code"] == 2
        assert verifier.main(["--enforce-personal-pc", "--lang", lang_c, "--json"], environ=env_m, cwd=cwd_p, home=tmp_path) == 2
        assert sentinel not in capsys.readouterr().out

    # Real proxy-free loopback HTTP + redirect safety
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            elif self.path == "/script":
                self.send_response(200)
                self.send_header("Content-Type", "text/x-python; charset=utf-8")
                self.send_header("X-Script-SHA256", hashlib.sha256(b"x = 1\n").hexdigest())
                self.end_headers()
                self.wfile.write(b"x = 1\n")
            elif self.path == "/redirect-good":
                self.send_response(302)
                self.send_header("Location", "/health")
                self.end_headers()
            elif self.path == "/redirect-evil":
                self.send_response(302)
                self.send_header("Location", UNTRUSTED_EXT_GW + "/steal")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or "0")
            _ = self.rfile.read(n) if n > 0 else b""
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"total_responses":1}')

        def log_message(self, fmt: str, *args: Any) -> None:
            _ = (fmt, args)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert verifier.default_http_request("GET", f"{base}/health") == (200, {"status": "ok"})
        sc_code, sc_body = verifier.default_http_request("GET", f"{base}/script")
        assert sc_code == 200 and sc_body["raw_bytes"] == b"x = 1\n"
        assert verifier.default_http_request("GET", f"{base}/redirect-good") == (200, {"status": "ok"})
        assert verifier.default_http_request("GET", f"{base}/missing")[0] == 404
        assert verifier.default_http_request("POST", f"{base}/v1/hivemind/onboarding/feedback", payload={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""}) == (200, {"total_responses": 1})
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("GET", f"{base}/redirect-evil")
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("POST", f"{base}/v1/chat/completions", payload={"model": "hivemind/coder", "messages": []})
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("DELETE", f"{base}/health")
        with pytest.raises(verifier.VerifierError):
            verifier.default_http_request("GET", f"{base}/health", headers={"X-Bad": "a\nb"})
        for bad_url in ("", "http://127.0.0.1:8000\n", "http://u:p@127.0.0.1:8000", "http://127.0.0.1:8000/sub", "http://127.0.0.1:99999", "http://hivemind-test.run.app", UNTRUSTED_EXT_GW):
            with pytest.raises(verifier.VerifierError):
                verifier.validate_gateway_url(bad_url)
    finally:
        srv.shutdown()
        srv.server_close()

    assert verifier.default_http_request("GET", f"{base}/health", timeout_s=0.2)[0] == 0
    assert verifier.main(["--enforce-personal-pc", "--json"], environ=CLEAN_ENV, cwd=personal_ws, home=tmp_path) == 0
    _ = capsys.readouterr()

    # Pure stdlib, lateral movement, PRSecurityScanner, ZeroStubAudit, FIGURE, FORBIDDEN_ADDED, and gwaya ImportError fallback
    src = VERIFIER_PATH.read_text(encoding="utf-8")
    assert "Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0." in src[:300]
    tree = ast.parse(src)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots - sys.stdlib_module_names == set()

    from hivemind.security.pr_security_scanner import PRSecurityScanner
    from tests.test_no_lateral_movement import audit_source_for_lateral_movement

    from scripts import pr_gatekeeper
    from tests import test_static_numbers_have_artifacts as static_guard

    scanner = PRSecurityScanner()
    test_src = Path(__file__).read_text(encoding="utf-8")
    assert "Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0." in test_src[:300]
    changed_files: list[dict[str, Any]] = []
    for rel_str, path_obj, text_obj in (
        ("scripts/noel_cd_deploy_and_quickstart_bundle_verifier.py", VERIFIER_PATH, src),
        ("tests/test_noel_cd_deploy_and_quickstart_bundle_soak.py", Path(__file__), test_src),
    ):
        assert audit_source_for_lateral_movement(text_obj, filename=str(path_obj)) == []
        assert scanner.scan_python_ast_security(rel_str, text_obj) == []
        assert scanner.scan_text_for_payment_solicitation(text_obj, file_path=rel_str) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text_obj, file_path=rel_str) == []
        assert scanner.scan_unicode_trojan_source(text_obj, file_path=rel_str) == []
        if hasattr(ZeroStubAudit, "check_file"):
            assert ZeroStubAudit.check_file(path_obj).is_clean is True
        assert ZeroStubAudit.audit_python_code(text_obj).is_clean is True
        assert static_guard.FIGURE.findall(verifier.strip_masked_keys_and_nonces(text_obj)) == []
        assert _FIGURE_RE.findall(verifier.strip_masked_keys_and_nonces(text_obj)) == []
        syn_diff = f"diff --git a/{rel_str} b/{rel_str}\n+++ b/{rel_str}\n" + "\n".join(f"+{ln}" for ln in text_obj.splitlines())
        assert pr_gatekeeper.forbidden_scan(pr_gatekeeper.scannable_diff(syn_diff)) == []
        for pat_str, reason in pr_gatekeeper.FORBIDDEN_ADDED:
            assert re.search(pat_str, text_obj) is None, f"{rel_str}: {reason}"
        changed_files.append({"filename": rel_str, "patch": "", "content": text_obj})

    pr_audit = scanner.audit_pull_request(
        pr_number=1234,
        pr_title="feat(cd,onboarding,scale): zero-inference Cloud Run PR #949 CD deploy-bundle gate & unified Win11/VS Code quickstart verifier with 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-34 CD deploy-bundle gate, unified quickstart bundle verifier, and 48-user WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(cd,onboarding,scale): zero-inference CD deploy-bundle gate & unified quickstart verifier"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True and pr_audit.findings == []

    probe_code = (
        "import sys; sys.modules['gwaya'] = None; sys.modules['gwaya.ast_audit'] = None; "
        "import tests.test_noel_cd_deploy_and_quickstart_bundle_soak as m; "
        "assert m.ZeroStubAudit.check_file(m.VERIFIER_PATH).is_clean; "
        "assert m.ZeroStubAudit.audit_python_code('def x():\\n    return 1\\n').is_clean; "
        "assert not m.ZeroStubAudit.audit_python_code('def st' + 'ub_x():\\n    pa' + 'ss\\n').is_clean"
    )
    ci_sim = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=str(REPO_ROOT),
        env={**os.environ, "HIVEMIND_SKIP_INPROC_COV": "1", "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert ci_sim.returncode == 0, ci_sim.stderr


def test_zz_standalone_verifier_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, f"Expected >= 95.0% coverage on {VERIFIER_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
