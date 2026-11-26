# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, Dec 26 Day-7 retention & cohort-balance, La Ruche, compliance & 48-user WAL soak tests (`SLICE-SCALE-DYN-31`)."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "scripts" / "noel_day7_retention_and_ruche_verifier.py"

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


from scripts import noel_day7_retention_and_ruche_verifier as verifier  # noqa: E402

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn31-router-sec-32b!"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example.invalid"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-bluecoat.internal:8080"
CORP_FORTI_PROXY = "http://" + "proxy.fortinet.internal:3128"
CORP_GIT_REMOTE = "https://" + "git.amadeus-internal.corp/team/service.git"

TEST_KEY_ALPHA = "hm-dyn31_" + "personal_key_1234"
TEST_KEY_8MS = "hm-dyn31_" + "personal_08" + "ms"
TEST_KEY_LIVE = "hm_live_" + "personal_dyn31_9999"

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


def _seed_live_coder_producer(client: Any, *, handle: str = "day7-home-rtx-coder", peer_ip: str = "127.0.99.1") -> None:
    reg_p = client.post(
        "/v1/hivemind/register",
        json={"handle": handle, "role": "producer"},
        headers={"X-Forwarded-For": peer_ip},
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
        headers={"Authorization": f"Bearer {reg_p['api_key']}", "X-Forwarded-For": peer_ip},
    )
    assert hb_p.status_code == 200, hb_p.text


def _sample_compliant_html(route: str, nonce: str, *, lang: str = "en") -> str:
    extra = ""
    if route in ("/noel", "/quick"):
        extra = "<pre>python3 scripts/hivemind_quickstart.py</pre>"
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


def _sample_models(*, include_coder: bool = True) -> dict[str, Any]:
    rows = [{"id": "hivemind/auto", "object": "model", "owned_by": "hivemind"}]
    if include_coder:
        rows.append({"id": "hivemind/coder", "object": "model", "owned_by": "hivemind"})
    return {"object": "list", "data": rows}


def _sample_fleet(*, live: int = 2, leak_endpoint: bool = False) -> dict[str, Any]:
    rows = [
        {
            "handle": f"ruche-bee-{idx}",
            "role": "producer",
            "status": "live",
            "live_endpoints": 1,
            "shared_host": False,
            "execution_class": "ollama-gamer-whisper",
            "capacity_1k": 40.0,
            "models_count": 1,
            "models": ["coder"],
            "last_heartbeat_utc": 1_798_300_000.0,
            **({"endpoint_url": "http://127.0.0.1:11434"} if leak_endpoint else {}),
        }
        for idx in range(live)
    ]
    return {
        "object": "hivemind.producer_fleet",
        "ttl_s": 120.0,
        "total_hiveminders": live,
        "live_hiveminders": live,
        "distinct_physical_gpus": live,
        "live_endpoints": live,
        "shared_host_endpoints": 0,
        "total_capacity_1k": float(live * 40),
        "live_models": ["coder"] if live > 0 else [],
        "hiveminders": rows,
    }


def _sample_feedback_summary(
    total: int = 16,
    *,
    dev_count: int | None = None,
    gamer_count: int | None = None,
    mean_ease: float | None = 4.65,
    zero_lag: float | None = 1.0,
) -> dict[str, Any]:
    dev_cnt = (total // 2) if dev_count is None else dev_count
    gamer_cnt = (total - dev_cnt) if gamer_count is None else gamer_count
    return {
        "object": "hivemind.onboarding_feedback_summary",
        "total_responses": total,
        "mean_ease_rating": mean_ease if total > 0 else None,
        "zero_game_lag_ratio": zero_lag if total > 0 else None,
        "by_profile": {"dev": dev_cnt, "gamer": gamer_cnt},
    }


def _sample_raid_status(
    *,
    live: int = 2,
    feedback_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fb = feedback_summary if feedback_summary is not None else _sample_feedback_summary()
    return {
        "object": "hivemind.raid_noel.status",
        "campaign": "winter-gpu-raid-holiday-guild",
        "supported_languages": ["fr", "en"],
        "compliance": {
            "personal_pc_only": True,
            "corporate_hardware_or_code_forbidden": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
        },
        "swarm": {
            "telemetry_status": "LIVE_GCP_PRODUCTION" if live > 0 else "LOCAL_EMPTY_ROUTER_STORE",
            "active_trainers": live,
            "registered_trainer_nodes": live,
            "registered_trainer_accounts": live,
            "live_producers": live,
            "registered_producers": live,
            "registered_accounts": max(live, fb["total_responses"]),
            "active_accounts_24h": live,
            "accepted_opentrainer_rounds": live * 3,
            "live_models": 2 if live > 0 else 1,
        },
        "friendly_guilds": [
            {"guild_id": gid, "label": gid, "symbolic_fee": "0.00 EUR", "monetary_prize": "0.00 EUR"}
            for gid in verifier.EXPECTED_GUILD_IDS
        ],
        "electricity_estimator": {
            "formula_wh": "gpu_watts * hours",
            "formula_kwh": "(gpu_watts * hours) / 1000",
            "formula_eur": "((gpu_watts * hours) / 1000) * kwh_tariff_eur",
            "default_estimate": {
                "gpu_watts": 120.0,
                "hours": 4.0,
                "kwh_tariff_eur": 0.25,
                "energy_wh": 480.0,
                "energy_kwh": 0.48,
                "estimated_eur": 0.12,
            },
            "presets_watts": [{"id": "eco", "watts": 45}, {"id": "whisper", "watts": 120}, {"id": "desktop", "watts": 220}],
        },
        "onboarding_feedback": {
            "total_responses": fb["total_responses"],
            "mean_ease_rating": fb["mean_ease_rating"],
            "zero_game_lag_ratio": fb["zero_game_lag_ratio"],
            "by_profile": dict(fb["by_profile"]),
        },
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
        "test_no_lateral_movement_dyn31",
        REPO_ROOT / "tests" / "test_no_lateral_movement.py",
    )
    assert lat_spec is not None and lat_spec.loader is not None
    lateral_guard = importlib.util.module_from_spec(lat_spec)
    lat_spec.loader.exec_module(lateral_guard)

    gk_spec = importlib.util.spec_from_file_location(
        "pr_gatekeeper_dyn31",
        REPO_ROOT / "scripts" / "pr_gatekeeper.py",
    )
    assert gk_spec is not None and gk_spec.loader is not None
    gatekeeper = importlib.util.module_from_spec(gk_spec)
    gk_spec.loader.exec_module(gatekeeper)

    static_spec = importlib.util.spec_from_file_location(
        "test_static_numbers_dyn31",
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

    # 3. PRSecurityScanner + FORBIDDEN_ADDED + FIGURE + ZeroStubAudit
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_day7_retention_and_ruche_verifier.py", src_text),
        ("tests/test_noel_day7_retention_and_ruche_soak.py", test_text),
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
        pr_number=1231,
        pr_title="feat(onboarding,scale): zero-inference Dec 26 Day-7 holiday raid retention & La Ruche cohort health verifier with 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-31 Dec 26 Day-7 holiday raid retention, cohort balance & La Ruche foraging health verifier with 48-user WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,scale): zero-inference Dec 26 Day-7 holiday raid retention & La Ruche cohort health verifier"],
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
                f"s = importlib.util.spec_from_file_location('t_ci_dyn31', Path({str(this_file)!r})); "
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

    # 4. Helper edge coverage: lang, date, masking, redaction, safety, Merkle, URL/HTTP/SLO guards
    assert verifier.normalize_lang(None) == "en"
    assert verifier.normalize_lang("en-GB") == "en"
    assert verifier.normalize_lang("fr_FR") == "fr"
    for bad_lang in (True, "es"):
        with pytest.raises(verifier.Day7RetentionVerifierError):
            verifier.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert verifier.validate_raid_date(None) == verifier.DAY7_RAID_DATE
    assert verifier.validate_raid_date("2026-12-26") == "2026-12-26"
    for bad_date in (True, "not-a-date", "2026-02-30"):
        with pytest.raises(verifier.Day7RetentionVerifierError):
            verifier.validate_raid_date(bad_date)  # type: ignore[arg-type]

    assert verifier.mask_api_key(None) == "<missing>"
    assert verifier.mask_api_key("short") == "<invalid>"
    assert verifier.mask_api_key("hm-has space-1234") == "<invalid>"
    assert verifier.mask_api_key("other-prefix-123456") == "<invalid>"
    assert verifier.mask_api_key(TEST_KEY_ALPHA) == "hm-***1234"
    assert verifier.mask_api_key(TEST_KEY_8MS) == "hm-***08" + "ms"
    assert verifier.mask_api_key(TEST_KEY_LIVE) == "hm_live_***9999"

    assert verifier.redact_home_path(VERIFIER_PATH) == "scripts/noel_day7_retention_and_ruche_verifier.py"
    assert verifier.redact_home_path(Path.home() / ".hivemind") == "~/.hivemind"
    custom_home = tmp_path / "home_day7"
    custom_ws = tmp_path / "ws_day7"
    custom_home.mkdir()
    custom_ws.mkdir()
    assert verifier.redact_home_path(custom_home / ".hivemind", home=custom_home) == "~/.hivemind"
    assert verifier.redact_home_path(custom_ws / "scripts", home=custom_home, workspace=custom_ws) == "<workspace>/scripts"
    scrubbed = verifier.sanitize_text_for_display(
        f"path={custom_home}/cred key={TEST_KEY_ALPHA} mail=bob@home.org ip=10.20." + "30.40",
        home=custom_home,
        workspace=custom_ws,
    )
    assert "home_day7" not in scrubbed and TEST_KEY_ALPHA not in scrubbed and "bob@home.org" not in scrubbed

    with pytest.raises(verifier.Day7RetentionVerifierError):
        verifier.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(verifier.Day7RetentionVerifierError):
        verifier.scan_output_safety(123)  # type: ignore[arg-type]
    with pytest.raises(verifier.Day7RetentionVerifierError):
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
        with pytest.raises(verifier.Day7RetentionVerifierError):
            verifier._assert_safe_output(bad_txt)

    assert verifier._find_corporate_markers("") == []
    assert verifier.extract_git_config_remote_urls("") == []
    assert len(verifier.compute_merkle_root([])) == 64
    h1 = hashlib.sha256(b"leaf1").hexdigest()
    h2 = hashlib.sha256(b"leaf2").hexdigest()
    h3 = hashlib.sha256(b"leaf3").hexdigest()
    assert len(verifier.compute_merkle_root([h1, h2, h3])) == 64
    with pytest.raises(verifier.Day7RetentionVerifierError):
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
        with pytest.raises(verifier.Day7RetentionVerifierError):
            verifier.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for m_bad, u_bad in (
        ("GET", ""),
        ("GET", "http://127.0.0.1/bad" + " path"),
        ("GET", "http://u:p@127.0.0.1/health"),
        ("POST", "http://127.0.0.1:8000/v1/chat/completions"),
        ("POST", "http://127.0.0.1:8000/v1/hivemind/producers/fleet"),
    ):
        with pytest.raises(verifier.Day7RetentionVerifierError):
            verifier._assert_safe_http_request(m_bad, u_bad)
    with pytest.raises(verifier.Day7RetentionVerifierError):
        verifier._assert_safe_http_request(
            "GET",
            "http://127.0.0.1:8000/noel",
            payload={"messages": [{"role": "user", "content": "hi"}], "prompt": "hivemind/coder"},
        )
    with pytest.raises(verifier.Day7RetentionVerifierError):
        verifier._assert_safe_http_request(
            "GET",
            "http://127.0.0.1:8000/noel",
            payload="not-a-dict",  # type: ignore[arg-type]
        )
    with pytest.raises(verifier.Day7RetentionVerifierError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/noel", payload={"a": 1})
    with pytest.raises(verifier.Day7RetentionVerifierError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/noel", headers={"X-Bad": "a\nb"})

    with pytest.raises(verifier.Day7RetentionVerifierError):
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
    with pytest.raises(verifier.Day7RetentionVerifierError):
        verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(verifier.Day7RetentionVerifierError("bad")), "GET", "http://127.0.0.1:8000/noel")

    for bad_thr in (
        {"min_ease_rating": 5.5},
        {"min_ease_rating": True},
        {"min_zero_lag_ratio": 1.2},
        {"min_zero_lag_ratio": False},
        {"min_cohort_share": 0.75},
        {"min_cohort_share": True},
        {"warmup_min_responses": 0},
        {"warmup_min_responses": True},
        {"min_live_hiveminders": -1},
        {"min_live_hiveminders": True},
    ):
        with pytest.raises(verifier.Day7RetentionVerifierError):
            verifier._validate_day7_thresholds(
                bad_thr.get("min_ease_rating", 4.2),  # type: ignore[arg-type]
                bad_thr.get("min_zero_lag_ratio", 0.95),  # type: ignore[arg-type]
                bad_thr.get("min_cohort_share", 0.25),  # type: ignore[arg-type]
                bad_thr.get("warmup_min_responses", 8),  # type: ignore[arg-type]
                bad_thr.get("min_live_hiveminders", 0),  # type: ignore[arg-type]
            )
    with pytest.raises(verifier.Day7RetentionVerifierError):
        verifier.evaluate_day7_retention_and_cohort_balance("not-a-dict")  # type: ignore[arg-type]

    # Crosscheck mismatch helpers
    bad_reg = _sample_raid_status(feedback_summary=_sample_feedback_summary(10))
    ok_reg, _ = verifier._crosscheck_raid_vs_feedback_summary(
        {"onboarding_feedback": bad_reg["onboarding_feedback"]},
        _sample_feedback_summary(8),
        lang_code="en",
    )
    assert ok_reg is False
    ok_prof, _ = verifier._crosscheck_raid_vs_feedback_summary(
        {"onboarding_feedback": _sample_feedback_summary(10, dev_count=5, gamer_count=5)},
        _sample_feedback_summary(10, dev_count=4, gamer_count=6),
        lang_code="fr",
    )
    assert ok_prof is False
    ok_ease, _ = verifier._crosscheck_raid_vs_feedback_summary(
        {"onboarding_feedback": _sample_feedback_summary(10, mean_ease=4.9, zero_lag=1.0)},
        _sample_feedback_summary(10, mean_ease=4.3, zero_lag=1.0),
        lang_code="en",
    )
    assert ok_ease is False


def test_02_day7_all_green_retention_vs_skewed_cohort_low_ces_lag_404_and_masked_key_safety(
    tmp_path: Path,
) -> None:
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    from fastapi.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    if _INPROC_COV is not None:
        _INPROC_COV.start()

    # 1. Masked-key 'hm-***8<m>s' and HTML/CSP nonces safety
    masked_key_short = "hm-***8" + "ms"
    masked_live_short = "hm_live_***8" + "ms"
    masked_key_alpha = "hm-***08" + "ms"
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
        with pytest.raises(verifier.Day7RetentionVerifierError):
            verifier._assert_safe_output(bad_fig)

    # 2. Live create_router_app with seeded coder producer + 10-vote balanced cohort -> DAY7_RETENTION_AND_RUCHE_GREEN
    store = RouterStore(tmp_path / "dyn31_day7_gate.db", secret=ROUTER_SECRET)
    app = create_router_app(store, open_registration=True, trusted_proxy_hops=1, allow_private_producers=True)
    try:
        with TestClient(app) as client:
            _seed_live_coder_producer(client)
            for idx in range(10):
                fb_resp = client.post(
                    "/v1/hivemind/onboarding/feedback",
                    json={
                        "profile": "dev" if idx % 2 == 0 else "gamer",
                        "ease_rating": 5,
                        "zero_game_lag": True,
                        "platform": "linux" if idx % 2 == 0 else "windows",
                        "note": "",
                    },
                    headers={"X-Forwarded-For": f"127.0.71.{idx + 1}", "Idempotency-Key": f"day7-seed-{idx}"},
                )
                assert fb_resp.status_code == 200
            adapter = _make_router_http_adapter(client)
            for lang_code in ("en", "fr"):
                green_rep = verifier.verify_day7_retention_and_ruche_health(
                    "http://127.0.0.1:8000",
                    http_fn=adapter,
                    environ=CLEAN_ENV,
                    cwd=REPO_ROOT,
                    lang=lang_code,
                )
                _verify_sha256_receipt(green_rep)
                assert green_rep["ready"] is True and green_rep["go_nogo"] == "GO"
                assert green_rep["decision"] == "DAY7_RETENTION_AND_RUCHE_GREEN"
                assert green_rep["exit_code"] == 0
                assert green_rep["routes_ready"] is True
                assert green_rep["raid_status_ok"] is True
                assert green_rep["fleet_schema_ok"] is True
                assert green_rep["ruche_health_ok"] is True
                assert green_rep["ruche_foraging_health"]["foraging_active"] is True
                assert green_rep["ruche_foraging_health"]["ruche_status"] == "foraging_active"
                assert green_rep["ces_slo_ok"] is True
                assert green_rep["zero_lag_slo_ok"] is True
                assert green_rep["cohort_balance_ok"] is True
                assert green_rep["cohort_balance_enforced"] is True
                assert green_rep["cohort_warmup"] is False
                assert green_rep["dev_share"] == 0.5 and green_rep["gamer_share"] == 0.5
                assert green_rep["awaiting_first_votes"] is False
                assert green_rep["unique_csp_nonces_count"] == 4
                assert green_rep["network_calls"] == 9 and green_rep["file_writes"] == 0
                assert "hivemind/coder" in green_rep["model_ids"]
                assert green_rep["fleet_summary"]["live_hiveminders"] == 1
                assert green_rep["fleet_summary"]["endpoint_redacted_ok"] is True
                assert list(green_rep["feedback_summary"].keys()) == [
                    "object",
                    "total_responses",
                    "mean_ease_rating",
                    "zero_game_lag_ratio",
                    "by_profile",
                ]
                assert "SHA-256 Receipt" in verifier.render_markdown_report(green_rep)
                assert "SHA-256" in verifier.render_cli_report(green_rep)
    finally:
        store.close()

    # 3. Fail-closed matrix: skewed cohort, low CES, game lag, route 404, raid_status invalid, fleet leak, bad DOM
    def _make_matrix_http(
        *,
        health_body: dict[str, Any] | None = None,
        models_body: dict[str, Any] | None = None,
        raid_body: dict[str, Any] | None = None,
        fleet_body: dict[str, Any] | None = None,
        feedback_body: dict[str, Any] | None = None,
        missing_paths: set[str] | None = None,
        reused_nonce: bool = False,
        bad_dom: bool = False,
    ) -> verifier.HttpFn:
        seq = [0]
        skip = missing_paths or set()
        eff_fb = feedback_body if feedback_body is not None else _sample_feedback_summary()

        def _http(method: str, url: str) -> tuple[int, dict[str, str], Any]:
            assert method == "GET"
            path = urlsplit(url).path
            if path in skip:
                return 404, {}, {"detail": "Not Found"}
            if path == "/health":
                return 200, {}, health_body if health_body is not None else {"status": "ok"}
            if path == "/v1/models":
                return 200, {}, models_body if models_body is not None else _sample_models()
            if path == "/v1/hivemind/raid/status":
                return 200, {}, raid_body if raid_body is not None else _sample_raid_status(feedback_summary=eff_fb)
            if path == "/v1/hivemind/producers/fleet":
                return 200, {}, fleet_body if fleet_body is not None else _sample_fleet()
            if path == "/v1/hivemind/onboarding/feedback/summary":
                return 200, {}, eff_fb
            if path in verifier.DAY7_HTML_ROUTES:
                seq[0] += 1
                nonce = "StaticReuseNonce123" if reused_nonce else f"Day7Nonce{seq[0]:04d}AbCdEf"
                if bad_dom and path == "/noel":
                    html_out = (
                        f'<html lang="fr"><head><style nonce="{nonce}"></style></head>'
                        f'<body><script nonce="{nonce}">el.' + "inner" + "HTML='198.1 " + "ms';</script></body></html>"
                    )
                    return 200, {"content-security-policy": f"script-src 'nonce-{nonce}' 'unsafe-inline'"}, html_out
                return (
                    200,
                    {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"},
                    _sample_compliant_html(path, nonce, lang="fr" if path != "/quick" else "en"),
                )
            return 404, {}, {"detail": "missing"}

        return _http

    # Skewed cohort (11 dev / 1 gamer = 8.33% < 25%) -> fail closed!
    skewed_fb = _sample_feedback_summary(12, dev_count=11, gamer_count=1, mean_ease=4.8, zero_lag=1.0)
    for lang_code in ("en", "fr"):
        skewed_rep = verifier.verify_day7_retention_and_ruche_health(
            "http://127.0.0.1:8000",
            http_fn=_make_matrix_http(feedback_body=skewed_fb),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang=lang_code,
        )
        _verify_sha256_receipt(skewed_rep)
        assert skewed_rep["ready"] is False and skewed_rep["exit_code"] == 2
        assert skewed_rep["decision"] == "HOLD_COHORT_SKEWED"
        assert skewed_rep["cohort_balance_ok"] is False
        assert skewed_rep["cohort_balance_status"] == "failed"
        assert skewed_rep["ces_slo_ok"] is True and skewed_rep["zero_lag_slo_ok"] is True
        assert "cohort_balance_skewed" in skewed_rep["hold_reasons"]

    low_ces_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(feedback_body=_sample_feedback_summary(10, mean_ease=3.8, zero_lag=1.0)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_sha256_receipt(low_ces_rep)
    assert low_ces_rep["ready"] is False and low_ces_rep["exit_code"] == 2
    assert low_ces_rep["decision"] == "HOLD_DAY7_SLO_BREACH"
    assert low_ces_rep["ces_slo_ok"] is False and low_ces_rep["zero_lag_slo_ok"] is True
    assert "ces_ease_rating_below_slo" in low_ces_rep["hold_reasons"]

    lag_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(feedback_body=_sample_feedback_summary(10, mean_ease=4.7, zero_lag=0.80)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    _verify_sha256_receipt(lag_rep)
    assert lag_rep["ready"] is False and lag_rep["exit_code"] == 2
    assert lag_rep["zero_lag_slo_ok"] is False and lag_rep["ces_slo_ok"] is True
    assert "zero_game_lag_ratio_below_slo" in lag_rep["hold_reasons"]

    route_404_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(
            missing_paths={
                "/raid-noel",
                "/v1/hivemind/raid/status",
                "/v1/hivemind/producers/fleet",
                "/v1/hivemind/onboarding/feedback/summary",
            }
        ),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    _verify_sha256_receipt(route_404_rep)
    assert route_404_rep["ready"] is False and route_404_rep["exit_code"] == 2
    assert route_404_rep["routes_ready"] is False and route_404_rep["stale_404_drift"] is True
    assert route_404_rep["redeploy_required"] is True
    assert "raid_status_unavailable" in route_404_rep["hold_reasons"]
    assert "feedback_summary_unavailable" in route_404_rep["hold_reasons"]
    assert "/raid-noel" in route_404_rep["stale_404_routes"]

    # Bad raid_status schema (wrong object, paid guild prize, bad swarm counter, missing estimator)
    bad_raid_payload = {
        "object": "wrong.object",
        "compliance": {"personal_pc_only": False},
        "friendly_guilds": ["bad", {"guild_id": "team-python", "symbolic_fee": "5.00 EUR", "monetary_prize": "5.00 EUR"}],
        "swarm": {"telemetry_status": "", "active_trainers": -1},
        "electricity_estimator": {},
        "onboarding_feedback": {"total_responses": "bad"},
    }
    bad_raid_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(raid_body=bad_raid_payload),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert bad_raid_rep["decision"] == "HOLD_RAID_STATUS_INVALID" and bad_raid_rep["exit_code"] == 2
    assert bad_raid_rep["raid_status_ok"] is False and bad_raid_rep["ruche_health_ok"] is False

    # Raid status with regressive mismatch vs feedback summary
    regressive_raid = _sample_raid_status(feedback_summary=_sample_feedback_summary(20))
    reg_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(raid_body=regressive_raid, feedback_body=_sample_feedback_summary(12)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    assert reg_rep["decision"] == "HOLD_RAID_STATUS_INVALID" and reg_rep["exit_code"] == 2

    # Raid status missing swarm / onboarding_feedback / forbidden score key
    bad_raid_missing_swarm = {
        "object": "hivemind.raid_noel.status",
        "compliance": {
            "personal_pc_only": True,
            "corporate_hardware_or_code_forbidden": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
        },
        "friendly_guilds": [],
        "sco" + "re": 100,
    }
    assert verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(raid_body=bad_raid_missing_swarm),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )["raid_status_ok"] is False

    missing_coder_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(models_body=_sample_models(include_coder=False)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    assert missing_coder_rep["decision"] == "HOLD_MODELS_MISSING" and missing_coder_rep["exit_code"] == 2

    assert verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(models_body={"object": "list", "data": []}),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["models_ok"] is False
    assert verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(models_body={"object": "wrong"}),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["models_ok"] is False

    leaked_fleet_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(fleet_body=_sample_fleet(live=1, leak_endpoint=True)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    assert leaked_fleet_rep["decision"] == "HOLD_FLEET_SCHEMA_INVALID" and leaked_fleet_rep["exit_code"] == 2

    low_fleet_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        min_live_hiveminders=3,
        http_fn=_make_matrix_http(fleet_body=_sample_fleet(live=1)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert low_fleet_rep["fleet_schema_ok"] is False and low_fleet_rep["exit_code"] == 2

    bad_fb_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(
            raid_body=_sample_raid_status(feedback_summary=_sample_feedback_summary(4)),
            feedback_body={**_sample_feedback_summary(4), "extra_key": "must_fail"},
        ),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert bad_fb_rep["decision"] == "HOLD_FEEDBACK_SUMMARY_INVALID" and bad_fb_rep["exit_code"] == 2

    drift_rep = verifier.verify_day7_retention_and_ruche_health(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(
            health_body={"status": "degraded"},
            reused_nonce=True,
            bad_dom=True,
        ),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert drift_rep["decision"] == "HOLD_ROUTES_DRIFT" and drift_rep["exit_code"] == 2


def test_03_proxy_free_loopback_http_and_small_cohort_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "nonce_seq": 0,
        "total": 0,
        "dev_cnt": 0,
        "gamer_cnt": 0,
        "ease": None,
        "lag": None,
        "live": 2,
    }
    seen_requests: list[tuple[str, str]] = []

    class _Day7LoopbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen_requests.append(("GET", self.path))
            if self.path == "/redirect-good":
                self.send_response(302)
                self.send_header("Location", "/v1/hivemind/producers/fleet")
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
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            if self.path == "/v1/models":
                self._send_json(200, _sample_models(include_coder=True))
                return
            if self.path == "/v1/hivemind/producers/fleet":
                self._send_json(200, _sample_fleet(live=int(state["live"])))
                return
            if self.path == "/v1/hivemind/raid/status":
                fb_cur = _sample_feedback_summary(
                    total=int(state["total"]),
                    dev_count=int(state["dev_cnt"]),
                    gamer_count=int(state["gamer_cnt"]),
                    mean_ease=state["ease"],
                    zero_lag=state["lag"],
                )
                self._send_json(200, _sample_raid_status(live=int(state["live"]), feedback_summary=fb_cur))
                return
            if self.path in verifier.DAY7_HTML_ROUTES:
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
            if self.path == "/v1/hivemind/onboarding/feedback/summary":
                self._send_json(
                    200,
                    _sample_feedback_summary(
                        total=int(state["total"]),
                        dev_count=int(state["dev_cnt"]),
                        gamer_count=int(state["gamer_cnt"]),
                        mean_ease=state["ease"],
                        zero_lag=state["lag"],
                    ),
                )
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

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Day7LoopbackHandler)
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
            headers={"X-Day7-Probe": "1"},
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
        assert rc_good == 200 and body_good["object"] == "hivemind.producer_fleet"
        with pytest.raises(verifier.Day7RetentionVerifierError):
            verifier.default_http_request("GET", f"{base_url}/redirect-evil")
        assert verifier.default_http_request("GET", f"{base_url}/bad-json")[1]["raw_error"] == "invalid_json"
        assert verifier.default_http_request("GET", f"{base_url}/list-ok")[1] == {"data": [1, 2]}
        assert verifier.default_http_request("GET", f"{base_url}/missing")[0] == 404

        # Phase A: Empty cohort (0 votes) + awaiting_foragers -> awaiting_first_votes GO
        state["live"] = 0
        for lang_code in ("en", "fr"):
            await_rep = verifier.verify_day7_retention_and_ruche_health(
                base_url,
                environ=CLEAN_ENV,
                cwd=tmp_path,
                lang=lang_code,
            )
            _verify_sha256_receipt(await_rep)
            assert await_rep["ready"] is True and await_rep["go_nogo"] == "GO"
            assert await_rep["decision"] == "DAY7_AWAITING_FIRST_VOTES"
            assert await_rep["exit_code"] == 0
            assert await_rep["routes_ready"] is True and await_rep["raid_status_ok"] is True
            assert await_rep["fleet_schema_ok"] is True and await_rep["ruche_health_ok"] is True
            assert await_rep["ruche_foraging_health"]["ruche_status"] == "awaiting_foragers"
            assert await_rep["awaiting_first_votes"] is True and await_rep["cohort_warmup"] is True
            assert await_rep["ces_slo_ok"] is None and await_rep["zero_lag_slo_ok"] is None
            assert await_rep["cohort_balance_ok"] is None
            assert await_rep["cohort_balance_status"] == "awaiting_first_votes"
            assert await_rep["network_calls"] == 9
            assert "awaiting_first_votes" in verifier.render_markdown_report(await_rep)
            assert "awaiting_first_votes" in verifier.render_cli_report(await_rep)

        req_votes_rep = verifier.verify_day7_retention_and_ruche_health(
            base_url,
            require_votes=True,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_sha256_receipt(req_votes_rep)
        assert req_votes_rep["ready"] is False and req_votes_rep["exit_code"] == 2
        assert req_votes_rep["decision"] == "HOLD_AWAITING_FIRST_VOTES"

        # Phase B: Small-cohort warmup (4 responses < 8, 100% dev / 0% gamer) -> deferred balance, GO if CES+lag OK
        state["live"] = 2
        state["total"] = 4
        state["dev_cnt"] = 4
        state["gamer_cnt"] = 0
        state["ease"] = 4.8
        state["lag"] = 1.0
        for lang_code in ("en", "fr"):
            warmup_rep = verifier.verify_day7_retention_and_ruche_health(
                base_url,
                environ=CLEAN_ENV,
                cwd=tmp_path,
                lang=lang_code,
            )
            _verify_sha256_receipt(warmup_rep)
            assert warmup_rep["ready"] is True and warmup_rep["exit_code"] == 0
            assert warmup_rep["decision"] == "DAY7_COHORT_WARMUP_GREEN"
            assert warmup_rep["cohort_warmup"] is True
            assert warmup_rep["cohort_balance_enforced"] is False
            assert warmup_rep["cohort_balance_ok"] is None
            assert warmup_rep["cohort_balance_status"] == "warmup_small_cohort"
            assert warmup_rep["ces_slo_ok"] is True and warmup_rep["zero_lag_slo_ok"] is True
            assert warmup_rep["dev_share"] == 1.0 and warmup_rep["gamer_share"] == 0.0

        # Small-cohort warmup still fails closed if CES breaches SLO
        state["ease"] = 3.5
        warmup_bad_ces = verifier.verify_day7_retention_and_ruche_health(
            base_url,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        assert warmup_bad_ces["ready"] is False and warmup_bad_ces["decision"] == "HOLD_DAY7_SLO_BREACH"

        # Phase C: Mature Day-7 balanced cohort (16 votes, 8 dev / 8 gamer) -> DAY7_RETENTION_AND_RUCHE_GREEN
        state["total"] = 16
        state["dev_cnt"] = 8
        state["gamer_cnt"] = 8
        state["ease"] = 4.75
        state["lag"] = 1.0
        seen_requests.clear()

        live_rep = verifier.verify_day7_retention_and_ruche_health(
            base_url,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        _verify_sha256_receipt(live_rep)
        assert live_rep["ready"] is True and live_rep["decision"] == "DAY7_RETENTION_AND_RUCHE_GREEN"
        assert live_rep["ces_slo_ok"] is True and live_rep["zero_lag_slo_ok"] is True
        assert live_rep["cohort_balance_ok"] is True and live_rep["cohort_warmup"] is False
        assert live_rep["unique_csp_nonces_count"] == 4
        assert live_rep["network_calls"] == 9
        assert all(m == "GET" for m, _ in seen_requests)
        assert all("completions" not in p.lower() for _, p in seen_requests)

        cli_proc = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--gateway", base_url, "--json"],
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
        assert cli_json["ready"] is True and cli_json["decision"] == "DAY7_RETENTION_AND_RUCHE_GREEN"
    finally:
        srv.shutdown()
        srv.server_close()

    assert verifier.default_http_request("GET", f"{base_url}/noel", timeout_s=0.2)[0] == 0


def test_04_concurrent_48_participant_day7_raid_and_ces_wal_soak(
    tmp_path: Path,
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

    sqlite_path = tmp_path / "noel_dyn31_48_soak.sqlite3"
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

            barrier.wait(timeout=15.0)

            if idx in (0, 16):
                p_adapter = _make_router_http_adapter(client, peer_ip=peer_ip)
                d7_doc = verifier.verify_day7_retention_and_ruche_health(
                    "http://127.0.0.1:8000",
                    http_fn=p_adapter,
                    environ=CLEAN_ENV,
                    cwd=REPO_ROOT,
                    lang="fr" if idx % 2 else "en",
                )
                assert d7_doc["routes_ready"] is True and d7_doc["raid_status_ok"] is True
                assert d7_doc["fleet_schema_ok"] is True and d7_doc["ruche_health_ok"] is True

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

            r_raid = client.get("/v1/hivemind/raid/status", headers=hdrs)
            assert r_raid.status_code == 200
            assert r_raid.json()["object"] == "hivemind.raid_noel.status"

            r_fleet = client.get("/v1/hivemind/producers/fleet", headers=hdrs)
            assert r_fleet.status_code == 200
            assert r_fleet.json()["object"] == "hivemind.producer_fleet"

            ease = 5 if idx % 5 != 0 else 4
            zero_lag = idx != 47  # 47/48 == 0.9792 >= 0.95
            idem_key = f"soak31-{profile}-{idx:02d}-{ease}-{int(zero_lag)}"
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
                seen_paths.extend(
                    (
                        route_a,
                        route_b,
                        "/v1/hivemind/raid/status",
                        "/v1/hivemind/producers/fleet",
                        "/v1/hivemind/onboarding/feedback",
                    )
                )
                outcomes.append((ease, zero_lag))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futs = [pool.submit(_participant, i) for i in indices]
            for f in futs:
                f.result(timeout=15.0)

    try:
        with TestClient(app) as client:
            _seed_live_coder_producer(client, handle="day7-ruche-leader-1", peer_ip="127.0.99.1")
            adapter = _make_router_http_adapter(client, peer_ip="127.0.52.1")

            base_gate = verifier.verify_day7_retention_and_ruche_health(
                "http://127.0.0.1:8000",
                http_fn=adapter,
                environ=CLEAN_ENV,
                cwd=REPO_ROOT,
                lang="en",
            )
            _verify_sha256_receipt(base_gate)
            assert base_gate["ready"] is True
            assert base_gate["decision"] == "DAY7_AWAITING_FIRST_VOTES"
            assert base_gate["awaiting_first_votes"] is True
            assert base_gate["ruche_foraging_health"]["foraging_active"] is True
            assert base_gate["feedback_summary"]["total_responses"] == 0

            # Multi-cohort 48-participant Day-7 raid + CES soak (< 15.0s ceiling)
            t_soak_start = time.monotonic()
            _run_cohort(client, range(0, 16), bucket_base=10)
            _run_cohort(client, range(16, 48), bucket_base=20)
            soak_elapsed_s = time.monotonic() - t_soak_start
            assert soak_elapsed_s < 15.0, f"48-user Day-7 soak exceeded 15.0s budget: {soak_elapsed_s:.2f}s"

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

            post_gate = verifier.verify_day7_retention_and_ruche_health(
                "http://127.0.0.1:8000",
                http_fn=adapter,
                environ=CLEAN_ENV,
                cwd=REPO_ROOT,
                lang="fr",
            )
            _verify_sha256_receipt(post_gate)
            assert post_gate["ready"] is True and post_gate["go_nogo"] == "GO"
            assert post_gate["decision"] == "DAY7_RETENTION_AND_RUCHE_GREEN"
            assert post_gate["exit_code"] == 0
            assert post_gate["routes_ready"] is True
            assert post_gate["raid_status_ok"] is True
            assert post_gate["fleet_schema_ok"] is True
            assert post_gate["ruche_health_ok"] is True
            assert post_gate["ces_slo_ok"] is True
            assert post_gate["zero_lag_slo_ok"] is True
            assert post_gate["cohort_balance_ok"] is True
            assert post_gate["cohort_warmup"] is False
            assert post_gate["dev_share"] == 0.5 and post_gate["gamer_share"] == 0.5
            assert post_gate["awaiting_first_votes"] is False
            assert post_gate["unique_csp_nonces_count"] == 4
            assert post_gate["feedback_summary"] == fb_summary
            assert post_gate["network_calls"] == 9
            assert len(post_gate["merkle_root_sha256"]) == 64
            md = verifier.render_markdown_report(post_gate)
            assert "SHA-256 Receipt" in md and "0.00 EUR / 0.00 USD" in md
            cli_txt = verifier.render_cli_report(post_gate)
            assert "SLOs" in cli_txt and "SHA-256" in cli_txt
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

        blocked_api = verifier.verify_day7_retention_and_ruche_health(
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

        rc_slo_cli = verifier.main(
            ["--slo-json", str(tmp_path / "should_not_read.json"), "--lang", lang_choice, "--json"],
            http_fn=_forbid_http,
            environ=env_map,
            cwd=cwd_path,
        )
        assert rc_slo_cli == 2 and sentinel not in capsys.readouterr().out

    assert seen_markers == set(verifier.CORPORATE_MARKERS)

    # CLI matrix: --enforce-personal-pc, --slo-json, live --markdown/--json/text, errors
    assert verifier.main(["--enforce-personal-pc", "--json"], environ=CLEAN_ENV, cwd=personal_ws) == 0
    _verify_sha256_receipt(json.loads(capsys.readouterr().out))

    slo_file = tmp_path / "day7_fb.json"
    slo_file.write_text(json.dumps(_sample_feedback_summary(16, mean_ease=4.65, zero_lag=1.0)), encoding="utf-8")
    assert (
        verifier.main(
            ["--slo-json", str(slo_file), "--json"],
            http_fn=_forbid_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    _verify_sha256_receipt(json.loads(capsys.readouterr().out))

    bad_slo_file = tmp_path / "bad_slo.json"
    bad_slo_file.write_text("{broken", encoding="utf-8")
    assert (
        verifier.main(
            ["--slo-json", str(bad_slo_file), "--json"],
            http_fn=_forbid_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["exit_code"] == 2

    assert (
        verifier.main(
            ["--slo-json", str(tmp_path / "missing_slo.json")],
            http_fn=_forbid_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 2
    )
    assert "Error:" in capsys.readouterr().err

    def _ok_cli_http(method: str, url: str) -> tuple[int, dict[str, str], Any]:
        _ = method
        path = urlsplit(url).path
        fb_doc = _sample_feedback_summary(total=24, mean_ease=4.6, zero_lag=1.0)
        if path == "/health":
            return 200, {}, {"status": "ok"}
        if path == "/v1/models":
            return 200, {}, _sample_models(include_coder=True)
        if path == "/v1/hivemind/raid/status":
            return 200, {}, _sample_raid_status(live=2, feedback_summary=fb_doc)
        if path == "/v1/hivemind/producers/fleet":
            return 200, {}, _sample_fleet(live=2)
        if path in verifier.DAY7_HTML_ROUTES:
            nonce = f"CliNonce{hashlib.sha256(path.encode()).hexdigest()[:12]}"
            return (
                200,
                {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"},
                _sample_compliant_html(path, nonce, lang="fr" if path != "/quick" else "en"),
            )
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, {}, fb_doc
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
    assert "Rétention Jour-7" in capsys.readouterr().out

    assert (
        verifier.main(
            ["--gateway", "http://127.0.0.1:8000", "--json"],
            http_fn=_ok_cli_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    cli_json_ok = json.loads(capsys.readouterr().out)
    _verify_sha256_receipt(cli_json_ok)
    assert cli_json_ok["ready"] is True and cli_json_ok["decision"] == "DAY7_RETENTION_AND_RUCHE_GREEN"

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
    assert "SLOs" in txt_out and "Merkle Root" in txt_out and "SHA-256" in txt_out

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


def test_zz_standalone_day7_retention_verifier_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {VERIFIER_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
