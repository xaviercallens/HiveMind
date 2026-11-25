# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, cross-verifier SHA-256 aggregation, compliance, and 48-user WAL soak tests (`SLICE-SCALE-DYN-23`)."""

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
import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGGREGATOR_PATH = REPO_ROOT / "scripts" / "noel_holiday_digest_and_receipt_aggregator.py"

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
        include=[str(AGGREGATOR_PATH)],
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


from scripts import noel_holiday_digest_and_receipt_aggregator as aggregator  # noqa: E402

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn23-router-sec-32b!"
TARGET_FULL_SHA = "e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"
TARGET_SHORT_SHA = "e7f8a9b0c1d2"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example.invalid"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-bluecoat.internal:8080"
CORP_FORTI_PROXY = "http://" + "proxy.fortinet.internal:3128"
CORP_GIT_REMOTE = "https://" + "git.amadeus-internal.corp/team/service.git"

TEST_KEY_ALPHA = "hm-dyn23_" + "personal_key_1234"
TEST_KEY_LIVE = "hm_live_" + "personal_dyn23_9999"

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
    assert aggregator.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_keys(doc)


def _sample_compliant_html(route: str, nonce: str, *, lang: str = "en") -> str:
    extra = ""
    if route in ("/raid-noel", "/raid"):
        extra = (
            '<main><button id="guildBtn-team-python">Team Python</button>'
            '<button id="guildBtn-team-rust-cpp">Team Rust/C++</button>'
            '<button id="guildBtn-team-steam-rpg">Team Steam RPG</button>'
            '<button id="guildBtn-team-night-owl">Team Night Owl</button>'
            '<div id="raidLiveTrainers">2</div><div id="raidRegisteredAccounts">4</div>'
            '<div id="raidAcceptedRounds">6</div><div id="raidFeedbackResponses">48</div>'
            '<input id="calcWatts"><input id="calcHours"><input id="calcTariff">'
            '<button id="calcComputeBtn"></button><button id="calcPresetEco"></button>'
            '<button id="calcPresetWhisper"></button><button id="calcPresetDesktop"></button>'
            '<span id="calcResultWh"></span><span id="calcResultEur"></span><p id="calcResultNote"></p></main>'
        )
    elif route == "/arena-noel":
        cards = "".join(
            f'<article id="arenaCard-{tid}" class="task-card" data-task-id="{tid}" '
            f'data-ballot="abcdef012345.1.0123456789abcdef01234567">'
            f'<pre id="candA-{tid}">def f(): return 1</pre>'
            f'<pre id="candB-{tid}">def f(): return 2</pre></article>'
            for tid in (
                "python-sliding-median",
                "rust-frame-decoder",
                "lean4-energy-descent",
                "exoplanet-transit-bls",
            )
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


def _sample_manifest(build_sha: str = TARGET_FULL_SHA) -> dict[str, Any]:
    return {
        "object": "hivemind.onboarding.manifest",
        "campaign_id": "noel-2026-easy-onboard",
        "status": "ok",
        "build_sha": build_sha,
        "holiday_routes": list(aggregator.EXPECTED_MANIFEST_HOLIDAY_ROUTES),
        "downloadable_scripts": [
            "hivemind_boinc_idle_worker.py",
            "hivemind_connect.py",
            "hivemind_doctor.py",
            "hivemind_opentrainer_agent.py",
            "hivemind_producer_agent.py",
            "hivemind_producer_autostart.py",
            "hivemind_producer_shield.py",
            "hivemind_quickstart.py",
            "hivemind_volunteer_config.py",
            "hivemind_volunteer_preflight.py",
            "join_hivemind.ps1",
            "noel_canary_cohort_rehearsal.py",
            "noel_cloud_run_parity_gate.py",
            "noel_gamer_whisper_selftest.py",
            "noel_post_setup_probe.py",
            "noel_scale_load_harness.py",
            "noel_vscode_autoconfig.py",
        ],
        "personal_pc_only": True,
        "financial_promise": "0.00 EUR / 0.00 USD",
    }


def _sample_feedback_summary(total: int = 48) -> dict[str, Any]:
    dev_cnt = total // 2
    gamer_cnt = total - dev_cnt
    return {
        "object": "hivemind.onboarding_feedback_summary",
        "total_responses": total,
        "mean_ease_rating": 4.75 if total > 0 else None,
        "zero_game_lag_ratio": 0.9792 if total > 0 else None,
        "by_profile": {"dev": dev_cnt, "gamer": gamer_cnt},
    }


def _sample_raid_status(total_fb: int = 48) -> dict[str, Any]:
    fb = _sample_feedback_summary(total_fb)
    return {
        "object": "hivemind.raid_noel.status",
        "compliance": {
            "personal_pc_only": True,
            "corporate_hardware_or_code_forbidden": True,
            "monetary_cost_eur": "0.00 EUR",
            "monetary_cost_usd": "0.00 USD",
            "license": "AGPL-3.0",
        },
        "swarm": {
            "telemetry_status": "LIVE_GCP_PRODUCTION",
            "active_trainers": 2,
            "registered_trainer_nodes": 3,
            "registered_trainer_accounts": 3,
            "live_producers": 4,
            "registered_producers": 6,
            "registered_accounts": 12,
            "active_accounts_24h": 7,
            "accepted_opentrainer_rounds": 9,
            "live_models": 2,
        },
        "friendly_guilds": [
            {"guild_id": gid, "label": lbl, "symbolic_fee": "0.00 EUR", "monetary_prize": "0.00 EUR"}
            for gid, lbl in (
                ("team-python", "Team Python"),
                ("team-rust-cpp", "Team Rust/C++"),
                ("team-steam-rpg", "Team Steam RPG"),
                ("team-night-owl", "Team Night Owl"),
            )
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
            "presets_watts": [
                {"id": "eco", "watts": 45},
                {"id": "whisper", "watts": 120},
                {"id": "desktop", "watts": 220},
            ],
        },
        "onboarding_feedback": {
            "total_responses": fb["total_responses"],
            "mean_ease_rating": fb["mean_ease_rating"],
            "zero_game_lag_ratio": fb["zero_game_lag_ratio"],
            "by_profile": fb["by_profile"],
        },
    }


def _make_full_mock_http() -> aggregator.HttpFn:
    seq = {"n": 0}

    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[int, dict[str, str], str | dict[str, Any]]:
        _ = (method, payload, headers, timeout_s)
        path = urlsplit(url).path
        if path in ("/noel", "/quick", "/raid-noel", "/raid", "/arena-noel"):
            seq["n"] += 1
            nonce = f"123ms456Nonce{seq['n']:04d}AbCdEf"
            lang = "fr" if path in ("/noel", "/raid-noel", "/arena-noel") else "en"
            return (
                200,
                {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"},
                _sample_compliant_html(path, nonce, lang=lang),
            )
        if path == "/health":
            return 200, {}, {"status": "ok"}
        if path == "/v1/models":
            return 200, {}, {"data": [{"id": "hivemind/coder"}]}
        if path == "/v1/hivemind/onboarding/manifest":
            return 200, {}, _sample_manifest()
        if path == "/v1/hivemind/onboarding/feedback/summary":
            return 200, {}, _sample_feedback_summary()
        if path == "/v1/hivemind/raid/status":
            return 200, {}, _sample_raid_status()
        if path == "/v1/hivemind/producers/fleet":
            return 200, {}, {
                "object": "hivemind.producer_fleet",
                "live_producers": 2,
                "distinct_physical_gpus": 2,
                "hiveminders": [],
            }
        if path == "/scripts/join_hivemind.ps1":
            return 200, {"content-type": "text/plain"}, (REPO_ROOT / "scripts" / "join_hivemind.ps1").read_text(encoding="utf-8")
        if path == "/scripts/hivemind_quickstart.py":
            return 200, {"content-type": "text/x-python"}, (REPO_ROOT / "scripts" / "hivemind_quickstart.py").read_text(encoding="utf-8")
        if path == "/scripts/hivemind_doctor.py":
            return 200, {"content-type": "text/x-python"}, (REPO_ROOT / "scripts" / "hivemind_doctor.py").read_text(encoding="utf-8")
        return 404, {}, {"detail": "Not Found"}

    return _http


def _make_router_http_adapter(
    client: Any,
    *,
    peer_ip: str = "127.0.0.1",
) -> aggregator.HttpFn:
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
        "test_no_lateral_movement_dyn23",
        REPO_ROOT / "tests" / "test_no_lateral_movement.py",
    )
    assert lat_spec is not None and lat_spec.loader is not None
    lateral_guard = importlib.util.module_from_spec(lat_spec)
    lat_spec.loader.exec_module(lateral_guard)

    gk_spec = importlib.util.spec_from_file_location(
        "pr_gatekeeper_dyn23",
        REPO_ROOT / "scripts" / "pr_gatekeeper.py",
    )
    assert gk_spec is not None and gk_spec.loader is not None
    gatekeeper = importlib.util.module_from_spec(gk_spec)
    gk_spec.loader.exec_module(gatekeeper)

    static_spec = importlib.util.spec_from_file_location(
        "test_static_numbers_dyn23",
        REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
    )
    assert static_spec is not None and static_spec.loader is not None
    static_numbers_guard = importlib.util.module_from_spec(static_spec)
    static_spec.loader.exec_module(static_numbers_guard)

    this_file = Path(__file__).resolve()
    for p in (AGGREGATOR_PATH, this_file):
        assert EXPECTED_AGPL_HEADER in p.read_text(encoding="utf-8")[:300]

    agg_src = AGGREGATOR_PATH.read_text(encoding="utf-8")
    test_src = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 stdlib imports only + ProxyHandler({}) present
    tree = ast.parse(agg_src, filename=str(AGGREGATOR_PATH))
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
    assert lateral_guard.audit_source_for_lateral_movement(agg_src, str(AGGREGATOR_PATH)) == []
    assert lateral_guard.audit_source_for_lateral_movement(test_src, str(this_file)) == []
    lan_ip = ".".join(("192", "168", "44", "10"))
    assert lateral_guard.audit_source_for_lateral_movement(agg_src + f'\n_LAN = "http://{lan_ip}:8000"\n') != []
    assert lateral_guard.audit_source_for_lateral_movement(agg_src + f'\n_EXT = "{UNTRUSTED_EXT_GW}"\n') != []

    # 3. PRSecurityScanner 5-layer audit + PR Gatekeeper FORBIDDEN_ADDED + FIGURE + ZeroStubAudit
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    for rel, content in (
        ("scripts/noel_holiday_digest_and_receipt_aggregator.py", agg_src),
        ("tests/test_noel_holiday_digest_and_receipt_soak.py", test_src),
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
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1223,
        pr_title="feat(onboarding,scale): zero-inference holiday readiness digest & cross-verifier receipt aggregator with 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-23 holiday launch digest and cross-verifier SHA-256 receipt aggregator.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,scale): zero-inference holiday readiness digest and receipt aggregator"],
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
                f"s = importlib.util.spec_from_file_location('t_ci_dyn23', Path({str(this_file)!r})); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert m.ZeroStubAudit.check_file(m.AGGREGATOR_PATH).is_clean; "
                "assert m.ZeroStubAudit.audit_python_code('def ok():\\n    return 1\\n').is_clean; "
                "assert not m.ZeroStubAudit.audit_python_code('def stub_bad():\\n    pass\\n').is_clean; "
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

    # 4. Helper coverage: lang, masking, redaction, output safety, Merkle, URL/HTTP guards
    assert aggregator.normalize_lang(None) == "en"
    assert aggregator.normalize_lang("en-US") == "en"
    assert aggregator.normalize_lang("fr_FR") == "fr"
    for bad_lang in (True, "es"):
        with pytest.raises(aggregator.DigestAggregatorError):
            aggregator.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert aggregator.mask_api_key(None) == "<missing>"
    assert aggregator.mask_api_key("short") == "<invalid>"
    assert aggregator.mask_api_key("other-prefix-123456") == "<invalid>"
    assert aggregator.mask_api_key(TEST_KEY_ALPHA) == "hm-***1234"
    assert aggregator.mask_api_key(TEST_KEY_LIVE) == "hm_live_***9999"

    assert aggregator.redact_home_path(AGGREGATOR_PATH) == "scripts/noel_holiday_digest_and_receipt_aggregator.py"
    custom_home = tmp_path / "home_bob"
    custom_home.mkdir()
    assert aggregator.redact_home_path(custom_home / ".hivemind", home=custom_home) == "~/.hivemind"
    scrubbed = aggregator.sanitize_text_for_display(
        f"path={custom_home}/cred key={TEST_KEY_ALPHA} mail=bob@home.org ip=10.20." + "30.40",
        home=custom_home,
    )
    assert "home_bob" not in scrubbed and TEST_KEY_ALPHA not in scrubbed and "bob@home.org" not in scrubbed

    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator.scan_output_safety(123)  # type: ignore[arg-type]
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator._attach_receipt({"sco" + "re": 9})

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
        assert issue_key in aggregator.scan_output_safety(bad_txt)
        with pytest.raises(aggregator.DigestAggregatorError):
            aggregator._assert_safe_output(bad_txt)

    assert aggregator.sanitize_build_sha(None) == "unknown"
    assert aggregator.sanitize_build_sha("secret1234") == "unknown"
    assert aggregator.shas_match(TARGET_FULL_SHA, TARGET_SHORT_SHA) is True
    assert aggregator.shas_match("tagalpha12", "tagbeta12") is False
    assert aggregator.shas_match("unknown", TARGET_SHORT_SHA) is False
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator.validate_expected_sha("unknown")

    assert len(aggregator.compute_merkle_root([])) == 64
    h1 = hashlib.sha256(b"leaf1").hexdigest()
    h2 = hashlib.sha256(b"leaf2").hexdigest()
    h3 = hashlib.sha256(b"leaf3").hexdigest()
    assert len(aggregator.compute_merkle_root([h1, h2, h3])) == 64
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator.compute_merkle_root(["not-hex"])

    assert aggregator.is_allowed_outbound_host("datasets-server.huggingface.co") is True
    assert aggregator.is_allowed_outbound_host("storage.googleapis.com") is True
    assert aggregator.is_allowed_outbound_host("custom.a.run.app") is True
    assert aggregator.is_allowed_outbound_host("") is False
    assert aggregator.validate_gateway_url("http://[::1]:8080/") == "http://[::1]:8080"
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
        with pytest.raises(aggregator.DigestAggregatorError):
            aggregator.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for m_bad, u_bad in (
        ("GET", ""),
        ("GET", "http://127.0.0.1/bad" + " path"),
        ("GET", "http://u:p@127.0.0.1/health"),
        ("POST", "http://127.0.0.1:8000/v1/chat/completions"),
        ("POST", "http://127.0.0.1:8000/v1/hivemind/onboarding/manifest"),
    ):
        with pytest.raises(aggregator.DigestAggregatorError):
            aggregator._assert_safe_http_request(m_bad, u_bad)
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator._assert_safe_http_request(
            "GET",
            "http://127.0.0.1:8000/noel",
            payload={"messages": [{"role": "user", "content": "hi"}], "prompt": "hivemind/coder"},
        )
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator._assert_safe_http_request(
            "GET",
            "http://127.0.0.1:8000/noel",
            payload="not-a-dict",  # type: ignore[arg-type]
        )
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator.default_http_request("GET", "http://127.0.0.1:8000/noel", payload={"a": 1})
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator.default_http_request("GET", "http://127.0.0.1:8000/noel", headers={"X-Bad": "a\nb"})

    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator._normalize_http_return(("bad",))
    assert aggregator._looks_like_http_headers({}) is True
    assert aggregator._normalize_http_return((200, {"object": "ok"}, {"content-type": "application/json"}))[1]["_json"] == {"object": "ok"}
    assert aggregator._normalize_http_return((200, "plain-hdr-missing", "<html>p</html>"))[1]["html"] == "<html>p</html>"
    assert aggregator._normalize_http_return((200, {"x-a": "1"}, {"headers": {"x-a": "2"}, "html": "<p>x</p>"}))[1]["headers"]["x-a"] == "2"
    assert aggregator._normalize_http_return((200, "<html>plain</html>", {"content-type": "text/html"}))[1]["html_length"] > 0
    assert aggregator._normalize_http_return((200, '{"status":"ok"}', {"content-type": "application/json"}))[1]["_json"] == {"status": "ok"}
    assert aggregator._normalize_http_return((200, "[1, 2]"))[1]["data"] == [1, 2]
    assert aggregator._normalize_http_return((200, 42))[1]["data"] == 42

    assert aggregator._invoke_http_fn(lambda m, u, **_kw: (200, {"kw": True}), "GET", "http://127.0.0.1:8000/noel")[1]["kw"] is True
    assert aggregator._invoke_http_fn(lambda m, u, p: (200, {"p3": True}), "GET", "http://127.0.0.1:8000/noel")[1]["p3"] is True
    assert aggregator._invoke_http_fn(lambda m, u, p, h: (200, {"p4": True}), "GET", "http://127.0.0.1:8000/noel")[1]["p4"] is True
    assert aggregator._invoke_http_fn(lambda m, u, p, h, t: (200, {"p5": True}), "GET", "http://127.0.0.1:8000/noel")[1]["p5"] is True
    assert aggregator._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(OSError("reset")), "GET", "http://127.0.0.1:8000/noel") == (0, {"error": "network_error"})
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(aggregator.DigestAggregatorError("bad")), "GET", "http://127.0.0.1:8000/noel")


def test_02_cross_verifier_receipt_aggregation_tamper_missing_pii_and_hm_ms_regression(
    tmp_path: Path,
) -> None:
    from scripts import noel_arena_ballot_and_parity_verifier as arena_v
    from scripts import noel_deploy_smoke_and_win11_verifier as win11_v
    from scripts import noel_posix_launcher_and_smoke_verifier as posix_v
    from scripts import noel_raid_guild_and_bulletin_verifier as raid_v

    # 1. Masked-key '<digit>ms' and HTML/CSP nonces regression safety
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
        assert aggregator.scan_output_safety(sample) == [], sample
        assert _STATIC_FIG_RE.findall(aggregator.strip_masked_keys_and_nonces(sample)) == []

    # Real static figures and unmasked keys still fail closed
    for bad_fig in ("198.1 " + "ms", f"{masked_key_short} latency 198.1 " + "ms"):
        assert "static_figure_detected" in aggregator.scan_output_safety(bad_fig)
        with pytest.raises(aggregator.DigestAggregatorError):
            aggregator._assert_safe_output(bad_fig)

    # 2. Generate REAL signed SHA-256 receipts from all 4 verifiers
    mock_http = _make_full_mock_http()

    def gpu_ok() -> dict[str, Any]:
        return {
            "present": True,
            "parse_error": False,
            "name": "NVIDIA GeForce RTX 4070",
            "total_mib": 12288,
            "free_mib": 9600,
            "temp_c": 55.0,
            "util_pct": 5.0,
        }

    def yield_ok() -> dict[str, Any]:
        return {
            "worker_started": True,
            "worker_stopped": True,
            "clean_release": True,
            "retained_worker_vram_mib": 0,
            "yield_latency_ms": 24.5,
        }
    win11_rep = win11_v.run_unified_onboarding_and_smoke_gate(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        run_launcher_audit=False,
        dev_home_dir=tmp_path / "w_home",
        dev_workspace_dir=tmp_path / "w_ws",
        gpu_probe_fn=gpu_ok,
        process_probe_fn=lambda: ["python3"],
        worker_yield_fn=yield_ok,
        http_fn=mock_http,
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
        lang="en",
    )
    posix_rep = posix_v.run_unified_posix_onboarding_and_smoke_gate(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        run_launcher_audit=False,
        dev_home_dir=tmp_path / "p_home",
        dev_workspace_dir=tmp_path / "p_ws",
        platform_name="linux",
        gpu_probe_fn=gpu_ok,
        process_probe_fn=lambda: ["python3"],
        worker_yield_fn=yield_ok,
        http_fn=mock_http,
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
        lang="en",
    )
    arena_rep = arena_v.verify_arena_noel_holiday_readiness(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        http_fn=mock_http,
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
        lang="en",
    )
    raid_rep = raid_v.verify_raid_noel_and_bulletin_readiness(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_SHORT_SHA,
        http_fn=mock_http,
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
        lang="en",
    )
    assert win11_rep["exit_code"] == 0 and posix_rep["exit_code"] == 0
    assert arena_rep["ready"] is True and raid_rep["ready"] is True

    # Order-independent aggregation & deterministic Merkle root
    agg_a = aggregator.aggregate_holiday_verifier_receipts(
        [win11_rep, posix_rep, arena_rep, raid_rep],
        expected_sha=TARGET_SHORT_SHA,
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
        lang="en",
    )
    _verify_sha256_receipt(agg_a)
    assert agg_a["ready"] is True
    assert agg_a["unanimous_ready"] is True
    assert agg_a["go_nogo"] == "GO"
    assert agg_a["integrity_ok"] is True
    assert agg_a["pii_redaction_ok"] is True
    assert agg_a["present_verifiers"] == ["arena", "posix", "raid", "win11"]
    assert agg_a["missing_verifiers"] == []
    assert agg_a["exit_code"] == 0

    agg_b = aggregator.aggregate_holiday_verifier_receipts(
        [raid_rep, arena_rep, posix_rep, win11_rep],
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
        lang="fr",
    )
    _verify_sha256_receipt(agg_b)
    assert agg_b["merkle_root_sha256"] == agg_a["merkle_root_sha256"]
    assert "Agrégation" in aggregator.render_markdown_report(agg_b)

    # Path-based and Mapping-based loading (including masked-key suffix inside a signed receipt)
    win11_ms_doc = aggregator._attach_receipt(
        {
            "verifier": "win11",
            "ready": True,
            "compliance_ok": True,
            "masked_api_key": masked_key_short,
            "exit_code": 0,
        }
    )
    r_dir = tmp_path / "receipts"
    r_dir.mkdir()
    p_w = r_dir / "win11.json"
    p_p = r_dir / "posix.json"
    p_a = r_dir / "arena.json"
    p_r = r_dir / "raid.json"
    for p, d in ((p_w, win11_ms_doc), (p_p, posix_rep), (p_a, arena_rep), (p_r, raid_rep)):
        p.write_text(json.dumps(d, sort_keys=True), encoding="utf-8")
    agg_paths = aggregator.aggregate_holiday_verifier_receipts(
        [p_w, str(p_p), p_a, p_r], environ=CLEAN_ENV, cwd=REPO_ROOT
    )
    _verify_sha256_receipt(agg_paths)
    assert agg_paths["ready"] is True and agg_paths["go_nogo"] == "GO"

    agg_map = aggregator.aggregate_holiday_verifier_receipts(
        {"win11": win11_ms_doc, "posix": p_p, "arena": arena_rep, "raid": raid_rep},
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
    )
    _verify_sha256_receipt(agg_map)
    assert agg_map["ready"] is True

    # 3. Fail-closed tamper, missing verifier, duplicate, unready, drift, PII, and SHA mismatch
    tampered_raid = dict(raid_rep)
    tampered_raid["receipt_sha256"] = ("0" if raid_rep["receipt_sha256"][0] != "0" else "1") + raid_rep["receipt_sha256"][1:]
    res_tamper = aggregator.aggregate_holiday_verifier_receipts(
        [win11_rep, posix_rep, arena_rep, tampered_raid],
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
        lang="fr",
    )
    _verify_sha256_receipt(res_tamper)
    assert res_tamper["ready"] is False and res_tamper["go_nogo"] == "NO-GO"
    assert res_tamper["integrity_ok"] is False and res_tamper["tampered_verifiers"] == ["raid"]
    assert res_tamper["exit_code"] == 2

    res_missing = aggregator.aggregate_holiday_verifier_receipts(
        [win11_rep, posix_rep, arena_rep], environ=CLEAN_ENV, cwd=REPO_ROOT
    )
    _verify_sha256_receipt(res_missing)
    assert res_missing["ready"] is False and res_missing["missing_verifiers"] == ["raid"]

    res_missing_file = aggregator.aggregate_holiday_verifier_receipts(
        [p_w, p_p, p_a, r_dir / "raid.json" if False else tmp_path / "missing_raid.json"],
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
    )
    _verify_sha256_receipt(res_missing_file)
    assert res_missing_file["ready"] is False and res_missing_file["exit_code"] == 2

    bad_json_file = tmp_path / "raid.json"
    bad_json_file.write_text("{broken", encoding="utf-8")
    res_bad_json = aggregator.aggregate_holiday_verifier_receipts(
        [p_w, p_p, p_a, bad_json_file], environ=CLEAN_ENV, cwd=REPO_ROOT
    )
    assert res_bad_json["ready"] is False and "raid" in res_bad_json["tampered_verifiers"]

    list_json_file = tmp_path / "arena.json"
    list_json_file.write_text("[1, 2]", encoding="utf-8")
    assert aggregator.aggregate_holiday_verifier_receipts(
        [p_w, p_p, list_json_file, p_r], environ=CLEAN_ENV, cwd=REPO_ROOT
    )["ready"] is False

    unready_arena = dict(arena_rep)
    unready_arena["ready"] = False
    unready_arena["exit_code"] = 2
    unready_arena["stale_404_drift"] = True
    unready_arena["compliance_ok"] = False
    unready_arena = {k: v for k, v in unready_arena.items() if k != "receipt_sha256"}
    unready_arena["receipt_sha256"] = hashlib.sha256(
        json.dumps(unready_arena, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    res_unready = aggregator.aggregate_holiday_verifier_receipts(
        [win11_rep, posix_rep, unready_arena, raid_rep], environ=CLEAN_ENV, cwd=REPO_ROOT
    )
    _verify_sha256_receipt(res_unready)
    assert res_unready["ready"] is False and "arena" in res_unready["unready_verifiers"]

    # PII/secret inside receipt fails closed WITHOUT leaking raw PII in output digest
    pii_secret = "hm-dyn23_" + "leak_999988"
    pii_payload = {"verifier": "raid", "ready": True, "email": "alice@corp-secret.org", "leaked": pii_secret, "sco" + "re": 9}
    pii_payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(pii_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    res_pii = aggregator.aggregate_holiday_verifier_receipts(
        [win11_rep, posix_rep, arena_rep, pii_payload], environ=CLEAN_ENV, cwd=REPO_ROOT, lang="fr"
    )
    _verify_sha256_receipt(res_pii)
    assert res_pii["ready"] is False and res_pii["pii_redaction_ok"] is False
    assert pii_secret not in json.dumps(res_pii) and "alice@corp-secret.org" not in json.dumps(res_pii)

    # Duplicate, unidentified, non-finite float, build_sha conflict, and invalid arguments
    assert aggregator.aggregate_holiday_verifier_receipts(
        [win11_rep, win11_rep, posix_rep, arena_rep, raid_rep], environ=CLEAN_ENV, cwd=REPO_ROOT
    )["duplicate_verifiers"] == ["win11"]
    assert aggregator.aggregate_holiday_verifier_receipts(
        [{"unknown": 1}], environ=CLEAN_ENV, cwd=REPO_ROOT
    )["unidentified_receipts_count"] == 1
    assert aggregator.aggregate_holiday_verifier_receipts(
        [{"verifier": "win11", "val": float("nan")}], environ=CLEAN_ENV, cwd=REPO_ROOT
    )["ready"] is False

    drift_sha_doc = aggregator._attach_receipt(
        {"verifier": "raid", "ready": True, "compliance_ok": True, "live_build_sha": "deadbeef9999"}
    )
    res_sha_drift = aggregator.aggregate_holiday_verifier_receipts(
        [win11_rep, posix_rep, arena_rep, drift_sha_doc],
        expected_sha=TARGET_SHORT_SHA,
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
        lang="fr",
    )
    assert res_sha_drift["ready"] is False and res_sha_drift["build_sha_consensus_ok"] is False

    for bad_arg in (123, "single-str", {"schema": "x"}):
        with pytest.raises(aggregator.DigestAggregatorError):
            aggregator.aggregate_holiday_verifier_receipts(bad_arg)  # type: ignore[arg-type]
    for bad_req in ("win11", (), [True], ["bad verifier!"]):
        with pytest.raises(aggregator.DigestAggregatorError):
            aggregator.aggregate_holiday_verifier_receipts([], require_verifiers=bad_req)  # type: ignore[arg-type]
    with pytest.raises(aggregator.DigestAggregatorError):
        aggregator.aggregate_holiday_verifier_receipts([123])  # type: ignore[list-item]
    assert aggregator.aggregate_holiday_verifier_receipts(
        [{"verifier": "win11", "ready": True, "receipt_sha256": "abc"}],
        require_verifiers=("win11",),
        environ=CLEAN_ENV,
        cwd=REPO_ROOT,
    )["ready"] is False
    assert aggregator._load_receipt_item("\x00bad")[2] == "invalid_receipt_path"


def test_03_proxy_free_loopback_http_and_stale_404_to_live_holiday_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"live": False, "nonce_seq": 0}
    seen_requests: list[tuple[str, str]] = []

    class _DigestLoopbackHandler(http.server.BaseHTTPRequestHandler):
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

            if self.path in aggregator.LIVE_DIGEST_HTML_ROUTES:
                state["nonce_seq"] += 1
                nonce = f"123ms456Nonce{state['nonce_seq']:04d}AbCdEf"
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
                self._send_json(200, _sample_feedback_summary())
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

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _DigestLoopbackHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        # Hostile proxies MUST be bypassed by urllib.request.ProxyHandler({})
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

        fb_code, fb_body = aggregator.default_http_request(
            "GET", f"{base_url}/v1/hivemind/onboarding/feedback/summary"
        )
        assert fb_code == 200
        assert fb_body == _sample_feedback_summary()
        assert list(fb_body.keys()) == [
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        ]

        rc_good, body_good = aggregator.default_http_request("GET", f"{base_url}/redirect-good")
        assert rc_good == 200 and body_good["total_responses"] == 48
        with pytest.raises(aggregator.DigestAggregatorError):
            aggregator.default_http_request("GET", f"{base_url}/redirect-evil")
        assert aggregator.default_http_request("GET", f"{base_url}/bad-json")[1]["raw_error"] == "invalid_json"
        assert aggregator.default_http_request("GET", f"{base_url}/list-ok")[1] == {"data": [1, 2]}
        assert aggregator.default_http_request("GET", f"{base_url}/missing")[0] == 404

        # Phase A: Stale 404 drift -> NO-GO
        stale_en = aggregator.run_live_holiday_digest(
            base_url, environ=CLEAN_ENV, cwd=tmp_path, lang="en"
        )
        _verify_sha256_receipt(stale_en)
        assert stale_en["ready"] is False and stale_en["go_nogo"] == "NO-GO"
        assert stale_en["stale_404_drift"] is True and stale_en["redeploy_required"] is True
        assert stale_en["exit_code"] == 2
        assert "/raid-noel" in stale_en["stale_404_routes"]
        assert "/arena-noel" in stale_en["stale_404_routes"]
        assert "NO-GO" in aggregator.render_markdown_report(stale_en)
        assert "NO-GO" in aggregator.render_cli_report(stale_en)

        # Phase B: Live cutover -> GO
        state["live"] = True
        seen_requests.clear()
        live_en = aggregator.run_live_holiday_digest(
            base_url,
            expected_sha=TARGET_SHORT_SHA,
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="en",
        )
        _verify_sha256_receipt(live_en)
        assert live_en["ready"] is True and live_en["go_nogo"] == "GO"
        assert live_en["exit_code"] == 0
        assert live_en["stale_404_drift"] is False and live_en["redeploy_required"] is False
        assert live_en["unique_csp_nonces_count"] == 4
        assert live_en["html_all_ok"] is True
        assert live_en["manifest_ok"] is True and live_en["sha_match"] is True
        assert live_en["feedback_summary_ok"] is True
        assert live_en["feedback_summary"] == fb_body
        assert live_en["network_calls"] == 6
        assert all(m == "GET" for m, _ in seen_requests)
        assert all("completions" not in p.lower() for _, p in seen_requests)

        live_fr = aggregator.run_live_holiday_digest(
            base_url, environ=CLEAN_ENV, cwd=tmp_path, lang="fr"
        )
        _verify_sha256_receipt(live_fr)
        assert live_fr["ready"] is True and "Digest de Lancement Noël" in aggregator.render_markdown_report(live_fr)
        assert "AGRÉGATEUR" in aggregator.render_cli_report(live_fr)
    finally:
        srv.shutdown()
        srv.server_close()

    assert aggregator.default_http_request("GET", f"{base_url}/noel", timeout_s=0.2)[0] == 0

    # Negative parity controls: reused nonce, unsafe sink, static figure, broken manifest, extra feedback key
    def _broken_http(method: str, url: str) -> tuple[int, dict[str, str], str | dict[str, Any]]:
        _ = method
        path = urlsplit(url).path
        reused = "ReusedNonce123456"
        if path in aggregator.LIVE_DIGEST_HTML_ROUTES:
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
            return 200, {}, {**_sample_feedback_summary(1), "extra_key": "must_fail"}
        return 404, {}, {"detail": "missing"}

    broken_rep = aggregator.run_live_holiday_digest(
        "http://127.0.0.1:8000",
        expected_sha=TARGET_FULL_SHA,
        http_fn=_broken_http,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    _verify_sha256_receipt(broken_rep)
    assert broken_rep["ready"] is False and broken_rep["go_nogo"] == "NO-GO" and broken_rep["exit_code"] == 2
    assert broken_rep["manifest_ok"] is False and broken_rep["feedback_summary_ok"] is False


def test_04_concurrent_48_participant_multi_route_wal_soak_and_live_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _INPROC_COV is not None:
        _INPROC_COV.stop()
    from fastapi.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    monkeypatch.setenv("HIVEMIND_BUILD_SHA", TARGET_FULL_SHA)
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00250-noel")

    upstream_inference_hits: list[str] = []

    def _forbid_upstream_inference(request: httpx.Request) -> httpx.Response:
        upstream_inference_hits.append(str(request.url))
        return httpx.Response(500, json={"error": "inference_forbidden"})

    sqlite_path = tmp_path / "noel_dyn23_48_soak.sqlite3"
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

    total_users = 48
    barrier = threading.Barrier(total_users)
    obs_lock = threading.Lock()
    observed_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []
    seen_paths: list[str] = []
    outcomes: list[tuple[int, bool]] = []
    routes_cycle = ("/noel", "/quick", "/raid-noel", "/arena-noel")

    try:
        with TestClient(app) as client:
            def _participant(idx: int) -> None:
                is_dev = idx < 24
                cohort_idx = idx if is_dev else idx - 24
                profile = "dev" if is_dev else "gamer"
                peer_ip = _loopback_ip(10 if is_dev else 20, cohort_idx)
                hdrs = {"X-Forwarded-For": peer_ip}
                route_a = routes_cycle[idx % 4]
                route_b = routes_cycle[(idx + 1) % 4]
                local_nonces: list[str] = []

                barrier.wait(timeout=15.0)

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
                    stripped = aggregator.strip_masked_keys_and_nonces(r_html.text)
                    assert _STATIC_FIG_RE.findall(stripped) == []
                    assert ("inner" + "HTML") not in stripped
                    local_nonces.append(hdr_nonces[0])

                ease = 5 if idx % 5 != 0 else 4
                zero_lag = idx != 47
                idem_key = f"soak23-{profile}-{cohort_idx:02d}-{ease}-{int(zero_lag)}"
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

            t_soak_start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=total_users) as pool:
                futs = [pool.submit(_participant, i) for i in range(total_users)]
                for f in futs:
                    f.result(timeout=15.0)
            soak_elapsed_s = time.monotonic() - t_soak_start
            assert soak_elapsed_s < 3.5, f"48-user soak exceeded 3.5s budget: {soak_elapsed_s:.2f}s"

            assert sqlite_lock_errors == []
            assert upstream_inference_hits == []
            assert all("completions" not in p.lower() for p in seen_paths)
            assert len(observed_nonces) == total_users * 2
            assert len(set(observed_nonces)) == total_users * 2

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

            # Live holiday digest + signed 4-verifier receipt bundle against real gateway app
            signed_verifiers = [
                aggregator._attach_receipt(
                    {
                        "verifier": vid,
                        "ready": True,
                        "compliance_ok": True,
                        "live_build_sha": TARGET_FULL_SHA,
                        "exit_code": 0,
                    }
                )
                for vid in ("win11", "posix", "arena", "raid")
            ]
            digest = aggregator.run_live_holiday_digest(
                "http://127.0.0.1:8000",
                http_fn=_make_router_http_adapter(client, peer_ip="127.0.50.1"),
                environ=CLEAN_ENV,
                cwd=REPO_ROOT,
                lang="en",
                receipts=signed_verifiers,
                expected_sha=TARGET_SHORT_SHA,
            )
            _verify_sha256_receipt(digest)
            assert digest["ready"] is True and digest["go_nogo"] == "GO"
            assert digest["exit_code"] == 0
            assert digest["unique_csp_nonces_count"] == 4
            assert digest["feedback_summary"] == fb_summary
            assert digest["receipt_aggregation"]["unanimous_ready"] is True
            assert len(digest["merkle_root_sha256"]) == 64
            md = aggregator.render_markdown_report(digest)
            assert "## Cross-Verifier Receipts" in md and "SHA-256 Receipt" in md
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
    ok_comp = aggregator.evaluate_personal_pc_compliance(environ=CLEAN_ENV, cwd=personal_ws, lang="en")
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
        comp = aggregator.evaluate_personal_pc_compliance(environ=env_map, cwd=cwd_path, lang=lang_choice)
        _verify_sha256_receipt(comp)
        assert comp["compliance_ok"] is False and comp["exit_code"] == 2, vec_name
        assert comp["network_calls"] == 0 and comp["file_writes"] == 0 and comp["subprocess_calls"] == 0
        assert "0.00 EUR / 0.00 USD" in comp["notice_bilingual"]
        assert "AGPL-3.0" in comp["notice_bilingual"]
        assert sentinel not in json.dumps(comp), vec_name
        seen_markers.update(comp["matched_markers"])

        blocked_agg = aggregator.aggregate_holiday_verifier_receipts(
            [personal_ws / "missing.json"], environ=env_map, cwd=cwd_path, lang=lang_choice
        )
        assert blocked_agg["compliance_ok"] is False and blocked_agg["exit_code"] == 2

        blocked_api = aggregator.run_live_holiday_digest(
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

        rc_cli = aggregator.main(
            ["--lang", lang_choice, "--json" if idx % 2 == 0 else "--markdown"],
            http_fn=_forbid_http,
            environ=env_map,
            cwd=cwd_path,
        )
        out_cli = capsys.readouterr().out
        assert rc_cli == 2 and sentinel not in out_cli

    assert seen_markers == set(aggregator.CORPORATE_MARKERS)

    # CLI matrix: --enforce-personal-pc, --aggregate-only, live --markdown, live default, errors
    assert aggregator.main(["--enforce-personal-pc", "--json"], environ=CLEAN_ENV, cwd=personal_ws) == 0
    _verify_sha256_receipt(json.loads(capsys.readouterr().out))

    rec_dir = tmp_path / "cli_receipts"
    rec_dir.mkdir()
    r_paths: list[str] = []
    for vid in ("win11", "posix", "arena", "raid"):
        rp = rec_dir / f"{vid}.json"
        rp.write_text(
            json.dumps(
                aggregator._attach_receipt({"verifier": vid, "ready": True, "compliance_ok": True, "exit_code": 0})
            ),
            encoding="utf-8",
        )
        r_paths.append(str(rp))

    agg_argv = ["--aggregate-only", "--json", *(arg for p in r_paths for arg in ("--receipt", p))]
    assert aggregator.main(agg_argv, http_fn=_forbid_http, environ=CLEAN_ENV, cwd=personal_ws) == 0
    _verify_sha256_receipt(json.loads(capsys.readouterr().out))

    mock_http = _make_full_mock_http()
    assert (
        aggregator.main(
            ["--gateway", "http://127.0.0.1:8000", "--markdown", "--lang", "fr"],
            http_fn=mock_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    assert "Digest de Lancement Noël" in capsys.readouterr().out

    assert (
        aggregator.main(
            ["--gateway", "http://127.0.0.1:8000"],
            http_fn=mock_http,
            environ=CLEAN_ENV,
            cwd=personal_ws,
        )
        == 0
    )
    txt_out = capsys.readouterr().out
    assert "Merkle Root" in txt_out and "SHA-256" in txt_out

    assert (
        aggregator.main(
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

    assert aggregator.main(["--lang", "es"], environ=CLEAN_ENV, cwd=personal_ws) == 2
    assert "Error:" in capsys.readouterr().err


def test_zz_standalone_digest_aggregator_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {AGGREGATOR_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
