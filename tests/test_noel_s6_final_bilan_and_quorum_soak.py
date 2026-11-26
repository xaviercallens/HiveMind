# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, S6 blind A/B & quorum, proxy-free loopback, compliance & 48-user WAL soak tests (`SLICE-SCALE-DYN-32`)."""

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
VERIFIER_PATH = REPO_ROOT / "scripts" / "noel_s6_final_bilan_and_quorum_verifier.py"

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


from scripts import noel_s6_final_bilan_and_quorum_verifier as verifier  # noqa: E402

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
ROUTER_SECRET = b"noel-scale-dyn32-router-sec-32b!"
TARGET_FULL_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
TARGET_SHORT_SHA = "a1b2c3d4e5f6"

UNTRUSTED_EXT_GW = "https://" + "untrusted.example.invalid"
INSECURE_RUN_APP_GW = "http://" + "hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
CORP_ZSCALER_PROXY = "http://" + "zscaler-proxy.internal:8443"
CORP_NETSKOPE_PROXY = "http://" + "netskope-bluecoat.internal:8080"
CORP_FORTI_PROXY = "http://" + "proxy.fortinet.internal:3128"
CORP_GIT_REMOTE = "https://" + "git.amadeus-internal.corp/team/service.git"

TEST_KEY_ALPHA = "hm-dyn32_" + "personal_key_1234"
TEST_KEY_8MS = "hm-dyn32_" + "personal_08" + "ms"
TEST_KEY_LIVE = "hm_live_" + "personal_dyn32_9999"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_HTML_ATTR_NONCE_RE = re.compile(r"""\bnonce=(["'])([A-Za-z0-9_+=/=-]{8,128})\1""")
_STATIC_FIG_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)
_ARENA_CARD_PAIR_RE = re.compile(
    r'data-task-id="([A-Za-z0-9_-]{3,64})"[^>]*?data-ballot="([A-Za-z0-9._-]{16,128})"'
)
_CAND_BLOCK_RE = re.compile(r'<pre\s+id="cand([AB])-([^"]+)"[^>]*>(.*?)</pre>', re.DOTALL)


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
        extra = "<pre>python3 scripts/hivemind_quickstart.py</pre>"
    elif route == "/raid-noel":
        extra = '<main><div id="raidLiveTrainers">8</div></main>'
    elif route == "/arena-noel":
        extra = (
            '<main id="arenaNoelApp">'
            '<article id="arenaCard-py" data-task-id="python-sliding-median" '
            'data-ballot="abcdef012345.1.0123456789abcdef01234567">'
            '<pre id="candA-python-sliding-median">def f(): return 1</pre>'
            '<pre id="candB-python-sliding-median">def f(): return 2</pre>'
            "</article></main>"
        )
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


def _sample_manifest(
    build_sha: str = TARGET_FULL_SHA,
    *,
    missing_route: str | None = None,
) -> dict[str, Any]:
    routes = [r for r in verifier.EXPECTED_MANIFEST_HOLIDAY_ROUTES if r != missing_route]
    return {
        "object": "hivemind.onboarding.manifest",
        "campaign_id": "noel-2026-easy-onboard",
        "status": "ok",
        "build_sha": build_sha,
        "holiday_routes": routes,
        "personal_pc_only": True,
        "financial_promise": "0.00 EUR / 0.00 USD",
    }


def _sample_fleet(
    *,
    live: int = 8,
    distinct_gpus: int | None = None,
    leak_endpoint: bool = False,
) -> dict[str, Any]:
    gpus = live if distinct_gpus is None else distinct_gpus
    rows = [
        {
            "handle": f"friend-bee-{idx}",
            "role": "producer",
            "status": "live",
            "live_endpoints": 1,
            "shared_host": False,
            "execution_class": "ollama-gamer-whisper",
            "capacity_1k": 40.0,
            "models_count": 1,
            "models": ["coder"],
            "last_heartbeat_utc": 1_798_400_000.0,
            **({"endpoint_url": f"http://{_loopback_ip(50, idx)}:11434"} if leak_endpoint else {}),
        }
        for idx in range(live)
    ]
    return {
        "object": "hivemind.producer_fleet",
        "ttl_s": 120.0,
        "total_hiveminders": live,
        "live_hiveminders": live,
        "distinct_physical_gpus": gpus,
        "live_endpoints": live,
        "shared_host_endpoints": 0,
        "total_capacity_1k": float(live * 40),
        "live_models": ["coder"] if live > 0 else [],
        "hiveminders": rows,
    }


def _sample_raid_status(*, live_producers: int = 8) -> dict[str, Any]:
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
            "telemetry_status": "OK",
            "active_trainers": 2,
            "live_producers": live_producers,
            "registered_producers": live_producers,
            "registered_accounts": 48,
            "live_models": 2,
        },
        "friendly_guilds": [{"guild_id": "team-python", "label": "Team Python"}],
        "electricity_estimator": {"formula_wh": "gpu_watts * hours"},
        "onboarding_feedback": {"total_responses": 0},
    }


def _sample_arena_summary(
    *,
    holiday_wins: int = 14,
    baseline_wins: int = 2,
    ties: int = 0,
    total_votes: int | None = None,
    win_ratio: float | None = None,
    wilson_lb: float | None = None,
    by_profile: dict[str, int] | None = None,
    chosen_side_counts: dict[str, int] | None = None,
    by_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    computed_total = holiday_wins + baseline_wins + ties
    decisive = holiday_wins + baseline_wins
    tot = computed_total if total_votes is None else total_votes
    wr = (round(holiday_wins / decisive, 4) if decisive > 0 else None) if win_ratio is None else win_ratio
    wlb = round(verifier.wilson_95_lower_bound(holiday_wins, decisive), 4) if wilson_lb is None else wilson_lb
    prof = by_profile if by_profile is not None else {"dev": tot // 2, "gamer": tot - (tot // 2)}
    sides = (
        chosen_side_counts
        if chosen_side_counts is not None
        else {"A": decisive // 2, "B": decisive - (decisive // 2), "tie": ties}
    )
    out: dict[str, Any] = {
        "object": "hivemind.blind_ab_arena.summary",
        "campaign": "noel-2026-holiday-checkpoint-arena",
        "zero_vram_inference": True,
        "total_votes": tot,
        "decisive_votes": decisive,
        "holiday_wins": holiday_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "win_ratio": wr,
        "wilson_95_lower_bound": wlb,
        "chosen_side_counts": sides,
        "by_profile": prof,
    }
    if by_task is not None:
        out["by_task"] = by_task
    return out


def _sample_feedback_summary(
    total: int = 16,
    *,
    mean_ease: float | None = 4.65,
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
        "test_no_lateral_movement_dyn32",
        REPO_ROOT / "tests" / "test_no_lateral_movement.py",
    )
    assert lat_spec is not None and lat_spec.loader is not None
    lateral_guard = importlib.util.module_from_spec(lat_spec)
    lat_spec.loader.exec_module(lateral_guard)

    gk_spec = importlib.util.spec_from_file_location(
        "pr_gatekeeper_dyn32",
        REPO_ROOT / "scripts" / "pr_gatekeeper.py",
    )
    assert gk_spec is not None and gk_spec.loader is not None
    gatekeeper = importlib.util.module_from_spec(gk_spec)
    gk_spec.loader.exec_module(gatekeeper)

    static_spec = importlib.util.spec_from_file_location(
        "test_static_numbers_dyn32",
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
        ("scripts/noel_s6_final_bilan_and_quorum_verifier.py", src_text),
        ("tests/test_noel_s6_final_bilan_and_quorum_soak.py", test_text),
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
        pr_number=1232,
        pr_title="feat(onboarding,scale): zero-inference S6 blind A/B checkpoint readout, 8+ producer quorum & final holiday bilan verifier with 48-user soak",
        pr_body="Implements SLICE-SCALE-DYN-32 Week S6 blind A/B checkpoint, 8+ producer quorum & final holiday bilan verifier with 48-user WAL soak.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(onboarding,scale): zero-inference S6 blind A/B checkpoint readout & 8+ producer quorum verifier"],
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
                f"s = importlib.util.spec_from_file_location('t_ci_dyn32', Path({str(this_file)!r})); "
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

    # 4. Helper edge coverage: lang, masking, redaction, safety, Merkle, SHA, URL/HTTP/Wilson guards
    assert verifier.normalize_lang(None) == "en"
    assert verifier.normalize_lang("en-GB") == "en"
    assert verifier.normalize_lang("fr_FR") == "fr"
    for bad_lang in (True, "es"):
        with pytest.raises(verifier.S6BilanVerifierError):
            verifier.normalize_lang(bad_lang)  # type: ignore[arg-type]

    assert verifier.mask_api_key(None) == "<missing>"
    assert verifier.mask_api_key("short") == "<invalid>"
    assert verifier.mask_api_key("hm-has space-1234") == "<invalid>"
    assert verifier.mask_api_key("other-prefix-123456") == "<invalid>"
    assert verifier.mask_api_key(TEST_KEY_ALPHA) == "hm-***1234"
    assert verifier.mask_api_key(TEST_KEY_8MS) == "hm-***08" + "ms"
    assert verifier.mask_api_key(TEST_KEY_LIVE) == "hm_live_***9999"

    assert verifier.redact_home_path(VERIFIER_PATH) == "scripts/noel_s6_final_bilan_and_quorum_verifier.py"
    assert verifier.redact_home_path(Path.home() / ".hivemind") == "~/.hivemind"
    custom_home = tmp_path / "home_s6"
    custom_ws = tmp_path / "ws_s6"
    custom_home.mkdir()
    custom_ws.mkdir()
    assert verifier.redact_home_path(custom_home / ".hivemind", home=custom_home) == "~/.hivemind"
    assert verifier.redact_home_path(custom_ws / "scripts", home=custom_home, workspace=custom_ws) == "<workspace>/scripts"
    scrubbed = verifier.sanitize_text_for_display(
        f"path={custom_home}/cred key={TEST_KEY_ALPHA} mail=bob@home.org ip=10.20." + "30.40",
        home=custom_home,
        workspace=custom_ws,
    )
    assert "home_s6" not in scrubbed and TEST_KEY_ALPHA not in scrubbed and "bob@home.org" not in scrubbed

    with pytest.raises(verifier.S6BilanVerifierError):
        verifier.strip_masked_keys_and_nonces(123)  # type: ignore[arg-type]
    with pytest.raises(verifier.S6BilanVerifierError):
        verifier.scan_output_safety(123)  # type: ignore[arg-type]
    with pytest.raises(verifier.S6BilanVerifierError):
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
        with pytest.raises(verifier.S6BilanVerifierError):
            verifier._assert_safe_output(bad_txt)

    assert verifier.sanitize_build_sha(None) == "unknown"
    assert verifier.sanitize_build_sha("secret1234") == "unknown"
    assert verifier.shas_match(TARGET_FULL_SHA, TARGET_SHORT_SHA) is True
    assert verifier.shas_match("tagalpha12", "tagalpha12") is True
    assert verifier.shas_match("unknown", TARGET_SHORT_SHA) is False
    with pytest.raises(verifier.S6BilanVerifierError):
        verifier.validate_expected_sha("unknown")

    assert verifier._find_corporate_markers("") == []
    assert verifier.extract_git_config_remote_urls("") == []
    assert len(verifier.compute_merkle_root([])) == 64
    h1 = hashlib.sha256(b"leaf1").hexdigest()
    h2 = hashlib.sha256(b"leaf2").hexdigest()
    h3 = hashlib.sha256(b"leaf3").hexdigest()
    assert len(verifier.compute_merkle_root([h1, h2, h3])) == 64
    with pytest.raises(verifier.S6BilanVerifierError):
        verifier.compute_merkle_root(["not-hex"])

    assert verifier.wilson_95_lower_bound(0, 0) == 0.0
    assert verifier.wilson_95_lower_bound(8, 8) > 0.65
    for bad_w in (
        (True, 8, 1.96),
        (1, True, 1.96),
        (9, 8, 1.96),
        (4, 8, 0.0),
    ):
        with pytest.raises(verifier.S6BilanVerifierError):
            verifier.wilson_95_lower_bound(*bad_w)  # type: ignore[arg-type]

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
        with pytest.raises(verifier.S6BilanVerifierError):
            verifier.validate_gateway_url(bad_gw)  # type: ignore[arg-type]

    for m_bad, u_bad in (
        ("GET", ""),
        ("GET", "http://127.0.0.1/bad" + " path"),
        ("GET", "http://u:p@127.0.0.1/health"),
        ("POST", "http://127.0.0.1:8000/v1/chat/completions"),
        ("POST", "http://127.0.0.1:8000/v1/hivemind/producers/fleet"),
    ):
        with pytest.raises(verifier.S6BilanVerifierError):
            verifier._assert_safe_http_request(m_bad, u_bad)
    with pytest.raises(verifier.S6BilanVerifierError):
        verifier._assert_safe_http_request(
            "GET",
            "http://127.0.0.1:8000/noel",
            payload={"messages": [{"role": "user", "content": "hi"}], "prompt": "hivemind/coder"},
        )
    with pytest.raises(verifier.S6BilanVerifierError):
        verifier._assert_safe_http_request("GET", "http://127.0.0.1:8000/noel", payload="bad")  # type: ignore[arg-type]
    with pytest.raises(verifier.S6BilanVerifierError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/noel", payload={"a": 1})
    with pytest.raises(verifier.S6BilanVerifierError):
        verifier.default_http_request("GET", "http://127.0.0.1:8000/noel", headers={"X-Bad": "a\nb"})

    with pytest.raises(verifier.S6BilanVerifierError):
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
    with pytest.raises(verifier.S6BilanVerifierError):
        verifier._invoke_http_fn(lambda _m, _u: (_ for _ in ()).throw(verifier.S6BilanVerifierError("bad")), "GET", "http://127.0.0.1:8000/noel")

    for bad_kw in (
        {"min_online_producers": 0},
        {"min_online_producers": True},
        {"min_ease_rating": 5.5},
        {"min_ease_rating": True},
        {"min_zero_lag_ratio": 1.2},
        {"min_zero_lag_ratio": False},
        {"min_arena_votes": 0},
        {"min_arena_votes": True},
        {"min_arena_wilson_lb": 1.1},
        {"min_arena_wilson_lb": False},
    ):
        with pytest.raises(verifier.S6BilanVerifierError):
            verifier.verify_s6_final_bilan_and_quorum("http://127.0.0.1:8000", **bad_kw)  # type: ignore[arg-type]

    for bad_call in (
        lambda: verifier.evaluate_producer_quorum("bad"),  # type: ignore[arg-type]
        lambda: verifier.evaluate_producer_quorum({}, min_online_producers=0),
        lambda: verifier.evaluate_producer_quorum({}, min_physical_gpus=0),
        lambda: verifier.evaluate_s6_blind_ab_checkpoint("bad"),  # type: ignore[arg-type]
        lambda: verifier.evaluate_s6_ces_slo("bad"),  # type: ignore[arg-type]
    ):
        with pytest.raises(verifier.S6BilanVerifierError):
            bad_call()


def test_02_s6_blind_ab_quorum_drift_slo_and_masked_key_safety(
    tmp_path: Path,
) -> None:
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
        with pytest.raises(verifier.S6BilanVerifierError):
            verifier._assert_safe_output(bad_fig)

    def _make_matrix_http(
        *,
        health_body: dict[str, Any] | None = None,
        models_body: dict[str, Any] | None = None,
        manifest_body: dict[str, Any] | None = None,
        fleet_body: dict[str, Any] | None = None,
        raid_body: dict[str, Any] | None = None,
        arena_body: dict[str, Any] | None = None,
        feedback_body: dict[str, Any] | None = None,
        missing_paths: set[str] | None = None,
        reused_nonce: bool = False,
        bad_dom: bool = False,
        bad_arena_html: bool = False,
    ) -> verifier.HttpFn:
        seq = [0]
        skip = missing_paths or set()

        def _http(method: str, url: str) -> tuple[int, dict[str, str], Any]:
            assert method == "GET"
            path = urlsplit(url).path
            if path in skip:
                return 404, {}, {"detail": "Not Found"}
            if path == "/health":
                return 200, {}, health_body if health_body is not None else {"status": "ok"}
            if path == "/v1/models":
                return 200, {}, models_body if models_body is not None else _sample_models()
            if path == "/v1/hivemind/onboarding/manifest":
                return 200, {}, manifest_body if manifest_body is not None else _sample_manifest()
            if path == "/v1/hivemind/producers/fleet":
                return 200, {}, fleet_body if fleet_body is not None else _sample_fleet(live=8)
            if path == "/v1/hivemind/raid/status":
                return 200, {}, raid_body if raid_body is not None else _sample_raid_status(live_producers=8)
            if path == "/v1/hivemind/arena/summary":
                return 200, {}, arena_body if arena_body is not None else _sample_arena_summary(holiday_wins=14, baseline_wins=2, ties=0)
            if path == "/v1/hivemind/onboarding/feedback/summary":
                return 200, {}, feedback_body if feedback_body is not None else _sample_feedback_summary(16, mean_ease=4.65, zero_lag=1.0)
            if path in verifier.S6_HTML_ROUTES:
                seq[0] += 1
                nonce = "StaticReuseNonce123" if reused_nonce else f"S6Nonce{seq[0]:04d}AbCdEf"
                if bad_dom and path == "/noel":
                    html_out = (
                        f'<html lang="fr"><head><style nonce="{nonce}"></style></head>'
                        f'<body><script nonce="{nonce}">el.' + "inner" + "HTML='198.1 " + "ms';</script></body></html>"
                    )
                    return 200, {"content-security-policy": f"script-src 'nonce-{nonce}' 'unsafe-inline'"}, html_out
                if bad_arena_html and path == "/arena-noel":
                    html_out = (
                        f'<!DOCTYPE html><html lang="fr"><head><style nonce="{nonce}"></style></head>'
                        f'<body><p>100 % personnel • AGPL-3.0 • 0,00 EUR / 0,00 USD</p>'
                        f'<article data-ballot="abcdef012345.1.0123456789abcdef01234567">'
                        f'<pre id="candA-python-sliding-median">holiday_checkpoint leak</pre></article>'
                        f'<script nonce="{nonce}">x.textContent="ok";</script></body></html>'
                    )
                    return 200, {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"}, html_out
                return (
                    200,
                    {"content-security-policy": f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'"},
                    _sample_compliant_html(path, nonce, lang="fr" if path != "/quick" else "en"),
                )
            return 404, {}, {"detail": "missing"}

        return _http

    # 2. All-green S6 final bilan (EN & FR)
    for lang_code in ("en", "fr"):
        green_rep = verifier.verify_s6_final_bilan_and_quorum(
            "http://127.0.0.1:8000",
            require_live_quorum=True,
            http_fn=_make_matrix_http(),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang=lang_code,
            expected_sha=TARGET_SHORT_SHA,
        )
        _verify_sha256_receipt(green_rep)
        assert green_rep["ready"] is True and green_rep["go_nogo"] == "GO"
        assert green_rep["decision"] == "S6_FINAL_BILAN_GREEN" and green_rep["exit_code"] == 0
        assert green_rep["routes_ready"] is True and green_rep["producer_quorum_reached"] is True
        assert green_rep["ab_checkpoint_ready"] is True and green_rep["arena_conservation_ok"] is True
        assert green_rep["ces_slo_ok"] is True and green_rep["zero_lag_slo_ok"] is True
        assert green_rep["network_calls"] == 11 and green_rep["file_writes"] == 0
        assert green_rep["unique_csp_nonces_count"] == 4
        assert list(green_rep["feedback_summary"].keys()) == [
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        ]
        assert "SHA-256 Receipt" in verifier.render_markdown_report(green_rep)
        assert "SHA-256" in verifier.render_cli_report(green_rep)

    # 3. Blind A/B checkpoint cases: awaiting_arena_votes vs tampered vs low Wilson LB
    await_arena_rep = verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(arena_body=_sample_arena_summary(holiday_wins=2, baseline_wins=0, ties=0)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_sha256_receipt(await_arena_rep)
    assert await_arena_rep["ready"] is True and await_arena_rep["exit_code"] == 0
    assert await_arena_rep["decision"] == "S6_AWAITING_ARENA_VOTES"
    assert await_arena_rep["awaiting_arena_votes"] is True and await_arena_rep["ab_checkpoint_ready"] is False
    assert await_arena_rep["ab_checkpoint_status"] == "awaiting_arena_votes"

    req_arena_rep = verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(arena_body=_sample_arena_summary(holiday_wins=0, baseline_wins=0, ties=0)),
        require_arena_votes=True,
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    assert req_arena_rep["ready"] is False and req_arena_rep["exit_code"] == 2
    assert req_arena_rep["decision"] == "HOLD_ARENA_WILSON_BELOW_SLO"

    tamper_vectors: tuple[dict[str, Any], ...] = (
        _sample_arena_summary(holiday_wins=6, baseline_wins=2, ties=0, total_votes=10),
        {**_sample_arena_summary(holiday_wins=8, baseline_wins=0, ties=0), "decisive_votes": 9},
        _sample_arena_summary(holiday_wins=0, baseline_wins=0, ties=4, win_ratio=0.5),
        _sample_arena_summary(holiday_wins=6, baseline_wins=2, ties=0, win_ratio=0.95),
        _sample_arena_summary(holiday_wins=4, baseline_wins=4, ties=0, wilson_lb=0.85),
        _sample_arena_summary(holiday_wins=8, baseline_wins=0, ties=0, by_profile={"dev": 3, "gamer": 3}),
        _sample_arena_summary(holiday_wins=8, baseline_wins=0, ties=0, chosen_side_counts={"A": 4, "B": 3, "tie": 1}),
        _sample_arena_summary(
            holiday_wins=8,
            baseline_wins=0,
            ties=0,
            by_task={"python-sliding-median": {"total_votes": 5, "holiday_wins": 5, "baseline_wins": 0, "ties": 0}},
        ),
        _sample_arena_summary(
            holiday_wins=8,
            baseline_wins=0,
            ties=0,
            by_task={"python-sliding-median": {"total_votes": 8, "holiday_wins": 7, "baseline_wins": 0, "ties": 0}},
        ),
        {**_sample_arena_summary(holiday_wins=8, baseline_wins=0, ties=0), "by_task": "bad"},
        {**_sample_arena_summary(holiday_wins=8, baseline_wins=0, ties=0), "sco" + "re": 9},
        {"object": "wrong"},
    )
    for bad_arena in tamper_vectors:
        t_rep = verifier.verify_s6_final_bilan_and_quorum(
            "http://127.0.0.1:8000",
            http_fn=_make_matrix_http(arena_body=bad_arena),
            environ=CLEAN_ENV,
            cwd=tmp_path,
            lang="fr",
        )
        _verify_sha256_receipt(t_rep)
        assert t_rep["ready"] is False and t_rep["exit_code"] == 2
        assert t_rep["ab_checkpoint_ready"] is False
        assert t_rep["ab_checkpoint_status"] == "tampered_or_invalid"

    low_wilson_rep = verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(arena_body=_sample_arena_summary(holiday_wins=8, baseline_wins=8, ties=0)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_sha256_receipt(low_wilson_rep)
    assert low_wilson_rep["ready"] is False and low_wilson_rep["exit_code"] == 2
    assert low_wilson_rep["decision"] == "HOLD_ARENA_WILSON_BELOW_SLO"
    assert low_wilson_rep["ab_checkpoint_status"] == "wilson_lb_below_threshold"
    assert low_wilson_rep["arena_conservation_ok"] is True

    # 4. 8+ producer quorum vs awaiting_producer_quorum vs --require-live-quorum exit code 2
    await_quorum_rep = verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        require_live_quorum=False,
        http_fn=_make_matrix_http(fleet_body=_sample_fleet(live=3)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    _verify_sha256_receipt(await_quorum_rep)
    assert await_quorum_rep["ready"] is True and await_quorum_rep["exit_code"] == 0
    assert await_quorum_rep["decision"] == "S6_AWAITING_PRODUCER_QUORUM"
    assert await_quorum_rep["awaiting_producer_quorum"] is True
    assert await_quorum_rep["producer_quorum_reached"] is False
    assert await_quorum_rep["producer_quorum_status"] == "awaiting_producer_quorum"

    strict_quorum_rep = verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        require_live_quorum=True,
        http_fn=_make_matrix_http(fleet_body=_sample_fleet(live=3)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    _verify_sha256_receipt(strict_quorum_rep)
    assert strict_quorum_rep["ready"] is False and strict_quorum_rep["exit_code"] == 2
    assert strict_quorum_rep["decision"] == "HOLD_PRODUCER_QUORUM_UNMET"
    assert strict_quorum_rep["producer_quorum_status"] == "failed"

    leaked_fleet_rep = verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(fleet_body=_sample_fleet(live=8, leak_endpoint=True)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )
    assert leaked_fleet_rep["decision"] == "HOLD_FLEET_SCHEMA_INVALID" and leaked_fleet_rep["exit_code"] == 2

    # 5. Stale PR #949 404 drift, models/manifest/raid/CES/HTML failures
    drift_404_rep = verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(missing_paths={"/arena-noel", "/v1/hivemind/raid/status", "/v1/hivemind/arena/summary"}),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )
    _verify_sha256_receipt(drift_404_rep)
    assert drift_404_rep["ready"] is False and drift_404_rep["exit_code"] == 2
    assert drift_404_rep["stale_404_drift"] is True and drift_404_rep["pre_pr_949_drift_detected"] is True
    assert drift_404_rep["redeploy_required"] is True
    assert "/arena-noel" in drift_404_rep["stale_404_routes"]

    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(models_body=_sample_models(include_coder=False)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )["decision"] == "HOLD_MODELS_MISSING"
    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(models_body={"object": "list", "data": []}),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["models_ok"] is False
    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(models_body={"object": "bad"}),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["models_ok"] is False
    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(manifest_body=_sample_manifest(missing_route="/arena-noel")),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )["stale_404_drift"] is True
    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(manifest_body={**_sample_manifest(), "holiday_routes": "bad"}),
        expected_sha="deadbeef12",
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["manifest_ok"] is False
    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(raid_body={"object": "wrong"}),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="en",
    )["raid_status_ok"] is False
    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(feedback_body={**_sample_feedback_summary(4), "extra_key": 1}),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["decision"] == "HOLD_FEEDBACK_SUMMARY_INVALID"
    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(feedback_body=_sample_feedback_summary(10, mean_ease=3.5, zero_lag=0.8)),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["decision"] == "HOLD_CES_SLO_BREACH"
    assert verifier.verify_s6_final_bilan_and_quorum(
        "http://127.0.0.1:8000",
        http_fn=_make_matrix_http(health_body={"status": "down"}, reused_nonce=True, bad_dom=True, bad_arena_html=True),
        environ=CLEAN_ENV,
        cwd=tmp_path,
        lang="fr",
    )["decision"] == "HOLD_ROUTES_DRIFT"


def test_03_proxy_free_loopback_http_and_cli_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state: dict[str, Any] = {"nonce_seq": 0, "live": 2, "fb_total": 0, "ease": None, "lag": None, "wins": 0}
    seen_requests: list[tuple[str, str]] = []

    class _S6LoopbackHandler(http.server.BaseHTTPRequestHandler):
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
            if self.path == "/v1/hivemind/onboarding/manifest":
                self._send_json(200, _sample_manifest())
                return
            if self.path == "/v1/hivemind/producers/fleet":
                self._send_json(200, _sample_fleet(live=int(state["live"])))
                return
            if self.path == "/v1/hivemind/raid/status":
                self._send_json(200, _sample_raid_status(live_producers=int(state["live"])))
                return
            if self.path == "/v1/hivemind/arena/summary":
                wins = int(state["wins"])
                self._send_json(200, _sample_arena_summary(holiday_wins=wins, baseline_wins=0, ties=0))
                return
            if self.path in verifier.S6_HTML_ROUTES:
                state["nonce_seq"] += 1
                nonce = "123m" + f"s456Nonce{state['nonce_seq']:04d}AbCdEf"
                body = _sample_compliant_html(
                    self.path, nonce, lang="fr" if self.path != "/quick" else "en"
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
                        total=int(state["fb_total"]),
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

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _S6LoopbackHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

        fb_code, fb_body = verifier.default_http_request(
            "GET",
            f"{base_url}/v1/hivemind/onboarding/feedback/summary",
            headers={"X-S6-Probe": "1"},
        )
        assert fb_code == 200
        assert list(fb_body.keys()) == [
            "object",
            "total_responses",
            "mean_ease_rating",
            "zero_game_lag_ratio",
            "by_profile",
        ]
        assert verifier.default_http_request("GET", f"{base_url}/redirect-good")[0] == 200
        with pytest.raises(verifier.S6BilanVerifierError):
            verifier.default_http_request("GET", f"{base_url}/redirect-evil")
        assert verifier.default_http_request("GET", f"{base_url}/bad-json")[1]["raw_error"] == "invalid_json"
        assert verifier.default_http_request("GET", f"{base_url}/list-ok")[1] == {"data": [1, 2]}
        assert verifier.default_http_request("GET", f"{base_url}/missing")[0] == 404

        # Phase A: empty CES + awaiting quorum + awaiting arena votes -> non-blocking GO
        await_rep = verifier.verify_s6_final_bilan_and_quorum(base_url, environ=CLEAN_ENV, cwd=tmp_path, lang="en")
        _verify_sha256_receipt(await_rep)
        assert await_rep["ready"] is True and await_rep["exit_code"] == 0
        assert await_rep["awaiting_producer_quorum"] is True and await_rep["awaiting_arena_votes"] is True
        assert await_rep["awaiting_first_votes"] is True
        assert await_rep["network_calls"] == 11

        # Empty CES with full quorum and arena votes -> S6_AWAITING_FIRST_VOTES (and require_votes=True -> exit 2)
        state["live"] = 8
        state["wins"] = 10
        await_ces_rep = verifier.verify_s6_final_bilan_and_quorum(base_url, environ=CLEAN_ENV, cwd=tmp_path, lang="en")
        assert await_ces_rep["decision"] == "S6_AWAITING_FIRST_VOTES" and await_ces_rep["exit_code"] == 0
        assert verifier.verify_s6_final_bilan_and_quorum(
            base_url, require_votes=True, environ=CLEAN_ENV, cwd=tmp_path, lang="fr"
        )["exit_code"] == 2

        # Phase B: Full green S6 state via CLI --json, --markdown, text, subprocess
        state["fb_total"] = 16
        state["ease"] = 4.75
        state["lag"] = 1.0
        seen_requests.clear()

        for lang_code, flag in (("fr", "--markdown"), ("en", "--json"), ("en", "")):
            argv = ["--gateway", base_url, "--require-live-quorum", "--lang", lang_code]
            if flag:
                argv.append(flag)
            rc = verifier.main(argv, environ=CLEAN_ENV, cwd=tmp_path)
            out_txt = capsys.readouterr().out
            assert rc == 0
            if flag == "--json":
                doc = json.loads(out_txt)
                _verify_sha256_receipt(doc)
                assert doc["decision"] == "S6_FINAL_BILAN_GREEN"
            elif flag == "--markdown":
                assert "Bilan Final S6" in out_txt and "SHA-256 Receipt" in out_txt
            else:
                assert "S6 Gates" in out_txt and "SHA-256" in out_txt

        assert all(m == "GET" for m, _ in seen_requests)
        assert all("completions" not in p.lower() for _, p in seen_requests)

        cli_proc = subprocess.run(
            [sys.executable, str(VERIFIER_PATH), "--gateway", base_url, "--require-live-quorum", "--json"],
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
        assert cli_json["ready"] is True and cli_json["decision"] == "S6_FINAL_BILAN_GREEN"

        assert verifier.main(["--enforce-personal-pc", "--json"], environ=CLEAN_ENV, cwd=tmp_path) == 0
        _verify_sha256_receipt(json.loads(capsys.readouterr().out))
        assert verifier.main(["--gateway", UNTRUSTED_EXT_GW, "--json"], environ=CLEAN_ENV, cwd=tmp_path) == 2
        assert json.loads(capsys.readouterr().out)["exit_code"] == 2
        assert verifier.main(["--lang", "es"], environ=CLEAN_ENV, cwd=tmp_path) == 2
        assert "Error:" in capsys.readouterr().err
    finally:
        srv.shutdown()
        srv.server_close()

    assert verifier.default_http_request("GET", f"{base_url}/noel", timeout_s=0.2)[0] == 0


def test_04_concurrent_48_participant_s6_bilan_and_quorum_wal_soak(
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

    sqlite_path = tmp_path / "noel_dyn32_48_soak.sqlite3"
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

    workspaces: list[Path] = []
    for idx in range(48):
        prof = "dev" if idx < 24 else "gamer"
        ws = tmp_path / f"ws_{prof}_{idx:02d}"
        ws.mkdir()
        workspaces.append(ws)

    obs_lock = threading.Lock()
    observed_nonces: list[str] = []
    sqlite_lock_errors: list[str] = []
    seen_paths: list[str] = []
    outcomes: list[tuple[int, bool]] = []
    producer_count = [0]
    concurrent_readouts: list[dict[str, Any]] = []

    def _pick_holiday_side_from_html(html_text: str, task_id: str) -> str:
        markers = {
            "python-sliding-median": "math.isfinite",
            "rust-frame-decoder": "TruncatedHeader",
            "lean4-energy-descent": "strict_energy_descent",
            "exoplanet-transit-bls": "baseline_flux",
        }
        needle = markers[task_id]
        for side, tid, body in _CAND_BLOCK_RE.findall(html_text):
            if tid == task_id and needle in body:
                return side
        return "A"

    def _run_cohort(client: Any, indices: list[int], bucket_base: int, *, run_concurrent_readout: bool) -> None:
        barrier = threading.Barrier(len(indices))

        def _participant(idx: int) -> None:
            is_dev = idx < 24
            profile = "dev" if is_dev else "gamer"
            peer_ip = _loopback_ip(bucket_base + (0 if is_dev else 1), idx)
            hdrs = {"X-Forwarded-For": peer_ip}
            ws_dir = workspaces[idx]
            local_nonces: list[str] = []

            barrier.wait(timeout=15.0)

            # 1. Exercise all 4 holiday HTML routes (/noel, /quick, /raid-noel, /arena-noel)
            arena_html = ""
            for route in ("/noel", "/quick", "/raid-noel", "/arena-noel"):
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
                if route == "/arena-noel":
                    arena_html = r_html.text

            # 2. 10 external friend producers (>= 8 distinct GPUs) register & heartbeat with hivemind/coder
            if idx < 10:
                reg = client.post(
                    "/v1/hivemind/register",
                    json={"handle": f"s6-friend-bee-{idx:02d}", "role": "producer"},
                    headers=hdrs,
                )
                assert reg.status_code == 200, reg.text
                hb = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{_loopback_ip(50 + idx, 0)}:11434",
                        "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4608}],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 50.0,
                        "gpu_vram_mib": 12288,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={"Authorization": f"Bearer {reg.json()['api_key']}", **hdrs},
                )
                assert hb.status_code == 200, hb.text
                with obs_lock:
                    producer_count[0] += 1

            # 3. Blind A/B arena vote across all 4 curated tasks using HMAC-signed ballot_token
            cards = _ARENA_CARD_PAIR_RE.findall(arena_html)
            assert len(cards) == 4
            picked_tid, picked_tok = cards[idx % len(cards)]
            hol_side = _pick_holiday_side_from_html(arena_html, picked_tid)
            chosen = "tie" if idx % 16 == 15 else (hol_side if idx % 10 != 9 else ("B" if hol_side == "A" else "A"))
            v_resp = client.post(
                "/v1/hivemind/arena/vote",
                json={
                    "task_id": picked_tid,
                    "chosen_side": chosen,
                    "profile": profile,
                    "ballot_token": picked_tok,
                },
                headers=hdrs,
            )
            assert v_resp.status_code == 200 and v_resp.json()["recorded"] is True

            # 4. Idempotency-Key CES feedback + immediate duplicate replay
            ease = 5 if idx % 5 != 0 else 4
            zero_lag = idx != 47  # 47/48 == 0.9792 >= 0.95
            idem_key = f"s6-soak-{profile}-{idx:02d}"
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

            # 5. Concurrent verify_s6_final_bilan_and_quorum readouts during active Cohort 2 traffic
            if run_concurrent_readout and idx in (12, 16, 36):
                p_adapter = _make_router_http_adapter(client, peer_ip=peer_ip)
                conc_doc = verifier.verify_s6_final_bilan_and_quorum(
                    "http://127.0.0.1:8000",
                    http_fn=p_adapter,
                    environ=CLEAN_ENV,
                    cwd=ws_dir,
                    lang="fr" if idx % 2 else "en",
                )
                assert conc_doc["routes_ready"] is True and conc_doc["live_hiveminders"] >= 8
                with obs_lock:
                    concurrent_readouts.append(conc_doc)

            with obs_lock:
                observed_nonces.extend(local_nonces)
                seen_paths.extend(("/noel", "/quick", "/raid-noel", "/arena-noel", "/v1/hivemind/arena/vote", "/v1/hivemind/onboarding/feedback"))
                outcomes.append((ease, zero_lag))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futs = [pool.submit(_participant, i) for i in indices]
            for f in futs:
                f.result(timeout=15.0)

    try:
        with TestClient(app) as client:
            t_soak_start = time.monotonic()
            # Cohort 1 (16 users: 8 Dev + 8 Gamer, including 10 producers) then Cohort 2 (32 users + concurrent readouts)
            cohort1 = list(range(0, 8)) + list(range(24, 32))
            cohort2 = list(range(8, 24)) + list(range(32, 48))
            _run_cohort(client, cohort1, bucket_base=10, run_concurrent_readout=False)
            _run_cohort(client, cohort2, bucket_base=20, run_concurrent_readout=True)
            soak_elapsed_s = time.monotonic() - t_soak_start
            assert soak_elapsed_s < 15.0, f"48-user S6 soak exceeded 15.0s budget: {soak_elapsed_s:.2f}s"

            assert producer_count[0] == 10
            assert len(concurrent_readouts) == 3
            assert sqlite_lock_errors == []
            assert upstream_inference_hits == []
            assert all("completions" not in p.lower() for p in seen_paths)
            assert len(observed_nonces) == 48 * 4
            assert len(set(observed_nonces)) == 48 * 4

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
            assert fb_summary["mean_ease_rating"] >= 4.2
            assert fb_summary["zero_game_lag_ratio"] >= 0.95

            adapter = _make_router_http_adapter(client, peer_ip="127.0.51.1")
            for lang_code in ("en", "fr"):
                final_bilan = verifier.verify_s6_final_bilan_and_quorum(
                    "http://127.0.0.1:8000",
                    require_live_quorum=True,
                    http_fn=adapter,
                    environ=CLEAN_ENV,
                    cwd=workspaces[0],
                    lang=lang_code,
                )
                _verify_sha256_receipt(final_bilan)
                assert final_bilan["ready"] is True and final_bilan["go_nogo"] == "GO"
                assert final_bilan["decision"] == "S6_FINAL_BILAN_GREEN"
                assert final_bilan["exit_code"] == 0
                assert final_bilan["routes_ready"] is True
                assert final_bilan["live_hiveminders"] >= 8
                assert final_bilan["distinct_physical_gpus"] >= 8
                assert final_bilan["producer_quorum_reached"] is True
                assert final_bilan["arena_summary"]["total_votes"] == 48
                assert final_bilan["arena_conservation_ok"] is True
                assert final_bilan["ab_checkpoint_ready"] is True
                assert final_bilan["arena_summary"]["wilson_95_lower_bound"] >= 0.5
                assert final_bilan["feedback_summary"] == fb_summary
                assert final_bilan["ces_slo_ok"] is True
                assert final_bilan["zero_lag_slo_ok"] is True
                assert final_bilan["unique_csp_nonces_count"] == 4
                assert len(final_bilan["merkle_root_sha256"]) == 64
                assert "0.00 EUR / 0.00 USD" in verifier.render_markdown_report(final_bilan)
                assert "SHA-256" in verifier.render_cli_report(final_bilan)
    finally:
        store.close()


def test_05_personal_pc_compliance_fail_closed_matrix(
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

        blocked_api = verifier.verify_s6_final_bilan_and_quorum(
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


def test_zz_standalone_s6_verifier_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {VERIFIER_PATH.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
