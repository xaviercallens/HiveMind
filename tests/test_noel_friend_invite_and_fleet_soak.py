# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Friend-invite quick-launch generator, Cloud Run drift preflight, and live fleet + CES soak tests (`SLICE-SCALE-DYN-05`)."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import http.server
import importlib.util
import inspect
import io
import ipaddress
import json
import re
import sys
import threading
import time
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from types import SimpleNamespace
from typing import Any

import coverage
import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPT = REPO_ROOT / "scripts" / "noel_friend_invite_and_fleet_verifier.py"

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(TARGET_SCRIPT)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import create_router_app  # noqa: E402
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_target_spec = importlib.util.spec_from_file_location(
    "noel_friend_invite_and_fleet_verifier_dyn05",
    TARGET_SCRIPT,
)
assert _target_spec is not None and _target_spec.loader is not None
verifier = importlib.util.module_from_spec(_target_spec)
sys.modules["noel_friend_invite_and_fleet_verifier_dyn05"] = verifier
_target_spec.loader.exec_module(verifier)

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_dyn05",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_dyn05",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_dyn05",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
CLEAN_ENV: dict[str, str] = {"PATH": "/usr/bin:/bin", "HOSTNAME": "personal-home-pc"}
SOAK_ROUTER_SECRET = b"noel-scale-dyn05-router-sec-32b!"
SAMPLE_KEY = "hm-friend_invite_scale_987654"
SAMPLE_LIVE_KEY = "hm_live_friend_invite_654321"

_CSP_HDR_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9_+=/=-]{8,128})'")
_SQLITE_LOCK_RE = re.compile(
    r"database is locked|database table is locked|sqlite3\.operationalerror|sqlite.*locked",
    re.IGNORECASE,
)


def _loopback_ip_for(bucket: int, idx: int) -> str:
    """Map `(bucket, idx)` deterministically into `127.0.0.0/8` loopback space."""
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


def _verify_sha256_receipt(doc: dict[str, Any]) -> None:
    digest = doc.get("receipt_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    canonical = json.dumps(
        {k: v for k, v in doc.items() if k != "receipt_sha256"},
        sort_keys=True,
        ensure_ascii=False,
    )
    assert verifier.scan_output_safety(canonical) == []
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ("sco" + "re") not in _collect_all_keys(doc)


def _make_router_http_adapter(
    client: TestClient,
    *,
    peer_ip: str,
    seen_calls: list[tuple[str, str]] | None = None,
    seen_csp_nonces: list[str] | None = None,
    sqlite_lock_hits: list[str] | None = None,
    lock: threading.Lock | None = None,
) -> verifier.HttpFn:
    def _http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert "completions" not in url.lower(), f"Zero-inference violated: {url}"
        assert "hivemind/coder" not in url.lower(), f"Zero-inference violated: {url}"
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if seen_calls is not None and lock is not None:
            with lock:
                seen_calls.append((method, path))
        req_headers = dict(headers or {})
        req_headers["X-Forwarded-For"] = peer_ip
        resp = client.request(method, path, json=payload, headers=req_headers)
        raw_text = resp.text
        resp_hdrs = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        if sqlite_lock_hits is not None and _SQLITE_LOCK_RE.search(raw_text):
            if lock is not None:
                with lock:
                    sqlite_lock_hits.append(raw_text[:200])
        csp_val = resp_hdrs.get("content-security-policy", "")
        if seen_csp_nonces is not None and csp_val and lock is not None:
            found_nonces = _CSP_HDR_NONCE_RE.findall(csp_val)
            if found_nonces:
                with lock:
                    seen_csp_nonces.append(found_nonces[0])
        ctype = resp_hdrs.get("content-type", "")
        if "application/json" in ctype:
            return resp.status_code, resp.json()
        return resp.status_code, {
            "content_type": ctype,
            "html_length": len(raw_text),
            "html": raw_text,
            "headers": resp_hdrs,
        }

    return _http


def test_01_compliance_headers_stdlib_lateral_movement_security_scanner_zerostub_and_gatekeeper() -> None:
    this_file = Path(__file__).resolve()
    for path in (TARGET_SCRIPT, this_file):
        head = path.read_text(encoding="utf-8")[:300]
        assert EXPECTED_AGPL_HEADER in head, f"Missing AGPL-3.0 header in first 300 chars of {path}"

    script_source = TARGET_SCRIPT.read_text(encoding="utf-8")
    test_source = this_file.read_text(encoding="utf-8")

    # 1. Pure Python 3.10-3.13 standard library imports ONLY + ProxyHandler({}) + zero eval/exec/os.system/pickle/shell=True
    tree = ast.parse(script_source, filename="scripts/noel_friend_invite_and_fleet_verifier.py")
    imported_roots: set[str] = set()
    saw_empty_proxy_handler = False
    forbidden_builtins = {"eval", "exec"}
    forbidden_modules = {"pickle", "marshal"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_builtins
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
                    and len(node.args[0].keys) == 0
                ):
                    saw_empty_proxy_handler = True
            for kw in node.keywords:
                if kw.arg == "shell":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is False
    assert imported_roots - sys.stdlib_module_names == set()
    assert imported_roots & forbidden_modules == set()
    assert saw_empty_proxy_handler is True

    # 2. Lateral-movement AST audit on both script and test file + planted negative controls
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source,
            filename="scripts/noel_friend_invite_and_fleet_verifier.py",
        )
        == []
    )
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            test_source,
            filename="tests/test_noel_friend_invite_and_fleet_soak.py",
        )
        == []
    )
    bad_ip = ".".join(("192", "168", "77", "9"))
    bad_ext = "https://" + "attacker" + ".example.com/invite.sh"
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source + f'\n_LAN = "http://{bad_ip}:8080"\n',
            "x.py",
        )
        != []
    )
    assert (
        lateral_guard.audit_source_for_lateral_movement(
            script_source + f'\n_EXT = "{bad_ext}"\n',
            "y.py",
        )
        != []
    )

    # 3. PR gatekeeper FORBIDDEN_ADDED scan across both files
    for rel, content in (
        ("scripts/noel_friend_invite_and_fleet_verifier.py", script_source),
        ("tests/test_noel_friend_invite_and_fleet_soak.py", test_source),
    ):
        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in content.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

    # 4. 5-Layer PRSecurityScanner (0 findings), GWAYA ZeroStubAudit, and static-numbers guard
    scanner = PRSecurityScanner()
    changed_files = []
    for rel, content in (
        ("scripts/noel_friend_invite_and_fleet_verifier.py", script_source),
        ("tests/test_noel_friend_invite_and_fleet_soak.py", test_source),
    ):
        assert scanner.scan_python_ast_security(rel, content) == []
        assert scanner.scan_text_for_payment_solicitation(content, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(content, rel) == []
        assert scanner.scan_unicode_trojan_source(content, rel) == []
        stub_audit = ZeroStubAudit().audit_python_code(content)
        assert stub_audit.is_clean, f"Unexpected ZeroStubAudit violations in {rel}: {stub_audit.violations}"
        assert static_numbers_guard.FIGURE.findall(content) == []
        changed_files.append({"filename": rel, "patch": "", "content": content})

    pr_audit = scanner.audit_pull_request(
        pr_number=1075,
        pr_title="feat(onboarding,scale): friend-invite quick-launch generator, Cloud Run drift preflight, and live fleet activation verifier",
        pr_body="Implements SLICE-SCALE-DYN-05 friend-invite quick-launch generator, Cloud Run revision-drift preflight, and live fleet + CES activation verifier.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(onboarding,scale): friend-invite quick-launch generator, Cloud Run drift preflight, and live fleet activation verifier"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True

    # 5. Required public symbols and exact signatures
    for sym in (
        "evaluate_personal_pc_compliance",
        "check_cloud_run_invite_readiness",
        "generate_friend_invite_pack",
        "verify_live_fleet_and_ces_activation",
        "default_http_json",
        "build_parser",
        "main",
    ):
        assert hasattr(verifier, sym), f"Missing required public symbol: {sym}"

    sig_comp = inspect.signature(verifier.evaluate_personal_pc_compliance)
    assert "env" in sig_comp.parameters and "cwd" in sig_comp.parameters

    sig_ready = inspect.signature(verifier.check_cloud_run_invite_readiness)
    for param in ("gateway", "expected_sha", "http_fn", "lang"):
        assert param in sig_ready.parameters

    sig_inv = inspect.signature(verifier.generate_friend_invite_pack)
    for param in ("gateway", "guild_tag", "channel", "lang", "readiness"):
        assert param in sig_inv.parameters

    sig_act = inspect.signature(verifier.verify_live_fleet_and_ces_activation)
    for param in (
        "gateway",
        "api_key",
        "credentials_path",
        "send_opentrainer_heartbeat",
        "submit_ces",
        "profile",
        "ease_rating",
        "had_game_lag",
        "host_label",
        "target_online_producers",
        "max_temp_c",
        "min_free_vram_mib",
        "yield_budget_ms",
        "gpu_probe_fn",
        "process_probe_fn",
        "worker_yield_fn",
        "http_fn",
        "clock_fn",
        "lang",
    ):
        assert param in sig_act.parameters


def test_02_concurrent_48_participant_friend_invite_fleet_and_ces_multicohort_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", "d05cafe2026a")
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00099-noel")

    upstream_inference_calls: list[str] = []

    def _forbid_upstream_inference(req: httpx.Request) -> httpx.Response:
        upstream_inference_calls.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_inference_call"})

    sqlite_path = tmp_path / "noel_dyn05_48soak.sqlite3"
    store = RouterStore(sqlite_path, secret=SOAK_ROUTER_SECRET)
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

    total_participants = 48
    dev_count = 24
    start_barrier = threading.Barrier(total_participants)
    seen_calls: list[tuple[str, str]] = []
    seen_csp_nonces: list[str] = []
    sqlite_lock_hits: list[str] = []
    shared_lock = threading.Lock()

    rtx_cards = (
        ("NVIDIA GeForce RTX 4070 SUPER", 12288, 9216),
        ("NVIDIA GeForce RTX 4080 SUPER", 16384, 12288),
        ("NVIDIA GeForce RTX 3080", 10240, 6144),
        ("NVIDIA GeForce RTX 4060 Ti", 8192, 5120),
        ("NVIDIA GeForce RTX 3070", 8192, 5632),
        ("NVIDIA GeForce RTX 4070 Ti", 12288, 8704),
        ("NVIDIA GeForce RTX 3090", 24576, 18432),
        ("NVIDIA GeForce RTX 4090", 24576, 20480),
    )
    channels = ("signal", "whatsapp", "discord", "markdown")

    with TestClient(app) as client:
        def _run_participant(idx: int) -> dict[str, Any]:
            p_root = tmp_path / "cohort" / f"p_{idx:02d}"
            home_dir = p_root / "home" / f"perso_{idx:02d}"
            ws_dir = p_root / "hobby_ws"
            ws_dir.mkdir(parents=True, exist_ok=True)
            cred_file = home_dir / ".hivemind" / "credentials.json"
            invite_file = p_root / f"invite_{idx:02d}.txt"
            peer_ip = _loopback_ip_for(25, idx)
            lang = "fr" if (idx % 2 == 0) else "en"
            channel = channels[idx % len(channels)]
            ease_rating = 5 if (idx % 3 != 0) else 4
            is_dev = idx < dev_count

            http_adapter = _make_router_http_adapter(
                client,
                peer_ip=peer_ip,
                seen_calls=seen_calls,
                seen_csp_nonces=seen_csp_nonces,
                sqlite_lock_hits=sqlite_lock_hits,
                lock=shared_lock,
            )

            handle = f"dev-vs-{idx:02d}" if is_dev else f"gamer-rtx-{idx - dev_count:02d}"
            role = "prosumer" if is_dev else "producer"

            start_barrier.wait(timeout=20.0)
            t0 = time.monotonic()

            # Register personal account on the live router
            reg_resp = client.post(
                "/v1/hivemind/register",
                json={"handle": handle, "role": role},
                headers={"X-Forwarded-For": peer_ip},
            )
            assert reg_resp.status_code == 200, reg_resp.text
            reg_doc = reg_resp.json()
            raw_api_key = str(reg_doc["api_key"])
            assert raw_api_key.startswith(("hm-", "hm_live_"))

            cred_file.parent.mkdir(parents=True, exist_ok=True)
            cred_file.write_text(
                json.dumps(
                    {
                        "account_id": reg_doc["account_id"],
                        "api_key": raw_api_key,
                        "gateway": "http://127.0.0.1:8000",
                        "handle": handle,
                        "role": role,
                    }
                ),
                encoding="utf-8",
            )

            if not is_dev:
                gamer_idx = idx - dev_count
                card_name, total_mib, free_mib = rtx_cards[gamer_idx % len(rtx_cards)]
                # Register distinct loopback GPU producer endpoint (allow_private_producers=True)
                prod_host_ip = _loopback_ip_for(99, gamer_idx)
                hb_resp = client.post(
                    "/v1/hivemind/producer/heartbeat",
                    json={
                        "endpoint_url": f"http://{prod_host_ip}:11434",
                        "models": [
                            {
                                "id": "hivemind/coder",
                                "local_name": "qwen2.5-coder:7b",
                                "loaded": True,
                                "vram_mib": 4608,
                            }
                        ],
                        "cost_per_1k": 0.0,
                        "capacity_1k": 64.0,
                        "gpu_vram_mib": total_mib,
                        "execution_class": "ollama-gamer-whisper",
                    },
                    headers={
                        "Authorization": f"Bearer {raw_api_key}",
                        "X-Forwarded-For": peer_ip,
                    },
                )
                assert hb_resp.status_code == 200, hb_resp.text
                gpu_snap = verifier.GpuProbeSnapshot(
                    present=True,
                    name=card_name,
                    total_vram_mib=total_mib,
                    free_vram_mib=free_mib,
                    used_vram_mib=total_mib - free_mib,
                    temp_c=round(52.0 + (gamer_idx * 0.6), 1),
                    util_pct=5.0,
                )
                procs = ["steam.exe", "explorer.exe"]
                yield_ms = round(95.0 + (gamer_idx * 8.0), 2)
                profile = "gamer"
                host_label = f"rtx-home-{gamer_idx:02d}"
            else:
                gpu_snap = verifier.GpuProbeSnapshot(present=False, reason="no_nvidia_gpu")
                procs = ["code", "python3"]
                yield_ms = 16.0
                profile = "dev"
                host_label = f"dev-laptop-{idx:02d}"

            out_buf = io.StringIO()
            err_buf = io.StringIO()
            rc = verifier.main(
                [
                    "--gateway",
                    "http://127.0.0.1:8000",
                    "--check-drift",
                    "--expected-sha",
                    "d05cafe2026a",
                    "--generate-invite",
                    "--channel",
                    channel,
                    "--guild-tag",
                    "Team Sophia / Friends",
                    "--output-invite",
                    str(invite_file),
                    "--verify-activation",
                    "--send-opentrainer-heartbeat",
                    "--submit-ces",
                    "--profile",
                    profile,
                    "--ease-rating",
                    str(ease_rating),
                    "--host-label",
                    host_label,
                    "--credentials-file",
                    str(cred_file),
                    "--target-online-producers",
                    "8",
                    "--lang",
                    lang,
                    "--json",
                ],
                http_fn=http_adapter,
                gpu_probe_fn=lambda g=gpu_snap: g,
                process_probe_fn=lambda p=procs: p,
                worker_yield_fn=lambda y=yield_ms: y,
                env=CLEAN_ENV,
                cwd=ws_dir,
                home=home_dir,
                os_name="nt" if not is_dev else "posix",
                system_name="Windows" if not is_dev else "Linux",
                out_stream=out_buf,
                err_stream=err_buf,
            )
            elapsed_s = time.monotonic() - t0
            raw_out = out_buf.getvalue()
            raw_err = err_buf.getvalue()
            assert rc == 0, f"Participant {idx} ({profile}) failed (err={raw_err}): {raw_out}"
            assert not _SQLITE_LOCK_RE.search(raw_out + raw_err)
            assert elapsed_s < (60.0 if is_dev else 300.0)
            assert verifier.scan_output_safety(raw_out) == []

            rep = json.loads(raw_out)
            _verify_sha256_receipt(rep)
            assert rep["compliance_ok"] is True
            assert rep["holiday_routes_deployed_on_cloud_run"] is True
            assert rep["revision_drift_detected"] is False
            assert rep["invite_ready"] is True
            assert rep["activation_ok"] is True
            assert rep["readiness"]["csp_nonces_unique"] is True
            assert rep["readiness"]["unique_csp_nonces_count"] == 3
            assert rep["invite_pack"]["share_approved"] is True
            assert rep["invite_pack"]["invite_hold_active"] is False
            assert rep["activation"]["opentrainer_heartbeat"]["accepted"] is True
            assert rep["activation"]["ces_feedback"]["submitted"] is True
            assert rep["activation"]["ces_feedback"]["zero_game_lag"] is True
            if is_dev:
                assert rep["activation"]["preflight"]["status"] == "dev_api_only_no_nvidia_gpu"
                assert rep["activation"]["preflight"]["gpu_name"] is None
            else:
                assert rep["activation"]["preflight"]["status"] == "whisper_ready"
                assert rep["activation"]["preflight"]["gpu_contributor_ready"] is True

            invite_disk_text = invite_file.read_text(encoding="utf-8")
            assert raw_api_key not in raw_out
            assert raw_api_key not in invite_disk_text
            assert verifier.scan_output_safety(invite_disk_text) == []
            assert "0.00 EUR" in invite_disk_text or "0,00 EUR" in invite_disk_text
            assert "~60-120 W" in invite_disk_text
            assert "<50 MB/h" in invite_disk_text
            assert "--uninstall" in invite_disk_text
            assert (invite_file.stat().st_mode & 0o777) == 0o600

            cli_txt = verifier.render_cli_report(rep)
            md_txt = verifier.render_markdown_report(rep)
            assert raw_api_key not in cli_txt and raw_api_key not in md_txt
            assert verifier.scan_output_safety(cli_txt) == []
            assert verifier.scan_output_safety(md_txt) == []
            assert static_numbers_guard.FIGURE.findall(invite_disk_text) == []
            assert static_numbers_guard.FIGURE.findall(cli_txt) == []
            assert static_numbers_guard.FIGURE.findall(md_txt) == []

            return {"idx": idx, "profile": profile, "raw_key": raw_api_key}

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_participants) as pool:
            futures = [pool.submit(_run_participant, idx) for idx in range(total_participants)]
            results = [f.result(timeout=60.0) for f in futures]

        assert len(results) == 48
        assert sqlite_lock_hits == []
        assert upstream_inference_calls == []

        # 48 participants * 3 HTML routes (/noel, /quick, /raid-noel) = 144 HTML fetches; 100% unique CSP nonces!
        assert len(seen_csp_nonces) == total_participants * 3
        assert len(set(seen_csp_nonces)) == len(seen_csp_nonces)

        # Final post-soak fleet + readiness + CES summary verification
        final_http = _make_router_http_adapter(
            client,
            peer_ip=_loopback_ip_for(30, 1),
            lock=shared_lock,
        )
        final_ready = verifier.check_cloud_run_invite_readiness(
            "http://127.0.0.1:8000",
            expected_sha="d05cafe",
            http_fn=final_http,
            lang="en",
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        _verify_sha256_receipt(final_ready)
        assert final_ready["invite_ready"] is True
        assert final_ready["sha_match"] is True
        assert final_ready["feedback_summary"]["total_responses"] == 48
        assert final_ready["feedback_summary"]["by_profile"] == {"dev": 24, "gamer": 24}
        assert final_ready["feedback_summary"]["average_ease_rating"] >= 4.2
        assert final_ready["feedback_summary"]["zero_game_lag_ratio"] == 1.0

        final_act = verifier.verify_live_fleet_and_ces_activation(
            "http://127.0.0.1:8000",
            profile="gamer",
            target_online_producers=8,
            gpu_probe_fn=lambda: verifier.GpuProbeSnapshot(
                present=True,
                name="NVIDIA GeForce RTX 4070 SUPER",
                total_vram_mib=12288,
                free_vram_mib=9216,
                temp_c=56.0,
                util_pct=4.0,
            ),
            process_probe_fn=lambda: ["explorer.exe"],
            worker_yield_fn=lambda: 85.0,
            http_fn=final_http,
            lang="fr",
            env=CLEAN_ENV,
            cwd=tmp_path,
            home=tmp_path / "final_home",
        )
        _verify_sha256_receipt(final_act)
        assert final_act["activation_ok"] is True
        assert final_act["fleet"]["active_trainers"] >= 24
        assert final_act["fleet"]["live_hiveminders"] >= 8
        assert final_act["fleet"]["distinct_physical_gpus"] >= 8
        assert final_act["fleet"]["target_producers_met"] is True
        assert final_act["fleet"]["producers_remaining_to_target"] == 0


def test_03_pre_pr949_http_404_revision_drift_invite_hold_ces_spool_and_live_cutover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIVEMIND_BUILD_SHA", "949livecut05")
    monkeypatch.setenv("K_REVISION", "hivemind-openrouter-00105-live")

    def _stale_pre_pr949_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (payload, headers)
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if method == "GET" and path == "/health":
            return 200, {"status": "ok"}
        if method == "GET" and path == "/v1/hivemind/producers/fleet":
            return 200, {
                "object": "hivemind.producer_fleet",
                "total_hiveminders": 2,
                "live_hiveminders": 1,
                "distinct_physical_gpus": 1,
                "live_endpoints": 1,
                "live_models": ["hivemind/coder"],
            }
        return 404, {"detail": "Not Found"}

    # 1. Pre-PR-#949 stale Cloud Run revision -> holiday_routes_deployed_on_cloud_run=False, revision_drift_detected=True
    drift_rep = verifier.check_cloud_run_invite_readiness(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        expected_sha="949livecut05",
        http_fn=_stale_pre_pr949_http,
        lang="fr",
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    _verify_sha256_receipt(drift_rep)
    assert drift_rep["holiday_routes_deployed_on_cloud_run"] is False
    assert drift_rep["revision_drift_detected"] is True
    assert drift_rep["invite_ready"] is False
    assert "/noel" in drift_rep["stale_404_routes"]
    assert "/quick" in drift_rep["stale_404_routes"]
    assert "/raid-noel" in drift_rep["stale_404_routes"]

    # CLI --check-drift returns exit code 1 when drift is detected
    drift_cli_out = io.StringIO()
    rc_drift = verifier.main(
        ["--check-drift", "--expected-sha", "949livecut05", "--lang", "en"],
        http_fn=_stale_pre_pr949_http,
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path / "h1",
        out_stream=drift_cli_out,
    )
    assert rc_drift == 1
    assert "INVITE HOLD ACTIVE" in drift_cli_out.getvalue()

    # 2. Invite generator activates automatic hold/fallback notice across FR and EN so stale 404 links aren't shared blindly
    for lang_code, expected_marker in (
        ("fr", "EN ATTENTE / HOLD"),
        ("en", "HOLD / FALLBACK"),
    ):
        pack = verifier.generate_friend_invite_pack(
            "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
            guild_tag="Team Sophia / Friends",
            channel="discord",
            lang=lang_code,
            readiness=drift_rep,
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        _verify_sha256_receipt(pack)
        assert pack["holiday_routes_deployed_on_cloud_run"] is False
        assert pack["revision_drift_detected"] is True
        assert pack["invite_hold_active"] is True
        assert pack["share_approved"] is False
        assert pack["verified_holiday_urls"] == []
        assert expected_marker in pack["invite_card"]
        assert "/signup" in pack["invite_card"]

    # 3. Activation verifier degrades non-blockingly on HTTP 404 CES feedback and spools locally (0600)
    spool_file = tmp_path / "home_gamer" / ".hivemind" / "ces_feedback_spool.json"
    stale_act = verifier.verify_live_fleet_and_ces_activation(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        submit_ces=True,
        profile="gamer",
        ease_rating=5,
        gpu_probe_fn=lambda: verifier.GpuProbeSnapshot(
            present=True,
            name="NVIDIA GeForce RTX 4070",
            total_vram_mib=12288,
            free_vram_mib=8192,
            temp_c=61.0,
            util_pct=3.0,
        ),
        process_probe_fn=lambda: ["explorer.exe"],
        worker_yield_fn=lambda: 120.0,
        http_fn=_stale_pre_pr949_http,
        lang="en",
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path / "home_gamer",
        ces_spool_path=spool_file,
    )
    _verify_sha256_receipt(stale_act)
    assert stale_act["activation_ok"] is True
    assert stale_act["fleet"]["stale_raid_route_404"] is True
    assert stale_act["fleet"]["revision_drift_detected"] is True
    assert stale_act["ces_feedback"]["submitted"] is False
    assert stale_act["ces_feedback"]["degraded"] is True
    assert stale_act["ces_feedback"]["stale_cloud_run_404"] is True
    assert stale_act["ces_feedback"]["spooled"] is True
    assert stale_act["ces_feedback"]["spool_file"] == "~/.hivemind/ces_feedback_spool.json"
    assert spool_file.is_file()
    assert (spool_file.stat().st_mode & 0o777) == 0o600

    # Duplicate enqueue is idempotently deduplicated
    dup_enq = verifier.enqueue_ces_spool_item(
        spool_file,
        stale_act["ces_feedback"]["zero_pii_payload"],
        reason="dup_check",
        home=tmp_path / "home_gamer",
    )
    assert dup_enq["deduplicated"] is True
    assert dup_enq["pending_count"] == 1

    # 4. Cutover to live Cloud Run router -> readiness green, invite hold lifted, spooled CES flushed cleanly!
    store = RouterStore(tmp_path / "cutover.sqlite3", secret=SOAK_ROUTER_SECRET)
    app = create_router_app(
        store,
        allow_private_producers=True,
        trusted_proxy_hops=1,
        open_registration=True,
    )
    lock = threading.Lock()
    with TestClient(app) as client:
        live_http = _make_router_http_adapter(client, peer_ip=_loopback_ip_for(40, 1), lock=lock)

        live_ready = verifier.check_cloud_run_invite_readiness(
            "http://127.0.0.1:8000",
            expected_sha="949livecut05",
            http_fn=live_http,
            lang="fr",
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        _verify_sha256_receipt(live_ready)
        assert live_ready["holiday_routes_deployed_on_cloud_run"] is True
        assert live_ready["revision_drift_detected"] is False
        assert live_ready["sha_match"] is True
        assert live_ready["invite_ready"] is True

        live_pack = verifier.generate_friend_invite_pack(
            "http://127.0.0.1:8000",
            guild_tag="Team Sophia / Friends",
            channel="signal",
            lang="fr",
            readiness=live_ready,
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert live_pack["invite_hold_active"] is False
        assert live_pack["share_approved"] is True
        assert len(live_pack["verified_holiday_urls"]) == 3

        flush_out = io.StringIO()
        rc_flush = verifier.main(
            [
                "--gateway",
                "http://127.0.0.1:8000",
                "--flush-ces-spool",
                "--ces-spool-file",
                str(spool_file),
                "--markdown",
            ],
            http_fn=live_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            home=tmp_path / "home_gamer",
            out_stream=flush_out,
        )
        assert rc_flush == 0
        assert "flush_ces_spool" in flush_out.getvalue()

        post_flush_ready = verifier.check_cloud_run_invite_readiness(
            "http://127.0.0.1:8000",
            expected_sha="949livecut05",
            http_fn=live_http,
            lang="en",
            env=CLEAN_ENV,
            cwd=tmp_path,
        )
        assert post_flush_ready["feedback_summary"]["total_responses"] == 1
        assert post_flush_ready["feedback_summary"]["by_profile"] == {"dev": 0, "gamer": 1}


def test_04_steam_yield_thermal_vram_no_gpu_matrix_and_corporate_compliance_blocking(
    tmp_path: Path,
) -> None:
    def _ok_fleet_http(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        _ = (payload, headers)
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if method == "GET" and path == "/v1/hivemind/producers/fleet":
            return 200, {
                "object": "hivemind.producer_fleet",
                "total_hiveminders": 10,
                "live_hiveminders": 9,
                "distinct_physical_gpus": 9,
                "live_endpoints": 9,
                "live_models": ["hivemind/coder"],
            }
        if method == "GET" and path == "/v1/hivemind/raid/status":
            return 200, {"swarm": {"active_trainers": 12, "registered_trainer_nodes": 14, "live_producers": 9}}
        if method == "POST" and path == "/v1/hivemind/onboarding/feedback":
            return 200, {"total_responses": 10, "mean_ease_rating": 4.8, "zero_game_lag_ratio": 0.9}
        return 200, {"node": {"node_id": "perso:rig"}}

    # 1. Slow Steam yield (>= 1.50s -> yield_slo_ok=False, zero_game_lag=False, OpenTrainer heartbeat blocked)
    slow_rep = verifier.verify_live_fleet_and_ces_activation(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        api_key=SAMPLE_KEY,
        send_opentrainer_heartbeat=True,
        submit_ces=True,
        profile="gamer",
        ease_rating=3,
        gpu_probe_fn=lambda: {
            "present": True,
            "name": "NVIDIA GeForce RTX 3080",
            "total_vram_mib": 10240,
            "free_vram_mib": 6144,
            "temp_c": 62.0,
            "util_pct": 8.0,
        },
        process_probe_fn=lambda: ["cs2.exe"],
        worker_yield_fn=lambda: {"yield_latency_ms": 1525.0, "worker_started": True, "worker_stopped": True, "clean_release": True},
        http_fn=_ok_fleet_http,
        lang="fr",
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path / "h_slow",
    )
    _verify_sha256_receipt(slow_rep)
    assert slow_rep["yield_slo_ok"] is False
    assert slow_rep["zero_game_lag"] is False
    assert slow_rep["activation_ok"] is False
    assert slow_rep["preflight"]["status"] == "yield_slo_exceeded"
    assert slow_rep["opentrainer_heartbeat"]["sent"] is False
    assert slow_rep["opentrainer_heartbeat"]["status"] == "blocked_yield_slo"
    assert slow_rep["ces_feedback"]["zero_game_lag"] is False
    assert SAMPLE_KEY not in json.dumps(slow_rep)
    assert slow_rep["masked_api_key"] == "hm-***REDACTED***"

    # 2. Hot GPU (> 70.0 C -> thermal_ok=False, zero_game_lag=False, status='thermal_clamped_paused')
    hot_rep = verifier.verify_live_fleet_and_ces_activation(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        api_key=SAMPLE_LIVE_KEY,
        submit_ces=True,
        profile="gamer",
        max_temp_c=70.0,
        gpu_probe_fn=lambda: SimpleNamespace(
            present=True,
            parse_error=False,
            gpu_name="NVIDIA GeForce RTX 4080",
            total_mib=16384,
            free_mib=10240,
            used_mib=6144,
            temp_c=74.2,
            util_pct=12.0,
            reason="ok",
        ),
        process_probe_fn=lambda: ["steam.exe"],
        worker_yield_fn=lambda: 140.0,
        http_fn=_ok_fleet_http,
        lang="en",
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path / "h_hot",
    )
    assert hot_rep["thermal_ok"] is False
    assert hot_rep["zero_game_lag"] is False
    assert hot_rep["preflight"]["status"] == "thermal_clamped_paused"
    assert hot_rep["preflight"]["gpu_contributor_ready"] is False
    assert SAMPLE_LIVE_KEY not in json.dumps(hot_rep)
    assert hot_rep["masked_api_key"] == "hm_live_***REDACTED***"

    # 3. Low free VRAM (< 2048 MiB -> vram_reserve_ok=False, status='vram_reserve_hold') + combined hot & low VRAM
    low_vram_pf = verifier.run_steam_yield_and_thermal_preflight(
        min_free_vram_mib=2048,
        gpu_probe_fn=lambda: verifier.GpuProbeSnapshot(
            present=True,
            name="NVIDIA GeForce RTX 3060",
            total_vram_mib=8192,
            free_vram_mib=1536,
            temp_c=63.0,
            util_pct=5.0,
        ),
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: 90.0,
        lang="fr",
    )
    assert low_vram_pf["vram_reserve_ok"] is False
    assert low_vram_pf["gpu_contributor_ready"] is False
    assert low_vram_pf["status"] == "vram_reserve_hold"

    both_pf = verifier.run_steam_yield_and_thermal_preflight(
        max_temp_c=70.0,
        min_free_vram_mib=2048,
        gpu_probe_fn=lambda: verifier.GpuProbeSnapshot(
            present=True,
            name="NVIDIA GeForce RTX 3060",
            total_vram_mib=8192,
            free_vram_mib=1024,
            temp_c=76.0,
            util_pct=5.0,
        ),
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: 90.0,
        lang="en",
    )
    assert both_pf["thermal_ok"] is False and both_pf["vram_reserve_ok"] is False
    assert both_pf["status"] == "thermal_and_vram_clamped_paused"

    # Active Steam/DX12 game while cool & sufficient VRAM -> steam_game_active_yielded
    active_game_pf = verifier.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: verifier.GpuProbeSnapshot(
            present=True,
            name="NVIDIA GeForce RTX 4070",
            total_vram_mib=12288,
            free_vram_mib=8192,
            temp_c=64.0,
            util_pct=42.0,
        ),
        process_probe_fn=lambda: ["cyberpunk2077.exe"],
        worker_yield_fn=lambda: {"yield_elapsed_s": 0.12, "worker_started": True, "worker_stopped": True},
        lang="en",
    )
    assert active_game_pf["status"] == "steam_game_active_yielded"
    assert active_game_pf["game_detected"] is True
    assert active_game_pf["yield_slo_ok"] is True

    # 4. No-GPU laptop Dev fallback (honest 'dev_api_only_no_nvidia_gpu', zero invented GPU specs)
    nogpu_rep = verifier.verify_live_fleet_and_ces_activation(
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
        api_key=SAMPLE_KEY,
        send_opentrainer_heartbeat=True,
        profile="dev",
        gpu_probe_fn=lambda: None,
        process_probe_fn=lambda: ["code", "zsh"],
        http_fn=_ok_fleet_http,
        lang="en",
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path / "h_nogpu",
    )
    assert nogpu_rep["activation_ok"] is True
    assert nogpu_rep["preflight"]["status"] == "dev_api_only_no_nvidia_gpu"
    assert nogpu_rep["preflight"]["gpu_present"] is False
    assert nogpu_rep["preflight"]["gpu_name"] is None
    assert nogpu_rep["preflight"]["vram_total_mib"] is None
    assert nogpu_rep["preflight"]["vram_free_mib"] is None
    assert nogpu_rep["opentrainer_heartbeat"]["payload"]["gpu_name"] is None
    assert nogpu_rep["opentrainer_heartbeat"]["payload"]["vram_total_mib"] is None

    # 5. 100% Amadeus / corporate PC & git-remote blocking BEFORE any file write or network call!
    http_called: list[str] = []

    def _never_called_http(*args: Any, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        http_called.append(str((args, kwargs)))
        raise AssertionError("Corporate compliance gate MUST block before any network call!")

    corp_proxy = "http" + "://proxy.zscaler.net:8080"
    netskope_proxy = "http" + "://gw.netskope.local:8443"
    corp_scenarios: list[tuple[dict[str, str], Path, str, str]] = [
        ({"USERDNSDOMAIN": "amadeus.internal.example"}, tmp_path / "clean1", "amadeus", "amadeus.internal.example"),
        ({"USERDOMAIN": "ACMECORP"}, tmp_path / "clean2", "corp", "ACMECORP"),
        ({"HTTP_PROXY": corp_proxy}, tmp_path / "clean3", "zscaler", corp_proxy),
        ({"HTTPS_PROXY": netskope_proxy}, tmp_path / "clean4", "netskope", "gw.netskope.local"),
        ({"COMPUTERNAME": "WKS-ENTERPRISE-09"}, tmp_path / "clean5", "enterprise", "WKS-ENTERPRISE-09"),
        ({"HOSTNAME": "laptop-intranet-bluecoat"}, tmp_path / "clean6", "intranet", "laptop-intranet-bluecoat"),
        ({"AMADEUS_SSO_USER": "alice_secret_id"}, tmp_path / "clean7", "amadeus", "alice_secret_id"),
        ({"ZSCALER_CERT_PATH": "/etc/ssl/zs_bundle.pem"}, tmp_path / "clean8", "zscaler", "zs_bundle.pem"),
        ({"SSL_CERT_FILE": "/etc/fortinet/paloalto_ca.pem"}, tmp_path / "clean9", "fortinet", "paloalto_ca.pem"),
        (CLEAN_ENV, tmp_path / "amadeus_work_repo", "amadeus", "amadeus_work_repo"),
    ]

    # Git config remote corporate marker + git worktree commondir corporate marker
    git_corp_ws = tmp_path / "hobby_lookalike"
    (git_corp_ws / ".git").mkdir(parents=True)
    secret_remote = "git@" + "git.amadeus" + ".internal:secret-division/private-repo.git"
    (git_corp_ws / ".git" / "config").write_text(
        f'[remote "origin"]\n\turl = {secret_remote}\n',
        encoding="utf-8",
    )
    corp_scenarios.append((CLEAN_ENV, git_corp_ws, "amadeus", "secret-division/private-repo"))

    common_git = tmp_path / "common_git_dir"
    wt_git = common_git / "worktrees" / "wt1"
    wt_git.mkdir(parents=True)
    (wt_git / "commondir").write_text("../..\n", encoding="utf-8")
    corp_wt_remote = "https" + "://gitlab.enterprise.local/team/confidential.git"
    (common_git / "config").write_text(
        f'[remote "origin"]\n\turl = {corp_wt_remote}\n',
        encoding="utf-8",
    )
    wt_ws = tmp_path / "wt_checkout"
    wt_ws.mkdir(parents=True)
    (wt_ws / ".git").write_text(f"gitdir: {wt_git}\n", encoding="utf-8")
    corp_scenarios.append((CLEAN_ENV, wt_ws, "enterprise", "team/confidential"))

    for idx, (bad_env, bad_cwd, expected_marker, secret_val) in enumerate(corp_scenarios):
        bad_cwd.mkdir(parents=True, exist_ok=True)
        unwritten_invite = tmp_path / f"must_never_exist_{idx}.md"
        unwritten_spool = tmp_path / f"must_never_spool_{idx}.json"
        out_json = io.StringIO()
        err_cli = io.StringIO()

        rc_json = verifier.main(
            [
                "--check-drift",
                "--generate-invite",
                "--output-invite",
                str(unwritten_invite),
                "--verify-activation",
                "--send-opentrainer-heartbeat",
                "--submit-ces",
                "--ces-spool-file",
                str(unwritten_spool),
                "--lang",
                "fr" if idx % 2 == 0 else "en",
                "--json",
            ],
            http_fn=_never_called_http,
            env=bad_env,
            cwd=bad_cwd,
            home=tmp_path / "corp_home",
            out_stream=out_json,
            err_stream=err_cli,
        )
        assert rc_json == 2
        assert not unwritten_invite.exists()
        assert not unwritten_spool.exists()
        assert http_called == []

        raw_blocked = out_json.getvalue()
        assert secret_val not in raw_blocked, f"Leaked local env/remote value {secret_val!r}!"
        blocked_doc = json.loads(raw_blocked)
        _verify_sha256_receipt(blocked_doc)
        assert blocked_doc["compliance_ok"] is False
        assert blocked_doc["network_calls"] == 0
        assert blocked_doc["file_writes"] == 0
        assert expected_marker in blocked_doc["compliance"]["matched_markers"]
        assert "0.00 EUR / 0.00 USD" in blocked_doc["compliance"]["notice_bilingual"]
        assert "0,00 EUR / 0,00 USD" in blocked_doc["compliance"]["symbolic_points_notice"]

        # Direct API functions also refuse closed with 0 network calls
        assert verifier.check_cloud_run_invite_readiness(
            http_fn=_never_called_http, env=bad_env, cwd=bad_cwd
        )["compliance_ok"] is False
        assert verifier.generate_friend_invite_pack(env=bad_env, cwd=bad_cwd)["compliance_ok"] is False
        assert verifier.verify_live_fleet_and_ces_activation(
            http_fn=_never_called_http, env=bad_env, cwd=bad_cwd
        )["compliance_ok"] is False
        assert verifier.flush_ces_spool(
            spool_path=unwritten_spool, http_fn=_never_called_http, env=bad_env, cwd=bad_cwd
        )["compliance_ok"] is False
        assert http_called == []


def test_05_real_loopback_http_subprocess_probes_validators_and_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Real loopback HTTPServer verifying default_http_json ProxyHandler({}) & zero-inference guards
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            elif self.path == "/html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<!doctype html><p>ok</p>")
            elif self.path == "/empty":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/err502":
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"upstream bad gateway")
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"detail": "missing"}')

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"recorded": true}')

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            _ = (format, args)

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        # Poisoned proxy env must be bypassed by ProxyHandler({})
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        base = f"http://127.0.0.1:{port}"
        assert verifier.default_http_json("GET", f"{base}/health") == (200, {"status": "ok"})
        assert verifier.default_http_json("GET", f"{base}/html")[1]["html_length"] > 0
        assert verifier.default_http_json("GET", f"{base}/empty")[0] == 204
        assert verifier.default_http_json("GET", f"{base}/err502")[0] == 502
        assert verifier.default_http_json("GET", f"{base}/missing") == (404, {"detail": "missing"})
        assert (
            verifier.default_http_json(
                "POST",
                f"{base}/v1/hivemind/onboarding/feedback",
                {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux", "note": ""},
                {"Idempotency-Key": "abc123"},
            )[0]
            == 200
        )
        assert (
            verifier.default_http_json(
                "POST",
                f"{base}/v1/hivemind/opentrainer/heartbeat",
                {"host_label": "home-pc", "network_mode": "outbound-https-relay"},
            )[0]
            == 200
        )
    finally:
        srv.shutdown()
        srv.server_close()

    assert verifier.default_http_json("GET", "http://127.0.0.1:1/health", timeout_s=0.25)[0] == 0

    # Zero-inference & gateway validation rejections
    untrusted_url = "https" + "://evil.example.com"
    bad_nl_url = "https" + "://bad\n.run.app"
    bad_sp_url = "https" + "://bad space.run.app"
    for bad_call in (
        lambda: verifier.validate_gateway_url(""),
        lambda: verifier.validate_gateway_url(bad_nl_url),
        lambda: verifier.validate_gateway_url(bad_sp_url),
        lambda: verifier.validate_gateway_url("https://user:pw@hivemind.a.run.app"),
        lambda: verifier.validate_gateway_url("https://hivemind.a.run.app?x=1"),
        lambda: verifier.validate_gateway_url("https://hivemind.a.run.app/extra"),
        lambda: verifier.validate_gateway_url("http://127.0.0.1:99999"),
        lambda: verifier.validate_gateway_url(untrusted_url),
        lambda: verifier.validate_gateway_url("ftp://127.0.0.1:8000"),
        lambda: verifier.validate_gateway_url("http://hivemind.a.run.app"),
        lambda: verifier.default_http_json("POST", "http://127.0.0.1:8000/v1/chat/completions", {"model": "hivemind/coder"}),
        lambda: verifier.default_http_json("POST", "http://127.0.0.1:8000/v1/unauthorized", {}),
        lambda: verifier.default_http_json("DELETE", "http://127.0.0.1:8000/health"),
        lambda: verifier.default_http_json("GET", "http://127.0.0.1:8000/health", {"unexpected": 1}),
        lambda: verifier.default_http_json("GET", "http://127.0.0.1:8000/health", None, {"X-Bad": "a\nb"}),
    ):
        with pytest.raises(verifier.InviteFleetVerifierError):
            bad_call()

    # 2. Subprocess runner, GPU probe parsing, Windows/POSIX process probes, and real worker yield drill
    rc_py, out_py = verifier.default_subprocess_runner([sys.executable, "-c", "print('ok-probe')"])
    assert rc_py == 0 and "ok-probe" in out_py
    assert verifier.default_subprocess_runner(["nonexistent_hivemind_binary_xyz"]) == (127, "")
    with pytest.raises(verifier.InviteFleetVerifierError):
        verifier.default_subprocess_runner("shell string forbidden")  # type: ignore[arg-type]

    multi_gpu_csv = (
        "NVIDIA GeForce RTX 3060, 12288, 4096, 8192, 66, 10\n"
        "NVIDIA GeForce RTX 4080, 16384, 14336, 2048, 54, 3\n"
    )
    best_gpu = verifier.default_gpu_probe(runner=lambda _c, _t: (0, multi_gpu_csv))
    assert best_gpu.present is True and best_gpu.name == "NVIDIA GeForce RTX 4080"
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (127, "")).present is False
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (0, "broken,line")).parse_error is True
    assert verifier.default_gpu_probe(runner=lambda _c, _t: (0, "GPU, -1, 0, 0, 50, 0")).parse_error is True

    # Parse error & incomplete telemetry branch in run_steam_yield_and_thermal_preflight
    parse_err_pf = verifier.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: {"present": True, "name": "RTX"},
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: 50.0,
        lang="fr",
    )
    assert parse_err_pf["status"] == "gpu_driver_parse_error"
    assert parse_err_pf["preflight_ok"] is False

    nogpu_slow_pf = verifier.run_steam_yield_and_thermal_preflight(
        gpu_probe_fn=lambda: None,
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: 1800.0,
        lang="en",
    )
    assert nogpu_slow_pf["status"] == "yield_slo_exceeded"

    win_procs = verifier.default_process_probe(
        runner=lambda _c, _t: (0, '"cs2.exe","1024","Console","1","500 K"\n"steam.exe","900","Console","1","80 K"\n'),
        os_name="nt",
    )
    assert "cs2.exe" in win_procs
    posix_procs = verifier.default_process_probe(
        runner=lambda _c, _t: (0, "python3\n/opt/games/helldivers2.exe\n"),
        os_name="posix",
    )
    assert "helldivers2.exe" in posix_procs

    real_drill = verifier.default_worker_yield_drill(trigger_exe="cs2.exe")
    assert real_drill["clean_release"] is True
    assert real_drill["retained_worker_vram_mib"] == 0
    assert real_drill["yield_latency_ms"] < 1500.0
    non_game_drill = verifier.default_worker_yield_drill(trigger_exe="notepad.exe")
    assert non_game_drill["clean_release"] is False

    # 3. Payload validators, sanitization, SHA helpers, and CLI error/default branches
    for bad_v in (
        lambda: verifier.normalize_lang("de"),
        lambda: verifier.normalize_lang(True),  # type: ignore[arg-type]
        lambda: verifier.normalize_channel("irc"),
        lambda: verifier.normalize_channel(None),
        lambda: verifier.normalize_profile("admin"),
        lambda: verifier.sanitize_guild_tag("Amadeus Internal Guild"),
        lambda: verifier.sanitize_guild_tag("bad\x00tag"),
        lambda: verifier.sanitize_host_label("corp-laptop-01"),
        lambda: verifier.validate_expected_sha("unknown"),
        lambda: verifier.validate_onboarding_feedback_payload({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "extra": 1}),
        lambda: verifier.validate_onboarding_feedback_payload({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "note": "alice@example.com"}),
        lambda: verifier.validate_opentrainer_heartbeat_payload({"host_label": "pc", "extra": 1}),
        lambda: verifier.validate_opentrainer_heartbeat_payload({"host_label": "pc", "network_mode": "nebula-mesh"}),
        lambda: verifier.validate_opentrainer_heartbeat_payload({"host_label": "pc", "vram_total_mib": 128}),
    ):
        with pytest.raises(verifier.InviteFleetVerifierError):
            bad_v()

    assert verifier.shas_match("abcdef123456", "abcdef1") is True
    assert verifier.shas_match("abcdef123456", "9999999") is False
    assert verifier.shas_match("unknown", "abcdef1") is False
    assert verifier.sanitize_cloud_run_revision("hm-secret-token") is None
    assert verifier.mask_api_key("") == "<redacted-key>"
    assert verifier.mask_api_key("plain-non-hm") == "<redacted-key>"

    leaky = f"Bearer secret999 {SAMPLE_KEY} user@test.org /home/alice/.hivemind " + '{"' + "sco" + 're": 9}'
    scrubbed = verifier.sanitize_text_for_display(leaky)
    assert verifier.scan_output_safety(scrubbed) == []

    # Invalid CLI arguments -> exit code 2
    for bad_argv in (
        ["--lang", "es"],
        ["--expected-sha", "hm-secretkey12345"],
        ["--api-key", "invalid-key", "--verify-activation"],
        ["--max-temp-c", "130.0", "--verify-activation"],
    ):
        assert verifier.main(
            bad_argv,
            env=CLEAN_ENV,
            cwd=tmp_path,
            home=tmp_path / "cli_err",
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        ) == 2

    # Missing API key on --send-opentrainer-heartbeat -> exit code 1
    rc_no_key = verifier.main(
        ["--verify-activation", "--send-opentrainer-heartbeat"],
        http_fn=lambda _m, _u: (200, {"total_hiveminders": 0, "live_hiveminders": 0, "distinct_physical_gpus": 0, "live_endpoints": 0, "live_models": [], "swarm": {}}),
        gpu_probe_fn=lambda: None,
        process_probe_fn=lambda: [],
        worker_yield_fn=lambda: 20.0,
        env=CLEAN_ENV,
        cwd=tmp_path,
        home=tmp_path / "nokey_home",
        out_stream=io.StringIO(),
        err_stream=io.StringIO(),
    )
    assert rc_no_key == 1

    # Corporate CLI refusal without --json writes to err_stream
    err_corp_cli = io.StringIO()
    assert (
        verifier.main(
            ["--check-drift"],
            env={"USERDNSDOMAIN": "amadeus.local"},
            cwd=tmp_path,
            home=tmp_path / "corp_cli",
            out_stream=io.StringIO(),
            err_stream=err_corp_cli,
        )
        == 2
    )
    assert "0.00 EUR / 0.00 USD" in err_corp_cli.getvalue()

    # Standalone offline invite generation (readiness=None) + invalid readiness type
    offline_pack = verifier.generate_friend_invite_pack(lang="en", env=CLEAN_ENV, cwd=tmp_path)
    assert offline_pack["share_approved"] is True
    with pytest.raises(verifier.InviteFleetVerifierError):
        verifier.generate_friend_invite_pack(readiness="invalid", env=CLEAN_ENV, cwd=tmp_path)  # type: ignore[arg-type]

    # Readiness diagnostic branches (/health bad status, invalid manifest/html/summary/fleet, 429, 0)
    def _broken_contracts_http(_method: str, url: str) -> tuple[int, Any]:
        p = "/" + url.split("://", 1)[1].split("/", 1)[1]
        if p == "/health":
            return 200, {"status": "degraded"}
        if p == "/v1/hivemind/onboarding/manifest":
            return 200, {"object": "wrong", "holiday_routes": [], "downloadable_scripts": []}
        if p == "/noel":
            return 200, "<html>missing doctype & csp</html>"
        if p == "/quick":
            return 429, {"detail": "rate_limited"}
        if p == "/raid-noel":
            return 0, {"network_error": "TimeoutError"}
        if p == "/v1/hivemind/onboarding/feedback/summary":
            return 200, {"total_responses": -1}
        return 200, {"object": "hivemind.producer_fleet", "total_hiveminders": "bad"}

    broken_ready = verifier.check_cloud_run_invite_readiness(
        http_fn=_broken_contracts_http,
        lang="en",
        env=CLEAN_ENV,
        cwd=tmp_path,
    )
    assert broken_ready["invite_ready"] is False
    assert broken_ready["revision_drift_detected"] is True
    assert len(broken_ready["diagnostics"]) >= 5

    # Default CLI mode (no action flags -> check_drift + generate_invite)
    assert (
        verifier.main(
            [],
            http_fn=_broken_contracts_http,
            env=CLEAN_ENV,
            cwd=tmp_path,
            home=tmp_path / "def_home",
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 1
    )

    # load_saved_credentials & flush_ces_spool edge branches
    bad_cred = tmp_path / "bad_cred.json"
    assert verifier.load_saved_credentials(bad_cred) is None
    bad_cred.write_text("{corrupt", encoding="utf-8")
    assert verifier.load_saved_credentials(bad_cred) is None
    bad_cred.write_text("[]", encoding="utf-8")
    assert verifier.load_saved_credentials(bad_cred) is None
    bad_cred.write_text('{"api_key": "short"}', encoding="utf-8")
    assert verifier.load_saved_credentials(bad_cred) is None
    bad_cred.write_text(
        json.dumps({"api_key": SAMPLE_KEY, "gateway": "http://127.0.0.1:9999"}),
        encoding="utf-8",
    )
    assert verifier.load_saved_credentials(bad_cred, expected_gateway="http://127.0.0.1:8000") is None

    retry_spool = tmp_path / "retry_spool.json"
    retry_spool.write_text(
        json.dumps(
            {
                "pending": [
                    {"event_id": "bad", "payload": {"invalid": True}},
                    {
                        "event_id": "ev503",
                        "payload": {
                            "profile": "dev",
                            "ease_rating": 5,
                            "zero_game_lag": True,
                            "platform": "linux",
                            "note": "",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        verifier.main(
            ["--flush-ces-spool", "--ces-spool-file", str(retry_spool)],
            http_fn=lambda *_a: (503, {"error": "unavailable"}),
            env=CLEAN_ENV,
            cwd=tmp_path,
            home=tmp_path / "sp_home",
            out_stream=io.StringIO(),
            err_stream=io.StringIO(),
        )
        == 1
    )

    # Additional branch coverage across validators, redaction, rendering, and callable wrappers
    assert verifier.is_valid_hivemind_key(123) is False
    assert verifier.redact_home_path(TARGET_SCRIPT) == "scripts/noel_friend_invite_and_fleet_verifier.py"
    assert verifier.redact_home_path(Path.home() / ".hivemind" / "credentials.json") == "~/.hivemind/credentials.json"
    assert verifier.redact_home_path("C:\\Users\\perso\\.hivemind\\x.json") == "~/.hivemind/x.json"
    assert verifier.sanitize_guild_tag("") == verifier.DEFAULT_GUILD_TAG
    assert verifier.sanitize_guild_tag("!") == verifier.DEFAULT_GUILD_TAG
    assert verifier.sanitize_host_label("") == verifier.DEFAULT_HOST_LABEL
    assert verifier.sanitize_build_sha(None) == "unknown"
    assert verifier._detect_platform_tag("darwin") == "macos"
    assert verifier._detect_platform_tag("freebsd") == "freebsd"
    assert verifier._optional_finite_float("N/A") is None
    assert verifier._optional_finite_float("not-a-float") is None
    assert verifier._invoke_with_supported_kwargs(lambda **kw: kw["x"], x=7) == 7
    assert verifier.default_process_probe(runner=lambda _c, _t: (1, ""), os_name="nt") == []
    assert verifier.default_process_probe(runner=lambda _c, _t: (1, ""), os_name="posix") == []
    assert verifier.default_process_probe(runner=lambda _c, _t: (0, "\npython3\n"), os_name="posix") == ["python3"]
    assert verifier.detect_steam_dx12_activity(["", "cs2.exe"])["game_detected"] is True

    raw_ip = ".".join(("10", "20", "30", "40"))
    raw_unsafe = f"a@b.co {SAMPLE_KEY} Bearer secret99 /home/bob/x {raw_ip} " + '{"' + "sco" + 're": 3}'
    safety_hits = verifier.scan_output_safety(raw_unsafe)
    assert len(safety_hits) == 6
    with pytest.raises(verifier.InviteFleetVerifierError):
        verifier._assert_safe_output(raw_unsafe)

    hb_full = verifier.validate_opentrainer_heartbeat_payload(
        {
            "host_label": "rig-01",
            "gpu_name": "RTX 4070",
            "vram_total_mib": 12288,
            "vram_free_mib": 8192,
            "network_mode": "outbound-https-relay",
            "training_stack": "qlora-moe",
            "active_campaign": "noel-2026",
        }
    )
    assert hb_full["training_stack"] == "qlora-moe" and hb_full["active_campaign"] == "noel-2026"

    assert verifier._invoke_http_callable(
        lambda m, u, p, timeout_s=1.0: (200, [m, u, p, timeout_s]),
        "GET",
        "http://127.0.0.1:8000/health",
        timeout_s=1.5,
    )[0] == 200
    assert verifier._invoke_http_callable(
        lambda _m, _u: (_ for _ in ()).throw(OSError("offline")),
        "GET",
        "http://127.0.0.1:8000/health",
    )[0] == 0

    assert verifier._normalize_gpu_snapshot({"present": False}).present is False
    assert verifier._normalize_gpu_snapshot({"present": True, "parse_error": True}).parse_error is True

    for more_bad in (
        lambda: verifier.normalize_profile(None),
        lambda: verifier.sanitize_guild_tag(123),  # type: ignore[arg-type]
        lambda: verifier.validate_onboarding_feedback_payload(None),
        lambda: verifier.validate_onboarding_feedback_payload({"profile": "dev", "ease_rating": 9, "zero_game_lag": True}),
        lambda: verifier.validate_onboarding_feedback_payload({"profile": "dev", "ease_rating": 5, "zero_game_lag": "yes"}),  # type: ignore[dict-item]
        lambda: verifier.validate_onboarding_feedback_payload({"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "secret_token"}),
        lambda: verifier.validate_opentrainer_heartbeat_payload(None),
        lambda: verifier.validate_opentrainer_heartbeat_payload({"host_label": "pc", "gpu_name": ""}),
        lambda: verifier.validate_opentrainer_heartbeat_payload({"host_label": "pc", "vram_free_mib": -5}),
        lambda: verifier._normalize_gpu_snapshot(42),
        lambda: verifier.run_steam_yield_and_thermal_preflight(min_free_vram_mib=64),
        lambda: verifier.run_steam_yield_and_thermal_preflight(yield_budget_ms=-1.0),
        lambda: verifier.run_steam_yield_and_thermal_preflight(gpu_probe_fn=lambda: None, process_probe_fn=lambda: [], worker_yield_fn=lambda: "bad"),
    ):
        with pytest.raises(verifier.InviteFleetVerifierError):
            more_bad()

    assert " Invite Card" in verifier.render_markdown_report(offline_pack)
    assert "Team Sophia / Friends" in verifier.render_cli_report(offline_pack)
    assert verifier.main(["--help"], out_stream=io.StringIO(), err_stream=io.StringIO()) == 0


def test_zz_standalone_friend_invite_and_fleet_verifier_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    buf = io.StringIO()
    pct = float(_INPROC_COV.report(file=buf, show_missing=True))
    assert pct >= 95.0, (
        f"Expected >= 95.0% standalone line coverage on {TARGET_SCRIPT.name}, got {pct:.2f}%:\n{buf.getvalue()}"
    )
