# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression & negative-control test suite for IP-141, IP-142, IP-150, IP-156, IP-157, and IP-161."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import hivemind.config as hm_cfg
from hivemind.game.exoplanet_hunter_cockpit import (
    create_exoplanet_hunter_fastapi_app,
    render_citizen_scientist_community_hub_html,
    render_traffic_and_tflops_monitor_html,
)
from hivemind.game.webgpu_galaxy_cockpit import execute_server_webgpu_verification_and_sync
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.swarm.rusty_sundials_scientific_bee import SolverTaughtSurrogateExpert
from paper.autoevolve_peer_review_pipeline import (
    LEAN4_RDA_MULTIPLIER_PATH,
    LEAN4_RDA_RECEIPT_PATH,
    verify_manuscript_data_consistency,
    verify_rda_multiplier_lean_receipt,
)


def test_ip141_ip142_cockpit_steam_and_webgpu_auth_and_no_unverified_minting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-141 & IP-142 (P0 Issue #493):
    - Calling `/v1/hivemind/steam/register-login` or `/v1/hivemind/webgpu/register-login` on an existing handle
      without its valid `Bearer hm-...` key returns 401 and never rotates `key_hash` or leaks the key.
    - New registrations start with 0.0 HMT (no 2400 HMT faucet).
    - `/v1/hivemind/steam/contribute-boinc` requires `Authorization: Bearer hm-...` (401 without) and awards
      0.0 HMT (`credit_status == "pending_verification"`).
    """
    monkeypatch.setattr("hivemind.game.exoplanet_hunter_cockpit.ARTIFACTS_ROOT", tmp_path)
    app = create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)

    # 1. Register victim on Steam endpoint
    reg = client.post("/v1/hivemind/steam/register-login", json={"handle": "victim_steam_pilot"})
    assert reg.status_code == 200
    body = reg.json()
    assert body["created"] is True
    assert body["balance_hmt"] == 0.0
    victim_key = body["api_key"]
    assert isinstance(victim_key, str) and victim_key.startswith("hm-")

    # 2. Attacker attempts unauthenticated takeover by calling register-login with victim's handle
    takeover = client.post("/v1/hivemind/steam/register-login", json={"handle": "victim_steam_pilot"})
    assert takeover.status_code == 401
    assert "api_key" not in takeover.json()

    # 3. Victim's original key still works on /v1/hivemind/me and /v1/hivemind/steam/register-login
    me = client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {victim_key}"})
    assert me.status_code == 200
    assert me.json()["handle"] == "victim_steam_pilot"

    resume = client.post(
        "/v1/hivemind/steam/register-login",
        json={"handle": "victim_steam_pilot"},
        headers={"Authorization": f"Bearer {victim_key}"},
    )
    assert resume.status_code == 200
    assert resume.json()["created"] is False
    assert resume.json()["api_key"] is None

    # 4. Same check for WebGPU register-login
    wreg = client.post("/v1/hivemind/webgpu/register-login", json={"handle": "victim_webgpu_pilot"})
    assert wreg.status_code == 200
    assert wreg.json()["balance_hmt"] == 0.0
    wkey = wreg.json()["api_key"]
    assert client.post("/v1/hivemind/webgpu/register-login", json={"handle": "victim_webgpu_pilot"}).status_code == 401
    assert (
        client.post(
            "/v1/hivemind/webgpu/register-login",
            json={"handle": "victim_webgpu_pilot"},
            headers={"Authorization": f"Bearer {wkey}"},
        ).status_code
        == 200
    )

    # 5. IP-142: Unauthenticated /v1/hivemind/steam/contribute-boinc is rejected with 401
    unauth_boinc = client.post("/v1/hivemind/steam/contribute-boinc", json={"handle": "victim_steam_pilot"})
    assert unauth_boinc.status_code == 401

    # Authenticated contribute-boinc awards 0.0 HMT (pending_verification)
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        auth_boinc = client.post(
            "/v1/hivemind/steam/contribute-boinc",
            json={"target_planet": "K2-18 b"},
            headers={"Authorization": f"Bearer {victim_key}"},
        )
    assert auth_boinc.status_code == 200
    bjson = auth_boinc.json()
    assert bjson["credit_status"] == "pending_verification"
    assert bjson["hmt_awarded"] == 0.0
    assert bjson["balance_hmt"] == 0.0


def test_ip156_webgpu_contribute_validates_inputs_and_records_sync_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-156 (P1 Issue #499):
    - Negative transit depth (`recovered_transit_depth_ppm: -1e9`), zero points (`n_points_folded: 0`),
      or non-positive `kernel_exec_ms` are rejected with ValueError / HTTP 400 on both gateway and cockpit.
    - When GCP Ruche sync fails, `gcp_sync_error` and `verification_status: CLIENT_REPORTED_UNVERIFIED` are recorded.
    """
    with pytest.raises(ValueError, match="n_points_folded"):
        execute_server_webgpu_verification_and_sync(
            handle="tester",
            target_planet="K2-18 b",
            gpu_adapter="WebGPU",
            backend="wgsl",
            n_points_folded=0,
            kernel_exec_ms=0.5,
            recovered_transit_depth_ppm=2500.0,
        )

    with pytest.raises(ValueError, match="recovered_transit_depth_ppm"):
        execute_server_webgpu_verification_and_sync(
            handle="tester",
            target_planet="K2-18 b",
            gpu_adapter="WebGPU",
            backend="wgsl",
            n_points_folded=16384,
            kernel_exec_ms=0.5,
            recovered_transit_depth_ppm=-1e9,
        )

    # Gateway endpoint rejects invalid parameters with 400
    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    _, key = store.register("webgpu_tester", "prosumer")
    gw_client = TestClient(create_router_app(store, allow_private_producers=True))
    bad_gw = gw_client.post(
        "/v1/hivemind/webgpu/contribute",
        json={"n_points_folded": 0, "recovered_transit_depth_ppm": -1e9},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert bad_gw.status_code == 400

    # Cockpit endpoint rejects invalid parameters with 400 and awards 0 HMT on valid call
    monkeypatch.setattr("hivemind.game.exoplanet_hunter_cockpit.ARTIFACTS_ROOT", tmp_path)
    cockpit_client = TestClient(create_exoplanet_hunter_fastapi_app())
    ck_key = cockpit_client.post("/v1/hivemind/webgpu/register-login", json={"handle": "ck_webgpu"}).json()["api_key"]
    bad_ck = cockpit_client.post(
        "/v1/hivemind/webgpu/contribute",
        json={"n_points_folded": 0, "recovered_transit_depth_ppm": -1e9},
        headers={"Authorization": f"Bearer {ck_key}"},
    )
    assert bad_ck.status_code == 400

    with patch("urllib.request.urlopen", side_effect=OSError("ruche unreachable")):
        ok_ck = cockpit_client.post(
            "/v1/hivemind/webgpu/contribute",
            json={"n_points_folded": 16384, "kernel_exec_ms": 0.42, "recovered_transit_depth_ppm": 2789.0},
            headers={"Authorization": f"Bearer {ck_key}"},
        )
    assert ok_ck.status_code == 200
    ok_body = ok_ck.json()
    assert ok_body["hmt_awarded"] == 0.0
    assert ok_body["credit_status"] == "pending_verification"
    assert ok_body["verification_status"] == "CLIENT_REPORTED_UNVERIFIED"
    assert ok_body["gcp_cloud_synced"] is False
    assert "ruche unreachable" in str(ok_body["gcp_sync_error"])


def test_ip157_community_hub_and_monitor_escape_stored_xss_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-157 (P1 Issue #499):
    - Stored notes and BOINC sync entries escape `<`, `>`, `"`, `'` on storage and in frontend `escHtml(...)`.
    """
    monkeypatch.setattr("hivemind.game.exoplanet_hunter_cockpit.ARTIFACTS_ROOT", tmp_path)
    client = TestClient(create_exoplanet_hunter_fastapi_app())

    xss_payload = "<img src=x onerror=alert(1)>"
    res = client.post(
        "/api/v1/community/post-note",
        json={
            "handle": xss_payload,
            "role": "<script>evil()</script>",
            "target": "K2-18 b <svg onload=alert(2)>",
            "note": "Transit <img src=x onerror=alert(3)>",
        },
    )
    assert res.status_code == 200
    entry = res.json()["entry"]
    assert "<img" not in entry["gamer_handle"]
    assert "&lt;img" in entry["gamer_handle"]
    assert "<script>" not in entry["scientist_handle"]
    assert "<svg" not in entry["target"]
    assert "<img" not in entry["summary"]

    comm_html = render_citizen_scientist_community_hub_html()
    mon_html = render_traffic_and_tflops_monitor_html()
    assert "function escHtml(" in comm_html
    assert "escHtml(f.summary" in comm_html
    assert "function escHtml(" in mon_html
    assert "escHtml(u.gamer_tag)" in mon_html


def test_ip150_solver_taught_surrogate_rejects_mismatched_initial_state_or_grid() -> None:
    """IP-150 (P1 Issue #501):
    - `SolverTaughtSurrogateExpert.predict_and_verify` verifies the candidate trajectory against the exact
      solver trajectory for the queried initial state and step grid, rather than echoing `model['teacher_drift']`.
    """
    expert = SolverTaughtSurrogateExpert()
    expert.fit_all_four_domains(steps=32, dt=0.01)

    # Matching default initial state (e=0.6) and grid (steps=32, dt=0.01) -> verified
    ok_res = expert.predict_and_verify("Kepler orbit e=0.6", domain="astronomy", steps=32, dt=0.01)
    assert ok_res["surrogate_verified"] is True
    assert ok_res["served_by"] == "surrogate_fast_path_verified"

    # Different eccentricity e=0.25 -> trajectory differs from cached e=0.6 anchors -> rejected & rescued by solver
    diff_ecc = expert.predict_and_verify("Kepler orbit e=0.25", domain="astronomy", steps=32, dt=0.01)
    assert diff_ecc["surrogate_verified"] is False
    assert diff_ecc["surrogate_rejected_by_verifier"] is True
    assert diff_ecc["served_by"] == "rusty_sundials_deterministic_teacher_fallback"
    assert diff_ecc["surrogate_invariant_drift"] > diff_ecc["max_surrogate_drift_tol"]
    # Returned fallback trajectory has x(0) near 1 - 0.25 = 0.75 (not 1 - 0.6 = 0.4)
    assert abs(diff_ecc["sample_trajectory"][0]["x"] - 0.75) < 0.05


def test_ip161_rda_multiplier_lean_receipt_and_prov31_detect_tampering(tmp_path: Path) -> None:
    """IP-161 (P1 Issue #510):
    - `verify_rda_multiplier_lean_receipt` verifies `lean4/mathlib/RDAMultiplier.lean` SHA-256, theorem count,
      zero `sorry`/`axiom` tokens, and allowed axioms against `artifacts/lean_rda_multiplier_check.json`.
    - Tampering with `RDAMultiplier.lean` (e.g. injecting `sorry`) or forging `file_sha256` fails closed,
      including `PROV-31` in `verify_manuscript_data_consistency`.
    """
    clean_audit = verify_rda_multiplier_lean_receipt()
    assert clean_audit["verified"] is True
    assert clean_audit["theorem_count"] == 3
    assert clean_audit["sorry_count"] == 0
    assert clean_audit["user_axiom_count"] == 0

    # Negative control 1: inject `sorry` into a copy of RDAMultiplier.lean
    tampered_lean = tmp_path / "RDAMultiplier.lean"
    tampered_lean.write_text(
        LEAN4_RDA_MULTIPLIER_PATH.read_text(encoding="utf-8") + "\ntheorem bad_thm : True := by sorry\n",
        encoding="utf-8",
    )
    bad_audit = verify_rda_multiplier_lean_receipt(rda_lean_path=tampered_lean)
    assert bad_audit["verified"] is False
    assert bad_audit["sorry_count"] == 1
    assert any("sorry" in f or "SHA-256" in f for f in bad_audit["failures"])

    # Negative control 2: PROV-31 fails when `__lean_rda__` has a mismatched SHA-256
    tex_text = (hm_cfg.PROJECT_ROOT / "paper" / "hivemind_foundation_article.tex").read_text(encoding="utf-8")
    forged_rda = {
        "file_sha256": "0" * 64,
        "theorems": 10,
        "compiled": 10,
        "sorry_tokens": 0,
        "axiom_tokens": 0,
        "sorry_ax_references": 0,
        "records": [],
    }
    consistency = verify_manuscript_data_consistency(
        tex_text=tex_text,
        poc_data={"verdict": {}},
        game_data={"aggregated_metrics": {}},
        arch_data={},
        exp_v3_data={"__lean_rda__": forged_rda},
        lean_audit={"theorem_count": 162},
    )
    prov31 = next(c for c in consistency["checks"] if c["check_id"] == "PROV-31")
    assert prov31["passed"] is False
