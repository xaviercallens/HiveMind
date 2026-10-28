# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""End-to-End Hardware & Collaborator Test Suite for BOINC CUDA PTX Folding & Steam Fat Client (`tests/test_rtx_boinc_and_steam_fat_client.py`).

Tests:
1. Real NVIDIA RTX GPU detection and CUDA Driver PTX kernel (`bls_fold_kernel`) execution across all 8 NASA Exoplanet Archive MCP targets.
2. Automatic Game Preemption (0% BOINC GPU overhead and 0 CUDA allocations during active gameplay) vs. Idle-GPU Folding.
3. Live NASA Exoplanet Archive MCP catalog (`artifacts/nasa_exoplanet_mcp_catalog.json` & `src/hivemind/game/ui/exoplanetCatalog.ts`).
4. Native Steam Fat Client (`HiveMind_Exoplanet_Hunter.exe --self-test`) 5-stage render benchmark and Steamworks VDF depot verification.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("hivemind_boinc_idle_worker", ROOT / "scripts" / "hivemind_boinc_idle_worker.py")
worker = importlib.util.module_from_spec(_spec)
sys.modules["hivemind_boinc_idle_worker"] = worker
_spec.loader.exec_module(worker)  # type: ignore[union-attr]


def test_rtx_gpu_status_and_telemetry() -> None:
    gpu = worker.query_nvidia_gpu_status()
    assert gpu["total_mib"] >= 3500.0
    assert 0.0 <= gpu["utilization_pct"] <= 100.0
    assert 0.0 < gpu["temp_c"] < 105.0
    assert "NVIDIA" in gpu["gpu_model"].upper()


@pytest.mark.parametrize(
    "planet,expected_ppm",
    [
        ("K2-18 b", 2789.0),
        ("TRAPPIST-1 e", 4996.0),
        ("LHS 1140 b", 5385.0),
        ("TOI-700 d", 653.0),
        ("HD 80606 b", 10179.0),
        ("WASP-39 b", 19549.0),
        ("Kepler-11 f", 458.0),
        ("Kepler-186 f", 420.0),
    ],
)
def test_cuda_ptx_bls_kernel_folds_all_eight_nasa_exoplanet_targets(planet: str, expected_ppm: float) -> None:
    res = worker.execute_cuda_ptx_bls_fold(target_planet=planet, n_pts=16384)
    assert res["n_points_folded"] == 16384
    assert abs(res["recovered_transit_depth_ppm"] - expected_ppm) < 1.0
    assert res["kernel_exec_ms"] < 250.0
    assert len(res["result_sha256"]) == 64
    if sys.platform == "win32":
        assert res["backend"] == "nvidia_cuda_driver_ptx_sm50"
        assert "RTX" in res["cuda_device"] or "NVIDIA" in res["cuda_device"]
        assert res["vram_allocated_bytes"] == 131072


def test_active_gameplay_preempts_boinc_gpu_with_zero_cuda_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda: 0.15)
    rep = worker.run_one_idle_gpu_boinc_cycle(
        commander="Cmdr_Gamer_Active_60FPS",
        idle_threshold_seconds=4.0,
        force_fold=False,
    )
    assert rep["boinc_active"] is False
    assert rep["cuda_fold"]["backend"] == "paused_for_active_gameplay"
    assert rep["cuda_fold"]["vram_allocated_bytes"] == 0
    assert rep["cuda_fold"]["kernel_exec_ms"] == 0.0


def test_idle_gpu_automatically_resumes_cuda_ptx_boinc_folding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda: 8.5)
    rep = worker.run_one_idle_gpu_boinc_cycle(
        commander="Cmdr_Gamer_Idle_BOINC",
        idle_threshold_seconds=4.0,
        target_planet="LHS 1140 b",
        force_fold=False,
    )
    assert rep["boinc_active"] is True
    assert rep["cuda_fold"]["n_points_folded"] == 16384
    assert abs(rep["cuda_fold"]["recovered_transit_depth_ppm"] - 5385.0) < 1.0


def test_cli_pause_now_and_resume_now_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert worker.main(["--pause-now"]) == 0
    out_pause = json.loads(capsys.readouterr().out)
    assert out_pause["status"] == "paused"

    assert worker.main(["--resume-now"]) == 0
    out_resume = json.loads(capsys.readouterr().out)
    assert out_resume["status"] == "resumed"


def test_stitch_design_and_nasa_mcp_catalog_artifacts_exist() -> None:
    design_md = ROOT / ".stitch" / "DESIGN.md"
    mcp_cfg = ROOT / ".agents" / "mcp_config.json"
    ts_cat = ROOT / "src" / "hivemind" / "game" / "ui" / "exoplanetCatalog.ts"
    json_cat = ROOT / "artifacts" / "nasa_exoplanet_mcp_catalog.json"

    assert design_md.exists() and "#030914" in design_md.read_text(encoding="utf-8")
    assert mcp_cfg.exists() and "exoplanet-science" in mcp_cfg.read_text(encoding="utf-8")
    assert ts_cat.exists() and "EXOPLANET_CATALOG" in ts_cat.read_text(encoding="utf-8")
    assert json_cat.exists()
    data = json.loads(json_cat.read_text(encoding="utf-8"))
    assert len(data["targets"]) == 8


@pytest.mark.skipif(sys.platform != "win32", reason="Native Win32 GDI+ Steam Fat Client exe runs on Windows")
def test_steam_fat_client_exe_self_test_renders_all_five_stages(tmp_path: Path) -> None:
    exe_path = ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "HiveMind_Exoplanet_Hunter.exe"
    assert exe_path.exists(), f"Missing {exe_path}"
    report_file = tmp_path / "fat_client_self_test_report.json"
    proc = subprocess.run([str(exe_path), "--self-test", str(report_file)], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["app_id"] == 3133700
    assert report["targets_loaded"] >= 8
    assert len(report["stages"]) == 5
    stage_names = [s["stage_name"] for s in report["stages"]]
    assert stage_names == ["1.GALAXIE", "2.ORBIT_BEE", "3.OOLITE_LAND", "4.JWST_LOIS", "5.BOINC_GCP"]


def test_portal_free_steam_game_registration_login_and_boinc_contribution(tmp_path: Path) -> None:
    from starlette.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    store = RouterStore(tmp_path / "router.db", b"s" * 32)
    app = create_router_app(store)
    client = TestClient(app)

    # 1. Verify Portal page includes the Free Steam Game (#steam-free-game) registration section
    portal_html = client.get("/").text
    assert 'id="steam-free-game"' in portal_html
    assert "AppID 3133700" in portal_html
    assert "FreeOnDemand" in portal_html

    # 2. Verify /v1/hivemind/steam/status
    st = client.get("/v1/hivemind/steam/status").json()
    assert st["app_id"] == 3133700
    assert st["portal_registration_status"] == "REGISTERED_AND_LIVE"

    # 3. Simulate first-time Registration of Free Steam Game player
    reg = client.post(
        "/v1/hivemind/steam/register-login",
        json={
            "handle": "Cmdr_RTX2070_Test",
            "steam_id64": "76561198031337001",
            "app_id": 3133700,
            "gpu_model": "NVIDIA GeForce RTX 2070",
            "role": "prosumer",
        },
    ).json()
    assert reg["ok"] is True
    assert reg["created"] is True
    assert reg["api_key"].startswith("hm-")
    assert reg["steam_session_ticket"].startswith("STEAM-3133700-")
    initial_bal = float(reg["balance_hmt"])

    # 4. Simulate returning Login with the same handle
    login = client.post(
        "/v1/hivemind/steam/register-login",
        json={
            "handle": "Cmdr_RTX2070_Test",
            "steam_id64": "76561198031337001",
            "app_id": 3133700,
        },
    ).json()
    assert login["ok"] is True
    assert login["created"] is False
    assert login["account_id"] == reg["account_id"]

    # 5. Contribute a real CUDA PTX BOINC workunit and verify HMT points are credited
    contrib = client.post(
        "/v1/hivemind/steam/contribute-boinc",
        headers={"Authorization": f"Bearer {login['api_key']}"},
        json={"handle": "Cmdr_RTX2070_Test", "target_planet": "K2-18 b", "app_id": 3133700},
    ).json()
    assert contrib["ok"] is True
    assert contrib["hmt_awarded"] == 250.0
    assert contrib["cobblestones_awarded"] == 45.0
    assert abs(float(contrib["balance_hmt"]) - (initial_bal + 250.0)) < 1e-6

    # 6. Verify /v1/hivemind/me reflects the updated balance
    me = client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {login['api_key']}"}).json()
    assert abs(float(me["balance"]) - (initial_bal + 250.0)) < 1e-6


def test_webgpu_zero_install_requires_login_and_credits_wgsl_compute(tmp_path: Path, monkeypatch):
    """Verify /webgpu zero-install page, mandatory registration/login gate (401), and WGSL @compute contribution."""
    from starlette.testclient import TestClient
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app
    from hivemind.game import webgpu_galaxy_cockpit

    monkeypatch.setattr(
        webgpu_galaxy_cockpit,
        "execute_server_webgpu_verification_and_sync",
        lambda **kwargs: {
            "ok": True,
            "receipt_sha256": "489f9ffd9ac611fb3133700000000000489f9ffd9ac611fb3133700000000000",
            "target_planet": kwargs.get("target_planet", "K2-18 b"),
            "n_points_folded": kwargs.get("n_points_folded", 16384),
            "workgroups_dispatched": 256,
            "recovered_transit_depth_ppm": 2789.0,
            "wgsl_entrypoint": "bls_fold_main",
            "gcp_sync": {"ok": True, "http_status": 200},
        },
    )

    store = RouterStore(tmp_path / "webgpu_router.db", b"w" * 32)
    app = create_router_app(store)
    client = TestClient(app)

    # 1. Verify /webgpu page renders with WGSL @compute shader, 5 languages, and mandatory login gate
    resp = client.get("/webgpu")
    assert resp.status_code == 200
    html = resp.text
    assert "@compute @workgroup_size(64)" in html
    assert "fn bls_fold_main" in html
    assert "navigator.gpu" in html
    assert "GALAXIE WEBGPU" in html
    for lang in ("fr", "en", "es", "de", "zh"):
        assert f'id="lang-{lang}"' in html

    # 2. Verify /v1/hivemind/webgpu/status reports mandatory registration & login
    st = client.get("/v1/hivemind/webgpu/status").json()
    assert st["zero_install"] is True
    assert st["registration_and_login_required"] is True
    assert st["workgroup_size"] == 64

    # 3. Verify unauthenticated POST /v1/hivemind/webgpu/contribute is rejected with HTTP 401
    unauth = client.post(
        "/v1/hivemind/webgpu/contribute",
        json={"target_planet": "K2-18 b", "n_points_folded": 16384},
    )
    assert unauth.status_code == 401

    # 4. Register a new WebGPU user
    reg = client.post(
        "/v1/hivemind/webgpu/register-login",
        json={
            "handle": "Cmdr_WebGPU_ZeroInstall",
            "gpu_adapter": "NVIDIA GeForce RTX 2070 (WebGPU / Dawn D3D12)",
        },
    ).json()
    assert reg["ok"] is True
    assert reg["created"] is True
    assert reg["webgpu_session_ticket"].startswith("WEBGPU-")
    assert reg["api_key"].startswith("hm-")
    start_bal = float(reg["balance_hmt"])

    # 5. Returning login with the same handle reuses the account
    login = client.post(
        "/v1/hivemind/webgpu/register-login",
        json={
            "handle": "Cmdr_WebGPU_ZeroInstall",
            "gpu_adapter": "NVIDIA GeForce RTX 2070 (WebGPU / Dawn D3D12)",
        },
    ).json()
    assert login["ok"] is True
    assert login["created"] is False
    assert login["account_id"] == reg["account_id"]

    # 6. Contribute a WebGPU WGSL @compute workunit using Bearer token
    contrib = client.post(
        "/v1/hivemind/webgpu/contribute",
        headers={"Authorization": f"Bearer {login['api_key']}"},
        json={
            "target_planet": "K2-18 b",
            "n_points_folded": 16384,
            "recovered_transit_depth_ppm": 2789.0,
            "kernel_exec_ms": 1.84,
            "gpu_adapter": "NVIDIA GeForce RTX 2070 (WebGPU)",
        },
    )
    assert contrib.status_code == 200
    payload = contrib.json()
    assert payload["ok"] is True
    assert payload["hmt_awarded"] == 250.0
    assert payload["cobblestones_awarded"] == 45.0
    assert abs(float(payload["balance_hmt"]) - (start_bal + 250.0)) < 1e-6
    assert payload["receipt_sha256"].startswith("489f9ffd9ac611fb")


