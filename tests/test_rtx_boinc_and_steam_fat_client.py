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


_T4_CSV = "Tesla T4, 5, 52, 15360, 612, 28.4, 550.54.15\n"
_REAL_PROBE_GPU = worker.probe_gpu  # captured before any monkeypatch replaces worker.probe_gpu


def _ready_probe() -> object:
    """A READY probe built from real nvidia-smi csv through the injected-runner path (external I/O faked only)."""
    return _REAL_PROBE_GPU(
        run=lambda *a, **k: subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=_T4_CSV, stderr=""),
        which=lambda _name: "/usr/bin/nvidia-smi",
    )


def test_rtx_gpu_status_and_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive: a real nvidia-smi line is parsed faithfully. Negative: no nvidia-smi means unavailable, never an invented card."""
    monkeypatch.setattr(worker.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=_T4_CSV, stderr=""),
    )
    gpu = worker.query_nvidia_gpu_status()
    assert gpu["available"] is True
    assert gpu["total_mib"] >= 3500.0
    assert 0.0 <= gpu["utilization_pct"] <= 100.0
    assert 0.0 < gpu["temp_c"] < 105.0
    assert gpu["gpu_model"] == "Tesla T4"

    monkeypatch.setattr(worker.shutil, "which", lambda _name: None)
    assert worker.query_nvidia_gpu_status()["available"] is False


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
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _ready_probe())
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 0.15)
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
    """On a host with a working CUDA driver this runs the real kernel and must be READY on the CUDA backend; on a host without one the
    cycle must be BLOCKED (cuda_fold_unavailable) and never report boinc_active=True for a CPU loop."""
    monkeypatch.setattr(worker, "get_user_idle_seconds", lambda *a, **k: 8.5)
    monkeypatch.setattr(worker, "probe_gpu", lambda *a, **k: _ready_probe())
    boinc_calls: list[bool] = []
    monkeypatch.setattr(worker, "set_boinc_client_gpu_mode", lambda enable: boinc_calls.append(enable) or "test_driver")
    rep = worker.run_one_idle_gpu_boinc_cycle(
        commander="Cmdr_Gamer_Idle_BOINC",
        idle_threshold_seconds=4.0,
        target_planet="LHS 1140 b",
        force_fold=False,
    )
    if rep["status"] == "READY":
        assert rep["boinc_active"] is True and boinc_calls == [True]
        assert rep["cuda_fold"]["backend"] == worker.CUDA_BACKEND
        assert rep["cuda_fold"]["cuda_device"] not in ("", "CPU")
        assert rep["cuda_fold"]["n_points_folded"] == 16384
        assert abs(rep["cuda_fold"]["recovered_transit_depth_ppm"] - 5385.0) < 1.0
    else:
        assert rep["status"] == "BLOCKED" and rep["reason"] == "cuda_fold_unavailable"
        assert rep["boinc_active"] is False and boinc_calls == []
        assert rep["cuda_fold"]["backend"] == "cpu_fallback"


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


def test_stitch_v46_cockpit_and_photorealistic_assets_from_project_10018415280079892710() -> None:
    """Verify the upgraded game at `/` uses the latest Stitch v2 screens and local photorealistic assets."""
    from starlette.testclient import TestClient
    from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app

    app = create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)

    # 1. Main cockpit `/` renders USS HIVEMIND NCC-31337 v4.6 across all 5 Stitch stages (Screens 00..11)
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "USS HIVEMIND NCC-31337 v4.6" in html
    # Stage 1 (Screens 02, 03, 07, 11): All 10 Star Systems + Sub-Spatial Astrometric HUD
    assert "[STAGE 1: GALAXIE ENGAGÉ]" in html
    assert "SPECTRO-SCAN TACTIQUE" in html
    for planet_key in (
        "trappist1e",
        "k218b",
        "kepler11f",
        "hd80606b",
        "toi700d",
        "lhs1140b",
        "wasp39b",
        "kepler186f",
        "koi14",
        "koibin",
    ):
        assert f'data-planet="{planet_key}"' in html
    # Stage 2 (Screen 00): Ion Propulsion Telemetry, Orbital Eccentricity Decay (e), Lagrange Ring L1-L5, Target Dossier, Fleet Rendezvous
    assert "ION PROPULSION TELEMETRY" in html
    assert "ORBITAL ECCENTRICITY DECAY (e)" in html
    assert "VIEWPORT // ASTROMETRIC-02" in html
    assert "ORBITAL DOCK // FLEET RENDEZVOUS" in html
    # Stage 3 (Screens 10 & 06): Cobra MK-III MFD, Retro-Propulsion Slider, [SYS.SPEC.01] Atmo & Litho, Radar Contacts
    assert "COBRA MK-III // HONOUR OF HIVEMIND" in html
    assert "ÉQUIPEMENTS PRIMEABLES" in html
    assert "RETRO-PROPULSION OUTPUT:" in html
    assert "[SYS.SPEC.01] ATMO &amp; LITHO" in html
    assert "RADAR CIBLES &amp; CONTACTS" in html
    # Stage 4 (Screen 01): Beryllium 6.5M NIRSpec•MIRI, 6 Molecular Band Cards, NASA TAP ADQL Sync, Thermal Gradient Model (T_eq)
    assert "BERYLLIUM 6.5M NIRSpec•MIRI" in html
    assert "[NIRSPEC_SPECTRUM_WATERFALL]" in html
    assert "[NASA_TAP_ADQL_SYNC]" in html
    assert "THERMAL GRADIENT MODEL (T_eq)" in html
    # Stage 5 (Screen 04): Dual-Engine Architecture (Steam Fat Client RTX 2070 BOINC vs. Zero-Install WebGPU)
    assert "OPTION A: NATIVE FAT CLIENT (STEAM AppID 3133700)" in html
    assert "OPTION B: ZERO-INSTALL WEBGPU GALAXY" in html
    # Bottom Dual Oscilloscopes & 5-Language Switcher
    assert "CONSOLE SYSTÉMIQUE UCSC // VITESSE RADIALE DOPPLER" in html
    assert "PHOTOMÉTRIE DE TRANSIT KEPLER/TESS" in html
    for lang in ("fr", "es", "de", "en", "zh"):
        assert f'id="lang-{lang}"' in html

    # 2. Local photorealistic Stitch v2 PNG assets serve 200 OK with PNG magic header
    for asset in (
        "stitch_v2_07_photorealistic_milky_way_barred_spiral_galaxy_vi.png",
        "stitch_v2_08_photorealistic_descent_cockpit_view_of_spaceship.png",
        "stitch_v2_09_photorealistic_exoplanet_k2-18_b_hycean_world_vi.png",
        "stitch_v2_05_photorealistic_exoplanet_lhs_1140_b_super-earth.png",
    ):
        assert f"/stitch-assets/{asset}" in html
        img_resp = client.get(f"/stitch-assets/{asset}")
        assert img_resp.status_code == 200
        assert len(img_resp.content) > 500_000
        assert img_resp.content[:3] in (b"\x89PN", b"\xff\xd8\xff")

    # 3. Stitch project metadata endpoint reports project 10018415280079892710 and 12 screens
    meta = client.get("/api/astro/stitch_project").json()
    assert meta["project_id"] == "10018415280079892710"
    assert meta["total_screens"] == 12


def test_intense_all_10_stitch_planet_keys_vetting_and_bee_moe_scan() -> None:
    """Verify all 10 Stitch planet keys resolve to SECTOR-01..10 with accurate vetting and Bee-MoE GPU routing."""
    from starlette.testclient import TestClient
    from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app

    app = create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)

    expected_planets = [
        ("trappist1e", "SECTOR-01-TRAPPIST-1E", "PLANET"),
        ("k218b", "SECTOR-02-K2-18B", "PLANET"),
        ("kepler11f", "SECTOR-03-KEPLER-11F", "PLANET"),
        ("hd80606b", "SECTOR-04-HD-80606B", "PLANET"),
        ("toi700d", "SECTOR-05-TOI-700D", "PLANET"),
        ("lhs1140b", "SECTOR-06-LHS-1140B", "PLANET"),
        ("wasp39b", "SECTOR-07-WASP-39B", "PLANET"),
        ("kepler186f", "SECTOR-08-KEPLER-186F", "PLANET"),
        ("koi14", "SECTOR-09-KOI-CAND-14", "FALSE_POSITIVE"),
        ("koibin", "SECTOR-10-KOI-BIN-02", "FALSE_POSITIVE"),
    ]
    for key, expected_sector, classification in expected_planets:
        scan = client.post("/api/astro/bee-moe-scan", json={"target_id": key}).json()
        assert scan["target_id"] == expected_sector
        disc = client.post(
            "/api/astro/discover",
            json={"target_id": key, "gamer_tag": "IntensePytest", "player_classification": classification},
        ).json()
        assert disc["accurate"] is True
        assert disc["points_earned"] == 600
        assert disc["discovery_entry"]["target_id"] == expected_sector


def test_intense_all_html_and_json_routes_and_security_guards() -> None:
    """Verify all 19 HTML routes, 12 JSON telemetry routes, and security guards (400/401/404)."""
    from starlette.testclient import TestClient
    from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app

    app = create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)

    for html_route in (
        "/",
        "/classic",
        "/game",
        "/webgpu",
        "/portal",
        "/playground",
        "/join",
        "/rejoindre",
        "/community",
        "/boinc",
        "/monitor",
        "/stitch/v46-complete",
        "/stitch/v46-oolite",
        "/stitch/stage1",
        "/stitch/stage2",
        "/stitch/stage3",
        "/stitch/stage3-classic",
        "/stitch/stage4",
        "/stitch/architecture",
    ):
        r = client.get(html_route)
        assert r.status_code == 200, f"Route {html_route} returned {r.status_code}"
        assert len(r.text) > 1000

    manifest = client.get("/api/steam/fat-client-manifest")
    assert manifest.status_code == 200
    assert manifest.json()["app_id"] == 3133700

    # Security checks
    assert client.post("/v1/hivemind/webgpu/register-login", json={"handle": "sys:validators"}).status_code == 400
    assert client.get("/v1/hivemind/me").status_code == 401
    assert (
        client.post(
            "/v1/hivemind/webgpu/contribute",
            headers={"Authorization": "Bearer hm-invalid-forged-token"},
            json={"target_planet": "K2-18 b"},
        ).status_code
        == 401
    )
    assert client.get("/stitch-assets/..%2F..%2Fpyproject.toml").status_code == 404


def test_fat_client_one_time_chapter1_voice_and_planet_analysis_and_life() -> None:
    """Verify the native C# Fat Client and Stitch Web Cockpit speak Chapter 1 explanation only once and provide concise planet analysis + how life could live when moving between planets."""
    from pathlib import Path
    from hivemind.config import PROJECT_ROOT
    from hivemind.game.fat_steam_astro_builder import NATIVE_CSHARP_FAT_CLIENT_SOURCE
    from starlette.testclient import TestClient
    from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app

    # 1. Verify C# Fat Client tracks _chapterIntroSpoken and calls SpeakPlanetAnalysisOnly() in SelectTarget
    assert "private bool[] _chapterIntroSpoken = new bool[5];" in NATIVE_CSHARP_FAT_CLIENT_SOURCE
    assert "private void SpeakPlanetAnalysisOnly()" in NATIVE_CSHARP_FAT_CLIENT_SOURCE
    assert "private string GetOneTimeChapterExplanation(int stage)" in NATIVE_CSHARP_FAT_CLIENT_SOURCE
    assert "_chapterIntroSpoken[0] = true; // Never restart Chapter 1 voice when moving between planets" in NATIVE_CSHARP_FAT_CLIENT_SOURCE
    assert "SpeakPlanetAnalysisOnly();" in NATIVE_CSHARP_FAT_CLIENT_SOURCE

    # 2. Verify all targets (>= 10) have concise AnalysisEn, LifeEn, AnalysisFr, LifeFr
    assert NATIVE_CSHARP_FAT_CLIENT_SOURCE.count("AnalysisEn =") >= 10
    assert NATIVE_CSHARP_FAT_CLIENT_SOURCE.count("LifeEn =") >= 10
    assert NATIVE_CSHARP_FAT_CLIENT_SOURCE.count("AnalysisFr =") >= 10
    assert NATIVE_CSHARP_FAT_CLIENT_SOURCE.count("LifeFr =") >= 10
    assert "How life could live:" in NATIVE_CSHARP_FAT_CLIENT_SOURCE
    assert "Vie possible :" in NATIVE_CSHARP_FAT_CLIENT_SOURCE

    # 3. Verify compiled HiveMind_Exoplanet_Hunter.cs and HiveMind_Exoplanet_Hunter.exe are up to date on disk
    cs_file = PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "HiveMind_Exoplanet_Hunter.cs"
    exe_file = PROJECT_ROOT / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter" / "HiveMind_Exoplanet_Hunter.exe"
    assert cs_file.exists()
    if sys.platform == "win32":
        assert exe_file.exists()
    assert "SpeakPlanetAnalysisOnly();" in cs_file.read_text(encoding="utf-8")

    # 4. Verify Web Cockpit `/` also tracks chapter1IntroDone and PLANET_ANALYSIS_AND_LIFE
    app = create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)
    html = client.get("/").text
    assert "let chapter1IntroDone = false;" in html
    assert "PLANET_ANALYSIS_AND_LIFE" in html
    assert "Vie possible :" in html


def test_playable_web_cockpit_4_stage_cinematic_and_gamepad_keyboard_mouse_controls() -> None:
    """Verify the Stitch v4.6 Web Cockpit provides the full 4-stage playable cinematic loop, clear controls banner, Gamepad/Keyboard/Mouse support, progressive difficulty, and unseen NASA mystery planets."""
    from starlette.testclient import TestClient
    from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app

    app = create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)
    html = client.get("/").text

    # 1. Always-visible Controls & Cinematic Pipeline Banner + Modal
    assert 'id="playable-controls-banner"' in html
    assert 'id="controls-modal"' in html
    assert "⌨️ CLAVIER :" in html
    assert "🖱️ SOURIS :" in html
    assert "🎮 MANETTE" in html
    assert 'id="gamepad-status-badge"' in html
    assert "navigator.getGamepads" in html

    # 2. 4-Stage Playable Cinematic Flow & Interactive Canvases
    assert "launchBeeProbesCinematic()" in html
    assert 'id="orbit-bee-canvas"' in html
    assert 'id="orbit-sync-bar"' in html
    assert "completeStage2AndStartLanding()" in html
    assert 'id="oolite-glide-canvas"' in html
    assert "triggerAssistedTouchdown()" in html
    assert 'id="s4-astrobio-summary"' in html
    assert "advanceToNextPlanetFromStage4()" in html
    assert "requestAnimationFrame(masterGameLoop)" in html

    # 3. Progressive Difficulty & Unseen Real NASA Mystery Planet Pool
    assert 'id="difficulty-badge"' in html
    assert "UNSEEN_NASA_PLANETS" in html
    assert "selectRandomNasaMysteryPlanet" in html
    assert "Proxima Centauri b" in html
    assert "55 Cancri e" in html
    assert "GJ 1214 b" in html
    assert "LP 890-9 c" in html






