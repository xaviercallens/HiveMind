# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The game pages must not fabricate accounts, keys, balances or GPU contributions in the visitor's browser (audit 2026-11-03).

Before: the WebGPU page invented an ``hm-webgpu-*`` key and a 500 HMT balance when the gateway was unreachable, claimed "+250 HMT" for work
the server credits with 0 until it can be verified, and the Stitch page registered an account and posted a fabricated RTX 2070 contribution
(5.8% GPU use, 12 s idle) to the dashboard for every visitor on page load."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1] / "src/hivemind/game"
EXOPLANET = (ROOT / "exoplanet_hunter_cockpit.py").read_text(encoding="utf-8")
WEBGPU = (ROOT / "webgpu_galaxy_cockpit.py").read_text(encoding="utf-8")
STEAM = (ROOT / "steam_edition.py").read_text(encoding="utf-8")
STITCH = (ROOT / "stitch_v46_cockpit_template.py").read_text(encoding="utf-8")


def test_webgpu_page_invents_no_key_balance_or_credit() -> None:
    assert "hm-webgpu-" not in WEBGPU  # no key minted in the browser
    assert "balance_hmt: 500" not in WEBGPU and "balanceHmt + 250" not in WEBGPU and "500 HMT" not in WEBGPU
    assert "+250 HMT" not in WEBGPU and "+45 Cobblestones" not in WEBGPU
    assert "api/v1/community/boinc-sync" not in WEBGPU  # nothing mirrored to the dashboard from the browser
    assert "aucun compte n'a été créé" in WEBGPU  # unreachable gateway is an honest error
    assert "data.hmt_awarded" in WEBGPU  # what the server reports is what the page shows


def test_stitch_page_posts_nothing_and_does_not_run_on_load() -> None:
    body = STITCH[STITCH.index("window.simulateSteamRegisterLoginAndContribute"):]
    body = body[: body.index("};") + 2]
    for forbidden in ("fetch(", "register-login", "contribute-boinc", "boinc-sync", "gpu_utilization_pct", "hivemind_api_key"):
        assert forbidden not in body, forbidden
    assert "SIMULATION LOCALE" in body
    assert "\n  simulateSteamRegisterLoginAndContribute();\n" not in STITCH  # no automatic call for every visitor
    assert "+250 HMT" not in STITCH and "+500 HMT" not in STITCH


def test_ip197_ip206_no_numeric_or_fallback_literals_in_game_templates_and_telemetry_formatting() -> None:
    """IP-197 & IP-206:
    - No numeric `|| <digit>` fallback across exoplanet_hunter_cockpit.py, webgpu_galaxy_cockpit.py, or steam_edition.py.
    - Empty telemetry payload formats as `n/a`; populated payload formats with actual numbers.
    """
    from hivemind.game.webgpu_galaxy_cockpit import format_webgpu_live_gcp_telemetry_banner

    for name, script in (
        ("exoplanet_hunter_cockpit.py", EXOPLANET),
        ("webgpu_galaxy_cockpit.py", WEBGPU),
        ("steam_edition.py", STEAM),
    ):
        matches = re.findall(r"\|\| *[0-9]", script)
        assert matches == [], f"{name} still has numeric || fallbacks: {matches}"

    telemetry_block = WEBGPU[WEBGPU.index("syncWebGPULiveGcpTelemetry") :]
    assert re.search(r"\|\| *\d", telemetry_block) is None

    empty_banner = format_webgpu_live_gcp_telemetry_banner({})
    assert empty_banner == "🟢 GCP Live: n/a Comptes • n/a Modèles • T4 n/a MiB • n/a WUs • OpenTrainer n/a GPU"

    full_banner = format_webgpu_live_gcp_telemetry_banner(
        {
            "accounts": {"total_registered_accounts": 42},
            "models": {"live_models_count": 7},
            "t4_boinc": {"vram_free_mib": 12345, "assimilated_workunits": 88},
            "opentrainer": {"total_registered_nodes": 3},
        }
    )
    assert full_banner == "🟢 GCP Live: 42 Comptes • 7 Modèles • T4 12,345 MiB • 88 WUs • OpenTrainer 3 GPU"


def test_ip196_planet_forecast_html_renders_data_source_and_na_when_baseline_empty() -> None:
    """IP-196 negative & positive controls:
    - Default payload (`data_source: ASSUMED_BASELINE_SNAPSHOT_NOT_LIVE`) prints `data_source`,
      contains no `REAL LIVE`, no `(Real)`, and no `base.<field> || <digit>`.
    - Rendering with `current_live_baseline={}` renders `kpi-baseline-nodes` as `n/a` (not 59).
    """
    from hivemind.economy.international_launch_views import render_gcp_planet_forecast_html

    default_html = render_gcp_planet_forecast_html()
    assert "ASSUMED_BASELINE_SNAPSHOT_NOT_LIVE" in default_html
    assert "REAL LIVE" not in default_html
    assert "(Real)" not in default_html
    assert re.search(r"base\.\w+\s*\|\|\s*\d", default_html) is None

    empty_base_html = render_gcp_planet_forecast_html({"current_live_baseline": {}})
    m = re.search(r'id="kpi-baseline-nodes"[^>]*>([^<]+)<', empty_base_html)
    assert m is not None
    assert m.group(1).strip() == "n/a"


def test_ip205_fat_client_manifest_reports_not_built_when_manifest_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IP-205 negative & positive controls:
    - When `steam_fat_client_manifest.json` is absent under `PROJECT_ROOT`, `/api/steam/fat-client-manifest`
      does not report `status: READY`.
    - When the manifest file exists on disk, its contents are returned.
    """
    import hivemind.game.exoplanet_hunter_cockpit as cockpit_mod

    monkeypatch.setattr(cockpit_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cockpit_mod, "ARTIFACTS_ROOT", tmp_path)
    client = TestClient(cockpit_mod.create_exoplanet_hunter_fastapi_app())

    missing_resp = client.get("/api/steam/fat-client-manifest")
    assert missing_resp.status_code in (200, 404)
    missing_body = missing_resp.json()
    assert missing_body.get("status") == "NOT_BUILT"
    assert missing_body.get("status") != "READY"

    manifest_dir = tmp_path / "steam_dist" / "HiveMind_Astrophysics_Exoplanet_Hunter"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "steam_fat_client_manifest.json").write_text(
        '{"app_id": 3133700, "status": "READY", "stitch_planet_keys": ["k1"]}',
        encoding="utf-8",
    )
    present_resp = client.get("/api/steam/fat-client-manifest")
    assert present_resp.status_code == 200
    assert present_resp.json()["status"] == "READY"

