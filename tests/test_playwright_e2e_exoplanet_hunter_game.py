# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""End-to-End Playwright Browser Tests for the USS Hivemind v4.6 Exoplanet Hunter Web Game."""

from __future__ import annotations

import socket
import threading
import time
from typing import Generator

import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def live_game_server_url() -> Generator[str, None, None]:
    """Start the Exoplanet Hunter FastAPI app on a free local port for Playwright E2E testing."""
    port = _get_free_port()
    app = create_exoplanet_hunter_fastapi_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)

    yield base_url
    server.should_exit = True
    thread.join(timeout=5.0)


def test_playwright_e2e_full_4_stage_cinematic_keyboard_mouse_and_gamepad(
    live_game_server_url: str,
) -> None:
    """Verify the complete 4-stage playable web game loop in headless Chromium with Keyboard, Mouse & Gamepad."""
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Load Stage 1 (GALAXIE)
        page.goto(live_game_server_url, wait_until="domcontentloaded")
        page.wait_for_selector("#playable-controls-banner", state="visible")

        # Verify clear description of all controls (Keyboard, Mouse, Gamepad) is immediately visible
        banner_text = page.locator("#playable-controls-banner").inner_text()
        assert "CLAVIER" in banner_text
        assert "ZQSD" in banner_text
        assert "SOURIS" in banner_text
        assert "MANETTE" in banner_text

        # Toggle Controls Help Modal with [H] key and close with [Escape]
        assert page.locator("#controls-modal").is_hidden()
        page.keyboard.press("h")
        page.wait_for_selector("#controls-modal", state="visible")
        modal_text = page.locator("#controls-modal").inner_text()
        assert "GUIDE COMPLET DES COMMANDES" in modal_text
        page.keyboard.press("Escape")
        assert page.locator("#controls-modal").is_hidden()

        # Verify Mystery NASA Planet button [M] selects an unseen real NASA exoplanet and shows global toast
        page.click("#btn-new-mystery-planet")
        target_name = page.locator("#s1-hud-target-name").inner_text()
        unseen_prefixes = (
            "Proxima Centauri b",
            "55 Cancri e",
            "GJ 1214 b",
            "LP 890-9 c",
            "HD 189733 b",
            "TOI-1452 b",
        )
        assert any(p in target_name for p in unseen_prefixes)
        assert page.locator("#global-game-toast").is_visible()

        # Verify Keyboard steering in Stage 1 moves the USS Hivemind ship smoothly
        initial_left = page.evaluate(
            "parseFloat(document.getElementById('hivemind-marker').style.left)"
        )
        page.keyboard.down("ArrowRight")
        page.wait_for_timeout(140)
        page.keyboard.up("ArrowRight")
        moved_left = page.evaluate(
            "parseFloat(document.getElementById('hivemind-marker').style.left)"
        )
        assert moved_left > initial_left

        # Verify Mouse click on a planet card selects LHS 1140 b
        page.click('.planet-card[data-planet="lhs1140b"]')
        assert page.locator("#s1-hud-target-name").inner_text() == "LHS 1140 b"
        assert "1.53 g" in page.locator("#s1-hud-physics-tag").inner_text()

        # Launch Bee-MoE surveillance robots from Stage 1 -> transitions to Stage 2 after probe animation
        page.click("#btn-s1-launch-robots")
        page.wait_for_selector("#probe-launch-layer div", state="attached", timeout=2000)
        page.wait_for_selector("#viewport-stage-2", state="visible", timeout=3500)

        # 2. Verify Stage 2 (ORBITE BEE) interactive 60 FPS canvas & orbital telemetry
        assert page.locator("#orbit-bee-canvas").is_visible()
        orbit_explain = page.locator("#orbit-astro-explain").inner_text()
        assert "LHS 1140 b" in orbit_explain

        # Steer robots with mouse on the orbital radar canvas & click orbital sync actions
        canvas_box = page.locator("#orbit-bee-canvas").bounding_box()
        assert canvas_box is not None
        page.mouse.click(canvas_box["x"] + 160, canvas_box["y"] + 110)
        page.click("#btn-deploy-probe")
        page.click("#btn-sync-phase")

        sync_pct_text = page.locator("#orbit-sync-pct").inner_text()
        sync_val = int(sync_pct_text.replace("%", "").strip())
        assert sync_val >= 50

        # Transition from Stage 2 -> Stage 3 (Atmospheric Landing)
        page.click("#btn-go-stage3")
        page.wait_for_selector("#viewport-stage-3", state="visible", timeout=3000)

        # 3. Verify Stage 3 (OOLITE LAND) 60 FPS glide-slope canvas, atmospheric layers & retro-thrusters
        assert page.locator("#oolite-glide-canvas").is_visible()
        layer_text = page.locator("#oolite-atmo-layer-tag").inner_text()
        assert "COUCHE" in layer_text
        wind_text = page.locator("#oolite-wind-tag").inner_text()
        assert "VENT:" in wind_text
        atmo_explain = page.locator("#oolite-atmo-detail-text").inner_text()
        assert "LHS 1140 b" in atmo_explain

        # Fire retro-thrusters via Space key & button, verify speed decreases
        speed_before = float(page.locator("#gauge-speed").inner_text())
        page.click("#btn-retro-burn")
        page.keyboard.press("Space")
        speed_after = float(page.locator("#gauge-speed").inner_text())
        assert speed_after < speed_before

        # Trigger assisted touchdown -> awards +1,000 HMT and transitions to Stage 4
        score_before_land = int(
            page.locator("#score-hmt").inner_text().replace(",", "").strip()
        )
        page.click("#btn-auto-land")
        assert "ATTERRISSAGE RÉUSSI" in page.locator("#terrain-alert").inner_text()
        score_after_land = int(
            page.locator("#score-hmt").inner_text().replace(",", "").strip()
        )
        assert score_after_land >= score_before_land + 1000

        # Wait for automatic transition to Stage 4 (JWST/LOIS)
        page.wait_for_selector("#viewport-stage-4", state="visible", timeout=3500)

        # 4. Verify Stage 4 (JWST/LOIS) astrobiology analysis, molecular bands, voting & difficulty escalation
        assert page.locator("#s4-astrobio-summary").is_visible()
        life_text = page.locator("#s4-life-text").inner_text()
        assert "LHS 1140 b" in life_text or "Océan" in life_text

        # Click a molecular band card and verify spectral lock readout
        page.evaluate("window.highlightMolecule('O₃ (Ozone)', '9.6 μm', '96.4%')")
        assert "O₃ (Ozone)" in page.locator("#dms-detection-tag").inner_text()

        # Approve the planet -> verify gold medal stamp on bottom rack & difficulty level escalation (NIV. 1/10 -> NIV. 2/10)
        assert "NIV. 1/10" in page.locator("#difficulty-badge").inner_text()
        page.click("#btn-approve")
        assert "NIV. 2/10" in page.locator("#difficulty-badge").inner_text()
        medal_text = page.locator(
            '.planet-card[data-planet="lhs1140b"] .medal-stamp'
        ).inner_text()
        assert "VALIDÉ" in medal_text

        # 5. Simulate HTML5 Gamepad API connection and verify gamepad status badge turns green
        page.evaluate(
            """() => {
            navigator.getGamepads = () => [{
              id: 'Xbox 360 Controller (XInput STANDARD GAMEPAD)',
              index: 0,
              connected: true,
              axes: [0.65, -0.35, 0.0, 0.0],
              buttons: Array.from({ length: 16 }, () => ({ pressed: false, value: 0 }))
            }];
          }"""
        )
        page.wait_for_timeout(120)
        gp_badge_text = page.locator("#gamepad-status-badge").inner_text()
        assert "MANETTE ACTIVE" in gp_badge_text or "MANETTE CONNECTÉE" in gp_badge_text

        # Click "PROCHAINE PLANÈTE (NIV. +1)" to return to Stage 1 for the next planet
        page.click("#btn-next-planet-s1")
        page.wait_for_selector("#viewport-stage-1", state="visible", timeout=2000)

        assert page_errors == [], f"Unexpected browser JS errors: {page_errors}"
        browser.close()


def test_playwright_e2e_keyboard_only_and_false_positive_and_i18n(
    live_game_server_url: str,
) -> None:
    """Verify pure keyboard shortcuts (Space, Enter, Tab/N, 1-5), false-positive vetting, and 5-language switcher."""
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(live_game_server_url, wait_until="domcontentloaded")
        page.wait_for_selector("#playable-controls-banner", state="visible")

        # Test cycling planets via [N] key and [◀ / ▶] buttons
        before_planet = page.locator("#s1-hud-target-name").inner_text()
        page.keyboard.press("n")
        after_planet = page.locator("#s1-hud-target-name").inner_text()
        assert after_planet != before_planet

        # Test selecting Eclipsing Binary false positive (koibin) and running keyboard-only flow
        page.evaluate("window.selectPlanet('koibin')")
        assert "KOI-Bin #02" in page.locator("#s1-hud-target-name").inner_text()

        # Press Space in Stage 1 -> launches Bee-MoE robots and transitions to Stage 2
        page.keyboard.press("Space")
        page.wait_for_selector("#viewport-stage-2", state="visible", timeout=3500)

        # Press Space in Stage 2 -> boosts orbital sync, then Enter -> transitions to Stage 3
        page.keyboard.press("Space")
        page.keyboard.press("Enter")
        page.wait_for_selector("#viewport-stage-3", state="visible", timeout=3000)

        # Press Enter in Stage 3 -> assisted touchdown and transition to Stage 4
        page.keyboard.press("Enter")
        page.wait_for_selector("#viewport-stage-4", state="visible", timeout=3500)

        # Press 'r' (or click #btn-reject) in Stage 4 to correctly reject KOI-Bin #02 as a false positive (+800 HMT)
        score_before = int(
            page.locator("#score-hmt").inner_text().replace(",", "").strip()
        )
        page.keyboard.press("r")
        score_after = int(
            page.locator("#score-hmt").inner_text().replace(",", "").strip()
        )
        assert score_after == score_before + 800
        assert "BRAVO" in page.locator("#global-game-toast").inner_text()

        # Test 5-Language Switcher (FR / ES / DE / EN / ZH) and Stage 5 tab
        for lang in ("es", "de", "en", "zh", "fr"):
            page.click(f"#lang-{lang}")
        page.keyboard.press("5")
        page.wait_for_selector("#viewport-stage-5", state="visible", timeout=2000)

        assert page_errors == [], f"Unexpected browser JS errors: {page_errors}"
        browser.close()


def test_playwright_e2e_webgpu_stitch_4_stage_gameplay_and_wgsl_compute(
    live_game_server_url: str,
) -> None:
    """Verify the upgraded /webgpu interface with Stitch v2 dossier, 4-stage playable cinematic, NASA Mystery Planet, and WGSL @compute."""
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(f"{live_game_server_url}/webgpu", wait_until="domcontentloaded")
        page.wait_for_selector("#webgpu-cinematic-bar", state="visible")

        # 1. Verify Stitch v2 photorealistic preview & dual SVG oscilloscopes
        img_src = page.locator("#stitch-planet-preview-img").get_attribute("src") or ""
        assert "/stitch-assets/stitch_v2_" in img_src
        assert "Astrophysique" in page.locator("#webgpu-astro-explain").inner_text()
        assert "Exobiologie" in page.locator("#webgpu-life-analysis").inner_text()

        # 2. Register & login on /webgpu and run a WGSL @compute workunit (+250 HMT)
        page.fill("#commanderHandle", "Cmdr_Playwright_WebGPU")
        page.click("#btnRegisterLogin")
        page.wait_for_function(
            "() => !document.getElementById('btnContributeWebGPU').disabled",
            timeout=5000,
        )
        page.click("#btnContributeWebGPU")
        page.wait_for_function(
            "() => document.getElementById('computeStatusBox').textContent.includes('WORKUNIT WEBGPU VALIDÉE')",
            timeout=5000,
        )

        # 3. Roll a New Unseen NASA Mystery Planet ([M] / #btn-webgpu-mystery-planet)
        page.click("#btn-webgpu-mystery-planet")
        assert "Proxima Cen b" in page.locator("#t-name").inner_text()
        assert "DIFFICULTÉ NIV. 2/10" in page.locator("#webgpu-difficulty-badge").inner_text()

        # 4. Play through all 4 Stages via primary CTA & keyboard shortcuts
        # Stage 1 -> Stage 2 (Orbit Bee-MoE)
        page.click("#btn-webgpu-stage-cta")
        assert "active" in (page.locator("#webgpu-stage-2").get_attribute("class") or "")

        # Stage 2 -> Stage 3 (Oolite Atmospheric Descent)
        page.keyboard.press("Space")
        page.keyboard.press("Space")
        assert "active" in (page.locator("#webgpu-stage-3").get_attribute("class") or "")
        stage3_img = page.locator("#stitch-planet-preview-img").get_attribute("src") or ""
        assert "08_photorealistic_descent_cockpit_view_of_spaceship.png" in stage3_img

        # Stage 3 -> Stage 4 (JWST Analysis & Vote)
        page.keyboard.press("Enter")
        assert "active" in (page.locator("#webgpu-stage-4").get_attribute("class") or "")

        # Approve Proxima Cen b in Stage 4 -> awards +800 HMT, gold medal on #planet-dock, and Difficulty Niv. 3/10
        page.click("#btn-webgpu-approve")
        assert "DIFFICULTÉ NIV. 3/10" in page.locator("#webgpu-difficulty-badge").inner_text()
        assert "🏅" in page.locator("#planet-dock").inner_text()

        assert page_errors == [], f"Unexpected browser JS errors on /webgpu: {page_errors}"
        browser.close()


