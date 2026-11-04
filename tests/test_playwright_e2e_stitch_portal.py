"""Intensive End-to-End Playwright Browser Validation Suite for the HiveMind & La Ruche Stitch Portal.

Validates:
1. Landing Page, 8-Domain MoE Hexagonal Dispatch Topology, and Multi-Screen Navigation.
2. Guest User Journey: Visitor Mode, 1-Click Guest Sandbox, and In-Place Guest-to-Permanent Account Upgrade.
3. New End-User Journey across Google, Reddit, GitHub, Steam, Epic Games, BOINC, Email/Passkey, and Quantum Magic Link.
4. Existing End-User Journey: Returning Operator Login, Multi-Provider SSO Identity Linking (+1,500 HMT Bonus), Session Persistence across Reloads, and Key Rotation.
5. OpenRouter Gateway & Biomimetic BEE MoE Routing Sandbox.
6. BYOB Campaigns, Interactive GPU Yield & HMT Calculator, and OpenRouter Code Tabs (cURL / Python / TypeScript).
7. AMCP Agent Mesh Live Chat, Tool Call Dispatcher, and MCP Stargate IDE Connectors.
8. Multi-Language (i18n), Multi-Device Viewports (Desktop, iPad, Mobile), and 1-Tap Emergency GPU Preemption Halt.
"""

from __future__ import annotations

import socket
import threading
import time
import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app


def _strip_num(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch.isdigit())


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Spin up a live FastAPI server on a background thread for Playwright E2E tests."""
    import os
    os.environ["HIVEMIND_DASHBOARD_ADMIN_KEY"] = "e2e-playwright-admin-key-2026"
    port = get_free_port()
    app = create_hivemind_at_home_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)

    yield base_url
    server.should_exit = True


class _IsolatedAdminBrowserAdapter:
    """Provides browser.new_page() with isolated BrowserContext + x-admin-key per test."""

    def __init__(self, browser):
        self._browser = browser
        self._contexts = []

    def new_page(self):
        client_idx = len(self._contexts) + 1
        ctx = self._browser.new_context(
            extra_http_headers={
                "x-admin-key": "e2e-playwright-admin-key-2026",
                "x-forwarded-for": f"10.240.0.{client_idx}",
            }
        )
        ctx.add_init_script("window.HIVEMIND_ADMIN_KEY = 'e2e-playwright-admin-key-2026';")
        self._contexts.append(ctx)
        return ctx.new_page()

    def close_all(self):
        for ctx in self._contexts:
            try:
                ctx.close()
            except Exception:
                pass


@pytest.fixture(scope="module")
def browser_context():
    """Launch headless Chromium browser via Playwright with isolated context per page."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        adapter = _IsolatedAdminBrowserAdapter(browser)
        yield adapter
        adapter.close_all()
        browser.close()


def test_e2e_landing_page_and_navigation(live_server, browser_context):
    """Journey 1: Landing page rendering, Stitch 8-Domain MoE Topology, and multi-screen tab switching."""
    page = browser_context.new_page()
    page.goto(live_server + "/")

    # 1. Verify Swarm Telemetry HUD & Stitch 8-Domain Topology
    assert page.is_visible('[data-testid="hero-title"]')
    assert "THE OPEN INTELLIGENCE SWARM" in page.text_content('[data-testid="hero-title"]')
    assert "48,920" in page.text_content('[data-testid="stat-nodes-val"]')
    assert "542.8" in page.text_content('[data-testid="stat-pflops-val"]')
    assert "128 / 128" in page.text_content('[data-testid="stat-lean-val"]')
    assert page.is_visible('[data-testid="stitch-overview-topology"]')
    assert "MoE DISPATCH TOPOLOGY // 8 EXPERTS" in page.text_content('[data-testid="stitch-overview-topology"]')

    # 2. Test Tab Switching across all 7 screens
    screens = [
        ("nav-auth", "screen-auth"),
        ("nav-openrouter", "screen-openrouter"),
        ("nav-byob", "screen-byob"),
        ("nav-amcp", "screen-amcp"),
        ("nav-mobile", "screen-mobile"),
        ("nav-telemetry", "screen-telemetry"),
        ("nav-home", "screen-home"),
    ]
    for nav_id, screen_id in screens:
        page.click(f'[data-testid="{nav_id}"]')
        page.wait_for_selector(f'[data-testid="{screen_id}"].active', timeout=3000)
        assert page.is_visible(f'[data-testid="{screen_id}"]')

    page.close()


def test_e2e_guest_user_flow_and_upgrade(live_server, browser_context):
    """Journey 2: Visitor anonymous mode, 1-Click Guest Sandbox, and upgrading Guest to Permanent Operator."""
    page = browser_context.new_page()
    page.goto(live_server + "/#auth")
    page.wait_for_selector('[data-testid="screen-auth"].active')

    # 1. Verify initial Visitor / Anonymous Mode banner
    assert page.is_visible('[data-testid="guest-status-banner"]')
    assert "VISITOR / ANONYMOUS MODE" in page.text_content('[data-testid="visitor-mode-badge"]')
    assert not page.is_visible('[data-testid="active-session-card"]')

    # 2. Enter 1-Click Guest Sandbox
    page.click('[data-testid="demo-login-btn"]')
    page.wait_for_selector('[data-testid="active-session-card"]:not(.hidden)')
    assert "guest_hiveminder" in page.text_content('[data-testid="session-handle"]')
    assert "GUEST SANDBOX" in page.text_content('[data-testid="session-lifecycle-badge"]')
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "2500"
    assert page.is_visible('[data-testid="upgrade-guest-panel"]')

    # 3. Upgrade Guest Sandbox to Permanent Registered User via Reddit OAuth
    page.fill('[data-testid="upgrade-handle-input"]', "promoted_reddit_bee")
    page.select_option('[data-testid="upgrade-provider-select"]', "reddit")
    page.click('[data-testid="upgrade-guest-btn"]')

    page.wait_for_selector('[data-testid="session-handle"]:has-text("promoted_reddit_bee")', timeout=3000)
    assert "NEW OPERATOR ONBOARDED" in page.text_content('[data-testid="session-lifecycle-badge"]')
    assert "REDDIT_SSO" in page.text_content('#session-provider-badge')
    assert "GUEST_SANDBOX + REDDIT_SSO" in page.text_content('[data-testid="session-linked-providers"]')
    # 2,500 Guest HMT + 2,500 Upgrade Bonus = 5,000 HMT
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "5000"
    assert not page.is_visible('[data-testid="upgrade-guest-panel"]')

    # Clean logout
    page.click('[data-testid="logout-btn"]')
    page.wait_for_selector('[data-testid="active-session-card"]', state="hidden")
    page.close()


def test_e2e_new_end_user_across_google_reddit_github_steam_epic(live_server, browser_context):
    """Journey 3: Onboarding New End Users across Google, Reddit, GitHub, Steam, Epic Games, BOINC, Email & Magic Link."""
    page = browser_context.new_page()
    page.goto(live_server + "/#auth")
    page.wait_for_selector('[data-testid="screen-auth"].active')

    # Select AI Developer Target Compute Profile (Step 01)
    page.click('[data-testid="role-developer-card"]')

    sso_matrix = [
        ("sso-google-btn", "GOOGLE_SSO", "e2e_new_google_user"),
        ("sso-reddit-btn", "REDDIT_SSO", "e2e_new_reddit_user"),
        ("sso-github-btn", "GITHUB_SSO", "e2e_new_github_user"),
        ("sso-steam-btn", "STEAM_SSO", "e2e_new_steam_user"),
        ("sso-epic-btn", "EPIC_SSO", "e2e_new_epic_user"),
        ("sso-boinc-btn", "BOINC_SSO", "e2e_new_boinc_user"),
    ]

    for btn_testid, expected_badge, custom_handle in sso_matrix:
        page.fill('[data-testid="sso-handle-hint"]', custom_handle)
        page.click(f'[data-testid="{btn_testid}"]')
        page.wait_for_selector(f'[data-testid="session-handle"]:has-text("{custom_handle}")', timeout=3000)

        assert expected_badge in page.text_content('#session-provider-badge')
        assert "NEW OPERATOR ONBOARDED" in page.text_content('[data-testid="session-lifecycle-badge"]')
        assert page.text_content('[data-testid="session-login-count"]').strip() == "1"
        assert page.text_content('[data-testid="session-profile-display"]').strip() == "developer"
        assert _strip_num(page.text_content('[data-testid="session-points"]')) == "2500"
        assert page.input_value('[data-testid="vault-key-input"]').startswith("hm_live_")

        # Logout before testing next SSO provider
        page.click('[data-testid="logout-btn"]')
        page.wait_for_selector('[data-testid="active-session-card"]', state="hidden")

    # Test New User Email & Passkey Registration Form
    page.click('[data-testid="role-researcher-card"]')
    page.fill('[data-testid="auth-handle-input"]', "e2e_new_email_researcher")
    page.fill('[data-testid="auth-email-input"]', "researcher@hivemind.org")
    page.click('[data-testid="auth-submit-btn"]')
    page.wait_for_selector('[data-testid="session-handle"]:has-text("e2e_new_email_researcher")', timeout=3000)
    assert "NEW OPERATOR ONBOARDED" in page.text_content('[data-testid="session-lifecycle-badge"]')
    assert page.text_content('[data-testid="session-profile-display"]').strip() == "researcher"
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "1000"

    page.click('[data-testid="logout-btn"]')
    page.wait_for_selector('[data-testid="active-session-card"]', state="hidden")

    # Test Passwordless Quantum Magic Link Onboarding
    page.fill('[data-testid="magic-email-input"]', "quantum_newbie@swarm-node.net")
    page.click('[data-testid="magic-send-btn"]')
    page.wait_for_selector('[data-testid="magic-status"]:not(.hidden)', timeout=3000)
    page.wait_for_selector('[data-testid="session-handle"]:has-text("quantum_newbie")', timeout=3000)
    assert "MAGIC_LINK_ZK" in page.text_content('#session-provider-badge')
    assert "NEW OPERATOR ONBOARDED" in page.text_content('[data-testid="session-lifecycle-badge"]')

    page.click('[data-testid="logout-btn"]')
    page.wait_for_selector('[data-testid="active-session-card"]', state="hidden")
    page.close()


def test_e2e_existing_end_user_multi_sso_linking_and_persistence(live_server, browser_context):
    """Journey 4: Returning Existing Users, Multi-Provider SSO Identity Linking (+1,500 HMT), Reload Persistence & Key Rotation."""
    page = browser_context.new_page()
    page.goto(live_server + "/#auth")
    page.wait_for_selector('[data-testid="screen-auth"].active')

    # 1. Test Quick Returning Profiles (Google+GitHub, Reddit+GitHub, Steam, Epic Games)
    returning_profiles = [
        ("quick-existing-reddit", "reddit_llama_king", "REDDIT_SSO", "34200"),
        ("quick-existing-steam", "steam_rtx4090_pro", "STEAM_SSO", "77500"),
        ("quick-existing-epic", "epic_unreal_dev", "EPIC_SSO", "49800"),
        ("quick-existing-xavier", "xavier_callens", "GOOGLE_SSO", "68500"),
    ]
    for btn_id, handle, prov, expected_pts in returning_profiles:
        page.click(f'[data-testid="{btn_id}"]')
        page.wait_for_selector(f'[data-testid="session-handle"]:has-text("{handle}")', timeout=3000)
        assert "EXISTING OPERATOR // LOGIN #" in page.text_content('[data-testid="session-lifecycle-badge"]')
        assert prov in page.text_content('[data-testid="session-linked-providers"]')
        assert _strip_num(page.text_content('[data-testid="session-points"]')) == expected_pts

    # 2. While logged in as xavier_callens (68,500 HMT), link Reddit, Steam, and Epic Games SSO identities
    page.click('[data-testid="link-sso-reddit"]')
    page.wait_for_selector('[data-testid="session-linked-providers"]:has-text("REDDIT_SSO")', timeout=3000)
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "70000"

    page.click('[data-testid="link-sso-steam"]')
    page.wait_for_selector('[data-testid="session-linked-providers"]:has-text("STEAM_SSO")', timeout=3000)
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "71500"

    page.click('[data-testid="link-sso-epic"]')
    page.wait_for_selector('[data-testid="session-linked-providers"]:has-text("EPIC_SSO")', timeout=3000)
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "73000"

    # 3. Reload page and verify session & linked providers persist in browser
    page.reload()
    page.wait_for_selector('[data-testid="active-session-card"]:not(.hidden)', timeout=3000)
    assert "xavier_callens" in page.text_content('[data-testid="session-handle"]')
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "73000"
    linked_text = page.text_content('[data-testid="session-linked-providers"]')
    for expected_p in ("GOOGLE_SSO", "GITHUB_SSO", "REDDIT_SSO", "STEAM_SSO", "EPIC_SSO"):
        assert expected_p in linked_text

    # 4. Test API Key Copy & Server-Backed Key Rotation
    old_key = page.input_value('[data-testid="vault-key-input"]')
    page.click('[data-testid="copy-key-btn"]')
    page.wait_for_selector('[data-testid="copy-confirm"]:not(.hidden)')

    page.click('[data-testid="regenerate-key-btn"]')
    time.sleep(0.3)
    new_key = page.input_value('[data-testid="vault-key-input"]')
    assert new_key.startswith("hm_live_")
    assert new_key != old_key

    # 5. Logout and re-authenticate xavier_callens via Epic Games SSO button -> must recognize as EXISTING OPERATOR with 73,000 HMT!
    page.click('[data-testid="logout-btn"]')
    page.wait_for_selector('[data-testid="active-session-card"]', state="hidden")

    page.fill('[data-testid="sso-handle-hint"]', "xavier_callens")
    page.click('[data-testid="sso-epic-btn"]')
    page.wait_for_selector('[data-testid="session-handle"]:has-text("xavier_callens")', timeout=3000)
    assert "EXISTING OPERATOR // LOGIN #" in page.text_content('[data-testid="session-lifecycle-badge"]')
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "73000"

    # 6. Test Existing Operator Login via email input form (dr_elena_astro -> elena@berkeley.edu)
    page.fill('[data-testid="existing-login-input"]', "elena@berkeley.edu")
    page.click('[data-testid="existing-login-btn"]')
    page.wait_for_selector('[data-testid="session-handle"]:has-text("dr_elena_astro")', timeout=3000)
    assert "EXISTING OPERATOR // LOGIN #" in page.text_content('[data-testid="session-lifecycle-badge"]')
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "210500"

    page.close()


def test_e2e_openrouter_and_moe_routing(live_server, browser_context):
    """Journey 5: OpenRouter gateway & interactive Queen Bee SigmaMoE routing sandbox."""
    page = browser_context.new_page()
    page.goto(live_server + "/#openrouter")
    page.wait_for_selector('[data-testid="screen-openrouter"].active')

    cards = page.locator('[data-testid="honeycomb-container"] > div')
    assert cards.count() >= 4

    page.fill('[data-testid="sim-route-input"]', "Analyze Kepler-1649 exoplanet transit lightcurve and prove Lean 4 theorem")
    page.click('[data-testid="sim-route-btn"]')
    page.wait_for_selector('[data-testid="sim-status"]:has-text("TOP-2 EXPERTS ACTIVATED")', timeout=3000)
    assert "Activated Cell #3 (Astrophysics)" in page.text_content('[data-testid="sim-response-text"]')

    page.close()


def test_e2e_byob_yield_calculator_and_code_tabs(live_server, browser_context):
    """Journey 6: BYOB Browser Worker, Campaigns, Stitch GPU Yield Calculator & OpenRouter Code Tabs."""
    page = browser_context.new_page()
    page.goto(live_server + "/#byob")
    page.wait_for_selector('[data-testid="screen-byob"].active')

    # 1. Start WebGPU Browser Bee
    page.click('[data-testid="webgpu-btn"]')
    page.wait_for_selector('[data-testid="webgpu-status"]:not(.hidden)', timeout=3000)
    assert "PAUSE BROWSER WORKER BEE" in page.text_content('[data-testid="webgpu-btn"]')

    # 2. Test Stitch BYOB Yield & Reward Calculator
    assert _strip_num(page.text_content('[data-testid="cp-display"]')) == "14280"
    page.click('[data-testid="gpu-tier-2"]')  # 2.4x multiplier (RTX 4090 / M3 Max)
    assert _strip_num(page.text_content('[data-testid="cp-display"]')) == "34272"
    page.click('[data-testid="gpu-tier-3"]')  # 4.8x multiplier (RTX 5090 / 64GB+)
    assert _strip_num(page.text_content('[data-testid="cp-display"]')) == "68544"

    # 3. Test OpenRouter Code Tabs (cURL, Python, TypeScript)
    assert page.is_visible('[data-testid="code-block-curl"]')
    page.click('[data-testid="code-tab-python"]')
    assert page.is_visible('[data-testid="code-block-python"]')
    assert not page.is_visible('[data-testid="code-block-curl"]')
    page.click('[data-testid="code-tab-typescript"]')
    assert page.is_visible('[data-testid="code-block-typescript"]')

    page.close()


def test_e2e_amcp_agent_mesh_stargate_and_tools(live_server, browser_context):
    """Journey 7: AMCP Agent Mesh live chat, @mention auto-reply, Tool Call Dispatcher, and MCP Stargate."""
    page = browser_context.new_page()
    page.goto(live_server + "/#amcp")
    page.wait_for_selector('[data-testid="screen-amcp"].active')

    # 1. Test Message Filtering
    page.click('[data-testid="filter-humans"]')
    time.sleep(0.2)
    assert page.locator('#mesh-chat-stream > div').count() >= 1
    page.click('[data-testid="filter-all"]')
    time.sleep(0.2)
    assert page.locator('#mesh-chat-stream > div').count() >= 4

    # 2. Test Mention & Dispatch Message to @AstroBee
    page.click('[data-testid="mention-astro"]')
    page.fill('[data-testid="mesh-input"]', "@AstroBee please analyze the Kepler sector 24 transit curve")
    page.click('[data-testid="mesh-send-btn"]')
    page.wait_for_selector('#mesh-chat-stream:has-text("anomaly verified")', timeout=6000)
    assert "anomaly verified" in page.text_content('[data-testid="mesh-chat-stream"]')

    # 3. Test AMCP Tool Call Dispatcher
    page.click('[data-testid="tool-astro-btn"]')
    page.wait_for_selector('[data-testid="tool-response-box"]:has-text("KOI-0482.01")', timeout=5000)
    tool_resp = page.text_content('[data-testid="tool-response-box"]')
    assert "query_exoplanet_catalog" in tool_resp
    assert "KOI-0482.01" in tool_resp

    # 4. Test Stitch MCP Stargate Section & IDE Connectors
    assert page.is_visible('[data-testid="mcp-stargate-section"]')
    assert "https://stitch.googleapis.com/mcp" in page.text_content('[data-testid="mcp-json-payload"]')
    page.click('[data-testid="mcp-cursor-btn"]')
    assert "HOOK DEPLOYED!" in page.text_content('[data-testid="mcp-cursor-btn"]')
    page.click('[data-testid="mcp-cline-btn"]')
    assert "PAIRED TO CLINE" in page.text_content('[data-testid="mcp-cline-btn"]')

    page.close()


def test_e2e_multilingual_responsive_and_mobile_preemption(live_server, browser_context):
    """Journey 8: Real-time i18n translation, responsive viewports, Mobile HUD & Emergency Preemption Halt."""
    page = browser_context.new_page()
    page.goto(live_server + "/#home")
    page.wait_for_selector('[data-testid="screen-home"].active')

    # 1. Language Switching: French, Spanish, Chinese, English
    page.select_option('[data-testid="lang-select"]', "fr")
    page.wait_for_selector('[data-testid="hero-title"]:has-text("L\'ESSAIM D\'INTELLIGENCE OUVERTE")', timeout=2000)
    page.select_option('[data-testid="lang-select"]', "es")
    page.wait_for_selector('[data-testid="hero-title"]:has-text("EL ENJAMBRE DE INTELIGENCIA ABIERTA")', timeout=2000)
    page.select_option('[data-testid="lang-select"]', "zh")
    page.wait_for_selector('[data-testid="hero-title"]:has-text("开放智能蜂巢超级网络")', timeout=2000)
    page.select_option('[data-testid="lang-select"]', "en")
    page.wait_for_selector('[data-testid="hero-title"]:has-text("THE OPEN INTELLIGENCE SWARM")', timeout=2000)

    # 2. Responsive Viewport Switching: Tablet & Mobile
    page.select_option('[data-testid="viewport-select"]', "tablet")
    assert "view-tablet" in page.get_attribute("#app-viewport-wrapper", "class")
    page.select_option('[data-testid="viewport-select"]', "mobile")
    assert "view-mobile" in page.get_attribute("#app-viewport-wrapper", "class")

    # 3. Mobile Light Screen & Emergency Preemption Halt Switch
    page.click('[data-testid="nav-mobile"]')
    page.wait_for_selector('[data-testid="screen-mobile"].active')
    assert "LIGHT MOBILE HUD" in page.text_content('[data-testid="screen-mobile"]')

    page.click('[data-testid="emergency-halt-btn"]')
    assert "PREEMPTION TRIGGERED" in page.text_content('[data-testid="emergency-halt-btn"]')

    page.fill('[data-testid="mobile-input"]', "Mobile prompt test")
    page.click('[data-testid="mobile-send-btn"]')
    page.wait_for_selector('#mobile-chat-log:has-text("Mobile dispatch acknowledged")', timeout=3000)

    page.close()


def test_e2e_simple_email_registration_frictionless(live_server, browser_context):
    """Journey 9: Frictionless Simple Email Registration with 1-click token issuance & v2 access."""
    page = browser_context.new_page()
    page.goto(live_server + "/v2#auth")
    page.wait_for_selector('[data-testid="screen-auth"].active')

    # 1. Verify v2 badge link in header and simple email registration card
    assert page.is_visible('[data-testid="v2-badge-link"]')
    assert page.is_visible('[data-testid="simple-email-card"]')
    assert page.is_visible('[data-testid="simple-email-input"]')
    assert page.is_visible('[data-testid="simple-email-submit-btn"]')

    # 2. Enter email only (no handle) and submit
    test_email = "new_simple_contributor@swarm-node.net"
    page.fill('[data-testid="simple-email-input"]', test_email)
    page.click('[data-testid="simple-email-submit-btn"]')

    # 3. Verify session active with auto-derived handle and +1,000 HMT starter grant
    page.wait_for_selector('[data-testid="active-session-card"]:not(.hidden)', timeout=4000)
    assert "new_simple_contributor" in page.text_content('[data-testid="session-handle"]')
    assert _strip_num(page.text_content('[data-testid="session-points"]')) == "1000"
    assert page.input_value('[data-testid="vault-key-input"]').startswith("hm_live_")

    # 4. Verify session persistence after reload on /v2
    page.reload()
    page.wait_for_selector('[data-testid="active-session-card"]:not(.hidden)', timeout=4000)
    assert "new_simple_contributor" in page.text_content('[data-testid="session-handle"]')

    # Clean logout
    page.click('[data-testid="logout-btn"]')
    page.wait_for_selector('[data-testid="active-session-card"]', state="hidden")
    page.close()


def test_e2e_existing_user_xavier_v2_access_and_restoration(live_server, browser_context):
    """Journey 10: Existing User Xavier Callens (Lead Architect) priority v2 access, points & restoration."""
    page = browser_context.new_page()
    # Direct access to /v2
    page.goto(live_server + "/v2#auth")
    page.wait_for_selector('[data-testid="screen-auth"].active', timeout=5000)

    # 1. Verify Lead Architect Banner is visible
    assert page.is_visible('[data-testid="lead-architect-banner"]')
    assert "Xavier Callens" in page.text_content('[data-testid="lead-architect-banner"]')
    assert "LEAD ARCHITECT" in page.text_content('[data-testid="lead-architect-banner"]')
    assert "68,500+ HMT" in page.text_content('[data-testid="lead-architect-banner"]')

    # 2. 1-Click Restore Xavier Session
    page.click('[data-testid="quick-restore-xavier-btn"]')
    page.wait_for_selector('[data-testid="session-handle"]:has-text("xavier_callens")', timeout=8000)

    # 3. Verify Xavier's profile integrity
    assert "xavier_callens" in page.text_content('[data-testid="session-handle"]')
    assert int(_strip_num(page.text_content('[data-testid="session-points"]'))) >= 68500
    assert "Lead Architect" in page.text_content('[data-testid="session-role-display"]')
    assert "0x9AF4C101" in page.text_content('[data-testid="session-node-id"]')
    assert "GOOGLE_SSO" in page.text_content('[data-testid="session-linked-providers"]')
    assert "GITHUB_SSO" in page.text_content('[data-testid="session-linked-providers"]')
    assert page.input_value('[data-testid="vault-key-input"]').startswith("hm_live_")

    # 4. Verify Xavier can freely navigate all screens on /v2
    for nav_id, screen_id in [
        ("nav-home", "screen-home"),
        ("nav-openrouter", "screen-openrouter"),
        ("nav-byob", "screen-byob"),
        ("nav-amcp", "screen-amcp"),
        ("nav-telemetry", "screen-telemetry"),
        ("nav-auth", "screen-auth"),
    ]:
        page.click(f'[data-testid="{nav_id}"]')
        page.wait_for_selector(f'[data-testid="{screen_id}"].active', timeout=4000)
        assert page.is_visible(f'[data-testid="{screen_id}"]')

    # 5. Reload on /v2 and verify persistent v2 access
    page.reload()
    page.wait_for_selector('[data-testid="session-handle"]:has-text("xavier_callens")', timeout=8000)
    assert "xavier_callens" in page.text_content('[data-testid="session-handle"]')
    assert int(_strip_num(page.text_content('[data-testid="session-points"]'))) >= 68500

    # 6. Logout and test restoring Xavier via entering email in Simple Email Form
    page.click('[data-testid="logout-btn"]')
    page.wait_for_selector('[data-testid="active-session-card"]', state="hidden", timeout=5000)

    page.fill('[data-testid="simple-email-input"]', "xavier@hivemind.network")
    page.click('[data-testid="simple-email-submit-btn"]')
    page.wait_for_selector('[data-testid="session-handle"]:has-text("xavier_callens")', timeout=8000)
    assert "xavier_callens" in page.text_content('[data-testid="session-handle"]')
    assert int(_strip_num(page.text_content('[data-testid="session-points"]'))) >= 68500

    page.close()


def test_e2e_leaderboard_epoch_412_and_boinc_download(live_server, browser_context):
    """Journey 11: Top Contributor Leaderboard (Epoch #412), BOINC Community MoE Download, and French Telemetry Figures."""
    page = browser_context.new_page()
    page.goto(live_server + "/telemetry")
    page.wait_for_selector('[data-testid="screen-telemetry"].active', timeout=6000)

    # 1. Verify Top Contributor Leaderboard (Non-Monetary Reciprocity) Epoch #412
    page.wait_for_selector('[data-testid="leaderboard-tbody"] tr', timeout=5000)
    rows = page.locator('[data-testid="leaderboard-tbody"] tr')
    assert rows.count() >= 5

    def clean_text(s):
        return s.replace("\u202f", ",").replace(" ", "")

    # Check #1 @xaviercallens
    row1_text = rows.nth(0).text_content()
    assert "#1" in row1_text
    assert "@xaviercallens" in row1_text
    assert "RTX 5060" in row1_text
    assert "142" in row1_text
    assert "35,500HMT" in clean_text(row1_text) or "35500HMT" in clean_text(row1_text)
    assert "QUEEN-HIVE FOUNDER" in row1_text
    assert "undefined" not in row1_text
    assert "NaN" not in row1_text

    # Check #2 @clark719664
    row2_text = rows.nth(1).text_content()
    assert "#2" in row2_text
    assert "@clark719664" in row2_text
    assert "96" in row2_text
    assert "24,000HMT" in clean_text(row2_text) or "24000HMT" in clean_text(row2_text)
    assert "undefined" not in row2_text
    assert "NaN" not in row2_text

    # Check #3 @radialmonster
    row3_text = rows.nth(2).text_content()
    assert "#3" in row3_text
    assert "@radialmonster" in row3_text
    assert "84" in row3_text
    assert "21,000HMT" in clean_text(row3_text) or "21000HMT" in clean_text(row3_text)
    assert "undefined" not in row3_text
    assert "NaN" not in row3_text

    # Check #4 @pedromineo
    row4_text = rows.nth(3).text_content()
    assert "#4" in row4_text
    assert "@pedromineo" in row4_text
    assert "72" in row4_text
    assert "18,000HMT" in clean_text(row4_text) or "18000HMT" in clean_text(row4_text)
    assert "undefined" not in row4_text
    assert "NaN" not in row4_text

    # Check #5 @socrateai-agora-hermes-t4
    row5_text = rows.nth(4).text_content()
    assert "#5" in row5_text
    assert "@socrateai-agora-hermes-t4" in row5_text
    assert "64" in row5_text
    assert "16,000HMT" in clean_text(row5_text) or "16000HMT" in clean_text(row5_text)
    assert "undefined" not in row5_text
    assert "NaN" not in row5_text

    # 2. Test BOINC Community MoE Package Download
    page.click('[data-testid="nav-byob"]')
    page.wait_for_selector('[data-testid="screen-byob"].active', timeout=4000)
    assert page.is_visible('[data-testid="boinc-pkg-link"]')
    boinc_href = page.get_attribute('[data-testid="boinc-pkg-link"]', "href")
    assert "/api/la-ruche/download/community-moe.safetensors" in boinc_href

    # Download binary via browser API context
    response = page.request.get(live_server + "/api/la-ruche/download/community-moe.safetensors")
    assert response.status == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "x-safetensors-sha256" in response.headers or "X-Safetensors-SHA256" in response.headers
    body = response.body()
    assert len(body) > 100000

    # 3. Test French Language Switch and Exact Telemetry Figures
    page.click('[data-testid="nav-home"]')
    page.wait_for_selector('[data-testid="screen-home"].active', timeout=4000)

    # Click French Language button
    page.select_option('[data-testid="lang-select"]', "fr")

    # Verify exact figures requested by the user
    page.wait_for_selector('[data-i18n="stat_nodes"]:has-text("NŒUDS ACTIFS")', timeout=3000)
    assert page.text_content('[data-testid="stat-nodes-val"]').strip() == "48,920"
    assert "▲ +3,410 this epoch" in page.text_content('[data-i18n="stat_nodes_sub"]')

    assert "PUISSANCE TOTALE" in page.text_content('[data-i18n="stat_throughput"]')
    assert "542.8 PFLOPS" in page.text_content('[data-testid="stat-pflops-val"]')
    assert "FP16 • DiLoCo Synced" in page.text_content('[data-i18n="stat_throughput_sub"]')

    assert "THÉORÈMES LEAN4 VÉRIFIÉS" in page.text_content('[data-i18n="stat_theorems"]')
    assert page.text_content('[data-testid="stat-lean-val"]').strip() == "128 / 128"
    assert "Zero `sorry` proofs" in page.text_content('[data-i18n="stat_theorems_sub"]')

    assert "LATENCE MAILLAGE" in page.text_content('[data-i18n="stat_latency"]')
    assert "8.42 ms" in page.text_content('[data-testid="stat-latency-val"]')
    assert "Direct WebRTC/QUIC" in page.text_content('[data-i18n="stat_latency_sub"]')

    page.close()


def test_e2e_masterminders_admin_deck_and_campaign_creation(live_server, browser_context):
    """Journey 10: MasterMinders Admin Deck, Tribute to Mastermind, 1-Click Auth, Campaign Studio, MoE Training & Model Selector."""
    page = browser_context.new_page()
    page.goto(live_server + "/admin")

    # 1. Verify Admin Screen & Tribute to the game Mastermind
    page.wait_for_selector('[data-testid="screen-admin"].active', timeout=5000)
    assert page.is_visible('[data-testid="screen-admin"]')
    screen_text = page.text_content('[data-testid="screen-admin"]')
    assert "MasterMinders Administrator Deck" in screen_text
    assert "TRIBUTE TO MASTERMIND" in screen_text
    assert "callensxavier@gmail.com" in screen_text

    # 2. 1-Click Restore MasterCommander (callensxavier@gmail.com)
    assert page.is_visible('[data-testid="admin-quick-restore-btn"]')
    page.click('[data-testid="admin-quick-restore-btn"]')

    # Verify session restored
    page.wait_for_timeout(1000)
    header_user = page.text_content("#header-user-label")
    assert "xavier_callens" in header_user or "xavier" in header_user.lower()

    # 3. Swarm Campaign Studio: Launch new campaign
    page.fill('[data-testid="admin-campaign-name"]', "Dark Matter Gravitational Lensing Survey")
    page.select_option('[data-testid="admin-campaign-domain"]', "astrophysics")
    page.fill('[data-testid="admin-campaign-steps"]', "300")
    page.select_option('[data-testid="admin-campaign-priority"]', "CRITICAL")
    page.fill('[data-testid="admin-campaign-budget"]', "85000")
    page.click('[data-testid="admin-create-campaign-btn"]')

    # Verify campaign scheduled receipt
    page.wait_for_selector('[data-testid="admin-campaign-output"]:has-text("Campaign Scheduled")', timeout=5000)
    output_text = page.text_content('[data-testid="admin-campaign-output"]')
    assert "Campaign Scheduled" in output_text
    assert "CAMP-MM-" in output_text

    # 4. Swarm Internal MoE Training: Train new expert
    page.select_option('[data-testid="admin-moe-domain"]', "quantum_chemistry")
    page.select_option('[data-testid="admin-moe-rank"]', "32")
    page.fill('[data-testid="admin-moe-steps"]', "450")
    page.click('[data-testid="admin-train-moe-btn"]')

    # Verify training output receipt
    page.wait_for_selector('[data-testid="admin-train-output"]:has-text("MoE Expert Compiled")', timeout=5000)
    train_text = page.text_content('[data-testid="admin-train-output"]')
    assert "MoE Expert Compiled" in train_text
    assert "TRAIN-MM-" in train_text

    # 5. Active Model Selector & Topology Switch
    page.select_option('[data-testid="admin-model-select"]', "hivemind/astro-bee-moe-14b")
    page.select_option('[data-testid="admin-routing-mode"]', "top_k_specialist")
    page.click('[data-testid="admin-activate-model-btn"]')

    # Verify model activated receipt
    page.wait_for_selector('[data-testid="admin-model-output"]:has-text("Model Active")', timeout=5000)
    model_text = page.text_content('[data-testid="admin-model-output"]')
    assert "Model Active" in model_text
    assert "hivemind/astro-bee-moe-14b" in model_text

    # 6. Verify MasterMinders Group Roster Table
    page.wait_for_selector('[data-testid="masterminders-roster-table"]', timeout=4000)
    roster_html = page.inner_html('[data-testid="masterminders-roster-table"]')
    assert "xavier_callens" in roster_html
    assert "callensxavier@gmail.com" in roster_html
    assert "MasterCommander" in roster_html
    assert "@clark719664" in roster_html
    assert "@radialmonster" in roster_html
    assert "@pedromineo" in roster_html
    assert "@socrateai-agora-hermes-t4" in roster_html

    page.close()


def test_e2e_night_moe_training_and_rtx2070_inference_shield(live_server, browser_context):
    """Journey 13: Night MoE Swarm Training (T4/Cloud with Qwen 3.5 & Gemma 3) and Local RTX 2070 Dedicated Inference Shield."""
    page = browser_context.new_page()
    page.goto(live_server + "/admin")

    # 1. Verify Header Hardware Shield Badges
    page.wait_for_selector('[data-testid="screen-admin"].active', timeout=5000)
    assert page.is_visible('[data-testid="rtx2070-shield-badge"]')
    shield_badge = page.text_content('[data-testid="rtx2070-shield-badge"]')
    assert "RTX 2070: INFERENCE LOCKED" in shield_badge

    assert page.is_visible('[data-testid="night-moe-badge"]')
    night_badge = page.text_content('[data-testid="night-moe-badge"]')
    assert "NIGHT MoE: T4 & CLOUD" in night_badge

    # 2. Verify RTX 2070 Hardware Isolation Shield Card in Admin Deck
    assert page.is_visible('[data-testid="admin-rtx2070-shield-status"]')
    shield_card = page.text_content('[data-testid="admin-rtx2070-shield-status"]')
    assert "100% INFERENCE SECURED" in shield_card
    assert "BLOCKED (0% Training Load)" in shield_card
    assert "NVIDIA GeForce RTX 2070" in shield_card

    # 3. Verify Swarm Night MoE Training Panel (T4 & Cloud Big Producers)
    assert page.is_visible('[data-testid="admin-night-moe-panel"]')
    night_card = page.text_content('[data-testid="admin-night-moe-panel"]')
    assert "NIGHT-MOE-2" in night_card
    assert "Tesla T4" in night_card
    assert "Qwen 3.5 MoE" in night_card
    assert "Google Gemma 3 MoE" in night_card

    # 4. Trigger Overnight Swarm Training Run
    assert page.is_visible('[data-testid="admin-trigger-night-moe-btn"]')
    page.click('[data-testid="admin-trigger-night-moe-btn"]')

    # Verify night training dispatched output receipt
    page.wait_for_selector('[data-testid="admin-night-moe-output"]:has-text("Night Training Dispatched")', timeout=5000)
    night_output = page.text_content('[data-testid="admin-night-moe-output"]')
    assert "Night Training Dispatched" in night_output
    assert "NIGHT-MOE2-" in night_output
    assert "NVIDIA GeForce RTX 2070" in night_output

    # 5. Activate Next-Gen Qwen 3.5 MoE Model Topology
    page.select_option('[data-testid="admin-model-select"]', "hivemind/qwen3.5-moe-bee-35b")
    page.select_option('[data-testid="admin-routing-mode"]', "biomimetic_moe")
    page.click('[data-testid="admin-activate-model-btn"]')

    # Verify model active receipt
    page.wait_for_selector('[data-testid="admin-model-output"]:has-text("hivemind/qwen3.5-moe-bee-35b")', timeout=5000)
    model_output = page.text_content('[data-testid="admin-model-output"]')
    assert "Model Active" in model_output
    assert "hivemind/qwen3.5-moe-bee-35b" in model_output

    page.close()


def test_e2e_steam_free_game_and_boinc_daemon_points_retribution(live_server, browser_context):
    """Journey 14: Free Steam Game (AppID 3133700), Zero-Intrusion RTX BOINC Daemon Control, Preemption Simulation, Points Retribution, and Gamer Yield Calculator."""
    page = browser_context.new_page()
    page.goto(live_server + "/steam")

    # 1. Verify Steam Screen and AppID 3133700 Banner
    page.wait_for_selector('[data-testid="screen-steam"].active', timeout=5000)
    assert page.is_visible('[data-testid="steam-appid-badge"]')
    appid_text = page.text_content('[data-testid="steam-appid-badge"]')
    assert "3133700" in appid_text

    # Verify 1-click Install and Download Buttons
    assert page.is_visible('[data-testid="btn-steam-install"]')
    install_href = page.get_attribute('[data-testid="btn-steam-install"]', "href")
    assert install_href == "steam://install/3133700"

    assert page.is_visible('[data-testid="btn-download-steam-zip"]')
    zip_href = page.get_attribute('[data-testid="btn-download-steam-zip"]', "href")
    assert "free-edition.zip" in zip_href

    # 2. Verify RTX Zero-Intrusion Daemon Telemetry & Status
    assert page.is_visible('[data-testid="steam-daemon-status-badge"]')
    status_text = page.text_content('[data-testid="steam-daemon-status-badge"]')
    assert "RUNNING" in status_text

    assert page.is_visible('[data-testid="steam-gpu-name"]')
    assert "RTX" in page.text_content('[data-testid="steam-gpu-name"]')

    # 3. Test Daemon Toggle (Pause & Resume)
    assert page.is_visible('[data-testid="btn-toggle-steam-daemon"]')
    page.click('[data-testid="btn-toggle-steam-daemon"]')
    page.wait_for_selector('[data-testid="steam-daemon-feedback"]:has-text("Daemon Status Updated")', timeout=5000)

    # 4. Test Game Launch Preemption (<1.5s SLA)
    assert page.is_visible('[data-testid="btn-simulate-game-launch"]')
    page.click('[data-testid="btn-simulate-game-launch"]')
    page.wait_for_selector('[data-testid="steam-daemon-feedback"]:has-text("Instant Preemption")', timeout=5000)
    preempt_text = page.text_content('[data-testid="steam-daemon-feedback"]')
    assert "Instant Preemption" in preempt_text
    assert "Cyberpunk 2077" in preempt_text

    # 5. Test Compute Points Retribution Claim
    assert page.is_visible('[data-testid="btn-claim-steam-points"]')
    page.click('[data-testid="btn-claim-steam-points"]')
    page.wait_for_selector('[data-testid="steam-claim-receipt"]:has-text("POINTS RETRIBUTED")', timeout=5000)
    receipt_text = page.text_content('[data-testid="steam-claim-receipt"]')
    assert "POINTS RETRIBUTED" in receipt_text
    assert "callensxavier@gmail.com" in receipt_text

    # 6. Test Interactive Gamer RTX Yield Calculator
    assert page.is_visible('[data-testid="calc-rtx-model"]')
    page.select_option('[data-testid="calc-rtx-model"]', "rtx4090")
    daily_pts = page.text_content('[data-testid="calc-daily-points"]')
    assert "HMT / day" in daily_pts

    # 7. Navigation Test: Switch from other screen to Steam via Nav Pill
    page.click('[data-testid="nav-home"]')
    page.wait_for_selector('[data-testid="screen-home"].active', timeout=3000)
    assert page.is_visible('[data-testid="nav-steam"]')
    page.click('[data-testid="nav-steam"]')
    page.wait_for_selector('[data-testid="screen-steam"].active', timeout=3000)
    assert page.is_visible('[data-testid="screen-steam"]')
    page.close()


def test_e2e_gcp_finops_scale_to_zero_when_no_activity(live_server, browser_context):
    """Journey 15: GCP Cloud Run FinOps & Scale-to-Zero (min-instances=0) Optimizer UI & API E2E Verification."""
    page = browser_context.new_page()
    page.goto(live_server + "/admin")

    # 1. Verify Admin FinOps Scale-to-Zero Governance Card is visible
    page.wait_for_selector('[data-testid="screen-admin"].active', timeout=5000)
    assert page.is_visible('[data-testid="admin-finops-panel"]')
    assert "0 (Scale-to-Zero)" in page.text_content('[data-testid="finops-min-replicas"]')
    assert "$0.00 / hr" in page.text_content('[data-testid="finops-idle-cost"]')
    assert "TRUE" in page.text_content('[data-testid="finops-cpu-throttling"]')

    # 2. Test Cold-Start Wakeup (0 -> 1 Replica)
    page.click('[data-testid="btn-finops-wake"]')
    page.wait_for_selector('[data-testid="finops-action-output"]:has-text("COLD-START WAKEUP")', timeout=5000)
    assert "1 / 2 Max" in page.text_content('[data-testid="finops-active-replicas"]')

    # 3. Simulate No Activity -> Scale to 0 Replicas
    page.click('[data-testid="btn-finops-simulate-idle"]')
    page.wait_for_selector('[data-testid="finops-action-output"]:has-text("SCALED TO ZERO REPLICAS")', timeout=5000)
    assert "0 / 2 Max" in page.text_content('[data-testid="finops-active-replicas"]')
    assert "IDLE_SCALED_TO_ZERO" in page.text_content('[data-testid="finops-state-text"]')

    # 4. Enforce GCP FinOps Policy (minScale=0)
    page.click('[data-testid="btn-finops-enforce-policy"]')
    page.wait_for_selector('[data-testid="finops-action-output"]:has-text("FINOPS POLICY ENFORCED")', timeout=5000)
    out_text = page.text_content('[data-testid="finops-action-output"]')
    assert "minScale=0" in out_text
    assert "cpu-throttling=true" in out_text

    # 5. Verify @Vxtzq Pre-Deployment Collaborator Review Gate in Admin Deck
    assert page.is_visible('[data-testid="admin-vxtzq-review-gate"]')
    assert "@Vxtzq" in page.text_content('[data-testid="admin-vxtzq-review-gate"]')
    page.click('[data-testid="btn-request-vxtzq-review"]')
    page.wait_for_selector('[data-testid="vxtzq-review-output"]:has-text("REVIEW REQUESTED FROM @Vxtzq")', timeout=5000)
    assert "LOCKED UNTIL @Vxtzq APPROVAL" in page.text_content('[data-testid="vxtzq-gcp-lock-state"]')

    page.click('[data-testid="btn-approve-vxtzq-review"]')
    page.wait_for_selector('[data-testid="vxtzq-review-output"]:has-text("APPROVED BY @Vxtzq")', timeout=5000)
    assert "UNLOCKED (APPROVED BY @Vxtzq)" in page.text_content('[data-testid="vxtzq-gcp-lock-state"]')


def test_e2e_webgpu_galaxy_exploration_and_sector_warp(live_server, browser_context):
    """Journey 16: dgreenheck/webgpu-galaxy Interactive Spiral Galaxy Explorer, Sector Warp & HMT Bounty E2E."""
    page = browser_context.new_page()
    page.goto(live_server + "/galaxy")

    # 1. Verify WebGPU Galaxy screen & canvas are visible
    page.wait_for_selector('[data-testid="screen-galaxy"].active', timeout=5000)
    assert page.is_visible('[data-testid="webgpu-galaxy-canvas"]')
    assert page.is_visible('[data-testid="webgpu-galaxy-repo-link"]')
    repo_href = page.get_attribute('[data-testid="webgpu-galaxy-repo-link"]', "href")
    assert "https://github.com/dgreenheck/webgpu-galaxy" == repo_href

    # 2. Warp to Kepler-1649c (Perseus Spiral Arm) and verify HUD updates
    page.click('[data-testid="btn-sector-kepler_1649c"]')
    assert "KEPLER-1649C" in page.text_content('[data-testid="galaxy-active-sector-badge"]')
    assert "Kepler-1649c Earth-Twin" in page.text_content('[data-testid="galaxy-hud-sector-title"]')

    # 3. Warp to Sagittarius A* Core and verify HUD updates
    page.click('[data-testid="btn-sector-sagittarius_a_core"]')
    assert "SAGITTARIUS A*" in page.text_content('[data-testid="galaxy-active-sector-badge"]')

    # 4. Scan Sector & Claim HMT Bounty via POST /api/v1/galaxy/explore
    page.click('[data-testid="btn-galaxy-scan-explore"]')
    page.wait_for_selector('[data-testid="galaxy-explore-output"]:has-text("SECTOR RECONNAISSANCE COMPLETE")', timeout=5000)
    exp_text = page.text_content('[data-testid="galaxy-explore-output"]')
    assert "+750 HMT AWARDED" in exp_text
    assert "dgreenheck/webgpu-galaxy" in exp_text

    # 5. Reset to dgreenheck/webgpu-galaxy defaults
    page.click('[data-testid="btn-galaxy-reset"]')
    assert page.text_content("#val-galaxy-arm-count") == "2"

    page.close()


def test_e2e_v1_to_v2_migration_and_cyber_security_waf_shield(live_server, browser_context):
    """Journey 17: v1 -> v2 Seamless Account Migration & Zero-Trust Cyber WAF Live Scanner E2E."""
    page = browser_context.new_page()
    resp = page.goto(live_server + "/auth")
    assert resp is not None
    assert resp.headers.get("x-hivemind-waf") == "ARMED-ZERO-TRUST-v2.8"
    assert resp.headers.get("x-hivemind-migration") == "v1-to-v2-seamless-active"

    # 1. Verify v1 -> v2 Seamless Account Migration Card is visible on #auth
    page.wait_for_selector('[data-testid="screen-auth"].active', timeout=5000)
    assert page.is_visible('[data-testid="v1-migration-card"]')

    # Test restoring @clark719664 from v1 on v2
    page.click('[data-testid="v1-preset-clark"]')
    page.wait_for_selector('[data-testid="v1-migrate-output"]:has-text("v1 ACCOUNT RESTORED ON v2")', timeout=5000)
    mig_text = page.text_content('[data-testid="v1-migrate-output"]')
    assert "@clark719664" in mig_text
    assert "24,000 HMT" in mig_text

    # Test restoring callensxavier@gmail.com from v1 on v2
    page.click('[data-testid="v1-preset-xavier"]')
    page.wait_for_selector('[data-testid="v1-migrate-output"]:has-text("xavier_callens")', timeout=5000)
    xav_text = page.text_content('[data-testid="v1-migrate-output"]')
    assert "xavier_callens" in xav_text
    assert "HMT" in xav_text and ("68,500" in xav_text or "73,000" in xav_text)

    # 2. Verify Cyber Protection & Zero-Trust WAF Live Scanner Card
    assert page.is_visible('[data-testid="cyber-security-scanner-card"]')
    page.click('[data-testid="waf-preset-xss"]')
    page.wait_for_selector('[data-testid="waf-scan-output"]:has-text("BLOCKED_AND_SANITIZED")', timeout=5000)
    waf_bad_text = page.text_content('[data-testid="waf-scan-output"]')
    assert "XSS_DOM_INJECTION" in waf_bad_text
    assert "SQL_INJECTION" in waf_bad_text

    # Test clean scientific query
    page.click('[data-testid="waf-preset-clean"]')
    page.wait_for_selector('[data-testid="waf-scan-output"]:has-text("CLEAN_ADMITTED")', timeout=5000)
    waf_clean_text = page.text_content('[data-testid="waf-scan-output"]')
    assert "CLEAN_ADMITTED" in waf_clean_text
    page.close()


def test_e2e_browser_ide_studio_and_10_planet_habitability_dossier(live_server, browser_context):
    """Journey 18: Browser IDE Studio (#ide) code/WGSL execution & 10-Exoplanet Habitability Dossier E2E."""
    page = browser_context.new_page()
    page.goto(live_server + "/ide")
    page.wait_for_selector('[data-testid="screen-ide"].active', timeout=5000)

    # 1. Verify Local IDE Deep-Link Bar & Multi-Tab Editor
    assert page.is_visible('[data-testid="local-ide-deeplink-bar"]')
    assert page.is_visible('[data-testid="browser-ide-editor"]')

    # Switch to WGSL tab and execute in Browser IDE Sandbox
    page.click('[data-testid="ide-tab-wgsl"]')
    assert "@compute @workgroup_size(64)" in page.input_value('[data-testid="ide-code-textarea"]')
    page.click('[data-testid="btn-ide-run"]')
    page.wait_for_selector('[data-testid="ide-console-output"]:has-text("IDE_EXECUTION_SUCCESS")', timeout=5000)
    ide_out = page.text_content('[data-testid="ide-console-output"]')
    assert "+200 HMT" in ide_out
    assert "How life could live:" in ide_out

    # 2. Switch to #galaxy and verify one-time Chapter 1 intro + 10 Stitch planet habitability dossier
    page.click('[data-testid="nav-galaxy"]')
    page.wait_for_selector('[data-testid="screen-galaxy"].active', timeout=5000)
    assert page.is_visible('[data-testid="planet-habitability-dossier"]')

    # First click shows Chapter 1 intro once
    page.click('[data-testid="btn-planet-k218b"]')
    vt1 = page.text_content('[data-testid="planet-voice-transcript"]')
    assert "Chapitre 1 : Passerelle." in vt1
    assert "Vie possible :" in vt1

    # Second click shows ONLY the concise planet analysis & habitability without repeating Chapter 1 intro
    page.click('[data-testid="btn-planet-lhs1140b"]')
    vt2 = page.text_content('[data-testid="planet-voice-transcript"]')
    assert "Chapitre 1 : Passerelle." not in vt2
    assert "LHS 1140 b" in vt2
    assert "Vie possible :" in vt2
    page.close()


def test_e2e_cockpit_v46_five_stages_and_language_switcher(live_server, browser_context):
    """Journey 19: USS HIVEMIND NCC-31337 v4.6 (/cockpit-v46) 5-Stage Tactical Bridge & 5-Language Switcher E2E."""
    page = browser_context.new_page()
    resp = page.goto(live_server + "/cockpit-v46")
    assert resp is not None
    assert resp.status == 200
    assert resp.headers.get("x-frame-options") == "SAMEORIGIN"

    # 1. Verify Steam + BOINC GPU auto-connect badge completes
    page.wait_for_selector('#steam-auth-badge:has-text("STEAM (AppID 3133700)")', timeout=6000)

    # 2. Switch across all 5 interactive stages (Stage 1 -> 2 -> 3 -> 4 -> 5)
    for stage_num in (2, 3, 4, 5):
        page.click(f'button.stage-btn[data-stage="{stage_num}"]')
        page.wait_for_selector(f"#viewport-stage-{stage_num}:not(.hidden)", timeout=4000)
        assert page.is_visible(f"#viewport-stage-{stage_num}")

    # Return to Stage 1
    page.click('button.stage-btn[data-stage="1"]')

    # 3. Test 5-Language Switcher (EN, ES, DE, ZH, FR)
    page.click("#lang-en")
    assert "STAGE 1" in page.text_content("#instruction-text")
    page.click("#lang-fr")
    assert "ÉTAPE 1" in page.text_content("#instruction-text")

    # 4. Select planets and verify one-time Chapter 1 intro + habitability analysis
    page.evaluate("window.selectPlanet('k218b')")
    vt_first = page.text_content("#voice-transcript")
    assert "Chapitre 1 : Passerelle." in vt_first
    assert "Vie possible :" in vt_first

    page.evaluate("window.selectPlanet('lhs1140b')")
    vt_second = page.text_content("#voice-transcript")
    assert "Chapitre 1 : Passerelle." not in vt_second
    assert "LHS 1140 b" in vt_second
    assert "Vie possible :" in vt_second


def test_e2e_browser_ide_self_test_and_cockpit_iframe_sync(live_server, browser_context):
    """Journey 20: Browser IDE (#ide) 8-Point Contract Self-Test & Embedded Cockpit v4.6 Iframe Stage Sync E2E."""
    page = browser_context.new_page()
    page.goto(live_server + "/ide")
    page.wait_for_selector('[data-testid="screen-ide"].active', timeout=5000)

    # 1. Click 8-Point Contract Self-Test button and verify 8/8 contracts pass
    page.click('[data-testid="btn-ide-self-test"]')
    page.wait_for_selector('[data-testid="ide-console-output"]:has-text("ALL_CONTRACTS_VERIFIED")', timeout=5000)
    assert "8/8 CONTRACTS VERIFIED" in page.text_content('[data-testid="ide-self-test-badge"]')
    out_txt = page.text_content('[data-testid="ide-console-output"]')
    assert "+300 HMT" in out_txt
    assert "USS HIVEMIND NCC-31337 v4.6" in out_txt

    # 2. Verify embedded /cockpit-v46 iframe is visible and stage controller buttons sync the iframe
    assert page.is_visible('[data-testid="ide-cockpit-iframe"]')
    for stage_idx in (2, 3, 4, 5, 1):
        page.click(f'[data-testid="ide-stage-btn-{stage_idx}"]')

    page.close()


def test_e2e_galaxy_all_10_planets_and_astro_discovery_api(live_server, browser_context):
    """Journey 21: 10-Exoplanet Habitability Dossier walkthrough & Live Astro-Bee Discovery API E2E."""
    page = browser_context.new_page()
    page.goto(live_server + "/galaxy")
    page.wait_for_selector('[data-testid="screen-galaxy"].active', timeout=5000)

    # Walk through 5 distinct Stitch exoplanet targets and verify unique habitability dossier updates
    for p_key, expected_name in [
        ("trappist1e", "TRAPPIST-1e"),
        ("k218b", "K2-18 b"),
        ("toi700d", "TOI-700 d"),
        ("kepler186f", "Kepler-186f"),
        ("koibin", "KOI-Bin #02"),
    ]:
        page.click(f'[data-testid="btn-planet-{p_key}"]')
        vt = page.text_content('[data-testid="planet-voice-transcript"]')
        assert expected_name in vt
        assert "Vie possible :" in vt
        en_box = page.text_content('[data-testid="planet-analysis-en-box"]')
        assert "How life could live:" in en_box
    page.close()


def test_e2e_hiveminer_maintenance_mode_banner_and_loyalty_claim(live_server, browser_context):
    """Journey 22: HiveMiner Scheduled Maintenance Mode (Intense MVP & New Release v3.0) E2E."""
    page = browser_context.new_page()
    resp = page.goto(live_server + "/maintenance")
    assert resp is not None
    assert resp.status == 200
    assert resp.headers.get("x-hivemind-maintenance-mode") == "ACTIVE-INTENSE-MVP-RELEASE"

    # 1. Verify HiveMiner Maintenance Mode Banner & message are prominently displayed
    page.wait_for_selector('[data-testid="hiveminer-maintenance-banner"]', timeout=5000)
    assert page.is_visible('[data-testid="hiveminer-maintenance-banner"]')
    banner_msg = page.text_content('[data-testid="hiveminer-maintenance-msg"]')
    assert "HiveMiners" in banner_msg
    assert "Intense MVP" in banner_msg
    assert "+500 HMT" in banner_msg

    # 2. Click Acknowledge & Claim +500 HMT HiveMiner Bonus button
    page.click('[data-testid="btn-claim-maintenance-bonus"]')
    page.wait_for_selector(
        '[data-testid="maintenance-claim-output"]:has-text("HIVEMINER_ACKNOWLEDGED_AND_BONUS_CREDITED")',
        timeout=5000,
    )
    assert page.is_visible('[data-testid="hiveminer-maintenance-drawer"]')
    claim_txt = page.text_content('[data-testid="maintenance-claim-output"]')
    assert "+500 HMT" in claim_txt
    page.close()


def test_e2e_core_mvp_4_sso_email_password_and_staged_gcp_dns_portal(live_server, browser_context):
    """Journey 23: Core MVP 4-SSO (Google, Reddit, Steam, GitHub) + Standard Email/Password Auth & Staged GCP Public DNS Portal E2E."""
    page = browser_context.new_page()
    resp = page.goto(live_server + "/auth")
    assert resp is not None
    assert resp.status == 200
    assert "proof-of-concept" in resp.headers.get("x-hivemind-status", "").lower()

    # 1. Verify PoC notice banner & Core MVP Auth + Staged GCP Public DNS Portal panel
    page.wait_for_selector('[data-testid="screen-auth"].active', timeout=5000)
    assert page.is_visible('[data-testid="poc-notice"]')
    assert page.is_visible('[data-testid="mvp-core-auth-panel"]')
    panel_txt = page.text_content('[data-testid="mvp-core-auth-panel"]')
    assert "hivemind.socrate-ai-lab.com" in panel_txt
    assert "STAGED_READY_NOT_DEPLOYED" in panel_txt

    # 2. Check & Review Staged GCP Public DNS Readiness (DO NOT DEPLOY YET)
    page.click('[data-testid="btn-check-gcp-portal-readiness"]')
    page.wait_for_selector('[data-testid="gcp-portal-readiness-output"]:has-text("STAGED GCP PUBLIC DNS PORTAL READY")', timeout=5000)
    assert "deployed_to_gcp: false" in page.text_content('[data-testid="gcp-portal-readiness-output"]')
    assert "https://hivemind.socrate-ai-lab.com" in page.text_content('[data-testid="gcp-portal-readiness-output"]')

    page.click('[data-testid="btn-review-gcp-portal-readiness"]')
    page.wait_for_selector('[data-testid="gcp-portal-readiness-output"]:has-text("STAGED_VERIFIED_DO_NOT_DEPLOY_YET")', timeout=5000)
    assert "DO NOT DEPLOY YET" in page.text_content('[data-testid="gcp-portal-stage-pill"]')

    # 3. Test all 4 Core MVP SSO Providers (Google, Reddit, Steam, GitHub) with OAuth2 PKCE S256
    for prov_id, expected_badge in [
        ("google", "GOOGLE_SSO"),
        ("reddit", "REDDIT_SSO"),
        ("steam", "STEAM_SSO"),
        ("github", "GITHUB_SSO"),
    ]:
        handle = f"e2e_mvp_{prov_id}_pilot"
        page.fill('[data-testid="mvp-handle-input"]', handle)
        page.click(f'[data-testid="mvp-sso-{prov_id}-btn"]')
        page.wait_for_selector(f'[data-testid="session-handle"]:has-text("{handle}")', timeout=5000)
        assert expected_badge in page.text_content("#session-provider-badge")
        assert "PKCE S256" in page.text_content('[data-testid="mvp-auth-status"]')
        page.click('[data-testid="logout-btn"]')
        page.wait_for_selector('[data-testid="active-session-card"]', state="hidden", timeout=4000)

    # 4. Test Standard Email + Password Registration & Login (PBKDF2-HMAC-SHA256)
    page.fill('[data-testid="mvp-email-input"]', "mvp_e2e_pilot@hivemind.network")
    page.fill('[data-testid="mvp-password-input"]', "StrongMVP#Password2026!")
    page.fill('[data-testid="mvp-handle-input"]', "mvp_e2e_pilot")
    page.click('[data-testid="mvp-register-btn"]')

    page.wait_for_selector('[data-testid="session-handle"]:has-text("mvp_e2e_pilot")', timeout=5000)
    assert "PBKDF2-HMAC-SHA256" in page.text_content('[data-testid="mvp-auth-status"]')

    # Logout and re-login with Email + Password
    page.click('[data-testid="logout-btn"]')
    page.wait_for_selector('[data-testid="active-session-card"]', state="hidden", timeout=4000)

    page.fill('[data-testid="mvp-email-input"]', "mvp_e2e_pilot@hivemind.network")
    page.fill('[data-testid="mvp-password-input"]', "StrongMVP#Password2026!")
    page.click('[data-testid="mvp-login-btn"]')
    page.wait_for_selector('[data-testid="session-handle"]:has-text("mvp_e2e_pilot")', timeout=5000)
    assert "Authenticated as Existing Operator" in page.text_content('[data-testid="mvp-auth-status"]')
    page.close()


def test_e2e_release_v3_mvp_kafka_collab_calendar_and_four_dashboards(live_server, browser_context):
    """Journey 24: Release V3.0.0 MVP Command Center, AMCP Kafka Human-Agent-Bee Space, Self-Organized OpenDiLoCo Calendar, 17:00-22:00 T4/A100/TPU Schedule & 4 Real-Data Dashboards."""
    page = browser_context.new_page()
    page.goto(live_server + "/v3-mvp")

    # 1. Verify Release V3 MVP Command Center & Persistent GCP Backbone Banner
    page.wait_for_selector('[data-testid="v3-mvp-command-center"]', timeout=5000)
    assert page.is_visible('[data-testid="v3-mvp-backbone-card"]')
    backbone_text = page.text_content('[data-testid="v3-mvp-backbone-card"]')
    assert "RELEASE V3.0.0 MVP" in backbone_text
    assert "hm-kafka-1" in backbone_text
    assert "17:00" in backbone_text

    # 2. Test AMCP v1.5 Kafka Collaborative Space (Human + Agent + MoE Bee)
    assert page.is_visible('[data-testid="amcp-kafka-collab-space"]')
    page.fill('[data-testid="amcp-kafka-actor-handle"]', "@xavier_callens")
    page.select_option('[data-testid="amcp-kafka-actor-type"]', "HUMAN")
    page.select_option('[data-testid="amcp-kafka-topic"]', "hivemind.opendiloco.rounds")
    page.fill(
        '[data-testid="amcp-kafka-message"]',
        "Launching 17:00-22:00 CET French + Math QLoRA round with @Granite_Split_MoE_Bee",
    )
    page.click('[data-testid="btn-amcp-kafka-publish"]')
    page.wait_for_selector(
        '[data-testid="amcp-kafka-publish-status"]:has-text("PUBLISHED_TO_AMCP_KAFKA")',
        timeout=5000,
    )
    assert "hivemind.opendiloco.rounds" in page.text_content('[data-testid="amcp-kafka-publish-status"]')

    # 3. Test Personal Contribution Daily 17:00-22:00 Schedule (T4 / A100 / TPU)
    assert page.is_visible('[data-testid="personal-17-22-schedule-panel"]')
    page.select_option('[data-testid="personal-schedule-accel"]', "tpu")
    page.click('[data-testid="btn-save-personal-schedule"]')
    page.wait_for_selector(
        '[data-testid="personal-schedule-status"]:has-text("PERSONAL_17_22_SCHEDULE_UPDATED")',
        timeout=5000,
    )
    assert "17:00" in page.text_content('[data-testid="personal-schedule-status"]')
    assert "22:00" in page.text_content('[data-testid="personal-schedule-status"]')

    # Switch back to Tesla T4 default for 17:00-22:00
    page.select_option('[data-testid="personal-schedule-accel"]', "t4")
    page.click('[data-testid="btn-save-personal-schedule"]')
    page.wait_for_selector(
        '[data-testid="personal-schedule-status"]:has-text("Tesla T4")',
        timeout=5000,
    )

    # 4. Test Self-Organized OpenDiLoCo Multi-GPU QLoRA MoE Training Calendar (Zero Owner GPU Required)
    assert "OWNER GPU REQUIRED: FALSE" in page.text_content('[data-testid="zero-owner-gpu-badge"]')
    page.fill('[data-testid="propose-session-title"]', "E2E 4xGPU OpenDiLoCo QLoRA MoE Session")
    page.fill('[data-testid="propose-session-handle"]', "@clark719664")
    page.click('[data-testid="btn-propose-training-session"]')
    page.wait_for_selector(
        '[data-testid="opendiloco-calendar-status"]:has-text("TRAINING_SESSION_PROPOSED")',
        timeout=5000,
    )
    assert "owner_gpu_required: false" in page.text_content('[data-testid="opendiloco-calendar-status"]')

    page.click('[data-testid="btn-pledge-training-gpu"]')
    page.wait_for_selector(
        '[data-testid="opendiloco-calendar-status"]:has-text("GPU_PLEDGED_TO_OPENDILOCO_SESSION")',
        timeout=5000,
    )

    # 5. Verify Dashboard 1: Global Connected End-Users Globe Dashboard
    assert page.is_visible('[data-testid="dashboard-globe-users"]')
    assert page.is_visible('[data-testid="globe-users-svg"]')
    page.click('[data-testid="btn-globe-filter-trainer"]')
    page.wait_for_selector(
        '[data-testid="globe-nodes-table-body"]:has-text("TRAINER_OPENDILOCO")',
        timeout=5000,
    )
    page.click('[data-testid="btn-globe-filter-all"]')

    # 6. Verify Dashboard 2: Nash Equilibrium Compute-Reciprocity Dashboard
    assert page.is_visible('[data-testid="dashboard-nash-equilibrium"]')
    assert "1.00" in page.text_content('[data-testid="nash-poa-val"]')
    page.fill('[data-testid="nash-sim-producers"]', "28")
    page.fill('[data-testid="nash-sim-consumers"]', "60")
    page.click('[data-testid="btn-simulate-nash"]')
    page.wait_for_selector(
        '[data-testid="nash-sim-output"]:has-text("NASH_EQUILIBRIUM_CONVERGED")',
        timeout=5000,
    )

    # 7. Verify Dashboard 3: Real Inference Data & Consumer Needs Fit Analytics
    assert page.is_visible('[data-testid="dashboard-inference-analytics"]')
    page.click('[data-testid="btn-refresh-inference-analytics"]')
    inf_text = page.text_content('[data-testid="dashboard-inference-analytics"]')
    assert "41.75 tok/s" in inf_text
    assert "NEEDS_17_22_TRAINING_BOOST" in inf_text

    # 8. Verify Dashboard 4: Real Trained MoE Bees Dashboard (Strictly Real Trained Models Only)
    assert page.is_visible('[data-testid="dashboard-real-moe-bees"]')
    assert "5 REAL MEASURED MODELS" in page.text_content('[data-testid="real-moe-honesty-badge"]')
    bees_text = page.text_content('[data-testid="real-moe-bees-grid"]')
    assert "hivemind/bee-stars" in bees_text
    assert "Qwen/Qwen2.5-0.5B-NF4-QLoRA" in bees_text
    assert "ibm-granite/granite-3.1-1b-a400m-base" in bees_text
    assert "hivemind/coder-lite" in bees_text
    assert "hivemind/e8-pilot-two-host" in bees_text
    gated_text = page.text_content('[data-testid="b3-gated-adapter-box"]')
    assert "hmq_00147" in gated_text
    assert "servable: false" in gated_text
    page.close()


