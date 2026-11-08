"""v3.3 portal in a real browser (Chromium via Playwright). Skipped when Playwright is not installed; run by the
web-auth intense workflow, which also installs axe-core and sets AXE_JS for the accessibility scan.

Checks: no CSP violation or console error, live facts rendered, French toggle, keyboard-operable tabs, sign-in shows the
real balance from the account endpoint, a password change shows the new key once, and axe-core finds no serious issue.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright")
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from hivemind.web_v32 import mount_web_v32  # noqa: E402
from tests.test_web_v32 import SECRET, FakeGateway  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def portal_url():
    port = _free_port()
    origin = f"http://localhost:{port}"
    app = FastAPI()
    mount_web_v32(app, env={"HIVEMIND_WEB_SESSION_SECRET": SECRET, "HIVEMIND_PUBLIC_WEB_URL": origin}, gateway=FakeGateway())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield origin
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def test_portal_works_in_a_real_browser_without_csp_violations(portal_url: str, browser) -> None:
    page = browser.new_page()
    problems: list[str] = []
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(str(e)))
    page.goto(f"{portal_url}/v32", wait_until="networkidle")

    page.wait_for_selector("#facts-list .value")
    assert page.locator("#facts-list .value").first.inner_text() == "78"

    page.click("#lang-toggle")
    page.wait_for_function("document.documentElement.lang === 'fr'")
    assert page.locator("#tab-login").inner_text() == "Connexion"
    page.click("#lang-toggle")
    page.wait_for_function("document.documentElement.lang === 'en'")

    page.focus("#tab-login")
    page.keyboard.press("ArrowRight")
    assert page.locator("#tab-register").get_attribute("aria-selected") == "true"
    assert page.locator("#form-register").is_visible() and not page.locator("#form-login").is_visible()
    page.keyboard.press("ArrowLeft")
    assert page.locator("#tab-login").get_attribute("aria-selected") == "true"

    page.fill("#form-login input[name=identifier]", "alice@example.org")
    page.fill("#form-login input[name=password]", "AlicePassword#2026")
    page.click("#form-login button[type=submit]")
    page.wait_for_selector("#signed-in:not([hidden])")
    page.wait_for_selector("#balance-line:not([hidden])")
    assert "1,234.5" in page.locator("#balance").inner_text()
    assert page.locator("#usage-rows tr").count() == 1

    page.fill("#form-change input[name=new_password]", "AliceNewPassword#2026")
    page.click("#form-change button[type=submit]")
    page.wait_for_selector("#keybox:not([hidden])")
    assert page.locator("#key-once").inner_text().startswith("hm-")

    page.click("#logout")
    page.wait_for_selector("#signed-out:not([hidden])")
    csp_or_errors = [p for p in problems if "Content Security Policy" in p or "Refused" in p or "Error" in p]
    assert not csp_or_errors, csp_or_errors
    page.close()


def test_axe_core_finds_no_serious_accessibility_issue(portal_url: str, browser) -> None:
    axe_path = os.environ.get("AXE_JS", "")
    if not axe_path or not Path(axe_path).is_file():
        pytest.skip("AXE_JS not set (the intense workflow installs axe-core)")
    context = browser.new_context(bypass_csp=True)  # axe is injected by the test, not served by the site
    page = context.new_page()
    page.goto(f"{portal_url}/v32", wait_until="networkidle")
    page.wait_for_selector("#facts-list .value")
    page.add_script_tag(path=axe_path)
    result = page.evaluate("async () => await axe.run(document, {resultTypes: ['violations']})")
    serious = [
        {"id": v["id"], "impact": v["impact"], "nodes": len(v["nodes"])}
        for v in result["violations"] if v.get("impact") in ("serious", "critical")
    ]
    assert not serious, json.dumps(serious, indent=1)
    context.close()
