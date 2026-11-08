"""v3.3 portal: French/English completeness, language choice, error codes, and the account panel."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hivemind.web_v32.app import ERRORS
from tests.test_web_v32 import _client, _csrf

STATIC = Path(__file__).resolve().parents[1] / "src" / "hivemind" / "web_v32" / "static"
I18N = json.loads((STATIC / "i18n.json").read_text(encoding="utf-8"))
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
JS = (STATIC / "app.js").read_text(encoding="utf-8")


def test_english_and_french_have_the_same_keys_and_no_empty_string() -> None:
    assert set(I18N) == {"en", "fr"}
    assert set(I18N["en"]) == set(I18N["fr"]), set(I18N["en"]) ^ set(I18N["fr"])
    for lang, table in I18N.items():
        for key, text in table.items():
            assert isinstance(text, str) and text.strip(), (lang, key)


def test_every_key_used_by_the_page_and_the_script_exists() -> None:
    used = set(re.findall(r'data-i18n="([a-z0-9_]+)"', HTML))
    used |= set(re.findall(r'\bt\("([a-z0-9_]+)"\s*[,)]', JS))  # whole literal keys; prefixes like t("err_" + code) are dynamic
    used |= set(re.findall(r'(?:selectTab|errorText)\([^)]*"(err_[a-z0-9_]+)"', JS))
    missing = sorted(k for k in used if k not in I18N["en"])
    assert not missing, missing


def test_every_error_code_is_translated() -> None:
    for code in list(ERRORS) + ["wrong_host"]:
        assert f"err_{code}" in I18N["en"] and f"err_{code}" in I18N["fr"], code


def test_every_fact_and_statement_has_translations() -> None:
    client, _, _ = _client()
    data = client.get("/v32/api/facts").json()
    for f in data["facts"]:
        assert f"fact_{f['key']}" in I18N["fr"], f["key"]
        if f["note"]:
            assert f"fact_{f['key']}_note" in I18N["fr"], f["key"]
    for s in data["statements"]:
        assert f"stmt_{s['key']}" in I18N["fr"], s["key"]


def test_french_text_makes_no_claim_the_english_does_not() -> None:
    banned = ("démo", "demo", "pflops", "zk-snark", "steam", "reddit", "garanti")
    for text in I18N["fr"].values():
        assert not any(b in text.lower() for b in banned), text


@pytest.mark.parametrize("headers,query,cookie,expected", [
    ({"Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"}, "", None, "fr"),
    ({"Accept-Language": "de-DE,en;q=0.5"}, "", None, "en"),
    ({"Accept-Language": "fr-FR"}, "?lang=en", None, "en"),
    ({}, "?lang=fr", None, "fr"),
    ({}, "", "fr", "fr"),
    ({}, "?lang=xx", None, "en"),
])
def test_page_language_follows_query_cookie_then_browser(headers, query, cookie, expected) -> None:
    client, _, _ = _client()
    if cookie:
        client.cookies.set("hm_lang", cookie, domain="hivemind.example.org")
    page = client.get("/v32" + query, headers=headers).text
    assert f'<html lang="{expected}"' in page


def test_i18n_asset_is_served_with_strict_headers() -> None:
    client, _, _ = _client()
    r = client.get("/v32/static/i18n.json")
    assert r.status_code == 200 and r.json()["fr"]["tab_login"] == "Connexion"
    assert "default-src 'none'" in r.headers["content-security-policy"]


def test_errors_carry_a_stable_code() -> None:
    client, _, _ = _client()
    h = _csrf(client)
    r = client.post("/v32/api/login", json={"identifier": "nobody@example.org", "password": "wrong-password-1"}, headers=h)
    assert r.json() == {"error": ERRORS["bad_login"], "code": "bad_login"}
    r = client.post("/v32/api/login", json={}, headers={"Origin": "https://hivemind.example.org"})
    assert r.json()["code"] == "csrf"


def test_account_panel_needs_a_session_and_shows_real_balance_and_usage() -> None:
    client, _, _ = _client()
    assert client.get("/v32/api/account").json()["code"] == "sign_in_first"
    h = _csrf(client)
    client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h)
    data = client.get("/v32/api/account").json()
    assert data["balance"] == 1234.5 and data["source"].endswith("/v1/hivemind/me")
    assert data["usage"][0] == {"utc": 1.8e9, "model": "qwen2.5-coder:7b", "tokens": 321, "charge": 0.42, "as": "consumer"}
    assert "hm-" not in json.dumps(data)


def test_account_panel_reports_an_expired_key() -> None:
    client, gw, _ = _client()
    h = _csrf(client)
    client.post("/v32/api/login", json={"identifier": "alice@example.org", "password": "AlicePassword#2026"}, headers=h)
    gw.accounts["alice@example.org"]["key"] = "hm-replaced-elsewhere-000000"
    assert client.get("/v32/api/account").json()["code"] == "session_expired"


def test_accessibility_basics_are_in_the_markup() -> None:
    assert 'class="skip" href="#main"' in HTML and 'id="main" tabindex="-1"' in HTML
    assert HTML.count('role="tab"') == 3 and HTML.count('role="tabpanel"') == 3
    for tab in re.findall(r'<button role="tab"[^>]*>', HTML):
        assert "aria-controls=" in tab and "aria-selected=" in tab
    assert 'role="status" aria-live="polite"' in HTML
    assert "ArrowRight" in JS and "ArrowLeft" in JS, "tabs are keyboard operable"
