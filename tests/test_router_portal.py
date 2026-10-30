"""Tests for the OpenRouter portal, playground and landing page at ``/`` and ``/playground``."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.portal_page import portal_content_security_policy, render_portal_page


def test_portal_content_security_policy_structure() -> None:
    csp = portal_content_security_policy("test-nonce-12345")
    assert "default-src 'none'" in csp
    assert "script-src 'nonce-test-nonce-12345'" in csp
    assert "style-src 'nonce-test-nonce-12345'" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_render_portal_page_without_client_id() -> None:
    html = render_portal_page(None, "ephemeral-nonce-abc")
    assert "HiveMind OpenRouter" in html
    assert "Live AI Completion Console" in html
    assert "5 Community Capability Tiers" in html
    assert "qwen2.5-coder:7b" in html
    assert "ephemeral-nonce-abc" in html


def test_portal_and_playground_endpoints(tmp_path: pytest.TempPathFactory) -> None:
    store = RouterStore(tmp_path / "router.db", b"x" * 32)
    app = create_router_app(store)
    client = TestClient(app)

    # GET /
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "HiveMind OpenRouter" in res_root.text
    assert "script-src 'nonce-" in res_root.headers.get("content-security-policy", "")
    assert res_root.headers.get("x-content-type-options") == "nosniff"

    # HEAD /
    res_head = client.head("/")
    assert res_head.status_code == 200
    assert "text/html" in res_head.headers.get("content-type", "")

    # GET /playground
    res_play = client.get("/playground")
    assert res_play.status_code == 200
    assert "text/html" in res_play.headers.get("content-type", "")
    assert "Live AI Completion Console" in res_play.text

    # HEAD /signup
    res_signup_head = client.head("/signup")
    assert res_signup_head.status_code == 200

    # HEAD /health and /healthz
    assert client.head("/health").status_code == 200
    assert client.head("/healthz").status_code == 200


def test_portal_snippets_cover_the_documented_editors_and_keep_the_output_limit() -> None:
    html = render_portal_page(None, "n")
    for tab in ("opencode", "continue", "cline", "zed", "jetbrains", "aider", "env", "python", "curl"):
        assert f'data-tab="{tab}"' in html, tab  # every button has a snippet below
        assert f"  {tab}: `" in html, tab
    # the OpenCode snippet must carry the output limit, or a small balance is refused (observed on the live gateway)
    opencode = html.split("  opencode: `", 1)[1].split("`,", 1)[0]
    assert '"limit": { "context": 8192, "output": 700 }' in opencode
    # Continue now shows the YAML format the docs use, not the retired JSON "models" list
    assert "roles: [chat, edit, apply]" in html and '"title": "HiveMind Coder (7B)"' not in html
    # only tiers that can be served today are shown as ready snippets
    assert "hivemind/coder-plus" not in html.split("// Model ID:", 1)[1].split("\n", 1)[0]


def test_portal_page_renders_overnight_boinc_qlora_telemetry_and_caveats() -> None:
    html = render_portal_page(None, "nonce-night")
    assert 'id="night-telemetry"' in html
    assert "100 Assimilated WUs (98 hmq_*)" in html
    assert "5,632.11 Cobblestones" in html
    assert "2.2705 &rarr; 2.1380 (-5.8%)" in html
    assert "hmq_00092" in html
    assert "Plateau after ~30 rounds" in html
    assert "scripts/t4_training_report.py" in html


@pytest.mark.parametrize(
    "lang,expected_heading,expected_points_name",
    [
        ("en", "Community AI Inference.", "HiveMind Reflexion Points (HMT)"),
        ("fr", "Inférence IA Communautaire.", "Points de Réflexion HiveMind (HMT)"),
        ("es", "Inferencia de IA Comunitaria.", "Puntos de Reflexión HiveMind (HMT)"),
        ("de", "Community-KI-Inferenz.", "HiveMind-Reflexionspunkte (HMT)"),
        ("zh", "社区驱动的 AI 推理网络。", "HiveMind 反思积分 (HMT)"),
    ],
)
def test_portal_page_multilingual_rendering_and_zero_currency_leftovers(
    lang: str, expected_heading: str, expected_points_name: str
) -> None:
    from hivemind.router.points_wording import currency_leftovers

    html = render_portal_page(None, "nonce-i18n", lang=lang)
    assert f'<html lang="{lang}">' in html
    assert expected_heading in html
    assert expected_points_name in html
    assert 'id="lang-switcher"' in html
    for code in ("en", "fr", "es", "de", "zh"):
        assert f'data-lang="{code}"' in html
        assert f'hreflang="{code}"' in html
    assert currency_leftovers(html) == []


def test_gateway_multilingual_negotiation_routes_and_i18n_endpoint(tmp_path: pytest.TempPathFactory) -> None:
    store = RouterStore(tmp_path / "router.db", b"x" * 32)
    app = create_router_app(store)
    client = TestClient(app)

    # 1. Query string ?lang=fr|es|de|zh|en on / and /playground
    r_fr = client.get("/?lang=fr")
    assert r_fr.status_code == 200
    assert r_fr.headers.get("content-language") == "fr"
    assert r_fr.headers.get("vary") == "Accept-Language"
    assert '<html lang="fr">' in r_fr.text
    assert "Console de Complétion IA en Direct" in r_fr.text

    r_es = client.get("/playground?lang=es")
    assert r_es.status_code == 200 and r_es.headers.get("content-language") == "es"
    assert "Consola de Completado IA en Vivo" in r_es.text

    r_de = client.get("/?lang=de")
    assert r_de.status_code == 200 and r_de.headers.get("content-language") == "de"
    assert "Live-KI-Completion-Konsole" in r_de.text

    r_zh = client.get("/?lang=zh")
    assert r_zh.status_code == 200 and r_zh.headers.get("content-language") == "zh"
    assert "实时 AI 推理控制台" in r_zh.text

    # 2. Accept-Language header negotiation on /
    r_accept_fr = client.get("/", headers={"Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"})
    assert r_accept_fr.status_code == 200 and r_accept_fr.headers.get("content-language") == "fr"
    assert '<html lang="fr">' in r_accept_fr.text

    # 3. Dedicated language routes (/fr, /aire-de-jeu, /es, /de, /zh, /inscription)
    for path, expected_lang in (
        ("/fr", "fr"),
        ("/aire-de-jeu", "fr"),
        ("/es", "es"),
        ("/de", "de"),
        ("/zh", "zh"),
        ("/inscription", "fr"),
    ):
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.headers.get("content-language") == expected_lang
        assert f'<html lang="{expected_lang}">' in resp.text

    # 4. Public /v1/hivemind/i18n JSON catalog
    r_i18n = client.get("/v1/hivemind/i18n")
    assert r_i18n.status_code == 200
    payload = r_i18n.json()
    assert payload["object"] == "hivemind.i18n_catalog"
    assert payload["supported_languages"] == ["en", "fr", "es", "de", "zh"]
    for code in ("en", "fr", "es", "de", "zh"):
        assert code in payload["portal"]
        assert code in payload["signup"]
        assert code in payload["official_points_names"]


def test_router_overlay_multilingual_routes(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    from pathlib import Path

    orig_bids = RouterStore.funded_consumer_bids
    monkeypatch.setattr(RouterStore, "funded_consumer_bids", orig_bids)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("HIVEMIND_ROUTER_DB", str(tmp_path / "overlay_router.db"))
    monkeypatch.setenv("HIVEMIND_ROUTER_SECRET", "x" * 64)
    monkeypatch.setenv("HIVEMIND_ORIG_SERVER", str(root / "deploy" / "gcp_openrouter" / "server.py"))

    ns = runpy.run_path(str(root / "deploy" / "router_overlay" / "server.py"), run_name="test_overlay")
    monkeypatch.setattr(RouterStore, "funded_consumer_bids", orig_bids)
    overlay_app = ns["app"]
    client = TestClient(overlay_app)

    for path, expected_lang in (
        ("/", "en"),
        ("/?lang=fr", "fr"),
        ("/fr", "fr"),
        ("/aire-de-jeu", "fr"),
        ("/es", "es"),
        ("/de", "de"),
        ("/zh", "zh"),
        ("/signup?lang=es", "es"),
        ("/inscription", "fr"),
        ("/join?lang=de", "de"),
        ("/rejoindre", "fr"),
        ("/webgpu", "fr"),
        ("/webgpu?lang=en", "en"),
    ):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers.get("content-language") == expected_lang, path
        assert f'<html lang="{expected_lang}">' in r.text, path

    r_i18n = client.get("/v1/hivemind/i18n?lang=fr")
    assert r_i18n.status_code == 200
    assert "fr" in r_i18n.json()["portal"]


def test_webgpu_route_works_without_torch_installed(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "hivemind.game" or mod_name.startswith("hivemind.game."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "rich", None)

    store = RouterStore(tmp_path / "router.db", b"x" * 32)
    app = create_router_app(store)
    client = TestClient(app)

    r = client.get("/webgpu?lang=en")
    assert r.status_code == 200
    assert '<html lang="en">' in r.text
    assert "bls_fold_main" in r.text

    r_status = client.get("/v1/hivemind/webgpu/status")
    assert r_status.status_code == 200
    assert r_status.json()["wgsl_entrypoint"] == "bls_fold_main"




