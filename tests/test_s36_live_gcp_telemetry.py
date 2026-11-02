# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit & integration tests for unified real-life GCP Web UI telemetry across all pages."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_router_live_telemetry_endpoint_and_all_web_pages(tmp_path) -> None:
    """Verify /v1/hivemind/live-telemetry and all router Web UI pages render real-life telemetry."""
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    store = RouterStore(tmp_path / "accounts.sqlite3", b"k" * 32)
    store.register("pioneer_prosumer", "prosumer")
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    # 1. Verify unified live-telemetry JSON endpoint
    for ep in ("/v1/hivemind/live-telemetry", "/v1/live-telemetry"):
        resp = client.get(ep)
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"
        payload = resp.json()
        assert payload["status"] == "LIVE_GCP_PRODUCTION"
        assert payload["accounts"]["total_registered_accounts"] >= 1
        assert "by_role" in payload["accounts"]
        assert "trainer" in payload["accounts"]["by_role"]
        assert payload["models"]["live_count"] >= 1
        assert "producer_fleet" in payload
        assert "tesla_t4_boinc" in payload
        assert "opentrainer" in payload
        assert payload["opentrainer"]["coordinator"] == "opentrainer"

    # 2. Verify all router Web UI HTML pages include real-life GCP telemetry
    page_expectations = {
        "/": "live-gcp-telemetry",
        "/portal": "live-gcp-telemetry",
        "/playground": "live-gcp-telemetry",
        "/life-as-a-bee": "live-gcp-telemetry",
        "/benchmarks": "benchmarks-live-telemetry",
        "/comparatif": "benchmarks-live-telemetry",
        "/signup": "signupLiveTelemetry",
        "/login": "signupLiveTelemetry",
        "/register": "signupLiveTelemetry",
        "/connexion": "signupLiveTelemetry",
        "/join": "join-live-telemetry",
        "/rejoindre": "join-live-telemetry",
        "/webgpu": "webgpu-live-gcp-telemetry",
        "/game": "webgpu-live-gcp-telemetry",
        "/manifesto": "manifesto-live-telemetry",
        "/manifeste": "manifesto-live-telemetry",
    }
    for path, marker in page_expectations.items():
        r = client.get(path)
        assert r.status_code == 200, f"Failed GET {path}: {r.status_code}"
        assert marker in r.text, f"Missing marker {marker} in {path}"
        assert (
            "/v1/hivemind/live-telemetry" in r.text or "join-live-telemetry" in r.text
        ), f"Missing live-telemetry integration in {path}"


def test_ruche_live_telemetry_endpoint_and_all_web_pages() -> None:
    """Verify /v1/hivemind/live-telemetry and all hivemind-at-home-ruche Web UI pages."""
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    app = create_hivemind_at_home_app()
    client = TestClient(app)

    from hivemind.router.catalog import catalog_payload

    declared_tiers = len([t for t in catalog_payload()["tiers"] if t.get("id")])
    for ep in ("/v1/hivemind/live-telemetry", "/v1/live-telemetry", "/api/v1/live-telemetry"):
        resp = client.get(ep)
        assert resp.status_code == 200
        data = resp.json()
        # IP-148: the ruche holds no router ledger, usage log or T4 feed, so it must not call itself live production
        # telemetry nor carry a figure for them; only what this process measured is a number.
        assert data["status"] != "LIVE_GCP_PRODUCTION" and "NOT_ROUTER_TELEMETRY" in data["status"]
        assert data["accounts"]["total_registered_accounts"] is None and data["accounts"]["by_role"] == {}
        assert data["models"]["live_count"] == declared_tiers and "catalog" in data["models"]["data_source"]
        assert data["opentrainer"]["status"] is None and data["opentrainer"]["latest_eval_loss"] is None
        t4 = data["tesla_t4_boinc"]
        assert t4["assimilated_workunits"] is None and t4["total_cobblestones"] is None and t4["vram_free_mib"] is None
        assert t4["completed_receipts_count"] == 0 and t4["active_leases_count"] == 0
        for section in ("accounts", "producer_fleet", "usage_24h", "tesla_t4_boinc", "opentrainer"):
            assert "NOT_MEASURED_ON_RUCHE" in data[section]["data_source"], section
        assert "5632.11" not in resp.text and "161250" not in resp.text and "1.7772" not in resp.text

    ruche_pages = [
        "/",
        "/v1",
        "/v2",
        "/community",
        "/boinc",
        "/night-moe",
        "/planet-forecast",
        "/foundation-paper",
        "/join",
        "/rejoindre",
        "/webgpu",
        "/cockpit-v46",
        "/manifesto",
        "/manifeste",
    ]
    for path in ruche_pages:
        r = client.get(path)
        assert r.status_code == 200, f"Failed GET {path} on ruche: {r.status_code}"
        assert (
            "/v1/hivemind/live-telemetry" in r.text or "join-live-telemetry" in r.text
        ), f"Missing live telemetry integration on ruche page {path}"


def test_15_moe_catalog_and_balanced_producer_prosumer_distribution(tmp_path) -> None:
    """Verify all 15 proposed MoE models are online and load-balanced across producers & prosumers."""
    import hashlib
    import httpx
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    key_secret = b"m" * 32
    admin_token = "adm-test-key"
    store = RouterStore(tmp_path / "accounts_moe.sqlite3", key_secret)
    p1, _ = store.register("tibrib", "prosumer")
    p2, _ = store.register("rtx-2070-producer-9351", "producer")
    p3, _ = store.register("cybervalkyrie-fr-5ecb", "producer")
    p4, _ = store.register("xavier_callens", "prosumer")
    p5, _ = store.register("gcp-t4-producer", "producer")
    p6, _ = store.register("t4-big-gpu-moe-20261028", "producer")
    consumer, consumer_key = store.register("test_consumer", "consumer")
    store.grant(consumer, 5000.0, "test_topup")

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test-15moe",
                "object": "chat.completion",
                "created": 1730000000,
                "model": "qwen2.5-coder:7b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(upstream_handler))
    app = create_router_app(
        store,
        http_client=http_client,
        allow_private_producers=True,
        admin_key_hash=store.key_digest(admin_token),
        resolver=lambda host: ["34.48.48.154"],
    )
    client = TestClient(app)

    # Reconnect all producers & prosumers via admin endpoint
    rec = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"x-admin-key": admin_token},
        json={
            "replace_endpoints": True,
            "include_legacy_known_users": True,
            "endpoint_token": "test-endpoint-token-0123456789abcdefghijk",
            # The test declares the catalog it expects; the handler no longer assumes one.
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "bee-stars", "local_name": "hivemind-bee-stars", "loaded": True, "vram_mib": 0}, {"id": "crowd-v1", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "coder-lite", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "gemma-core", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "mistral-core", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "qwen35-core", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "bee-rusty-sundials", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "bee-lean-prover", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "bee-ruche-moe", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "coder-plus", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "moe-qwen35-bee", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "moe-general", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}, {"id": "moe-coder", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 0}],
        },
    )
    assert rec.status_code == 200
    rec_data = rec.json()
    assert rec_data["reconnected_count"] >= 6

    # Check /v1/models lists all 15 proposed MoE models
    expected_15_models = {
        "hivemind/auto",
        "hivemind/bee-lean-prover",
        "hivemind/bee-ruche-moe",
        "hivemind/bee-rusty-sundials",
        "hivemind/bee-stars",
        "hivemind/coder",
        "hivemind/coder-lite",
        "hivemind/coder-plus",
        "hivemind/crowd-v1",
        "hivemind/gemma-core",
        "hivemind/mistral-core",
        "hivemind/moe-coder",
        "hivemind/moe-general",
        "hivemind/moe-qwen35-bee",
        "hivemind/qwen35-core",
    }
    models_resp = client.get("/v1/models")
    assert models_resp.status_code == 200
    live_ids = {m["id"] for m in models_resp.json()["data"]}
    assert expected_15_models.issubset(live_ids), f"Missing models: {expected_15_models - live_ids}"

    # Verify /v1/chat/completions across all 15 models distributes load across multiple producers AND prosumers
    chosen_accounts: list[str] = []
    chosen_roles: set[str] = set()
    role_by_id = {
        p1: "prosumer",
        p2: "producer",
        p3: "producer",
        p4: "prosumer",
        p5: "producer",
        p6: "producer",
    }
    for model_id in sorted(expected_15_models):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {consumer_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": f"Validate MoE tier {model_id}"}],
                "max_tokens": 16,
            },
        )
        assert r.status_code == 200, f"Failed {model_id}: {r.status_code} {r.text}"
        body = r.json()
        producer_id = body["hivemind"]["producer_id"]
        chosen_accounts.append(producer_id)
        chosen_roles.add(role_by_id.get(producer_id, "unknown"))

    assert len(set(chosen_accounts)) >= 4, f"Expected load balanced across >=4 nodes, got {set(chosen_accounts)}"
    assert "producer" in chosen_roles and "prosumer" in chosen_roles, f"Expected both roles, got {chosen_roles}"

    # Verify usage_capture_summary includes by_producer and by_producer_role
    telem = client.get("/v1/hivemind/live-telemetry").json()
    usage = telem["usage_capture"]
    assert usage["totals"]["requests"] == 15
    assert len(usage["by_producer"]) >= 4
    roles_served = set(usage["by_producer_role"].keys())
    assert "producer" in roles_served and "prosumer" in roles_served


def test_manifesto_page_and_gcp_portal_links(tmp_path) -> None:
    """Verify /manifesto, /manifeste, /v1/hivemind/manifesto, and cross-links to the GCP Portal."""
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    store = RouterStore(tmp_path / "accounts_manifesto.sqlite3", b"z" * 32)
    store.register("reddit_pioneer", "prosumer")
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    # 1. Verify /manifesto (EN & FR) and /manifeste
    for route, lang_code, expected_phrase in (
        ("/manifesto?lang=fr", "fr", "Manifeste de l'IA Citoyenne"),
        ("/manifesto?lang=en", "en", "Citizen AI Manifesto"),
        ("/manifeste", "fr", "Manifeste de l'IA Citoyenne"),
    ):
        resp = client.get(route)
        assert resp.status_code == 200
        assert resp.headers.get("content-language") == lang_code
        assert expected_phrase in resp.text
        # Must link directly to the GCP Portal
        assert "https://hivemind-openrouter-1003063861791.us-central1.run.app/portal" in resp.text
        assert "/v1/hivemind/live-telemetry" in resp.text
        assert "HM-REDDIT-01" in resp.text
        assert "HM-REDDIT-08" in resp.text

    # 2. Verify /v1/hivemind/manifesto JSON endpoint
    api_resp = client.get("/v1/hivemind/manifesto")
    assert api_resp.status_code == 200
    data = api_resp.json()
    assert data["object"] == "hivemind.manifesto"
    assert data["version"] == "1.0.0-s36"
    assert data["portal_links"]["gcp_openrouter_portal"] == "https://hivemind-openrouter-1003063861791.us-central1.run.app/portal"
    assert len(data["manifesto_pillars"]) == 7
    assert len(data["reddit_summary"]["threads"]) == 5

    # 3. Verify Portal, Benchmarks, Signup, and Join pages link to /manifesto
    for page_path in ("/portal", "/benchmarks", "/signup", "/join"):
        page_resp = client.get(page_path)
        assert page_resp.status_code == 200
        assert "/manifesto" in page_resp.text, f"Expected /manifesto link on {page_path}"





def test_ruche_pages_read_only_the_routers_live_payload_and_carry_no_literal_fallbacks() -> None:
    """The community and night-MoE scripts skip a payload that is not the router's live telemetry and never print
    a literal (147 workunits, 5632.11 cobblestones, eval_loss 1.7772, 'ONLINE') in place of a missing figure."""
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    client = TestClient(create_hivemind_at_home_app())
    for path in ("/community", "/night-moe"):
        html = client.get(path).text
        assert "d.status==='LIVE_GCP_PRODUCTION'" in html, path
        for literal in ("||147", "||5632.11", "||10519", "||1.7772", "||'ONLINE'", "||2.2705", "||2.138"):
            assert literal not in html, (path, literal)


def test_pages_show_na_not_a_literal_when_a_telemetry_figure_is_missing() -> None:
    """IP-146 server side: the portal, join and bee renderers printed 69 accounts, 147 workunits, 5,632.11 cobblestones,
    71,809 tokens or eval_loss 1.7772 whenever the payload lacked the figure; now they print n/a."""
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app
    from hivemind.router.join_page import render_join_page
    from hivemind.router.portal_page import render_portal_page
    from hivemind.router.telemetry_figures import shown_figure

    assert shown_figure(None) == "n/a" and shown_figure(float("nan")) == "n/a" and shown_figure(True) == "n/a"
    assert shown_figure(71809) == "71,809" and shown_figure(5632.11, decimals=2) == "5,632.11" and shown_figure(2.12239, decimals=4) == "2.1224"
    for html_text in (render_portal_page(None, "n", live_telemetry=None), render_portal_page(None, "n", live_telemetry={"accounts": {}, "t4_boinc": {}})):
        assert "n/a" in html_text
        # (the dated 25-26 Oct night report section quotes its own measured figures and is not a fallback)
        for literal in ("71,809", "10,519", "eval_loss 1.7772", "69 registered", "557 "):
            assert literal not in html_text, literal
    join_html = render_join_page("en", "abcdefgh12345678", live_telemetry={"accounts": {"total_registered_accounts": None}})
    assert "n/a registered accounts" in join_html and "69 registered accounts" not in join_html and "147 assimilated" not in join_html
    # the ruche's /join page renders its own honest payload without crashing and without a literal
    ruche = TestClient(create_hivemind_at_home_app()).get("/join", headers={"Accept-Language": "en"})
    assert ruche.status_code == 200 and "n/a registered accounts" in ruche.text and "147 assimilated" not in ruche.text


def test_router_live_telemetry_opentrainer_eval_loss_is_none_until_a_round_is_recorded() -> None:
    """IP-146 (router side): with no OpenTrainer round recorded the block said latest_eval_loss 1.7772, the Stage-2
    soak's final loss, while its own moe_self_training block said NO_ROUNDS_RECORDED_YET."""
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    client = TestClient(create_router_app(RouterStore(":memory:", b"k" * 32)))
    ot = client.get("/v1/hivemind/live-telemetry").json()["opentrainer"]
    assert ot["accepted_rounds_count"] == 0 and ot["latest_eval_loss"] is None
    assert ot["latest_eval_loss_source"] == "NO_ROUNDS_RECORDED_YET"
    assert ot["moe_self_training"]["latest_eval_loss"] is None


def test_portal_page_script_prints_na_not_a_literal_for_a_missing_live_figure() -> None:
    """IP-146 (page-JS half): the portal's live-GCP script carried `|| 69`, `|| 147`, `|| 5632.11`, `|| 10519`,
    `|| 71809`, `|| 557`, `|| 20480`, `|| 1.7772` and `|| 'ONLINE'` fallbacks, so a null figure rendered as a number."""
    from hivemind.router.portal_page import render_portal_page

    html_text = render_portal_page(None, "n")
    assert "const v = function (x)" in html_text
    for literal in ("|| 69)", "|| 147)", "|| 5632.11)", "|| 10519)", "|| 71809)", "|| 557)", "|| 20480)", "|| 1.7772)", "|| 'ONLINE')", "|| 15360)", "|| 42)", "|| 210)"):
        assert literal not in html_text, literal
