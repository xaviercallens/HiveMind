"""Unit tests for the Mathematical, Social, and Economic International English Launch Model (HM-S19-INTL)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.economy.international_launch_model import (
    SwarmLiveBaseline,
    build_100_producer_international_fleet_plan,
    build_foundation_paper_v42_portal_payload,
    compute_congestion_price,
    compute_erlang_c_metrics,
    compute_progressive_starter_grant,
    evaluate_international_launch_readiness,
    forecast_200plus_node_international_launch,
    project_international_launch_wave,
    render_foundation_paper_portal_html,
    render_gcp_planet_forecast_html,
)
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app


def test_erlang_c_detects_saturation_under_unthrottled_english_spike() -> None:
    # Current baseline: 8 producers, 380 tok/s -> service rate ~2.2579 req/s
    stable = compute_erlang_c_metrics(active_producers_c=8, aggregate_tok_per_s=380.0, arrival_rate_req_per_s=0.40)
    assert stable.is_stable is True
    assert stable.utilization_rho < 0.25
    assert stable.p95_total_latency_s < 1.5

    # Unthrottled Hacker News / r/LocalLLaMA spike: 4.5 req/s exceeds 2.26 req/s capacity
    saturated = compute_erlang_c_metrics(active_producers_c=8, aggregate_tok_per_s=380.0, arrival_rate_req_per_s=4.5)
    assert saturated.is_stable is False
    assert saturated.utilization_rho > 1.0
    assert saturated.erlang_c_queue_prob == 1.0


def test_progressive_starter_grant_prevents_unbacked_inflation() -> None:
    quote = compute_progressive_starter_grant(epoch_supply_1k_tokens=200.0, epoch_consumer_demand_1k_tokens=400.0)
    assert quote.instant_grant_hmt == 15.0
    assert quote.prosumer_gpu_unlock_bonus_hmt == 85.0
    assert quote.effective_unbacked_inflation_reduction_pct == 85.0
    assert quote.daily_consumer_drip_hmt == 2.5  # 5.0 * (200 / 400)


def test_congestion_pricing_boosts_producer_rewards_during_peak_demand() -> None:
    calm = compute_congestion_price(base_posted_price_per_1k=0.40, utilization_rho=0.45)
    assert calm["surge_multiplier"] == 1.0
    assert calm["clearing_price_per_1k"] == 0.40

    busy = compute_congestion_price(base_posted_price_per_1k=0.40, utilization_rho=0.90)
    assert busy["surge_multiplier"] > 2.5
    assert busy["producer_reward_per_1k"] > calm["producer_reward_per_1k"] * 2.5


def test_staged_international_wave_simulations_prove_controls_prevent_collapse() -> None:
    # Stage 2 (r/LocalLLaMA, 8,000 visitors) after Stage 1 producer seed (37 producers, 2,150 tok/s)
    proj = project_international_launch_wave(
        wave_name="r/LocalLLaMA Technical Wave",
        new_visitors=8000,
        signup_rate=0.15,
        direct_producer_fraction_alpha_p=0.04,
        consumer_to_prosumer_conversion_kappa=0.09,
        baseline=SwarmLiveBaseline(
            total_accounts=287,
            producers=37,
            prosumers=52,
            consumers=195,
            admins=3,
            peak_tflops_fp16=1820.0,
            daytime_aggregate_tok_per_s=2150.0,
        ),
    )
    assert proj.actual_conversion_rate_kappa >= proj.critical_conversion_rate_kappa_star
    assert proj.utilization_rho_with_controls < 0.78
    assert proj.utilization_rho_without_controls > proj.utilization_rho_with_controls
    assert proj.is_self_sustaining is True


def test_gateway_international_launch_readiness_endpoint(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    resp = client.get("/v1/hivemind/international-launch-readiness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "hivemind.international_launch_readiness"
    assert data["ready_for_unthrottled_mass_english_launch_today"] is False
    assert data["ready_for_staged_producer_first_rollout"] is True
    assert len(data["four_stage_rollout_plan"]) == 4
    assert len(data["wave_simulations"]) == 3
    assert data["hundred_producer_scale_plan"]["target_active_producers"] == 100
    assert data["live_store_scale_status"]["target_producers"] == 100


def test_build_100_producer_international_fleet_plan_invariants() -> None:
    plan = build_100_producer_international_fleet_plan(
        active_producers=100,
        standby_prosumers=160,
        arrival_rate_req_per_s=24.0,
    )
    assert plan["plan_id"] == "HM-S27-100-PRODUCER-INTERNATIONAL-SCALE"
    assert plan["target_active_producers"] == 100
    assert plan["max_endpoints_supported"] == 800
    assert plan["aggregate_daytime_tok_per_s"] >= 6200.0

    # Regional shard partition conservation: 42 + 28 + 16 + 14 == 100
    shards = plan["regional_shards"]
    assert len(shards) == 4
    assert [s["active_producers"] for s in shards] == [42, 28, 16, 14]
    assert sum(s["active_producers"] for s in shards) == 100
    assert sum(s["standby_prosumers"] for s in shards) == 160

    # Hardware tier partition conservation: 55 + 30 + 15 == 100
    tiers = plan["hardware_tier_allocations"]
    assert len(tiers) == 3
    assert [t["node_count"] for t in tiers] == [55, 30, 15]
    assert sum(t["node_count"] for t in tiers) == 100
    assert tiers[0]["vram_ceiling_mib"] <= 6656
    assert tiers[1]["free_vram_guard_mib"] >= 4300

    # Queueing stability at c = 100
    q = plan["global_queue_diagnostics"]
    assert q["is_stable"] is True
    assert q["utilization_rho"] < 0.78
    assert q["max_safe_concurrent_users"] >= 3500

    # Two-lane QoS & zero fiat invariant
    qos = plan["two_lane_priority_qos"]
    assert qos["prosumer_fast_lane_share"] + qos["consumer_fair_queue_share"] == 1.0
    assert qos["fiat_currency_exposure_eur_usd"] == 0.0


def test_router_store_and_gateway_100_producer_registration_and_simulation(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router_100.db", b"k" * 32)
    now = 1_760_000_000.0
    runtimes = ("ollama", "vllm", "llamacpp")

    for i in range(100):
        acc_id, _ = store.register(f"prod-node-{i:03d}", "producer")
        runtime = runtimes[i % len(runtimes)]
        model_id = "qwen3.5-coder-14b" if i % 2 == 0 else "qwen3.5-4b-instruct"
        store.upsert_producer(
            account_id=acc_id,
            endpoint_url=f"http://10.20.{i // 50}.{(i % 50) + 10}:11434",
            models=({"id": model_id, "local_name": f"{model_id}:q4_k_m"},),
            cost_per_1k=0.35,
            capacity_1k=25.0,
            execution_class=runtime,
            now_utc=now,
            residency={
                "probed_utc": now,
                "vram_used_mib": 5400,
                "models": {
                    model_id: {
                        "loaded": (i % 3 != 2),
                        "size_vram_mib": 5400 if (i % 3 != 2) else 0,
                    }
                },
            },
        )

    status = store.producer_fleet_scale_status(now_utc=now, ttl_s=120.0, target_producers=100)
    assert status["live_producer_accounts"] == 100
    assert status["live_endpoints"] == 100
    assert status["scale_readiness_pct"] == 100.0
    assert status["target_reached"] is True
    assert status["residency_reporting_endpoints"] == 100
    assert status["warm_loaded_models_count"] == 2
    assert sum(status["execution_class_breakdown"].values()) == 100

    app = create_router_app(store, clock=lambda: now, allow_private_producers=True)
    client = TestClient(app)

    status_resp = client.get("/v1/hivemind/producers/scale-100-status")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["live_store_scale_status"]["live_producer_accounts"] == 100
    assert status_body["live_store_scale_status"]["target_reached"] is True

    sim_resp = client.post(
        "/v1/hivemind/producers/scale-100-simulate",
        json={
            "active_producers": 100,
            "standby_prosumers": 160,
            "arrival_rate_req_per_s": 26.0,
            "client_region": "us-east",
            "model_id": "hivemind/qwen3.5-coder-14b",
        },
    )
    assert sim_resp.status_code == 200
    sim_body = sim_resp.json()
    assert sim_body["fleet_plan"]["global_queue_diagnostics"]["is_stable"] is True
    selected = sim_body["p2c_dispatch_sample"]["selected_producer"]
    assert selected["region"] == "us-east"
    assert selected["residency_tier"] == 0
    assert selected["loaded"] is True


def test_forecast_200plus_node_international_launch_consumer_heavy_stability() -> None:
    report = forecast_200plus_node_international_launch(
        target_total_nodes=265,
        visitor_count=2400,
    )
    assert report["model_id"] == "HM-S33-200PLUS-NODE-INTERNATIONAL-LAUNCH-FORECAST"
    assert report["ready_for_200plus_international_launch"] is True
    assert report["charter_compliance"]["fiat_currency_exposure_eur_usd"] == 0.0

    swarm = report["forecast_200plus_swarm"]
    assert swarm["total_active_nodes"] >= 200
    assert swarm["visitor_funnel_count"] == 2400
    assert swarm["consumer_nodes"] > swarm["prosumer_nodes"] > swarm["dedicated_producers"]
    assert (
        swarm["dedicated_producers"] + swarm["prosumer_nodes"] + swarm["consumer_nodes"]
        == swarm["total_active_nodes"]
    )
    assert swarm["effective_inference_servers_c"] >= swarm["halfin_whitt_min_servers_c"]

    tp = report["throughput_and_compute_power"]
    assert tp["effective_daytime_tok_per_s"] > 3000.0
    assert tp["elastic_burst_max_tok_per_s"] > tp["effective_daytime_tok_per_s"]
    assert tp["nocturnal_diloco_pflops_fp16"] > 3.0

    bal = report["balancing_and_latency_sla"]
    assert bal["uncontrolled_naive_routing"]["utilization_rho"] > 0.85
    assert bal["hivemind_balanced_routing"]["is_stable"] is True
    assert bal["hivemind_balanced_routing"]["utilization_rho"] <= 0.78
    assert bal["hivemind_balanced_routing"]["fast_lane_prosumer_ttft_ms"] <= 200.0
    assert bal["hivemind_balanced_routing"]["blended_p50_response_ms"] <= 300.0

    globe = report["hiveminers_on_globe"]
    assert globe["total_plotted_nodes"] >= 200
    assert globe["current_pioneer_nodes_count"] == 59
    assert report["current_live_baseline"]["total_nodes"] == 59
    assert report["current_live_baseline"]["producers"] == 10
    assert report["current_live_baseline"]["prosumers"] == 22
    assert report["current_live_baseline"]["consumers"] == 24
    assert report["current_live_baseline"]["billed_completions"] == 2455
    assert len(globe["super_peer_hubs"]) == 6
    assert len(globe["arcs"]) >= 10
    assert len(report["literature_review_models"]) == 6


def test_gcp_planet_forecast_html_and_dashboard_routes() -> None:
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    html = render_gcp_planet_forecast_html()
    assert "hivemind-planet-canvas" in html
    assert "Kleinrock" in html
    assert "Mitzenmacher" in html
    assert "0.00 EUR / 0.00 USD" in html

    app = create_hivemind_at_home_app()
    client = TestClient(app)

    api_resp = client.get("/api/v1/launch/planet-forecast?target_total_nodes=280&visitor_count=3200")
    assert api_resp.status_code == 200
    payload = api_resp.json()
    assert payload["forecast_200plus_swarm"]["total_active_nodes"] >= 280
    assert payload["forecast_200plus_swarm"]["visitor_funnel_count"] == 3200
    assert payload["ready_for_200plus_international_launch"] is True

    post_resp = client.post(
        "/api/v1/launch/planet-forecast",
        json={
            "target_total_nodes": 310,
            "visitor_count": 4000,
            "dedicated_producers": 34,
            "prosumer_nodes": 66,
            "consumer_nodes": 210,
            "utc_hour": 18,
        },
    )
    assert post_resp.status_code == 200
    post_data = post_resp.json()
    assert post_data["forecast_200plus_swarm"]["total_active_nodes"] == 310
    assert post_data["ready_for_200plus_international_launch"] is True

    page_resp = client.get("/planet-forecast")
    assert page_resp.status_code == 200
    assert "hivemind-planet-canvas" in page_resp.text
    assert "Interactive 3D Planet HiveMiners Globe" in page_resp.text


def test_foundation_paper_v42_and_portal_integration() -> None:
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    paper_payload = build_foundation_paper_v42_portal_payload()
    assert paper_payload["paper_version"] == "4.2.0"
    assert paper_payload["page_count"] >= 69
    assert paper_payload["charter_fiat_exposure_eur_usd"] == 0.0
    assert (
        paper_payload["live_experimentation_v42"]["experiment_e20_international_launch"][
            "P20a_adaptive_multisplit_guard"
        ]["client_prelude_vram_mib"]
        <= 1536.0
    )
    assert (
        paper_payload["live_experimentation_v42"]["experiment_e20_international_launch"][
            "P20b_top1pct_2d_dct_ef21"
        ]["compression_ratio"]
        == 225.4
    )
    assert len(paper_payload["five_wave_international_roadmap"]) == 6

    paper_html = render_foundation_paper_portal_html()
    assert "Foundation Article" in paper_html
    assert "v4.2.0" in paper_html
    assert "/paper/hivemind_foundation_article.pdf" in paper_html
    assert "/planet-forecast" in paper_html

    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Verify main HiveMind Portal (/) includes the Foundation Paper, 3D Globe, Real vs. Forecast Traffic & Network Balancing deck
    portal_resp = client.get("/")
    assert portal_resp.status_code == 200
    assert "portal-planet-paper-balancing-deck" in portal_resp.text
    assert "/foundation-paper" in portal_resp.text
    assert "/planet-forecast" in portal_resp.text
    assert "/paper/hivemind_foundation_article.pdf" in portal_resp.text
    assert "Real Live WAN Traffic (47 nodes) vs. Forecast Traffic" in portal_resp.text
    assert "HiveMind Network Balancing" in portal_resp.text

    # 2. Verify JSON API for Foundation Paper v4.2.0
    api_paper = client.get("/api/v1/paper/foundation")
    assert api_paper.status_code == 200
    api_data = api_paper.json()
    assert api_data["paper_version"] == "4.2.0"
    assert (
        api_data["real_vs_forecast_traffic_and_balancing"]["hivemind_balanced_265_node_forecast"][
            "status"
        ]
        == "BALANCED_QED_GREEN"
    )

    # 3. Verify HTML reader routes (/foundation-paper and /paper)
    reader_resp = client.get("/foundation-paper")
    assert reader_resp.status_code == 200
    assert "v4.2.0" in reader_resp.text
    assert "E20" in reader_resp.text

    alias_resp = client.get("/paper")
    assert alias_resp.status_code == 200
    assert "v4.2.0" in alias_resp.text

    # 4. Verify PDF and LaTeX source serving routes
    pdf_resp = client.get("/paper/hivemind_foundation_article.pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"].startswith("application/pdf")
    assert pdf_resp.content.startswith(b"%PDF-")

    tex_resp = client.get("/paper/hivemind_foundation_article.tex")
    assert tex_resp.status_code == 200
    assert "Version 4.2.0" in tex_resp.text
    assert "tab:five_wave_international_roadmap" in tex_resp.text


def test_international_launch_views_modular_separation() -> None:
    from hivemind.economy import international_launch_model, international_launch_views

    html_planet = international_launch_views.render_gcp_planet_forecast_html()
    html_paper = international_launch_views.render_foundation_paper_portal_html()
    assert "hivemind-planet-canvas" in html_planet
    assert "Foundation Article" in html_paper
    assert html_planet == international_launch_model.render_gcp_planet_forecast_html()
    assert html_paper == international_launch_model.render_foundation_paper_portal_html()

