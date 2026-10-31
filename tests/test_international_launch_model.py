"""Unit tests for the Mathematical, Social, and Economic International English Launch Model (HM-S19-INTL)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.economy.international_launch_model import (
    SwarmLiveBaseline,
    build_100_producer_international_fleet_plan,
    compute_congestion_price,
    compute_erlang_c_metrics,
    compute_progressive_starter_grant,
    evaluate_international_launch_readiness,
    project_international_launch_wave,
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

