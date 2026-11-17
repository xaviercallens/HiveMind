# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression and falsification tests for PR-IP-18: Game, Economy & Lab Honesty.

Covers:
- IP-87: Policy-gradient preference update in hf_hack_miner.py
- IP-88: Stored XSS escaping in exoplanet_hunter_engine.py producer registry & traffic log
- IP-89 & IP-128: Community hub initial state, corrupt JSON 503, unverified credit rejection, and concurrent write safety
- IP-90: Erlang-C service_time_s = 1/mu_single and p95_wait_s = 0.0 when P_Q <= 0.05
- IP-91: Daily modelled BOINC cobblestones unit consistency (200 * GFLOPS-day)
- IP-92: Dynamic critical_kappa_star and verdict_summary overload inequality (> 0.78)
- IP-93: Pioneer globe node location_estimated flag
- IP-94: Foundation paper portal dynamic node counts and overload styling
- IP-95: Honest synthetic flag on swarm telemetry and speed-of-light bounded RTTs
- IP-96: Real test callable execution in evaluate_bee_candidate_energy and ask_la_ruche_bee_copilot_live
- IP-97: Safe default install_to_local_steam_library=False, no taskkill, and non-destructive shortcuts.vdf append
- IP-127: Kelly equilibrium capacity scaling
- IP-138: Swarm lab staleness penalty queue retention and drop counts
"""

from __future__ import annotations

import inspect
import math
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from hivemind.economy.compute_coin import kelly_equilibrium
from hivemind.economy.international_launch_model import (
    COBBLESTONES_PER_GFLOPS_DAY,
    compute_erlang_c_metrics,
    forecast_200plus_node_international_launch,
    render_foundation_paper_portal_html,
)
from hivemind.game import exoplanet_hunter_cockpit as cockpit_mod
from hivemind.game import exoplanet_hunter_engine as engine_mod
from hivemind.game import fat_steam_astro_builder as builder_mod
from hivemind.game import scientific_discovery_game as sci_mod
from hivemind.game.hf_hack_miner import (
    _update_preferences,
    build_georing_3d_topology_snapshot,
    build_swarm_telemetry_frame,
)
from hivemind.lab import swarm_lab
from hivemind.verification.bee_energy import MAX_PAIN_ENERGY, evaluate_bee_candidate_energy


def test_ip87_update_preferences_rewards_chosen_policy_relative_to_others() -> None:
    prefs = {"A": 1.0, "B": 1.0, "C": 1.0}
    policies = ("A", "B", "C")
    _update_preferences(prefs, chosen="A", reward=2.0, mean_reward=1.0, policies=policies, learning_rate=0.5)
    assert prefs["A"] > prefs["B"]
    assert math.isclose(prefs["B"], prefs["C"], rel_tol=1e-9)

    prefs2 = {"A": 1.0, "B": 1.0, "C": 1.0}
    _update_preferences(prefs2, chosen="A", reward=0.2, mean_reward=1.0, policies=policies, learning_rate=0.5)
    assert prefs2["A"] < prefs2["B"]


def test_ip88_exoplanet_monitor_escapes_producer_and_traffic_xss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cockpit_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(cockpit_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(engine_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(
        engine_mod.HiveMindMultiProducerInferencePool,
        "benchmark_live_tflops",
        lambda self, matrix_dim=3072: {"measured_fp32_tflops": 0.0},
    )

    app = cockpit_mod.create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)
    r_reg = client.post(
        "/api/v1/producers/register",
        json={
            "producer_id": "P-XSS",
            "name": "<img src=x onerror=alert(1)>",
            "hardware": "<script>evil()</script>",
            "notes": "<svg onload=alert(2)>",
        },
    )
    assert r_reg.status_code == 200
    client.get("/health?<script>pwn()</script>", headers={"X-Gamer-Tag": "<b onmouseover=x>tag</b>"})

    telem_text = client.get("/api/v1/monitor/telemetry").text
    assert "<img src=x onerror=alert(1)>" not in telem_text
    assert "<script>evil()</script>" not in telem_text
    assert "<script>pwn()</script>" not in telem_text
    assert "&lt;img src=x onerror=alert(1)&gt;" in telem_text


def test_ip89_ip128_community_hub_honest_init_corrupt_503_credit_reject_and_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cockpit_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(cockpit_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(engine_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(
        engine_mod.HiveMindMultiProducerInferencePool,
        "benchmark_live_tflops",
        lambda self, matrix_dim=3072: {"measured_fp32_tflops": 0.0},
    )

    app = cockpit_mod.create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)

    # Fresh community hub has no fabricated personas
    hub_init = client.get("/api/v1/community/hub")
    assert hub_init.status_code == 200
    assert hub_init.json()["community_members"] == []
    assert hub_init.json()["coop_discovery_feed"] == []

    # Self-declared inflated credit is rejected with 400
    bad_sync = client.post(
        "/api/v1/community/boinc-sync",
        json={"handle": "@Cheater", "cobblestones": 999999999, "hmt_bonus": 999999999},
    )
    assert bad_sync.status_code == 400
    assert bad_sync.json()["ok"] is False

    # 50 concurrent boinc-sync calls preserve all 50 workunit increments without lost updates or corrupt JSON
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        errors: list[Exception] = []

        def _worker(i: int) -> None:
            try:
                r = client.post(
                    "/api/v1/community/boinc-sync",
                    json={"handle": f"@Concurrent_{i % 5}", "idle_seconds": 8.0},
                )
                assert r.status_code == 200
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
        assert not errors

    hub_after = client.get("/api/v1/community/hub").json()
    total_wus = sum(m["boinc_workunits_completed"] for m in hub_after["community_members"])
    assert total_wus == 50

    # Corrupt JSON on disk returns 503 BLOCKED without clobbering the file
    state_file = tmp_path / "astro_boinc_community_hub_state.json"
    state_file.write_text("{broken_json", encoding="utf-8")
    corrupt_resp = client.get("/api/v1/community/hub")
    assert corrupt_resp.status_code == 503
    assert corrupt_resp.json()["status"] == "BLOCKED"
    assert state_file.read_text(encoding="utf-8") == "{broken_json"


def test_ip90_erlang_c_service_time_and_p95_wait_monotonicity() -> None:
    c1 = compute_erlang_c_metrics(
        active_producers_c=1,
        aggregate_tok_per_s=100.0,
        arrival_rate_req_per_s=0.01,
        mean_tokens_per_req=100.0,
    )
    c10 = compute_erlang_c_metrics(
        active_producers_c=10,
        aggregate_tok_per_s=1000.0,
        arrival_rate_req_per_s=0.01,
        mean_tokens_per_req=100.0,
    )
    # Identical per-node speed (100 tok/s) and negligible queue (P_Q << 0.05) -> identical p95_total_latency_s
    assert c1.erlang_c_queue_prob <= 0.05
    assert c10.erlang_c_queue_prob <= 0.05
    assert math.isclose(c1.p95_total_latency_s, c10.p95_total_latency_s, rel_tol=1e-4)
    assert math.isclose(c10.p95_total_latency_s, 1.0 + 0.085, rel_tol=1e-4)


def test_ip91_ip92_ip93_launch_forecast_cobblestones_overload_verdict_and_location_flag() -> None:
    rep = forecast_200plus_node_international_launch(target_total_nodes=265, visitor_count=2400)
    tp = rep["throughput_and_compute_power"]
    blended_tflops = tp["daytime_inference_tflops_fp16"] * 0.45 + tp["nocturnal_diloco_tflops_fp16"] * 0.55
    expected_cobb = round(blended_tflops * 1000.0 * COBBLESTONES_PER_GFLOPS_DAY, 1)
    assert tp["daily_modelled_boinc_cobblestones"] == expected_cobb

    nodes = rep["hiveminers_on_globe"]["nodes"]
    assert nodes and all(n.get("location_estimated") is True for n in nodes)

    overload = forecast_200plus_node_international_launch(target_total_nodes=265, visitor_count=900_000)
    assert overload["ready_for_200plus_international_launch"] is False
    assert "> 0.78" in overload["verdict_summary"]
    assert "<= 0.78" not in overload["verdict_summary"]
    assert (
        overload["balancing_and_latency_sla"]["economic_and_epidemic_stabilizers"][
            "critical_conversion_threshold_kappa_star"
        ]
        > rep["balancing_and_latency_sla"]["economic_and_epidemic_stabilizers"][
            "critical_conversion_threshold_kappa_star"
        ]
    )


def test_ip94_render_foundation_paper_portal_html_dynamic_nodes_and_overload() -> None:
    payload = {
        "live_experimentation_v42": {
            "live_47_node_wan_baseline": {
                "active_nodes": 7,
                "utilization_rho": 1.4,
            },
            "experiment_e20_international_launch": {},
            "nocturnal_vmoe_pareto_ratchet": {},
        },
        "real_vs_forecast_traffic_and_balancing": {
            "hivemind_balanced_265_node_forecast": {
                "utilization_rho": 1.4,
            }
        },
    }
    html = render_foundation_paper_portal_html(payload)
    assert "7-Node WAN" in html
    assert "Real Live Swarm Traffic (7 Nodes)" in html
    assert "(7 &rarr; 10,000+ Nodes)" in html
    assert "READY FOR LAUNCH" not in html
    assert "OVERLOAD" in html
    assert 'class="card card-green"' not in html
    assert 'class="card card-red"' in html


def test_ip95_swarm_telemetry_synthetic_flag_and_physical_georing_rtts() -> None:
    frame = build_swarm_telemetry_frame(seq=3)
    assert frame["synthetic"] is True
    topo = build_georing_3d_topology_snapshot()
    expected_mean_rtt = round(
        sum(float(link["hop_rtt_ms"]) for link in topo["ring_links"]) / len(topo["ring_links"]),
        3,
    )
    assert frame["mean_ring_rtt_ms"] == expected_mean_rtt
    # US-East Virginia <-> Tokyo (~10,900 km great circle) must exceed 80 ms round-trip speed of light in fiber
    us_jp = next(l for l in topo["ring_links"] if "virginia" in l["source"] and "tokyo" in l["target"])
    assert us_jp["hop_rtt_ms"] >= 80.0


def test_ip96_bee_candidate_energy_and_copilot_execute_real_test_callables() -> None:
    wrong_eval = evaluate_bee_candidate_energy(
        candidate_id="wrong-fn",
        response_text="def f(x):\n    return 0\n",
        test_callables=[lambda ns: ns["f"](2) == 4, lambda ns: ns["f"](3) == 9],
        latency_ms=5.0,
        memory_mb=10.0,
    )
    assert wrong_eval.tests_passed == 0
    assert wrong_eval.tests_total == 2
    assert wrong_eval.total_energy >= 100.0

    # Plain prose reply without a function definition is rejected as AST_NO_FUNCTION_OR_CLASS (MAX_PAIN_ENERGY)
    prose_eval = evaluate_bee_candidate_energy(
        candidate_id="copilot-prose",
        response_text="Hello, I am a conversational reply with no Python code.",
        test_callables=[lambda ns: "step_verlet" in ns],
    )
    assert prose_eval.is_stub_or_invalid is True
    assert prose_eval.total_energy == MAX_PAIN_ENERGY

    mission3 = sci_mod.run_anse_energy_reactor_mission()
    evals = {a["candidate_id"]: a for a in mission3["attempts"]}
    assert evals["attempt-1-stub"]["tests_passed"] == 0
    assert evals["attempt-2-euler"]["tests_passed"] == 3
    assert evals["attempt-3-symplectic-verlet"]["tests_passed"] == 4


def test_ip97_fat_steam_builder_defaults_and_non_destructive_shortcuts_vdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sig = inspect.signature(builder_mod.build_fat_steam_exoplanet_client)
    assert sig.parameters["install_to_local_steam_library"].default is False

    src_text = inspect.getsource(builder_mod.build_fat_steam_exoplanet_client)
    assert "taskkill" not in src_text

    monkeypatch.setattr(builder_mod, "PROJECT_ROOT", tmp_path)
    steam_common = tmp_path / "SteamLibrary" / "steamapps" / "common"
    steam_config = tmp_path / "Steam" / "userdata" / "1" / "config"
    steam_common.mkdir(parents=True, exist_ok=True)
    steam_config.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(builder_mod, "STEAM_LIBRARY_COMMON", steam_common)
    monkeypatch.setattr(builder_mod, "STEAM_USERDATA_CONFIG", steam_config)
    monkeypatch.setattr(
        builder_mod,
        "synthesize_gemini_or_gcp_free_tts",
        lambda **kw: {"ok": True, "cached_path": str(tmp_path / "dummy.mp3")},
    )

    existing_shortcut = (
        b"\x00shortcuts\x00"
        b"\x000\x00"
        b"\x02appid\x00\x01\x02\x03\x04"
        b"\x01AppName\x00ExistingUserGame\x00"
        b"\x08\x08"
    )
    vdf_path = steam_config / "shortcuts.vdf"
    vdf_path.write_bytes(existing_shortcut)

    builder_mod.build_fat_steam_exoplanet_client(install_to_local_steam_library=True)
    backup_path = steam_config / "shortcuts.vdf.bak"
    assert backup_path.exists()
    assert backup_path.read_bytes() == existing_shortcut
    updated_vdf = vdf_path.read_bytes()
    assert b"ExistingUserGame" in updated_vdf
    assert b"USS HiveMind: Astrophysics Exoplanet Hunter" in updated_vdf


def test_ip127_kelly_equilibrium_scales_with_capacity() -> None:
    r1 = kelly_equilibrium([10.0, 20.0], capacity=1.0)
    r100 = kelly_equilibrium([10.0, 20.0], capacity=100.0)
    assert math.isclose(r100.total_payment, r1.total_payment * 100.0, rel_tol=1e-9)
    assert math.isclose(r100.payments[0], r1.payments[0] * 100.0, rel_tol=1e-9)
    assert math.isclose(r100.efficiency, r1.efficiency, rel_tol=1e-9)
    with pytest.raises(ValueError, match="capacity"):
        kelly_equilibrium([10.0, 20.0], capacity=0.0)


def test_ip138_swarm_lab_staleness_penalty_varies_weights_and_drops() -> None:
    sc_low = swarm_lab.Scenario(rounds=12, straggle="persistent", policy="deferred", penalty=1.5, seed=7)
    sc_mid = swarm_lab.Scenario(rounds=12, straggle="persistent", policy="deferred", penalty=3.0, seed=7)
    sc_high = swarm_lab.Scenario(rounds=12, straggle="persistent", policy="deferred", penalty=4.0, seed=7)

    r_low = swarm_lab.simulate(sc_low)
    r_mid = swarm_lab.simulate(sc_mid)
    r_high = swarm_lab.simulate(sc_high)

    assert r_low["final_loss"] != r_mid["final_loss"]
    assert sum(row["dropped_stale"] for row in r_high["rounds"]) > 0
