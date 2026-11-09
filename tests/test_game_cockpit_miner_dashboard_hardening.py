# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control boundary hardening tests for Iteration 136 (`hf_hack_miner`, `hivemind_at_home_dashboard`, `nat_traversal`, `exoplanet_hunter_engine`)."""

from __future__ import annotations

import math
import random
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from hivemind.boinc import hivemind_at_home_dashboard as dash
from hivemind.game import exoplanet_hunter_engine as eng
from hivemind.game import hf_hack_miner as hfm
from hivemind.swarm import nat_traversal as nt


def test_hf_hack_miner_repair_arena_and_strategy_validation() -> None:
    with pytest.raises(ValueError, match="patch_code must be a string"):
        hfm._repair_patch_safety_error(123)  # type: ignore[arg-type]
    assert hfm._repair_patch_safety_error("x = 1 + 1") == ""

    for bad_tests in ("not-seq", [123], b"bytes"):
        with pytest.raises(ValueError, match="tests must be a sequence of strings"):
            hfm._failed_repair_results(bad_tests, "Err")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error must be a non-empty string"):
        hfm._failed_repair_results(["assert True"], "  ")

    with pytest.raises(ValueError, match="patch_code must be a string"):
        hfm._evaluate_repair_candidate(None, ["assert True"])  # type: ignore[arg-type]
    for bad_tests in ("not-seq", [], [""], [123]):
        with pytest.raises(ValueError, match="tests must be a non-empty sequence of non-empty strings"):
            hfm._evaluate_repair_candidate("x = 1", bad_tests)  # type: ignore[arg-type]

    rng = random.Random(42)
    with pytest.raises(ValueError, match="preferences must be a dict"):
        hfm._choose_strategy("bad", rng, ["p1"])  # type: ignore[arg-type]
    for bad_pol in ("p1", [], [""], [123]):
        with pytest.raises(ValueError, match="policies must be a non-empty sequence of non-empty strings"):
            hfm._choose_strategy({"p1": 1.0}, rng, bad_pol)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rng must provide a random"):
        hfm._choose_strategy({"p1": 1.0}, None, ["p1"])
    assert hfm._choose_strategy({"p1": 1.0}, rng, ["p1"]) == "p1"

    prefs = {"p1": 1.0, "p2": 1.0}
    with pytest.raises(ValueError, match="preferences must be a dict"):
        hfm._update_preferences("bad", "p1", 1.0, 0.5, ["p1", "p2"], 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chosen must be a non-empty string"):
        hfm._update_preferences(prefs, "", 1.0, 0.5, ["p1", "p2"], 1.0)
    with pytest.raises(ValueError, match="policies must be a non-empty sequence of non-empty strings"):
        hfm._update_preferences(prefs, "p1", 1.0, 0.5, [], 1.0)
    for bad_num in (True, math.nan, math.inf, "bad"):
        with pytest.raises(ValueError, match="must be a finite number"):
            hfm._update_preferences(prefs, "p1", bad_num, 0.5, ["p1", "p2"], 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        hfm._update_preferences(prefs, "p1", 1.0, 0.5, ["p1", "p2"], 0.0)

    for cls in (hfm.SelfPlayCodeRepairArena, hfm.Lean4AutoformalizerArena):
        with pytest.raises(ValueError, match="seed must be an integer"):
            cls(seed=True)  # type: ignore[arg-type]
        for bad_lr in (False, 0.0, -1.0, math.nan):
            with pytest.raises(ValueError, match="learning_rate must be a finite positive number"):
                cls(seed=1, learning_rate=bad_lr)  # type: ignore[arg-type]
        arena = cls(seed=1, learning_rate=1.2)
        for bad_r in (0, -1, True):
            with pytest.raises(ValueError, match="rounds must be a positive integer"):
                arena.run(rounds=bad_r)  # type: ignore[arg-type]


def test_hf_hack_miner_perplexity_and_guild_tournament_validation() -> None:
    for bad_corpus in ("not-list", [123]):
        with pytest.raises(ValueError, match="corpus must be a list of strings"):
            hfm.estimate_bigram_holdout_perplexity(bad_corpus)  # type: ignore[arg-type]
    for bad_ratio in (0.0, 1.0, -0.2, 1.5, True, math.nan):
        with pytest.raises(ValueError, match="holdout_ratio must be in"):
            hfm.estimate_bigram_holdout_perplexity(["alpha beta", "beta gamma"], holdout_ratio=bad_ratio)  # type: ignore[arg-type]
    for bad_smooth in (0.0, -0.1, False, math.inf):
        with pytest.raises(ValueError, match="smoothing must be a finite positive number"):
            hfm.estimate_bigram_holdout_perplexity(["alpha beta", "beta gamma"], smoothing=bad_smooth)  # type: ignore[arg-type]

    for bad_squads in ([], ["not-squad"], "bad"):
        with pytest.raises(ValueError, match="squads must be a non-empty list of GuildTournamentSquad"):
            hfm.GuildTournamentEngine(squads=bad_squads)  # type: ignore[arg-type]

    eng_gt = hfm.GuildTournamentEngine()
    with pytest.raises(ValueError, match="guild_id must be a non-empty string"):
        eng_gt.register_squad_update("")
    with pytest.raises(ValueError, match="verified_worker must be a boolean"):
        eng_gt.register_squad_update("GUILD-01", verified_worker=1)  # type: ignore[arg-type]
    for bad_ppl in (0.0, -1.0, True, math.nan):
        with pytest.raises(ValueError, match="multi_domain_holdout_perplexity must be a finite positive number"):
            eng_gt.register_squad_update("GUILD-01", multi_domain_holdout_perplexity=bad_ppl)  # type: ignore[arg-type]
    for bad_tps in (-0.1, False, math.inf):
        with pytest.raises(ValueError, match="speculative_tokens_per_sec must be a finite non-negative number"):
            eng_gt.register_squad_update("GUILD-01", speculative_tokens_per_sec=bad_tps)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gamer_tag must be a string or None"):
        eng_gt.register_squad_update("GUILD-01", gamer_tag=123)  # type: ignore[arg-type]


def test_dashboard_flag_math_and_inference_validation() -> None:
    with pytest.raises(ValueError, match="report must be a dict"):
        dash._flag("bad", "k1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="keys must be non-empty strings"):
        dash._flag({"k1": True})
    with pytest.raises(ValueError, match="keys must be non-empty strings"):
        dash._flag({"k1": True}, "")
    assert dash._flag({"k1": True, "k2": 1}, "k1", "k2") is True

    with pytest.raises(ValueError, match="gpu_campaigns_report_path must be a non-empty path"):
        dash.build_hivemind_at_home_state(gpu_campaigns_report_path="  ")
    with pytest.raises(ValueError, match="mining_ruche_report_path must be a non-empty path"):
        dash.build_hivemind_at_home_state(mining_ruche_report_path="")
    with pytest.raises(ValueError, match="gcp_tpu_node_id must be a non-empty string"):
        dash.build_hivemind_at_home_state(gcp_tpu_node_id="")

    arr = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="x and weight must be non-empty numpy arrays"):
        dash._rms_norm_np([1.0, 2.0], arr)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="x and weight must be non-empty numpy arrays"):
        dash._rms_norm_np(np.array([]), arr)
    for bad_eps in (0.0, -1e-6, True, math.nan):
        with pytest.raises(ValueError, match="eps must be a finite positive number"):
            dash._rms_norm_np(arr, arr, eps=bad_eps)  # type: ignore[arg-type]
    assert dash._rms_norm_np(arr, arr).shape == (2,)

    for fn in (dash._silu_np, dash._sigmoid_np):
        for bad_x in (True, "bad", b"bytes", None):
            with pytest.raises(ValueError, match="x must be a numeric array or scalar"):
                fn(bad_x)  # type: ignore[arg-type]
        assert np.isfinite(fn(0.5))

    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        dash.run_la_ruche_moe_inference("   ")
    with pytest.raises(ValueError, match="vault_dir must be a non-empty path"):
        dash.run_la_ruche_moe_inference("hello", vault_dir="")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        dash.run_la_ruche_moe_inference("hello", top_k=0)
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        dash.run_la_ruche_moe_inference("hello", max_new_tokens=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="state must be a dict"):
        dash.render_hivemind_at_home_html("not-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="api_base_url must be a string"):
        dash.render_hivemind_at_home_html({}, api_base_url=123)  # type: ignore[arg-type]


def test_nat_traversal_decoders_verifiers_and_resolve_validation() -> None:
    with pytest.raises(TypeError, match="value and transaction_id must be bytes"):
        nt.decode_xor_address("bad", b"x" * 12)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="value must be bytes"):
        nt.decode_plain_address("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="value must be bytes"):
        nt._address_family(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="data must be bytes"):
        nt.is_channel_data("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="value must be bytes"):
        nt.decode_channel_number("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="data must be bytes"):
        nt.decode_channel_data(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="raw must be bytes and key must be non-empty bytes"):
        nt.verify_message_integrity("bad", b"key")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="raw must be bytes and key must be non-empty bytes"):
        nt.verify_message_integrity(b"raw", b"")
    with pytest.raises(ValueError, match="raw must be bytes"):
        nt.verify_fingerprint("bad")  # type: ignore[arg-type]

    for bad_tuple in (("", 3478), ("127.0.0.1", 0), ("127.0.0.1", 70000), ("127.0.0.1", True), ("127.0.0.1",)):
        with pytest.raises(ValueError, match="server tuple must be"):
            nt.resolve(bad_tuple)  # type: ignore[arg-type]
    for bad_str in ("localhost", ":3478", "127.0.0.1:0", "127.0.0.1:70000", "127.0.0.1:abc"):
        with pytest.raises(ValueError, match="server"):
            nt.resolve(bad_str)
    with pytest.raises(TypeError, match="server must be a"):
        nt.resolve(12345)  # type: ignore[arg-type]
    assert nt.resolve("127.0.0.1:3478") == ("127.0.0.1", 3478)


def test_exoplanet_hunter_engine_service_and_pool_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = eng.AstroBeeMoEGPUService.__new__(eng.AstroBeeMoEGPUService)
    svc.catalog = [{"target_id": "SECTOR-01-TRAPPIST-1E", "name": "TRAPPIST-1 e", "disposition": "CONFIRMED"}]
    svc.discoveries_path = tmp_path / "discoveries.json"
    svc._awarded_discoveries = set()

    with pytest.raises(ValueError, match="target must be a non-empty dict"):
        svc.run_bee_moe_planet_analysis({})
    for bad_period in (0.0, -1.0, True, math.nan):
        with pytest.raises(ValueError, match="trial_period_days must be a finite positive number"):
            svc.run_bee_moe_planet_analysis({"id": 1}, trial_period_days=bad_period)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gamer_tag must be a non-empty string"):
        svc.run_bee_moe_planet_analysis({"id": 1}, gamer_tag="  ")

    with pytest.raises(ValueError, match="target_id must be a string"):
        svc.resolve_target(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="target_id must be a non-empty string"):
        svc.record_planet_discovery("", "Alice", "PLANET")
    with pytest.raises(ValueError, match="gamer_tag must be a string"):
        svc.record_planet_discovery("SECTOR-01-TRAPPIST-1E", 123, "PLANET")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="player_classification must be a non-empty string"):
        svc.record_planet_discovery("SECTOR-01-TRAPPIST-1E", "Alice", "")
    with pytest.raises(ValueError, match="notes must be a string"):
        svc.record_planet_discovery("SECTOR-01-TRAPPIST-1E", "Alice", "PLANET", notes=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="account_id must be a string or None"):
        svc.record_planet_discovery("SECTOR-01-TRAPPIST-1E", "Alice", "PLANET", account_id=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="prompt must be a string"):
        svc.run_general_hivemind_moe_inference(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        svc.run_general_hivemind_moe_inference("ok", max_new_tokens=0)
    with pytest.raises(ValueError, match="preferred_producer must be a non-empty string"):
        svc.run_general_hivemind_moe_inference("ok", preferred_producer="")

    with pytest.raises(ValueError, match="gpu_service must be an AstroBeeMoEGPUService"):
        eng.HiveMindMultiProducerInferencePool("not-service")  # type: ignore[arg-type]

    pool = eng.HiveMindMultiProducerInferencePool.__new__(eng.HiveMindMultiProducerInferencePool)
    pool.gpu_service = svc
    pool.total_requests_served = 0
    pool.total_http_requests = 0

    with pytest.raises(ValueError, match="prompt must be a string"):
        pool.serve_inference(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        pool.serve_inference("hi", max_new_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="preferred_producer must be a string"):
        pool.serve_inference("hi", preferred_producer=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_id must be a string or None"):
        pool.serve_inference("hi", target_id=123)  # type: ignore[arg-type]

    for bad_dim in (0, -64, True):
        with pytest.raises(ValueError, match="matrix_dim must be a positive integer"):
            pool.benchmark_live_tflops(matrix_dim=bad_dim)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="method and path must be non-empty strings"):
        pool.record_traffic_event("", "/health")
    with pytest.raises(ValueError, match="client_ip, user_agent, and gamer_tag must be strings"):
        pool.record_traffic_event("GET", "/health", client_ip=123)  # type: ignore[arg-type]
    for bad_lat in (-0.1, True, math.nan):
        with pytest.raises(ValueError, match="latency_ms must be a finite non-negative number"):
            pool.record_traffic_event("GET", "/health", latency_ms=bad_lat)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="served_by must be a string or None"):
        pool.record_traffic_event("GET", "/health", served_by=123)  # type: ignore[arg-type]
