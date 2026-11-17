# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for Iteration 32 (PR-IP-32: IP-31, IP-34, IP-42, IP-43, IP-44, IP-47, IP-48, IP-67, IP-68)."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from starlette.testclient import TestClient


def test_ip31_steam_status_reports_not_registered_placeholder(tmp_path: Path) -> None:
    """IP-31: /v1/hivemind/steam/status must report NOT_REGISTERED and PLACEHOLDER_NOT_REGISTERED."""
    from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    store = RouterStore(tmp_path / "router.db", b"s" * 32)
    router_client = TestClient(create_router_app(store))
    r_status = router_client.get("/v1/hivemind/steam/status")
    assert r_status.status_code == 200
    body = r_status.json()
    assert body["portal_registration_status"] == "NOT_REGISTERED"
    assert body["app_id_status"] == "PLACEHOLDER_NOT_REGISTERED"
    assert "REGISTERED_AND_LIVE" not in r_status.text

    cockpit_client = TestClient(create_exoplanet_hunter_fastapi_app(portal_secret=b"p" * 32))
    c_status = cockpit_client.get("/v1/hivemind/steam/status")
    assert c_status.status_code == 200
    c_body = c_status.json()
    assert c_body["portal_registration_status"] == "NOT_REGISTERED"
    assert c_body["app_id_status"] == "PLACEHOLDER_NOT_REGISTERED"
    assert "REGISTERED_AND_LIVE" not in c_status.text


def test_ip34_multikrum_clamps_f_to_validity_bound_and_reports_krum_guarantee() -> None:
    """IP-34: K=4 with max_byzantine_f=1 must clamp max_byzantine_f to 0 and report krum_guarantee=False;
    K=5 with max_byzantine_f=1 satisfies K >= 2f+3 and reports max_byzantine_f=1, krum_guarantee=True."""
    from hivemind.core.diloco_outer import filter_byzantine_deltas_multikrum

    deltas_k4 = {
        f"w{i}": {"p": torch.tensor([1.0, 0.5, 0.25, 0.125]) * (1.0 + 0.01 * i)}
        for i in range(4)
    }
    rep_k4 = filter_byzantine_deltas_multikrum(deltas_k4, max_byzantine_f=1)
    assert rep_k4.max_byzantine_f == 0
    assert rep_k4.krum_guarantee is False

    deltas_k5 = {
        f"w{i}": {"p": torch.tensor([1.0, 0.5, 0.25, 0.125]) * (1.0 + 0.01 * i)}
        for i in range(5)
    }
    rep_k5 = filter_byzantine_deltas_multikrum(deltas_k5, max_byzantine_f=1)
    assert rep_k5.max_byzantine_f == 1
    assert rep_k5.krum_guarantee is True


def test_ip42_stream_load_hf_causal_lm_enforces_pinned_revision_and_sha256(tmp_path: Path) -> None:
    """IP-42: require_pinned_revision rejects unpinned/missing revisions, and expected_sha256 rejects digest mismatch."""
    from safetensors.torch import save_file
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from hivemind.core.quantization import stream_load_hf_causal_lm

    with pytest.raises(ValueError, match="pinned commit revision"):
        stream_load_hf_causal_lm(
            model_id="Qwen/Qwen2.5-0.5B",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            require_pinned_revision=True,
        )

    with pytest.raises(ValueError, match="pinned commit revision"):
        stream_load_hf_causal_lm(
            model_id="Qwen/Qwen2.5-0.5B",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            revision="main",
        )

    tiny_cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    ref_model = Qwen2ForCausalLM(tiny_cfg).to("cpu")
    ckpt_path = tmp_path / "model.safetensors"
    save_file({k: v.detach().cpu().clone() for k, v in ref_model.state_dict().items()}, str(ckpt_path))
    true_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=tiny_cfg) as mock_cfg,
        patch("huggingface_hub.hf_hub_download", return_value=str(ckpt_path)) as mock_dl,
    ):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            stream_load_hf_causal_lm(
                model_id="Qwen/Qwen2.5-0.5B",
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
                apply_nf4=False,
                revision="a1b2c3d4e5f6",
                expected_sha256="0" * 64,
            )

        loaded, _, _ = stream_load_hf_causal_lm(
            model_id="Qwen/Qwen2.5-0.5B",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            apply_nf4=False,
            revision="a1b2c3d4e5f6",
            expected_sha256=true_sha,
            require_pinned_revision=True,
        )
        assert loaded is not None
        mock_cfg.assert_called_with("Qwen/Qwen2.5-0.5B", token=None, revision="a1b2c3d4e5f6")
        mock_dl.assert_called_with(
            repo_id="Qwen/Qwen2.5-0.5B", filename="model.safetensors", token=None, revision="a1b2c3d4e5f6"
        )


def test_ip43_configure_deterministic_runtime_defaults_to_strict_non_warn_only() -> None:
    """IP-43: configure_deterministic_runtime defaults to warn_only=False so spot-check verification is strict."""
    from hivemind.config import configure_deterministic_runtime

    sig = inspect.signature(configure_deterministic_runtime)
    assert sig.parameters["warn_only"].default is False

    configure_deterministic_runtime(seed=123)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False


def test_ip44_bee_energy_rejects_lean4_axioms_and_admits_but_allows_comments() -> None:
    """IP-44: Lean 4 candidates with `axiom h : False` or `admit` must receive MAX_PAIN_ENERGY;
    mentioning `sorry` or `axiom` inside a comment on an otherwise valid proof is allowed."""
    from hivemind.verification.bee_energy import MAX_PAIN_ENERGY, evaluate_bee_candidate_energy

    res_axiom = evaluate_bee_candidate_energy(
        candidate_id="cand-axiom",
        domain="lean4",
        response_text="axiom h : False\ntheorem bad : False := h",
        tests_passed=1,
        tests_total=1,
        latency_ms=10.0,
        memory_mb=16.0,
        invariant_residual=0.0,
    )
    assert res_axiom.is_stub_or_invalid is True
    assert res_axiom.total_energy == MAX_PAIN_ENERGY
    assert res_axiom.failure_reason is not None and "LEAN4_AXIOM_DETECTED" in res_axiom.failure_reason

    res_clean = evaluate_bee_candidate_energy(
        candidate_id="cand-clean",
        domain="lean4",
        response_text="-- Zero sorry and zero axiom proof\n/- block sorry axiom -/\ntheorem ok : 1 + 1 = 2 := rfl",
        tests_passed=1,
        tests_total=1,
        latency_ms=10.0,
        memory_mb=16.0,
        invariant_residual=0.0,
    )
    assert res_clean.is_stub_or_invalid is False
    assert res_clean.passed_all_tests is True


def test_ip47_h5_hypothesis_fails_when_dma_transfer_exceeds_attention_window() -> None:
    """IP-47: H5 with copy time above the attention window must report passed=False."""
    from hivemind.swarm.architecture_hypothesis_verifier import ArchitectureHypothesisVerifier

    verifier = ArchitectureHypothesisVerifier(seed=42)
    rec_pass = verifier.verify_h5_skip_attention_car_and_async_dma(
        simulated_attention_compute_ms=5.0,
        min_transfer_latency_ms=0.0,
    )
    assert rec_pass.passed is True

    rec_fail = verifier.verify_h5_skip_attention_car_and_async_dma(
        simulated_attention_compute_ms=1.0,
        min_transfer_latency_ms=5.0,
    )
    assert rec_fail.passed is False
    assert rec_fail.key_metrics["exposed_swap_latency_ms_simulated"] > 0.0


def test_ip48_lane_emden_integrates_rk4_matches_analytical_and_rejects_bad_polytrope() -> None:
    """IP-48: Lane-Emden integrates via RK4 (matching exact n=0 and n=1 solutions to <1e-5) and rejects invalid poly_n."""
    import math
    from hivemind.swarm.rusty_sundials_scientific_bee import solve_wave2_method_of_lines_pde

    # n = 0 exact solution: theta(xi) = 1 - xi^2 / 6
    res_n0 = solve_wave2_method_of_lines_pde("lane_emden_stellar", grid_points=32, param=0.0)
    assert res_n0["crate"] == "python_rk4_desingularized_lane_emden"
    assert res_n0["invariant_drift"] < 1e-6
    for xi, th, _ in res_n0["full_profile"]:
        expected_th = max(0.0, 1.0 - (xi * xi) / 6.0)
        assert abs(th - expected_th) < 1e-5

    # n = 1 exact solution: theta(xi) = sin(xi) / xi
    res_n1 = solve_wave2_method_of_lines_pde("lane_emden_stellar", grid_points=64, param=1.0)
    assert res_n1["invariant_drift"] < 1e-6
    for xi, th, _ in res_n1["full_profile"][1:]:
        expected_th = max(0.0, math.sin(xi) / xi)
        assert abs(th - expected_th) < 1e-5

    # Invalid polytropic index must raise ValueError
    with pytest.raises(ValueError, match="polytropic index"):
        solve_wave2_method_of_lines_pde("lane_emden_stellar", param=-0.5)
    with pytest.raises(ValueError, match="polytropic index"):
        solve_wave2_method_of_lines_pde("lane_emden_stellar", param=5.5)
    with pytest.raises(ValueError, match="polytropic index"):
        solve_wave2_method_of_lines_pde("lane_emden_stellar", param=float("nan"))


def test_ip67_and_ip68_no_hardcoded_developer_paths_and_annotations_present() -> None:
    """IP-67 & IP-68: No hardcoded `/home/xavkal/xdev` paths in `src/` and required type annotations present."""
    from hivemind.swarm.inference_ring import RegionalDHTSwarmRing
    from hivemind.swarm.quic_link import WebSocketHiddenStateServer

    src_root = Path(__file__).resolve().parents[1] / "src"
    offenders: list[str] = []
    for py_file in src_root.rglob("*.py"):
        if "/home/xavkal/xdev" in py_file.read_text(encoding="utf-8"):
            offenders.append(str(py_file.relative_to(src_root)))
    assert offenders == []

    assert "return" in WebSocketHiddenStateServer._handle.__annotations__
    assert "return" in RegionalDHTSwarmRing._make_prefetch_hook.__annotations__
    assert "return" in RegionalDHTSwarmRing._make_router_capture_hook.__annotations__
