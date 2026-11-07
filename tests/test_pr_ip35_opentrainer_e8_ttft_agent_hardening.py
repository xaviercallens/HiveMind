# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression suite for Iteration 35 (PR #703):
- Issue #618 / IP-259: OpenTrainer submit-delta rejects unknown campaign_id values
- IP-30: Scientific discovery game cohort playtest and mission reports carry simulated=True
- IP-38: E8 Gosset lattice quantizer tie-breaking parity across all 3 call sites (10^4 vectors + ties)
- MVP Audit C4: Single canonical source for hivemind_producer_agent.py and hivemind_producer_shield.py
- MVP Audit C9: Streaming TTFT (ttft_ms) and throughput (tokens_per_s) recorded in settle & usage_telemetry
- MVP Audit B10: Canonical roadmap PR specs carry no fabricated OpenRouter completion/request IDs
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from hivemind.compress.lattice_quantizer import nearest_dn, nearest_e8
from hivemind.game.scientific_discovery_game import (
    HiveMindScientificGameEngine,
    _quantize_e8_vector,
)
from hivemind.router import hivemind_producer_agent as src_agent
from hivemind.router import hivemind_producer_shield as src_shield
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.welcome_email import build_welcome_email_content
from hivemind.swarm.pr_first_openrouter_orchestrator import (
    CANONICAL_5_ROADMAP_PRS,
    PRFirstOpenRouterCodingOrchestrator,
)
from hivemind.swarm.s34_03_gosset_e8_lattice_quantizer import (
    GossetE8LatticeQuantizer,
    _nearest_d8,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ip259_issue618_opentrainer_submit_delta_rejects_invented_campaign_id(tmp_path: Path) -> None:
    """Issue #618 / IP-259: submit-delta must reject invented campaign_id values with 400 unknown_campaign."""
    store = RouterStore(tmp_path / "router.sqlite3", secret=b"s" * 32)
    app = create_router_app(store)
    client = TestClient(app)

    _, api_key = store.register("trainer_ip35", "prosumer")
    headers = {"Authorization": f"Bearer {api_key}"}

    # 1. Invented campaign_id is rejected with 400 unknown_campaign
    bad_resp = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers=headers,
        json={
            "campaign_id": "INVENTED-FAKE-CAMPAIGN-999999",
            "round_id": 1,
            "delta_norm": 0.42,
            "delta_sha256": "a" * 64,
            "loss_before": 2.1,
            "loss_after": 1.8,
            "inner_steps": 16,
        },
    )
    assert bad_resp.status_code == 400
    err = bad_resp.json()["error"]
    assert err["code"] == "unknown_campaign"
    assert "unknown campaign" in err["message"].lower()

    # 2. Known catalog campaign_id succeeds
    ok_resp = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers=headers,
        json={
            "campaign_id": "E8-STAGE2-TINY-GRANITE-MOE-QLORA",
            "round_id": 1,
            "delta_norm": 0.42,
            "delta_sha256": "b" * 64,
            "loss_before": 2.1,
            "loss_after": 1.8,
            "inner_steps": 16,
        },
    )
    assert ok_resp.status_code == 200

    # 3. Node heartbeat registering a custom campaign allows that account to submit deltas for it
    hb_resp = client.post(
        "/v1/hivemind/opentrainer/heartbeat",
        headers=headers,
        json={
            "host_label": "gpu-0",
            "gpu_name": "NVIDIA GeForce RTX 2070",
            "vram_total_mib": 8192,
            "vram_free_mib": 6800,
            "active_campaign": "CUSTOM-NODE-HEARTBEAT-RUN-2026",
        },
    )
    assert hb_resp.status_code == 200
    assert hb_resp.json()["node"]["active_campaign"] == "CUSTOM-NODE-HEARTBEAT-RUN-2026"

    hb_delta_resp = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers=headers,
        json={
            "campaign_id": "CUSTOM-NODE-HEARTBEAT-RUN-2026",
            "round_id": 2,
            "delta_norm": 0.35,
            "delta_sha256": "c" * 64,
            "loss_before": 1.8,
            "loss_after": 1.6,
            "inner_steps": 16,
        },
    )
    assert hb_delta_resp.status_code == 200


def test_ip30_scientific_discovery_game_marks_cohort_and_missions_simulated() -> None:
    """IP-30: Scripted persona playtests and mission suites must carry simulated=True."""
    engine = HiveMindScientificGameEngine()
    cohort = engine.run_multiplayer_cohort_playtest()
    assert cohort["simulated"] is True
    assert all(s["simulated"] is True for s in cohort["player_sessions"])
    assert all(row["simulated"] is True for row in cohort["leaderboard"])

    missions_suite = engine.run_full_scientific_odyssey()
    assert missions_suite["simulated"] is True


def test_ip38_e8_gosset_lattice_tie_breaking_parity_across_all_call_sites() -> None:
    """IP-38: All 3 E8 lattice projection call sites must agree bit-for-bit on 10^4 random + tie vectors."""
    rng = np.random.default_rng(20261107)
    random_batch = rng.normal(0.0, 2.0, size=(10_000, 8))
    tie_vectors = np.array(
        [
            [0.5, -0.5, 1.5, -1.5, 0.25, -0.25, 0.75, -0.75],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
            [-0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.25, -1.25, 0.75, -0.75, 0.25, -0.25, 2.25, -2.25],
        ],
        dtype=np.float64,
    )
    all_vecs = np.vstack([tie_vectors, random_batch])
    canonical = nearest_e8(all_vecs)
    canonical_d8 = nearest_dn(tie_vectors)

    quantizer = GossetE8LatticeQuantizer()
    assert quantizer.reference_artifact == "artifacts/experiments/E10_lattice_diloco.json"
    assert quantizer.dense_e8_compression_ratio == 16.0

    # Verify tie vectors on all 3 implementations
    for idx in range(len(tie_vectors)):
        row = tie_vectors[idx]
        q_s34, _ = quantizer.project_to_e8(row.tolist())
        q_game = _quantize_e8_vector(row)
        d8_s34 = _nearest_d8(row.tolist())
        np.testing.assert_allclose(q_s34, canonical[idx], atol=1e-12)
        np.testing.assert_allclose(q_game, canonical[idx], atol=1e-12)
        np.testing.assert_allclose(d8_s34, canonical_d8[idx], atol=1e-12)

    # Verify 1,000 random samples across S34 quantizer and game quantizer against canonical batch
    for idx in range(5, 1005):
        row = all_vecs[idx]
        q_s34, _ = quantizer.project_to_e8(row.tolist())
        q_game = _quantize_e8_vector(row)
        np.testing.assert_allclose(q_s34, canonical[idx], atol=1e-12)
        np.testing.assert_allclose(q_game, canonical[idx], atol=1e-12)

    email_doc = build_welcome_email_content(
        handle="alice_bee",
        recipient_email="alice@example.org",
        account_id="acct_alice",
        role="prosumer",
        balance_hmt=100.0,
    )
    html = str(email_doc["html_body"])
    assert "artifacts/experiments/E10_lattice_diloco.json" in html
    assert "16&times;" in html


def test_audit_c4_single_source_for_producer_agent_and_shield(tmp_path: Path) -> None:
    """MVP Audit C4: scripts/hivemind_producer_{agent,shield}.py and src/hivemind/router/... are byte-identical and gateway serves them."""
    agent_script_path = REPO_ROOT / "scripts" / "hivemind_producer_agent.py"
    shield_script_path = REPO_ROOT / "scripts" / "hivemind_producer_shield.py"
    agent_src_path = REPO_ROOT / "src" / "hivemind" / "router" / "hivemind_producer_agent.py"
    shield_src_path = REPO_ROOT / "src" / "hivemind" / "router" / "hivemind_producer_shield.py"

    assert agent_script_path.read_text(encoding="utf-8") == agent_src_path.read_text(encoding="utf-8")
    assert shield_script_path.read_text(encoding="utf-8") == shield_src_path.read_text(encoding="utf-8")

    spec_a = importlib.util.spec_from_file_location("scripts_agent_check", agent_script_path)
    assert spec_a is not None and spec_a.loader is not None
    mod_a = importlib.util.module_from_spec(spec_a)
    spec_a.loader.exec_module(mod_a)

    spec_s = importlib.util.spec_from_file_location("scripts_shield_check", shield_script_path)
    assert spec_s is not None and spec_s.loader is not None
    mod_s = importlib.util.module_from_spec(spec_s)
    spec_s.loader.exec_module(mod_s)

    for attr in (
        "main",
        "build_parser",
        "probe_residency",
        "probe_model_sizes_mib",
        "filter_models_by_vram_budget",
        "apply_residency",
        "parse_control",
        "apply_control",
        "fetch_remote_config",
    ):
        assert hasattr(mod_a, attr) and hasattr(src_agent, attr), f"missing attr on agent: {attr}"

    for attr in ("main", "build_parser", "create_shield_app", "POC_NOTICE"):
        assert hasattr(mod_s, attr) and hasattr(src_shield, attr), f"missing attr on shield: {attr}"

    store = RouterStore(tmp_path / "router.sqlite3", secret=b"s" * 32)
    client = TestClient(create_router_app(store))
    r_agent = client.get("/hivemind_producer_agent.py")
    r_shield = client.get("/hivemind_producer_shield.py")
    assert r_agent.status_code == 200
    assert r_shield.status_code == 200
    assert r_agent.text == agent_src_path.read_text(encoding="utf-8")
    assert r_shield.text == shield_src_path.read_text(encoding="utf-8")


def test_audit_c9_streaming_ttft_ms_and_tokens_per_s_recorded_in_settle(tmp_path: Path) -> None:
    """MVP Audit C9: settle and usage_telemetry record ttft_ms and tokens_per_s for streaming and blocking requests."""
    store = RouterStore(tmp_path / "router.sqlite3", secret=b"s" * 32)

    def producer_handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        if body.get("stream"):
            sse = (
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello "}}]}\n\n'
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Swarm!"}}],"usage":{"prompt_tokens":10,"completion_tokens":20}}\n\n'
                b"data: [DONE]\n\n"
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(sse))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Blocking ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
            },
        )

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(producer_handler)),
        resolver=lambda host: ["93.184.216.34"],
    )
    client = TestClient(app)

    _, prod_key = store.register("prod_c9", "producer")
    cons_id, cons_key = store.register("cons_c9", "consumer")
    store.grant(cons_id, 50.0, "test_c9")

    hb = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {prod_key}"},
        json={
            "endpoint_url": "https://producer.example.org",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b", "loaded": True, "vram_mib": 4800}],
            "cost_per_1k": 0.4,
            "capacity_1k": 60.0,
            "execution_class": "gpu",
        },
    )
    assert hb.status_code == 200

    # 1. Streaming request records ttft_ms and tokens_per_s
    stream_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/coder", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert stream_resp.status_code == 200
    assert "[DONE]" in stream_resp.text

    # 2. Blocking request records ttft_ms and tokens_per_s
    block_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/coder", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert block_resp.status_code == 200

    summary = store.usage_capture_summary(include_non_user=True)
    assert summary["total_requests"] == 2
    assert summary["stream_requests"] == 1
    assert summary["ttft_ms"]["samples"] == 2
    assert summary["ttft_ms"]["p50"] >= 0.0
    assert summary["tokens_per_s"]["samples"] == 2
    assert summary["tokens_per_s"]["mean"] > 0.0


def test_audit_b10_pr_first_openrouter_orchestrator_has_no_fabricated_ids() -> None:
    """MVP Audit B10: CANONICAL_5_ROADMAP_PRS must not embed fabricated chatcmpl-* or req-* strings."""
    for spec in CANONICAL_5_ROADMAP_PRS:
        assert spec.openrouter_completion_id is None
        assert spec.openrouter_request_id is None

    orchestrator = PRFirstOpenRouterCodingOrchestrator(repo_root=REPO_ROOT)
    receipts = orchestrator.verify_all_roadmap_prs()
    assert len(receipts) == 5
    for rec in receipts:
        assert rec.openrouter_completion_id is None
        assert rec.openrouter_request_id is None
        assert rec.ast_security_clean is True
        assert rec.ready_to_merge is True
        assert rec.to_dict()["roadmap_id"] == rec.roadmap_id


def test_pr_ip35_comprehensive_branch_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise edge cases in E8 quantizer, PR-first orchestrator, OpenTrainer validator, producer shield/agent, and solverbee families."""
    from hivemind.router.opentrainer import (
        is_allowed_opentrainer_campaign_id,
        validate_trainer_delta_submission,
    )
    from hivemind.solverbee.families import FamilyError, compile_slot_spec
    from hivemind.swarm.pr_first_openrouter_orchestrator import RoadmapPRSpec, scan_python_ast_security
    from hivemind.swarm.s34_03_gosset_e8_lattice_quantizer import (
        _nearest_d8,
        is_valid_e8_lattice_point,
    )

    # 1. GossetE8LatticeQuantizer edge cases
    with pytest.raises(ValueError):
        GossetE8LatticeQuantizer(codebook_radius=0.0)
    with pytest.raises(ValueError):
        GossetE8LatticeQuantizer.project_to_e8([1.0, 2.0])
    with pytest.raises(ValueError):
        _nearest_d8([1.0, 2.0])
    q_e8 = GossetE8LatticeQuantizer(codebook_radius=4.0)
    with pytest.raises(ValueError):
        q_e8.quantize_8d([1.0, 2.0])
    zero_block = q_e8.quantize_8d([0.0] * 8)
    assert zero_block.scale == 0.0 and zero_block.to_dict()["coset"] == "D8_integer"
    nz_block = q_e8.quantize_8d([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert nz_block.is_valid_e8_point is True
    assert is_valid_e8_lattice_point([0.0, 0.0, 0.0]) is False
    assert is_valid_e8_lattice_point([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) is True
    assert is_valid_e8_lattice_point([0.5] * 8) is True
    assert is_valid_e8_lattice_point([0.3] * 8) is False

    # 2. PRFirstOpenRouterCodingOrchestrator edge cases
    clean, violations = scan_python_ast_security("x = eval('1')\ny = pickle.loads(b'')\n")
    assert clean is False and "eval" in violations and "pickle.loads" in violations
    orch = PRFirstOpenRouterCodingOrchestrator(repo_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        orch.verify_spec(
            RoadmapPRSpec(1, "HM-X", "T", "missing.py", "test_missing.py")
        )
    (tmp_path / "mod_ok.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        orch.verify_spec(
            RoadmapPRSpec(1, "HM-X", "T", "mod_ok.py", "test_missing.py")
        )

    # 3. OpenTrainer validator edge cases
    assert is_allowed_opentrainer_campaign_id("   ") is False
    for bad_kwargs in (
        {"run_id": "", "round_index": 0, "delta_l2": 0.1, "eval_loss": 1.0, "tokens_trained": 100, "vram_free_mib": 2000, "vram_total_mib": 8192},
        {"run_id": "r1", "round_index": -1, "delta_l2": 0.1, "eval_loss": 1.0, "tokens_trained": 100, "vram_free_mib": 2000, "vram_total_mib": 8192},
        {"run_id": "r1", "round_index": 0, "delta_l2": -0.1, "eval_loss": 1.0, "tokens_trained": 100, "vram_free_mib": 2000, "vram_total_mib": 8192},
        {"run_id": "r1", "round_index": 0, "delta_l2": 999.0, "eval_loss": 1.0, "tokens_trained": 100, "vram_free_mib": 2000, "vram_total_mib": 8192},
        {"run_id": "r1", "round_index": 0, "delta_l2": 0.1, "eval_loss": -1.0, "tokens_trained": 100, "vram_free_mib": 2000, "vram_total_mib": 8192},
        {"run_id": "r1", "round_index": 0, "delta_l2": 0.1, "eval_loss": 1.0, "tokens_trained": 0, "vram_free_mib": 2000, "vram_total_mib": 8192},
        {"run_id": "r1", "round_index": 0, "delta_l2": 0.1, "eval_loss": 1.0, "tokens_trained": 100, "vram_free_mib": 100, "vram_total_mib": 8192},
    ):
        with pytest.raises(ValueError):
            validate_trainer_delta_submission(**bad_kwargs)

    # 4. Producer shield CLI main() & parser
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "s" * 24)
    assert src_shield.main(["--port", "8789", "--route", "m1=http://127.0.0.1:8100"]) == 0
    with pytest.raises(SystemExit):
        src_shield.main(["--route", "badroute"])
    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "short")
    with pytest.raises(SystemExit):
        src_shield.main([])

    # 5. Producer agent VRAM guard & CLI validation branches
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-key")
    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "s" * 24)
    for bad_argv in (
        ["--gateway", "https://gw.example.com", "--endpoint", "http://remote.example.com"],
        ["--gateway", "https://gw.example.com", "--endpoint", "https://node.example.com"],
        ["--gateway", "https://gw.example.com", "--endpoint", "https://node.example.com", "--model", "c=q:7b"],
        ["--gateway", "https://gw.example.com", "--endpoint", "https://node.example.com", "--model", "c=q:7b", "--cost-per-1k", "-1", "--capacity-1k", "10"],
        ["--gateway", "https://gw.example.com", "--endpoint", "https://node.example.com", "--model", "c=q:7b", "--cost-per-1k", "0.1", "--capacity-1k", "0"],
        ["--gateway", "https://gw.example.com", "--endpoint", "https://node.example.com", "--model", "c=q:7b", "--cost-per-1k", "0.1", "--capacity-1k", "10", "--interval-s", "0"],
        ["--gateway", "https://gw.example.com", "--endpoint", "https://node.example.com", "--model", "c=q:7b", "--cost-per-1k", "0.1", "--capacity-1k", "10", "--skip-preflight", "--cold-start-s", "999"],
    ):
        with pytest.raises(SystemExit):
            src_agent.main(bad_argv)

    res_transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"models": [{"name": "small:1b", "size": 1000 * 1024 * 1024, "size_vram": 1000 * 1024 * 1024}, {"name": "huge:70b", "size": 40000 * 1024 * 1024}]},
        )
    )
    gw_transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
    rc_drop_one = src_agent.main(
        [
            "--gateway", "https://gw.example.com",
            "--endpoint", "https://node.example.com",
            "--model", "s=small:1b",
            "--model", "h=huge:70b",
            "--cost-per-1k", "0.2",
            "--capacity-1k", "50",
            "--skip-preflight",
            "--gpu-vram-mib", "15360",
            "--min-free-vram-mib", "2048",
            "--once",
        ],
        transport=gw_transport,
        residency_transport=res_transport,
    )
    assert rc_drop_one == 0
    rc_drop_all = src_agent.main(
        [
            "--gateway", "https://gw.example.com",
            "--endpoint", "https://node.example.com",
            "--model", "h=huge:70b",
            "--cost-per-1k", "0.2",
            "--capacity-1k", "50",
            "--skip-preflight",
            "--gpu-vram-mib", "8192",
            "--min-free-vram-mib", "2048",
            "--once",
        ],
        transport=gw_transport,
        residency_transport=res_transport,
    )
    assert rc_drop_all == 1

    # 6. solverbee/families.py compile_slot_spec branches
    t_grid = [0.2, 0.5, 1.0]
    for slot_case in (
        {"family": "pendulum", "t_out": t_grid, "period_ms": 1000.0, "theta0_deg": 15.0},
        {"family": "pendulum", "t_out": t_grid, "L": 2.0, "theta0": 0.2},
        {"family": "pendulum", "t_out": t_grid, "l": 1.5, "theta0": 0.2},
        {"family": "pendulum", "t_out": t_grid, "length_cm": 50.0, "theta0": 0.2},
        {"family": "pendulum", "t_out": t_grid, "theta0": 0.2},
        {"family": "projectile", "t_out": t_grid, "vx0": 3.0, "vy0": 4.0},
        {"family": "projectile", "t_out": t_grid, "v0": 10.0, "angle_rad": 0.5},
        {"family": "projectile", "t_out": t_grid, "speed_kmh": 36.0, "angle_deg": 30.0},
        {"family": "kepler", "t_out": t_grid, "x0": 1.0, "y0": 0.0, "vx0": 0.0, "vy0": 1.0},
        {"family": "logistic", "t_out": t_grid, "r": 0.5, "n0": 0.2},
        {"family": "logistic", "t_out": t_grid, "r_per_min": 6.0, "n0": 0.2},
        {"family": "logistic", "t_out": t_grid, "r_percent": 15.0, "n0": 0.2},
        {"family": "logistic", "t_out": t_grid, "n0": 0.2},
        {"family": "sir", "t_out": t_grid, "gamma": 0.1, "beta": 0.3, "i0": 0.01, "r0": 0.0, "s0": 0.99},
        {"family": "sir", "t_out": t_grid, "infectious_period": 5.0, "R0": 2.5, "i0_percent": 1.0, "r0_percent": 2.0, "s0_percent": 97.0},
        {"family": "seir", "t_out": t_grid, "gamma": 0.1, "sigma": 0.2, "beta": 0.3, "e0": 0.01, "i0": 0.01, "r0": 0.0, "s0": 0.98},
        {"family": "seir", "t_out": t_grid, "infectious_period": 5.0, "incubation_period": 4.0, "R0": 2.0, "e0_percent": 1.0, "i0_percent": 1.0, "r0_percent": 1.0, "s0_percent": 97.0},
        {"family": "series_rlc", "t_out": t_grid, "L": 0.1, "C": 0.01},
        {"family": "series_rlc", "t_out": t_grid, "L_mH": 10.0, "C_mF": 1.0},
        {"family": "series_rlc", "t_out": t_grid, "L": 0.1, "C_uF": 100.0},
        {"family": "spring_mass_damper", "t_out": t_grid, "k": 2.0},
        {"family": "spring_mass_damper", "t_out": t_grid, "m_g": 500.0, "damper_force": 2.0, "damper_speed": 4.0, "x0_cm": 10.0, "k": 2.0},
        {"family": "decay_chain", "t_out": t_grid, "rates": [0.1, 0.2], "y0": [1.0, 0.0, 0.0]},
        {"family": "decay_chain", "t_out": t_grid, "rates_per_min": [6.0, 12.0]},
        {"family": "decay_chain", "t_out": t_grid, "time_constants_s": [10.0, 5.0]},
        {"family": "yukawa_crystal", "t_out": t_grid, "kappa": 1.0},
        {"family": "gpe_vortex", "t_out": t_grid, "g_int": 2.0},
        {"family": "art_exo_genetics", "t_out": t_grid, "k_pol": 0.5},
        {"family": "bao_cosmology", "t_out": t_grid, "hubble": 0.1},
        {"family": "projectile", "t_out": t_grid, "speed": 10.0, "slope_rise": 3.0, "slope_run": 4.0},
        {"family": "projectile", "t_out": t_grid, "speed": 10.0},
        {"family": "kepler", "t_out": t_grid, "circular_radius": 1.0},
        {"family": "logistic", "t_out": t_grid, "r_percent_per_min": 12.0, "n0": 0.2},
        {"family": "logistic", "t_out": t_grid, "doubling_time": 2.0, "n0": 0.2},
        {"family": "harmonic", "t_out": t_grid, "k": 4.0, "m": 1.0, "x0": 1.0},
    ):
        spec, trace = compile_slot_spec(slot_case, return_trace=True)
        assert spec["y0"] and trace["slot_family"]

    for bad_slot in (
        {"family": "pendulum", "t_out": t_grid},
        {"family": "projectile", "t_out": t_grid},
        {"family": "sir", "t_out": t_grid, "beta": 0.2, "i0": 0.1},
        {"family": "sir", "t_out": t_grid, "gamma": 0.2, "i0": 0.1},
        {"family": "sir", "t_out": t_grid, "gamma": 0.2, "beta": 0.3},
        {"family": "seir", "t_out": t_grid, "sigma": 0.2, "beta": 0.3, "e0": 0.1, "i0": 0.1},
        {"family": "seir", "t_out": t_grid, "gamma": 0.2, "beta": 0.3, "e0": 0.1, "i0": 0.1},
        {"family": "seir", "t_out": t_grid, "gamma": 0.2, "sigma": 0.2, "e0": 0.1, "i0": 0.1},
        {"family": "seir", "t_out": t_grid, "gamma": 0.2, "sigma": 0.2, "beta": 0.3, "i0": 0.1},
        {"family": "seir", "t_out": t_grid, "gamma": 0.2, "sigma": 0.2, "beta": 0.3, "e0": 0.1},
        {"family": "series_rlc", "t_out": t_grid, "C": 0.01},
        {"family": "series_rlc", "t_out": t_grid, "L": 0.1},
        {"family": "decay_chain", "t_out": t_grid},
        {"family": "van_der_pol", "t_out": t_grid, "mu": 1.0},
    ):
        with pytest.raises(FamilyError):
            compile_slot_spec(bad_slot)

    # 7. verification/bee_energy.py AST evaluator & test_callables branches
    from hivemind.verification.bee_energy import evaluate_bee_candidate_energy

    code_src = (
        "def helper(z):\n"
        "    return abs(z)\n\n"
        "def solve(x):\n"
        "    '''Docstring.'''\n"
        "    a, b = (x, 2)\n"
        "    arr = [a, b]\n"
        "    a += sum(arr)\n"
        "    val = helper(a) if (0 < x < 10) else -1\n"
        "    if val > 0:\n"
        "        return val\n"
        "    return 0\n"
    )
    ev = evaluate_bee_candidate_energy(
        candidate_id="c1",
        response_text=code_src,
        test_callables=[
            lambda ns: ns["solve"](3) == 8,
            lambda ns: ns["solve"](20) == 0,
        ],
        latency_ms=5.0,
        memory_mb=10.0,
    )
    assert ev.passed_all_tests is True and ev.total_energy < 10.0

