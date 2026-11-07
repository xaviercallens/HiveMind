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
    """MVP Audit C4: scripts/hivemind_producer_{agent,shield}.py re-export src/hivemind/router/... and gateway serves them."""
    agent_path = REPO_ROOT / "scripts" / "hivemind_producer_agent.py"
    shield_path = REPO_ROOT / "scripts" / "hivemind_producer_shield.py"

    spec_a = importlib.util.spec_from_file_location("scripts_agent_check", agent_path)
    assert spec_a is not None and spec_a.loader is not None
    mod_a = importlib.util.module_from_spec(spec_a)
    spec_a.loader.exec_module(mod_a)

    spec_s = importlib.util.spec_from_file_location("scripts_shield_check", shield_path)
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
        assert getattr(mod_a, attr) is getattr(src_agent, attr), f"diverged attr on agent: {attr}"

    for attr in ("main", "build_parser", "create_shield_app", "POC_NOTICE"):
        assert getattr(mod_s, attr) is getattr(src_shield, attr), f"diverged attr on shield: {attr}"

    store = RouterStore(tmp_path / "router.sqlite3", secret=b"s" * 32)
    client = TestClient(create_router_app(store))
    r_agent = client.get("/hivemind_producer_agent.py")
    r_shield = client.get("/hivemind_producer_shield.py")
    assert r_agent.status_code == 200
    assert r_shield.status_code == 200
    assert r_agent.text == (REPO_ROOT / "src" / "hivemind" / "router" / "hivemind_producer_agent.py").read_text(encoding="utf-8")
    assert r_shield.text == (REPO_ROOT / "src" / "hivemind" / "router" / "hivemind_producer_shield.py").read_text(encoding="utf-8")


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
