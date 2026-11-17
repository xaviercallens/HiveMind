# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Gateway integration tests for MVP Criterion C2: ExpertRegistry gate enforcement & domain routing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from hivemind.core.expert_registry import CANDIDATE, LIVE, SHADOW, ExpertRecord, ExpertRegistry
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

SECRET = b"gateway-c2-routing-test-secret-32b!"
ADMIN_KEY = "hm-admin-c2-test-key"
BASE_MODEL = "ibm-granite/granite-3.1-1b-a400m-base"


def _make_upstream(captured_models: list[str] | None = None, *, stream: bool = False):
    def _handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if captured_models is not None:
            captured_models.append(str(body.get("model", "")))
        if stream or body.get("stream"):
            sse = (
                'data: {"id":"cmpl-s1","choices":[{"delta":{"content":"hello"},"index":0}]}\n\n'
                'data: {"id":"cmpl-s1","choices":[],"usage":{"prompt_tokens":6,"completion_tokens":3,"total_tokens":9}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "cmpl-c2",
                "object": "chat.completion",
                "model": body.get("model", "local"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(_handle))


def _setup_app(tmp_path: Path, *, registry: ExpertRegistry | None = None, captured: list[str] | None = None):
    store = RouterStore(tmp_path / "router.db", secret=SECRET)
    _, prod_key = store.register("prod-c2", role="producer")
    _, cons_key = store.register("cons-c2", role="consumer")
    store.grant(store.account_id_for("cons-c2"), 50.0, "grant")
    now = 1_794_900_000.0
    app = create_router_app(
        store,
        http_client=_make_upstream(captured),
        allow_private_producers=True,
        admin_key_hash=store.key_digest(ADMIN_KEY),
        clock=lambda: now,
        expert_registry=registry,
    )
    client = TestClient(app)
    hb = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {prod_key}"},
        json={
            "endpoint_url": "https://producer.example.invalid",
            "models": [
                {"id": "hivemind/granite-moe", "local_name": "granite3.1-moe:3b", "loaded": True},
                {"id": "hivemind/boinc-astro", "local_name": "boinc-astro-lora", "loaded": True},
                {"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True},
            ],
            "cost_per_1k": 0.25,
            "capacity_1k": 60.0,
        },
    )
    assert hb.status_code == 200
    return client, cons_key


def test_c2_admin_endpoints_reject_failed_gate_and_non_admin(tmp_path: Path) -> None:
    client, cons_key = _setup_app(tmp_path)
    sha = hashlib.sha256(b"astro-adapter-bytes").hexdigest()

    # Non-admin cannot register or promote
    unauth = client.post(
        "/v1/hivemind/experts/register",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={
            "expert_id": "boinc-astro",
            "kind": "lora_domain",
            "cluster_id": "science_astronomy",
            "base_model_id": BASE_MODEL,
            "adapter_sha256": sha,
            "adapter_size_mib": 12,
            "keywords": ["exoplanet", "kepler"],
            "move_to_shadow": True,
        },
    )
    assert unauth.status_code == 403

    reg = client.post(
        "/v1/hivemind/experts/register",
        headers={"x-admin-key": ADMIN_KEY},
        json={
            "expert_id": "boinc-astro",
            "kind": "lora_domain",
            "cluster_id": "science_astronomy",
            "base_model_id": BASE_MODEL,
            "adapter_sha256": sha,
            "adapter_size_mib": 12,
            "keywords": ["exoplanet", "kepler"],
            "move_to_shadow": True,
        },
    )
    assert reg.status_code == 200
    assert reg.json()["expert"]["state"] == SHADOW

    # Failed real gate artifact must be refused
    real_gate = json.loads((Path(__file__).resolve().parents[1] / "artifacts/experiments/GRANITE_ADAPTER_hmq_00147_gate_f16.json").read_text())
    bad = client.post("/v1/hivemind/experts/promote", headers={"x-admin-key": ADMIN_KEY}, json={"expert_id": "boinc-astro", "gate": real_gate})
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "gate_not_passed"

    listed = client.get("/v1/hivemind/experts").json()
    assert listed["live_count"] == 0
    assert listed["experts"][0]["state"] == SHADOW


def test_c2_shadow_and_unregistered_experts_refused_while_live_routes_by_domain(tmp_path: Path) -> None:
    captured: list[str] = []
    client, cons_key = _setup_app(tmp_path, captured=captured)
    sha = hashlib.sha256(b"astro-adapter-v2").hexdigest()

    client.post(
        "/v1/hivemind/experts/register",
        headers={"x-admin-key": ADMIN_KEY},
        json={
            "expert_id": "boinc-astro",
            "kind": "lora_domain",
            "cluster_id": "science_astronomy",
            "base_model_id": BASE_MODEL,
            "adapter_sha256": sha,
            "adapter_size_mib": 12,
            "keywords": ["exoplanet", "kepler", "transit"],
            "move_to_shadow": True,
        },
    )

    # Even though producer advertised hivemind/boinc-astro, requesting it in SHADOW returns 403
    refused = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/boinc-astro", "messages": [{"role": "user", "content": "Kepler exoplanet transit?"}]},
    )
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "expert_not_servable"

    # Unregistered adapter is 404
    missing = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/never-registered-adapter", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert missing.status_code == 404

    # Promote with matching gate
    promoted = client.post(
        "/v1/hivemind/experts/promote",
        headers={"x-admin-key": ADMIN_KEY},
        json={
            "expert_id": "boinc-astro",
            "gate": {
                "servable": True,
                "base": BASE_MODEL,
                "adapter": {"lora_gguf_sha256": sha},
                "pooled": {"mean_nll_gain_nats": 0.04, "bootstrap95": [0.02, 0.06]},
                "worst_domain_regression": 0.0,
            },
        },
    )
    assert promoted.status_code == 200
    assert promoted.json()["expert"]["state"] == LIVE

    # Matching domain prompt routes through hivemind/auto to boinc-astro
    ok_auto = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/auto", "messages": [{"role": "user", "content": "Compute Kepler exoplanet transit depth."}]},
    )
    assert ok_auto.status_code == 200
    hm = ok_auto.json()["hivemind"]
    assert hm["expert_id"] == "boinc-astro"
    assert hm["cluster_id"] == "science_astronomy"
    assert hm["routing_mode"] == "expert_registry_live"
    assert ok_auto.headers["X-HiveMind-Expert-Id"] == "boinc-astro"
    assert captured[-1] == "boinc-astro-lora"

    # Streaming response also carries expert headers
    stream_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/auto", "stream": True, "messages": [{"role": "user", "content": "Analyze exoplanet transit curve."}]},
    )
    assert stream_resp.status_code == 200
    assert stream_resp.headers["X-HiveMind-Expert-Id"] == "boinc-astro"
    assert stream_resp.headers["X-HiveMind-Cluster-Id"] == "science_astronomy"

    # Unrelated prompt does not get routed to boinc-astro
    code_auto = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/auto", "messages": [{"role": "user", "content": "Write a Python sorting helper."}]},
    )
    assert code_auto.status_code == 200
    assert "expert_id" not in code_auto.json()["hivemind"]


def test_c2_explicit_live_expert_falls_back_to_base_tier_alias(tmp_path: Path) -> None:
    sha = hashlib.sha256(b"granite-base-bound").hexdigest()
    reg = ExpertRegistry(strict=True)
    reg.register(ExpertRecord("boinc-astro-only", "lora_domain", "science_astronomy", BASE_MODEL, adapter_sha256=sha, adapter_size_mib=8, keywords=("exoplanet",)))
    reg.transition("boinc-astro-only", SHADOW, "shadow")
    reg.promote_with_gate("boinc-astro-only", {"servable": True, "base": BASE_MODEL, "adapter": {"lora_gguf_sha256": sha}})

    captured: list[str] = []
    client, cons_key = _setup_app(tmp_path, registry=reg, captured=captured)
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/boinc-astro-only", "messages": [{"role": "user", "content": "Tell me about an exoplanet."}]},
    )
    assert resp.status_code == 200
    assert resp.json()["hivemind"]["expert_id"] == "boinc-astro-only"
    assert resp.json()["hivemind"]["routing_mode"] == "expert_registry_explicit"
    assert captured[-1] == "granite3.1-moe:3b"


def test_c2_script_runs_and_writes_passing_artifact(tmp_path: Path) -> None:
    import scripts.mvp_c2_check as c2

    out = tmp_path / "C2_expert_routing.json"
    assert c2.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True
