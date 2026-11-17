# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and integration tests for HM-S19-USAGE: privacy-preserving usage capture, smart auto-intent routing, and feedback."""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.usage_capture import (
    classify_prompt_intent_and_language,
    compute_nocturnal_moe_workunit_weights,
    detect_client_sdk,
)

SECRET = b"usage-capture-test-secret-32-bytes!!"


class _FakeProducer:
    def __init__(self, answer: str = "ok") -> None:
        self.answer = answer

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-test",
                "object": "chat.completion",
                "model": body.get("model", "local"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
            },
        )


def test_classify_prompt_intent_and_language_domains_and_fr_en() -> None:
    # 1. Rusty-SUNDIALS stiff ODE/PDE in English
    sundials = classify_prompt_intent_and_language(
        [{"role": "user", "content": "Solve this stiff ODE Navier-Stokes vorticity system using SUNDIALS CVODE BDF."}]
    )
    assert sundials.task_domain == "science_rusty_sundials"
    assert sundials.language == "en"
    assert sundials.preferred_tier == "bee-rusty-sundials"

    # 2. Astronomy in French
    astro_fr = classify_prompt_intent_and_language(
        [{"role": "user", "content": "Bonjour, peux-tu expliquer comment calculer l'orbite Kepler d'une exoplanet en astrophysique ?"}]
    )
    assert astro_fr.task_domain == "science_astronomy"
    assert astro_fr.language == "fr"
    assert astro_fr.preferred_tier == "bee-stars"

    # 3. Python coding
    py_code = classify_prompt_intent_and_language(
        [{"role": "user", "content": "Write a FastAPI endpoint in Python with pytest unit tests and asyncio."}]
    )
    assert py_code.task_domain == "coding_python"
    assert py_code.preferred_tier == "coder"

    # 4. Rust coding
    rust_code = classify_prompt_intent_and_language(
        [{"role": "user", "content": "Implement a zero-copy Cargo crate in Rust using tokio and #[derive(Clone)]."}]
    )
    assert rust_code.task_domain == "coding_rust"
    assert rust_code.preferred_tier == "coder"

    # 5. General chat has no specialist override so auto-supply fallback still applies
    general = classify_prompt_intent_and_language([{"role": "user", "content": "hello"}])
    assert general.task_domain == "general_chat"
    assert general.preferred_tier is None


def test_detect_client_sdk_parses_openrouter_and_ide_headers() -> None:
    assert detect_client_sdk({"x-title": "HiveMind-Playground"}).client_sdk == "playground_web"
    assert detect_client_sdk({"user-agent": "Cursor/0.42.0"}).client_sdk == "cursor"
    assert detect_client_sdk({"user-agent": "Continue-Dev/1.0"}).client_sdk == "continue_dev"
    assert detect_client_sdk({"x-title": "Cline VSCode"}).client_sdk == "cline_roo"
    assert detect_client_sdk({"x-title": "opencode"}).client_sdk == "opencode"
    assert detect_client_sdk({"user-agent": "aider/0.75"}).client_sdk == "aider"
    assert detect_client_sdk({"user-agent": "google-antigravity/2.0"}).client_sdk == "antigravity_sdk"
    assert detect_client_sdk({"user-agent": "OpenAI/Python 1.52.0"}).client_sdk == "openai_python"
    assert detect_client_sdk({"user-agent": "curl/8.5.0"}).client_sdk == "curl"


def test_smart_auto_intent_routing_and_usage_capture_telemetry() -> None:
    store = RouterStore(":memory:", SECRET)
    fake = _FakeProducer()
    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(fake.handle)),
        allow_private_producers=True,
        starter_grant_hmt=50.0,
    )
    client = TestClient(app)

    # Register producer serving coder (high capacity), bee-stars, and bee-rusty-sundials
    prod = client.post("/v1/hivemind/register", json={"handle": "prod1", "role": "prosumer"}).json()
    prod_key = prod["api_key"]
    hb = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {prod_key}"},
        json={
            "endpoint_url": "https://gpu.test",
            "models": [
                {"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b"},
                {"id": "hivemind/bee-stars", "local_name": "hivemind-bee-stars:4b"},
                {"id": "hivemind/bee-rusty-sundials", "local_name": "hivemind-bee-rusty-sundials:7b"},
            ],
            "cost_per_1k": 0.4,
            "capacity_1k": 200.0,
            "execution_class": "rtx-2070",
        },
    )
    assert hb.status_code == 200

    # Register real user consumer
    user = client.post("/v1/hivemind/register", json={"handle": "alice-fr", "role": "consumer"}).json()
    user_key = user["api_key"]

    # 1. Call hivemind/auto with a Rusty-SUNDIALS stiff ODE prompt from Continue IDE
    secret_marker = "SUPER_SECRET_USER_PROMPT_DO_NOT_STORE_99182"
    r_sundials = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {user_key}", "X-Title": "Continue"},
        json={
            "model": "hivemind/auto",
            "messages": [
                {
                    "role": "user",
                    "content": f"Solve this stiff ODE Navier-Stokes system with SUNDIALS CVODE BDF ({secret_marker}).",
                }
            ],
        },
    )
    assert r_sundials.status_code == 200
    body1: dict[str, Any] = r_sundials.json()
    assert body1["model"] == "hivemind/bee-rusty-sundials"

    # 2. Call hivemind/auto with a French astronomy prompt from Playground
    r_astro = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {user_key}", "X-Title": "HiveMind-Playground"},
        json={
            "model": "hivemind/auto",
            "messages": [
                {
                    "role": "user",
                    "content": "Bonjour, explique le calcul d'une orbite Kepler pour une exoplanet en astrophysique.",
                }
            ],
        },
    )
    assert r_astro.status_code == 200
    body2 = r_astro.json()
    assert body2["model"] == "hivemind/bee-stars"

    # 3. Submit thumbs-up feedback on the first request
    req_id = body1["hivemind"]["request_id"]
    fb = client.post(
        "/v1/hivemind/feedback",
        headers={"Authorization": f"Bearer {user_key}"},
        json={"request_id": req_id, "rating": 1, "comment": "Accurate CVODE BDF step advice"},
    )
    assert fb.status_code == 200
    assert fb.json()["rating"] == 1
    assert fb.json()["task_domain"] == "science_rusty_sundials"

    # 4. Query public aggregate usage-capture endpoint
    summary_res = client.get("/v1/hivemind/usage-capture")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["total_requests"] == 2
    assert summary["by_task_domain"]["science_rusty_sundials"]["requests"] == 1
    assert summary["by_task_domain"]["science_astronomy"]["requests"] == 1
    assert summary["by_language"]["en"] == 1
    assert summary["by_language"]["fr"] == 1
    assert summary["by_client_sdk"]["continue_dev"] == 1
    assert summary["by_client_sdk"]["playground_web"] == 1
    assert summary["by_routing_mode"]["auto_intent"] == 2
    assert summary["feedback"]["thumbs_up"] == 1
    assert summary["feedback"]["satisfaction_rate"] == 1.0
    assert "hivemind/bee-rusty-sundials" in summary["nocturnal_moe_workunit_weights"]

    # Privacy invariant: raw prompt text is never stored or exposed
    assert secret_marker not in summary_res.text


def test_portal_playground_exposes_sundials_and_french_presets() -> None:
    store = RouterStore(":memory:", SECRET)
    app = create_router_app(store)
    client = TestClient(app)
    html = client.get("/playground").text
    assert 'data-prompt="sundials"' in html
    assert 'data-prompt="francais"' in html
    assert "hivemind/bee-rusty-sundials" in html
    assert "HiveMind-Playground" in html


def test_compute_nocturnal_moe_workunit_weights_sums_to_one() -> None:
    weights = compute_nocturnal_moe_workunit_weights(
        {"science_rusty_sundials": 5000, "coding_python": 3000, "science_astronomy": 2000}
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-3
    assert weights["hivemind/bee-rusty-sundials"] > weights["hivemind/bee-stars"]
