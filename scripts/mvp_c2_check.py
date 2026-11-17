#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion C2 check: expert registry wired into gateway domain routing & train-to-serve gate enforcement.

Exercises the real FastAPI gateway (``create_router_app``) end-to-end with an HTTP producer mock transport:
  1. Registers a domain LoRA expert (`boinc-granite-astro`, `science_astronomy`, keywords `exoplanet`, `kepler`, `transit`)
     in `SHADOW` state; checks the real failed gate artifact `GRANITE_ADAPTER_hmq_00147_gate_f16.json` (`servable: false`)
     and a mismatched SHA-256 gate are both refused (`HTTP 400 gate_not_passed`).
  2. Verifies a `SHADOW` (ungated) or unregistered expert request is refused (`HTTP 403` / `HTTP 404`) and `hivemind/auto`
     does not route to an unpromoted adapter.
  3. Promotes the expert via `POST /v1/hivemind/experts/promote` with a matching `servable: true` gate for its exact SHA-256
     and base model, then verifies `hivemind/auto` routes matching astronomy prompts to that expert (`expert_registry_live`)
     while unrelated prompts fall through to normal routing without attaching the domain expert.

Writes ``artifacts/mvp/C2_expert_routing.json``.

    python scripts/mvp_c2_check.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.gateway import create_router_app  # noqa: E402

SECRET = b"c2-expert-routing-check-secret-32b"
ADMIN_KEY = "hm-admin-c2-check-key"
BASE_MODEL = "ibm-granite/granite-3.1-1b-a400m-base"
EXPERT_ID = "boinc-granite-astro"
CLUSTER_ID = "science_astronomy"


def _upstream_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "cmpl-c2-check",
            "object": "chat.completion",
            "model": body.get("model", "local"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        },
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/mvp/C2_expert_routing.json"))
    args = ap.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="c2_"))
    store = RouterStore(work / "router.db", secret=SECRET)
    now = 1_794_900_000.0

    _, prod_key = store.register("prod-granite", role="producer")
    _, cons_key = store.register("cons-c2", role="consumer")
    store.grant(store.account_id_for("cons-c2"), 50.0, "c2-check-grant")

    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_upstream_handler)),
        allow_private_producers=True,
        admin_key_hash=store.key_digest(ADMIN_KEY),
        clock=lambda: now,
    )
    client = TestClient(app)

    # Producer advertises base tier + attempted unpromoted adapter model name + coder fallback
    hb = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {prod_key}"},
        json={
            "endpoint_url": "https://producer.example.invalid",
            "models": [
                {"id": "hivemind/granite-moe", "local_name": "granite3.1-moe:3b", "loaded": True},
                {"id": f"hivemind/{EXPERT_ID}", "local_name": EXPERT_ID, "loaded": True},
                {"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True},
            ],
            "cost_per_1k": 0.25,
            "capacity_1k": 60.0,
        },
    )

    adapter_sha = hashlib.sha256(b"verified-gguf-astro-adapter-v1").hexdigest()
    reg_resp = client.post(
        "/v1/hivemind/experts/register",
        headers={"x-admin-key": ADMIN_KEY},
        json={
            "expert_id": EXPERT_ID,
            "kind": "lora_domain",
            "cluster_id": CLUSTER_ID,
            "base_model_id": BASE_MODEL,
            "adapter_sha256": adapter_sha,
            "adapter_size_mib": 16,
            "keywords": ["exoplanet", "kepler", "transit"],
            "source": "scripts/mvp_c2_check.py",
            "move_to_shadow": True,
        },
    )

    real_gate_path = ROOT / "artifacts/experiments/GRANITE_ADAPTER_hmq_00147_gate_f16.json"
    real_gate = json.loads(real_gate_path.read_text(encoding="utf-8"))
    failed_gate_resp = client.post(
        "/v1/hivemind/experts/promote",
        headers={"x-admin-key": ADMIN_KEY},
        json={"expert_id": EXPERT_ID, "gate": real_gate},
    )
    wrong_sha_resp = client.post(
        "/v1/hivemind/experts/promote",
        headers={"x-admin-key": ADMIN_KEY},
        json={
            "expert_id": EXPERT_ID,
            "gate": {
                "servable": True,
                "base": BASE_MODEL,
                "adapter": {"lora_gguf_sha256": hashlib.sha256(b"different-bytes").hexdigest()},
                "pooled": {"mean_nll_gain_nats": 0.03, "bootstrap95": [0.01, 0.05]},
                "worst_domain_regression": 0.001,
            },
        },
    )

    shadow_explicit_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": f"hivemind/{EXPERT_ID}", "messages": [{"role": "user", "content": "Explain Kepler exoplanet transit timing."}]},
    )
    unregistered_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/unregistered-domain-adapter", "messages": [{"role": "user", "content": "Hello"}]},
    )
    auto_before_promote = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/auto", "messages": [{"role": "user", "content": "Explain Kepler exoplanet transit photometry."}]},
    )

    passing_gate_resp = client.post(
        "/v1/hivemind/experts/promote",
        headers={"x-admin-key": ADMIN_KEY},
        json={
            "expert_id": EXPERT_ID,
            "gate": {
                "servable": True,
                "base": BASE_MODEL,
                "adapter": {"lora_gguf_sha256": adapter_sha},
                "pooled": {"mean_nll_gain_nats": 0.032, "bootstrap95": [0.014, 0.049]},
                "worst_domain_regression": 0.001,
            },
        },
    )

    auto_domain_routed = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/auto", "messages": [{"role": "user", "content": "Explain Kepler exoplanet transit photometry."}]},
    )
    auto_unrelated = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {cons_key}"},
        json={"model": "hivemind/auto", "messages": [{"role": "user", "content": "Write a tiny Python function to reverse a list."}]},
    )

    routed_hm = auto_domain_routed.json().get("hivemind", {}) if auto_domain_routed.status_code == 200 else {}
    unrelated_hm = auto_unrelated.json().get("hivemind", {}) if auto_unrelated.status_code == 200 else {}
    before_hm = auto_before_promote.json().get("hivemind", {}) if auto_before_promote.status_code == 200 else {}

    checks = {
        "heartbeat_ok": hb.status_code == 200,
        "register_shadow_ok": reg_resp.status_code == 200 and reg_resp.json()["expert"]["state"] == "shadow",
        "failed_gate_refused": failed_gate_resp.status_code == 400 and failed_gate_resp.json()["error"]["code"] == "gate_not_passed",
        "wrong_sha_gate_refused": wrong_sha_resp.status_code == 400 and wrong_sha_resp.json()["error"]["code"] == "gate_not_passed",
        "shadow_expert_request_refused": shadow_explicit_resp.status_code == 403 and shadow_explicit_resp.json()["error"]["code"] == "expert_not_servable",
        "unregistered_adapter_refused": unregistered_resp.status_code == 404 and unregistered_resp.json()["error"]["code"] == "model_not_found",
        "auto_skips_unpromoted_shadow_expert": auto_before_promote.status_code == 200 and before_hm.get("expert_id") is None,
        "passing_gate_promotes_live": passing_gate_resp.status_code == 200 and passing_gate_resp.json()["expert"]["state"] == "live",
        "auto_routes_domain_prompt_to_live_expert": (
            auto_domain_routed.status_code == 200
            and routed_hm.get("expert_id") == EXPERT_ID
            and routed_hm.get("cluster_id") == CLUSTER_ID
            and routed_hm.get("routing_mode") == "expert_registry_live"
            and auto_domain_routed.headers.get("X-HiveMind-Expert-Id") == EXPERT_ID
        ),
        "unrelated_prompt_does_not_hit_domain_expert": auto_unrelated.status_code == 200 and unrelated_hm.get("expert_id") is None,
    }

    result = {
        "criterion": "C2",
        "what": "Registered adapter is routed by domain; failed-gate and unregistered adapters refused in gateway router",
        "transport": "FastAPI TestClient + httpx.MockTransport upstream producer stub",
        "real_failed_gate_tested": str(real_gate_path.relative_to(ROOT)),
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
        "reason": "" if all(checks.values()) else f"failed checks: {[k for k, v in checks.items() if not v]}",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": checks}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
