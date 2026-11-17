# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-02-D2 (Consent & Pause Lean invariants + one-heartbeat integration)
and PS-10-D3 (Chain Base Weights Mirror Receipt with licence files and SHA-256 checksums)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from hivemind.router import hivemind_producer_agent as agent
from hivemind.router import hivemind_producer_shield as shield
import scripts.base_weights_mirror_receipt as mirror_receipt

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_SPEC_PATH = REPO_ROOT / "lean4" / "HiveMindSpec.lean"


def _count_sorry_tokens(lean_text: str) -> int:
    return lean_text.count("sorry")


def test_ps02_d2_lean_consent_and_pause_invariants_zero_sorry() -> None:
    """PS-02-D2: Consent and pause invariants exist in HiveMindSpec.lean with zero sorry."""
    text = LEAN_SPEC_PATH.read_text(encoding="utf-8")
    assert _count_sorry_tokens(text) == 0, "HiveMindSpec.lean must contain 0 occurrences of 'sorry'"
    assert "namespace Season30_VolunteerConsentAndPauseInvariants" in text
    assert "theorem no_node_starts_without_stored_owner_enrolment" in text
    assert "theorem pause_implies_no_new_work_within_one_heartbeat" in text
    assert (
        "#print axioms HiveMindSpec.Season30_VolunteerConsentAndPauseInvariants.no_node_starts_without_stored_owner_enrolment"
        in text
    )
    assert (
        "#print axioms HiveMindSpec.Season30_VolunteerConsentAndPauseInvariants.pause_implies_no_new_work_within_one_heartbeat"
        in text
    )


def test_ps02_d2_negative_control_sorry_detector_flags_sorry() -> None:
    """Negative control: a planted sorry token is caught by the zero-sorry scanner."""
    planted = "theorem bad : True := by\n  sorry\n"
    assert _count_sorry_tokens(planted) == 1


def test_ps02_d2_paused_producer_serves_nothing_within_one_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PS-02-D2 integration: within one heartbeat of receiving paused=True, shield serves 0 upstream requests."""
    upstream_calls: list[dict[str, Any]] = []

    async def fake_chat(request: Request) -> JSONResponse:
        body = await request.json()
        upstream_calls.append(body)
        return JSONResponse(
            {
                "id": "chatcmpl-ps02-d2",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            }
        )

    upstream_app = Starlette(routes=[Route("/v1/chat/completions", fake_chat, methods=["POST"])])
    transport = httpx.ASGITransport(app=upstream_app)
    real_async_client = httpx.AsyncClient

    def _patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(shield.httpx, "AsyncClient", _patched_async_client)

    control_file = tmp_path / "producer_control.json"
    token = "secret-shield-token-ps02-d2"
    app = shield.create_shield_app(
        upstream="http://127.0.0.1:11434",
        token=token,
        allowed_models={"qwen2.5-coder:7b"},
        control_state_file=str(control_file),
    )

    headers = {"Authorization": f"Bearer {token}"}
    req_body = {
        "model": "qwen2.5-coder:7b",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 4,
    }

    with TestClient(app) as client:
        # Heartbeat 0: producer is unpaused (paused=False)
        hb0_control = agent.parse_control({"paused": False, "vram_cap_mib": 6144, "version": 1})
        agent.write_control_state(str(control_file), hb0_control)
        hb0_payload = agent.apply_control(
            {"id": "node-ps02", "capacity_1k": 60.0, "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b"}]},
            hb0_control,
        )
        assert hb0_payload["capacity_1k"] == 60.0
        assert not hb0_payload.get("paused", False)

        resp0 = client.post("/v1/chat/completions", headers=headers, json=req_body)
        assert resp0.status_code == 200
        assert len(upstream_calls) == 1

        # Heartbeat 1: coordinator ack returns paused=True; agent writes control state and advertises 0 capacity
        hb1_control = agent.parse_control({"paused": True, "vram_cap_mib": 6144, "version": 2})
        agent.write_control_state(str(control_file), hb1_control)
        hb1_payload = agent.apply_control(
            {"id": "node-ps02", "capacity_1k": 60.0, "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b"}]},
            hb1_control,
        )
        assert hb1_payload["capacity_1k"] == 0.0
        assert hb1_payload["paused"] is True

        # Within that single heartbeat, shield rejects new work (503 producer_paused) and forwards 0 requests upstream
        for _ in range(3):
            resp_paused = client.post("/v1/chat/completions", headers=headers, json=req_body)
            assert resp_paused.status_code == 503
            assert resp_paused.json()["error"]["code"] == "producer_paused"
        assert len(upstream_calls) == 1, "Paused producer must serve zero upstream requests after pause heartbeat"

        # Heartbeat 2: coordinator unpauses (paused=False); serving resumes
        hb2_control = agent.parse_control({"paused": False, "vram_cap_mib": 6144, "version": 3})
        agent.write_control_state(str(control_file), hb2_control)
        resp2 = client.post("/v1/chat/completions", headers=headers, json=req_body)
        assert resp2.status_code == 200
        assert len(upstream_calls) == 2


def test_ps10_d3_base_weights_mirror_receipt_positive_and_artifact() -> None:
    """PS-10-D3: Every chain base model has a mirror path, licence, licence_file, and 64-hex sha256."""
    receipt = mirror_receipt.build_base_weights_mirror_receipt()
    assert receipt["deliverable_id"] == "PS-10-D3"
    assert receipt["all_chain_models_mirrored"] is True
    assert receipt["violations"] == []
    assert mirror_receipt.validate_base_weights_mirror_receipt(receipt) == []

    by_id = {m["model_id"]: m for m in receipt["base_models"]}
    for required_id in (
        "ibm-granite/granite-3.1-1b-a400m-base",
        "ibm-granite/granite-3.1-3b-a800m-base",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "google/gemma-2-2b-it",
    ):
        assert required_id in by_id
        entry = by_id[required_id]
        assert entry["mirror_path"].startswith("gs://")
        assert entry["licence"] in {"Apache-2.0", "Gemma-Terms-of-Use"}
        assert entry["licence_file"].endswith("/LICENSE")
        assert len(entry["sha256"]) == 64

    artifact_path = REPO_ROOT / "artifacts" / "experiments" / "BASE_WEIGHTS_MIRROR_2026-11-07.json"
    assert artifact_path.is_file()
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert mirror_receipt.validate_base_weights_mirror_receipt(saved) == []


def test_ps10_d3_base_weights_mirror_receipt_negative_controls() -> None:
    """PS-10-D3 negative controls: missing mirror_path, licence, or sha256 fails validation."""
    base = mirror_receipt.build_base_weights_mirror_receipt()

    # 1. Missing mirror_path
    bad_mirror = copy.deepcopy(base)
    bad_mirror["base_models"][0]["mirror_path"] = ""
    v1 = mirror_receipt.validate_base_weights_mirror_receipt(bad_mirror)
    assert any("mirror_path" in msg for msg in v1)

    # 2. Missing licence
    bad_licence = copy.deepcopy(base)
    bad_licence["base_models"][1]["licence"] = ""
    v2 = mirror_receipt.validate_base_weights_mirror_receipt(bad_licence)
    assert any("missing licence" in msg for msg in v2)

    # 3. Corrupted sha256
    bad_sha = copy.deepcopy(base)
    bad_sha["base_models"][2]["sha256"] = "not-a-valid-sha256"
    v3 = mirror_receipt.validate_base_weights_mirror_receipt(bad_sha)
    assert any("invalid sha256" in msg for msg in v3)

    # 4. Missing required chain base model
    missing_req = copy.deepcopy(base)
    missing_req["base_models"] = [
        m for m in missing_req["base_models"] if m["model_id"] != "ibm-granite/granite-3.1-1b-a400m-base"
    ]
    v4 = mirror_receipt.validate_base_weights_mirror_receipt(missing_req)
    assert any("ibm-granite/granite-3.1-1b-a400m-base" in msg for msg in v4)
