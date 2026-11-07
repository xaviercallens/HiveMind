# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for PR #698 (Iteration 31):
- IP-29: Exoplanet Hunter game server `serve_inference` respects `max_tokens` and reports honest local producer (`PRODUCER-LOCAL-RTX2070-CUDA0`).
- IP-32: `sync_game_with_la_ruche_live` does not auto-register accounts unless explicitly requested and stores session outside git with `0o600`.
- IP-39: Lean theorem count is computed from `lean4/HiveMindSpec.lean` rather than asserted as a stale literal.
- IP-45: `LIFE_AS_A_BEE_SCENARIOS` `underlying_tag` matches `TIERS` in `catalog.py`, and `summary` fields are derived from scenario data.
- IP-54: `steam_contribute_boinc` and `webgpu_contribute` offload synchronous blocking work via `run_in_threadpool`.
- IP-55: `/v1/hivemind/bees/swat-telepathy/plan` validates JSON body and finite numeric bounds, returning HTTP 400 instead of 500.
- IP-56: Header-only authentication (`Authorization: Bearer` / `X-API-Key`); query `?api_key=` and `/k/{api_key}/...` URLs are rejected, and `AntigravityHiveMindCoPilot.authenticated_base_url` embeds no key.
- IP-58: `/v1/hivemind/bees/coordinate` and `/v1/hivemind/bees/trade` enforce `check_publish` and `amcp_publish_limiter`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import _count_lean4_spec_theorems
from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app
from hivemind.game.exoplanet_hunter_engine import (
    AstroBeeMoEGPUService,
    HiveMindMultiProducerInferencePool,
)
from hivemind.game.scientific_discovery_game import (
    _default_la_ruche_session_path,
    sync_game_with_la_ruche_live,
)
from hivemind.router.accounts import RouterStore
from hivemind.router.catalog import TIERS
from hivemind.router.gateway import create_router_app
from hivemind.router.life_as_a_bee import (
    LIFE_AS_A_BEE_SCENARIOS,
    BeeScenarioResult,
    life_as_a_bee_payload,
)
from hivemind.swarm.s32_01_verlet_kepler442b_copilot import (
    DEFAULT_HIVEMIND_BASE_URL,
    AntigravityHiveMindCoPilot,
)

SECRET = b"ip31-router-game-lean-catalog-32b"


# ---------------------------------------------------------------------------
# IP-29: Exoplanet Hunter game server respects max_tokens and honest producer
# ---------------------------------------------------------------------------
def test_ip29_chat_completion_tokens_and_served_by_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hivemind.game.exoplanet_hunter_cockpit as cockpit_mod
    import hivemind.game.exoplanet_hunter_engine as engine_mod

    monkeypatch.setattr(engine_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(cockpit_mod, "ARTIFACTS_ROOT", tmp_path)
    app = create_exoplanet_hunter_fastapi_app()
    client = TestClient(app)

    reg = client.post(
        "/api/v1/producers/register",
        json={
            "producer_id": "P0",
            "name": "External-P0",
            "hardware": "NVIDIA RTX 4090",
            "priority": 0,
            "status": "ONLINE_SERVING",
            "endpoint_url": "http://127.0.0.1:59999",
        },
    )
    assert reg.status_code == 200

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "hivemind-exoplanet-350m-moe",
            "messages": [{"role": "user", "content": "Write a Python function for orbital mechanics."}],
            "max_tokens": 5,
            "preferred_producer": "P0",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    tokens = content.split()
    assert len(tokens) == 5
    assert body["usage"]["completion_tokens"] == len(tokens) == 5
    assert body["served_by_producer"] == "PRODUCER-LOCAL-RTX2070-CUDA0"
    assert body["served_by_producer"] != "P0"

    pool_status = client.get("/api/v1/producers/status").json()
    p0_entry = next(p for p in pool_status["producers"] if p["producer_id"] == "P0")
    assert p0_entry["requests_served"] == 0


# ---------------------------------------------------------------------------
# IP-32: La Ruche gamer session path & opt-in registration
# ---------------------------------------------------------------------------
def test_ip32_la_ruche_sync_does_not_auto_register_without_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_path = _default_la_ruche_session_path()
    assert not default_path.is_relative_to(repo_root)

    gitignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/*session*.json" in gitignore_text

    post_calls: list[str] = []

    class DummyClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> DummyClient:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def post(self, url: str, **kwargs: Any) -> Any:
            post_calls.append(url)

            class Resp:
                status_code = 200

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, Any]:
                    return {
                        "account_id": "acct-test-32",
                        "api_key": "hm-" + "x" * 40,
                        "handle": "rtx2070-exoplanet-commander",
                        "role": "prosumer",
                    }

            return Resp()

        def get(self, url: str, **kwargs: Any) -> Any:
            class Resp:
                status_code = 200

                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, Any]:
                    return {"balance_hmt": 100.0, "data": [{"id": "hivemind/coder"}]}

            return Resp()

    monkeypatch.delenv("HIVEMIND_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "Client", DummyClient)

    missing_session = tmp_path / "missing_session.json"
    res_no_reg = sync_game_with_la_ruche_live(
        session_path=missing_session,
        allow_auto_register=False,
    )
    assert res_no_reg["la_ruche_cloud"]["connected"] is False
    assert post_calls == []
    assert not missing_session.exists()

    res_opt_in = sync_game_with_la_ruche_live(
        session_path=missing_session,
        allow_auto_register=True,
    )
    assert res_opt_in["la_ruche_cloud"]["connected"] is True
    assert missing_session.exists()
    mode = os.stat(missing_session).st_mode & 0o777
    assert mode == 0o600


# ---------------------------------------------------------------------------
# IP-39: Lean theorem count is dynamically computed from HiveMindSpec.lean
# ---------------------------------------------------------------------------
def test_ip39_lean_counts_are_computed_and_no_test_asserts_literal() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    lean_spec = (repo_root / "lean4" / "HiveMindSpec.lean").read_text(encoding="utf-8")
    stripped = re.sub(r"/-.*?-/", "", lean_spec, flags=re.DOTALL)
    stripped = re.sub(r"--[^\n]*", "", stripped)
    expected = len(re.findall(r"^\s*theorem\s+\w+", stripped, flags=re.MULTILINE))
    assert _count_lean4_spec_theorems() == expected
    assert expected >= 160

    tests_dir = repo_root / "tests"
    offenders: list[str] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"lean4_theorems\s*==\s*1\d\d\b|total_theorems\"\]\s*==\s*1\d\d\b", text):
            offenders.append(path.name)
    assert offenders == []


# ---------------------------------------------------------------------------
# IP-45: Life-as-a-Bee underlying_tag matches catalog TIERS & dynamic summary
# ---------------------------------------------------------------------------
def test_ip45_life_as_a_bee_tags_match_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    tier_by_public_id = {f"hivemind/{t.tier_id}": t for t in TIERS}
    matched = 0
    for scen in LIFE_AS_A_BEE_SCENARIOS:
        tier = tier_by_public_id.get(scen.model_id)
        if tier is not None:
            assert scen.underlying_tag == tier.ollama_tag, (
                f"{scen.scenario_id} ({scen.model_id}): {scen.underlying_tag!r} != {tier.ollama_tag!r}"
            )
            matched += 1
    assert matched >= 5

    modified_scenarios = list(LIFE_AS_A_BEE_SCENARIOS)
    modified_scenarios[0] = dataclasses.replace(
        modified_scenarios[0],
        model_id="hivemind/custom-fastest-bee",
        tokens_per_sec=999.9,
    )
    import hivemind.router.life_as_a_bee as lab_mod

    monkeypatch.setattr(lab_mod, "LIFE_AS_A_BEE_SCENARIOS", tuple(modified_scenarios))
    payload = life_as_a_bee_payload()
    assert payload["summary"]["fastest_bee_model"] == "hivemind/custom-fastest-bee"
    assert payload["summary"]["fastest_bee_tokens_per_sec"] == 999.9


# ---------------------------------------------------------------------------
# IP-54: webgpu_contribute & steam_contribute_boinc offload blocking work
# ---------------------------------------------------------------------------
def test_ip54_webgpu_and_steam_contribute_run_in_threadpool() -> None:
    import hivemind.router.gateway as gw_mod

    src = inspect.getsource(gw_mod.create_router_app)
    webgpu_block = src.split("async def webgpu_contribute", 1)[1].split("async def ", 1)[0]
    assert "await run_in_threadpool(" in webgpu_block
    assert "execute_server_webgpu_verification_and_sync" in webgpu_block

    steam_block = src.split("async def steam_contribute_boinc", 1)[1].split("@app.", 1)[0]
    assert "await run_in_threadpool(" in steam_block


# ---------------------------------------------------------------------------
# IP-55: swat-telepathy/plan validates JSON body & finite numeric parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_payload",
    [
        {"input_tokens": "abc"},
        {"am_compaction_ratio": "nan"},
        {"ocr_text_density": "not-a-float"},
        ["not", "a", "dict"],
    ],
)
def test_ip55_swat_telepathy_plan_bad_numeric_input_returns_400(bad_payload: Any) -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store))
    resp = client.post("/v1/hivemind/bees/swat-telepathy/plan", json=bad_payload)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# IP-56: Header-only authentication; query & path keys rejected; CoPilot clean
# ---------------------------------------------------------------------------
def test_ip56_header_only_auth_rejects_query_and_url_api_keys() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=25.0))
    reg = client.post("/v1/hivemind/register", json={"handle": "header-only-user", "role": "consumer"}).json()
    key = reg["api_key"]

    assert client.get("/v1/hivemind/me", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert client.get(f"/v1/hivemind/me?api_key={key}").status_code == 401
    assert (
        client.post(
            f"/v1/chat/completions?api_key={key}",
            json={"model": "hivemind/coder", "messages": [{"role": "user", "content": "hi"}]},
        ).status_code
        == 401
    )
    assert client.get(f"/k/{key}/v1/models").status_code == 404
    assert (
        client.post(
            f"/k/{key}/v1/chat/completions",
            json={"model": "hivemind/coder", "messages": [{"role": "user", "content": "hi"}]},
        ).status_code
        == 404
    )

    copilot = AntigravityHiveMindCoPilot(api_key=key, base_url=DEFAULT_HIVEMIND_BASE_URL)
    assert copilot.authenticated_base_url == DEFAULT_HIVEMIND_BASE_URL
    assert key not in copilot.authenticated_base_url


# ---------------------------------------------------------------------------
# IP-58: bees/trade and bees/coordinate enforce check_publish & limiter
# ---------------------------------------------------------------------------
def test_ip58_bees_trade_and_coordinate_enforce_check_publish_and_limiter() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=100.0))

    consumer = client.post("/v1/hivemind/register", json={"handle": "pure-consumer", "role": "consumer"}).json()
    prosumer = client.post("/v1/hivemind/register", json={"handle": "active-prosumer", "role": "prosumer"}).json()

    trade_body = {
        "bee_id": "hivemind/coder",
        "side": "BID",
        "subject_id": "SUBJ-01-PYTHON-RUST-CODER",
        "capacity_1k": 1.0,
        "price_per_1k": 0.5,
    }
    coord_body = {
        "primary_bee": "hivemind/coder",
        "prompt": "Integrate stiff ODE system with CVODE BDF",
        "steps": 8,
        "dt": 0.01,
    }

    assert (
        client.post(
            "/v1/hivemind/bees/trade",
            headers={"Authorization": f"Bearer {consumer['api_key']}"},
            json=trade_body,
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/hivemind/bees/coordinate",
            headers={"Authorization": f"Bearer {consumer['api_key']}"},
            json=coord_body,
        ).status_code
        == 403
    )

    assert (
        client.post(
            "/v1/hivemind/bees/coordinate",
            headers={"Authorization": f"Bearer {prosumer['api_key']}"},
            json=coord_body,
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/hivemind/bees/trade",
            headers={"Authorization": f"Bearer {prosumer['api_key']}"},
            json=trade_body,
        ).status_code
        == 200
    )
