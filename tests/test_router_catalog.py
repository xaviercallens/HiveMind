"""Capability tiers: which model fits which GPU, and that every tier id is routable by the gateway."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hivemind.router.accounts import RouterStore
from hivemind.router.catalog import (
    PREWIRED_PARTNER_MODELS,
    TIERS,
    ModelTier,
    catalog_payload,
    normalize_model_alias,
    recommend_tier,
    tier_by_id,
    tiers_for_vram,
)
from hivemind.router.gateway import _MODEL_ID_RE, create_router_app


def test_every_tier_id_is_routable_and_unique_and_ordered_by_memory() -> None:
    ids = [tier.public_id for tier in TIERS]
    assert len(set(ids)) == len(ids)
    assert all(_MODEL_ID_RE.match(public_id) for public_id in ids)
    memory = [tier.min_vram_gb for tier in TIERS]
    assert memory == sorted(memory) and len(TIERS) >= 5


def test_moe_tiers_exist_and_declare_their_active_parameters() -> None:
    moe = [tier for tier in TIERS if tier.moe]
    assert {tier.tier_id for tier in moe} == {"bee-ruche-moe", "moe-qwen35-bee", "moe-general", "moe-coder"}
    assert all(tier.active_params_b is not None and tier.active_params_b < 10 for tier in moe)
    assert all(tier.active_params_b is None for tier in TIERS if not tier.moe)


@pytest.mark.parametrize(
    ("vram", "expected"),
    [
        (3.9, None),
        (4.0, "coder-lite"),
        (4.5, "gemma-core"),
        (5.5, "bee-stars"),
        (6.0, "coder"),
        (8.0, "coder"),
        (8.5, "bee-lean-prover"),
        (10.0, "bee-ruche-moe"),
        (12.0, "coder-plus"),
        (14.9, "coder-plus"),
        (15.0, "moe-general"),
        (15.36, "moe-general"),
        (16.0, "moe-general"),
        (24.0, "moe-coder"),
        (80.0, "moe-coder"),
    ],
)
def test_recommend_tier_picks_the_biggest_tier_that_fits(vram: float, expected: str | None) -> None:
    tier = recommend_tier(vram)
    assert (tier.tier_id if tier else None) == expected


def test_core_models_and_moe_bees_aliases_are_normalized() -> None:
    for alias, expected in (
        ("hivemind/gemma", "gemma-core"),
        ("google/gemma-3-4b", "gemma-core"),
        ("gemma3:4b", "gemma-core"),
        ("hivemind/qwen3.5", "qwen35-core"),
        ("qwen/qwen3.5-8b", "qwen35-core"),
        ("qwen3:8b", "qwen35-core"),
        ("hivemind/mistral", "mistral-core"),
        ("mistral:7b-instruct", "mistral-core"),
        ("hivemind/bee-lean", "bee-lean-prover"),
        ("hivemind/la-ruche-moe", "bee-ruche-moe"),
        ("hivemind/moe-bee", "moe-qwen35-bee"),
    ):
        assert normalize_model_alias(alias) == expected
        tier = tier_by_id(alias)
        assert tier is not None and tier.tier_id == expected


@given(st.floats(min_value=0.0, max_value=200.0), st.floats(min_value=0.0, max_value=200.0))
def test_more_memory_never_recommends_a_smaller_tier(a: float, b: float) -> None:
    low, high = sorted((a, b))
    small, big = recommend_tier(low), recommend_tier(high)
    assert len(tiers_for_vram(low)) <= len(tiers_for_vram(high))
    if small is not None:
        assert big is not None and big.min_vram_gb >= small.min_vram_gb
        assert big.min_vram_gb <= high  # a recommendation always fits the card


def test_invalid_memory_and_invalid_tiers_are_rejected() -> None:
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            tiers_for_vram(bad)
    with pytest.raises(ValueError):
        ModelTier("Bad Id", "x", "t", 1.0, 1.0, False, None, 0.1, 1.0, "x")  # uppercase and space are not routable
    with pytest.raises(ValueError):
        ModelTier("moe-without-active", "x", "t", 1.0, 1.0, True, None, 0.1, 1.0, "x")
    with pytest.raises(ValueError):
        ModelTier("dense-with-active", "x", "t", 1.0, 1.0, False, 3.0, 0.1, 1.0, "x")


def test_lookup_accepts_bare_and_public_ids_and_payload_is_json() -> None:
    assert tier_by_id("coder") is tier_by_id("hivemind/coder") and tier_by_id("coder") is not None
    assert tier_by_id("nope") is None
    payload = json.loads(json.dumps(catalog_payload()))
    assert [t["public_id"] for t in payload["tiers"]] == [t.public_id for t in TIERS]
    assert all(t["suggested_cost_per_1k"] >= 0 and t["suggested_capacity_1k"] > 0 for t in payload["tiers"])


def test_a_database_created_before_the_endpoint_token_is_upgraded_in_place(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE producers(account_id TEXT PRIMARY KEY, endpoint_url TEXT NOT NULL, models_json TEXT NOT NULL,"
        " cost_per_1k REAL NOT NULL, capacity_1k REAL NOT NULL, last_heartbeat_utc REAL NOT NULL, execution_class TEXT NOT NULL);"
        "INSERT INTO producers VALUES('acc-old','https://old.example.com','[{\"id\":\"coder\",\"local_name\":\"q\"}]',0.4,10,100.0,'gpu');"
    )
    old.commit()
    old.close()

    store = RouterStore(path, b"s" * 32)
    rows = store.active_producers("coder", now_utc=110.0)

    assert [r.account_id for r in rows] == ["acc-old"] and rows[0].endpoint_token is None
    account_id, _ = store.register("new-producer", "producer")
    store.upsert_producer(account_id, "https://new.example.com", [{"id": "coder", "local_name": "q"}], 0.4, 10, "gpu", now_utc=111.0, endpoint_token="t" * 20)
    assert {r.account_id: r.endpoint_token for r in store.active_producers("coder", now_utc=112.0)}[account_id] == "t" * 20
    assert "t" * 20 not in repr(store.active_producers("coder", now_utc=112.0)[0])  # a record never prints its token


def test_crowdgpt_crowd_v1_partner_model_is_prewired_and_routable(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    assert len(PREWIRED_PARTNER_MODELS) >= 1
    crowd = next(m for m in PREWIRED_PARTNER_MODELS if m.tier_id == "crowd-v1")
    assert crowd.hf_repo == "Vxtzq/Crowd-v1"
    assert crowd.github_repo == "Vxtzq/CrowdGPT"
    assert crowd.public_id == "hivemind/crowd-v1"
    assert crowd.partner_id == "crowdgpt/crowd-v1"

    for alias in ("crowd-v1", "hivemind/crowd-v1", "crowdgpt/crowd-v1", "Vxtzq/Crowd-v1", "vxtzq/crowd-v1"):
        assert normalize_model_alias(alias) == "crowd-v1"
        tier = tier_by_id(alias)
        assert tier is not None and tier.tier_id == "crowd-v1" and tier.public_id == "hivemind/crowd-v1"

    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    cat = client.get("/v1/hivemind/catalog").json()
    assert any(m["hf_repo"] == "Vxtzq/Crowd-v1" for m in cat["prewired_partner_models"])

    partners = client.get("/v1/hivemind/partner-models").json()
    assert partners["object"] == "hivemind.partner_models"
    crowd_entry = next(m for m in partners["models"] if m["tier_id"] == "crowd-v1")
    assert crowd_entry["live_on_swarm"] is False

    prod_id, api_key = store.register("crowdgpt-node", "producer")
    store.upsert_producer(
        prod_id,
        "https://crowdgpt.net",
        [{"id": "crowd-v1", "local_name": "Vxtzq/Crowd-v1"}],
        0.15,
        50.0,
        "gpu",
    )
    partners_live = client.get("/v1/hivemind/partner-models").json()
    crowd_live = next(m for m in partners_live["models"] if m["tier_id"] == "crowd-v1")
    assert crowd_live["live_on_swarm"] is True
    assert "hivemind/crowd-v1" in [m["id"] for m in client.get("/v1/models").json()["data"]]
    for alias in ("hivemind/crowd-v1", "crowdgpt/crowd-v1", "Vxtzq/Crowd-v1"):
        mkt = client.get("/v1/hivemind/market", params={"model": alias}, headers={"Authorization": f"Bearer {api_key}"})
        assert mkt.status_code == 200
        assert mkt.json()["model"] == "hivemind/crowd-v1"


def test_season19_swarm_plan_endpoint_for_bees_and_miners(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    store = RouterStore(tmp_path / "router.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    cat = client.get("/v1/hivemind/catalog").json()
    assert cat["season"] == "HM-S19-Q35"
    assert cat["swarm_plan_endpoint"] == "/v1/hivemind/swarm-plan"

    plan = client.get("/v1/hivemind/swarm-plan").json()
    assert plan["object"] == "hivemind.swarm_plan"
    assert plan["season"] == "HM-S19-Q35"
    assert set(plan["miner_hardware_tiers"].keys()) == {"std_8gb", "big_12gb", "xl_16gb_plus"}
    assert len(plan["specialized_bees"]) == 14
    assert len(plan["seven_day_roadmap"]["days"]) == 6
    assert plan["target_cluster_shares"]["coding"] > 0.55


def test_wave2_scientific_moe_models_and_producer_quorum_activation(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from hivemind.router.catalog import (
        CPU_HYBRID_SOLVER_CATALOG,
        normalize_model_alias,
        resolve_producer_quorum_moe_state,
    )

    # Verify new Wave 2 Scientific MoE and frontier MoE aliases resolve to valid tiers
    assert normalize_model_alias("hivemind/moe-scientific-wave2") == "bee-rusty-sundials"
    assert normalize_model_alias("hivemind/moe-surrogate-sundials") == "bee-rusty-sundials"
    assert normalize_model_alias("hivemind/moe-pde-fluid-quantum") == "moe-qwen35-bee"
    assert normalize_model_alias("hivemind/moe-cosmology-dualscale") == "bee-stars"
    assert normalize_model_alias("hivemind/moe-genetics-hopfield") == "bee-rusty-sundials"
    assert normalize_model_alias("hivemind/moe-deepseek-prover-v2") == "bee-lean-prover"
    assert normalize_model_alias("hivemind/moe-qwen3-coder-next") == "moe-coder"
    assert normalize_model_alias("hivemind/moe-glm45-air-science") == "moe-general"

    assert len(CPU_HYBRID_SOLVER_CATALOG) == 3
    assert CPU_HYBRID_SOLVER_CATALOG[0].min_vram_gb == 0.0
    assert CPU_HYBRID_SOLVER_CATALOG[1].min_quorum == 2

    store = RouterStore(tmp_path / "router_quorum.db", b"k" * 32)
    app = create_router_app(store, allow_private_producers=True)
    client = TestClient(app)

    cat = client.get("/v1/hivemind/catalog").json()
    assert len(cat["cpu_hybrid_solver_tiers"]) == 3
    assert len(cat["wave2_scientific_moe_quorum_stages"]) == 5

    # Register 2 live producers and verify /v1/hivemind/moe/quorum-status activates Wave 2 Scientific MoE
    for idx in range(2):
        pid, _ = store.register(f"wave2-prod-{idx}", "producer")
        store.upsert_producer(
            pid,
            f"https://prod-{idx}.hivemind.net",
            [{"id": "bee-rusty-sundials", "local_name": "rusty-sundials-7b"}],
            0.35,
            40.0,
            "gpu",
        )

    quorum = client.get("/v1/hivemind/moe/quorum-status").json()
    assert quorum["connected_producers"] == 2
    assert quorum["wave2_scientific_moe_active"] is True
    assert quorum["quorum_sweep_enabled"] is True
    assert quorum["boinc_min_quorum"] == 2
    assert "hivemind/moe-scientific-wave2" in quorum["unlocked_moe_models"]

    # Check 8-producer full swarm stage
    full_state = resolve_producer_quorum_moe_state(8, total_vram_gb=64.0)
    assert full_state["stage_id"] == "stage_4_full_swarm_20_expert_super_moe"
    assert full_state["max_concurrent_experts"] == 20



