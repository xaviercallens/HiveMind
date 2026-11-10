from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.opentrainer import (
    CROWDGPT_TRAINER_PRODUCER_CO_TRAINING_PLAN,
    DEFAULT_MIN_FREE_VRAM_MIB_8GB,
    DEFAULT_MIN_FREE_VRAM_MIB_T4,
    build_crowdgpt_trainer_producer_plan,
    build_opentrainer_status_payload,
    is_allowed_opentrainer_campaign_id,
)
from hivemind.swarm.crowdgpt_bridge import classify_gpu_precision_cohort


def test_crowdgpt_campaign_ids_allowed() -> None:
    for cid in (
        "CROWDGPT-V1-TRUNK-DILOCO",
        "CROWDGPT-V1-QLORA-GRAFT",
        "CROWDGPT-V1-LOSS-CHECK",
    ):
        assert is_allowed_opentrainer_campaign_id(cid) is True


def test_build_crowdgpt_trainer_producer_plan_metrics_and_vram_floors() -> None:
    now = 1_762_800_000.0
    nodes = [
        {"node_id": "n1", "host_label": "XAVGAMSRV-RTX2070", "last_heartbeat_utc": now - 30.0},
        {"node_id": "n2", "host_label": "hivemind-t4", "last_heartbeat_utc": now - 60.0},
        {"node_id": "n3", "host_label": "stale-node", "last_heartbeat_utc": now - 900.0},
    ]
    rounds = [
        {"run_id": "CROWDGPT-V1-TRUNK-DILOCO", "round_index": 1, "tokens_trained": 32768, "eval_loss": 8.42},
        {"run_id": "CROWDGPT-V1-QLORA-GRAFT", "round_index": 2, "tokens_trained": 16384, "eval_loss": 7.91},
        {"run_id": "E8-STAGE2-TINY-GRANITE-MOE-QLORA", "round_index": 3, "tokens_trained": 4096, "eval_loss": 2.15},
    ]
    plan = build_crowdgpt_trainer_producer_plan(
        opentrainer_nodes=nodes,
        recent_rounds=rounds,
        now_utc=now,
    )
    assert plan["object"] == "hivemind.opentrainer.crowdgpt_trainer_producer_plan"
    assert plan["plan_id"] == "CROWDGPT-HIVEMIND-CO-TRAIN-V1"
    assert plan["partner_project"]["hf_model_id"] == "Vxtzq/Crowd-v1"
    assert plan["partner_project"]["openrouter_model_id"] == "hivemind/crowd-v1"
    assert plan["partner_project"]["discord_invite"] == "https://discord.gg/ZeFtZpKHf"

    roles_by_id = {r["role_id"]: r for r in plan["gpu_role_partitioning"]}
    assert roles_by_id["producer_gpu_dual_schedule"]["vram_safety_floor_mib"] == DEFAULT_MIN_FREE_VRAM_MIB_8GB == 1536
    assert roles_by_id["dedicated_and_spot_trainer_gpu"]["vram_safety_floor_mib"] == DEFAULT_MIN_FREE_VRAM_MIB_T4 == 4300
    assert roles_by_id["lowcost_cpu_orchestrator_and_coordinator"]["vram_safety_floor_mib"] == 0

    telemetry = plan["live_campaign_telemetry"]
    assert telemetry["active_trainer_and_producer_nodes"] == 2
    assert telemetry["crowdgpt_rounds_recorded"] == 2
    assert telemetry["crowdgpt_tokens_trained"] == 32768 + 16384


def test_precision_cohorts_align_with_crowdgpt_bridge() -> None:
    cohorts = {c["cohort_key"]: c for c in CROWDGPT_TRAINER_PRODUCER_CO_TRAINING_PLAN["precision_cohorts"]}
    sm75 = classify_gpu_precision_cohort("nvidia", (7, 5), requested_dtype="auto", optimizer_family="adamw_8bit")
    assert sm75.cohort_key in cohorts
    assert sm75.effective_dtype == "fp32"
    assert sm75.pre_ampere_fp32_enforced is True

    sm86 = classify_gpu_precision_cohort("nvidia", (8, 6), requested_dtype="auto", optimizer_family="adamw_8bit")
    assert sm86.cohort_key in cohorts
    assert sm86.effective_dtype == "bf16"
    assert sm86.pre_ampere_fp32_enforced is False


def test_gateway_crowdgpt_plan_and_opentrainer_status_endpoints(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router.db", b"test-secret-tag-32bytes-00000000")
    app = create_router_app(store)
    client = TestClient(app)

    r_plan = client.get("/v1/hivemind/opentrainer/crowdgpt-plan")
    assert r_plan.status_code == 200
    body = r_plan.json()
    assert body["plan_id"] == "CROWDGPT-HIVEMIND-CO-TRAIN-V1"
    assert len(body["rollout_phases"]) == 4

    r_status = client.get("/v1/hivemind/opentrainer/status")
    assert r_status.status_code == 200
    st_body = r_status.json()
    assert "crowdgpt_trainer_producer_plan" in st_body
    assert st_body["crowdgpt_trainer_producer_plan"]["partner_project"]["hf_model_id"] == "Vxtzq/Crowd-v1"

    r_hub = client.get("/v1/hivemind/contributive-hub")
    assert r_hub.status_code == 200
    hub_body = r_hub.json()
    assert "crowdgpt_trainer_producer_plan" in hub_body
    assert hub_body["crowdgpt_trainer_producer_plan"]["plan_id"] == "CROWDGPT-HIVEMIND-CO-TRAIN-V1"
