# Copyright (C) 2026 Xavier Callens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Tests for Contributive & Community Help Portal, OpenTrainer Self-Training MoE, and Producer Heartbeat Precedence."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.opentrainer import (
    COMMUNITY_HELP_AND_CONTRIBUTIVE_GUIDE,
    MULTI_GPU_2_TO_4_BYOM_CO_TRAINING_PLAYBOOK,
    summarize_moe_self_training_progress,
)


def _make_app_and_store() -> tuple[TestClient, RouterStore, str]:
    secret = b"hivemind-test-secret-at-least-32-bytes-long!"
    store = RouterStore(":memory:", secret)
    admin_key = "admin-secret-key-for-hivemaster"
    admin_hash = store.key_digest(admin_key)
    app = create_router_app(
        store,
        allow_private_producers=True,
        admin_key_hash=admin_hash,
        resolver=lambda _host: ["34.48.48.154"],
    )
    return TestClient(app), store, admin_key


def test_summarize_moe_self_training_progress_empty_and_monotone() -> None:
    empty = summarize_moe_self_training_progress(recent_rounds=[], opentrainer_nodes=[], now_utc=1000.0)
    assert empty["rounds_recorded"] == 0
    assert empty["initial_eval_loss"] is None
    assert empty["latest_eval_loss"] is None
    assert empty["monotone_ratchet_improved"] is False
    assert empty["data_source"] == "NO_ROUNDS_RECORDED_YET"

    rounds = [
        {
            "round_index": 760,
            "eval_loss": 0.9412,
            "tokens_trained": 10240,
            "run_id": "e8-rtx2070-failover-co-train-v1",
            "handle": "xavier_callens",
            "submitted_utc": 950.0,
        },
        {
            "round_index": 761,
            "eval_loss": 0.8925,
            "tokens_trained": 10240,
            "run_id": "e8-rtx2070-failover-co-train-v1",
            "handle": "xavier_callens",
            "submitted_utc": 985.0,
        },
    ]
    nodes = [{"node_id": "xavier_callens:XAVGAMSRV-RTX2070", "last_heartbeat_utc": 990.0}]
    prog = summarize_moe_self_training_progress(
        recent_rounds=rounds,
        opentrainer_nodes=nodes,
        now_utc=1000.0,
    )
    assert prog["rounds_recorded"] == 2
    assert prog["latest_round_index"] == 761
    assert prog["initial_eval_loss"] == 0.9412
    assert prog["latest_eval_loss"] == 0.8925
    assert prog["best_ratcheted_eval_loss"] == 0.8925
    assert prog["loss_improvement_delta"] > 0
    assert prog["monotone_ratchet_improved"] is True
    assert prog["total_tokens_trained"] == 20480
    assert prog["active_trainer_nodes_count"] == 1
    assert len(COMMUNITY_HELP_AND_CONTRIBUTIVE_GUIDE) >= 5
    assert len(MULTI_GPU_2_TO_4_BYOM_CO_TRAINING_PLAYBOOK["supported_topologies"]) == 3


def test_contributive_hub_and_opentrainer_moe_endpoints() -> None:
    client, store, _ = _make_app_and_store()
    _, api_key = store.register("trainer_qc_node", "prosumer")

    hb_ot = client.post(
        "/v1/hivemind/opentrainer/heartbeat",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "host_label": "XAVGAMSRV-RTX2070",
            "gpu_name": "NVIDIA GeForce RTX 2070",
            "vram_total_mib": 8192,
            "vram_free_mib": 6400,
            "network_mode": "outbound-https-relay",
        },
    )
    assert hb_ot.status_code == 200

    for r_idx, loss_val in ((101, 1.42), (102, 1.31), (103, 1.19)):
        sub = client.post(
            "/v1/hivemind/opentrainer/submit-delta",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "campaign_id": "byom-custom-moe-4gpu-v1",
                "round_id": r_idx,
                "learner_index": 1,
                "inner_steps": 24,
                "loss_before": loss_val + 0.08,
                "loss_after": loss_val,
                "delta_norm": 0.42,
                "delta_sha256": f"{r_idx:064x}",
                "network_mode": "outbound-https-relay",
            },
        )
        assert sub.status_code == 200

    moe_res = client.get("/v1/hivemind/opentrainer/moe-self-training")
    assert moe_res.status_code == 200
    moe_json = moe_res.json()
    assert moe_json["rounds_recorded"] == 3
    assert moe_json["latest_round_index"] == 103
    assert moe_json["monotone_ratchet_improved"] is True
    assert moe_json["best_ratcheted_eval_loss"] == 1.19
    assert "byom-custom-moe-4gpu-v1" in moe_json["active_campaigns"]

    hub_res = client.get("/v1/hivemind/contributive-hub")
    assert hub_res.status_code == 200
    hub_json = hub_res.json()
    assert hub_json["object"] == "hivemind.contributive_hub"
    assert hub_json["integration_cycle"]["cadence_hours"] == 3
    assert hub_json["opentrainer_moe_self_training"]["rounds_recorded"] == 3
    assert len(hub_json["community_help_guide"]) >= 5
    assert "multi_gpu_2_to_4_byom_playbook" in hub_json


def test_producer_heartbeat_overrides_reconnected_fleet_keepalive() -> None:
    client, store, admin_key = _make_app_and_store()
    store.register("gcp-t4-bee", "producer")

    rec = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"x-admin-key": admin_key},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io/u",
            "bee_endpoint_url": "https://34-48-48-154.sslip.io/bee",
            "endpoint_token": "test-endpoint-token-0123456789abcdefghijk",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct", "loaded": True, "vram_mib": 4841}],
            "replace_endpoints": True,
            "include_legacy_known_users": True,
        },
    )
    assert rec.status_code == 200

    # Now heartbeat gcp-t4-bee directly to /bee with only bee-stars
    hb = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"x-admin-key": admin_key},
        json={
            "handle": "gcp-t4-bee",
            "endpoint_url": "https://34-48-48-154.sslip.io/bee",
            "models": [{"id": "bee-stars", "local_name": "qwen2.5-coder-7b-instruct-q4_k_m.gguf"}],
            "cost_per_1k": 0.5,
            "capacity_1k": 30,
            "gpu_vram_mib": 15360,
            "replace_endpoints": True,
        },
    )
    assert hb.status_code == 200
    ack = hb.json()
    assert ack["models"] == ["bee-stars"]
    assert ack["endpoints"] == 1
    fleet = client.get("/v1/hivemind/admin/producers", headers={"x-admin-key": admin_key}).json()
    by_handle = {r["handle"]: r for r in fleet["hiveminders"]}
    assert by_handle["gcp-t4-bee"]["endpoint_url"] == "https://34-48-48-154.sslip.io/bee"


def test_portal_renders_contributive_community_hub_section() -> None:
    client, _, _ = _make_app_and_store()
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert 'id="contributive-community-hub"' in html
    assert 'id="communityHelpTopicSelect"' in html
    assert 'id="contributiveHubBtn"' in html
    assert 'value="byom-2-to-4-local-gpus"' in html
    assert "/v1/hivemind/contributive-hub" in html
