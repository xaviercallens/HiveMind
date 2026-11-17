# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for base_model_catalog, btm_moe, dataset_registry, international_launch_model, and swarm_lab (IP-172)."""

from __future__ import annotations

import pytest
import torch

from hivemind.core.base_model_catalog import (
    architecture_from_config,
    build_catalog,
    fetch_catalog_entry,
    geometry_from_entry,
    render_plan,
    urllib_get,
    validate_catalog,
)
from hivemind.core.btm_moe import project_lora_rank_svd_with_report
from hivemind.data.dataset_registry import EvalIndex, MixtureConfig, build_mixture, decontaminate
from hivemind.economy.international_launch_model import (
    SwarmLiveBaseline,
    build_100_producer_international_fleet_plan,
    build_foundation_paper_v42_portal_payload,
    evaluate_international_launch_readiness,
    render_foundation_paper_portal_html,
    render_gcp_planet_forecast_html,
)
from hivemind.lab.swarm_lab import Scenario, contract_digest, simulate


def test_base_model_catalog_boundary_validation() -> None:
    with pytest.raises(ValueError, match="url"):
        urllib_get("", {})
    with pytest.raises(TypeError, match="headers"):
        urllib_get("https://huggingface.co", None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="cfg"):
        architecture_from_config(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        architecture_from_config(
            {
                "hidden_size": 0,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "intermediate_size": 1024,
                "vocab_size": 32000,
            }
        )
    arch = architecture_from_config(
        {
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
            "vocab_size": 32000,
        }
    )
    assert arch["hidden_size"] == 512
    assert arch["head_dim"] == 64

    with pytest.raises(ValueError, match="model_id"):
        fetch_catalog_entry("", "incumbent", lambda _u, _h: (404, b""), "2026-11-10T00:00:00Z")
    with pytest.raises(ValueError, match="role"):
        fetch_catalog_entry("org/m", "", lambda _u, _h: (404, b""), "2026-11-10T00:00:00Z")
    with pytest.raises(TypeError, match="http_get"):
        fetch_catalog_entry("org/m", "incumbent", "bad", "2026-11-10T00:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="retrieved_at"):
        fetch_catalog_entry("org/m", "incumbent", lambda _u, _h: (404, b""), "")

    with pytest.raises(TypeError, match="http_get"):
        build_catalog("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="requested"):
        build_catalog(lambda _u, _h: (404, b""), requested="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="now"):
        build_catalog(lambda _u, _h: (404, b""), requested=[], now="bad")  # type: ignore[arg-type]

    cat = build_catalog(lambda _u, _h: (404, b""), requested=[("org/missing", "candidate")])
    assert len(cat["not_found"]) == 1
    assert cat["models"] == []

    with pytest.raises(TypeError, match="catalog"):
        validate_catalog(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entry"):
        geometry_from_entry(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="plan"):
        render_plan(None)  # type: ignore[arg-type]


def test_btm_moe_project_lora_rank_svd_with_report_validation() -> None:
    with pytest.raises(ValueError, match="lora_state_dict"):
        project_lora_rank_svd_with_report(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_rank_r"):
        project_lora_rank_svd_with_report({}, target_rank_r=0)
    with pytest.raises(ValueError, match="target_rank_r"):
        project_lora_rank_svd_with_report({}, target_rank_r=True)  # type: ignore[arg-type]

    a = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(16, 8, dtype=torch.float32)
    projected, report = project_lora_rank_svd_with_report(
        {"layers.0.mlp.up_proj.lora_A.weight": a, "layers.0.mlp.up_proj.lora_B.weight": b},
        target_rank_r=4,
    )
    assert projected["layers.0.mlp.up_proj.lora_A.weight"].shape == (4, 16)
    assert projected["layers.0.mlp.up_proj.lora_B.weight"].shape == (16, 4)
    assert report.target_rank_r == 4


def test_dataset_registry_decontaminate_and_build_mixture_validation() -> None:
    idx = EvalIndex()
    with pytest.raises(TypeError, match="rows"):
        decontaminate("not-a-sequence", idx)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="index"):
        decontaminate([], "bad-index")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seen"):
        decontaminate([], idx, seen=["not-a-set"])  # type: ignore[arg-type]

    cfg = MixtureConfig(total_rows=2, domain_weights={"code": 1.0})
    with pytest.raises(TypeError, match="Mappings"):
        build_mixture(None, {}, {}, cfg, idx, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MixtureConfig"):
        build_mixture({}, {}, {}, "bad", idx, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EvalIndex"):
        build_mixture({}, {}, {}, cfg, "bad", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="shards"):
        build_mixture({}, {}, {}, cfg, idx, "bad")  # type: ignore[arg-type]


def test_international_launch_model_boundary_validation() -> None:
    with pytest.raises(TypeError, match="baseline"):
        evaluate_international_launch_readiness(baseline="bad")  # type: ignore[arg-type]
    readiness = evaluate_international_launch_readiness(SwarmLiveBaseline())
    assert readiness["ready_for_staged_producer_first_rollout"] is True

    with pytest.raises(ValueError, match="active_producers"):
        build_100_producer_international_fleet_plan(active_producers=0)
    with pytest.raises(ValueError, match="standby_prosumers"):
        build_100_producer_international_fleet_plan(standby_prosumers=-1)
    with pytest.raises(ValueError, match="arrival_rate_req_per_s"):
        build_100_producer_international_fleet_plan(arrival_rate_req_per_s=0.0)
    with pytest.raises(ValueError, match="mean_tokens_per_request"):
        build_100_producer_international_fleet_plan(mean_tokens_per_request=0.0)

    plan = build_100_producer_international_fleet_plan(active_producers=100, standby_prosumers=160)
    assert plan["target_active_producers"] == 100

    with pytest.raises(ValueError, match="base_gcp_url"):
        build_foundation_paper_v42_portal_payload(base_gcp_url="  ")
    with pytest.raises(ValueError, match="e20_artifact_path"):
        build_foundation_paper_v42_portal_payload(e20_artifact_path="  ")

    with pytest.raises(TypeError, match="initial_forecast"):
        render_gcp_planet_forecast_html(initial_forecast="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="payload"):
        render_foundation_paper_portal_html(payload="bad")  # type: ignore[arg-type]


def test_swarm_lab_contract_digest_and_simulate_validation() -> None:
    with pytest.raises(ValueError, match="Scenario or dict"):
        contract_digest("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Scenario or dict"):
        simulate(123)  # type: ignore[arg-type]

    sc = Scenario(M=2, K=2, rounds=2, H=2, dim=4)
    digest = contract_digest(sc)
    sim = simulate(sc)
    assert len(digest) == 16
    assert sim["digest"] == digest
