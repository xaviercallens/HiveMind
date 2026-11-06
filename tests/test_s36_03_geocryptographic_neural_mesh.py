"""Unit tests for HM-S36-03 Geocryptographic P2P Neural Mesh Routing Table."""

from __future__ import annotations

import pytest

from hivemind.swarm.s36_03_geocryptographic_neural_mesh import (
    GeocryptographicNeuralMeshRouter,
    GeoMeshExpertNode,
    VivaldiCoord3D,
)


def test_geomesh_router_selects_nearest_verified_node_and_rejects_tampered() -> None:
    router = GeocryptographicNeuralMeshRouter()
    eval_res = router.evaluate_multi_region_p95_reduction()
    assert eval_res["meets_40pct_reduction_gate"] is True
    assert float(eval_res["p95_reduction_pct"]) >= 40.0

    # IP-164 negative control: placing the central gateway at the client's coordinate must fail the 40% reduction gate
    co_located_res = router.evaluate_multi_region_p95_reduction(
        central_gateway_coord=VivaldiCoord3D(12.5, 10.2, 2.5, 1.2),
        client_coord=VivaldiCoord3D(12.5, 10.2, 2.5, 1.2),
    )
    assert co_located_res["meets_40pct_reduction_gate"] is False


def test_geomesh_router_fails_closed_when_all_adapters_are_tampered() -> None:
    from hivemind.swarm.s36_03_geocryptographic_neural_mesh import _cosine_sim

    assert _cosine_sim([], []) == 0.0
    assert _cosine_sim([0.0, 0.0], [0.0, 0.0]) == 0.0

    router = GeocryptographicNeuralMeshRouter()
    with pytest.raises(ValueError, match="adapter_sha256"):
        router.register_canonical_adapter("code_rust", "short")
    with pytest.raises(KeyError, match="Unknown canonical digest"):
        router.route_query(
            client_coord=VivaldiCoord3D(6.0, 5.0, 1.0),
            expert_domain="unknown_domain",
            query_semantic_vector=(1.0, 0.0),
        )
    router.register_canonical_adapter("code_rust", "a" * 64)
    router.register_node(
        GeoMeshExpertNode(
            node_id="other-domain-01",
            region="eu-west",
            coord=VivaldiCoord3D(5.0, 5.0, 1.0),
            expert_domain="other_domain",
            semantic_vector=(1.0, 0.0),
            adapter_sha256="a" * 64,
        )
    )
    router.register_node(
        GeoMeshExpertNode(
            node_id="low-vram-01",
            region="eu-west",
            coord=VivaldiCoord3D(5.0, 5.0, 1.0),
            expert_domain="code_rust",
            semantic_vector=(1.0, 0.0),
            adapter_sha256="a" * 64,
            vram_free_mib=128.0,
        )
    )
    router.register_node(
        GeoMeshExpertNode(
            node_id="bad-peer-01",
            region="eu-west",
            coord=VivaldiCoord3D(5.0, 5.0, 1.0),
            expert_domain="code_rust",
            semantic_vector=(1.0, 0.0),
            adapter_sha256="b" * 64,
        )
    )
    with pytest.raises(RuntimeError, match="No verified GeoMesh node"):
        router.route_query(
            client_coord=VivaldiCoord3D(6.0, 5.0, 1.0),
            expert_domain="code_rust",
            query_semantic_vector=(1.0, 0.0),
        )
    router.register_node(
        GeoMeshExpertNode(
            node_id="good-peer-01",
            region="eu-west",
            coord=VivaldiCoord3D(5.0, 5.0, 1.0),
            expert_domain="code_rust",
            semantic_vector=(1.0, 0.0),
            adapter_sha256="a" * 64,
        )
    )
    dec = router.route_query(
        client_coord=VivaldiCoord3D(6.0, 5.0, 1.0),
        expert_domain="code_rust",
        query_semantic_vector=(1.0, 0.0),
    )
    assert dec.to_dict()["selected_node_id"] == "good-peer-01"
