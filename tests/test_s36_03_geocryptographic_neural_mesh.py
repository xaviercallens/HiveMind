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


def test_geomesh_router_fails_closed_when_all_adapters_are_tampered() -> None:
    router = GeocryptographicNeuralMeshRouter()
    router.register_canonical_adapter("code_rust", "a" * 64)
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
