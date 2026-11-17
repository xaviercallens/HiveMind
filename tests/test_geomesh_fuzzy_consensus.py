"""Unit tests for HM-S31-01 & HM-S31-02 (GeoMesh + Fuzzy Consensus) and PR-First Orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hivemind.swarm.geomesh_fuzzy_consensus import (
    GeoMeshNode,
    fuzzy_subswarm_consensus,
    haversine_km,
    partition_into_fuzzy_subswarms,
    triangulate_rtt_consistency,
)
from hivemind.swarm.pr_first_openrouter_orchestrator import (
    CANONICAL_5_ROADMAP_PRS,
    PRFirstOpenRouterCodingOrchestrator,
    scan_python_ast_security,
)


ANCHORS = {
    "us-east": (39.0438, -77.4874),
    "eu-west": (50.1109, 8.6821),
    "ap-east": (35.6762, 139.6503),
}


def _realistic_rtts(lat: float, lon: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for aid, (alat, alon) in ANCHORS.items():
        d_km = haversine_km(lat, lon, alat, alon)
        out[aid] = round((2.0 * d_km) / 198.0 * 1.35 + 4.0, 2)
    return out


def test_triangulate_rtt_consistency_accepts_honest_and_rejects_spoofed_coords() -> None:
    honest_node = GeoMeshNode(
        node_id="paris-bee-01",
        lat_deg=48.8566,
        lon_deg=2.3522,
        asn=3215,
        anchor_rtts_ms=_realistic_rtts(48.8566, 2.3522),
    )
    ok, reason = triangulate_rtt_consistency(honest_node, ANCHORS)
    assert ok is True
    assert reason == "ok"

    # Node claims to be in Sydney (-33.8688, 151.2093) while physically in Virginia (1 ms to us-east)
    spoofed_node = GeoMeshNode(
        node_id="sybil-spoof-01",
        lat_deg=-33.8688,
        lon_deg=151.2093,
        asn=16509,
        anchor_rtts_ms={"us-east": 1.2, "eu-west": 75.0, "ap-east": 150.0},
    )
    ok_spoof, reason_spoof = triangulate_rtt_consistency(spoofed_node, ANCHORS)
    assert ok_spoof is False
    assert "speed_of_light_violation" in reason_spoof


def test_partition_into_fuzzy_subswarms_enforces_15_to_20_nodes_per_swarm() -> None:
    rng = np.random.default_rng(19)
    nodes: list[GeoMeshNode] = []
    for i in range(54):
        lat = float(rng.uniform(-55.0, 65.0))
        lon = float(rng.uniform(-140.0, 150.0))
        nodes.append(
            GeoMeshNode(
                node_id=f"bee-{i:03d}",
                lat_deg=lat,
                lon_deg=lon,
                asn=1000 + (i % 18),
                anchor_rtts_ms=_realistic_rtts(lat, lon),
            )
        )
    subswarms = partition_into_fuzzy_subswarms(nodes, epoch_seed="epoch-42")
    assert len(subswarms) == 3
    for sw in subswarms:
        assert 15 <= len(sw.nodes) <= 20
        assert sw.mean_pairwise_km > 1500.0
        assert sw.unique_asn_count >= 10


def test_fuzzy_subswarm_consensus_rejects_colluding_outliers_and_preserves_signal() -> None:
    rng = np.random.default_rng(123)
    dim = 128
    true_grad = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    node_grads: dict[str, np.ndarray] = {}

    for i in range(14):
        node_grads[f"honest-{i:02d}"] = true_grad + 0.04 * rng.standard_normal(dim, dtype=np.float32)
    for j in range(4):
        node_grads[f"byz-{j:02d}"] = -8.0 * true_grad + 0.5 * rng.standard_normal(dim, dtype=np.float32)

    res = fuzzy_subswarm_consensus(
        subswarm_id="subswarm-epoch-42-000",
        node_gradients=node_grads,
        reference_gradient=true_grad,
    )
    assert len(res.rejected_node_ids) == 4
    assert all(nid.startswith("byz-") for nid in res.rejected_node_ids)
    assert res.honest_cosine_fidelity > 0.99


def test_pr_first_openrouter_orchestrator_verifies_all_5_roadmap_prs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    orchestrator = PRFirstOpenRouterCodingOrchestrator(repo_root=repo_root)
    receipts = orchestrator.verify_all_roadmap_prs()
    assert len(receipts) == len(CANONICAL_5_ROADMAP_PRS) == 5
    for rec in receipts:
        assert rec.ast_security_clean is True
        assert rec.forbidden_calls_found == ()
        assert rec.ready_to_merge is True

    clean, violations = scan_python_ast_security("x = eval('1 + 1')\n")
    assert clean is False
    assert "eval" in violations
