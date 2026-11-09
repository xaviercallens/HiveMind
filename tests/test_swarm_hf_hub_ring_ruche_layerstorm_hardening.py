# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression and input-validation tests for HF community hub, inference ring, La Ruche vault, and LayerStoRm backend."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from hivemind.swarm.hf_community_hub import (
    CommunityAdapterBundle,
    HFCommunityModelCardExporter,
    SharedCommunityExpertEntry,
)
from hivemind.swarm.inference_ring import RegionalDHTSwarmRing, SwarmPeerNode
from hivemind.swarm.la_ruche_vault import (
    _serialize_tensors_safetensors,
    load_community_moe_from_gcs_or_https,
)
from hivemind.swarm.layerstorm_backend import (
    HybridHiSparseE8KVCache,
    RealCUDAPCIeExpertStreamer,
)


def _sample_bundle() -> CommunityAdapterBundle:
    entry = SharedCommunityExpertEntry(
        domain_name="code_python",
        dataset_id="hivemind/code_python",
        author_gamer_tag="CyberValkyrie_FR",
        author_contributor_handle="@xaviercallens",
        expert_cid="cid://sha256/expert-code-python-" + ("a" * 24),
        source_rank_r=16,
        canonical_rank_r=16,
        lora_alpha=32,
        spectral_energy_retained=0.98,
        baseline_loss=4.2,
        trained_loss=1.5,
        relative_loss_reduction_pct=64.29,
        tensor_sha256="b" * 64,
    )
    return CommunityAdapterBundle(
        bundle_id="bundle-01",
        bundle_title="Sample Community Bundle",
        hf_repo_id="hivemind-swarm/bundle-01",
        base_model_id="Qwen/Qwen2.5-0.5B",
        canonical_rank_r=16,
        created_at_epoch=1760000000.0,
        guild_id="GUILD-01-EURO-CYBER-SYNDICATE",
        authors=["CyberValkyrie_FR"],
        contributors=["@xaviercallens"],
        entries=[entry],
        bundle_sha256="c" * 64,
    )


def test_hf_community_hub_exporter_validation(tmp_path: Path) -> None:
    exporter = HFCommunityModelCardExporter(mock_hub_root=tmp_path / "mock_hub")
    bundle = _sample_bundle()
    card = exporter.generate_model_card_markdown(bundle)
    assert "Sample Community Bundle" in card

    with pytest.raises(TypeError, match="bundle must be a CommunityAdapterBundle"):
        exporter.generate_model_card_markdown({"bad": True})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="bundle must be a CommunityAdapterBundle"):
        exporter.export_bundle_to_directory({"bad": True}, {"code_python": {}}, tmp_path / "out")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expert_states must be a non-empty dict"):
        exporter.export_bundle_to_directory(bundle, {}, tmp_path / "out")

    with pytest.raises(ValueError, match="offline_fallback must be a boolean"):
        exporter.push_bundle_to_hf_hub(tmp_path / "out", offline_fallback="yes")  # type: ignore[arg-type]


def test_inference_ring_routing_validation() -> None:
    ring = RegionalDHTSwarmRing(region="EU-West-Paris")
    p1 = SwarmPeerNode("peer-eu", "76561198000000001", "EU-West-Paris", 12.0, 8.0, "code_python", (0, 4), True)
    p2 = SwarmPeerNode("peer-us", "76561198000000002", "US-East-Virginia", 14.0, 8.0, "code_python", (0, 4), True)
    ring.register_super_ring_peer(p1)
    ring.register_super_ring_peer(p2)

    route = ring.resolve_hierarchical_route("peer-eu", "peer-us")
    assert route["source_peer_id"] == "peer-eu"

    with pytest.raises(ValueError, match="source_peer_id must be a non-empty string"):
        ring.resolve_hierarchical_route("  ", "peer-us")

    with pytest.raises(ValueError, match="target_peer_id must be a non-empty string"):
        ring.resolve_hierarchical_route("peer-eu", "")

    with pytest.raises(ValueError, match="source_peer_id must be a non-empty string"):
        ring.route_to_region("", "US-East-Virginia")

    with pytest.raises(ValueError, match="target_region must be a non-empty string"):
        ring.route_to_region("peer-eu", "   ")


def test_la_ruche_vault_safetensors_and_gcs_loader_validation(tmp_path: Path) -> None:
    raw = _serialize_tensors_safetensors({"w": torch.tensor([1.0, 2.0])})
    assert len(raw) > 0

    with pytest.raises(ValueError, match="tensors must be a non-empty dict"):
        _serialize_tensors_safetensors({})

    with pytest.raises(ValueError, match="gcs_public_base_url must be a non-empty string"):
        load_community_moe_from_gcs_or_https(cache_dir=tmp_path, gcs_public_base_url="   ")

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        load_community_moe_from_gcs_or_https(cache_dir=tmp_path, seed=-1)


def test_layerstorm_streamer_and_kv_cache_validation() -> None:
    streamer = RealCUDAPCIeExpertStreamer(
        num_total_experts=4,
        zone_a_slots=2,
        zone_b_slots=2,
        expert_rows=16,
        expert_cols=32,
        device="cpu",
    )
    h = torch.randn(2, 16)
    res = streamer.forward_step(h, [0, 1])
    assert res["output_norm"] >= 0.0

    with pytest.raises(ValueError, match="hidden_state must be a 2D tensor"):
        streamer.forward_step(torch.randn(16), [0])

    with pytest.raises(ValueError, match="hidden_state must be a 2D tensor"):
        streamer.forward_step(torch.randn(2, 8), [0])

    kv = HybridHiSparseE8KVCache(sink_tokens=4, sliding_window_tokens=8, heavy_hitter_tokens=4)
    attn = torch.softmax(torch.randn(1, 20), dim=-1)
    idx = kv.select_hot_indices(attn, random_control=False, seed=7)
    assert idx.numel() > 0

    with pytest.raises(ValueError, match="random_control must be a boolean"):
        kv.select_hot_indices(attn, random_control="no")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        kv.select_hot_indices(attn, random_control=True, seed=-5)

    with pytest.raises(ValueError, match="random_control must be a boolean"):
        q = torch.randn(1, 8, 16)
        kv.evaluate_attention_fidelity(q, q, q, random_control=1)  # type: ignore[arg-type]
