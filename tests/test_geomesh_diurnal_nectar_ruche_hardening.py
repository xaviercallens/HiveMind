# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening and negative-control tests for GeoMesh fuzzy consensus, Diurnal MoE engine, Swarm Nectar Cache, and La Ruche Vault."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hivemind.swarm.dynamic_diurnal_moe_engine import (
    DiurnalMoETrainingScheduler,
    DiurnalProducerProfile,
    LongTailIntentScanner,
)
from hivemind.swarm.geomesh_fuzzy_consensus import (
    GeoMeshNode,
    fuzzy_subswarm_consensus,
    haversine_km,
    partition_into_fuzzy_subswarms,
    triangulate_rtt_consistency,
)
from hivemind.swarm.la_ruche_vault import (
    LaRucheModelVault,
    _verify_and_measure_adapter_svd,
    load_community_moe_from_ruche,
)
from hivemind.swarm.swarm_nectar_cache import (
    InMemoryClusterNectarHub,
    MultiNodeNectarSyncBackend,
    NectarEntry,
    SwarmNectarCache,
    canonical_query_hash,
)


class TestGeoMeshFuzzyConsensusHardening:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"node_id": "  ", "lat_deg": 48.8, "lon_deg": 2.3, "asn": 1234, "anchor_rtts_ms": {}}, "node_id"),
            ({"node_id": "n1", "lat_deg": True, "lon_deg": 2.3, "asn": 1234, "anchor_rtts_ms": {}}, "lat_deg"),
            ({"node_id": "n1", "lat_deg": 95.0, "lon_deg": 2.3, "asn": 1234, "anchor_rtts_ms": {}}, "lat_deg"),
            ({"node_id": "n1", "lat_deg": 48.8, "lon_deg": False, "asn": 1234, "anchor_rtts_ms": {}}, "lon_deg"),
            ({"node_id": "n1", "lat_deg": 48.8, "lon_deg": -185.0, "asn": 1234, "anchor_rtts_ms": {}}, "lon_deg"),
            ({"node_id": "n1", "lat_deg": 48.8, "lon_deg": 2.3, "asn": True, "anchor_rtts_ms": {}}, "asn"),
            ({"node_id": "n1", "lat_deg": 48.8, "lon_deg": 2.3, "asn": 0, "anchor_rtts_ms": {}}, "asn"),
            ({"node_id": "n1", "lat_deg": 48.8, "lon_deg": 2.3, "asn": 1234, "anchor_rtts_ms": []}, "anchor_rtts_ms"),
        ],
    )
    def test_geomesh_node_rejects_invalid_fields(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            GeoMeshNode(**kwargs)

    def test_haversine_and_triangulation_reject_invalid_coordinates_and_bounds(self) -> None:
        assert haversine_km(48.8566, 2.3522, 51.5074, -0.1278) > 300.0
        with pytest.raises(ValueError, match="lat1"):
            haversine_km(True, 0.0, 0.0, 0.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="lon2"):
            haversine_km(0.0, 0.0, 0.0, 200.0)

        node = GeoMeshNode("n1", 48.85, 2.35, 3215, {"eu": 12.0})
        with pytest.raises(ValueError, match="max_routing_stretch"):
            triangulate_rtt_consistency(node, {"eu": (50.1, 8.6)}, max_routing_stretch=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="base_switching_jitter_ms"):
            triangulate_rtt_consistency(node, {"eu": (50.1, 8.6)}, base_switching_jitter_ms=-1.0)

        bad_rtt_node = GeoMeshNode("n2", 48.85, 2.35, 3215, {"eu": True})  # type: ignore[dict-item]
        ok, reason = triangulate_rtt_consistency(bad_rtt_node, {"eu": (50.1, 8.6)})
        assert not ok and reason == "non_finite_rtt:eu"

    def test_partition_and_consensus_reject_invalid_inputs(self) -> None:
        node = GeoMeshNode("n1", 48.85, 2.35, 3215, {"eu": 12.0})
        with pytest.raises(ValueError, match="epoch_seed"):
            partition_into_fuzzy_subswarms([node], epoch_seed="   ")
        with pytest.raises(ValueError, match="Invalid target sub-swarm bounds"):
            partition_into_fuzzy_subswarms([node], target_min_size=True, target_max_size=20)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid target sub-swarm bounds"):
            partition_into_fuzzy_subswarms([node], target_min_size=15, target_max_size=False)  # type: ignore[arg-type]

        g1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        g2 = np.array([1.1, 1.9, 3.1], dtype=np.float32)
        with pytest.raises(ValueError, match="subswarm_id"):
            fuzzy_subswarm_consensus("  ", {"n1": g1, "n2": g2})
        with pytest.raises(ValueError, match="temperature"):
            fuzzy_subswarm_consensus("s1", {"n1": g1, "n2": g2}, temperature=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="rejection_threshold"):
            fuzzy_subswarm_consensus("s1", {"n1": g1, "n2": g2}, rejection_threshold=1.5)
        with pytest.raises(ValueError, match="finite"):
            fuzzy_subswarm_consensus("s1", {"n1": g1, "n2": np.array([1.0, float("nan"), 3.0])})
        with pytest.raises(ValueError, match="dimensionality"):
            fuzzy_subswarm_consensus("s1", {"n1": g1, "n2": np.array([1.0, 2.0])})
        with pytest.raises(ValueError, match="reference_gradient"):
            fuzzy_subswarm_consensus("s1", {"n1": g1, "n2": g2}, reference_gradient=np.array([1.0]))


class TestDiurnalMoEEngineHardening:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"account_id": "", "handle": "h1", "vram_gb": 16.0, "hardware_class": "cuda", "timezone_offset_hours": 1.0}, "account_id"),
            ({"account_id": "a1", "handle": "  ", "vram_gb": 16.0, "hardware_class": "cuda", "timezone_offset_hours": 1.0}, "handle"),
            ({"account_id": "a1", "handle": "h1", "vram_gb": True, "hardware_class": "cuda", "timezone_offset_hours": 1.0}, "vram_gb"),
            ({"account_id": "a1", "handle": "h1", "vram_gb": 0.0, "hardware_class": "cuda", "timezone_offset_hours": 1.0}, "vram_gb"),
            ({"account_id": "a1", "handle": "h1", "vram_gb": 16.0, "hardware_class": "", "timezone_offset_hours": 1.0}, "hardware_class"),
            ({"account_id": "a1", "handle": "h1", "vram_gb": 16.0, "hardware_class": "cuda", "timezone_offset_hours": True}, "timezone_offset_hours"),
            ({"account_id": "a1", "handle": "h1", "vram_gb": 16.0, "hardware_class": "cuda", "timezone_offset_hours": 18.0}, "timezone_offset_hours"),
            ({"account_id": "a1", "handle": "h1", "vram_gb": 16.0, "hardware_class": "cuda", "timezone_offset_hours": 1.0, "is_gamer_node": 1}, "is_gamer_node"),
        ],
    )
    def test_diurnal_producer_profile_rejects_invalid_fields(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            DiurnalProducerProfile(**kwargs)

    def test_profile_scanner_and_scheduler_reject_invalid_runtime_args(self) -> None:
        prof = DiurnalProducerProfile("a1", "h1", 16.0, "cuda", 1.0)
        with pytest.raises(ValueError, match="now_utc"):
            prof.local_hour(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="now_utc"):
            prof.local_hour(-1.0)
        with pytest.raises(ValueError, match="system_load_ratio"):
            prof.resolve_mode(1000.0, 1.5, 0)
        with pytest.raises(ValueError, match="surplus_producers_count"):
            prof.resolve_mode(1000.0, 0.5, True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="is_preempted"):
            prof.resolve_mode(1000.0, 0.5, 0, is_preempted=1)  # type: ignore[arg-type]

        scanner = LongTailIntentScanner()
        with pytest.raises(ValueError, match="domain"):
            scanner.record_request("   ")
        with pytest.raises(ValueError, match="timestamp_utc"):
            scanner.record_request("maths_formal", timestamp_utc=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="now_utc"):
            scanner.score_all_domains(float("nan"))
        with pytest.raises(ValueError, match="target_domains"):
            scanner.score_all_domains(1000.0, target_domains=())

        sched = DiurnalMoETrainingScheduler(scanner=scanner)
        with pytest.raises(ValueError, match="system_load_ratio"):
            sched.schedule_diurnal_batch(1000.0, [], system_load_ratio=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="now_utc"):
            sched.plan_producer_quorum_moe_serving([], now_utc=-5.0)
        with pytest.raises(ValueError, match="system_load_ratio"):
            sched.plan_producer_quorum_moe_serving([], now_utc=1000.0, system_load_ratio=2.0)


class TestSwarmNectarCacheHardening:
    def test_canonical_query_hash_and_nectar_entry_validation(self) -> None:
        with pytest.raises(ValueError, match="tool_name"):
            canonical_query_hash("   ", "query")

        with pytest.raises(ValueError, match="nectar_richness"):
            NectarEntry("id1", "tool", "hash", {}, True, "bee-1")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="nectar_richness"):
            NectarEntry("id1", "tool", "hash", {}, 1.5, "bee-1")
        with pytest.raises(ValueError, match="access_count"):
            NectarEntry("id1", "tool", "hash", {}, 0.9, "bee-1", access_count=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="hmt_royalties_earned"):
            NectarEntry("id1", "tool", "hash", {}, 0.9, "bee-1", hmt_royalties_earned=-0.1)
        with pytest.raises(ValueError, match="created_utc"):
            NectarEntry("id1", "tool", "hash", {}, 0.9, "bee-1", created_utc=0.0)
        with pytest.raises(ValueError, match="ttl_s"):
            NectarEntry("id1", "tool", "hash", {}, 0.9, "bee-1", ttl_s=False)  # type: ignore[arg-type]

    def test_hub_sync_backend_and_cache_reject_invalid_parameters(self) -> None:
        hub = InMemoryClusterNectarHub()
        with pytest.raises(ValueError, match="key"):
            hub.setex("   ", 60.0, "val")
        with pytest.raises(ValueError, match="ttl_seconds"):
            hub.setex("k1", True, "val")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ttl_seconds"):
            hub.setex("k1", 0.0, "val")

        with pytest.raises(ValueError, match="namespace"):
            MultiNodeNectarSyncBackend(namespace=":::")

        with pytest.raises(ValueError, match="default_ttl_s"):
            SwarmNectarCache(default_ttl_s=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="default_ttl_s"):
            SwarmNectarCache(default_ttl_s=0.0)

        cache = SwarmNectarCache()
        with pytest.raises(ValueError, match="nectar_richness"):
            cache.put("search", "q", {}, "bee-1", nectar_richness=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ttl_s"):
            cache.put("search", "q", {}, "bee-1", ttl_s=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="discovered_by_bee"):
            cache.put("search", "q", {}, "   ")


class TestLaRucheVaultHardening:
    def test_vault_init_deposit_assemble_and_path_traversal_guards(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="canonical_rank_r"):
            _verify_and_measure_adapter_svd({"a": torch.ones(2, 2)}, canonical_rank_r=True)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="hidden_size"):
            LaRucheModelVault(tmp_path / "v1", hidden_size=30)
        with pytest.raises(ValueError, match="vocab_size"):
            LaRucheModelVault(tmp_path / "v1", vocab_size=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="canonical_rank_r"):
            LaRucheModelVault(tmp_path / "v1", canonical_rank_r=0)

        vault = LaRucheModelVault(tmp_path / "vault")
        tensors = {
            "layers.0.self_attn.q_proj.lora_A": torch.ones((4, 32), dtype=torch.float32) * 0.1,
            "layers.0.self_attn.q_proj.lora_B": torch.ones((32, 4), dtype=torch.float32) * 0.1,
        }
        with pytest.raises(ValueError, match="contributor_handle"):
            vault.deposit_qlora_expert(
                "code_python",
                tensors,
                "   ",
                "RTX 3090",
                "cid-1",
                2.5,
                2.0,
                2.4,
                1.9,
                11.0,
                6.7,
            )

        with pytest.raises(ValueError, match="release_version"):
            vault.assemble_community_btm_moe(release_version="  ")
        with pytest.raises(ValueError, match="top_k"):
            vault.assemble_community_btm_moe(top_k=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="seed"):
            vault.assemble_community_btm_moe(seed=-1)

        # Path traversal guard in load_community_moe_from_ruche
        rel_dir = tmp_path / "vault_escape" / "community_moe_release"
        rel_dir.mkdir(parents=True, exist_ok=True)
        (rel_dir / "ruche_manifest.json").write_text(
            json.dumps(
                {
                    "combined_safetensors_path": "../outside.safetensors",
                    "combined_safetensors_sha256": "a" * 64,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unsafe combined_safetensors_path"):
            load_community_moe_from_ruche(tmp_path / "vault_escape")
        with pytest.raises(ValueError, match="seed"):
            load_community_moe_from_ruche(tmp_path / "vault_escape", seed=True)  # type: ignore[arg-type]
