# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and boundary hardening tests for Exemplar BYOB Bees, Identity Bridge,
AMCP Kafka Campaign Bus, and HF Community Hub (Iteration 73).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hivemind.swarm.amcp_kafka_campaign_bus import (
    InMemoryKafkaBridge,
    build_amcp_kafka_event,
    clear_bee_capacity_market,
    compute_campaign_poll_summary,
    coordinate_inter_bee_inference,
    is_allowed_bee_id,
)
from hivemind.swarm.exemplar_byob_bees import (
    EXEMPLAR_BYOB_BEES,
    ExemplarBYOBBee,
    ExemplarBYOBRegistryManager,
)
from hivemind.swarm.hf_community_hub import (
    CommunityAdapterBundleManager,
    HFCommunityModelCardExporter,
    SwarmMiningGuildEngine,
    _compute_state_dict_sha256,
)
from hivemind.swarm.identity_bridge import (
    SHARED_ED25519_VC_TEST_VECTORS,
    _b58decode,
    _b58encode,
    did_key_from_ed25519_public_bytes,
    ed25519_public_bytes_from_did_key,
    hash_hmt_key_commitment,
    issue_producer_verifiable_credential,
    verify_producer_verifiable_credential,
)


# ============================================================================
# 1. Exemplar BYOB Bees Hardening Tests
# ============================================================================


def test_exemplar_byob_bee_and_registry_manager_reject_invalid_inputs() -> None:
    base_bee = EXEMPLAR_BYOB_BEES[0]
    base_kwargs = {
        "bee_id": base_bee.bee_id,
        "public_id": base_bee.public_id,
        "title": base_bee.title,
        "domain": base_bee.domain,
        "base_hf_repo": base_bee.base_hf_repo,
        "base_tier_id": base_bee.base_tier_id,
        "hf_repo_id": base_bee.hf_repo_id,
        "revision": base_bee.revision,
        "author_handle": base_bee.author_handle,
        "parameters_b": base_bee.parameters_b,
        "description": base_bee.description,
        "license": base_bee.license,
        "test_accuracy_gain": base_bee.test_accuracy_gain,
        "mcnemar_p_value": base_bee.mcnemar_p_value,
        "benchmark_prompt": base_bee.benchmark_prompt,
        "sample_completion": base_bee.sample_completion,
    }
    for bad in (
        {"bee_id": ""},
        {"public_id": "   "},
        {"title": ""},
        {"domain": ""},
        {"parameters_b": 0.0},
        {"parameters_b": True},
        {"parameters_b": float("nan")},
        {"test_accuracy_gain": True},
        {"test_accuracy_gain": float("inf")},
        {"mcnemar_p_value": -0.01},
        {"mcnemar_p_value": 1.01},
        {"mcnemar_p_value": True},
    ):
        kw = dict(base_kwargs)
        kw.update(bad)
        with pytest.raises(ValueError):
            ExemplarBYOBBee(**kw)  # type: ignore[arg-type]

    mgr = ExemplarBYOBRegistryManager()
    with pytest.raises(ValueError):
        mgr.get_bee("")
    with pytest.raises(ValueError):
        mgr.get_bee("   ")
    with pytest.raises(TypeError):
        mgr.register_and_attest_bee("not-a-bee")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mgr.register_and_attest_bee(base_bee, adapter_weights_bytes=b"")
    with pytest.raises(ValueError):
        mgr.register_and_attest_bee(base_bee, adapter_sha256="not_64_hex")
    with pytest.raises(ValueError):
        mgr.register_and_attest_bee(base_bee, paired_outcomes=[])
    with pytest.raises(ValueError):
        mgr.register_and_attest_bee(base_bee, paired_outcomes=[(1, 0)])  # type: ignore[list-item]

    res = mgr.register_and_attest_bee(base_bee)
    assert res["status"] == "APPROVED_LIVE"
    assert res["gate_passed"] is True


# ============================================================================
# 2. Identity Bridge Hardening Tests
# ============================================================================


def test_identity_bridge_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        _b58encode(b"")
    with pytest.raises(ValueError):
        _b58encode("not_bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _b58decode("")
    with pytest.raises(ValueError):
        _b58decode("0OIl")  # invalid base58 chars

    with pytest.raises(ValueError):
        did_key_from_ed25519_public_bytes(b"short")
    with pytest.raises(ValueError):
        ed25519_public_bytes_from_did_key("did:web:example.com")

    with pytest.raises(ValueError):
        hash_hmt_key_commitment("")
    with pytest.raises(ValueError):
        hash_hmt_key_commitment("   ")
    with pytest.raises(ValueError):
        hash_hmt_key_commitment(True)  # type: ignore[arg-type]

    seed = b"\x11" * 32
    base_vc_kw = {
        "private_key_seed": seed,
        "node_id": "node-paris-01",
        "hmt_key": "hmt_live_secret_01",
        "kwaainet_peer_id": "12D3KooWParis01",
        "tiers": ["hivemind/coder"],
        "issuance_date": "2026-11-07T00:00:00Z",
        "expiration_date": "2027-11-07T00:00:00Z",
    }
    for bad_vc in (
        {"private_key_seed": b"short"},
        {"node_id": ""},
        {"node_id": True},
        {"hmt_key": ""},
        {"kwaainet_peer_id": "   "},
        {"kwaainet_peer_id": 123},
        {"tiers": []},
        {"tiers": [""]},
        {"issuance_date": "not-a-date"},
        {"expiration_date": "2025-01-01T00:00:00Z"},  # before issuance_date
    ):
        kw = dict(base_vc_kw)
        kw.update(bad_vc)
        with pytest.raises(ValueError):
            issue_producer_verifiable_credential(**kw)  # type: ignore[arg-type]

    vc = SHARED_ED25519_VC_TEST_VECTORS[0]["credential"]
    ok, reason = verify_producer_verifiable_credential(
        vc,
        expected_node_id="node-paris-rtx2070-01",
        expected_hmt_key="hmt_live_vector01_paris_rtx2070",
        now_utc="2026-12-01T00:00:00Z",
    )
    assert ok is True and reason == "ok"


# ============================================================================
# 3. AMCP Kafka Campaign Bus Hardening Tests
# ============================================================================


def test_amcp_kafka_campaign_bus_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        is_allowed_bee_id(123, "xavier")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        is_allowed_bee_id("hivemind/coder", None)  # type: ignore[arg-type]
    assert is_allowed_bee_id("hivemind/coder", "xavier") is True
    assert is_allowed_bee_id("   ", "xavier") is False

    for bad_ev in (
        {"topic_key": ""},
        {"event_type": "  "},
        {"sender_id": ""},
        {"sender_kind": ""},
        {"recipient_id": ""},
        {"subject_id": ""},
        {"correlation_id": "   "},
        {"now_utc": 0.0},
        {"now_utc": True},
        {"now_utc": float("nan")},
    ):
        kw = dict(
            topic_key="campaigns",
            event_type="io.hivemind.amcp.test",
            sender_id="s1",
            sender_kind="ai_agent",
        )
        kw.update(bad_ev)
        with pytest.raises(ValueError):
            build_amcp_kafka_event(**kw)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        compute_campaign_poll_summary("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_campaign_poll_summary(["not-a-dict"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        compute_campaign_poll_summary(
            [{"subject_id": "SUBJ-01-PYTHON-RUST-CODER", "pledged_vram_gb": True}]
        )
    with pytest.raises(ValueError):
        compute_campaign_poll_summary(
            [{"subject_id": "SUBJ-01-PYTHON-RUST-CODER", "pledged_vram_gb": -4.0}]
        )

    for bad_coord in (
        {"primary_bee": ""},
        {"prompt": "   "},
        {"steps": 0},
        {"steps": True},
        {"dt": 0.0},
        {"dt": True},
        {"dt": float("nan")},
    ):
        kw = dict(primary_bee="hivemind/coder", prompt="Solve Robertson ODE", steps=8, dt=0.01)
        kw.update(bad_coord)
        with pytest.raises(ValueError):
            coordinate_inter_bee_inference(**kw)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        clear_bee_capacity_market("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        clear_bee_capacity_market([{"side": "OFFER", "capacity_1k": True, "price_per_1k": 1.0}])
    with pytest.raises(ValueError):
        clear_bee_capacity_market([{"side": "BID", "capacity_1k": 10.0, "price_per_1k": -1.0}])

    for bad_maxlen in (0, -1, True):
        with pytest.raises(ValueError):
            InMemoryKafkaBridge(maxlen_per_topic=bad_maxlen)  # type: ignore[arg-type]

    bridge = InMemoryKafkaBridge(maxlen_per_topic=8)
    with pytest.raises(TypeError):
        bridge.publish({"not": "an_event"})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bridge.recent(limit=0)
    with pytest.raises(ValueError):
        bridge.recent(limit=True)  # type: ignore[arg-type]


# ============================================================================
# 4. HF Community Hub Hardening Tests
# ============================================================================


def test_hf_community_hub_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _compute_state_dict_sha256({})

    with pytest.raises(ValueError):
        HFCommunityModelCardExporter(hf_org_or_user="")
    with pytest.raises(ValueError):
        HFCommunityModelCardExporter(base_model_id="   ")

    exporter = HFCommunityModelCardExporter(mock_hub_root=tmp_path / "mock_hub")
    with pytest.raises(ValueError):
        exporter.pull_bundle_from_hf_hub("", tmp_path / "out")

    for bad_mgr in (
        {"royalty_hive_per_import": -1.0},
        {"royalty_hive_per_import": True},
        {"royalty_hive_per_import": float("nan")},
        {"royalty_pts_per_import": -5},
        {"royalty_pts_per_import": True},
    ):
        with pytest.raises(ValueError):
            CommunityAdapterBundleManager(store_dir=tmp_path / "cas", **bad_mgr)  # type: ignore[arg-type]

    mgr = CommunityAdapterBundleManager(store_dir=tmp_path / "cas")
    adapter_spec = {
        "Python_Systems_Code": {
            "lora_state_dict": {
                "layers.0.mlp.gate_proj.lora_A": torch.randn(8, 16),
                "layers.0.mlp.gate_proj.lora_B": torch.randn(16, 8),
            }
        }
    }
    for bad_pub in (
        {"bundle_id": ""},
        {"bundle_title": "  "},
        {"guild_id": ""},
        {"domain_adapters": {}},
        {"canonical_rank_r": 0},
        {"canonical_rank_r": True},
        {"base_model_id": ""},
    ):
        kw = dict(
            bundle_id="b1",
            bundle_title="Bundle 1",
            guild_id="GUILD-01-EURO-CYBER-SYNDICATE",
            domain_adapters=adapter_spec,
            output_dir=tmp_path / "bundle_out",
            canonical_rank_r=8,
        )
        kw.update(bad_pub)
        with pytest.raises(ValueError):
            mgr.create_and_publish_bundle(**kw)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        mgr.import_community_bundle(tmp_path / "bundle_out", importer_agent_id="")
    with pytest.raises(ValueError):
        mgr.seed_and_replicate_bundle_chunks(tmp_path / "bundle_out", source_node_id="")
    with pytest.raises(ValueError):
        mgr.seed_and_replicate_bundle_chunks(
            tmp_path / "bundle_out", source_node_id="node-1", chunk_size=0
        )
    with pytest.raises(ValueError):
        mgr.seed_and_replicate_bundle_chunks(
            tmp_path / "bundle_out", source_node_id="node-1", chunk_size=True  # type: ignore[arg-type]
        )

    guild_eng = SwarmMiningGuildEngine()
    for bad_act in (
        {"guild_id": ""},
        {"base_pts_earned": -1},
        {"base_pts_earned": True},
        {"base_hive_earned": -0.5},
        {"base_hive_earned": True},
        {"base_hive_earned": float("nan")},
        {"peer_imports_count": -1},
        {"peer_imports_count": True},
    ):
        kw = dict(
            guild_id="GUILD-01-EURO-CYBER-SYNDICATE",
            base_pts_earned=100,
            base_hive_earned=10.0,
            peer_imports_count=1,
        )
        kw.update(bad_act)
        with pytest.raises(ValueError):
            guild_eng.record_guild_bundle_activity(**kw)  # type: ignore[arg-type]
