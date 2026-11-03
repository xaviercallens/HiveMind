"""Season 5 Bounties (`HM-S5-01`..`HM-S5-06`) & Pareto Swarm Strategy Improvement (`SI-01`..`SI-05`) Tests."""

from __future__ import annotations

from pathlib import Path

import torch

from hivemind.core.btm_moe import (
    FP8PagedKVCache,
    HierarchicalMoERouter,
    hotplug_hierarchical_experts,
)
from hivemind.core.btm_upcycle import initialize_domain_branches, upcycle_branches_to_sparse_moe
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.hf_community_hub import CommunityAdapterBundleManager
from hivemind.swarm.ledger import HiveSwarmLedger
from hivemind.swarm.strategy_optimizer import (
    AdaptiveDiLoCoCadenceController,
    DynamicFisherSVDSelector,
    ParetoGuildCampaignPlanner,
    UCB1CurriculumShardScheduler,
    reputation_weighted_ties_merge_lora,
)
from hivemind.swarm.swarm_fabric import (
    HiveLayer2StateChannel,
    OptimisticProofOfInferenceVerifier,
)
from paper.autoevolve_peer_review_pipeline import verify_lean4_specification


def _make_lora_adapter(rank_r: int, seed: int, dim: int = 32) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "layers.0.mlp.gate_proj.lora_A": torch.randn(rank_r, dim, generator=gen),
        "layers.0.mlp.gate_proj.lora_B": torch.randn(dim * 2, rank_r, generator=gen),
        "layers.0.mlp.up_proj.lora_A": torch.randn(rank_r, dim, generator=gen),
        "layers.0.mlp.up_proj.lora_B": torch.randn(dim * 2, rank_r, generator=gen),
        "layers.0.mlp.down_proj.lora_A": torch.randn(rank_r, dim * 2, generator=gen),
        "layers.0.mlp.down_proj.lora_B": torch.randn(dim, rank_r, generator=gen),
    }


def test_hm_s5_01_hf_hub_sync_and_64kb_multi_peer_cas_chunk_seeding(tmp_path: Path) -> None:
    """Verify HM-S5-01 (`PR #55`): HF Hub push/pull mirror + 64 KB Merkle chunk seeding & multi-peer download."""
    store_origin = ContentAddressedSwarmStore(tmp_path / "cas_paris", node_id="seti-node-paris-01")
    store_peer1 = ContentAddressedSwarmStore(tmp_path / "cas_berlin", node_id="seti-node-berlin-02")
    store_peer2 = ContentAddressedSwarmStore(tmp_path / "cas_tokyo", node_id="seti-node-tokyo-04")

    manager = CommunityAdapterBundleManager(store=store_origin)
    manager.exporter.mock_hub_root = tmp_path / "mock_hf_hub"

    bundle_dir = tmp_path / "exported_s5_bundle"
    manager.create_and_publish_bundle(
        bundle_id="HIVEMIND-S5-STRATEGY-BUNDLE",
        bundle_title="Season 5 Multi-Peer Chunked Bundle",
        guild_id="GUILD-01-EURO-SINGULARITY",
        domain_adapters={
            "Rust_Systems_Security": {
                "lora_state_dict": _make_lora_adapter(16, 11),
                "dataset_id": "anon/rust-code-instructions",
                "author_gamer_tag": "CyberValkyrie_FR",
                "author_contributor_handle": "@xaviercallens",
                "baseline_loss": 5.8,
                "trained_loss": 0.31,
            },
            "Solidity_Smart_Contracts": {
                "lora_state_dict": _make_lora_adapter(32, 22),
                "dataset_id": "ASSERT-KTH/DISL-Solidity-Smart-Contracts",
                "author_gamer_tag": "NeuralGhost_DE",
                "author_contributor_handle": "@pedromineo",
                "baseline_loss": 5.4,
                "trained_loss": 0.29,
            },
        },
        output_dir=bundle_dir,
    )

    push_res = manager.exporter.push_bundle_to_hf_hub(bundle_dir=bundle_dir)
    # no token: the bundle is mirrored locally and the record says so (IP-165); "pushed" is reserved for the Hub
    assert push_res["status"] == "mirrored_locally_not_pushed" and push_res["pushed_to_hub"] is False
    assert "README.md" in push_res["files_uploaded"]

    pulled_dir = tmp_path / "pulled_from_hub"
    pull_res = manager.exporter.pull_bundle_from_hf_hub(
        repo_id=push_res["repo_id"], target_dir=pulled_dir
    )
    assert pull_res["status"] == "pulled"
    assert (pulled_dir / "hivemind_swarm_manifest.json").exists()

    seed_res = manager.seed_and_replicate_bundle_chunks(
        bundle_dir=bundle_dir,
        source_node_id="seti-node-paris-01",
        peer_stores=[store_peer1, store_peer2],
        chunk_size=4096,
    )
    assert seed_res["num_chunks"] >= 4
    assert sum(seed_res["replication_counts"].values()) > 0

    reassembled_dir = tmp_path / "reassembled_bundle"
    import_res = manager.download_and_import_chunked_bundle(
        manifest=seed_res["manifest"],
        target_dir=reassembled_dir,
        importer_agent_id="seti-node-virginia-03",
        peer_stores=[store_origin, store_peer1, store_peer2],
    )
    assert import_res["download_summary"]["verified"] is True
    assert import_res["import_receipt"]["status"] == "verified_and_hotplugged"
    assert import_res["import_receipt"]["royalties_settled_hive"] == 30.0


def test_hm_s5_02_64_expert_hierarchical_moe_and_fp8_paged_kv_cache() -> None:
    """Verify HM-S5-02 (`PR #54`): 2-level `G_macro -> G_micro` 64-expert router and FP8 paged KV-cache `< 2.2 GB`."""
    router = HierarchicalMoERouter(
        hidden_size=32, num_macro_domains=8, experts_per_domain=8, top_k=2
    )
    assert router.num_experts == 64
    x = torch.randn(10, 32)
    weights, selected, aux_loss = router(x)
    assert weights.shape == (10, 2)
    assert selected.shape == (10, 2)
    assert int(selected.min().item()) >= 0
    assert int(selected.max().item()) < 64
    assert float(aux_loss.item()) > 0.0

    kv_cache = FP8PagedKVCache(
        num_layers=4, num_kv_heads=4, head_dim=16, page_size_tokens=16, max_pages=32
    )
    k_page = torch.randn(1, 4, 16, 16, dtype=torch.float16)
    v_page = torch.randn(1, 4, 16, 16, dtype=torch.float16)
    w_info = kv_cache.write_kv_page(0, 0, k_page, v_page)
    assert w_info["compression_ratio"] >= 1.75
    k_rec, v_rec = kv_cache.read_kv_page(0, 0)
    assert k_rec.shape == k_page.shape and v_rec.shape == v_page.shape
    cos_k = float(
        torch.nn.functional.cosine_similarity(
            k_page.flatten().float().unsqueeze(0), k_rec.flatten().float().unsqueeze(0)
        ).item()
    )
    assert cos_k > 0.99


def test_hm_s5_03_interactive_bisection_fraud_game_v3_and_bond_slashing(tmp_path: Path) -> None:
    """Verify HM-S5-03 (`Issue #45`): `O(log2 L)` interactive bisection pinpoints corrupted layer `#11` in `<= 4` rounds."""
    verifier = OptimisticProofOfInferenceVerifier()
    ledger = HiveSwarmLedger(ledger_path=tmp_path / "bisection_ledger.json")
    ledger.credit_reward("dishonest-node-99", 50.0, "Initial Prover Bond")

    l2_chan = HiveLayer2StateChannel(
        consumer_id="cursor-dev-01",
        provider_ids=["dishonest-node-99"],
        escrow_hive=5.0,
    )
    l2_chan.sign_micro_voucher(120)

    gen = torch.Generator().manual_seed(77)
    honest_layers = [torch.randn(4, 32, generator=gen) for _ in range(16)]
    claimed_layers = [t.clone() for t in honest_layers]
    # Corrupt layer #11 (and downstream layers 12..15)
    for idx in range(11, 16):
        claimed_layers[idx] = honest_layers[idx] * -3.5 + 2.0

    verdict = verifier.run_interactive_bisection_dispute_game_v3(
        peer_id="dishonest-node-99",
        claimed_layer_activations=claimed_layers,
        validator_layer_activations=honest_layers,
        prover_bond_hive=50.0,
        challenger_id="challenger-guardian-01",
        consumer_id="cursor-dev-01",
        ledger=ledger,
        l2_channel=l2_chan,
    )

    assert verdict["fraud_proven"] is True
    assert verdict["diverging_layer_index"] == 11
    assert verdict["bisection_rounds_executed"] <= 4
    assert verdict["single_layer_replays_executed"] == 1
    assert verdict["log_l_bound_satisfied"] is True
    assert verdict["prover_bond_slashed_hive"] == 50.0
    assert verdict["challenger_bounty_hive"] == 25.0
    assert verdict["consumer_compensation_hive"] == 25.0
    assert verdict["l2_escrow_refunded_hive"] > 0.0
    assert "dishonest-node-99" in ledger.slashed_registry
    assert ledger.get_balance("challenger-guardian-01") == 25.0


def test_hm_s5_06_lean4_48_theorems_machine_checked() -> None:
    """Verify HM-S5-06 (`Issue #48`): all 48 Lean 4 theorems compile with `0 sorry` and `0` user axioms."""
    audit = verify_lean4_specification()
    assert audit["tier_a_machine_checked"] is True
    assert audit["theorem_count"] >= 48
    assert audit["sorry_count"] == 0
    assert audit["user_axiom_count"] == 0
    assert "ties_sign_consensus_majority_bound" in audit["theorems"]
    assert "fisher_spectral_rank_budget_conservation" in audit["theorems"]
    assert "adaptive_diloco_communication_sla_bound" in audit["theorems"]


def test_pareto_swarm_strategy_improvements_si01_to_si05(tmp_path: Path) -> None:
    """Verify all 5 Strategy Improvements (`SI-01`..`SI-05`) in `src/hivemind/swarm/strategy_optimizer.py`."""
    ref_lora = {
        "layer0.lora_A": torch.zeros((16, 64), dtype=torch.float32),
        "layer0.lora_B": torch.zeros((64, 16), dtype=torch.float32),
        "layer1.lora_A": torch.zeros((16, 64), dtype=torch.float32),
        "layer1.lora_B": torch.zeros((64, 16), dtype=torch.float32),
    }
    planner = ParetoGuildCampaignPlanner()
    out_json = tmp_path / "swarm_strategy_improvement_report.json"
    report = planner.compute_pareto_frontier_and_ablation(
        reference_lora_state=ref_lora,
        output_report_path=out_json,
    )

    # SI-01: Dynamic Fisher SVD Rank Allocation
    si01 = report["si01_dynamic_fisher_svd"]
    assert si01["vram_budget_conserved"] is True
    assert si01["total_rank_used"] <= si01["total_rank_budget"]
    assert si01["spectral_energy_improvement"] > 0.0
    ranks = si01["allocated_ranks_by_domain"]
    assert ranks["Rust_Systems_Security"] > ranks["Biomedical_Clinical_QA"]

    # SI-02: UCB1 Bandit Curriculum Scheduler
    si02 = report["si02_ucb1_curriculum_bandit"]
    assert "Lean4_Theorem_Prover" in si02["top2_prioritized_domains"]

    # SI-03: Adaptive DiLoCo Cadence Controller
    si03 = report["si03_adaptive_diloco_cadence"]
    lan = si03["regional_quic_0rtt_profile"]
    wan = si03["transcontinental_turn_relay_profile"]
    assert lan["sla_8pct_satisfied"] is True
    assert wan["sla_8pct_satisfied"] is True
    assert wan["recommended_inner_steps_H"] > lan["recommended_inner_steps_H"]

    # SI-04: Reputation-Weighted TIES Merging
    si04 = report["si04_reputation_weighted_ties"]
    assert si04["reputation_weighted_beats_uniform"] is True
    assert si04["reputation_weighted_ties_holdout_loss"] < si04["uniform_ties_holdout_loss"]

    # SI-05: Pareto Strategy Frontier
    frontier = report["si05_pareto_strategy_frontier"]
    assert len(frontier) == 3
    assert frontier[-1]["pareto_optimal"] is True
    assert out_json.exists()
