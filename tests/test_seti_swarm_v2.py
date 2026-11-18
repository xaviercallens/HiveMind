"""Unit and Integration Tests for HiveMind v2 ('SETI-Swarm') 4-Pillar Architecture.

Covers:
1. ContentAddressedSwarmStore (Kademlia DHT + CAS + Merkle Byte-Range Proof-of-Storage)
2. HiveSwarmLedger ($HIVE PoUW Block Mining, Halving, Byzantine Slashing & Inference Settlement)
3. Autonomous 3-State SETI Agent Lifecycle (Gaming/Seed-Only, Idle Miner, Swarm Consumer)
4. Dynamic Domain Expert Hot-Plug (BTM MoE v2 E -> E+1 Router Expansion)
"""

from __future__ import annotations

from pathlib import Path

import torch

from hivemind.client.daemon import SETIAgentState, ZeroIntrusionDaemon
from hivemind.core.btm_upcycle import UpcycledSparseMoEModel, upcycle_branches_to_sparse_moe
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.ledger import HiveSwarmLedger, WorkReceipt


def test_cas_vault_and_pouw_ledger_tokenomics(tmp_path: Path) -> None:
    """Verify CAS shard storage, Merkle Proof-of-Storage, PoUW Block #1 mining, and fee settlement."""
    store = ContentAddressedSwarmStore(tmp_path / "swarm_store")
    ledger_path = tmp_path / "swarm_store" / "hive_pouw_ledger.json"
    ledger = HiveSwarmLedger(ledger_path=ledger_path)

    # Check Bitcoin-style halving schedule
    assert ledger.block_reward_at_height(1) == 50.0
    assert ledger.block_reward_at_height(10_000) == 25.0
    assert ledger.block_reward_at_height(20_000) == 12.5

    nodes = ["seti-node-paris-01", "seti-node-lyon-02", "seti-node-nice-03"]
    storage_receipts = []
    work_receipts = []

    for idx, nid in enumerate(nodes):
        samples = [f"def fn_{idx}_{j}(x): return x + {j}" for j in range(12)]
        cid, path, nbytes = store.store_hf_dataset_shard(
            node_id=nid,
            domain_name="Python_Systems_Code",
            shard_id=idx,
            samples=samples,
        )
        assert cid.startswith("cid://sha256/")
        assert nbytes == store.SHARD_BLOCK_ALIGNMENT_BYTES

        s_receipt = store.verify_proof_of_storage(
            node_id=nid,
            shard_cid=cid,
            shard_path=path,
            prev_block_hash=ledger.chain[-1].block_hash,
        )
        assert s_receipt.verified is True
        storage_receipts.append(s_receipt)

        work_receipts.append(
            WorkReceipt(
                node_id=nid,
                steam_id64=f"7656119800000000{idx + 1}",
                branch_name="Python_Systems_Code",
                flops_contributed=3.84e11,
                cosine_alignment=0.98,
                spot_check_passed=True,
                multikrum_accepted=True,
                delta_sha256=f"honest_delta_{idx}",
            )
        )

    # Inject 1 Byzantine Poisoner Node
    work_receipts.append(
        WorkReceipt(
            node_id="byzantine-botnet-99",
            steam_id64="76561198999999999",
            branch_name="Python_Systems_Code",
            flops_contributed=3.84e11,
            cosine_alignment=-0.45,
            spot_check_passed=True,
            multikrum_accepted=False,
            delta_sha256="poison_delta_99",
        )
    )

    dummy_lora = {"layers.0.mlp.gate_proj.lora_A.weight": torch.randn(8, 32)}
    expert_cid, expert_path = store.publish_lora_expert_checkpoint(
        domain_name="Python_Systems_Code",
        lora_state_dict=dummy_lora,
        metadata={"dataset": "sahil2801/CodeAlpaca-20k", "outer_step": 1},
    )
    assert expert_path.exists()
    loaded_cid, loaded_meta, loaded_state = store.load_lora_expert_checkpoint(expert_cid)
    assert loaded_cid == expert_cid
    assert loaded_meta["dataset"] == "sahil2801/CodeAlpaca-20k"
    assert "layers.0.mlp.gate_proj.lora_A.weight" in loaded_state

    block = ledger.mine_pouw_block(
        outer_step=1,
        expert_cid=expert_cid,
        work_receipts=work_receipts,
        storage_receipts=storage_receipts,
        validator_node_ids=["validator-node-eu-01"],
    )
    assert block.block_height == 1
    assert block.total_block_reward == 50.0
    assert "byzantine-botnet-99" in block.slashed_nodes
    assert ledger.balances["seti-node-paris-01"] == 15.0
    assert ledger.balances["seti-node-lyon-02"] == 15.0
    assert ledger.balances["seti-node-nice-03"] == 15.0
    assert ledger.balances["validator-node-eu-01"] == 5.0
    assert ledger.balances["byzantine-botnet-99"] == 0.0

    settlement = ledger.settle_swarm_inference(
        consumer_node_id="seti-node-paris-01",
        provider_node_ids=["seti-node-lyon-02", "seti-node-nice-03"],
        domain_expert="Python_Systems_Code",
        verified_tokens=45,
        rtt_speedup=5.0,
        rate_per_token=0.005,
    )
    assert settlement.fee_paid_hive == 0.225
    assert ledger.balances["seti-node-paris-01"] == 14.775
    assert ledger.balances["seti-node-lyon-02"] == 15.1125
    assert ledger.balances["seti-node-nice-03"] == 15.1125
    assert ledger.verify_chain_integrity() is True
    assert ledger_path.exists()


def test_autonomous_3state_seti_daemon_and_dynamic_moe_hotplug() -> None:
    """Verify 3-state SETI daemon switching and dynamic BTM MoE v2 expert hot-plugging (E -> E+1)."""
    daemon = ZeroIntrusionDaemon(
        idle_threshold_seconds=30.0,
        enforce_os_idle_hook=True,
        simulated_idle_seconds=120.0,
        simulated_active_game=None,
    )
    assert daemon.resolve_seti_agent_state() == SETIAgentState.STATE_2_IDLE_SETI_MINER
    assert daemon.resolve_seti_agent_state(consumer_query_active=True) == SETIAgentState.STATE_3_SWARM_CONSUMER

    daemon.simulated_active_game = "cyberpunk2077.exe"
    assert daemon.resolve_seti_agent_state(consumer_query_active=False) == SETIAgentState.STATE_1_GAMING_SEED_ONLY

    # Test Dynamic Domain Expert Hot-Plug (E=2 -> E=3)
    base_model, _, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model, rank_r=8, lora_alpha=16, gradient_checkpointing=False)
    init_lora = extract_lora_state_dict(bundle.model)

    branch_states = {
        "Code": {k: v.clone() for k, v in init_lora.items()},
        "Mathematics": {k: v.clone() + 0.01 for k, v in init_lora.items()},
    }
    moe_model, report = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states=branch_states,
        lora_rank_r=8,
        lora_alpha=16,
        top_k=2,
        in_place=True,
    )
    assert moe_model.num_experts == 2
    assert report.num_experts == 2

    new_domain_lora = {k: v.clone() - 0.01 for k, v in init_lora.items()}
    new_count = moe_model.hotplug_domain_expert(
        domain_name="Python_Systems_Code",
        lora_state_dict=new_domain_lora,
        lora_rank_r=8,
        lora_alpha=16,
    )
    assert new_count == 3
    assert moe_model.expert_domains == ["Code", "Mathematics", "Python_Systems_Code"]
    for blk in moe_model.get_moe_blocks():
        assert blk.router.wg.out_features == 3

    x = torch.randint(0, 256, (2, 12), dtype=torch.long)
    out = moe_model(input_ids=x, labels=x)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_three_hf_datasets_streaming_and_3block_circular_economy(tmp_path: Path) -> None:
    """Verify multi-dataset Hugging Face streaming (CodeAlpaca, PubMedQA, GSM8K) and 3-block symmetric circular economy."""
    from hivemind.cli.demo_hf_domain_swarm import fetch_hf_domain_subset

    code_samples = fetch_hf_domain_subset("sahil2801/CodeAlpaca-20k", num_samples=6)
    pubmed_samples = fetch_hf_domain_subset("qiaojin/PubMedQA", num_samples=6)
    gsm8k_samples = fetch_hf_domain_subset("openai/gsm8k", num_samples=6)

    assert len(code_samples) == 6
    assert len(pubmed_samples) == 6
    assert len(gsm8k_samples) == 6

    store = ContentAddressedSwarmStore(tmp_path / "swarm_store")
    ledger = HiveSwarmLedger(ledger_path=tmp_path / "swarm_store" / "hive_pouw_ledger.json")
    nodes = ["seti-node-paris-01", "seti-node-lyon-02", "seti-node-nice-03"]

    for exp_idx, dom in enumerate(
        ["Python_Systems_Code", "Biomedical_Clinical_QA", "Mathematical_Reasoning"], start=1
    ):
        s_receipts = []
        w_receipts = []
        for idx, nid in enumerate(nodes):
            cid, path, _ = store.store_hf_dataset_shard(nid, dom, idx, [f"{dom}_{idx}"])
            s_receipts.append(
                store.verify_proof_of_storage(nid, cid, path, ledger.chain[-1].block_hash)
            )
            w_receipts.append(
                WorkReceipt(
                    node_id=nid,
                    steam_id64=f"7656119800000000{idx + 1}",
                    branch_name=dom,
                    flops_contributed=1.0e11,
                    cosine_alignment=0.98,
                    spot_check_passed=True,
                    multikrum_accepted=True,
                    delta_sha256=f"delta_{exp_idx}_{idx}",
                )
            )
        ledger.mine_pouw_block(
            outer_step=exp_idx,
            expert_cid=f"cid://sha256/expert-{dom.lower()}",
            work_receipts=w_receipts,
            storage_receipts=s_receipts,
            validator_node_ids=["validator-node-eu-01"],
        )
        consumer = nodes[exp_idx - 1]
        providers = [n for n in nodes if n != consumer]
        ledger.settle_swarm_inference(
            consumer_node_id=consumer,
            provider_node_ids=providers,
            domain_expert=dom,
            verified_tokens=45,
            rtt_speedup=5.0,
            rate_per_token=0.005,
        )

    assert len(ledger.chain) == 4  # Genesis + 3 PoUW blocks
    assert ledger.verify_chain_integrity() is True
    # Symmetric circular economy: each node mined 45 HIVE, spent 0.225 HIVE, and earned 2 * 0.1125 HIVE = 45.0 HIVE
    for nid in nodes:
        assert ledger.balances[nid] == 45.0
    assert ledger.balances["validator-node-eu-01"] == 15.0


def test_hf_hack_miner_gamification_and_free_steam_edition(tmp_path: Path) -> None:
    """Verify Hugging Face Hack-Miner data scanner, gamer scoring, Steam achievements, and Free Steam Edition packaging."""
    from fastapi.testclient import TestClient
    from hivemind.game.hf_hack_miner import (
        GamerMinerProfile,
        HackMiningGameEngine,
        HFDataReconScanner,
        execute_interactive_bounty_probe,
        execute_interactive_hf_shard_scan,
    )
    from hivemind.game.steam_edition import (
        STEAM_APP_ID,
        build_free_steam_edition_package,
        create_hack_miner_fastapi_app,
    )

    raw_rows = [
        "### Instruction:\nWrite binary_search in Python.\n### Response:\ndef binary_search(arr, target):\n    while low <= high: return mid",
        "### Instruction:\nSort list.\n### Response:\nreturn sorted(arr)",
        "### Instruction:\nBinary search helper.\n### Response:\ndef helper(low, high): return (low + high) // 2",
    ]
    curated, sig_score = HFDataReconScanner.score_and_curate_shard(
        raw_samples=raw_rows,
        gamer_idx=0,
        num_gamers=1,
        target_keywords=["def", "low", "high", "while", "mid", "return"],
    )
    assert len(curated) == 3
    assert 0.50 <= sig_score <= 1.0

    engine = HackMiningGameEngine(store_dir=tmp_path / "engine_store")
    audit = engine.verify_all_steam_accounts()
    assert len(audit["verified_gamers"]) == 3
    assert all(g["verified"] is True for g in audit["verified_gamers"])
    assert len(audit["blocked_sybil_bots"]) == 2
    assert all(b["verified"] is False for b in audit["blocked_sybil_bots"])

    gamer = GamerMinerProfile(
        gamer_tag="CyberValkyrie_FR",
        steam_id64="76561198000000001",
        node_id="seti-node-paris-01",
        region="EU-West-Paris",
        specialty_focus="Code & Algorithmic Breaching",
        total_points=8500,
        shards_mined=3,
        bounties_solved=3,
        boss_raids_cleared=1,
        swarm_tokens_served=45,
        best_loss_reduction_pct=78.5,
    )
    gamer.update_rank_and_achievements(is_top_miner=True)
    assert "Singularity Core" in gamer.rank_title
    assert "TOP_1_APEX_MINER" in gamer.achievements_unlocked
    assert "SWARM_ARCHITECT_E3" in gamer.achievements_unlocked
    assert "MOE_BOSS_SLAYER" in gamer.achievements_unlocked

    receipt_hmac = engine.steamworks.upload_hack_miner_score(
        steam_id64=gamer.steam_id64,
        persona_name=gamer.gamer_tag,
        total_points=gamer.total_points,
        swarm_contribution_points=2500,
        hive_balance=45.0,
        achievements=gamer.achievements_unlocked,
    )
    assert receipt_hmac.startswith("steam_lb_hmac_")

    pkg = build_free_steam_edition_package(
        game_report={
            "game_title": "HiveMind: Neural Breach",
            "hardware": {"gpu_name": "NVIDIA GeForce RTX 2070", "vram_total_gb": 8.0},
            "vram_telemetry": {"peak_allocated_mb": 1326.0, "free_vram_gb": 5.59},
            "leaderboard": [],
            "missions": [],
            "moe_boss_fusion": {
                "num_experts": 3,
                "expert_domains": ["Code", "Bio", "Math"],
                "router_initial_loss": 1.8,
                "router_final_loss": 1.3,
                "expert_utilization": {"Code": 0.33, "Bio": 0.33, "Math": 0.34},
            },
        },
        output_root=tmp_path / "HiveMind_Steam_FreeEdition",
    )
    assert pkg["steam_app_id"] == STEAM_APP_ID
    assert pkg["is_free_to_play"] is True
    assert Path(pkg["steam_appid_file"]).read_text(encoding="utf-8").strip() == str(STEAM_APP_ID)
    assert Path(pkg["app_build_vdf"]).exists()
    assert Path(pkg["depot_build_vdf"]).exists()
    assert Path(pkg["steamworks_partner_schema"]).exists()
    assert Path(pkg["steam_cloud_save_file"]).exists()
    assert Path(pkg["windows_launcher_bat"]).exists()
    assert Path(pkg["windows_hud_launcher_bat"]).exists()
    assert Path(pkg["player_readme_md"]).exists()
    assert Path(pkg["interactive_hud_html"]).exists()

    # Verify interactive shard scan & bounty probe helpers and FastAPI routes
    scan_res = execute_interactive_hf_shard_scan(
        gamer_tag="CyberValkyrie_FR",
        dataset_id="sahil2801/CodeAlpaca-20k",
        custom_rows=raw_rows,
    )
    assert scan_res["merkle_pos_verified"] is True
    assert scan_res["data_discovery_pts_awarded"] >= 200

    bounty_res = execute_interactive_bounty_probe(
        gamer_tag="CyberValkyrie_FR",
        prompt="Write a Python function to calculate HIVE minted across 3 blocks.",
    )
    assert len(bounty_res["top2_active_experts"]) == 2
    assert bounty_res["total_pts_awarded"] > 700

    app = create_hack_miner_fastapi_app()
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/leaderboard").status_code == 200
    assert client.get("/api/missions").status_code == 200
    assert client.get("/api/steam-manifest").status_code == 200
    assert client.get("/api/steam-cloud").status_code == 200
    r_bounty = client.post(
        "/api/solve-bounty",
        json={"gamer_tag": "CyberValkyrie_FR", "prompt": "Mitochondria ATP synthesis in Python"},
    )
    assert r_bounty.status_code == 200
    assert r_bounty.json()["total_pts_awarded"] > 0


def test_phase2_swarm_fabric_m1_to_m4_and_openai_gateway(tmp_path: Path) -> None:
    """Verify Swarm Fabric M1 (Geo-Ring), M2 (Pinned RAM DMA + Macro/Micro Router), M3 (FP8 + Tree Speculation gamma=6), and M4 (OPoI + L2 State Channel + /v1/chat/completions)."""
    from fastapi.testclient import TestClient
    from hivemind.boinc.coordinator_api import BOINCCoordinatorService, create_coordinator_fastapi_app
    from hivemind.config import HiveMindConfig
    from hivemind.swarm.swarm_fabric import (
        GeoRingPeer,
        HiveLayer2StateChannel,
        OptimisticProofOfInferenceAuditor,
        PinnedRAMExpertDMAStreamer,
        RegionalGeoRingTransport,
        TokenTreeSpeculativeVerifier,
        TwoLevelMacroMicroRouter,
        compress_hidden_states_fp8,
        decompress_hidden_states_fp8,
    )

    # M1: Sub-20ms Kademlia Geo-Ring & PR #9 AsyncQUICGeoRingTransport 0-RTT Loopback
    from hivemind.swarm.quic_transport import AsyncQUICGeoRingTransport

    ring = RegionalGeoRingTransport(region="EU-West-Paris", max_rtt_ms=20.0)
    assert ring.register_peer(GeoRingPeer("peer-paris-01", "EU-West-Paris", 11.2)) is True
    assert ring.register_peer(GeoRingPeer("peer-slow-99", "EU-West-Paris", 42.0)) is False
    closest = ring.find_closest_peers("Python_Systems_Code", k=1)
    assert len(closest) == 1 and closest[0].peer_id == "peer-paris-01"

    quic_ring = AsyncQUICGeoRingTransport(region="EU-West-Paris", max_rtt_ms=20.0)
    udp_bench = quic_ring.benchmark_loopback_0rtt(
        torch.randn(1, 8, 32, dtype=torch.float16), num_frames=120
    )
    # Real QUIC: the second connection resumed the TLS session and the server took frames as 0-RTT data.
    assert udp_bench["protocol"] == "quic_0rtt"
    assert udp_bench["session_resumed"] is True
    assert udp_bench["early_data_accepted"] is True
    assert udp_bench["server_early_frames"] > 0
    assert udp_bench["frames_delivered"] == udp_bench["frames_verified"] == 120
    assert udp_bench["frames_per_sec"] >= 100.0
    assert udp_bench["serialization_overhead_ms"] < 1.5
    ws_bench = quic_ring.benchmark_loopback_0rtt(
        torch.randn(1, 8, 32, dtype=torch.float16), num_frames=20, simulate_udp_blocked=True
    )
    assert ws_bench["protocol"] == "libp2p_websocket"
    assert ws_bench["fallback_reason"] == "TimeoutError"
    assert ws_bench["frames_verified"] == 20

    # M2: 2-Level Macro/Micro Router & Pinned Host RAM Async DMA Streamer (PR #7 Lookahead Prefetch)
    router = TwoLevelMacroMicroRouter(
        hidden_size=32,
        macro_clusters={
            "Code_Math": ["Python_Systems_Code", "Mathematical_Reasoning"],
            "Bio_Clinical": ["Biomedical_Clinical_QA", "Legal_Medical"],
        },
        top_k_micro=2,
    )
    hidden = torch.randn(2, 6, 32)
    route_out = router(hidden)
    assert route_out["selected_macro_cluster"] in ["Code_Math", "Bio_Clinical"]
    assert len(route_out["selected_micro_experts"]) == 2

    streamer = PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    for exp_name in [
        "Python_Systems_Code",
        "Mathematical_Reasoning",
        "Biomedical_Clinical_QA",
        "Legal_Medical",
    ]:
        nbytes = streamer.stage_expert_in_pinned_ram(
            exp_name,
            {"lora_A": torch.randn(8, 32), "lora_B": torch.randn(32, 8)},
        )
        assert nbytes > 0
    dma_info = streamer.async_dma_prefetch_to_vram(
        "Python_Systems_Code", simulated_attention_compute_ms=5.0
    )
    assert dma_info["exposed_swap_latency_ms"] == 0.0
    lookahead_l0 = streamer.prefetch_lookahead_layer(
        router, hidden, layer_idx=0, simulated_attention_compute_ms=5.0
    )
    lookahead_l1 = streamer.prefetch_lookahead_layer(
        router, hidden, layer_idx=1, simulated_attention_compute_ms=5.0
    )
    assert lookahead_l0["lookahead_dma_slot"] == "slot_B"
    assert lookahead_l1["lookahead_dma_slot"] == "slot_A"
    assert lookahead_l0["exposed_swap_latency_ms"] == 0.0
    assert len(lookahead_l0["prefetched_experts"]) == 2

    # M3: FP8 Hidden-State Wire Compression & Token Tree Speculation (PR #8 Prefix-Trie gamma = 6..8)
    pkt = compress_hidden_states_fp8(hidden.to(torch.float16))
    assert pkt.compression_ratio >= 1.8
    restored = decompress_hidden_states_fp8(pkt)
    assert restored.shape == hidden.shape

    tree_verifier = TokenTreeSpeculativeVerifier(gamma=6, branching_factor=2)
    tree_res = tree_verifier.verify_token_tree(
        draft_tree_candidates=[[10, 20, 30, 99, 100, 101], [10, 20, 30, 40, 50, 60]],
        target_reference_tokens=[10, 20, 30, 40, 50, 60, 70],
    )
    assert tree_res["tree_depth_gamma"] == 6
    assert tree_res["accepted_length"] == 7  # 6 draft tokens + 1 bonus token
    assert tree_res["num_tree_nodes"] == 10
    mask_info = TokenTreeSpeculativeVerifier(gamma=8).build_causal_tree_attention_mask(
        [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 9, 10, 11, 12]]
    )
    assert mask_info["tree_depth_gamma"] == 8
    assert mask_info["causal_tree_mask"].shape == (13, 13)
    assert bool(mask_info["causal_tree_mask"][-1, 0]) is True

    # M4: OPoI 5% Activation Auditor, Layer-2 $HIVE State Channel & OpenAI /v1/chat/completions (PR #6 SSE)
    auditor = OptimisticProofOfInferenceAuditor(audit_probability=0.05)
    audit_pass = auditor.verify_activation_commitment("peer-paris-01", hidden, hidden)
    assert audit_pass["opoi_passed"] is True
    audit_fail = auditor.verify_activation_commitment("peer-poison-99", hidden, -hidden)
    assert audit_fail["opoi_passed"] is False and audit_fail["slashed"] is True

    ledger = HiveSwarmLedger(ledger_path=tmp_path / "l2_ledger.json")
    ledger.balances["seti-node-paris-01"] = 10.0
    l2 = HiveLayer2StateChannel(
        consumer_id="seti-node-paris-01",
        provider_ids=["seti-node-lyon-02", "seti-node-nice-03"],
        escrow_hive=5.0,
    )
    v1 = l2.sign_micro_voucher(20)
    v2 = l2.sign_micro_voucher(25)
    assert v2.nonce == 2 and v2.cumulative_tokens == 45
    settled = l2.settle_to_l1_ledger(ledger)
    assert settled["settled_hive"] == 0.225
    assert ledger.balances["seti-node-paris-01"] == 9.775

    cfg = HiveMindConfig.from_yaml()
    coord = BOINCCoordinatorService(
        config=cfg,
        initial_lora_state={"lora_A": torch.zeros(4, 8)},
        base_model_sha256="sha256_test",
        shards_dir=tmp_path / "shards",
    )
    app = create_coordinator_fastapi_app(coord)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "hivemind-qwen2.5-sparse-moe-top2",
            "messages": [{"role": "user", "content": "Write a Python function to compute 3 blocks * 50 HIVE."}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["hivemind_swarm_fabric"]["tree_speculation_gamma"] == 6
    # The endpoint has no activation commitment to audit, so it must say so rather than report a pass.
    assert body["hivemind_swarm_fabric"]["opoi_audit"]["status"] == "NOT_PERFORMED"
    assert body["hivemind_swarm_fabric"]["opoi_audit"]["opoi_passed"] is None

    import json

    stream_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "hivemind-qwen2.5-sparse-moe-top2",
            "stream": True,
            "messages": [{"role": "user", "content": "Write a Python function to compute 3 blocks * 50 HIVE."}],
        },
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    sse_lines = [line for line in stream_resp.text.splitlines() if line.startswith("data: ")]
    assert sse_lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in sse_lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assembled = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert assembled == body["choices"][0]["message"]["content"]
    closing_chunk = chunks[-1]
    assert closing_chunk["choices"][0]["finish_reason"] == "stop"
    assert closing_chunk["hivemind_swarm_fabric"]["opoi_audit"]["status"] == "NOT_PERFORMED"
    assert closing_chunk["hivemind_swarm_fabric"]["opoi_audit"]["opoi_passed"] is None
    assert closing_chunk["hivemind_swarm_fabric"]["l2_state_channel_voucher"]["nonce"] >= 1
    assert closing_chunk["usage"]["completion_tokens"] > 0


def test_lean4_specification_and_foundation_article_autoevolve_pipeline() -> None:
    """Verify Lean 4 formal specification (0 sorry, 0 axiom), foundation .tex/.pdf article, and AutoevolveAI peer review."""
    from paper.autoevolve_peer_review_pipeline import (
        AUTOEVOLVE_MANIFEST_JSON,
        PDF_PATH,
        PEER_REVIEW_JSON,
        TEX_PATH,
        run_autoevolve_peer_review,
        verify_lean4_specification,
    )

    lean_res = verify_lean4_specification()
    assert lean_res["sorry_count"] == 0
    assert lean_res["user_axiom_count"] == 0
    assert lean_res["theorem_count"] >= 20
    assert "prefix_trie_dedup_node_bound" in lean_res["theorems"]
    assert "speculative_rejection_sampling_target_invariance" in lean_res["theorems"]
    assert "prefix_trie_speculative_progress_bounds" in lean_res["theorems"]
    assert "rfc9001_0rtt_monotonic_nonce_anti_replay" in lean_res["theorems"]
    assert "rfc9001_0rtt_ticket_expiration_bound" in lean_res["theorems"]
    assert "wan_nat_hole_punch_rtt_bound" in lean_res["theorems"]
    assert lean_res["tier_a_machine_checked"] is True

    assert TEX_PATH.exists() and TEX_PATH.stat().st_size > 10_000
    assert PDF_PATH.exists() and PDF_PATH.stat().st_size > 100_000

    review_report = run_autoevolve_peer_review()
    # The pipeline is a scripted static audit, not a peer review: it must say so and must not emit scores.
    assert review_report["meta_review"]["decision"] == "SCRIPTED_GATES_PASSED_NOT_A_PEER_REVIEW"
    assert review_report["is_peer_review"] is False
    assert "aggregate_score" not in review_report["meta_review"]
    assert "calibrated_epistemic_confidence" not in review_report["meta_review"]
    assert all("score_out_of_10" not in ref for ref in review_report["referee_reviews"])
    assert review_report["manuscript_provenance_audit"]["all_passed"] is True
    assert PEER_REVIEW_JSON.exists()
    assert AUTOEVOLVE_MANIFEST_JSON.exists()


def test_season2_hf_hack_miner_missions_and_steamworks_rich_presence(tmp_path: Path) -> None:
    """Verify HM-M5-01 (Steam Rich Presence, Overlay, UserStats, native DLL bridge) and HM-M5-02 (Season 2 HF Missions & Live QLoRA Micro-Forge)."""
    from fastapi.testclient import TestClient
    from hivemind.cli.demo_hf_domain_swarm import fetch_hf_domain_subset
    from hivemind.client.steamworks import SteamworksSDKBridge
    from hivemind.game.hf_hack_miner import (
        ALL_HF_BOUNTY_CONTRACTS,
        SEASON_1_HF_BOUNTY_CONTRACTS,
        SEASON_2_HF_BOUNTY_CONTRACTS,
        GamerMinerProfile,
        execute_interactive_bounty_probe,
        execute_interactive_lora_micro_forge,
    )
    from hivemind.game.steam_edition import create_hack_miner_fastapi_app

    # 1. Verify HM-M5-01 Steamworks Native Bridge, Rich Presence, Overlay & UserStats
    sw = SteamworksSDKBridge()
    bridge_status = sw.get_native_bridge_status()
    assert bridge_status["app_id"] == 3133700
    assert "native_dll_loaded" in bridge_status
    assert "native_dll_path" in bridge_status

    rp = sw.set_rich_presence(
        steam_id64="76561198000000001",
        persona_name="CyberValkyrie_FR",
        mission_name="Operation: Borrow-Checker Bastion (Rust_Systems_Security)",
        rank_title="Rank VI: Singularity Core // Omega Hive Sovereign",
        hive_balance=90.0,
        total_points=21800,
    )
    assert rp.rich_presence_hmac.startswith("steam_rp_hmac_")
    assert "Rust_Systems_Security" in rp.active_mission

    # Verify PR #10 (@clark719664) positional 5-arg set_rich_presence and rich_presence_log
    rp_pr10 = sw.set_rich_presence(
        "76561198000000002",
        "PubMedQA",
        "RTX 2070",
        "Rank V: Singularity Core",
        45.0,
    )
    assert isinstance(rp_pr10, str)
    assert "Hack-Mining PubMedQA on RTX 2070" in rp_pr10
    assert sw.rich_presence_log["76561198000000002"] == str(rp_pr10)

    # Verify PR #11 (@radialmonster) docs/STEAM_STATUS.md
    steam_status_doc = Path(__file__).resolve().parents[1] / "docs" / "STEAM_STATUS.md"
    assert steam_status_doc.exists()
    assert "3133700" in steam_status_doc.read_text(encoding="utf-8")

    overlay_ev = sw.trigger_steam_overlay_notification(
        steam_id64="76561198000000001",
        persona_name="CyberValkyrie_FR",
        achievement_id="OMEGA_6_EXPERT_SLAYER",
        title="Omega 6-Expert Boss Slayer",
        description="Fuse all 6 Season 1 & Season 2 QLoRA experts.",
    )
    assert overlay_ev["achievement_id"] == "OMEGA_6_EXPERT_SLAYER"
    assert overlay_ev["overlay_corner"] == "k_EPositionBottomRight"

    stats_ev = sw.update_user_stats(
        steam_id64="76561198000000001",
        persona_name="CyberValkyrie_FR",
        shards_mined=6,
        bounties_cracked=6,
        best_loss_drop_pct=92.4,
        boss_raids_cleared=1,
        hive_mined=90.0,
    )
    assert stats_ev["stat_bounties_cracked"] == 6
    assert stats_ev["stat_hive_mined"] == 90.0

    # 2. Verify HM-M5-02 Season 2 HF Datasets & 6-Mission Contracts
    assert len(SEASON_1_HF_BOUNTY_CONTRACTS) == 3
    assert len(SEASON_2_HF_BOUNTY_CONTRACTS) == 3
    assert len(ALL_HF_BOUNTY_CONTRACTS) == 6

    rust_samples = fetch_hf_domain_subset("anon/rust-code-instructions", num_samples=6)
    sec_samples = fetch_hf_domain_subset("CyberNative/Code_Vulnerability_Security_DPO", num_samples=6)
    fr_samples = fetch_hf_domain_subset("CohereForAI/aya_collection", num_samples=6)
    assert len(rust_samples) == 6
    assert len(sec_samples) == 6
    assert len(fr_samples) == 6

    gamer_s2 = GamerMinerProfile(
        gamer_tag="CyberValkyrie_FR",
        steam_id64="76561198000000001",
        node_id="seti-node-paris-01",
        region="EU-West-Paris",
        specialty_focus="Code & Algorithmic Breaching",
        total_points=21800,
        shards_mined=6,
        bounties_solved=6,
        boss_raids_cleared=1,
        swarm_tokens_served=45,
        best_loss_reduction_pct=92.4,
        domains_forged=[
            "Python_Systems_Code",
            "Biomedical_Clinical_QA",
            "Mathematical_Reasoning",
            "Rust_Systems_Security",
            "CyberSecurity_Vulnerability_Recon",
            "Multilingual_FR_Synthesis",
        ],
    )
    gamer_s2.update_rank_and_achievements(is_top_miner=True, num_moe_experts=6)
    assert gamer_s2.rank_title == "Rank VI: Singularity Core // Omega Hive Sovereign"
    assert "RUST_BORROW_CHECKER" in gamer_s2.achievements_unlocked
    assert "ZERO_DAY_SENTINEL" in gamer_s2.achievements_unlocked
    assert "POLYGLOT_ARCHITECT_FR" in gamer_s2.achievements_unlocked
    assert "OMEGA_6_EXPERT_SLAYER" in gamer_s2.achievements_unlocked

    # 3. Verify 6-Expert routing probe & interactive QLoRA micro-forge + FastAPI routes
    probe_rust = execute_interactive_bounty_probe(
        gamer_tag="CyberValkyrie_FR",
        prompt="Write a zero-copy Rust RingBuffer with Arc<Mutex> and audit SQL injection CVE.",
    )
    assert (
        "Rust_Systems_Security" in probe_rust["top2_active_experts"]
        or "CyberSecurity_Vulnerability_Recon" in probe_rust["top2_active_experts"]
    )

    forge_res = execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-04-RUST-SYSTEMS",
    )
    assert forge_res["domain_expert"] == "Rust_Systems_Security"
    assert forge_res["loss_reduction_pct"] > 0.0
    assert forge_res["rich_presence_hmac"].startswith("steam_rp_hmac_")

    app = create_hack_miner_fastapi_app()
    client = TestClient(app)
    r_rp = client.get("/api/rich-presence")
    assert r_rp.status_code == 200
    assert "native_dll_bridge" in r_rp.json()

    r_forge = client.post(
        "/api/forge-lora-mission",
        json={"gamer_tag": "CyberValkyrie_FR", "contract_id": "CONTRACT-HF-05-CYBERSEC-AUDIT"},
    )
    assert r_forge.status_code == 200
    assert r_forge.json()["domain_expert"] == "CyberSecurity_Vulnerability_Recon"


def test_swarm_fabric_advanced_milestones_m1_to_m4(tmp_path: Path) -> None:
    """Verify HM-M1-01 (UDP 0-RTT QUIC transport), HM-M1-02 (<25ms GossipSub failover), HM-M2-01/02 (2-slot double-buffered DMA & LRU), HM-M3-01 (2D tree mask), HM-M4-01 (SSE stream), and HM-M4-02 (Merkle fraud dispute game)."""
    from fastapi.testclient import TestClient
    from hivemind.boinc.coordinator_api import BOINCCoordinatorService, create_coordinator_fastapi_app
    from hivemind.config import HiveMindConfig
    from hivemind.swarm.swarm_fabric import (
        AsyncQUICGeoRingTransport,
        GeoRingPeer,
        HiveLayer2StateChannel,
        OptimisticProofOfInferenceAuditor,
        PinnedRAMExpertDMAStreamer,
        RegionalGeoRingTransport,
        TokenTreeSpeculativeVerifier,
        compress_hidden_states_fp8,
    )

    # HM-M1-01: AsyncQUICGeoRingTransport (STUN/ICE + UDP 0-RTT datagram loopback)
    quic_tx = AsyncQUICGeoRingTransport(bind_host="127.0.0.1")
    ice_info = quic_tx.negotiate_stun_ice_candidate("peer-paris-01", local_port=4433)
    assert ice_info["protocol"] == "quic_0rtt"
    assert len(ice_info["session_ticket_0rtt"]) == 24
    assert ice_info["ice_candidate"].startswith("candidate:1 1 UDP 2130706431 127.0.0.1 ")
    assert ice_info["ice_candidate"].endswith(" typ host")

    pkt = compress_hidden_states_fp8(torch.randn(1, 4, 16, dtype=torch.float16))
    udp_res = quic_tx.transmit_compressed_hidden_state_loopback(pkt, simulate_udp_blocked=False)
    assert udp_res["frame_verified"] is True
    assert udp_res["transport_used"] == "quic_0rtt_stream"
    assert udp_res["early_data_accepted"] is True

    ws_res = quic_tx.transmit_compressed_hidden_state_loopback(pkt, simulate_udp_blocked=True)
    assert ws_res["frame_verified"] is True
    assert ws_res["transport_used"] == "libp2p_websocket_fallback"

    # HM-M1-02: <25ms GossipSub Ring Re-Slotting on Game Launch
    ring = RegionalGeoRingTransport(region="EU-West-Paris", max_rtt_ms=20.0)
    ring.register_peer(GeoRingPeer("peer-paris-01", "EU-West-Paris", 11.2, hosted_experts=["Rust_Systems_Security"]))
    ring.register_peer(GeoRingPeer("peer-lyon-02", "EU-West-Paris", 14.5, hosted_experts=["Rust_Systems_Security"]))
    failover = ring.broadcast_game_preemption_failover(
        preempting_peer_id="peer-paris-01",
        expert_domain="Rust_Systems_Security",
        active_game_exe="cyberpunk2077.exe",
    )
    assert failover["within_25ms_sla"] is True
    assert failover["promoted_standby_peer_id"] == "peer-lyon-02"

    # HM-M2-01 & HM-M2-02: 2-Slot Double-Buffered CUDA Stream Lookahead & Bounded LRU Eviction
    streamer = PinnedRAMExpertDMAStreamer(device=torch.device("cpu"), max_pinned_ram_bytes=256)
    streamer.stage_expert_in_pinned_ram("Exp1", {"w": torch.zeros(32, dtype=torch.float16)})  # 64 B
    streamer.stage_expert_in_pinned_ram("Exp2", {"w": torch.zeros(32, dtype=torch.float16)})  # 64 B
    streamer.stage_expert_in_pinned_ram("Exp3", {"w": torch.zeros(32, dtype=torch.float16)})  # 64 B
    streamer.stage_expert_in_pinned_ram("Exp4", {"w": torch.zeros(32, dtype=torch.float16)})  # 64 B (256 B full)
    streamer.stage_expert_in_pinned_ram("Exp5", {"w": torch.zeros(32, dtype=torch.float16)})  # triggers LRU eviction of Exp1
    assert "Exp1" not in streamer.pinned_host_cache
    assert "Exp5" in streamer.pinned_host_cache

    dma_l0 = streamer.async_dma_prefetch_to_vram("Exp2", simulated_attention_compute_ms=5.0, layer_idx=0)
    dma_l1 = streamer.async_dma_prefetch_to_vram("Exp3", simulated_attention_compute_ms=5.0, layer_idx=1)
    assert dma_l0["double_buffer_slot"] == "slot_A"
    assert dma_l1["double_buffer_slot"] == "slot_B"

    # HM-M3-01: 2D Causal Tree-Attention Mask (with PR #8 prefix-trie deduplication)
    verifier = TokenTreeSpeculativeVerifier(gamma=6, branching_factor=2)
    mask = verifier.build_causal_tree_attention_mask(
        draft_tree_candidates=[[10, 20, 30, 40, 50, 60], [10, 20, 30, 91, 92, 93]]
    )
    assert mask.ndim == 4
    assert mask.shape == (1, 1, 10, 10)

    # HM-M4-02: Merkle Fraud Dispute Game & L2 Escrow Refund
    auditor = OptimisticProofOfInferenceAuditor(audit_probability=0.05)
    honest_layers = [torch.ones(1, 4, 8) * (i + 1) for i in range(6)]
    byzantine_layers = [t.clone() for t in honest_layers]
    byzantine_layers[3] = -byzantine_layers[3]  # Corrupt layer 3

    ledger = HiveSwarmLedger(ledger_path=tmp_path / "l2_fraud_ledger.json")
    ledger.balances["peer-byzantine-99"] = 25.0
    l2 = HiveLayer2StateChannel(
        consumer_id="consumer-01",
        provider_ids=["peer-byzantine-99"],
        escrow_hive=5.0,
    )
    l2.sign_micro_voucher(30)

    dispute = auditor.run_merkle_fraud_dispute_game(
        peer_id="peer-byzantine-99",
        claimed_layer_activations=byzantine_layers,
        validator_layer_activations=honest_layers,
        ledger=ledger,
        l2_channel=l2,
    )
    assert dispute["fraud_proven"] is True
    assert dispute["diverging_layer_index"] == 3
    assert dispute["slashed_in_l1_ledger"] is True
    assert dispute["l2_escrow_refunded_hive"] == 0.15
    assert ledger.balances["peer-byzantine-99"] == 0.0
    assert "peer-byzantine-99" in ledger.slashed_registry

    # HM-M4-01: OpenAI /v1/chat/completions SSE Streaming (stream=True)
    cfg = HiveMindConfig.from_yaml()
    coord = BOINCCoordinatorService(
        config=cfg,
        initial_lora_state={"lora_A": torch.zeros(4, 8)},
        base_model_sha256="sha256_test",
        shards_dir=tmp_path / "shards_sse",
    )
    app = create_coordinator_fastapi_app(coord)
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "hivemind-qwen2.5-sparse-moe-top2",
            "stream": True,
            "messages": [{"role": "user", "content": "Stream token tree verification"}],
        },
    ) as sse_resp:
        assert sse_resp.status_code == 200
        assert "text/event-stream" in sse_resp.headers.get("content-type", "")
        raw_sse = "".join(sse_resp.iter_text())
        assert "chat.completion.chunk" in raw_sse
        assert "data: [DONE]" in raw_sse


def test_s3_01_multi_node_wan_nat_traversal_stun_turn_hole_punching() -> None:
    """
    Validates Season 3 Milestone HM-S3-01:
    - RFC 5389 STUN reflexive endpoint discovery & RFC 5766 TURN relay allocation
    - Simultaneous UDP hole-punching (< 5 ms/hop) across restricted/port-restricted cone NATs
    - Automatic 0-RTT TURN relay fallback when both peers are behind symmetric CGNAT
    - RFC 9001 monotonic 0-RTT anti-replay nonce enforcement
    """
    from hivemind.swarm.nat_traversal import StunServer, TurnRelayServer
    from hivemind.swarm.quic_transport import AsyncQUICGeoRingTransport

    class SymmetricNatStunServer(StunServer):
        """Fixture: reports a mapping the other peer cannot reach, as a symmetric NAT does."""

        def mapped_address_for(self, addr):
            return addr[0], (addr[1] + 7919) % 64512 + 1024

    quic = AsyncQUICGeoRingTransport(region="EU-West-Paris", max_rtt_ms=15.0)
    with StunServer() as stun, StunServer() as stun_alt, TurnRelayServer({"hive": "relay-secret"}) as turn:
        ep = quic.discover_stun_turn_endpoint(
            peer_id="wan-node-berlin-01",
            stun_servers=[stun.url, stun_alt.url],
            turn_server=turn.url,
            turn_username="hive",
            turn_password="relay-secret",
        )
        assert ep["reflexive_ip"] == "127.0.0.1"
        assert ep["stun_servers_answered"] == 2
        assert ep["mapping_behaviour"] == "no_nat" and ep["hole_punchable"] is True
        assert ep["turn_relay_endpoint"].startswith("127.0.0.1:")
        assert len(turn.allocations) == 1
        quic.release_turn_allocations()
        assert len(turn.allocations) == 0

        punch_res = quic.execute_udp_hole_punch(
            peer_a_id="wan-node-berlin-01",
            peer_b_id="wan-node-tokyo-02",
            stun_server=stun.url,
        )
        assert punch_res["direct_hole_punch_succeeded"] is True
        assert punch_res["fallback_to_turn"] is False
        assert punch_res["transport_mode"] == "direct_udp_hole_punch"
        assert punch_res["hole_punch_latency_ms"] < 5.0
        assert punch_res["sub_5ms_verified"] is True

    with SymmetricNatStunServer() as symmetric_stun, TurnRelayServer({"hive": "relay-secret"}) as turn:
        cgnat_res = quic.execute_udp_hole_punch(
            peer_a_id="wan-cgnat-mobile-01",
            peer_b_id="wan-cgnat-mobile-02",
            stun_server=symmetric_stun.url,
            turn_server=turn.url,
            turn_username="hive",
            turn_password="relay-secret",
            direct_timeout=0.5,
        )
    assert cgnat_res["direct_hole_punch_succeeded"] is False
    assert cgnat_res["fallback_to_turn"] is True
    assert cgnat_res["relay_verified"] is True
    assert cgnat_res["transport_mode"] == "turn_relay"
    assert cgnat_res["hole_punch_latency_ms"] < 5.0

    assert quic.verify_and_advance_0rtt_nonce("wan-node-berlin-01", 101) is True
    assert quic.verify_and_advance_0rtt_nonce("wan-node-berlin-01", 101) is False
    assert quic.verify_and_advance_0rtt_nonce("wan-node-berlin-01", 99) is False
    assert quic.verify_and_advance_0rtt_nonce("wan-node-berlin-01", 102) is True


def test_s3_02_triton_int4_fp8_fused_moe_router_kernel() -> None:
    """
    Validates Season 3 Milestone HM-S3-02:
    - 4-bit INT4/NF4 weight packing (2 nibbles per uint8 byte + FP16 scale) and on-the-fly dequantization
    - Fused Top-2 Sparse MoE routing + FP8 (`float8_e4m3fn`) hidden-state activation projection
    - Numerical parity (`cosine_similarity >= 0.99`) and >= 3.2x VRAM weight compression
    """
    from hivemind.swarm.swarm_fabric import (
        benchmark_fused_int4_fp8_moe_kernel,
        fused_int4_fp8_moe_route_and_project,
        pack_int4_nf4_weights,
        unpack_int4_nf4_weights,
    )

    torch.manual_seed(7)
    w = torch.randn(32, 64, dtype=torch.float32)
    packed_u8, scale_fp16 = pack_int4_nf4_weights(w)
    assert packed_u8.dtype == torch.uint8
    assert packed_u8.shape == (32, 32)
    w_restored = unpack_int4_nf4_weights(packed_u8, scale_fp16, original_in_dim=64)
    w_cos = torch.nn.functional.cosine_similarity(
        w.flatten().unsqueeze(0), w_restored.flatten().unsqueeze(0)
    ).item()
    assert w_cos >= 0.99

    x = torch.randn(2, 8, 64, dtype=torch.float32)
    router_w = torch.randn(8, 64, dtype=torch.float32)
    experts = [torch.randn(64, 64, dtype=torch.float32) for _ in range(8)]
    res = fused_int4_fp8_moe_route_and_project(x, router_w, experts, top_k=2)
    assert res["output"].shape == (2, 8, 64)
    assert res["top_k_indices"].shape == (16, 2)
    assert res["cosine_similarity"] >= 0.99
    assert res["vram_compression_ratio"] >= 3.2

    bench = benchmark_fused_int4_fp8_moe_kernel(batch_tokens=32, hidden_dim=64, num_experts=8, top_k=2)
    assert bench["parity_verified"] is True
    assert bench["cosine_similarity"] >= 0.99
    assert bench["throughput_tokens_per_sec"] > 0


def test_s3_03_season3_missions_9_12_and_webgl_georing_hud() -> None:
    """
    Validates Season 3 Milestone HM-S3-03:
    - Season 3 Hack-Miner Campaign Missions 9-12 (`SEASON_3_HF_BOUNTY_CONTRACTS`)
    - Interactive QLoRA Micro-Forge on Season 3 Mission 9-12 contracts
    - WebGL 3D Geo-Ring Topology Map in Cyberpunk HUD (`#webgl-georing-map`) and `/api/georing-topology`
    """
    from fastapi.testclient import TestClient
    from hivemind.game.hf_hack_miner import (
        ALL_HF_BOUNTY_CONTRACTS,
        FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS,
        SEASON_3_HF_BOUNTY_CONTRACTS,
        build_georing_3d_topology_snapshot,
        execute_interactive_lora_micro_forge,
    )
    from hivemind.game.steam_edition import create_hack_miner_fastapi_app

    assert len(ALL_HF_BOUNTY_CONTRACTS) == 6
    assert len(SEASON_3_HF_BOUNTY_CONTRACTS) == 4
    assert len(FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS) == 10
    s3_ids = [c.contract_id for c in SEASON_3_HF_BOUNTY_CONTRACTS]
    assert s3_ids == [
        "CONTRACT-HF-09-CROSS-CONTINENT-RELAY",
        "CONTRACT-HF-10-ZK-SPOTCHECK-AUDIT",
        "CONTRACT-HF-11-DYNAMIC-16EXPERT-FUSION",
        "CONTRACT-HF-12-SWARM-DAO-GOVERNANCE",
    ]

    forge_res = execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-09-CROSS-CONTINENT-RELAY",
    )
    assert forge_res["contract_id"] == "CONTRACT-HF-09-CROSS-CONTINENT-RELAY"
    assert forge_res["season"] == 3
    assert forge_res["total_pts_awarded"] > 1200

    topo = build_georing_3d_topology_snapshot()
    assert topo["renderer"] == "webgl_3d_georing_globe_v3"
    assert topo["num_nodes"] == 4
    assert len(topo["season3_missions"]) == 4

    app = create_hack_miner_fastapi_app()
    client = TestClient(app)
    html_resp = client.get("/")
    assert html_resp.status_code == 200
    assert "webgl-georing-map" in html_resp.text
    assert "webgl-georing-canvas" in html_resp.text

    topo_resp = client.get("/api/georing-topology")
    assert topo_resp.status_code == 200
    topo_json = topo_resp.json()
    assert topo_json["num_nodes"] == 4
    assert len(topo_json["season3_missions"]) == 4


def test_s3_04_additional_domain_expert_lora_adapters(tmp_path: Path) -> None:
    """
    Validates Season 3 Milestone HM-S3-04:
    - Training & Top-k (`k=0.05`) + FP8 (`float8_e4m3fn`) compression of the 4 Season 3
      domain LoRA adapters (`Rust_Systems_Security`, `Solidity_Smart_Contracts`,
      `Biomedical_Clinical_QA`, `Lean4_Theorem_Prover`)
    - Content-Addressed Swarm Store (`cid://sha256/expert-...`) publication & reload
    - Dynamic hot-plugging into `UpcycledSparseMoEModel` and Top-2 domain routing
    """
    from hivemind.core.btm_moe import (
        SEASON_3_DOMAIN_EXPERT_SPECS,
        upcycle_branches_to_sparse_moe,
    )
    from hivemind.data.hf_real_trainer import train_and_publish_season3_domain_experts
    from hivemind.swarm.dht_storage import ContentAddressedSwarmStore

    base_model, _, _ = build_local_qwen2_backbone(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model, rank_r=8, lora_alpha=16, gradient_checkpointing=False)
    ref_lora = extract_lora_state_dict(bundle.model)

    moe_model, _ = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states={
            "Python_Systems_Code": {k: v.clone() for k, v in ref_lora.items()},
            "Mathematical_Reasoning": {k: v.clone() + 0.01 for k, v in ref_lora.items()},
        },
        lora_rank_r=8,
        lora_alpha=16,
        top_k=2,
        in_place=True,
    )
    assert moe_model.num_experts == 2

    store = ContentAddressedSwarmStore(tmp_path / "s3_cas_store")
    registry, summary = train_and_publish_season3_domain_experts(
        store=store,
        reference_lora_state=ref_lora,
        moe_model=moe_model,
        topk_ratio=0.05,
        inner_steps=10,
    )
    assert summary["num_domains_trained"] == 4
    assert set(registry.list_domains()) == set(SEASON_3_DOMAIN_EXPERT_SPECS.keys())
    assert moe_model.num_experts == 6

    for dom, rec in summary["receipts"].items():
        assert rec["expert_cid"].startswith(f"cid://sha256/expert-{dom.lower()}-")
        assert rec["loss_decreased"] is True
        assert rec["compression_ratio"] > 1.0
        loaded_cid, loaded_meta, loaded_state = store.load_lora_expert_checkpoint(rec["expert_cid"])
        assert loaded_cid == rec["expert_cid"]
        assert set(loaded_state.keys()) == set(ref_lora.keys())
        assert loaded_meta["domain_name"] == dom

    top2_solidity, _ = registry.route_domain_query(
        "Audit this Solidity ERC20 smart contract for reentrancy in msg.sender", top_k=2
    )
    assert top2_solidity[0] == "Solidity_Smart_Contracts"

    top2_lean, _ = registry.route_domain_query(
        "Prove this Lean 4 theorem by induction and omega tactic", top_k=2
    )
    assert top2_lean[0] == "Lean4_Theorem_Prover"

    x = torch.randint(0, 256, (2, 8), dtype=torch.long)
    out = moe_model(input_ids=x, labels=x)
    assert out.loss is not None


def _s4_05_synthetic_lora_pair(
    rank_r: int,
    in_features: int,
    out_features: int,
    decay: float = 0.85,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a LoRA `(A, B)` pair whose effective delta `B @ A` has a controlled singular spectrum."""
    gen = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(out_features, rank_r, generator=gen))
    v, _ = torch.linalg.qr(torch.randn(in_features, rank_r, generator=gen))
    singular = decay ** torch.arange(rank_r, dtype=torch.float32)
    sqrt_s = torch.sqrt(singular)
    lora_b = u * sqrt_s.unsqueeze(0)  # [out_features, r]
    lora_a = sqrt_s.unsqueeze(1) * v.t()  # [r, in_features]
    return lora_a, lora_b


def _s4_05_synthetic_adapter(
    rank_r: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Build a PEFT-style LoRA state dict of rank `rank_r` for every MLP projection."""
    state: dict[str, torch.Tensor] = {}
    proj_idx = 0
    for layer_idx in range(num_layers):
        for proj, (in_f, out_f) in {
            "gate_proj": (hidden_size, intermediate_size),
            "up_proj": (hidden_size, intermediate_size),
            "down_proj": (intermediate_size, hidden_size),
        }.items():
            lora_a, lora_b = _s4_05_synthetic_lora_pair(
                rank_r, in_f, out_f, seed=seed + 17 * layer_idx + proj_idx
            )
            prefix = f"base_model.model.model.layers.{layer_idx}.mlp.{proj}"
            state[f"{prefix}.lora_A.default.weight"] = lora_a
            state[f"{prefix}.lora_B.default.weight"] = lora_b
            proj_idx += 1
    return state


def _s4_05_build_moe(
    seed: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    vocab_size: int,
) -> UpcycledSparseMoEModel:
    """Build a 2-expert upcycled Sparse MoE with a shared (VRAM-efficient) base FFN."""
    base_model, _, _ = build_local_qwen2_backbone(
        seed=seed,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model, rank_r=16, lora_alpha=32, gradient_checkpointing=False
    )
    ref_lora = extract_lora_state_dict(bundle.model)
    moe_model, _ = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states={
            "Code": {k: v.clone() for k, v in ref_lora.items()},
            "Mathematics": {k: v.clone() + 0.01 for k, v in ref_lora.items()},
        },
        lora_rank_r=16,
        lora_alpha=32,
        top_k=2,
        in_place=True,
    )
    return moe_model


def test_s4_01_multi_vendor_rocm_mps_dma_and_moe_dispatch() -> None:
    from hivemind.config import (
        detect_gpu_backend,
        detect_gpu_vendor,
        detect_local_hardware,
        rocm_supports_native_bf16,
    )
    from hivemind.swarm.swarm_fabric import (
        PinnedRAMExpertDMAStreamer,
        TwoLevelMacroMicroRouter,
        fused_int4_fp8_moe_route_and_project,
    )

    hw = detect_local_hardware()
    assert hw["compute_backend"] in {"cpu", "cuda", "mps", "rocm"}
    assert hw["gpu_backend"] == hw["compute_backend"]
    assert detect_gpu_vendor(detect_gpu_backend()) in {"nvidia", "amd", "apple", "cpu"}
    assert rocm_supports_native_bf16("gfx1100") is True
    assert rocm_supports_native_bf16("gfx1030") is False
    assert "mps_available" in hw and "rocm_available" in hw

    torch.manual_seed(4)
    device = torch.device("cpu")
    if hw["compute_backend"] == "cuda":
        device = torch.device("cuda:0")
    elif hw["compute_backend"] == "mps":
        device = torch.device("mps")
    elif hw["compute_backend"] == "rocm":
        device = torch.device("cuda:0")

    x = torch.randn(2, 4, 32, device=device)
    router_w = torch.randn(4, 32, device=device)
    experts = [torch.randn(32, 32, device=device) for _ in range(4)]
    routed = fused_int4_fp8_moe_route_and_project(x, router_w, experts, top_k=2)
    assert routed["cosine_similarity"] >= 0.99

    router = TwoLevelMacroMicroRouter(
        hidden_size=32,
        macro_clusters={"Code_Math": ["Python_Systems_Code", "Mathematical_Reasoning"]},
        top_k_micro=1,
    )
    streamer = PinnedRAMExpertDMAStreamer(device=device)
    assert streamer.staging_mode in {"cpu_copy", "cuda_stream", "mps_unified", "hip_stream"}
    streamer.stage_expert_in_pinned_ram("Python_Systems_Code", {"lora_A": torch.randn(4, 32)})
    streamer.stage_expert_in_pinned_ram("Mathematical_Reasoning", {"lora_A": torch.randn(4, 32)})
    prefetch = streamer.prefetch_lookahead_layer(router, torch.randn(1, 2, 32), layer_idx=0)
    assert prefetch["staging_mode"] == streamer.staging_mode
    assert prefetch["prefetched_layer"] == 1


def test_s4_02_hierarchical_continental_super_ring_routing() -> None:
    from hivemind.swarm.quic_transport import (
        AdaptiveCongestionController,
        ContinentalSuperRingRouter,
        RegionalBridgePeer,
        SuperRingPeerNode,
    )

    router = ContinentalSuperRingRouter(
        [
            RegionalBridgePeer("paris", "EU", 12.0),
            RegionalBridgePeer("berlin", "EU", 9.5),
            RegionalBridgePeer("virginia", "US", 18.0),
            RegionalBridgePeer("ohio", "US", 14.0, stun_verified=False),
            RegionalBridgePeer("tokyo", "APAC", 22.0),
            RegionalBridgePeer("osaka", "APAC", 19.0),
        ]
    )
    bridges = router.elect_bridges()
    assert bridges["EU"].peer_id == "berlin"
    assert bridges["US"].peer_id == "virginia"
    assert bridges["APAC"].peer_id == "osaka"
    route = router.route("EU", "APAC")
    assert route["hop_count"] == 3
    assert route["hops"] == ["regional", "continental_bridge", "remote_regional"]
    assert route["rtt_ms"] == round(9.5 + max(9.5, 19.0) + 19.0, 3)
    assert ContinentalSuperRingRouter.bbr_window(16, jitter_ms=0.2, loss=0.0) == 16
    assert ContinentalSuperRingRouter.bbr_window(16, jitter_ms=8.0, loss=0.5) < 16

    # Also test peer-to-peer 2-Tier Super-Ring & BBRv1 AdaptiveCongestionController (PR #37)
    peer_router = ContinentalSuperRingRouter(max_regional_rtt_ms=20.0)
    assert peer_router.register_peer(SuperRingPeerNode("eu-paris-1", "EU-West-Paris", 4.2, stun_verified=True))
    assert peer_router.register_peer(SuperRingPeerNode("us-va-1", "US-East-Virginia", 6.1, stun_verified=True))
    p_route = peer_router.route("eu-paris-1", "us-va-1")
    assert p_route["tier"] == "continental_bridge"
    assert p_route["cross_continent"] is True
    bbr = AdaptiveCongestionController()
    bbr.record_rtt_sample(12.0)
    bbr.record_delivery(24_000, 12.0)
    w_clean = bbr.compute_target_window(rtt_ms=12.0, jitter_ms=0.2, loss_ratio=0.0)
    w_lossy = bbr.compute_target_window(rtt_ms=12.0, jitter_ms=8.0, loss_ratio=0.25)
    assert w_lossy.cwnd_packets <= w_clean.cwnd_packets


def test_s4_03_opoi_merkle_inclusion_and_dual_bound_audit() -> None:
    from hivemind.swarm.swarm_fabric import OptimisticProofOfInferenceAuditor

    torch.manual_seed(11)
    layers = [torch.randn(2, 8) for _ in range(4)]
    auditor = OptimisticProofOfInferenceAuditor(max_normalized_linf=0.05)
    proof = auditor.build_merkle_inclusion_proof(2, layers)
    assert auditor.verify_merkle_inclusion_proof(proof) is True
    assert proof["depth"] <= len(layers) - 1
    tampered = dict(proof)
    tampered["leaf"] = "0" * 64
    assert auditor.verify_merkle_inclusion_proof(tampered) is False

    # Also verify PR #35 quantized layer commitment & dual-bound audit
    commit_info = auditor.commit_layer_activations(layers)
    assert commit_info["num_layers"] == 4
    q_proof = auditor.build_merkle_inclusion_proof(1)
    q_ver = auditor.verify_merkle_inclusion_proof(q_proof)
    assert q_ver["included"] is True and q_ver["is_log_l_proof"] is True
    scale_forged = layers[0] * 5.0
    q_audit = auditor.audit_quantized_activation("peer", scale_forged, layers[0])
    assert q_audit["cosine_passed"] is True
    assert q_audit["linf_passed"] is False and q_audit["slashed"] is True

    honest = auditor.verify_activation_commitment("peer", layers[0], layers[0])
    assert honest["opoi_passed"] is True
    spiked = layers[0].clone()
    spiked.view(-1)[0] += 50.0
    fraud = auditor.verify_activation_commitment("peer", spiked, layers[0])
    assert fraud["opoi_passed"] is False and fraud["slashed"] is True


def test_s4_04_season4_missions_13_16_and_live_telemetry() -> None:
    """
    Validates Season 4 Milestone HM-S4-04:
    - Season 4 Hack-Miner Campaign Missions 13-16 (`SEASON_4_HF_BOUNTY_CONTRACTS`)
    - `/api/season4-missions` REST catalog of the Season 4 bounty contracts
    - Live `/ws/swarm-telemetry` WebSocket HUD stream (`swarm-telemetry` telemetry frames)
    """
    from fastapi.testclient import TestClient

    from hivemind.game.hf_hack_miner import (
        SEASON_4_EXTENDED_HF_BOUNTY_CONTRACTS,
        SEASON_4_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS,
        SEASON_4_HF_BOUNTY_CONTRACTS,
        build_season4_mission_catalog,
        build_swarm_telemetry_frame,
        execute_interactive_lora_micro_forge,
    )
    from hivemind.game.steam_edition import create_hack_miner_fastapi_app

    ids = [c.contract_id for c in SEASON_4_HF_BOUNTY_CONTRACTS]
    assert ids == [
        "CONTRACT-HF-13-TRITON-KERNEL-SYNTH",
        "CONTRACT-HF-14-ZK-CIRCUIT-PROVER",
        "CONTRACT-HF-15-QUANTUM-QASM-OPT",
        "CONTRACT-HF-16-AUTONOMOUS-AGENT-SWARM",
    ]
    assert all(c.season == 4 for c in SEASON_4_HF_BOUNTY_CONTRACTS)
    assert len(SEASON_4_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS) == 14
    assert len(SEASON_4_EXTENDED_HF_BOUNTY_CONTRACTS) == 16

    catalog = build_season4_mission_catalog()
    assert catalog["season"] == 4
    assert catalog["num_missions"] == 4
    assert catalog["mission_ids"] == ids
    assert catalog["bounty_pool_hive"] == 1800.0

    frame = build_swarm_telemetry_frame(0)
    assert frame["stream"] == "swarm-telemetry"
    assert frame["season4_missions"] == 4
    assert frame["mission_ids"] == ids

    forge_res = execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-13-TRITON-KERNEL-SYNTH",
    )
    assert forge_res["contract_id"] == "CONTRACT-HF-13-TRITON-KERNEL-SYNTH"
    assert forge_res["season"] == 4
    assert forge_res["total_pts_awarded"] > 1800

    app = create_hack_miner_fastapi_app()
    client = TestClient(app)

    body = client.get("/api/season4-missions").json()
    assert body["missions"] == ids
    assert body["season"] == 4
    assert len(body["contracts"]) == 4

    html_resp = client.get("/")
    assert html_resp.status_code == 200
    assert "season4-missions-panel" in html_resp.text
    assert "swarm-telemetry" in html_resp.text

    with client.websocket_connect("/ws/swarm-telemetry") as socket:
        ws_frame = socket.receive_json()
    assert ws_frame["stream"] == "swarm-telemetry"
    assert ws_frame["season4_missions"] == 4
    assert ws_frame["mission_ids"] == ids


def test_s4_05_svd_rank_adaptive_lora_projection_and_moe_scaling() -> None:
    """
    Validates Season 4 Milestone HM-S4-05 (PR #33 & PR #34):
    - `project_lora_rank_svd` truncated-SVD rank-adaptive projection (`r=8, 16, 32, 64 -> r=16`)
    - Eckart-Young optimality of the compressed delta, exact padding of low-rank adapters
      and preservation of the `alpha / r` scaling convention across ranks
    - Hot-plugging 32 heterogeneous-rank community experts into `UpcycledSparseMoEModel`
      while staying inside the `< 2.0 GB` VRAM budget (no rank-induced VRAM bloat)
    """
    from hivemind.core.btm_moe import project_lora_rank_svd, project_lora_rank_svd_with_report
    from hivemind.core.btm_upcycle import (
        UpcycledSparseMoEModel,
        detect_lora_rank,
        hotplug_rank_adaptive_experts,
        measure_moe_vram_footprint_gb,
    )

    # --- 0. Standalone 2D factor projection (PR #33 API) ---
    torch.manual_seed(5)
    wide = {"lora_B": torch.randn(64, 32)}
    narrow = {"lora_B": torch.randn(64, 8)}
    wide_out = project_lora_rank_svd(wide, target_rank_r=16)
    narrow_out = project_lora_rank_svd(narrow, target_rank_r=16)
    assert tuple(wide_out["state_dict"]["lora_B"].shape) == (64, 16)
    assert tuple(narrow_out["state_dict"]["lora_B"].shape) == (64, 16)
    assert wide_out["factors"][0]["spectral_energy_kept"] <= 1.0

    target_r = 16
    in_f, out_f = 64, 96
    a_key = "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight"
    b_key = "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight"

    # --- 1. r=8 adapters are zero-padded to the canonical r=16 with an exact delta ---
    a8, b8 = _s4_05_synthetic_lora_pair(8, in_f, out_f, seed=8)
    state8 = {a_key: a8, b_key: b8}
    projected8, report8 = project_lora_rank_svd_with_report(state8, target_rank_r=target_r)
    assert set(projected8.keys()) == set(state8.keys())
    assert projected8[a_key].shape == (target_r, in_f)
    assert projected8[b_key].shape == (out_f, target_r)
    assert report8.source_rank_r == 8
    assert report8.target_rank_r == target_r
    assert report8.num_padded_pairs == 1
    assert report8.num_identity_pairs == 0
    assert report8.mean_spectral_energy_retained == 1.0
    assert report8.max_relative_delta_error < 1e-4
    assert torch.allclose(projected8[b_key] @ projected8[a_key], b8 @ a8, atol=1e-4)

    # --- 2. r=64 adapters are compressed onto the Eckart-Young optimal rank-16 delta ---
    a64, b64 = _s4_05_synthetic_lora_pair(64, in_f, out_f, seed=64)
    state64 = {a_key: a64, b_key: b64}
    projected64, report64 = project_lora_rank_svd_with_report(state64, target_rank_r=target_r)
    assert projected64[a_key].shape == (target_r, in_f)
    assert projected64[b_key].shape == (out_f, target_r)
    assert report64.source_rank_r == 64
    assert report64.num_projected_pairs == 1
    assert report64.num_padded_pairs == 0
    assert report64.compression_ratio == 4.0

    delta64 = b64 @ a64
    u, s, vh = torch.linalg.svd(delta64, full_matrices=False)
    optimal_delta = (u[:, :target_r] * s[:target_r]) @ vh[:target_r, :]
    assert torch.allclose(projected64[b_key] @ projected64[a_key], optimal_delta, atol=1e-4)
    expected_energy = float((s[:target_r] ** 2).sum() / (s**2).sum())
    assert abs(report64.mean_spectral_energy_retained - expected_energy) < 1e-4
    assert report64.mean_spectral_energy_retained > 0.99

    # --- 3. r=32 adapters are compressed and r=16 adapters stay at the canonical rank ---
    a32, b32 = _s4_05_synthetic_lora_pair(32, in_f, out_f, seed=32)
    projected32, report32 = project_lora_rank_svd_with_report(
        {a_key: a32, b_key: b32}, target_rank_r=target_r
    )
    assert report32.source_rank_r == 32
    assert projected32[a_key].shape == (target_r, in_f)
    assert report32.mean_spectral_energy_retained > 0.99

    a16, b16 = _s4_05_synthetic_lora_pair(16, in_f, out_f, seed=16)
    projected16, report16 = project_lora_rank_svd_with_report(
        {a_key: a16, b_key: b16}, target_rank_r=target_r
    )
    assert report16.num_identity_pairs == 1
    assert report16.mean_spectral_energy_retained == 1.0
    assert torch.allclose(projected16[b_key] @ projected16[a_key], b16 @ a16, atol=1e-4)

    # --- 4. The `alpha / r` scaling convention is preserved across ranks ---
    rescaled = project_lora_rank_svd(
        {a_key: a16, b_key: b16}, target_rank_r=target_r, source_lora_alpha=16, target_lora_alpha=32
    )
    assert torch.allclose(rescaled[b_key] @ rescaled[a_key], 0.5 * (b16 @ a16), atol=1e-4)

    source_snapshot = {k: v.clone() for k, v in state64.items()}
    project_lora_rank_svd(state64, target_rank_r=target_r)
    for key, tensor in state64.items():
        assert torch.equal(tensor, source_snapshot[key])

    # --- 5. Hot-plug 32 heterogeneous-rank community experts inside a < 2.0 GB VRAM budget ---
    hidden_size, intermediate_size, num_layers, vocab_size = 512, 1024, 6, 512
    moe_model = _s4_05_build_moe(
        seed=42,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    assert moe_model.num_experts == 2

    ranks = [8, 16, 32, 64]
    community_adapters = {
        f"Community_Domain_{idx:02d}": _s4_05_synthetic_adapter(
            rank_r=ranks[idx % len(ranks)],
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            seed=1_000 + idx,
        )
        for idx in range(30)
    }

    hotplug_result = hotplug_rank_adaptive_experts(
        moe_model,
        community_adapters,
        target_rank_r=target_r,
        vram_budget_gb=2.0,
    )
    assert hotplug_result["initial_num_experts"] == 2
    assert hotplug_result["final_num_experts"] == 32
    assert hotplug_result["num_plugged_experts"] == 30
    assert hotplug_result["within_vram_budget"] is True
    assert hotplug_result["vram_footprint_after_gb"] < 2.0

    for blk in moe_model.get_moe_blocks():
        assert blk.router.wg.out_features == 32
        for expert in blk.experts:
            assert expert.gate_A.shape[0] == target_r
            assert expert.up_A.shape[0] == target_r
            assert expert.down_A.shape[0] == target_r

    for idx, domain in enumerate(sorted(community_adapters.keys())):
        report = hotplug_result["projection_reports"][domain]
        assert report["source_rank_r"] == ranks[idx % len(ranks)]
        assert report["target_rank_r"] == target_r
        assert report["mean_spectral_energy_retained"] > 0.95
        assert report["max_relative_delta_error"] < 0.5

    projected_footprint = measure_moe_vram_footprint_gb(moe_model)
    assert projected_footprint["total_gb"] < 2.0
    assert projected_footprint["num_experts"] == 32

    raw_moe_model = _s4_05_build_moe(
        seed=42,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    for domain, adapter in community_adapters.items():
        raw_moe_model.hotplug_domain_expert(
            domain_name=domain,
            lora_state_dict=adapter,
            lora_rank_r=detect_lora_rank(adapter),
            lora_alpha=2 * detect_lora_rank(adapter),
        )
    raw_footprint = measure_moe_vram_footprint_gb(raw_moe_model)
    assert raw_moe_model.num_experts == 32
    assert raw_footprint["total_gb"] > projected_footprint["total_gb"]
    assert raw_footprint["total_gb"] < 2.0

    x = torch.randint(0, vocab_size, (2, 8), dtype=torch.long)
    out = moe_model(input_ids=x, labels=x)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    util = moe_model.get_expert_utilization()
    assert abs(sum(util.values()) - 1.0) < 1e-2
    assert len(util) == 32


def test_s4_06_lean4_season4_dao_and_superring_theorems() -> None:
    from paper.autoevolve_peer_review_pipeline import verify_lean4_specification

    audit = verify_lean4_specification()
    assert audit["theorem_count"] >= 32
    assert audit["sorry_count"] == 0
    assert audit["user_axiom_count"] == 0
    names = set(audit["theorems"])
    for theorem in (
        "quadratic_voting_cost_bound",
        "steamworks_sybil_account_cost_lower_bound",
        "continental_super_ring_hop_bound",
        "svd_rank_truncation_parameter_savings",
        "opoi_merkle_inclusion_depth_bound",
    ):
        assert theorem in names



