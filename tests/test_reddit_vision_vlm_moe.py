# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and integration tests for the Reddit-Vision HiveMind Culturally-Grounded VLM MoE Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app
from hivemind.core.reddit_vision_vlm_moe import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    MAX_CONSUMER_8GB_VRAM_MIB,
    REDDIT_VISION_BASE_MODELS,
    REDDIT_VISION_EXPERT_CLUSTERS,
    ByzantineMultiKrumAggregator,
    DHTDeltaRegistry,
    FrozenVisionLanguageBackbone,
    QLoRAVisionAdapter,
    UpcycledRedditVisionSparseMoE,
    build_cluster_synthetic_shard,
    dequantize_adamw_state_8bit,
    dequantize_nf4_blockwise,
    dequantize_sparseloco_dct_2bit,
    quantize_adamw_state_8bit,
    quantize_nf4_blockwise,
    quantize_sparseloco_dct_2bit,
    run_diloco_qlora_worker_training,
    run_nocturnal_reddit_vision_moe_campaign,
)


def test_reddit_vision_catalogs_and_vram_budgets() -> None:
    """Verify the 6 open-weight VLM backbones and 3 culturally-grounded expert clusters."""
    assert len(REDDIT_VISION_BASE_MODELS) == 6
    expected_backbones = {
        "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "dinov2-vit-b14": "facebook/dinov2-base",
        "siglip-so400m": "google/siglip-so400m-patch14-384",
        "janus-pro-1b": "deepseek-ai/Janus-Pro-1B",
        "janus-pro-7b": "deepseek-ai/Janus-Pro-7B",
    }
    for key, model_id in expected_backbones.items():
        assert key in REDDIT_VISION_BASE_MODELS
        meta = REDDIT_VISION_BASE_MODELS[key]
        assert meta["model_id"] == model_id
        assert meta["peak_vram_mib_8gb_safe"] <= MAX_CONSUMER_8GB_VRAM_MIB

    assert len(REDDIT_VISION_EXPERT_CLUSTERS) == 3
    assert "laion/aesthetic-visual-dataset" in (
        REDDIT_VISION_EXPERT_CLUSTERS["expert_1_aesthetics_art"]["primary_hf_dataset"]
    )
    assert "poloclub/diffusiondb" in (
        REDDIT_VISION_EXPERT_CLUSTERS["expert_1_aesthetics_art"]["secondary_hf_datasets"]
    )
    assert "HuggingFaceM4/Docmatix" in (
        REDDIT_VISION_EXPERT_CLUSTERS["expert_2_social_meme_culture"]["primary_hf_dataset"]
    )
    assert "lmms-lab/ChartQA" in (
        REDDIT_VISION_EXPERT_CLUSTERS["expert_3_technical_infographics"]["primary_hf_dataset"]
    )
    assert "lmms-lab/MathVision" in (
        REDDIT_VISION_EXPERT_CLUSTERS["expert_3_technical_infographics"]["secondary_hf_datasets"]
    )


def test_stage1_frozen_backbone_and_nf4_8bit_adamw_quantization() -> None:
    """Verify Stage 1 base freeze (ViT + LLM Attention) and NF4 / 8-bit AdamW quantizers."""
    backbone = FrozenVisionLanguageBackbone(vision_dim=48, hidden_dim=64, intermediate_dim=128)
    assert backbone.verify_base_freeze() is True

    # Test non-multiple-of-64 padding in NF4 blockwise quantizer
    rng = np.random.default_rng(42)
    w = rng.standard_normal((17, 19)).astype(np.float32) * 0.5
    idx, absmax = quantize_nf4_blockwise(w, block_size=64)
    w_rec = dequantize_nf4_blockwise(idx, absmax, w.shape)
    assert w_rec.shape == w.shape
    assert np.all(np.isfinite(w_rec))

    # Test 8-bit AdamW moment quantizer
    m = torch.randn(16, 64) * 0.1
    q8, sc = quantize_adamw_state_8bit(m)
    assert q8.dtype == np.int8
    m_rec = dequantize_adamw_state_8bit(q8, sc, torch.device("cpu"))
    assert torch.allclose(m, m_rec, atol=2e-3)


def test_stage3_sparseloco_dct_2bit_and_dht_registry() -> None:
    """Verify 2-bit SparseLoCo + 2D DCT-II compression and DHT content-addressed exchange."""
    rng = np.random.default_rng(99)
    delta = rng.standard_normal((64, 64)).astype(np.float32) * 0.08
    pkt = quantize_sparseloco_dct_2bit(
        delta_matrix=delta,
        bee_id="bee_vlm_01_aesthetic_composition",
        cluster_id="expert_1_aesthetics_art",
        backbone_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    assert pkt.compression_ratio > 8.0
    assert pkt.sqnr_db > 10.0

    dht = DHTDeltaRegistry()
    key = dht.publish(pkt)
    assert key in dht.list_keys()
    fetched = dht.fetch(key)
    recon = dequantize_sparseloco_dct_2bit(fetched)
    assert recon.shape == (64, 64)
    assert np.all(np.isfinite(recon))

    with pytest.raises(KeyError):
        dht.fetch("dht://hivemind/reddit-vision/missing/key")


def test_end_to_end_5_stage_reddit_vision_campaign_and_fastapi_routes(tmp_path: Path) -> None:
    """Verify all 5 stages of the Reddit-Vision VLM MoE campaign and FastAPI Night-MoE endpoints."""
    out_json = tmp_path / "nocturnal_reddit_vision_report.json"
    report = run_nocturnal_reddit_vision_moe_campaign(
        inner_steps=45,
        lora_rank=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
        output_json_path=out_json,
    )
    assert out_json.exists()
    assert report.stage1_all_base_frozen is True
    assert report.stage2_mean_loss_reduction_pct > 25.0
    assert report.stage2_invariant_i1_satisfied is True
    assert report.stage3_mean_compression_ratio > 8.0
    assert report.stage3_mean_sqnr_db > 10.0
    assert len(report.stage3_dht_keys_published) == 6
    assert report.stage4_poison_rejection_rate_pct == 100.0
    assert report.stage4_all_consensus_passed is True
    assert report.stage5_top_k_active == 2
    assert report.stage5_total_experts == 3
    assert report.stage5_cluster_routing_accuracy >= 0.85
    assert report.stage5_upcycled_moe_loss < report.stage5_single_model_baseline_loss
    assert report.overall_gate_passed is True
    assert len(report.report_sha256) == 64

    # Verify FastAPI Night-MoE Reddit-Vision endpoints
    app = create_hivemind_at_home_app(vault_dir=tmp_path / "vault")
    client = TestClient(app)
    r_status = client.get("/api/v1/night-moe/reddit-vision/status")
    assert r_status.status_code == 200
    assert r_status.json()["status"] == "REDDIT_VISION_VLM_READY"
    assert len(r_status.json()["base_models"]) == 6
    assert len(r_status.json()["expert_clusters"]) == 3

    r_start = client.post("/api/v1/night-moe/reddit-vision/start", json={"inner_steps": 15})
    assert r_start.status_code == 200
    assert r_start.json()["campaign_id"] == "NIGHT-REDDIT-VISION-MOE-V1"
    assert r_start.json()["stage1_all_base_frozen"] is True


def test_la_ruche_1500mb_vmoe_muon_btm_zk_opoi_and_fomoe() -> None:
    """Verify < 1.5 GB (<= 1,536 MiB) La Ruche V-MoE with Muon, BTM, ZK-OPoI, and FoMoE Top-2."""
    from hivemind.core.reddit_vision_vlm_moe import (
        LA_RUCHE_1500MB_VMOE_BACKBONES,
        LA_RUCHE_VMOE_4_SPECIALIZATIONS,
        MAX_LA_RUCHE_BACKGROUND_VRAM_MIB,
        generate_zk_opoi_training_receipt,
        merge_btm_attention_weights,
        newton_schulz_orthogonalize_muon,
        run_la_ruche_1500mb_patch_vmoe_pipeline,
        verify_zk_opoi_training_receipt,
    )

    assert len(LA_RUCHE_1500MB_VMOE_BACKBONES) == 4
    for info in LA_RUCHE_1500MB_VMOE_BACKBONES.values():
        assert info["peak_vram_mib"] <= MAX_LA_RUCHE_BACKGROUND_VRAM_MIB
    assert set(LA_RUCHE_VMOE_4_SPECIALIZATIONS.keys()) == {
        "art_aesthetics",
        "scientific_diagrams",
        "semantic_segmentation",
        "object_detection",
    }

    # Muon Newton-Schulz orthogonalization (2D, tall 2D, 3D, and zero tensor)
    g_tall = np.random.default_rng(11).normal(size=(16, 8)).astype(np.float32)
    o_tall = newton_schulz_orthogonalize_muon(g_tall, steps=5)
    assert o_tall.shape == (16, 8)
    g_3d = np.random.default_rng(12).normal(size=(4, 4, 4)).astype(np.float32)
    o_3d = newton_schulz_orthogonalize_muon(g_3d, steps=3)
    assert o_3d.shape == (4, 4, 4)
    g_zero = np.zeros((8, 8), dtype=np.float32)
    assert np.allclose(newton_schulz_orthogonalize_muon(g_zero), 0.0)

    # BTM attention weight averaging
    t1 = {"q": torch.ones((4, 4)), "v": torch.zeros((4, 4))}
    t2 = {"q": torch.ones((4, 4)) * 3.0, "v": torch.ones((4, 4)) * 2.0}
    merged = merge_btm_attention_weights([t1, t2])
    assert torch.allclose(merged["q"], torch.ones((4, 4)) * 2.0)
    assert torch.allclose(merged["v"], torch.ones((4, 4)) * 1.0)
    with pytest.raises(ValueError, match="at least one branch"):
        merge_btm_attention_weights([])

    # ZK-OPoI receipt generation and verification
    d0 = np.zeros((8, 8), dtype=np.float32)
    d1 = np.ones((8, 8), dtype=np.float32)
    receipt = generate_zk_opoi_training_receipt(
        bee_id="bee_01",
        specialization_id="art_aesthetics",
        inner_steps=500,
        optimizer_name="Muon",
        seed=42,
        initial_delta=d0,
        final_delta=d1,
        dct_packet_sha256="abc123",
    )
    assert verify_zk_opoi_training_receipt(receipt, expected_final_delta=d1) is True
    assert verify_zk_opoi_training_receipt(receipt, expected_final_delta=d0) is False

    # End-to-end < 1.5 GB La Ruche V-MoE pipeline
    summary = run_la_ruche_1500mb_patch_vmoe_pipeline(inner_steps=50, seed=99)
    assert summary["strictly_under_1500mb_vram"] is True
    assert summary["max_worker_vram_mib"] <= MAX_LA_RUCHE_BACKGROUND_VRAM_MIB
    assert summary["all_zk_opoi_receipts_verified"] is True
    assert summary["fomoe_patch_routing_accuracy"] == 1.0
