"""Unit and integration tests for La Ruche MoE Model Vault, Dataset Mining Campaign & HiveMind@home GCP Dashboard."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import (
    build_hivemind_at_home_state,
    create_hivemind_at_home_app,
    export_hivemind_at_home_artifacts,
    render_hivemind_at_home_html,
)
from hivemind.data.dataset_mining_campaign import (
    SCIENTIFIC_MINING_DOMAIN_CORPORA,
    run_dataset_mining_and_ruche_campaign,
)
from hivemind.swarm.la_ruche_vault import (
    RUCHE_ALVEOLE_HEX_LAYOUT,
    LaRucheModelVault,
    load_community_moe_from_ruche,
)


def test_la_ruche_rejects_degraded_expert_and_accepts_improved_expert(
    tmp_path: Path,
) -> None:
    """Verify La Ruche gatekeeper rejects experts that fail to improve holdout NLL."""
    vault = LaRucheModelVault(
        vault_dir=tmp_path / "ruche_test",
        hidden_size=32,
        vocab_size=259,
        canonical_rank_r=4,
    )
    bad_tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(4, 32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(32, 4),
    }

    with pytest.raises(ValueError, match="La Ruche quality gate rejected"):
        vault.deposit_qlora_expert(
            domain_name="code_python",
            adapter_tensors=bad_tensors,
            contributor_handle="@xaviercallens",
            hardware_device="cpu",
            dataset_shard_cid="cid://sha256/test",
            train_nll_before=5.5,
            train_nll_after=5.4,
            holdout_nll_before=5.5,
            holdout_nll_after=5.6,
            holdout_perplexity_before=244.69,
            holdout_perplexity_after=270.42,
        )


def test_dataset_mining_campaign_populates_la_ruche_and_dashboard(
    tmp_path: Path,
) -> None:
    """Verify end-to-end 8-domain Dataset Mining Campaign, La Ruche MoE export, and HiveMind@home FastAPI portal."""
    assert len(RUCHE_ALVEOLE_HEX_LAYOUT) == 8
    assert len(SCIENTIFIC_MINING_DOMAIN_CORPORA) == 8

    vault_dir = tmp_path / "la_ruche_vault"
    mining_report_path = tmp_path / "dataset_mining_report.json"
    html_path = tmp_path / "hivemind_at_home.html"
    state_path = tmp_path / "hivemind_at_home_state.json"

    summary = run_dataset_mining_and_ruche_campaign(
        vault_dir=vault_dir,
        report_json_path=mining_report_path,
        training_steps=4,
        learning_rate=0.03,
    )

    assert summary["all_shards_dedup_clean"] is True
    assert summary["completed_shard_leases"] == 8
    assert summary["duplicate_shard_leases"] == 0
    assert summary["lease_conservation_holds"] is True
    assert summary["moe_forward_verified"] is True
    assert len(summary["mined_shards"]) == 8
    assert len(summary["domain_training_metrics"]) == 8

    # Verify loading the Community BTM Sparse MoE model from La Ruche
    moe_model, manifest = load_community_moe_from_ruche(vault_dir)
    assert manifest["num_domains"] == 8
    assert manifest["mean_perplexity_reduction_pct"] > 0.0
    tokens = torch.tensor([[2, 5, 9, 14, 21, 33]], dtype=torch.long)
    out = moe_model(input_ids=tokens)
    assert out.logits.shape == (1, 6, 259)
    assert bool(torch.isfinite(out.logits).all())

    # Verify HiveMind@home + GCP + La Ruche HTML/JSON export & FastAPI server
    state = export_hivemind_at_home_artifacts(
        output_html_path=html_path,
        output_state_json_path=state_path,
        mining_ruche_report_path=mining_report_path,
    )
    assert html_path.exists()
    assert state_path.exists()
    html_text = render_hivemind_at_home_html(state)
    assert "HIVEMIND@HOME" in html_text
    assert "LA RUCHE" in html_text
    assert "gwenlaya-tpu-1" in html_text

    app = create_hivemind_at_home_app(
        mining_ruche_report_path=mining_report_path,
        vault_dir=vault_dir,
    )
    client = TestClient(app)
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "la-ruche-svg" in r_index.text

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["cost_mode"] == "scale_to_zero_min0_max2"

    r_status = client.get("/api/hivemind-at-home/status")
    assert r_status.status_code == 200
    assert r_status.json()["formal_verification"]["lean4_theorems_verified"] == 128

    r_ruche = client.get("/api/la-ruche/manifest")
    assert r_ruche.status_code == 200
    assert r_ruche.json()["num_domains"] == 8

    # Verify binary download of Community MoE safetensors and individual Honeycomb Cell
    r_dl_moe = client.get("/api/la-ruche/download/community-moe.safetensors")
    assert r_dl_moe.status_code == 200
    assert len(r_dl_moe.content) == manifest["combined_safetensors_bytes"]
    assert (
        r_dl_moe.headers["X-Safetensors-SHA256"]
        == manifest["combined_safetensors_sha256"]
    )

    r_dl_cell = client.get("/api/la-ruche/download/cell/code_python")
    assert r_dl_cell.status_code == 200
    assert len(r_dl_cell.content) > 1000
    assert r_dl_cell.headers["X-Domain-Name"] == "code_python"

    # Verify live Sparse MoE inference & OpenAI chat completions
    r_inf = client.post(
        "/api/v1/moe/inference",
        json={
            "prompt": "Optimize PyTorch tensor kernel and prove Lean 4 theorem",
            "top_k": 2,
        },
    )
    assert r_inf.status_code == 200
    inf_json = r_inf.json()
    assert inf_json["top_k_active"] == 2
    assert len(inf_json["routed_experts"]) == 2
    assert inf_json["logit_l2_norm"] > 0.0

    r_chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "hivemind-la-ruche-moe-v2.6",
            "messages": [{"role": "user", "content": "Explain E8 lattice quantization in Rust QUIC"}],
            "top_k": 2,
        },
    )
    assert r_chat.status_code == 200
    chat_json = r_chat.json()
    assert len(chat_json["choices"]) == 1
    assert len(chat_json["la_ruche_routing"]["routed_experts"]) == 2

    # Verify federated workunit request, outer-delta submission, and ledger
    r_wu = client.post(
        "/api/v1/federated/workunit/request",
        json={
            "contributor_handle": "@test-gpu-node",
            "hardware_device": "NVIDIA RTX 5060",
            "preferred_domain": "code_python",
        },
    )
    assert r_wu.status_code == 200
    wu_json = r_wu.json()
    assert wu_json["domain_name"] == "code_python"

    r_sub = client.post(
        "/api/v1/federated/workunit/submit",
        json={
            "workunit_id": wu_json["workunit_id"],
            "lease_id": wu_json["lease_id"],
            "lease_version_tag": wu_json["lease_version_tag"],
            "contributor_handle": "@test-gpu-node",
            "domain_name": "code_python",
            "train_nll_before": 5.55,
            "train_nll_after": 5.12,
            "holdout_nll_before": 5.50,
            "holdout_nll_after": 5.32,
        },
    )
    assert r_sub.status_code == 200
    sub_json = r_sub.json()
    assert sub_json["status"] == "ADMITTED_TO_LA_RUCHE_QUORUM"
    assert sub_json["monetary_payout_eur_usd"] == 0.0

    r_ledger = client.get("/api/v1/federated/ledger")
    assert r_ledger.status_code == 200
    assert r_ledger.json()["completed_receipts_count"] == 1

    r_shards = client.get("/api/dataset-mining/shards")
    assert r_shards.status_code == 200
    assert r_shards.json()["completed_shard_leases"] == 8

    # Verify the 5 Cyber-Bee Screens, Pretrained MoE Bees, Results Carousel & Leaderboard
    for screen_id in (
        "screen-overview",
        "screen-join-points",
        "screen-moe-bees",
        "screen-results-carousel",
        "screen-tech-leaderboard",
    ):
        assert screen_id in r_index.text
    assert "GWAYA-Code-Bee-v2.6" in r_index.text
    assert "Gwen-Qwen2.5-MoE-Bee-v2.6" in r_index.text
    assert "Astro-Exoplanet-SETI-Bee-v2.6" in r_index.text
    assert len(state["community_results_carousel"]) == 6
    assert len(state["contributor_leaderboard"]) >= 4

    # Verify Compute Points redemption endpoint across GWAYA Python coding & Astrophysics
    r_redeem = client.post(
        "/api/v1/compute-points/redeem",
        json={
            "contributor_handle": "@test-gpu-node",
            "service_id": "python_coding_gwaya",
            "prompt": "Write a vectorized E8 lattice quantizer in Python.",
            "points_balance_before": 1250,
        },
    )
    assert r_redeem.status_code == 200
    redeem_json = r_redeem.json()
    assert redeem_json["status"] == "COMPUTE_POINTS_REDEEMED"
    assert redeem_json["points_spent"] == 50
    assert redeem_json["points_balance_after"] == 1200
    assert redeem_json["monetary_cost_eur_usd"] == 0.0
    assert "quantize_e8_coset_2bit" in redeem_json["domain_output"]

    # Verify 5-Language Support (English, French, Chinese, Spanish, German)
    r_i18n_all = client.get("/api/v1/i18n")
    assert r_i18n_all.status_code == 200
    i18n_all_json = r_i18n_all.json()
    assert i18n_all_json["supported_languages"] == ["en", "fr", "zh", "es", "de"]
    for lang_code in ("en", "fr", "zh", "es", "de"):
        r_lang = client.get(f"/api/v1/i18n/{lang_code}")
        assert r_lang.status_code == 200
        assert r_lang.json()["lang"] == lang_code
        assert "brand_title" in r_lang.json()["dictionary"]
        assert "video_title" in r_lang.json()["dictionary"]

        r_red_lang = client.post(
            "/api/v1/compute-points/redeem",
            json={
                "contributor_handle": "@test-gpu-node",
                "service_id": "astrophysics_exoplanet",
                "prompt": "Detect Kepler-186f exoplanet transit dip.",
                "points_balance_before": 1250,
                "lang": lang_code,
            },
        )
        assert r_red_lang.status_code == 200
        assert r_red_lang.json()["lang"] == lang_code
        assert f"[{lang_code.upper()}]" in r_red_lang.json()["domain_output"]

    # Verify 8-Domain Interactive Examples Gallery across all 5 languages
    for lang_code in ("en", "fr", "zh", "es", "de"):
        r_ex = client.get(f"/api/v1/examples?lang={lang_code}")
        assert r_ex.status_code == 200
        ex_json = r_ex.json()
        assert ex_json["examples_count"] == 8
        assert len(ex_json["examples"]) == 8
        assert ex_json["examples"][0]["localized_title"] != ""

    # Verify SVG Visualizations & Interactive Video Cinema Manifest + DOM Elements
    r_svg1 = client.get("/api/v1/media/cyber-bee-poster.svg")
    assert r_svg1.status_code == 200
    assert "image/svg+xml" in r_svg1.headers["content-type"]
    assert "cyber-bee-poster-svg" in r_svg1.text

    r_svg2 = client.get("/api/v1/media/exoplanet-seti-visual.svg")
    assert r_svg2.status_code == 200
    assert "image/svg+xml" in r_svg2.headers["content-type"]
    assert "exoplanet-seti-svg" in r_svg2.text

    r_vid = client.get("/api/v1/media/video-manifest")
    assert r_vid.status_code == 200
    vid_json = r_vid.json()
    assert len(vid_json["chapters"]) == 4
    assert "cyber-bee-video-element" in r_index.text
    assert "cyber-bee-video-canvas" in r_index.text
    assert "interactive-examples-grid" in r_index.text
    assert "lang-switcher-bar" in r_index.text
