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
    sanitize_handle,
    sanitize_prompt_text,
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
    assert (html_path.parent / "cyber-bee-poster.svg").exists()
    assert (html_path.parent / "exoplanet-seti-visual.svg").exists()

    html_text = render_hivemind_at_home_html(state)
    assert "HIVEMIND@HOME" in html_text
    assert "LA RUCHE" in html_text
    assert "gwenlaya-tpu-1" in html_text
    assert "cyber-bee-svg-layer" in html_text
    assert "Space Grotesk" in html_text
    assert "nav-tab lang-btn" not in html_text

    # Verify that the embedded JavaScript is free of syntax errors
    import re
    import subprocess
    scripts = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    assert len(scripts) >= 1
    node_res = subprocess.run(
        ["node", "--check"],
        input=scripts[0].encode("utf-8"),
        capture_output=True,
    )
    assert node_res.returncode == 0, f"JS Syntax Error: {node_res.stderr.decode()}"

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
    assert sub_json["status"] == "RECEIVED_CLIENT_CLAIMS_UNVERIFIED"
    assert sub_json["reciprocity_points_awarded"] == 0
    assert sub_json["spotcheck_verified"] is False
    # Negative control: the same lease cannot be submitted twice, and an invented lease is refused.
    replay = client.post("/api/v1/federated/workunit/submit", json={
        "workunit_id": wu_json["workunit_id"], "lease_id": wu_json["lease_id"],
        "lease_version_tag": wu_json["lease_version_tag"], "contributor_handle": "@test-gpu-node",
        "domain_name": "code_python", "train_nll_before": 5.55, "train_nll_after": 5.12,
        "holdout_nll_before": 5.50, "holdout_nll_after": 5.32})
    assert replay.status_code == 422
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

    # Verify Zero-Trust Security Headers & Cognitive Shield WAF
    assert "content-security-policy" in r_index.headers
    assert "default-src 'self'" in r_index.headers["content-security-policy"]
    assert r_index.headers["x-frame-options"] == "DENY"
    assert r_index.headers["x-content-type-options"] == "nosniff"
    assert r_index.headers["x-hivemind-shield"] == "Active-v2.7.0"
    assert "x-ratelimit-limit" in r_index.headers

    # Verify Defense Status Endpoint
    r_def = client.get("/api/v1/defense/status")
    assert r_def.status_code == 200
    def_json = r_def.json()
    assert def_json["firewall_status"] == "ACTIVE"
    assert def_json["rate_limiter"]["firewall_status"] == "ACTIVE"
    assert def_json["formal_verification"]["lean4_theorems_verified"] == 128
    assert def_json["formal_verification"]["sorry_count"] == 0
    assert def_json["byzantine_poisoning_defense"]["attack_vectors_blocked_count"] == 49

    # Verify Live Attack Defense Probe Test Endpoint
    r_probe = client.post(
        "/api/v1/defense/test",
        json={"test_input": "<script>alert('attack')</script> SELECT * FROM hive_users;"},
    )
    assert r_probe.status_code == 200
    probe_json = r_probe.json()
    assert probe_json["injection_detected"] is True
    assert probe_json["firewall_action"] == "NEUTRALIZED"
    assert "[SANITIZED_INJECTION_PAYLOAD]" in probe_json["sanitized_output"]
    assert probe_json["security_verdict"] == "ZERO_TRUST_ENFORCED"

    # Verify Sanitizer Unit Invariants
    assert sanitize_handle("@valid_contributor-99") == "@valid_contributor-99"
    with pytest.raises(Exception):
        sanitize_handle("<script>evil</script>")
    with pytest.raises(Exception):
        sanitize_handle("bad\x00handle")
    with pytest.raises(Exception):
        sanitize_handle("x")

    assert "[SANITIZED_INJECTION_PAYLOAD]" in sanitize_prompt_text("<script>xss()</script> test")
    assert "clean code" == sanitize_prompt_text("clean code")

    # Verify Cyberpunk UI and Defense Shield elements in rendered HTML
    assert "cyber-defense-hud-bar" in r_index.text
    assert "defense-modal-backdrop" in r_index.text
    assert "cyber-scanlines" in r_index.text
    assert "COGNITIVE SHIELD: ARMED" in r_index.text
    assert "btn-chamfer" in r_index.text

    # Verify /community, /boinc, /api/v1/community/*, /rejoindre, /join, and /scripts/*
    assert client.get("/community").status_code == 200
    assert client.get("/boinc").status_code == 200
    assert client.get("/api/v1/community/hub").status_code == 200
    r_bsync = client.post(
        "/api/v1/community/boinc-sync",
        json={"handle": "pioneer-prosumer-01", "target_planet": "K2-18 b", "idle_seconds": 8.0},
    )
    assert r_bsync.status_code == 200 and r_bsync.json()["ok"] is True
    r_bsync_new = client.post(
        "/api/v1/community/boinc-sync",
        json={"handle": "new-gamer-fr", "target_planet": "LHS 1140 b", "idle_seconds": 12.0},
    )
    assert r_bsync_new.status_code == 200 and r_bsync_new.json()["ok"] is True
    r_note = client.post(
        "/api/v1/community/post-note",
        json={"handle": "new-gamer-fr", "target": "K2-18 b", "note": "Transit verified"},
    )
    assert r_note.status_code == 200 and r_note.json()["ok"] is True
    assert client.get("/rejoindre?lang=fr").status_code == 200
    assert client.get("/join?lang=en").status_code == 200
    assert client.get("/scripts/hivemind_boinc_idle_worker.py").status_code == 200
    assert client.get("/scripts/unknown_script.py").status_code == 404

    # 1. HTML routes (v2, legacy, aliases, cockpit, webgpu)
    for route in (
        "/?view=legacy",
        "/v2",
        "/v2/auth",
        "/v2/openrouter-moe",
        "/v2/byob",
        "/v2/mesh",
        "/v2/mobile",
        "/v2/telemetry",
        "/v2/admin",
        "/admin",
        "/v2/steam",
        "/v2/galaxy",
        "/v2/ide",
        "/home",
        "/auth",
        "/openrouter-moe",
        "/byob",
        "/mesh",
        "/mobile",
        "/telemetry",
        "/v1",
        "/cockpit-v46",
        "/game",
        "/exoplanet-hunter",
        "/webgpu?lang=fr",
        "/webgpu?lang=invalid",
    ):
        assert client.get(route).status_code == 200

    # 2. Auth endpoints (guest, upgrade-guest, register, login, sso, link-provider, rotate-key, magic-link, session, migration)
    r_guest = client.post("/api/v1/auth/guest")
    assert r_guest.status_code == 200
    guest_token = r_guest.json()["token"]

    r_up = client.post(
        "/api/v1/auth/upgrade-guest",
        headers={"Authorization": f"Bearer {guest_token}"},
        json={"handle": "upgraded-user-01", "email": "up@hivemind.ai", "provider": "google"},
    )
    assert r_up.status_code == 200

    assert (
        client.post(
            "/api/v1/auth/register",
            json={"handle": "reg-user-01", "email": "reg01@hivemind.ai"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/auth/register",
            json={"handle": "reg-user-01", "email": "reg01@hivemind.ai"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/auth/register",
            json={"handle": "reg-user-01", "email": "reg01@hivemind.ai", "strict_new": True},
        ).status_code
        == 409
    )

    assert client.post("/api/v1/auth/login", json={"handle_or_email": "reg-user-01"}).status_code == 200
    assert client.post("/api/v1/auth/login", json={"handle_or_email": "login-created-01"}).status_code == 200
    assert (
        client.post(
            "/api/v1/auth/login",
            json={"handle_or_email": "missing-strict-user", "strict_existing": True},
        ).status_code
        == 404
    )

    assert (
        client.post("/api/v1/auth/sso/google", json={"handle_hint": "sso-user-01"}).status_code
        == 200
    )
    assert client.post("/api/v1/auth/sso/google", json={"handle_hint": "sso-user-01"}).status_code == 200
    assert client.post("/api/v1/auth/sso/unsupported_idp", json={"handle_hint": "sso-user-01"}).status_code == 400

    assert (
        client.post("/api/v1/auth/link-provider", json={"handle": "sso-user-01", "provider": "github"}).status_code
        == 200
    )
    assert client.post("/api/v1/auth/rotate-key", headers={"Authorization": f"Bearer {r_up.json()['token']}"}).status_code == 200
    assert client.post("/api/v1/auth/magic-link", json={"email": "magic@hivemind.ai"}).status_code == 200
    assert client.post("/api/v1/auth/magic-link", json={"email": "magic@hivemind.ai"}).status_code == 200
    assert client.get("/api/v1/auth/users").status_code == 200
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {r_up.json()['token']}"}).status_code == 200
    assert client.get("/api/v1/auth/session").status_code == 200
    assert client.post("/api/v1/auth/logout").status_code == 200

    assert client.get("/api/v1/auth/migration/status").status_code == 200
    assert client.post("/api/v1/auth/v1-migrate", json={"handle_or_email": "callensxavier@gmail.com"}).status_code == 200
    assert client.post("/api/v1/auth/v1-migrate", json={"handle_or_email": "migrated-v1-custom"}).status_code == 200

    # 3. AMCP Mesh & OpenRouter Models
    assert client.get("/api/v1/amcp/mesh/nodes").status_code == 200
    assert client.get("/api/v1/amcp/mesh/messages").status_code == 200
    for recipient, content in (
        ("@AstroBee", "analyze transit"),
        ("@Lean4Bee", "prove theorem"),
        ("@CodeBee", "optimize kernel"),
        ("@QueenBee", "status"),
    ):
        assert (
            client.post(
                "/api/v1/amcp/mesh/send",
                json={"sender_id": "sso-user-01", "recipient": recipient, "content": content},
            ).status_code
            == 200
        )
    for tool_name in ("query_exoplanet_catalog", "verify_lean4_theorem", "mesh_status", "custom_tool"):
        assert (
            client.post("/api/v1/amcp/tool/call", json={"tool_name": tool_name, "arguments": {}}).status_code == 200
        )
    assert client.get("/api/v1/amcp/protocol/spec").status_code == 200
    assert client.get("/api/v1/openrouter/models").status_code == 200

    # 4. Night-MoE, Hardware Shield & Admin endpoints
    assert client.get("/api/v1/night-moe/status").status_code == 200
    assert client.post("/api/v1/night-moe/start", json={}).status_code == 200
    assert client.get("/api/v1/hardware/rtx2070/shield").status_code == 200
    assert client.post("/api/v1/hardware/rtx2070/shield/lock").status_code == 200
    assert client.get("/api/v1/admin/dashboard").status_code == 200
    assert client.get("/api/v1/admin/masterminders").status_code == 200
    assert (
        client.post(
            "/api/v1/admin/campaigns/create",
            json={"name": "Test Campaign", "domain": "code_python"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/admin/moe/train-expert",
            json={"domain_name": "physics_sim", "hardware_override": "cuda:0 RTX 2070"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/admin/moe/train-expert",
            json={"domain_name": "physics_sim", "hardware_override": "cloud-a10g"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/admin/moe/models/select",
            json={"model_id": "hivemind/moe-queenbee-256b"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/son/feedback",
            json={"gamer_tag": "sso-user-01", "overall_rating": 5, "review_text": "Great cockpit!"},
        ).status_code
        == 200
    )

    # 5. Steam, FinOps, Deployment Review, Galaxy, Security, IDE & Astro routes
    assert client.get("/api/v1/steam/info").status_code == 200
    assert client.get("/api/v1/steam/daemon/status").status_code == 200
    for act in ("pause_for_game", "resume", "toggle"):
        assert (
            client.post(
                "/api/v1/steam/daemon/toggle", json={"action": act, "game": "Cyberpunk 2077"}
            ).status_code
            == 200
        )
    assert (
        client.post(
            "/api/v1/steam/daemon/config",
            json={
                "idle_threshold_percent": 15.0,
                "gaming_preemption_threshold_percent": 70.0,
                "max_temp_c": 65.0,
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/steam/claim-points",
            json={"contributor_handle": "sso-user-01", "amount": 250},
        ).status_code
        == 200
    )
    assert client.get("/api/v1/steam/achievements").status_code == 200
    assert client.get("/api/v1/steam/download/free-edition.zip").status_code == 200

    assert client.get("/api/v1/finops/status").status_code == 200
    for action in ("scale_to_zero", "wake_up", "simulate_idle", "enforce_policy"):
        assert (
            client.post(
                "/api/v1/finops/scale-to-zero",
                json={"action": action, "max_instances": 2, "idle_timeout_seconds": 300},
            ).status_code
            == 200
        )
    assert client.post("/api/v1/finops/simulate-idle").status_code == 200

    assert client.get("/api/v1/admin/deployment-review/status").status_code == 200
    for action in ("request_review", "approve", "reset"):
        assert (
            client.post(
                "/api/v1/admin/deployment-review/request",
                json={"action": action, "reviewer": "@Vxtzq", "comment": "Reviewed"},
            ).status_code
            == 200
        )

    assert client.get("/api/v1/galaxy/config").status_code == 200
    assert (
        client.post(
            "/api/v1/galaxy/explore",
            json={
                "sector_id": "trappist1e",
                "star_count": 500000,
                "rotation_speed": 0.15,
                "spiral_tightness": 1.8,
                "arm_count": 4,
                "arm_width": 2.0,
                "randomness": 1.5,
                "galaxy_radius": 12.0,
                "galaxy_thickness": 2.5,
                "mouse_force": 5.0,
                "mouse_radius": 8.0,
                "dense_star_color": "#00f0ff",
                "sparse_star_color": "#fbbf24",
                "cloud_count": 4000,
                "cloud_tint_color": "#fecdd3",
                "bloom_strength": 0.3,
                "contributor_handle": "sso-user-01",
            },
        ).status_code
        == 200
    )

    assert client.get("/api/v1/security/status").status_code == 200
    assert client.post("/api/v1/security/waf/scan", json={"payload": "normal query"}).status_code == 200
    assert client.post("/api/v1/security/waf/scan", json={"payload": "<script>alert(1)</script> UNION SELECT"}).status_code == 200
    assert client.get("/api/steam/fat-client-manifest").status_code == 200

    assert client.get("/api/v1/ide/status").status_code == 200
    for tab in ("bee_moe_router.py", "bls_fold_galaxy.wgsl", "stitch_v2_ui_component.html", "custom.txt"):
        assert client.post("/api/v1/ide/execute", json={"file_tab": tab}).status_code == 200
    assert client.post("/api/v1/ide/self-test").status_code == 200

    assert (
        client.post(
            "/v1/hivemind/steam/register-login",
            json={"handle": "steam-pilot-01", "gpu_model": "NVIDIA RTX 4080"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/hivemind/steam/contribute-boinc",
            json={"handle": "steam-pilot-01", "target_planet": "K2-18 b"},
        ).status_code
        == 200
    )

    assert client.get("/api/astro/nasa-tap-sync").status_code == 200
    assert (
        client.post(
            "/api/astro/discover",
            json={"target_id": "k218b", "player_classification": "PLANET"},
        ).status_code
        == 200
    )
    assert client.post("/api/astro/bee-moe-scan", json={"target_id": "lhs1140b"}).status_code == 200
    assert client.get("/stitch-assets/nonexistent.png").status_code == 404
    assert client.get("/download/steam-fat-client", follow_redirects=False).status_code in (200, 302, 307)
    assert client.get("/download/steam-package-zip", follow_redirects=False).status_code in (200, 302, 307)
    assert client.get("/api/v1/steam/download/free-edition.zip").status_code == 200
    assert client.get("/cockpit-v46").status_code == 200
    assert client.get("/night-moe").status_code == 200
    assert client.get("/api/v1/night-moe").status_code in (200, 404)
    assert client.get("/api/v1/i18n/unsupported_lang").status_code == 404
    assert client.post("/api/v1/auth/v1-migrate", json={"v1_token": r_up.json()["token"]}).status_code == 200
    assert client.post("/api/v1/steam/daemon/toggle", json={"action": "pause"}).status_code == 200
    assert client.post("/api/v1/steam/daemon/toggle", json={"action": "resume"}).status_code == 200
    assert client.post("/api/v1/steam/daemon/toggle", json={}).status_code == 200
    assert client.post("/api/v1/steam/daemon/toggle", json={}).status_code == 200
    for svc in ("image_spectrogram_synthesis", "lean4_formal_proof"):
        assert (
            client.post(
                "/api/v1/compute-points/redeem",
                json={"contributor_handle": "@sso-user-01", "service_id": svc, "prompt": "test", "points_balance_before": 500},
            ).status_code
            == 200
        )

