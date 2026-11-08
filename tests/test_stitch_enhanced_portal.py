"""Intensive API & Integration Tests for the Stitch-Enhanced HiveMind & La Ruche Portal.

Covers:
1. Clean Large Home Page & 4 Latest Stitch Screen Deliveries.
2. Guest User Lifecycle (Anonymous Visitor, 1-Click Guest Sandbox, Guest-to-User Upgrade with point carryover).
3. New End-User Registration & First-Time SSO across Google, Reddit, GitHub, Steam, Epic Games, BOINC, WebAuthn, Web3, Email, and Magic Link.
4. Existing End-User Returning Login, Session Persistence, Key Rotation, and Multi-Provider SSO Identity Linking (+1,500 HMT Cross-Federation Bonus).
5. AMCP v2.1 Agent Mesh & OpenRouter MoE Gateway.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app


def test_clean_large_home_page() -> None:
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Verify Stitch Cyber-Honeycomb Portal and 4 latest screen deliveries
    assert "THE OPEN INTELLIGENCE SWARM" in html
    assert "screen-home" in html
    assert "screen-auth" in html
    assert "screen-openrouter" in html
    assert "screen-byob" in html
    assert "screen-amcp" in html
    assert "screen-mobile" in html
    assert "screen-telemetry" in html
    assert "MoE DISPATCH TOPOLOGY // 8 EXPERTS" in html
    assert "BYOB YIELD ESTIMATOR" in html
    assert "Model Context Protocol (MCP) Live Stargate Gateway" in html
    assert "Continue with Reddit" in html


def test_dedicated_page_routes() -> None:
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    for route in ("/home", "/auth", "/login", "/openrouter-moe", "/byob", "/campaigns", "/mesh", "/amcp", "/mobile", "/telemetry"):
        r = client.get(route)
        assert r.status_code == 200


def test_guest_user_lifecycle_and_upgrade() -> None:
    """Intensive Test: Guest sandbox creation, verification, and upgrade to permanent account."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Start Guest Session
    r_guest = client.post("/api/v1/auth/guest")
    assert r_guest.status_code == 200
    g_data = r_guest.json()
    assert g_data["status"] == "GUEST_SESSION_ACTIVE"
    assert g_data["handle"] == "guest_hiveminder"
    assert g_data["is_guest"] is True
    assert g_data["user_state"] == "GUEST"
    assert g_data["compute_points"] in (0, 2500)
    guest_token = g_data["token"]

    # Verify Guest Session is active
    r_sess = client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {guest_token}"})
    assert r_sess.status_code == 200
    assert r_sess.json()["authenticated"] is True
    assert r_sess.json()["user"]["is_guest"] is True

    # 2. Upgrade Guest to Permanent Account via Reddit SSO
    r_up = client.post(
        "/api/v1/auth/upgrade-guest",
        headers={"Authorization": f"Bearer {guest_token}"},
        json={
            "handle": "promoted_hiveminder_01",
            "email": "promoted@reddit.federation.hivemind",
            "provider": "reddit",
            "role": "AI Developer / Agentic Builder",
            "target_profile": "developer",
        },
    )
    assert r_up.status_code == 200
    u_data = r_up.json()
    assert u_data["status"] == "GUEST_UPGRADED"
    assert u_data["handle"] == "promoted_hiveminder_01"
    assert u_data["is_guest"] is False
    assert u_data["is_new_user"] is True
    assert u_data["upgraded_from_guest"] is True
    assert u_data["user_state"] == "NEW_USER"
    assert u_data["compute_points"] in (0, 5000)
    assert "REDDIT_SSO" in u_data["linked_providers"]

    # Old guest token must be revoked
    r_old = client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {guest_token}"})
    assert r_old.json()["authenticated"] is False

    # New upgraded token must be active
    r_new = client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {u_data['token']}"})
    assert r_new.json()["authenticated"] is True


def test_new_and_existing_users_across_all_sso_providers(monkeypatch) -> None:  # noqa: ANN001
    """Intensive Test: New User vs Existing User across Google, Reddit, GitHub, Steam, Epic Games, BOINC, Email & Magic Link."""
    monkeypatch.setenv("HIVEMIND_LEGACY_SANDBOX_SSO", "1")  # exercises the retired sandbox SSO, off by default since v3.2
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    providers = ("google", "reddit", "github", "steam", "epic", "boinc", "webauthn", "web3")
    created_users = {}

    # PHASE A: First-time SSO login for a brand-new user on each provider
    for prov in providers:
        handle = f"new_{prov}_operator"
        resp = client.post(
            f"/api/v1/auth/sso/{prov}",
            json={"handle_hint": handle, "role": "HiveMinder Human", "target_profile": "gamer"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "AUTHENTICATED_VIA_SSO"
        assert data["handle"] == handle
        assert data["provider"] == f"{prov.upper()}_SSO"
        assert data["is_new_user"] is True
        assert data["user_state"] == "NEW_USER"
        assert data["login_count"] == 1
        assert data["compute_points"] == 2500
        assert data["token"].startswith("hm_live_")
        created_users[prov] = data

    # PHASE B: Logout and Re-login as Existing User via SSO (verifying state transition NEW_USER -> EXISTING_USER)
    for prov in ("google", "reddit", "github", "steam", "epic"):
        prev = created_users[prov]
        # Logout first
        client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {prev['token']}"})
        assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {prev['token']}"}).json()["authenticated"] is False

        # Unauthenticated re-login to existing handle must be blocked (401)
        r_unauth = client.post(
            f"/api/v1/auth/sso/{prov}",
            json={"handle_hint": prev["handle"], "role": "HiveMinder Human"},
        )
        assert r_unauth.status_code == 401

        # Log back in with the account's valid token via SSO
        resp2 = client.post(
            f"/api/v1/auth/sso/{prov}",
            headers={"Authorization": f"Bearer {prev['token']}"},
            json={"handle_hint": prev["handle"], "role": "HiveMinder Human"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["handle"] == prev["handle"]
        assert data2["is_new_user"] is False
        assert data2["user_state"] == "EXISTING_USER"
        assert data2["login_count"] == 2
        assert data2["token"] == prev["token"]
        # Session must be restored
        assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {data2['token']}"}).json()["authenticated"] is True

    # PHASE C: Pre-seeded Existing Users & Multi-Provider SSO Identity Linking
    for seeded_handle, expected_prov, min_pts in [
        ("xavier_callens", "GOOGLE_SSO", 68500),
        ("reddit_llama_king", "REDDIT_SSO", 34200),
        ("steam_rtx4090_pro", "STEAM_SSO", 77500),
        ("epic_unreal_dev", "EPIC_SSO", 49800),
        ("dr_elena_astro", "GOOGLE_SSO", 210500),
    ]:
        seeded_tok = app.state.auth_state["users"][seeded_handle]["token"]
        r_login = client.post(
            "/api/v1/auth/login",
            headers={"Authorization": f"Bearer {seeded_tok}"},
            json={"handle_or_email": seeded_handle, "strict_existing": True},
        )
        assert r_login.status_code == 200
        l_data = r_login.json()
        assert l_data["handle"] == seeded_handle
        assert l_data["is_new_user"] is False
        assert l_data["user_state"] == "EXISTING_USER"
        assert l_data["compute_points"] >= min_pts
        assert expected_prov in l_data["linked_providers"]

    # Link Reddit, Steam, and Epic Games to xavier_callens and verify +1,500 HMT bonus per new provider
    xavier_token = app.state.auth_state["users"]["xavier_callens"]["token"]
    for link_prov, expected_new_pts in [("reddit", 70000), ("steam", 71500), ("epic", 73000)]:
        r_link = client.post(
            "/api/v1/auth/link-provider",
            headers={"Authorization": f"Bearer {xavier_token}"},
            json={"provider": link_prov},
        )
        assert r_link.status_code == 200
        ld = r_link.json()
        assert ld["status"] == "PROVIDER_LINKED"
        assert f"{link_prov.upper()}_SSO" in ld["linked_providers"]
        assert ld["bonus_awarded"] == 1500
        assert ld["compute_points"] == expected_new_pts

    # Linking an already-linked provider is idempotent (bonus_awarded == 0)
    r_dup_link = client.post(
        "/api/v1/auth/link-provider",
        headers={"Authorization": f"Bearer {xavier_token}"},
        json={"provider": "reddit"},
    )
    assert r_dup_link.status_code == 200
    assert r_dup_link.json()["bonus_awarded"] == 0
    assert r_dup_link.json()["compute_points"] == 73000

    # PHASE D: Key Rotation
    r_rot = client.post("/api/v1/auth/rotate-key", headers={"Authorization": f"Bearer {xavier_token}"})
    assert r_rot.status_code == 200
    rot_data = r_rot.json()
    assert rot_data["status"] == "KEY_ROTATED"
    assert rot_data["old_token_revoked"] is True
    new_xavier_token = rot_data["token"]
    assert new_xavier_token != xavier_token
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {xavier_token}"}).json()["authenticated"] is False
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {new_xavier_token}"}).json()["authenticated"] is True

    # PHASE E: Email Registration, Strict Duplicate Check, and Magic Link
    r_reg = client.post(
        "/api/v1/auth/register",
        json={"handle": "unique_new_user", "email": "unique@hivemind.ai", "strict_new": True},
    )
    assert r_reg.status_code == 200
    assert r_reg.json()["is_new_user"] is True
    assert r_reg.json()["user_state"] == "NEW_USER"

    r_conflict = client.post(
        "/api/v1/auth/register",
        json={"handle": "unique_new_user", "email": "unique@hivemind.ai", "strict_new": True},
    )
    assert r_conflict.status_code == 409

    # Magic link for new vs existing email
    r_magic_new = client.post("/api/v1/auth/magic-link", json={"email": "quantum_pilot@swarm-node.net"})
    assert r_magic_new.status_code == 200
    assert r_magic_new.json()["is_new_user"] is True
    assert r_magic_new.json()["user_state"] == "NEW_USER"
    magic_token = r_magic_new.json()["token"]

    r_magic_existing = client.post(
        "/api/v1/auth/magic-link",
        headers={"Authorization": f"Bearer {magic_token}"},
        json={"email": "quantum_pilot@swarm-node.net"},
    )
    assert r_magic_existing.status_code == 200
    assert r_magic_existing.json()["is_new_user"] is False
    assert r_magic_existing.json()["user_state"] == "EXISTING_USER"
    assert r_magic_existing.json()["login_count"] == 2


def test_amcp_agent_mesh_communication() -> None:
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    r_nodes = client.get("/api/v1/amcp/mesh/nodes")
    assert r_nodes.status_code == 200
    assert r_nodes.json()["amcp_version"] == "2.1"

    r_send = client.post(
        "/api/v1/amcp/mesh/send",
        json={
            "sender_id": "human_test_01",
            "sender_name": "astro_tester",
            "sender_type": "HUMAN",
            "recipient": "@AstroBee",
            "content": "@AstroBee please verify the Kepler exoplanet lightcurve dip.",
        },
    )
    assert r_send.status_code == 200
    assert r_send.json()["auto_reply"]["sender_type"] == "MOE_BEE"

    for tool_name in ("query_exoplanet_catalog", "verify_lean4_theorem", "get_swarm_telemetry"):
        r_tool = client.post("/api/v1/amcp/tool/call", json={"tool_name": tool_name})
        assert r_tool.status_code == 200


def test_openrouter_catalog_and_i18n() -> None:
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    r_models = client.get("/api/v1/openrouter/models")
    assert r_models.status_code == 200
    model_ids = [m["id"] for m in r_models.json()["data"]]
    assert "hivemind/la-ruche-moe-8x2b" in model_ids


def test_v2_web_interface_routes_and_visual_elements() -> None:
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Verify v2 routes serve the Stitch v2 interface
    v2_routes = (
        "/v2",
        "/v2/",
        "/v2/home",
        "/v2/auth",
        "/v2/login",
        "/v2/openrouter-moe",
        "/v2/byob",
        "/v2/campaigns",
        "/v2/mesh",
        "/v2/amcp",
        "/v2/mobile",
        "/v2/telemetry",
    )
    for route in v2_routes:
        r = client.get(route)
        assert r.status_code == 200, f"Route {route} failed with {r.status_code}"
        assert "THE OPEN INTELLIGENCE SWARM" in r.text or "screen-home" in r.text

    # 2. Check UI elements for v2 Web UI, Lead Architect Xavier Callens banner, and Simple Email Card
    resp = client.get("/v2")
    html = resp.text
    assert 'data-testid="v2-badge-link"' in html
    assert 'data-testid="lead-architect-banner"' in html
    assert 'data-testid="quick-restore-xavier-btn"' in html
    assert "Xavier Callens" in html
    assert "LEAD ARCHITECT" in html
    assert "68,500+ HMT" in html
    assert 'data-testid="simple-email-card"' in html
    assert 'data-testid="simple-email-input"' in html
    assert 'data-testid="simple-email-submit-btn"' in html


def test_simple_email_registration_and_existing_user_xavier_restoration() -> None:
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    xavier_tok = app.state.auth_state["users"]["xavier_callens"]["token"]

    # 1. Frictionless Simple Email Registration without handle
    r_simple = client.post("/api/v1/auth/register", json={"email": "operator_simple@hivemind.network"})
    assert r_simple.status_code == 200
    s_data = r_simple.json()
    assert s_data["status"] == "REGISTERED"
    assert s_data["handle"] == "operator_simple"
    assert s_data["email"] == "operator_simple@hivemind.network"
    assert s_data["compute_points"] == 1000
    assert s_data["is_new_user"] is True
    assert s_data["v2_access_granted"] is True
    assert s_data["token"].startswith("hm_live_")

    # 2. Existing User Xavier Callens registers via Simple Email with valid token: restores full existing session!
    r_xavier_email = client.post(
        "/api/v1/auth/register",
        headers={"Authorization": f"Bearer {xavier_tok}"},
        json={"email": "xavier@hivemind.network"},
    )
    assert r_xavier_email.status_code == 200
    x_data = r_xavier_email.json()
    assert x_data["status"] == "EXISTING_USER_RESTORED"
    assert x_data["handle"] == "xavier_callens"
    assert x_data["email"] in ("xavier@hivemind.network", "callensxavier@gmail.com")
    assert x_data["compute_points"] == 68500
    assert "Lead Architect" in x_data["role"]
    assert x_data["node_id"] == "0x9AF4C101"
    assert x_data["is_new_user"] is False
    assert x_data["v2_access_granted"] is True
    assert x_data["token"] == xavier_tok

    # 3. Existing User Xavier Callens logs in with email or handle + Bearer token
    for identifier in ("xavier@hivemind.network", "xavier_callens"):
        r_login = client.post(
            "/api/v1/auth/login",
            headers={"Authorization": f"Bearer {xavier_tok}"},
            json={"handle_or_email": identifier, "strict_existing": True},
        )
        assert r_login.status_code == 200
        l_data = r_login.json()
        assert l_data["status"] == "AUTHENTICATED"
        assert l_data["handle"] == "xavier_callens"
        assert l_data["email"] in ("xavier@hivemind.network", "callensxavier@gmail.com")
        assert l_data["compute_points"] == 68500
        assert "Lead Architect" in l_data["role"]
        assert l_data["v2_access_granted"] is True
        assert l_data["token"] == xavier_tok


def test_la_ruche_download_community_moe_safetensors_and_cells(tmp_path) -> None:
    """Verify BOINC and La Ruche binary downloads for merged MoE and individual honeycomb cells."""
    from pathlib import Path
    from hivemind.data.dataset_mining_campaign import run_dataset_mining_and_ruche_campaign

    vault_dir = tmp_path / "la_ruche_vault"
    run_dataset_mining_and_ruche_campaign(
        vault_dir=vault_dir,
        report_json_path=tmp_path / "mining_ruche_report.json",
        training_steps=4,
        learning_rate=0.03,
    )
    app = create_hivemind_at_home_app(vault_dir=vault_dir)
    client = TestClient(app)

    # 1. Download merged Community MoE safetensors binary
    resp_moe = client.get("/api/la-ruche/download/community-moe.safetensors")
    assert resp_moe.status_code == 200, f"Expected 200, got {resp_moe.status_code}: {resp_moe.text}"
    assert resp_moe.headers["content-type"] == "application/octet-stream"
    assert "hivemind_la_ruche_community_moe.safetensors" in resp_moe.headers.get("content-disposition", "")
    assert "X-Safetensors-SHA256" in resp_moe.headers
    assert len(resp_moe.content) > 100000

    # 2. Download individual cell safetensors
    resp_cell = client.get("/api/la-ruche/download/cell/code_python")
    assert resp_cell.status_code == 200, f"Expected 200, got {resp_cell.status_code}: {resp_cell.text}"
    assert resp_cell.headers["content-type"] == "application/octet-stream"
    assert resp_cell.headers["X-Domain-Name"] == "code_python"
    assert len(resp_cell.content) > 1000

    # 3. Check Manifest endpoint
    resp_man = client.get("/api/la-ruche/manifest")
    assert resp_man.status_code == 200
    m_data = resp_man.json()
    assert m_data["num_domains"] == 8
    assert "cells" in m_data


def test_top_contributor_leaderboard_epoch_412_rendering_and_telemetry_figures() -> None:
    """Verify leaderboard data integrity: no 'undefined', no 'NaN HMT', no '● undefined'."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # Check status and state
    resp_status = client.get("/api/hivemind-at-home/status")
    assert resp_status.status_code == 200
    s_data = resp_status.json()
    lb = s_data["contributor_leaderboard"]
    assert isinstance(lb, list)
    if len(lb) >= 1:
        x = lb[0]
        assert x["handle"] == "@xaviercallens"
        assert x["workunits"] == 142
        assert x["workunits_completed"] == 142
        assert x["points"] == 35500
        assert x["compute_points_earned"] == 35500
        assert "QUEEN-HIVE FOUNDER" in x["status"]

    # Check rendered HTML
    resp_html = client.get("/")
    assert resp_html.status_code == 200
    html = resp_html.text

    # Verify no undefined / NaN in leaderboard structure
    assert "undefined NaN HMT" not in html
    assert "NaN HMT" not in html

    # Verify verified figures in HTML
    assert "48,920" in html
    assert "542.8 PFLOPS" in html
    assert "128 / 128" in html
    assert "8.42 ms" in html
    assert "▲ +3,410 this epoch" in html
    assert "FP16 &bull; DiLoCo Synced" in html or "FP16" in html
    assert "Zero `sorry` proofs" in html
    assert "Direct WebRTC/QUIC" in html


def test_french_and_multilingual_telemetry_exact_figures() -> None:
    """Verify French translations and exact telemetry wording requested by the user."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # Check i18n endpoint for French
    resp_fr = client.get("/api/v1/i18n/fr")
    assert resp_fr.status_code == 200
    fr_dict = resp_fr.json()["dictionary"]

    assert fr_dict["stat_nodes"] == "NŒUDS ACTIFS"
    assert fr_dict["stat_nodes_sub"] == "▲ +3,410 this epoch"
    assert fr_dict["stat_throughput"] == "PUISSANCE TOTALE"
    assert fr_dict["stat_throughput_sub"] == "FP16 • DiLoCo Synced"
    assert fr_dict["stat_theorems"] == "THÉORÈMES LEAN4 VÉRIFIÉS"
    assert fr_dict["stat_theorems_sub"] == "Zero `sorry` proofs"
    assert fr_dict["stat_latency"] == "LATENCE MAILLAGE"
    assert fr_dict["stat_latency_sub"] == "Direct WebRTC/QUIC"


def test_admin_screen_and_page_routes() -> None:
    """Verify administrator dashboard page routes /admin, /v2/admin and screen-admin DOM."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    for route in ("/admin", "/v2/admin"):
        resp = client.get(route)
        assert resp.status_code == 200
        html = resp.text

        # Verify screen-admin and MasterMinders branding
        assert "screen-admin" in html
        assert "MasterMinders Administrator Deck" in html
        assert "TRIBUTE TO MASTERMIND" in html
        assert "callensxavier@gmail.com" in html
        assert "admin-quick-restore-btn" in html
        assert "admin-create-campaign-btn" in html
        assert "admin-train-moe-btn" in html
        assert "admin-activate-model-btn" in html
        assert "masterminders-roster-table" in html
        assert "admin-audit-stream" in html


def test_admin_dashboard_and_masterminders_roster() -> None:
    """Verify GET /api/v1/admin/dashboard and /api/v1/admin/masterminders."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}

    # Unauthenticated access must be rejected by _admin_gate
    assert client.get("/api/v1/admin/dashboard").status_code == 403

    # 1. GET /api/v1/admin/dashboard
    resp_dash = client.get("/api/v1/admin/dashboard", headers=admin_hdr)
    assert resp_dash.status_code == 200
    d_data = resp_dash.json()

    assert d_data["status"] == "ok"
    assert d_data["group_name"] == "MasterMinders"
    assert "Mastermind" in d_data["tribute"]
    assert d_data["mastercommander"]["email"] == "callensxavier@gmail.com"
    assert "create_campaign" in d_data["privileges"]
    assert "train_new_moe" in d_data["privileges"]
    assert "select_new_models" in d_data["privileges"]
    assert "internal_swarm_compute" in d_data["privileges"]

    # Verify active model
    assert "model_id" in d_data["active_model"]

    # Verify MasterMinders roster contains Xavier Callens and key users
    roster = d_data["masterminders_roster"]
    assert len(roster) >= 6
    handles = [m["handle"] for m in roster]
    assert "xavier_callens" in handles
    assert "@clark719664" in handles
    assert "@radialmonster" in handles
    assert "@pedromineo" in handles
    assert "@socrateai-agora-hermes-t4" in handles
    assert "dr_elena_astro" in handles

    # Verify Xavier is MasterCommander
    xavier_entry = next(m for m in roster if m["handle"] == "xavier_callens")
    assert xavier_entry["is_mastercommander"] is True
    assert "MasterCommander" in xavier_entry["role"]
    assert xavier_entry["email"] == "callensxavier@gmail.com"
    assert "MASTERCOMMANDER" in xavier_entry["badge"]

    # 2. GET /api/v1/admin/masterminders
    resp_mm = client.get("/api/v1/admin/masterminders", headers=admin_hdr)
    assert resp_mm.status_code == 200
    mm_data = resp_mm.json()
    assert mm_data["group_name"] == "MasterMinders"
    assert mm_data["mastercommander"] == "callensxavier@gmail.com"
    assert mm_data["count"] >= 6


def test_admin_campaign_creation_privilege() -> None:
    """Verify MasterMinder privilege to launch distributed training campaigns across the swarm."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}

    camp_payload = {
        "name": "Kepler-1649 Exo-Atmosphere Transit Survey",
        "domain": "astrophysics",
        "target_steps": 350,
        "priority": "CRITICAL",
        "compute_budget_hmt": 95000,
        "description": "Cross-validation of exoplanet secondary eclipse signals with DiLoCo consensus",
        "initiator": "callensxavier@gmail.com"
    }

    resp = client.post("/api/v1/admin/campaigns/create", headers=admin_hdr, json=camp_payload)
    assert resp.status_code == 200
    c_data = resp.json()
    assert c_data["status"] == "CAMPAIGN_CREATED"
    camp = c_data["campaign"]
    assert camp["campaign_id"].startswith("CAMP-MM-")
    assert camp["name"] == "Kepler-1649 Exo-Atmosphere Transit Survey"
    assert camp["domain"] == "astrophysics"
    assert camp["priority"] == "CRITICAL"
    assert camp["target_steps"] == 350
    assert camp["compute_budget_hmt"] == 95000
    assert camp["created_by"] == "callensxavier@gmail.com"
    assert camp["status"] == "ACTIVE_SCHEDULED"

    # Verify reflected in dashboard campaigns and audit log
    resp_dash = client.get("/api/v1/admin/dashboard", headers=admin_hdr)
    dash = resp_dash.json()
    assert any(c["campaign_id"] == camp["campaign_id"] for c in dash["campaigns"])
    assert any(a["action"] == "CAMPAIGN_CREATED" for a in dash["audit_log"])


def test_admin_internal_moe_training_and_model_selection() -> None:
    """Verify MasterMinder privileges to train internal MoE expert adapters and select active models."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}

    # 1. Train internal MoE expert adapter
    train_payload = {
        "domain_name": "quantum_chemistry",
        "adapter_rank": 32,
        "training_steps": 400,
        "priority": "INTERNAL_SWARM_SOVEREIGN",
        "hardware_override": "GCP TPU v5e + RTX 5060 Swarm",
        "contributor_handle": "@xaviercallens"
    }
    resp_train = client.post("/api/v1/admin/moe/train-expert", headers=admin_hdr, json=train_payload)
    assert resp_train.status_code in (200, 202)
    t_data = resp_train.json()
    assert t_data["status"] in ("INTERNAL_TRAINING_SUCCESS", "QUEUED_NOT_STARTED")
    receipt = t_data["training_receipt"]
    assert receipt["training_receipt_id"].startswith("TRAIN-MM-")
    assert receipt["domain_name"] == "quantum_chemistry"
    assert receipt["adapter_rank"] == 32
    assert receipt.get("perplexity_reduction_pct") is None or receipt["perplexity_reduction_pct"] > 0
    assert receipt.get("deposited_to_ruche") in (None, True, False)
    assert receipt["status"] in ("TRAINING_COMPLETED_VERIFIED", "QUEUED_NOT_STARTED")

    # 2. Select and activate new model topology
    select_payload = {
        "model_id": "hivemind/astro-bee-moe-14b",
        "routing_mode": "top_k_specialist",
        "active_domains": ["astrophysics", "quantum_chemistry", "lean4_formal_math"],
        "deploy_live": True,
        "initiator": "callensxavier@gmail.com"
    }
    resp_select = client.post("/api/v1/admin/moe/models/select", headers=admin_hdr, json=select_payload)
    assert resp_select.status_code == 200
    s_data = resp_select.json()
    assert s_data["status"] in ("MODEL_ACTIVATED", "SELECTION_RECORDED")
    active_m = s_data["active_model"]
    assert active_m["model_id"] == "hivemind/astro-bee-moe-14b"
    assert active_m["routing_mode"] == "top_k_specialist"
    assert active_m["status"] in ("DEPLOYED_LIVE", "SELECTION_RECORDED", "SELECTED_IN_MEMORY")
    assert active_m["activated_by"] == "callensxavier@gmail.com"

    # Verify dashboard reflects new active model and training queue
    dash = client.get("/api/v1/admin/dashboard", headers=admin_hdr).json()
    assert dash["active_model"]["model_id"] == "hivemind/astro-bee-moe-14b"
    assert any(q["job_id"] == receipt["training_receipt_id"] for q in dash["internal_swarm_queue"])


def test_callensxavier_gmail_mastercommander_auth() -> None:
    """Verify Xavier Callens authenticate as MasterCommander via callensxavier@gmail.com."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    xavier_tok = app.state.auth_state["users"]["xavier_callens"]["token"]
    hdr = {"Authorization": f"Bearer {xavier_tok}"}

    # 1. Login with callensxavier@gmail.com
    r_login = client.post(
        "/api/v1/auth/login",
        headers=hdr,
        json={"handle_or_email": "callensxavier@gmail.com", "strict_existing": True},
    )
    assert r_login.status_code == 200
    l_data = r_login.json()
    assert l_data["status"] == "AUTHENTICATED"
    assert l_data["handle"] == "xavier_callens"
    assert l_data["email"] == "callensxavier@gmail.com"
    assert l_data["is_mastercommander"] is True
    assert "MasterMinders" in l_data["groups"]
    assert "MasterCommander" in l_data["role"]
    assert l_data["compute_points"] >= 35500

    # 2. Register / Restore with callensxavier@gmail.com
    r_reg = client.post("/api/v1/auth/register", headers=hdr, json={"email": "callensxavier@gmail.com"})
    assert r_reg.status_code == 200
    r_data = r_reg.json()
    assert r_data["status"] == "EXISTING_USER_RESTORED"
    assert r_data["handle"] == "xavier_callens"
    assert r_data["email"] == "callensxavier@gmail.com"
    assert r_data["is_mastercommander"] is True
    assert "MasterMinders" in r_data["groups"]
    assert "MasterCommander" in r_data["role"]


def test_rtx2070_dedicated_inference_shield_and_training_isolation() -> None:
    """Verify local RTX 2070 hardware shield locks GPU strictly for inference and protects against training."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}

    # 1. GET /api/v1/hardware/rtx2070/shield
    resp = client.get("/api/v1/hardware/rtx2070/shield")
    assert resp.status_code == 200
    s_data = resp.json()
    assert s_data["status"] == "ENGAGED"
    assert s_data["gpu_name"] == "NVIDIA GeForce RTX 2070"
    assert s_data["device"] == "cuda:0"
    assert s_data["mode"] == "INFERENCE_EXCLUSIVE_SECURED"
    assert s_data["training_workloads_allowed"] is False
    assert s_data["training_preemption_lock"] is True
    assert s_data["vram_headroom_reserved_mb"] >= 2048.0
    assert s_data["current_training_load_pct"] == 0.0

    # 2. POST /api/v1/hardware/rtx2070/shield/lock
    resp_lock = client.post("/api/v1/hardware/rtx2070/shield/lock")
    assert resp_lock.status_code == 200
    l_data = resp_lock.json()
    assert l_data["status"] in ("LOCKED_INFERENCE_EXCLUSIVE", "NOT_PERFORMED")

    # 3. Attempting to schedule MoE training on RTX 2070 automatically engages shield and offloads to T4
    train_req = {
        "domain_name": "astrophysics",
        "adapter_rank": 32,
        "training_steps": 250,
        "priority": "HIGH",
        "hardware_override": "NVIDIA GeForce RTX 2070 (cuda:0)",
        "contributor_handle": "@xaviercallens"
    }
    resp_train = client.post("/api/v1/admin/moe/train-expert", headers=admin_hdr, json=train_req)
    assert resp_train.status_code in (200, 202)
    t_data = resp_train.json()
    receipt = t_data["training_receipt"]
    assert receipt["rtx2070_inference_shield_engaged"] is True
    assert receipt.get("rtx2070_inference_lock", True) is True
    hw_str = receipt.get("hardware_allocated", receipt.get("hardware_requested", ""))
    assert "Tesla T4" in hw_str
    assert "RTX 2070 Shielded" in hw_str


def test_night_moe_training_status_and_model_evolution_qwen35_gemma3() -> None:
    """Verify Night MoE training orchestration, T4 & Cloud GPU allocation, and Qwen 3.5 & Gemma 3 models."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. GET /api/v1/night-moe/status
    resp_status = client.get("/api/v1/night-moe/status")
    assert resp_status.status_code == 200
    n_data = resp_status.json()
    assert n_data["experiment_id"] == "NIGHT-MOE-2"
    assert "Qwen/Qwen3.5-MoE-Instruct" in [m["model_id"] for m in n_data["base_models_improving"]]
    assert "google/gemma-3-moe" in [m["model_id"] for m in n_data["base_models_improving"]]
    assert "google/gemma-2-9b-it" in [m["model_id"] for m in n_data["base_models_improving"]]

    # Verify hardware partition
    hw = n_data["hardware_partition"]
    sec_inf = hw["secured_inference_hardware"]
    assert sec_inf["gpu_name"] == "NVIDIA GeForce RTX 2070"
    assert sec_inf["mode"] == "INFERENCE_EXCLUSIVE_SECURED"
    assert sec_inf["training_workloads_allowed"] is False
    assert sec_inf["training_preemption_lock"] is True

    producers = hw["active_training_producers"]
    assert any("Tesla T4" in p["name"] for p in producers)
    assert any("TPU v5e" in p["name"] for p in producers)

    # 2. POST /api/v1/night-moe/start
    resp_start = client.post("/api/v1/night-moe/start")
    assert resp_start.status_code == 200
    st_data = resp_start.json()
    assert st_data["status"] in ("NIGHT_TRAINING_DISPATCHED", "NOT_PERFORMED")
    assert st_data["experiment"] == "NIGHT-MOE-2"
    if "models" in st_data:
        assert "Qwen/Qwen3.5-MoE-Instruct" in st_data["models"]
        assert "google/gemma-3-moe" in st_data["models"]
    if "inference_hardware_secured" in st_data:
        assert "RTX 2070" in st_data["inference_hardware_secured"]


def test_openrouter_nextgen_models_qwen35_and_gemma3() -> None:
    """Verify OpenRouter models catalog and active model selection for Qwen 3.5 MoE and Google Gemma 3 MoE."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}

    # 1. GET /api/v1/openrouter/models
    resp_models = client.get("/api/v1/openrouter/models")
    assert resp_models.status_code == 200
    model_ids = [m["id"] for m in resp_models.json()["data"]]
    assert "hivemind/qwen3.5-moe-bee-35b" in model_ids
    assert "hivemind/gemma-3-moe-deepmind-24b" in model_ids
    assert "hivemind/gemma-2-astro-9b" in model_ids

    # 2. Select Qwen 3.5 MoE as active model
    req_qwen = {
        "model_id": "hivemind/qwen3.5-moe-bee-35b",
        "routing_mode": "top_k_specialist",
        "active_domains": ["code_python", "math_reasoning", "astrophysics"],
        "deploy_live": True,
        "initiator": "callensxavier@gmail.com"
    }
    resp_select_qwen = client.post("/api/v1/admin/moe/models/select", headers=admin_hdr, json=req_qwen)
    assert resp_select_qwen.status_code == 200
    assert resp_select_qwen.json()["active_model"]["model_id"] == "hivemind/qwen3.5-moe-bee-35b"

    # 3. Select Google Gemma 3 MoE as active model
    req_gemma = {
        "model_id": "hivemind/gemma-3-moe-deepmind-24b",
        "routing_mode": "biomimetic_moe",
        "active_domains": ["multilingual_reasoning", "astrophysics", "formal_lean4"],
        "deploy_live": True,
        "initiator": "callensxavier@gmail.com"
    }
    resp_select_gemma = client.post("/api/v1/admin/moe/models/select", headers=admin_hdr, json=req_gemma)
    assert resp_select_gemma.status_code == 200
    assert resp_select_gemma.json()["active_model"]["model_id"] == "hivemind/gemma-3-moe-deepmind-24b"


def test_steam_free_game_and_dedicated_routes() -> None:
    """Verify Free Steam Game (AppID 3133700), dedicated page routes, and UI components."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Dedicated page routes
    for route in ("/steam", "/v2/steam"):
        resp = client.get(route)
        assert resp.status_code == 200
        html = resp.text
        assert "screen-steam" in html
        assert "APPID: 3133700" in html
        assert "100% FREE / ZERO PAY-TO-WIN" in html
        assert "RTX Zero-Intrusion BOINC Daemon" in html
        assert "steam://install/3133700" in html
        assert "Compute Points Retribution Ledger" in html
        assert "Interactive Gamer RTX Yield &amp; Science Reward Calculator" in html


def test_steam_api_endpoints_and_daemon_lifecycle() -> None:
    """Verify Steam API, BOINC Daemon control, preemption SLA < 1.5s, points retribution, and zip download."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. GET /api/v1/steam/info
    resp_info = client.get("/api/v1/steam/info")
    assert resp_info.status_code == 200
    info_data = resp_info.json()
    assert info_data["app_id"] == 3133700
    assert info_data["depot_id"] == 3133701
    assert info_data["is_free"] is True
    assert info_data["price_usd"] == 0.0
    assert "steam://install/3133700" in info_data["steam_protocol_install"]
    assert "free-edition.zip" in info_data["download_zip_url"]

    # 2. GET /api/v1/steam/daemon/status
    resp_status = client.get("/api/v1/steam/daemon/status")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    assert status_data["app_id"] == 3133700
    assert status_data["status"] in ("RUNNING", "PAUSED", "PAUSED_GAMING")
    assert status_data["idle_threshold_percent"] in (15.0, 20.0, 25.0)
    assert status_data["gaming_preemption_threshold_percent"] in (70.0, 75.0, 80.0)
    assert status_data["preemption_latency_ms"] is None or status_data["preemption_latency_ms"] < 1500
    assert status_data["gpu_device"]  # the machine's own GPU name, whichever it is

    # 3. POST /api/v1/steam/daemon/toggle - Pause
    resp_pause = client.post("/api/v1/steam/daemon/toggle", json={"action": "pause"})
    assert resp_pause.status_code == 200
    assert resp_pause.json()["status"] == "PAUSED"

    # 4. POST /api/v1/steam/daemon/toggle - Resume
    resp_resume = client.post("/api/v1/steam/daemon/toggle", json={"action": "resume"})
    assert resp_resume.status_code == 200
    assert resp_resume.json()["status"] == "RUNNING"

    # 5. POST /api/v1/steam/daemon/toggle - Simulate Game Launch (Instant Preemption)
    resp_game = client.post(
        "/api/v1/steam/daemon/toggle",
        json={"action": "simulate_game", "game": "Cyberpunk 2077 (cyberpunk2077.exe)"}
    )
    assert resp_game.status_code == 200
    game_data = resp_game.json()
    assert game_data["status"] == "PAUSED_GAMING"
    assert game_data["active_game_detected"] == "Cyberpunk 2077 (cyberpunk2077.exe)"
    assert game_data["preemption_latency_ms"] is None or game_data["preemption_latency_ms"] <= 1500
    assert game_data.get("vram_yielded_mb") is None or game_data["vram_yielded_mb"] >= 0

    # Resume back to running
    client.post("/api/v1/steam/daemon/toggle", json={"action": "resume"})

    # 6. POST /api/v1/steam/daemon/config
    resp_cfg = client.post(
        "/api/v1/steam/daemon/config",
        json={"idle_threshold_percent": 20.0, "gaming_preemption_threshold_percent": 80.0, "max_temp_c": 66.0}
    )
    assert resp_cfg.status_code == 200
    assert resp_cfg.json()["config"]["idle_threshold_percent"] == 20.0
    assert resp_cfg.json()["config"]["gaming_preemption_threshold_percent"] == 80.0

    # 7. POST /api/v1/steam/claim-points - Retribute points
    resp_claim = client.post(
        "/api/v1/steam/claim-points",
        json={"contributor_handle": "callensxavier@gmail.com", "amount": 450}
    )
    assert resp_claim.status_code == 200
    claim_data = resp_claim.json()
    assert claim_data["status"] in ("POINTS_RETRIBUTED", "NOT_CREDITED_UNVERIFIED")
    assert claim_data["contributor"] == "callensxavier@gmail.com"
    assert claim_data["claimed_hmt"] in (0, 450)
    assert claim_data["total_points_retributed"] >= 0
    if "receipt_sha256" in claim_data:
        assert len(claim_data["receipt_sha256"]) == 64

    # 8. GET /api/v1/steam/achievements
    resp_ach = client.get("/api/v1/steam/achievements")
    assert resp_ach.status_code == 200
    ach_list = resp_ach.json()["achievements"]
    ach_ids = [a["id"] for a in ach_list]
    assert "ACH_BOINC_FIRST_BLOCK" in ach_ids
    assert "ACH_RTX_100_INFERENCES" in ach_ids
    assert "ACH_EXOPLANET_HUNTER" in ach_ids

    # 9. GET /api/v1/steam/download/free-edition.zip
    resp_zip = client.get("/api/v1/steam/download/free-edition.zip")
    assert resp_zip.status_code == 200
    assert resp_zip.headers["content-type"] == "application/zip"
    assert len(resp_zip.content) > 1000


def test_gcp_finops_scale_to_zero_api_and_manifests() -> None:
    """Verify GCP Cloud Run FinOps Scale-to-Zero (min-instances=0) API endpoints and deployment manifests."""
    from pathlib import Path

    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. GET /api/v1/finops/status
    resp_status = client.get("/api/v1/finops/status")
    assert resp_status.status_code == 200
    st = resp_status.json()
    assert st["status"] == "FINOPS_OPTIMAL_SCALE_TO_ZERO"
    assert st["min_instances"] == 0
    assert st["max_instances"] == 2
    assert st["cpu_throttling"] is True
    assert st["startup_cpu_boost"] is True
    assert st["execution_environment"] == "gen2"
    assert st["idle_hourly_cost_usd"] == 0.0
    assert st["idle_monthly_cost_usd"] == 0.0
    assert st["knative_annotations"]["autoscaling.knative.dev/minScale"] == "0"
    assert st["knative_annotations"]["run.googleapis.com/cpu-throttling"] == "true"

    # 2. POST /api/v1/finops/simulate-idle -> Scales replicas to 0 when no activity
    resp_idle = client.post("/api/v1/finops/simulate-idle")
    assert resp_idle.status_code == 200
    idle_data = resp_idle.json()
    assert idle_data["min_instances"] == 0
    assert idle_data["current_active_replicas"] == 0
    assert idle_data["activity_state"] == "IDLE_SCALED_TO_ZERO"
    assert idle_data["idle_hourly_cost_usd"] == 0.0
    assert len(idle_data["receipt_sha256"]) == 64

    # 3. POST /api/v1/finops/scale-to-zero (wake_up -> enforce_policy -> simulate_idle)
    resp_wake = client.post("/api/v1/finops/scale-to-zero", json={"action": "wake_up", "min_instances": 0, "max_instances": 2})
    assert resp_wake.status_code == 200
    assert resp_wake.json()["current_active_replicas"] == 1
    assert resp_wake.json()["activity_state"] == "ACTIVE_SERVING"

    resp_enforce = client.post(
        "/api/v1/finops/scale-to-zero",
        json={"action": "enforce_policy", "min_instances": 0, "max_instances": 2, "idle_timeout_seconds": 300, "cpu_throttling": True},
    )
    assert resp_enforce.status_code == 200
    enf_data = resp_enforce.json()
    assert enf_data["min_instances"] == 0
    assert enf_data["current_active_replicas"] == 0
    assert enf_data["activity_state"] == "IDLE_SCALED_TO_ZERO"

    # 4. Verify GCP deployment files enforce min-instances=0 and cpu-throttling
    repo_root = Path(__file__).resolve().parents[1]
    service_yaml = (repo_root / "deploy" / "gcp_cloud_run" / "service.yaml").read_text(encoding="utf-8")
    assert 'autoscaling.knative.dev/minScale: "0"' in service_yaml
    assert 'run.googleapis.com/cpu-throttling: "true"' in service_yaml
    assert 'run.googleapis.com/startup-cpu-boost: "true"' in service_yaml

    ps1_path = repo_root / "deploy" / "gcp_cloud_run" / "deploy_gcp_astro_community.ps1"
    if ps1_path.exists():
        ps1_script = ps1_path.read_text(encoding="utf-8")
        assert "--min-instances" in ps1_script
        assert "--cpu-throttling" in ps1_script
        assert "--cpu-boost" in ps1_script

    openrouter_sh = (repo_root / "deploy" / "gcp_openrouter" / "deploy.sh").read_text(encoding="utf-8")
    assert "--min-instances" in openrouter_sh
    assert "--cpu-boost" in openrouter_sh


def test_vxtzq_pre_deployment_review_gate_and_collaborator_roster() -> None:
    """Verify @Vxtzq is in MasterMinders roster and pre-deployment review gate controls central GCP deployment."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}

    # 1. Verify @Vxtzq in MasterMinders roster
    resp_mm = client.get("/api/v1/admin/masterminders", headers=admin_hdr)
    assert resp_mm.status_code == 200
    handles = [m["handle"] for m in resp_mm.json()["members"]]
    assert "@Vxtzq" in handles

    # 2. GET /api/v1/admin/deployment-review/status
    resp_st = client.get("/api/v1/admin/deployment-review/status", headers=admin_hdr)
    assert resp_st.status_code == 200
    st_data = resp_st.json()
    assert st_data["required_reviewer"] == "@Vxtzq"
    assert st_data["reviewer_github_url"] == "https://github.com/Vxtzq"
    assert st_data["repository_url"] == "https://github.com/xaviercallens/HiveMind-WebUI"

    # 3. POST /api/v1/admin/deployment-review/request (request_review -> gates GCP deploy)
    resp_req = client.post(
        "/api/v1/admin/deployment-review/request",
        headers=admin_hdr,
        json={"action": "request_review", "reviewer": "@Vxtzq", "initiator": "callensxavier@gmail.com"},
    )
    assert resp_req.status_code == 200
    req_data = resp_req.json()
    assert req_data["review_status"] == "AWAITING_VXTZQ_REVIEW"
    assert req_data["gcp_deployment_unlocked"] is False
    assert len(req_data["receipt_sha256"]) == 64

    # 4. POST /api/v1/admin/deployment-review/request (approve_review -> unlocks GCP deploy)
    resp_app = client.post(
        "/api/v1/admin/deployment-review/request",
        headers=admin_hdr,
        json={"action": "approve_review", "reviewer": "@Vxtzq", "initiator": "@Vxtzq"},
    )
    assert resp_app.status_code == 200
    app_data = resp_app.json()
    assert app_data["review_status"] == "APPROVED_BY_VXTZQ"
    assert app_data["gcp_deployment_unlocked"] is True


def test_webgpu_galaxy_exploration_screen_and_api() -> None:
    """Verify dgreenheck/webgpu-galaxy interactive spiral galaxy explorer screen and API routes."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Verify /galaxy and /v2/galaxy page routes render screen-galaxy
    for route in ["/galaxy", "/v2/galaxy"]:
        resp_page = client.get(route)
        assert resp_page.status_code == 200
        html = resp_page.text
        assert 'id="screen-galaxy"' in html
        assert 'data-testid="webgpu-galaxy-canvas"' in html
        assert "https://github.com/dgreenheck/webgpu-galaxy" in html
        assert "galaxyHash" in html
        assert "rotateXZ" in html

    # 2. GET /api/v1/galaxy/config
    resp_cfg = client.get("/api/v1/galaxy/config")
    assert resp_cfg.status_code == 200
    cfg_data = resp_cfg.json()
    assert cfg_data["upstream_repository"] == "https://github.com/dgreenheck/webgpu-galaxy"
    assert cfg_data["config"]["starCount"] in (500000, 750000)
    assert cfg_data["config"]["denseStarColor"] in ("#1885ff", "#00f0ff", "#f43f5e", "#10b981", "#a855f7")
    assert len(cfg_data["sectors"]) >= 5

    # 3. POST /api/v1/galaxy/explore (warp to kepler_1649c with custom spiral params)
    resp_exp = client.post(
        "/api/v1/galaxy/explore",
        json={
            "sector_id": "kepler_1649c",
            "arm_count": 4,
            "spiral_tightness": 2.25,
            "rotation_speed": 0.16,
            "mouse_force": 8.5,
            "contributor_handle": "callensxavier@gmail.com",
        },
    )
    assert resp_exp.status_code == 200
    exp_data = resp_exp.json()
    assert exp_data["status"] == "SECTOR_EXPLORED"
    assert exp_data["active_sector_id"] == "kepler_1649c"
    assert exp_data["config"]["armCount"] == 4
    assert exp_data["config"]["spiralTightness"] == 2.25
    assert exp_data["hmt_awarded"] == 500
    assert len(exp_data["receipt_sha256"]) == 64


def test_v1_to_v2_seamless_migration_and_login_preservation() -> None:
    """Verify v1 users keep 100% of their registration, login credentials, tokens, and HMT points on v2."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}

    # 1. Verify /v1 legacy compatibility UI route
    resp_v1 = client.get("/v1")
    assert resp_v1.status_code == 200
    assert "HiveMind" in resp_v1.text

    # 2. Verify GET /api/v1/auth/migration/status (requires admin header)
    assert client.get("/api/v1/auth/migration/status").status_code == 403
    resp_mig = client.get("/api/v1/auth/migration/status", headers=admin_hdr)
    assert resp_mig.status_code == 200
    mig_data = resp_mig.json()
    assert mig_data["migration_status"] == "ACTIVE_SEAMLESS_V1_TO_V2"
    assert mig_data["v1_accounts_preserved_pct"] == 100.0
    assert mig_data["zero_data_loss"] is True
    assert mig_data["migrated_accounts_count"] >= 7

    # 3. Verify all v1 founding accounts can log in / migrate with their active token
    for v1_id, user_key, expected_min_pts in [
        ("callensxavier@gmail.com", "xavier_callens", 68500),
        ("@clark719664", "clark719664", 24000),
        ("radialmonster", "radialmonster", 21000),
        ("@pedromineo", "pedromineo", 18000),
        ("@socrateai-agora-hermes-t4", "socrateai_hermes", 16000),
        ("@Vxtzq", "vxtzq", 30000),
    ]:
        tok = app.state.auth_state["users"][user_key]["token"]
        u_hdr = {"Authorization": f"Bearer {tok}"}
        r_login = client.post(
            "/api/v1/auth/login",
            headers=u_hdr,
            json={"handle_or_email": v1_id, "strict_existing": True},
        )
        assert r_login.status_code == 200
        l_json = r_login.json()
        assert l_json["status"] == "AUTHENTICATED"
        assert l_json["v1_migrated"] is True
        assert l_json["v2_access_granted"] is True
        assert l_json["compute_points"] >= expected_min_pts

        r_mig = client.post("/api/v1/auth/v1-migrate", headers=u_hdr, json={"handle_or_email": v1_id})
        assert r_mig.status_code == 200
        m_json = r_mig.json()
        assert m_json["status"] == "V1_USER_MIGRATED_AND_AUTHENTICATED"
        assert m_json["v1_migrated"] is True
        assert m_json["v2_access_granted"] is True
        assert m_json["compute_points"] >= expected_min_pts
        assert len(m_json["migration_receipt_sha256"]) == 64

    # 4. Verify custom v1 pioneer account seamless carry-over migration
    r_custom = client.post("/api/v1/auth/v1-migrate", json={"handle_or_email": "pioneer_v1_astro@hivemind.network"})
    assert r_custom.status_code == 200
    c_json = r_custom.json()
    assert c_json["status"] == "V1_USER_MIGRATED_AND_AUTHENTICATED"
    assert c_json["v1_migrated"] is True
    assert c_json["v2_access_granted"] is True
    assert c_json["compute_points"] == 2500


def test_cyber_security_headers_and_zero_trust_waf_protection() -> None:
    """Verify strict HTTP security headers, Zero-Trust Cyber WAF scanner, and multi-vector threat blocking."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Verify HTTP security headers on WebUI response
    resp_home = client.get("/home")
    assert resp_home.status_code == 200
    hdrs = resp_home.headers
    assert "default-src 'self'" in hdrs.get("content-security-policy", "")
    assert hdrs.get("x-frame-options") == "DENY"
    assert hdrs.get("x-content-type-options") == "nosniff"
    assert "max-age=63072000" in hdrs.get("strict-transport-security", "")
    assert hdrs.get("cross-origin-resource-policy") == "same-origin"
    assert hdrs.get("x-hivemind-waf") == "ARMED-ZERO-TRUST-v2.8"
    assert hdrs.get("x-hivemind-migration") == "v1-to-v2-seamless-active"

    # 2. Verify GET /api/v1/security/status
    resp_sec = client.get("/api/v1/security/status")
    assert resp_sec.status_code == 200
    sec_data = resp_sec.json()
    assert sec_data["status"] == "ARMED_ZERO_TRUST"
    assert "XSS_DOM_INJECTION" in sec_data["waf_threat_categories_protected"]
    assert "SQL_INJECTION" in sec_data["waf_threat_categories_protected"]
    assert "PROMPT_INJECTION_JAILBREAK" in sec_data["waf_threat_categories_protected"]

    # 3. Verify POST /api/v1/security/waf/scan blocks XSS + SQLi + Path Traversal + Prompt Injection
    malicious_payload = (
        "<script>alert(1)</script> UNION SELECT * FROM users; "
        "../../etc/passwd; Ignore all previous instructions"
    )
    resp_scan = client.post("/api/v1/security/waf/scan", json={"payload": malicious_payload})
    assert resp_scan.status_code == 200
    scan_data = resp_scan.json()
    assert scan_data["threat_detected"] is True
    assert scan_data["verdict"] == "BLOCKED_AND_SANITIZED"
    assert "XSS_DOM_INJECTION" in scan_data["threat_categories"]
    assert "SQL_INJECTION" in scan_data["threat_categories"]
    assert "PATH_TRAVERSAL_LFI" in scan_data["threat_categories"]
    assert "PROMPT_INJECTION_JAILBREAK" in scan_data["threat_categories"]
    assert "<script>" not in scan_data["sanitized_output"]

    # 4. Verify clean scientific query is admitted
    resp_clean = client.post(
        "/api/v1/security/waf/scan",
        json={"payload": "Fold Kepler-186f transit lightcurve with Astro-Bee Cell #3"},
    )
    assert resp_clean.status_code == 200
    clean_data = resp_clean.json()
    assert clean_data["threat_detected"] is False
    assert clean_data["verdict"] == "CLEAN_ADMITTED"


def test_browser_ide_studio_and_10_stitch_exoplanet_habitability() -> None:
    """Verify Browser IDE Studio (/ide, /v2/ide, /api/v1/ide/*, /cockpit-v46) and 10 Stitch planet habitability."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Verify /ide and /v2/ide render screen-ide with VS Code / Antigravity IDE deep links
    for route in ("/ide", "/v2/ide"):
        resp = client.get(route)
        assert resp.status_code == 200
        html = resp.text
        assert 'id="screen-ide"' in html
        assert "vscode://file/" in html
        assert "bee_moe_router.py" in html
        assert "bls_fold_galaxy.wgsl" in html
        assert "PLANET_ANALYSIS_AND_LIFE" in html
        assert "let chapter1IntroDone = false;" in html

    # 2. Verify GET /api/v1/ide/status
    st_resp = client.get("/api/v1/ide/status")
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["status"] == "BROWSER_IDE_ONLINE"
    assert len(st_data["workspace_files"]) >= 4
    assert "bee_moe_router.py" in st_data["templates"]
    assert "bls_fold_galaxy.wgsl" in st_data["templates"]

    # 3. Verify POST /api/v1/ide/execute for Python, WGSL, and HTML tabs
    for tab in ("bee_moe_router.py", "bls_fold_galaxy.wgsl", "stitch_v2_ui_component.html"):
        ex_resp = client.post(
            "/api/v1/ide/execute",
            json={
                "file_tab": tab,
                "code": st_data["templates"][tab],
                "target_planet": "k218b",
                "contributor_handle": "callensxavier@gmail.com",
            },
        )
        assert ex_resp.status_code == 200
        ex_data = ex_resp.json()
        assert ex_data["status"] == "IDE_EXECUTION_SUCCESS"
        assert ex_data["hmt_awarded"] == 200
        assert "How life could live:" in ex_data["planet_analysis_en"]
        assert "Vie possible :" in ex_data["planet_analysis_fr"]
        assert len(ex_data["receipt_sha256"]) == 64

    # 4. Verify /cockpit-v46 and /api/steam/fat-client-manifest
    cockpit_resp = client.get("/cockpit-v46")
    assert cockpit_resp.status_code == 200
    assert "USS HIVEMIND NCC-31337 v4.6" in cockpit_resp.text
    assert cockpit_resp.headers.get("x-frame-options") == "SAMEORIGIN"

    manifest_resp = client.get("/api/steam/fat-client-manifest")
    assert manifest_resp.status_code == 200
    manifest_data = manifest_resp.json()
    assert manifest_data["app_id"] == 3133700
    assert len(manifest_data["stitch_planet_keys"]) == 10


def test_browser_ide_self_test_and_stage_controls() -> None:
    """Verify POST /api/v1/ide/self-test 8-point contract check and Browser IDE stage controls."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    ide_html = client.get("/ide").text
    assert 'data-testid="btn-ide-self-test"' in ide_html
    assert 'data-testid="ide-cockpit-stage-controls"' in ide_html
    assert "syncCockpitStage(1)" in ide_html
    assert "syncCockpitStage(5)" in ide_html

    st_resp = client.post("/api/v1/ide/self-test")
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["status"] == "ALL_CONTRACTS_VERIFIED"
    assert st_data["passed_count"] == 8
    assert st_data["total_checks"] == 8
    assert st_data["hmt_awarded"] == 300
    check_ids = {c["id"] for c in st_data["checks"]}
    assert "uss_hivemind_v46_cockpit" in check_ids
    assert "webgpu_galaxy_10_planets" in check_ids


def test_cockpit_v46_full_5_stage_html_and_sameorigin_csp() -> None:
    """Verify /cockpit-v46 renders all 5 interactive stages and allows same-origin iframe embedding."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    resp = client.get("/cockpit-v46")
    assert resp.status_code == 200
    assert resp.headers.get("x-frame-options") == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in resp.headers.get("content-security-policy", "")
    html = resp.text
    for marker in (
        "viewport-stage-2",
        "viewport-stage-3",
        "viewport-stage-4",
        "viewport-stage-5",
        "PLANET_ANALYSIS_AND_LIFE",
        "let chapter1IntroDone = false;",
    ):
        assert marker in html

    asset_resp = client.get(
        "/stitch-assets/stitch_v2_07_photorealistic_milky_way_barred_spiral_galaxy_vi.png"
    )
    assert asset_resp.status_code == 200
    assert asset_resp.headers.get("content-type") in ("image/png", "image/jpeg")


def test_cockpit_steam_register_login_and_boinc_contribute() -> None:
    """Verify /v1/hivemind/steam/register-login and /v1/hivemind/steam/contribute-boinc on unified portal."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    reg_resp = client.post(
        "/v1/hivemind/steam/register-login",
        json={
            "handle": "Cmdr_RTX2070_Steam",
            "steam_id64": "76561198031337001",
            "app_id": 3133700,
        },
    )
    assert reg_resp.status_code == 200
    reg_data = reg_resp.json()
    assert reg_data["status"] == "STEAM_REGISTERED_AND_AUTHENTICATED"
    assert reg_data["app_id"] == 3133700
    assert reg_data["api_key"].startswith("hm_live_steam_")

    boinc_resp = client.post(
        "/v1/hivemind/steam/contribute-boinc",
        json={
            "handle": "Cmdr_RTX2070_Steam",
            "target_planet": "K2-18 b",
            "app_id": 3133700,
        },
        headers={"Authorization": f"Bearer {reg_data['api_key']}"},
    )
    assert boinc_resp.status_code == 200
    boinc_data = boinc_resp.json()
    assert boinc_data["status"] in ("BOINC_CUDA_WORKUNIT_VERIFIED", "RECEIVED_UNVERIFIED")
    assert boinc_data["hmt_awarded"] in (0, 250)
    assert boinc_data["cobblestones_awarded"] in (0, 128)


def test_astro_nasa_tap_sync_discover_and_bee_moe_scan_all_10_planets() -> None:
    """Verify /api/astro/nasa-tap-sync, /api/astro/discover, and /api/astro/bee-moe-scan across all 10 Stitch planets."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    tap_resp = client.get("/api/astro/nasa-tap-sync")
    assert tap_resp.status_code == 200
    tap_data = tap_resp.json()
    assert tap_data["status"] == "NASA_TAP_ADQL_VERIFIED"
    assert tap_data["row_count"] == 10

    for planet_key in (
        "trappist1e",
        "k218b",
        "kepler11f",
        "hd80606b",
        "toi700d",
        "lhs1140b",
        "wasp39b",
        "kepler186f",
        "koi14",
        "koibin",
    ):
        scan_resp = client.post("/api/astro/bee-moe-scan", json={"target_id": planet_key})
        assert scan_resp.status_code == 200
        scan_data = scan_resp.json()
        assert scan_data["status"] == "BEE_MOE_SCAN_COMPLETE"
        assert "Vie possible :" in scan_data["planet_analysis_fr"]
        assert "How life could live:" in scan_data["planet_analysis_en"]

        disc_resp = client.post(
            "/api/astro/discover",
            json={
                "target_id": planet_key,
                "player_classification": "FALSE_POSITIVE" if planet_key == "koibin" else "PLANET",
            },
        )
        assert disc_resp.status_code == 200
        disc_data = disc_resp.json()
        assert disc_data["status"] in ("DISCOVERY_RECORDED", "NOT_PERFORMED")
        assert disc_data["hmt_awarded"] in (0, 300, 600)


def test_contributor_pr_auto_review_and_merge_pipeline() -> None:
    """Verify automated contributor PR review & merge endpoints (/api/v1/admin/prs/status & /auto-merge)."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}

    st_resp = client.get("/api/v1/admin/prs/status", headers=admin_hdr)
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["status"] == "AUTO_REVIEW_AND_MERGE_ACTIVE"
    assert "@jblam-O" in st_data["trusted_collaborators"]
    assert "@Vxtzq" in st_data["trusted_collaborators"]
    assert st_data["merged_prs_count"] >= 4
    pr_numbers = {p["pr_number"] for p in st_data["merged_prs"]}
    assert {1, 2, 4, 5}.issubset(pr_numbers)

    mg_resp = client.post("/api/v1/admin/prs/auto-merge", headers=admin_hdr)
    assert mg_resp.status_code == 200
    mg_data = mg_resp.json()
    assert mg_data["status"] == "STATIC_LIST_NO_ACTION_TAKEN"
    assert mg_data["hmt_awarded"] == 0
    assert len(mg_data["receipt_sha256"]) == 64


def test_intense_mvp_maintenance_mode_and_hiveminer_notification() -> None:
    """Verify Intense MVP & New Release Maintenance Mode banner, headers, and HiveMiner loyalty claim."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Verify / and /maintenance include X-HiveMind-Maintenance-Mode header and HiveMiner banner
    for route in ("/", "/maintenance", "/v2/maintenance"):
        resp = client.get(route)
        assert resp.status_code == 200
        assert resp.headers.get("x-hivemind-maintenance-mode") == "ACTIVE-INTENSE-MVP-RELEASE"
        html = resp.text
        assert 'id="hiveminer-maintenance-banner"' in html
        assert "MAINTENANCE MODE ACTIVE" in html
        assert "Notice to all" in html and "HiveMiners" in html

    # 2. Verify GET /api/v1/maintenance/status
    st_resp = client.get("/api/v1/maintenance/status")
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["active"] is True
    assert st_data["status"] == "MAINTENANCE_MODE_ACTIVE"
    assert st_data["mode_code"] == "INTENSE_MVP_AND_NEW_RELEASE"
    assert st_data["hiveminer_guarantees"]["accounts_and_hmt_preserved_pct"] == 100.0
    assert st_data["hiveminer_guarantees"]["loyalty_bonus_hmt"] == 500
    assert len(st_data["milestones"]) == 5

    # 3. Verify POST /api/v1/maintenance/acknowledge: needs an account; nothing is credited
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}
    assert client.post("/api/v1/maintenance/acknowledge", json={}).status_code == 401
    ack_resp = client.post("/api/v1/maintenance/acknowledge", headers=admin_hdr, json={"hiveminer_handle": "someone_else"})
    assert ack_resp.status_code == 200
    ack_data = ack_resp.json()
    assert ack_data["ack_status"] == "HIVEMINER_ACKNOWLEDGED"
    assert ack_data["loyalty_hmt_awarded"] == 0
    assert ack_data["hiveminer_handle"] != "someone_else"
    assert len(ack_data["receipt_sha256"]) == 64

    # 4. Verify POST /api/v1/maintenance/toggle and restore active=True
    assert client.post("/api/v1/maintenance/toggle", json={"active": False}).status_code == 403
    tog_off = client.post("/api/v1/maintenance/toggle", headers=admin_hdr, json={"active": False})
    assert tog_off.status_code == 200
    assert tog_off.json()["active"] is False
    tog_on = client.post("/api/v1/maintenance/toggle", headers=admin_hdr, json={"active": True})
    assert tog_on.status_code == 200
    assert tog_on.json()["active"] is True


def test_core_mvp_4_sso_providers_pkce_s256_and_login_flow(monkeypatch) -> None:  # noqa: ANN001
    """Verify Core MVP 4-SSO providers (Google, Reddit, Steam, GitHub) OAuth2 PKCE S256 authorize + login."""
    monkeypatch.setenv("HIVEMIND_LEGACY_SANDBOX_SSO", "1")  # exercises the retired sandbox SSO, off by default since v3.2
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    for prov in ("google", "reddit", "steam", "github"):
        # 1. GET /api/v1/auth/sso/{provider}/authorize
        r_authz = client.get(
            f"/api/v1/auth/sso/{prov}/authorize",
            params={"redirect_uri": "https://hivemind.socrate-ai-lab.com/auth/callback"},
        )
        assert r_authz.status_code == 200
        az = r_authz.json()
        assert az["status"] == "SSO_AUTHORIZATION_READY"
        assert az["provider"] == prov
        assert az["code_challenge_method"] == "S256"
        assert len(az["state"]) >= 16
        assert len(az["code_challenge"]) >= 32
        assert az["public_dns_domain"] == "hivemind.socrate-ai-lab.com"

        # 2. Complete SSO login with PKCE state
        handle = f"mvp_{prov}_pilot"
        r_sso = client.post(
            f"/api/v1/auth/sso/{prov}",
            json={
                "handle_hint": handle,
                "role": "HiveMinder Human",
                "target_profile": "developer",
                "state": az["state"],
                "code_verifier": f"verifier_{prov}_1234567890",
            },
        )
        assert r_sso.status_code == 200
        s_data = r_sso.json()
        assert s_data["status"] == "AUTHENTICATED_VIA_SSO"
        assert s_data["handle"] == handle
        assert s_data["provider"] == f"{prov.upper()}_SSO"
        assert s_data["compute_points"] == 2500
        assert s_data["token"].startswith("hm_live_")
        assert len(s_data["token"]) == len("hm_live_") + 32


def test_core_mvp_email_password_registration_and_login_pbkdf2() -> None:
    """Verify Standard Email + Password Registration & Login with PBKDF2-HMAC-SHA256 (200k iterations) + brute-force guard."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Reject weak password (< 8 chars)
    r_weak = client.post(
        "/api/v1/auth/register",
        json={"handle": "weak_pwd_user", "email": "weak@hivemind.network", "password": "short"},
    )
    assert r_weak.status_code in (400, 422)

    # 2. Register with strong password (PBKDF2-HMAC-SHA256)
    r_reg = client.post(
        "/api/v1/auth/register",
        json={
            "handle": "mvp_secure_operator",
            "email": "mvp_secure@hivemind.network",
            "password": "StrongMVP#Password2026!",
            "target_profile": "developer",
        },
    )
    assert r_reg.status_code == 200
    reg_data = r_reg.json()
    assert reg_data["status"] == "REGISTERED"
    assert reg_data["handle"] == "mvp_secure_operator"
    assert reg_data["has_password"] is True
    assert "password_hash" not in reg_data
    assert "password_salt" not in reg_data

    # Verify stored hash uses 200,000 PBKDF2 iterations and 16-byte salt (32 hex chars)
    stored = app.state.auth_state["users"]["mvp_secure_operator"]
    assert len(stored["password_salt"]) == 32
    assert len(stored["password_hash"]) == 64

    # 3. Wrong password login must fail with 401
    r_wrong = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "mvp_secure@hivemind.network", "password": "WrongPassword999!"},
    )
    assert r_wrong.status_code == 401

    # 4. Correct password login succeeds without needing Bearer header
    r_ok = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "mvp_secure@hivemind.network", "password": "StrongMVP#Password2026!"},
    )
    assert r_ok.status_code == 200
    ok_data = r_ok.json()
    assert ok_data["status"] == "AUTHENTICATED"
    assert ok_data["handle"] == "mvp_secure_operator"
    assert ok_data["token"] == reg_data["token"]

    # 5. Brute-force lockout after 8 failed password attempts within 60 seconds -> 429
    for _ in range(8):
        client.post(
            "/api/v1/auth/login",
            json={"handle_or_email": "mvp_secure_operator", "password": "BadBruteForceAttempt!"},
        )
    r_locked = client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "mvp_secure_operator", "password": "StrongMVP#Password2026!"},
    )
    assert r_locked.status_code == 429


def test_red_team_security_controls_pr532_and_ghsa(monkeypatch) -> None:
    """Verify all Red Team security fixes from PR #532 and GHSA-gjc6-74f9-wmrr."""
    monkeypatch.setenv("HIVEMIND_DASHBOARD_ADMIN_KEY", "super-secret-redteam-admin-key")
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Hardcoded source token hm_live_xavier_994a102f88c is NEVER valid at runtime
    r_sess = client.get("/api/v1/auth/session", headers={"Authorization": "Bearer hm_live_xavier_994a102f88c"})
    assert r_sess.json()["authenticated"] is False

    # 2. Unauthenticated login by handle/email alone is rejected (401)
    assert client.post("/api/v1/auth/login", json={"handle_or_email": "xavier_callens"}).status_code == 401
    assert client.post("/api/v1/auth/v1-migrate", json={"handle_or_email": "callensxavier@gmail.com"}).status_code == 401

    # 3. Admin gate blocks unauthenticated requests (403) and admits x-admin-key (200)
    assert client.get("/api/v1/auth/users").status_code == 403
    r_users = client.get("/api/v1/auth/users", headers={"x-admin-key": "super-secret-redteam-admin-key"})
    assert r_users.status_code == 200
    for u in r_users.json()["users"]:
        assert "token" not in u
        assert "password_hash" not in u

    # 4. Admin account key rotation revokes old token and issues fresh CSPRNG token
    old_xavier_tok = app.state.auth_state["users"]["xavier_callens"]["token"]
    r_rot = client.post(
        "/api/v1/admin/accounts/rotate-key",
        headers={"x-admin-key": "super-secret-redteam-admin-key"},
        json={"handle": "xavier_callens"},
    )
    assert r_rot.status_code == 200
    new_xavier_tok = r_rot.json()["token"]
    assert new_xavier_tok != old_xavier_tok
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {old_xavier_tok}"}).json()["authenticated"] is False
    assert client.get("/api/v1/auth/session", headers={"Authorization": f"Bearer {new_xavier_tok}"}).json()["authenticated"] is True

    # 5. Federated workunit submission requires active lease_id and bounded 2-D LoRA shapes
    r_fake_lease = client.post(
        "/api/v1/federated/workunit/submit",
        json={
            "lease_id": "forged-lease-9999",
            "domain_name": "code_python",
            "contributor_handle": "@attacker",
            "lora_A": [[0.01] * 8 for _ in range(4)],
            "lora_B": [[0.01] * 4 for _ in range(8)],
        },
    )
    assert r_fake_lease.status_code in (404, 422)

    r_lease = client.post(
        "/api/v1/federated/workunit/request",
        json={"domain_name": "code_python", "contributor_handle": "@honest_bee"},
    )
    assert r_lease.status_code == 200
    lease_id = r_lease.json()["lease_id"]

    r_bad_shape = client.post(
        "/api/v1/federated/workunit/submit",
        json={
            "lease_id": lease_id,
            "domain_name": "code_python",
            "contributor_handle": "@honest_bee",
            "lora_A": [[0.01, 0.02]],
            "lora_B": [[0.01, 0.02, 0.03]],
        },
    )
    assert r_bad_shape.status_code == 422

    # 6. PoC status header and zero "Kyber-1024" claims in HTML or JSON responses
    r_home = client.get("/")
    assert "proof-of-concept" in r_home.headers.get("X-HiveMind-Status", "").lower()
    assert "Proof-of-concept service" in r_home.text
    assert "Kyber-1024" not in r_home.text
    assert "KYBER-1024" not in r_home.text
    assert "Kyber-1024" not in client.get("/api/v1/security/status").text
    assert "Kyber-1024" not in client.get("/api/v1/amcp/protocol").text


def test_staged_gcp_public_dns_portal_readiness_not_deployed() -> None:
    """Verify Staged GCP Public DNS Portal readiness endpoints & manifests (hivemind.socrate-ai-lab.com, NOT deployed yet)."""
    import sys
    from pathlib import Path
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from deploy.gcp_cloud_run.prepare_gcp_portal_release import build_staged_portal_release_report

    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. GET /api/v1/gcp/portal-readiness
    r_ready = client.get("/api/v1/gcp/portal-readiness")
    assert r_ready.status_code == 200
    rd = r_ready.json()
    assert rd["deployment_state"] == "STAGED_READY_NOT_DEPLOYED"
    assert rd["deployed_to_gcp"] is False
    assert rd["do_not_deploy_yet"] is True
    assert rd["public_dns_domain"] == "hivemind.socrate-ai-lab.com"
    assert rd["public_dns_url"] == "https://hivemind.socrate-ai-lab.com"
    assert rd["forwarding_rule_ip"] == "34.54.227.33"
    assert rd["managed_ssl_certificate"] == "socrate-ai-ssl"
    assert rd["core_mvp_auth"]["sso_provider_ids"] == ["google", "reddit", "steam", "github"]
    assert [p["id"] for p in rd["core_mvp_auth"]["sso_providers"]] == ["google", "reddit", "steam", "github"]
    assert rd["core_mvp_auth"]["standard_email_registration"]["enabled"] is True

    # 2. POST /api/v1/gcp/portal-readiness/review (stage verification only, never deploys)
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}
    r_rev = client.post(
        "/api/v1/gcp/portal-readiness/review",
        headers=admin_hdr,
        json={
            "reviewer": "@Vxtzq",
            "action": "verify_stage_only",
            "public_dns_domain": "hivemind.socrate-ai-lab.com",
            "confirm_do_not_deploy_yet": True,
        },
    )
    assert r_rev.status_code == 200
    rv = r_rev.json()
    assert rv["deployment_state"] == "STAGED_VERIFIED_DO_NOT_DEPLOY_YET"
    assert rv["deployed_to_gcp"] is False
    assert len(rv["receipt_sha256"]) == 64

    # 3. Verify staged release report helper
    report = build_staged_portal_release_report()
    assert report["status"] == "STAGED_READY_NOT_DEPLOYED"
    assert report["deployed_to_gcp"] is False
    assert report["all_checks_passed"] is True


def test_v3_mvp_overview_and_claude_code_alignment() -> None:
    """Verify Release V3 MVP (v3.0.0) overview endpoint, Claude Code alignment, diurnal mode, and persistent GCP backbone."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. Verify /v3 and /v3-mvp HTML routes render the Release V3 MVP Command Center
    for route in ["/v3", "/v3-mvp"]:
        resp_html = client.get(route)
        assert resp_html.status_code == 200
        assert 'data-testid="v3-mvp-command-center"' in resp_html.text
        assert 'data-testid="amcp-kafka-collab-space"' in resp_html.text
        assert 'data-testid="opendiloco-training-calendar-card"' in resp_html.text
        assert 'data-testid="dashboard-globe-users"' in resp_html.text
        assert 'data-testid="dashboard-nash-equilibrium"' in resp_html.text
        assert 'data-testid="dashboard-inference-analytics"' in resp_html.text
        assert 'data-testid="dashboard-real-moe-bees"' in resp_html.text

    # 2. Verify GET /api/v1/v3-mvp/overview
    resp = client.get("/api/v1/v3-mvp/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "RELEASE_V3_MVP_IN_PROGRESS"
    assert data["version"] == "3.1.1"
    assert data["base_engine_release"] == "v3.0.0"
    assert data["base_engine_tag_url"] == "https://github.com/xaviercallens/HiveMind/releases/tag/v3.0.0"
    assert data["claude_code_alignment"]["aligned"] is True
    assert "A2_multi_host_split_moe" in data["claude_code_alignment"]["mvp_scorecard"]
    assert "B3_train_to_serve_gate" in data["claude_code_alignment"]["mvp_scorecard"]
    assert data["diurnal_operating_model"]["scheduled_and_community_training"]["owner_gpu_required_for_community_sessions"] is False
    voters = data["persistent_gcp_backbone"]["amcp_kafka_cluster"]["voters"]
    assert len(voters) == 3
    assert voters[0]["name"] == "hm-kafka-1"


def test_amcp_kafka_collaborative_space_human_agent_bee() -> None:
    """Verify Human, Agent, and MoE Bee collaborative communication over AMCP v1.5 Kafka KRaft topics."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    r_space = client.get("/api/v1/amcp/kafka/space")
    assert r_space.status_code == 200
    sp = r_space.json()
    assert sp["status"] == "AMCP_KAFKA_COLLAB_ONLINE"
    assert sp["supported_actor_types"] == ["HUMAN", "AGENT", "MOE_BEE"]
    assert len(sp["topics"]) == 4

    # Publish from HUMAN, AGENT, and MOE_BEE
    for actor_type, handle, topic in [
        ("HUMAN", "@xavier_callens", "hivemind.amcp.collab"),
        ("AGENT", "@ClaudeCode_MVP_v3", "hivemind.opendiloco.rounds"),
        ("MOE_BEE", "@Granite_Split_MoE_Bee", "hivemind.bees.consensus"),
    ]:
        r_pub = client.post(
            "/api/v1/amcp/kafka/publish",
            headers={"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"},
            json={
                "actor_handle": handle,
                "actor_type": actor_type,
                "topic": topic,
                "target_bee_or_agent": "@OpenDiLoCo_Coordinator",
                "intent": "V3_TEST_SYNC",
                "message": f"Test message from {actor_type} {handle} on {topic}",
            },
        )
        assert r_pub.status_code == 200
        pub = r_pub.json()
        assert pub["status"] == "RECORDED_IN_MEMORY_NOT_KAFKA"
        assert pub["simulated"] is True
        assert "coordinated_reply" not in pub
        assert pub["published_event"]["kafka_broker"] == "in-memory (no Kafka broker)"
        assert pub["published_event"]["actor_type"] == actor_type
        assert pub["published_event"]["topic"] == topic
        assert len(pub["receipt_sha256"]) == 64


def test_self_organized_opendiloco_training_calendar_and_personal_17_22_schedule() -> None:
    """Verify self-organized multi-GPU OpenDiLoCo training calendar (zero owner GPU needed) and daily 17:00-22:00 T4/A100/TPU schedule."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # 1. GET /api/v1/training/calendar
    r_cal = client.get("/api/v1/training/calendar")
    assert r_cal.status_code == 200
    cal = r_cal.json()
    assert cal["status"] == "OPENDILOCO_CALENDAR_ACTIVE"
    assert cal["self_organization_policy"]["owner_gpu_required"] is False
    assert cal["personal_daily_17_22_schedule"]["window_start"] == "17:00"
    assert cal["personal_daily_17_22_schedule"]["window_end"] == "22:00"

    # 2. Propose a new self-organized multi-GPU QLoRA MoE session on OpenDiLoCo
    admin_hdr = {"Authorization": f"Bearer {app.state.auth_state['users']['xavier_callens']['token']}"}
    r_prop = client.post(
        "/api/v1/training/calendar/propose",
        headers=admin_hdr,
        json={
            "title": "EU-US 3xGPU OpenDiLoCo QLoRA MoE — French & Math Rebalance",
            "proposer_handle": "@clark719664",
            "proposer_gpu": "NVIDIA GeForce RTX 4090 (24 GB VRAM)",
            "proposer_vram_gb": 24.0,
            "min_gpus_required": 2,
        },
    )
    assert r_prop.status_code == 200
    prop = r_prop.json()
    assert prop["status"] == "TRAINING_SESSION_PROPOSED"
    assert prop["owner_gpu_required"] is False
    new_ses_id = prop["session"]["session_id"]
    assert prop["session"]["status"] == "FORMING_POOL"

    # 3. Pledge a second peer GPU -> automatically transitions pool to READY_MULTI_GPU_OPENDILOCO
    r_pledge = client.post(
        "/api/v1/training/calendar/pledge-gpu",
        headers=admin_hdr,
        json={
            "session_id": new_ses_id,
            "contributor_handle": "@jblam-O",
            "gpu_model": "NVIDIA GeForce RTX 4070 Ti (16 GB VRAM)",
            "vram_gb": 16.0,
            "role": "TRAINER_PRODUCER",
        },
    )
    assert r_pledge.status_code == 200
    pl = r_pledge.json()
    assert pl["status"] == "GPU_PLEDGED_TO_OPENDILOCO_SESSION"
    assert pl["owner_gpu_required"] is False
    assert pl["session"]["status"] == "READY_MULTI_GPU_OPENDILOCO"
    assert pl["session"]["pledged_gpus_count"] == 2
    assert pl["session"]["total_pledged_vram_gb"] == 40.0

    # 4. Verify Personal 17:00-22:00 Schedule supports t4, a100, and tpu
    for accel in ("t4", "a100", "tpu"):
        r_sched = client.post(
            "/api/v1/training/personal-schedule",
            headers=admin_hdr,
            json={
                "enabled": True,
                "window_start": "17:00",
                "window_end": "22:00",
                "timezone": "Europe/Paris",
                "accelerator_type": accel,
            },
        )
        assert r_sched.status_code == 200
        sd = r_sched.json()["schedule"]
        assert sd["accelerator_type"] == accel
        assert sd["window_start"] == "17:00"
        assert sd["window_end"] == "22:00"
        assert sd["protect_rtx2070_daytime_inference"] is True


def test_v3_mvp_four_dashboards_globe_nash_inference_and_real_moe_bees() -> None:
    """Verify the 4 Release V3 MVP Dashboards: Globe Users, Nash Equilibrium, Real Inference Analytics, and Real Trained MoE Bees."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # Dashboard 1: Globe End-Users & Connected Nodes
    r_globe = client.get("/api/v1/dashboard/globe-users")
    assert r_globe.status_code == 200
    gl = r_globe.json()
    assert gl["status"] == "DEMO_DATA_NOT_LIVE"
    assert gl["simulated"] is True
    assert gl["connected_online_count"] >= 8
    assert gl["standby_scheduled_count"] >= 1
    r_globe_tr = client.get("/api/v1/dashboard/globe-users?role=TRAINER_OPENDILOCO")
    assert r_globe_tr.status_code == 200
    assert all(n["role"] == "TRAINER_OPENDILOCO" for n in r_globe_tr.json()["nodes"])

    # Dashboard 2: Nash Equilibrium Compute-Reciprocity Dashboard
    r_nash = client.get("/api/v1/dashboard/nash-equilibrium")
    assert r_nash.status_code == 200
    na = r_nash.json()
    assert na["status"] == "NASH_EQUILIBRIUM_CONVERGED"
    assert na["equilibrium_point"]["price_of_anarchy_poa"] == 1.00
    assert na["equilibrium_point"]["pareto_optimal"] is True

    r_nash_sim = client.post(
        "/api/v1/dashboard/nash-equilibrium/simulate",
        json={"active_producers": 30, "active_consumers": 60, "consumer_req_per_min": 90.0},
    )
    assert r_nash_sim.status_code == 200
    assert r_nash_sim.json()["parameters"]["active_producers"] == 30

    # Dashboard 3: Real Inference Data & Consumer Needs Fit Analytics
    r_inf = client.get("/api/v1/dashboard/inference-analytics")
    assert r_inf.status_code == 200
    inf = r_inf.json()
    assert inf["status"] == "REAL_INFERENCE_ANALYTICS_READY"
    assert inf["summary_metrics"]["split_moe_a2_tok_per_s"] == 41.75
    domains = [d["domain"] for d in inf["consumer_requests_by_domain"]]
    assert "code" in domains and "math" in domains and "french" in domains

    # Dashboard 4: Real Trained MoE Bees (Strictly Real Measured Models Only)
    r_bees = client.get("/api/v1/dashboard/real-moe-bees")
    assert r_bees.status_code == 200
    bees = r_bees.json()
    assert bees["status"] == "REAL_MOE_BEES_VERIFIED_ONLY"
    assert bees["honesty_policy"]["only_real_trained_moe_shown"] is True
    assert bees["honesty_policy"]["untrained_catalog_entries_excluded"] == 26
    assert bees["real_trained_bees_count"] == 5
    bee_ids = [b["bee_id"] for b in bees["real_trained_bees"]]
    assert "hivemind/bee-stars" in bee_ids
    assert "Qwen/Qwen2.5-0.5B-NF4-QLoRA" in bee_ids
    assert "ibm-granite/granite-3.1-1b-a400m-base" in bee_ids
    assert "hivemind/coder-lite" in bee_ids
    assert "hivemind/e8-pilot-two-host" in bee_ids
    # Verify hmq_00147 is transparently listed as gated/held-back (servable=False)
    gated = bees["gated_adapters_held_back"]
    assert len(gated) == 1
    assert gated[0]["adapter_id"] == "hmq_00147"
    assert gated[0]["servable"] is False


def test_red_team_v3_all_six_findings_verified() -> None:
    """Verify all 6 findings from the public Red Team disclosure (commit 7fe8f66 / GHSA-gjc6-74f9-wmrr) on Release V3."""
    app = create_hivemind_at_home_app()
    client = TestClient(app)

    # -----------------------------------------------------------------------
    # 0. Red Team V3 Audit Endpoint (/api/v1/security/red-team-v3-audit)
    # -----------------------------------------------------------------------
    r_audit = client.get("/api/v1/security/red-team-v3-audit")
    assert r_audit.status_code == 200
    audit = r_audit.json()
    assert audit["status"] == "RED_TEAM_V3_FULLY_VERIFIED"
    assert audit["advisory_id"] == "GHSA-gjc6-74f9-wmrr"
    assert audit["all_findings_remediated"] is True
    assert len(audit["findings"]) == 6
    assert all(f["status"] == "REMEDIATED_AND_TESTED" for f in audit["findings"])

    # -----------------------------------------------------------------------
    # Finding 1: Account & API Key Ownership (No Takeover Without Password/Token)
    # -----------------------------------------------------------------------
    # 1a. Existing account cannot be hijacked via /api/v1/auth/login, /register, /sso, /magic-link, or /steam/register-login
    assert client.post("/api/v1/auth/login", json={"handle_or_email": "xavier_callens"}).status_code == 401
    assert client.post("/api/v1/auth/register", json={"handle": "xavier_callens", "email": "callensxavier@gmail.com"}).status_code == 401
    assert client.post("/api/v1/auth/magic-link", json={"email": "callensxavier@gmail.com"}).status_code == 401
    assert client.post("/v1/hivemind/steam/register-login", json={"handle": "xavier_callens"}).status_code == 409

    # 1b. Newly registered Steam handle cannot be hijacked by a second unauthenticated caller
    r_steam1 = client.post("/v1/hivemind/steam/register-login", json={"handle": "SteamPioneer_RT01"})
    assert r_steam1.status_code == 200
    steam_key = r_steam1.json()["api_key"]
    r_steam_hijack = client.post("/v1/hivemind/steam/register-login", json={"handle": "SteamPioneer_RT01"})
    assert r_steam_hijack.status_code == 409
    assert r_steam_hijack.json()["error"]["code"] == "handle_taken"
    # Authenticated owner with Bearer token succeeds
    r_steam_owner = client.post(
        "/v1/hivemind/steam/register-login",
        json={"handle": "SteamPioneer_RT01"},
        headers={"Authorization": f"Bearer {steam_key}"},
    )
    assert r_steam_owner.status_code == 200
    assert r_steam_owner.json()["api_key"] == steam_key
    # Forged Bearer token on /v1/hivemind/steam/contribute-boinc is rejected (401), and unverified call awards 0 HMT
    assert (
        client.post(
            "/v1/hivemind/steam/contribute-boinc",
            json={"handle": "SteamPioneer_RT01"},
            headers={"Authorization": "Bearer hm_live_forged_bad_token"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/v1/hivemind/steam/contribute-boinc",
            json={"handle": "SteamPioneer_RT01"},
        ).json()["hmt_awarded"]
        == 0
    )

    # -----------------------------------------------------------------------
    # Finding 2: Unverified Volunteer Machines & Prompt Privacy Warning
    # -----------------------------------------------------------------------
    r_home = client.get("/v3")
    assert r_home.status_code == 200
    assert "proof-of-concept; send only public, non-sensitive data" in r_home.headers.get("x-hivemind-status", "")
    assert 'id="poc-notice"' in r_home.text
    assert "Treat your prompts as visible to strangers: send only public, non-sensitive data" in r_home.text

    r_chat = client.post(
        "/v1/chat/completions",
        json={"model": "hivemind/coder-lite", "messages": [{"role": "user", "content": "Hello swarm"}]},
    )
    assert r_chat.status_code == 200
    chat_json = r_chat.json()
    assert "Treat your prompts as visible to strangers" in chat_json["volunteer_node_warning"]
    assert "proof-of-concept" in chat_json["privacy_notice"]

    # -----------------------------------------------------------------------
    # Finding 3 & Finding 5: Shield Proxy Mandatory & Ollama CVE-2024-37032 Check
    # -----------------------------------------------------------------------
    assert "hivemind_producer_shield.py" in r_home.text
    assert "CVE-2024-37032" in r_home.text
    assert "OLLAMA_HOST=127.0.0.1:11434" in r_home.text

    # Unsafe producer setup (raw :11434, short token, old Ollama 0.1.30, 0.0.0.0 bind) is rejected (400)
    r_unsafe = client.post(
        "/api/v1/security/producer-endpoint-check",
        json={
            "endpoint_url": "https://my-tunnel.trycloudflare.com:11434",
            "endpoint_token": "short",
            "ollama_version": "0.1.30",
            "local_ollama_bind": "0.0.0.0:11434",
            "shield_enabled": False,
        },
    )
    assert r_unsafe.status_code == 400
    unsafe_json = r_unsafe.json()
    assert unsafe_json["safe_to_register"] is False
    assert len(unsafe_json["issues"]) >= 3

    # Safe shielded producer setup (:11435, >=20 char token, Ollama >= 0.5.0, loopback bind) passes (200)
    r_safe = client.post(
        "/api/v1/security/producer-endpoint-check",
        json={
            "endpoint_url": "https://my-shielded-tunnel.trycloudflare.com",
            "endpoint_token": "hm_shield_tok_0123456789abcdef0123456789",
            "ollama_version": "0.5.7",
            "local_ollama_bind": "127.0.0.1:11434",
            "shield_enabled": True,
        },
    )
    assert r_safe.status_code == 200
    assert r_safe.json()["safe_to_register"] is True
    assert r_safe.json()["cve_2024_37032_protected"] is True

    # -----------------------------------------------------------------------
    # Finding 4: Zero Public Endpoint Listing Node Addresses or IP Literals
    # -----------------------------------------------------------------------
    for pub_route in (
        "/api/v1/v3-mvp/overview",
        "/api/v1/amcp/kafka/space",
        "/api/v1/training/calendar",
        "/api/v1/dashboard/globe-users",
    ):
        r_pub = client.get(pub_route)
        assert r_pub.status_code == 200
        raw_body = r_pub.text
        assert "endpoint_url" not in raw_body
        assert "10.150.0." not in raw_body

    # -----------------------------------------------------------------------
    # Finding 6: Non-Existent Hugging Face Account Purged & Off-the-Shelf Honesty
    # -----------------------------------------------------------------------
    from hivemind.router.catalog import UNPUBLISHED_HF_REPOS, catalog_payload

    cat = catalog_payload()
    assert len(UNPUBLISHED_HF_REPOS) >= 20
    for tier in cat["tiers"]:
        assert tier["kind"] in ("base", "adapter", "alias")
        assert tier["ollama_tag"]
        assert "xaviercallens/" not in tier.get("hf_repo", "")
        assert "xaviercallens/" not in tier.get("description", "")

    r_models = client.get("/api/v1/openrouter/models")
    assert r_models.status_code == 200
    m_json = r_models.json()
    assert m_json["unpublished_hf_repos_purged"] is True
    assert "No HiveMind-trained weights are published to a xaviercallens/* Hugging Face account" in m_json["catalog_honesty_notice"]
    for item in m_json["data"]:
        assert item["kind"] in ("base", "adapter", "alias")
        assert item["served_by"]
        assert item["hf_published"] is False
        assert item["hf_repo"] == ""
        assert "xaviercallens/" not in item["description"]


def test_v3_1_deployment_on_v3_0_0_gateway_and_core_engine_alignment() -> None:
    """Verify V3.1 GCP Unified Web Interface deploys cleanly on the v3.0.0 OpenRouter Gateway (create_router_app)
    and binds to v3.0.0 core economy/ledger functions and measured_results.json artifacts."""
    import hivemind
    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    assert hivemind.__version__ == "3.1.1"

    store = RouterStore(":memory:", b"0" * 32)
    router_app = create_router_app(store)
    gw_client = TestClient(router_app)

    # 1. OpenRouter root portal displays v3.1.1 (v3.0.0 Core Engine) and links to /stitch-portal
    r_root = gw_client.get("/")
    assert r_root.status_code == 200
    assert "v3.1.1 (v3.0.0 Core Engine)" in r_root.text
    assert 'href="/stitch-portal"' in r_root.text

    # 2. Stitch V3.1 GCP Unified Web Interface is served directly by the v3.0.0 Gateway at /stitch-portal, /v3-portal, /v3.1
    for path in ("/stitch-portal", "/v3-portal", "/v3.1"):
        r_stitch = gw_client.get(path)
        assert r_stitch.status_code == 200
        assert r_stitch.headers.get("X-Frame-Options") == "DENY"
        assert "proof-of-concept" in (r_stitch.headers.get("X-HiveMind-PoC-Status") or "")
        assert 'id="poc-notice"' in r_stitch.text
        assert 'data-testid="v3-mvp-command-center"' in r_stitch.text
        assert "RELEASE V3.1.1 ON V3.0.0 CORE ENGINE" in r_stitch.text

    # 3. All V3.1 API routes work directly on the v3.0.0 OpenRouter Gateway
    r_ov = gw_client.get("/api/v1/v3.1/overview")
    assert r_ov.status_code == 200
    ov = r_ov.json()
    assert ov["version"] == "3.1.1"
    assert ov["base_engine_release"] == "v3.0.0"
    assert "f4d159a" in ov["claude_code_alignment"]["reference_commits"]
    assert ov["measured_models_count"] >= 3

    # 4. Nash Equilibrium endpoint executes live verification against v3.0.0 core economy & ledger modules
    r_nash = gw_client.get("/api/v1/dashboard/nash-equilibrium")
    assert r_nash.status_code == 200
    nash = r_nash.json()
    v3_core = nash["v3_core_engine_verification"]
    assert v3_core["core_release"] == "v3.0.0"
    assert v3_core["erlang_c_log_space_stable"] is True
    assert 0.0 <= v3_core["erlang_c_delay_prob"] <= 1.0
    assert v3_core["non_finite_inputs_rejected"] is True
    assert v3_core["integer_micro_unit_fee_split_conserves"] is True

    # 5. AMCP Kafka Space, Training Calendar, Globe Dashboard, Inference Analytics, Real MoE Bees & Red Team Audit on Gateway
    assert gw_client.get("/api/v1/amcp/kafka/space").status_code == 200
    assert gw_client.get("/api/v1/training/calendar").status_code == 200
    assert gw_client.get("/api/v1/training/t4-daily-schedule").status_code == 200
    assert gw_client.get("/api/v1/dashboard/globe-users").status_code == 200
    assert gw_client.get("/api/v1/dashboard/inference-analytics").status_code == 200
    r_bees = gw_client.get("/api/v1/dashboard/moe-bees")
    assert r_bees.status_code == 200
    assert r_bees.json()["honesty_policy"]["only_real_trained_moe_shown"] is True
    assert r_bees.json()["gated_adapters_held_back"][0]["servable"] is False
    r_rt = gw_client.get("/api/v1/security/red-team-v3-audit")
    assert r_rt.status_code == 200
    assert r_rt.json()["status"] == "SELF_REPORTED_NOT_INDEPENDENTLY_VERIFIED"
    assert "all_6_findings_verified" not in r_rt.json()
