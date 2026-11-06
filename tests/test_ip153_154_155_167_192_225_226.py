"""Regression tests for IP-153, IP-154, IP-155, IP-167, IP-192, IP-225, and IP-226."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import hivemind.game.exoplanet_hunter_cockpit as cockpit_mod
import hivemind.game.hf_hack_miner as hf_miner_mod
import hivemind.game.rtx_steam_tool_deployer as rtx_deployer_mod
from hivemind.game.exoplanet_hunter_cockpit import create_exoplanet_hunter_fastapi_app
from hivemind.game.hf_hack_miner import (
    DEFAULT_GUILD_TOURNAMENT_SQUADS,
    execute_interactive_lora_micro_forge,
)
from hivemind.game.steam_edition import create_hack_miner_fastapi_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ip153_guild_tournament_unauthenticated_post_cannot_override_champion() -> None:
    """IP-153: Unauthenticated POST /api/guild-tournament cannot alter champion_guild_id; no external GitHub handles."""
    all_contributors = [
        handle
        for squad in DEFAULT_GUILD_TOURNAMENT_SQUADS
        for handle in squad.member_contributors
    ]
    assert "@pedromineo" not in all_contributors
    assert "@clark719664" not in all_contributors

    client = TestClient(create_hack_miner_fastapi_app(worker_token="worker-secret-token-32b"))
    before = client.get("/api/guild-tournament").json()
    initial_champion = before["champion_guild_id"]
    assert initial_champion == "GUILD-02"

    r_unauth = client.post(
        "/api/guild-tournament",
        json={
            "guild_id": "GUILD-03",
            "multi_domain_holdout_perplexity": 0.001,
            "speculative_tokens_per_sec": 500.0,
        },
    )
    assert r_unauth.status_code == 200
    after = r_unauth.json()
    assert after["champion_guild_id"] == initial_champion
    g3 = next(g for g in after["guilds"] if g["guild_id"] == "GUILD-03")
    assert g3["squad_report_verified"] is False
    assert g3["self_report_status"] == "UNVERIFIED_IGNORED_FOR_RANKING"
    assert g3["multi_domain_holdout_perplexity"] > 1.0

    # Authenticated worker update is honored.
    r_auth = client.post(
        "/api/guild-tournament",
        headers={"Authorization": "Bearer worker-secret-token-32b"},
        json={
            "guild_id": "GUILD-03",
            "multi_domain_holdout_perplexity": 4.25,
            "speculative_tokens_per_sec": 31.5,
        },
    )
    assert r_auth.status_code == 200
    assert r_auth.json()["champion_guild_id"] == "GUILD-03"


def test_ip154_forge_lora_blocks_empty_corpus_and_no_cuda_and_unauth_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """IP-154: Empty corpus or missing CUDA sets spot_check_passed=False and hive_minted=0.0; unauth calls do not mutate leaderboard."""
    empty_res = execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-04-RUST-SYSTEMS",
        corpus_override=[],
    )
    assert empty_res["spot_check_passed"] is False
    assert empty_res["status"] == "BLOCKED_EMPTY_CORPUS"
    assert empty_res["hive_minted"] == 0.0
    assert empty_res["total_pts_awarded"] == 0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    no_cuda_res = execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-04-RUST-SYSTEMS",
        require_cuda=True,
    )
    assert no_cuda_res["spot_check_passed"] is False
    assert no_cuda_res["status"] == "BLOCKED_NO_CUDA"
    assert no_cuda_res["hive_minted"] == 0.0

    # HTTP route /api/forge-lora-mission and /api/solve-bounty do not mutate leaderboard when unauthenticated
    monkeypatch.setattr(hf_miner_mod, "ARTIFACTS_ROOT", tmp_path)
    report_path = tmp_path / "hf_hack_miner_game_report.json"
    initial_payload = {
        "leaderboard": [{"gamer_tag": "CyberValkyrie_FR", "total_points": 100, "hive_balance": 10.0}],
    }
    report_path.write_text(json.dumps(initial_payload), encoding="utf-8")

    client = TestClient(create_hack_miner_fastapi_app(worker_token="worker-secret-token-32b"))
    r_http_forge = client.post(
        "/api/forge-lora-mission",
        json={"gamer_tag": "CyberValkyrie_FR", "contract_id": "CONTRACT-HF-04-RUST-SYSTEMS"},
    )
    assert r_http_forge.status_code == 200
    assert r_http_forge.json()["spot_check_passed"] is False
    assert r_http_forge.json()["hive_minted"] == 0.0
    assert r_http_forge.json()["leaderboard_persisted"] is False

    r_http_bounty = client.post(
        "/api/solve-bounty",
        json={"gamer_tag": "CyberValkyrie_FR", "prompt": "def compute_hive(): return 1"},
    )
    assert r_http_bounty.status_code == 200
    assert r_http_bounty.json()["leaderboard_persisted"] is False
    assert r_http_bounty.json()["scoring_basis"] == "keyword_heuristic_demo"
    assert json.loads(report_path.read_text(encoding="utf-8")) == initial_payload


def test_ip155_scientific_game_solve_caller_spend_not_steam_sybil_verified() -> None:
    """IP-155: POST /api/scientific-game/solve with caller-supplied spend has steam_sybil_verified=False."""
    client = TestClient(create_hack_miner_fastapi_app())
    r_solve = client.post(
        "/api/scientific-game/solve",
        json={
            "mission_id": "MISSION-SCI-01-SYMPLECTIC-EXOPLANET",
            "steam_id64": "1",
            "account_spend_usd": 1e9,
            "is_vac_banned": False,
        },
    )
    assert r_solve.status_code == 200
    body = r_solve.json()
    assert body["steam_sybil_verified"] is False
    assert body["sybil_verification_source"] == "caller_supplied_unverified"
    assert body["local_sybil_rule_passed"] is True

    r_cohort = client.post("/api/scientific-game/playtest-cohort")
    assert r_cohort.status_code == 200
    cohort = r_cohort.json()
    assert cohort["cohort_basis"] == "scripted_local_rule_simulation"
    assert cohort["honest_players_verified"] == 0
    assert cohort["scripted_honest_profiles_passed_local_rule"] == 4


def test_ip167_no_committed_hmac_secrets_and_portal_secret_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """IP-167: No committed HMAC literals in exoplanet_hunter_cockpit.py or rtx_steam_tool_deployer.py."""
    cockpit_src = Path(cockpit_mod.__file__).read_text(encoding="utf-8")
    deployer_src = Path(rtx_deployer_mod.__file__).read_text(encoding="utf-8")
    assert "hivemind-local-steam-portal-secret-32b" not in cockpit_src
    assert "HiveMind-Steamworks-RTX-Deploy-Key" not in deployer_src

    monkeypatch.setattr(cockpit_mod, "ARTIFACTS_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="HIVEMIND_PORTAL_SECRET"):
        create_exoplanet_hunter_fastapi_app(require_portal_secret=True)

    monkeypatch.setenv("HIVEMIND_PORTAL_SECRET", "short")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        create_exoplanet_hunter_fastapi_app()

    monkeypatch.setenv("HIVEMIND_PORTAL_SECRET", "p" * 36)
    app = create_exoplanet_hunter_fastapi_app(require_portal_secret=True)
    assert app is not None


def test_pr_security_scanner_resolves_head_sha_author_email_for_trusted_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRSecurityScanner resolves commit author email from HEAD_SHA in GitHub Actions while keeping None unverified otherwise."""
    import subprocess
    from hivemind.security.pr_security_scanner import PRSecurityScanner

    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    scanner = PRSecurityScanner()

    # Without HEAD_SHA and with author_email=None, workflow modification is blocked by CB-04.
    monkeypatch.delenv("HEAD_SHA", raising=False)
    res_no_env = scanner.audit_pull_request(
        pr_number=678,
        pr_title="ci: update workflow",
        pr_body="Update workflow",
        author="xaviercallens",
        author_email=None,
        changed_files=[
            {
                "filename": ".github/workflows/ci_swarm_tests.yml",
                "status": "modified",
                "additions": 2,
                "patch": "@@ -1,1 +1,2 @@\n+cancel-in-progress: false\n",
            }
        ],
    )
    assert any(f.rule_id == "CB-04" for f in res_no_env.findings)

    # With HEAD_SHA pointing to an owner-authored commit, CB-04 is not triggered.
    monkeypatch.setenv("HEAD_SHA", head_sha)
    res_with_sha = scanner.audit_pull_request(
        pr_number=678,
        pr_title="ci: update workflow",
        pr_body="Update workflow",
        author="xaviercallens",
        author_email=None,
        changed_files=[
            {
                "filename": ".github/workflows/ci_swarm_tests.yml",
                "status": "modified",
                "additions": 2,
                "patch": "@@ -1,1 +1,2 @@\n+cancel-in-progress: false\n",
            }
        ],
    )
    assert not any(f.rule_id == "CB-04" for f in res_with_sha.findings)

