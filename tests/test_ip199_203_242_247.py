# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and verification tests for IP-199, IP-203, IP-242, and IP-247."""

from __future__ import annotations

from pathlib import Path
from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
import torch

from hivemind.boinc.hivemind_at_home_dashboard import (
    ComputePointsRedeemRequest,
    create_hivemind_at_home_app,
    redeem_compute_points_for_task,
)
from hivemind.game import hf_hack_miner as hf_miner_mod
from hivemind.game.hf_hack_miner import (
    ALLOWED_HF_SCAN_DATASETS,
    execute_interactive_hf_shard_scan,
    execute_interactive_lora_micro_forge,
)
from hivemind.game.rtx_steam_tool_deployer import run_intense_rtx_stress_suite
from hivemind.game.steam_edition import create_hack_miner_fastapi_app


def test_ip199_scan_hf_shard_allowlist_and_offline_block_zero_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-199: unallowed dataset or offline fetch failure returns BLOCKED, 0 points, no write."""
    monkeypatch.setattr(hf_miner_mod, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(hf_miner_mod, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    res_unallowed = execute_interactive_hf_shard_scan(
        gamer_tag="CyberValkyrie_FR",
        dataset_id="nonexistent/repo",
    )
    assert res_unallowed["status"] == "BLOCKED"
    assert res_unallowed["data_discovery_pts_awarded"] == 0
    assert res_unallowed["merkle_pos_verified"] is False
    assert not (tmp_path / "cache" / "swarm_store").exists()

    # Allowed dataset that is not in the offline curated corpus and has no local cache when HF_HUB_OFFLINE=1
    assert "sahil2801/CodeAlpaca-20k" in ALLOWED_HF_SCAN_DATASETS
    res_offline_uncached = execute_interactive_hf_shard_scan(
        gamer_tag="CyberValkyrie_FR",
        dataset_id="sahil2801/CodeAlpaca-20k",
    )
    assert res_offline_uncached["status"] == "BLOCKED"
    assert res_offline_uncached["data_discovery_pts_awarded"] == 0
    assert res_offline_uncached["merkle_pos_verified"] is False

    # FastAPI endpoint returns 400 on BLOCKED
    app = create_hack_miner_fastapi_app()
    client = TestClient(app)
    resp_bad = client.post(
        "/api/scan-hf-shard",
        json={"gamer_tag": "CyberValkyrie_FR", "dataset_id": "nonexistent/repo"},
    )
    assert resp_bad.status_code == 400
    assert resp_bad.json()["status"] == "BLOCKED"
    assert resp_bad.json()["data_discovery_pts_awarded"] == 0

    # No PYTEST_CURRENT_TEST fork in hf_hack_miner.py
    src_text = Path(hf_miner_mod.__file__).read_text(encoding="utf-8")
    assert "PYTEST_CURRENT_TEST" not in src_text


def test_ip203_stage9_fails_when_spotcheck_or_cas_roundtrip_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IP-203: forge returning spot_check_passed=False or corrupted CAS tensor fails Stage 9 and all_stages_passed."""
    bad_forge = execute_interactive_lora_micro_forge(
        gamer_tag="CyberValkyrie_FR",
        contract_id="CONTRACT-HF-07-LEAN4-FORMAL",
        worker_delta_override={
            "layers.0.lora_A": torch.ones((4, 16), dtype=torch.float32) * 99.0,
            "layers.0.lora_B": torch.ones((16, 4), dtype=torch.float32) * 99.0,
        },
    )
    assert bad_forge["spot_check_passed"] is False
    assert bad_forge["total_pts_awarded"] == 0
    assert bad_forge["hive_minted"] == 0.0

    orig_forge = hf_miner_mod.execute_interactive_lora_micro_forge
    monkeypatch.setattr(
        hf_miner_mod,
        "execute_interactive_lora_micro_forge",
        lambda gamer_tag="CyberValkyrie_FR", contract_id="CONTRACT-HF-04-RUST-SYSTEMS": orig_forge(
            gamer_tag=gamer_tag,
            contract_id=contract_id,
            spot_check_passed_override=False,
        ),
    )
    report = run_intense_rtx_stress_suite(
        vram_cycles=1,
        udp_burst_packets=10,
        num_hotplug_experts=4,
        steam_concurrent_accounts=10,
    )
    assert report["stage9_season3_hack_miner_and_16_expert_cas"]["season3_all_spotchecks_passed"] is False
    assert report["all_stages_passed"] is False


def test_ip242_redeem_uses_ledger_balance_and_prompt_dependent_moe_output(
    tmp_path: Path,
) -> None:
    """IP-242: unknown handle with points_balance_before=999999 is refused; two prompts yield distinct domain_output."""
    with pytest.raises(HTTPException) as exc_info:
        redeem_compute_points_for_task(
            ComputePointsRedeemRequest(
                contributor_handle="@unknown-handle-xyz",
                service_id="python_coding_gwaya",
                prompt="Prompt A",
                points_balance_before=999999,
            )
        )
    assert exc_info.value.status_code == 422

    ledger = {"@alice_tester": 200}
    res_a = redeem_compute_points_for_task(
        ComputePointsRedeemRequest(
            contributor_handle="@alice_tester",
            service_id="astrophysics_exoplanet",
            prompt="Analyze Kepler-186f habitable zone transit period.",
            points_balance_before=999999,
        ),
        ledger=ledger,
    )
    res_b = redeem_compute_points_for_task(
        ComputePointsRedeemRequest(
            contributor_handle="@alice_tester",
            service_id="astrophysics_exoplanet",
            prompt="Compute relativistic periastron precession for binary pulsar.",
            points_balance_before=999999,
        ),
        ledger=ledger,
    )
    assert res_a["points_balance_before"] == 200
    assert res_a["points_balance_after"] == 150
    assert res_b["points_balance_before"] == 150
    assert res_b["points_balance_after"] == 100
    assert res_a["domain_output"] != res_b["domain_output"]
    assert "0.000422" not in res_a["domain_output"]
    assert "SNR 4.30" not in res_a["domain_output"]


def test_ip247_son_feedback_bounded_deduped_and_no_bonus_points(
    tmp_path: Path,
) -> None:
    """IP-247: 10,000 feedback posts stay within the 200-entry cap, dedupe per handle, and claim no bonus_points=500."""
    app = create_hivemind_at_home_app(vault_dir=tmp_path / "vault")
    client = TestClient(app)

    r1 = client.post(
        "/api/v1/son/feedback",
        json={"gamer_tag": "PlayerOne", "overall_rating": 4, "review_text": "First review"},
    )
    assert r1.status_code == 200
    j1 = r1.json()
    assert "bonus_points" not in j1
    assert j1["bonus_points_status"] == "not_credited"
    assert j1["hmt_awarded"] == 0

    # Duplicate handle updates in-place instead of growing the store
    r2 = client.post(
        "/api/v1/son/feedback",
        json={"gamer_tag": "PlayerOne", "overall_rating": 5, "review_text": "Updated review"},
    )
    assert r2.status_code == 200
    assert len(app.state.son_feedback_store) == 1
    assert app.state.son_feedback_store["playerone"]["review_text"] == "Updated review"

    # 10,000 posts stay bounded within the 200-entry cap
    for i in range(10_000):
        resp = client.post(
            "/api/v1/son/feedback",
            json={"gamer_tag": f"gamer_{i % 500}", "overall_rating": 5, "review_text": "ok"},
        )
        assert resp.status_code == 200
    assert len(app.state.son_feedback_store) <= 200
