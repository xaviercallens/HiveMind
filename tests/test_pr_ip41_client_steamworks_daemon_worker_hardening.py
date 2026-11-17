"""Negative-control tests for Iteration 41: client Steamworks, ZeroIntrusionDaemon, and HiveMindWorkerNode hardening."""

from __future__ import annotations

import hashlib
import hmac
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hivemind.client import daemon as daemon_mod
from hivemind.client.daemon import SETIAgentState, ZeroIntrusionDaemon
from hivemind.client.steamworks import (
    STEAMWORKS_APP_ID,
    SteamAuthSessionTicket,
    SteamworksSDKBridge,
)
from hivemind.client.worker_node import HiveMindWorkerNode


def test_steamworks_ticket_and_sybil_guard_reject_non_finite_or_negative_spend() -> None:
    """SteamAuthSessionTicket and SteamworksSDKBridge must reject NaN/Inf/negative account_spend_usd."""
    bridge = SteamworksSDKBridge(signing_secret=b"s" * 32)

    for bad_spend in (float("nan"), float("inf"), float("-inf"), -5.0):
        with pytest.raises(ValueError, match="account_spend_usd"):
            bridge.get_auth_session_ticket("76561198000000101", account_spend_usd=bad_spend)

    with pytest.raises(ValueError, match="steam_id64"):
        bridge.get_auth_session_ticket("   ", account_spend_usd=25.0)

    # Even if an attacker crafts a ticket with spend=inf and a matching HMAC signature, verify_sybil_resistance rejects it
    inf_payload = f"76561198000000101:{STEAMWORKS_APP_ID}:{float('inf'):.2f}:0"
    inf_sig = hmac.new(b"s" * 32, inf_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    forged_inf_ticket = SteamAuthSessionTicket(
        steam_id64="76561198000000101",
        app_id=STEAMWORKS_APP_ID,
        account_spend_usd=float("inf"),
        is_vac_banned=False,
        ticket_hex=inf_sig,
    )
    assert forged_inf_ticket.is_non_limited is False
    ok, reason = bridge.verify_sybil_resistance(forged_inf_ticket)
    assert ok is False
    assert "non-finite" in reason.lower() or "invalid" in reason.lower()


def test_steamworks_record_validated_workunit_deduplicates_and_rejects_non_finite() -> None:
    """record_validated_workunit must deduplicate (steam_id64, wu_id) and reject NaN/Inf/negative inputs."""
    bridge = SteamworksSDKBridge(signing_secret=b"k" * 32)

    e1 = bridge.record_validated_workunit(
        steam_id64="76561198000000101",
        persona_name="MinerOne",
        wu_id="wu-unique-001",
        tflops_contributed=6.0,
        credits_minted=12.5,
    )
    assert e1.workunits_completed == 1
    assert e1.verified_tflops == 6.0
    assert e1.api_credits_balance == 12.5
    assert len(e1.credit_receipts) == 1

    # Replaying the exact same workunit ID for the same SteamID64 must be a no-op
    e1_replay = bridge.record_validated_workunit(
        steam_id64="76561198000000101",
        persona_name="MinerOne",
        wu_id="wu-unique-001",
        tflops_contributed=6.0,
        credits_minted=12.5,
    )
    assert e1_replay.workunits_completed == 1
    assert e1_replay.verified_tflops == 6.0
    assert e1_replay.api_credits_balance == 12.5
    assert len(e1_replay.credit_receipts) == 1

    for bad_val in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            bridge.record_validated_workunit(
                steam_id64="76561198000000101",
                persona_name="MinerOne",
                wu_id="wu-bad-tflops",
                tflops_contributed=bad_val,
                credits_minted=10.0,
            )
        with pytest.raises(ValueError):
            bridge.record_validated_workunit(
                steam_id64="76561198000000101",
                persona_name="MinerOne",
                wu_id="wu-bad-credits",
                tflops_contributed=5.0,
                credits_minted=bad_val,
            )


def test_steamworks_stats_leaderboard_and_rich_presence_reject_non_finite_or_negative() -> None:
    """update_user_stats, upload_hack_miner_score, and set_rich_presence must reject NaN/Inf/negative values."""
    bridge = SteamworksSDKBridge(signing_secret=b"m" * 32)

    for bad_float in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError):
            bridge.update_user_stats(
                steam_id64="76561198000000101",
                persona_name="Player1",
                best_loss_drop_pct=bad_float,
            )
        with pytest.raises(ValueError):
            bridge.update_user_stats(
                steam_id64="76561198000000101",
                persona_name="Player1",
                hive_mined=bad_float,
            )
        with pytest.raises(ValueError):
            bridge.upload_hack_miner_score(
                steam_id64="76561198000000101",
                persona_name="Player1",
                total_points=100,
                swarm_contribution_points=50,
                hive_balance=bad_float,
                achievements=[],
            )
        with pytest.raises(ValueError):
            bridge.set_rich_presence(
                steam_id64="76561198000000101",
                persona_name="Player1",
                hive_balance=bad_float,
            )

    with pytest.raises(ValueError):
        bridge.update_user_stats("76561198000000101", "Player1", shards_mined=-1)
    with pytest.raises(ValueError):
        bridge.upload_hack_miner_score("76561198000000101", "Player1", -10, 0, 1.0, [])


def test_zero_intrusion_daemon_fails_closed_on_nan_or_negative_idle_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZeroIntrusionDaemon must validate thresholds and fail closed (yield VRAM) if OS idle reading is NaN/negative."""
    for bad_thresh in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="idle_threshold_seconds"):
            ZeroIntrusionDaemon(idle_threshold_seconds=bad_thresh)

    for bad_sim in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="simulated_idle_seconds"):
            ZeroIntrusionDaemon(simulated_idle_seconds=bad_sim)

    daemon = ZeroIntrusionDaemon(idle_threshold_seconds=30.0, enforce_os_idle_hook=True)
    monkeypatch.setattr(daemon_mod, "detect_running_game_process", lambda *a, **kw: None)
    monkeypatch.setattr(daemon_mod, "get_windows_idle_seconds", lambda: float("nan"))

    tel = daemon.poll()
    assert tel.should_yield_vram is True
    assert tel.seti_agent_state == SETIAgentState.STATE_1_GAMING_SEED_ONLY.value


def test_worker_node_validates_init_and_ef21_telemetry_finiteness() -> None:
    """HiveMindWorkerNode must reject non-finite constructor knobs and non-finite EF21 telemetry norms."""
    tiny_model = nn.Linear(4, 4)

    for bad_val in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="account_spend_usd"):
            HiveMindWorkerNode("w1", "76561198000000101", "Node1", tiny_model, account_spend_usd=bad_val)

    for bad_decay in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="error_feedback_stale_decay"):
            HiveMindWorkerNode(
                "w1",
                "76561198000000101",
                "Node1",
                tiny_model,
                error_feedback_stale_decay=bad_decay,
            )

    node = HiveMindWorkerNode(
        "w1",
        "76561198000000101",
        "Node1",
        tiny_model,
        enable_error_feedback=True,
    )
    node._error_feedback_for("Code", topk_ratio=0.2)
    nan_result = SimpleNamespace(
        branch_name="Code",
        outer_step=1,
        raw_delta={"w": torch.tensor([1.0, 2.0])},
        error_feedback_residual_norm=float("nan"),
    )
    with pytest.raises(ValueError, match="EF21 telemetry norms must be finite"):
        node._record_ef21_telemetry(nan_result)  # type: ignore[arg-type]
    assert node.ef21_telemetry() == []
