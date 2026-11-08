# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control regression tests for client daemon, Steam identity, Steamworks bridge, and WorkerNode hardening."""

from __future__ import annotations

import pytest
import torch.nn as nn

from hivemind.client.daemon import ZeroIntrusionDaemon
from hivemind.client.steam_identity import (
    IdentityTokens,
    SteamIdentityGate,
    SteamOpenIDVerifier,
    login_url,
)
from hivemind.client.steamworks import SteamworksSDKBridge
from hivemind.client.worker_node import HiveMindWorkerNode


def test_zero_intrusion_daemon_rejects_boolean_and_blank_inputs() -> None:
    d = ZeroIntrusionDaemon(idle_threshold_seconds=15.0, simulated_idle_seconds=20.0)
    assert d.should_preempt() is False

    with pytest.raises(ValueError, match="idle_threshold_seconds"):
        ZeroIntrusionDaemon(idle_threshold_seconds=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="enforce_os_idle_hook"):
        ZeroIntrusionDaemon(enforce_os_idle_hook=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="consumer_query_active"):
        ZeroIntrusionDaemon(consumer_query_active=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="simulated_idle_seconds"):
        ZeroIntrusionDaemon(simulated_idle_seconds=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="simulated_active_game"):
        ZeroIntrusionDaemon(simulated_active_game="   ")
    with pytest.raises(ValueError, match="consumer_query_active"):
        d.resolve_seti_agent_state(consumer_query_active=1)  # type: ignore[arg-type]


def test_steam_identity_rejects_boolean_and_invalid_inputs() -> None:
    url = login_url("https://example.com/cb", "https://example.com")
    assert "steamcommunity.com" in url
    with pytest.raises(ValueError, match="return_to"):
        login_url("   ", "https://example.com")
    with pytest.raises(ValueError, match="return_to"):
        login_url("https://example.com/cb", "   ")

    with pytest.raises(ValueError, match="nonce_ttl_s"):
        SteamOpenIDVerifier("https://example.com/cb", nonce_ttl_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_nonces"):
        SteamOpenIDVerifier("https://example.com/cb", max_nonces=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_attempts_per_ip"):
        SteamOpenIDVerifier("https://example.com/cb", max_attempts_per_ip=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="identity secret"):
        IdentityTokens(secret="not-bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ttl_s"):
        IdentityTokens(ttl_s=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="allow_limited"):
        SteamIdentityGate("https://example.com/cb", allow_limited=1)  # type: ignore[arg-type]


def test_steamworks_sdk_bridge_rejects_boolean_and_blank_inputs() -> None:
    bridge = SteamworksSDKBridge()
    sid = "76561198000000001"

    with pytest.raises(ValueError, match="steam_id64"):
        bridge.set_rich_presence("   ")
    with pytest.raises(ValueError, match="hive_balance"):
        bridge.set_rich_presence(sid, hive_balance=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="total_points"):
        bridge.set_rich_presence(sid, total_points=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="steam_id64"):
        bridge.update_user_stats("   ", "p")
    with pytest.raises(ValueError, match="best_loss_drop_pct"):
        bridge.update_user_stats(sid, "p", best_loss_drop_pct=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Steam user stats"):
        bridge.update_user_stats(sid, "p", shards_mined=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="account_spend_usd"):
        bridge.get_auth_session_ticket(sid, account_spend_usd=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is_vac_banned"):
        bridge.get_auth_session_ticket(sid, is_vac_banned=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tflops_contributed"):
        bridge.record_validated_workunit(sid, "p", "wu-1", tflops_contributed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tflops_contributed"):
        bridge.record_validated_workunit(sid, "p", "wu-1", tflops_contributed=1.0, credits_minted=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="leaderboard score"):
        bridge.upload_hack_miner_score(sid, "p", total_points=True, swarm_contribution_points=1, hive_balance=1.0, achievements=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="leaderboard score"):
        bridge.upload_hack_miner_score(sid, "p", total_points=1, swarm_contribution_points=1, hive_balance=True, achievements=[])  # type: ignore[arg-type]


def test_worker_node_rejects_boolean_and_blank_inputs() -> None:
    model = nn.Linear(4, 4)
    sid = "76561198000000001"
    with pytest.raises(ValueError, match="worker_id"):
        HiveMindWorkerNode("   ", sid, "p", model)
    with pytest.raises(ValueError, match="steam_id64"):
        HiveMindWorkerNode("w1", "   ", "p", model)
    with pytest.raises(ValueError, match="persona_name"):
        HiveMindWorkerNode("w1", sid, "   ", model)
    with pytest.raises(ValueError, match="enable_error_feedback"):
        HiveMindWorkerNode("w1", sid, "p", model, enable_error_feedback=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="account_spend_usd"):
        HiveMindWorkerNode("w1", sid, "p", model, account_spend_usd=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_feedback_stale_decay"):
        HiveMindWorkerNode("w1", sid, "p", model, error_feedback_stale_decay=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_feedback_max_residual_norm"):
        HiveMindWorkerNode("w1", sid, "p", model, error_feedback_max_residual_norm=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lr"):
        HiveMindWorkerNode._validate_inner_hyperparams(
            lr=True, weight_decay=0.01, micro_batch_size=1, topk_ratio=0.5, muon_ns_steps=5  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="micro_batch_size"):
        HiveMindWorkerNode._validate_inner_hyperparams(
            lr=1e-4, weight_decay=0.01, micro_batch_size=True, topk_ratio=0.5, muon_ns_steps=5  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="topk_ratio"):
        HiveMindWorkerNode._validate_inner_hyperparams(
            lr=1e-4, weight_decay=0.01, micro_batch_size=1, topk_ratio=True, muon_ns_steps=5  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="byzantine_poison_scale"):
        HiveMindWorkerNode._validate_inner_hyperparams(
            lr=1e-4, weight_decay=0.01, micro_batch_size=1, topk_ratio=0.5, muon_ns_steps=5, byzantine_poison_scale=True  # type: ignore[arg-type]
        )
