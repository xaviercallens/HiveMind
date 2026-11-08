# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for client package security & numerical guards:
  1. SteamOpenIDVerifier finite/positive nonce_ttl_s, max_nonces, max_attempts_per_ip, and clock readings.
  2. IdentityTokens finite/positive ttl_s and clock readings (preventing NaN clock expiry bypass).
  3. SteamworksSDKBridge.upload_hack_miner_score KeepBest semantics on api_credits_balance.
  4. HiveMindWorkerNode hyperparameter & byzantine_poison_scale finite/bounds validation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hivemind.client.steam_identity import IdentityError, IdentityTokens, SteamOpenIDVerifier
from hivemind.client.steamworks import SteamworksSDKBridge
from hivemind.client.worker_node import HiveMindWorkerNode


def test_steam_openid_verifier_rejects_nonfinite_ttl_and_clock() -> None:
    """Negative control: non-finite or non-positive nonce_ttl_s, max_nonces, max_attempts_per_ip, or clock must fail closed."""
    with pytest.raises(ValueError, match="nonce_ttl_s"):
        SteamOpenIDVerifier("https://hivemind.example/cb", nonce_ttl_s=float("nan"))
    with pytest.raises(ValueError, match="nonce_ttl_s"):
        SteamOpenIDVerifier("https://hivemind.example/cb", nonce_ttl_s=0.0)
    with pytest.raises(ValueError, match="max_nonces"):
        SteamOpenIDVerifier("https://hivemind.example/cb", max_nonces=0)
    with pytest.raises(ValueError, match="max_attempts_per_ip"):
        SteamOpenIDVerifier("https://hivemind.example/cb", max_attempts_per_ip=0)

    # Non-finite clock reading during verification must raise IdentityError instead of bypassing nonce/rate checks
    bad_clock_verifier = SteamOpenIDVerifier(
        "https://hivemind.example/cb",
        clock=lambda: float("nan"),
    )
    with pytest.raises(IdentityError, match="clock reading is non-finite"):
        bad_clock_verifier.verify_assertion({}, client_ip="203.0.113.10")


def test_identity_tokens_rejects_nonfinite_ttl_and_nan_clock_expiry_bypass() -> None:
    """Negative control: IdentityTokens must reject non-finite ttl_s and fail closed if clock returns NaN on verify."""
    secret = b"k" * 32
    with pytest.raises(ValueError, match="ttl_s"):
        IdentityTokens(secret=secret, ttl_s=float("nan"))
    with pytest.raises(ValueError, match="ttl_s"):
        IdentityTokens(secret=secret, ttl_s=-10.0)

    t_now = 1_700_000_000.0
    issuer = IdentityTokens(secret=secret, ttl_s=60.0, clock=lambda: t_now)
    token = issuer.issue("76561198000000001")
    assert issuer.verify(token) == "76561198000000001"

    # Expired clock rejects
    expired_verifier = IdentityTokens(secret=secret, ttl_s=60.0, clock=lambda: t_now + 120.0)
    with pytest.raises(IdentityError, match="expired"):
        expired_verifier.verify(token)

    # NaN clock must NOT let an expired token pass
    nan_verifier = IdentityTokens(secret=secret, ttl_s=60.0, clock=lambda: float("nan"))
    with pytest.raises(IdentityError, match="clock reading is non-finite"):
        nan_verifier.verify(token)


def test_steamworks_upload_hack_miner_score_preserves_keep_best_credits() -> None:
    """KeepBest leaderboard upload must not wipe previously earned workunit API credits."""
    bridge = SteamworksSDKBridge(signing_secret=b"s" * 32)
    steam_id = "76561198000000042"
    bridge.record_validated_workunit(
        steam_id64=steam_id,
        persona_name="Ada",
        wu_id="wu_001",
        tflops_contributed=12.5,
        credits_minted=25.0,
    )
    assert bridge.leaderboard[steam_id].api_credits_balance == 25.0

    # Uploading a score with a lower hive_balance (e.g. 5.0) must keep 25.0 under KeepBest semantics
    bridge.upload_hack_miner_score(
        steam_id64=steam_id,
        persona_name="Ada",
        total_points=500,
        swarm_contribution_points=100,
        hive_balance=5.0,
        achievements=["ACH_TEST"],
    )
    assert bridge.leaderboard[steam_id].api_credits_balance == 25.0

    with pytest.raises(ValueError, match="steam_id64"):
        bridge.upload_hack_miner_score(
            steam_id64="   ",
            persona_name="Ada",
            total_points=10,
            swarm_contribution_points=10,
            hive_balance=1.0,
            achievements=[],
        )


def test_worker_node_hyperparameter_and_poison_scale_validation() -> None:
    """HiveMindWorkerNode must reject non-finite or out-of-range inner-loop hyperparameters up front."""
    node = HiveMindWorkerNode(
        worker_id="w1",
        steam_id64="76561198000000099",
        persona_name="Turing",
        model=nn.Linear(4, 4),
    )
    dummy_tokens = torch.ones((2, 4), dtype=torch.long)

    for bad_kwargs, match in (
        ({"lr": 0.0}, "lr"),
        ({"lr": float("nan")}, "lr"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"micro_batch_size": 0}, "micro_batch_size"),
        ({"topk_ratio": 0.0}, "topk_ratio"),
        ({"topk_ratio": 1.5}, "topk_ratio"),
        ({"muon_ns_steps": 0}, "muon_ns_steps"),
    ):
        with pytest.raises(ValueError, match=match):
            node.execute_cas_shard(
                domain_name="math",
                token_sequences=dummy_tokens,
                global_lora_state={},
                **bad_kwargs,
            )

    with pytest.raises(ValueError, match="byzantine_poison_scale"):
        node._validate_inner_hyperparams(
            lr=1e-4,
            weight_decay=0.01,
            micro_batch_size=1,
            topk_ratio=0.2,
            muon_ns_steps=5,
            byzantine_poison_scale=float("inf"),
        )
