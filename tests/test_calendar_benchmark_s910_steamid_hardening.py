# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control boundary tests for campaign_calendar_quorum,
community_gpu_benchmark, season9_10_gpu_swarm_campaigns, and steam_identity."""

from __future__ import annotations

import pytest
import torch

from hivemind.cli.campaign_calendar_quorum import (
    evaluate_quorum_readiness,
    main as calendar_main,
)
from hivemind.cli.community_gpu_benchmark import (
    detect_community_accelerator,
    main as benchmark_main,
    run_community_gpu_benchmark,
)
from hivemind.cli.season9_10_gpu_swarm_campaigns import (
    _build_and_train_domain_branches,
    main as s910_main,
    run_season9_10_gpu_swarm_campaigns,
)
from hivemind.client.steam_identity import (
    IdentityError,
    IdentityTokens,
    SteamIdentityGate,
    SteamOpenIDVerifier,
    fetch_account_facts,
)


def test_campaign_calendar_quorum_rejects_non_string_elements_and_raw_argv() -> None:
    for bad_seq in (["alice", 123], ["alice", True], ["alice", None]):
        with pytest.raises(ValueError, match="human_participants elements must all be strings"):
            evaluate_quorum_readiness(bad_seq, ["a1"], ["d1"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ai_agent_participants elements must all be strings"):
            evaluate_quorum_readiness(["h1"], bad_seq, ["d1"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="covered_domains elements must all be strings"):
            evaluate_quorum_readiness(["h1"], ["a1"], bad_seq)  # type: ignore[arg-type]

    for bad_argv in ("--output out.json", b"--output", 42):
        with pytest.raises(ValueError, match="argv must be a sequence"):
            calendar_main(bad_argv)  # type: ignore[arg-type]

    res = evaluate_quorum_readiness(
        ["alice", "bob", "charlie", ""],
        ["agent-1", "agent-2"],
        ["d1", "d2", "d3", "d4"],
    )
    assert res["tier1_reached"] is True
    assert res["total_nodes"] == 5


def test_community_gpu_benchmark_rejects_non_string_device_and_non_int_dimensions() -> None:
    for bad_dev in (True, 123, ["auto"], {"device": "cpu"}):
        with pytest.raises(ValueError, match="Unsupported preferred_device"):
            detect_community_accelerator(bad_dev)  # type: ignore[arg-type]

    assert detect_community_accelerator("cpu")["device"] == "cpu"

    for bad_kwargs in (
        {"token_steps": 4.5},
        {"token_steps": "8"},
        {"router_tokens": 16.0},
        {"router_tokens": "16"},
        {"hidden_size": 16.0},
        {"hidden_size": "16"},
        {"pool_holdings_override": "200.0"},
    ):
        with pytest.raises(ValueError):
            run_community_gpu_benchmark(preferred_device="cpu", output_path=None, **bad_kwargs)  # type: ignore[arg-type]

    for bad_argv in ("--device cpu", b"--device", 42):
        with pytest.raises(ValueError, match="argv must be a list/tuple"):
            benchmark_main(bad_argv)  # type: ignore[arg-type]


def test_season9_10_gpu_swarm_campaigns_rejects_invalid_branch_builder_and_cli_args() -> None:
    with pytest.raises(ValueError, match="dev must be a torch.device"):
        _build_and_train_domain_branches("cpu")  # type: ignore[arg-type]
    for bad_domains in (True, 1, 9, 4.0, "4"):
        with pytest.raises(ValueError, match="num_domains"):
            _build_and_train_domain_branches(torch.device("cpu"), num_domains=bad_domains)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="num_domains"):
            run_season9_10_gpu_swarm_campaigns(num_domains=bad_domains, output_path=None)  # type: ignore[arg-type]
    for bad_steps in (True, 0, 2.5, "2"):
        with pytest.raises(ValueError, match="inner_steps"):
            _build_and_train_domain_branches(torch.device("cpu"), inner_steps=bad_steps)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="inner_steps"):
            run_season9_10_gpu_swarm_campaigns(inner_steps=bad_steps, output_path=None)  # type: ignore[arg-type]
    for bad_seed in (True, -1, 3.0):
        with pytest.raises(ValueError, match="seed"):
            _build_and_train_domain_branches(torch.device("cpu"), seed=bad_seed)  # type: ignore[arg-type]

    for bad_argv in ("--device cpu", b"--device", 99):
        with pytest.raises(ValueError, match="argv must be a sequence"):
            s910_main(bad_argv)  # type: ignore[arg-type]


def test_steam_identity_rejects_non_callable_and_non_mapping_inputs() -> None:
    cb = "https://coord.example/identity/steam/callback"
    with pytest.raises(ValueError, match="http must be callable"):
        SteamOpenIDVerifier(cb, http="not-callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clock must be callable"):
        SteamOpenIDVerifier(cb, clock=12345)  # type: ignore[arg-type]

    verifier = SteamOpenIDVerifier(cb)
    with pytest.raises(IdentityError, match="params must be a Mapping"):
        verifier.verify_assertion(["not-a-mapping"])  # type: ignore[arg-type]
    with pytest.raises(IdentityError, match="client_ip must be a string or None"):
        verifier.verify_assertion({}, client_ip=True)  # type: ignore[arg-type]
    with pytest.raises(IdentityError, match="client_ip must be a string or None"):
        verifier.verify_assertion({}, client_ip=12345)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="http must be callable"):
        fetch_account_facts("76561197960287930", http=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="clock must be callable"):
        IdentityTokens(clock="not-callable")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="http must be callable"):
        SteamIdentityGate(cb, http="not-callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tokens must be an IdentityTokens"):
        SteamIdentityGate(cb, tokens="bad-tokens")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="verifier must be a SteamOpenIDVerifier"):
        SteamIdentityGate(cb, verifier="bad-verifier")  # type: ignore[arg-type]

    gate = SteamIdentityGate(cb)
    token = gate.tokens.issue("76561197960287930")
    assert gate.authorize(token, "76561197960287930") == "76561197960287930"
