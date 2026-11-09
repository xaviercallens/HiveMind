# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-functional tests for Iteration 135 hardening.

Covers boundary validation across:
- hivemind.queenbee.strategies (BetaArm, BanditStrategy.arm/observe,
  PreferencePairStrategy.__init__/observe_correction/training_mix,
  aggregate_lora_product_exact)
- hivemind.boinc.real_client.resident_validator (send_msg, _recv_exact,
  recv_msg, ResidentValidatorServer.__init__/dispatch)
- hivemind.lab.swarm_lab (_norm, build_domains, domain_loss, losses,
  carried_weights, compare_policies)
- hivemind.core.btm_moe (_keyword_hit, RegisteredDomainExpert,
  Season3DomainExpertRegistry.register_expert, _even_cluster_sizes,
  _grow_gate, HierarchicalMoERouter.forward, ThreeTierHierarchicalMoERouter.forward)
"""

from __future__ import annotations

import math
import socket
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hivemind.boinc.real_client import resident_validator as rv
from hivemind.core import btm_moe as bm
from hivemind.lab import swarm_lab as sl
from hivemind.queenbee import strategies as qb


# =====================================================================
# 1. queenbee/strategies.py hardening tests
# =====================================================================


def test_beta_arm_and_bandit_strategy_boundary_validation() -> None:
    for bad_ab in ((0.0, 1.0), (1.0, -1.0), (math.nan, 1.0), (True, 1.0), ("1", 1.0)):
        with pytest.raises(ValueError, match="alpha and beta must be finite and > 0"):
            qb.BetaArm(*bad_ab)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="verified_n must be an integer >= 0"):
        qb.BetaArm(1.0, 1.0, verified_n=-1)
    with pytest.raises(ValueError, match="verified_n must be an integer >= 0"):
        qb.BetaArm(1.0, 1.0, verified_n=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="verified_successes cannot exceed verified_n"):
        qb.BetaArm(1.0, 1.0, verified_n=2, verified_successes=3)

    bs = qb.BanditStrategy()
    for bad_id in ("", "   ", 123, None):
        with pytest.raises(ValueError, match="cluster_id and expert_id must be non-empty strings"):
            bs.arm(bad_id, "e1")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cluster_id and expert_id must be non-empty strings"):
            bs.arm("c1", bad_id)  # type: ignore[arg-type]

    for bad_outcome in (None, "outcome", {"cluster_id": "c1"}):
        with pytest.raises(ValueError, match="outcome must be an Outcome"):
            bs.observe(bad_outcome)  # type: ignore[arg-type]

    bs.observe(qb.Outcome("c1", "e1", verified=True, success=True))
    assert bs.arm("c1", "e1").verified_successes == 1


def test_preference_pair_strategy_and_lora_product_exact_validation() -> None:
    with pytest.raises(ValueError, match="router must be a BanditStrategy or None"):
        qb.PreferencePairStrategy(router="bad")  # type: ignore[arg-type]

    pps = qb.PreferencePairStrategy()
    with pytest.raises(ValueError, match="prompt_id and cluster_id must be non-empty strings"):
        pps.observe_correction("", "c1", "bad", "good", verifier_kind="unit_tests", chosen_verified=True)
    with pytest.raises(ValueError, match="rejected and chosen must be strings"):
        pps.observe_correction("p1", "c1", 123, "good", verifier_kind="unit_tests", chosen_verified=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chosen_verified must be a boolean"):
        pps.observe_correction("p1", "c1", "bad", "good", verifier_kind="unit_tests", chosen_verified=1)  # type: ignore[arg-type]

    for bad_seq in ("str", b"bytes", [123], None):
        with pytest.raises(ValueError, match="unverified_pairs must be a sequence of PreferencePair"):
            pps.training_mix(bad_seq)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="admitted_deltas must be a Mapping"):
        qb.aggregate_lora_product_exact(None, ["w1"])  # type: ignore[arg-type]
    for bad_workers in ("w1", b"w1", 123):
        with pytest.raises(ValueError, match="accepted_workers must be a Sequence"):
            qb.aggregate_lora_product_exact({}, bad_workers)  # type: ignore[arg-type]
    assert qb.aggregate_lora_product_exact({}, []) == {}


# =====================================================================
# 2. resident_validator.py hardening tests
# =====================================================================


def test_resident_validator_protocol_and_server_validation(tmp_path: Path) -> None:
    s1, s2 = socket.socketpair()
    try:
        with pytest.raises(rv.ProtocolError, match="message must be a dict"):
            rv.send_msg(s1, ["not-dict"])  # type: ignore[arg-type]
        for bad_lim in (True, 0, -5, "100"):
            with pytest.raises(ValueError, match="limit must be a positive integer"):
                rv.send_msg(s1, {"ok": True}, limit=bad_lim)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="limit must be a positive integer"):
                rv.recv_msg(s2, limit=bad_lim)  # type: ignore[arg-type]
        for bad_n in (True, -1, "4"):
            with pytest.raises(ValueError, match="n must be a non-negative integer"):
                rv._recv_exact(s2, bad_n)  # type: ignore[arg-type]

        rv.send_msg(s1, {"op": "ping"})
        assert rv.recv_msg(s2) == {"op": "ping"}
    finally:
        s1.close()
        s2.close()

    sock_path = tmp_path / "val.sock"
    work_root = tmp_path / "work"
    work_root.mkdir()
    for bad_path in ("", "   ", 123, None):
        with pytest.raises(ValueError, match="socket_path must be a non-empty path"):
            rv.ResidentValidatorServer(bad_path, work_root)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="work_root must be a non-empty path"):
            rv.ResidentValidatorServer(sock_path, bad_path)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cache must be a ModelCache or None"):
        rv.ResidentValidatorServer(sock_path, work_root, cache="not-cache")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="validate_fn must be callable or None"):
        rv.ResidentValidatorServer(sock_path, work_root, validate_fn="not-callable")  # type: ignore[arg-type]

    srv = rv.ResidentValidatorServer(sock_path, work_root)
    with pytest.raises(ValueError, match="req must be a dict"):
        srv.dispatch("not-a-dict")  # type: ignore[arg-type]
    assert srv.dispatch({"op": "ping"})["pong"] is True


# =====================================================================
# 3. lab/swarm_lab.py hardening tests
# =====================================================================


def test_swarm_lab_boundary_validation() -> None:
    for bad_v in (None, "bad", [1.0, math.nan], [True]):
        with pytest.raises(ValueError, match="v must be a sequence of finite numbers"):
            sl._norm(bad_v)  # type: ignore[arg-type]
    assert sl._norm([3.0, 4.0]) == 5.0

    with pytest.raises(ValueError, match="sc must be a Scenario"):
        sl.build_domains({"M": 4})  # type: ignore[arg-type]

    for bad_args in (([], [1.0], [0.0]), ([1.0], [1.0, 2.0], [0.0]), ("bad", [1.0], [0.0])):
        with pytest.raises(ValueError, match="theta, a, and c must be equal-length non-empty lists"):
            sl.domain_loss(*bad_args)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="a_all and c_all must be equal-length non-empty lists"):
        sl.losses([1.0], [], [])

    with pytest.raises(ValueError, match="queue must be a list"):
        sl.carried_weights("not-list", 1)  # type: ignore[arg-type]
    for bad_rnd in (True, 0, -1, "1"):
        with pytest.raises(ValueError, match="rnd must be an integer >= 1"):
            sl.carried_weights([], bad_rnd)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="policies must be a non-empty subset"):
        sl.compare_policies(policies=("invalid",))
    with pytest.raises(ValueError, match="straggles must be a non-empty subset"):
        sl.compare_policies(straggles=())


# =====================================================================
# 4. core/btm_moe.py hardening tests
# =====================================================================


def test_btm_moe_keyword_registry_and_router_validation() -> None:
    with pytest.raises(ValueError, match="keyword and text_lower must be strings"):
        bm._keyword_hit(123, "rust code")  # type: ignore[arg-type]
    assert bm._keyword_hit("rust", "safe rust code") is True

    with pytest.raises(ValueError, match="domain_name and dataset_id must be non-empty strings"):
        bm.RegisteredDomainExpert("", "ds", "c1", "s1", 16, 32, 2.0, 1.0, 4.0, 1024, {})
    with pytest.raises(ValueError, match="lora_rank_r and lora_alpha must be positive integers"):
        bm.RegisteredDomainExpert("Rust", "ds", "c1", "s1", 0, 32, 2.0, 1.0, 4.0, 1024, {})
    with pytest.raises(ValueError, match="lora_state_dict must be a dict"):
        bm.RegisteredDomainExpert("Rust", "ds", "c1", "s1", 16, 32, 2.0, 1.0, 4.0, 1024, "bad")  # type: ignore[arg-type]

    reg = bm.Season3DomainExpertRegistry()
    with pytest.raises(ValueError, match="expert must be a RegisteredDomainExpert"):
        reg.register_expert("not-expert")  # type: ignore[arg-type]

    for bad_args in ((0, 4), (4, 0), (True, 4), (4, False)):
        with pytest.raises(ValueError, match="total and num_clusters must be integers >= 1"):
            bm._even_cluster_sizes(*bad_args)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gate must be an nn.Linear"):
        bm._grow_gate("not-linear")  # type: ignore[arg-type]
    grown = bm._grow_gate(nn.Linear(8, 4, bias=False))
    assert grown.out_features == 5

    r2 = bm.HierarchicalMoERouter(hidden_size=8, num_macro_domains=2, experts_per_domain=4, top_k=2)
    r3 = bm.ThreeTierHierarchicalMoERouter(hidden_size=8, num_super_clusters=2, macro_per_super=2, micro_per_macro=2, top_k=2)
    for router in (r2, r3):
        for bad_hs in (
            "not-tensor",
            torch.zeros((2, 4, 8)),
            torch.zeros((0, 8)),
            torch.zeros((2, 6)),
            torch.tensor([[1.0, math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        ):
            with pytest.raises(ValueError, match="hidden_states must be a finite 2D tensor"):
                router(bad_hs)  # type: ignore[arg-type]
        w, idx, aux = router(torch.randn(3, 8))
        assert w.shape == (3, 2)
        assert idx.shape == (3, 2)
        assert bool(torch.isfinite(aux).item())
