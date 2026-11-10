"""Negative-control and invariant tests for hf_hack_miner, queenbee.dance, queenbee.knowledge_market, and queenbee.strategies dataclasses."""

from __future__ import annotations

import hashlib
import math

import pytest

from hivemind.game.hf_hack_miner import (
    CodeRepairTask,
    GamerMinerProfile,
    GuildTournamentSquad,
    Lean4ProofTask,
    MinerScoreBreakdown,
)
from hivemind.queenbee.dance import RecruitmentReport
from hivemind.queenbee.knowledge_market import (
    Admission,
    KnowledgeCertificate,
    KnowledgeShare,
)
from hivemind.queenbee.strategies import (
    ClipReport,
    PrivacyBudgetReport,
    ServedExpertTask,
    ShareAggregation,
)


def test_hf_hack_miner_dataclasses_validation() -> None:
    valid_score = dict(
        gamer_tag="CyberValkyrie_FR",
        contract_id="HM-C-01",
        shard_cid="bafybeigdyrzt5s",
        shard_signal_score=0.92,
        initial_loss=2.1,
        final_loss=1.3,
        loss_reduction_pct=38.1,
        data_discovery_pts=100,
        lora_training_pts=250,
        solution_bounty_pts=150,
        swarm_contribution_pts=80,
        total_mission_pts=580,
        hive_earned=45.0,
    )
    ms = MinerScoreBreakdown(**valid_score)
    assert ms.total_mission_pts == 580

    with pytest.raises(ValueError, match="gamer_tag must be a non-empty string"):
        MinerScoreBreakdown(**{**valid_score, "gamer_tag": " "})
    with pytest.raises(ValueError, match="shard_signal_score must be a finite non-negative number"):
        MinerScoreBreakdown(**{**valid_score, "shard_signal_score": -0.1})
    with pytest.raises(ValueError, match="loss_reduction_pct must be a finite number"):
        MinerScoreBreakdown(**{**valid_score, "loss_reduction_pct": math.nan})
    with pytest.raises(ValueError, match="data_discovery_pts must be a non-negative integer"):
        MinerScoreBreakdown(**{**valid_score, "data_discovery_pts": -5})

    valid_prof = dict(
        gamer_tag="CyberValkyrie_FR",
        steam_id64="76561198000000001",
        node_id="node-fr-01",
        region="EU-West",
        specialty_focus="Rust",
    )
    prof = GamerMinerProfile(**valid_prof)
    assert prof.reputation_score == 100

    with pytest.raises(ValueError, match="gamer_tag must be a non-empty string"):
        GamerMinerProfile(**{**valid_prof, "gamer_tag": ""})
    with pytest.raises(ValueError, match="total_points must be a non-negative integer"):
        GamerMinerProfile(**{**valid_prof, "total_points": -1})
    with pytest.raises(ValueError, match="hive_balance must be a finite non-negative number"):
        GamerMinerProfile(**{**valid_prof, "hive_balance": -10.0})
    with pytest.raises(ValueError, match="steam_sybil_verified must be a boolean"):
        GamerMinerProfile(**{**valid_prof, "steam_sybil_verified": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="domains_forged must be a list of non-empty strings"):
        GamerMinerProfile(**{**valid_prof, "domains_forged": [""]})

    valid_repair = dict(
        task_id="REPAIR-01",
        title="Off by one",
        bug_family="boundary",
        buggy_code="def f(x): return x - 1",
        failing_test="assert f(1) == 2",
        reference_fix="def f(x): return x + 1",
        tests=("assert f(1) == 2",),
        candidate_fixes={"fix_a": "def f(x): return x + 1"},
    )
    crt = CodeRepairTask(**valid_repair)
    assert crt.task_id == "REPAIR-01"

    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        CodeRepairTask(**{**valid_repair, "task_id": ""})
    with pytest.raises(ValueError, match="tests must be a non-empty tuple of non-empty strings"):
        CodeRepairTask(**{**valid_repair, "tests": ()})
    with pytest.raises(ValueError, match="candidate_fixes must be a non-empty dict of non-empty strings"):
        CodeRepairTask(**{**valid_repair, "candidate_fixes": {}})

    valid_lean = dict(
        task_id="LEAN-01",
        title="Commutativity",
        natural_language="a + b = b + a",
        reference_proof="theorem add_comm_test (a b : Nat) : a + b = b + a := by omega",
        candidate_proofs={"p1": "theorem add_comm_test (a b : Nat) : a + b = b + a := by omega"},
    )
    lpt = Lean4ProofTask(**valid_lean)
    assert lpt.task_id == "LEAN-01"

    with pytest.raises(ValueError, match="title must be a non-empty string"):
        Lean4ProofTask(**{**valid_lean, "title": "  "})
    with pytest.raises(ValueError, match="candidate_proofs must be a non-empty dict of non-empty strings"):
        Lean4ProofTask(**{**valid_lean, "candidate_proofs": {"": "proof"}})

    valid_squad = dict(
        guild_id="GUILD-01",
        guild_name="Euro-Cyber",
        full_guild_id="GUILD-01-EURO",
        guild_motto="Zero-Copy",
        region_hub="EU-West",
        home_node="node-paris-01",
        lat=48.8566,
        lon=2.3522,
        member_miners=["Miner1"],
        member_contributors=["@c1"],
        synergy_multiplier=1.15,
        domain_weights={"rust": 0.5, "math": 0.5},
    )
    sq = GuildTournamentSquad(**valid_squad)
    assert sq.synergy_multiplier == 1.15

    with pytest.raises(ValueError, match="guild_id must be a non-empty string"):
        GuildTournamentSquad(**{**valid_squad, "guild_id": ""})
    with pytest.raises(ValueError, match="lat must be a finite number in \\[-90.0, 90.0\\]"):
        GuildTournamentSquad(**{**valid_squad, "lat": 95.0})
    with pytest.raises(ValueError, match="lon must be a finite number in \\[-180.0, 180.0\\]"):
        GuildTournamentSquad(**{**valid_squad, "lon": -190.0})
    with pytest.raises(ValueError, match="member_miners must be a non-empty list of non-empty strings"):
        GuildTournamentSquad(**{**valid_squad, "member_miners": []})
    with pytest.raises(ValueError, match="synergy_multiplier must be a finite positive number"):
        GuildTournamentSquad(**{**valid_squad, "synergy_multiplier": 0.0})
    with pytest.raises(ValueError, match="domain_weights must be a non-empty dict"):
        GuildTournamentSquad(**{**valid_squad, "domain_weights": {}})


def test_queenbee_dance_knowledge_and_strategies_dataclasses_validation() -> None:
    rep = RecruitmentReport(
        weights={"c1": {"e1": 0.5}},
        accepted=2,
        rejected={"expired": 1},
        trial_weights={"c1": {"e1": 10.0}},
        trial_successes={"c1": {"e1": 8.0}},
        signer_trials_used={"s1": 10.0},
    )
    assert rep.accepted == 2

    with pytest.raises(ValueError, match="accepted must be a non-negative integer"):
        RecruitmentReport(weights={}, accepted=-1, rejected={})
    with pytest.raises(ValueError, match="weights cluster keys must be non-empty strings"):
        RecruitmentReport(weights={"": {"e1": 0.5}}, accepted=1, rejected={})
    with pytest.raises(ValueError, match="weights values must be finite non-negative numbers"):
        RecruitmentReport(weights={"c1": {"e1": -0.1}}, accepted=1, rejected={})
    with pytest.raises(ValueError, match="rejected counts must be non-negative integers"):
        RecruitmentReport(weights={}, accepted=0, rejected={"bad": -1})
    with pytest.raises(ValueError, match="signer_trials_used values must be finite non-negative numbers"):
        RecruitmentReport(weights={}, accepted=0, rejected={}, signer_trials_used={"s1": -1.0})

    text = "def add(a, b): return a + b"
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    share = KnowledgeShare(
        account_id="acc_1",
        kind="verified_code",
        text_sha256=sha,
        licence="CC-BY-4.0",
        verified=True,
        epoch=1,
        verifier_receipt_sha256="a" * 64,
    )
    adm = Admission(True, "admitted", share)
    assert adm.admitted is True

    with pytest.raises(ValueError, match="admitted must be a boolean"):
        Admission(1, "admitted", share)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        Admission(True, "  ", share)
    with pytest.raises(ValueError, match="share must be a KnowledgeShare"):
        Admission(True, "admitted", "not_share")  # type: ignore[arg-type]

    cert = KnowledgeCertificate(True, True, True, True, True, True)
    assert cert.ok is True
    with pytest.raises(ValueError, match="price_unchanged must be a boolean"):
        KnowledgeCertificate(1, True, True, True, True, True)  # type: ignore[arg-type]

    clip = ClipReport(pre_norm=1.5, post_norm=1.0, clipped=True, sigma=0.05)
    assert clip.clipped is True
    with pytest.raises(ValueError, match="pre_norm must be a finite non-negative number"):
        ClipReport(pre_norm=-0.1, post_norm=1.0, clipped=False, sigma=0.05)
    with pytest.raises(ValueError, match="clipped must be a boolean"):
        ClipReport(pre_norm=1.0, post_norm=1.0, clipped="no", sigma=0.05)  # type: ignore[arg-type]

    pbr = PrivacyBudgetReport(steps=5, delta=1e-5, epsilon=1.2, max_epsilon=8.0, best_alpha=4.0, within_budget=True)
    assert pbr.within_budget is True
    with pytest.raises(ValueError, match="steps must be a non-negative integer"):
        PrivacyBudgetReport(steps=-1, delta=1e-5, epsilon=1.2, max_epsilon=8.0, best_alpha=4.0, within_budget=True)
    with pytest.raises(ValueError, match="delta must be a finite number in \\(0.0, 1.0\\)"):
        PrivacyBudgetReport(steps=1, delta=1.0, epsilon=1.2, max_epsilon=8.0, best_alpha=4.0, within_budget=True)
    with pytest.raises(ValueError, match="best_alpha must be a finite number > 1.0"):
        PrivacyBudgetReport(steps=1, delta=1e-5, epsilon=1.2, max_epsilon=8.0, best_alpha=1.0, within_budget=True)

    sa = ShareAggregation(
        quorum_reached=True,
        admitted=("w1", "w2"),
        deferred=(),
        accepted_by_krum=("w1",),
        rejected_by_krum=("w2",),
        aggregate={"w": 1},
    )
    assert sa.quorum_reached is True
    with pytest.raises(ValueError, match="quorum_reached must be a boolean"):
        ShareAggregation(1, ("w1",), (), ("w1",), ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="admitted must be a tuple of non-empty strings"):
        ShareAggregation(True, ("",), (), (), ())
    with pytest.raises(ValueError, match="aggregate must be a Mapping or None"):
        ShareAggregation(True, ("w1",), (), ("w1",), (), aggregate="bad")  # type: ignore[arg-type]

    task = ServedExpertTask(
        task_id="t1",
        cluster_id="c1",
        prompt="Solve x",
        tests=("assert f(1) == 2",),
        expert_candidates={"e1": "def f(x): return x + 1"},
        cost_per_1k={"e1": 0.25},
    )
    assert task.task_id == "t1"
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        ServedExpertTask("", "c1", "p", ("assert True",), {"e1": "x=1"})
    with pytest.raises(ValueError, match="tests must be a non-empty tuple of non-empty strings"):
        ServedExpertTask("t1", "c1", "p", (), {"e1": "x=1"})
    with pytest.raises(ValueError, match="expert_candidates must be a non-empty mapping of non-empty strings"):
        ServedExpertTask("t1", "c1", "p", ("assert True",), {})
    with pytest.raises(ValueError, match="cost_per_1k values must be finite non-negative numbers"):
        ServedExpertTask("t1", "c1", "p", ("assert True",), {"e1": "x=1"}, cost_per_1k={"e1": -0.5})
