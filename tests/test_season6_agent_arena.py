"""Season 6 Milestone HM-S6-04: Autonomous Agent Arena, Missions 21-24 & SI-01..SI-05 HUD tab.

Validates:
- the four Season 6 Hack-Miner missions 21-24 and their curated offline corpora,
- the multi-agent self-play code-repair arena (real assertions in an isolated subprocess),
- the structural Lean 4 autoformalizer arena (sorry/axiom rejection without kernel claims),
- live `SI-01`..`SI-05` Pareto strategy telemetry and its POST update path,
- the `/api/season6-missions`, `/api/agent-arena` and `/api/strategy-optimizer` HUD endpoints.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from hivemind.game.hf_hack_miner import (  # noqa: E402
    LEAN4_PROOF_TASKS,
    SELF_PLAY_CODE_REPAIR_TASKS,
    SEASON_6_EXTENDED_HF_BOUNTY_CONTRACTS,
    SEASON_6_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS,
    SEASON_6_HF_BOUNTY_CONTRACTS,
    STEAM_ACHIEVEMENTS_CATALOG,
    GamerMinerProfile,
    Lean4AutoformalizerArena,
    SelfPlayCodeRepairArena,
    audit_lean4_proof,
    build_agent_arena_state,
    build_season6_mission_catalog,
    build_swarm_telemetry_frame,
    execute_interactive_lora_micro_forge,
    fetch_offline_curated_corpus,
    fetch_season6_curated_corpus,
)
from hivemind.game.steam_edition import create_hack_miner_fastapi_app  # noqa: E402
from hivemind.swarm.strategy_optimizer import (  # noqa: E402
    STRATEGY_OPTIMIZER_SCHEMA_VERSION,
    StrategyOptimizerTelemetryEngine,
)

SEASON_6_MISSION_IDS = [
    "CONTRACT-HF-21-SELFPLAY-CODE-REPAIR",
    "CONTRACT-HF-22-LEAN4-AGENT-FORMALIZER",
    "CONTRACT-HF-23-PARETO-STRATEGY-OPTIMIZER",
    "CONTRACT-HF-24-AUTONOMOUS-AGENT-ARENA",
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_hack_miner_fastapi_app(worker_token="test-worker-token-not-a-secret"))


def test_hm_s6_04_season6_missions_catalog_and_offline_corpora() -> None:
    """The four Season 6 contracts exist, are catalogued, and ship offline corpora."""
    ids = [c.contract_id for c in SEASON_6_HF_BOUNTY_CONTRACTS]
    assert ids == SEASON_6_MISSION_IDS
    assert all(c.season == 6 for c in SEASON_6_HF_BOUNTY_CONTRACTS)
    assert all(c.difficulty_skulls >= 4 for c in SEASON_6_HF_BOUNTY_CONTRACTS)
    assert all(c.base_bounty_pts > 0 for c in SEASON_6_HF_BOUNTY_CONTRACTS)

    domains = [c.domain_expert for c in SEASON_6_HF_BOUNTY_CONTRACTS]
    assert domains == [
        "SelfPlay_Code_Repair",
        "Lean4_Agent_Autoformalizer",
        "Pareto_Guild_Strategy_Optimizer",
        "Autonomous_Agent_Arena",
    ]

    # Season 6 extends (never mutates) the Season 1-5 campaigns.
    assert len(SEASON_6_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS) == 22
    assert len(SEASON_6_EXTENDED_HF_BOUNTY_CONTRACTS) == 24
    assert len(set(c.contract_id for c in SEASON_6_EXTENDED_HF_BOUNTY_CONTRACTS)) == 24

    catalog = build_season6_mission_catalog()
    assert catalog["season"] == 6
    assert catalog["milestone"] == "HM-S6-04"
    assert catalog["num_missions"] == 4
    assert catalog["mission_ids"] == SEASON_6_MISSION_IDS
    assert catalog["bounty_pool_hive"] == 2500.0
    assert catalog["total_bounty_pts"] == sum(
        c.base_bounty_pts for c in SEASON_6_HF_BOUNTY_CONTRACTS
    )
    assert catalog["agent_arena_endpoint"] == "/api/agent-arena"
    assert catalog["strategy_optimizer_endpoint"] == "/api/strategy-optimizer"

    for contract in SEASON_6_HF_BOUNTY_CONTRACTS:
        corpus = fetch_season6_curated_corpus(contract.dataset_id, num_samples=12)
        assert len(corpus) == 12
        assert corpus[0].startswith(contract.bounty_question_prompt)
        assert contract.reference_solution in corpus[0]
        assert fetch_offline_curated_corpus(contract.dataset_id, num_samples=3) == corpus[:3]


def test_hm_s6_04_season6_missions_are_completable() -> None:
    """Each Season 6 mission is completable through the interactive QLoRA Micro-Forge."""
    for contract in SEASON_6_HF_BOUNTY_CONTRACTS:
        result = execute_interactive_lora_micro_forge(
            gamer_tag="CyberValkyrie_FR",
            contract_id=contract.contract_id,
        )
        assert result["contract_id"] == contract.contract_id
        assert result["season"] == 6
        assert result["dataset_id"] == contract.dataset_id
        assert result["domain_expert"] == contract.domain_expert
        assert result["spot_check_passed"] is True
        assert result["final_loss"] < result["initial_loss"]
        assert result["loss_reduction_pct"] > 0.0
        assert result["bounty_pts_awarded"] > contract.base_bounty_pts
        assert result["total_pts_awarded"] > contract.base_bounty_pts
        assert result["hive_minted"] > 0.0
        assert result["steam_rich_presence"]
        assert result["agent_arena_badge"] in {
            "S6_LEAN4_AUTOFORMALIZER",
            "S6_AGENT_ARENA_CHAMPION",
        }

    assert "S6_AGENT_ARENA_CHAMPION" in STEAM_ACHIEVEMENTS_CATALOG
    assert "S6_LEAN4_AUTOFORMALIZER" in STEAM_ACHIEVEMENTS_CATALOG


def test_hm_s6_04_self_play_code_repair_verifier_and_arena() -> None:
    """The arena executes real curated tests and rejects untrusted imports."""
    for task in SELF_PLAY_CODE_REPAIR_TASKS:
        correct = next(iter(task.candidate_fixes.values()))
        assert correct  # every task ships at least one candidate patch
        assert task.reference_fix

    from hivemind.game.hf_hack_miner import _evaluate_repair_candidate

    task = SELF_PLAY_CODE_REPAIR_TASKS[0]
    perfect = _evaluate_repair_candidate(task.reference_fix, task.tests)
    assert perfect["score"] == 1.0
    assert perfect["tests_passed"] == perfect["tests_total"] == len(task.tests)

    buggy = _evaluate_repair_candidate(task.buggy_code, task.tests)
    assert buggy["score"] < 1.0

    # Arbitrary imports are not available inside the restricted verifier namespace.
    untrusted = _evaluate_repair_candidate(
        "import os\ndef binary_search(arr, target):\n    return -1\n",
        task.tests,
    )
    assert untrusted["score"] == 0.0

    arena = SelfPlayCodeRepairArena(seed=42)
    report = arena.run(rounds=16)
    assert report["schema_version"] == "hm-s6-04/self-play-code-repair/v1"
    assert report["milestone"] == "HM-S6-04"
    assert report["rounds_executed"] == 16
    assert report["num_agents"] == 4
    assert report["num_tasks"] == 4
    assert len(report["round_telemetry"]) == 16
    assert len(report["learning_curve"]) == 16
    assert report["champion_agent"] in [row["agent_id"] for row in report["agent_standing"]]
    assert report["champion_policy"] in report["policies"]
    assert report["validated_repairs"] >= 3
    assert report["total_tests_executed"] > 0
    assert 0.0 <= report["mean_final_round_score"] <= 1.0
    assert set(report["best_patch_by_task"]) == {
        task.task_id for task in SELF_PLAY_CODE_REPAIR_TASKS
    }
    assert any(patch["score"] == 1.0 for patch in report["best_patch_by_task"].values())

    # Deterministic seed => identical champion and policy weights.
    replay = SelfPlayCodeRepairArena(seed=42).run(rounds=16)
    assert replay["champion_agent"] == report["champion_agent"]
    assert replay["final_policy_weights"] == report["final_policy_weights"]


def test_hm_s6_04_lean4_autoformalizer_audit_and_arena() -> None:
    """The Lean 4 arena rejects `sorry`/custom axioms and never claims kernel verification."""
    reference = LEAN4_PROOF_TASKS[0].reference_proof
    audit = audit_lean4_proof(reference)
    assert audit["valid"] is True
    assert audit["kernel_checked"] is False
    assert audit["score"] == 1.0

    sorry_proof = "theorem bad (n : Nat) : n = n := by\n  sorry"
    sorry_audit = audit_lean4_proof(sorry_proof)
    assert sorry_audit["valid"] is False
    assert sorry_audit["checks"]["no_sorry_or_admit"] is False

    axiom_proof = (
        "axiom cheat (n : Nat) : n = n\n"
        "theorem bad (n : Nat) : n = n := by\n  exact cheat n"
    )
    assert audit_lean4_proof(axiom_proof)["checks"]["no_custom_axiom"] is False

    unbalanced = "theorem bad (n : Nat : n = n := by\n  rfl"
    assert audit_lean4_proof(unbalanced)["checks"]["balanced_delimiters"] is False
    assert audit_lean4_proof("")["valid"] is False

    arena = Lean4AutoformalizerArena(seed=7)
    report = arena.run(rounds=9)
    assert report["schema_version"] == "hm-s6-04/lean4-agent-autoformalizer/v1"
    assert report["rounds_executed"] == 9
    assert report["num_tasks"] == len(LEAN4_PROOF_TASKS)
    assert report["validated_proofs"] == len(LEAN4_PROOF_TASKS)
    assert report["zero_sorry_proofs"] == len(LEAN4_PROOF_TASKS)
    assert report["all_proofs_sorry_free"] is True
    assert report["honesty"]["verification"].startswith("structural Lean 4 lint")
    assert report["honesty"]["not_verified"]
    assert all(not item["valid"] or "sorry" not in item["proof"] for item in report["best_proof_by_task"].values())


def test_hm_s6_04_agent_arena_and_achievements() -> None:
    """The combined arena state is deterministic and awards the Season 6 badges."""
    state = build_agent_arena_state(seed=42, rounds=8)
    assert state["schema_version"] == "hm-s6-04/agent-arena/v1"
    assert state["milestone"] == "HM-S6-04"
    assert state["season"] == 6
    assert state["status"] == "settled"
    assert state["mission_ids"] == SEASON_6_MISSION_IDS
    assert state["num_agents"] == 4
    assert state["self_play_code_repair"]["rounds_executed"] == 8
    assert state["lean4_autoformalization"]["zero_sorry_proofs"] >= 1
    assert state["strategy_optimizer_endpoint"] == "/api/strategy-optimizer"

    profile = GamerMinerProfile(
        gamer_tag="CyberValkyrie_FR",
        steam_id64="76561198000000001",
        node_id="seti-node-paris-01",
        region="EU-West-Paris",
        specialty_focus="Code",
        season6_missions_forged=4,
        agent_arena_wins=1,
    )
    profile.domains_forged.append("Lean4_Agent_Autoformalizer")
    profile.update_rank_and_achievements()
    assert "S6_AGENT_ARENA_CHAMPION" in profile.achievements_unlocked
    assert "S6_LEAN4_AUTOFORMALIZER" in profile.achievements_unlocked


def test_hm_s6_04_live_strategy_optimizer_engine() -> None:
    """SI-01..SI-05 are computed from real optimizer components with an explicit honesty section."""
    engine = StrategyOptimizerTelemetryEngine(seed=2026)
    snapshot = engine.snapshot()
    assert snapshot["schema_version"] == STRATEGY_OPTIMIZER_SCHEMA_VERSION
    assert snapshot["milestone"] == "HM-S6-04"
    assert snapshot["strategy_ids"] == ["SI-01", "SI-02", "SI-03", "SI-04", "SI-05"]

    si01 = snapshot["si01_dynamic_fisher_svd"]
    assert si01["vram_budget_conserved"] is True
    assert si01["total_rank_used"] <= si01["total_rank_budget"]
    assert si01["spectral_energy_improvement"] >= 0.0
    assert set(si01["allocated_ranks_by_domain"]) == {
        "Rust_Systems_Security",
        "Solidity_Smart_Contracts",
        "Biomedical_Clinical_QA",
        "Lean4_Theorem_Prover",
    }

    si02 = snapshot["si02_ucb1_curriculum_bandit"]
    assert len(si02["top2_prioritized_domains"]) == 2
    assert set(si02["ucb1_scores"]) == set(si02["pull_counts"])

    si03 = snapshot["si03_adaptive_diloco_cadence"]
    assert si03["current_link_profile"]["sla_8pct_satisfied"] is True
    assert si03["transcontinental_turn_relay_profile"]["recommended_inner_steps_H"] > si03[
        "regional_quic_0rtt_profile"
    ]["recommended_inner_steps_H"]

    si04 = snapshot["si04_reputation_weighted_ties"]
    assert si04["reputation_weighted_beats_uniform"] is True
    assert (
        si04["reputation_weighted_ties_holdout_loss"]
        < si04["uniform_ties_holdout_loss"]
    )

    frontier = snapshot["si05_pareto_strategy_frontier"]
    assert len(frontier) == 3
    assert [p["strategy_profile"] for p in frontier][-1].startswith("Proposed_v1.7")
    assert frontier[-1]["pareto_optimal"] is True
    assert all(p["coordinates_modeled"] is True for p in frontier)
    assert snapshot["honesty"]["modeled"]

    # Live link/curriculum updates recompute SI-02/SI-03 and advance the sequence.
    updated = engine.update(
        {
            "ring_rtt_ms": 145.0,
            "bbr_pacing_mbps": 18.0,
            "packet_loss_rate": 0.03,
            "turn_relay_active": True,
            "domain_name": "Solidity_Smart_Contracts",
            "loss_before": 2.0,
            "loss_after": 0.5,
            "gpu_seconds": 0.25,
        }
    )
    assert updated["sequence"] == 1
    assert updated["live_inputs"]["turn_relay_active"] is True
    assert updated["si03_adaptive_diloco_cadence"]["current_link_profile"]["turn_relay_active"] is True
    assert updated["si02_ucb1_curriculum_bandit"]["pull_counts"]["Solidity_Smart_Contracts"] == 2

    with pytest.raises(ValueError, match="ring_rtt_ms"):
        engine.update({"ring_rtt_ms": 0})
    with pytest.raises(ValueError, match="bbr_pacing_mbps"):
        engine.update({"bbr_pacing_mbps": -1})
    with pytest.raises(ValueError, match="packet_loss_rate"):
        engine.update({"packet_loss_rate": 1.5})
    with pytest.raises(ValueError, match="unknown domain_name"):
        engine.update({"domain_name": "NotADomain"})
    with pytest.raises(ValueError, match="loss_before"):
        engine.update({"domain_name": "Rust_Systems_Security"})
    with pytest.raises(ValueError, match="gpu_seconds"):
        engine.update(
            {
                "domain_name": "Rust_Systems_Security",
                "loss_before": 1.0,
                "loss_after": 0.5,
                "gpu_seconds": 0,
            }
        )


def test_hm_s6_04_hud_and_api_endpoints(client: TestClient) -> None:
    """The HUD exposes Season 6 missions, arena runs, strategy telemetry and offline fallbacks."""
    catalog = client.get("/api/season6-missions").json()
    assert catalog["season"] == 6
    assert catalog["num_missions"] == 4
    assert catalog["missions"] == SEASON_6_MISSION_IDS
    assert catalog["agent_arena_endpoint"] == "/api/agent-arena"
    assert catalog["strategy_optimizer_endpoint"] == "/api/strategy-optimizer"
    assert len(catalog["contracts"]) == 4
    assert all(contract["season"] == 6 for contract in catalog["contracts"])

    arena = client.get("/api/agent-arena", params={"rounds": 6, "seed": 3})
    assert arena.status_code == 200
    arena_body = arena.json()
    assert arena_body["season"] == 6
    assert arena_body["steam_app_id"] == 3133700
    assert arena_body["self_play_code_repair"]["rounds_executed"] == 6

    bad_arena = client.get("/api/agent-arena", params={"rounds": 99})
    assert bad_arena.status_code == 400
    bad_arena_payload = client.post("/api/agent-arena", json={"rounds": "many"})
    assert bad_arena_payload.status_code == 400

    posted_arena = client.post(
        "/api/agent-arena",
        json={"rounds": 6, "seed": 3, "gamer_tag": "NeuralGhost_DE"},
    )
    assert posted_arena.status_code == 200
    assert posted_arena.json()["gamer_tag"] == "NeuralGhost_DE"

    strategy = client.get("/api/strategy-optimizer")
    assert strategy.status_code == 200
    strategy_body = strategy.json()
    assert strategy_body["milestone"] == "HM-S6-04"
    assert strategy_body["steam_app_id"] == 3133700
    assert strategy_body["si05_pareto_strategy_frontier"]

    posted_strategy = client.post(
        "/api/strategy-optimizer",
        headers={"Authorization": "Bearer test-worker-token-not-a-secret"},
        json={
            "ring_rtt_ms": 145.0,
            "bbr_pacing_mbps": 18.0,
            "packet_loss_rate": 0.03,
            "turn_relay_active": True,
        },
    )
    assert posted_strategy.status_code == 200
    assert posted_strategy.json()["sequence"] >= 1
    invalid_strategy = client.post("/api/strategy-optimizer", headers={"Authorization": "Bearer test-worker-token-not-a-secret"}, json={"ring_rtt_ms": -5})
    assert invalid_strategy.status_code == 400

    missions = client.get("/api/missions").json()
    assert len(missions["season6_missions"]) == 4
    assert missions["season6_missions"][0]["contract_id"] == SEASON_6_MISSION_IDS[0]

    frame = build_swarm_telemetry_frame(0)
    assert frame["season6_missions"] == 4
    assert frame["season6_mission_ids"] == SEASON_6_MISSION_IDS

    html = client.get("/").text
    for needle in (
        "season6-missions-panel",
        "/api/season6-missions",
        "/api/agent-arena",
        "/api/strategy-optimizer",
        "runAgentArena",
        "runLean4Arena",
        "loadStrategyOptimizer",
        "pushStrategyTelemetry",
        "SI-01 FISHER RANKS",
        "SI-05 PARETO FRONTIER",
    ):
        assert needle in html, needle
    assert "<<<<<<<" not in html
    assert ">>>>>>>" not in html
