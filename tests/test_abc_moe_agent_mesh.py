# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit & Integration Tests for Two-Layer Honeybee Hive Mind Mesh (AMCP Kafka + ABC MoE)."""

import pytest

from hivemind.swarm.abc_moe_agent_mesh import (
    ABCMoEAgent,
    ABCState,
    AgentMeshOrchestrator,
    FoodSource,
    QuorumConsensusVerdict,
    WaggleDanceMessage,
    WaggleDanceScrutinyProtocol,
)


def test_waggle_dance_message_duration_and_signature():
    """Verify dance duration is strictly proportional to confidence Q (tau_0 * Q) and HMAC verified."""
    protocol = WaggleDanceScrutinyProtocol(tau_0_ms=2000)

    # Confidence 0.85 -> 1700 ms
    dance = protocol.create_waggle_dance(
        task_id="task-test-01",
        scout_bee_id="hivemind/qwen35-4b",
        task_domain="scientific_research",
        confidence_score=0.85,
        estimated_tokens=400,
        required_tools=["web_search", "solver"],
        affinity_vector={"web_search": 0.9},
        quorum_threshold=2,
    )

    assert dance.confidence_score == 0.85
    assert dance.dance_duration_ms == 1700
    assert dance.verify_signature() is True

    # Tampering with confidence score breaks signature
    dance.confidence_score = 0.50
    assert dance.verify_signature() is False


def test_quorum_sensing_and_blind_recruitment_imitation():
    """Verify scout bees recruit peers until quorum threshold triggers consumer group lock."""
    protocol = WaggleDanceScrutinyProtocol(tau_0_ms=1000)

    dance = protocol.create_waggle_dance(
        task_id="task-test-quorum",
        scout_bee_id="hivemind/qwen35-4b",
        task_domain="data_extraction",
        confidence_score=0.90,
        estimated_tokens=300,
        required_tools=["web_search", "json_extractor"],
        affinity_vector={"data_extraction": 0.95},
        quorum_threshold=3,
    )

    peers = [
        {"bee_id": "hivemind/phi4-mini", "affinity": 0.80},
        {"bee_id": "hivemind/gemma4-e4b", "affinity": 0.85},
    ]

    updated_dance, verdict = protocol.process_peer_recruitment(dance, peers)

    assert updated_dance.quorum_status == "QUORUM_REACHED"
    assert len(updated_dance.current_recruits) == 3
    assert verdict is not None
    assert verdict.status == "LOCKED"
    assert "hivemind/qwen35-4b" in verdict.assigned_bees
    assert "hivemind/phi4-mini" in verdict.assigned_bees
    assert "hivemind/gemma4-e4b" in verdict.assigned_bees
    assert verdict.elected_consumer_group.startswith("cg-hivemind-data_extraction-")


def test_abc_employed_observer_scout_transition():
    """Verify Employed exploitation, Observer stagnation detection, and Scout tool trigger."""
    agent = ABCMoEAgent(agent_id="hivemind/qwen35-4b", abandonment_limit=1, quality_threshold=0.80)

    # Initial low nectar source
    init_source = FoodSource(
        source_id="src-0",
        source_type="initial_prompt",
        nectar_richness=0.20,
        content={"prompt": "Rechercher batterie"},
    )

    state = ABCState(
        task_id="task-abc-01",
        user_query="Rechercher batterie",
        target_schema={"battery_formulation": "str", "stability_score": "float"},
        current_phase="EMPLOYED_EXPLOIT",
        food_sources=[init_source],
        quality_threshold=0.80,
        abandonment_limit=1,
    )

    # 1. Employed step activates MoE
    state = agent.employed_exploit_step(state)
    assert state.current_phase == "OBSERVER_EVALUATE"
    assert state.best_solution is not None

    # 2. Observer evaluates: fitness will be low -> increments trial -> reaches abandonment limit (1) -> SCOUT_EXPLORE
    state = agent.observer_evaluate_step(state)
    assert state.current_phase == "SCOUT_EXPLORE"
    assert init_source.exhausted is True

    # 3. Scout step explores tools: fetches fresh web snippets -> creates new FoodSource -> EMPLOYED_EXPLOIT
    state = agent.scout_explore_step(state)
    assert state.current_phase == "EMPLOYED_EXPLOIT"
    assert len(state.food_sources) == 2
    assert state.food_sources[1].source_type == "web_search_snippets"
    assert state.food_sources[1].nectar_richness == 0.92


def test_agent_mesh_orchestrator_end_to_end_convergence():
    """Verify complete two-layer pipeline from Waggle Dance consensus to ABC Web deliverable."""
    orchestrator = AgentMeshOrchestrator(lead_bee_id="hivemind/qwen35-4b")

    result = orchestrator.execute_mesh_workflow(
        task_id="task-battery-2026",
        user_query="Rechercher les dernières formulations chimiques de batteries et valider la stabilité",
        target_schema={"battery_formulation": "str", "stability_score": "float"},
        confidence_prior=0.85,
        required_tools=["web_search", "json_extractor"],
    )

    assert result["status"] == "COMPLETED"
    assert result["layer_1_consensus"]["quorum_verdict"]["status"] == "LOCKED"
    assert len(result["layer_1_consensus"]["quorum_verdict"]["assigned_bees"]) >= 2

    execution = result["layer_2_abc_execution"]
    assert execution["best_quality_score"] >= 0.75
    assert execution["food_sources_count"] >= 2  # Discovered at least one fresh web food source
    assert execution["solution"] is not None
    assert "battery_formulation" in execution["solution"]["data"]
