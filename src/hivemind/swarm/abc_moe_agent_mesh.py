# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Two-Layer Honeybee Hive Mind Architecture for Agent Mesh with Kafka & MoE LLMs.

Implements HM-S19-BI-LAYER-ABC:
1. **Layer 1: Network / Consensus via Kafka (Nesting Scrutiny & Quorum Sensing / Waggle Dance)**
   - Topologies:
     * `hivemind.mesh.task.inbox.v1` (Incoming raw multi-step user tasks)
     * `hivemind.mesh.waggle_dance.v1` (Scout evaluation metadata, confidence Q, token cost, tools)
     * `hivemind.mesh.quorum_consensus.v1` (Quorum aggregation & dedicated consumer group lock)
     * `hivemind.mesh.food_sources.v1` (Shared information nectar & tool outputs between bees)
     * `hivemind.mesh.result.v1` (Final validated deliverable to end-user)
   - Dynamic Waggle Dance protocol: Scout bees broadcast dance duration and intensity proportional to site quality
     (Duration = tau_0 * Q). Other bees listen and enroll (MCL recruitment) until a critical quorum threshold is reached.

2. **Layer 2: Individual Agent Execution (Adaptive Asynchronous Artificial Bee Colony / ABC)**
   - **Employed Bees (Internal MoE Exploitation):**
     Activates specialized internal MoE experts (e.g., JSON extraction expert, code expert, reasoning expert)
     to exploit the current context / food source.
   - **Observer Bees (Quality & Hallucination Assessment):**
     Reflection / critique module that evaluates solution fitness. Tracks stagnation counter (trials);
     if trials >= abandonment_limit, marks source exhausted.
   - **Scout Bees (External Tool Exploration):**
     Switches to exploration when internal experts stagnate: invokes external tools (Web Search, Solvers,
     Image generation, RAG) to discover fresh high-nectar information, restocking contextual memory.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# ==============================================================================
# Layer 1: Kafka & AMCP Topology Constants
# ==============================================================================

MESH_TOPICS: dict[str, str] = {
    "task_inbox": "hivemind.mesh.task.inbox.v1",
    "waggle_dance": "hivemind.mesh.waggle_dance.v1",
    "quorum_consensus": "hivemind.mesh.quorum_consensus.v1",
    "food_sources": "hivemind.mesh.food_sources.v1",
    "result": "hivemind.mesh.result.v1",
}

DEFAULT_HMAC_SECRET = "hivemind-amcp-mesh-hmac-sha256-secret-key-32b!"


# ==============================================================================
# Layer 1: Schemas & Pydantic Models for Waggle Dance & Quorum
# ==============================================================================


class WaggleDanceMessage(BaseModel):
    """AMCP v1.5 Waggle Dance schema broadcasted by scout bees to coordinate consensus."""

    dance_id: str = Field(default_factory=lambda: f"dance-{uuid.uuid4().hex[:12]}")
    task_id: str
    scout_bee_id: str
    task_domain: str
    confidence_score: float = Field(ge=0.0, le=1.0, description="Perceived site/task quality Q in [0, 1]")
    dance_duration_ms: int = Field(description="Dance duration in ms, proportional to confidence Q (tau_0 * Q)")
    estimated_tokens_cost: int = Field(ge=0)
    required_tools: list[str] = Field(default_factory=list)
    affinity_vector: dict[str, float] = Field(
        default_factory=dict, description="Alignment scores with internal MoE experts"
    )
    food_source_quality_prior: float = Field(default=0.5, ge=0.0, le=1.0)
    quorum_threshold: int = Field(default=2, ge=1)
    current_recruits: list[str] = Field(default_factory=list)
    quorum_status: Literal["RECRUITING", "QUORUM_REACHED", "DISMISSED"] = "RECRUITING"
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    signature_hmac: str = ""

    def sign(self, secret_key: str = DEFAULT_HMAC_SECRET) -> WaggleDanceMessage:
        """Sign canonical message content with HMAC-SHA256."""
        payload = (
            f"{self.dance_id}:{self.task_id}:{self.scout_bee_id}:{self.confidence_score}:"
            f"{self.dance_duration_ms}:{self.quorum_status}:{self.timestamp_utc}"
        )
        sig = hmac.new(secret_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        self.signature_hmac = sig
        return self

    def verify_signature(self, secret_key: str = DEFAULT_HMAC_SECRET) -> bool:
        """Verify HMAC-SHA256 signature."""
        if not self.signature_hmac:
            return False
        payload = (
            f"{self.dance_id}:{self.task_id}:{self.scout_bee_id}:{self.confidence_score}:"
            f"{self.dance_duration_ms}:{self.quorum_status}:{self.timestamp_utc}"
        )
        expected = hmac.new(secret_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature_hmac, expected)


class QuorumConsensusVerdict(BaseModel):
    """Emitted on hivemind.mesh.quorum_consensus.v1 when the waggle dance reaches threshold."""

    task_id: str
    elected_consumer_group: str
    lead_bee_id: str
    assigned_bees: list[str]
    confidence_aggregate: float
    locked_tools: list[str]
    locked_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    status: Literal["LOCKED", "REJECTED"] = "LOCKED"


# ==============================================================================
# Layer 2: Artificial Bee Colony (ABC) Food Sources & State
# ==============================================================================


@dataclass
class FoodSource:
    """Represents an information site / nectar source (web snippets, solver proofs, context)."""

    source_id: str
    source_type: Literal["web_search_snippets", "solver_proof", "rag_documents", "initial_prompt"]
    nectar_richness: float  # Quality / fitness score in [0.0, 1.0]
    content: dict[str, Any]
    exhausted: bool = False
    trial_count: int = 0
    discovered_by: str = "initial"
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )

    def increment_trial(self) -> int:
        self.trial_count += 1
        return self.trial_count

    def reset_trials(self) -> None:
        self.trial_count = 0


@dataclass
class ABCState:
    """State graph of the Artificial Bee Colony optimization loop."""

    task_id: str
    user_query: str
    target_schema: dict[str, str] = field(default_factory=dict)
    current_phase: Literal[
        "QUORUM_VOTING",
        "EMPLOYED_EXPLOIT",
        "OBSERVER_EVALUATE",
        "SCOUT_EXPLORE",
        "COMPLETED",
        "FAILED",
    ] = "QUORUM_VOTING"
    food_sources: list[FoodSource] = field(default_factory=list)
    current_source_idx: int = 0
    best_solution: dict[str, Any] | None = None
    best_quality_score: float = 0.0
    quality_threshold: float = 0.75
    abandonment_limit: int = 2  # Max trials before scout abandons site
    max_cycles: int = 12
    cycle_count: int = 0
    kafka_event_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Layer 1 Implementation: Waggle Dance Protocol & Quorum Sensing
# ==============================================================================


class WaggleDanceScrutinyProtocol:
    """Implements honeybee nesting scrutiny over Kafka topic topologies."""

    def __init__(
        self,
        tau_0_ms: int = 1000,
        secret_key: str = DEFAULT_HMAC_SECRET,
    ) -> None:
        self.tau_0_ms = tau_0_ms
        self.secret_key = secret_key

    def create_waggle_dance(
        self,
        task_id: str,
        scout_bee_id: str,
        task_domain: str,
        confidence_score: float,
        estimated_tokens: int,
        required_tools: list[str],
        affinity_vector: dict[str, float],
        quorum_threshold: int = 2,
    ) -> WaggleDanceMessage:
        """Calculate dance duration proportional to site quality (tau_0 * Q) and broadcast."""
        clamped_q = max(0.0, min(1.0, float(confidence_score)))
        duration_ms = int(round(self.tau_0_ms * clamped_q))

        dance = WaggleDanceMessage(
            task_id=task_id,
            scout_bee_id=scout_bee_id,
            task_domain=task_domain,
            confidence_score=clamped_q,
            dance_duration_ms=duration_ms,
            estimated_tokens_cost=estimated_tokens,
            required_tools=required_tools,
            affinity_vector=affinity_vector,
            quorum_threshold=quorum_threshold,
            current_recruits=[scout_bee_id],
            quorum_status="RECRUITING",
        )
        return dance.sign(self.secret_key)

    def process_peer_recruitment(
        self,
        dance: WaggleDanceMessage,
        available_peers: list[dict[str, Any]],
    ) -> tuple[WaggleDanceMessage, QuorumConsensusVerdict | None]:
        """Simulate blind recruitment imitation: peers join if peer affinity matches required tools."""
        recruits = list(dance.current_recruits)
        for peer in available_peers:
            pid = str(peer.get("bee_id", "peer-bee"))
            if pid in recruits:
                continue
            affinity = float(peer.get("affinity", 0.5))
            # Probability of recruitment proportional to dance confidence Q and peer affinity
            prob_recruit = dance.confidence_score * affinity
            if prob_recruit >= 0.35:
                recruits.append(pid)
                if len(recruits) >= dance.quorum_threshold:
                    break

        dance.current_recruits = recruits
        verdict: QuorumConsensusVerdict | None = None

        if len(recruits) >= dance.quorum_threshold:
            dance.quorum_status = "QUORUM_REACHED"
            verdict = QuorumConsensusVerdict(
                task_id=dance.task_id,
                elected_consumer_group=f"cg-hivemind-{dance.task_domain}-{dance.task_id[:8]}",
                lead_bee_id=dance.scout_bee_id,
                assigned_bees=recruits,
                confidence_aggregate=round(dance.confidence_score, 4),
                locked_tools=dance.required_tools,
                status="LOCKED",
            )
        else:
            dance.quorum_status = "RECRUITING"

        dance.sign(self.secret_key)
        return dance, verdict


# ==============================================================================
# Layer 2 Implementation: Artificial Bee Colony (ABC) MoE Execution Loop
# ==============================================================================


class ABCMoEAgent:
    """Autonomous MoE Agent executing the Artificial Bee Colony optimization loop."""

    def __init__(
        self,
        agent_id: str = "hivemind/qwen35-4b",
        abandonment_limit: int = 2,
        quality_threshold: float = 0.75,
        web_search_fn: Callable[[str], list[dict[str, str]]] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.abandonment_limit = abandonment_limit
        self.quality_threshold = quality_threshold
        self.web_search_fn = web_search_fn or self._default_mock_web_search

    def _default_mock_web_search(self, query: str) -> list[dict[str, str]]:
        """Fallback deterministic research snippets when real search is omitted."""
        terms = query.lower()
        if "battery" in terms or "batterie" in terms:
            return [
                {
                    "title": "Solid-State Li-S Battery Formulations (2026)",
                    "snippet": "Solid-state lithium-sulfur with sulfide electrolytes (Li6PS5Cl) exhibit 450 Wh/kg and stable cycling under 3.5 MPa pressure.",
                    "source": "https://doi.org/10.1038/s41560-026-00123-x",
                },
                {
                    "title": "Silicon-Graphene Anode Chemistry & Binder Stabilities",
                    "snippet": "Polyacrylic acid (PAA) crosslinked with carboxymethyl cellulose prevents 300% volume expansion cracking during delithiation.",
                    "source": "https://doi.org/10.1016/j.ensm.2026.04.015",
                },
            ]
        return [
            {
                "title": f"Recent Empirical Findings on {query[:30]}",
                "snippet": f"Validated experimental parameters and schema requirements for {query[:40]}.",
                "source": "https://arxiv.org/abs/2609.35958",
            }
        ]

    # --- ABC Step 1: Employed Bee Phase (Internal MoE Expert Exploitation) ---
    def employed_exploit_step(self, state: ABCState) -> ABCState:
        """Exploits the current food source using internal MoE experts."""
        if not state.food_sources:
            # No food source yet: trigger scout immediately
            state.current_phase = "SCOUT_EXPLORE"
            return state

        current_source = state.food_sources[state.current_source_idx]
        # Simulate MoE internal expert forward pass on current source content
        extracted_fields: dict[str, Any] = {}
        for key in state.target_schema:
            content_str = json.dumps(current_source.content).lower()
            if key.lower() in content_str or "formulation" in key or "stability" in key:
                extracted_fields[key] = f"Extracted[{key}] from {current_source.source_type}"
            else:
                extracted_fields[key] = None

        candidate_solution = {
            "source_id": current_source.source_id,
            "source_type": current_source.source_type,
            "data": extracted_fields,
            "moe_expert_activated": "expert_json_semantic_extractor",
        }

        state.best_solution = candidate_solution
        state.current_phase = "OBSERVER_EVALUATE"
        return state

    # --- ABC Step 2: Observer Bee Phase (Quality Assessment & Abandonment Test) ---
    def observer_evaluate_step(self, state: ABCState) -> ABCState:
        """Assesses quality of candidate solution and increments abandonment counter if stagnant."""
        current_source = state.food_sources[state.current_source_idx]
        solution = state.best_solution or {}
        data = solution.get("data", {})

        # Compute nectar richness: proportion of non-null fields + source quality prior
        matched_count = sum(1 for v in data.values() if v is not None)
        total_fields = max(1, len(state.target_schema))
        field_coverage = matched_count / total_fields

        # Combined fitness function: f(source)
        fitness = round(0.6 * field_coverage + 0.4 * current_source.nectar_richness, 4)

        if fitness > state.best_quality_score:
            state.best_quality_score = fitness
            current_source.reset_trials()

        if fitness >= state.quality_threshold:
            state.current_phase = "COMPLETED"
            state.kafka_event_log.append(
                {
                    "topic": MESH_TOPICS["result"],
                    "event_type": "io.hivemind.mesh.task_completed",
                    "fitness": fitness,
                    "solution": solution,
                }
            )
            return state

        # If below threshold, increment trials on current food source
        trials = current_source.increment_trial()
        if trials >= state.abandonment_limit:
            current_source.exhausted = True
            state.current_phase = "SCOUT_EXPLORE"
            state.kafka_event_log.append(
                {
                    "topic": MESH_TOPICS["food_sources"],
                    "event_type": "io.hivemind.mesh.food_source_exhausted",
                    "source_id": current_source.source_id,
                    "trials": trials,
                    "action": "trigger_scout_external_tools",
                }
            )
        else:
            state.current_phase = "EMPLOYED_EXPLOIT"

        return state

    # --- ABC Step 3: Scout Bee Phase (External Tool Exploration) ---
    def scout_explore_step(self, state: ABCState) -> ABCState:
        """Invokes external tools (Web Search / API / Solvers) to discover fresh high-nectar source."""
        query = state.user_query
        web_snippets = self.web_search_fn(query)

        new_source_id = f"source-web-{uuid.uuid4().hex[:8]}"
        fresh_source = FoodSource(
            source_id=new_source_id,
            source_type="web_search_snippets",
            nectar_richness=0.92,  # Fresh web snippets provide high potential nectar
            content={"query": query, "snippets": web_snippets},
            exhausted=False,
            trial_count=0,
            discovered_by=self.agent_id,
        )

        state.food_sources.append(fresh_source)
        state.current_source_idx = len(state.food_sources) - 1
        state.current_phase = "EMPLOYED_EXPLOIT"

        state.kafka_event_log.append(
            {
                "topic": MESH_TOPICS["food_sources"],
                "event_type": "io.hivemind.mesh.new_food_source_discovered",
                "source_id": new_source_id,
                "tool_used": "web_search",
                "snippets_count": len(web_snippets),
            }
        )
        return state

    # --- StateGraph Run Cycle ---
    def run(self, initial_state: ABCState) -> ABCState:
        """Executes the ABC loop until completion or maximum cycle threshold."""
        state = initial_state
        while state.current_phase not in ("COMPLETED", "FAILED") and state.cycle_count < state.max_cycles:
            state.cycle_count += 1
            if state.current_phase == "EMPLOYED_EXPLOIT":
                state = self.employed_exploit_step(state)
            elif state.current_phase == "OBSERVER_EVALUATE":
                state = self.observer_evaluate_step(state)
            elif state.current_phase == "SCOUT_EXPLORE":
                state = self.scout_explore_step(state)
            elif state.current_phase == "QUORUM_VOTING":
                # Start by exploring or exploiting depending on initial food sources
                if not state.food_sources:
                    state.current_phase = "SCOUT_EXPLORE"
                else:
                    state.current_phase = "EMPLOYED_EXPLOIT"

        if state.current_phase != "COMPLETED":
            state.current_phase = "FAILED"
        return state


# ==============================================================================
# End-to-End Orchestrator: Layer 1 (Kafka Consensus) + Layer 2 (ABC MoE Loop)
# ==============================================================================


class AgentMeshOrchestrator:
    """Orchestrates Layer 1 Waggle Dance Quorum and Layer 2 ABC MoE Execution Mesh."""

    def __init__(
        self,
        lead_bee_id: str = "hivemind/qwen35-4b",
        available_peers: list[dict[str, Any]] | None = None,
    ) -> None:
        self.lead_bee_id = lead_bee_id
        self.protocol = WaggleDanceScrutinyProtocol(tau_0_ms=1000)
        self.available_peers = available_peers or [
            {"bee_id": "hivemind/phi4-mini", "affinity": 0.85},
            {"bee_id": "hivemind/gemma4-e4b", "affinity": 0.90},
            {"bee_id": "hivemind/qwen35-sub2b", "affinity": 0.70},
        ]
        self.abc_agent = ABCMoEAgent(agent_id=lead_bee_id)

    def execute_mesh_workflow(
        self,
        task_id: str,
        user_query: str,
        target_schema: dict[str, str],
        confidence_prior: float = 0.85,
        required_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Runs the complete two-layer workflow from Waggle Dance consensus to ABC deliverable."""
        tools = required_tools or ["web_search", "json_extractor"]

        # Step 1: Couche 1 - Émission de la Danse Frétillante (Waggle Dance)
        dance = self.protocol.create_waggle_dance(
            task_id=task_id,
            scout_bee_id=self.lead_bee_id,
            task_domain="scientific_exploration_and_synthesis",
            confidence_score=confidence_prior,
            estimated_tokens=512,
            required_tools=tools,
            affinity_vector={"web_search": 0.90, "moe_extract": 0.85, "solver": 0.40},
            quorum_threshold=2,
        )

        # Step 2: Couche 1 - Recrutement et Atteinte du Quorum
        recruited_dance, verdict = self.protocol.process_peer_recruitment(
            dance=dance,
            available_peers=self.available_peers,
        )

        if not verdict or verdict.status != "LOCKED":
            return {
                "task_id": task_id,
                "status": "QUORUM_FAILED",
                "dance_message": recruited_dance.model_dump(),
                "verdict": None,
                "result": None,
            }

        # Step 3: Couche 2 - Initialisation de l'état ABC
        initial_food_source = FoodSource(
            source_id=f"source-init-{task_id[:6]}",
            source_type="initial_prompt",
            nectar_richness=0.30,  # Faible nectar initial pour forcer l'exploration Éclaireuse
            content={"prompt": user_query},
            exhausted=False,
            trial_count=0,
            discovered_by="user_inbox",
        )

        abc_state = ABCState(
            task_id=task_id,
            user_query=user_query,
            target_schema=target_schema,
            current_phase="QUORUM_VOTING",
            food_sources=[initial_food_source],
            quality_threshold=0.75,
            abandonment_limit=2,
            max_cycles=12,
        )

        # Step 4: Couche 2 - Exécution de la boucle ABC MoE
        final_state = self.abc_agent.run(abc_state)

        return {
            "task_id": task_id,
            "status": final_state.current_phase,
            "layer_1_consensus": {
                "waggle_dance": recruited_dance.model_dump(),
                "quorum_verdict": verdict.model_dump(),
            },
            "layer_2_abc_execution": {
                "total_cycles": final_state.cycle_count,
                "best_quality_score": final_state.best_quality_score,
                "food_sources_count": len(final_state.food_sources),
                "solution": final_state.best_solution,
                "event_log": final_state.kafka_event_log,
            },
        }
