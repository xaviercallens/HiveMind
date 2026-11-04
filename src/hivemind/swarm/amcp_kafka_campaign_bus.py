# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Joint Subject-by-Subject Improvement Campaigns, Community Poll, Joint Calendar & AMCP v1.5 / Kafka Bus.

Implements `HM-S19-CAMPAIGNS`:
1. **Joint Subject-by-Subject Improvement Campaigns (`JOINT_SUBJECT_CAMPAIGNS`)**:
   Focuses humans (researchers, developers, citizen GPU miners), autonomous agents (Claude Code, Antigravity,
   local RTX 2070 / GCP T4 agents), and Specialized Bees on **one flagship subject per campaign sprint**,
   with explicit quantitative gates and nocturnal BOINC/QLoRA workunit allocation.
2. **Community Interest & Registration Poll (`CampaignPollVote`)**:
   Allows humans, GPU producers, AI agents, and Bees to vote and pledge VRAM for the subjects they want to
   prioritize next, dynamically re-weighting nocturnal training and updating the **Joint Campaign Calendar (`.ics`)**.
3. **AMCP v1.5 + CloudEvents 1.0 over Kafka Event Bus (`AMCPKafkaCampaignBus`)**:
   Enables structured, authenticated cross-human, cross-agent, and inter-bee communication across 4 Kafka topics:
   - `hivemind.amcp.campaigns.v1` — Poll registrations, subject sprint transitions, and calendar sync.
   - `hivemind.amcp.human_agent.v1` — Cross-human <-> cross-agent problem proposals, dataset sharing, and status updates.
   - `hivemind.amcp.bee_coord.v1` — Inter-bee (`bee <-> bee`) inference coordination & specialist sub-query handoffs
     (e.g. `hivemind/coder` delegating stiff ODE verification to `hivemind/bee-rusty-sundials` or orbital mechanics
     to `hivemind/bee-stars`).
   - `hivemind.amcp.bee_trading.v1` — Inter-bee & producer HMT capacity/adapter trading (`OFFER` vs `BID` double-auction
     clearing via Kelly/Walrasian midpoint pricing so overloaded bees can lease standby GPU slots).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

KAFKA_TOPICS: dict[str, str] = {
    "campaigns": "hivemind.amcp.campaigns.v1",
    "human_agent": "hivemind.amcp.human_agent.v1",
    "bee_coord": "hivemind.amcp.bee_coord.v1",
    "bee_trading": "hivemind.amcp.bee_trading.v1",
    "federation": "hivemind.amcp.federation.v1",  # signed task offers, claims, results and aborts for co-training rounds (see docs/FEDERATION_V2_AMCP_AGLETS.md)
}

ParticipantKind = Literal["human_researcher", "gpu_producer", "ai_agent", "bee_specialist"]
AvailabilitySlot = Literal["eu_evening_cet", "us_evening_est", "apac_evening_jst", "24x7_autonomous"]
TradeSide = Literal["OFFER", "BID"]


@dataclass(frozen=True)
class SubjectCampaign:
    """One focused Subject-by-Subject joint improvement campaign between humans, agents, and Bees."""

    subject_id: str
    order_index: int
    title: str
    domain_key: str
    lead_bee: str
    co_bees: tuple[str, ...]
    dtstart_utc: str
    dtend_utc: str
    human_window: str
    objective: str
    quantitative_gate: str
    dataset_ref: str
    base_nocturnal_weight: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["co_bees"] = list(self.co_bees)
        return d


JOINT_SUBJECT_CAMPAIGNS: tuple[SubjectCampaign, ...] = (
    SubjectCampaign(
        subject_id="SUBJ-01-PYTHON-RUST-CODER",
        order_index=1,
        title="Subject 1: Python & Rust Systems Coding Mastery",
        domain_key="coding_python",
        lead_bee="hivemind/coder",
        co_bees=("hivemind/moe-coder", "hivemind/coder-plus"),
        dtstart_utc="20261028T180000Z",
        dtend_utc="20261030T220000Z",
        human_window="Oct 28–30, 2026 · 19:00 CET / 14:00 EDT",
        objective=(
            "Co-train and benchmark Qwen 3.5 coding adapters across Python FastAPI/PyTorch and Rust Cargo/SIMD "
            "workflows with zero-stub AST verification."
        ),
        quantitative_gate="Pass >= 95% unit-test synthesis on held-out Python/Rust tasks with p95 latency < 650 ms.",
        dataset_ref="data/qwen35_coding_s19.jsonl",
        base_nocturnal_weight=0.28,
    ),
    SubjectCampaign(
        subject_id="SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE",
        order_index=2,
        title="Subject 2: Stiff ODE/PDE & Navier-Stokes Fluid Rescue (Sunday Launch Anchor)",
        domain_key="science_rusty_sundials",
        lead_bee="hivemind/bee-rusty-sundials",
        co_bees=("hivemind/coder", "hivemind/moe-general"),
        dtstart_utc="20261031T160000Z",
        dtend_utc="20261102T220000Z",
        human_window="Oct 31 – Nov 2, 2026 · 17:00 CET / 12:00 EST (International Producer Launch)",
        objective=(
            "Rescue unstable explicit Euler/RK4 fluid & chemical kinetics codes using Rusty-SUNDIALS CVODE BDF "
            "and certified Wave-1 numeric residual verification."
        ),
        quantitative_gate="100% certified residual pass (r_inf <= 1e-6) and CFL/divergence stability on Robertson & Burgers.",
        dataset_ref="data/rusty_sundials_scientific_bee/rusty_sundials_numeric_rescue_dataset.jsonl",
        base_nocturnal_weight=0.24,
    ),
    SubjectCampaign(
        subject_id="SUBJ-03-ASTRO-SYMPLECTIC-STARS",
        order_index=3,
        title="Subject 3: Exoplanet Transit Timing (TTV) & Symplectic N-Body Astrophysics",
        domain_key="science_astronomy",
        lead_bee="hivemind/bee-stars",
        co_bees=("hivemind/bee-rusty-sundials", "hivemind/coder"),
        dtstart_utc="20261103T180000Z",
        dtend_utc="20261105T220000Z",
        human_window="Nov 3–5, 2026 · 19:00 CET / 13:00 EST",
        objective=(
            "Improve orbital resonance TTV inversion and Yoshida 4th-order / Velocity-Verlet symplectic integrators "
            "with strict Hamiltonian conservation."
        ),
        quantitative_gate="Relative Hamiltonian energy drift |dH/H0| < 1e-6 across 10,000 orbital steps.",
        dataset_ref="data/rusty_sundials_scientific_bee/rusty_sundials_numeric_rescue_dataset.jsonl#astronomy",
        base_nocturnal_weight=0.18,
    ),
    SubjectCampaign(
        subject_id="SUBJ-04-QUANTUM-BEC-GROSS-PITAEVSKII",
        order_index=4,
        title="Subject 4: Quantum Fluids & Bose-Einstein Condensate Solitons",
        domain_key="science_rusty_sundials",
        lead_bee="hivemind/bee-rusty-sundials",
        co_bees=("hivemind/bee-stars", "hivemind/coder"),
        dtstart_utc="20261106T180000Z",
        dtend_utc="20261108T220000Z",
        human_window="Nov 6–8, 2026 · 19:00 CET / 13:00 EST",
        objective=(
            "Co-solve 1D/2D Gross-Pitaevskii superfluid vortex and dark-soliton dynamics using unitary Crank-Nicolson "
            "and Split-Step Fourier integrators."
        ),
        quantitative_gate="Wavefunction L2 norm conservation |dN/N0| < 1e-8 and chemical potential convergence.",
        dataset_ref="data/rusty_sundials_scientific_bee/rusty_sundials_numeric_rescue_dataset.jsonl#quantum_fluid",
        base_nocturnal_weight=0.12,
    ),
    SubjectCampaign(
        subject_id="SUBJ-05-GENETICS-WRIGHT-FISHER",
        order_index=5,
        title="Subject 5: Population Genetics & Fisher-KPP Reaction-Diffusion",
        domain_key="science_rusty_sundials",
        lead_bee="hivemind/bee-rusty-sundials",
        co_bees=("hivemind/moe-general", "hivemind/coder"),
        dtstart_utc="20261109T180000Z",
        dtend_utc="20261111T220000Z",
        human_window="Nov 9–11, 2026 · 19:00 CET / 13:00 EST",
        objective=(
            "Model multi-allele Wright-Fisher selection-mutation-drift and spatial Fisher-KPP wave propagation "
            "while preserving probability simplex invariants."
        ),
        quantitative_gate="Exact simplex conservation |sum(p_i) - 1| < 1e-12 and non-negativity p_i >= 0.",
        dataset_ref="data/rusty_sundials_scientific_bee/rusty_sundials_numeric_rescue_dataset.jsonl#genetics",
        base_nocturnal_weight=0.10,
    ),
    SubjectCampaign(
        subject_id="SUBJ-06-LEAN4-FORMAL-PROOFS",
        order_index=6,
        title="Subject 6: Formal Lean 4 Proofs & Certified Numeric Invariants",
        domain_key="coding_general",
        lead_bee="hivemind/moe-coder",
        co_bees=("hivemind/coder", "hivemind/bee-rusty-sundials"),
        dtstart_utc="20261112T180000Z",
        dtend_utc="20261115T220000Z",
        human_window="Nov 12–15, 2026 · 19:00 CET / 13:00 EST",
        objective=(
            "Eliminate sorry obligations and synthesize machine-checked Lean 4 theorems for physical conservation "
            "laws and decentralized market-clearing bounds."
        ),
        quantitative_gate="100% lake build pass with zero sorry tokens on target theorem batch.",
        dataset_ref="lean4/HiveMind/",
        base_nocturnal_weight=0.08,
    ),
)

SUBJECT_BY_ID: dict[str, SubjectCampaign] = {c.subject_id: c for c in JOINT_SUBJECT_CAMPAIGNS}


@dataclass(frozen=True)
class AMCPKafkaEvent:
    """AMCP v1.5 CloudEvents 1.0 message envelope transported over a Kafka topic."""

    event_id: str
    topic: str
    event_type: str
    sender_id: str
    sender_kind: str
    recipient_id: str
    subject_id: str
    correlation_id: str
    payload: dict[str, Any]
    utc: float
    specversion: str = "1.0"
    protocol: str = "AMCP/1.5+Kafka"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_amcp_kafka_event(
    *,
    topic_key: str,
    event_type: str,
    sender_id: str,
    sender_kind: str,
    recipient_id: str = "broadcast",
    subject_id: str = "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE",
    correlation_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
    now_utc: float | None = None,
) -> AMCPKafkaEvent:
    topic = KAFKA_TOPICS.get(topic_key, topic_key)
    stamp = time.time() if now_utc is None else float(now_utc)
    ev_id = "amcp-" + uuid.uuid4().hex[:16]
    corr = correlation_id or ("corr-" + uuid.uuid4().hex[:12])
    return AMCPKafkaEvent(
        event_id=ev_id,
        topic=topic,
        event_type=event_type,
        sender_id=str(sender_id)[:64],
        sender_kind=str(sender_kind)[:32],
        recipient_id=str(recipient_id)[:64],
        subject_id=str(subject_id)[:64],
        correlation_id=str(corr)[:64],
        payload=dict(payload or {}),
        utc=stamp,
    )


def compute_campaign_poll_summary(
    votes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate community poll registrations per subject and compute dynamic nocturnal BOINC/QLoRA weights."""
    counts_by_subject: dict[str, dict[str, Any]] = {}
    for camp in JOINT_SUBJECT_CAMPAIGNS:
        counts_by_subject[camp.subject_id] = {
            "subject_id": camp.subject_id,
            "title": camp.title,
            "lead_bee": camp.lead_bee,
            "co_bees": list(camp.co_bees),
            "dtstart_utc": camp.dtstart_utc,
            "dtend_utc": camp.dtend_utc,
            "human_window": camp.human_window,
            "total_votes": 0,
            "human_votes": 0,
            "producer_votes": 0,
            "agent_votes": 0,
            "bee_votes": 0,
            "pledged_vram_gb": 0.0,
            "availability_slots": {},
        }

    for v in votes:
        sid = str(v.get("subject_id", ""))
        if sid not in counts_by_subject:
            continue
        entry = counts_by_subject[sid]
        entry["total_votes"] += 1
        kind = str(v.get("participant_kind", "human_researcher"))
        if kind == "human_researcher":
            entry["human_votes"] += 1
        elif kind == "gpu_producer":
            entry["producer_votes"] += 1
        elif kind == "ai_agent":
            entry["agent_votes"] += 1
        elif kind == "bee_specialist":
            entry["bee_votes"] += 1
        vram = float(v.get("pledged_vram_gb", 0.0) or 0.0)
        entry["pledged_vram_gb"] = round(entry["pledged_vram_gb"] + max(0.0, vram), 2)
        slot = str(v.get("availability_slot", "eu_evening_cet"))
        slots = entry["availability_slots"]
        slots[slot] = slots.get(slot, 0) + 1

    total_votes_all = sum(e["total_votes"] for e in counts_by_subject.values())
    dynamic_weights: dict[str, float] = {}
    for camp in JOINT_SUBJECT_CAMPAIGNS:
        entry = counts_by_subject[camp.subject_id]
        if total_votes_all > 0:
            vote_share = entry["total_votes"] / total_votes_all
            # Blend 60% community poll vote share with 40% baseline roadmap schedule weight
            raw_w = 0.60 * vote_share + 0.40 * camp.base_nocturnal_weight
        else:
            raw_w = camp.base_nocturnal_weight
        dynamic_weights[camp.subject_id] = raw_w

    norm = sum(dynamic_weights.values()) or 1.0
    for sid, w in dynamic_weights.items():
        w_norm = round(w / norm, 4)
        dynamic_weights[sid] = w_norm
        counts_by_subject[sid]["nocturnal_workunit_weight"] = w_norm

    ranked = sorted(
        counts_by_subject.values(),
        key=lambda e: (e["total_votes"], e["pledged_vram_gb"], e["nocturnal_workunit_weight"]),
        reverse=True,
    )
    leading_subject = ranked[0]["subject_id"] if ranked else JOINT_SUBJECT_CAMPAIGNS[0].subject_id

    return {
        "total_registrations": total_votes_all,
        "total_pledged_vram_gb": round(sum(e["pledged_vram_gb"] for e in counts_by_subject.values()), 2),
        "leading_subject_id": leading_subject,
        "nocturnal_workunit_weights": dynamic_weights,
        "subjects": [counts_by_subject[c.subject_id] for c in JOINT_SUBJECT_CAMPAIGNS],
    }


def coordinate_inter_bee_inference(
    *,
    primary_bee: str,
    prompt: str,
    subject_id: str | None = None,
    steps: int = 32,
    dt: float = 0.01,
) -> dict[str, Any]:
    """Coordinate multi-bee inference handoff (`bee <-> bee`) over `hivemind.amcp.bee_coord.v1`.

    When a generalist or coding bee (`hivemind/coder`) receives a prompt with scientific or numeric constraints,
    it dispatches an AMCP sub-task to `hivemind/bee-rusty-sundials` or `hivemind/bee-stars`, verifies the numeric
    invariants deterministically, and returns a joint co-inference plan + CloudEvents trace.
    """
    from hivemind.router.usage_capture import classify_prompt_intent_and_language
    from hivemind.swarm.rusty_sundials_scientific_bee import (
        SolverTaughtSurrogateExpert,
        rescue_numeric_calculation,
    )

    intent = classify_prompt_intent_and_language([{"role": "user", "content": prompt}])
    specialist_bee = intent.preferred_tier or "bee-rusty-sundials"
    if not specialist_bee.startswith("hivemind/"):
        specialist_bee = f"hivemind/{specialist_bee}"
    norm_primary = primary_bee if primary_bee.startswith("hivemind/") else f"hivemind/{primary_bee}"

    # Determine matching campaign subject
    resolved_subject = subject_id
    if not resolved_subject or resolved_subject not in SUBJECT_BY_ID:
        if intent.task_domain == "science_astronomy":
            resolved_subject = "SUBJ-03-ASTRO-SYMPLECTIC-STARS"
        elif intent.task_domain == "science_rusty_sundials":
            resolved_subject = "SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE"
        else:
            resolved_subject = "SUBJ-01-PYTHON-RUST-CODER"

    corr_id = "coord-" + uuid.uuid4().hex[:12]
    handoff_event = build_amcp_kafka_event(
        topic_key="bee_coord",
        event_type="io.hivemind.amcp.bee.handoff_request",
        sender_id=norm_primary,
        sender_kind="bee_specialist",
        recipient_id=specialist_bee,
        subject_id=resolved_subject,
        correlation_id=corr_id,
        payload={
            "task_domain": intent.task_domain,
            "language": intent.language,
            "delegated_action": "numeric_invariant_rescue_and_co_inference",
            "capabilities": ["solver", "wave2_surrogate", "mol_pde"],
        },
    )

    rescue_cert = rescue_numeric_calculation(prompt, None, max(4, min(steps, 128)), dt)
    surrogate = SolverTaughtSurrogateExpert()
    surrogate_verdict = surrogate.predict_and_verify(
        prompt,
        domain=rescue_cert.domain,
        steps=max(4, min(steps, 128)),
        dt=dt,
    )
    verdict = "PASS" if rescue_cert.rescued_succeeded else "FAIL"
    reply_event = build_amcp_kafka_event(
        topic_key="bee_coord",
        event_type="io.hivemind.amcp.bee.handoff_certified",
        sender_id=specialist_bee,
        sender_kind="bee_specialist",
        recipient_id=norm_primary,
        subject_id=resolved_subject,
        correlation_id=corr_id,
        payload={
            "solver_family": rescue_cert.rescued_scheme_name,
            "domain": rescue_cert.domain,
            "recommended_crate": rescue_cert.diagnosis.recommended_rusty_sundials_crate,
            "relative_invariant_drift": rescue_cert.rescued_invariant_drift,
            "improvement_factor": rescue_cert.improvement_factor,
            "verdict": verdict,
            "capabilities": ["solver", "wave2_surrogate", "mol_pde"],
            "surrogate_accepted": surrogate_verdict["surrogate_verified"],
            "surrogate_served_by": surrogate_verdict["served_by"],
            "wilson_lower_bound_95": surrogate_verdict["wilson_lower_bound"],
        },
    )

    return {
        "object": "hivemind.bee_coordination",
        "correlation_id": corr_id,
        "subject_id": resolved_subject,
        "primary_bee": norm_primary,
        "specialist_bee": specialist_bee,
        "task_domain": intent.task_domain,
        "language": intent.language,
        "capabilities": ["solver", "wave2_surrogate", "mol_pde"],
        "numeric_certificate": {
            "domain": rescue_cert.domain,
            "recommended_solver": rescue_cert.rescued_scheme_name,
            "recommended_crate": rescue_cert.diagnosis.recommended_rusty_sundials_crate,
            "relative_invariant_drift": rescue_cert.rescued_invariant_drift,
            "improvement_factor": rescue_cert.improvement_factor,
            "verdict": verdict,
            "surrogate_verification": {
                "surrogate_accepted": surrogate_verdict["surrogate_verified"],
                "served_by": surrogate_verdict["served_by"],
                "surrogate_invariant_drift": surrogate_verdict["surrogate_invariant_drift"],
                "returned_invariant_drift": surrogate_verdict["final_verified_invariant_drift"],
                "invariant_drift_tolerance": surrogate_verdict["max_surrogate_drift_tol"],
                "wilson_lower_bound_95": surrogate_verdict["wilson_lower_bound"],
            },
        },
        "amcp_kafka_events": [handoff_event.to_dict(), reply_event.to_dict()],
    }


def clear_bee_capacity_market(
    orders: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Clear inter-bee & producer HMT capacity/adapter orders (`OFFER` vs `BID`) via double-auction midpoint pricing.

    Allows a high-demand Bee (e.g. `hivemind/coder` during Subject 1) to bid for idle GPU capacity or nocturnal
    QLoRA slots offered by another Bee/Producer (`hivemind/bee-stars` or `rtx-2070-producer`), maximizing utilization.
    """
    offers = [
        dict(o)
        for o in orders
        if str(o.get("side", "")).upper() == "OFFER" and float(o.get("capacity_1k", 0.0)) > 0.0
    ]
    bids = [
        dict(b)
        for b in orders
        if str(b.get("side", "")).upper() == "BID" and float(b.get("capacity_1k", 0.0)) > 0.0
    ]

    # Sort offers ascending by ask price, bids descending by bid price
    offers.sort(key=lambda o: (float(o.get("price_per_1k", 0.0)), float(o.get("utc", 0.0))))
    bids.sort(key=lambda b: (-float(b.get("price_per_1k", 0.0)), float(b.get("utc", 0.0))))

    matches: list[dict[str, Any]] = []
    i, j = 0, 0
    rem_offer = float(offers[0]["capacity_1k"]) if offers else 0.0
    rem_bid = float(bids[0]["capacity_1k"]) if bids else 0.0

    while i < len(offers) and j < len(bids):
        ask = float(offers[i]["price_per_1k"])
        bid = float(bids[j]["price_per_1k"])
        if bid < ask:
            break
        traded_cap = min(rem_offer, rem_bid)
        clearing_price = round(0.5 * (ask + bid), 6)
        if traded_cap > 0.0:
            matches.append(
                {
                    "offer_id": str(offers[i].get("order_id", f"off-{i}")),
                    "bid_id": str(bids[j].get("order_id", f"bid-{j}")),
                    "seller_bee": str(offers[i].get("bee_id", "hivemind/bee-stars")),
                    "buyer_bee": str(bids[j].get("bee_id", "hivemind/coder")),
                    "subject_id": str(bids[j].get("subject_id", offers[i].get("subject_id", "SUBJ-01-PYTHON-RUST-CODER"))),
                    "traded_capacity_1k": round(traded_cap, 4),
                    "clearing_price_per_1k": clearing_price,
                    "settled_hmt_per_1k_window": round(traded_cap * clearing_price, 6),
                }
            )
        rem_offer -= traded_cap
        rem_bid -= traded_cap
        if rem_offer <= 1e-9:
            i += 1
            rem_offer = float(offers[i]["capacity_1k"]) if i < len(offers) else 0.0
        if rem_bid <= 1e-9:
            j += 1
            rem_bid = float(bids[j]["capacity_1k"]) if j < len(bids) else 0.0

    total_traded = round(sum(m["traded_capacity_1k"] for m in matches), 4)
    mean_price = (
        round(sum(m["clearing_price_per_1k"] * m["traded_capacity_1k"] for m in matches) / total_traded, 6)
        if total_traded > 0.0
        else 0.0
    )

    return {
        "object": "hivemind.bee_capacity_market",
        "open_offers": len(offers),
        "open_bids": len(bids),
        "matched_trades_count": len(matches),
        "total_traded_capacity_1k": total_traded,
        "volume_weighted_clearing_price_per_1k": mean_price,
        "matches": matches,
    }


def render_joint_campaign_ics(poll_summary: Mapping[str, Any] | None = None) -> str:
    """Render an RFC 5545 iCalendar (`.ics`) feed for all 6 Subject-by-Subject Improvement Campaigns."""
    by_sid: dict[str, Mapping[str, Any]] = {}
    if poll_summary and isinstance(poll_summary.get("subjects"), list):
        for item in poll_summary["subjects"]:
            if isinstance(item, Mapping) and "subject_id" in item:
                by_sid[str(item["subject_id"])] = item

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Project HiveMind//Subject-by-Subject Joint Campaigns S19-S20//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:HiveMind Joint Subject-by-Subject Improvement Campaigns",
        "X-WR-TIMEZONE:UTC",
    ]
    for camp in JOINT_SUBJECT_CAMPAIGNS:
        stats = by_sid.get(camp.subject_id, {})
        votes = int(stats.get("total_votes", 0))
        vram = float(stats.get("pledged_vram_gb", 0.0))
        weight = float(stats.get("nocturnal_workunit_weight", camp.base_nocturnal_weight))
        desc = (
            f"Lead Bee: {camp.lead_bee} | Co-Bees: {', '.join(camp.co_bees)}\\n"
            f"Objective: {camp.objective}\\n"
            f"Gate: {camp.quantitative_gate}\\n"
            f"Community Poll: {votes} registered participants ({vram:.1f} GB VRAM pledged) | "
            f"Nocturnal BOINC/QLoRA Weight: {weight * 100:.1f}%\\n"
            f"AMCP/Kafka Topic: {KAFKA_TOPICS['campaigns']}"
        )
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{camp.subject_id.lower()}@hivemind.openrouter",
                f"DTSTAMP:20261027T160000Z",
                f"DTSTART:{camp.dtstart_utc}",
                f"DTEND:{camp.dtend_utc}",
                f"SUMMARY:[HiveMind {camp.subject_id}] {camp.title}",
                f"DESCRIPTION:{desc}",
                "LOCATION:https://hivemind-openrouter-1003063861791.us-central1.run.app/playground#campaigns",
                "STATUS:CONFIRMED",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


class InMemoryKafkaBridge:
    """Thread-safe Kafka topic ring buffer with optional `KAFKA_BOOTSTRAP_SERVERS` metadata export."""

    def __init__(self, maxlen_per_topic: int = 256) -> None:
        self._lock = threading.Lock()
        self._maxlen = maxlen_per_topic
        self._topics: dict[str, deque[dict[str, Any]]] = {
            topic: deque(maxlen=maxlen_per_topic) for topic in KAFKA_TOPICS.values()
        }
        self.bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "in-memory-amcp-broker:9092")

    def publish(self, event: AMCPKafkaEvent) -> dict[str, Any]:
        record = event.to_dict()
        with self._lock:
            bucket = self._topics.setdefault(event.topic, deque(maxlen=self._maxlen))
            bucket.append(record)
            offset = len(bucket) - 1
        return {"topic": event.topic, "partition": 0, "offset": offset, "event_id": event.event_id}

    def recent(self, topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if topic:
                resolved = KAFKA_TOPICS.get(topic, topic)
                items = list(self._topics.get(resolved, ()))
            else:
                items = [ev for bucket in self._topics.values() for ev in bucket]
        items.sort(key=lambda x: float(x.get("utc", 0.0)), reverse=True)
        return items[: max(1, min(int(limit), 200))]
