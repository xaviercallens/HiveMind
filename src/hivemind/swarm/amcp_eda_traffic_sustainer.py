# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""AMCP v1.5 (Agent Mesh Communication Protocol) & CloudEvents 1.0 Event-Driven Architecture (EDA) Traffic Sustainer.

Integrates the core architectural primitives from `xaviercallens/amcp` & `xaviercallens/amcp-v1.5-opensource`
with `xaviercallens/HiveMind` to sustain high-concurrency international traffic waves starting with the
**Sunday, November 1, 2026 Producer-First Launch** and the **Nov 2-8 Consumer + Prosumer Mix Wave**:

1. Leveraged AMCP Pull Requests (`xaviercallens/amcp` & `xaviercallens/amcp-v1.5-opensource`):
   - `amcp-v1.5-opensource#1` ("AMCP v1.5 Release - CloudEvents 1.0 Compliance & Developer Experience"):
     Standardizes asynchronous event envelopes (`io.hivemind.amcp.*`) across router, coordinator, and bees.
   - `amcp#1` ("Enhanced Kafka/NATS/InMemory EventBroker - Enterprise Integration"):
     Decouples synchronous HTTP bursts (`10-25 req/s`) from GPU execution via topic-partitioned EDA queues.
   - `amcp#2` ("Advanced Security Suite - MFA, mTLS, JWT/OIDC, RBAC & Audit"):
     Reinforces producer heartbeat attestation, OIDC sign-in (`#151`), and non-monetary HMT audit trails.
   - `amcp#3` & `amcp#6` ("Enterprise Testing Framework & Chaos/Load Suite"):
     Pre-launch GameDay chaos and burst-load verification before each international wave.
   - `amcp#8` ("AMCP v1.5 Multi-Agent Communication System with Ollama Integration & Strong Mobility"):
     Provides IBM Aglet-style strong mobility primitives (`dispatch()`, `clone()`, `replicate()`, `migrate()`,
     `retract()`, `federateWith()`) and Google A2A bridge to dynamically replicate hot 18 MB QLoRA Bee adapters
     across standby Ollama/vLLM producer GPUs when topic queue depth rises.

2. Leveraged HiveMind Pull Requests (`xaviercallens/HiveMind`):
   - `HiveMind#249` (`HM-S16-06`): Loaded-model residency probing (`GET /api/ps`), `503 model_loading` with
     `Retry-After`, and non-evicting background warm-up (`max_tokens=1`).
   - `HiveMind#206`: Multi-endpoint producer heartbeats (up to 8 GPUs/Bees per account).
   - `HiveMind#228` & `#245`: Big-GPU plan class (`12-24 GB+` Qwen 3.5 14B & 30B-A3B-MoE) + SVD Rank-16 projection.
   - `HiveMind#230` & `#232`: Nocturnal community contributors, prorated fair-share leaderboard & BYOB (`Crowd-v1`).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AMCPMobilityVerb = Literal["dispatch", "clone", "replicate", "migrate", "retract", "federateWith"]

LEVERAGED_AMCP_AND_HIVEMIND_PRS: tuple[dict[str, str], ...] = (
    {
        "pr_ref": "xaviercallens/amcp-v1.5-opensource#1",
        "component": "CloudEvents 1.0 EDA Envelope & Async Topic Broker",
        "role_in_launch": "Wraps every inference burst, warm-up signal, and prosumer activation in a CloudEvents 1.0 envelope with correlation tracking.",
    },
    {
        "pr_ref": "xaviercallens/amcp#1",
        "component": "Enhanced Multi-Broker EventDriven Architecture (InMemory / NATS / Kafka)",
        "role_in_launch": "Absorbs 10-25 req/s viral English traffic spikes into non-blocking topic partitions so HTTP workers never exhaust thread pools.",
    },
    {
        "pr_ref": "xaviercallens/amcp#2",
        "component": "Advanced Security Suite (OIDC/JWT, RBAC, Audit & Rate-Limiting)",
        "role_in_launch": "Protects the 15+85 HMT progressive starter drip against Sybil/bot farms while auditing every cross-node federation event.",
    },
    {
        "pr_ref": "xaviercallens/amcp#3 & amcp#6",
        "component": "Enterprise Chaos & High-Concurrency Load Testing Suite",
        "role_in_launch": "Executes synthetic 8,000-to-25,000 visitor chaos drills on Saturday Oct 31 before the Sunday Nov 1 Producer-First Launch.",
    },
    {
        "pr_ref": "xaviercallens/amcp#8",
        "component": "AMCP v1.5 Multi-Agent System with Ollama Integration & IBM Aglet Strong Mobility",
        "role_in_launch": "Executes `replicate()`, `migrate()`, and `federateWith()` to auto-clone hot 18 MB QLoRA coding Bees onto standby Ollama GPUs.",
    },
    {
        "pr_ref": "xaviercallens/HiveMind#249 (HM-S16-06)",
        "component": "Loaded-Model Residency Routing & Non-Evicting Background Warm-Up",
        "role_in_launch": "Routes live traffic strictly to resident VRAM models first and triggers 1-token background warm-ups on idle Ollama nodes instead of 504 timeouts.",
    },
    {
        "pr_ref": "xaviercallens/HiveMind#206, #228, #232, #245",
        "component": "Multi-Endpoint Producers, Big-GPU Qwen 3.5 Tiers & Prorated BYOB Leaderboard",
        "role_in_launch": "Allows Big Producers (12-24 GB+) to serve multiple Qwen 3.5 Bees simultaneously and rewards Sunday Nov 1 producers with prorated HMT.",
    },
)


@dataclass(frozen=True)
class CloudEventEnvelope:
    """CloudEvents v1.0 compliant event envelope (ported from AMCP v1.5 PR #1)."""

    id: str
    source: str
    type: str
    time: float
    correlation_id: str
    subject: str
    data: dict[str, Any]
    specversion: str = "1.0"
    datacontenttype: str = "application/json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AMCPMobilityCommand:
    """IBM Aglet-style strong agent mobility command (ported from AMCP v1.5 PR #8)."""

    verb: AMCPMobilityVerb
    bee_id: str
    model_id: str
    source_node_id: str
    target_node_id: str
    adapter_size_mib: float
    expected_warm_latency_s: float
    reason: str
    event_id: str


@dataclass(frozen=True)
class ElasticEDAScalingResult:
    """Quantitative result of sustaining a traffic wave using AMCP Strong Mobility + EDA + PR #249 residency."""

    wave_name: str
    base_resident_producers_c0: int
    standby_prosumer_pool: int
    activated_amcp_replicas: int
    effective_active_producers_c: int
    peak_arrival_req_per_s: float
    static_service_rate_req_per_s: float
    elastic_service_rate_req_per_s: float
    static_utilization_rho: float
    elastic_eda_utilization_rho: float
    eda_buffered_burst_requests: int
    cold_start_504_prevented_by_pr249: int
    mobility_commands_emitted: list[dict[str, Any]]
    sustained_without_drops: bool


class AMCPEventDrivenTrafficSustainer:
    """Event-Driven Architecture (EDA) broker and AMCP Strong Mobility controller for HiveMind."""

    def __init__(self, max_buffer_events: int = 4096) -> None:
        self.max_buffer_events = max_buffer_events
        self._event_log: deque[CloudEventEnvelope] = deque(maxlen=max_buffer_events)
        self._mobility_history: list[AMCPMobilityCommand] = []

    def publish_cloudevent(
        self,
        event_type: str,
        source: str,
        subject: str,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> CloudEventEnvelope:
        """Publish a CloudEvents 1.0 compliant event onto the AMCP/EDA bus."""
        if not event_type.startswith("io.hivemind.amcp."):
            raise ValueError("event_type must use the 'io.hivemind.amcp.*' CloudEvents namespace")
        env = CloudEventEnvelope(
            id=f"ce-{uuid.uuid4().hex[:16]}",
            source=source,
            type=event_type,
            time=time.time(),
            correlation_id=correlation_id or f"corr-{uuid.uuid4().hex[:12]}",
            subject=subject,
            data=dict(data),
        )
        self._event_log.append(env)
        return env

    def plan_elastic_bee_mobility(
        self,
        wave_name: str,
        base_resident_producers_c0: int,
        standby_prosumer_pool: int,
        peak_arrival_req_per_s: float,
        mean_tokens_per_req: float = 168.3,
        mean_producer_tok_per_s: float = 55.0,
        target_rho: float = 0.65,
        hot_bee_id: str = "bee_qwen35_py_ts_fullstack",
        hot_model_id: str = "hivemind/coder",
    ) -> ElasticEDAScalingResult:
        """Compute and emit AMCP `replicate()`, `migrate()`, and `federateWith()` events to sustain peak traffic."""
        mu_single = mean_producer_tok_per_s / mean_tokens_per_req
        static_mu_tot = base_resident_producers_c0 * mu_single
        static_rho = peak_arrival_req_per_s / max(static_mu_tot, 1e-6)

        # Required total servers c(t) so utilization stays <= target_rho (0.65)
        required_c = int(math.ceil(peak_arrival_req_per_s / max(target_rho * mu_single, 1e-6)))
        needed_replicas = max(0, required_c - base_resident_producers_c0)
        activated_replicas = min(standby_prosumer_pool, needed_replicas)
        effective_c = base_resident_producers_c0 + activated_replicas

        elastic_mu_tot = effective_c * mu_single
        elastic_rho = peak_arrival_req_per_s / max(elastic_mu_tot, 1e-6)

        commands: list[AMCPMobilityCommand] = []
        for idx in range(activated_replicas):
            verb: AMCPMobilityVerb = "replicate" if idx % 3 != 2 else "federateWith"
            ev = self.publish_cloudevent(
                event_type=f"io.hivemind.amcp.bee.{verb.lower()}",
                source="urn:hivemind:queenbee:amcp-coordinator",
                subject=f"{hot_model_id}:{hot_bee_id}",
                data={
                    "verb": verb,
                    "bee_id": hot_bee_id,
                    "model_id": hot_model_id,
                    "replica_index": idx + 1,
                    "pr249_warmup_max_tokens": 1,
                    "adapter_size_mib": 18.0,
                },
            )
            cmd = AMCPMobilityCommand(
                verb=verb,
                bee_id=hot_bee_id,
                model_id=hot_model_id,
                source_node_id="queenbee-vault-eu-west1",
                target_node_id=f"amcp-ollama-prosumer-{idx + 1:03d}",
                adapter_size_mib=18.0,
                expected_warm_latency_s=1.25,
                reason=f"Sustaining {wave_name}: static_rho={static_rho:.2f} -> elastic_rho={elastic_rho:.2f}",
                event_id=ev.id,
            )
            commands.append(cmd)
            self._mobility_history.append(cmd)

        # EDA burst buffer absorbs short sub-minute spikes while PR #249 background warm-up loads replicas
        buffered_requests = int(round(peak_arrival_req_per_s * 12.0))
        prevented_504s = activated_replicas * 14

        return ElasticEDAScalingResult(
            wave_name=wave_name,
            base_resident_producers_c0=base_resident_producers_c0,
            standby_prosumer_pool=standby_prosumer_pool,
            activated_amcp_replicas=activated_replicas,
            effective_active_producers_c=effective_c,
            peak_arrival_req_per_s=round(peak_arrival_req_per_s, 4),
            static_service_rate_req_per_s=round(static_mu_tot, 4),
            elastic_service_rate_req_per_s=round(elastic_mu_tot, 4),
            static_utilization_rho=round(static_rho, 4),
            elastic_eda_utilization_rho=round(elastic_rho, 4),
            eda_buffered_burst_requests=buffered_requests,
            cold_start_504_prevented_by_pr249=prevented_504s,
            mobility_commands_emitted=[asdict(c) for c in commands[:6]],
            sustained_without_drops=elastic_rho <= 0.78,
        )

    @property
    def recent_events(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._event_log]


def build_sunday_producer_first_international_plan() -> dict[str, Any]:
    """Build the complete Sunday Nov 1 Producer-First -> Next Week Nov 2-8 Consumer/Mix Launch Plan."""
    sustainer = AMCPEventDrivenTrafficSustainer()
    wave_results = [
        asdict(
            sustainer.plan_elastic_bee_mobility(
                wave_name="Sunday Nov 1: Producer-First Weekend Launch (r/BOINC, r/homelab, r/gpumining, CrowdGPT)",
                base_resident_producers_c0=16,
                standby_prosumer_pool=30,
                peak_arrival_req_per_s=2.4,
            )
        ),
        asdict(
            sustainer.plan_elastic_bee_mobility(
                wave_name="Tue-Thu Nov 3-5: Technical Prosumer/Mix Wave (r/LocalLLaMA, r/MachineLearning, 3000 Devs)",
                base_resident_producers_c0=38,
                standby_prosumer_pool=75,
                peak_arrival_req_per_s=11.5,
            )
        ),
        asdict(
            sustainer.plan_elastic_bee_mobility(
                wave_name="Fri-Sun Nov 6-8: Broad International Consumer/Mix Wave (Hacker News Show HN, X/Twitter, Dev.to)",
                base_resident_producers_c0=85,
                standby_prosumer_pool=160,
                peak_arrival_req_per_s=24.0,
            )
        ),
        asdict(
            sustainer.plan_elastic_bee_mobility(
                wave_name="Global Scale Gate: 100 Resident Producer Nodes + 160 Standby AMCP Prosumers (EU/US/APAC)",
                base_resident_producers_c0=100,
                standby_prosumer_pool=160,
                peak_arrival_req_per_s=28.5,
            )
        ),
    ]

    schedule = [
        {
            "phase_id": "PHASE-0-PREP",
            "dates": "Tue Oct 27 – Sat Oct 31, 2026",
            "focus": "Pre-Weekend Engineering & AMCP/EDA Integration",
            "audience_mix": "100% Internal & Existing 47 Pioneer Nodes",
            "actions": [
                "Merge HiveMind PR #249 (HM-S16-06 loaded-model residency, 503 Retry-After & 1-token background warm-up).",
                "Wire AMCP v1.5 CloudEvents 1.0 EDA broker (`amcp-v1.5-opensource#1`, `amcp#1`) + Aglet strong mobility (`amcp#8`).",
                "Deploy Qwen 3.5 14B/30B-A3B Big-GPU tiers (#239-#242) and run Saturday Oct 31 AMCP Chaos/Load Drill (`amcp#3/#6`).",
            ],
            "target_gate": "0% HTTP 5xx under synthetic 15 req/s burst; PR #249 warm-up verified; 15+85 HMT progressive drip active.",
        },
        {
            "phase_id": "PHASE-1-SUNDAY-PRODUCER-LAUNCH",
            "dates": "Sunday, November 1, 2026 (Weekend Launch Day)",
            "focus": "PRODUCER-FIRST International Weekend Launch (Build the Supply Cushion First)",
            "audience_mix": "80% GPU Producers & Prosumers / 20% Technical Early Adopters",
            "actions": [
                "10:00 UTC — Launch on GPU & Distributed Computing communities: r/BOINC, r/homelab, r/gpumining, and @Vxtzq's 100-node CrowdGPT pool.",
                "14:00 UTC — Open Sunday Producer Onboarding for the 3,000 internal developers ('Plug in your idle home GPU on Sunday, get unlimited Fast-Lane Qwen 3.5 Coding all week').",
                "22:00 CET (Sunday Night) — Execute the first 35+ GPU Nocturnal Qwen 3.5 MoE + SVD Rank-32 -> Rank-16 Distillation run.",
            ],
            "target_gate": ">= 38 active resident GPU producers + >= 30 standby AMCP prosumer nodes (>= 2,200 tok/s daytime capacity, >= 1.8 PFLOPS peak).",
        },
        {
            "phase_id": "PHASE-2-PROSUMER-MIX-WAVE",
            "dates": "Mon Nov 2 – Thu Nov 5, 2026 (Early/Mid Next Week)",
            "focus": "PROSUMER + CONSUMER MIX Wave (Technical AI & Developer Communities)",
            "audience_mix": "35% Prosumers & GPU Builders / 65% Technical Consumers",
            "actions": [
                "Publish v3.9.0 Foundation Paper & Qwen 3.5 6.5 GB WDDM Engineering Showcase on r/LocalLLaMA, r/MachineLearning, and r/SelfHosted.",
                "AMCP Strong Mobility (`replicate()`, `federateWith()`) auto-clones hot 18 MB Coding Bee adapters onto standby Ollama nodes whenever queue utilization rho > 0.65.",
                "Progressive 15 HMT trial + 85 HMT instant GPU-heartbeat unlock converts >= 9% of pure consumers into Night/Idle Prosumers.",
            ],
            "target_gate": "Sustain 11.5 req/s peak (8,000+ visitors) at elastic_rho <= 0.50, p50 < 220 ms, >= 85 resident producers + 150 prosumers (~5.6 PFLOPS).",
        },
        {
            "phase_id": "PHASE-3-GLOBAL-CONSUMER-MIX-WAVE",
            "dates": "Fri Nov 6 – Sun Nov 8, 2026 (Next Weekend Global Wave)",
            "focus": "BROAD CONSUMER + PROSUMER MIX Launch (Show HN, X/Twitter, Dev.to)",
            "audience_mix": "15% Producers & Converted Prosumers / 85% Global Developer Consumers",
            "actions": [
                "Launch 'Show HN: HiveMind@Home — A Citizen BOINC + OpenRouter Swarm where your idle night GPU pays for your daytime coding AI'.",
                "EDA CloudEvents broker buffers 24 req/s bursts; Congestion Kelly Surge pricing (omega(rho)) boosts daytime producer HMT rewards by 2.8x to pull all 160 standby prosumer GPUs online.",
                "Sunday Night Nov 8 — Global 100-Producer Daytime / 200+ GPU Nocturnal MoE Consolidation (> 11 PFLOPS).",
            ],
            "target_gate": "Sustain 24.0-28.5 req/s peak (25,000+ visitors) across 100 active producer nodes at elastic_rho <= 0.45, 0% cold-start 504s (PR #249), self-sustaining supply ratio >= 12.5%.",
        },
    ]

    payload = {
        "plan_id": "HM-S19-AMCP-EDA-SUNDAY-LAUNCH",
        "sunday_launch_date": "2026-11-01",
        "leveraged_prs": list(LEVERAGED_AMCP_AND_HIVEMIND_PRS),
        "elastic_eda_wave_simulations": wave_results,
        "sunday_to_next_week_schedule": schedule,
        "hundred_producer_mesh_scaling": plan_100_producer_mesh_scaling(active_producers=100),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["plan_sha256"] = digest
    return payload


def select_100_node_optimal_producer(
    candidates: list[dict[str, Any]],
    model_id: str,
    client_region: str = "eu-west",
    seed: int = 42,
) -> dict[str, Any]:
    """Power-of-Two-Choices (P2C) + HM-S16-06 Residency-Aware + Geo-Affinity Producer Dispatcher (#249, #251).

    Given up to 100 active producer accounts (up to 800 endpoints at `MAX_ENDPOINTS_PER_ACCOUNT = 8`):
    1. Filters out paused nodes (`paused == True`) or zero-capacity endpoints (`capacity_1k <= 0`).
    2. Partition candidates by HM-S16-06 loaded-model residency tier:
       - Tier 0 (`loaded is True`): model confirmed warm in VRAM (`GET /api/ps`) -> zero cold-start latency.
       - Tier 1 (`loaded is None`): legacy/unprobed endpoint -> fallback candidate.
       - Tier 2 (`loaded is False`): model currently cold on disk -> avoided if Tier 0/1 exist, triggers background warm-up.
    3. Within the best non-empty residency tier, evaluates candidates using deterministic Power-of-Two-Choices (P2C)
       augmented by regional affinity (`region == client_region`) and normalized inflight queue load:
       `score = (inflight + 1.0) / max(1.0, capacity_1k) + (0.0 if same_region else 0.15) + rtt_ms / 1000.0`.
    """
    eligible: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        if bool(cand.get("paused", False)):
            continue
        cap = int(cand.get("capacity_1k", 10))
        if cap <= 0:
            continue
        models = cand.get("models")
        if isinstance(models, (list, tuple, set)) and model_id and model_id not in models:
            continue
        residency_map = cand.get("residency")
        if isinstance(residency_map, dict) and model_id in residency_map:
            loaded_val = residency_map[model_id]
        else:
            loaded_val = cand.get("loaded")
        if loaded_val is True:
            res_tier = 0
        elif loaded_val is None:
            res_tier = 1
        else:
            res_tier = 2

        reg = str(cand.get("region", "eu-west"))
        same_region = reg == client_region
        inflight = max(0, int(cand.get("inflight", 0)))
        rtt_ms = float(cand.get("rtt_ms", 20.0 if same_region else 85.0))
        load_score = (inflight + 1.0) / max(1.0, float(cap)) + (0.0 if same_region else 0.15) + (rtt_ms / 1000.0)
        eligible.append(
            {
                "index": idx,
                "account_id": str(cand.get("account_id", f"producer-{idx}")),
                "endpoint_id": str(cand.get("endpoint_id", "default")),
                "region": reg,
                "same_region": same_region,
                "residency_tier": res_tier,
                "loaded": loaded_val,
                "inflight": inflight,
                "capacity_1k": cap,
                "rtt_ms": round(rtt_ms, 2),
                "load_score": round(load_score, 6),
            }
        )

    if not eligible:
        raise ValueError(f"no eligible producer endpoints available for model={model_id!r}")

    best_tier = min(c["residency_tier"] for c in eligible)
    tier_pool = [c for c in eligible if c["residency_tier"] == best_tier]

    # Prefer same-region pool within the best residency tier when available, else full tier pool
    regional_tier_pool = [c for c in tier_pool if c["same_region"]]
    active_pool = regional_tier_pool if regional_tier_pool else tier_pool

    if len(active_pool) == 1:
        chosen = active_pool[0]
        p2c_pair = [chosen["account_id"]]
    else:
        # Deterministic P2C tournament across the top-ranked candidates + pseudo-random probe
        sorted_pool = sorted(active_pool, key=lambda item: (item["load_score"], item["account_id"], item["endpoint_id"]))
        probe_idx = ( abs(int(seed)) % (len(sorted_pool) - 1) ) + 1
        choice_a = sorted_pool[0]
        choice_b = sorted_pool[probe_idx]
        chosen = choice_a if choice_a["load_score"] <= choice_b["load_score"] else choice_b
        p2c_pair = [choice_a["account_id"], choice_b["account_id"]]

    return {
        "selected_producer": chosen,
        "model_id": model_id,
        "client_region": client_region,
        "best_residency_tier": best_tier,
        "total_eligible_candidates": len(eligible),
        "tier_pool_size": len(tier_pool),
        "p2c_pair_evaluated": p2c_pair,
        "warmup_recommended": best_tier == 2,
    }


def plan_100_producer_mesh_scaling(
    active_producers: int = 100,
    endpoints_per_producer: int = 2,
    heartbeat_interval_s: float = 30.0,
) -> dict[str, Any]:
    """Compute AMCP v1.5 topic-partitioned heartbeat & P2C dispatch scaling metrics for 100 producer nodes."""
    c = max(1, int(active_producers))
    ep_per_prod = max(1, min(8, int(endpoints_per_producer)))
    total_endpoints = c * ep_per_prod
    interval_s = max(5.0, float(heartbeat_interval_s))

    unbatched_wal_writes_per_s = round(total_endpoints / interval_s, 4)
    regional_partitions = 4  # eu-west, us-east, us-west, ap-northeast
    batched_wal_commits_per_s = round(regional_partitions / interval_s, 4)
    sqlite_lock_reduction_factor = round(total_endpoints / regional_partitions, 2)

    return {
        "active_producers": c,
        "endpoints_per_producer": ep_per_prod,
        "total_live_endpoints": total_endpoints,
        "max_supported_endpoints": c * 8,
        "amcp_topic_partitions": [
            "io.hivemind.amcp.heartbeat.eu-west",
            "io.hivemind.amcp.heartbeat.us-east",
            "io.hivemind.amcp.heartbeat.us-west",
            "io.hivemind.amcp.heartbeat.ap-northeast",
        ],
        "unbatched_wal_writes_per_s": unbatched_wal_writes_per_s,
        "batched_wal_commits_per_s": batched_wal_commits_per_s,
        "sqlite_lock_reduction_factor": sqlite_lock_reduction_factor,
        "p2c_routing_complexity": "O(1) per request after residency tier partitioning",
    }

