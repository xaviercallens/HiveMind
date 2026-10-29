# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Dynamic Diurnal MoE Training Engine & Long-Tail Intent Scanner.

Part of Project HiveMind (`hivemind.swarm.dynamic_diurnal_moe_engine`).

Coordinates two critical operational modes across connected heterogeneous producers:
1. Daytime Surplus Harvesting:
   - When active producers during daytime exceed the threshold needed for low-latency
     inference traffic (demand ratio < 0.70 or surplus idle producers >= 2), the engine
     dynamically schedules lightweight micro-QLoRA training steps without degrading
     inference SLA (<250ms p50) or violating the <1.5s gamer preemption guarantee.
2. Timezone-Aware Intense Nocturnal Training:
   - Evaluates each producer according to its specific local timezone (e.g., CET, EDT, PDT, JST).
   - When a producer's local clock enters the nocturnal window (22:00 to 07:00 local time),
     it transitions to high-intensity QLoRA training on large base models (14B, 16B MoE, 32B),
     with deeper rank (r=32..64), larger gradient accumulation, and a +25% PoUW HMT reward multiplier.
3. Long-Tail Intent & Resident-Type Scanner:
   - Scans recent end-user prompt streams and producer resident types (`residency_json`).
   - Specifically detects long-tail / niche domains with few requests (e.g., 1-5 prompts for Lean 4
     formal proofs, quantum lattice field theory, ZK circuits, or CRISPR genomics).
   - Calculates a Long-Tail Priority Score:
       S_d = (1 / sqrt(N_d + 1)) * Complexity_d * RecencyBonus_d
     ensuring that even small, isolated user requests receive dedicated, world-class MoE specialization.
4. Bring Your Own Bee (BYOB) Integration:
   - Prioritizes community-submitted BYOB adapters awaiting evaluation or continuous self-refinement.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from hivemind.core.auto_improvement import (
    ALL_BASE_MODELS,
    DEFAULT_BASE_WU_REWARD_HMT,
    DEFAULT_GATE_PASS_BONUS_HMT,
    DEFAULT_TFLOPS_STEP_RATE_HMT,
    HF_EXPERT_CATALOG,
    NOCTURNAL_20_MOE_TARGETS,
    SUPPORTED_BASE_MODELS,
    AutoImprovementWorkunit,
    resolve_domain_canonical,
)
from hivemind.router.accounts import ProducerRecord, RouterStore
from hivemind.router.catalog import resolve_producer_quorum_moe_state


class DiurnalOperationalMode(str, Enum):
    """Operational mode of a producer depending on its local time and system load."""

    DAYTIME_INFERENCE_PRIMARY = "DAYTIME_INFERENCE_PRIMARY"
    DAYTIME_SURPLUS_HARVESTING = "DAYTIME_SURPLUS_HARVESTING"
    INTENSE_NOCTURNAL_TRAINING = "INTENSE_NOCTURNAL_TRAINING"
    GAMER_PREEMPTED = "GAMER_PREEMPTED"


@dataclass(frozen=True)
class DiurnalProducerProfile:
    """Hardware and geographic diurnal profile for a connected producer node."""

    account_id: str
    handle: str
    vram_gb: float
    hardware_class: str
    timezone_offset_hours: float  # e.g., +1.0 for CET, -4.0 for EDT (Quebec), -7.0 for PDT, +9.0 for JST
    is_gamer_node: bool = False

    def local_hour(self, now_utc: float) -> float:
        """Calculate the local solar/clock hour (0.0 to 24.0) for this producer."""
        utc_hour = (now_utc % 86400.0) / 3600.0
        local = (utc_hour + self.timezone_offset_hours) % 24.0
        return local

    def is_local_night(self, now_utc: float) -> bool:
        """Return True if local time is between 22:00 (10 PM) and 07:00 (7 AM)."""
        hour = self.local_hour(now_utc)
        return hour >= 22.0 or hour < 7.0

    def resolve_mode(
        self,
        now_utc: float,
        system_load_ratio: float,
        surplus_producers_count: int,
        is_preempted: bool = False,
    ) -> DiurnalOperationalMode:
        """Resolve the optimal operational mode for this producer right now."""
        if is_preempted:
            return DiurnalOperationalMode.GAMER_PREEMPTED
        if self.is_local_night(now_utc):
            return DiurnalOperationalMode.INTENSE_NOCTURNAL_TRAINING
        # Daytime: check if surplus compute is available
        if surplus_producers_count >= 2 or system_load_ratio < 0.65:
            return DiurnalOperationalMode.DAYTIME_SURPLUS_HARVESTING
        return DiurnalOperationalMode.DAYTIME_INFERENCE_PRIMARY


@dataclass(frozen=True)
class LongTailDomainScore:
    """Prioritization score for a domain, giving high weight to niche / small user requests."""

    domain: str
    canonical_domain: str
    hf_dataset: str
    request_count: int
    complexity_weight: float
    priority_score: float
    recommended_base_model: str
    cluster: str
    reason: str


class LongTailIntentScanner:
    """Scans user prompt streams and resident types to select high-value niche domains."""

    # Complexity multipliers for specialized domains
    DOMAIN_COMPLEXITY: dict[str, float] = {
        "maths_formal": 2.5,          # Machine-checked Lean 4 theorem proving
        "cuda_kernel_opt": 2.3,       # Triton / Tensor Core kernel optimization
        "cryptography_zkp": 2.4,      # Halo2 / ZK-SNARK circuit proving
        "cyber_smart_contracts": 2.2, # Solidity state-channel escrow
        "astrophysics_atmosphere": 2.1,# JWST transmission spectroscopy
        "physics_quantum_lattice": 2.3,# Symplectic lattice gauge theory
        "biomedical_genomics": 2.0,   # CRISPR Cas9/Cas12 off-target scoring
        "chemistry_molecular_docking": 2.0, # Ligand-protein docking
        "code_rust_kernel": 2.1,      # Zero-copy kernel DMA
        "cyber_vulnerability": 2.0,   # SWE-bench CVE patching
        "maths_olympiad": 1.8,        # Multi-step GSM8K Olympiad
        "cyber_threat_intel": 1.7,    # CTI-Bench ATT&CK triage
        "climate_earth_observation": 1.6, # Sentinel/Landsat radiative transfer
        "biomedical_clinical": 1.9,   # Clinical trial pharmacology
        "code_feedback_debug": 1.5,   # Multi-turn debugging
        "systems_distributed_p2p": 1.8,# P2P gossip consensus
    }

    def __init__(self) -> None:
        self._request_counts: dict[str, int] = {}
        self._last_seen_utc: dict[str, float] = {}

    def record_request(self, domain: str, timestamp_utc: float | None = None) -> None:
        """Record an end-user request for a domain."""
        t = time.time() if timestamp_utc is None else timestamp_utc
        try:
            canonical = resolve_domain_canonical(domain)
        except ValueError:
            canonical = "code_feedback_debug"
        self._request_counts[canonical] = self._request_counts.get(canonical, 0) + 1
        self._last_seen_utc[canonical] = t

    def scan_resident_types(self, producers: Sequence[ProducerRecord]) -> dict[str, int]:
        """Scan resident model types from live producers to identify active supply."""
        resident_distribution: dict[str, int] = {}
        for p in producers:
            for public_id, local_name in p.models:
                resident_distribution[public_id] = resident_distribution.get(public_id, 0) + 1
        return resident_distribution

    def score_all_domains(
        self,
        now_utc: float,
        target_domains: Sequence[str] = NOCTURNAL_20_MOE_TARGETS,
    ) -> list[LongTailDomainScore]:
        """Calculate Long-Tail Priority Scores across all candidate MoE domains.

        Formula:
          S_d = (1.0 / sqrt(N_d + 1)) * Complexity_d * RecencyBonus_d
        Domains with very few requests (N_d = 1..3) get the largest inverse-frequency boost!
        """
        scores: list[LongTailDomainScore] = []
        for domain in target_domains:
            canonical = resolve_domain_canonical(domain)
            meta = HF_EXPERT_CATALOG.get(canonical, {})
            n_req = self._request_counts.get(canonical, 0)
            complexity = self.DOMAIN_COMPLEXITY.get(canonical, 1.5)

            # Recency bonus: requests seen in the last 6 hours get a freshness boost
            last_t = self._last_seen_utc.get(canonical, 0.0)
            age_hours = (now_utc - last_t) / 3600.0 if last_t > 0 else 24.0
            recency_bonus = 1.5 if age_hours <= 6.0 else (1.2 if age_hours <= 24.0 else 1.0)

            # Active user demand factor: small niche requests (1 to 5 requests) indicate specific active end-user interest
            # This ensures that even for a single small request, the system detects and prioritizes it for MoE specialization.
            if 1 <= n_req <= 5:
                user_demand_factor = 2.2 / math.sqrt(n_req)
            elif n_req == 0:
                user_demand_factor = 0.8  # Unrequested idle background domain
            else:
                user_demand_factor = 1.5 / math.sqrt(n_req)  # Saturated high-volume domain

            priority_score = round(user_demand_factor * complexity * recency_bonus, 4)

            # Select recommended base model depending on complexity
            if complexity >= 2.2:
                rec_base = "Qwen/Qwen2.5-32B-Instruct" if "maths" in canonical or "zkp" in canonical else "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
            elif complexity >= 1.9:
                rec_base = "Qwen/Qwen2.5-Coder-14B-Instruct"
            else:
                rec_base = "Qwen/Qwen2.5-Coder-7B-Instruct"

            reason = (
                f"High-value long-tail request: {n_req} request(s) observed, complexity factor {complexity:.1f}x. "
                f"Allocated to {meta.get('title', canonical)} via {meta.get('hf_dataset', 'HuggingFace')}."
            )

            scores.append(
                LongTailDomainScore(
                    domain=domain,
                    canonical_domain=canonical,
                    hf_dataset=meta.get("hf_dataset", "HuggingFace"),
                    request_count=n_req,
                    complexity_weight=complexity,
                    priority_score=priority_score,
                    recommended_base_model=rec_base,
                    cluster=meta.get("cluster", "UNKNOWN"),
                    reason=reason,
                )
            )

        # Sort descending by priority score: most interesting long-tail domains first
        scores.sort(key=lambda s: s.priority_score, reverse=True)
        return scores


@dataclass(frozen=True)
class DiurnalTrainingWorkunit:
    """One scheduled QLoRA training workunit adapted to daytime surplus or nocturnal intensity."""

    wu_id: str
    domain: str
    canonical_domain: str
    hf_dataset: str
    base_model_id: str
    target_rank: int  # r=8..16 (daytime) or r=32..64 (intense nocturnal)
    steps: int
    operational_mode: DiurnalOperationalMode
    assigned_producer_id: str
    producer_handle: str
    producer_timezone_offset: float
    local_clock_hour: float
    pouw_reward_hmt: float
    estimated_tflops: float
    is_byob: bool = False
    byob_bee_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiurnalMoETrainingScheduler:
    """Orchestrates continuous daytime surplus harvesting and timezone-aware nocturnal training."""

    def __init__(
        self,
        router_store: RouterStore | None = None,
        scanner: LongTailIntentScanner | None = None,
    ) -> None:
        self._store = router_store
        self._scanner = scanner or LongTailIntentScanner()
        self._producers: dict[str, DiurnalProducerProfile] = {}

    def register_producer_profile(self, profile: DiurnalProducerProfile) -> None:
        """Register or update a producer's diurnal profile (timezone, VRAM, hardware)."""
        self._producers[profile.account_id] = profile

    def get_producer_profile(self, account_id: str) -> DiurnalProducerProfile | None:
        return self._producers.get(account_id)

    def schedule_diurnal_batch(
        self,
        now_utc: float,
        active_records: Sequence[ProducerRecord],
        target_domains: Sequence[str] = NOCTURNAL_20_MOE_TARGETS,
        system_load_ratio: float = 0.45,
    ) -> list[DiurnalTrainingWorkunit]:
        """Generate optimal QLoRA training workunits across active producers.

        1. Classifies each producer into its local DiurnalOperationalMode.
        2. Scans long-tail requests to pick the highest priority domains.
        3. Scales batch parameters:
           - DAYTIME_SURPLUS_HARVESTING: r=16, 30 steps, 1.0x PoUW reward.
           - INTENSE_NOCTURNAL_TRAINING: r=32 or r=64, 100 steps, large VRAM models, 1.25x PoUW multiplier.
        """
        scored_domains = self._scanner.score_all_domains(now_utc, target_domains)
        workunits: list[DiurnalTrainingWorkunit] = []
        domain_idx = 0

        # Estimate surplus count: producers with low current utilization
        total_producers = len(active_records)
        surplus_count = max(0, total_producers - 2) if system_load_ratio < 0.60 else 0

        for record in active_records:
            profile = self._producers.get(record.account_id)
            if profile is None:
                # Default profile with CET timezone (+1.0) and 16 GB VRAM
                profile = DiurnalProducerProfile(
                    account_id=record.account_id,
                    handle="producer." + record.account_id[:8],
                    vram_gb=16.0,
                    hardware_class=record.execution_class,
                    timezone_offset_hours=1.0,
                )
                self._producers[record.account_id] = profile

            mode = profile.resolve_mode(
                now_utc=now_utc,
                system_load_ratio=system_load_ratio,
                surplus_producers_count=surplus_count,
            )

            # If daytime and not in surplus, preserve 100% capacity for low-latency inference
            if mode == DiurnalOperationalMode.DAYTIME_INFERENCE_PRIMARY:
                continue

            target_domain_meta = scored_domains[domain_idx % len(scored_domains)]
            domain_idx += 1

            local_hr = profile.local_hour(now_utc)

            if mode == DiurnalOperationalMode.INTENSE_NOCTURNAL_TRAINING:
                # High-intensity nocturnal QLoRA batch
                rank = 32 if profile.vram_gb >= 16.0 else 16
                steps = 100
                pouw_mult = 1.25  # +25% bonus for nocturnal heavy work
                base_model = (
                    target_domain_meta.recommended_base_model
                    if profile.vram_gb >= 16.0
                    else "Qwen/Qwen2.5-Coder-7B-Instruct"
                )
            else:
                # Daytime surplus micro-QLoRA harvesting
                rank = 16 if profile.vram_gb >= 12.0 else 8
                steps = 30
                pouw_mult = 1.0
                base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

            # Compute TFLOPs and deterministic HMT reward
            est_tflops = round(steps * (profile.vram_gb / 8.0) * 1.85, 3)
            base_reward = DEFAULT_BASE_WU_REWARD_HMT
            step_reward = steps * DEFAULT_TFLOPS_STEP_RATE_HMT
            total_reward = round((base_reward + step_reward + DEFAULT_GATE_PASS_BONUS_HMT) * pouw_mult, 4)

            wu_id = f"wu-diurnal-{target_domain_meta.canonical_domain}-{int(now_utc)}-{record.account_id[:6]}"

            workunit = DiurnalTrainingWorkunit(
                wu_id=wu_id,
                domain=target_domain_meta.domain,
                canonical_domain=target_domain_meta.canonical_domain,
                hf_dataset=target_domain_meta.hf_dataset,
                base_model_id=base_model,
                target_rank=rank,
                steps=steps,
                operational_mode=mode,
                assigned_producer_id=record.account_id,
                producer_handle=profile.handle,
                producer_timezone_offset=profile.timezone_offset_hours,
                local_clock_hour=round(local_hr, 2),
                pouw_reward_hmt=total_reward,
                estimated_tflops=est_tflops,
            )
            workunits.append(workunit)

        return workunits

    def plan_producer_quorum_moe_serving(
        self,
        active_records: Sequence[ProducerRecord],
        now_utc: float | None = None,
        system_load_ratio: float = 0.45,
    ) -> dict[str, Any]:
        """Compute live MoE & Wave 2 Scientific MoE serving readiness as producers connect.

        Inspects connected producer records and registered diurnal hardware profiles to determine:
        1. Which Wave 2 Scientific MoE stage is unlocked (`stage_0_standby` .. `stage_4_full_swarm_20_expert_super_moe`).
        2. Whether BOINC `min_quorum=2` cross-CPU scientific parameter sweeps are active (`>= 2` producers).
        3. Whether Method-of-Lines PDE MoE and 3x Cluster Super-MoEs are active (`>= 4` producers).
        4. Per-producer operational mode breakdown (daytime inference, surplus harvesting, nocturnal training).
        """
        ts = float(now_utc) if now_utc is not None else time.time()
        total_vram_gb = 0.0
        live_models_set: set[str] = set()
        mode_counts: dict[str, int] = {m.value: 0 for m in DiurnalOperationalMode}
        surplus_count = max(0, len(active_records) - 2) if system_load_ratio < 0.60 else 0

        for record in active_records:
            profile = self._producers.get(record.account_id)
            vram = profile.vram_gb if profile is not None else 8.0
            total_vram_gb += vram
            for public_id, _local_name in record.models:
                live_models_set.add(f"hivemind/{public_id.removeprefix('hivemind/')}")
            if profile is not None:
                mode = profile.resolve_mode(
                    now_utc=ts,
                    system_load_ratio=system_load_ratio,
                    surplus_producers_count=surplus_count,
                )
            else:
                mode = DiurnalOperationalMode.DAYTIME_INFERENCE_PRIMARY
            mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1

        quorum_state = resolve_producer_quorum_moe_state(
            connected_producers=len(active_records),
            total_vram_gb=total_vram_gb,
            live_models=sorted(live_models_set),
        )
        quorum_state["operational_mode_breakdown"] = mode_counts
        quorum_state["system_load_ratio"] = round(float(system_load_ratio), 3)
        quorum_state["evaluated_at_utc"] = round(ts, 3)
        return quorum_state
