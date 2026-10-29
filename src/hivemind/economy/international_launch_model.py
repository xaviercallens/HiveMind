# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Mathematical, Social, and Economic Model for the HiveMind International English Launch (HM-S19-INTL).

Why an unthrottled English viral launch is dangerous without two-sided market controls:
1. Supply-Demand Asymmetry ("The Consumer Tsunami"):
   - In the French/community pilot (`47` accounts), `8` producers + `14` prosumers = `46.8%` supply participation.
   - In a large English viral wave (`r/LocalLLaMA`, Hacker News `Show HN`), immediate producer signup fraction
     `alpha_p` drops to `2% - 5%` while pure consumers represent `90% - 95%`.
2. Queueing Collapse (`M/M/c` Erlang-C Saturation):
   - With `c` active daytime producers serving `mu_tot` requests/sec, whenever consumer arrival rate
     `lambda_c >= mu_tot` (`rho >= 1.0`), queue length and latency diverge to infinity (`502/504` timeouts).
3. Starter-Grant Tragedy of the Commons:
   - Granting `100 HMT` unconditionally to `2,000` pure consumers mints `200,000 HMT` (`500M` tokens of demand)
     with zero added GPU supply.

This module implements the 4-Mechanism Self-Balancing International Launch Model:
- Mechanism 1: M/M/c Erlang-C Queueing & Safe Concurrency Calculator (`compute_erlang_c_metrics`).
- Mechanism 2: Supply-Pegged Progressive Starter Grant (`15 HMT` instant + `85 HMT` Prosumer GPU unlock or
  supply-backed daily drip).
- Mechanism 3: Congestion-Adjusted Kelly & Surge Pricing (`compute_congestion_price`) that rewards producers
  `2x - 4x` during peak consumer traffic while throttling pure-consumer scraping.
- Mechanism 4: Coupled Bass/Compartmental Social Conversion Model (`Consumer -> Prosumer -> Producer`) &
  4-Stage Wave-Gated International Launch Readiness Evaluator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SwarmLiveBaseline:
    """Empirical baseline measured on hivemind-openrouter and hivemind-at-home-ruche (2026-10-27)."""

    total_accounts: int = 47
    producers: int = 8
    prosumers: int = 14
    consumers: int = 22
    admins: int = 3
    peak_tflops_fp16: float = 518.4
    cumulative_tflops_hours: float = 4525.0
    boinc_cobblestones: float = 5632.11
    billed_completions: int = 1427
    billed_tokens: int = 240162
    mean_tokens_per_request: float = 168.3
    daytime_aggregate_tok_per_s: float = 380.0
    p50_latency_s: float = 0.1766
    p95_latency_s: float = 0.5636


@dataclass(frozen=True)
class ErlangCQueueDiagnostic:
    """M/M/c queueing diagnostic for a given producer fleet and consumer arrival rate."""

    active_producers_c: int
    aggregate_tok_per_s: float
    service_rate_req_per_s: float
    arrival_rate_req_per_s: float
    utilization_rho: float
    erlang_c_queue_prob: float
    expected_queue_wait_s: float
    p95_total_latency_s: float
    max_safe_arrival_req_per_s: float
    max_safe_concurrent_users: int
    is_stable: bool


@dataclass(frozen=True)
class ProgressiveGrantQuote:
    """Supply-pegged progressive starter grant to prevent pure-consumer HMT dilution."""

    instant_grant_hmt: float
    prosumer_gpu_unlock_bonus_hmt: float
    daily_consumer_drip_hmt: float
    supply_demand_ratio: float
    effective_unbacked_inflation_reduction_pct: float


@dataclass(frozen=True)
class SocialFunnelProjection:
    """Coupled Consumer -> Prosumer conversion projection for an international traffic wave."""

    wave_name: str
    new_visitors: int
    new_signups: int
    initial_producers: int
    converted_prosumers: int
    pure_consumers: int
    effective_supply_ratio: float
    critical_conversion_rate_kappa_star: float
    actual_conversion_rate_kappa: float
    post_wave_daytime_tok_per_s: float
    post_wave_nocturnal_tflops: float
    utilization_rho_with_controls: float
    utilization_rho_without_controls: float
    is_self_sustaining: bool


def compute_erlang_c_metrics(
    active_producers_c: int,
    aggregate_tok_per_s: float,
    arrival_rate_req_per_s: float,
    mean_tokens_per_req: float = 168.3,
    base_network_overhead_s: float = 0.085,
    safe_rho_ceiling: float = 0.78,
) -> ErlangCQueueDiagnostic:
    """Compute exact M/M/c Erlang-C waiting probability and latency under consumer demand."""
    if active_producers_c < 1 or aggregate_tok_per_s <= 0.0 or mean_tokens_per_req <= 0.0:
        raise ValueError("active_producers_c >= 1, aggregate_tok_per_s > 0, and mean_tokens_per_req > 0 required")
    if arrival_rate_req_per_s < 0.0:
        raise ValueError("arrival_rate_req_per_s must be >= 0")

    c = int(active_producers_c)
    mu_tot = aggregate_tok_per_s / mean_tokens_per_req
    mu_single = mu_tot / c
    rho = arrival_rate_req_per_s / mu_tot
    max_safe_lambda = safe_rho_ceiling * mu_tot
    # Typical active human developer sends ~1 request every 125 seconds (~28.8 req/hr)
    max_safe_users = int(max_safe_lambda * 125.0)

    if rho >= 1.0:
        return ErlangCQueueDiagnostic(
            active_producers_c=c,
            aggregate_tok_per_s=round(aggregate_tok_per_s, 2),
            service_rate_req_per_s=round(mu_tot, 4),
            arrival_rate_req_per_s=round(arrival_rate_req_per_s, 4),
            utilization_rho=round(rho, 4),
            erlang_c_queue_prob=1.0,
            expected_queue_wait_s=float("inf"),
            p95_total_latency_s=float("inf"),
            max_safe_arrival_req_per_s=round(max_safe_lambda, 4),
            max_safe_concurrent_users=max_safe_users,
            is_stable=False,
        )

    a = c * rho  # offered load in Erlangs
    # Compute sum_{k=0}^{c-1} a^k / k! using log-space or iterative terms for numerical stability
    sum_terms = 1.0
    term = 1.0
    for k in range(1, c):
        term *= a / k
        sum_terms += term
    last_term = (term * (a / c)) / (1.0 - rho)
    prob_queue = last_term / (sum_terms + last_term) if a > 0.0 else 0.0

    expected_wait_s = prob_queue / (mu_tot - arrival_rate_req_per_s)
    service_time_s = 1.0 / mu_single if c == 1 else (1.0 / mu_tot) * min(c, 2.2)
    # 95th percentile of waiting time in M/M/c queue: W_q(0.95) = max(0, -ln(0.05 / P_Q) / (mu_tot - lambda))
    if prob_queue > 0.05:
        p95_wait_s = -math.log(0.05 / prob_queue) / (mu_tot - arrival_rate_req_per_s)
    else:
        p95_wait_s = expected_wait_s * 2.5

    p95_total_s = base_network_overhead_s + service_time_s + p95_wait_s
    return ErlangCQueueDiagnostic(
        active_producers_c=c,
        aggregate_tok_per_s=round(aggregate_tok_per_s, 2),
        service_rate_req_per_s=round(mu_tot, 4),
        arrival_rate_req_per_s=round(arrival_rate_req_per_s, 4),
        utilization_rho=round(rho, 4),
        erlang_c_queue_prob=round(prob_queue, 4),
        expected_queue_wait_s=round(expected_wait_s, 4),
        p95_total_latency_s=round(p95_total_s, 4),
        max_safe_arrival_req_per_s=round(max_safe_lambda, 4),
        max_safe_concurrent_users=max_safe_users,
        is_stable=rho <= safe_rho_ceiling,
    )


def compute_progressive_starter_grant(
    epoch_supply_1k_tokens: float,
    epoch_consumer_demand_1k_tokens: float,
    legacy_lump_sum_hmt: float = 100.0,
    instant_grant_hmt: float = 15.0,
    base_daily_drip_hmt: float = 5.0,
) -> ProgressiveGrantQuote:
    """Compute the Supply-Pegged Progressive Starter Grant (`15 HMT` instant + `85 HMT` Prosumer GPU unlock)."""
    ratio = epoch_supply_1k_tokens / max(epoch_consumer_demand_1k_tokens, 1e-6)
    clamped_ratio = min(1.0, max(0.0, ratio))
    drip = round(base_daily_drip_hmt * clamped_ratio, 4)
    unlock_bonus = round(max(0.0, legacy_lump_sum_hmt - instant_grant_hmt), 4)
    inflation_reduction_pct = round(100.0 * (1.0 - (instant_grant_hmt / legacy_lump_sum_hmt)), 2)
    return ProgressiveGrantQuote(
        instant_grant_hmt=instant_grant_hmt,
        prosumer_gpu_unlock_bonus_hmt=unlock_bonus,
        daily_consumer_drip_hmt=drip,
        supply_demand_ratio=round(ratio, 4),
        effective_unbacked_inflation_reduction_pct=inflation_reduction_pct,
    )


def compute_congestion_price(
    base_posted_price_per_1k: float,
    utilization_rho: float,
    rho_threshold: float = 0.60,
    rho_cap: float = 0.95,
    surge_gamma: float = 2.5,
    validator_plus_burn_fee: float = 0.15,
) -> dict[str, float]:
    """Compute congestion-adjusted price omega(rho) and producer reward per 1k tokens."""
    if base_posted_price_per_1k <= 0.0:
        raise ValueError("base_posted_price_per_1k must be > 0")
    clamped_rho = min(max(utilization_rho, 0.0), rho_cap)
    if clamped_rho <= rho_threshold:
        multiplier = 1.0
    else:
        excess = (clamped_rho - rho_threshold) / max(rho_cap - rho_threshold, 1e-6)
        multiplier = 1.0 + surge_gamma * (excess**2)
    price = base_posted_price_per_1k * multiplier
    producer_payout = price * (1.0 - validator_plus_burn_fee)
    return {
        "base_price_per_1k": round(base_posted_price_per_1k, 4),
        "utilization_rho": round(utilization_rho, 4),
        "surge_multiplier": round(multiplier, 4),
        "clearing_price_per_1k": round(price, 4),
        "producer_reward_per_1k": round(producer_payout, 4),
    }


def simulate_international_launch_wave(
    wave_name: str,
    new_visitors: int,
    signup_rate: float = 0.18,
    direct_producer_fraction_alpha_p: float = 0.03,
    consumer_to_prosumer_conversion_kappa: float = 0.08,
    baseline: SwarmLiveBaseline = SwarmLiveBaseline(),
    mean_prosumer_tok_per_s: float = 48.0,
    prosumer_daytime_duty_cycle: float = 0.35,
    mean_consumer_active_req_per_s: float = 0.0022,
) -> SocialFunnelProjection:
    """Simulate the social, queueing, and economic equilibrium of an English launch wave."""
    new_signups = max(1, int(round(new_visitors * signup_rate)))
    init_producers = int(round(new_signups * direct_producer_fraction_alpha_p))
    initial_consumers = max(0, new_signups - init_producers)
    converted_prosumers = int(round(initial_consumers * consumer_to_prosumer_conversion_kappa))
    pure_consumers = initial_consumers - converted_prosumers

    # Effective daytime tok/s added per new producer (full duty ~0.75) and converted prosumer (micro-idle duty ~0.35)
    effective_tok_s_per_prosumer = mean_prosumer_tok_per_s * prosumer_daytime_duty_cycle
    effective_tok_s_per_producer = mean_prosumer_tok_per_s * 0.75

    # Critical conversion rate kappa* needed so newly added supply >= newly added consumer demand
    demand_tok_s_per_consumer = mean_consumer_active_req_per_s * baseline.mean_tokens_per_request
    raw_kappa_star = (
        (demand_tok_s_per_consumer / max(effective_tok_s_per_prosumer, 1e-6))
        - (direct_producer_fraction_alpha_p / max(1.0 - direct_producer_fraction_alpha_p, 1e-6))
    )
    kappa_star = max(0.0, min(1.0, raw_kappa_star))

    # Post-wave fleet capacity
    post_tok_per_s = (
        baseline.daytime_aggregate_tok_per_s
        + init_producers * effective_tok_s_per_producer
        + converted_prosumers * effective_tok_s_per_prosumer
    )
    post_nocturnal_tflops = (
        baseline.peak_tflops_fp16
        + (init_producers + converted_prosumers) * 18.5  # average ~18.5 FP16 TFLOPS per consumer GPU
    )

    # Without controls: no progressive drip, no 2-lane QoS, low conversion (kappa=0.01), 2.2x higher scraping rate
    uncontrolled_arrival_req_s = (baseline.consumers + initial_consumers) * (mean_consumer_active_req_per_s * 2.2)
    uncontrolled_mu_req_s = (
        baseline.daytime_aggregate_tok_per_s + init_producers * effective_tok_s_per_producer
    ) / baseline.mean_tokens_per_request
    rho_uncontrolled = uncontrolled_arrival_req_s / max(uncontrolled_mu_req_s, 1e-6)

    # With 4-Mechanism controls: progressive 15 HMT drip + 2-lane QoS + converted prosumers expanding supply
    controlled_arrival_req_s = (baseline.consumers + pure_consumers + 0.5 * converted_prosumers) * mean_consumer_active_req_per_s
    controlled_mu_req_s = post_tok_per_s / baseline.mean_tokens_per_request
    rho_controlled = controlled_arrival_req_s / max(controlled_mu_req_s, 1e-6)

    total_post_accounts = baseline.total_accounts + new_signups
    total_post_supply_nodes = baseline.producers + baseline.prosumers + init_producers + converted_prosumers
    supply_ratio = total_post_supply_nodes / max(total_post_accounts, 1)

    return SocialFunnelProjection(
        wave_name=wave_name,
        new_visitors=new_visitors,
        new_signups=new_signups,
        initial_producers=init_producers,
        converted_prosumers=converted_prosumers,
        pure_consumers=pure_consumers,
        effective_supply_ratio=round(supply_ratio, 4),
        critical_conversion_rate_kappa_star=round(kappa_star, 4),
        actual_conversion_rate_kappa=round(consumer_to_prosumer_conversion_kappa, 4),
        post_wave_daytime_tok_per_s=round(post_tok_per_s, 2),
        post_wave_nocturnal_tflops=round(post_nocturnal_tflops, 2),
        utilization_rho_with_controls=round(rho_controlled, 4),
        utilization_rho_without_controls=round(rho_uncontrolled, 4),
        is_self_sustaining=(
            consumer_to_prosumer_conversion_kappa >= kappa_star and rho_controlled <= 0.78
        ),
    )


def evaluate_international_launch_readiness(
    baseline: SwarmLiveBaseline = SwarmLiveBaseline(),
) -> dict[str, Any]:
    """Return the complete readiness diagnosis, 4-stage rollout schedule, and wave simulations."""
    current_queue = compute_erlang_c_metrics(
        active_producers_c=baseline.producers,
        aggregate_tok_per_s=baseline.daytime_aggregate_tok_per_s,
        arrival_rate_req_per_s=0.18,
    )
    grant_policy = compute_progressive_starter_grant(
        epoch_supply_1k_tokens=480.0,
        epoch_consumer_demand_1k_tokens=320.0,
    )
    waves = [
        asdict(
            simulate_international_launch_wave(
                wave_name="Stage 1: Producer-First Seed (r/BOINC, r/homelab, CrowdGPT, 3000-Dev Internal)",
                new_visitors=1200,
                signup_rate=0.20,
                direct_producer_fraction_alpha_p=0.12,
                consumer_to_prosumer_conversion_kappa=0.18,
                baseline=baseline,
            )
        ),
        asdict(
            simulate_international_launch_wave(
                wave_name="Stage 2: Technical ML/GPU Communities (r/LocalLLaMA, r/MachineLearning, r/SelfHosted)",
                new_visitors=8000,
                signup_rate=0.15,
                direct_producer_fraction_alpha_p=0.04,
                consumer_to_prosumer_conversion_kappa=0.09,
                baseline=SwarmLiveBaseline(
                    total_accounts=287,
                    producers=37,
                    prosumers=52,
                    consumers=195,
                    admins=3,
                    peak_tflops_fp16=1820.0,
                    daytime_aggregate_tok_per_s=2150.0,
                ),
            )
        ),
        asdict(
            simulate_international_launch_wave(
                wave_name="Stage 3: Broad International Viral Wave (Hacker News Show HN, X/Twitter, Dev.to — 100 Producer Scale)",
                new_visitors=25000,
                signup_rate=0.14,
                direct_producer_fraction_alpha_p=0.025,
                consumer_to_prosumer_conversion_kappa=0.075,
                baseline=SwarmLiveBaseline(
                    total_accounts=1487,
                    producers=100,
                    prosumers=160,
                    consumers=1224,
                    admins=3,
                    peak_tflops_fp16=6480.0,
                    daytime_aggregate_tok_per_s=6400.0,
                ),
            )
        ),
    ]
    rollout_stages = [
        {
            "stage": "Stage 0 (Days 1-5: Oct 27 - Oct 31)",
            "name": "Pre-Launch Hardening & Economic Shield (DO NOT post mass English consumer links yet)",
            "target_channels": "Internal Repo, Existing 47 Pioneer Nodes, @Vxtzq CrowdGPT Bridge",
            "entry_gate": "Current State (47 accounts, 8 producers, 518.4 TFLOPS peak)",
            "exit_gate": "Qwen3.5 14B/30B-A3B live (#239-#243), 15+85 HMT Progressive Drip active, 2-Lane Prosumer QoS active",
        },
        {
            "stage": "Stage 1 (Weekend Launch — Sunday Nov 1, 2026)",
            "name": "Sunday Producer-First International Seed Wave (Recruit GPU Miners Before Pure Consumers)",
            "target_channels": "r/BOINC, r/homelab, r/gpumining, CrowdGPT 100-node pool, 3,000 Enterprise Developers (BYOB)",
            "entry_gate": "Stage 0 complete + PR #249 loaded-model routing + AMCP/EDA traffic sustainer live",
            "exit_gate": ">= 38 active GPU producers + >= 30 standby AMCP prosumers, >= 1,800 TFLOPS peak, >= 2,200 tok/s daytime serving capacity",
        },
        {
            "stage": "Stage 2 (Early/Mid Next Week — Mon Nov 2 - Thu Nov 5, 2026)",
            "name": "Prosumer + Consumer Mix Technical Launch (Paper v3.9.0 + Qwen 3.5 SVD Distillation + AMCP Auto-Replication)",
            "target_channels": "r/LocalLLaMA, r/MachineLearning, r/SelfHosted, HuggingFace Community",
            "entry_gate": "Sunday Nov 1 Producer-First exit gate met (rho < 0.65 under 1,200 visitors)",
            "exit_gate": ">= 150 supply nodes (producers + prosumers), p50 < 250ms, 0% HTTP 5xx, kappa >= 7.5% conversion",
        },
        {
            "stage": "Stage 3 (Next Weekend — Fri Nov 6 - Sun Nov 8, 2026)",
            "name": "Broad International Consumer + Prosumer Mix Launch at 100-Producer Global Scale (Show HN: HiveMind@Home)",
            "target_channels": "Hacker News (Show HN), X/Twitter, Dev.to, LinkedIn Engineering",
            "entry_gate": "Stage 2 exit gate met (>= 100 active daytime producer nodes across 4 regions, >= 6,400 tok/s daytime capacity)",
            "exit_gate": "Self-sustaining 100-producer global swarm (rho <= 0.78, supply ratio >= 12%, zero unbacked HMT inflation)",
        },
    ]
    from hivemind.swarm.amcp_eda_traffic_sustainer import build_sunday_producer_first_international_plan

    return {
        "object": "hivemind.international_launch_readiness",
        "ready_for_unthrottled_mass_english_launch_today": False,
        "ready_for_staged_producer_first_rollout": True,
        "sunday_producer_first_launch_date": "2026-11-01",
        "recommended_mass_english_launch_window": "Sunday Nov 1 (Producer-First Seed) -> Mon-Thu Nov 2-5 (Prosumer/Consumer Mix on r/LocalLLaMA) -> Fri-Sun Nov 6-8 (Global Consumer/Mix on Show HN)",
        "current_baseline": asdict(baseline),
        "current_queue_diagnostics": asdict(current_queue),
        "progressive_grant_policy": asdict(grant_policy),
        "wave_simulations": waves,
        "four_stage_rollout_plan": rollout_stages,
        "amcp_eda_sunday_launch_plan": build_sunday_producer_first_international_plan(),
        "hundred_producer_scale_plan": build_100_producer_international_fleet_plan(),
    }


@dataclass(frozen=True)
class RegionalProducerShard:
    """Geo-affinity regional shard within the 100-producer international launch topology."""

    region_id: str
    display_name: str
    active_producers: int
    standby_prosumers: int
    daytime_aggregate_tok_per_s: float
    median_intra_region_rtt_ms: float
    erlang_c_utilization_rho: float
    p95_total_latency_s: float


@dataclass(frozen=True)
class ProducerHardwareTierAllocation:
    """Hardware tier composition across the 100-producer fleet respecting consumer VRAM guards."""

    tier_id: str
    display_name: str
    node_count: int
    vram_ceiling_mib: int
    free_vram_guard_mib: int
    avg_tok_per_s_per_node: float
    primary_models: tuple[str, ...]
    supported_runtimes: tuple[str, ...]


def build_100_producer_international_fleet_plan(
    active_producers: int = 100,
    standby_prosumers: int = 160,
    arrival_rate_req_per_s: float = 24.0,
    mean_tokens_per_request: float = 168.3,
) -> dict[str, Any]:
    """Build the 100-Producer International Launch Fleet Scaling & Multi-Region Topology Plan (Issues #247, #251).

    Invariants enforced:
    1. Regional partition across 4 continental shards sums exactly to `active_producers`
       (42 EU-West + 28 US-East + 16 US-West + 14 AP-Northeast = 100 at target scale).
    2. Hardware tier partition sums exactly to `active_producers`
       (55 `std_8gb` + 30 `big_12gb` + 15 `xl_16gb_plus` = 100 at target scale):
       - `std_8gb` strictly respects the `<= 6,656 MiB` WDDM ceiling (`< 500 ms` gamer preemption SLA).
       - `big_12gb` (including GCP Tesla T4 15,360 MiB) strictly preserves `>= 4,300 MiB` free VRAM.
    3. Two-Lane Priority QoS reserves `70%` fast-lane capacity for active Prosumers/Producers and
       `30%` fair-queue capacity for pure Consumers, with zero fiat currency (`0.00 EUR / 0.00 USD`).
    """
    c = max(4, int(active_producers))
    prosumers = max(0, int(standby_prosumers))

    # Proportional regional shard allocation that preserves exact 42/28/16/14 split at c = 100
    eu_w = max(1, round(c * 0.42))
    us_e = max(1, round(c * 0.28))
    us_w = max(1, round(c * 0.16))
    ap_ne = max(1, c - (eu_w + us_e + us_w))
    if eu_w + us_e + us_w + ap_ne != c:
        eu_w += c - (eu_w + us_e + us_w + ap_ne)

    # Hardware tier partition that preserves exact 55/30/15 split at c = 100
    t_std8 = max(1, round(c * 0.55))
    t_big12 = max(1, round(c * 0.30))
    t_xl16 = max(1, c - (t_std8 + t_big12))
    if t_std8 + t_big12 + t_xl16 != c:
        t_std8 += c - (t_std8 + t_big12 + t_xl16)

    hardware_tiers = [
        ProducerHardwareTierAllocation(
            tier_id="std_8gb",
            display_name="Consumer 8 GB GPU (RTX 2070 / 3060 8G / 4060 — WDDM <= 6,656 MiB)",
            node_count=t_std8,
            vram_ceiling_mib=6656,
            free_vram_guard_mib=1536,
            avg_tok_per_s_per_node=48.0,
            primary_models=(
                "hivemind/qwen3.5-4b-instruct",
                "hivemind/qwen3.5-9b-fast",
                "hivemind/bee-scientific-ode",
                "hivemind/bee-rusty-sundials",
            ),
            supported_runtimes=("ollama", "llamacpp", "hivemind-solver"),
        ),
        ProducerHardwareTierAllocation(
            tier_id="big_12gb",
            display_name="Mid-Range 12-16 GB GPU (RTX 3060 12G / 4070 / GCP Tesla T4 >= 4,300 MiB Free Guard)",
            node_count=t_big12,
            vram_ceiling_mib=11060,
            free_vram_guard_mib=4300,
            avg_tok_per_s_per_node=72.0,
            primary_models=(
                "hivemind/qwen3.5-14b-instruct",
                "hivemind/qwen3.5-coder-14b",
                "hivemind/bee-SymbolicMath-Lean4",
                "hivemind/bee-ClimateFluid-PDE",
            ),
            supported_runtimes=("vllm", "ollama", "llamacpp", "hivemind-solver"),
        ),
        ProducerHardwareTierAllocation(
            tier_id="xl_16gb_plus",
            display_name="Enthusiast & Prosumer 24 GB+ GPU (RTX 3090 / 4090 / L4 / A5000)",
            node_count=t_xl16,
            vram_ceiling_mib=22528,
            free_vram_guard_mib=2048,
            avg_tok_per_s_per_node=106.6667,
            primary_models=(
                "hivemind/qwen3.5-30b-a3b-moe",
                "hivemind/deepseek-r1-distill-32b",
                "Vxtzq/Crowd-v1",
                "hivemind/bee-QuantumLattice-BEC",
            ),
            supported_runtimes=("vllm", "ollama", "llamacpp", "hivemind-solver"),
        ),
    ]

    aggregate_tok_per_s = round(
        sum(t.node_count * t.avg_tok_per_s_per_node for t in hardware_tiers),
        2,
    )
    fleet_queue = compute_erlang_c_metrics(
        active_producers_c=c,
        aggregate_tok_per_s=aggregate_tok_per_s,
        arrival_rate_req_per_s=arrival_rate_req_per_s,
        mean_tokens_per_req=mean_tokens_per_request,
    )

    region_specs = [
        ("eu-west", "Europe West (Paris / Frankfurt / Amsterdam)", eu_w, round(prosumers * 0.425), 18.0),
        ("us-east", "North America East (Virginia / Montreal)", us_e, round(prosumers * 0.28125), 22.0),
        ("us-west", "North America West (Oregon / Silicon Valley)", us_w, round(prosumers * 0.15625), 24.0),
        (
            "ap-northeast",
            "Asia-Pacific Northeast (Tokyo / Seoul / Singapore)",
            ap_ne,
            max(0, prosumers - (round(prosumers * 0.425) + round(prosumers * 0.28125) + round(prosumers * 0.15625))),
            28.0,
        ),
    ]
    regional_shards: list[dict[str, Any]] = []
    for reg_id, reg_name, reg_c, reg_pros, rtt_ms in region_specs:
        share = reg_c / c
        reg_tok_s = round(aggregate_tok_per_s * share, 2)
        reg_arrival = round(arrival_rate_req_per_s * share, 4)
        reg_q = compute_erlang_c_metrics(
            active_producers_c=reg_c,
            aggregate_tok_per_s=reg_tok_s,
            arrival_rate_req_per_s=reg_arrival,
            mean_tokens_per_req=mean_tokens_per_request,
        )
        shard = RegionalProducerShard(
            region_id=reg_id,
            display_name=reg_name,
            active_producers=reg_c,
            standby_prosumers=reg_pros,
            daytime_aggregate_tok_per_s=reg_tok_s,
            median_intra_region_rtt_ms=rtt_ms,
            erlang_c_utilization_rho=reg_q.utilization_rho,
            p95_total_latency_s=reg_q.p95_total_latency_s,
        )
        regional_shards.append(asdict(shard))

    return {
        "plan_id": "HM-S27-100-PRODUCER-INTERNATIONAL-SCALE",
        "target_active_producers": c,
        "standby_amcp_prosumers": prosumers,
        "max_endpoints_supported": c * 8,
        "aggregate_daytime_tok_per_s": aggregate_tok_per_s,
        "peak_nocturnal_tflops_fp16": round(c * 64.8, 1),
        "regional_shards": regional_shards,
        "hardware_tier_allocations": [asdict(t) for t in hardware_tiers],
        "two_lane_priority_qos": {
            "prosumer_fast_lane_share": 0.70,
            "consumer_fair_queue_share": 0.30,
            "fast_lane_ttft_sla_ms": 200,
            "gamer_preemption_sla_ms": 500,
            "wddm_8gb_vram_cap_mib": 6656,
            "t4_min_free_vram_guard_mib": 4300,
            "fiat_currency_exposure_eur_usd": 0.0,
        },
        "routing_architecture": {
            "dispatcher": "Power-of-Two-Choices (P2C) + HM-S16-06 Loaded-Model Residency + Geo-Shard Affinity",
            "heartbeat_protocol": "AMCP v1.5 CloudEvents 1.0 Topic-Partitioned WAL Batching",
            "supported_runtimes": ["ollama", "vllm", "llamacpp", "hivemind-solver"],
            "partner_federation_bridges": ["Vxtzq/Crowd-v1 (100-node CrowdGPT pool)", "3,000 Enterprise Developers BYOB"],
        },
        "global_queue_diagnostics": asdict(fleet_queue),
    }

