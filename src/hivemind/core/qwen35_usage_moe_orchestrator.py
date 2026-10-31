# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Season 19 (`HM-S19-Q35`): Qwen 3.5 Coding Mastery & Usage-Driven Idle/Nocturnal MoE Orchestrator.

Aligns decentralized MoE training and inference capacity with measured end-user demand on
`hivemind-openrouter` (61.2% coding completions / 71.0% coding tokens, 38.6% astrophysics/science,
and growing multilingual/BYOB traffic) while leveraging both Large Producers (12 GB - 24 GB+ VRAM)
and Standard Consumer Producers (6 GB - 8 GB VRAM).

Key capabilities:
1. Hardware-stratified Qwen 3.5 model & LoRA profile assignment (`std_8gb`, `big_12gb`, `xl_16gb_plus`),
   keeping 8 GB GPUs strictly below the 6,656 MiB (6.5 GB) Windows WDDM ceiling.
2. Usage-weighted workunit allocation across 3 clusters (Coding 65%, Science 25%, Multilingual/BYOB 10%)
   driven by live OpenRouter completion telemetry, energy-gap failure signals, and an exploration floor.
3. Dual-window scheduling:
   - Nocturnal deep QLoRA + GWAYA RFT window (22:00-07:00 CET, 80-160 inner steps).
   - Daytime micro-idle window (>= 30s idle with < 1.0s preemption circuit breaker, 20-40 inner steps).
4. SVD Rank-32 -> Rank-16 subspace projection (`project_lora_rank_svd`) so 8 GB consumer GPUs inherit
   high-rank adapters trained overnight on 12 GB - 24 GB+ producers.
5. 7-Day weekly orchestration schedule generator for contributor & producer execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

try:
    import torch
except ImportError:  # Slim router container without torch
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Hardware-Stratified Qwen 3.5 Backbone & LoRA Profiles
# ---------------------------------------------------------------------------

QWEN35_HARDWARE_TIERS: dict[str, dict[str, Any]] = {
    "std_8gb": {
        "tier_id": "std_8gb",
        "plan_class": "cuda_hivemind",
        "min_vram_mib": 5500,
        "max_wddm_safe_mib": 6656,
        "primary_backbone": "Qwen/Qwen3.5-9B-Base",
        "coder_backbone": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "draft_backbone": "Qwen/Qwen3.5-0.8B",
        "estimated_vram_nf4_mib": 5460,
        "lora_rank": 16,
        "lora_alpha": 32,
        "seq_len": 512,
        "nocturnal_inner_steps": 80,
        "idle_micro_steps": 20,
        "openrouter_model_ids": ["hivemind/coder", "hivemind/qwen3.5-coder", "hivemind/coder-lite"],
    },
    "big_12gb": {
        "tier_id": "big_12gb",
        "plan_class": "cuda_hivemind_big",
        "min_vram_mib": 10000,
        "max_wddm_safe_mib": 11264,
        "primary_backbone": "Qwen/Qwen3.5-9B",
        "coder_backbone": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "draft_backbone": "Qwen/Qwen3.5-0.8B",
        "estimated_vram_nf4_mib": 8396,
        "lora_rank": 24,
        "lora_alpha": 48,
        "seq_len": 1024,
        "nocturnal_inner_steps": 120,
        "idle_micro_steps": 30,
        "openrouter_model_ids": ["hivemind/qwen3.5-coder-14b", "hivemind/qwen3.5-coder"],
    },
    "xl_16gb_plus": {
        "tier_id": "xl_16gb_plus",
        "plan_class": "cuda_hivemind_xl",
        "min_vram_mib": 15000,
        "max_wddm_safe_mib": 22528,
        "primary_backbone": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "coder_backbone": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "draft_backbone": "Qwen/Qwen3.5-4B",
        "estimated_vram_nf4_mib": 10035,
        "lora_rank": 32,
        "lora_alpha": 64,
        "seq_len": 2048,
        "nocturnal_inner_steps": 160,
        "idle_micro_steps": 40,
        "openrouter_model_ids": ["hivemind/qwen3.5-coder-pro", "hivemind/qwen3.5-coder-14b"],
    },
}


# ---------------------------------------------------------------------------
# 2. Usage-Driven Specialized Coding, Science & Multilingual Bees
# ---------------------------------------------------------------------------

QWEN35_SPECIALIZED_BEES: tuple[dict[str, Any], ...] = (
    # Cluster A: Software Engineering, IDE FIM & Formal Verification (Target share: ~65%)
    {
        "bee_id": "bee-qwen35-python-swe",
        "cluster": "coding",
        "domain": "python_swe_verified",
        "hf_dataset": "SWE-bench/SWE-bench_Verified",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.16,
        "verification_oracle": "gwaya_pytest_sandbox",
        "description": "Multi-file Python bug repair, unit test synthesis, and zero-stub AST verification.",
    },
    {
        "bee_id": "bee-qwen35-fim-autocomplete",
        "cluster": "coding",
        "domain": "ide_fim_low_latency",
        "hf_dataset": "HuggingFaceH4/CodeAlpaca_20K",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.12,
        "verification_oracle": "fim_syntax_and_latency_gate",
        "description": "Sub-150ms Fill-In-the-Middle (FIM) IDE completion for VS Code, Continue.dev & Cursor.",
    },
    {
        "bee_id": "bee-qwen35-rust-systems",
        "cluster": "coding",
        "domain": "rust_systems_kernel",
        "hf_dataset": "bigcode/the-stack-v2-dedup",
        "min_tier": "big_12gb",
        "base_demand_weight": 0.09,
        "verification_oracle": "cargo_check_clippy_sandbox",
        "description": "Zero-allocation Rust systems programming, SIMD kernels, and memory-safe concurrency.",
    },
    {
        "bee_id": "bee-qwen35-ts-fullstack",
        "cluster": "coding",
        "domain": "typescript_react_fullstack",
        "hf_dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.08,
        "verification_oracle": "tsc_ast_typecheck",
        "description": "TypeScript 5, React/Next.js, FastAPI, and REST/GraphQL fullstack application synthesis.",
    },
    {
        "bee_id": "bee-qwen35-cuda-triton",
        "cluster": "coding",
        "domain": "gpu_kernels_int4_fp8",
        "hf_dataset": "neuralmagic/LLM_Compression_Code",
        "min_tier": "big_12gb",
        "base_demand_weight": 0.06,
        "verification_oracle": "tensor_cosine_sqnr_gate",
        "description": "CUDA, ROCm HIP, and Triton quantized INT4/FP8 MoE kernel optimization.",
    },
    {
        "bee_id": "bee-qwen35-sql-data",
        "cluster": "coding",
        "domain": "sql_data_pipelines",
        "hf_dataset": "b-mc2/sql-create-context",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.05,
        "verification_oracle": "sqlite_execution_equivalence",
        "description": "SQL query optimization, DuckDB/BigQuery analytics, and PyArrow schema transformations.",
    },
    {
        "bee_id": "bee-qwen35-devops-sec",
        "cluster": "coding",
        "domain": "devops_cyber_hardening",
        "hf_dataset": "ASSERT-KTH/DISL-Solidity-Smart-Contracts",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.05,
        "verification_oracle": "bandit_sast_zero_findings",
        "description": "CI/CD, Docker, Cloud Run hardening, and automated SAST vulnerability remediation.",
    },
    {
        "bee_id": "bee-qwen35-lean4-prover",
        "cluster": "coding",
        "domain": "lean4_formal_verification",
        "hf_dataset": "hoskinson-center/proofnet",
        "min_tier": "xl_16gb_plus",
        "base_demand_weight": 0.04,
        "verification_oracle": "lean4_lake_zero_sorry",
        "description": "Machine-checked Lean 4 theorem proving with zero sorry and zero user axioms.",
    },
    # Cluster B: Astrophysics, Math & Scientific Discovery (Target share: ~25%)
    {
        "bee_id": "bee-stars-gaia-v2",
        "cluster": "science",
        "domain": "astrophysics_gaia_spectra",
        "hf_dataset": "BrunoBarreto/sdss_dr12_stars_regression",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.11,
        "verification_oracle": "paired_mcnemar_stellar_gate",
        "description": "GAIA DR3 & Hipparcos stellar classification and exoplanet transit discovery.",
    },
    {
        "bee_id": "bee-rusty-sundials",
        "cluster": "science",
        "domain": "rusty_sundials_numeric_physics",
        "hf_dataset": "xaviercallens/rusty-SUNDIALS",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.08,
        "verification_oracle": "rusty_sundials_invariant_conservation_gate",
        "description": (
            "BYOB Scientific & Numeric Rescue Bee (CVODE BDF, ARKode ARS(2,2,2) IMEX, Dual-RRK, qf-pgpe IF-RK4) "
            "for astronomy, fluid mechanics, quantum fluids, and genetics."
        ),
    },
    {
        "bee_id": "bee-math-olympiad-qwen35",
        "cluster": "science",
        "domain": "math_reasoning_gsm8k",
        "hf_dataset": "openai/gsm8k",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.03,
        "verification_oracle": "exact_q_numeric_match",
        "description": "Multi-step mathematical reasoning and symbolic proof verification.",
    },
    {
        "bee_id": "bee-biomed-clinical",
        "cluster": "science",
        "domain": "biomedical_pubmed",
        "hf_dataset": "qiaojin/PubMedQA",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.03,
        "verification_oracle": "paired_mcnemar_qa_gate",
        "description": "Biomedical literature synthesis and clinical evidence grounding.",
    },
    # Cluster C: Multilingual (FR/EN/ZH) & Partner BYOB (Target share: ~10%)
    {
        "bee_id": "bee-french-enterprise-dev",
        "cluster": "multilingual_byob",
        "domain": "french_technical_engineering",
        "hf_dataset": "machelreid/mimo-french-instruct",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.06,
        "verification_oracle": "heldout_nll_and_french_ast",
        "description": "Bilingual French/English software architecture and technical documentation.",
    },
    {
        "bee_id": "bee-crowdgpt-trunk-graft",
        "cluster": "multilingual_byob",
        "domain": "crowdgpt_1b_graft",
        "hf_dataset": "Vxtzq/Crowd-v1",
        "min_tier": "std_8gb",
        "base_demand_weight": 0.04,
        "verification_oracle": "crowdgpt_cpu_spot_check_ratchet",
        "description": "18 MB QLoRA domain graft and GPU held-out loss check for CrowdGPT Vxtzq/Crowd-v1.",
    },
)


@dataclass(frozen=True)
class ProducerNodeCapability:
    """Hardware & availability descriptor for a registered HiveMind GPU producer."""

    node_id: str
    gpu_model: str
    vram_mib: int
    fp16_tflops: float
    idle_seconds: float
    is_european_night: bool
    active_inference_requests: int = 0

    @property
    def hardware_tier(self) -> str:
        if self.vram_mib >= QWEN35_HARDWARE_TIERS["xl_16gb_plus"]["min_vram_mib"]:
            return "xl_16gb_plus"
        if self.vram_mib >= QWEN35_HARDWARE_TIERS["big_12gb"]["min_vram_mib"]:
            return "big_12gb"
        if self.vram_mib >= QWEN35_HARDWARE_TIERS["std_8gb"]["min_vram_mib"]:
            return "std_8gb"
        return "cpu_audit_only"

    @property
    def training_mode(self) -> str:
        """Determine whether the node should serve inference, run micro-idle steps, or run nocturnal deep training."""
        if self.active_inference_requests > 0 or self.idle_seconds < 30.0:
            return "inference_priority"
        if self.hardware_tier == "cpu_audit_only":
            return "zk_opoi_cpu_audit"
        if self.is_european_night:
            return "nocturnal_deep_qlora"
        return "daytime_idle_micro_qlora"


@dataclass(frozen=True)
class ScheduledMoEWorkunit:
    """Single usage-weighted Qwen 3.5 / MoE training workunit assigned to an idle or nocturnal producer."""

    workunit_id: str
    node_id: str
    hardware_tier: str
    training_mode: str
    backbone_model: str
    bee_id: str
    cluster: str
    hf_dataset: str
    lora_rank: int
    lora_alpha: int
    seq_len: int
    inner_steps: int
    estimated_vram_mib: int
    verification_oracle: str
    prorated_reward_hmt: float


class UsageDrivenQwen35MoEOrchestrator:
    """Computes usage-proportional MoE training allocations and schedules idle/nocturnal producers."""

    def __init__(
        self,
        observed_completions_by_cluster: Mapping[str, int] | None = None,
        energy_failure_rate_by_bee: Mapping[str, float] | None = None,
    ) -> None:
        # Default to live v3.9.0 OpenRouter telemetry (873 coding, 551 science, 3 multilingual/general)
        self.observed_completions = dict(
            observed_completions_by_cluster
            or {"coding": 873, "science": 551, "multilingual_byob": 3}
        )
        self.energy_failure_rate_by_bee = dict(energy_failure_rate_by_bee or {})

    def compute_cluster_target_shares(self) -> dict[str, float]:
        """Blend live OpenRouter usage (70% weight) with a 10% strategic multilingual/BYOB floor."""
        total = max(sum(self.observed_completions.values()), 1)
        raw_coding = self.observed_completions.get("coding", 0) / total
        raw_science = self.observed_completions.get("science", 0) / total
        raw_multi = self.observed_completions.get("multilingual_byob", 0) / total

        multi_share = max(0.08, 0.85 * raw_multi + 0.15 * 0.15)
        raw_cs = (0.85 * raw_coding + 0.15 * 0.60) + (0.85 * raw_science + 0.15 * 0.25)
        scale_cs = (1.0 - multi_share) / max(raw_cs, 1e-9)
        coding_share = (0.85 * raw_coding + 0.15 * 0.60) * scale_cs
        science_share = (0.85 * raw_science + 0.15 * 0.25) * scale_cs
        return {
            "coding": round(coding_share, 4),
            "science": round(science_share, 4),
            "multilingual_byob": round(multi_share, 4),
        }

    def compute_bee_priority_scores(self) -> dict[str, float]:
        """Rank all 13 specialized Bees by usage share and sandbox energy failure rate."""
        cluster_shares = self.compute_cluster_target_shares()
        cluster_base_totals: dict[str, float] = {}
        for b in QWEN35_SPECIALIZED_BEES:
            c = b["cluster"]
            cluster_base_totals[c] = cluster_base_totals.get(c, 0.0) + float(b["base_demand_weight"])

        scores: dict[str, float] = {}
        for b in QWEN35_SPECIALIZED_BEES:
            bee_id = b["bee_id"]
            c = b["cluster"]
            within_cluster = float(b["base_demand_weight"]) / max(cluster_base_totals[c], 1e-9)
            usage_term = cluster_shares[c] * within_cluster
            fail_boost = 1.0 + 0.5 * min(max(self.energy_failure_rate_by_bee.get(bee_id, 0.0), 0.0), 1.0)
            scores[bee_id] = usage_term * fail_boost

        total_score = sum(scores.values())
        return {k: round(v / total_score, 4) for k, v in scores.items()}

    def schedule_workunits_for_producers(
        self,
        producers: Sequence[ProducerNodeCapability],
        round_index: int = 1,
    ) -> list[ScheduledMoEWorkunit]:
        """Assign usage-prioritized Qwen 3.5 QLoRA workunits to eligible idle and nocturnal producers."""
        scores = self.compute_bee_priority_scores()
        tier_rank = {"std_8gb": 1, "big_12gb": 2, "xl_16gb_plus": 3}
        ordered_bees = sorted(
            QWEN35_SPECIALIZED_BEES,
            key=lambda b: (-tier_rank[b["min_tier"]], -scores[b["bee_id"]]),
        )

        scheduled: list[ScheduledMoEWorkunit] = []
        bee_cursor = 0

        for idx, node in enumerate(producers):
            mode = node.training_mode
            if mode in {"inference_priority", "zk_opoi_cpu_audit"}:
                continue

            tier_cfg = QWEN35_HARDWARE_TIERS[node.hardware_tier]
            node_tier_level = tier_rank[node.hardware_tier]

            # Pick the highest-priority compatible Bee for this producer's VRAM tier
            eligible_bees = [
                b for b in ordered_bees if tier_rank[b["min_tier"]] <= node_tier_level
            ]
            chosen_bee = eligible_bees[(bee_cursor + idx) % len(eligible_bees)]
            bee_cursor += 1

            inner_steps = (
                int(tier_cfg["nocturnal_inner_steps"])
                if mode == "nocturnal_deep_qlora"
                else int(tier_cfg["idle_micro_steps"])
            )
            backbone = (
                str(tier_cfg["coder_backbone"])
                if chosen_bee["cluster"] == "coding" and node.hardware_tier != "xl_16gb_plus"
                else str(tier_cfg["primary_backbone"])
            )
            # Prorated fair-share reward (~112-115 HMT/hr normalized by steps)
            prorated_hmt = round(10.0 + 0.125 * inner_steps, 2)
            wu_id = f"hm_s19_r{round_index:03d}_{idx + 1:02d}_{chosen_bee['bee_id']}"

            scheduled.append(
                ScheduledMoEWorkunit(
                    workunit_id=wu_id,
                    node_id=node.node_id,
                    hardware_tier=node.hardware_tier,
                    training_mode=mode,
                    backbone_model=backbone,
                    bee_id=chosen_bee["bee_id"],
                    cluster=chosen_bee["cluster"],
                    hf_dataset=chosen_bee["hf_dataset"],
                    lora_rank=int(tier_cfg["lora_rank"]),
                    lora_alpha=int(tier_cfg["lora_alpha"]),
                    seq_len=int(tier_cfg["seq_len"]),
                    inner_steps=inner_steps,
                    estimated_vram_mib=int(tier_cfg["estimated_vram_nf4_mib"]),
                    verification_oracle=chosen_bee["verification_oracle"],
                    prorated_reward_hmt=prorated_hmt,
                )
            )

        return scheduled


def project_lora_rank_svd(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    target_rank: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Project a high-rank Big-GPU LoRA adapter (e.g. rank 32) down to a consumer 8 GB GPU rank (rank 16).

    Computes the best rank-`target_rank` Frobenius approximation of `delta_W = lora_b @ lora_a`
    via truncated SVD and returns `(student_a, student_b, retained_energy_ratio)`.
    """
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("lora_a and lora_b must be 2D matrices")
    if lora_b.shape[1] != lora_a.shape[0]:
        raise ValueError(f"Inner rank mismatch: lora_b {lora_b.shape} vs lora_a {lora_a.shape}")
    if target_rank <= 0 or target_rank > lora_a.shape[0]:
        raise ValueError(f"target_rank {target_rank} must be in [1, {lora_a.shape[0]}]")

    delta_w = (lora_b.to(torch.float32)) @ (lora_a.to(torch.float32))
    u, s, vh = torch.linalg.svd(delta_w, full_matrices=False)

    total_energy = float(torch.sum(s * s).item())
    s_top = s[:target_rank]
    retained_energy = float(torch.sum(s_top * s_top).item())
    energy_ratio = retained_energy / total_energy if total_energy > 0.0 else 1.0

    sqrt_s = torch.sqrt(s_top)
    student_b = (u[:, :target_rank] * sqrt_s.unsqueeze(0)).to(lora_b.dtype)
    student_a = (sqrt_s.unsqueeze(1) * vh[:target_rank, :]).to(lora_a.dtype)
    return student_a, student_b, round(energy_ratio, 6)


def build_seven_day_orchestration_plan() -> dict[str, Any]:
    """Generate the structured 7-day Season 19 execution schedule for the coming week."""
    days = [
        {
            "day": "Day 1",
            "task_id": "HM-S19-Q35-01",
            "title": "Onboard Qwen 3.5 9B / 14B / 30B-A3B Hardware-Stratified Profiles & OpenRouter Tiers",
            "target_nodes": "Big-GPU (12-24 GB) + Standard-GPU (8 GB)",
            "deliverable": "Register `hivemind/qwen3.5-coder` (9B/14B) and `hivemind/qwen3.5-coder-pro` (30B-A3B MoE) in catalog and BOINC profiles.",
        },
        {
            "day": "Day 2",
            "task_id": "HM-S19-Q35-02",
            "title": "Usage-Driven Nocturnal (22:00-07:00 CET) & Daytime Micro-Idle (>=30s) Scheduler",
            "target_nodes": "All 8 Producer + 14 Prosumer Nodes",
            "deliverable": "Allocate 65% Coding, 25% Science, 10% Multilingual/BYOB workunits with <1.0s inference preemption.",
        },
        {
            "day": "Day 3",
            "task_id": "HM-S19-Q35-03",
            "title": "8-Bee Qwen 3.5 Coding Specialization with GWAYA Zero-Stub RFT & D6 Energy-Gap Pairs",
            "target_nodes": "Big-GPU (RTX 3090/4070/T4) & RTX 2070",
            "deliverable": "Train Python SWE, FIM Autocomplete, Rust Systems, TypeScript, CUDA/Triton, SQL, DevOps & Lean 4 Bees.",
        },
        {
            "day": "Day 4",
            "task_id": "HM-S19-Q35-04",
            "title": "SVD Rank-32 -> Rank-16 Adapter Distillation (Big-GPU Teacher to 8 GB Consumer Student)",
            "target_nodes": "8 GB Consumer GPUs (RTX 2070 / 3060 / 4060)",
            "deliverable": "Project nocturnal rank-32 Big-GPU adapters to rank-16 (<5.4 GB VRAM) with >=95% spectral energy retention.",
        },
        {
            "day": "Day 5",
            "task_id": "HM-S19-Q35-05",
            "title": "Sub-150ms IDE Fill-In-the-Middle (FIM) Speculative Decoding & 3,000-Developer Onboarding Kit",
            "target_nodes": "hivemind-openrouter + VS Code / Continue.dev / Cursor / OpenCode",
            "deliverable": "Pair Qwen 3.5 3B draft with Qwen 3.5 14B/9B target and publish 1-click enterprise developer setup.",
        },
        {
            "day": "Day 6",
            "task_id": "HM-S19-Q35-06",
            "title": "Paired McNemar & HumanEval/MBPP-Sanitized Promotion Gate + Prorated Leaderboard Audit",
            "target_nodes": "BOINC Validator & La Ruche Vault",
            "deliverable": "Enforce zero parent regression, exact McNemar p < 0.05, and fair-share ~112-115 HMT/hr credit.",
        },
    ]
    payload = {
        "season": "Season 19 (Wave 18 — Operation Qwen 3.5 Coding Mastery & Usage-Driven Idle/Nocturnal MoE)",
        "epic_code": "HM-S19-Q35",
        "days": days,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["plan_sha256"] = digest
    return payload
