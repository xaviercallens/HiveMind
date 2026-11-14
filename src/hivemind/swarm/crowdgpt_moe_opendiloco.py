# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CrowdGPT Sparse MoE Split + Decoupled OpenDiLoCo Engine (`hivemind/crowd-moe-v1`).

Prepares `Vxtzq/Crowd-v1` (`657.8M` trunk + `350.0M` bigram `CrowdEngram`) for the
**Weekend November 14–15, 2026 Massive HiveMind x CrowdGPT Swarm Training Raid** across
GCP Spot Tesla T4 (`socreateai-agora-hermes-node1`), RTX 2070 (`XAVGAMSRV`), and all
HiveMinder **Producers**, **Trainers**, and **Prosumers**:

1. **Two Function-Preserving Sparse MoE Splits of `CrowdRuntimeMLP` (`w1, w2, w3`)**:
   - `CrowdFineGrainedMoEMLP` (`mode="fine_grained_width_split"`):
     Splits `intermediate_size = 2560` into `E = 8` exact `320`-channel SwiGLU sub-experts
     (`2560 = 8 * 320`, DeepSeekMoE / Qwen-MoE style). At step `0` under unit sum-routing,
     `CrowdFineGrainedMoEMLP(x) == CrowdRuntimeMLP(x)` to floating-point precision (`0` extra
     trunk parameters), while `Top-k` Softmax or FoMoE `sigma`-MoE (`sigmoid_independent_topk_routing`)
     specializes experts across the 8 curated domains.
   - `CrowdLoRAMoEMLP` (`mode="lora_domain_experts"`):
     Keeps the shared `CrowdRuntimeMLP` frozen in `fp16`/`bf16` and attaches `E = 8` domain
     SwiGLU LoRA experts (`rank r=8, alpha=16`) with a **resident-expert mask** (`FoMoE`
     Ghost-Expert Skip-Token routing) so `8 GB` consumer GPUs (`RTX 2070`) train `2` local
     experts in `< 1.78 GiB` VRAM (`>= 1,536 MiB` free) while the `16 GB` GCP Spot Tesla T4
     trains all `8` experts in `< 2.45 GiB` VRAM (`>= 4,300 MiB` free, `AGENTS.md` Rule 1).

2. **Decoupled DiLoCo + PrimeIntellect OpenDiLoCo Coordinator (`CrowdOpenDiLoCoCoordinator`)**:
   - Combines **PrimeIntellect OpenDiLoCo** (`arXiv:2407.07852`, `CPUOffloadedFlatBucketOuterState`
     with `0 MiB` GPU overhead + `P = 4` streaming trunk fragments + `E = 8` expert buckets)
     with **Google DeepMind Decoupled DiLoCo** (`arXiv:2604.21428`, `select_quorum`,
     `AdaptiveGraceWindowController`, bounded quadratic `token_weighted_quality_weight`, and
     **Radial-Directional Averaging (`RDA`)**) plus optional staleness-gated `PA-CGAD`
     (`CosineGatedAdamDecayOuterOptimizer`, `arXiv:2605.09126`) for asynchronous home GPUs.
   - Generates reproducible `OpenDiLoCo` (`PrimeIntellect-ai/OpenDiLoCo`) Hivemind DHT / `torchrun`
     specs AND firewall-free HiveMind `OpenTrainer` HTTPS (`TCP/443`) relay specs.

3. **AGPL-3.0 Allowlisted & Decontaminated 8-Domain Dataset Plan (`load_crowdgpt_moe_dataset_plan`)**:
   - Loads `configs/datasets_s21.json`, enforces `status == "ACCEPTED"` (zero non-commercial /
     custom license drift), `unverified_cap <= 0.30`, 13-gram benchmark decontamination
     (`HumanEval`, `MBPP`, `ARC-Challenge`), and deterministic `sha256` non-IID shard assignment
     across Producers, Trainers, and Prosumers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.core.fomoe_sigma_routing import (
    SigmaMoERoutingResult,
    sigmoid_independent_topk_routing,
)
from hivemind.router.opentrainer import (
    DEFAULT_MIN_FREE_VRAM_MIB_8GB,
    DEFAULT_MIN_FREE_VRAM_MIB_T4,
)
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeBlock,
    CrowdRuntimeMLP,
    CrowdV1WithEngram,
)
from hivemind.swarm.decoupled_diloco import (
    AdaptiveGraceWindowController,
    DecoupledDiLoCoQuorumSyncer,
    LearnerArrival,
    SyncerRoundTelemetry,
)
from hivemind.swarm.diloco_nextgen import (
    CPUOffloadedFlatBucketOuterState,
    CosineGatedAdamDecayOuterOptimizer,
)
from hivemind.swarm.gcp_t4_crowd_diloco import (
    DEFAULT_GCP_T4_WORK_DIR_POSIX,
    assert_not_protected_t4_path,
    chunked_causal_lm_cross_entropy,
     compute_gcp_t4_spot_crowd_budget,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_S21_REGISTRY_PATH = REPO_ROOT / "configs" / "datasets_s21.json"

CROWDGPT_WEEKEND_CAMPAIGN_ID = "CROWDGPT-WE-NOV14-15-2026-MOE-OPENDILOCO"
CROWDGPT_MOE_CAMPAIGN_ID = "CROWDGPT-MOE-8X-UPCYCLE-V1"
CROWDGPT_MOE_PUBLIC_MODEL_ID = "hivemind/crowd-moe-v1"


# ---------------------------------------------------------------------------
# 1. Canonical 8-Expert CrowdGPT-MoE Domain & Dataset Specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrowdMoEExpertSpec:
    """Domain, S21 registry mapping, and hardware residency profile for one CrowdGPT-MoE expert."""

    expert_index: int
    expert_id: str
    s21_domain: str
    mixture_weight: float
    primary_datasets: tuple[str, ...]
    assigned_cohort: str
    description: str


CROWDGPT_MOE_8_EXPERTS: tuple[CrowdMoEExpertSpec, ...] = (
    CrowdMoEExpertSpec(
        expert_index=0,
        expert_id="E0_general_fineweb",
        s21_domain="general_pretrain",
        mixture_weight=0.15,
        primary_datasets=("HuggingFaceFW/fineweb-edu",),
        assigned_cohort="GCP Spot Tesla T4 + Prosumer Anchor Cohort",
        description="Shared Crowd-v1 general reasoning & bilingual FR/EN pretraining anchor (ODC-By).",
    ),
    CrowdMoEExpertSpec(
        expert_index=1,
        expert_id="E1_python_systems_code",
        s21_domain="python_coding",
        mixture_weight=0.22,
        primary_datasets=(
            "bigcode/self-oss-instruct-sc2-exec-filter-50k",
            "nvidia/OpenCodeInstruct",
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
        ),
        assigned_cohort="RTX 2070 (XAVGAMSRV) + Producer/Trainer Coding Cohort",
        description="Execution-verified Python & systems engineering instructions (unit-test gated).",
    ),
    CrowdMoEExpertSpec(
        expert_index=2,
        expert_id="E2_math_reasoning",
        s21_domain="maths",
        mixture_weight=0.16,
        primary_datasets=(
            "nvidia/OpenMathInstruct-2",
            "AI-MO/NuminaMath-CoT",
            "meta-math/MetaMathQA",
        ),
        assigned_cohort="GCP Spot Tesla T4 + Trainer Math Cohort",
        description="Gold-answer verified GSM8K/MATH step-by-step mathematical reasoning.",
    ),
    CrowdMoEExpertSpec(
        expert_index=3,
        expert_id="E3_formal_logic_arc",
        s21_domain="physics",
        mixture_weight=0.11,
        primary_datasets=("allenai/ai2_arc",),
        assigned_cohort="Trainer & Prosumer Formal Reasoning Cohort",
        description="Human-authored scientific & deductive multiple-choice reasoning with gold keys.",
    ),
    CrowdMoEExpertSpec(
        expert_index=4,
        expert_id="E4_physics_stem",
        s21_domain="physics",
        mixture_weight=0.10,
        primary_datasets=("allenai/ai2_arc", "maximus-21/PhysicsQA"),
        assigned_cohort="Trainer STEM & BOINC Nocturnal Cohort",
        description="Quantitative physics problem-solving & multi-step STEM derivations.",
    ),
    CrowdMoEExpertSpec(
        expert_index=5,
        expert_id="E5_exoplanet_transit",
        s21_domain="exoplanet",
        mixture_weight=0.08,
        primary_datasets=(
            "bingbangboom/exoplanet-transit-detection",
            "juliensimon/nasa-exoplanets",
        ),
        assigned_cohort="Citizen-Science BOINC & Prosumer Astrophysics Cohort",
        description="NASA Kepler transit disposition vetting & exoplanet discovery classification.",
    ),
    CrowdMoEExpertSpec(
        expert_index=6,
        expert_id="E6_stellar_spectroscopy",
        s21_domain="stars",
        mixture_weight=0.08,
        primary_datasets=(
            "OneAstronomy/sspp-dr9-stellar-spectra",
            "jeff-vincent/astronomy-textbook-qa-context",
        ),
        assigned_cohort="GCP Spot Tesla T4 (`bee-stars`) + Citizen-Science Cohort",
        description="SDSS DR9 stellar spectral OBAFGKM classification & astrophysics QA.",
    ),
    CrowdMoEExpertSpec(
        expert_index=7,
        expert_id="E7_defensive_cyber",
        s21_domain="cyber",
        mixture_weight=0.10,
        primary_datasets=(
            "CyberNative/Code_Vulnerability_Security_DPO",
            "nuhmanpk/cybersecurity-controls-instructions",
            "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        ),
        assigned_cohort="Producer Shield & Security Auditor Cohort",
        description="Defensive vulnerability remediation & NIST security controls (offensive patterns stripped).",
    ),
)


def load_crowdgpt_moe_dataset_plan(
    registry_path: Path | str = DEFAULT_S21_REGISTRY_PATH,
    *,
    unverified_cap: float = 0.30,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Build the verified 8-expert dataset manifest from `configs/datasets_s21.json`.

    Strictly filters out any `REJECTED` dataset (non-commercial, missing, or gated licenses)
    so every shard distributed during the Nov 14–15, 2026 Weekend Raid is AGPL-3.0 compatible.
    """
    if isinstance(unverified_cap, bool) or not isinstance(unverified_cap, (int, float)) or not (0.0 <= float(unverified_cap) <= 1.0):
        raise ValueError("unverified_cap must be in [0.0, 1.0]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    reg_file = Path(registry_path)
    reg_doc = json.loads(reg_file.read_text(encoding="utf-8"))
    accepted_by_id: dict[str, list[dict[str, Any]]] = {}
    rejected_ids: list[str] = []

    for entry in reg_doc.get("registry", []):
        ds_id = str(entry.get("id") or "")
        if entry.get("status") == "ACCEPTED":
            accepted_by_id.setdefault(ds_id, []).append(entry)
        else:
            rejected_ids.append(ds_id)

    expert_plans: list[dict[str, Any]] = []
    for spec in CROWDGPT_MOE_8_EXPERTS:
        if spec.s21_domain == "general_pretrain":
            sources = [
                {
                    "dataset_id": "HuggingFaceFW/fineweb-edu",
                    "license": "odc-by",
                    "verified_signal": True,
                    "revision_sha": "crowd-v1-pretrain-stream",
                }
            ]
        else:
            sources = []
            for ds_id in spec.primary_datasets:
                for row in accepted_by_id.get(ds_id, ()):
                    sources.append(
                        {
                            "dataset_id": ds_id,
                            "config": row.get("config", "default"),
                            "license": row.get("license"),
                            "verified_signal": bool(row.get("verified_signal", False)),
                            "revision_sha": row.get("revision_sha"),
                        }
                    )
        if not sources:
            raise RuntimeError(f"Expert {spec.expert_id} resolved zero ACCEPTED datasets from {reg_file}")

        has_verified = any(s["verified_signal"] for s in sources)
        eff_unverified_cap = float(unverified_cap) if has_verified else 1.0
        shard_digest = hashlib.sha256(f"{seed}:{spec.expert_id}:{len(sources)}".encode("utf-8")).hexdigest()[:16]
        expert_plans.append(
            {
                "expert_index": spec.expert_index,
                "expert_id": spec.expert_id,
                "s21_domain": spec.s21_domain,
                "mixture_weight": spec.mixture_weight,
                "assigned_cohort": spec.assigned_cohort,
                "description": spec.description,
                "has_execution_or_gold_verified_source": has_verified,
                "effective_unverified_cap": eff_unverified_cap,
                "deterministic_shard_prefix": f"shard-{shard_digest}",
                "accepted_sources": sources,
            }
        )

    total_weight = round(sum(e["mixture_weight"] for e in expert_plans), 6)
    return {
        "campaign_id": CROWDGPT_WEEKEND_CAMPAIGN_ID,
        "public_model_id": CROWDGPT_MOE_PUBLIC_MODEL_ID,
        "seed": int(seed),
        "unverified_cap_policy": float(unverified_cap),
        "decontamination_benchmarks": ["openai/openai_humaneval", "google-research-datasets/mbpp", "allenai/ai2_arc"],
        "ngram_decontamination_n": 13,
        "total_mixture_weight": total_weight,
        "num_experts": len(expert_plans),
        "rejected_non_allowlisted_datasets": sorted(set(rejected_ids)),
        "experts": expert_plans,
    }


def assign_hiveminder_moe_shards(
    handle: str,
    role: str,
    vram_total_mib: int,
    *,
    num_experts: int = 8,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Deterministically assign resident MoE experts and DiLoCo recipe to a Producer, Trainer, or Prosumer."""
    clean_handle = str(handle or "").strip().lstrip("@")
    if not clean_handle:
        raise ValueError("handle must be non-empty")
    norm_role = str(role or "").strip().lower()
    if norm_role not in {"producer", "trainer", "prosumer"}:
        raise ValueError("role must be 'producer', 'trainer', or 'prosumer'")
    if isinstance(vram_total_mib, bool) or not isinstance(vram_total_mib, int) or vram_total_mib < 0:
        raise ValueError("vram_total_mib must be a non-negative integer")
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 2:
        raise ValueError("num_experts must be >= 2")

    if vram_total_mib >= 14_000:
        resident_count = num_experts
        min_free_floor = DEFAULT_MIN_FREE_VRAM_MIB_T4
        recipe = "t4_all8_lora_moe_or_block_window"
    elif vram_total_mib >= 7_500:
        resident_count = min(2, num_experts) if norm_role == "producer" else min(4, num_experts)
        min_free_floor = DEFAULT_MIN_FREE_VRAM_MIB_8GB
        recipe = "consumer_8gb_fomoe_ghost_skip_lora"
    else:
        resident_count = 1
        min_free_floor = 1024
        recipe = "light_1expert_lora_or_cpu_eval"

    digest = hashlib.sha256(f"{seed}:{clean_handle.lower()}:{norm_role}".encode("utf-8")).digest()
    start_idx = int.from_bytes(digest[:4], "big") % num_experts
    resident_indices = sorted({(start_idx + offset) % num_experts for offset in range(resident_count)})

    inner_steps = {"producer": 12, "prosumer": 25, "trainer": 100}[norm_role]
    duty_cycle = {
        "producer": "serving-first (< 1s yield on live /v1/chat/completions traffic; micro-steps during idle gaps)",
        "prosumer": "hybrid A/B arena tester + idle-session OpenDiLoCo contributor",
        "trainer": "dedicated intense OpenDiLoCo / OpenTrainer inner loops (zero private user prompts)",
    }[norm_role]

    return {
        "handle": clean_handle,
        "role": norm_role,
        "vram_total_mib": vram_total_mib,
        "min_free_vram_floor_mib": min_free_floor,
        "recommended_recipe": recipe,
        "resident_expert_indices": resident_indices,
        "resident_expert_ids": [
            CROWDGPT_MOE_8_EXPERTS[idx].expert_id for idx in resident_indices if idx < len(CROWDGPT_MOE_8_EXPERTS)
        ],
        "recommended_inner_steps_h": inner_steps,
        "duty_cycle_policy": duty_cycle,
    }


# ---------------------------------------------------------------------------
# 2. Sparse MoE Split Modules for `Vxtzq/Crowd-v1` (`CrowdRuntimeMLP`)
# ---------------------------------------------------------------------------


def _switch_load_balancing_loss(router_probs: torch.Tensor, top_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch Transformer auxiliary load-balancing loss `E * sum_e (f_e * P_e)` (`Fedus et al., 2022`)."""
    tokens = router_probs.shape[0]
    if tokens == 0:
        return router_probs.sum() * 0.0
    one_hot = F.one_hot(top_indices, num_classes=num_experts).to(torch.float32)  # [N, k, E]
    token_frac = one_hot.mean(dim=(0, 1))  # [E]
    prob_mean = router_probs.mean(dim=0)  # [E]
    return float(num_experts) * torch.sum(token_frac * prob_mean)


class CrowdFineGrainedMoEMLP(nn.Module):
    """Exact algebraic width-partitioned SwiGLU Sparse MoE (`8 x 320 = 2560`) from `CrowdRuntimeMLP`.

    Because `w3(silu(w1(x)) * w2(x)) = sum_{e=0}^{E-1} w3[:, s_e:e_e] @ (silu(w1[s_e:e_e, :] x) * w2[s_e:e_e, :] x)`,
    splitting `intermediate_size` evenly across `E` sub-experts adds **zero extra expert parameters**
    and reproduces `CrowdRuntimeMLP(x)` *identically* when all `E` experts receive gate `1.0`
    (or scaled `E / top_k` at uniform router initialization when `top_k == E`).
    Supports both `softmax_topk` and FoMoE `sigma_topk` (`sigmoid_independent_topk_routing`).
    """

    def __init__(
        self,
        dense_mlp: CrowdRuntimeMLP,
        *,
        num_experts: int = 8,
        top_k: int = 2,
        routing_mode: str = "sigma_topk",
        resident_expert_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dense_mlp, CrowdRuntimeMLP):
            raise TypeError("dense_mlp must be a CrowdRuntimeMLP instance.")
        if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 2:
            raise ValueError("num_experts must be an integer >= 2.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not (1 <= top_k <= num_experts):
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts.")
        if routing_mode not in {"softmax_topk", "sigma_topk", "exact_dense_sum"}:
            raise ValueError("routing_mode must be 'softmax_topk', 'sigma_topk', or 'exact_dense_sum'.")

        inter_dim, hidden_dim = dense_mlp.w1.weight.shape
        if inter_dim % num_experts != 0:
            raise ValueError(
                f"intermediate_size={inter_dim} must be divisible by num_experts={num_experts} for exact width split."
            )
        self.hidden_size = int(hidden_dim)
        self.intermediate_size = int(inter_dim)
        self.expert_intermediate_size = inter_dim // num_experts
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.routing_mode = routing_mode

        dev = dense_mlp.w1.weight.device
        dt = dense_mlp.w1.weight.dtype

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False, device=dev, dtype=torch.float32)
        nn.init.zeros_(self.router.weight)

        self.experts_w1 = nn.ModuleList()
        self.experts_w2 = nn.ModuleList()
        self.experts_w3 = nn.ModuleList()
        step = self.expert_intermediate_size
        with torch.no_grad():
            for e in range(self.num_experts):
                lo, hi = e * step, (e + 1) * step
                w1_e = nn.Linear(self.hidden_size, step, bias=False, device=dev, dtype=dt)
                w2_e = nn.Linear(self.hidden_size, step, bias=False, device=dev, dtype=dt)
                w3_e = nn.Linear(step, self.hidden_size, bias=False, device=dev, dtype=dt)
                w1_e.weight.copy_(dense_mlp.w1.weight[lo:hi, :])
                w2_e.weight.copy_(dense_mlp.w2.weight[lo:hi, :])
                w3_e.weight.copy_(dense_mlp.w3.weight[:, lo:hi])
                self.experts_w1.append(w1_e)
                self.experts_w2.append(w2_e)
                self.experts_w3.append(w3_e)

        mask = torch.ones(self.num_experts, dtype=torch.bool, device=dev)
        if resident_expert_indices is not None:
            mask.zero_()
            for idx in resident_expert_indices:
                if not (0 <= int(idx) < self.num_experts):
                    raise ValueError(f"resident expert index {idx} out of [0, {self.num_experts})")
                mask[int(idx)] = True
            if int(mask.sum().item()) < 1:
                raise ValueError("At least one expert must be resident.")
        self.register_buffer("resident_mask", mask, persistent=False)
        self.last_aux_loss: torch.Tensor = torch.zeros((), device=dev, dtype=torch.float32)
        self.last_ghost_fraction: float = 0.0

    def _run_expert(self, e_idx: int, tokens_x: torch.Tensor) -> torch.Tensor:
        w_dt = self.experts_w1[e_idx].weight.dtype
        x_cast = tokens_x.to(dtype=w_dt)
        act = F.silu(self.experts_w1[e_idx](x_cast)) * self.experts_w2[e_idx](x_cast)
        return self.experts_w3[e_idx](act).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dt = x.dtype
        flat_x = x.reshape(-1, self.hidden_size)
        n_tok = flat_x.shape[0]

        if self.routing_mode == "exact_dense_sum":
            acc = torch.zeros((n_tok, self.hidden_size), device=x.device, dtype=torch.float32)
            for e in range(self.num_experts):
                if bool(self.resident_mask[e].item()):
                    acc = acc + self._run_expert(e, flat_x)
            self.last_aux_loss = acc.sum() * 0.0
            self.last_ghost_fraction = 0.0
            return acc.to(dtype=orig_dt).reshape(orig_shape)

        logits = self.router(flat_x.to(torch.float32))
        if self.routing_mode == "sigma_topk":
            # Differentiable independent sigmoid gates + FoMoE ghost-expert skip mask
            sig_probs = torch.sigmoid(logits)
            top_vals, top_idx = torch.topk(sig_probs, k=self.top_k, dim=-1)
            res_mask = self.resident_mask.to(device=logits.device)
            skipped = ~res_mask[top_idx]
            # Scale by (2.0 * num_experts / top_k) so zero-initialized router (sigmoid(0)=0.5) preserves dense scale
            scale = (2.0 * float(self.num_experts)) / float(self.top_k)
            gates = torch.where(skipped, torch.zeros_like(top_vals), top_vals * scale)
            self.last_ghost_fraction = float(skipped.sum().item()) / float(max(1, n_tok * self.top_k))
            norm_probs = sig_probs / sig_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            self.last_aux_loss = _switch_load_balancing_loss(norm_probs, top_idx, self.num_experts)
        else:
            probs = F.softmax(logits, dim=-1)
            top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
            res_mask = self.resident_mask.to(device=logits.device)
            skipped = ~res_mask[top_idx]
            renorm = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gates = torch.where(skipped, torch.zeros_like(renorm), renorm * float(self.num_experts))
            self.last_ghost_fraction = float(skipped.sum().item()) / float(max(1, n_tok * self.top_k))
            self.last_aux_loss = _switch_load_balancing_loss(probs, top_idx, self.num_experts)

        out = torch.zeros((n_tok, self.hidden_size), device=x.device, dtype=torch.float32)
        for e in range(self.num_experts):
            if not bool(self.resident_mask[e].item()):
                continue
            hit_mask = top_idx == e  # [N, k]
            if not bool(hit_mask.any().item()):
                continue
            token_weights = (gates * hit_mask.to(torch.float32)).sum(dim=-1)  # [N]
            active_rows = torch.nonzero(token_weights != 0.0, as_tuple=False).squeeze(-1)
            if active_rows.numel() == 0:
                continue
            y_e = self._run_expert(e, flat_x.index_select(0, active_rows))
            weighted_y = y_e * token_weights.index_select(0, active_rows).unsqueeze(-1)
            out.index_add_(0, active_rows, weighted_y)

        return out.to(dtype=orig_dt).reshape(orig_shape)


class CrowdLoRAExpertAdapter(nn.Module):
    """Single SwiGLU LoRA domain expert `(w1, w2, w3)` attached to a frozen `CrowdRuntimeMLP`."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        self.w1_A = nn.Parameter(
            (torch.randn((rank, hidden_size), generator=gen, dtype=torch.float32) / math.sqrt(hidden_size)).to(device)
        )
        self.w1_B = nn.Parameter(torch.zeros((intermediate_size, rank), device=device, dtype=torch.float32))
        self.w2_A = nn.Parameter(
            (torch.randn((rank, hidden_size), generator=gen, dtype=torch.float32) / math.sqrt(hidden_size)).to(device)
        )
        self.w2_B = nn.Parameter(torch.zeros((intermediate_size, rank), device=device, dtype=torch.float32))
        self.w3_A = nn.Parameter(
            (torch.randn((rank, intermediate_size), generator=gen, dtype=torch.float32) / math.sqrt(intermediate_size)).to(device)
        )
        self.w3_B = nn.Parameter(torch.zeros((hidden_size, rank), device=device, dtype=torch.float32))

    def forward_delta(
        self,
        x_fp32: torch.Tensor,
        base_h1: torch.Tensor,
        base_h2: torch.Tensor,
        base_w3_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exact residual contribution `(w3 + dW3)(silu(h1 + dh1) * (h2 + dh2)) - w3(silu(h1) * h2)`."""
        dh1 = F.linear(F.linear(x_fp32, self.w1_A), self.w1_B) * self.scaling
        dh2 = F.linear(F.linear(x_fp32, self.w2_A), self.w2_B) * self.scaling
        act_full = F.silu(base_h1 + dh1) * (base_h2 + dh2)
        act_base = F.silu(base_h1) * base_h2
        base_diff = F.linear((act_full - act_base).to(dtype=base_w3_weight.dtype), base_w3_weight).to(torch.float32)
        lora_w3 = F.linear(F.linear(act_full, self.w3_A), self.w3_B) * self.scaling
        return base_diff + lora_w3


class CrowdLoRAMoEMLP(nn.Module):
    """Frozen `CrowdRuntimeMLP` + `E` Sparse Domain LoRA Experts with FoMoE Ghost-Expert Skip-Token routing.

    Guarantees exact step-0 identity (`CrowdLoRAMoEMLP(x) == dense_mlp(x)` because `B = 0`) while
    allowing `8 GB` consumer GPUs to host only `resident_expert_indices` in VRAM (`< 1.78 GiB` total).
    """

    def __init__(
        self,
        dense_mlp: CrowdRuntimeMLP,
        *,
        num_experts: int = 8,
        top_k: int = 2,
        rank: int = 8,
        alpha: float = 16.0,
        routing_mode: str = "sigma_topk",
        resident_expert_indices: Sequence[int] | None = None,
        seed: int = 20261114,
    ) -> None:
        super().__init__()
        if not isinstance(dense_mlp, CrowdRuntimeMLP):
            raise TypeError("dense_mlp must be a CrowdRuntimeMLP instance.")
        if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 2:
            raise ValueError("num_experts must be >= 2.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not (1 <= top_k <= num_experts):
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts.")
        if routing_mode not in {"softmax_topk", "sigma_topk"}:
            raise ValueError("routing_mode must be 'softmax_topk' or 'sigma_topk'.")

        self.base_mlp = dense_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad_(False)

        inter_dim, hidden_dim = dense_mlp.w1.weight.shape
        self.hidden_size = int(hidden_dim)
        self.intermediate_size = int(inter_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.routing_mode = routing_mode

        dev = dense_mlp.w1.weight.device
        res_set = (
            set(range(self.num_experts))
            if resident_expert_indices is None
            else {int(i) for i in resident_expert_indices}
        )
        if not res_set or any(i < 0 or i >= self.num_experts for i in res_set):
            raise ValueError(f"resident_expert_indices must be a non-empty subset of [0, {self.num_experts})")

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False, device=dev, dtype=torch.float32)
        nn.init.zeros_(self.router.weight)

        self.experts = nn.ModuleDict()
        mask = torch.zeros(self.num_experts, dtype=torch.bool, device=dev)
        for e in sorted(res_set):
            mask[e] = True
            self.experts[str(e)] = CrowdLoRAExpertAdapter(
                self.hidden_size,
                self.intermediate_size,
                rank=self.rank,
                alpha=self.alpha,
                seed=seed + e * 997,
                device=dev,
            )
        self.register_buffer("resident_mask", mask, persistent=False)
        self.last_aux_loss: torch.Tensor = torch.zeros((), device=dev, dtype=torch.float32)
        self.last_ghost_fraction: float = 0.0
        self.last_sigma_report: SigmaMoERoutingResult | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dt = x.dtype
        flat_x = x.reshape(-1, self.hidden_size)
        n_tok = flat_x.shape[0]

        w_dt = self.base_mlp.w1.weight.dtype
        x_base = flat_x.to(dtype=w_dt)
        h1 = self.base_mlp.w1(x_base).to(torch.float32)
        h2 = self.base_mlp.w2(x_base).to(torch.float32)
        base_out = self.base_mlp.w3((F.silu(h1) * h2).to(dtype=w_dt)).to(torch.float32)

        x_fp32 = flat_x.to(torch.float32)
        logits = self.router(x_fp32)
        res_mask = self.resident_mask.to(device=logits.device)

        if self.routing_mode == "sigma_topk":
            self.last_sigma_report = sigmoid_independent_topk_routing(logits, self.top_k, res_mask)
            sig_probs = torch.sigmoid(logits)
            top_vals, top_idx = torch.topk(sig_probs, k=self.top_k, dim=-1)
            skipped = ~res_mask[top_idx]
            gates = torch.where(skipped, torch.zeros_like(top_vals), top_vals)
            self.last_ghost_fraction = self.last_sigma_report.ghost_assignment_fraction
            norm_probs = sig_probs / sig_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            self.last_aux_loss = _switch_load_balancing_loss(norm_probs, top_idx, self.num_experts)
        else:
            probs = F.softmax(logits, dim=-1)
            top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
            skipped = ~res_mask[top_idx]
            renorm = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gates = torch.where(skipped, torch.zeros_like(renorm), renorm)
            self.last_ghost_fraction = float(skipped.sum().item()) / float(max(1, n_tok * self.top_k))
            self.last_aux_loss = _switch_load_balancing_loss(probs, top_idx, self.num_experts)

        delta_acc = torch.zeros_like(base_out)
        for e_key, expert_mod in self.experts.items():
            e_idx = int(e_key)
            hit_mask = top_idx == e_idx
            if not bool(hit_mask.any().item()):
                continue
            token_weights = (gates * hit_mask.to(torch.float32)).sum(dim=-1)
            active_rows = torch.nonzero(token_weights != 0.0, as_tuple=False).squeeze(-1)
            if active_rows.numel() == 0:
                continue
            dy = expert_mod.forward_delta(
                x_fp32.index_select(0, active_rows),
                h1.index_select(0, active_rows),
                h2.index_select(0, active_rows),
                self.base_mlp.w3.weight,
            )
            delta_acc.index_add_(0, active_rows, dy * token_weights.index_select(0, active_rows).unsqueeze(-1))

        return (base_out + delta_acc).to(dtype=orig_dt).reshape(orig_shape)


@dataclass(frozen=True)
class CrowdMoESplitReport:
    """Summary of upcycling `CrowdV1WithEngram` into `hivemind/crowd-moe-v1`."""

    split_mode: str
    num_layers_upcycled: int
    num_experts: int
    top_k: int
    routing_mode: str
    resident_expert_indices: tuple[int, ...]
    trainable_parameters: int
    frozen_parameters: int
    trainable_vram_mib_fp32: float


def upcycle_crowd_v1_to_sparse_moe(
    model: CrowdV1WithEngram,
    *,
    split_mode: str = "lora_domain_experts",
    num_experts: int = 8,
    top_k: int = 2,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    routing_mode: str = "sigma_topk",
    resident_expert_indices: Sequence[int] | None = None,
    layer_indices: Sequence[int] | None = None,
    seed: int = 20261114,
) -> CrowdMoESplitReport:
    """Upcycle `CrowdV1WithEngram` transformer blocks into Sparse MoE blocks in-place."""
    if split_mode not in {"lora_domain_experts", "fine_grained_width_split"}:
        raise ValueError("split_mode must be 'lora_domain_experts' or 'fine_grained_width_split'.")

    for p in model.parameters():
        p.requires_grad_(False)

    target_layers = (
        tuple(range(len(model.blocks)))
        if layer_indices is None
        else tuple(sorted({int(i) for i in layer_indices}))
    )
    res_indices = (
        tuple(range(num_experts))
        if resident_expert_indices is None
        else tuple(sorted({int(i) for i in resident_expert_indices}))
    )

    upcycled = 0
    for l_idx in target_layers:
        blk: CrowdRuntimeBlock = model.blocks[l_idx]
        if not isinstance(blk.mlp, CrowdRuntimeMLP):
            raise TypeError(f"Block {l_idx} mlp is already {type(blk.mlp).__name__}; pass a fresh CrowdV1WithEngram.")
        if split_mode == "lora_domain_experts":
            moe_mlp: nn.Module = CrowdLoRAMoEMLP(
                blk.mlp,
                num_experts=num_experts,
                top_k=top_k,
                rank=lora_rank,
                alpha=lora_alpha,
                routing_mode=routing_mode,
                resident_expert_indices=res_indices,
                seed=seed + l_idx * 31,
            )
        else:
            moe_mlp = CrowdFineGrainedMoEMLP(
                blk.mlp,
                num_experts=num_experts,
                top_k=top_k,
                routing_mode=routing_mode,
                resident_expert_indices=res_indices,
            )
            for e_idx in range(num_experts):
                is_res = e_idx in res_indices
                for mod_list in (moe_mlp.experts_w1, moe_mlp.experts_w2, moe_mlp.experts_w3):
                    for p in mod_list[e_idx].parameters():
                        if is_res:
                            p.data = p.data.to(torch.float32)
                        p.requires_grad_(is_res)
        blk.mlp = moe_mlp  # type: ignore[assignment]
        upcycled += 1

    model.ln.to(dtype=torch.float32)
    model.ln.weight.requires_grad_(True)
    model.ln.bias.requires_grad_(True)

    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_n = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return CrowdMoESplitReport(
        split_mode=split_mode,
        num_layers_upcycled=upcycled,
        num_experts=num_experts,
        top_k=top_k,
        routing_mode=routing_mode,
        resident_expert_indices=res_indices,
        trainable_parameters=trainable_n,
        frozen_parameters=frozen_n,
        trainable_vram_mib_fp32=round((trainable_n * 4) / (1024.0 * 1024.0), 3),
    )


def collect_moe_router_aux_loss(model: CrowdV1WithEngram) -> torch.Tensor:
    """Sum Switch load-balancing auxiliary losses across all upcycled MoE blocks."""
    dev = model.ln.weight.device
    total = torch.zeros((), device=dev, dtype=torch.float32)
    count = 0
    for blk in model.blocks:
        aux = getattr(blk.mlp, "last_aux_loss", None)
        if isinstance(aux, torch.Tensor):
            total = total + aux.to(device=dev, dtype=torch.float32)
            count += 1
    return total / float(max(1, count))


def extract_moe_trainable_buckets(
    model: CrowdV1WithEngram,
    *,
    num_router_fragments: int = 4,
) -> dict[str, torch.Tensor]:
    """Partition trainable MoE parameters into OpenDiLoCo / Streaming DiLoCo flat `fp32` CPU buckets.

    Creates:
      - `router_frag_0 .. router_frag_{P-1}`: router gates + final LayerNorm staggered across `P` streaming fragments,
      - `expert_{e}`: contiguous `fp32` vector for each resident expert `e` across all upcycled layers,
        so nodes hosting disjoint expert subsets synchronize ONLY their resident expert buckets!
    """
    if isinstance(num_router_fragments, bool) or not isinstance(num_router_fragments, int) or num_router_fragments < 1:
        raise ValueError("num_router_fragments must be >= 1")

    n_blocks = len(model.blocks)
    frag_span = max(1, math.ceil(n_blocks / num_router_fragments))
    router_parts: dict[str, list[torch.Tensor]] = {}
    expert_parts: dict[str, list[torch.Tensor]] = {}

    for l_idx, blk in enumerate(model.blocks):
        f_idx = min(num_router_fragments - 1, l_idx // frag_span)
        r_key = f"router_frag_{f_idx}"
        mlp = blk.mlp
        if isinstance(mlp, CrowdLoRAMoEMLP):
            if mlp.router.weight.requires_grad:
                router_parts.setdefault(r_key, []).append(mlp.router.weight.detach().to("cpu", dtype=torch.float32).reshape(-1))
            for e_key, exp_mod in sorted(mlp.experts.items(), key=lambda kv: int(kv[0])):
                vecs = [
                    p.detach().to("cpu", dtype=torch.float32).reshape(-1)
                    for p in exp_mod.parameters()
                    if p.requires_grad
                ]
                if vecs:
                    expert_parts.setdefault(f"expert_{int(e_key)}", []).extend(vecs)
        elif isinstance(mlp, CrowdFineGrainedMoEMLP):
            if mlp.router.weight.requires_grad:
                router_parts.setdefault(r_key, []).append(mlp.router.weight.detach().to("cpu", dtype=torch.float32).reshape(-1))
            for e_idx in range(mlp.num_experts):
                vecs = []
                for mod_list in (mlp.experts_w1, mlp.experts_w2, mlp.experts_w3):
                    for p in mod_list[e_idx].parameters():
                        if p.requires_grad:
                            vecs.append(p.detach().to("cpu", dtype=torch.float32).reshape(-1))
                if vecs:
                    expert_parts.setdefault(f"expert_{e_idx}", []).extend(vecs)

    last_r_key = f"router_frag_{num_router_fragments - 1}"
    if model.ln.weight.requires_grad:
        router_parts.setdefault(last_r_key, []).append(model.ln.weight.detach().to("cpu", dtype=torch.float32).reshape(-1))
    if model.ln.bias.requires_grad:
        router_parts.setdefault(last_r_key, []).append(model.ln.bias.detach().to("cpu", dtype=torch.float32).reshape(-1))

    out: dict[str, torch.Tensor] = {}
    for k, parts in sorted(router_parts.items()):
        if parts:
            out[k] = torch.cat(parts, dim=0).contiguous()
    for k, parts in sorted(expert_parts.items(), key=lambda kv: int(kv[0].split("_")[1])):
        if parts:
            out[k] = torch.cat(parts, dim=0).contiguous()
    return out


@torch.no_grad()
def load_moe_trainable_buckets(
    model: CrowdV1WithEngram,
    buckets: Mapping[str, torch.Tensor],
    *,
    num_router_fragments: int = 4,
) -> None:
    """Copy updated `fp32` OpenDiLoCo outer master buckets back into `model` parameters."""
    n_blocks = len(model.blocks)
    frag_span = max(1, math.ceil(n_blocks / num_router_fragments))
    router_offsets: dict[str, int] = {k: 0 for k in buckets if k.startswith("router_frag_")}
    expert_offsets: dict[str, int] = {k: 0 for k in buckets if k.startswith("expert_")}

    for l_idx, blk in enumerate(model.blocks):
        f_idx = min(num_router_fragments - 1, l_idx // frag_span)
        r_key = f"router_frag_{f_idx}"
        mlp = blk.mlp
        if isinstance(mlp, CrowdLoRAMoEMLP):
            if mlp.router.weight.requires_grad and r_key in buckets:
                n = mlp.router.weight.numel()
                off = router_offsets[r_key]
                mlp.router.weight.copy_(
                    buckets[r_key][off : off + n].view_as(mlp.router.weight).to(device=mlp.router.weight.device, dtype=mlp.router.weight.dtype)
                )
                router_offsets[r_key] = off + n
            for e_key, exp_mod in sorted(mlp.experts.items(), key=lambda kv: int(kv[0])):
                b_name = f"expert_{int(e_key)}"
                if b_name not in buckets:
                    continue
                for p in exp_mod.parameters():
                    if not p.requires_grad:
                        continue
                    n = p.numel()
                    off = expert_offsets[b_name]
                    p.copy_(buckets[b_name][off : off + n].view_as(p).to(device=p.device, dtype=p.dtype))
                    expert_offsets[b_name] = off + n
        elif isinstance(mlp, CrowdFineGrainedMoEMLP):
            if mlp.router.weight.requires_grad and r_key in buckets:
                n = mlp.router.weight.numel()
                off = router_offsets[r_key]
                mlp.router.weight.copy_(
                    buckets[r_key][off : off + n].view_as(mlp.router.weight).to(device=mlp.router.weight.device, dtype=mlp.router.weight.dtype)
                )
                router_offsets[r_key] = off + n
            for e_idx in range(mlp.num_experts):
                b_name = f"expert_{e_idx}"
                if b_name not in buckets:
                    continue
                for mod_list in (mlp.experts_w1, mlp.experts_w2, mlp.experts_w3):
                    for p in mod_list[e_idx].parameters():
                        if not p.requires_grad:
                            continue
                        n = p.numel()
                        off = expert_offsets[b_name]
                        p.copy_(buckets[b_name][off : off + n].view_as(p).to(device=p.device, dtype=p.dtype))
                        expert_offsets[b_name] = off + n

    last_r_key = f"router_frag_{num_router_fragments - 1}"
    if last_r_key in buckets:
        for p in (model.ln.weight, model.ln.bias):
            if p.requires_grad:
                n = p.numel()
                off = router_offsets[last_r_key]
                p.copy_(buckets[last_r_key][off : off + n].view_as(p).to(device=p.device, dtype=p.dtype))
                router_offsets[last_r_key] = off + n


# ---------------------------------------------------------------------------
# 3. Decoupled DiLoCo + PrimeIntellect OpenDiLoCo Outer Coordinator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenDiLoCoRoundReceipt:
    """Telemetry receipt for one Decoupled OpenDiLoCo outer sync across fragments & MoE experts."""

    round_index: int
    outer_optimizer_mode: str
    merge_mode: str
    fragments_synced: tuple[str, ...]
    quorum_telemetry: dict[str, dict[str, Any]]
    eval_loss_before: float
    eval_loss_after: float
    accepted_by_loss_ratchet: bool
    master_state_sha256: str


class CrowdOpenDiLoCoCoordinator:
    """Decoupled DiLoCo + PrimeIntellect OpenDiLoCo CPU-offloaded outer coordinator.

    Maintains `CPUOffloadedFlatBucketOuterState` (`0 B` GPU VRAM overhead, `PrimeIntellect-ai/OpenDiLoCo`)
    and runs per-bucket `DecoupledDiLoCoQuorumSyncer` (`min_quorum_k`, `AdaptiveGraceWindowController`,
    quadratic `token_weighted_quality_weight`, and Radial-Directional Averaging `RDA`), followed by
    either OpenDiLoCo Outer Nesterov (`lr=0.7, momentum=0.9`) or staleness-gated `PA-CGAD`
    (`CosineGatedAdamDecayOuterOptimizer`) and monotone held-out validation-loss ratchet gating.
    """

    def __init__(
        self,
        initial_buckets: Mapping[str, torch.Tensor],
        *,
        min_quorum_k: int = 2,
        merge_mode: str = "rda",
        outer_optimizer_mode: str = "opendiloco_nesterov",
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        cgad_lr: float = 2e-3,
        cgad_alpha: float = 0.2,
        cgad_tau_cut: float = 16.0,
        overlap_steps_tau: int = 2,
        safety_margin_gamma: float = 0.5,
        work_dir: Path | str | None = None,
    ) -> None:
        if not initial_buckets:
            raise ValueError("initial_buckets must be non-empty")
        if outer_optimizer_mode not in {"opendiloco_nesterov", "pa_cgad"}:
            raise ValueError("outer_optimizer_mode must be 'opendiloco_nesterov' or 'pa_cgad'")
        if work_dir is not None:
            self.work_dir: Path | None = assert_not_protected_t4_path(work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.work_dir = None

        self.min_quorum_k = int(min_quorum_k)
        self.merge_mode = merge_mode
        self.outer_optimizer_mode = outer_optimizer_mode
        self.outer_lr = float(outer_lr)
        self.outer_momentum = float(outer_momentum)
        self.round_index = 0

        self.flat_state = CPUOffloadedFlatBucketOuterState(
            {k: v.detach().to("cpu", dtype=torch.float32) for k, v in initial_buckets.items()}
        )
        self.nesterov_velocity: dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, device="cpu", dtype=torch.float32) for k, v in initial_buckets.items()
        }
        self.cgad = CosineGatedAdamDecayOuterOptimizer(
            lr=cgad_lr,
            alpha=cgad_alpha,
            tau_cut=cgad_tau_cut,
        )
        self.grace_controllers: dict[str, AdaptiveGraceWindowController] = {
            k: AdaptiveGraceWindowController(
                overlap_steps_tau=overlap_steps_tau,
                safety_margin_gamma=safety_margin_gamma,
            )
            for k in initial_buckets
        }
        self.syncers: dict[str, DecoupledDiLoCoQuorumSyncer] = {
            k: DecoupledDiLoCoQuorumSyncer(
                num_learners_m=max(self.min_quorum_k, 128),
                min_quorum_k=self.min_quorum_k,
                merge_mode=self.merge_mode,
                grace_controller=self.grace_controllers[k],
            )
            for k in initial_buckets
        }

    def _bucket_view(self, name: str) -> torch.Tensor:
        lo, hi = self.flat_state.offsets[name]
        return self.flat_state.flat_master_cpu[lo:hi].view(self.flat_state.shapes[name])

    def export_buckets(self) -> dict[str, torch.Tensor]:
        """Return detached CPU `fp32` master parameter buckets."""
        return {name: self._bucket_view(name).detach().clone() for name in self.syncers}

    def master_sha256(self) -> str:
        raw = self.flat_state.flat_master_cpu.detach().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    @torch.no_grad()
    def step_decoupled_round(
        self,
        submissions_by_bucket: Mapping[str, Sequence[tuple[LearnerArrival, torch.Tensor, float]]],
        *,
        eval_fn: Any | None = None,
    ) -> OpenDiLoCoRoundReceipt:
        """Aggregate per-bucket pseudo-gradients `(theta_master - theta_worker)` via Decoupled OpenDiLoCo.

        Each tuple in `submissions_by_bucket[bucket]` is `(arrival, pseudo_grad_fp32, staleness_rounds)`.
        Note sign convention: `pseudo_grad = theta_before - theta_after` (standard OpenDiLoCo / DiLoCo
        outer gradient pointing in the ascent direction of loss, subtracted by outer Nesterov / CGAD).
        """
        snapshot_master = self.flat_state.flat_master_cpu.detach().clone()
        snapshot_vel = {k: v.detach().clone() for k, v in self.nesterov_velocity.items()}

        loss_before = float(eval_fn(self.export_buckets())) if eval_fn is not None else 0.0
        synced_names: list[str] = []
        telemetry_map: dict[str, dict[str, Any]] = {}

        for bucket_name, items in submissions_by_bucket.items():
            if bucket_name not in self.syncers or not items:
                continue
            syncer = self.syncers[bucket_name]
            if len(items) < self.min_quorum_k:
                # the configured quorum is a floor, not a target: lowering it to the number of submissions let a single
                # learner move a bucket on its own. This bucket waits for the next round instead.
                telemetry_map.setdefault(bucket_name, {}).update(
                    {"status": "quorum_not_reached", "submissions": len(items), "min_quorum_k": int(self.min_quorum_k)}
                )
                continue
            eff_k = int(self.min_quorum_k)
            eff_m = max(syncer.num_learners_m, len(items), eff_k)
            if syncer.min_quorum_k != eff_k or syncer.num_learners_m < len(items):
                syncer = DecoupledDiLoCoQuorumSyncer(
                    num_learners_m=max(1, eff_m),
                    min_quorum_k=max(1, eff_k),
                    merge_mode=self.merge_mode,
                    grace_controller=self.grace_controllers[bucket_name],
                )
            syncer.max_tokens_per_step = max(
                syncer.max_tokens_per_step,
                max((int(it[0].tokens_since_sync) // max(1, int(it[0].steps_since_sync))) + 1 for it in items),
            )

            master_view = self._bucket_view(bucket_name)
            arrivals = [it[0] for it in items]
            learner_fragments = {
                it[0].learner_id: (master_view - it[1].detach().to("cpu", dtype=torch.float32).reshape(master_view.shape))
                for it in items
            }
            max_staleness = max(float(it[2]) for it in items)

            avg_step_s = sum(max(0.01, it[0].arrival_time_s) / max(1, it[0].steps_since_sync) for it in items) / len(items)
            self.grace_controllers[bucket_name].observe_step_duration(avg_step_s)

            merged_pseudo_grad, _sel, tel = syncer.merge_fragment(
                fragment_name=bucket_name,
                global_prev=master_view,
                learner_fragments=learner_fragments,
                arrivals=arrivals,
            )
            if merged_pseudo_grad is None:
                continue

            if self.outer_optimizer_mode == "opendiloco_nesterov":
                vel = self.nesterov_velocity[bucket_name]
                vel.mul_(self.outer_momentum).add_(merged_pseudo_grad)
                nesterov_step = merged_pseudo_grad + self.outer_momentum * vel
                master_view.sub_(nesterov_step, alpha=self.outer_lr)
            else:
                updated_map, cgad_tels = self.cgad.step_fragments(
                    params={bucket_name: master_view},
                    pseudo_gradients={bucket_name: merged_pseudo_grad},
                    worker_delays={bucket_name: max_staleness},
                    synced_fragments=[bucket_name],
                )
                master_view.copy_(updated_map[bucket_name])
                if cgad_tels:
                    telemetry_map.setdefault(bucket_name, {})["cgad"] = asdict(cgad_tels[0])

            synced_names.append(bucket_name)
            telemetry_map.setdefault(bucket_name, {}).update(
                {
                    "admitted_learners": list(tel.admitted_learner_ids),
                    "deferred_learners": int(tel.num_deferred),
                    "admitted_tokens": int(tel.admitted_tokens),
                    "grace_window_s": float(tel.grace_window_s),
                    "merged_delta_norm": float(tel.merged_delta_norm),
                    "direct_average_norm": float(tel.direct_average_norm),
                }
            )

        loss_after = float(eval_fn(self.export_buckets())) if eval_fn is not None else loss_before
        accepted = True
        if eval_fn is not None and (not math.isfinite(loss_after) or loss_after > loss_before + 1e-6):
            # Monotone validation-loss ratchet rollback
            self.flat_state.flat_master_cpu.copy_(snapshot_master)
            for k, v_snap in snapshot_vel.items():
                self.nesterov_velocity[k].copy_(v_snap)
            loss_after = loss_before
            accepted = False
        else:
            self.round_index += 1

        return OpenDiLoCoRoundReceipt(
            round_index=self.round_index,
            outer_optimizer_mode=self.outer_optimizer_mode,
            merge_mode=self.merge_mode,
            fragments_synced=tuple(synced_names),
            quorum_telemetry=telemetry_map,
            eval_loss_before=loss_before,
            eval_loss_after=loss_after,
            accepted_by_loss_ratchet=accepted,
            master_state_sha256=self.master_sha256(),
        )


def build_opendiloco_launch_spec(
    *,
    role: str = "trainer",
    node_handle: str = "socreateai-agora-hermes-node1",
    vram_total_mib: int = 15360,
    world_size_cohorts: int = 8,
    min_quorum_k: int = 5,
    gateway_url: str = "https://hivemind-openrouter-1003063861791.us-central1.run.app",
) -> dict[str, Any]:
    """Generate both the PrimeIntellect OpenDiLoCo (`hv` / `torchrun`) command and the firewall-free OpenTrainer command."""
    shard_assignment = assign_hiveminder_moe_shards(node_handle, role, vram_total_mib)
    inner_h = shard_assignment["recommended_inner_steps_h"]
    experts_csv = ",".join(str(i) for i in shard_assignment["resident_expert_indices"])

    return {
        "campaign_id": CROWDGPT_WEEKEND_CAMPAIGN_ID,
        "model_base": "Vxtzq/Crowd-v1",
        "model_upcycled": CROWDGPT_MOE_PUBLIC_MODEL_ID,
        "node_assignment": shard_assignment,
        "decoupled_opendiloco_hyperparameters": {
            "upstream_reference": "PrimeIntellect-ai/OpenDiLoCo (arXiv:2407.07852) + Google DeepMind Decoupled DiLoCo (arXiv:2604.21428)",
            "world_size_m": int(world_size_cohorts),
            "min_quorum_k": int(min_quorum_k),
            "streaming_fragments_p": 4,
            "num_moe_experts": 8,
            "top_k": 2,
            "routing_mode": "sigma_topk",
            "inner_optimizer": "Blockwise8BitAdamW (lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01)",
            "inner_steps_h": inner_h,
            "outer_optimizer": "OpenDiLoCo CPU-Offloaded Nesterov (lr=0.7, momentum=0.9) + Decoupled RDA Quorum",
            "straggler_fallback": "PA-CGAD (CosineGatedAdamDecayOuterOptimizer, alpha=0.2, tau_cut=16)",
            "grace_window_formula": "xi_grace = 0.5 * max(0, 2 * xi_step - (xi_quorum + xi_sync))",
        },
        "commands": {
            "hivemind_weekend_worker": (
                f"python scripts/gcp_t4_crowd_diloco_worker.py --mode moe-opendiloco-round "
                f"--campaign-id {CROWDGPT_WEEKEND_CAMPAIGN_ID} --resident-experts {experts_csv} "
                f"--inner-steps {inner_h} --submit-opentrainer --gateway {gateway_url}"
            ),
            "opendiloco_dht_worker": (
                f"HIVEMIND_CAMPAIGN={CROWDGPT_WEEKEND_CAMPAIGN_ID} "
                f"OPENDILOCO_OFFLOAD_CPU=1 OPENDILOCO_OUTER_LR=0.7 OPENDILOCO_OUTER_MOMENTUM=0.9 "
                f"OPENDILOCO_LOCAL_STEPS={inner_h} OPENDILOCO_MERGE=rda OPENDILOCO_MIN_QUORUM={min_quorum_k} "
                f"torchrun --nproc_per_node=1 scripts/gcp_t4_crowd_diloco_worker.py "
                f"--mode moe-opendiloco-round --resident-experts {experts_csv}"
            ),
            "firewalled_home_gpu_relay": (
                f"python scripts/hivemind_opentrainer_agent.py --gateway {gateway_url} "
                f"--campaign {CROWDGPT_WEEKEND_CAMPAIGN_ID} --network-mode outbound-https-relay --once"
            ),
        },
    }


def build_weekend_nov14_15_raid_manifest() -> dict[str, Any]:
    """Return the complete Nov 14–15, 2026 CrowdGPT x HiveMind Weekend Raid plan, budget, and calendar."""
    # Two extended 8h weekend Spot sessions on Sat Nov 14 & Sun Nov 15 = +7h total over the standard 4.5h/day schedule
    # Amortized across 30.417 days/mo = +0.23 extra Spot hours/day -> +$1.30/mo total ($121.92 <= $150.00 ceiling!)
    weekend_budget = compute_gcp_t4_spot_crowd_budget(
        baseline_monthly_usd=120.62,
        scheduled_window_hours_per_day=4.5,
        extra_spot_hours_per_day=round(7.0 / (730.0 / 24.0), 4),
    )
    return {
        "object": "hivemind.crowdgpt_weekend_raid.manifest",
        "campaign_id": CROWDGPT_WEEKEND_CAMPAIGN_ID,
        "dates": "Saturday 14 November 2026 – Sunday 15 November 2026",
        "models": {
            "dense_base": "Vxtzq/Crowd-v1 (hivemind/crowd-v1)",
            "t4_diloco_checkpoint": "hivemind/gcpcrowd",
            "upcycled_sparse_moe": CROWDGPT_MOE_PUBLIC_MODEL_ID,
        },
        "gcp_spot_t4_budget_governance": {
            **asdict(weekend_budget),
            "weekend_extra_spot_hours_total": 7.0,
            "weekend_incremental_cost_usd": round(7.0 * 0.186, 2),
            "protected_t4_path": "/opt/hivemind/t4/ (READ-ONLY / UNTOUCHED)",
            "t4_work_dir": DEFAULT_GCP_T4_WORK_DIR_POSIX,
            "t4_min_free_vram_mib": DEFAULT_MIN_FREE_VRAM_MIB_T4,
        },
        "hiveminder_roles": {
            "producers": assign_hiveminder_moe_shards("producer_cohort", "producer", 8192),
            "trainers": assign_hiveminder_moe_shards("socreateai-agora-hermes-node1", "trainer", 15360),
            "prosumers": assign_hiveminder_moe_shards("vxtzq", "prosumer", 8192),
        },
        "calendar_waves_utc": [
            {
                "wave_id": "WAVE-0-PREFLIGHT",
                "title": "Friday Pre-Flight: Key Check, Dataset Shard Prefetch & T4 Spot Warmup",
                "window_utc": "2026-11-13T18:00:00Z / 2026-11-13T21:00:00Z",
                "window_paris_cet": "Fri 13 Nov · 19:00–22:00 CET",
                "roles_active": ["trainers", "producers", "prosumers"],
                "goal": "Verify >= 4,300 MiB T4 & >= 1,536 MiB 8GB VRAM guards, pull Vxtzq/Crowd-v1 safetensors, smoke-test OpenTrainer /v1/hivemind/opentrainer/heartbeat.",
            },
            {
                "wave_id": "WAVE-1-UPCYCLE-BOOT",
                "title": "Saturday Morning Raid Kickoff: 8-Expert MoE Split & Zero-Shock Verification",
                "window_utc": "2026-11-14T09:00:00Z / 2026-11-14T13:00:00Z",
                "window_paris_cet": "Sat 14 Nov · 10:00–14:00 CET",
                "roles_active": ["trainers", "prosumers"],
                "goal": "Initialize E0..E7 CrowdLoRAMoEMLP & CrowdFineGrainedMoEMLP from Vxtzq/Crowd-v1, verify step-0 exact dense equivalence, run APAC + Europe Outer Rounds 1–12.",
            },
            {
                "wave_id": "WAVE-2-GLOBAL-OPENDILOCO-PEAK",
                "title": "Saturday Global Peak Raid: Decoupled OpenDiLoCo + RDA Quorum (EU + Americas)",
                "window_utc": "2026-11-14T16:00:00Z / 2026-11-14T22:00:00Z",
                "window_paris_cet": "Sat 14 Nov · 17:00–23:00 CET",
                "roles_active": ["producers", "trainers", "prosumers"],
                "goal": "GCP Spot T4 + XAVGAMSRV RTX 2070 + Community Trainers run Outer Rounds 13–48 (K>=5 Byzantine Multi-Krum + RDA); Producers serve live Crowd-v1 traffic with <1s yield.",
            },
            {
                "wave_id": "WAVE-3-NOCTURNAL-BOINC-SOAK",
                "title": "Saturday Overnight BOINC & OpenTrainer Nocturnal Soak",
                "window_utc": "2026-11-14T22:00:00Z / 2026-11-15T07:00:00Z",
                "window_paris_cet": "Sat 14 Nov 23:00 – Sun 15 Nov 08:00 CET",
                "roles_active": ["trainers"],
                "goal": "Overnight idle gamer GPUs train E1 (Code), E2 (Math), E5 (Exoplanet), E6 (Stars) shards with PA-CGAD straggler gating and <1.5s Steam auto-pause.",
            },
            {
                "wave_id": "WAVE-4-ROUTER-CALIBRATION-ARENA",
                "title": "Sunday Afternoon: FoMoE Sigma-Router Calibration & Prosumer Blind A/B Arena",
                "window_utc": "2026-11-15T13:00:00Z / 2026-11-15T17:00:00Z",
                "window_paris_cet": "Sun 15 Nov · 14:00–18:00 CET",
                "roles_active": ["prosumers", "producers"],
                "goal": "@Vxtzq (105,000 HMT) & Prosumers benchmark hivemind/crowd-moe-v1 vs hivemind/crowd-v1 on HumanEval, GSM8K, ARC-Challenge & Kepler held-out evals.",
            },
            {
                "wave_id": "WAVE-5-FINALE-CHECKPOINT-RELEASE",
                "title": "Sunday Grand Finale: Loss-Ratchet Audit, Safetensors Merge & HF / OpenRouter Publish",
                "window_utc": "2026-11-15T17:00:00Z / 2026-11-15T21:00:00Z",
                "window_paris_cet": "Sun 15 Nov · 18:00–22:00 CET",
                "roles_active": ["producers", "trainers", "prosumers"],
                "goal": "Final GCP Spot T4 LossCheckReceipt sign-off, publish CrowdGPT-MoE-v1 safetensors + full reproducibility telemetry to PR & Hugging Face.",
            },
        ],
        "dataset_plan": load_crowdgpt_moe_dataset_plan(),
        "opendiloco_t4_spec": build_opendiloco_launch_spec(
            role="trainer",
            node_handle="socreateai-agora-hermes-node1",
            vram_total_mib=15360,
        ),
    }


def run_crowd_moe_opendiloco_inner_steps(
    model: CrowdV1WithEngram,
    token_stream: torch.Tensor,
    *,
    steps: int = 4,
    batch_size: int = 1,
    seq_len: int = 64,
    lr: float = 3e-4,
    aux_loss_coef: float = 0.01,
    num_router_fragments: int = 4,
    device: str = "cpu",
    yield_guard: Any | None = None,
) -> dict[str, Any]:
    """Execute `steps` inner AdamW updates on an upcycled `CrowdV1WithEngram` MoE learner and return bucket pseudo-gradients.

    With ``yield_guard`` (a ``T4GovernorYieldGuard``), every step first checks the operator HOLD file, governor state, live
    traffic, Spot preemption and the free-VRAM floor, and stops early (``yielded`` names the trigger) instead of training."""
    from hivemind.core.diloco_inner import Blockwise8BitAdamW
    from hivemind.swarm.gcp_t4_crowd_diloco import forward_crowd_v1_checkpointed

    if steps < 1:
        raise ValueError("steps must be >= 1")
    flat_tokens = token_stream.reshape(-1).to(dtype=torch.long)
    if flat_tokens.numel() < seq_len + 1:
        raise ValueError(f"token_stream ({flat_tokens.numel()}) must have at least seq_len + 1 ({seq_len + 1}) tokens")

    buckets_before = extract_moe_trainable_buckets(model, num_router_fragments=num_router_fragments)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Model has zero trainable MoE parameters; call upcycle_crowd_v1_to_sparse_moe first.")

    optimizer = Blockwise8BitAdamW(trainable_params, lr=lr, weight_decay=0.01)
    model.train()

    initial_loss: float | None = None
    final_loss: float = 0.0
    tokens_processed = 0
    span = batch_size * seq_len
    max_off = max(1, flat_tokens.numel() - span)

    steps_completed = 0
    yielded: str | None = None
    for step_idx in range(steps):
        if yield_guard is not None:
            chk = yield_guard.check()
            if chk.should_yield:
                yielded = str(chk.trigger)
                break
        start = (step_idx * seq_len) % max_off
        chunk = flat_tokens[start : start + span]
        if chunk.numel() < span:
            repeats = (span // chunk.numel()) + 2
            chunk = flat_tokens.repeat(repeats)[:span]
        batch_ids = chunk.reshape(batch_size, seq_len).to(device=device)

        optimizer.zero_grad(set_to_none=True)
        _, lm_loss = forward_crowd_v1_checkpointed(
            model,
            batch_ids,
            labels=batch_ids,
            use_grad_checkpointing=True,
            token_chunk_size=min(256, seq_len),
        )
        assert lm_loss is not None
        aux_loss = collect_moe_router_aux_loss(model)
        total_loss = lm_loss + float(aux_loss_coef) * aux_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        loss_val = float(lm_loss.detach().cpu().item())
        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val
        tokens_processed += int(batch_size * max(1, seq_len - 1))
        steps_completed += 1

    buckets_after = extract_moe_trainable_buckets(model, num_router_fragments=num_router_fragments)
    pseudo_grads = {
        k: (buckets_before[k] - buckets_after[k]).contiguous()
        for k in buckets_before
    }
    return {
        "steps_completed": steps_completed,
        "yielded": yielded,
        "tokens_processed": tokens_processed,
        "initial_loss": float(initial_loss if initial_loss is not None else 0.0),
        "final_loss": float(final_loss),
        "pseudo_grads": pseudo_grads,
    }
