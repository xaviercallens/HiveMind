"""Season 3 (`HM-S3-04`), Season 4 (`HM-S4-05`) & Season 5 (`HM-S5-02`) BTM Sparse MoE.

Re-exports the core Branch-Train-Merge (`btm_upcycle`) primitives and provides
`Season3DomainExpertRegistry` for managing and routing specialized domain QLoRA
adapters (`Rust_Systems_Security`, `Solidity_Smart_Contracts`,
`Biomedical_Clinical_QA`, and `Lean4_Theorem_Prover`).

Season 4 (`HM-S4-05`) adds `project_lora_rank_svd`: a truncated-SVD rank-adaptive
LoRA projection that compresses higher-rank community adapters (`r=32`, `r=64`) and
zero-pads lower-rank ones (`r=8`) onto a single canonical rank (`r=16`), so that
16-32 heterogeneous community experts can be hot-plugged into one unified
`UpcycledSparseMoEModel` inside a `< 2.0 GB` VRAM budget.

Season 5 (`HM-S5-02`) adds the **64-Expert 2-Level Hierarchical Sparse MoE**:

- `HierarchicalMoERouter`: a `G_macro -> G_micro` router over `8` macro-domains x
  `8` micro-experts (`E = 64`) that selects the **Top-2 micro-experts inside the
  winning macro-cluster**. It implements the exact `TopKMoERouter` contract
  (`forward(hidden_states) -> (routing_weights, selected_experts, aux_loss)`) and
  returns **global** expert indices, so it is a drop-in replacement inside
  `SparseMoEMLPBlock`.
- `hotplug_hierarchical_experts` / `build_hierarchical_sparse_moe`: hot-plug 64
  heterogeneous community adapters (all SVD-projected onto the canonical `r=16`)
  and install the hierarchical router, keeping the whole model inside the
  `< 2.2 GB` VRAM ceiling enforced by the FP8 paged KV-cache of
  `hivemind.swarm.swarm_fabric`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.core.btm_upcycle import (
    ExpertBranchState,
    MoEUpcycleReport,
    SparseMoEMLPBlock,
    SVDLoARankProjectionReport,
    TopKMoERouter,
    UpcycledFactoredSwiGLUExpert,
    UpcycledSparseMoEModel,
    UpcycledSwiGLUExpert,
    detect_lora_rank,
    hotplug_rank_adaptive_experts,
    initialize_domain_branches,
    measure_moe_vram_footprint_gb,
    pair_lora_ab_keys,
    project_lora_state_dict_to_rank,
    svd_project_lora_pair,
    train_moe_router_gating,
    upcycle_branches_to_sparse_moe,
)

SEASON_3_DOMAIN_EXPERT_SPECS: Dict[str, Dict[str, Any]] = {
    "Rust_Systems_Security": {
        "dataset_id": "anon/rust-code-instructions",
        "lora_rank_r": 16,
        "lora_alpha": 32,
        "keywords": ["fn", "pub", "mut", "arc", "mutex", "unsafe", "borrow", "lifetime", "rust"],
        "description": "Zero-copy systems programming, borrow-checker invariants, and memory-safe concurrency in Rust.",
    },
    "Solidity_Smart_Contracts": {
        "dataset_id": "hivemind/solidity-audit-invariants",
        "lora_rank_r": 16,
        "lora_alpha": 32,
        "keywords": ["solidity", "pragma", "contract", "reentrancy", "erc20", "uint256", "msg.sender", "require"],
        "description": "EVM smart contract security auditing, CEI reentrancy defense, and fixed-point tokenomics invariants.",
    },
    "Biomedical_Clinical_QA": {
        "dataset_id": "qiaojin/PubMedQA",
        "lora_rank_r": 16,
        "lora_alpha": 32,
        "keywords": ["clinical", "patient", "atp", "mitochondria", "apoptosis", "therapy", "trial", "enzyme"],
        "description": "Evidence-based clinical synthesis, cellular bioenergetics, and PubMedQA diagnostic reasoning.",
    },
    "Lean4_Theorem_Prover": {
        "dataset_id": "hivemind/lean4-mathlib-proofs",
        "lora_rank_r": 16,
        "lora_alpha": 32,
        "keywords": ["theorem", "lemma", "omega", "simp", "nat", "lean", "by", "induction", "proof"],
        "description": "Machine-checked Lean 4 formal verification with zero sorrys and zero custom axioms.",
    },
}


@dataclass
class RegisteredDomainExpert:
    """Metadata and CAS location for a trained Season 3 domain LoRA adapter."""

    domain_name: str
    dataset_id: str
    expert_cid: str
    shard_cid: str
    lora_rank_r: int
    lora_alpha: int
    initial_loss: float
    final_loss: float
    compression_ratio: float
    compressed_bytes: int
    lora_state_dict: Dict[str, torch.Tensor] = field(repr=False)


class ProjectedLoRAStateDict(dict):
    """Dict of projected LoRA tensors that also exposes metadata keys (`state_dict`, `factors`, `target_rank`, `parameter_bytes`)."""

    def __init__(self, tensors: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> None:
        super().__init__(tensors)
        self._meta = meta

    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        if key == "state_dict":
            return dict(self)
        if key in self._meta:
            return self._meta[key]
        raise KeyError(key)


def project_lora_rank_svd(
    lora_state_dict: Dict[str, torch.Tensor],
    target_rank_r: int = 16,
    source_lora_alpha: float | None = None,
    target_lora_alpha: float | None = None,
) -> Dict[str, Any]:
    """Season 4 (`HM-S4-05`): truncated-SVD rank-adaptive LoRA projection.

    Supports both paired `(lora_A, lora_B)` PEFT state dicts (PR #34) and standalone
    2D LoRA factor dictionaries (PR #33), returning a `ProjectedLoRAStateDict` that behaves
    as a tensor `state_dict` while also exposing `["state_dict"]`, `["factors"]`,
    `["target_rank"]`, and `["parameter_bytes"]`.
    """
    if target_rank_r < 1:
        raise ValueError("target_rank_r must be positive")

    pairs = pair_lora_ab_keys(lora_state_dict)
    if pairs:
        projected, report = project_lora_state_dict_to_rank(
            lora_state_dict,
            target_rank_r=target_rank_r,
            source_lora_alpha=source_lora_alpha,
            target_lora_alpha=target_lora_alpha,
        )
        factors: List[Dict[str, Any]] = [
            {
                "name": name,
                "source_rank": report.source_rank_r,
                "kept_rank": min(report.source_rank_r, target_rank_r),
                "target_rank": target_rank_r,
                "output_shape": list(t.shape),
                "spectral_energy_kept": report.mean_spectral_energy_retained,
            }
            for name, t in projected.items()
            if isinstance(t, torch.Tensor)
        ]
        param_bytes = sum(
            t.numel() * t.element_size() for t in projected.values() if isinstance(t, torch.Tensor)
        )
        return ProjectedLoRAStateDict(
            projected,
            {
                "target_rank": target_rank_r,
                "factors": factors,
                "parameter_bytes": param_bytes,
                "report": report,
            },
        )

    projected_single: Dict[str, torch.Tensor] = {}
    reports: List[Dict[str, Any]] = []
    for name, weight in lora_state_dict.items():
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            projected_single[name] = weight.detach().clone() if isinstance(weight, torch.Tensor) else weight
            continue
        matrix = weight.detach().to(torch.float32)
        rank_on_rows = matrix.shape[0] <= matrix.shape[1]
        current_rank = int(matrix.shape[0] if rank_on_rows else matrix.shape[1])
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        keep = min(target_rank_r, int(singular.shape[0]), current_rank)
        core = (u[:, :keep] * singular[:keep].unsqueeze(0)) @ vh[:keep, :]
        if rank_on_rows:
            if keep < target_rank_r:
                core = torch.nn.functional.pad(core, (0, 0, 0, target_rank_r - keep))
            else:
                core = core[:target_rank_r, :]
        else:
            if keep < target_rank_r:
                core = torch.nn.functional.pad(core, (0, target_rank_r - keep, 0, 0))
            else:
                core = core[:, :target_rank_r]
        projected_single[name] = core.to(dtype=weight.dtype)
        kept_energy = float(torch.linalg.vector_norm(singular[:keep]).item())
        full_energy = float(torch.linalg.vector_norm(singular).item())
        reports.append(
            {
                "name": name,
                "source_rank": current_rank,
                "kept_rank": keep,
                "target_rank": target_rank_r,
                "output_shape": list(core.shape),
                "spectral_energy_kept": round(kept_energy / max(full_energy, 1e-8), 6),
            }
        )
    parameter_bytes = sum(
        t.numel() * t.element_size() for t in projected_single.values() if isinstance(t, torch.Tensor)
    )
    return ProjectedLoRAStateDict(
        projected_single,
        {
            "target_rank": target_rank_r,
            "factors": reports,
            "parameter_bytes": parameter_bytes,
        },
    )


def project_lora_rank_svd_with_report(
    lora_state_dict: Dict[str, torch.Tensor],
    target_rank_r: int = 16,
    source_lora_alpha: float | None = None,
    target_lora_alpha: float | None = None,
) -> Tuple[Dict[str, torch.Tensor], SVDLoARankProjectionReport]:
    """Same as `project_lora_rank_svd` but also returns the spectral-energy report."""
    return project_lora_state_dict_to_rank(
        lora_state_dict,
        target_rank_r=target_rank_r,
        source_lora_alpha=source_lora_alpha,
        target_lora_alpha=target_lora_alpha,
    )


class Season3DomainExpertRegistry:
    """Registry managing Season 3 domain LoRA adapters and dynamic Top-2 MoE hot-plugging (`HM-S3-04`)."""

    def __init__(self) -> None:
        self.experts: Dict[str, RegisteredDomainExpert] = {}

    def register_expert(self, expert: RegisteredDomainExpert) -> None:
        self.experts[expert.domain_name] = expert

    def list_domains(self) -> List[str]:
        return list(self.experts.keys())

    def hot_plug_all_into_moe(
        self,
        moe_model: UpcycledSparseMoEModel,
        target_rank_r: int | None = None,
    ) -> Dict[str, Any]:
        """Hot-plugs all registered Season 3 domain adapters into a live `UpcycledSparseMoEModel`.

        Args:
            moe_model: Live Sparse MoE receiving the domain experts.
            target_rank_r: When set, every adapter is first projected onto this
                canonical rank with `project_lora_rank_svd` (`HM-S4-05`), so
                heterogeneous community ranks can coexist without VRAM bloat.
        """
        initial_experts = moe_model.num_experts
        plugged_domains: List[str] = []
        for domain_name, exp in self.experts.items():
            if domain_name not in moe_model.expert_domains:
                lora_state = exp.lora_state_dict
                if target_rank_r is not None:
                    lora_state = project_lora_rank_svd(
                        lora_state,
                        target_rank_r=target_rank_r,
                        source_lora_alpha=float(exp.lora_alpha),
                    )
                moe_model.hotplug_domain_expert(
                    domain_name=domain_name,
                    lora_state_dict=lora_state,
                    lora_rank_r=exp.lora_rank_r,
                    lora_alpha=exp.lora_alpha,
                )
                plugged_domains.append(domain_name)
        return {
            "initial_num_experts": initial_experts,
            "final_num_experts": moe_model.num_experts,
            "plugged_domains": plugged_domains,
            "active_branch_names": list(moe_model.expert_domains),
        }

    def route_domain_query(self, prompt: str, top_k: int = 2) -> Tuple[List[str], Dict[str, float]]:
        """Scores a prompt against Season 3 domain keyword signatures and returns Top-k expert weights."""
        p_low = prompt.lower()
        raw_scores: Dict[str, float] = {}
        for domain, spec in SEASON_3_DOMAIN_EXPERT_SPECS.items():
            hits = sum(1 for kw in spec["keywords"] if kw in p_low)
            raw_scores[domain] = 1.0 + hits * 2.5

        ranked = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = ranked[: max(1, int(top_k))]
        total = sum(score for _, score in selected) or 1.0
        weights = {dom: round(score / total, 4) for dom, score in selected}
        return [dom for dom, _ in selected], weights




def _even_cluster_sizes(total: int, num_clusters: int) -> List[int]:
    """Split `total` experts into `num_clusters` groups whose sizes differ by at most one."""
    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")
    base, extra = divmod(int(total), int(num_clusters))
    return [base + (1 if i < extra else 0) for i in range(int(num_clusters))]


def _stack_ragged_gates(
    gates: "nn.ModuleList", hidden_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad per-cluster gate matrices to a common width; padded rows are masked out."""
    width = max(g.out_features for g in gates)
    device = gates[0].weight.device
    stacked = torch.zeros((len(gates), width, hidden_size), dtype=torch.float32, device=device)
    valid = torch.zeros((len(gates), width), dtype=torch.bool, device=device)
    for i, g in enumerate(gates):
        stacked[i, : g.out_features] = g.weight
        valid[i, : g.out_features] = True
    return stacked, valid


def _grow_gate(gate: nn.Linear) -> nn.Linear:
    """Return a copy of `gate` with one extra output row initialised to the mean row."""
    old_out = gate.out_features
    new_gate = nn.Linear(
        gate.in_features, old_out + 1, bias=False, dtype=torch.float32, device=gate.weight.device
    )
    with torch.no_grad():
        new_gate.weight[:old_out].copy_(gate.weight)
        new_gate.weight[old_out:].copy_(gate.weight.mean(dim=0, keepdim=True))
    return new_gate


class HierarchicalMoERouter(nn.Module):
    """Season 5 (`HM-S5-02`): 2-Level `G_macro -> G_micro` Router for 64-Expert Sparse MoE.

    Partitions `E = sum(cluster_sizes)` experts into `M` macro-clusters (default `8 * 8 = 64`).
    For each token:
      1. `macro_gate(x)` selects the winning macro-domain `m* in {0..M-1}` (`M` inner products).
      2. `micro_gates[m*](x)` selects the Top-`k` micro-experts inside cluster `m*`.
    Gating cost per token is `M + max(cluster_sizes)` length-`hidden_size` inner products
    instead of `E` for a flat router. Clusters may be ragged after hot-plugging, so global
    expert indices come from per-cluster offsets rather than `m * experts_per_domain`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_macro_domains: int = 8,
        experts_per_domain: int = 8,
        top_k: int = 2,
        dtype: torch.dtype = torch.float32,
        cluster_sizes: List[int] | None = None,
    ) -> None:
        super().__init__()
        _ = dtype
        if num_macro_domains < 1 or experts_per_domain < 1:
            raise ValueError("num_macro_domains and experts_per_domain must be >= 1")
        sizes = (
            [int(s) for s in cluster_sizes]
            if cluster_sizes is not None
            else [int(experts_per_domain)] * int(num_macro_domains)
        )
        if not sizes or any(s < 1 for s in sizes):
            raise ValueError("every macro cluster needs at least one expert")
        self.hidden_size = int(hidden_size)
        self.cluster_sizes = sizes
        self.num_macro_domains = len(sizes)
        self.top_k = max(1, min(int(top_k), min(sizes)))

        self.macro_gate = nn.Linear(
            self.hidden_size, self.num_macro_domains, bias=False, dtype=torch.float32
        )
        self.micro_gates = nn.ModuleList(
            [nn.Linear(self.hidden_size, s, bias=False, dtype=torch.float32) for s in sizes]
        )
        nn.init.normal_(self.macro_gate.weight, mean=0.0, std=0.02)
        for mg in self.micro_gates:
            nn.init.normal_(mg.weight, mean=0.0, std=0.02)

    @property
    def num_experts(self) -> int:
        return sum(self.cluster_sizes)

    @num_experts.setter
    def num_experts(self, value: int) -> None:
        if int(value) != sum(self.cluster_sizes):
            raise ValueError("num_experts is derived from cluster_sizes; use expand_for_new_expert()")

    @property
    def experts_per_domain(self) -> int:
        return max(self.cluster_sizes)

    @property
    def cluster_offsets(self) -> List[int]:
        offsets, acc = [], 0
        for s in self.cluster_sizes:
            offsets.append(acc)
            acc += s
        return offsets

    @property
    def gating_inner_products_per_token(self) -> int:
        """Exact count of length-`hidden_size` inner products per token (not a timing)."""
        return self.num_macro_domains + max(self.cluster_sizes)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hierarchical Top-k routing weights, global expert indices `[0..E-1]`, and aux loss."""
        hs_fp32 = hidden_states.to(torch.float32)
        device = hs_fp32.device

        macro_logits = self.macro_gate(hs_fp32)  # [N, M]
        macro_probs = F.softmax(macro_logits, dim=-1, dtype=torch.float32)
        winning_macro_prob, winning_macro_idx = torch.max(macro_probs, dim=-1)  # [N]

        micro_weight_stack, valid = _stack_ragged_gates(self.micro_gates, self.hidden_size)
        token_micro_weights = micro_weight_stack[winning_macro_idx]  # [N, W, d]
        micro_logits = torch.bmm(token_micro_weights, hs_fp32.unsqueeze(-1)).squeeze(-1)
        micro_logits = micro_logits.masked_fill(~valid[winning_macro_idx], float("-inf"))

        top_micro_logits, top_micro_indices = torch.topk(micro_logits, k=self.top_k, dim=-1)
        micro_probs = F.softmax(top_micro_logits, dim=-1, dtype=torch.float32)
        routing_weights = micro_probs * winning_macro_prob.unsqueeze(-1)
        routing_weights = (
            routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ).to(hidden_states.dtype)

        offsets = torch.tensor(self.cluster_offsets, dtype=torch.long, device=device)
        global_selected_experts = offsets[winning_macro_idx].unsqueeze(-1) + top_micro_indices

        # Switch-style hierarchical load-balancing auxiliary loss
        one_hot_macro = F.one_hot(
            winning_macro_idx, num_classes=self.num_macro_domains
        ).to(torch.float32)
        frac_macro = one_hot_macro.mean(dim=0)
        mean_macro_prob = macro_probs.mean(dim=0)
        aux_loss = self.num_macro_domains * torch.sum(frac_macro * mean_macro_prob)
        return routing_weights, global_selected_experts, aux_loss

    def expand_for_new_expert(self) -> None:
        """Grow the last macro cluster's micro-gate by one row for a newly hot-plugged expert."""
        self.micro_gates[-1] = _grow_gate(self.micro_gates[-1])
        self.cluster_sizes[-1] += 1


class ThreeTierHierarchicalMoERouter(nn.Module):
    """Season 6 (`HM-S6-02`): 3-Tier `G_super -> G_macro -> G_micro` Router for 128-Expert Sparse MoE.

    Experts are grouped as `S` super-clusters x `M` macro-domains x micro-experts (default
    `4 x 4 x 8 = 128`). Per token:
      1. `super_gate(x)` picks the super-cluster `s*` (`S` inner products).
      2. `macro_gates[s*](x)` picks the macro-domain `m*` inside it (`M` inner products).
      3. `micro_gates[s*, m*](x)` picks the Top-`k` micro-experts (`max micro size` inner products).
    That is `S + M + max_micro` length-`hidden_size` inner products per token (`4 + 4 + 8 = 16`)
    against `E` for a flat softmax router (`128`): an exact 8x reduction in gating inner
    products. It is a count of operations, not a measured latency.

    Micro clusters may be ragged (hot-plugging appends to the last one), so global indices
    come from per-cluster offsets. `predict_candidate_experts` exposes path probabilities for
    speculative expert prefetching.
    """

    def __init__(
        self,
        hidden_size: int,
        num_super_clusters: int = 4,
        macro_per_super: int = 4,
        micro_per_macro: int = 8,
        top_k: int = 2,
        micro_cluster_sizes: List[int] | None = None,
    ) -> None:
        super().__init__()
        if num_super_clusters < 1 or macro_per_super < 1 or micro_per_macro < 1:
            raise ValueError("num_super_clusters, macro_per_super and micro_per_macro must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_super_clusters = int(num_super_clusters)
        self.macro_per_super = int(macro_per_super)
        num_micro_clusters = self.num_super_clusters * self.macro_per_super
        sizes = (
            [int(s) for s in micro_cluster_sizes]
            if micro_cluster_sizes is not None
            else [int(micro_per_macro)] * num_micro_clusters
        )
        if len(sizes) != num_micro_clusters or any(s < 1 for s in sizes):
            raise ValueError(
                f"micro_cluster_sizes must have {num_micro_clusters} entries, each >= 1"
            )
        self.micro_cluster_sizes = sizes
        self.top_k = max(1, min(int(top_k), min(sizes)))

        self.super_gate = nn.Linear(
            self.hidden_size, self.num_super_clusters, bias=False, dtype=torch.float32
        )
        self.macro_gates = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.macro_per_super, bias=False, dtype=torch.float32)
                for _ in range(self.num_super_clusters)
            ]
        )
        self.micro_gates = nn.ModuleList(
            [nn.Linear(self.hidden_size, s, bias=False, dtype=torch.float32) for s in sizes]
        )
        for gate in [self.super_gate, *self.macro_gates, *self.micro_gates]:
            nn.init.normal_(gate.weight, mean=0.0, std=0.02)

    @property
    def num_experts(self) -> int:
        return sum(self.micro_cluster_sizes)

    @property
    def micro_cluster_offsets(self) -> List[int]:
        offsets, acc = [], 0
        for s in self.micro_cluster_sizes:
            offsets.append(acc)
            acc += s
        return offsets

    @property
    def gating_inner_products_per_token(self) -> int:
        """Exact count of length-`hidden_size` inner products per token (not a timing)."""
        return self.num_super_clusters + self.macro_per_super + max(self.micro_cluster_sizes)

    @property
    def flat_gating_inner_products_per_token(self) -> int:
        return self.num_experts

    @property
    def gating_reduction_factor(self) -> float:
        return round(self.flat_gating_inner_products_per_token / self.gating_inner_products_per_token, 4)

    def _route_path(
        self, hs_fp32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return super probs/idx, macro probs/idx and the masked micro logits of the winning path."""
        super_probs = F.softmax(self.super_gate(hs_fp32), dim=-1, dtype=torch.float32)
        super_prob, super_idx = torch.max(super_probs, dim=-1)

        macro_weight_stack = torch.stack([g.weight for g in self.macro_gates], dim=0)  # [S, M, d]
        macro_logits = torch.bmm(macro_weight_stack[super_idx], hs_fp32.unsqueeze(-1)).squeeze(-1)
        macro_probs = F.softmax(macro_logits, dim=-1, dtype=torch.float32)
        macro_prob, macro_idx = torch.max(macro_probs, dim=-1)

        cluster_idx = super_idx * self.macro_per_super + macro_idx
        micro_weight_stack, valid = _stack_ragged_gates(self.micro_gates, self.hidden_size)
        micro_logits = torch.bmm(micro_weight_stack[cluster_idx], hs_fp32.unsqueeze(-1)).squeeze(-1)
        micro_logits = micro_logits.masked_fill(~valid[cluster_idx], float("-inf"))
        return super_probs, super_prob, macro_probs, macro_prob, cluster_idx, micro_logits

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 3-tier Top-k routing weights, global expert indices `[0..E-1]`, and aux loss."""
        hs_fp32 = hidden_states.to(torch.float32)
        super_probs, super_prob, macro_probs, macro_prob, cluster_idx, micro_logits = self._route_path(hs_fp32)

        top_micro_logits, top_micro_indices = torch.topk(micro_logits, k=self.top_k, dim=-1)
        micro_probs = F.softmax(top_micro_logits, dim=-1, dtype=torch.float32)
        routing_weights = micro_probs * (super_prob * macro_prob).unsqueeze(-1)
        routing_weights = (
            routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ).to(hidden_states.dtype)

        offsets = torch.tensor(self.micro_cluster_offsets, dtype=torch.long, device=hs_fp32.device)
        global_selected_experts = offsets[cluster_idx].unsqueeze(-1) + top_micro_indices

        # Switch-style balance terms on the super and macro decisions
        super_idx = super_probs.argmax(dim=-1)
        frac_super = F.one_hot(super_idx, num_classes=self.num_super_clusters).float().mean(dim=0)
        macro_idx = macro_probs.argmax(dim=-1)
        frac_macro = F.one_hot(macro_idx, num_classes=self.macro_per_super).float().mean(dim=0)
        aux_loss = self.num_super_clusters * torch.sum(frac_super * super_probs.mean(dim=0)) + (
            self.macro_per_super * torch.sum(frac_macro * macro_probs.mean(dim=0))
        )
        return routing_weights, global_selected_experts, aux_loss

    @torch.no_grad()
    def predict_candidate_experts(
        self, hidden_states: torch.Tensor, top_n: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token candidate global expert indices and their path probabilities.

        The probability of expert `e` on the winning path is
        `P(super) * P(macro | super) * softmax(micro logits)[e]`, so the values are a proper
        sub-distribution (they sum to at most 1). Used to decide which experts to prefetch.
        """
        hs_fp32 = hidden_states.to(torch.float32).reshape(-1, self.hidden_size)
        _, super_prob, _, macro_prob, cluster_idx, micro_logits = self._route_path(hs_fp32)
        micro_probs = F.softmax(micro_logits, dim=-1, dtype=torch.float32)
        n = min(int(top_n or self.top_k), min(self.micro_cluster_sizes))
        top_probs, top_idx = torch.topk(micro_probs, k=n, dim=-1)
        offsets = torch.tensor(self.micro_cluster_offsets, dtype=torch.long, device=hs_fp32.device)
        global_idx = offsets[cluster_idx].unsqueeze(-1) + top_idx
        return global_idx, top_probs * (super_prob * macro_prob).unsqueeze(-1)

    def expand_for_new_expert(self) -> None:
        """Grow the last micro cluster's gate by one row for a newly hot-plugged expert."""
        self.micro_gates[-1] = _grow_gate(self.micro_gates[-1])
        self.micro_cluster_sizes[-1] += 1


class FP8PagedKVCache:
    """Season 5 (`HM-S5-02`) & Season 10 (`HM-S10-05`): Pluggable Paged KV-Cache Manager.

    Stores attention Key/Value states in fixed-size pages (`page_size_tokens=16`).
    Supports five pluggable page codecs (`HM-S10-05` / Experiment E11):
      - `"fp8_e4m3"`: default block-scaled FP8 (`float8_e4m3fn`) with per-page FP16 absmax scales.
      - `"a2_rope_plane"`: 2D hexagonal lattice `A2` quantizer across RoPE coordinate planes `(x_{2i}, x_{2i+1})`.
      - `"e8_lattice"`: 8D Gosset `E8` lattice quantizer with factorised rate accounting.
      - `"scalar_2bit"`: 2-bit uniform scalar quantizer (E11 positive control: high distortion).
      - `"a2_8bit"`: high-rate 8-bit `A2` lattice quantizer (E11 negative control: near-lossless).
    """

    VALID_CODECS = ("fp8_e4m3", "a2_rope_plane", "e8_lattice", "scalar_2bit", "a2_8bit")

    def __init__(
        self,
        num_layers: int = 12,
        num_kv_heads: int = 8,
        head_dim: int = 64,
        page_size_tokens: int = 16,
        max_pages: int = 128,
        device: torch.device | str = "cpu",
        codec: str = "fp8_e4m3",
        target_bits_per_dim: float = 4.0,
    ) -> None:
        if codec not in self.VALID_CODECS:
            raise ValueError(f"codec must be one of {self.VALID_CODECS}, got {codec!r}")
        if target_bits_per_dim <= 0.0:
            raise ValueError("target_bits_per_dim must be > 0")
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size_tokens = int(page_size_tokens)
        self.max_pages = int(max_pages)
        self.device = torch.device(device)
        self.codec = codec
        self.target_bits_per_dim = float(target_bits_per_dim)
        self._pages_k: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pages_v: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._page_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}

    @staticmethod
    def _quantize_page_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_fp32 = tensor.detach().to(torch.float32)
        absmax = t_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (absmax / 448.0).to(torch.float16)
        scaled = (t_fp32 / scale.to(torch.float32)).clamp(-448.0, 448.0)
        if hasattr(torch, "float8_e4m3fn"):
            q = scaled.to(torch.float8_e4m3fn)
        else:
            q = torch.round(scaled / 448.0 * 127.0).to(torch.int8)
        return q, scale

    @staticmethod
    def _dequantize_page_fp8(
        q: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        if hasattr(torch, "float8_e4m3fn") and q.dtype == torch.float8_e4m3fn:
            return (q.to(torch.float32) * scale.to(torch.float32)).to(target_dtype)
        return ((q.to(torch.float32) / 127.0) * 448.0 * scale.to(torch.float32)).to(target_dtype)

    @classmethod
    def _encode_decode_with_codec(
        cls,
        tensor: torch.Tensor,
        codec: str,
        target_bits_per_dim: float = 4.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """Quantize `tensor` under `codec`, returning `(stored_q, scale_t, rate_bits_per_dim, relative_mse)`."""
        t_fp32 = tensor.detach().to(torch.float32)
        orig_shape = t_fp32.shape
        flat = t_fp32.reshape(-1).cpu()
        vn = float(torch.dot(flat, flat).item())

        if codec == "fp8_e4m3":
            q, s = cls._quantize_page_fp8(tensor)
            deq = cls._dequantize_page_fp8(q, s, target_dtype=torch.float32).reshape(-1).cpu()
            err = float(torch.dot(flat - deq, flat - deq).item()) / vn if vn > 0.0 else 0.0
            return q, s, 8.0, err

        if codec == "scalar_2bit":
            # 2-bit uniform scalar quantizer over [-absmax, +absmax] with 4 levels {-1.5, -0.5, +0.5, +1.5} * step
            absmax = t_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            step = (absmax / 1.5).to(torch.float16)
            idx = torch.round((t_fp32 / step.to(torch.float32)).clamp(-1.5, 1.5) + 1.5).clamp(0.0, 3.0).to(torch.int8)
            deq_full = ((idx.to(torch.float32) - 1.5) * step.to(torch.float32)).reshape(-1).cpu()
            err = float(torch.dot(flat - deq_full, flat - deq_full).item()) / vn if vn > 0.0 else 0.0
            return idx, step, 2.0, err

        from hivemind.compress.lattice_quantizer import (
            LatticeDeltaQuantizer,
            calibrate_scale_for_rate,
        )

        kind = "A2" if codec in ("a2_rope_plane", "a2_8bit") else "E8"
        block_dim = 2 if kind == "A2" else 8
        desired_bits = 8.0 if codec == "a2_8bit" else float(target_bits_per_dim)
        std_val = float(flat.std().item()) if flat.numel() > 1 else 1.0
        fallback_scale = max(1e-6, std_val / float(2.0 ** max(0.5, desired_bits - 0.5)))
        try:
            scale_val, _ = calibrate_scale_for_rate(
                kind, flat, target_bits_per_dim=desired_bits, block=block_dim
            )
        except ValueError:
            scale_val = fallback_scale

        q_res = LatticeDeltaQuantizer(
            kind=kind, scale=scale_val, block=block_dim, error_feedback=False
        ).quantize(flat)
        deq_tensor = q_res.dequantized.to(device=tensor.device, dtype=torch.float16).reshape(orig_shape)
        scale_t = torch.tensor([scale_val], dtype=torch.float16, device=tensor.device)
        return deq_tensor, scale_t, float(q_res.rate_bits_per_dim), float(q_res.relative_mse)

    def write_kv_page(
        self, layer_idx: int, page_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """Quantize and store a `(K, V)` page using the configured `self.codec`."""
        if len(self._pages_k) >= self.max_pages and (layer_idx, page_idx) not in self._pages_k:
            oldest_key = next(iter(self._pages_k))
            self._pages_k.pop(oldest_key, None)
            self._pages_v.pop(oldest_key, None)
            self._page_metrics.pop(oldest_key, None)

        k_dev = k_tensor.to(self.device)
        v_dev = v_tensor.to(self.device)
        qk, sk, rate_k, mse_k = self._encode_decode_with_codec(
            k_dev, self.codec, self.target_bits_per_dim
        )
        qv, sv, rate_v, mse_v = self._encode_decode_with_codec(
            v_dev, self.codec, self.target_bits_per_dim
        )
        self._pages_k[(layer_idx, page_idx)] = (qk, sk)
        self._pages_v[(layer_idx, page_idx)] = (qv, sv)
        mean_rate = 0.5 * (rate_k + rate_v)
        mean_mse = 0.5 * (mse_k + mse_v)
        self._page_metrics[(layer_idx, page_idx)] = {
            "rate_bits_per_dim": round(mean_rate, 4),
            "relative_mse": round(mean_mse, 8),
        }

        fp16_bytes = (k_tensor.numel() + v_tensor.numel()) * 2
        if self.codec == "fp8_e4m3":
            paged_bytes = (
                qk.numel() * qk.element_size()
                + qv.numel() * qv.element_size()
                + (sk.numel() + sv.numel()) * 2
            )
        else:
            total_dims = k_tensor.numel() + v_tensor.numel()
            paged_bytes = max(1, int(round(total_dims * mean_rate / 8.0)) + (sk.numel() + sv.numel()) * 2)
        return {
            "layer_idx": layer_idx,
            "page_idx": page_idx,
            "codec": self.codec,
            "fp16_equivalent_bytes": fp16_bytes,
            "fp8_paged_bytes": paged_bytes,
            "measured_rate_bits_per_dim": round(mean_rate, 4),
            "relative_mse": round(mean_mse, 8),
            "compression_ratio": round(fp16_bytes / max(1, paged_bytes), 4),
        }

    def read_kv_page(
        self, layer_idx: int, page_idx: int, target_dtype: torch.dtype = torch.float16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return a stored `(K, V)` page."""
        qk, sk = self._pages_k[(layer_idx, page_idx)]
        qv, sv = self._pages_v[(layer_idx, page_idx)]
        if self.codec == "fp8_e4m3":
            return (
                self._dequantize_page_fp8(qk, sk, target_dtype=target_dtype),
                self._dequantize_page_fp8(qv, sv, target_dtype=target_dtype),
            )
        if self.codec == "scalar_2bit":
            dk = ((qk.to(torch.float32) - 1.5) * sk.to(torch.float32)).to(target_dtype)
            dv = ((qv.to(torch.float32) - 1.5) * sv.to(torch.float32)).to(target_dtype)
            return dk, dv
        return qk.to(target_dtype), qv.to(target_dtype)

    @classmethod
    def benchmark_kv_page_codecs(
        cls,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        target_bits_per_dim: float = 4.0,
    ) -> Dict[str, Any]:
        """Compare all 5 KV-cache page codecs (`HM-S10-05` / E11) with positive/negative controls."""
        results: Dict[str, Dict[str, float]] = {}
        for codec in cls.VALID_CODECS:
            _, _, rk, mk = cls._encode_decode_with_codec(k_tensor, codec, target_bits_per_dim)
            _, _, rv, mv = cls._encode_decode_with_codec(v_tensor, codec, target_bits_per_dim)
            results[codec] = {
                "rate_bits_per_dim": round(0.5 * (rk + rv), 4),
                "relative_mse": round(0.5 * (mk + mv), 8),
            }
        pos_ctrl_passed = (
            results["scalar_2bit"]["relative_mse"] > results["a2_rope_plane"]["relative_mse"]
            and results["scalar_2bit"]["relative_mse"] > results["e8_lattice"]["relative_mse"]
        )
        neg_ctrl_passed = (
            results["a2_8bit"]["relative_mse"] < results["a2_rope_plane"]["relative_mse"]
        )
        return {
            "strategy_id": "HM-S10-05-LATTICE-KV-CACHE-CODEC-BENCHMARK",
            "target_bits_per_dim": float(target_bits_per_dim),
            "codecs": results,
            "positive_control_scalar_2bit_worse": bool(pos_ctrl_passed),
            "negative_control_a2_8bit_better": bool(neg_ctrl_passed),
            "controls_all_passed": bool(pos_ctrl_passed and neg_ctrl_passed),
        }

    def total_kv_cache_bytes(self) -> int:
        total = 0
        for q, s in list(self._pages_k.values()) + list(self._pages_v.values()):
            total += q.numel() * q.element_size() + s.numel() * s.element_size()
        return total


def _install_router_on_blocks(moe_model: UpcycledSparseMoEModel, make_router) -> None:
    for block in moe_model.moe_blocks:
        dev = next(block.router.parameters()).device
        block.router = make_router().to(dev)


def _footprint_report(moe_model: UpcycledSparseMoEModel, vram_ceiling_gb: float) -> Dict[str, Any]:
    """Parameter-byte footprint, plus live CUDA allocation only when CUDA is actually in use."""
    stats = measure_moe_vram_footprint_gb(moe_model)
    on_cuda = next(moe_model.parameters()).device.type == "cuda" and torch.cuda.is_available()
    report: Dict[str, Any] = {
        "parameter_footprint_gb": stats["total_gb"],
        "vram_ceiling_gb": float(vram_ceiling_gb),
    }
    if on_cuda:
        report["vram_measured_gb"] = stats["cuda_allocated_gb"]
        report["fits_under_vram_ceiling"] = stats["cuda_allocated_gb"] < vram_ceiling_gb
        report["vram_measurement"] = "measured: torch.cuda.memory_allocated on the model device"
    else:
        report["vram_measured_gb"] = None
        report["fits_under_vram_ceiling"] = None
        report["vram_measurement"] = "not_measured: model is not on CUDA; parameter bytes only"
    return report


def hotplug_hierarchical_experts(
    moe_model: UpcycledSparseMoEModel,
    community_adapters: Dict[str, Dict[str, torch.Tensor]],
    canonical_rank_r: int = 16,
    num_macro_domains: int = 8,
    experts_per_domain: int = 8,
    top_k: int = 2,
    vram_ceiling_gb: float = 2.2,
) -> Dict[str, Any]:
    """Season 5 (`HM-S5-02`): Hot-plug up to 64 heterogeneous community adapters into `moe_model`
    using SVD rank normalization (`r -> canonical_rank_r`) and install `HierarchicalMoERouter`.

    The router is sized to the model's actual expert count (ragged clusters), so every global
    index it emits names a real expert.
    """
    hotplug_report = hotplug_rank_adaptive_experts(
        moe_model=moe_model,
        expert_lora_states=community_adapters,
        target_rank_r=canonical_rank_r,
        vram_budget_gb=vram_ceiling_gb,
    )
    total_experts = moe_model.num_experts
    top_k = moe_model.moe_blocks[0].top_k if moe_model.moe_blocks else int(top_k)
    # every cluster must hold at least top_k experts or the block gets fewer columns than it expects
    eff_macro = max(1, min(int(num_macro_domains), total_experts // max(1, top_k)))
    sizes = _even_cluster_sizes(total_experts, eff_macro)
    _ = experts_per_domain
    _install_router_on_blocks(
        moe_model,
        lambda: HierarchicalMoERouter(
            hidden_size=moe_model.hidden_size, top_k=top_k, cluster_sizes=list(sizes)
        ),
    )
    router = moe_model.moe_blocks[0].router if moe_model.moe_blocks else None
    gating = router.gating_inner_products_per_token if router is not None else total_experts
    return {
        **hotplug_report,
        "router_type": "HierarchicalMoERouter",
        "num_macro_domains": eff_macro,
        "cluster_sizes": sizes,
        "experts_per_domain": max(sizes),
        "total_experts": total_experts,
        "gating_inner_products_per_token": gating,
        "flat_gating_inner_products_per_token": total_experts,
        "gating_flops_reduction_factor": round(float(total_experts) / float(gating), 2),
        "gating_reduction_basis": "exact count of length-hidden_size inner products per token; not a measured latency",
        **_footprint_report(moe_model, vram_ceiling_gb),
    }


def hotplug_three_tier_experts(
    moe_model: UpcycledSparseMoEModel,
    community_adapters: Dict[str, Dict[str, torch.Tensor]],
    canonical_rank_r: int = 16,
    num_super_clusters: int = 4,
    macro_per_super: int = 4,
    micro_per_macro: int = 8,
    top_k: int = 2,
    vram_ceiling_gb: float = 2.4,
) -> Dict[str, Any]:
    """Season 6 (`HM-S6-02`): Hot-plug community adapters and install the 3-tier
    `G_super -> G_macro -> G_micro` router sized exactly to the model's expert count.

    With the default `4 x 4 x 8` layout and 128 experts the router performs 16 gating inner
    products per token instead of 128 (`8.0x`). The report states what was counted and what
    was measured; VRAM is only reported as measured when the model lives on CUDA.
    """
    hotplug_report = hotplug_rank_adaptive_experts(
        moe_model=moe_model,
        expert_lora_states=community_adapters,
        target_rank_r=canonical_rank_r,
        vram_budget_gb=vram_ceiling_gb,
    )
    total_experts = moe_model.num_experts
    top_k = moe_model.moe_blocks[0].top_k if moe_model.moe_blocks else int(top_k)
    max_clusters = max(1, total_experts // max(1, top_k))
    eff_super = max(1, min(int(num_super_clusters), max_clusters))
    eff_macro = max(1, min(int(macro_per_super), max_clusters // eff_super))
    sizes = _even_cluster_sizes(total_experts, eff_super * eff_macro)
    _ = micro_per_macro
    _install_router_on_blocks(
        moe_model,
        lambda: ThreeTierHierarchicalMoERouter(
            hidden_size=moe_model.hidden_size,
            num_super_clusters=eff_super,
            macro_per_super=eff_macro,
            top_k=top_k,
            micro_cluster_sizes=list(sizes),
        ),
    )
    router = moe_model.moe_blocks[0].router if moe_model.moe_blocks else None
    gating = router.gating_inner_products_per_token if router is not None else total_experts
    return {
        **hotplug_report,
        "router_type": "ThreeTierHierarchicalMoERouter",
        "num_super_clusters": eff_super,
        "macro_per_super": eff_macro,
        "micro_cluster_sizes": sizes,
        "total_experts": total_experts,
        "gating_inner_products_per_token": gating,
        "flat_gating_inner_products_per_token": total_experts,
        "gating_flops_reduction_factor": round(float(total_experts) / float(gating), 2),
        "gating_reduction_basis": "exact count of length-hidden_size inner products per token; not a measured latency",
        **_footprint_report(moe_model, vram_ceiling_gb),
    }


__all__ = [
    "ExpertBranchState",
    "FP8PagedKVCache",
    "HierarchicalMoERouter",
    "MoEUpcycleReport",
    "ProjectedLoRAStateDict",
    "RegisteredDomainExpert",
    "SEASON_3_DOMAIN_EXPERT_SPECS",
    "SVDLoARankProjectionReport",
    "Season3DomainExpertRegistry",
    "SparseMoEMLPBlock",
    "ThreeTierHierarchicalMoERouter",
    "TopKMoERouter",
    "UpcycledFactoredSwiGLUExpert",
    "UpcycledSparseMoEModel",
    "UpcycledSwiGLUExpert",
    "detect_lora_rank",
    "hotplug_hierarchical_experts",
    "hotplug_rank_adaptive_experts",
    "hotplug_three_tier_experts",
    "initialize_domain_branches",
    "measure_moe_vram_footprint_gb",
    "pair_lora_ab_keys",
    "project_lora_rank_svd",
    "project_lora_rank_svd_with_report",
    "project_lora_state_dict_to_rank",
    "svd_project_lora_pair",
    "train_moe_router_gating",
    "upcycle_branches_to_sparse_moe",
]
