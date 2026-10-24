"""Phase A & Phase C: Branch-Train-Mix (BTM) Domain Partitioning & Sparse MoE Upcycling.

Implements:
- Phase A: Initializing E=8 domain-specific expert branches from a shared base checkpoint W_0.
- Phase C: Upcycling E converged QLoRA adapters into a unified Sparse Mixture-of-Experts (MoE):
    1. Self-Attention projections (q_proj, k_proj, v_proj, o_proj) are uniformly averaged
       across all E adapters and merged into the shared attention backbone:
         W_attn = W_0_attn + (1/E) * sum_{e=1}^E (alpha/r) * B_{e,attn} @ A_{e,attn}
    2. FFN/MLP projections (gate_proj, up_proj, down_proj) are merged per expert e in {1..E}
       to create E distinct SwiGLU FFN experts:
         W_{ffn, e} = W_0_ffn + (alpha/r) * B_{e,ffn} @ A_{e,ffn}
    3. A Top-2 Softmax Router G(x) = Softmax(TopK(x @ W_g, k=2)) routes each token to its
       top-2 domain experts with Switch-style load-balancing auxiliary loss.
- Season 7 (`HM-S7-01`): a looped `Prelude -> RecurrentCore x T -> Coda` wrapper that reuses
  the core decoder layers T times instead of copying them, with optional per-step LoRA adapters.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EXPERT_DOMAINS = [
    "Code",
    "Mathematics",
    "Creative_Writing",
    "STEM",
    "Legal_Medical",
    "Multilingual_FR",
    "Reasoning",
    "General",
]

# Season 8 (`HM-S8-03`): attention merge strategies for the shared backbone.
ATTENTION_MERGE_MODES = ("uniform", "iso_cts", "tsv")


@dataclass
class ExpertBranchState:
    """Tracks the outer-loop LoRA state for one BTM domain branch."""

    branch_id: int
    domain_name: str
    outer_step: int
    lora_state: dict[str, torch.Tensor] = field(repr=False)


@dataclass
class MoEUpcycleReport:
    """Telemetry report produced when upcycling E QLoRA branches into a Sparse MoE."""

    num_experts: int
    top_k: int
    expert_domains: list[str]
    attention_layers_merged: int
    ffn_expert_modules_created: int
    total_moe_parameters: int
    active_parameters_per_token: int
    router_initial_loss: float = 0.0
    router_final_loss: float = 0.0
    expert_utilization_distribution: dict[str, float] = field(default_factory=dict)
    attention_merge_mode: str = "uniform"


def initialize_domain_branches(
    initial_lora_state: dict[str, torch.Tensor],
    expert_domains: list[str] | None = None,
) -> dict[str, ExpertBranchState]:
    """Phase A: Initialize E=8 independent expert branches from identical base checkpoint W_0."""
    domains = expert_domains or DEFAULT_EXPERT_DOMAINS
    branches: dict[str, ExpertBranchState] = {}
    for idx, domain in enumerate(domains):
        branches[domain] = ExpertBranchState(
            branch_id=idx,
            domain_name=domain,
            outer_step=0,
            lora_state={k: v.detach().cpu().clone() for k, v in initial_lora_state.items()},
        )
    return branches


def _dequantize_linear_weight(module: nn.Module) -> torch.Tensor:
    """Extract dense FP32 weight matrix from either nn.Linear or bitsandbytes Linear4bit."""
    base_layer = getattr(module, "base_layer", module)
    weight = base_layer.weight
    if hasattr(weight, "quant_state") and weight.quant_state is not None:
        import bitsandbytes.functional as bnb_F

        return bnb_F.dequantize_4bit(weight.data, weight.quant_state).to(torch.float32).cpu()
    return weight.detach().to(torch.float32).cpu()


def _extract_ab_pair(
    lora_state: dict[str, torch.Tensor],
    layer_idx: int,
    subpath: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Locate (lora_A, lora_B) tensors for a specific decoder layer and projection."""
    prefix = f"layers.{layer_idx}.{subpath}."
    a_tensor: torch.Tensor | None = None
    b_tensor: torch.Tensor | None = None
    for k, v in lora_state.items():
        if prefix in k and "lora_A" in k:
            a_tensor = v.detach().to(torch.float32).cpu()
        elif prefix in k and "lora_B" in k:
            b_tensor = v.detach().to(torch.float32).cpu()
    if a_tensor is not None and b_tensor is not None:
        return a_tensor, b_tensor
    return None


@dataclass
class SVDLoARankProjectionReport:
    """Telemetry describing a truncated-SVD LoRA rank projection (`HM-S4-05`).

    Attributes:
        source_rank_r: Rank `r_src` detected on the incoming adapter.
        target_rank_r: Canonical rank `r_dst` the adapter was projected onto.
        num_projected_pairs: Number of (A, B) LoRA factor pairs reprojected.
        num_padded_pairs: Number of pairs whose source rank was below `target_rank_r`.
        num_identity_pairs: Number of pairs already at `target_rank_r`.
        mean_spectral_energy_retained: Mean fraction of the Frobenius spectral
            energy of `D = B @ A` preserved by the truncated SVD.
        min_spectral_energy_retained: Minimum per-pair spectral energy retention.
        max_relative_delta_error: Maximum relative Frobenius error between the
            source effective delta and the projected effective delta.
        source_params: Total number of LoRA parameters before projection.
        target_params: Total number of LoRA parameters after projection.
        source_lora_alpha: `alpha` assumed for the source adapter.
        target_lora_alpha: `alpha` the projected adapter is meant to be used with.
    """

    source_rank_r: int
    target_rank_r: int
    num_projected_pairs: int
    num_padded_pairs: int
    num_identity_pairs: int
    mean_spectral_energy_retained: float
    min_spectral_energy_retained: float
    max_relative_delta_error: float
    source_params: int
    target_params: int
    source_lora_alpha: float
    target_lora_alpha: float

    @property
    def compression_ratio(self) -> float:
        """Parameter compression factor achieved by the rank projection."""
        if self.target_params <= 0:
            return 1.0
        return round(self.source_params / self.target_params, 4)


def _orient_lora_pair(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return a LoRA (A, B) pair in the canonical PEFT layout `A=[r, in]`, `B=[out, r]`."""
    if lora_a.dim() != 2 or lora_b.dim() != 2:
        raise ValueError(
            "LoRA factors must be 2-D tensors, got "
            f"{tuple(lora_a.shape)} and {tuple(lora_b.shape)}."
        )
    if lora_a.shape[0] == lora_b.shape[1]:
        return lora_a, lora_b, False
    if lora_a.shape[1] == lora_b.shape[0]:
        return lora_a.t().contiguous(), lora_b.t().contiguous(), True
    raise ValueError(
        f"Incompatible LoRA factor shapes {tuple(lora_a.shape)} / {tuple(lora_b.shape)}: "
        "expected A=[r, in_features] and B=[out_features, r] (or the transposed layout)."
    )


def pair_lora_ab_keys(lora_state_dict: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """Pair every `lora_A` key of a LoRA state dict with its matching `lora_B` key."""
    pairs: list[tuple[str, str]] = []
    for key in lora_state_dict:
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key in lora_state_dict:
            pairs.append((key, b_key))
    return pairs


def detect_lora_rank(lora_state_dict: dict[str, torch.Tensor]) -> int | None:
    """Detect the rank `r` of a LoRA state dict (`None` when it holds no LoRA pair).

    Adapters mixing several ranks across layers report the **maximum** rank; every
    factor pair is nonetheless projected independently by
    `project_lora_state_dict_to_rank`.
    """
    ranks: set[int] = set()
    for a_key, b_key in pair_lora_ab_keys(lora_state_dict):
        a_t = lora_state_dict[a_key]
        b_t = lora_state_dict[b_key]
        if a_t.shape[0] == b_t.shape[1]:
            ranks.add(int(a_t.shape[0]))
        elif a_t.shape[1] == b_t.shape[0]:
            ranks.add(int(a_t.shape[1]))
    if not ranks:
        return None
    return max(ranks)


def svd_project_lora_pair(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    target_rank_r: int,
    scale_b: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Truncated-SVD projection of a LoRA (A, B) factor pair onto `target_rank_r`.

    Computes the Eckart-Young optimal rank-`target_rank_r` approximation of the
    effective weight delta `D = B @ A` and refactors it back into a LoRA pair
    `(A', B')` such that `B' @ A' ~= D_k` (the truncated spectral expansion of `D`).

    The computation never materializes the dense `D` matrix: a QR decomposition of
    `B` is combined with the SVD of the (small) `R @ A` core, which is
    mathematically equivalent to the SVD of `D` but costs `O(r * n)` instead of
    `O(min(m, n)^3)` — critical to stay inside the `< 2.0 GB` VRAM budget while
    hot-plugging 32 experts.

    Args:
        lora_a: LoRA down-projection factor `A` with shape `[r, in_features]`.
        lora_b: LoRA up-projection factor `B` with shape `[out_features, r]`.
        target_rank_r: Canonical rank the pair is projected onto.
        scale_b: Multiplicative factor applied to `B'` (used to keep the
            `alpha / r` scaling convention consistent across ranks).

    Returns:
        `(lora_a_new, lora_b_new, spectral_energy_retained)` where
        `spectral_energy_retained = sum(s[:k]^2) / sum(s^2)` is the fraction of the
        Frobenius spectral energy of `D` captured by the truncated expansion.
    """
    if target_rank_r < 1:
        raise ValueError(f"target_rank_r must be >= 1, got {target_rank_r}.")

    a_t = lora_a.detach().to(torch.float32)
    b_t = lora_b.detach().to(torch.float32)

    q_b, r_core = torch.linalg.qr(b_t)  # q_b: [out, k], r_core: [k, r]
    core = r_core @ a_t  # [k, in] - the only matrix we ever factorize
    u, s, vh = torch.linalg.svd(core, full_matrices=False)

    total_energy = float(torch.sum(s * s).item())
    k_eff = int(min(target_rank_r, s.numel()))
    u_k = u[:, :k_eff]
    s_k = s[:k_eff]
    vh_k = vh[:k_eff, :]

    sqrt_s = torch.sqrt(s_k.clamp_min(0.0))
    left = q_b @ u_k  # [out, k_eff] left singular vectors of D
    b_new = (left * sqrt_s.unsqueeze(0)) * float(scale_b)
    a_new = sqrt_s.unsqueeze(1) * vh_k

    if k_eff < target_rank_r:
        # Pad lower-rank adapters up to the canonical rank with zero rows/columns:
        # the effective delta is preserved exactly while the nominal rank matches.
        pad = target_rank_r - k_eff
        a_new = torch.cat([a_new, a_new.new_zeros((pad, a_new.shape[1]))], dim=0)
        b_new = torch.cat([b_new, b_new.new_zeros((b_new.shape[0], pad))], dim=1)

    retained = 1.0 if total_energy <= 0.0 else float(torch.sum(s_k * s_k).item()) / total_energy
    return a_new, b_new, retained


def project_lora_state_dict_to_rank(
    lora_state_dict: dict[str, torch.Tensor],
    target_rank_r: int = 16,
    source_lora_alpha: float | None = None,
    target_lora_alpha: float | None = None,
    source_rank_r: int | None = None,
) -> tuple[dict[str, torch.Tensor], SVDLoARankProjectionReport]:
    """Project a whole LoRA state dict onto a canonical rank via truncated SVD.

    Higher-rank adapters (`r=32`, `r=64`) are compressed and lower-rank adapters
    (`r=8`) are zero-padded, always landing on the same canonical rank so that
    heterogeneous community adapters can be hot-plugged into one unified
    `UpcycledSparseMoEModel` without VRAM bloat.

    The `alpha / r` scaling convention is preserved: the returned adapter obeys

        (target_lora_alpha / target_rank_r) * B' @ A'
            ~= (source_lora_alpha / source_rank_r) * B @ A

    so the effective weight delta of the caller is unchanged (up to the truncation
    error of the discarded spectral tail).

    Args:
        lora_state_dict: LoRA state dict with `lora_A` / `lora_B` tensor pairs.
        target_rank_r: Canonical rank to project onto.
        source_lora_alpha: `alpha` of the source adapter (defaults to the
            community-standard `2 * r` convention).
        target_lora_alpha: `alpha` the projected adapter will be used with
            (defaults to the community-standard `2 * target_rank_r` convention).
        source_rank_r: Explicit source rank override (auto-detected by default).

    Returns:
        `(projected_state_dict, report)`.
    """
    if target_rank_r < 1:
        raise ValueError(f"target_rank_r must be >= 1, got {target_rank_r}.")

    pairs = pair_lora_ab_keys(lora_state_dict)
    detected_rank = detect_lora_rank(lora_state_dict)
    src_r = int(source_rank_r) if source_rank_r is not None else int(detected_rank or 0)
    if src_r <= 0:
        empty = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in lora_state_dict.items()}
        report = SVDLoARankProjectionReport(
            source_rank_r=0,
            target_rank_r=int(target_rank_r),
            num_projected_pairs=0,
            num_padded_pairs=0,
            num_identity_pairs=0,
            mean_spectral_energy_retained=1.0,
            min_spectral_energy_retained=1.0,
            max_relative_delta_error=0.0,
            source_params=0,
            target_params=0,
            source_lora_alpha=0.0,
            target_lora_alpha=0.0,
        )
        return empty, report

    src_alpha = float(source_lora_alpha) if source_lora_alpha is not None else 2.0 * src_r
    dst_alpha = float(target_lora_alpha) if target_lora_alpha is not None else 2.0 * float(target_rank_r)
    target_scaling = dst_alpha / float(target_rank_r)

    projected: dict[str, torch.Tensor] = {}
    retentions: list[float] = []
    rel_errors: list[float] = []
    source_params = 0
    target_params = 0
    num_padded = 0
    num_identity = 0
    consumed: set[str] = set()

    for a_key, b_key in pairs:
        consumed.add(a_key)
        consumed.add(b_key)
        raw_a = lora_state_dict[a_key]
        raw_b = lora_state_dict[b_key]
        a_t, b_t, transposed = _orient_lora_pair(raw_a, raw_b)
        pair_rank = int(a_t.shape[0])

        # The `alpha / r` convention is applied per factor pair so that adapters
        # mixing several ranks across layers keep their exact effective delta.
        pair_src_alpha = float(source_lora_alpha) if source_lora_alpha is not None else 2.0 * pair_rank
        scale_b = (pair_src_alpha / float(pair_rank)) / target_scaling if target_scaling != 0.0 else 1.0

        a_new, b_new, retained = svd_project_lora_pair(
            a_t, b_t, target_rank_r=target_rank_r, scale_b=scale_b
        )
        retentions.append(retained)
        if pair_rank < target_rank_r:
            num_padded += 1
        elif pair_rank == target_rank_r:
            num_identity += 1

        # Relative Frobenius error between the source and the projected delta.
        delta_src = b_t @ a_t
        delta_dst = b_new @ a_new
        denom = float(torch.linalg.norm(delta_src).item())
        err = float(torch.linalg.norm(delta_dst - delta_src).item())
        rel_errors.append(err / denom if denom > 0.0 else 0.0)

        source_params += int(raw_a.numel()) + int(raw_b.numel())
        target_params += int(a_new.numel()) + int(b_new.numel())

        out_a = a_new.t().contiguous() if transposed else a_new.contiguous()
        out_b = b_new.t().contiguous() if transposed else b_new.contiguous()
        projected[a_key] = out_a.to(dtype=raw_a.dtype)
        projected[b_key] = out_b.to(dtype=raw_b.dtype)

    for key, value in lora_state_dict.items():
        if key in consumed:
            continue
        projected[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value

    report = SVDLoARankProjectionReport(
        source_rank_r=src_r,
        target_rank_r=int(target_rank_r),
        num_projected_pairs=len(pairs),
        num_padded_pairs=num_padded,
        num_identity_pairs=num_identity,
        mean_spectral_energy_retained=round(sum(retentions) / max(1, len(retentions)), 6),
        min_spectral_energy_retained=round(min(retentions) if retentions else 1.0, 6),
        max_relative_delta_error=round(max(rel_errors) if rel_errors else 0.0, 8),
        source_params=source_params,
        target_params=target_params,
        source_lora_alpha=round(src_alpha, 6),
        target_lora_alpha=round(dst_alpha, 6),
    )
    return projected, report


class UpcycledSwiGLUExpert(nn.Module):
    """Single domain-specialized SwiGLU FFN expert constructed via W_e = W_0 + (alpha/r) B_e A_e."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dev = gate_weight.device
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=dev)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=dev)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=dev)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate_weight.to(device=dev, dtype=dtype))
            self.up_proj.weight.copy_(up_weight.to(device=dev, dtype=dtype))
            self.down_proj.weight.copy_(down_weight.to(device=dev, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class UpcycledFactoredSwiGLUExpert(nn.Module):
    """VRAM-efficient SwiGLU FFN expert evaluating (W_0 + (alpha/r) B_e A_e) x on shared NF4/FP16 W_0."""

    def __init__(
        self,
        shared_base_ffn: tuple[nn.Linear, nn.Linear, nn.Linear],
        ab_gate: tuple[torch.Tensor, torch.Tensor] | None,
        ab_up: tuple[torch.Tensor, torch.Tensor] | None,
        ab_down: tuple[torch.Tensor, torch.Tensor] | None,
        scaling: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        # Keep reference in a plain Python tuple so PyTorch does not duplicate shared base parameters
        self._shared_base_ffn = shared_base_ffn
        for prefix, ab in (("gate", ab_gate), ("up", ab_up), ("down", ab_down)):
            if ab is not None:
                a_buf = ab[0].to(device=device, dtype=dtype).contiguous()
                b_buf = (ab[1].to(device=device, dtype=torch.float32) * scaling).to(dtype=dtype).contiguous()
                self.register_buffer(f"{prefix}_A", a_buf)
                self.register_buffer(f"{prefix}_B", b_buf)
            else:
                self.register_buffer(f"{prefix}_A", None)
                self.register_buffer(f"{prefix}_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_gate, base_up, base_down = self._shared_base_ffn
        g = base_gate(x)
        if self.gate_A is not None and self.gate_B is not None:
            g = g + (x @ self.gate_A.t()) @ self.gate_B.t()
        u = base_up(x)
        if self.up_A is not None and self.up_B is not None:
            u = u + (x @ self.up_A.t()) @ self.up_B.t()
        h = F.silu(g) * u
        d = base_down(h)
        if self.down_A is not None and self.down_B is not None:
            d = d + (h @ self.down_A.t()) @ self.down_B.t()
        return d


class TopKMoERouter(nn.Module):
    """Lightweight Top-k Softmax Gating Network: G(x) = Softmax(TopK(x @ W_g, k=2))."""

    def __init__(self, hidden_size: int, num_experts: int = 8, top_k: int = 2, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        _ = dtype
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        # Keep tiny router matrix W_g in FP32 so AdamW (eps=1e-8) never underflows in FP16
        self.wg = nn.Linear(hidden_size, num_experts, bias=False, dtype=torch.float32)
        nn.init.normal_(self.wg.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Top-k gating weights, selected expert indices, and Switch load-balancing loss.

        Args:
            hidden_states: [batch * seq_len, hidden_size]
        Returns:
            routing_weights: [N, top_k] normalized via Softmax over top-k logits
            selected_experts: [N, top_k] integer indices in [0, E-1]
            aux_loss: scalar load-balancing loss
        """
        logits = self.wg(hidden_states.to(torch.float32))  # [N, E]
        probs_all = F.softmax(logits, dim=-1, dtype=torch.float32)

        top_logits, selected_experts = torch.topk(logits, k=self.top_k, dim=-1)
        routing_weights = F.softmax(top_logits, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # Load-balancing auxiliary loss: E * sum_i (f_i * P_i)
        num_tokens = max(1, hidden_states.size(0))
        one_hot = F.one_hot(selected_experts[:, 0], num_classes=self.num_experts).to(torch.float32)
        fraction_routed = one_hot.mean(dim=0)
        mean_prob = probs_all.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(fraction_routed * mean_prob) * (num_tokens / num_tokens)

        return routing_weights, selected_experts, aux_loss

    def expand_for_new_expert(self) -> None:
        """Incrementally expand the Top-k gating matrix W_g from R^{d x E} to R^{d x (E+1)}."""
        old_wg = self.wg
        device = old_wg.weight.device
        hidden_size = old_wg.in_features
        new_num_experts = self.num_experts + 1

        new_wg = nn.Linear(hidden_size, new_num_experts, bias=False, dtype=torch.float32, device=device)
        with torch.no_grad():
            new_wg.weight[: self.num_experts].copy_(old_wg.weight)
            # Initialize the new expert's gating column near the mean of existing columns
            mean_col = old_wg.weight.mean(dim=0, keepdim=True)
            new_wg.weight[self.num_experts :].copy_(mean_col + torch.randn_like(mean_col) * 0.01)

        self.wg = new_wg
        self.num_experts = new_num_experts


def _normalize_local_expert_indices(
    local_expert_indices: Sequence[int] | None,
    num_experts: int,
) -> tuple[int, ...] | None:
    """Validate and canonicalize a FoMoE local-expert subset (`HM-S8-06`)."""
    if local_expert_indices is None:
        return None
    try:
        normalized = sorted({int(idx) for idx in local_expert_indices})
    except (TypeError, ValueError) as exc:
        raise ValueError("local_expert_indices must be a sequence of integers") from exc
    if not normalized:
        raise ValueError("local_expert_indices must contain at least one expert index")
    if normalized[0] < 0 or normalized[-1] >= num_experts:
        raise ValueError(
            f"local_expert_indices must be within [0, {num_experts - 1}], got {normalized}"
        )
    return tuple(normalized)


def _merge_attention_projection_deltas(
    deltas: list[torch.Tensor],
    weights: list[float],
    mode: str,
) -> torch.Tensor:
    """Merge per-branch attention deltas with the requested `HM-S8-03` strategy."""
    if not deltas:
        raise ValueError("at least one attention delta is required")
    if len(weights) != len(deltas):
        raise ValueError("weights and deltas must have the same length")
    if mode == "iso_cts":
        # Import lazily: `hivemind.swarm` imports the full swarm stack.
        from hivemind.swarm.diloco_nextgen import ReputationWeightedIsoCTSMerger

        key = "__btm_attention_delta__"
        base_state = {key: torch.zeros_like(deltas[0])}
        worker_states = [{key: delta} for delta in deltas]
        # Uniform scaling is applied by the caller; keep the merger's TIES/ISO-CTS
        # spectral isotropy without inflating the merged delta by `scaling_lambda`.
        merger = ReputationWeightedIsoCTSMerger(scaling_lambda=1.0)
        merged_state, _ = merger.merge_worker_deltas(
            base_lora_state=base_state,
            worker_lora_states=worker_states,
            worker_reputation_weights=weights,
        )
        return merged_state[key]
    if mode == "tsv":
        from hivemind.swarm.diloco_nextgen import TSVDecorrelatedMerger

        merged, _ = TSVDecorrelatedMerger.merge_2d_deltas(deltas, weights)
        return merged
    raise ValueError(f"Unsupported attention merge mode: {mode!r}")


class SparseMoEMLPBlock(nn.Module):
    """Phase C Upcycled Sparse MoE Layer replacing a dense Qwen/Gemma MLP block.

    Evaluates:
      y = sum_{i in Top-2(G(x))} G(x)_i * FFN_i(x)
    """

    def __init__(
        self,
        hidden_size: int,
        experts: list[nn.Module],
        expert_domains: list[str],
        top_k: int = 2,
        dtype: torch.dtype = torch.float32,
        shared_base_ffn: tuple[nn.Linear, nn.Linear, nn.Linear] | None = None,
        routing_mode: str = "softmax",
        peri_ln: bool = False,
    ) -> None:
        super().__init__()
        if routing_mode not in ("softmax", "sigma_moe"):
            raise ValueError(f"routing_mode must be 'softmax' or 'sigma_moe', got {routing_mode!r}")
        self.hidden_size = hidden_size
        self.num_experts = len(experts)
        self.top_k = min(top_k, self.num_experts)
        self.expert_domains = list(expert_domains)
        self.routing_mode = routing_mode
        self.peri_ln = bool(peri_ln)
        self.peri_pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.peri_post_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        if shared_base_ffn is not None:
            self.shared_gate_proj = shared_base_ffn[0]
            self.shared_up_proj = shared_base_ffn[1]
            self.shared_down_proj = shared_base_ffn[2]
        self.router = TopKMoERouter(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=dtype,
        )
        self.experts = nn.ModuleList(experts)
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.register_buffer("expert_token_counts", torch.zeros(self.num_experts, dtype=torch.long))
        # Season 8/9 (`HM-S8-06` & `HM-S9-02`): FoMoE partial-replica & sigma-MoE routing state.
        self.local_expert_indices: tuple[int, ...] | None = None
        self.ghost_skipped_assignments: int = 0
        self.total_routed_assignments: int = 0
        self.last_ghost_assignment_fraction: float = 0.0
        self.last_pre_peri_rms: float = 0.0
        self.last_post_peri_rms: float = 0.0

    def set_peri_ln(self, enabled: bool) -> None:
        """Enable or disable Peri-LayerNorm around the MoE block (`HM-S9-02`)."""
        self.peri_ln = bool(enabled)

    def set_local_expert_subset(self, local_expert_indices: Sequence[int] | None) -> None:
        """Restrict this worker to `M < E` resident experts (`HM-S8-06`).

        Routing assignments that target any non-resident Ghost Expert are dropped
        locally (`Skip-Token`), so no remote FFN activation RPC is ever issued.
        Passing `None` restores the full-replica default.
        """
        self.local_expert_indices = _normalize_local_expert_indices(
            local_expert_indices, self.num_experts
        )

    def hotplug_expert(self, domain_name: str, expert_module: nn.Module) -> None:
        """Hot-plug a newly converged domain expert E_{N+1} into this MoE layer."""
        self.experts.append(expert_module)
        self.expert_domains.append(domain_name)
        self.num_experts = len(self.experts)
        self.router.expand_for_new_expert()
        old_counts = self.expert_token_counts
        new_counts = torch.zeros(self.num_experts, dtype=torch.long, device=old_counts.device)
        new_counts[: old_counts.numel()] = old_counts
        self.expert_token_counts = new_counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.hidden_size)
        if self.peri_ln:
            flat_x = self.peri_pre_norm(flat_x.to(torch.float32)).to(x.dtype)
        num_slots = int(flat_x.shape[0] * self.top_k)
        self.total_routed_assignments += num_slots

        if self.routing_mode == "sigma_moe":
            from hivemind.core.fomoe_sigma_routing import sigmoid_independent_topk_routing

            x_fp32 = flat_x.to(torch.float32)
            if hasattr(self.router, "wg"):
                router_logits = self.router.wg(x_fp32)
            elif hasattr(self.router, "micro_gates"):
                router_logits = torch.cat([g(x_fp32) for g in self.router.micro_gates], dim=-1)
            else:
                router_logits = torch.zeros(
                    (x_fp32.shape[0], self.num_experts), dtype=torch.float32, device=x_fp32.device
                )
            if self.local_expert_indices is None:
                resident_mask = None
                resident_indices: Sequence[int] = range(self.num_experts)
            else:
                resident_indices = self.local_expert_indices
                resident_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=router_logits.device)
                resident_mask[list(self.local_expert_indices)] = True
            sigma_res = sigmoid_independent_topk_routing(
                router_logits=router_logits,
                top_k=self.top_k,
                resident_mask=resident_mask,
            )
            routing_weights = sigma_res.gate_values
            selected_experts = sigma_res.expert_indices
            self.last_aux_loss = torch.tensor(0.0, device=x.device)
            skipped_here = int(sigma_res.skipped_slots.sum().item())
            self.ghost_skipped_assignments += skipped_here
            self.last_ghost_assignment_fraction = sigma_res.ghost_assignment_fraction
        else:
            routing_weights, selected_experts, aux_loss = self.router(flat_x)
            self.last_aux_loss = aux_loss

            if self.local_expert_indices is None:
                resident_indices = range(self.num_experts)
                self.last_ghost_assignment_fraction = 0.0
            else:
                resident_indices = self.local_expert_indices
                resident_tensor = torch.tensor(
                    self.local_expert_indices,
                    dtype=selected_experts.dtype,
                    device=selected_experts.device,
                )
                ghost_mask = ~torch.isin(selected_experts, resident_tensor)
                skipped_here = int(ghost_mask.sum().item())
                self.ghost_skipped_assignments += skipped_here
                self.last_ghost_assignment_fraction = (
                    float(skipped_here) / float(num_slots) if num_slots > 0 else 0.0
                )

        final_hidden = torch.zeros_like(flat_x)
        with torch.no_grad():
            for k_idx in range(self.top_k):
                binc = torch.bincount(selected_experts[:, k_idx], minlength=self.num_experts)
                self.expert_token_counts.add_(binc.to(self.expert_token_counts.device))

        for expert_idx in resident_indices:
            expert_layer = self.experts[expert_idx]
            # Find tokens routed to expert_idx in any of the top_k slots
            mask = selected_experts == expert_idx  # [N, top_k]
            if not mask.any():
                continue
            token_indices, slot_indices = torch.where(mask)
            expert_in = flat_x[token_indices]
            expert_out = expert_layer(expert_in)
            weights = routing_weights[token_indices, slot_indices].unsqueeze(-1).to(expert_out.dtype)
            final_hidden.index_add_(0, token_indices, expert_out * weights)

        with torch.no_grad():
            self.last_pre_peri_rms = float(
                final_hidden.detach().to(torch.float32).pow(2).mean().sqrt().item()
            )
        if self.peri_ln and bool(final_hidden.abs().max().item() > 0.0):
            final_hidden = self.peri_post_norm(final_hidden.to(torch.float32)).to(x.dtype)
        with torch.no_grad():
            self.last_post_peri_rms = float(
                final_hidden.detach().to(torch.float32).pow(2).mean().sqrt().item()
            )

        return final_hidden.reshape(orig_shape)


class UpcycledSparseMoEModel(nn.Module):
    """Unified Sparse MoE Causal LM upcycled from E domain-specific QLoRA branches."""

    def __init__(
        self,
        upcycled_backbone: nn.Module,
        expert_domains: list[str],
        top_k: int = 2,
        router_aux_loss_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.backbone = upcycled_backbone
        self.expert_domains = list(expert_domains)
        self.num_experts = len(expert_domains)
        self.top_k = top_k
        self.router_aux_loss_coef = router_aux_loss_coef
        # Season 4 (`HM-S4-05`): telemetry of the last SVD rank-adaptive projection.
        self.last_rank_projection_report: SVDLoARankProjectionReport | None = None
        # Season 8 (`HM-S8-06`): FoMoE partial-replica expert residency.
        self.local_expert_indices: tuple[int, ...] | None = None

    def get_moe_blocks(self) -> list[SparseMoEMLPBlock]:
        blocks: list[SparseMoEMLPBlock] = []
        for mod in self.backbone.modules():
            if isinstance(mod, SparseMoEMLPBlock):
                blocks.append(mod)
        return blocks

    def set_local_expert_subset(self, local_expert_indices: Sequence[int] | None) -> None:
        """Set the resident-expert subset on every MoE block (`HM-S8-06`).

        Workers holding only `M < E` experts locally drop Ghost-Expert routing
        assignments (`Skip-Token`) instead of issuing remote FFN activation RPCs.
        """
        self.local_expert_indices = _normalize_local_expert_indices(
            local_expert_indices, self.num_experts
        )
        for blk in self.get_moe_blocks():
            blk.set_local_expert_subset(self.local_expert_indices)

    def set_routing_mode(self, routing_mode: str) -> None:
        """Switch all MoE blocks between `'softmax'` and FoMoE `'sigma_moe'` routing (`HM-S9-02`)."""
        if routing_mode not in ("softmax", "sigma_moe"):
            raise ValueError(f"routing_mode must be 'softmax' or 'sigma_moe', got {routing_mode!r}")
        for blk in self.get_moe_blocks():
            blk.routing_mode = routing_mode

    def set_peri_ln(self, enabled: bool) -> None:
        """Enable or disable Peri-LayerNorm across all MoE blocks (`HM-S9-02`)."""
        for blk in self.get_moe_blocks():
            blk.set_peri_ln(enabled)

    def measured_ghost_fraction_per_layer(self) -> list[float]:
        """Return the empirical ghost-assignment fraction observed on each MoE block (`HM-S9-02`)."""
        return [float(blk.last_ghost_assignment_fraction) for blk in self.get_moe_blocks()]

    def skip_token_flops_report(self) -> dict[str, Any]:
        """HM-S9-02: Report empirical Skip-Token FLOP savings gated on `measured_ghost_fraction_per_layer()`.

        Never claims the theoretical `(E - M) / E` formula without first executing at least one
        forward pass (`realized_from_forward_pass` must be True).
        """
        blocks = self.get_moe_blocks()
        per_layer_ghost = self.measured_ghost_fraction_per_layer()
        total_routed = sum(int(blk.total_routed_assignments) for blk in blocks)
        total_skipped = sum(int(blk.ghost_skipped_assignments) for blk in blocks)
        realized = total_routed > 0
        mean_ghost = (
            sum(per_layer_ghost) / float(len(per_layer_ghost))
            if (realized and per_layer_ghost)
            else 0.0
        )
        active_flops_multiplier = 1.0 - mean_ghost if realized else 1.0
        resident_count = (
            len(self.local_expert_indices) if self.local_expert_indices is not None else self.num_experts
        )
        return {
            "strategy_id": "HM-S9-02-SIGMA-MOE-SKIP-TOKEN-REPORT",
            "realized_from_forward_pass": realized,
            "num_experts": self.num_experts,
            "resident_experts": resident_count,
            "total_routed_assignments": total_routed,
            "ghost_skipped_assignments": total_skipped,
            "measured_ghost_fraction_per_layer": [round(g, 6) for g in per_layer_ghost],
            "mean_measured_ghost_fraction": round(mean_ghost, 6),
            "measured_expert_flops_multiplier": round(active_flops_multiplier, 6),
            "measured_expert_flops_saved_fraction": round(mean_ghost if realized else 0.0, 6),
            "peri_ln_enabled": bool(blocks and blocks[0].peri_ln),
        }

    @property
    def ghost_skipped_assignments(self) -> int:
        """Total token-to-expert assignments dropped by FoMoE `Skip-Token` routing."""
        return sum(blk.ghost_skipped_assignments for blk in self.get_moe_blocks())

    def reset_routing_telemetry(self) -> None:
        for blk in self.get_moe_blocks():
            blk.expert_token_counts.zero_()
            blk.ghost_skipped_assignments = 0
            blk.total_routed_assignments = 0
            blk.last_ghost_assignment_fraction = 0.0

    def get_expert_utilization(self) -> dict[str, float]:
        counts = torch.zeros(self.num_experts, dtype=torch.float64)
        for blk in self.get_moe_blocks():
            counts += blk.expert_token_counts.detach().cpu().to(torch.float64)
        total = float(counts.sum().item())
        if total <= 0:
            return {dom: round(1.0 / self.num_experts, 4) for dom in self.expert_domains}
        return {
            dom: round(float(counts[i].item()) / total, 4)
            for i, dom in enumerate(self.expert_domains)
        }

    def hotplug_domain_expert(
        self,
        domain_name: str,
        lora_state_dict: dict[str, torch.Tensor],
        lora_rank_r: int = 16,
        lora_alpha: int = 32,
        target_rank_r: int | None = None,
        source_lora_alpha: float | None = None,
    ) -> int:
        """Phase 3.5 (BTM MoE v2): Hot-plug a new domain LoRA expert (E -> E+1) without retraining existing experts.

        Args:
            domain_name: Human-readable domain identifier of the new expert.
            lora_state_dict: LoRA state dict of the incoming community adapter.
            lora_rank_r: Rank declared by the caller (drives the `alpha / r` scaling).
            lora_alpha: LoRA alpha declared by the caller.
            target_rank_r: When set, the adapter is first projected onto this
                canonical rank with a truncated SVD (`HM-S4-05`) so that
                heterogeneous `r=8..64` adapters can share one canonical layout.
            source_lora_alpha: `alpha` of the incoming adapter (defaults to `lora_alpha`).

        Returns:
            The new total number of experts `E+1`.
        """
        scaling = float(lora_alpha) / float(lora_rank_r)
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        cfg = self.backbone.config
        blocks = self.get_moe_blocks()

        if target_rank_r is not None:
            src_alpha = float(source_lora_alpha) if source_lora_alpha is not None else float(lora_alpha)
            # The model keeps applying `lora_alpha / lora_rank_r` to the projected
            # adapter, so the projection must compensate with the same scaling.
            dst_alpha = float(lora_alpha) * float(target_rank_r) / float(lora_rank_r)
            lora_state_dict, projection_report = project_lora_state_dict_to_rank(
                lora_state_dict,
                target_rank_r=target_rank_r,
                source_lora_alpha=src_alpha,
                target_lora_alpha=dst_alpha,
            )
            self.last_rank_projection_report = projection_report

        for layer_idx, blk in enumerate(blocks):
            ab_gate = _extract_ab_pair(lora_state_dict, layer_idx, "mlp.gate_proj")
            ab_up = _extract_ab_pair(lora_state_dict, layer_idx, "mlp.up_proj")
            ab_down = _extract_ab_pair(lora_state_dict, layer_idx, "mlp.down_proj")

            if hasattr(blk, "shared_gate_proj"):
                shared_ffn = (blk.shared_gate_proj, blk.shared_up_proj, blk.shared_down_proj)
                expert_mod: nn.Module = UpcycledFactoredSwiGLUExpert(
                    shared_base_ffn=shared_ffn,
                    ab_gate=ab_gate,
                    ab_up=ab_up,
                    ab_down=ab_down,
                    scaling=scaling,
                    dtype=dtype,
                    device=device,
                )
            else:
                ref_exp: UpcycledSwiGLUExpert = blk.experts[0]  # type: ignore[assignment]
                w0_gate = ref_exp.gate_proj.weight.detach().to(torch.float32)
                w0_up = ref_exp.up_proj.weight.detach().to(torch.float32)
                w0_down = ref_exp.down_proj.weight.detach().to(torch.float32)
                w_gate = w0_gate + (
                    scaling * (ab_gate[1].to(device) @ ab_gate[0].to(device)) if ab_gate else 0.0
                )
                w_up = w0_up + (
                    scaling * (ab_up[1].to(device) @ ab_up[0].to(device)) if ab_up else 0.0
                )
                w_down = w0_down + (
                    scaling * (ab_down[1].to(device) @ ab_down[0].to(device)) if ab_down else 0.0
                )
                expert_mod = UpcycledSwiGLUExpert(
                    hidden_size=cfg.hidden_size,
                    intermediate_size=cfg.intermediate_size,
                    gate_weight=w_gate,
                    up_weight=w_up,
                    down_weight=w_down,
                    dtype=dtype,
                ).to(device)

            blk.hotplug_expert(domain_name=domain_name, expert_module=expert_mod)

        self.expert_domains.append(domain_name)
        self.num_experts = len(self.expert_domains)
        return self.num_experts

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: Any) -> Any:
        out = self.backbone(input_ids=input_ids, labels=labels, **kwargs)
        if labels is not None and hasattr(out, "loss") and out.loss is not None:
            aux_losses = [blk.last_aux_loss.to(out.loss.device) for blk in self.get_moe_blocks()]
            if aux_losses:
                total_aux = torch.stack(aux_losses).mean()
                out.loss = out.loss + self.router_aux_loss_coef * total_aux
        return out


LOOPED_INPUT_INJECTION_MODES = ("add", "none", "orthogonal")


def orthogonal_input_injection(
    recurrent_state: torch.Tensor,
    prelude_state: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """HM-S9-03: Orthogonal Input Injection (`OrthoInj`, `arXiv:2606.15039`).

    Projects `prelude_state` (`u`) onto the subspace orthogonal to `recurrent_state` (`h_t`)
    along the hidden dimension (`dim=-1`) before adding:
      u_perp = u - (<u, h_t> / (||h_t||_2^2 + eps)) * h_t
      h_next = h_t + u_perp
    Prevents collinear variance explosion across `T` recurrent passes while preserving
    exact orthogonality `<u_perp, h_t> = 0` when `||h_t||_2 > 0`.
    """
    h = recurrent_state.to(torch.float32)
    u = prelude_state.to(torch.float32)
    h_norm_sq = torch.sum(h * h, dim=-1, keepdim=True)
    dot_uh = torch.sum(u * h, dim=-1, keepdim=True)
    proj_coeff = torch.where(
        h_norm_sq > eps,
        dot_uh / h_norm_sq.clamp_min(eps),
        torch.zeros_like(h_norm_sq),
    )
    u_perp = u - proj_coeff * h
    return (h + u_perp).to(dtype=recurrent_state.dtype)


class LearnedDepthPrior(nn.Module):
    """HM-S9-03: Learned Categorical Depth Prior with Shannon Entropy & Budget Regularisation (`arXiv:2510.25741`).

    Predicts a per-token categorical probability distribution `q_phi(t | x)` over
    `loop_steps` (`t in {1..T}`) from the prelude hidden representation and computes:
      L_depth = budget_coef * max(0, E_q[t] - target_budget)^2 - entropy_coef * H(q_phi)
    """

    def __init__(
        self,
        hidden_size: int,
        loop_steps: int,
        entropy_coef: float = 0.01,
        budget_coef: float = 0.05,
        target_budget_steps: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or loop_steps < 1:
            raise ValueError("hidden_size >= 1 and loop_steps >= 1 required")
        if entropy_coef < 0.0 or budget_coef < 0.0:
            raise ValueError("entropy_coef and budget_coef must be >= 0")
        self.hidden_size = int(hidden_size)
        self.loop_steps = int(loop_steps)
        self.entropy_coef = float(entropy_coef)
        self.budget_coef = float(budget_coef)
        self.target_budget_steps = (
            float(target_budget_steps)
            if target_budget_steps is not None
            else max(1.0, 0.65 * float(loop_steps))
        )
        self.gate = nn.Linear(hidden_size, loop_steps, bias=True, dtype=torch.float32)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def depth_probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return categorical depth probabilities `q(t | x)` of shape `[..., T]`."""
        logits = self.gate(hidden_states.to(torch.float32))
        return F.softmax(logits, dim=-1)

    @staticmethod
    def expected_depth(probs: torch.Tensor) -> torch.Tensor:
        """Compute expected recurrent depth `E_q[t] = sum_{t=1}^T t * q_t` in `[1, T]`."""
        t_steps = probs.shape[-1]
        step_grid = torch.arange(1, t_steps + 1, dtype=probs.dtype, device=probs.device)
        return torch.sum(probs * step_grid, dim=-1)

    @staticmethod
    def shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Compute per-token Shannon entropy `H(q) = -sum_t q_t ln(q_t)` in `[0, ln(T)]`."""
        p_clamped = probs.clamp_min(eps)
        return -torch.sum(probs * torch.log(p_clamped), dim=-1)

    def regularization_loss(
        self,
        hidden_states: torch.Tensor,
        target_budget_steps: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate `(loss, telemetry)` for the learned depth prior."""
        probs = self.depth_probabilities(hidden_states)
        exp_depth = self.expected_depth(probs)
        ent = self.shannon_entropy(probs)
        budget = (
            float(target_budget_steps)
            if target_budget_steps is not None
            else self.target_budget_steps
        )
        excess = F.relu(exp_depth.mean() - budget)
        budget_penalty = self.budget_coef * (excess * excess)
        entropy_bonus = self.entropy_coef * ent.mean()
        total_reg = budget_penalty - entropy_bonus
        return total_reg, {
            "expected_depth_steps": round(float(exp_depth.detach().mean().item()), 6),
            "target_budget_steps": round(budget, 6),
            "shannon_entropy_nats": round(float(ent.detach().mean().item()), 6),
            "max_possible_entropy_nats": round(math.log(max(1, self.loop_steps)), 6),
            "depth_regularization_loss": round(float(total_reg.detach().item()), 6),
        }


def _unique_weight_bytes(module: nn.Module) -> int:
    """Bytes of parameters and buffers as stored, counting tied storage once."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in [*module.parameters(), *module.buffers()]:
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        if key in seen:
            continue
        seen.add(key)
        total += tensor.numel() * tensor.element_size()
    return total


def _causal_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive 4-D causal mask `[B, 1, S, S]`, with padded key positions masked out."""
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device).triu(1)
    mask = mask[None, None].expand(batch_size, 1, seq_len, seq_len).clone()
    if attention_mask is not None:
        padded_keys = attention_mask[:, None, None, :].to(device=device, dtype=torch.bool).logical_not()
        mask = mask.masked_fill(padded_keys, min_value)
    return mask


class LoopStepLoRAAdapter(nn.Module):
    """Residual low-rank adapter `h + B_t A_t h` applied before loop step t.

    `B_t` starts at zero, so an untrained adapter is the identity and the loop
    begins as plain weight-tied recurrence.
    """

    def __init__(self, hidden_size: int, rank: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=dtype, device=device)
        nn.init.normal_(self.down.weight, std=1.0 / hidden_size)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.down(hidden_states))


@dataclass
class LoopedBackboneReport:
    """Season 7 (`HM-S7-01`): shape and measured weight bytes of a looped backbone.

    Byte counts are tensors as stored on the device they were measured on (NF4
    packed when the backbone is quantized, fp32 on a CPU test model). They are
    not a VRAM measurement: activations and KV cache still grow with the loop.
    """

    source_depth: int
    num_prelude_layers: int
    num_core_layers: int
    num_coda_layers: int
    loop_steps: int
    effective_depth: int
    depth_multiplier: float
    input_injection: str
    per_step_lora_rank: int
    source_weight_bytes: int
    core_weight_bytes: int
    loop_adapter_bytes: int
    looped_weight_bytes: int
    unrolled_weight_bytes: int
    last_mean_effective_steps: float | None = None
    last_early_exit_fraction: float | None = None
    last_fixed_point_residual: float | None = None

    @property
    def extra_weight_bytes(self) -> int:
        """Bytes the looped model stores beyond the source model (only the step adapters)."""
        return self.looped_weight_bytes - self.source_weight_bytes

    @property
    def saved_weight_bytes(self) -> int:
        """Bytes saved versus unrolling the core into `loop_steps` separate copies."""
        return self.unrolled_weight_bytes - self.looped_weight_bytes


class LoopedExpertBackbone(nn.Module):
    """Season 7 (`HM-S7-01`): `Prelude -> RecurrentCore x T -> Coda` over a decoder stack.

    The first `num_prelude_layers` decoder layers run once, the middle (core)
    layers run `loop_steps` times with the *same* modules (weight-tied, no
    copies), and the last `num_coda_layers` layers plus the final norm and LM
    head form the coda. Before each core pass after the first, `input_injection="add"`
    adds the prelude output back into the recurrent state, and an optional
    per-step LoRA adapter lets each pass differ.

    Wraps either an `UpcycledSparseMoEModel` (routing telemetry, router aux loss
    and `train_moe_router_gating` keep working) or a plain Qwen2-style causal LM.
    Runs without a KV cache.

    Optional fixed-point halting freezes recurrent token states, retaining them
    as attention context. Effective steps describe logical depth: decoder kernels
    remain dense until the whole batch halts. Truncated BPTT tracks a global
    suffix of at most k executed core passes; tokens halted before its boundary
    may stay detached. With adaptive halting and k < T, gradient-enabled forwards
    discover the depth without gradients, then recompute at most k passes with
    RNG restored. Replay replaces auxiliary losses and does not count routes twice.
    """

    def __init__(
        self,
        model: nn.Module,
        num_prelude_layers: int = 1,
        num_coda_layers: int = 1,
        loop_steps: int = 2,
        per_step_lora_rank: int = 4,
        input_injection: str = "add",
        router_aux_loss_coef: float | None = None,
        fixed_point_tol: float | None = None,
        truncated_bptt_steps: int | None = None,
        use_learned_depth_prior: bool = False,
        depth_entropy_coef: float = 0.01,
        depth_budget_coef: float = 0.05,
        target_depth_budget: float | None = None,
    ) -> None:
        super().__init__()
        moe_model = model if isinstance(model, UpcycledSparseMoEModel) else None
        causal_lm = moe_model.backbone if moe_model is not None else model
        decoder = getattr(causal_lm, "model", None)
        layers = getattr(decoder, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not hasattr(causal_lm, "lm_head"):
            raise ValueError("model must be a causal LM exposing .model.layers and .lm_head")
        if loop_steps < 1:
            raise ValueError("loop_steps must be at least 1")
        if num_prelude_layers < 0 or num_coda_layers < 0:
            raise ValueError("prelude and coda layer counts must be non-negative")
        num_core = len(layers) - num_prelude_layers - num_coda_layers
        if num_core < 1:
            raise ValueError(
                f"{len(layers)} decoder layers leave no core after {num_prelude_layers} prelude "
                f"and {num_coda_layers} coda layers"
            )
        if input_injection not in LOOPED_INPUT_INJECTION_MODES:
            raise ValueError(f"input_injection must be one of {LOOPED_INPUT_INJECTION_MODES}")
        if per_step_lora_rank < 0:
            raise ValueError("per_step_lora_rank must be non-negative")
        if fixed_point_tol is not None and (not math.isfinite(fixed_point_tol) or fixed_point_tol < 0):
            raise ValueError("fixed_point_tol must be finite and non-negative")
        if truncated_bptt_steps is not None and (
            isinstance(truncated_bptt_steps, bool)
            or not isinstance(truncated_bptt_steps, int)
            or not 1 <= truncated_bptt_steps <= loop_steps
        ):
            raise ValueError("truncated_bptt_steps must be an integer in [1, loop_steps]")

        self.source_model = model
        self.moe_model = moe_model
        self.embed_tokens = decoder.embed_tokens
        self.rotary_emb = getattr(decoder, "rotary_emb", None)
        self.norm = decoder.norm
        self.lm_head = causal_lm.lm_head
        self.prelude = nn.ModuleList(layers[:num_prelude_layers])
        self.core = nn.ModuleList(layers[num_prelude_layers : num_prelude_layers + num_core])
        self.coda = nn.ModuleList(layers[num_prelude_layers + num_core :])
        self.loop_steps = loop_steps
        self.input_injection = input_injection
        self.per_step_lora_rank = per_step_lora_rank
        self.fixed_point_tol = fixed_point_tol
        self.truncated_bptt_steps = truncated_bptt_steps
        self.last_mean_effective_steps: float | None = None
        self.last_early_exit_fraction: float | None = None
        self.last_fixed_point_residual: float | None = None
        self.last_depth_prior_telemetry: dict[str, float] | None = None
        if router_aux_loss_coef is None:
            router_aux_loss_coef = moe_model.router_aux_loss_coef if moe_model is not None else 0.0
        self.router_aux_loss_coef = router_aux_loss_coef

        embed_weight = self.embed_tokens.weight
        self.step_adapters = nn.ModuleList(
            LoopStepLoRAAdapter(embed_weight.shape[1], per_step_lora_rank, embed_weight.dtype, embed_weight.device)
            for _ in range(loop_steps if per_step_lora_rank > 0 else 0)
        )
        self.depth_prior: LearnedDepthPrior | None = (
            LearnedDepthPrior(
                hidden_size=embed_weight.shape[1],
                loop_steps=loop_steps,
                entropy_coef=depth_entropy_coef,
                budget_coef=depth_budget_coef,
                target_budget_steps=target_depth_budget,
            ).to(device=embed_weight.device)
            if use_learned_depth_prior
            else None
        )
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def get_moe_blocks(self) -> list[SparseMoEMLPBlock]:
        return self.moe_model.get_moe_blocks() if self.moe_model is not None else []

    def reset_routing_telemetry(self) -> None:
        if self.moe_model is not None:
            self.moe_model.reset_routing_telemetry()

    def get_expert_utilization(self) -> dict[str, float]:
        return self.moe_model.get_expert_utilization() if self.moe_model is not None else {}

    def _run_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        aux_losses: list[torch.Tensor],
    ) -> torch.Tensor:
        out = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden_states = out[0] if isinstance(out, tuple) else out
        aux_losses.extend(blk.last_aux_loss for blk in layer.modules() if isinstance(blk, SparseMoEMLPBlock))
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        from transformers.modeling_outputs import CausalLMOutputWithPast

        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.rotary_emb is not None else None
        mask = _causal_attention_mask(attention_mask, batch_size, seq_len, hidden_states.dtype, hidden_states.device)
        run_args = (mask, position_ids, position_embeddings)
        aux_losses: list[torch.Tensor] = []

        for layer in self.prelude:
            hidden_states = self._run_layer(layer, hidden_states, *run_args, aux_losses)
        injected = hidden_states
        real = torch.ones_like(input_ids, dtype=torch.bool, device=hidden_states.device)
        if attention_mask is not None:
            real = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        active = real.clone()
        effective_steps = torch.zeros_like(input_ids, dtype=torch.float32, device=hidden_states.device)
        residual = torch.zeros_like(effective_steps)
        adaptive = self.fixed_point_tol is not None
        grad_enabled = torch.is_grad_enabled()
        k = self.truncated_bptt_steps or self.loop_steps
        replay = adaptive and k < self.loop_steps and grad_enabled
        history = deque(maxlen=k)
        for step in range(self.loop_steps):
            if adaptive and not active.any():
                break
            if replay:
                history.append((step, hidden_states.detach(), active.clone(), torch.get_rng_state(),
                                torch.cuda.get_rng_state(hidden_states.device) if hidden_states.is_cuda else None,
                                len(aux_losses)))
            if k < self.loop_steps and step == self.loop_steps - k:
                hidden_states, injected = hidden_states.detach(), injected.detach()
            previous = hidden_states
            with torch.set_grad_enabled(grad_enabled and not replay and step >= self.loop_steps - k):
                hidden_states = self._run_core(step, previous, injected, active if adaptive else None,
                                               run_args, aux_losses)
            # Detached FP32 drift compares complete passes to the pre-injection state.
            with torch.no_grad():
                prev = previous.detach().float()
                drift = (hidden_states.detach().float() - prev).norm(dim=-1) / prev.norm(dim=-1).clamp_min(1e-6)
                residual = torch.where(active, drift, residual)
                effective_steps += active
                if adaptive:
                    active = active & (drift > self.fixed_point_tol)
        if replay and history:
            _, hidden_states, _, cpu_rng, cuda_rng, aux_start = history[0]
            injected = injected.detach()
            aux_losses = aux_losses[:aux_start]
            blocks = self.get_moe_blocks()
            counts = [blk.expert_token_counts.clone() for blk in blocks]
            ghost_counts = [blk.ghost_skipped_assignments for blk in blocks]
            devices = [hidden_states.device.index] if hidden_states.is_cuda else []
            # fork_rng leaves the caller's RNG at the end of logical execution.
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, hidden_states.device)
                try:
                    for step, _, step_active, _, _, _ in history:
                        hidden_states = self._run_core(step, hidden_states, injected, step_active,
                                                       run_args, aux_losses)
                finally:
                    with torch.no_grad():
                        for blk, count, ghost_count in zip(blocks, counts, ghost_counts):
                            blk.expert_token_counts.copy_(count)
                            blk.ghost_skipped_assignments = ghost_count
        if real.any():
            self.last_mean_effective_steps = effective_steps[real].mean().item()
            self.last_early_exit_fraction = (effective_steps[real] < self.loop_steps).float().mean().item()
            self.last_fixed_point_residual = residual[real].mean().item()
        else:
            self.last_mean_effective_steps = self.last_early_exit_fraction = self.last_fixed_point_residual = 0.0
        for layer in self.coda:
            hidden_states = self._run_layer(layer, hidden_states, *run_args, aux_losses)

        logits = self.lm_head(self.norm(hidden_states))
        self.last_aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=logits.device)
        depth_reg_loss = torch.tensor(0.0, device=logits.device)
        if self.depth_prior is not None:
            depth_reg_loss, depth_telem = self.depth_prior.regularization_loss(injected)
            self.last_depth_prior_telemetry = depth_telem
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
            loss = F.cross_entropy(shift_logits, labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = loss + self.router_aux_loss_coef * self.last_aux_loss.to(loss.device)
            if self.depth_prior is not None:
                loss = loss + depth_reg_loss.to(loss.device)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def _run_core(
        self,
        step: int,
        previous: torch.Tensor,
        injected: torch.Tensor,
        active: torch.Tensor | None,
        run_args: tuple,
        aux_losses: list[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = previous
        if step > 0 and self.input_injection == "add":
            hidden_states = hidden_states + injected
        elif step > 0 and self.input_injection == "orthogonal":
            hidden_states = orthogonal_input_injection(hidden_states, injected)
        if self.step_adapters:
            hidden_states = self.step_adapters[step](hidden_states)
        if active is not None:
            hidden_states = torch.where(active.unsqueeze(-1), hidden_states, previous)
        for layer in self.core:
            hidden_states = self._run_layer(layer, hidden_states, *run_args, aux_losses)
        return hidden_states if active is None else torch.where(active.unsqueeze(-1), hidden_states, previous)

    @torch.no_grad()
    def evaluate_terminal_kv_sharing(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """HM-S9-03: Evaluate terminal-KV sharing (`Ouro`, `arXiv:2510.25741`) across `loop_steps`.

        Compares the representations and output logits produced by the terminal recurrent
        pass (`t = T - 1`) against intermediate passes (`t < T - 1`) and quantifies the
        KV-cache memory reduction ratio `(T - 1) / T` achieved when non-terminal loop KV
        caches are discarded/shared from the terminal core pass.
        """
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.rotary_emb is not None else None
        mask = _causal_attention_mask(attention_mask, batch_size, seq_len, hidden_states.dtype, hidden_states.device)
        run_args = (mask, position_ids, position_embeddings)
        aux_losses: list[torch.Tensor] = []

        for layer in self.prelude:
            hidden_states = self._run_layer(layer, hidden_states, *run_args, aux_losses)
        injected = hidden_states

        pass_states: list[torch.Tensor] = []
        for step in range(self.loop_steps):
            hidden_states = self._run_core(step, hidden_states, injected, None, run_args, aux_losses)
            pass_states.append(hidden_states.detach().to(torch.float32))

        terminal_state = pass_states[-1]
        penultimate_state = pass_states[-2] if len(pass_states) >= 2 else terminal_state
        cos_sim = float(
            F.cosine_similarity(
                penultimate_state.reshape(-1, penultimate_state.shape[-1]),
                terminal_state.reshape(-1, terminal_state.shape[-1]),
                dim=-1,
            ).mean().item()
        )
        kv_saved_fraction = float(max(0, self.loop_steps - 1)) / float(max(1, self.loop_steps))
        return {
            "strategy_id": "HM-S9-03-TERMINAL-KV-SHARING",
            "loop_steps": self.loop_steps,
            "num_core_layers": len(self.core),
            "input_injection": self.input_injection,
            "terminal_vs_penultimate_cosine": round(cos_sim, 6),
            "core_kv_cache_saved_fraction": round(kv_saved_fraction, 6),
            "core_kv_cache_multiplier": round(1.0 / float(max(1, self.loop_steps)), 6),
        }

    def report(self) -> LoopedBackboneReport:
        source_depth = len(self.prelude) + len(self.core) + len(self.coda)
        effective_depth = len(self.prelude) + len(self.core) * self.loop_steps + len(self.coda)
        source_bytes = _unique_weight_bytes(self.source_model)
        core_bytes = _unique_weight_bytes(self.core)
        adapter_bytes = _unique_weight_bytes(self.step_adapters)
        return LoopedBackboneReport(
            source_depth=source_depth,
            num_prelude_layers=len(self.prelude),
            num_core_layers=len(self.core),
            num_coda_layers=len(self.coda),
            loop_steps=self.loop_steps,
            effective_depth=effective_depth,
            depth_multiplier=round(effective_depth / source_depth, 4),
            input_injection=self.input_injection,
            per_step_lora_rank=self.per_step_lora_rank,
            source_weight_bytes=source_bytes,
            core_weight_bytes=core_bytes,
            loop_adapter_bytes=adapter_bytes,
            looped_weight_bytes=_unique_weight_bytes(self),
            unrolled_weight_bytes=source_bytes + core_bytes * (self.loop_steps - 1) + adapter_bytes,
            last_mean_effective_steps=self.last_mean_effective_steps,
            last_early_exit_fraction=self.last_early_exit_fraction,
            last_fixed_point_residual=self.last_fixed_point_residual,
        )


def measure_moe_vram_footprint_gb(
    moe_model: UpcycledSparseMoEModel,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Measure the resident VRAM footprint of an upcycled Sparse MoE in GiB.

    The factored (VRAM-efficient) expert layout stores only the LoRA factors
    `(A, B)` of every expert on top of the *shared* base FFN, so the footprint
    grows linearly with the number of hot-plugged experts while the base weights
    are paid for only once.

    Args:
        moe_model: The upcycled Sparse MoE to profile.
        device: Device to query for live CUDA allocations (auto-detected by default).

    Returns:
        Dictionary with the weight/buffer breakdown (MiB & GiB) and, when running
        on CUDA, the live and peak allocated VRAM.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in moe_model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in moe_model.buffers())
    total_bytes = param_bytes + buffer_bytes

    if device is None:
        try:
            device = next(moe_model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless model
            device = torch.device("cpu")

    footprint: dict[str, float] = {
        "num_experts": float(moe_model.num_experts),
        "param_mb": round(param_bytes / (1024**2), 4),
        "buffer_mb": round(buffer_bytes / (1024**2), 4),
        "total_mb": round(total_bytes / (1024**2), 4),
        "total_gb": round(total_bytes / (1024**3), 6),
        "cuda_allocated_gb": 0.0,
        "cuda_peak_gb": 0.0,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        footprint["cuda_allocated_gb"] = round(torch.cuda.memory_allocated(device) / (1024**3), 6)
        footprint["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024**3), 6)
    return footprint


def hotplug_rank_adaptive_experts(
    moe_model: UpcycledSparseMoEModel,
    expert_lora_states: dict[str, dict[str, torch.Tensor]],
    target_rank_r: int = 16,
    lora_alpha: int | None = None,
    expert_lora_alphas: dict[str, float] | None = None,
    vram_budget_gb: float = 2.0,
) -> dict[str, Any]:
    """Season 4 (`HM-S4-05`): hot-plug heterogeneous community adapters at scale.

    Every incoming adapter is projected onto the canonical `target_rank_r` with a
    truncated SVD (`project_lora_state_dict_to_rank`) before being hot-plugged,
    so `r=8`, `r=16`, `r=32` and `r=64` community adapters can all live inside a
    single `UpcycledSparseMoEModel` without VRAM bloat.

    Args:
        moe_model: Target Sparse MoE receiving the new experts.
        expert_lora_states: Mapping `domain_name -> LoRA state dict`.
        target_rank_r: Canonical rank every adapter is projected onto.
        lora_alpha: Alpha the projected adapters are used with (defaults to the
            community-standard `2 * target_rank_r`).
        expert_lora_alphas: Optional per-domain source alpha of the incoming adapter.
        vram_budget_gb: VRAM budget used to flag budget compliance.

    Returns:
        Telemetry dictionary with the expert counts, per-expert projection reports
        and the VRAM footprint measured before and after the hot-plug.
    """
    if target_rank_r < 1:
        raise ValueError(f"target_rank_r must be >= 1, got {target_rank_r}.")

    dst_alpha = float(lora_alpha) if lora_alpha is not None else 2.0 * float(target_rank_r)
    footprint_before = measure_moe_vram_footprint_gb(moe_model)

    projection_reports: dict[str, dict[str, Any]] = {}
    plugged_domains: list[str] = []
    for domain_name, lora_state in expert_lora_states.items():
        src_alpha = (
            float(expert_lora_alphas[domain_name])
            if expert_lora_alphas is not None and domain_name in expert_lora_alphas
            else None  # fall back to the per-pair community-standard `alpha = 2 * r`
        )
        projected, report = project_lora_state_dict_to_rank(
            lora_state,
            target_rank_r=target_rank_r,
            source_lora_alpha=src_alpha,
            target_lora_alpha=dst_alpha,
        )
        projection_reports[domain_name] = {
            "source_rank_r": report.source_rank_r,
            "target_rank_r": report.target_rank_r,
            "num_projected_pairs": report.num_projected_pairs,
            "num_padded_pairs": report.num_padded_pairs,
            "num_identity_pairs": report.num_identity_pairs,
            "mean_spectral_energy_retained": report.mean_spectral_energy_retained,
            "min_spectral_energy_retained": report.min_spectral_energy_retained,
            "max_relative_delta_error": report.max_relative_delta_error,
            "compression_ratio": report.compression_ratio,
            "source_params": report.source_params,
            "target_params": report.target_params,
        }
        moe_model.hotplug_domain_expert(
            domain_name=domain_name,
            lora_state_dict=projected,
            lora_rank_r=target_rank_r,
            lora_alpha=int(dst_alpha),
        )
        plugged_domains.append(domain_name)

    footprint_after = measure_moe_vram_footprint_gb(moe_model)
    return {
        "initial_num_experts": int(footprint_before["num_experts"]),
        "final_num_experts": int(moe_model.num_experts),
        "num_plugged_experts": len(plugged_domains),
        "plugged_domains": plugged_domains,
        "target_rank_r": int(target_rank_r),
        "lora_alpha": dst_alpha,
        "projection_reports": projection_reports,
        "vram_footprint_before_gb": footprint_before["total_gb"],
        "vram_footprint_after_gb": footprint_after["total_gb"],
        "vram_footprint_after_mb": footprint_after["total_mb"],
        "vram_budget_gb": float(vram_budget_gb),
        "within_vram_budget": bool(footprint_after["total_gb"] < float(vram_budget_gb)),
    }


def upcycle_branches_to_sparse_moe(
    peft_or_base_model: nn.Module,
    branch_lora_states: dict[str, dict[str, torch.Tensor]],
    lora_rank_r: int = 64,
    lora_alpha: int = 128,
    top_k: int = 2,
    router_aux_loss_coef: float = 0.01,
    target_device: torch.device | None = None,
    target_dtype: torch.dtype = torch.float32,
    in_place: bool = False,
    attention_merge_mode: str = "uniform",
    branch_reputations: dict[str, float] | None = None,
) -> tuple[UpcycledSparseMoEModel, MoEUpcycleReport]:
    """Phase C: Convert E converged QLoRA domain branches into a unified Sparse MoE model.

    `attention_merge_mode` selects how the shared self-attention LoRA deltas are
    merged (`HM-S8-03`):
      - `"uniform"`: arithmetic mean across branches (legacy behavior).
      - `"iso_cts"`: reputation-weighted TIES + ISO-CTS spectral isotropy.
      - `"tsv"`: Task Singular Vector subspace decorrelation.

    `branch_reputations` optionally supplies a non-negative reputation weight per
    domain for the `"iso_cts"` and `"tsv"` strategies.
    """
    expert_domains = list(branch_lora_states.keys())
    E = len(expert_domains)
    if E < 2:
        raise ValueError(f"Sparse MoE upcycling requires at least 2 expert branches, got {E}.")
    if attention_merge_mode not in ATTENTION_MERGE_MODES:
        raise ValueError(
            f"attention_merge_mode must be one of {ATTENTION_MERGE_MODES}, "
            f"got {attention_merge_mode!r}"
        )

    branch_weights = [1.0] * E
    if branch_reputations is not None:
        unknown_domains = sorted(set(branch_reputations) - set(expert_domains))
        if unknown_domains:
            raise ValueError(
                f"branch_reputations contains unknown domains: {unknown_domains}"
            )
        for idx, domain in enumerate(expert_domains):
            try:
                weight = float(branch_reputations.get(domain, 1.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"branch_reputations[{domain!r}] must be a finite non-negative number"
                ) from exc
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"branch_reputations[{domain!r}] must be finite and non-negative, "
                    f"got {weight}"
                )
            branch_weights[idx] = weight
        if not any(weight > 0.0 for weight in branch_weights):
            raise ValueError("branch_reputations must contain at least one positive weight")

    scaling = float(lora_alpha) / float(lora_rank_r)

    # Unwrap PeftModel to obtain the underlying CausalLM decoder structure
    raw_causal_lm = peft_or_base_model
    if not hasattr(getattr(raw_causal_lm, "model", None), "layers"):
        raw_causal_lm = getattr(peft_or_base_model, "base_model", peft_or_base_model)
    if hasattr(raw_causal_lm, "model") and hasattr(raw_causal_lm.model, "model"):
        raw_causal_lm = raw_causal_lm.model

    if target_device is None:
        target_device = next(raw_causal_lm.parameters()).device

    cfg = copy.deepcopy(raw_causal_lm.config)

    if in_place:
        merged_causal_lm = raw_causal_lm
    else:
        from transformers import Qwen2ForCausalLM

        merged_causal_lm = Qwen2ForCausalLM(cfg).to(dtype=target_dtype)
        with torch.no_grad():
            merged_causal_lm.model.embed_tokens.weight.copy_(
                raw_causal_lm.model.embed_tokens.weight.detach().to(dtype=target_dtype, device="cpu")
            )
            merged_causal_lm.model.norm.weight.copy_(
                raw_causal_lm.model.norm.weight.detach().to(dtype=target_dtype, device="cpu")
            )
            if not cfg.tie_word_embeddings:
                merged_causal_lm.lm_head.weight.copy_(
                    raw_causal_lm.lm_head.weight.detach().to(dtype=target_dtype, device="cpu")
                )

    attn_merged_count = 0
    ffn_experts_created = 0
    work_dev = target_device if in_place else torch.device("cpu")

    with torch.no_grad():
        for layer_idx, src_layer in enumerate(raw_causal_lm.model.layers):
            dst_layer = merged_causal_lm.model.layers[layer_idx]
            if not in_place:
                dst_layer.input_layernorm.weight.copy_(
                    src_layer.input_layernorm.weight.detach().to(dtype=target_dtype, device="cpu")
                )
                dst_layer.post_attention_layernorm.weight.copy_(
                    src_layer.post_attention_layernorm.weight.detach().to(dtype=target_dtype, device="cpu")
                )

            # 1. Merge self-attention LoRA adapters across all E branches into W_0
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                src_proj = getattr(src_layer.self_attn, proj_name)
                base_lin = getattr(src_proj, "base_layer", src_proj)
                avg_delta = (
                    torch.zeros_like(base_lin.weight.data, dtype=torch.float32, device=work_dev)
                    if attention_merge_mode == "uniform"
                    else None
                )
                projection_deltas: list[torch.Tensor] = []
                projection_weights: list[float] = []
                for branch_idx, domain in enumerate(expert_domains):
                    ab = _extract_ab_pair(
                        branch_lora_states[domain],
                        layer_idx=layer_idx,
                        subpath=f"self_attn.{proj_name}",
                    )
                    if ab is not None:
                        a_mat = ab[0].to(device=work_dev, dtype=torch.float32)
                        b_mat = ab[1].to(device=work_dev, dtype=torch.float32)
                        delta = scaling * (b_mat @ a_mat)
                        if attention_merge_mode == "uniform":
                            assert avg_delta is not None
                            avg_delta.add_(delta)
                        else:
                            projection_deltas.append(delta)
                            projection_weights.append(branch_weights[branch_idx])
                        del a_mat, b_mat, delta

                if attention_merge_mode == "uniform":
                    assert avg_delta is not None
                    # Missing adapters contribute a zero delta, preserving the 1/E mean.
                    avg_delta.div_(float(E))
                    merged_delta = avg_delta
                elif projection_deltas:
                    merged_delta = _merge_attention_projection_deltas(
                        projection_deltas,
                        projection_weights,
                        attention_merge_mode,
                    )
                    del projection_deltas
                else:
                    merged_delta = torch.zeros_like(
                        base_lin.weight.data, dtype=torch.float32, device=work_dev
                    )
                merged_delta = merged_delta.to(device=work_dev, dtype=torch.float32)

                if in_place:
                    base_lin.weight.data.copy_(
                        (base_lin.weight.data.to(torch.float32) + merged_delta).to(target_dtype)
                    )
                    setattr(dst_layer.self_attn, proj_name, base_lin)
                else:
                    w0 = base_lin.weight.detach().to(device=work_dev, dtype=torch.float32)
                    new_attn_lin = nn.Linear(
                        base_lin.in_features,
                        base_lin.out_features,
                        bias=base_lin.bias is not None,
                        device=work_dev,
                        dtype=target_dtype,
                    )
                    new_attn_lin.weight.copy_((w0 + merged_delta).to(target_dtype))
                    if base_lin.bias is not None and new_attn_lin.bias is not None:
                        new_attn_lin.bias.copy_(base_lin.bias.detach().to(device=work_dev, dtype=target_dtype))
                    setattr(dst_layer.self_attn, proj_name, new_attn_lin)
                    del w0
                del merged_delta
                attn_merged_count += 1

            # 2. Construct E distinct FFN experts: W_{ffn, e} = W_0_ffn + (alpha/r) * B_e @ A_e
            base_gate = getattr(src_layer.mlp.gate_proj, "base_layer", src_layer.mlp.gate_proj)
            base_up = getattr(src_layer.mlp.up_proj, "base_layer", src_layer.mlp.up_proj)
            base_down = getattr(src_layer.mlp.down_proj, "base_layer", src_layer.mlp.down_proj)

            layer_experts: list[nn.Module] = []
            if in_place:
                shared_ffn = (base_gate, base_up, base_down)
                for domain in expert_domains:
                    b_state = branch_lora_states[domain]
                    ab_gate = _extract_ab_pair(b_state, layer_idx, "mlp.gate_proj")
                    ab_up = _extract_ab_pair(b_state, layer_idx, "mlp.up_proj")
                    ab_down = _extract_ab_pair(b_state, layer_idx, "mlp.down_proj")
                    expert_mod = UpcycledFactoredSwiGLUExpert(
                        shared_base_ffn=shared_ffn,
                        ab_gate=ab_gate,
                        ab_up=ab_up,
                        ab_down=ab_down,
                        scaling=scaling,
                        dtype=target_dtype,
                        device=work_dev,
                    )
                    layer_experts.append(expert_mod)
                    ffn_experts_created += 1

                dst_layer.mlp = SparseMoEMLPBlock(
                    hidden_size=cfg.hidden_size,
                    experts=layer_experts,
                    expert_domains=expert_domains,
                    top_k=top_k,
                    dtype=target_dtype,
                    shared_base_ffn=shared_ffn,
                ).to(work_dev)
            else:
                w0_gate = _dequantize_linear_weight(src_layer.mlp.gate_proj).to(work_dev)
                w0_up = _dequantize_linear_weight(src_layer.mlp.up_proj).to(work_dev)
                w0_down = _dequantize_linear_weight(src_layer.mlp.down_proj).to(work_dev)

                for domain in expert_domains:
                    b_state = branch_lora_states[domain]
                    ab_gate = _extract_ab_pair(b_state, layer_idx, "mlp.gate_proj")
                    ab_up = _extract_ab_pair(b_state, layer_idx, "mlp.up_proj")
                    ab_down = _extract_ab_pair(b_state, layer_idx, "mlp.down_proj")

                    w_gate = w0_gate + (
                        scaling * (ab_gate[1].to(work_dev) @ ab_gate[0].to(work_dev)) if ab_gate else 0.0
                    )
                    w_up = w0_up + (
                        scaling * (ab_up[1].to(work_dev) @ ab_up[0].to(work_dev)) if ab_up else 0.0
                    )
                    w_down = w0_down + (
                        scaling * (ab_down[1].to(work_dev) @ ab_down[0].to(work_dev)) if ab_down else 0.0
                    )

                    expert_mod = UpcycledSwiGLUExpert(
                        hidden_size=cfg.hidden_size,
                        intermediate_size=cfg.intermediate_size,
                        gate_weight=w_gate,
                        up_weight=w_up,
                        down_weight=w_down,
                        dtype=target_dtype,
                    ).to(work_dev)
                    layer_experts.append(expert_mod)
                    del w_gate, w_up, w_down
                    ffn_experts_created += 1

                del w0_gate, w0_up, w0_down
                dst_layer.mlp = SparseMoEMLPBlock(
                    hidden_size=cfg.hidden_size,
                    experts=layer_experts,
                    expert_domains=expert_domains,
                    top_k=top_k,
                    dtype=target_dtype,
                ).to(work_dev)

    moe_model = UpcycledSparseMoEModel(
        upcycled_backbone=merged_causal_lm.to(target_device),
        expert_domains=expert_domains,
        top_k=top_k,
        router_aux_loss_coef=router_aux_loss_coef,
    )

    num_moe_layers = len(moe_model.get_moe_blocks())
    single_ffn_params = num_moe_layers * 3 * int(cfg.hidden_size) * int(cfg.intermediate_size)
    raw_param_sum = sum(p.numel() for p in moe_model.parameters())
    if in_place:
        non_ffn_params = raw_param_sum - single_ffn_params
        total_moe_params = non_ffn_params + single_ffn_params * E
    else:
        total_moe_params = raw_param_sum
        non_ffn_params = total_moe_params - single_ffn_params * E
    active_params = non_ffn_params + single_ffn_params * min(top_k, E)

    report = MoEUpcycleReport(
        num_experts=E,
        top_k=min(top_k, E),
        expert_domains=expert_domains,
        attention_layers_merged=attn_merged_count,
        ffn_expert_modules_created=ffn_experts_created,
        total_moe_parameters=total_moe_params,
        active_parameters_per_token=active_params,
        attention_merge_mode=attention_merge_mode,
    )
    return moe_model, report


def train_moe_router_gating(
    moe_model: UpcycledSparseMoEModel,
    domain_token_batches: dict[str, torch.Tensor],
    steps: int = 40,
    lr: float = 3e-3,
) -> tuple[float, float, dict[str, float]]:
    """Train only the lightweight Top-2 gating network W_g while freezing upcycled experts."""
    device = next(moe_model.parameters()).device
    moe_model.train()
    moe_model.reset_routing_telemetry()

    router_params: list[nn.Parameter] = []
    for name, param in moe_model.named_parameters():
        if "router.wg" in name:
            param.requires_grad = True
            router_params.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(router_params, lr=lr, weight_decay=0.0)
    domains = list(domain_token_batches.keys())
    num_doms = max(1, len(domains))
    first_epoch_losses: list[float] = []
    last_epoch_losses: list[float] = []

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        dom_idx = (step - 1) % num_doms
        domain = domains[dom_idx]
        tokens = domain_token_batches[domain]
        batch = tokens[0:1].to(device, dtype=torch.long)
        out = moe_model(input_ids=batch, labels=batch)
        loss: torch.Tensor = out.loss
        loss.backward()
        optimizer.step()

        lval = float(loss.detach().cpu().item())
        if step <= num_doms:
            first_epoch_losses.append(lval)
        if step > max(0, steps - num_doms):
            last_epoch_losses.append(lval)

    for blk in moe_model.get_moe_blocks():
        blk.last_aux_loss = blk.last_aux_loss.detach()

    initial_loss = sum(first_epoch_losses) / max(1, len(first_epoch_losses))
    final_loss = sum(last_epoch_losses) / max(1, len(last_epoch_losses))
    return round(initial_loss, 6), round(final_loss, 6), moe_model.get_expert_utilization()

