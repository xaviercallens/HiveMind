# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Runtime thermodynamic dense-fallback guard for open-weight Sparse MoE neural control.

Implements inference-time and hot-swap safety primitives for upcycled open-weight
Sparse MoE models (``CrowdV1WithEngram``, ``CrowdLoRAMoEMLP``, and ``CrowdFineGrainedMoEMLP``):

1. ``ThermodynamicHotSwapGate``:
   Verifies the ANSE thermodynamic condition ``Delta E = E_child - E_parent < 0`` and the
   strict ``<= +1.0%`` worst-domain NLL regression guard before activating edited adapter/router
   buckets, executing atomic rollback to the verified parent or dense base state on any
   violation, non-finite tensor, shape mismatch, or evaluation failure.
2. ``ProseConfidenceFallbackController`` & ``AdaptiveSteeredMoELayer``:
   Computes per-token calibrated prose-vs-specialist confidence from a learned uncentered
   linear probe ``u`` (compatible with ``fit_uncentered_prose_probe_with_oof``) and smoothly
   damps the residual MoE delta norm whenever out-of-distribution general prose confidence
   exceeds threshold or the per-token ``||delta|| / ||base||`` norm ratio spikes abnormally,
   with zero-overhead bypass when disabled.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.core.fomoe_sigma_routing import sigmoid_independent_topk_routing
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.sigma_router_gate_control import fit_uncentered_prose_probe_with_oof

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_MAX_DOMAIN_DRIFT: float = 0.01
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.0 + DEFAULT_MAX_DOMAIN_DRIFT
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1e6
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05

DEFAULT_PROSE_CONFIDENCE_THRESHOLD: float = 0.55
DEFAULT_PROSE_TRANSITION_WIDTH: float = 0.30
DEFAULT_PROBE_MIDPOINT: float = 0.50
DEFAULT_PROBE_LOGIT_SCALE: float = 8.0
DEFAULT_MAX_DELTA_BASE_RATIO: float = 0.50
DEFAULT_SPIKE_DECAY: float = 1.0
DEFAULT_NORM_EPS: float = 1e-6


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a strictly positive finite real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str) -> float:
    """Validate that ``value`` lies in ``[0.0, 1.0]``."""
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return out


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a positive integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _validate_probe_direction(tensor: torch.Tensor, name: str = "probe_direction") -> torch.Tensor:
    """Validate a 1D non-empty finite tensor and return a detached CPU float32 copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D tensor, got shape {tuple(tensor.shape)}")
    out = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _switch_load_balancing_loss(router_probs: torch.Tensor, top_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Compute Switch Transformer auxiliary load-balancing loss ``E * sum_e (f_e * P_e)``."""
    tokens = router_probs.shape[0]
    if tokens == 0:
        return router_probs.sum() * 0.0
    one_hot = F.one_hot(top_indices, num_classes=num_experts).to(torch.float32)
    token_frac = one_hot.mean(dim=(0, 1))
    prob_mean = router_probs.mean(dim=0)
    return float(num_experts) * torch.sum(token_frac * prob_mean)


def compute_buckets_sha256(buckets: Mapping[str, torch.Tensor]) -> str:
    """Compute deterministic SHA-256 provenance digest over a bucket mapping."""
    if not isinstance(buckets, Mapping) or not buckets:
        raise ValueError("buckets must be a non-empty Mapping[str, torch.Tensor]")
    hasher = hashlib.sha256()
    for key in sorted(buckets.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"Bucket key must be a non-empty string, got {key!r}")
        tensor = buckets[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"buckets[{key!r}] must be a torch.Tensor, got {type(tensor).__name__}")
        t_cpu = tensor.detach().to(device="cpu").contiguous()
        header = f"{key}:{tuple(t_cpu.shape)}:{t_cpu.dtype}:".encode()
        hasher.update(header)
        hasher.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# 1. Thermodynamic Energy & Atomic Hot-Swap Gate
# ---------------------------------------------------------------------------


def compute_thermodynamic_energy(
    baseline_domain_nll: Mapping[str, float],
    candidate_domain_nll: Mapping[str, float],
    *,
    domain_weights: Mapping[str, float] | None = None,
    general_domain_prefix: str = GENERAL_EXPERT_PREFIX,
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT,
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO,
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN,
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY,
) -> dict[str, Any]:
    """Compute per-domain NLL drift, worst-domain guard verdict, and objective ANSE energy."""
    if not isinstance(baseline_domain_nll, Mapping) or not baseline_domain_nll:
        raise ValueError("baseline_domain_nll must be a non-empty mapping")
    if not isinstance(candidate_domain_nll, Mapping) or not candidate_domain_nll:
        raise ValueError("candidate_domain_nll must be a non-empty mapping")
    if set(baseline_domain_nll.keys()) != set(candidate_domain_nll.keys()):
        raise ValueError("baseline_domain_nll and candidate_domain_nll must have identical domain keys")

    max_drift = _validate_nonneg_finite(max_domain_drift, "max_domain_drift")
    safe_e0_ratio = _validate_positive_finite(anse_safe_general_ratio, "anse_safe_general_ratio")
    pain_val = _validate_nonneg_finite(maximum_pain, "maximum_pain")
    soft_val = _validate_nonneg_finite(soft_margin_penalty, "soft_margin_penalty")

    weights: dict[str, float] = {}
    if domain_weights is not None:
        if set(domain_weights.keys()) != set(baseline_domain_nll.keys()):
            raise ValueError("domain_weights must have identical domain keys")
        for dom, raw_w in domain_weights.items():
            weights[dom] = _validate_nonneg_finite(raw_w, f"domain_weights[{dom!r}]")
        weight_sum = sum(weights.values())
        if weight_sum <= 0.0:
            raise ValueError("domain_weights must have positive total weight")
    else:
        weight_sum = float(len(baseline_domain_nll))
        weights = {dom: 1.0 for dom in baseline_domain_nll}

    domain_ratios: dict[str, float] = {}
    domain_drifts: dict[str, float] = {}
    domain_drifts_pct: dict[str, float] = {}
    mean_base = 0.0
    mean_cand = 0.0
    worst_domain = ""
    worst_drift = -math.inf

    for dom in sorted(baseline_domain_nll.keys()):
        b_val = _validate_positive_finite(baseline_domain_nll[dom], f"baseline_domain_nll[{dom!r}]")
        c_val = _validate_positive_finite(candidate_domain_nll[dom], f"candidate_domain_nll[{dom!r}]")
        w = weights[dom] / weight_sum
        mean_base += w * b_val
        mean_cand += w * c_val
        ratio = c_val / b_val
        drift = ratio - 1.0
        domain_ratios[dom] = ratio
        domain_drifts[dom] = drift
        domain_drifts_pct[dom] = 100.0 * drift
        if drift > worst_drift:
            worst_drift = drift
            worst_domain = dom

    mean_diff = mean_cand - mean_base
    max_ratio = 1.0 + max_drift
    worst_ratio = domain_ratios[worst_domain]
    guard_ok = worst_ratio <= max_ratio + 1e-12

    general_keys = [dom for dom in sorted(domain_ratios.keys()) if dom.startswith(general_domain_prefix)]
    if not general_keys and "general" in domain_ratios:
        general_keys = ["general"]
    general_ratio = max(domain_ratios[k] for k in general_keys) if general_keys else 1.0

    penalty = pain_val if not guard_ok else (soft_val if general_ratio > safe_e0_ratio else 0.0)
    anse_energy = mean_diff + penalty
    return {
        "mean_base_nll": mean_base,
        "mean_cand_nll": mean_cand,
        "mean_diff_vs_base": mean_diff,
        "worst_domain": worst_domain,
        "worst_domain_drift": worst_drift,
        "worst_domain_drift_pct": 100.0 * worst_drift,
        "worst_domain_ratio": worst_ratio,
        "general_ratio": general_ratio,
        "guard_ok": guard_ok,
        "anse_energy": anse_energy,
        "domain_drifts": domain_drifts,
        "domain_drifts_pct": domain_drifts_pct,
    }


@dataclass(frozen=True)
class HotSwapGateConfig:
    """Configuration governing thermodynamic bucket acceptance and atomic rollback."""

    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT
    min_energy_drop: float = 0.0
    require_strict_energy_drop: bool = True
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    rollback_to_dense_on_violation: bool = False
    verify_rollback_hash: bool = True
    general_domain_prefix: str = GENERAL_EXPERT_PREFIX

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.max_domain_drift, "max_domain_drift")
        _validate_nonneg_finite(self.min_energy_drop, "min_energy_drop")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")
        _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        _validate_nonneg_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.rollback_to_dense_on_violation, "rollback_to_dense_on_violation")
        _validate_bool(self.verify_rollback_hash, "verify_rollback_hash")
        if not isinstance(self.general_domain_prefix, str) or not self.general_domain_prefix:
            raise ValueError("general_domain_prefix must be a non-empty string")


@dataclass(frozen=True)
class HotSwapDecisionReceipt:
    """Immutable audit receipt produced for every hot-swap gate decision or fallback."""

    accepted: bool
    rolled_back: bool
    rollback_target: str | None
    energy_parent: float
    energy_child: float
    delta_energy: float
    worst_domain: str | None
    worst_domain_drift: float
    worst_domain_drift_pct: float
    domain_guard_ok: bool
    thermodynamic_ok: bool
    active_state_sha256: str
    candidate_sha256: str
    reasons: tuple[str, ...]
    domain_drifts_pct: dict[str, float]
    receipt_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize receipt into a plain JSON-compatible dictionary."""
        return asdict(self)


def _seal_receipt(receipt: HotSwapDecisionReceipt) -> HotSwapDecisionReceipt:
    body = receipt.to_dict()
    body["receipt_sha256"] = ""
    raw = repr(sorted(body.items())).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return replace(receipt, receipt_sha256=digest)


def unwrap_moe_mlp(mlp: nn.Module) -> nn.Module:
    """Return the underlying MoE module when wrapped in ``AdaptiveSteeredMoELayer``."""
    if isinstance(mlp, AdaptiveSteeredMoELayer):
        return mlp.moe_mlp
    return mlp


def _clone_buckets(buckets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().to(device="cpu").clone() for k, v in buckets.items()}


def _infer_router_fragments(buckets: Mapping[str, torch.Tensor]) -> int:
    return max(1, sum(1 for k in buckets if k.startswith("router_frag_")))


def extract_guarded_moe_buckets(
    model: nn.Module,
    *,
    num_router_fragments: int = 4,
    only_trainable: bool = True,
) -> dict[str, torch.Tensor]:
    """Extract OpenDiLoCo MoE buckets across ``CrowdV1WithEngram`` even when layers are guarded."""
    nfrag = _validate_positive_int(num_router_fragments, "num_router_fragments")
    _validate_bool(only_trainable, "only_trainable")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

    n_blocks = len(blocks)
    frag_span = max(1, math.ceil(n_blocks / nfrag))
    router_parts: dict[str, list[torch.Tensor]] = {}
    expert_parts: dict[str, list[torch.Tensor]] = {}

    for l_idx, blk in enumerate(blocks):
        f_idx = min(nfrag - 1, l_idx // frag_span)
        r_key = f"router_frag_{f_idx}"
        mlp = unwrap_moe_mlp(blk.mlp)
        if isinstance(mlp, CrowdLoRAMoEMLP):
            if not only_trainable or mlp.router.weight.requires_grad:
                router_parts.setdefault(r_key, []).append(
                    mlp.router.weight.detach().to("cpu", dtype=torch.float32).reshape(-1)
                )
            for e_key, exp_mod in sorted(mlp.experts.items(), key=lambda kv: int(kv[0])):
                vecs = [
                    p.detach().to("cpu", dtype=torch.float32).reshape(-1)
                    for p in exp_mod.parameters()
                    if (not only_trainable or p.requires_grad)
                ]
                if vecs:
                    expert_parts.setdefault(f"expert_{int(e_key)}", []).extend(vecs)
        elif isinstance(mlp, CrowdFineGrainedMoEMLP):
            if not only_trainable or mlp.router.weight.requires_grad:
                router_parts.setdefault(r_key, []).append(
                    mlp.router.weight.detach().to("cpu", dtype=torch.float32).reshape(-1)
                )
            for e_idx in range(mlp.num_experts):
                vecs = [
                    p.detach().to("cpu", dtype=torch.float32).reshape(-1)
                    for mod_list in (mlp.experts_w1, mlp.experts_w2, mlp.experts_w3)
                    for p in mod_list[e_idx].parameters()
                    if (not only_trainable or p.requires_grad)
                ]
                if vecs:
                    expert_parts.setdefault(f"expert_{e_idx}", []).extend(vecs)

    last_r_key = f"router_frag_{nfrag - 1}"
    ln = getattr(model, "ln", None)
    if ln is not None:
        for p in (ln.weight, ln.bias):
            if not only_trainable or p.requires_grad:
                router_parts.setdefault(last_r_key, []).append(p.detach().to("cpu", dtype=torch.float32).reshape(-1))

    out: dict[str, torch.Tensor] = {}
    for k, parts in sorted(router_parts.items()):
        if parts:
            out[k] = torch.cat(parts, dim=0).contiguous()
    for k, parts in sorted(expert_parts.items(), key=lambda kv: int(kv[0].split("_")[1])):
        if parts:
            out[k] = torch.cat(parts, dim=0).contiguous()
    return out


@torch.no_grad()
def load_guarded_moe_buckets(
    model: nn.Module,
    buckets: Mapping[str, torch.Tensor],
    *,
    num_router_fragments: int | None = None,
    only_trainable: bool = True,
) -> None:
    """Load OpenDiLoCo MoE buckets into ``model`` safely even when layers are guarded."""
    if not isinstance(buckets, Mapping) or not buckets:
        raise ValueError("buckets must be a non-empty Mapping[str, torch.Tensor]")
    _validate_bool(only_trainable, "only_trainable")

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        state = model.state_dict()
        for key, val in buckets.items():
            if key in state:
                state[key].copy_(val.to(device=state[key].device, dtype=state[key].dtype))
        return

    nfrag = _infer_router_fragments(buckets) if num_router_fragments is None else _validate_positive_int(
        num_router_fragments, "num_router_fragments"
    )
    n_blocks = len(blocks)
    frag_span = max(1, math.ceil(n_blocks / nfrag))
    router_offsets: dict[str, int] = {k: 0 for k in buckets if k.startswith("router_frag_")}
    expert_offsets: dict[str, int] = {k: 0 for k in buckets if k.startswith("expert_")}

    def _copy_slice(param: torch.Tensor, bucket_key: str, offsets: dict[str, int]) -> None:
        n = param.numel()
        off = offsets[bucket_key]
        src = buckets[bucket_key]
        if off + n > src.numel():
            raise ValueError(f"Bucket {bucket_key!r} underflow: need {off + n} elements, have {src.numel()}")
        param.copy_(src[off : off + n].view_as(param).to(device=param.device, dtype=param.dtype))
        offsets[bucket_key] = off + n

    for l_idx, blk in enumerate(blocks):
        f_idx = min(nfrag - 1, l_idx // frag_span)
        r_key = f"router_frag_{f_idx}"
        mlp = unwrap_moe_mlp(blk.mlp)
        if isinstance(mlp, CrowdLoRAMoEMLP):
            if (not only_trainable or mlp.router.weight.requires_grad) and r_key in buckets:
                _copy_slice(mlp.router.weight, r_key, router_offsets)
            for e_key, exp_mod in sorted(mlp.experts.items(), key=lambda kv: int(kv[0])):
                b_name = f"expert_{int(e_key)}"
                if b_name not in buckets:
                    continue
                for p in exp_mod.parameters():
                    if not only_trainable or p.requires_grad:
                        _copy_slice(p, b_name, expert_offsets)
        elif isinstance(mlp, CrowdFineGrainedMoEMLP):
            if (not only_trainable or mlp.router.weight.requires_grad) and r_key in buckets:
                _copy_slice(mlp.router.weight, r_key, router_offsets)
            for e_idx in range(mlp.num_experts):
                b_name = f"expert_{e_idx}"
                if b_name not in buckets:
                    continue
                for mod_list in (mlp.experts_w1, mlp.experts_w2, mlp.experts_w3):
                    for p in mod_list[e_idx].parameters():
                        if not only_trainable or p.requires_grad:
                            _copy_slice(p, b_name, expert_offsets)

    last_r_key = f"router_frag_{nfrag - 1}"
    ln = getattr(model, "ln", None)
    if ln is not None and last_r_key in buckets:
        for p in (ln.weight, ln.bias):
            if not only_trainable or p.requires_grad:
                _copy_slice(p, last_r_key, router_offsets)


def extract_dense_fallback_buckets(
    model: nn.Module,
    *,
    num_router_fragments: int = 4,
    only_trainable: bool = True,
) -> dict[str, torch.Tensor] | None:
    """Build exact zero-delta LoRA ``B=0`` dense fallback buckets without mutating ``model``."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        state = model.state_dict()
        has_lora_b = any(k.endswith((".w1_B", ".w2_B", ".w3_B")) for k in state)
        if not has_lora_b:
            return None
        return {
            k: (torch.zeros_like(v, device="cpu") if k.endswith((".w1_B", ".w2_B", ".w3_B")) else v.detach().cpu().clone())
            for k, v in state.items()
        }

    if not any(isinstance(unwrap_moe_mlp(blk.mlp), CrowdLoRAMoEMLP) for blk in blocks):
        return None

    nfrag = _validate_positive_int(num_router_fragments, "num_router_fragments")
    base = extract_guarded_moe_buckets(model, num_router_fragments=nfrag, only_trainable=only_trainable)
    expert_parts: dict[str, list[torch.Tensor]] = {}
    for blk in blocks:
        mlp = unwrap_moe_mlp(blk.mlp)
        if not isinstance(mlp, CrowdLoRAMoEMLP):
            continue
        for e_key, exp_mod in sorted(mlp.experts.items(), key=lambda kv: int(kv[0])):
            vecs: list[torch.Tensor] = []
            for p_name, p in exp_mod.named_parameters():
                if only_trainable and not p.requires_grad:
                    continue
                if p_name in ("w1_B", "w2_B", "w3_B"):
                    vecs.append(torch.zeros(p.numel(), dtype=torch.float32, device="cpu"))
                else:
                    vecs.append(p.detach().to("cpu", dtype=torch.float32).reshape(-1))
            if vecs:
                expert_parts.setdefault(f"expert_{int(e_key)}", []).extend(vecs)

    out = {k: v for k, v in base.items() if not k.startswith("expert_")}
    for k, parts in sorted(expert_parts.items(), key=lambda kv: int(kv[0].split("_")[1])):
        out[k] = torch.cat(parts, dim=0).contiguous()
    return out


class ThermodynamicHotSwapGate:
    """Thermodynamic hot-swap gate enforcing ``Delta E < 0`` and ``<= +1.0%`` domain drift."""

    def __init__(
        self,
        config: HotSwapGateConfig | None = None,
        *,
        parent_buckets: Mapping[str, torch.Tensor] | None = None,
        dense_buckets: Mapping[str, torch.Tensor] | None = None,
        parent_energy: float | None = None,
        baseline_domain_nll: Mapping[str, float] | None = None,
    ) -> None:
        self.config: HotSwapGateConfig = config or HotSwapGateConfig()
        self._parent_buckets: dict[str, torch.Tensor] | None = (
            _clone_buckets(parent_buckets) if parent_buckets is not None else None
        )
        self._dense_buckets: dict[str, torch.Tensor] | None = (
            _clone_buckets(dense_buckets) if dense_buckets is not None else None
        )
        self._parent_energy: float | None = (
            _validate_finite_float(parent_energy, "parent_energy") if parent_energy is not None else None
        )
        self._baseline_domain_nll: dict[str, float] | None = None
        if baseline_domain_nll is not None:
            self.set_baseline_domain_nll(baseline_domain_nll)
        self.swap_history: list[HotSwapDecisionReceipt] = []
        self.accepted_count: int = 0
        self.rejected_count: int = 0
        self.rollback_count: int = 0

    @property
    def active_buckets(self) -> dict[str, torch.Tensor] | None:
        """Return a cloned copy of the currently active verified parent buckets."""
        return _clone_buckets(self._parent_buckets) if self._parent_buckets is not None else None

    @property
    def dense_buckets(self) -> dict[str, torch.Tensor] | None:
        """Return a cloned copy of the registered dense fallback buckets, if present."""
        return _clone_buckets(self._dense_buckets) if self._dense_buckets is not None else None

    @property
    def parent_energy(self) -> float | None:
        """Return the incumbent thermodynamic energy ``E_parent``."""
        return self._parent_energy

    @property
    def baseline_domain_nll(self) -> dict[str, float] | None:
        """Return a shallow copy of baseline per-domain NLL values."""
        return dict(self._baseline_domain_nll) if self._baseline_domain_nll is not None else None

    def set_baseline_domain_nll(self, baseline_domain_nll: Mapping[str, float]) -> None:
        """Register validated positive finite baseline per-domain NLLs."""
        if not isinstance(baseline_domain_nll, Mapping) or not baseline_domain_nll:
            raise ValueError("baseline_domain_nll must be a non-empty mapping")
        self._baseline_domain_nll = {
            str(dom): _validate_positive_finite(val, f"baseline_domain_nll[{dom!r}]")
            for dom, val in baseline_domain_nll.items()
        }

    def snapshot_model_state(
        self,
        model: nn.Module,
        *,
        extract_buckets_fn: Callable[[nn.Module], Mapping[str, torch.Tensor]] | None = None,
        num_router_fragments: int = 4,
    ) -> dict[str, torch.Tensor]:
        """Extract a detached CPU snapshot of the active MoE buckets from ``model``."""
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        if extract_buckets_fn is not None:
            return _clone_buckets(extract_buckets_fn(model))
        return extract_guarded_moe_buckets(model, num_router_fragments=num_router_fragments)

    def _load_model_state(
        self,
        model: nn.Module,
        buckets: Mapping[str, torch.Tensor],
        *,
        load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    ) -> None:
        if load_buckets_fn is not None:
            load_buckets_fn(model, buckets)
        else:
            load_guarded_moe_buckets(model, buckets)

    def _resolve_domain_drifts_and_energy(
        self,
        raw_eval: Any,
        *,
        baseline_domain_nll: Mapping[str, float] | None,
    ) -> tuple[float, dict[str, float], str | None, float]:
        """Normalize evaluator outputs into ``(energy, domain_drifts, worst_domain, worst_drift)``."""
        base_map = baseline_domain_nll if baseline_domain_nll is not None else self._baseline_domain_nll
        explicit_energy: float | None = None
        domain_nll: Mapping[str, float] | None = None
        domain_drifts_raw: Mapping[str, float] | None = None

        if isinstance(raw_eval, tuple) and len(raw_eval) == 2:
            explicit_energy = _validate_finite_float(raw_eval[0], "eval_energy")
            if isinstance(raw_eval[1], Mapping):
                domain_nll = raw_eval[1]
        elif isinstance(raw_eval, Mapping):
            for key in ("anse_energy", "energy", "E"):
                if key in raw_eval and raw_eval[key] is not None:
                    explicit_energy = _validate_finite_float(raw_eval[key], f"eval[{key!r}]")
                    break
            for key in ("domain_nll", "per_domain_nll"):
                if key in raw_eval and isinstance(raw_eval[key], Mapping):
                    domain_nll = raw_eval[key]
                    break
            if "domain_drifts" in raw_eval and isinstance(raw_eval["domain_drifts"], Mapping):
                domain_drifts_raw = raw_eval["domain_drifts"]
        else:
            raise TypeError(
                "eval_fn must return a mapping with 'domain_nll'/'domain_drifts' or a (energy, domain_nll) tuple"
            )

        if domain_nll is not None:
            if base_map is None:
                raise ValueError("baseline_domain_nll must be set when eval_fn returns 'domain_nll'")
            thermo = compute_thermodynamic_energy(
                base_map,
                domain_nll,
                general_domain_prefix=self.config.general_domain_prefix,
                max_domain_drift=self.config.max_domain_drift,
                anse_safe_general_ratio=self.config.anse_safe_general_ratio,
                maximum_pain=self.config.maximum_pain_energy,
                soft_margin_penalty=self.config.soft_margin_penalty,
            )
            energy_val = explicit_energy if explicit_energy is not None else float(thermo["anse_energy"])
            return (
                energy_val,
                dict(thermo["domain_drifts"]),
                str(thermo["worst_domain"]),
                float(thermo["worst_domain_drift"]),
            )

        if domain_drifts_raw is not None and domain_drifts_raw:
            if explicit_energy is None:
                raise ValueError("eval_fn returning 'domain_drifts' must also supply 'energy' or 'anse_energy'")
            drifts = {str(k): _validate_finite_float(v, f"domain_drifts[{k!r}]") for k, v in domain_drifts_raw.items()}
            worst_dom = max(drifts.keys(), key=lambda d: drifts[d])
            return explicit_energy, drifts, worst_dom, drifts[worst_dom]

        raise ValueError("eval_fn output must contain per-domain NLLs or domain_drifts to verify <= +1.0% guard")

    def evaluate_candidate_metrics(
        self,
        energy_parent: float,
        energy_child: float,
        *,
        domain_nll_child: Mapping[str, float] | None = None,
        domain_nll_baseline: Mapping[str, float] | None = None,
        domain_drifts: Mapping[str, float] | None = None,
        active_state_sha256: str = "",
        candidate_sha256: str = "",
    ) -> HotSwapDecisionReceipt:
        """Pure thermodynamic decision verifying ``Delta E < 0`` and worst-domain drift ``<= max_domain_drift``."""
        e_p = _validate_finite_float(energy_parent, "energy_parent")
        e_c = _validate_finite_float(energy_child, "energy_child")
        delta_e = e_c - e_p

        if domain_nll_child is not None:
            base_map = domain_nll_baseline if domain_nll_baseline is not None else self._baseline_domain_nll
            if base_map is None:
                raise ValueError("domain_nll_baseline must be provided when domain_nll_child is passed")
            thermo = compute_thermodynamic_energy(
                base_map,
                domain_nll_child,
                general_domain_prefix=self.config.general_domain_prefix,
                max_domain_drift=self.config.max_domain_drift,
                anse_safe_general_ratio=self.config.anse_safe_general_ratio,
                maximum_pain=self.config.maximum_pain_energy,
                soft_margin_penalty=self.config.soft_margin_penalty,
            )
            drifts = dict(thermo["domain_drifts"])
            worst_dom: str | None = str(thermo["worst_domain"])
            worst_drift = float(thermo["worst_domain_drift"])
        elif domain_drifts is not None and domain_drifts:
            drifts = {str(k): _validate_finite_float(v, f"domain_drifts[{k!r}]") for k, v in domain_drifts.items()}
            worst_dom = max(drifts.keys(), key=lambda d: drifts[d])
            worst_drift = drifts[worst_dom]
        else:
            raise ValueError("Either domain_nll_child or non-empty domain_drifts must be provided")

        domain_guard_ok = worst_drift <= self.config.max_domain_drift + 1e-12
        if self.config.require_strict_energy_drop:
            thermo_ok = delta_e < -self.config.min_energy_drop
        else:
            thermo_ok = delta_e <= -self.config.min_energy_drop

        reasons: list[str] = []
        if not thermo_ok:
            reasons.append(
                f"THERMODYNAMIC_VIOLATION: delta_E={delta_e:+.6f} >= required {-self.config.min_energy_drop:+.6f}"
            )
        if not domain_guard_ok:
            reasons.append(
                f"DOMAIN_DRIFT_VIOLATION: {worst_dom} drifted {100.0 * worst_drift:+.3f}% "
                f"(limit {100.0 * self.config.max_domain_drift:+.3f}%)"
            )
        accepted = thermo_ok and domain_guard_ok
        if accepted:
            reasons.append("ACCEPTED: Delta E < 0 and worst-domain drift within guard")

        return _seal_receipt(
            HotSwapDecisionReceipt(
                accepted=accepted,
                rolled_back=False,
                rollback_target=None,
                energy_parent=e_p,
                energy_child=e_c,
                delta_energy=delta_e,
                worst_domain=worst_dom,
                worst_domain_drift=worst_drift,
                worst_domain_drift_pct=100.0 * worst_drift,
                domain_guard_ok=domain_guard_ok,
                thermodynamic_ok=thermo_ok,
                active_state_sha256=active_state_sha256,
                candidate_sha256=candidate_sha256,
                reasons=tuple(reasons),
                domain_drifts_pct={k: round(100.0 * v, 4) for k, v in drifts.items()},
            )
        )

    @staticmethod
    def _validate_candidate_buckets(
        child_buckets: Mapping[str, torch.Tensor],
        parent_snapshot: Mapping[str, torch.Tensor],
    ) -> list[str]:
        """Validate candidate bucket keys, shapes, and tensor finiteness prior to activation."""
        errors: list[str] = []
        if set(child_buckets.keys()) != set(parent_snapshot.keys()):
            missing = sorted(set(parent_snapshot.keys()) - set(child_buckets.keys()))
            extra = sorted(set(child_buckets.keys()) - set(parent_snapshot.keys()))
            errors.append(f"BUCKET_KEY_MISMATCH: missing={missing[:3]}, extra={extra[:3]}")
            return errors
        for key in sorted(parent_snapshot.keys()):
            c_ten = child_buckets[key]
            p_ten = parent_snapshot[key]
            if not isinstance(c_ten, torch.Tensor):
                errors.append(f"INVALID_BUCKET_TYPE: {key!r} is {type(c_ten).__name__}")
                break
            if tuple(c_ten.shape) != tuple(p_ten.shape):
                errors.append(f"BUCKET_SHAPE_MISMATCH: {key!r} {tuple(c_ten.shape)} != {tuple(p_ten.shape)}")
                break
            if not bool(torch.isfinite(c_ten).all().item()):
                errors.append(f"NON_FINITE_CANDIDATE_WEIGHTS: {key!r} contains NaN or Inf")
                break
        return errors

    def _perform_atomic_rollback(
        self,
        model: nn.Module,
        parent_snapshot: Mapping[str, torch.Tensor],
        *,
        prefer_dense: bool,
        load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None,
        extract_buckets_fn: Callable[[nn.Module], Mapping[str, torch.Tensor]] | None,
        num_router_fragments: int,
    ) -> tuple[str, str]:
        """Atomically restore either dense fallback buckets or parent buckets and verify hash."""
        target_name = "parent"
        target_buckets: Mapping[str, torch.Tensor] = parent_snapshot
        if prefer_dense:
            if self._dense_buckets is not None:
                target_buckets = self._dense_buckets
                target_name = "dense"
            else:
                inferred_dense = extract_dense_fallback_buckets(model, num_router_fragments=num_router_fragments)
                if inferred_dense is not None:
                    target_buckets = inferred_dense
                    target_name = "dense"

        self._load_model_state(model, target_buckets, load_buckets_fn=load_buckets_fn)
        if self.config.verify_rollback_hash:
            restored = self.snapshot_model_state(
                model,
                extract_buckets_fn=extract_buckets_fn,
                num_router_fragments=num_router_fragments,
            )
            aligned_target = {
                k: (v.to(dtype=restored[k].dtype) if k in restored else v)
                for k, v in target_buckets.items()
            }
            expected_sha = compute_buckets_sha256(aligned_target)
            restored_sha = compute_buckets_sha256(restored)
            if restored_sha != expected_sha:
                raise RuntimeError(
                    f"Atomic rollback verification failed for target={target_name}: "
                    f"{restored_sha[:12]} != {expected_sha[:12]}"
                )
            return target_name, restored_sha
        return target_name, compute_buckets_sha256(target_buckets)

    def _reject_and_rollback(
        self,
        model: nn.Module,
        parent_snapshot: Mapping[str, torch.Tensor],
        *,
        e_parent: float,
        candidate_sha256: str,
        reason: str | Sequence[str],
        prefer_dense: bool,
        load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None,
        extract_buckets_fn: Callable[[nn.Module], Mapping[str, torch.Tensor]] | None,
        num_router_fragments: int,
    ) -> HotSwapDecisionReceipt:
        """Atomically roll back, record counters, and return a sealed rejection receipt."""
        target_name, active_sha = self._perform_atomic_rollback(
            model,
            parent_snapshot,
            prefer_dense=prefer_dense,
            load_buckets_fn=load_buckets_fn,
            extract_buckets_fn=extract_buckets_fn,
            num_router_fragments=num_router_fragments,
        )
        self.rejected_count += 1
        self.rollback_count += 1
        reasons_tuple = (reason,) if isinstance(reason, str) else tuple(reason)
        receipt = _seal_receipt(
            HotSwapDecisionReceipt(
                accepted=False,
                rolled_back=True,
                rollback_target=target_name,
                energy_parent=e_parent,
                energy_child=self.config.maximum_pain_energy,
                delta_energy=self.config.maximum_pain_energy - e_parent,
                worst_domain=None,
                worst_domain_drift=math.inf,
                worst_domain_drift_pct=math.inf,
                domain_guard_ok=False,
                thermodynamic_ok=False,
                active_state_sha256=active_sha,
                candidate_sha256=candidate_sha256,
                reasons=reasons_tuple,
                domain_drifts_pct={},
            )
        )
        self.swap_history.append(receipt)
        return receipt

    @torch.no_grad()
    def try_hot_swap(
        self,
        model: nn.Module,
        child_buckets: Mapping[str, torch.Tensor],
        *,
        eval_fn: Callable[[nn.Module], Any],
        baseline_domain_nll: Mapping[str, float] | None = None,
        load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
        extract_buckets_fn: Callable[[nn.Module], Mapping[str, torch.Tensor]] | None = None,
        rollback_to_dense: bool | None = None,
    ) -> HotSwapDecisionReceipt:
        """Tentatively evaluate ``child_buckets`` on ``model`` and commit iff thermodynamic guards pass."""
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        if not isinstance(child_buckets, Mapping) or not child_buckets:
            raise ValueError("child_buckets must be a non-empty Mapping[str, torch.Tensor]")
        if not callable(eval_fn):
            raise TypeError("eval_fn must be callable")
        if baseline_domain_nll is not None and self._baseline_domain_nll is None:
            self.set_baseline_domain_nll(baseline_domain_nll)

        nfrag = _infer_router_fragments(child_buckets)
        parent_snapshot = self.snapshot_model_state(
            model,
            extract_buckets_fn=extract_buckets_fn,
            num_router_fragments=nfrag,
        )
        if self._parent_buckets is None:
            self._parent_buckets = _clone_buckets(parent_snapshot)

        prefer_dense = self.config.rollback_to_dense_on_violation if rollback_to_dense is None else _validate_bool(
            rollback_to_dense, "rollback_to_dense"
        )

        # Ensure parent energy is known before mutating the model
        if self._parent_energy is None:
            parent_eval = eval_fn(model)
            e_parent, _, _, _ = self._resolve_domain_drifts_and_energy(
                parent_eval,
                baseline_domain_nll=baseline_domain_nll,
            )
            self._parent_energy = e_parent
        else:
            e_parent = self._parent_energy

        val_errors = self._validate_candidate_buckets(child_buckets, parent_snapshot)
        if val_errors:
            return self._reject_and_rollback(
                model,
                parent_snapshot,
                e_parent=e_parent,
                candidate_sha256="",
                reason=val_errors,
                prefer_dense=prefer_dense,
                load_buckets_fn=load_buckets_fn,
                extract_buckets_fn=extract_buckets_fn,
                num_router_fragments=nfrag,
            )

        child_sha = compute_buckets_sha256(child_buckets)
        try:
            self._load_model_state(model, child_buckets, load_buckets_fn=load_buckets_fn)
            child_eval = eval_fn(model)
            e_child, drifts, worst_dom, worst_drift = self._resolve_domain_drifts_and_energy(
                child_eval,
                baseline_domain_nll=baseline_domain_nll,
            )
        except Exception as exc:
            return self._reject_and_rollback(
                model,
                parent_snapshot,
                e_parent=e_parent,
                candidate_sha256=child_sha,
                reason=f"CANDIDATE_EVAL_FAILURE: {type(exc).__name__}: {exc}",
                prefer_dense=prefer_dense,
                load_buckets_fn=load_buckets_fn,
                extract_buckets_fn=extract_buckets_fn,
                num_router_fragments=nfrag,
            )

        decision = self.evaluate_candidate_metrics(
            e_parent,
            e_child,
            domain_drifts=drifts,
            active_state_sha256=child_sha,
            candidate_sha256=child_sha,
        )
        if decision.accepted:
            self._parent_buckets = _clone_buckets(child_buckets)
            self._parent_energy = e_child
            self.accepted_count += 1
            self.swap_history.append(decision)
            return decision

        target_name, active_sha = self._perform_atomic_rollback(
            model,
            parent_snapshot,
            prefer_dense=prefer_dense,
            load_buckets_fn=load_buckets_fn,
            extract_buckets_fn=extract_buckets_fn,
            num_router_fragments=nfrag,
        )
        self.rejected_count += 1
        self.rollback_count += 1
        rolled_receipt = _seal_receipt(
            replace(
                decision,
                rolled_back=True,
                rollback_target=target_name,
                active_state_sha256=active_sha,
                worst_domain=worst_dom,
                worst_domain_drift=worst_drift,
                worst_domain_drift_pct=100.0 * worst_drift,
            )
        )
        self.swap_history.append(rolled_receipt)
        return rolled_receipt

    @torch.no_grad()
    def force_dense_fallback(
        self,
        model: nn.Module,
        *,
        load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
        extract_buckets_fn: Callable[[nn.Module], Mapping[str, torch.Tensor]] | None = None,
        num_router_fragments: int = 4,
        reason: str = "MANUAL_DENSE_FALLBACK",
    ) -> HotSwapDecisionReceipt:
        """Atomically revert ``model`` to its dense fallback state."""
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        dense_target = (
            self._dense_buckets
            if self._dense_buckets is not None
            else extract_dense_fallback_buckets(model, num_router_fragments=num_router_fragments)
        )
        if dense_target is None:
            raise ValueError("No dense_buckets registered and model has no CrowdLoRAMoEMLP layers to zero")

        nfrag = _infer_router_fragments(dense_target)
        self._load_model_state(model, dense_target, load_buckets_fn=load_buckets_fn)
        active_sha = compute_buckets_sha256(
            self.snapshot_model_state(model, extract_buckets_fn=extract_buckets_fn, num_router_fragments=nfrag)
        )
        self._parent_buckets = _clone_buckets(dense_target)
        self.rollback_count += 1
        e_prev = 0.0 if self._parent_energy is None else self._parent_energy
        receipt = _seal_receipt(
            HotSwapDecisionReceipt(
                accepted=False,
                rolled_back=True,
                rollback_target="dense",
                energy_parent=e_prev,
                energy_child=0.0,
                delta_energy=-e_prev,
                worst_domain=None,
                worst_domain_drift=0.0,
                worst_domain_drift_pct=0.0,
                domain_guard_ok=True,
                thermodynamic_ok=False,
                active_state_sha256=active_sha,
                candidate_sha256=active_sha,
                reasons=(reason,),
                domain_drifts_pct={},
            )
        )
        self.swap_history.append(receipt)
        return receipt


# ---------------------------------------------------------------------------
# 2. Per-Token Prose-Confidence & Norm-Ratio Smooth Fallback Controller
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FallbackControllerConfig:
    """Configuration for per-token prose-confidence and delta/base norm-ratio damping."""

    enabled: bool = True
    prose_confidence_threshold: float = DEFAULT_PROSE_CONFIDENCE_THRESHOLD
    prose_transition_width: float = DEFAULT_PROSE_TRANSITION_WIDTH
    probe_midpoint: float = DEFAULT_PROBE_MIDPOINT
    probe_logit_scale: float = DEFAULT_PROBE_LOGIT_SCALE
    rel_weight: float = 1.0
    max_delta_base_ratio: float = DEFAULT_MAX_DELTA_BASE_RATIO
    spike_decay: float = DEFAULT_SPIKE_DECAY
    min_delta_scale: float = 0.0
    norm_eps: float = DEFAULT_NORM_EPS

    def __post_init__(self) -> None:
        _validate_bool(self.enabled, "enabled")
        thresh = _validate_unit_interval(self.prose_confidence_threshold, "prose_confidence_threshold")
        if thresh >= 1.0:
            raise ValueError(f"prose_confidence_threshold must be < 1.0, got {thresh}")
        _validate_positive_finite(self.prose_transition_width, "prose_transition_width")
        _validate_finite_float(self.probe_midpoint, "probe_midpoint")
        _validate_positive_finite(self.probe_logit_scale, "probe_logit_scale")
        _validate_unit_interval(self.rel_weight, "rel_weight")
        _validate_positive_finite(self.max_delta_base_ratio, "max_delta_base_ratio")
        _validate_nonneg_finite(self.spike_decay, "spike_decay")
        _validate_unit_interval(self.min_delta_scale, "min_delta_scale")
        _validate_positive_finite(self.norm_eps, "norm_eps")


@dataclass(frozen=True)
class FallbackTelemetry:
    """Per-forward telemetry snapshot summarizing damping triggers and scaling."""

    bypassed: bool
    num_tokens: int
    mean_prose_confidence: float
    max_prose_confidence: float
    prose_trigger_fraction: float
    mean_delta_base_ratio: float
    max_delta_base_ratio: float
    post_damping_max_ratio: float
    ratio_trigger_fraction: float
    mean_gamma: float
    min_gamma: float
    damped_token_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize telemetry into a plain JSON-compatible dictionary."""
        return asdict(self)


def _calibrate_probe_logistic_params(
    probe_dir: torch.Tensor,
    prose_x: torch.Tensor,
    spec_x: torch.Tensor,
    *,
    rel_weight: float,
    min_scale: float = 4.0,
    max_scale: float = 16.0,
) -> tuple[float, float]:
    """Calibrate ``(midpoint, logit_scale)`` from uncentered probe projections."""
    if float(probe_dir.norm().item()) == 0.0 or rel_weight == 0.0:
        return DEFAULT_PROBE_MIDPOINT, DEFAULT_PROBE_LOGIT_SCALE
    u32 = probe_dir.to(device=prose_x.device, dtype=torch.float32)
    p_proj = prose_x.to(torch.float32) @ u32
    s_proj = spec_x.to(device=prose_x.device, dtype=torch.float32) @ u32
    mu_p = float(p_proj.mean().item())
    mu_s = float(s_proj.mean().item())
    if mu_p <= mu_s:
        return DEFAULT_PROBE_MIDPOINT, DEFAULT_PROBE_LOGIT_SCALE
    midpoint = 0.5 * (mu_p + mu_s)
    pooled_var = 0.5 * (float(p_proj.var(unbiased=False).item()) + float(s_proj.var(unbiased=False).item()))
    raw_scale = (mu_p - mu_s) / max(pooled_var, 1e-4)
    logit_scale = max(min_scale, min(max_scale, raw_scale))
    return midpoint, logit_scale


class ProseConfidenceFallbackController(nn.Module):
    """Calibrated per-token OOD prose-confidence and norm-ratio damping controller."""

    probe_direction: torch.Tensor | None

    def __init__(
        self,
        probe_direction: torch.Tensor | None = None,
        config: FallbackControllerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config: FallbackControllerConfig = config or FallbackControllerConfig()
        if probe_direction is not None:
            validated = _validate_probe_direction(probe_direction, "probe_direction")
            self.register_buffer("probe_direction", validated, persistent=True)
        else:
            self.register_buffer("probe_direction", None, persistent=True)
        self.last_telemetry: FallbackTelemetry | None = None

    @classmethod
    def from_probe_folds(
        cls,
        prose_fold0: torch.Tensor,
        prose_fold1: torch.Tensor,
        spec_fold0: torch.Tensor,
        spec_fold1: torch.Tensor,
        *,
        omega_spec: float = 6.0,
        ridge_scale: float = 0.75,
        config: FallbackControllerConfig | None = None,
    ) -> ProseConfidenceFallbackController:
        """Fit an uncentered OOF prose probe and calibrate confidence parameters automatically."""
        u_probe, stats = fit_uncentered_prose_probe_with_oof(
            prose_fold0,
            prose_fold1,
            spec_fold0,
            spec_fold1,
            omega_spec=omega_spec,
            ridge_scale=ridge_scale,
        )
        base_cfg = config or FallbackControllerConfig()
        rel_w = float(stats["rel_weight"])
        prose_all = torch.cat([prose_fold0.detach(), prose_fold1.detach()], dim=0)
        spec_all = torch.cat([spec_fold0.detach(), spec_fold1.detach()], dim=0)
        u_f32 = u_probe.to(dtype=torch.float32)
        midpoint, logit_scale = _calibrate_probe_logistic_params(u_f32, prose_all, spec_all, rel_weight=rel_w)
        cal_cfg = replace(
            base_cfg,
            probe_midpoint=midpoint,
            probe_logit_scale=logit_scale,
            rel_weight=rel_w,
        )
        return cls(probe_direction=u_f32, config=cal_cfg)

    def set_enabled(self, enabled: bool) -> None:
        """Toggle controller activation while preserving probe and calibration state."""
        self.config = replace(self.config, enabled=_validate_bool(enabled, "enabled"))

    def update_probe(
        self,
        probe_direction: torch.Tensor,
        *,
        probe_midpoint: float | None = None,
        probe_logit_scale: float | None = None,
        rel_weight: float | None = None,
    ) -> None:
        """Replace the uncentered probe direction and optional calibration parameters."""
        validated = _validate_probe_direction(probe_direction, "probe_direction")
        target_dev = self.probe_direction.device if self.probe_direction is not None else torch.device("cpu")
        self.probe_direction = validated.to(device=target_dev)
        self.config = replace(
            self.config,
            probe_midpoint=self.config.probe_midpoint if probe_midpoint is None else _validate_finite_float(
                probe_midpoint, "probe_midpoint"
            ),
            probe_logit_scale=self.config.probe_logit_scale if probe_logit_scale is None else _validate_positive_finite(
                probe_logit_scale, "probe_logit_scale"
            ),
            rel_weight=self.config.rel_weight if rel_weight is None else _validate_unit_interval(
                rel_weight, "rel_weight"
            ),
        )

    @torch.no_grad()
    def compute_prose_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-token ``(linear_probe_projection, calibrated_prose_confidence)``."""
        if not isinstance(x, torch.Tensor) or x.ndim < 1:
            raise ValueError("x must be a torch.Tensor with ndim >= 1")
        out_shape = x.shape[:-1]
        if self.probe_direction is None or self.config.rel_weight == 0.0:
            zeros = torch.zeros(out_shape, device=x.device, dtype=torch.float32)
            return zeros, zeros

        feat_dim = int(x.shape[-1])
        if int(self.probe_direction.numel()) != feat_dim:
            raise ValueError(
                f"x feature width {feat_dim} does not match probe_direction length {self.probe_direction.numel()}"
            )
        u_vec = self.probe_direction.to(device=x.device, dtype=torch.float32)
        if float(u_vec.norm().item()) == 0.0:
            zeros = torch.zeros(out_shape, device=x.device, dtype=torch.float32)
            return zeros, zeros

        proj = torch.matmul(x.to(dtype=torch.float32), u_vec)
        raw_prob = torch.sigmoid(self.config.probe_logit_scale * (proj - self.config.probe_midpoint))
        conf = (self.config.rel_weight * raw_prob).clamp(0.0, 1.0)
        return proj, conf

    @torch.no_grad()
    def compute_damping_factors(
        self,
        x: torch.Tensor,
        base_out: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, FallbackTelemetry]:
        """Compute smooth per-token residual scale ``gamma in [0, 1]`` and sanitized delta."""
        if not isinstance(x, torch.Tensor) or not isinstance(base_out, torch.Tensor) or not isinstance(delta, torch.Tensor):
            raise TypeError("x, base_out, and delta must be torch.Tensor instances")
        if x.ndim < 1 or base_out.shape != delta.shape or x.shape[:-1] != delta.shape[:-1]:
            raise ValueError(
                f"Incompatible tensor shapes: x={tuple(x.shape)}, base_out={tuple(base_out.shape)}, delta={tuple(delta.shape)}"
            )

        lead_shape = x.shape[:-1]
        n_tok = int(math.prod(lead_shape)) if lead_shape else 1
        x_flat = x.reshape(n_tok, x.shape[-1])
        base_flat = base_out.reshape(n_tok, base_out.shape[-1]).to(torch.float32)
        delta_flat = delta.reshape(n_tok, delta.shape[-1]).to(torch.float32)

        if n_tok == 0:
            empty_g = torch.empty(lead_shape, device=x.device, dtype=torch.float32)
            telemetry = FallbackTelemetry(
                bypassed=False,
                num_tokens=0,
                mean_prose_confidence=0.0,
                max_prose_confidence=0.0,
                prose_trigger_fraction=0.0,
                mean_delta_base_ratio=0.0,
                max_delta_base_ratio=0.0,
                post_damping_max_ratio=0.0,
                ratio_trigger_fraction=0.0,
                mean_gamma=1.0,
                min_gamma=1.0,
                damped_token_fraction=0.0,
            )
            return empty_g, delta.to(torch.float32), telemetry

        _, conf = self.compute_prose_confidence(x_flat)
        t_prose = ((conf - self.config.prose_confidence_threshold) / self.config.prose_transition_width).clamp(
            0.0, 1.0
        )
        smooth_step = t_prose * t_prose * (3.0 - 2.0 * t_prose)
        gamma_prose = 1.0 - (1.0 - self.config.min_delta_scale) * smooth_step

        nonfinite_delta = ~torch.isfinite(delta_flat).all(dim=-1)
        safe_delta_flat = torch.where(nonfinite_delta.unsqueeze(-1), torch.zeros_like(delta_flat), delta_flat)

        base_norm = base_flat.norm(dim=-1).clamp_min(self.config.norm_eps)
        delta_norm = safe_delta_flat.norm(dim=-1)
        norm_ratio = delta_norm / base_norm

        rho_max = self.config.max_delta_base_ratio
        excess_ratio = ((norm_ratio - rho_max) / max(rho_max, 1e-8)).clamp_min(0.0)
        raw_ratio_gamma = (rho_max / norm_ratio.clamp_min(1e-12)) / (1.0 + self.config.spike_decay * excess_ratio)
        ratio_exceeded = norm_ratio > rho_max
        gamma_ratio = torch.where(ratio_exceeded, raw_ratio_gamma.clamp(0.0, 1.0), torch.ones_like(norm_ratio))

        gamma = (gamma_prose * gamma_ratio).clamp(0.0, 1.0)
        gamma = torch.where(nonfinite_delta, torch.zeros_like(gamma), gamma)

        post_ratio = (gamma * delta_norm) / base_norm
        prose_triggered = conf > self.config.prose_confidence_threshold
        damped_mask = (gamma < (1.0 - 1e-6)) | nonfinite_delta

        telemetry = FallbackTelemetry(
            bypassed=False,
            num_tokens=n_tok,
            mean_prose_confidence=float(conf.mean().item()),
            max_prose_confidence=float(conf.max().item()),
            prose_trigger_fraction=float(prose_triggered.to(torch.float32).mean().item()),
            mean_delta_base_ratio=float(norm_ratio.mean().item()),
            max_delta_base_ratio=float(norm_ratio.max().item()),
            post_damping_max_ratio=float(post_ratio.max().item()),
            ratio_trigger_fraction=float((ratio_exceeded | nonfinite_delta).to(torch.float32).mean().item()),
            mean_gamma=float(gamma.mean().item()),
            min_gamma=float(gamma.min().item()),
            damped_token_fraction=float(damped_mask.to(torch.float32).mean().item()),
        )
        return gamma.reshape(lead_shape), safe_delta_flat.reshape(delta.shape), telemetry

    @staticmethod
    def _make_bypass_telemetry(x: torch.Tensor) -> FallbackTelemetry:
        n_tok = int(math.prod(x.shape[:-1])) if x.ndim > 1 else (1 if x.ndim == 1 else 0)
        return FallbackTelemetry(
            bypassed=True,
            num_tokens=n_tok,
            mean_prose_confidence=0.0,
            max_prose_confidence=0.0,
            prose_trigger_fraction=0.0,
            mean_delta_base_ratio=0.0,
            max_delta_base_ratio=0.0,
            post_damping_max_ratio=0.0,
            ratio_trigger_fraction=0.0,
            mean_gamma=1.0,
            min_gamma=1.0,
            damped_token_fraction=0.0,
        )

    @torch.no_grad()
    def apply_damping_to_delta(
        self,
        x: torch.Tensor,
        base_out: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, FallbackTelemetry]:
        """Apply smooth per-token damping directly to ``base_out + gamma * delta``."""
        if not self.config.enabled:
            tel = self._make_bypass_telemetry(x)
            self.last_telemetry = tel
            return (base_out.to(torch.float32) + delta.to(torch.float32)).to(dtype=base_out.dtype), tel

        gamma, safe_delta, telemetry = self.compute_damping_factors(x, base_out, delta)
        self.last_telemetry = telemetry
        guarded = (base_out.to(torch.float32) + gamma.unsqueeze(-1) * safe_delta).to(dtype=base_out.dtype)
        return guarded, telemetry

    @torch.no_grad()
    def apply_damping(
        self,
        x: torch.Tensor,
        base_out: torch.Tensor,
        moe_out: torch.Tensor,
    ) -> tuple[torch.Tensor, FallbackTelemetry]:
        """Smoothly damp ``moe_out - base_out`` back toward ``base_out`` when guards trigger."""
        if not self.config.enabled:
            tel = self._make_bypass_telemetry(x)
            self.last_telemetry = tel
            return moe_out, tel

        delta = moe_out.to(torch.float32) - base_out.to(torch.float32)
        guarded, telemetry = self.apply_damping_to_delta(x, base_out, delta)
        return guarded.to(dtype=moe_out.dtype), telemetry


# ---------------------------------------------------------------------------
# 3. Runtime MoE Layer & Model-Level Guard Installers
# ---------------------------------------------------------------------------


class AdaptiveSteeredMoELayer(nn.Module):
    """Runtime thermodynamic dense-fallback wrapper around a Sparse MoE MLP layer.

    When ``guard_enabled`` is ``False`` (or ``controller.config.enabled`` is ``False``),
    calls ``self.moe_mlp(x)`` directly with zero extra dense passes, zero probe matmuls,
    and zero telemetry allocations.
    """

    def __init__(
        self,
        moe_mlp: nn.Module,
        controller: ProseConfidenceFallbackController,
        *,
        dense_mlp: nn.Module | None = None,
        guard_enabled: bool = True,
        record_telemetry: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(moe_mlp, nn.Module):
            raise TypeError(f"moe_mlp must be an nn.Module, got {type(moe_mlp).__name__}")
        if not isinstance(controller, ProseConfidenceFallbackController):
            raise TypeError(
                f"controller must be a ProseConfidenceFallbackController, got {type(controller).__name__}"
            )
        _validate_bool(guard_enabled, "guard_enabled")
        _validate_bool(record_telemetry, "record_telemetry")

        self.moe_mlp = moe_mlp
        self.controller = controller
        self.guard_enabled = guard_enabled
        self.record_telemetry = record_telemetry
        self.last_telemetry: FallbackTelemetry | None = None

        base_attr = getattr(moe_mlp, "base_mlp", None)
        if dense_mlp is not None and dense_mlp is not base_attr:
            if not isinstance(dense_mlp, nn.Module):
                raise TypeError(f"dense_mlp must be an nn.Module, got {type(dense_mlp).__name__}")
            self._external_dense_mlp: nn.Module | None = dense_mlp
        else:
            self._external_dense_mlp = None
            if dense_mlp is None and base_attr is None and not isinstance(moe_mlp, CrowdFineGrainedMoEMLP):
                raise ValueError("dense_mlp must be provided when moe_mlp does not expose 'base_mlp'")

    def set_guard_enabled(self, enabled: bool) -> None:
        """Enable or disable runtime damping (zero-overhead bypass when ``False``)."""
        self.guard_enabled = _validate_bool(enabled, "enabled")

    def _compute_dense_output(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the dense baseline FFN output matching ``self.moe_mlp``."""
        if self._external_dense_mlp is not None:
            return self._external_dense_mlp(x)
        base_attr = getattr(self.moe_mlp, "base_mlp", None)
        if isinstance(base_attr, nn.Module):
            return base_attr(x)
        fg = self.moe_mlp
        if isinstance(fg, CrowdFineGrainedMoEMLP):
            orig_shape = x.shape
            orig_dt = x.dtype
            flat_x = x.reshape(-1, fg.hidden_size)
            acc = torch.zeros((flat_x.shape[0], fg.hidden_size), device=x.device, dtype=torch.float32)
            for e_idx in range(fg.num_experts):
                acc = acc + fg._run_expert(e_idx, flat_x)
            return acc.to(dtype=orig_dt).reshape(orig_shape)
        raise RuntimeError("No dense fallback path available for wrapped MoE module")  # pragma: no cover

    def _forward_guarded_crowd_lora(self, x: torch.Tensor) -> torch.Tensor:
        """Single-pass CrowdLoRAMoEMLP evaluation avoiding a duplicate ``base_mlp`` pass."""
        moe = self.moe_mlp
        assert isinstance(moe, CrowdLoRAMoEMLP)
        orig_shape = x.shape
        orig_dt = x.dtype
        flat_x = x.reshape(-1, moe.hidden_size)
        n_tok = flat_x.shape[0]

        w_dt = moe.base_mlp.w1.weight.dtype
        x_base = flat_x.to(dtype=w_dt)
        h1 = moe.base_mlp.w1(x_base).to(torch.float32)
        h2 = moe.base_mlp.w2(x_base).to(torch.float32)
        base_out = moe.base_mlp.w3((F.silu(h1) * h2).to(dtype=w_dt)).to(torch.float32)

        x_fp32 = flat_x.to(torch.float32)
        logits = moe.router(x_fp32)
        res_mask = moe.resident_mask.to(device=logits.device)

        if moe.routing_mode == "sigma_topk":
            moe.last_sigma_report = sigmoid_independent_topk_routing(logits, moe.top_k, res_mask)
            sig_probs = torch.sigmoid(logits)
            top_vals, top_idx = torch.topk(sig_probs, k=moe.top_k, dim=-1)
            skipped = ~res_mask[top_idx]
            gates = torch.where(skipped, torch.zeros_like(top_vals), top_vals)
            moe.last_ghost_fraction = moe.last_sigma_report.ghost_assignment_fraction
            norm_probs = sig_probs / sig_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            moe.last_aux_loss = _switch_load_balancing_loss(norm_probs, top_idx, moe.num_experts)
        else:
            probs = F.softmax(logits, dim=-1)
            top_vals, top_idx = torch.topk(probs, k=moe.top_k, dim=-1)
            skipped = ~res_mask[top_idx]
            renorm = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gates = torch.where(skipped, torch.zeros_like(renorm), renorm)
            moe.last_ghost_fraction = float(skipped.sum().item()) / float(max(1, n_tok * moe.top_k))
            moe.last_aux_loss = _switch_load_balancing_loss(probs, top_idx, moe.num_experts)

        delta_acc = torch.zeros_like(base_out)
        for e_key, expert_mod in moe.experts.items():
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
                moe.base_mlp.w3.weight,
            )
            delta_acc.index_add_(0, active_rows, dy * token_weights.index_select(0, active_rows).unsqueeze(-1))

        guarded, telemetry = self.controller.apply_damping_to_delta(x_fp32, base_out, delta_acc)
        if self.record_telemetry:
            self.last_telemetry = telemetry
        return guarded.to(dtype=orig_dt).reshape(orig_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run MoE inference with zero-overhead bypass when disabled or smooth damping when active."""
        if not self.guard_enabled or not self.controller.config.enabled:
            return self.moe_mlp(x)

        if type(self.moe_mlp) is CrowdLoRAMoEMLP and self._external_dense_mlp is None:
            return self._forward_guarded_crowd_lora(x)

        moe_out = self.moe_mlp(x)
        base_out = self._compute_dense_output(x)
        guarded, telemetry = self.controller.apply_damping(x, base_out, moe_out)
        if self.record_telemetry:
            self.last_telemetry = telemetry
        return guarded


def install_adaptive_steering_guard(
    model: nn.Module,
    controllers: Mapping[int, ProseConfidenceFallbackController] | ProseConfidenceFallbackController,
    *,
    layer_indices: Sequence[int] | None = None,
    guard_enabled: bool = True,
    record_telemetry: bool = True,
) -> dict[int, AdaptiveSteeredMoELayer]:
    """Wrap transformer block MLPs in ``model`` with ``AdaptiveSteeredMoELayer``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    n_blocks = len(blocks)

    if layer_indices is None:
        if isinstance(controllers, Mapping):
            targets = sorted(int(k) for k in controllers.keys())
        else:
            targets = list(range(n_blocks))
    else:
        targets = sorted({int(i) for i in layer_indices})

    if not targets:
        raise ValueError("layer_indices resolved to an empty target set")

    wrapped: dict[int, AdaptiveSteeredMoELayer] = {}
    for l_idx in targets:
        if l_idx < 0 or l_idx >= n_blocks:
            raise ValueError(f"layer index {l_idx} out of range [0, {n_blocks})")
        if isinstance(controllers, Mapping):
            if l_idx not in controllers:
                raise KeyError(f"Missing ProseConfidenceFallbackController for layer {l_idx}")
            ctrl = controllers[l_idx]
        else:
            ctrl = controllers

        raw_mlp = unwrap_moe_mlp(blocks[l_idx].mlp)
        layer_wrap = AdaptiveSteeredMoELayer(
            raw_mlp,
            ctrl,
            guard_enabled=guard_enabled,
            record_telemetry=record_telemetry,
        )
        blocks[l_idx].mlp = layer_wrap
        wrapped[l_idx] = layer_wrap
    return wrapped


def set_adaptive_steering_enabled(model: nn.Module, enabled: bool) -> int:
    """Toggle ``guard_enabled`` across all ``AdaptiveSteeredMoELayer`` blocks in ``model``."""
    flag = _validate_bool(enabled, "enabled")
    blocks = getattr(model, "blocks", ())
    toggled = 0
    for blk in blocks:
        mlp = getattr(blk, "mlp", None)
        if isinstance(mlp, AdaptiveSteeredMoELayer):
            mlp.set_guard_enabled(flag)
            toggled += 1
    return toggled


def remove_adaptive_steering_guard(model: nn.Module) -> int:
    """Restore original unwrapped MoE modules across all guarded blocks in ``model``."""
    blocks = getattr(model, "blocks", ())
    removed = 0
    for blk in blocks:
        mlp = getattr(blk, "mlp", None)
        if isinstance(mlp, AdaptiveSteeredMoELayer):
            blk.mlp = mlp.moe_mlp
            removed += 1
    return removed


def collect_adaptive_steering_telemetry(model: nn.Module) -> dict[str, Any]:
    """Aggregate per-layer damping statistics across all guarded MoE layers in ``model``."""
    blocks = getattr(model, "blocks", ())
    per_layer: dict[int, dict[str, Any]] = {}
    gammas: list[float] = []
    damped_fracs: list[float] = []
    max_post_ratio = 0.0
    max_raw_ratio = 0.0

    for l_idx, blk in enumerate(blocks):
        mlp = getattr(blk, "mlp", None)
        if isinstance(mlp, AdaptiveSteeredMoELayer) and mlp.last_telemetry is not None:
            tel = mlp.last_telemetry
            per_layer[l_idx] = tel.to_dict()
            gammas.append(tel.mean_gamma)
            damped_fracs.append(tel.damped_token_fraction)
            max_post_ratio = max(max_post_ratio, tel.post_damping_max_ratio)
            max_raw_ratio = max(max_raw_ratio, tel.max_delta_base_ratio)

    return {
        "guarded_layers_with_telemetry": len(per_layer),
        "mean_gamma": (sum(gammas) / len(gammas)) if gammas else 1.0,
        "mean_damped_token_fraction": (sum(damped_fracs) / len(damped_fracs)) if damped_fracs else 0.0,
        "max_raw_delta_base_ratio": max_raw_ratio,
        "max_post_damping_ratio": max_post_ratio,
        "per_layer": per_layer,
    }


__all__ = [
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_MAX_DELTA_BASE_RATIO",
    "DEFAULT_MAX_DOMAIN_DRIFT",
    "DEFAULT_NORM_EPS",
    "DEFAULT_PROBE_LOGIT_SCALE",
    "DEFAULT_PROBE_MIDPOINT",
    "DEFAULT_PROSE_CONFIDENCE_THRESHOLD",
    "DEFAULT_PROSE_TRANSITION_WIDTH",
    "DEFAULT_SPIKE_DECAY",
    "GENERAL_EXPERT_PREFIX",
    "AdaptiveSteeredMoELayer",
    "FallbackControllerConfig",
    "FallbackTelemetry",
    "HotSwapDecisionReceipt",
    "HotSwapGateConfig",
    "ProseConfidenceFallbackController",
    "ThermodynamicHotSwapGate",
    "collect_adaptive_steering_telemetry",
    "compute_buckets_sha256",
    "compute_thermodynamic_energy",
    "extract_dense_fallback_buckets",
    "extract_guarded_moe_buckets",
    "install_adaptive_steering_guard",
    "load_guarded_moe_buckets",
    "remove_adaptive_steering_guard",
    "set_adaptive_steering_enabled",
    "unwrap_moe_mlp",
]
