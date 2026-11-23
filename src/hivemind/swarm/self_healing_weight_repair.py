# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Self-Rewriting Weight-Block Corruption Scanner and Closed-Loop ANSE Spectral Repair.

Implements training-free, gradient-free diagnosis and surgical sub-0.05% self-rewriting
weight-block repair for open-weight Sparse MoEs (``CrowdV1WithEngram`` +
``CrowdLoRAMoEMLP`` SwiGLU rank-``r`` experts with ``sigma_topk`` / ``softmax_topk``
routers), inspired by *Language Models Can Rewrite Their Own Weights* (Nov 2026):

1. ``scan_moe_adapter_health``:
   Inspects SwiGLU LoRA factor pairs (``w1_A/B``, ``w2_A/B``, ``w3_A/B``) and router
   rows without gradients to detect:
   - ``NaN`` / ``Inf`` corruptions and finite BF16 exponent bit-flips,
   - heavy-tailed spectral spikes (``sigma_1 / median(sigma)``) via reduced QR-SVD,
   - effective stable-rank collapse (``||W||_F^2 / sigma_1^2``),
   - dead or globally saturated ``sigma_topk`` router rows.
2. ``propose_surgical_block_repairs`` & ``apply_surgical_block_repairs_inplace``:
   Synthesizes budget-checked sub-0.05% parameter edits:
   - finite zero-perturbation quarantine (sanitizing non-finite/bit-flip ``A``/``B``
     entries and enforcing exact ``w1_B = w2_B = w3_B = 0 => dy_e(x) == 0``),
   - QR-SVD singular-value winsorization preserving healthy singular modes,
   - saturated/dead ``sigma_topk`` router logit recentering orthogonal to healthy
     domain subspaces.
3. ``execute_self_rewriting_repair_loop``:
   Runs the closed-loop ``inspect -> propose -> deterministic probe eval -> accept iff
   Delta E < 0 and worst_domain_drift <= +1.0% (else atomic bitwise rollback)`` cycle
   with deterministic SHA-256 weight-state provenance.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.wsp_alphaedit_control import (
    ANSE_SAFE_E0_RATIO,
    DOMAIN_GUARD_MAX_RATIO,
    GENERAL_EXPERT_PREFIX,
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
    second_moment_and_mean,
)

DEFAULT_MAX_EDITED_PARAM_RATIO: float = 0.0005
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = DOMAIN_GUARD_MAX_RATIO
DEFAULT_MAX_DOMAIN_DRIFT: float = DOMAIN_GUARD_MAX_RATIO - 1.0
DEFAULT_ANSE_MAX_PAIN: float = 1_000_000.0
DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD: float = 10.0
DEFAULT_MIN_STABLE_RANK: float = 1.5
DEFAULT_WINSOR_CAP_MULTIPLIER: float = 2.5
DEFAULT_WINSOR_TRIGGER_MODE_RATIO: float = 4.0
DEFAULT_EXTREME_WEIGHT_THRESHOLD: float = 1e4
DEFAULT_ISOLATED_JUMP_MULTIPLIER: float = 500.0
DEFAULT_ISOLATED_JUMP_FLOOR: float = 100.0
DEFAULT_MIN_SINGULAR_VALUE: float = 1e-10
DEFAULT_SINGLE_MODE_SPIKE_NORM: float = 50.0
DEFAULT_ROUTER_SATURATION_LOGIT_THRESHOLD: float = 5.0
DEFAULT_ROUTER_SATURATED_GATE_THRESHOLD: float = 0.98
DEFAULT_ROUTER_DEAD_GATE_THRESHOLD: float = 0.02
DEFAULT_ROUTER_SATURATED_WEIGHT_NORM: float = 15.0
DEFAULT_ROUTER_NORM_OUTLIER_RATIO: float = 8.0
DEFAULT_ROUTER_TARGET_OFF_LOGIT: float = -3.5
DEFAULT_DEAD_ROUTER_TARGET_LOGIT: float = 0.0

_EXPERT_PREFIX_RE = re.compile(r"^E(\d+)(?:_|$)")
_SWIGLU_PROJECTIONS: tuple[str, ...] = ("w1", "w2", "w3")


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_pos_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_nonneg_finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str) -> float:
    """Validate that ``value`` is finite and within ``[0.0, 1.0]``."""
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return out


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_pos_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding bool)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


@dataclass(frozen=True)
class AdapterHealthScanConfig:
    """Thresholds governing gradient-free MoE adapter and router health inspection."""

    spectral_spike_ratio_threshold: float = DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD
    min_stable_rank: float = DEFAULT_MIN_STABLE_RANK
    extreme_weight_threshold: float = DEFAULT_EXTREME_WEIGHT_THRESHOLD
    isolated_jump_multiplier: float = DEFAULT_ISOLATED_JUMP_MULTIPLIER
    isolated_jump_floor: float = DEFAULT_ISOLATED_JUMP_FLOOR
    min_singular_value: float = DEFAULT_MIN_SINGULAR_VALUE
    single_mode_spike_norm: float = DEFAULT_SINGLE_MODE_SPIKE_NORM
    router_saturation_logit_threshold: float = DEFAULT_ROUTER_SATURATION_LOGIT_THRESHOLD
    router_saturated_gate_threshold: float = DEFAULT_ROUTER_SATURATED_GATE_THRESHOLD
    router_dead_gate_threshold: float = DEFAULT_ROUTER_DEAD_GATE_THRESHOLD
    router_saturated_weight_norm: float = DEFAULT_ROUTER_SATURATED_WEIGHT_NORM
    router_norm_outlier_ratio: float = DEFAULT_ROUTER_NORM_OUTLIER_RATIO
    score_from: int = 64
    batch_size: int = 24

    def __post_init__(self) -> None:
        _validate_pos_finite(self.spectral_spike_ratio_threshold, "spectral_spike_ratio_threshold")
        _validate_pos_finite(self.min_stable_rank, "min_stable_rank")
        _validate_pos_finite(self.extreme_weight_threshold, "extreme_weight_threshold")
        _validate_pos_finite(self.isolated_jump_multiplier, "isolated_jump_multiplier")
        _validate_pos_finite(self.isolated_jump_floor, "isolated_jump_floor")
        _validate_nonneg_finite(self.min_singular_value, "min_singular_value")
        _validate_pos_finite(self.single_mode_spike_norm, "single_mode_spike_norm")
        _validate_pos_finite(self.router_saturation_logit_threshold, "router_saturation_logit_threshold")
        _validate_unit_interval(self.router_saturated_gate_threshold, "router_saturated_gate_threshold")
        _validate_unit_interval(self.router_dead_gate_threshold, "router_dead_gate_threshold")
        _validate_pos_finite(self.router_saturated_weight_norm, "router_saturated_weight_norm")
        _validate_pos_finite(self.router_norm_outlier_ratio, "router_norm_outlier_ratio")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_pos_int(self.batch_size, "batch_size")


@dataclass(frozen=True)
class RepairProposalConfig:
    """Hyperparameters for sub-0.05% surgical weight-block repair synthesis."""

    max_edited_param_ratio: float = DEFAULT_MAX_EDITED_PARAM_RATIO
    enforce_param_budget: bool = True
    quarantine_strategy: str = "auto"
    winsor_cap_multiplier: float = DEFAULT_WINSOR_CAP_MULTIPLIER
    winsor_trigger_mode_ratio: float = DEFAULT_WINSOR_TRIGGER_MODE_RATIO
    router_target_off_logit: float = DEFAULT_ROUTER_TARGET_OFF_LOGIT
    dead_router_target_logit: float = DEFAULT_DEAD_ROUTER_TARGET_LOGIT
    router_preserve_weight: float = 50.0
    router_ridge_scale: float = 0.01
    router_max_off_gate_after: float = 0.08
    extreme_weight_threshold: float = DEFAULT_EXTREME_WEIGHT_THRESHOLD
    isolated_jump_multiplier: float = DEFAULT_ISOLATED_JUMP_MULTIPLIER
    isolated_jump_floor: float = DEFAULT_ISOLATED_JUMP_FLOOR
    min_singular_value: float = DEFAULT_MIN_SINGULAR_VALUE

    def __post_init__(self) -> None:
        _validate_unit_interval(self.max_edited_param_ratio, "max_edited_param_ratio")
        if not isinstance(self.enforce_param_budget, bool):
            raise TypeError(f"enforce_param_budget must be a bool, got {type(self.enforce_param_budget).__name__}")
        if self.quarantine_strategy not in {"auto", "zero_b", "elementwise"}:
            raise ValueError("quarantine_strategy must be 'auto', 'zero_b', or 'elementwise'")
        _validate_pos_finite(self.winsor_cap_multiplier, "winsor_cap_multiplier")
        _validate_pos_finite(self.winsor_trigger_mode_ratio, "winsor_trigger_mode_ratio")
        _validate_finite_float(self.router_target_off_logit, "router_target_off_logit")
        _validate_finite_float(self.dead_router_target_logit, "dead_router_target_logit")
        _validate_nonneg_finite(self.router_preserve_weight, "router_preserve_weight")
        _validate_nonneg_finite(self.router_ridge_scale, "router_ridge_scale")
        _validate_unit_interval(self.router_max_off_gate_after, "router_max_off_gate_after")
        _validate_pos_finite(self.extreme_weight_threshold, "extreme_weight_threshold")
        _validate_pos_finite(self.isolated_jump_multiplier, "isolated_jump_multiplier")
        _validate_pos_finite(self.isolated_jump_floor, "isolated_jump_floor")
        _validate_nonneg_finite(self.min_singular_value, "min_singular_value")


@dataclass(frozen=True)
class SelfHealingLoopConfig:
    """Closed-loop thermodynamic acceptance and atomic rollback configuration."""

    max_iterations: int = 3
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT
    health_penalty_weight: float = 1e-5
    anse_safe_drift_threshold: float = ANSE_SAFE_E0_RATIO - 1.0
    general_domain_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = 64
    batch_size: int = 24

    def __post_init__(self) -> None:
        _validate_pos_int(self.max_iterations, "max_iterations")
        _validate_nonneg_finite(self.max_domain_drift, "max_domain_drift")
        _validate_nonneg_finite(self.health_penalty_weight, "health_penalty_weight")
        _validate_nonneg_finite(self.anse_safe_drift_threshold, "anse_safe_drift_threshold")
        if not isinstance(self.general_domain_prefix, str) or not self.general_domain_prefix.strip():
            raise ValueError("general_domain_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_pos_int(self.batch_size, "batch_size")


# ---------------------------------------------------------------------------
# Internal Tensor, Layer & Bitwise Provenance Utilities
# ---------------------------------------------------------------------------


def _is_moe_mlp(mod: Any) -> bool:
    """Return True when ``mod`` exposes MoE ``experts`` and ``router`` submodules."""
    if not isinstance(mod, nn.Module):
        return False
    experts = getattr(mod, "experts", None)
    router = getattr(mod, "router", None)
    return isinstance(experts, (Mapping, nn.ModuleDict)) and isinstance(router, nn.Linear)


def _normalize_moe_layers(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
) -> tuple[list[tuple[str, nn.Module]], nn.Module | None]:
    """Normalize model, layer sequence, or mapping into ``[(layer_key, moe_mlp), ...]``."""
    if isinstance(moe_layers, nn.Module):
        if _is_moe_mlp(moe_layers):
            return [("0", moe_layers)], moe_layers
        blocks = getattr(moe_layers, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            layers: list[tuple[str, nn.Module]] = []
            for idx, blk in enumerate(blocks):
                cand = getattr(blk, "mlp", blk)
                if _is_moe_mlp(cand):
                    layers.append((str(idx), cand))
            if layers:
                return layers, moe_layers
        raise ValueError(
            f"Module {type(moe_layers).__name__} does not expose CrowdLoRAMoEMLP layers or blocks"
        )

    if isinstance(moe_layers, Mapping):
        if not moe_layers:
            raise ValueError("moe_layers mapping must be non-empty")
        layers_map: list[tuple[str, nn.Module]] = []
        for raw_key in sorted(moe_layers.keys(), key=lambda k: str(k)):
            mod = moe_layers[raw_key]
            cand = getattr(mod, "mlp", mod) if isinstance(mod, nn.Module) else mod
            if not _is_moe_mlp(cand):
                raise ValueError(f"moe_layers[{raw_key!r}] must be an MoE MLP with 'experts' and 'router'")
            layers_map.append((str(raw_key), cand))
        return layers_map, None

    if not isinstance(moe_layers, (str, bytes)) and isinstance(moe_layers, Sequence):
        if not moe_layers:
            raise ValueError("moe_layers sequence must be non-empty")
        layers_seq: list[tuple[str, nn.Module]] = []
        for idx, mod in enumerate(moe_layers):
            cand = getattr(mod, "mlp", mod) if isinstance(mod, nn.Module) else mod
            if not _is_moe_mlp(cand):
                raise ValueError(f"moe_layers[{idx}] must be an MoE MLP with 'experts' and 'router'")
            layers_seq.append((str(idx), cand))
        return layers_seq, None

    raise TypeError(
        "moe_layers must be an nn.Module, sequence of MoE modules, or mapping of layer_key -> MoE module"
    )


def _count_unique_params(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
    layers: Sequence[tuple[str, nn.Module]],
) -> int:
    """Count unique trainable+frozen parameters across ``moe_layers`` without double-counting tied weights."""
    seen: set[int] = set()
    total = 0

    def _add_module(mod: nn.Module) -> None:
        nonlocal total
        for p in mod.parameters():
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                total += int(p.numel())

    if isinstance(moe_layers, nn.Module):
        _add_module(moe_layers)
    else:
        for _, layer_mod in layers:
            _add_module(layer_mod)
    return max(1, total)


def _resolve_factor_tensor(exp_mod: nn.Module, attr_name: str) -> torch.Tensor:
    """Resolve a LoRA factor Tensor or Linear weight on ``exp_mod``."""
    attr = getattr(exp_mod, attr_name, None)
    if isinstance(attr, nn.Linear):
        attr = attr.weight
    if not isinstance(attr, torch.Tensor):
        raise ValueError(f"{type(exp_mod).__name__} must expose tensor or nn.Linear attribute {attr_name!r}")
    return attr


def _raw_bit_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return an integer bit-pattern view for exact NaN-safe bitwise comparison and hashing."""
    t_cpu = tensor.detach().cpu().contiguous()
    elem_bytes = t_cpu.element_size()
    if elem_bytes == 1:
        return t_cpu.view(torch.uint8)
    if elem_bytes == 2:
        return t_cpu.view(torch.int16)
    if elem_bytes == 4:
        return t_cpu.view(torch.int32)
    return t_cpu.view(torch.int64)


def _count_bitwise_modified(before: torch.Tensor, after: torch.Tensor) -> int:
    """Count exact bitwise-different elements between two same-shape tensors (NaN-safe)."""
    return int(torch.ne(_raw_bit_view(before), _raw_bit_view(after)).sum().item())


def _collect_state_tensors(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
) -> dict[str, torch.Tensor]:
    """Collect sorted canonical state tensor references for router, experts, and optional final LN."""
    layers, root_mod = _normalize_moe_layers(moe_layers)
    refs: dict[str, torch.Tensor] = {}
    for layer_key, moe_mlp in layers:
        refs[f"layers.{layer_key}.router.weight"] = moe_mlp.router.weight
        if moe_mlp.router.bias is not None:
            refs[f"layers.{layer_key}.router.bias"] = moe_mlp.router.bias
        for e_key in sorted(moe_mlp.experts.keys(), key=lambda k: (int(k) if str(k).isdigit() else 0, str(k))):
            exp_mod = moe_mlp.experts[e_key]
            for proj in _SWIGLU_PROJECTIONS:
                for suffix in ("A", "B"):
                    attr = f"{proj}_{suffix}"
                    if hasattr(exp_mod, attr):
                        refs[f"layers.{layer_key}.experts.{e_key}.{attr}"] = _resolve_factor_tensor(exp_mod, attr)
    if root_mod is not None and hasattr(root_mod, "ln"):
        ln_mod = root_mod.ln
        if isinstance(getattr(ln_mod, "weight", None), torch.Tensor):
            refs["model.ln.weight"] = ln_mod.weight
        if isinstance(getattr(ln_mod, "bias", None), torch.Tensor):
            refs["model.ln.bias"] = ln_mod.bias
    return refs


def compute_moe_state_sha256(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
) -> str:
    """Compute deterministic SHA-256 digest over raw weight bits of all MoE experts and routers."""
    refs = _collect_state_tensors(moe_layers)
    hasher = hashlib.sha256()
    for name in sorted(refs.keys()):
        t = refs[name].detach().cpu().contiguous()
        header = f"{name}|{tuple(t.shape)}|{t.dtype}|".encode()
        hasher.update(header)
        hasher.update(_raw_bit_view(t).reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


@torch.no_grad()
def snapshot_moe_state(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
) -> dict[str, torch.Tensor]:
    """Create an atomic bitwise clone of all MoE expert, router, and optional LN tensors."""
    refs = _collect_state_tensors(moe_layers)
    return {name: tensor.detach().clone() for name, tensor in refs.items()}


@torch.no_grad()
def restore_moe_state_inplace(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
    snapshot: Mapping[str, torch.Tensor],
) -> str:
    """Atomically restore MoE tensors from ``snapshot`` and return verified SHA-256 digest."""
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise ValueError("snapshot must be a non-empty mapping of tensor_key -> Tensor")
    refs = _collect_state_tensors(moe_layers)
    if set(refs.keys()) != set(snapshot.keys()):
        raise ValueError("snapshot keys do not match moe_layers state layout")
    for key, target in refs.items():
        src = snapshot[key]
        if target.shape != src.shape or target.dtype != src.dtype:
            raise ValueError(
                f"Snapshot mismatch on {key!r}: expected {tuple(target.shape)}/{target.dtype}, "
                f"got {tuple(src.shape)}/{src.dtype}"
            )
        target.copy_(src.to(device=target.device, dtype=target.dtype))
    return compute_moe_state_sha256(moe_layers)


def _detect_corruption_mask(
    tensor: torch.Tensor,
    *,
    extreme_threshold: float,
    jump_multiplier: float,
    jump_floor: float,
) -> tuple[torch.Tensor, int, int]:
    """Return ``(combined_bad_mask, non_finite_count, bitflip_outlier_count)`` for ``tensor``."""
    t64 = tensor.detach().to(device="cpu", dtype=torch.float64)
    finite_mask = torch.isfinite(t64)
    non_finite_mask = ~finite_mask
    non_finite_count = int(non_finite_mask.sum().item())

    abs_t = t64.abs()
    extreme_mask = finite_mask & (abs_t >= extreme_threshold)

    nz_finite = abs_t[finite_mask & (abs_t > 0.0) & (abs_t < extreme_threshold)]
    if nz_finite.numel() >= 4:
        med = float(torch.median(nz_finite).item())
        robust_scale = max(med, 1e-3)
        jump_thr = max(jump_floor, jump_multiplier * robust_scale)
        isolated_mask = finite_mask & (abs_t >= jump_thr)
        bitflip_mask = extreme_mask | isolated_mask
    else:
        bitflip_mask = extreme_mask

    bitflip_count = int(bitflip_mask.sum().item())
    combined = (non_finite_mask | bitflip_mask).to(device=tensor.device)
    return combined, non_finite_count, bitflip_count


def _parse_domain_index(domain_key: Any) -> int | None:
    """Parse integer expert index from either integer key or ``E{idx}_...`` domain name."""
    if isinstance(domain_key, int) and not isinstance(domain_key, bool):
        return domain_key if domain_key >= 0 else None
    text = str(domain_key).strip()
    if text.isdigit():
        return int(text)
    match = _EXPERT_PREFIX_RE.match(text)
    return int(match.group(1)) if match is not None else None


@torch.no_grad()
def _collect_layer_probe_activations(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
    layers: Sequence[tuple[str, nn.Module]],
    probe_activations: Mapping[Any, Any] | None,
    probe_chunks: Sequence[Mapping[str, Any]] | None,
    *,
    score_from: int,
    batch_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalize explicit or chunk-traced per-layer domain activation tensors ``layer_key -> {domain: 2D Tensor}``."""
    out: dict[str, dict[str, torch.Tensor]] = {lk: {} for lk, _ in layers}
    if probe_activations is not None:
        if not isinstance(probe_activations, Mapping) or not probe_activations:
            raise ValueError("probe_activations must be a non-empty mapping")
        first_val = next(iter(probe_activations.values()))
        if isinstance(first_val, torch.Tensor):
            # Flat mapping domain_id -> 2D Tensor shared across layers
            flat_dom: dict[str, torch.Tensor] = {}
            for dk, tens in probe_activations.items():
                if not isinstance(tens, torch.Tensor) or tens.ndim != 2 or tens.shape[0] == 0:
                    raise ValueError(f"probe_activations[{dk!r}] must be a non-empty 2D Tensor")
                flat_dom[str(dk)] = tens.detach().to("cpu", dtype=torch.float32)
            for lk, _ in layers:
                out[lk] = flat_dom
            return out
        for lk, _ in layers:
            raw_layer_map = probe_activations.get(lk)
            if raw_layer_map is None and lk.isdigit():
                raw_layer_map = probe_activations.get(int(lk))
            if raw_layer_map is not None:
                if not isinstance(raw_layer_map, Mapping):
                    raise ValueError(f"probe_activations[{lk!r}] must be a domain -> Tensor mapping")
                dom_map: dict[str, torch.Tensor] = {}
                for dk, tens in raw_layer_map.items():
                    if not isinstance(tens, torch.Tensor) or tens.ndim != 2 or tens.shape[0] == 0:
                        raise ValueError(f"probe_activations[{lk!r}][{dk!r}] must be a non-empty 2D Tensor")
                    dom_map[str(dk)] = tens.detach().to("cpu", dtype=torch.float32)
                out[lk] = dom_map
        return out

    if probe_chunks is None:
        return out

    if not isinstance(moe_layers, nn.Module) or not hasattr(moe_layers, "blocks") or not hasattr(moe_layers, "wte"):
        raise ValueError("probe_chunks tracing requires moe_layers to be a CrowdV1WithEngram-compatible model")
    if isinstance(probe_chunks, (str, bytes)) or not isinstance(probe_chunks, Sequence) or not probe_chunks:
        raise ValueError("probe_chunks must be a non-empty sequence of chunk dicts")

    ids_all = torch.stack([torch.as_tensor(ch["ids"], dtype=torch.long) for ch in probe_chunks], dim=0)
    seq_len = int(ids_all.shape[1])
    start_pos = min(max(0, score_from - 1), max(0, seq_len - 2))
    cos = moe_layers.cos_cached.to(ids_all.device)
    sin = moe_layers.sin_cached.to(ids_all.device)

    dom_indices: dict[str, list[int]] = {}
    for idx, ch in enumerate(probe_chunks):
        dom_indices.setdefault(str(ch["expert_id"]), []).append(idx)

    x_seq = torch.cat(
        [compute_embeddings_batch(moe_layers, ids_all[b0 : b0 + batch_size]) for b0 in range(0, ids_all.shape[0], batch_size)],
        dim=0,
    )
    for l_idx, blk in enumerate(moe_layers.blocks):
        xm_parts: list[torch.Tensor] = []
        mi_parts: list[torch.Tensor] = []
        for b0 in range(0, x_seq.shape[0], batch_size):
            xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + batch_size], cos, sin)
            xm_parts.append(xm)
            mi_parts.append(mi)
        x_mid = torch.cat(xm_parts, dim=0)
        mlp_in = torch.cat(mi_parts, dim=0)
        lk = str(l_idx)
        if lk in out:
            scored = mlp_in[:, start_pos:-1, :]
            hd = int(scored.shape[-1])
            out[lk] = {
                dom: scored[idxs].reshape(-1, hd).detach().cpu().to(torch.float32)
                for dom, idxs in dom_indices.items()
            }

        next_parts: list[torch.Tensor] = []
        for b0 in range(0, x_mid.shape[0], batch_size):
            xm_b = x_mid[b0 : b0 + batch_size]
            mi_b = mlp_in[b0 : b0 + batch_size]
            try:
                post_b = forward_block_post_mlp(blk, xm_b, mi_b)
                post_finite = bool(torch.isfinite(post_b).all().item())
            except ValueError:
                post_b = xm_b
                post_finite = False
            # Prevent corrupted earlier layer NaNs from blinding downstream router diagnostics.
            if not post_finite:
                base_mod = getattr(blk.mlp, "base_mlp", None)
                if base_mod is not None:
                    w_dt = base_mod.w1.weight.dtype if hasattr(base_mod, "w1") else mi_b.dtype
                    post_b = (xm_b + base_mod(mi_b.to(dtype=w_dt)).to(torch.float32)).to(torch.float32)
                else:
                    post_b = xm_b
            next_parts.append(post_b)
        x_seq = torch.cat(next_parts, dim=0)

    return out


# ---------------------------------------------------------------------------
# (a) Training-Free MoE Adapter & Router Health Scanner
# ---------------------------------------------------------------------------


def _inspect_lora_factor_pair(
    exp_mod: nn.Module,
    proj: str,
    cfg: AdapterHealthScanConfig,
) -> dict[str, Any]:
    """Inspect one SwiGLU LoRA factor pair ``(proj_B, proj_A)`` without gradients."""
    w_a = _resolve_factor_tensor(exp_mod, f"{proj}_A")
    w_b = _resolve_factor_tensor(exp_mod, f"{proj}_B")

    mask_a, nf_a, bf_a = _detect_corruption_mask(
        w_a,
        extreme_threshold=cfg.extreme_weight_threshold,
        jump_multiplier=cfg.isolated_jump_multiplier,
        jump_floor=cfg.isolated_jump_floor,
    )
    mask_b, nf_b, bf_b = _detect_corruption_mask(
        w_b,
        extreme_threshold=cfg.extreme_weight_threshold,
        jump_multiplier=cfg.isolated_jump_multiplier,
        jump_floor=cfg.isolated_jump_floor,
    )
    del mask_a, mask_b

    non_finite_total = nf_a + nf_b
    bitflip_total = bf_a + bf_b
    anomalies: list[str] = []
    if non_finite_total > 0 or bitflip_total > 0:
        anomalies.append("non_finite_or_bitflip")
        return {
            "projection": proj,
            "finite": non_finite_total == 0,
            "non_finite_count": non_finite_total,
            "bitflip_outlier_count": bitflip_total,
            "is_zero_adapter": False,
            "rank": int(w_a.shape[0]),
            "effective_capacity": int(min(w_b.shape[0], w_a.shape[1], w_a.shape[0])),
            "sigma_1": None,
            "median_sigma": None,
            "spectral_outlier_ratio": None,
            "stable_rank": None,
            "singular_values": [],
            "anomalies": anomalies,
        }

    _, s_vals, _ = decompose_lora_qr_svd(w_b, w_a)
    s_pos = s_vals[s_vals > cfg.min_singular_value]
    eff_cap = int(min(w_b.shape[0], w_a.shape[1], w_a.shape[0]))
    if s_pos.numel() == 0:
        return {
            "projection": proj,
            "finite": True,
            "non_finite_count": 0,
            "bitflip_outlier_count": 0,
            "is_zero_adapter": True,
            "rank": int(w_a.shape[0]),
            "effective_capacity": eff_cap,
            "sigma_1": 0.0,
            "median_sigma": 0.0,
            "spectral_outlier_ratio": 0.0,
            "stable_rank": 0.0,
            "singular_values": [float(x) for x in s_vals.tolist()],
            "anomalies": [],
        }

    sigma_1 = float(s_vals[0].item())
    median_sigma = float(torch.median(s_pos).item())
    spectral_ratio = sigma_1 / max(median_sigma, 1e-12)
    frob_sq = float((s_vals * s_vals).sum().item())
    stable_rank = frob_sq / max(sigma_1 * sigma_1, 1e-24)

    if s_pos.numel() >= 2 and spectral_ratio > cfg.spectral_spike_ratio_threshold:
        anomalies.append("spectral_spike")
    elif s_pos.numel() == 1 and sigma_1 > cfg.single_mode_spike_norm:
        anomalies.append("spectral_spike")

    if eff_cap >= 2 and (
        (s_pos.numel() >= 2 and stable_rank < cfg.min_stable_rank)
        or (s_pos.numel() == 1 and sigma_1 > cfg.single_mode_spike_norm)
    ):
        anomalies.append("stable_rank_collapse")

    return {
        "projection": proj,
        "finite": True,
        "non_finite_count": 0,
        "bitflip_outlier_count": 0,
        "is_zero_adapter": False,
        "rank": int(w_a.shape[0]),
        "effective_capacity": eff_cap,
        "sigma_1": sigma_1,
        "median_sigma": median_sigma,
        "spectral_outlier_ratio": spectral_ratio,
        "stable_rank": stable_rank,
        "singular_values": [float(x) for x in s_vals.tolist()],
        "anomalies": anomalies,
    }


def _inspect_router_health(
    moe_mlp: nn.Module,
    domain_acts: Mapping[str, torch.Tensor],
    cfg: AdapterHealthScanConfig,
) -> dict[str, Any]:
    """Inspect ``sigma_topk`` / ``softmax_topk`` router rows for bit-flips, saturation, and dead gates."""
    router: nn.Linear = moe_mlp.router
    num_experts = int(router.weight.shape[0])
    w_r = router.weight.detach().to("cpu", dtype=torch.float64)
    b_r = router.bias.detach().to("cpu", dtype=torch.float64) if router.bias is not None else None
    res_mask_raw = getattr(moe_mlp, "resident_mask", None)
    res_mask = (
        res_mask_raw.detach().cpu().to(torch.bool)
        if isinstance(res_mask_raw, torch.Tensor) and res_mask_raw.shape == (num_experts,)
        else torch.ones(num_experts, dtype=torch.bool)
    )
    top_k = max(1, min(num_experts, int(getattr(moe_mlp, "top_k", 1))))

    finite_rows_mask = torch.isfinite(w_r).all(dim=1)
    if b_r is not None:
        finite_rows_mask = finite_rows_mask & torch.isfinite(b_r)
    row_norms = torch.where(finite_rows_mask, w_r.norm(dim=1), torch.zeros(num_experts, dtype=torch.float64))
    nz_norms = row_norms[(finite_rows_mask) & (row_norms > 1e-6)]
    median_row_norm = float(torch.median(nz_norms).item()) if nz_norms.numel() > 0 else 0.0

    # Compute per-domain logit and sigmoid gate matrices when probe activations are available.
    domain_logits: dict[str, torch.Tensor] = {}
    domain_gates: dict[str, torch.Tensor] = {}
    topk_hits_total = torch.zeros(num_experts, dtype=torch.float64)
    total_probe_tokens = 0
    for dom_name, x_dom in sorted(domain_acts.items()):
        x_f64 = x_dom.detach().to("cpu", dtype=torch.float64)
        valid = torch.isfinite(x_f64).all(dim=1)
        x_clean = x_f64[valid]
        if x_clean.shape[0] == 0:
            continue
        logits_d = x_clean @ w_r.T
        if b_r is not None:
            logits_d = logits_d + b_r.unsqueeze(0)
        # Replace non-finite columns from corrupted router rows with 0 before topk accounting.
        safe_logits = torch.where(torch.isfinite(logits_d), logits_d, torch.full_like(logits_d, -1e9))
        gates_d = torch.sigmoid(safe_logits)
        _, top_idx = torch.topk(gates_d, k=top_k, dim=-1)
        for e_idx in range(num_experts):
            topk_hits_total[e_idx] += float((top_idx == e_idx).any(dim=-1).sum().item())
        total_probe_tokens += int(x_clean.shape[0])
        domain_logits[dom_name] = logits_d
        domain_gates[dom_name] = gates_d

    rows_report: list[dict[str, Any]] = []
    for e_idx in range(num_experts):
        row_w = router.weight[e_idx]
        _, nf_w, bf_w = _detect_corruption_mask(
            row_w,
            extreme_threshold=cfg.extreme_weight_threshold,
            jump_multiplier=cfg.isolated_jump_multiplier,
            jump_floor=cfg.isolated_jump_floor,
        )
        nf_b = 0
        bf_b = 0
        if router.bias is not None:
            b_slice = router.bias[e_idx : e_idx + 1]
            _, nf_b, bf_b = _detect_corruption_mask(
                b_slice,
                extreme_threshold=cfg.extreme_weight_threshold,
                jump_multiplier=cfg.isolated_jump_multiplier,
                jump_floor=cfg.isolated_jump_floor,
            )

        anomalies: list[str] = []
        row_norm = float(row_norms[e_idx].item()) if bool(finite_rows_mask[e_idx].item()) else None
        bias_val = float(b_r[e_idx].item()) if (b_r is not None and bool(torch.isfinite(b_r[e_idx]).item())) else None
        is_resident = bool(res_mask[e_idx].item())

        mean_logit_by_dom: dict[str, float] = {}
        mean_gate_by_dom: dict[str, float] = {}
        min_dom_logit: float | None = None
        max_dom_logit: float | None = None
        min_dom_gate: float | None = None
        max_dom_gate: float | None = None
        topk_rate = (float(topk_hits_total[e_idx].item()) / float(total_probe_tokens)) if total_probe_tokens > 0 else None

        if nf_w + nf_b + bf_w + bf_b > 0:
            anomalies.append("router_non_finite_or_bitflip")
        elif domain_logits:
            for dom_name in sorted(domain_logits.keys()):
                mean_logit_by_dom[dom_name] = float(domain_logits[dom_name][:, e_idx].mean().item())
                mean_gate_by_dom[dom_name] = float(domain_gates[dom_name][:, e_idx].mean().item())
            min_dom_logit = min(mean_logit_by_dom.values())
            max_dom_logit = max(mean_logit_by_dom.values())
            min_dom_gate = min(mean_gate_by_dom.values())
            max_dom_gate = max(mean_gate_by_dom.values())

            off_logits = [
                val for dname, val in mean_logit_by_dom.items() if _parse_domain_index(dname) != e_idx
            ]
            off_gates = [
                val for dname, val in mean_gate_by_dom.items() if _parse_domain_index(dname) != e_idx
            ]
            check_sat_logit = min(off_logits) if len(off_logits) >= 2 else min_dom_logit
            check_sat_gate = min(off_gates) if len(off_gates) >= 2 else min_dom_gate
            if (len(mean_logit_by_dom) >= 2 or len(off_logits) >= 1) and (
                check_sat_logit >= cfg.router_saturation_logit_threshold
                or check_sat_gate >= cfg.router_saturated_gate_threshold
            ):
                anomalies.append("saturated_router_row")
            elif is_resident and max_dom_gate <= cfg.router_dead_gate_threshold and (topk_rate or 0.0) <= 1e-4:
                anomalies.append("dead_router_row")
        else:
            # Static weight-only fallback when no probe activations are available.
            if bias_val is not None and bias_val >= cfg.router_saturation_logit_threshold:
                anomalies.append("saturated_router_row")
            elif is_resident and bias_val is not None and bias_val <= -cfg.router_saturation_logit_threshold:
                anomalies.append("dead_router_row")
            elif row_norm is not None and (
                row_norm >= cfg.router_saturated_weight_norm
                or (median_row_norm > 1e-6 and row_norm / median_row_norm >= cfg.router_norm_outlier_ratio)
            ):
                anomalies.append("saturated_router_row")

        rows_report.append(
            {
                "expert_id": e_idx,
                "resident": is_resident,
                "finite": (nf_w + nf_b) == 0,
                "non_finite_count": nf_w + nf_b,
                "bitflip_outlier_count": bf_w + bf_b,
                "row_norm": row_norm,
                "bias_value": bias_val,
                "min_domain_mean_logit": min_dom_logit,
                "max_domain_mean_logit": max_dom_logit,
                "min_domain_mean_gate": min_dom_gate,
                "max_domain_mean_gate": max_dom_gate,
                "topk_selection_rate": topk_rate,
                "mean_logit_by_domain": mean_logit_by_dom,
                "mean_gate_by_domain": mean_gate_by_dom,
                "anomalies": anomalies,
            }
        )

    return {"num_experts": num_experts, "top_k": top_k, "rows": rows_report}


@torch.no_grad()
def scan_moe_adapter_health(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
    *,
    config: AdapterHealthScanConfig | None = None,
    probe_activations: Mapping[Any, Any] | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scan SwiGLU LoRA adapters and router rows for bit-flips, spectral spikes, collapse, and saturation."""
    if config is not None and not isinstance(config, AdapterHealthScanConfig):
        raise TypeError(f"config must be AdapterHealthScanConfig or None, got {type(config).__name__}")
    cfg = config or AdapterHealthScanConfig()
    layers, _ = _normalize_moe_layers(moe_layers)
    total_params = _count_unique_params(moe_layers, layers)

    layer_acts = _collect_layer_probe_activations(
        moe_layers,
        layers,
        probe_activations,
        probe_chunks,
        score_from=cfg.score_from,
        batch_size=cfg.batch_size,
    )

    non_finite_adapters: list[dict[str, Any]] = []
    spectral_spike_adapters: list[dict[str, Any]] = []
    rank_collapsed_adapters: list[dict[str, Any]] = []
    saturated_router_rows: list[dict[str, Any]] = []
    dead_router_rows: list[dict[str, Any]] = []
    corrupted_router_rows: list[dict[str, Any]] = []
    layers_out: list[dict[str, Any]] = []

    severity = 0.0
    for layer_key, moe_mlp in layers:
        experts_report: dict[str, dict[str, Any]] = {}
        for e_key in sorted(moe_mlp.experts.keys(), key=lambda k: (int(k) if str(k).isdigit() else 0, str(k))):
            exp_mod = moe_mlp.experts[e_key]
            proj_reports: dict[str, dict[str, Any]] = {}
            for proj in _SWIGLU_PROJECTIONS:
                if not (hasattr(exp_mod, f"{proj}_A") and hasattr(exp_mod, f"{proj}_B")):
                    continue
                prep = _inspect_lora_factor_pair(exp_mod, proj, cfg)
                proj_reports[proj] = prep
                item = {"layer": layer_key, "expert_id": int(e_key) if str(e_key).isdigit() else str(e_key), **prep}
                if "non_finite_or_bitflip" in prep["anomalies"]:
                    non_finite_adapters.append(item)
                    severity += 100.0 + float(prep["non_finite_count"] + prep["bitflip_outlier_count"])
                if "spectral_spike" in prep["anomalies"]:
                    spectral_spike_adapters.append(item)
                    severity += min(100.0, float(prep["spectral_outlier_ratio"] or 10.0))
                if "stable_rank_collapse" in prep["anomalies"]:
                    rank_collapsed_adapters.append(item)
                    severity += max(0.5, cfg.min_stable_rank - float(prep["stable_rank"] or 0.0))
            experts_report[str(e_key)] = {"projections": proj_reports}

        router_rep = _inspect_router_health(moe_mlp, layer_acts.get(layer_key, {}), cfg)
        for row_rep in router_rep["rows"]:
            r_item = {"layer": layer_key, **row_rep}
            if "router_non_finite_or_bitflip" in row_rep["anomalies"]:
                corrupted_router_rows.append(r_item)
                severity += 100.0
            if "saturated_router_row" in row_rep["anomalies"]:
                saturated_router_rows.append(r_item)
                severity += 20.0
            if "dead_router_row" in row_rep["anomalies"]:
                dead_router_rows.append(r_item)
                severity += 5.0

        layers_out.append({"layer": layer_key, "experts": experts_report, "router": router_rep})

    total_anomalies = (
        len(non_finite_adapters)
        + len(spectral_spike_adapters)
        + len(rank_collapsed_adapters)
        + len(saturated_router_rows)
        + len(dead_router_rows)
        + len(corrupted_router_rows)
    )
    return {
        "healthy": total_anomalies == 0,
        "state_sha256": compute_moe_state_sha256(moe_layers),
        "num_layers": len(layers),
        "total_params_in_layers": total_params,
        "total_anomalies": total_anomalies,
        "health_severity": float(severity),
        "non_finite_or_bitflip_adapters": non_finite_adapters,
        "spectral_spike_adapters": spectral_spike_adapters,
        "stable_rank_collapsed_adapters": rank_collapsed_adapters,
        "saturated_router_rows": saturated_router_rows,
        "dead_router_rows": dead_router_rows,
        "corrupted_router_rows": corrupted_router_rows,
        "layers": layers_out,
    }


# ---------------------------------------------------------------------------
# (b) Surgical Weight-Block Repair Synthesis & Application
# ---------------------------------------------------------------------------


def _synthesize_winsorized_lora_a(
    exp_mod: nn.Module,
    proj: str,
    cfg: RepairProposalConfig,
) -> tuple[torch.Tensor, dict[str, Any]] | None:
    """Synthesize a QR-SVD winsorized ``proj_A`` tensor leaving healthy singular modes untouched."""
    w_a = _resolve_factor_tensor(exp_mod, f"{proj}_A")
    w_b = _resolve_factor_tensor(exp_mod, f"{proj}_B")
    u_r, s_vals, v_r = decompose_lora_qr_svd(w_b, w_a)
    del u_r

    s_pos = s_vals[s_vals > cfg.min_singular_value]
    if s_pos.numel() == 0:
        return None
    median_sigma = float(torch.median(s_pos).item())
    sigma_cap = cfg.winsor_cap_multiplier * median_sigma
    trigger_thr = max(sigma_cap, cfg.winsor_trigger_mode_ratio * median_sigma)

    k_eff = int(s_vals.numel())
    damped_indices = [k for k in range(k_eff) if float(s_vals[k].item()) > trigger_thr]
    if not damped_indices:
        damped_indices = [k for k in range(k_eff) if float(s_vals[k].item()) > sigma_cap]
    if not damped_indices:
        return None

    idx_t = torch.tensor(damped_indices, dtype=torch.long)
    scales = torch.tensor([min(1.0, sigma_cap / float(s_vals[k].item())) for k in damped_indices], dtype=torch.float64)
    v_d = v_r[:, idx_t]
    shrink_d = 1.0 - scales

    a64 = w_a.detach().to("cpu", dtype=torch.float64)
    delta_a = ((a64 @ v_d) * shrink_d.unsqueeze(0)) @ v_d.T
    a_new = (a64 - delta_a).to(device=w_a.device, dtype=w_a.dtype)
    mod_elems = _count_bitwise_modified(w_a, a_new)
    if mod_elems == 0:
        return None

    s_after = s_vals.clone()
    s_after[idx_t] = s_vals[idx_t] * scales
    frob_after = float((s_after * s_after).sum().item())
    s1_after = float(s_after.max().item())
    stable_rank_after = frob_after / max(s1_after * s1_after, 1e-24)

    return a_new, {
        "winsorized_modes": damped_indices,
        "sigma_cap": sigma_cap,
        "sigma_1_before": float(s_vals[0].item()),
        "sigma_1_after": s1_after,
        "median_sigma_before": median_sigma,
        "stable_rank_after": stable_rank_after,
        "modified_elements": mod_elems,
    }


def _solve_orthogonal_router_recenter_delta(
    w_row_64: torch.Tensor,
    b_val: float,
    x_off_64: torch.Tensor,
    x_preserve_64: torch.Tensor | None,
    cfg: RepairProposalConfig,
) -> tuple[torch.Tensor, float]:
    """Compute minimum-distortion router row update orthogonal to healthy domain subspaces."""
    d = int(w_row_64.numel())
    eye_d = torch.eye(d, dtype=torch.float64)
    z_off = (x_off_64 @ w_row_64) + b_val
    residual = z_off - cfg.router_target_off_logit

    m_off, _ = second_moment_and_mean(x_off_64)
    g_off = (x_off_64.T @ residual) / float(x_off_64.shape[0])
    tr_off = max(float(torch.diagonal(m_off).mean().item()), 1e-6)
    lam = max(cfg.router_ridge_scale * tr_off, 1e-5)

    def _solve_with_penalty(m_pres: torch.Tensor | None, omega: float) -> torch.Tensor:
        comb = m_off if m_pres is None else (m_off + omega * m_pres)
        comb = 0.5 * (comb + comb.T)
        return torch.linalg.solve(comb + lam * eye_d, g_off)

    if x_preserve_64 is not None and x_preserve_64.shape[0] > 0:
        m_pres_uncentered, _ = second_moment_and_mean(x_preserve_64)
        dw = _solve_with_penalty(m_pres_uncentered, cfg.router_preserve_weight)
        mean_gate_after = float(torch.sigmoid((x_off_64 @ (w_row_64 - dw)) + b_val).mean().item())
        if mean_gate_after <= cfg.router_max_off_gate_after:
            return dw, mean_gate_after

        # Collinear common carrier present in both off-domain and own-domain: protect centered specialist variation.
        x_pres_centered = x_preserve_64 - x_preserve_64.mean(dim=0, keepdim=True)
        if float(x_pres_centered.pow(2).sum().item()) > 1e-10:
            m_pres_centered, _ = second_moment_and_mean(x_pres_centered)
            dw = _solve_with_penalty(m_pres_centered, cfg.router_preserve_weight)
            mean_gate_after = float(torch.sigmoid((x_off_64 @ (w_row_64 - dw)) + b_val).mean().item())
            if mean_gate_after <= cfg.router_max_off_gate_after:
                return dw, mean_gate_after

    dw = _solve_with_penalty(None, 0.0)
    mean_gate_after = float(torch.sigmoid((x_off_64 @ (w_row_64 - dw)) + b_val).mean().item())
    return dw, mean_gate_after


def _synthesize_router_recenter_patch(
    moe_mlp: nn.Module,
    layer_key: str,
    expert_idx: int,
    kind: str,
    domain_acts: Mapping[str, torch.Tensor],
    healthy_subspace: torch.Tensor | None,
    cfg: RepairProposalConfig,
) -> list[dict[str, Any]]:
    """Synthesize surgical weight/bias recentering edits for a saturated or dead router row."""
    router: nn.Linear = moe_mlp.router
    w_row = router.weight[expert_idx]
    w_row_64 = w_row.detach().to("cpu", dtype=torch.float64)
    has_bias = router.bias is not None
    b_val = float(router.bias[expert_idx].item()) if has_bias and router.bias is not None else 0.0
    patches: list[dict[str, Any]] = []

    if kind == "saturated":
        own_parts: list[torch.Tensor] = []
        off_parts: list[torch.Tensor] = []
        for dom_name, tens in sorted(domain_acts.items()):
            t64 = tens.detach().to("cpu", dtype=torch.float64)
            valid = torch.isfinite(t64).all(dim=1)
            t_clean = t64[valid]
            if t_clean.shape[0] == 0:
                continue
            if _parse_domain_index(dom_name) == expert_idx:
                own_parts.append(t_clean)
            else:
                off_parts.append(t_clean)

        if not off_parts and own_parts:
            off_parts = own_parts
            own_parts = []

        if has_bias and router.bias is not None:
            if off_parts:
                x_off_cat = torch.cat(off_parts, dim=0)
                off_mean = float(((x_off_cat @ w_row_64) + b_val).mean().item())
                shift = off_mean - cfg.router_target_off_logit
                new_b = b_val - shift
            else:
                new_b = cfg.router_target_off_logit
            b_tensor = torch.tensor(new_b, device=router.bias.device, dtype=router.bias.dtype)
            mod_b = _count_bitwise_modified(router.bias[expert_idx], b_tensor)
            if mod_b > 0:
                patches.append(
                    {
                        "layer": layer_key,
                        "expert_id": expert_idx,
                        "target_attr": "router.bias",
                        "row_index": expert_idx,
                        "repair_kind": "saturated_router_recenter",
                        "new_tensor": b_tensor,
                        "modified_elements": mod_b,
                        "meta": {"bias_before": b_val, "bias_after": float(new_b)},
                    }
                )
            b_val = float(new_b)
            # If bias recentering already brings all off-domain gates below threshold, skip weight edits.
            if off_parts:
                x_off_cat = torch.cat(off_parts, dim=0)
                gate_after_bias = float(torch.sigmoid((x_off_cat @ w_row_64) + b_val).mean().item())
                if gate_after_bias <= cfg.router_max_off_gate_after:
                    return patches

        if off_parts:
            x_off_64 = torch.cat(off_parts, dim=0)
            pres_parts: list[torch.Tensor] = list(own_parts)
            if healthy_subspace is not None:
                pres_parts.append(healthy_subspace.detach().to("cpu", dtype=torch.float64))
            x_pres_64 = torch.cat(pres_parts, dim=0) if pres_parts else None
            dw, off_gate_after = _solve_orthogonal_router_recenter_delta(
                w_row_64, b_val, x_off_64, x_pres_64, cfg
            )
            w_new = (w_row_64 - dw).to(device=w_row.device, dtype=w_row.dtype)
            mod_w = _count_bitwise_modified(w_row, w_new)
            if mod_w > 0:
                patches.append(
                    {
                        "layer": layer_key,
                        "expert_id": expert_idx,
                        "target_attr": "router.weight",
                        "row_index": expert_idx,
                        "repair_kind": "saturated_router_recenter",
                        "new_tensor": w_new,
                        "modified_elements": mod_w,
                        "meta": {"off_domain_gate_after": off_gate_after, "orthogonal_projected": x_pres_64 is not None},
                    }
                )
            return patches

        # Weight-only fallback when no probe activations exist: damp component outside healthy router rows.
        all_w = router.weight.detach().to("cpu", dtype=torch.float64)
        other_rows = [all_w[k] for k in range(all_w.shape[0]) if k != expert_idx and bool(torch.isfinite(all_w[k]).all())]
        healthy_norms = [float(r.norm().item()) for r in other_rows if float(r.norm().item()) > 1e-6]
        target_norm = float(np.median(healthy_norms)) if healthy_norms else 1.0
        cur_norm = float(w_row_64.norm().item())
        if cur_norm > max(target_norm, 1e-8):
            if other_rows:
                w_h = torch.stack(other_rows, dim=0)
                _, s_h, vh_h = torch.linalg.svd(w_h, full_matrices=False)
                active_vh = vh_h[s_h > 1e-8]
                w_par = (w_row_64 @ active_vh.T) @ active_vh if active_vh.numel() > 0 else torch.zeros_like(w_row_64)
                w_perp = w_row_64 - w_par
                w_new_64 = w_par + w_perp * min(1.0, target_norm / max(float(w_perp.norm().item()), 1e-8))
                if float(w_new_64.norm().item()) > target_norm:
                    w_new_64 = w_new_64 * (target_norm / float(w_new_64.norm().item()))
            else:
                w_new_64 = w_row_64 * (target_norm / cur_norm)
            w_new = w_new_64.to(device=w_row.device, dtype=w_row.dtype)
            mod_w = _count_bitwise_modified(w_row, w_new)
            if mod_w > 0:
                patches.append(
                    {
                        "layer": layer_key,
                        "expert_id": expert_idx,
                        "target_attr": "router.weight",
                        "row_index": expert_idx,
                        "repair_kind": "saturated_router_recenter",
                        "new_tensor": w_new,
                        "modified_elements": mod_w,
                        "meta": {"target_norm": target_norm, "weight_only_fallback": True},
                    }
                )
        return patches

    # Dead router row recentering: restore neutral sigma(0)=0.50 responsiveness cleanly.
    if has_bias and router.bias is not None and b_val < -2.0:
        b_tensor = torch.tensor(cfg.dead_router_target_logit, device=router.bias.device, dtype=router.bias.dtype)
        mod_b = _count_bitwise_modified(router.bias[expert_idx], b_tensor)
        if mod_b > 0:
            patches.append(
                {
                    "layer": layer_key,
                    "expert_id": expert_idx,
                    "target_attr": "router.bias",
                    "row_index": expert_idx,
                    "repair_kind": "dead_router_recenter",
                    "new_tensor": b_tensor,
                    "modified_elements": mod_b,
                    "meta": {"bias_before": b_val, "bias_after": cfg.dead_router_target_logit},
                }
            )
    else:
        w_new = torch.zeros_like(w_row)
        mod_w = _count_bitwise_modified(w_row, w_new)
        if mod_w > 0:
            patches.append(
                {
                    "layer": layer_key,
                    "expert_id": expert_idx,
                    "target_attr": "router.weight",
                    "row_index": expert_idx,
                    "repair_kind": "dead_router_recenter",
                    "new_tensor": w_new,
                    "modified_elements": mod_w,
                    "meta": {"restored_neutral_gate": 0.5},
                }
            )
    return patches


def _build_quarantine_patches(
    exp_mod: nn.Module,
    layer_key: str,
    expert_key: str,
    cfg: RepairProposalConfig,
    remaining_budget: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Synthesize finite zero-perturbation quarantine or elementwise sanitization patches."""
    elem_patches: list[dict[str, Any]] = []
    elem_cost = 0
    zero_b_patches: list[dict[str, Any]] = []
    zero_b_cost = 0

    for proj in _SWIGLU_PROJECTIONS:
        for suffix in ("A", "B"):
            attr = f"{proj}_{suffix}"
            if not hasattr(exp_mod, attr):
                continue
            w = _resolve_factor_tensor(exp_mod, attr)
            bad_mask, nf_cnt, bf_cnt = _detect_corruption_mask(
                w,
                extreme_threshold=cfg.extreme_weight_threshold,
                jump_multiplier=cfg.isolated_jump_multiplier,
                jump_floor=cfg.isolated_jump_floor,
            )
            has_bad = (nf_cnt + bf_cnt) > 0
            if has_bad:
                w_sanitized = torch.where(bad_mask, torch.zeros_like(w), w)
                san_cost = _count_bitwise_modified(w, w_sanitized)
                if san_cost > 0:
                    elem_patches.append(
                        {
                            "layer": layer_key,
                            "expert_id": int(expert_key) if expert_key.isdigit() else expert_key,
                            "target_attr": f"experts.{expert_key}.{attr}",
                            "repair_kind": "elementwise_bitflip_sanitize",
                            "new_tensor": w_sanitized,
                            "modified_elements": san_cost,
                            "meta": {"non_finite_count": nf_cnt, "bitflip_outlier_count": bf_cnt},
                        }
                    )
                    elem_cost += san_cost

            if suffix == "B":
                w_zero = torch.zeros_like(w)
                zb_cost = _count_bitwise_modified(w, w_zero)
                if zb_cost > 0:
                    zero_b_patches.append(
                        {
                            "layer": layer_key,
                            "expert_id": int(expert_key) if expert_key.isdigit() else expert_key,
                            "target_attr": f"experts.{expert_key}.{attr}",
                            "repair_kind": "finite_zero_perturbation_quarantine",
                            "new_tensor": w_zero,
                            "modified_elements": zb_cost,
                            "meta": {"exact_zero_delta_quarantine": True},
                        }
                    )
                    zero_b_cost += zb_cost
            elif has_bad:
                w_sanitized = torch.where(bad_mask, torch.zeros_like(w), w)
                san_cost = _count_bitwise_modified(w, w_sanitized)
                if san_cost > 0:
                    zero_b_patches.append(
                        {
                            "layer": layer_key,
                            "expert_id": int(expert_key) if expert_key.isdigit() else expert_key,
                            "target_attr": f"experts.{expert_key}.{attr}",
                            "repair_kind": "finite_zero_perturbation_quarantine",
                            "new_tensor": w_sanitized,
                            "modified_elements": san_cost,
                            "meta": {"sanitized_a_for_finite_quarantine": True},
                        }
                    )
                    zero_b_cost += san_cost

    if cfg.quarantine_strategy == "elementwise":
        if not cfg.enforce_param_budget or elem_cost <= remaining_budget:
            return elem_patches, elem_cost, True
        return [], 0, False

    if cfg.quarantine_strategy == "zero_b":
        if not cfg.enforce_param_budget or zero_b_cost <= remaining_budget:
            return zero_b_patches, zero_b_cost, True
        return [], 0, False

    # strategy == "auto": prefer exact finite zero-perturbation quarantine when within budget,
    # otherwise fall back to minimal elementwise sanitization so sparse bit-flips still heal.
    if not cfg.enforce_param_budget or zero_b_cost <= remaining_budget:
        return zero_b_patches, zero_b_cost, True
    if elem_cost <= remaining_budget:
        return elem_patches, elem_cost, True
    return [], 0, False


@torch.no_grad()
def propose_surgical_block_repairs(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
    health_report: Mapping[str, Any] | None = None,
    *,
    config: RepairProposalConfig | None = None,
    scan_config: AdapterHealthScanConfig | None = None,
    probe_activations: Mapping[Any, Any] | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    healthy_subspaces: Mapping[Any, torch.Tensor] | None = None,
    total_model_params: int | None = None,
    max_edited_param_ratio: float | None = None,
) -> dict[str, Any]:
    """Synthesize sub-0.05% surgical weight-block repair patches from an adapter health report."""
    if config is not None and not isinstance(config, RepairProposalConfig):
        raise TypeError(f"config must be RepairProposalConfig or None, got {type(config).__name__}")
    cfg = config or RepairProposalConfig()
    s_cfg = scan_config or AdapterHealthScanConfig()
    layers, _ = _normalize_moe_layers(moe_layers)
    layer_lookup = dict(layers)

    denom_params = (
        _validate_pos_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else _count_unique_params(moe_layers, layers)
    )
    ratio_cap = (
        _validate_unit_interval(max_edited_param_ratio, "max_edited_param_ratio")
        if max_edited_param_ratio is not None
        else cfg.max_edited_param_ratio
    )
    max_allowed_elements = (
        int(math.floor(ratio_cap * float(denom_params))) if cfg.enforce_param_budget else denom_params
    )

    report = (
        health_report
        if health_report is not None
        else scan_moe_adapter_health(
            moe_layers,
            config=s_cfg,
            probe_activations=probe_activations,
            probe_chunks=probe_chunks,
        )
    )
    if not isinstance(report, Mapping) or "layers" not in report:
        raise ValueError("health_report must be a valid scan_moe_adapter_health report mapping")

    layer_acts = _collect_layer_probe_activations(
        moe_layers,
        layers,
        probe_activations,
        probe_chunks,
        score_from=s_cfg.score_from,
        batch_size=s_cfg.batch_size,
    )

    repairs: list[dict[str, Any]] = []
    skipped_budget: list[dict[str, Any]] = []
    remaining = max_allowed_elements

    # 1. Priority 1: Non-finite / BF16 exponent bit-flips in experts & router rows.
    corrupted_experts: set[tuple[str, str]] = set()
    for item in report.get("non_finite_or_bitflip_adapters", []):
        corrupted_experts.add((str(item["layer"]), str(item["expert_id"])))

    for layer_key, expert_key in sorted(corrupted_experts):
        moe_mlp = layer_lookup[layer_key]
        exp_mod = moe_mlp.experts[expert_key]
        patches, cost, fits = _build_quarantine_patches(exp_mod, layer_key, expert_key, cfg, remaining)
        if fits and patches:
            repairs.extend(patches)
            remaining = max(0, remaining - cost)
        elif not fits:
            skipped_budget.append({"layer": layer_key, "expert_id": expert_key, "reason": "param_budget_exceeded"})

    for item in report.get("corrupted_router_rows", []):
        layer_key = str(item["layer"])
        e_idx = int(item["expert_id"])
        moe_mlp = layer_lookup[layer_key]
        router: nn.Linear = moe_mlp.router
        row_w = router.weight[e_idx]
        bad_w, _, _ = _detect_corruption_mask(
            row_w,
            extreme_threshold=cfg.extreme_weight_threshold,
            jump_multiplier=cfg.isolated_jump_multiplier,
            jump_floor=cfg.isolated_jump_floor,
        )
        w_clean = torch.where(bad_w, torch.zeros_like(row_w), row_w)
        cost_w = _count_bitwise_modified(row_w, w_clean)
        if cost_w > 0:
            if not cfg.enforce_param_budget or cost_w <= remaining:
                repairs.append(
                    {
                        "layer": layer_key,
                        "expert_id": e_idx,
                        "target_attr": "router.weight",
                        "row_index": e_idx,
                        "repair_kind": "elementwise_bitflip_sanitize",
                        "new_tensor": w_clean,
                        "modified_elements": cost_w,
                        "meta": {"router_sanitized": True},
                    }
                )
                remaining = max(0, remaining - cost_w)
            else:
                skipped_budget.append({"layer": layer_key, "expert_id": e_idx, "reason": "param_budget_exceeded"})
        if router.bias is not None:
            bad_b, _, _ = _detect_corruption_mask(
                router.bias[e_idx : e_idx + 1],
                extreme_threshold=cfg.extreme_weight_threshold,
                jump_multiplier=cfg.isolated_jump_multiplier,
                jump_floor=cfg.isolated_jump_floor,
            )
            if bool(bad_b.any().item()):
                b_clean = torch.tensor(0.0, device=router.bias.device, dtype=router.bias.dtype)
                if not cfg.enforce_param_budget or 1 <= remaining:
                    repairs.append(
                        {
                            "layer": layer_key,
                            "expert_id": e_idx,
                            "target_attr": "router.bias",
                            "row_index": e_idx,
                            "repair_kind": "elementwise_bitflip_sanitize",
                            "new_tensor": b_clean,
                            "modified_elements": 1,
                            "meta": {"router_bias_sanitized": True},
                        }
                    )
                    remaining = max(0, remaining - 1)
                else:
                    skipped_budget.append({"layer": layer_key, "expert_id": e_idx, "reason": "param_budget_exceeded"})

    # 2. Priority 2: Saturated and dead sigma_topk router rows.
    router_targets: list[tuple[str, int, str]] = []
    for item in report.get("saturated_router_rows", []):
        router_targets.append((str(item["layer"]), int(item["expert_id"]), "saturated"))
    for item in report.get("dead_router_rows", []):
        router_targets.append((str(item["layer"]), int(item["expert_id"]), "dead"))

    for layer_key, e_idx, r_kind in router_targets:
        moe_mlp = layer_lookup[layer_key]
        h_sub = None
        if healthy_subspaces is not None:
            h_sub = healthy_subspaces.get(layer_key, healthy_subspaces.get(e_idx))
        r_patches = _synthesize_router_recenter_patch(
            moe_mlp,
            layer_key,
            e_idx,
            r_kind,
            layer_acts.get(layer_key, {}),
            h_sub,
            cfg,
        )
        r_cost = sum(int(p["modified_elements"]) for p in r_patches)
        if r_patches and (not cfg.enforce_param_budget or r_cost <= remaining):
            repairs.extend(r_patches)
            remaining = max(0, remaining - r_cost)
        elif r_patches:
            skipped_budget.append({"layer": layer_key, "expert_id": e_idx, "reason": "param_budget_exceeded"})

    # 3. Priority 3: Heavy-tailed spectral spikes & stable-rank collapse via QR-SVD winsorization.
    spectral_targets: set[tuple[str, str, str]] = set()
    for key_name in ("spectral_spike_adapters", "stable_rank_collapsed_adapters"):
        for item in report.get(key_name, []):
            lk, ek, pj = str(item["layer"]), str(item["expert_id"]), str(item["projection"])
            if (lk, ek) not in corrupted_experts:
                spectral_targets.add((lk, ek, pj))

    for layer_key, expert_key, proj in sorted(spectral_targets):
        moe_mlp = layer_lookup[layer_key]
        exp_mod = moe_mlp.experts[expert_key]
        win = _synthesize_winsorized_lora_a(exp_mod, proj, cfg)
        if win is None:
            continue
        a_new, meta = win
        cost = int(meta["modified_elements"])
        if not cfg.enforce_param_budget or cost <= remaining:
            repairs.append(
                {
                    "layer": layer_key,
                    "expert_id": int(expert_key) if expert_key.isdigit() else expert_key,
                    "target_attr": f"experts.{expert_key}.{proj}_A",
                    "repair_kind": "qr_svd_winsorization",
                    "new_tensor": a_new,
                    "modified_elements": cost,
                    "meta": meta,
                }
            )
            remaining = max(0, remaining - cost)
        else:
            skipped_budget.append(
                {"layer": layer_key, "expert_id": expert_key, "projection": proj, "reason": "param_budget_exceeded"}
            )

    total_modified = sum(int(p["modified_elements"]) for p in repairs)
    param_fraction = float(total_modified) / float(denom_params)
    within_budget = (not cfg.enforce_param_budget) or (total_modified <= max_allowed_elements)

    repair_kinds_count: dict[str, int] = {}
    for p in repairs:
        rk = str(p["repair_kind"])
        repair_kinds_count[rk] = repair_kinds_count.get(rk, 0) + 1

    return {
        "repairs": repairs,
        "repair_count": len(repairs),
        "repair_kinds_count": repair_kinds_count,
        "skipped_budget_count": len(skipped_budget),
        "skipped_due_to_budget": skipped_budget,
        "total_modified_elements": total_modified,
        "total_model_params": denom_params,
        "max_allowed_elements": max_allowed_elements,
        "max_edited_param_ratio": ratio_cap,
        "parameter_fraction_modified": param_fraction,
        "within_budget": within_budget,
        "source_state_sha256": report["state_sha256"],
    }


@torch.no_grad()
def apply_surgical_block_repairs_inplace(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a synthesized repair proposal to ``moe_layers`` in-place and return SHA-256 provenance."""
    if not isinstance(proposal, Mapping) or "repairs" not in proposal:
        raise ValueError("proposal must be a mapping containing 'repairs'")
    layers, _ = _normalize_moe_layers(moe_layers)
    layer_lookup = dict(layers)
    before_sha = compute_moe_state_sha256(moe_layers)

    applied_elements = 0
    applied_repairs = 0
    for patch in proposal["repairs"]:
        layer_key = str(patch["layer"])
        if layer_key not in layer_lookup:
            raise ValueError(f"Unknown layer key {layer_key!r} in repair proposal")
        moe_mlp = layer_lookup[layer_key]
        target_attr = str(patch["target_attr"])
        new_tensor = patch["new_tensor"]
        if not isinstance(new_tensor, torch.Tensor):
            raise TypeError("repair patch 'new_tensor' must be a torch.Tensor")

        if target_attr == "router.weight":
            row_idx = int(patch["row_index"])
            target = moe_mlp.router.weight[row_idx]
            applied_elements += _count_bitwise_modified(target, new_tensor)
            target.copy_(new_tensor.to(device=target.device, dtype=target.dtype))
        elif target_attr == "router.bias":
            if moe_mlp.router.bias is None:
                raise ValueError(f"Layer {layer_key!r} router has no bias to patch")
            row_idx = int(patch["row_index"])
            target = moe_mlp.router.bias[row_idx]
            applied_elements += _count_bitwise_modified(target, new_tensor)
            target.copy_(new_tensor.to(device=target.device, dtype=target.dtype))
        elif target_attr.startswith("experts."):
            parts = target_attr.split(".")
            if len(parts) != 3:
                raise ValueError(f"Malformed expert target_attr {target_attr!r}")
            _, expert_key, factor_attr = parts
            exp_mod = moe_mlp.experts[expert_key]
            target = _resolve_factor_tensor(exp_mod, factor_attr)
            applied_elements += _count_bitwise_modified(target, new_tensor)
            target.copy_(new_tensor.to(device=target.device, dtype=target.dtype))
        else:
            raise ValueError(f"Unsupported target_attr {target_attr!r}")
        applied_repairs += 1

    after_sha = compute_moe_state_sha256(moe_layers)
    return {
        "applied_repairs": applied_repairs,
        "modified_elements": applied_elements,
        "before_sha256": before_sha,
        "after_sha256": after_sha,
    }


# ---------------------------------------------------------------------------
# (c) Closed-Loop Self-Rewriting Inspection -> Repair -> Eval -> Gate
# ---------------------------------------------------------------------------


@torch.no_grad()
def _evaluate_model_probe(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
    *,
    probe_chunks: Sequence[Mapping[str, Any]] | None,
    probe_eval_fn: Callable[[Any], Mapping[str, Any]] | None,
    score_from: int,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate per-domain finite NLLs via custom callback or native ``CrowdV1WithEngram`` forward pass."""
    if probe_eval_fn is not None:
        raw = probe_eval_fn(moe_layers)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("probe_eval_fn must return a non-empty mapping")
        if "domain_nlls" in raw:
            dom_map = {str(k): float(v) for k, v in dict(raw["domain_nlls"]).items()}
            nf_outputs = int(raw.get("non_finite_outputs", 0))
        else:
            dom_map = {
                str(k): float(v)
                for k, v in raw.items()
                if str(k) not in {"non_finite_outputs", "nll_mean"}
            }
            nf_outputs = int(raw.get("non_finite_outputs", 0))
        if not dom_map:
            raise ValueError("probe_eval_fn result must contain at least one domain NLL")
        all_finite = nf_outputs == 0 and all(math.isfinite(v) for v in dom_map.values())
        finite_vals = [v for v in dom_map.values() if math.isfinite(v)]
        nll_mean = float(np.mean(finite_vals)) if (all_finite and finite_vals) else None
        return {
            "domain_nlls": {k: (v if math.isfinite(v) else None) for k, v in dom_map.items()},
            "non_finite_outputs": nf_outputs,
            "all_finite": all_finite,
            "nll_mean": nll_mean,
        }

    if probe_chunks is None:
        raise ValueError("Either probe_chunks or probe_eval_fn must be provided")
    if not isinstance(moe_layers, nn.Module) or not hasattr(moe_layers, "blocks") or not hasattr(moe_layers, "ln"):
        raise ValueError("probe_chunks evaluation requires a CrowdV1WithEngram-compatible model")
    if isinstance(probe_chunks, (str, bytes)) or not isinstance(probe_chunks, Sequence) or not probe_chunks:
        raise ValueError("probe_chunks must be a non-empty sequence")

    ids_all = torch.stack([torch.as_tensor(ch["ids"], dtype=torch.long) for ch in probe_chunks], dim=0)
    cos = moe_layers.cos_cached.to(ids_all.device)
    sin = moe_layers.sin_cached.to(ids_all.device)
    was_training = moe_layers.training
    moe_layers.eval()
    dom_names_sorted = sorted({str(ch["expert_id"]) for ch in probe_chunks})
    try:
        try:
            parts: list[torch.Tensor] = []
            for b0 in range(0, ids_all.shape[0], batch_size):
                xb = compute_embeddings_batch(moe_layers, ids_all[b0 : b0 + batch_size])
                for blk in moe_layers.blocks:
                    xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
                    xb = forward_block_post_mlp(blk, xm, mi)
                parts.append(xb)
            h_pre = torch.cat(parts, dim=0)
            nf_states = int((~torch.isfinite(h_pre)).sum().item())
            nlls = per_chunk_nll_from_pre_ln(
                h_pre,
                moe_layers.ln.weight,
                moe_layers.ln.bias,
                float(getattr(moe_layers.ln, "eps", 1e-5)),
                moe_layers.head.weight,
                ids_all,
                score_from=score_from,
            )
        except ValueError as exc:
            if "finite" not in str(exc).lower():
                raise
            return {
                "domain_nlls": {dom: None for dom in dom_names_sorted},
                "non_finite_outputs": int(ids_all.shape[0]),
                "all_finite": False,
                "nll_mean": None,
            }
    finally:
        moe_layers.train(was_training)

    nf_nlls = int((~torch.isfinite(nlls)).sum().item())
    nlls_list = [float(x) for x in nlls.detach().cpu().tolist()]
    grouped: dict[str, list[float]] = {}
    for ch, val in zip(probe_chunks, nlls_list):
        grouped.setdefault(str(ch["expert_id"]), []).append(val)

    dom_nlls: dict[str, float | None] = {}
    for dom, vals in sorted(grouped.items()):
        dom_nlls[dom] = float(np.mean(vals)) if all(math.isfinite(v) for v in vals) else None

    all_finite = (nf_states == 0) and (nf_nlls == 0) and all(v is not None for v in dom_nlls.values())
    nll_mean = float(np.mean(nlls_list)) if all_finite else None
    return {
        "domain_nlls": dom_nlls,
        "non_finite_outputs": nf_states + nf_nlls,
        "all_finite": all_finite,
        "nll_mean": nll_mean,
    }


def _compute_drift_and_energy(
    eval_res: Mapping[str, Any],
    ref_nlls: Mapping[str, float | None],
    health_report: Mapping[str, Any],
    cfg: SelfHealingLoopConfig,
) -> dict[str, Any]:
    """Compute per-domain NLL drift, general-domain drift, guard status, and ANSE energy."""
    dom_nlls: Mapping[str, float | None] = eval_res["domain_nlls"]
    all_finite = bool(eval_res["all_finite"])
    per_domain_drift: dict[str, float | None] = {}
    general_drifts: list[float] = []

    for dom, cand_val in dom_nlls.items():
        ref_val = ref_nlls.get(dom)
        if cand_val is None or not math.isfinite(float(cand_val)):
            per_domain_drift[dom] = None
            continue
        if ref_val is None or not math.isfinite(float(ref_val)):
            # Recovering from non-finite baseline to a finite NLL yields -1.0 (-100% pain recovery).
            drift = -1.0
        else:
            drift = (float(cand_val) - float(ref_val)) / max(abs(float(ref_val)), 1e-8)
        per_domain_drift[dom] = drift
        if dom.startswith(cfg.general_domain_prefix):
            general_drifts.append(drift)

    finite_drifts = [d for d in per_domain_drift.values() if d is not None]
    if not all_finite or len(finite_drifts) != len(dom_nlls):
        worst_drift: float | None = None
        general_drift: float | None = None
        guard_ok = False
        anse_energy = DEFAULT_ANSE_MAX_PAIN
    else:
        worst_drift = max(finite_drifts) if finite_drifts else 0.0
        general_drift = max(general_drifts) if general_drifts else worst_drift
        guard_ok = worst_drift <= cfg.max_domain_drift
        if not guard_ok:
            anse_energy = DEFAULT_ANSE_MAX_PAIN
        else:
            nll_m = float(eval_res["nll_mean"])
            soft_pen = 0.05 if worst_drift > cfg.anse_safe_drift_threshold else 0.0
            h_pen = cfg.health_penalty_weight * float(health_report.get("health_severity", 0.0))
            anse_energy = nll_m + soft_pen + h_pen

    return {
        "per_domain_drift": per_domain_drift,
        "worst_domain_drift": worst_drift,
        "general_domain_drift": general_drift,
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
    }


@torch.no_grad()
def execute_self_rewriting_repair_loop(
    moe_layers: nn.Module | Sequence[nn.Module] | Mapping[int | str, nn.Module],
    *,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    probe_eval_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    reference_domain_nlls: Mapping[str, float] | None = None,
    probe_activations: Mapping[Any, Any] | None = None,
    healthy_subspaces: Mapping[Any, torch.Tensor] | None = None,
    scan_config: AdapterHealthScanConfig | None = None,
    proposal_config: RepairProposalConfig | None = None,
    loop_config: SelfHealingLoopConfig | None = None,
    total_model_params: int | None = None,
    candidate_proposal_fn: Callable[[Any, Mapping[str, Any], int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute closed-loop inspect -> propose -> probe eval -> accept/rollback with SHA-256 provenance."""
    if loop_config is not None and not isinstance(loop_config, SelfHealingLoopConfig):
        raise TypeError(f"loop_config must be SelfHealingLoopConfig or None, got {type(loop_config).__name__}")
    l_cfg = loop_config or SelfHealingLoopConfig()
    s_cfg = scan_config or AdapterHealthScanConfig(score_from=l_cfg.score_from, batch_size=l_cfg.batch_size)
    p_cfg = proposal_config or RepairProposalConfig()

    layers, _ = _normalize_moe_layers(moe_layers)
    denom_params = (
        _validate_pos_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else _count_unique_params(moe_layers, layers)
    )
    budget_limit = int(math.floor(p_cfg.max_edited_param_ratio * float(denom_params)))

    initial_sha = compute_moe_state_sha256(moe_layers)
    parent_health = scan_moe_adapter_health(
        moe_layers, config=s_cfg, probe_activations=probe_activations, probe_chunks=probe_chunks
    )
    parent_eval = _evaluate_model_probe(
        moe_layers,
        probe_chunks=probe_chunks,
        probe_eval_fn=probe_eval_fn,
        score_from=l_cfg.score_from,
        batch_size=l_cfg.batch_size,
    )
    ref_nlls: dict[str, float | None] = (
        {str(k): _validate_finite_float(v, f"reference_domain_nlls[{k!r}]") for k, v in reference_domain_nlls.items()}
        if reference_domain_nlls is not None
        else dict(parent_eval["domain_nlls"])
    )
    parent_metrics = _compute_drift_and_energy(parent_eval, ref_nlls, parent_health, l_cfg)
    initial_health = parent_health
    initial_eval = parent_eval
    initial_energy = float(parent_metrics["anse_energy"])

    current_sha = initial_sha
    cumulative_modified = 0
    iterations: list[dict[str, Any]] = []
    accepted_count = 0
    rolled_back_count = 0
    termination_reason = "max_iterations_reached"

    for it in range(l_cfg.max_iterations):
        if it > 0:
            parent_health = scan_moe_adapter_health(
                moe_layers, config=s_cfg, probe_activations=probe_activations, probe_chunks=probe_chunks
            )
        if parent_health["healthy"] and candidate_proposal_fn is None:
            termination_reason = "converged_healthy"
            break

        if candidate_proposal_fn is not None:
            proposal = candidate_proposal_fn(moe_layers, parent_health, it)
        else:
            rem_ratio = (
                max(0.0, float(budget_limit - cumulative_modified) / float(denom_params))
                if p_cfg.enforce_param_budget
                else p_cfg.max_edited_param_ratio
            )
            proposal = propose_surgical_block_repairs(
                moe_layers,
                parent_health,
                config=p_cfg,
                scan_config=s_cfg,
                probe_activations=probe_activations,
                probe_chunks=probe_chunks,
                healthy_subspaces=healthy_subspaces,
                total_model_params=denom_params,
                max_edited_param_ratio=rem_ratio,
            )

        if not proposal.get("repairs"):
            termination_reason = (
                "budget_exhausted" if proposal.get("skipped_budget_count", 0) > 0 else "no_repairs_proposed"
            )
            break

        before_sha = current_sha
        snap = snapshot_moe_state(moe_layers)
        apply_res = apply_surgical_block_repairs_inplace(moe_layers, proposal)
        candidate_sha = str(apply_res["after_sha256"])

        child_health = scan_moe_adapter_health(
            moe_layers, config=s_cfg, probe_activations=probe_activations, probe_chunks=probe_chunks
        )
        child_eval = _evaluate_model_probe(
            moe_layers,
            probe_chunks=probe_chunks,
            probe_eval_fn=probe_eval_fn,
            score_from=l_cfg.score_from,
            batch_size=l_cfg.batch_size,
        )
        child_metrics = _compute_drift_and_energy(child_eval, ref_nlls, child_health, l_cfg)

        e_parent = float(parent_metrics["anse_energy"])
        e_child = float(child_metrics["anse_energy"])
        delta_e = e_child - e_parent
        mod_elems = int(apply_res["modified_elements"])
        budget_ok = (not p_cfg.enforce_param_budget) or ((cumulative_modified + mod_elems) <= budget_limit)
        accepted = bool(child_eval["all_finite"] and child_metrics["guard_ok"] and (delta_e < 0.0) and budget_ok)

        if accepted:
            accepted_count += 1
            cumulative_modified += mod_elems
            current_sha = candidate_sha
            parent_eval = child_eval
            parent_metrics = child_metrics
            parent_health = child_health
            iterations.append(
                {
                    "iteration": it,
                    "status": "accepted",
                    "rolled_back": False,
                    "before_sha256": before_sha,
                    "candidate_sha256": candidate_sha,
                    "final_sha256": candidate_sha,
                    "e_parent": e_parent,
                    "e_child": e_child,
                    "delta_e": delta_e,
                    "worst_domain_drift": child_metrics["worst_domain_drift"],
                    "general_domain_drift": child_metrics["general_domain_drift"],
                    "guard_ok": child_metrics["guard_ok"],
                    "modified_elements": mod_elems,
                    "parameter_fraction_modified": float(mod_elems) / float(denom_params),
                    "repair_kinds_count": proposal.get("repair_kinds_count", {}),
                }
            )
        else:
            rolled_back_count += 1
            restored_sha = restore_moe_state_inplace(moe_layers, snap)
            bitwise_verified = restored_sha == before_sha
            if not bitwise_verified:
                raise RuntimeError(
                    f"Atomic bitwise rollback SHA-256 mismatch: expected {before_sha}, got {restored_sha}"
                )
            reject_reason = (
                "non_finite_child_output"
                if not child_eval["all_finite"]
                else (
                    "domain_guard_exceeded"
                    if not child_metrics["guard_ok"]
                    else ("param_budget_exceeded" if not budget_ok else "non_negative_delta_energy")
                )
            )
            iterations.append(
                {
                    "iteration": it,
                    "status": "rolled_back",
                    "rolled_back": True,
                    "rollback_bitwise_verified": bitwise_verified,
                    "reject_reason": reject_reason,
                    "before_sha256": before_sha,
                    "candidate_sha256": candidate_sha,
                    "restored_sha256": restored_sha,
                    "final_sha256": restored_sha,
                    "e_parent": e_parent,
                    "e_child": e_child,
                    "delta_e": delta_e,
                    "worst_domain_drift": child_metrics["worst_domain_drift"],
                    "general_domain_drift": child_metrics["general_domain_drift"],
                    "guard_ok": child_metrics["guard_ok"],
                    "modified_elements": 0,
                    "rejected_candidate_elements": mod_elems,
                }
            )
            termination_reason = f"rolled_back:{reject_reason}"
            break

    final_sha = compute_moe_state_sha256(moe_layers)
    final_health = parent_health
    final_energy = float(parent_metrics["anse_energy"])
    status = (
        "accepted_and_repaired"
        if accepted_count > 0
        else ("rolled_back" if rolled_back_count > 0 else ("already_healthy" if initial_health["healthy"] else "no_change"))
    )

    return {
        "status": status,
        "termination_reason": termination_reason,
        "initial_sha256": initial_sha,
        "final_sha256": final_sha,
        "initial_anse_energy": initial_energy,
        "final_anse_energy": final_energy,
        "delta_anse_energy": final_energy - initial_energy,
        "initial_nll_mean": initial_eval["nll_mean"],
        "final_nll_mean": parent_eval["nll_mean"],
        "initial_all_finite": initial_eval["all_finite"],
        "final_all_finite": parent_eval["all_finite"],
        "accepted_iterations": accepted_count,
        "rolled_back_iterations": rolled_back_count,
        "total_modified_elements": cumulative_modified,
        "total_model_params": denom_params,
        "parameter_fraction_modified": float(cumulative_modified) / float(denom_params),
        "within_budget": (not p_cfg.enforce_param_budget) or (cumulative_modified <= budget_limit),
        "worst_domain_drift": parent_metrics["worst_domain_drift"],
        "general_domain_drift": parent_metrics["general_domain_drift"],
        "guard_ok": parent_metrics["guard_ok"],
        "initial_domain_nlls": initial_eval["domain_nlls"],
        "final_domain_nlls": parent_eval["domain_nlls"],
        "initial_health": initial_health,
        "final_health": final_health,
        "iterations": iterations,
        "loop_config": asdict(l_cfg),
    }


__all__ = [
    "AdapterHealthScanConfig",
    "DEFAULT_ANSE_MAX_PAIN",
    "DEFAULT_DEAD_ROUTER_TARGET_LOGIT",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_EXTREME_WEIGHT_THRESHOLD",
    "DEFAULT_ISOLATED_JUMP_FLOOR",
    "DEFAULT_ISOLATED_JUMP_MULTIPLIER",
    "DEFAULT_MAX_DOMAIN_DRIFT",
    "DEFAULT_MAX_EDITED_PARAM_RATIO",
    "DEFAULT_MIN_SINGULAR_VALUE",
    "DEFAULT_MIN_STABLE_RANK",
    "DEFAULT_ROUTER_DEAD_GATE_THRESHOLD",
    "DEFAULT_ROUTER_NORM_OUTLIER_RATIO",
    "DEFAULT_ROUTER_SATURATED_GATE_THRESHOLD",
    "DEFAULT_ROUTER_SATURATED_WEIGHT_NORM",
    "DEFAULT_ROUTER_SATURATION_LOGIT_THRESHOLD",
    "DEFAULT_ROUTER_TARGET_OFF_LOGIT",
    "DEFAULT_SINGLE_MODE_SPIKE_NORM",
    "DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD",
    "DEFAULT_WINSOR_CAP_MULTIPLIER",
    "DEFAULT_WINSOR_TRIGGER_MODE_RATIO",
    "GENERAL_EXPERT_PREFIX",
    "RepairProposalConfig",
    "SelfHealingLoopConfig",
    "apply_surgical_block_repairs_inplace",
    "compute_moe_state_sha256",
    "execute_self_rewriting_repair_loop",
    "propose_surgical_block_repairs",
    "restore_moe_state_inplace",
    "scan_moe_adapter_health",
    "snapshot_moe_state",
]
