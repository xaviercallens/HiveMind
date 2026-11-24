# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Causal Logit-Lens Layer-Localized and Leave-One-Expert-Out (LOEO) Attribution for Open-Weight MoEs.

Implements training-free causal attribution primitives for upcycled ``CrowdV1WithEngram`` +
``CrowdLoRAMoEMLP`` (``sigma_topk`` / ``softmax_topk`` SwiGLU rank-r adapters), inspired by
Takehi et al. (Weight-Space Programming, Nov 2026):

1. ``compute_leave_one_expert_out_attribution``:
   Ablates each expert's LoRA ``B`` matrices (``w1_B = w2_B = w3_B = 0 => dy_e(x) == 0``) and
   measures exact causal mid-row delta NLL on general vs specialist domains, classifying:
   - harmful off-target experts (Regime-A ``expert_0`` or leaking off-domain ghost sinks),
   - helpful general anchors (Regime-B ``expert_0``, preserved and protected),
   - specialist experts with off-target general interference (preserved for AlphaEdit / router surgery),
   - high-norm unscored rogue experts (e.g., ``expert_3`` with extreme ``||w1_B||`` and 0 probe chunks).
2. ``compute_per_layer_moe_bypass_attribution``:
   Ablates MoE deltas layer-by-layer against the frozen dense ``base_mlp`` to rank exact causal
   general-domain harm and specialist-domain benefit per layer, paired with intermediate
   logit-lens NLL deltas and MoE residual RMS norms.
3. ``select_surgical_layer_mask``:
   Localizes weight edits to the top-K harmed mid/late layers above ``min_harm_nats`` while
   protecting early shared-syntax layers below ``protect_early_layers_below``.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
)

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_GENERAL_EXPERT_INDEX: int = 0
DEFAULT_SCORE_FROM: int = 64
DEFAULT_LOEO_E0_HARM_THRESHOLD: float = -0.0015
DEFAULT_LOEO_GENERAL_HELP_THRESHOLD: float = 1e-4
DEFAULT_LOEO_MAX_SPEC_LOSS_INCREASE: float = 0.03
DEFAULT_ROGUE_B_NORM_THRESHOLD: float = 10.0
DEFAULT_ROGUE_NORM_RATIO_THRESHOLD: float = 4.0
DEFAULT_ROGUE_MIN_NORM_FLOOR: float = 1.0

_EXPERT_PREFIX_RE = re.compile(r"^E(\d+)(?:_|$)")


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding bool)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_model(model: nn.Module) -> None:
    """Verify required CrowdV1-compatible transformer attributes on ``model``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")


def _resolve_cal_ln_params(
    model: nn.Module,
    cal_ln_w: torch.Tensor | None,
    cal_ln_b: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Validate and return CPU float32 final LayerNorm weight, bias, and epsilon."""
    if (cal_ln_w is None) != (cal_ln_b is None):
        raise ValueError("cal_ln_w and cal_ln_b must either both be provided or both be None")
    h_dim = int(model.ln.weight.numel())
    w_src = model.ln.weight if cal_ln_w is None else cal_ln_w
    b_src = model.ln.bias if cal_ln_b is None else cal_ln_b
    if not isinstance(w_src, torch.Tensor) or not isinstance(b_src, torch.Tensor):
        raise TypeError("cal_ln_w and cal_ln_b must be torch.Tensor instances")
    if w_src.shape != (h_dim,) or b_src.shape != (h_dim,):
        raise ValueError(f"Expected 1D LN params of length {h_dim}, got {tuple(w_src.shape)} and {tuple(b_src.shape)}")
    w_f32 = w_src.detach().to("cpu", dtype=torch.float32)
    b_f32 = b_src.detach().to("cpu", dtype=torch.float32)
    if not bool(torch.isfinite(w_f32).all().item()) or not bool(torch.isfinite(b_f32).all().item()):
        raise ValueError("cal_ln_w and cal_ln_b must contain only finite values")
    ln_eps = _validate_nonneg_finite(float(getattr(model.ln, "eps", 1e-5)), "ln_eps")
    if ln_eps <= 0.0:
        raise ValueError(f"ln_eps must be > 0, got {ln_eps}")
    return w_f32, b_f32, ln_eps


def _resolve_bucket_io(
    buckets: Mapping[str, torch.Tensor],
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None,
) -> tuple[Callable[[nn.Module, Mapping[str, torch.Tensor]], None], Callable[[nn.Module], dict[str, torch.Tensor]]]:
    """Bind default OpenDiLoCo bucket loader/extractor when custom callbacks are omitted."""
    nfrag = max(1, len([k for k in buckets if k.startswith("router_frag_")]))
    loader = load_buckets_fn or (lambda m, b: load_moe_trainable_buckets(m, b, num_router_fragments=nfrag))
    extractor = extract_buckets_fn or (lambda m: extract_moe_trainable_buckets(m, num_router_fragments=nfrag))
    return loader, extractor


def _parse_expert_domain_index(expert_id: str) -> int | None:
    """Extract numeric expert index from domain identifiers such as ``E3_formal_logic_arc``."""
    match = _EXPERT_PREFIX_RE.match(expert_id.strip())
    return int(match.group(1)) if match is not None else None


def _prepare_probe_chunks(
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate probe chunks and return ``(ids_2d, is_general_mask, expert_id_strings)``."""
    if not isinstance(general_prefix, str) or not general_prefix.strip():
        raise ValueError("general_prefix must be a non-empty string")
    if isinstance(probe_chunks, (str, bytes)) or not isinstance(probe_chunks, Sequence) or len(probe_chunks) == 0:
        raise ValueError("probe_chunks must be a non-empty sequence of chunk mappings")

    id_rows: list[torch.Tensor] = []
    expert_ids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(probe_chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"probe_chunks[{idx}] must be a mapping with 'expert_id' and 'ids'")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"probe_chunks[{idx}]['ids'] must be 1D with length >= 2, got shape {tuple(ids_t.shape)}")
        if bool((ids_t < 0).any().item()):
            raise ValueError(f"probe_chunks[{idx}]['ids'] must contain non-negative token IDs")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All probe chunks must share sequence length {seq_len}, got {ids_t.numel()} at {idx}")
        id_rows.append(ids_t)
        expert_ids.append(str(ch["expert_id"]))

    ids_all = torch.stack(id_rows, dim=0)
    is_general = np.array([eid.startswith(general_prefix) for eid in expert_ids], dtype=bool)
    if not bool(np.any(is_general)):
        raise ValueError(f"probe_chunks must contain at least one general chunk starting with {general_prefix!r}")
    return ids_all, is_general, expert_ids


@torch.no_grad()
def compute_embeddings_batch(model: nn.Module, ids_2d: torch.Tensor) -> torch.Tensor:
    """Compute initial ``wte + engram`` float32 sequence states compatible with ``CrowdV1WithEngram``."""
    x = model.wte(ids_2d).to(torch.float32)
    if getattr(getattr(model, "cfg", None), "engram_buckets", 0) > 0:
        x = x + model.engram(ids_2d).to(device=x.device, dtype=torch.float32)
    return x


@torch.no_grad()
def forward_block_pre_mlp(
    blk: nn.Module,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``x_mid = x + attn(ln1(x))`` and return ``(x_mid, ln2(x_mid))`` in ``float32``."""
    blk_dtype = blk.ln1.weight.dtype
    x_in = x.to(dtype=blk_dtype)
    h, _ = blk.attn(blk.ln1(x_in), cos, sin, past_kv=None, use_cache=False)
    x_mid = x_in + h
    mlp_in = blk.ln2(x_mid)
    return x_mid.to(torch.float32), mlp_in.to(torch.float32)


@torch.no_grad()
def forward_block_dense_base_mlp(blk: nn.Module, mlp_in: torch.Tensor) -> torch.Tensor:
    """Evaluate the frozen dense MLP (zero MoE delta) at ``blk`` in ``float32``."""
    blk_dtype = blk.ln1.weight.dtype
    base_mod = getattr(blk.mlp, "base_mlp", None)
    if base_mod is None:
        if (
            hasattr(blk.mlp, "w1")
            and hasattr(blk.mlp, "w2")
            and hasattr(blk.mlp, "w3")
            and not hasattr(blk.mlp, "experts")
        ):
            base_mod = blk.mlp
        else:
            raise ValueError("Block mlp must expose 'base_mlp' or dense SwiGLU weights (w1, w2, w3)")

    if hasattr(base_mod, "w1") and hasattr(base_mod, "w2") and hasattr(base_mod, "w3"):
        w_dt = base_mod.w1.weight.dtype
        x_base = mlp_in.to(dtype=w_dt)
        h1 = base_mod.w1(x_base).to(torch.float32)
        h2 = base_mod.w2(x_base).to(torch.float32)
        return base_mod.w3((F.silu(h1) * h2).to(dtype=w_dt)).to(torch.float32)
    return base_mod(mlp_in.to(dtype=blk_dtype)).to(torch.float32)


@torch.no_grad()
def forward_block_post_mlp(blk: nn.Module, x_mid: torch.Tensor, mlp_in: torch.Tensor) -> torch.Tensor:
    """Finish transformer block ``x_out = x_mid + mlp(mlp_in)`` in ``float32``."""
    blk_dtype = blk.ln1.weight.dtype
    mlp_out = blk.mlp(mlp_in.to(dtype=blk_dtype)).to(torch.float32)
    return (x_mid + mlp_out).to(torch.float32)


@torch.no_grad()
def per_chunk_causal_midrow_nll(
    h_pre_ln: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    ln_eps: float,
    head_weight: torch.Tensor,
    ids_2d: torch.Tensor,
    *,
    score_from: int = DEFAULT_SCORE_FROM,
    seq_batch_size: int = 8,
) -> torch.Tensor:
    """Compute exact causal mid-row NLL over positions ``score_from - 1 .. T - 2`` predicting ``score_from .. T - 1``."""
    if not isinstance(h_pre_ln, torch.Tensor) or not isinstance(ids_2d, torch.Tensor):
        raise TypeError("h_pre_ln and ids_2d must be torch.Tensor instances")
    if h_pre_ln.ndim != 3 or ids_2d.ndim != 2:
        raise ValueError(
            f"Expected 3D h_pre_ln and 2D ids_2d, got {tuple(h_pre_ln.shape)} and {tuple(ids_2d.shape)}"
        )
    n_chunks, seq_len, hidden_dim = h_pre_ln.shape
    if n_chunks == 0 or ids_2d.shape != (n_chunks, seq_len):
        raise ValueError(f"Mismatched h_pre_ln {tuple(h_pre_ln.shape)} and ids_2d {tuple(ids_2d.shape)}")
    if not bool(torch.isfinite(h_pre_ln).all().item()):
        raise ValueError("h_pre_ln must contain only finite values")
    if ln_weight.shape != (hidden_dim,) or ln_bias.shape != (hidden_dim,):
        raise ValueError(f"Expected 1D LN params of length {hidden_dim}")
    if head_weight.ndim != 2 or head_weight.shape[1] != hidden_dim:
        raise ValueError(f"Expected 2D head_weight with width {hidden_dim}, got {tuple(head_weight.shape)}")
    vocab = int(head_weight.shape[0])
    if bool((ids_2d < 0).any().item()) or bool((ids_2d >= vocab).any().item()):
        raise ValueError(f"ids_2d token IDs must lie in [0, {vocab})")
    eps_f = _validate_nonneg_finite(ln_eps, "ln_eps")
    if eps_f <= 0.0:
        raise ValueError(f"ln_eps must be > 0, got {ln_eps}")

    offset = _validate_nonneg_int(score_from, "score_from")
    start_pos = max(0, offset - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(f"Sequence length {seq_len} too short for causal start_pos={start_pos}")
    bsz = _validate_positive_int(seq_batch_size, "seq_batch_size")

    ln_w = ln_weight.detach().to(device=h_pre_ln.device, dtype=torch.float32)
    ln_b = ln_bias.detach().to(device=h_pre_ln.device, dtype=torch.float32)
    hw_dt = head_weight.dtype if head_weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    hw = head_weight.detach().to(device=h_pre_ln.device, dtype=hw_dt)

    out_nlls = torch.empty(n_chunks, dtype=torch.float32, device=h_pre_ln.device)
    for c0 in range(0, n_chunks, bsz):
        c1 = min(n_chunks, c0 + bsz)
        sub_h = h_pre_ln[c0:c1, start_pos:-1, :].to(torch.float32)
        h_norm = F.layer_norm(sub_h, (hidden_dim,), ln_w, ln_b, eps=eps_f)
        targets = ids_2d[c0:c1, start_pos + 1 :]
        logits = F.linear(h_norm.to(dtype=hw_dt), hw).to(torch.float32)
        tok_loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none")
        out_nlls[c0:c1] = tok_loss.view(c1 - c0, -1).mean(dim=1)
    return out_nlls


@torch.no_grad()
def zero_expert_lora_b_inplace(moe_mlp: nn.Module, expert_indices: Sequence[int]) -> int:
    """Zero ``w1_B = w2_B = w3_B = 0`` for ``expert_indices`` on a MoE MLP (bitwise ``dy == 0``)."""
    if not isinstance(moe_mlp, nn.Module):
        raise TypeError(f"moe_mlp must be an nn.Module, got {type(moe_mlp).__name__}")
    experts_dict = getattr(moe_mlp, "experts", None)
    if experts_dict is None or not hasattr(experts_dict, "__contains__"):
        raise ValueError("moe_mlp must expose an 'experts' mapping")

    unique_indices = sorted({_validate_nonneg_int(idx, "expert_index") for idx in expert_indices})
    zeroed = 0
    for e_idx in unique_indices:
        key = str(e_idx)
        if key not in experts_dict:
            continue
        exp = experts_dict[key]
        for attr in ("w1_B", "w2_B", "w3_B"):
            param = getattr(exp, attr, None)
            if isinstance(param, torch.Tensor):
                param.zero_()
                zeroed += param.numel()
    return zeroed


@torch.no_grad()
def compute_expert_weight_norms(model: nn.Module) -> dict[int, dict[str, Any]]:
    """Aggregate per-expert LoRA ``B`` Frobenius norms and router row norms across all MoE blocks."""
    _validate_model(model)
    accum: dict[int, dict[str, Any]] = {}

    for l_idx, blk in enumerate(model.blocks):
        mlp = getattr(blk, "mlp", None)
        experts_dict = getattr(mlp, "experts", None)
        if experts_dict is None or not hasattr(experts_dict, "items"):
            continue
        router_w = getattr(getattr(mlp, "router", None), "weight", None)
        for e_key, exp_mod in experts_dict.items():
            e_idx = _validate_nonneg_int(int(e_key), "expert_key")
            entry = accum.setdefault(
                e_idx,
                {
                    "sq_w1": 0.0,
                    "sq_w2": 0.0,
                    "sq_w3": 0.0,
                    "sq_router": 0.0,
                    "max_layer_w1_b_norm": 0.0,
                    "max_layer_b_norm": 0.0,
                    "max_layer_w1_b_layer": l_idx,
                    "per_layer_w1_b_norms": {},
                },
            )
            layer_sq: dict[str, float] = {}
            for attr in ("w1_B", "w2_B", "w3_B"):
                param = getattr(exp_mod, attr, None)
                sq_val = float(param.detach().to(torch.float64).pow(2).sum().item()) if isinstance(param, torch.Tensor) else 0.0
                layer_sq[attr] = sq_val
            entry["sq_w1"] += layer_sq["w1_B"]
            entry["sq_w2"] += layer_sq["w2_B"]
            entry["sq_w3"] += layer_sq["w3_B"]

            w1_norm_l = math.sqrt(layer_sq["w1_B"])
            b_norm_l = math.sqrt(layer_sq["w1_B"] + layer_sq["w2_B"] + layer_sq["w3_B"])
            entry["per_layer_w1_b_norms"][l_idx] = w1_norm_l
            if w1_norm_l > entry["max_layer_w1_b_norm"]:
                entry["max_layer_w1_b_norm"] = w1_norm_l
                entry["max_layer_w1_b_layer"] = l_idx
            if b_norm_l > entry["max_layer_b_norm"]:
                entry["max_layer_b_norm"] = b_norm_l

            if isinstance(router_w, torch.Tensor) and e_idx < int(router_w.shape[0]):
                entry["sq_router"] += float(router_w[e_idx].detach().to(torch.float64).pow(2).sum().item())

    if not accum:
        raise ValueError("model must contain at least one MoE block with an 'experts' mapping")

    out: dict[int, dict[str, Any]] = {}
    for e_idx in sorted(accum.keys()):
        raw = accum[e_idx]
        w1_n = math.sqrt(raw["sq_w1"])
        w2_n = math.sqrt(raw["sq_w2"])
        w3_n = math.sqrt(raw["sq_w3"])
        tot_n = math.sqrt(raw["sq_w1"] + raw["sq_w2"] + raw["sq_w3"])
        out[e_idx] = {
            "w1_b_norm": w1_n,
            "w2_b_norm": w2_n,
            "w3_b_norm": w3_n,
            "total_b_norm": tot_n,
            "max_layer_w1_b_norm": float(raw["max_layer_w1_b_norm"]),
            "max_layer_b_norm": float(raw["max_layer_b_norm"]),
            "max_layer_w1_b_layer": int(raw["max_layer_w1_b_layer"]),
            "router_row_norm": math.sqrt(raw["sq_router"]),
            "per_layer_w1_b_norms": raw["per_layer_w1_b_norms"],
        }
    return out


def _snapshot_and_zero_expert_b(model: nn.Module, expert_idx: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot and zero ``w1_B, w2_B, w3_B`` for ``expert_idx`` across all MoE blocks."""
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    key = str(expert_idx)
    for blk in model.blocks:
        mlp = getattr(blk, "mlp", None)
        experts_dict = getattr(mlp, "experts", None)
        if experts_dict is None or key not in experts_dict:
            continue
        exp = experts_dict[key]
        for attr in ("w1_B", "w2_B", "w3_B"):
            param = getattr(exp, attr, None)
            if isinstance(param, torch.Tensor):
                saved.append((param, param.detach().clone()))
        zero_expert_lora_b_inplace(mlp, [expert_idx])
    return saved


def _restore_expert_b(saved: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Restore snapshotted LoRA ``B`` tensors in-place."""
    for param, backup in saved:
        param.copy_(backup)


@torch.no_grad()
def _forward_model_pre_ln(model: nn.Module, ids_2d: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Run all transformer blocks and return pre-LN hidden states in ``float32``."""
    cos = model.cos_cached.to(ids_2d.device)
    sin = model.sin_cached.to(ids_2d.device)
    parts: list[torch.Tensor] = []
    for b0 in range(0, ids_2d.shape[0], batch_size):
        xb = compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size])
        for blk in model.blocks:
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            xb = forward_block_post_mlp(blk, xm, mi)
        parts.append(xb)
    return torch.cat(parts, dim=0)


def _select_balanced_chunk_indices(
    expert_ids: Sequence[str],
    *,
    general_prefix: str,
    cap_per_domain: int,
) -> list[int]:
    """Select deterministic balanced indices across domains for LOEO attribution."""
    selected: list[int] = []
    seen_per_dom: dict[str, int] = {}
    for idx, dom in enumerate(expert_ids):
        cap = cap_per_domain * 2 if dom.startswith(general_prefix) else cap_per_domain
        if seen_per_dom.get(dom, 0) < cap:
            selected.append(idx)
            seen_per_dom[dom] = seen_per_dom.get(dom, 0) + 1
    return selected


def _classify_loeo_expert(
    *,
    e_idx: int,
    general_expert_index: int,
    delta_e0: float,
    delta_spec: float,
    is_unscored_rogue: bool,
    harm_thresh: float,
    help_thresh: float,
    max_spec_inc: float,
) -> str:
    """Classify an expert from LOEO general/specialist NLL deltas and rogue norm status."""
    if delta_e0 < harm_thresh:
        if delta_spec <= max_spec_inc:
            return "rogue_unscored_high_norm" if is_unscored_rogue else "harmful_off_target"
        return "specialist_with_general_interference"
    if is_unscored_rogue:
        return "rogue_unscored_high_norm"
    if delta_e0 > help_thresh and (e_idx == general_expert_index or delta_spec <= max_spec_inc):
        return "helpful_general"
    if delta_spec > max_spec_inc:
        return "helpful_specialist"
    return "neutral"


def _setup_attribution_run(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    buckets: Mapping[str, torch.Tensor] | None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None,
    cal_ln_w: torch.Tensor | None,
    cal_ln_b: torch.Tensor | None,
    score_from: int,
    batch_size: int,
    general_prefix: str,
) -> tuple[torch.Tensor, np.ndarray, list[str], torch.Tensor, torch.Tensor, float, int, int]:
    """Validate inputs, optionally load OpenDiLoCo buckets, and resolve probe & LN state."""
    _validate_model(model)
    ids_all, is_general, expert_id_strs = _prepare_probe_chunks(probe_chunks, general_prefix=general_prefix)
    bsz = _validate_positive_int(batch_size, "batch_size")
    offset = _validate_nonneg_int(score_from, "score_from")
    if buckets is not None:
        if not isinstance(buckets, Mapping):
            raise TypeError(f"buckets must be a mapping, got {type(buckets).__name__}")
        loader, _ = _resolve_bucket_io(buckets, load_buckets_fn, None)
        loader(model, {k: v.clone() for k, v in buckets.items()})
    cal_w, cal_b, ln_eps = _resolve_cal_ln_params(model, cal_ln_w, cal_ln_b)
    return ids_all, is_general, expert_id_strs, cal_w, cal_b, ln_eps, offset, bsz


@torch.no_grad()
def compute_leave_one_expert_out_attribution(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    buckets: Mapping[str, torch.Tensor] | None = None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    cal_ln_w: torch.Tensor | None = None,
    cal_ln_b: torch.Tensor | None = None,
    score_from: int = DEFAULT_SCORE_FROM,
    loeo_chunks_per_domain: int = 8,
    batch_size: int = 24,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    general_expert_index: int = DEFAULT_GENERAL_EXPERT_INDEX,
    scored_expert_indices: Sequence[int] | None = None,
    loeo_e0_harm_threshold: float = DEFAULT_LOEO_E0_HARM_THRESHOLD,
    loeo_general_help_threshold: float = DEFAULT_LOEO_GENERAL_HELP_THRESHOLD,
    loeo_max_spec_loss_increase: float = DEFAULT_LOEO_MAX_SPEC_LOSS_INCREASE,
    rogue_b_norm_threshold: float = DEFAULT_ROGUE_B_NORM_THRESHOLD,
    rogue_norm_ratio_threshold: float = DEFAULT_ROGUE_NORM_RATIO_THRESHOLD,
    rogue_min_norm_floor: float = DEFAULT_ROGUE_MIN_NORM_FLOOR,
) -> dict[str, Any]:
    """Measure per-expert causal delta NLL on general vs specialist domains when expert ``B`` matrices are zeroed.

    Simultaneously inspects per-expert ``w1_B/w2_B/w3_B`` Frobenius norms against domain coverage in
    ``probe_chunks`` to flag high-norm unscored rogue experts (e.g., ``expert_3`` with ``||w1_B|| = 42.5``
    and zero probe chunks) and distinguishes Regime-A harmful ``expert_0`` from Regime-B helpful ``expert_0``.
    """
    gen_idx = _validate_nonneg_int(general_expert_index, "general_expert_index")
    cap_per_dom = _validate_positive_int(loeo_chunks_per_domain, "loeo_chunks_per_domain")
    harm_thresh = _validate_finite_float(loeo_e0_harm_threshold, "loeo_e0_harm_threshold")
    if harm_thresh > 0.0:
        raise ValueError(f"loeo_e0_harm_threshold must be <= 0.0, got {harm_thresh}")
    help_thresh = _validate_nonneg_finite(loeo_general_help_threshold, "loeo_general_help_threshold")
    max_spec_inc = _validate_nonneg_finite(loeo_max_spec_loss_increase, "loeo_max_spec_loss_increase")
    rogue_b_thresh = _validate_nonneg_finite(rogue_b_norm_threshold, "rogue_b_norm_threshold")
    rogue_ratio_thresh = _validate_nonneg_finite(rogue_norm_ratio_threshold, "rogue_norm_ratio_threshold")
    rogue_floor = _validate_nonneg_finite(rogue_min_norm_floor, "rogue_min_norm_floor")

    was_training = model.training
    ids_all, is_general, expert_id_strs, cal_w, cal_b, ln_eps, offset, bsz = _setup_attribution_run(
        model,
        probe_chunks,
        buckets=buckets,
        load_buckets_fn=load_buckets_fn,
        cal_ln_w=cal_ln_w,
        cal_ln_b=cal_ln_b,
        score_from=score_from,
        batch_size=batch_size,
        general_prefix=general_prefix,
    )
    norm_inventory = compute_expert_weight_norms(model)
    expert_ids = sorted(norm_inventory.keys())

    sub_indices = _select_balanced_chunk_indices(
        expert_id_strs, general_prefix=general_prefix, cap_per_domain=cap_per_dom
    )
    sub_ids = ids_all[sub_indices]
    sub_is_gen = is_general[sub_indices]
    sub_is_spec = ~sub_is_gen
    has_sub_spec = bool(np.any(sub_is_spec))

    # Determine domain coverage per expert index
    domain_chunk_counts: dict[int, int] = {}
    for eid in expert_id_strs:
        parsed_idx = _parse_expert_domain_index(eid)
        if parsed_idx is not None:
            domain_chunk_counts[parsed_idx] = domain_chunk_counts.get(parsed_idx, 0) + 1

    if scored_expert_indices is not None:
        scored_set: set[int] | None = {_validate_nonneg_int(i, "scored_expert_index") for i in scored_expert_indices}
    else:
        has_spec_overall = bool(np.any(~is_general))
        any_spec_parsed = any(idx != gen_idx for idx in domain_chunk_counts)
        scored_set = set(domain_chunk_counts.keys()) if (any_spec_parsed or not has_spec_overall) else None

    nonzero_w1 = [norm_inventory[e]["w1_b_norm"] for e in expert_ids if norm_inventory[e]["w1_b_norm"] > 0.0]
    nonzero_tot = [norm_inventory[e]["total_b_norm"] for e in expert_ids if norm_inventory[e]["total_b_norm"] > 0.0]
    med_w1 = float(statistics.median(nonzero_w1)) if nonzero_w1 else 0.0
    med_tot = float(statistics.median(nonzero_tot)) if nonzero_tot else 0.0

    model.eval()
    try:
        def _eval_subset() -> tuple[float, float]:
            hp = _forward_model_pre_ln(model, sub_ids, batch_size=bsz)
            nlls = (
                per_chunk_causal_midrow_nll(
                    hp, cal_w, cal_b, ln_eps, model.head.weight, sub_ids, score_from=offset, seq_batch_size=bsz
                )
                .cpu()
                .numpy()
            )
            return float(nlls[sub_is_gen].mean()), (float(nlls[sub_is_spec].mean()) if has_sub_spec else 0.0)

        ref_e0, ref_spec = _eval_subset()

        expert_loeo: dict[str, dict[str, Any]] = {}
        harmful_sink_experts: list[int] = []
        rogue_high_norm_experts: list[int] = []
        interfering_specialist_experts: list[int] = []
        helpful_general_experts: list[int] = []
        helpful_specialist_experts: list[int] = []
        neutral_experts: list[int] = []

        for e_idx in expert_ids:
            saved_b = _snapshot_and_zero_expert_b(model, e_idx)
            try:
                e0_m, sp_m = _eval_subset()
            finally:
                _restore_expert_b(saved_b)

            d_e0 = e0_m - ref_e0
            d_sp = (sp_m - ref_spec) if has_sub_spec else 0.0
            norms = norm_inventory[e_idx]
            w1_n = float(norms["w1_b_norm"])
            tot_n = float(norms["total_b_norm"])
            w1_ratio = w1_n / max(med_w1, 1e-8) if med_w1 > 0.0 else 0.0
            tot_ratio = tot_n / max(med_tot, 1e-8) if med_tot > 0.0 else 0.0

            if len(nonzero_tot) >= 3:
                is_high_norm = (
                    (w1_n >= rogue_b_thresh and w1_ratio >= 2.0)
                    or (tot_n >= rogue_b_thresh and tot_ratio >= 2.0)
                    or (w1_ratio >= rogue_ratio_thresh and w1_n >= rogue_floor)
                    or (tot_ratio >= rogue_ratio_thresh and tot_n >= rogue_floor)
                )
            else:
                is_high_norm = (w1_n >= rogue_b_thresh) or (tot_n >= rogue_b_thresh)

            own_count = domain_chunk_counts.get(e_idx, 0)
            is_scored = (e_idx in scored_set) if scored_set is not None else True
            is_unscored_rogue = bool((not is_scored) and is_high_norm and (d_sp <= max_spec_inc))

            label = _classify_loeo_expert(
                e_idx=e_idx,
                general_expert_index=gen_idx,
                delta_e0=d_e0,
                delta_spec=d_sp,
                is_unscored_rogue=is_unscored_rogue,
                harm_thresh=harm_thresh,
                help_thresh=help_thresh,
                max_spec_inc=max_spec_inc,
            )

            if d_e0 < harm_thresh and d_sp <= max_spec_inc:
                harmful_sink_experts.append(e_idx)
            if is_unscored_rogue:
                rogue_high_norm_experts.append(e_idx)

            if label == "specialist_with_general_interference":
                interfering_specialist_experts.append(e_idx)
            elif label == "helpful_general":
                helpful_general_experts.append(e_idx)
            elif label == "helpful_specialist":
                helpful_specialist_experts.append(e_idx)
            elif label == "neutral":
                neutral_experts.append(e_idx)

            expert_loeo[f"expert_{e_idx}"] = {
                "expert_index": e_idx,
                "e0_nll_without": e0_m,
                "spec_nll_without": sp_m,
                "delta_e0_when_zeroed": d_e0,
                "delta_spec_when_zeroed": d_sp,
                "general_harm_nats": -d_e0,
                "w1_b_norm": w1_n,
                "w2_b_norm": float(norms["w2_b_norm"]),
                "w3_b_norm": float(norms["w3_b_norm"]),
                "total_b_norm": tot_n,
                "max_layer_w1_b_norm": float(norms["max_layer_w1_b_norm"]),
                "max_layer_w1_b_layer": int(norms["max_layer_w1_b_layer"]),
                "router_row_norm": float(norms["router_row_norm"]),
                "w1_norm_ratio_to_median": w1_ratio,
                "total_norm_ratio_to_median": tot_ratio,
                "own_domain_chunk_count": own_count,
                "is_scored_domain": is_scored,
                "is_high_norm_outlier": is_high_norm,
                "is_rogue_high_norm": is_unscored_rogue,
                "classification": label,
            }
    finally:
        model.train(was_training)

    gen_entry = expert_loeo.get(f"expert_{gen_idx}")
    if gen_entry is not None and gen_idx in harmful_sink_experts:
        expert0_regime = "regime_a_harmful"
        protect_general = False
    elif gen_entry is not None and gen_idx in helpful_general_experts:
        expert0_regime = "regime_b_helpful"
        protect_general = True
    else:
        expert0_regime = "neutral"
        protect_general = False

    recommended_sink = sorted(set(harmful_sink_experts) | set(rogue_high_norm_experts))
    return {
        "ref_general_nll": ref_e0,
        "ref_spec_nll": ref_spec,
        "score_from": offset,
        "eval_chunk_count": len(sub_indices),
        "general_expert_index": gen_idx,
        "median_nonzero_w1_b_norm": med_w1,
        "median_nonzero_total_b_norm": med_tot,
        "expert0_regime": expert0_regime,
        "protect_general_expert": protect_general,
        "harmful_sink_experts": harmful_sink_experts,
        "rogue_high_norm_experts": rogue_high_norm_experts,
        "recommended_sink_experts": recommended_sink,
        "interfering_specialist_experts": interfering_specialist_experts,
        "helpful_general_experts": helpful_general_experts,
        "helpful_specialist_experts": helpful_specialist_experts,
        "neutral_experts": neutral_experts,
        "expert_loeo": expert_loeo,
    }


def _select_eval_subsets_for_layer_attribution(
    ids_all: torch.Tensor,
    is_general: np.ndarray,
    expert_id_strs: Sequence[str],
    *,
    max_general: int,
    max_spec: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select balanced general and specialist token batches for per-layer bypass attribution."""
    gen_indices = [i for i, flag in enumerate(is_general) if bool(flag)][:max_general]
    ids_gen = ids_all[gen_indices]

    spec_by_dom: dict[str, list[int]] = {}
    for idx, flag in enumerate(is_general):
        if not bool(flag):
            spec_by_dom.setdefault(expert_id_strs[idx], []).append(idx)

    spec_indices: list[int] = []
    if spec_by_dom:
        dom_queues = list(spec_by_dom.values())
        round_idx = 0
        while len(spec_indices) < max_spec:
            added = False
            for q in dom_queues:
                if round_idx < len(q):
                    spec_indices.append(q[round_idx])
                    added = True
                    if len(spec_indices) >= max_spec:
                        break
            if not added:
                break
            round_idx += 1

    ids_spec = ids_all[spec_indices] if spec_indices else None
    return ids_gen, ids_spec


@torch.no_grad()
def compute_per_layer_moe_bypass_attribution(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    buckets: Mapping[str, torch.Tensor] | None = None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    cal_ln_w: torch.Tensor | None = None,
    cal_ln_b: torch.Tensor | None = None,
    score_from: int = DEFAULT_SCORE_FROM,
    max_general_probe_chunks: int = 24,
    max_spec_probe_chunks: int = 24,
    batch_size: int = 24,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
) -> dict[str, Any]:
    """Ablate MoE deltas layer-by-layer against the frozen dense MLP to rank exact causal harm per layer.

    At every transformer block ``l``, compares:
    1. Exact final-layer causal NLL when only layer ``l`` reverts to ``base_mlp`` (MoE delta = 0);
    2. Direct intermediate Logit-Lens NLL immediately before vs after layer ``l``'s MoE delta;
    3. Exact MoE residual RMS norm on causal scored positions ``[score_from - 1 : T - 1]``.
    """
    max_gen = _validate_positive_int(max_general_probe_chunks, "max_general_probe_chunks")
    max_sp = _validate_positive_int(max_spec_probe_chunks, "max_spec_probe_chunks")
    was_training = model.training
    ids_all, is_general, expert_id_strs, cal_w, cal_b, ln_eps, offset, bsz = _setup_attribution_run(
        model,
        probe_chunks,
        buckets=buckets,
        load_buckets_fn=load_buckets_fn,
        cal_ln_w=cal_ln_w,
        cal_ln_b=cal_ln_b,
        score_from=score_from,
        batch_size=batch_size,
        general_prefix=general_prefix,
    )
    ids_gen, ids_spec = _select_eval_subsets_for_layer_attribution(
        ids_all, is_general, expert_id_strs, max_general=max_gen, max_spec=max_sp
    )
    start_pos = max(0, offset - 1)
    seq_len = int(ids_gen.shape[1])
    if start_pos >= seq_len - 1:
        raise ValueError(f"Sequence length {seq_len} too short for causal start_pos={start_pos}")

    cos = model.cos_cached.to(ids_gen.device)
    sin = model.sin_cached.to(ids_gen.device)
    n_layers = len(model.blocks)

    def _mean_nll(h_states: torch.Tensor, ids_sub: torch.Tensor) -> float:
        return float(
            per_chunk_causal_midrow_nll(
                h_states, cal_w, cal_b, ln_eps, model.head.weight, ids_sub, score_from=offset, seq_batch_size=bsz
            )
            .mean()
            .item()
        )

    model.eval()
    try:
        def _trace_unablated_with_logit_lens(
            ids_sub: torch.Tensor,
        ) -> tuple[float, list[torch.Tensor], list[float], list[float], list[float]]:
            xb = torch.cat(
                [compute_embeddings_batch(model, ids_sub[b0 : b0 + bsz]) for b0 in range(0, ids_sub.shape[0], bsz)],
                dim=0,
            )
            dense_post_states: list[torch.Tensor] = []
            lens_before: list[float] = []
            lens_after: list[float] = []
            delta_rms_list: list[float] = []

            for blk in model.blocks:
                xm_parts: list[torch.Tensor] = []
                dense_parts: list[torch.Tensor] = []
                moe_parts: list[torch.Tensor] = []
                blk_dtype = blk.ln1.weight.dtype
                for b0 in range(0, xb.shape[0], bsz):
                    xm_b, mi_b = forward_block_pre_mlp(blk, xb[b0 : b0 + bsz], cos, sin)
                    d_out_b = forward_block_dense_base_mlp(blk, mi_b)
                    m_out_b = blk.mlp(mi_b.to(dtype=blk_dtype)).to(torch.float32)
                    xm_parts.append(xm_b)
                    dense_parts.append(d_out_b)
                    moe_parts.append(m_out_b)
                xm = torch.cat(xm_parts, dim=0)
                dense_out = torch.cat(dense_parts, dim=0)
                moe_out = torch.cat(moe_parts, dim=0)

                x_dense_l = xm + dense_out
                x_moe_l = xm + moe_out
                delta_l = moe_out[:, start_pos:-1, :] - dense_out[:, start_pos:-1, :]

                dense_post_states.append(x_dense_l)
                lens_before.append(_mean_nll(x_dense_l, ids_sub))
                lens_after.append(_mean_nll(x_moe_l, ids_sub))
                delta_rms_list.append(float(delta_l.pow(2).mean().sqrt().item()))
                xb = x_moe_l

            return _mean_nll(xb, ids_sub), dense_post_states, lens_before, lens_after, delta_rms_list

        def _finish_from_layer(x_state: torch.Tensor, start_layer: int, ids_sub: torch.Tensor) -> float:
            xb = x_state
            for tail_blk in model.blocks[start_layer:]:
                next_parts: list[torch.Tensor] = []
                for b0 in range(0, xb.shape[0], bsz):
                    xm_b, mi_b = forward_block_pre_mlp(tail_blk, xb[b0 : b0 + bsz], cos, sin)
                    next_parts.append(forward_block_post_mlp(tail_blk, xm_b, mi_b))
                xb = torch.cat(next_parts, dim=0)
            return _mean_nll(xb, ids_sub)

        full_gen_nll, gen_dense_states, gen_lens_before, gen_lens_after, gen_rms = _trace_unablated_with_logit_lens(ids_gen)
        if ids_spec is not None:
            full_spec_nll, sp_dense_states, sp_lens_before, sp_lens_after, sp_rms = _trace_unablated_with_logit_lens(ids_spec)
        else:
            full_spec_nll = None
            sp_dense_states = []
            sp_lens_before = []
            sp_lens_after = []
            sp_rms = []

        layers_out: list[dict[str, Any]] = []
        for l_idx in range(n_layers):
            gen_bypass_nll = _finish_from_layer(gen_dense_states[l_idx], l_idx + 1, ids_gen)
            gen_harm_nats = full_gen_nll - gen_bypass_nll
            lens_gen_harm = gen_lens_after[l_idx] - gen_lens_before[l_idx]

            if ids_spec is not None and full_spec_nll is not None:
                sp_bypass_nll = _finish_from_layer(sp_dense_states[l_idx], l_idx + 1, ids_spec)
                sp_benefit_nats: float | None = sp_bypass_nll - full_spec_nll
                sp_l_before: float | None = sp_lens_before[l_idx]
                sp_l_after: float | None = sp_lens_after[l_idx]
                sp_l_benefit: float | None = sp_lens_before[l_idx] - sp_lens_after[l_idx]
                sp_delta_rms: float | None = sp_rms[l_idx]
            else:
                sp_bypass_nll = None
                sp_benefit_nats = None
                sp_l_before = None
                sp_l_after = None
                sp_l_benefit = None
                sp_delta_rms = None

            layers_out.append(
                {
                    "layer": l_idx,
                    "general_nll_bypassed": gen_bypass_nll,
                    "general_harm_nats": gen_harm_nats,
                    "spec_nll_bypassed": sp_bypass_nll,
                    "spec_benefit_nats": sp_benefit_nats,
                    "logit_lens_general_nll_before_moe": gen_lens_before[l_idx],
                    "logit_lens_general_nll_after_moe": gen_lens_after[l_idx],
                    "logit_lens_general_harm_nats": lens_gen_harm,
                    "logit_lens_spec_nll_before_moe": sp_l_before,
                    "logit_lens_spec_nll_after_moe": sp_l_after,
                    "logit_lens_spec_benefit_nats": sp_l_benefit,
                    "moe_delta_rms_general": gen_rms[l_idx],
                    "moe_delta_rms_spec": sp_delta_rms,
                }
            )
    finally:
        model.train(was_training)

    ranked_layers = [
        int(row["layer"])
        for row in sorted(layers_out, key=lambda d: (-float(d["general_harm_nats"]), int(d["layer"])))
    ]
    return {
        "num_layers": n_layers,
        "num_general_chunks": int(ids_gen.shape[0]),
        "num_spec_chunks": int(ids_spec.shape[0]) if ids_spec is not None else 0,
        "score_from": offset,
        "scored_positions": (seq_len - 1) - start_pos,
        "full_general_nll": full_gen_nll,
        "full_spec_nll": full_spec_nll,
        "layers": layers_out,
        "ranked_layers_by_general_harm": ranked_layers,
    }


def select_surgical_layer_mask(
    attribution: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    top_k_layers: int | None = None,
    min_harm_nats: float = 0.0,
    protect_early_layers_below: int = 0,
) -> dict[str, Any]:
    """Localize surgical weight edits to harmed mid/late layers while protecting early syntax layers."""
    if isinstance(attribution, Mapping):
        raw_layers = attribution.get("layers")
        if isinstance(raw_layers, (str, bytes)) or not isinstance(raw_layers, Sequence) or len(raw_layers) == 0:
            raise ValueError("attribution mapping must contain a non-empty 'layers' sequence")
        explicit_num_layers = attribution.get("num_layers")
    elif not isinstance(attribution, (str, bytes)) and isinstance(attribution, Sequence):
        if len(attribution) == 0:
            raise ValueError("attribution sequence must be non-empty")
        raw_layers = attribution
        explicit_num_layers = None
    else:
        raise TypeError(f"attribution must be a mapping or sequence of layer dicts, got {type(attribution).__name__}")

    top_k = None if top_k_layers is None else _validate_nonneg_int(top_k_layers, "top_k_layers")
    min_harm = _validate_nonneg_finite(min_harm_nats, "min_harm_nats")
    protect_below = _validate_nonneg_int(protect_early_layers_below, "protect_early_layers_below")

    harm_by_layer: dict[int, float] = {}
    for idx, item in enumerate(raw_layers):
        if not isinstance(item, Mapping) or "layer" not in item:
            raise ValueError(f"attribution[{idx}] must be a mapping containing 'layer'")
        l_idx = _validate_nonneg_int(item["layer"], f"attribution[{idx}]['layer']")
        if l_idx in harm_by_layer:
            raise ValueError(f"Duplicate layer index {l_idx} in attribution")
        raw_harm = item.get("general_harm_nats", item.get("e0_harm_nats"))
        if raw_harm is None:
            raise ValueError(f"attribution[{idx}] must contain 'general_harm_nats' or 'e0_harm_nats'")
        harm_by_layer[l_idx] = _validate_finite_float(raw_harm, f"layer_{l_idx}_harm_nats")

    max_layer = max(harm_by_layer.keys())
    if explicit_num_layers is not None:
        num_layers = _validate_positive_int(explicit_num_layers, "num_layers")
        if num_layers <= max_layer:
            raise ValueError(f"num_layers ({num_layers}) must exceed max layer index ({max_layer})")
    else:
        num_layers = max_layer + 1

    protected_layers = sorted(l_idx for l_idx in harm_by_layer if l_idx < protect_below)
    eligible_items = [(l_idx, h) for l_idx, h in harm_by_layer.items() if l_idx >= protect_below]
    eligible_items.sort(key=lambda pair: (-pair[1], pair[0]))
    ranked_eligible = [l_idx for l_idx, _ in eligible_items]

    qualifying: list[int] = []
    rejected_below_threshold: list[int] = []
    for l_idx, harm_val in eligible_items:
        passes = (harm_val >= min_harm) if min_harm > 0.0 else (harm_val > 0.0)
        if passes:
            qualifying.append(l_idx)
        else:
            rejected_below_threshold.append(l_idx)

    chosen = qualifying if top_k is None else qualifying[:top_k]
    selected_set = set(chosen)
    selected_layers = sorted(selected_set)
    layer_mask = [idx in selected_set for idx in range(num_layers)]

    selected_harm_total = sum(max(0.0, harm_by_layer[idx]) for idx in selected_layers)
    total_positive_harm = sum(max(0.0, h) for h in harm_by_layer.values())
    harm_coverage = (selected_harm_total / total_positive_harm) if total_positive_harm > 0.0 else 0.0

    return {
        "num_layers": num_layers,
        "layer_mask": layer_mask,
        "selected_layers": selected_layers,
        "protected_layers": protected_layers,
        "rejected_below_threshold_layers": sorted(rejected_below_threshold),
        "ranked_eligible_layers": ranked_eligible,
        "top_k_layers": top_k,
        "min_harm_nats": min_harm,
        "protect_early_layers_below": protect_below,
        "selected_harm_nats_total": float(selected_harm_total),
        "total_positive_harm_nats": float(total_positive_harm),
        "harm_coverage_fraction": float(harm_coverage),
    }


@torch.no_grad()
def build_surgical_attribution_plan(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    buckets: Mapping[str, torch.Tensor] | None = None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    cal_ln_w: torch.Tensor | None = None,
    cal_ln_b: torch.Tensor | None = None,
    score_from: int = DEFAULT_SCORE_FROM,
    top_k_layers: int | None = None,
    min_harm_nats: float = 0.0,
    protect_early_layers_below: int = 0,
    loeo_chunks_per_domain: int = 8,
    max_general_probe_chunks: int = 24,
    max_spec_probe_chunks: int = 24,
    batch_size: int = 24,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    general_expert_index: int = DEFAULT_GENERAL_EXPERT_INDEX,
    scored_expert_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run full expert LOEO attribution, per-layer MoE bypass/logit-lens attribution, and surgical mask selection."""
    expert_attr = compute_leave_one_expert_out_attribution(
        model,
        probe_chunks,
        buckets=buckets,
        load_buckets_fn=load_buckets_fn,
        cal_ln_w=cal_ln_w,
        cal_ln_b=cal_ln_b,
        score_from=score_from,
        loeo_chunks_per_domain=loeo_chunks_per_domain,
        batch_size=batch_size,
        general_prefix=general_prefix,
        general_expert_index=general_expert_index,
        scored_expert_indices=scored_expert_indices,
    )
    layer_attr = compute_per_layer_moe_bypass_attribution(
        model,
        probe_chunks,
        buckets=buckets,
        load_buckets_fn=load_buckets_fn,
        cal_ln_w=cal_ln_w,
        cal_ln_b=cal_ln_b,
        score_from=score_from,
        max_general_probe_chunks=max_general_probe_chunks,
        max_spec_probe_chunks=max_spec_probe_chunks,
        batch_size=batch_size,
        general_prefix=general_prefix,
    )
    mask_summary = select_surgical_layer_mask(
        layer_attr,
        top_k_layers=top_k_layers,
        min_harm_nats=min_harm_nats,
        protect_early_layers_below=protect_early_layers_below,
    )
    return {
        "expert_attribution": expert_attr,
        "layer_attribution": layer_attr,
        "layer_mask_summary": mask_summary,
        "selected_layers": mask_summary["selected_layers"],
        "layer_mask": mask_summary["layer_mask"],
        "protected_layers": mask_summary["protected_layers"],
        "harmful_sink_experts": expert_attr["harmful_sink_experts"],
        "rogue_high_norm_experts": expert_attr["rogue_high_norm_experts"],
        "recommended_sink_experts": expert_attr["recommended_sink_experts"],
        "interfering_specialist_experts": expert_attr["interfering_specialist_experts"],
        "helpful_general_experts": expert_attr["helpful_general_experts"],
        "expert0_regime": expert_attr["expert0_regime"],
        "protect_general_expert": expert_attr["protect_general_expert"],
    }


__all__ = [
    "DEFAULT_GENERAL_EXPERT_INDEX",
    "DEFAULT_LOEO_E0_HARM_THRESHOLD",
    "DEFAULT_LOEO_GENERAL_HELP_THRESHOLD",
    "DEFAULT_LOEO_MAX_SPEC_LOSS_INCREASE",
    "DEFAULT_ROGUE_B_NORM_THRESHOLD",
    "DEFAULT_ROGUE_MIN_NORM_FLOOR",
    "DEFAULT_ROGUE_NORM_RATIO_THRESHOLD",
    "DEFAULT_SCORE_FROM",
    "GENERAL_EXPERT_PREFIX",
    "build_surgical_attribution_plan",
    "compute_embeddings_batch",
    "compute_expert_weight_norms",
    "compute_leave_one_expert_out_attribution",
    "compute_per_layer_moe_bypass_attribution",
    "forward_block_dense_base_mlp",
    "forward_block_post_mlp",
    "forward_block_pre_mlp",
    "per_chunk_causal_midrow_nll",
    "select_surgical_layer_mask",
    "zero_expert_lora_b_inplace",
]
