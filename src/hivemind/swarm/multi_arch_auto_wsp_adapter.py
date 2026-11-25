# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Multi-Architecture Open-Weight WSP Adapter for Granite, Qwen, and Llama State Dicts.

Bridges heterogeneous open-weight PyTorch checkpoint layouts into the canonical
``AutoOpenWeightEngine`` tensor schema and materializes verified ``S5`` PASS-Synergy
edits back into the exact native layout without weight copying corruption:

1. **Granite-MoE** (``blocks.{l}.mlp.experts.{e}.w{1,2,3}_{A,B}`` +
   ``blocks.{l}.mlp.router.weight`` + ``ln.{weight,bias}``):
   Maps LoRA SwiGLU output factor ``w3_{A,B}`` to canonical ``layer_{l}.w2_{A,B}``
   and propagates ``S5.1`` FoMoE ghost-sinking across all three ``w1_B, w2_B, w3_B``
   factors so ``dy_e(x) == 0`` holds bitwise.
2. **Qwen-MoE / Qwen-Dense** (``model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj``
   + ``model.layers.{l}.mlp.gate.weight`` + ``model.norm.weight``, or dense
   ``model.layers.{l}.mlp.{gate,up,down}_proj`` with QKV biases / QK-norms):
   Supports both full HuggingFace weight matrices (via deterministic sign-canonical
   thin SVD factorization with lossless delta round-trip) and PEFT/LoRA factors.
3. **Llama-GQA / Dense** (``model.layers.{l}.self_attn.{q,k,v,o}_proj`` +
   ``model.layers.{l}.mlp.{gate,up,down}_proj`` + biasless RMSNorm ``model.norm.weight``):
   Detects Grouped-Query Attention (GQA) ``kv_head`` ratio directly from Q/K weights,
   maps ``down_proj`` LoRA or full weights to canonical ``layer_{l}.w2_{A,B}`` for
   tail-layer QR-SVD prose null-space projection (``S5.3``), and folds ``1 / tau_temp``
   calibration directly into biasless RMSNorm ``model.norm.weight`` (``S5.4``).

Every untouched tensor (embeddings, attention projections, shared experts, layer
norms, and beneficial anchor layers) is preserved 100% bitwise in its native dtype.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from hivemind.swarm.auto_open_weight_engine import (
    ANSE_MAXIMUM_PAIN,
    AutoOpenWeightConfig,
    AutoOpenWeightResult,
    hash_state_dict,
    run_auto_open_weight_s5_synergy,
)

SCHEMA_VERSION: str = "hivemind.multi_arch_auto_wsp_adapter/1"
RECEIPT_OBJECT: str = "hivemind.multi_arch_auto_wsp_receipt"

ARCH_GRANITE_MOE: str = "granite_moe"
ARCH_QWEN_MOE: str = "qwen_moe"
ARCH_QWEN_DENSE: str = "qwen_dense"
ARCH_LLAMA_GQA: str = "llama_gqa"
SUPPORTED_ARCH_FAMILIES: frozenset[str] = frozenset(
    {ARCH_GRANITE_MOE, ARCH_QWEN_MOE, ARCH_QWEN_DENSE, ARCH_LLAMA_GQA}
)

PARAM_LORA: str = "lora"
PARAM_FULL_WEIGHT: str = "full_weight"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_GRANITE_EXPERT_RE = re.compile(r"^blocks\.(\d+)\.mlp\.experts\.(\d+)\.(w[123]_[AB])$")
_GRANITE_ROUTER_RE = re.compile(r"^blocks\.(\d+)\.mlp\.(?:router|gate)\.weight$")
_QWEN_EXPERT_LORA_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(lora_[AB])(?:\.(?:default\.)?weight)?$"
)
_QWEN_EXPERT_FULL_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_HF_DENSE_LORA_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(lora_[AB])(?:\.(?:default\.)?weight)?$"
)
_HF_DENSE_FULL_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
_HF_ROUTER_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(?:gate|router)\.weight$")
_HF_QWEN_BIAS_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(?:q|k|v)_proj\.bias$")
_HF_QWEN_QKNORM_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(?:q|k)_norm\.weight$")

_FINAL_NORM_CANDIDATES: tuple[tuple[str, str | None], ...] = (
    ("ln.weight", "ln.bias"),
    ("model.ln.weight", "model.ln.bias"),
    ("model.norm.weight", "model.norm.bias"),
    ("norm.weight", "norm.bias"),
    ("ln_f.weight", "ln_f.bias"),
)

_ALLOWED_S5_KWARGS: frozenset[str] = frozenset(
    {
        "layer_e0_harm_nats",
        "expert_e0_harm_nats",
        "expert_spec_loss_when_zeroed",
        "layer_prose_inputs",
        "layer_specialist_inputs",
        "tail_layer_prose_ffn_mid",
        "base_final_ln_w",
        "base_final_ln_b",
        "base_state",
        "extra_custom_stages",
        "config",
        "arch_hint",
        "full_weight_max_rank",
        "auto_temperature_calibration",
    }
)


@dataclasses.dataclass(frozen=True)
class OpenWeightArchitectureSpec:
    """Deterministic structural specification detected from a native PyTorch state dict."""

    arch_family: str
    parameterization: str
    layer_count: int
    expert_count: int
    is_moe: bool
    gqa_kv_head_ratio: int
    kv_to_q_head_ratio: float
    hidden_size: int
    intermediate_size: int
    lora_ranks: tuple[int, ...]
    final_norm_key: str | None
    final_norm_bias_key: str | None
    norm_type: str
    has_attention: bool
    q_out_dim: int | None
    kv_out_dim: int | None
    router_keys: tuple[str, ...]
    untouched_keys: tuple[str, ...]
    layout_signature_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize architecture metadata into a canonical JSON-safe dictionary."""
        return {
            "arch_family": self.arch_family,
            "parameterization": self.parameterization,
            "layer_count": self.layer_count,
            "expert_count": self.expert_count,
            "is_moe": self.is_moe,
            "gqa_kv_head_ratio": self.gqa_kv_head_ratio,
            "kv_to_q_head_ratio": self.kv_to_q_head_ratio,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "lora_ranks": list(self.lora_ranks),
            "final_norm_key": self.final_norm_key,
            "final_norm_bias_key": self.final_norm_bias_key,
            "norm_type": self.norm_type,
            "has_attention": self.has_attention,
            "q_out_dim": self.q_out_dim,
            "kv_out_dim": self.kv_out_dim,
            "router_keys": list(self.router_keys),
            "untouched_keys": list(self.untouched_keys),
            "layout_signature_sha256": self.layout_signature_sha256,
        }


@dataclasses.dataclass(frozen=True)
class MultiArchAutoWSPResult:
    """End-to-end result and provenance receipt for multi-architecture S5 WSP surgery."""

    committed: bool
    verdict: str
    arch_family: str
    parameterization: str
    layer_count: int
    expert_count: int
    gqa_kv_head_ratio: int
    final_norm_key: str | None
    norm_type: str
    anse_energy: float
    nll_mean: float
    mean_diff_vs_base: float
    general_rel_drift: float
    all_domains_negative: bool
    native_total_params: int
    native_edited_params: int
    native_edited_fraction: float
    untouched_keys_count: int
    untouched_keys_bitwise_preserved: bool
    selected_lambda_ln: float
    selected_tau_temp: float
    pre_native_sha256: str
    post_native_sha256: str
    canonical_pre_sha256: str
    canonical_post_sha256: str
    roundtrip_verified: bool
    arch_spec: OpenWeightArchitectureSpec
    s5_result: AutoOpenWeightResult
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize result and nested S5 telemetry into a JSON-safe dictionary."""
        return {
            "schema": SCHEMA_VERSION,
            "object": RECEIPT_OBJECT,
            "committed": self.committed,
            "verdict": self.verdict,
            "arch_family": self.arch_family,
            "parameterization": self.parameterization,
            "layer_count": self.layer_count,
            "expert_count": self.expert_count,
            "gqa_kv_head_ratio": self.gqa_kv_head_ratio,
            "final_norm_key": self.final_norm_key,
            "norm_type": self.norm_type,
            "anse_energy": self.anse_energy,
            "nll_mean": self.nll_mean,
            "mean_diff_vs_base": self.mean_diff_vs_base,
            "general_rel_drift": self.general_rel_drift,
            "all_domains_negative": self.all_domains_negative,
            "native_total_params": self.native_total_params,
            "native_edited_params": self.native_edited_params,
            "native_edited_fraction": self.native_edited_fraction,
            "untouched_keys_count": self.untouched_keys_count,
            "untouched_keys_bitwise_preserved": self.untouched_keys_bitwise_preserved,
            "selected_lambda_ln": self.selected_lambda_ln,
            "selected_tau_temp": self.selected_tau_temp,
            "pre_native_sha256": self.pre_native_sha256,
            "post_native_sha256": self.post_native_sha256,
            "canonical_pre_sha256": self.canonical_pre_sha256,
            "canonical_post_sha256": self.canonical_post_sha256,
            "roundtrip_verified": self.roundtrip_verified,
            "arch_spec": self.arch_spec.to_dict(),
            "s5_result": self.s5_result.to_dict(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclasses.dataclass(frozen=True)
class _ResolvedLayout:
    """Internal binding between a native state dict and canonical S5 tensor keys."""

    spec: OpenWeightArchitectureSpec
    down_lora_pairs: dict[tuple[int, int], tuple[str, str]]
    all_lora_b_keys: dict[tuple[int, int], tuple[str, ...]]
    down_full_keys: dict[tuple[int, int], str]
    router_key_by_layer: dict[int, str]
    layer_ranks: tuple[int, ...]


# ---------------------------------------------------------------------------
# 1. Validation, Hashing, and Deterministic SVD Helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` into deterministic UTF-8 JSON without NaN/Inf."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest over ``_canonical_json_bytes(payload)``."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _is_tensor_finite(tensor: torch.Tensor) -> bool:
    """Return True iff every floating/complex entry of ``tensor`` is finite."""
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    return bool(torch.isfinite(tensor.detach()).all().item())


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    """Return exact raw bytes of a CPU tensor across all dtypes including bfloat16."""
    cpu_t = tensor.detach().to(device="cpu").contiguous()
    if cpu_t.numel() == 0:
        return b""
    if cpu_t.dtype == torch.bool:
        return cpu_t.to(torch.uint8).numpy().tobytes()
    return cpu_t.reshape(-1).view(torch.uint8).numpy().tobytes()


def hash_native_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over native PyTorch tensors."""
    validated = _validate_native_state_dict(state_dict, name="state_dict")
    digest = hashlib.sha256()
    for key in sorted(validated.keys()):
        t = validated[key]
        digest.update(f"{key}:{t.dtype}:{tuple(t.shape)}:".encode())
        digest.update(_tensor_raw_bytes(t))
    return digest.hexdigest()


def clone_native_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Deep-clone a validated native PyTorch state dict preserving key order, dtype, and device."""
    validated = _validate_native_state_dict(state_dict, name="state_dict")
    return {k: v.detach().clone() for k, v in validated.items()}


def count_native_edited_parameters(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> tuple[int, int]:
    """Return ``(edited_params, total_params)`` counting exact elementwise changes."""
    if set(before.keys()) != set(after.keys()):
        raise ValueError("before and after native state dicts must share identical keys")
    edited = 0
    total = 0
    for key in sorted(before.keys()):
        b = before[key]
        a = after[key]
        if b.shape != a.shape:
            raise ValueError(f"Shape mismatch on native key {key!r}: {tuple(b.shape)} vs {tuple(a.shape)}")
        total += int(b.numel())
        if b.numel() == 0:
            continue
        b_cpu = b.detach().to(device="cpu")
        a_cpu = a.detach().to(device="cpu")
        if b_cpu.dtype == a_cpu.dtype and torch.equal(b_cpu, a_cpu):
            continue
        edited += int(torch.ne(b_cpu.to(torch.float64), a_cpu.to(torch.float64)).sum().item())
    return edited, total


def _validate_native_state_dict(
    state_dict: Any,
    *,
    name: str = "state_dict",
) -> dict[str, torch.Tensor]:
    """Validate that ``state_dict`` is a non-empty mapping of finite ``torch.Tensor`` values."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{name} must be a non-empty mapping of str -> torch.Tensor")
    out: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings, got {key!r}")
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"{name}[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        if int(val.numel()) == 0:
            raise ValueError(f"{name}[{key!r}] must be non-empty")
        if not _is_tensor_finite(val):
            raise ValueError(f"{name}[{key!r}] contains non-finite values")
        out[key] = val
    return out


def _tensor_to_float64_np(tensor: torch.Tensor, name: str) -> np.ndarray:
    """Validate finiteness and convert a floating-point ``torch.Tensor`` to CPU ``float64`` NumPy."""
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor, got {tensor.dtype}")
    arr = tensor.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite entries")
    return arr


def _deterministic_thin_svd_factor(
    weight_2d: np.ndarray,
    *,
    max_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize ``W = B @ A`` via sign-canonical thin SVD with ``B: (H, r)``, ``A: (r, I)``."""
    w64 = np.asarray(weight_2d, dtype=np.float64)
    if w64.ndim != 2 or w64.shape[0] == 0 or w64.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D weight matrix, got shape {w64.shape}")
    full_r = min(int(w64.shape[0]), int(w64.shape[1]))
    target_r = full_r if max_rank is None else min(full_r, int(max_rank))
    if target_r <= 0:
        raise ValueError("full_weight_max_rank must be >= 1 when provided")

    u_mat, s_vals, vt_mat = np.linalg.svd(w64, full_matrices=False)
    u_r = u_mat[:, :target_r]
    s_r = s_vals[:target_r]
    vt_r = vt_mat[:target_r, :]

    # Canonicalize column signs by largest-magnitude entry of U so SVD is platform-deterministic.
    for col in range(target_r):
        pivot = int(np.argmax(np.abs(u_r[:, col])))
        if u_r[pivot, col] < 0.0:
            u_r[:, col] *= -1.0
            vt_r[col, :] *= -1.0

    sqrt_s = np.sqrt(np.maximum(s_r, 0.0))
    b_mat = np.ascontiguousarray(u_r * sqrt_s[None, :], dtype=np.float64)
    a_mat = np.ascontiguousarray(sqrt_s[:, None] * vt_r, dtype=np.float64)
    return b_mat, a_mat


def _validate_contiguous_indices(indices: set[int], kind: str) -> tuple[int, ...]:
    """Validate that discovered layer/expert indices form ``0..N-1`` without gaps."""
    if not indices:
        raise ValueError(f"No {kind} indices discovered in state_dict")
    ordered = tuple(sorted(indices))
    expected = tuple(range(len(ordered)))
    if ordered != expected:
        raise ValueError(f"Non-contiguous {kind} indices {ordered}; expected contiguous {expected}")
    return ordered


def _resolve_final_norm_keys(
    state_dict: Mapping[str, torch.Tensor],
    hidden_size: int,
) -> tuple[str | None, str | None, str]:
    """Locate and validate final LayerNorm / RMSNorm weight and optional bias keys."""
    for w_key, b_key in _FINAL_NORM_CANDIDATES:
        if w_key in state_dict:
            w_t = state_dict[w_key]
            if w_t.ndim != 1 or int(w_t.shape[0]) != hidden_size or not w_t.is_floating_point():
                raise ValueError(
                    f"Final norm key {w_key!r} must be 1D float tensor of length {hidden_size}, "
                    f"got {tuple(w_t.shape)} ({w_t.dtype})"
                )
            if b_key is not None and b_key in state_dict:
                b_t = state_dict[b_key]
                if b_t.ndim != 1 or int(b_t.shape[0]) != hidden_size or not b_t.is_floating_point():
                    raise ValueError(
                        f"Final norm bias key {b_key!r} must be 1D float tensor of length {hidden_size}, "
                        f"got {tuple(b_t.shape)} ({b_t.dtype})"
                    )
                return w_key, b_key, "layernorm"
            return w_key, None, "rmsnorm"
    return None, None, "none"


def _lookup_first_tensor(
    state_dict: Mapping[str, torch.Tensor],
    candidates: Sequence[str],
) -> tuple[str, torch.Tensor] | None:
    """Return the first ``(key, tensor)`` present in ``state_dict`` from ``candidates``."""
    for key in candidates:
        if key in state_dict:
            return key, state_dict[key]
    return None


def _detect_gqa_attention_dims(
    state_dict: Mapping[str, torch.Tensor],
    *,
    is_granite: bool,
    layer_count: int,
    hidden_size: int,
) -> tuple[bool, int | None, int | None, int, float]:
    """Inspect Q/K/V/O attention weights across layer 0..L-1 and compute GQA KV-head ratio."""
    q_out_ref: int | None = None
    kv_out_ref: int | None = None

    for l_idx in range(layer_count):
        if is_granite:
            q_cands = (
                f"blocks.{l_idx}.attn.q.weight",
                f"blocks.{l_idx}.self_attn.q_proj.weight",
                f"blocks.{l_idx}.self_attn.q_proj.lora_B.weight",
                f"blocks.{l_idx}.self_attn.q_proj.lora_B",
            )
            k_cands = (
                f"blocks.{l_idx}.attn.k.weight",
                f"blocks.{l_idx}.self_attn.k_proj.weight",
                f"blocks.{l_idx}.self_attn.k_proj.lora_B.weight",
                f"blocks.{l_idx}.self_attn.k_proj.lora_B",
            )
            v_cands = (
                f"blocks.{l_idx}.attn.v.weight",
                f"blocks.{l_idx}.self_attn.v_proj.weight",
                f"blocks.{l_idx}.self_attn.v_proj.lora_B.weight",
                f"blocks.{l_idx}.self_attn.v_proj.lora_B",
            )
            o_cands = (
                f"blocks.{l_idx}.attn.o.weight",
                f"blocks.{l_idx}.self_attn.o_proj.weight",
                f"blocks.{l_idx}.self_attn.o_proj.lora_B.weight",
                f"blocks.{l_idx}.self_attn.o_proj.lora_B",
            )
        else:
            q_cands = (
                f"model.layers.{l_idx}.self_attn.q_proj.weight",
                f"model.layers.{l_idx}.self_attn.q_proj.base_layer.weight",
                f"model.layers.{l_idx}.self_attn.q_proj.lora_B.weight",
                f"model.layers.{l_idx}.self_attn.q_proj.lora_B.default.weight",
                f"model.layers.{l_idx}.self_attn.q_proj.lora_B",
            )
            k_cands = (
                f"model.layers.{l_idx}.self_attn.k_proj.weight",
                f"model.layers.{l_idx}.self_attn.k_proj.base_layer.weight",
                f"model.layers.{l_idx}.self_attn.k_proj.lora_B.weight",
                f"model.layers.{l_idx}.self_attn.k_proj.lora_B.default.weight",
                f"model.layers.{l_idx}.self_attn.k_proj.lora_B",
            )
            v_cands = (
                f"model.layers.{l_idx}.self_attn.v_proj.weight",
                f"model.layers.{l_idx}.self_attn.v_proj.base_layer.weight",
                f"model.layers.{l_idx}.self_attn.v_proj.lora_B.weight",
                f"model.layers.{l_idx}.self_attn.v_proj.lora_B.default.weight",
                f"model.layers.{l_idx}.self_attn.v_proj.lora_B",
            )
            o_cands = (
                f"model.layers.{l_idx}.self_attn.o_proj.weight",
                f"model.layers.{l_idx}.self_attn.o_proj.base_layer.weight",
                f"model.layers.{l_idx}.self_attn.o_proj.lora_B.weight",
                f"model.layers.{l_idx}.self_attn.o_proj.lora_B.default.weight",
                f"model.layers.{l_idx}.self_attn.o_proj.lora_B",
            )

        q_hit = _lookup_first_tensor(state_dict, q_cands)
        k_hit = _lookup_first_tensor(state_dict, k_cands)
        v_hit = _lookup_first_tensor(state_dict, v_cands)
        o_hit = _lookup_first_tensor(state_dict, o_cands)

        if q_hit is None and k_hit is None:
            continue
        if q_hit is None or k_hit is None:
            raise ValueError(f"Incomplete Q/K attention projections at layer {l_idx}")

        q_key, q_t = q_hit
        k_key, k_t = k_hit
        if q_t.ndim != 2 or k_t.ndim != 2:
            raise ValueError(f"Attention projections {q_key!r}/{k_key!r} must be 2D")
        q_out = int(q_t.shape[0])
        k_out = int(k_t.shape[0])
        if q_out <= 0 or k_out <= 0 or q_out < k_out or q_out % k_out != 0:
            raise ValueError(
                f"Invalid GQA dimensions at layer {l_idx}: q_out={q_out}, kv_out={k_out}"
            )
        if "lora_B" not in q_key and int(q_t.shape[1]) != hidden_size:
            raise ValueError(f"{q_key!r} input width {q_t.shape[1]} != hidden_size {hidden_size}")
        if "lora_B" not in k_key and int(k_t.shape[1]) != hidden_size:
            raise ValueError(f"{k_key!r} input width {k_t.shape[1]} != hidden_size {hidden_size}")

        if v_hit is not None:
            v_key, v_t = v_hit
            if v_t.ndim != 2 or int(v_t.shape[0]) != k_out:
                raise ValueError(
                    f"V projection {v_key!r} out_dim {tuple(v_t.shape)} does not match K out_dim {k_out}"
                )
        if o_hit is not None:
            o_key, o_t = o_hit
            if o_t.ndim != 2 or int(o_t.shape[0]) != hidden_size:
                raise ValueError(
                    f"O projection {o_key!r} out_dim {tuple(o_t.shape)} does not match hidden_size {hidden_size}"
                )
            if "lora_B" not in o_key and int(o_t.shape[1]) != q_out:
                raise ValueError(
                    f"O projection {o_key!r} in_dim {o_t.shape[1]} does not match Q out_dim {q_out}"
                )

        if q_out_ref is None:
            q_out_ref, kv_out_ref = q_out, k_out
        elif q_out != q_out_ref or k_out != kv_out_ref:
            raise ValueError(
                f"Inconsistent attention dimensions at layer {l_idx}: ({q_out}, {k_out}) vs ({q_out_ref}, {kv_out_ref})"
            )

    if q_out_ref is None or kv_out_ref is None:
        return False, None, None, 1, 1.0
    return True, q_out_ref, kv_out_ref, q_out_ref // kv_out_ref, float(kv_out_ref) / float(q_out_ref)


# ---------------------------------------------------------------------------
# 2. Unified Architecture Detection & Layout Resolution
# ---------------------------------------------------------------------------


def _validate_down_lora_pair(
    b_t: torch.Tensor,
    a_t: torch.Tensor,
    hidden_size: int | None,
    intermediate_size: int | None,
    layer_rank: int | None,
    *,
    label: str,
) -> tuple[int, int, int]:
    """Validate a 2D floating ``(B, A)`` down-projection LoRA pair and return ``(H, I, r)``."""
    if a_t.ndim != 2 or b_t.ndim != 2 or not (a_t.is_floating_point() and b_t.is_floating_point()):
        raise ValueError(f"Down-projection LoRA factors must be 2D floating tensors at {label}")
    h_e, r_b = int(b_t.shape[0]), int(b_t.shape[1])
    r_a, i_e = int(a_t.shape[0]), int(a_t.shape[1])
    if r_b != r_a or r_b <= 0 or h_e <= 0 or i_e <= 0:
        raise ValueError(f"Mismatched LoRA shapes at {label}: B={tuple(b_t.shape)}, A={tuple(a_t.shape)}")
    if hidden_size is not None and (h_e != hidden_size or i_e != intermediate_size):
        raise ValueError(f"Inconsistent dims at {label}: ({h_e}, {i_e}) != ({hidden_size}, {intermediate_size})")
    if layer_rank is not None and r_b != layer_rank:
        raise ValueError(f"Inconsistent LoRA rank at {label}: {r_b} != {layer_rank}")
    return h_e, i_e, r_b


def _finalize_resolved_layout(
    validated: Mapping[str, torch.Tensor],
    *,
    arch_family: str,
    parameterization: str,
    layers: tuple[int, ...],
    expert_count: int,
    is_moe: bool,
    is_granite: bool,
    hidden_size: int,
    intermediate_size: int,
    layer_ranks: tuple[int, ...],
    down_lora_pairs: dict[tuple[int, int], tuple[str, str]],
    all_lora_b_keys: dict[tuple[int, int], tuple[str, ...]],
    down_full_keys: dict[tuple[int, int], str],
    router_keys_raw: Mapping[int, str],
    surgical_keys: set[str],
) -> _ResolvedLayout:
    """Validate router/norm/attention keys and assemble ``OpenWeightArchitectureSpec``."""
    for l_idx in layers:
        if l_idx in router_keys_raw:
            r_key = router_keys_raw[l_idx]
            r_t = validated[r_key]
            if r_t.ndim != 2 or tuple(r_t.shape) != (expert_count, hidden_size) or not r_t.is_floating_point():
                raise ValueError(
                    f"Router key {r_key!r} must have shape ({expert_count}, {hidden_size}), got {tuple(r_t.shape)}"
                )
            surgical_keys.add(r_key)

    fn_w, fn_b, norm_type = _resolve_final_norm_keys(validated, hidden_size)
    if fn_w is not None:
        surgical_keys.add(fn_w)
    if fn_b is not None:
        surgical_keys.add(fn_b)

    has_attn, q_out, kv_out, gqa_ratio, kv_q_ratio = _detect_gqa_attention_dims(
        validated, is_granite=is_granite, layer_count=len(layers), hidden_size=hidden_size
    )
    router_tuple = tuple(router_keys_raw[l_idx] for l_idx in layers if l_idx in router_keys_raw)
    untouched = tuple(sorted(k for k in validated if k not in surgical_keys))
    sig_payload = [(k, str(validated[k].dtype), list(validated[k].shape)) for k in sorted(validated.keys())]

    spec = OpenWeightArchitectureSpec(
        arch_family=arch_family,
        parameterization=parameterization,
        layer_count=len(layers),
        expert_count=expert_count,
        is_moe=is_moe,
        gqa_kv_head_ratio=gqa_ratio,
        kv_to_q_head_ratio=kv_q_ratio,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        lora_ranks=layer_ranks,
        final_norm_key=fn_w,
        final_norm_bias_key=fn_b,
        norm_type=norm_type,
        has_attention=has_attn,
        q_out_dim=q_out,
        kv_out_dim=kv_out,
        router_keys=router_tuple,
        untouched_keys=untouched,
        layout_signature_sha256=_compute_payload_sha256(sig_payload),
    )
    return _ResolvedLayout(
        spec=spec,
        down_lora_pairs=down_lora_pairs,
        all_lora_b_keys=all_lora_b_keys,
        down_full_keys=down_full_keys,
        router_key_by_layer=dict(router_keys_raw),
        layer_ranks=layer_ranks,
    )


def _resolve_granite_layout(
    validated: Mapping[str, torch.Tensor],
    *,
    arch_hint: str | None,
) -> _ResolvedLayout:
    """Resolve Granite-MoE (``blocks.{l}.mlp.experts.{e}.w{1,2,3}_{A,B}``) tensors."""
    if arch_hint is not None and arch_hint != ARCH_GRANITE_MOE:
        raise ValueError(f"arch_hint={arch_hint!r} conflicts with Granite-MoE state_dict layout")

    layer_indices: set[int] = set()
    expert_indices_by_layer: dict[int, set[int]] = {}
    expert_factors: dict[tuple[int, int], dict[str, str]] = {}
    router_keys_raw: dict[int, str] = {}

    for key in sorted(validated.keys()):
        m_exp = _GRANITE_EXPERT_RE.match(key)
        if m_exp is not None:
            l_idx, e_idx, factor = int(m_exp.group(1)), int(m_exp.group(2)), m_exp.group(3)
            layer_indices.add(l_idx)
            expert_indices_by_layer.setdefault(l_idx, set()).add(e_idx)
            expert_factors.setdefault((l_idx, e_idx), {})[factor] = key
            continue
        m_rt = _GRANITE_ROUTER_RE.match(key)
        if m_rt is not None:
            l_idx = int(m_rt.group(1))
            if l_idx in router_keys_raw:
                raise ValueError(f"Duplicate Granite router key at layer {l_idx}")
            router_keys_raw[l_idx] = key

    layers = _validate_contiguous_indices(layer_indices, "Granite layer")
    first_experts = _validate_contiguous_indices(expert_indices_by_layer[layers[0]], "Granite expert")
    expert_count = len(first_experts)

    hidden_size: int | None = None
    intermediate_size: int | None = None
    layer_ranks: list[int] = []
    down_lora_pairs: dict[tuple[int, int], tuple[str, str]] = {}
    all_lora_b_keys: dict[tuple[int, int], tuple[str, ...]] = {}
    surgical_keys: set[str] = set()

    for l_idx in layers:
        l_experts = _validate_contiguous_indices(expert_indices_by_layer[l_idx], f"Granite layer {l_idx} expert")
        if len(l_experts) != expert_count:
            raise ValueError(f"Granite layer {l_idx} has {len(l_experts)} experts; expected {expert_count}")
        rank_l: int | None = None
        for e_idx in l_experts:
            fmap = expert_factors[(l_idx, e_idx)]
            if "w3_A" not in fmap or "w3_B" not in fmap:
                raise ValueError(f"Missing w3_A/w3_B pair at Granite layer {l_idx}, expert {e_idx}")
            hidden_size, intermediate_size, rank_l = _validate_down_lora_pair(
                validated[fmap["w3_B"]],
                validated[fmap["w3_A"]],
                hidden_size,
                intermediate_size,
                rank_l,
                label=f"Granite layer {l_idx} expert {e_idx}",
            )

            b_keys_e: list[str] = []
            for prefix_w in ("w1", "w2"):
                k_a, k_b = f"{prefix_w}_A", f"{prefix_w}_B"
                if (k_a in fmap) != (k_b in fmap):
                    raise ValueError(f"Incomplete {prefix_w} LoRA pair at layer {l_idx} expert {e_idx}")
                if k_a in fmap:
                    ta, tb = validated[fmap[k_a]], validated[fmap[k_b]]
                    if ta.shape != (rank_l, hidden_size) or tb.shape != (intermediate_size, rank_l):
                        raise ValueError(
                            f"Invalid {prefix_w} LoRA shapes at layer {l_idx} expert {e_idx}: "
                            f"A={tuple(ta.shape)}, B={tuple(tb.shape)}"
                        )
                    b_keys_e.append(fmap[k_b])
                    surgical_keys.add(fmap[k_b])
            b_keys_e.append(fmap["w3_B"])
            down_lora_pairs[(l_idx, e_idx)] = (fmap["w3_B"], fmap["w3_A"])
            all_lora_b_keys[(l_idx, e_idx)] = tuple(b_keys_e)
            surgical_keys.update((fmap["w3_A"], fmap["w3_B"]))

        assert rank_l is not None
        layer_ranks.append(rank_l)

    assert hidden_size is not None and intermediate_size is not None
    return _finalize_resolved_layout(
        validated,
        arch_family=ARCH_GRANITE_MOE,
        parameterization=PARAM_LORA,
        layers=layers,
        expert_count=expert_count,
        is_moe=True,
        is_granite=True,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_ranks=tuple(layer_ranks),
        down_lora_pairs=down_lora_pairs,
        all_lora_b_keys=all_lora_b_keys,
        down_full_keys={},
        router_keys_raw=router_keys_raw,
        surgical_keys=surgical_keys,
    )


def _validate_hf_gate_up_shapes(
    validated: Mapping[str, torch.Tensor],
    proj_map: Mapping[str, str],
    *,
    parameterization: str,
    hidden_size: int,
    intermediate_size: int,
    label: str,
) -> list[str]:
    """Validate optional ``gate_proj`` and ``up_proj`` weights/LoRAs and return their B keys."""
    extra_b_keys: list[str] = []
    for proj_name in ("gate_proj", "up_proj"):
        if parameterization == PARAM_LORA:
            a_tag, b_tag = f"{proj_name}.lora_A", f"{proj_name}.lora_B"
            if (a_tag in proj_map) != (b_tag in proj_map):
                raise ValueError(f"Incomplete LoRA pair for {proj_name} at {label}")
            if a_tag in proj_map:
                ta, tb = validated[proj_map[a_tag]], validated[proj_map[b_tag]]
                if ta.ndim != 2 or tb.ndim != 2 or not (ta.is_floating_point() and tb.is_floating_point()):
                    raise ValueError(f"{proj_name} LoRA factors must be 2D floating tensors at {label}")
                if int(tb.shape[0]) != intermediate_size or int(ta.shape[1]) != hidden_size or tb.shape[1] != ta.shape[0]:
                    raise ValueError(
                        f"Invalid {proj_name} LoRA shapes at {label}: B={tuple(tb.shape)}, A={tuple(ta.shape)}, "
                        f"expected ({intermediate_size}, r) and (r, {hidden_size})"
                    )
                extra_b_keys.append(proj_map[b_tag])
        else:
            w_tag = f"{proj_name}.weight"
            if w_tag in proj_map:
                tw = validated[proj_map[w_tag]]
                if tw.ndim != 2 or tuple(tw.shape) != (intermediate_size, hidden_size) or not tw.is_floating_point():
                    raise ValueError(
                        f"Invalid {proj_name}.weight shape at {label}: {tuple(tw.shape)} != "
                        f"({intermediate_size}, {hidden_size})"
                    )
    return extra_b_keys


def _resolve_hf_layout(
    validated: Mapping[str, torch.Tensor],
    *,
    arch_hint: str | None,
) -> _ResolvedLayout:
    """Resolve Qwen-MoE, Qwen-Dense, and Llama-GQA ``model.layers.{l}.*`` layouts."""
    qwen_exp_lora: dict[tuple[int, int], dict[str, str]] = {}
    qwen_exp_full: dict[tuple[int, int], dict[str, str]] = {}
    dense_lora: dict[int, dict[str, str]] = {}
    dense_full: dict[int, dict[str, str]] = {}
    router_keys_raw: dict[int, str] = {}
    has_qwen_marker = False

    for key in sorted(validated.keys()):
        if _HF_QWEN_BIAS_RE.match(key) or _HF_QWEN_QKNORM_RE.match(key):
            has_qwen_marker = True
        m_qe_lora = _QWEN_EXPERT_LORA_RE.match(key)
        if m_qe_lora is not None:
            l_idx, e_idx = int(m_qe_lora.group(1)), int(m_qe_lora.group(2))
            qwen_exp_lora.setdefault((l_idx, e_idx), {})[f"{m_qe_lora.group(3)}.{m_qe_lora.group(4)}"] = key
            continue
        m_qe_full = _QWEN_EXPERT_FULL_RE.match(key)
        if m_qe_full is not None:
            l_idx, e_idx = int(m_qe_full.group(1)), int(m_qe_full.group(2))
            qwen_exp_full.setdefault((l_idx, e_idx), {})[f"{m_qe_full.group(3)}.weight"] = key
            continue
        m_dl = _HF_DENSE_LORA_RE.match(key)
        if m_dl is not None:
            l_idx = int(m_dl.group(1))
            dense_lora.setdefault(l_idx, {})[f"{m_dl.group(2)}.{m_dl.group(3)}"] = key
            continue
        m_df = _HF_DENSE_FULL_RE.match(key)
        if m_df is not None:
            l_idx = int(m_df.group(1))
            dense_full.setdefault(l_idx, {})[f"{m_df.group(2)}.weight"] = key
            continue
        m_rt = _HF_ROUTER_RE.match(key)
        if m_rt is not None:
            l_idx = int(m_rt.group(1))
            if l_idx in router_keys_raw:
                raise ValueError(f"Duplicate router/gate key at model.layers.{l_idx}")
            router_keys_raw[l_idx] = key

    is_moe = bool(qwen_exp_lora or qwen_exp_full)
    if is_moe:
        if arch_hint is not None and arch_hint != ARCH_QWEN_MOE:
            raise ValueError(f"arch_hint={arch_hint!r} conflicts with Qwen-MoE layout")
        arch_family = ARCH_QWEN_MOE
        use_lora = bool(qwen_exp_lora)
        if use_lora and qwen_exp_full and not set(qwen_exp_full.keys()).issubset(set(qwen_exp_lora.keys())):
            raise ValueError("Inconsistent mixed LoRA and full-weight parameterization across Qwen-MoE experts")
        parameterization = PARAM_LORA if use_lora else PARAM_FULL_WEIGHT
        source_map = qwen_exp_lora if use_lora else qwen_exp_full
        layers = _validate_contiguous_indices({k[0] for k in source_map}, "Qwen-MoE layer")
        exp_by_layer: dict[int, set[int]] = {}
        for l_idx, e_idx in source_map:
            exp_by_layer.setdefault(l_idx, set()).add(e_idx)
        expert_count = len(_validate_contiguous_indices(exp_by_layer[layers[0]], "Qwen-MoE expert"))
    else:
        if router_keys_raw:
            raise ValueError("Router/gate weights found without MoE expert tensors")
        use_lora = bool(dense_lora)
        if use_lora and dense_full and not set(dense_full.keys()).issubset(set(dense_lora.keys())):
            raise ValueError("Inconsistent mixed LoRA and full-weight parameterization across dense layers")
        parameterization = PARAM_LORA if use_lora else PARAM_FULL_WEIGHT
        dense_map = dense_lora if use_lora else dense_full
        if not dense_map:
            raise ValueError("Unsupported open-weight architecture layout: no MLP projections found")
        if arch_hint is not None:
            if arch_hint not in (ARCH_QWEN_DENSE, ARCH_LLAMA_GQA):
                raise ValueError(f"arch_hint={arch_hint!r} conflicts with dense model.layers layout")
            arch_family = arch_hint
        else:
            arch_family = ARCH_QWEN_DENSE if has_qwen_marker else ARCH_LLAMA_GQA
        layers = _validate_contiguous_indices(set(dense_map.keys()), "HF dense layer")
        expert_count = 1
        exp_by_layer = {l_idx: {0} for l_idx in layers}
        source_map = {(l_idx, 0): dense_map[l_idx] for l_idx in layers}

    hidden_size: int | None = None
    intermediate_size: int | None = None
    layer_ranks: list[int] = []
    down_lora_pairs: dict[tuple[int, int], tuple[str, str]] = {}
    all_lora_b_keys: dict[tuple[int, int], tuple[str, ...]] = {}
    down_full_keys: dict[tuple[int, int], str] = {}
    surgical_keys: set[str] = set()

    for l_idx in layers:
        l_exps = _validate_contiguous_indices(exp_by_layer[l_idx], f"layer {l_idx} expert")
        if len(l_exps) != expert_count:
            raise ValueError(f"Layer {l_idx} has {len(l_exps)} experts; expected {expert_count}")
        rank_l: int | None = None
        for e_idx in l_exps:
            pmap = source_map[(l_idx, e_idx)]
            label = f"model.layers.{l_idx}" + (f".mlp.experts.{e_idx}" if is_moe else ".mlp")
            if use_lora:
                if "down_proj.lora_A" not in pmap or "down_proj.lora_B" not in pmap:
                    raise ValueError(f"Missing down_proj LoRA A/B pair at {label}")
                hidden_size, intermediate_size, rank_l = _validate_down_lora_pair(
                    validated[pmap["down_proj.lora_B"]],
                    validated[pmap["down_proj.lora_A"]],
                    hidden_size,
                    intermediate_size,
                    rank_l,
                    label=label,
                )
                extra_b = _validate_hf_gate_up_shapes(
                    validated,
                    pmap,
                    parameterization=PARAM_LORA,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    label=label,
                )
                down_lora_pairs[(l_idx, e_idx)] = (pmap["down_proj.lora_B"], pmap["down_proj.lora_A"])
                all_lora_b_keys[(l_idx, e_idx)] = tuple([*extra_b, pmap["down_proj.lora_B"]])
                surgical_keys.update((*extra_b, pmap["down_proj.lora_A"], pmap["down_proj.lora_B"]))
            else:
                if "down_proj.weight" not in pmap:
                    raise ValueError(f"Missing down_proj.weight at {label}")
                dw = validated[pmap["down_proj.weight"]]
                if dw.ndim != 2 or not dw.is_floating_point():
                    raise ValueError(f"down_proj.weight must be 2D floating tensor at {label}")
                h_e, i_e = int(dw.shape[0]), int(dw.shape[1])
                if hidden_size is None:
                    hidden_size, intermediate_size = h_e, i_e
                elif h_e != hidden_size or i_e != intermediate_size:
                    raise ValueError(
                        f"Inconsistent dims at {label}: ({h_e}, {i_e}) != ({hidden_size}, {intermediate_size})"
                    )
                rank_l = min(hidden_size, intermediate_size)
                _validate_hf_gate_up_shapes(
                    validated,
                    pmap,
                    parameterization=PARAM_FULL_WEIGHT,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    label=label,
                )
                down_full_keys[(l_idx, e_idx)] = pmap["down_proj.weight"]
                surgical_keys.add(pmap["down_proj.weight"])

        assert rank_l is not None
        layer_ranks.append(rank_l)

    assert hidden_size is not None and intermediate_size is not None
    return _finalize_resolved_layout(
        validated,
        arch_family=arch_family,
        parameterization=parameterization,
        layers=layers,
        expert_count=expert_count,
        is_moe=is_moe,
        is_granite=False,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_ranks=tuple(layer_ranks),
        down_lora_pairs=down_lora_pairs,
        all_lora_b_keys=all_lora_b_keys,
        down_full_keys=down_full_keys,
        router_keys_raw=router_keys_raw,
        surgical_keys=surgical_keys,
    )


def _resolve_layout(
    state_dict: Mapping[str, torch.Tensor],
    *,
    arch_hint: str | None = None,
) -> _ResolvedLayout:
    """Validate ``state_dict`` and deterministically resolve its architecture layout."""
    if arch_hint is not None and arch_hint not in SUPPORTED_ARCH_FAMILIES:
        raise ValueError(
            f"Unsupported arch_hint {arch_hint!r}; expected one of {sorted(SUPPORTED_ARCH_FAMILIES)}"
        )
    validated = _validate_native_state_dict(state_dict, name="state_dict")
    has_granite = any(_GRANITE_EXPERT_RE.match(k) or _GRANITE_ROUTER_RE.match(k) for k in validated)
    has_hf = any(k.startswith("model.layers.") for k in validated)
    if has_granite and has_hf:
        raise ValueError("Ambiguous mixed architecture layouts: both 'blocks.' and 'model.layers.' keys found")
    if has_granite:
        return _resolve_granite_layout(validated, arch_hint=arch_hint)
    if has_hf:
        return _resolve_hf_layout(validated, arch_hint=arch_hint)
    raise ValueError(
        "Unsupported open-weight architecture layout: expected Granite ('blocks.{l}.mlp.experts.'), "
        "Qwen-MoE/Dense ('model.layers.{l}.mlp.*'), or Llama-GQA ('model.layers.{l}.self_attn.*')"
    )


def detect_open_weight_architecture(
    state_dict: Mapping[str, torch.Tensor],
    *,
    arch_hint: str | None = None,
) -> OpenWeightArchitectureSpec:
    """Detect architecture family, layer/expert counts, GQA KV-head ratio, and final norm key."""
    return _resolve_layout(state_dict, arch_hint=arch_hint).spec


# ---------------------------------------------------------------------------
# 3. Canonical S5 State-Dict Bridge & Bitwise-Preserving Materialization
# ---------------------------------------------------------------------------


def _canonicalize_with_layout(
    state_dict: Mapping[str, torch.Tensor],
    layout: _ResolvedLayout,
    *,
    full_weight_max_rank: int | None = None,
) -> dict[str, np.ndarray]:
    """Canonicalize ``state_dict`` using an already-resolved layout descriptor."""
    spec = layout.spec
    canonical: dict[str, np.ndarray] = {}

    for l_idx in range(spec.layer_count):
        b_list: list[np.ndarray] = []
        a_list: list[np.ndarray] = []
        for e_idx in range(spec.expert_count):
            if spec.parameterization == PARAM_LORA:
                b_key, a_key = layout.down_lora_pairs[(l_idx, e_idx)]
                b_arr = _tensor_to_float64_np(state_dict[b_key], b_key)
                a_arr = _tensor_to_float64_np(state_dict[a_key], a_key)
            else:
                w_key = layout.down_full_keys[(l_idx, e_idx)]
                w_arr = _tensor_to_float64_np(state_dict[w_key], w_key)
                b_arr, a_arr = _deterministic_thin_svd_factor(w_arr, max_rank=full_weight_max_rank)
            b_list.append(b_arr)
            a_list.append(a_arr)

        canonical[f"layer_{l_idx}.w2_B"] = np.ascontiguousarray(np.stack(b_list, axis=0), dtype=np.float64)
        canonical[f"layer_{l_idx}.w2_A"] = np.ascontiguousarray(np.stack(a_list, axis=0), dtype=np.float64)

        if l_idx in layout.router_key_by_layer:
            r_key = layout.router_key_by_layer[l_idx]
            canonical[f"layer_{l_idx}.router_w"] = np.ascontiguousarray(
                _tensor_to_float64_np(state_dict[r_key], r_key),
                dtype=np.float64,
            )

    if spec.final_norm_key is not None:
        canonical["final_ln_w"] = np.ascontiguousarray(
            _tensor_to_float64_np(state_dict[spec.final_norm_key], spec.final_norm_key),
            dtype=np.float64,
        )
    if spec.final_norm_bias_key is not None:
        canonical["final_ln_b"] = np.ascontiguousarray(
            _tensor_to_float64_np(state_dict[spec.final_norm_bias_key], spec.final_norm_bias_key),
            dtype=np.float64,
        )
    return canonical


def canonicalize_state_dict_for_s5(
    state_dict: Mapping[str, torch.Tensor],
    *,
    arch_hint: str | None = None,
    full_weight_max_rank: int | None = None,
) -> dict[str, np.ndarray]:
    """Bridge Granite, Qwen, or Llama state dicts into canonical ``AutoOpenWeightEngine`` NumPy arrays."""
    if full_weight_max_rank is not None and (
        isinstance(full_weight_max_rank, bool) or not isinstance(full_weight_max_rank, int) or full_weight_max_rank <= 0
    ):
        raise ValueError("full_weight_max_rank must be a positive integer when provided")

    layout = _resolve_layout(state_dict, arch_hint=arch_hint)
    return _canonicalize_with_layout(state_dict, layout, full_weight_max_rank=full_weight_max_rank)


def _validate_canonical_edited_state(
    canonical_pre: Mapping[str, np.ndarray],
    canonical_edited: Any,
) -> dict[str, np.ndarray]:
    """Validate that ``canonical_edited`` matches expected canonical keys, shapes, and finiteness."""
    if not isinstance(canonical_edited, Mapping) or not canonical_edited:
        raise ValueError("canonical_edited_state must be a non-empty mapping")
    if set(canonical_edited.keys()) != set(canonical_pre.keys()):
        missing = sorted(set(canonical_pre.keys()) - set(canonical_edited.keys()))
        extra = sorted(set(canonical_edited.keys()) - set(canonical_pre.keys()))
        raise ValueError(
            f"canonical_edited_state keys mismatch canonical schema (missing={missing[:3]}, extra={extra[:3]})"
        )
    out: dict[str, np.ndarray] = {}
    for key in sorted(canonical_pre.keys()):
        raw_val = canonical_edited[key]
        arr = (
            raw_val.detach().to(device="cpu", dtype=torch.float64).numpy()
            if isinstance(raw_val, torch.Tensor)
            else np.asarray(raw_val, dtype=np.float64)
        )
        expected_shape = canonical_pre[key].shape
        if arr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch on canonical tensor {key!r}: {arr.shape} != expected {expected_shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Canonical tensor {key!r} contains NaN or Inf values")
        out[key] = np.ascontiguousarray(arr, dtype=np.float64)
    return out


def _np_to_native_tensor(arr: np.ndarray, template: torch.Tensor) -> torch.Tensor:
    """Convert a CPU ``float64`` NumPy array back to ``template.dtype`` and ``template.device``."""
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float64)).to(
        device=template.device,
        dtype=template.dtype,
    )


def _materialize_with_resolved_layout(
    native_state: Mapping[str, torch.Tensor],
    layout: _ResolvedLayout,
    canonical_pre: Mapping[str, np.ndarray],
    canonical_edited_state: Mapping[str, Any],
    *,
    clone_untouched: bool = True,
) -> dict[str, torch.Tensor]:
    """Materialize canonical S5 edits back into ``native_state`` using a pre-resolved layout."""
    spec = layout.spec
    edited = _validate_canonical_edited_state(canonical_pre, canonical_edited_state)
    out = clone_native_state_dict(native_state) if clone_untouched else dict(native_state)

    for l_idx in range(spec.layer_count):
        b_key_c = f"layer_{l_idx}.w2_B"
        a_key_c = f"layer_{l_idx}.w2_A"
        pre_b_layer = canonical_pre[b_key_c]
        pre_a_layer = canonical_pre[a_key_c]
        ed_b_layer = edited[b_key_c]
        ed_a_layer = edited[a_key_c]

        for e_idx in range(spec.expert_count):
            pre_b = pre_b_layer[e_idx]
            pre_a = pre_a_layer[e_idx]
            ed_b = ed_b_layer[e_idx]
            ed_a = ed_a_layer[e_idx]
            if np.array_equal(ed_b, pre_b) and np.array_equal(ed_a, pre_a):
                continue

            was_nonzero = float(np.max(np.abs(pre_b))) > 1e-12
            now_zero = float(np.max(np.abs(ed_b))) <= 1e-12
            if spec.parameterization == PARAM_LORA:
                if was_nonzero and now_zero:
                    for b_nat_key in layout.all_lora_b_keys[(l_idx, e_idx)]:
                        out[b_nat_key] = torch.zeros_like(out[b_nat_key])
                else:
                    b_nat_key, a_nat_key = layout.down_lora_pairs[(l_idx, e_idx)]
                    out[b_nat_key] = _np_to_native_tensor(ed_b, native_state[b_nat_key])
                    out[a_nat_key] = _np_to_native_tensor(ed_a, native_state[a_nat_key])
            else:
                down_nat_key = layout.down_full_keys[(l_idx, e_idx)]
                orig_t = native_state[down_nat_key]
                if was_nonzero and now_zero:
                    out[down_nat_key] = torch.zeros_like(orig_t)
                else:
                    orig_64 = _tensor_to_float64_np(orig_t, down_nat_key)
                    delta_w = (ed_b @ ed_a) - (pre_b @ pre_a)
                    out[down_nat_key] = _np_to_native_tensor(orig_64 + delta_w, orig_t)

        if l_idx in layout.router_key_by_layer:
            r_key_c = f"layer_{l_idx}.router_w"
            if not np.array_equal(edited[r_key_c], canonical_pre[r_key_c]):
                r_nat_key = layout.router_key_by_layer[l_idx]
                out[r_nat_key] = _np_to_native_tensor(edited[r_key_c], native_state[r_nat_key])

    if spec.final_norm_key is not None and not np.array_equal(edited["final_ln_w"], canonical_pre["final_ln_w"]):
        out[spec.final_norm_key] = _np_to_native_tensor(edited["final_ln_w"], native_state[spec.final_norm_key])
    if (
        spec.final_norm_bias_key is not None
        and "final_ln_b" in edited
        and not np.array_equal(edited["final_ln_b"], canonical_pre["final_ln_b"])
    ):
        out[spec.final_norm_bias_key] = _np_to_native_tensor(
            edited["final_ln_b"], native_state[spec.final_norm_bias_key]
        )

    return out


def materialize_s5_back_to_native_layout(
    native_state: Mapping[str, torch.Tensor],
    canonical_edited_state: Mapping[str, Any],
    *,
    arch_hint: str | None = None,
    full_weight_max_rank: int | None = None,
) -> dict[str, torch.Tensor]:
    """Materialize canonical S5 edits back into ``native_state`` layout without weight corruption."""
    layout = _resolve_layout(native_state, arch_hint=arch_hint)
    canonical_pre = _canonicalize_with_layout(native_state, layout, full_weight_max_rank=full_weight_max_rank)
    return _materialize_with_resolved_layout(
        native_state,
        layout,
        canonical_pre,
        canonical_edited_state,
        clone_untouched=True,
    )


# ---------------------------------------------------------------------------
# 4. End-to-End Multi-Architecture S5 Surgery Runner & Provenance Verifier
# ---------------------------------------------------------------------------


def _to_cpu_float64_np(val: Any, name: str) -> np.ndarray:
    """Convert a ``torch.Tensor`` or array-like input into a validated finite ``float64`` NumPy array."""
    arr = (
        val.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        if isinstance(val, torch.Tensor)
        else np.asarray(val, dtype=np.float64)
    )
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be non-empty and contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _normalize_activation_dict(
    raw_map: Mapping[int, Any] | None,
    *,
    expected_width: int,
    layer_count: int,
    name: str,
) -> dict[int, np.ndarray] | None:
    """Normalize per-layer 2D activation tensors to ``dict[int, np.ndarray]``."""
    if raw_map is None:
        return None
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise ValueError(f"{name} must be a non-empty mapping when provided")
    out: dict[int, np.ndarray] = {}
    for l_idx, val in raw_map.items():
        if isinstance(l_idx, bool) or not isinstance(l_idx, int) or not (0 <= l_idx < layer_count):
            raise ValueError(f"{name} layer key {l_idx!r} out of range [0, {layer_count})")
        arr = _to_cpu_float64_np(val, f"{name}[{l_idx}]")
        if arr.ndim != 2 or int(arr.shape[1]) != expected_width:
            raise ValueError(f"{name}[{l_idx}] must have shape (N, {expected_width}), got {arr.shape}")
        out[l_idx] = arr
    return out


def _normalize_specialist_dict(
    raw_map: Mapping[int, Mapping[str, Any]] | None,
    *,
    expected_width: int,
    layer_count: int,
) -> dict[int, dict[str, np.ndarray]] | None:
    """Normalize per-layer specialist domain activations to ``dict[int, dict[str, np.ndarray]]``."""
    if raw_map is None:
        return None
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise ValueError("layer_specialist_inputs must be a non-empty mapping when provided")
    out: dict[int, dict[str, np.ndarray]] = {}
    for l_idx, dom_map in raw_map.items():
        if isinstance(l_idx, bool) or not isinstance(l_idx, int) or not (0 <= l_idx < layer_count):
            raise ValueError(f"layer_specialist_inputs layer key {l_idx!r} out of range [0, {layer_count})")
        if not isinstance(dom_map, Mapping) or not dom_map:
            raise ValueError(f"layer_specialist_inputs[{l_idx}] must be a non-empty mapping")
        dom_out: dict[str, np.ndarray] = {}
        for dom_name, val in dom_map.items():
            if not isinstance(dom_name, str) or not dom_name.strip():
                raise ValueError(f"Specialist domain name at layer {l_idx} must be a non-empty string")
            arr = _to_cpu_float64_np(val, f"layer_specialist_inputs[{l_idx}][{dom_name!r}]")
            if arr.ndim != 2 or int(arr.shape[1]) != expected_width:
                raise ValueError(
                    f"layer_specialist_inputs[{l_idx}][{dom_name!r}] must have shape (N, {expected_width}), "
                    f"got {arr.shape}"
                )
            dom_out[dom_name] = arr
        out[l_idx] = dom_out
    return out


def verify_multi_arch_receipt_provenance(result: MultiArchAutoWSPResult | Mapping[str, Any]) -> bool:
    """Independently verify SHA-256 provenance of a ``MultiArchAutoWSPResult`` receipt."""
    if isinstance(result, MultiArchAutoWSPResult):
        payload = result.to_dict()
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        raise TypeError(f"result must be MultiArchAutoWSPResult or Mapping, got {type(result).__name__}")

    claimed = payload.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if payload.get("schema") != SCHEMA_VERSION or payload.get("object") != RECEIPT_OBJECT:
        return False
    core = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    try:
        expected = _compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def run_multi_arch_auto_open_weight_s5(
    native_state: Mapping[str, torch.Tensor],
    base_domain_nll: Mapping[str, float],
    native_eval_fn: Callable[[Mapping[str, torch.Tensor]], Mapping[str, float]],
    **s5_kwargs: Any,
) -> tuple[dict[str, torch.Tensor], MultiArchAutoWSPResult]:
    """Execute training-free ``S5`` PASS-Synergy surgery end-to-end on Granite, Qwen, or Llama layouts."""
    if not callable(native_eval_fn):
        raise TypeError("native_eval_fn must be callable")
    unknown_keys = sorted(set(s5_kwargs.keys()) - _ALLOWED_S5_KWARGS)
    if unknown_keys:
        raise TypeError(f"Unexpected keyword argument(s) for run_multi_arch_auto_open_weight_s5: {unknown_keys}")
    if "layer_e0_harm_nats" not in s5_kwargs:
        raise ValueError("layer_e0_harm_nats is required")

    arch_hint = s5_kwargs.get("arch_hint")
    full_weight_max_rank = s5_kwargs.get("full_weight_max_rank")
    if full_weight_max_rank is not None and (
        isinstance(full_weight_max_rank, bool) or not isinstance(full_weight_max_rank, int) or full_weight_max_rank <= 0
    ):
        raise ValueError("full_weight_max_rank must be a positive integer when provided")

    layout = _resolve_layout(native_state, arch_hint=arch_hint)
    spec = layout.spec
    pre_native_sha256 = hash_native_state_dict(native_state)

    canon_pre = _canonicalize_with_layout(
        native_state,
        layout,
        full_weight_max_rank=full_weight_max_rank,
    )
    canonical_pre_sha256 = hash_state_dict(canon_pre)

    # Verify strict lossless pre-flight roundtrip before any surgery begins.
    roundtrip_native = _materialize_with_resolved_layout(
        native_state,
        layout,
        canon_pre,
        canon_pre,
        clone_untouched=True,
    )
    roundtrip_ok = hash_native_state_dict(roundtrip_native) == pre_native_sha256

    layer_harm = [float(x) for x in s5_kwargs["layer_e0_harm_nats"]]
    if len(layer_harm) != spec.layer_count:
        raise ValueError(
            f"layer_e0_harm_nats length {len(layer_harm)} != detected layer_count {spec.layer_count}"
        )

    raw_exp_harm = s5_kwargs.get("expert_e0_harm_nats")
    raw_exp_spec = s5_kwargs.get("expert_spec_loss_when_zeroed")
    if raw_exp_harm is None or raw_exp_spec is None:
        if spec.expert_count == 1:
            exp_harm = [0.0] if raw_exp_harm is None else [float(x) for x in raw_exp_harm]
            exp_spec = [0.0] if raw_exp_spec is None else [float(x) for x in raw_exp_spec]
        else:
            raise ValueError("expert_e0_harm_nats and expert_spec_loss_when_zeroed are required for MoE models")
    else:
        exp_harm = [float(x) for x in raw_exp_harm]
        exp_spec = [float(x) for x in raw_exp_spec]

    if len(exp_harm) != spec.expert_count or len(exp_spec) != spec.expert_count:
        raise ValueError(
            f"Expert attribution length ({len(exp_harm)}, {len(exp_spec)}) != detected expert_count {spec.expert_count}"
        )

    prose_inputs = _normalize_activation_dict(
        s5_kwargs.get("layer_prose_inputs"),
        expected_width=spec.hidden_size,
        layer_count=spec.layer_count,
        name="layer_prose_inputs",
    )
    spec_inputs = _normalize_specialist_dict(
        s5_kwargs.get("layer_specialist_inputs"),
        expected_width=spec.hidden_size,
        layer_count=spec.layer_count,
    )
    tail_mid = _normalize_activation_dict(
        s5_kwargs.get("tail_layer_prose_ffn_mid"),
        expected_width=spec.intermediate_size,
        layer_count=spec.layer_count,
        name="tail_layer_prose_ffn_mid",
    )

    cfg: AutoOpenWeightConfig = s5_kwargs.get("config") or AutoOpenWeightConfig()
    if not isinstance(cfg, AutoOpenWeightConfig):
        raise TypeError(f"config must be AutoOpenWeightConfig or None, got {type(cfg).__name__}")

    base_state = s5_kwargs.get("base_state")
    raw_base_ln_w = s5_kwargs.get("base_final_ln_w")
    raw_base_ln_b = s5_kwargs.get("base_final_ln_b")
    if raw_base_ln_w is None and isinstance(base_state, Mapping) and spec.final_norm_key is not None:
        if spec.final_norm_key in base_state:
            raw_base_ln_w = base_state[spec.final_norm_key]
    if raw_base_ln_b is None and isinstance(base_state, Mapping) and spec.final_norm_bias_key is not None:
        if spec.final_norm_bias_key in base_state:
            raw_base_ln_b = base_state[spec.final_norm_bias_key]

    base_ln_w_np: np.ndarray | None = None
    base_ln_b_np: np.ndarray | None = None
    auto_temp = bool(s5_kwargs.get("auto_temperature_calibration", True))

    if raw_base_ln_b is not None and raw_base_ln_w is None:
        raise ValueError("base_final_ln_b requires base_final_ln_w")
    if raw_base_ln_b is not None and spec.final_norm_bias_key is None:
        raise ValueError("base_final_ln_b cannot be supplied for biasless norm architecture")

    if raw_base_ln_w is not None:
        base_ln_w_np = _to_cpu_float64_np(raw_base_ln_w, "base_final_ln_w")
        if base_ln_w_np.shape != (spec.hidden_size,):
            raise ValueError(f"base_final_ln_w shape {base_ln_w_np.shape} != ({spec.hidden_size},)")
    elif auto_temp and "final_ln_w" in canon_pre:
        base_ln_w_np = np.array(canon_pre["final_ln_w"], copy=True)
        cfg = dataclasses.replace(cfg, lambda_ln_grid=(1.0,))

    if raw_base_ln_b is not None:
        base_ln_b_np = _to_cpu_float64_np(raw_base_ln_b, "base_final_ln_b")
        if base_ln_b_np.shape != (spec.hidden_size,):
            raise ValueError(f"base_final_ln_b shape {base_ln_b_np.shape} != ({spec.hidden_size},)")
    elif auto_temp and raw_base_ln_w is None and "final_ln_b" in canon_pre:
        base_ln_b_np = np.array(canon_pre["final_ln_b"], copy=True)

    def _canon_eval(canon_candidate: Mapping[str, np.ndarray]) -> dict[str, float]:
        native_cand = _materialize_with_resolved_layout(
            native_state,
            layout,
            canon_pre,
            canon_candidate,
            clone_untouched=True,
        )
        raw_nll = native_eval_fn(native_cand)
        return {str(k): float(v.item() if isinstance(v, torch.Tensor) else v) for k, v in raw_nll.items()}

    canon_edited, s5_res = run_auto_open_weight_s5_synergy(
        canon_pre,
        base_domain_nll,
        _canon_eval,
        layer_e0_harm_nats=layer_harm,
        expert_e0_harm_nats=exp_harm,
        expert_spec_loss_when_zeroed=exp_spec,
        layer_prose_inputs=prose_inputs,
        layer_specialist_inputs=spec_inputs,
        tail_layer_prose_ffn_mid=tail_mid,
        base_final_ln_w=base_ln_w_np,
        base_final_ln_b=base_ln_b_np,
        extra_custom_stages=s5_kwargs.get("extra_custom_stages", ()),
        config=cfg,
    )

    if s5_res.committed:
        final_native = _materialize_with_resolved_layout(
            native_state,
            layout,
            canon_pre,
            canon_edited,
            clone_untouched=True,
        )
    else:
        final_native = clone_native_state_dict(native_state)

    post_native_sha256 = hash_native_state_dict(final_native)
    canonical_post_sha256 = hash_state_dict(canon_edited)
    edited_params, total_params = count_native_edited_parameters(native_state, final_native)

    untouched_preserved = all(
        native_state[k].dtype == final_native[k].dtype
        and torch.equal(native_state[k].detach().cpu(), final_native[k].detach().cpu())
        for k in spec.untouched_keys
    )

    nll_vals = list(s5_res.final_domain_nll.values())
    nll_mean = float(np.mean(nll_vals)) if nll_vals else float(ANSE_MAXIMUM_PAIN)
    if not math.isfinite(nll_mean):
        nll_mean = float(ANSE_MAXIMUM_PAIN)

    result = MultiArchAutoWSPResult(
        committed=s5_res.committed,
        verdict=s5_res.verdict,
        arch_family=spec.arch_family,
        parameterization=spec.parameterization,
        layer_count=spec.layer_count,
        expert_count=spec.expert_count,
        gqa_kv_head_ratio=spec.gqa_kv_head_ratio,
        final_norm_key=spec.final_norm_key,
        norm_type=spec.norm_type,
        anse_energy=s5_res.anse_energy,
        nll_mean=nll_mean,
        mean_diff_vs_base=s5_res.mean_diff_vs_base,
        general_rel_drift=s5_res.general_rel_drift,
        all_domains_negative=s5_res.all_domains_negative,
        native_total_params=total_params,
        native_edited_params=edited_params,
        native_edited_fraction=float(edited_params) / float(max(total_params, 1)),
        untouched_keys_count=len(spec.untouched_keys),
        untouched_keys_bitwise_preserved=untouched_preserved,
        selected_lambda_ln=s5_res.selected_lambda_ln,
        selected_tau_temp=s5_res.selected_tau_temp,
        pre_native_sha256=pre_native_sha256,
        post_native_sha256=post_native_sha256,
        canonical_pre_sha256=canonical_pre_sha256,
        canonical_post_sha256=canonical_post_sha256,
        roundtrip_verified=roundtrip_ok,
        arch_spec=spec,
        s5_result=s5_res,
        receipt_sha256="",
    )
    core_payload = {k: v for k, v in result.to_dict().items() if k != "receipt_sha256"}
    receipt_sha = _compute_payload_sha256(core_payload)
    return final_native, dataclasses.replace(result, receipt_sha256=receipt_sha)
