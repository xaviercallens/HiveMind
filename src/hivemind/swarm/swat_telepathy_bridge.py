# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Season 34 (Wave 32) — Swarm Telepathy (`SwaT`) Prefill-Free Cross-Family KV-Cache Bridge.

Implements `HM-S34-01`, `HM-S34-03`, `HM-S34-04`, `HM-S34-05`, and `HM-S34-06`, bridging
`third_party/SwaT/multiagent/` (`arXiv:2610.15095v1`, Alabdulkareem et al. 2026) and
Attention Matching (`arXiv:2602.16284`, Zweiger et al. 2026) into Project HiveMind's
Bi-Layer Honeybee Agent Mesh under strict hardware VRAM guards:
- Local NVIDIA GeForce RTX 2070 (`8 GB`): `< 6,656 MiB` Windows WDDM ceiling (`>= 2,048 MiB` free).
- Remote NVIDIA Tesla T4 (`16 GB`): `>= 4,300 MiB` free VRAM headroom guard.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


RTX2070_WDDM_CEILING_MIB: float = 6656.0
RTX2070_FREE_HEADROOM_GUARD_MIB: float = 2048.0
TESLA_T4_FREE_HEADROOM_GUARD_MIB: float = 4300.0


@dataclass(frozen=True)
class ModelKVSpec:
    """KV-cache geometry for a frozen Sender or Receiver Bee model."""

    model_id: str
    family: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    rope_theta: float = 10000.0
    max_context_tokens: int = 8192
    is_vlm: bool = False

    @property
    def flat_layer_dim(self) -> int:
        return 2 * self.num_kv_heads * self.head_dim

    @property
    def flat_total_dim(self) -> int:
        return self.num_layers * self.flat_layer_dim

    def kv_cache_mib(self, seq_len: int, bytes_per_elem: int = 2) -> float:
        total_bytes = seq_len * self.flat_total_dim * bytes_per_elem
        return float(total_bytes) / float(1024 * 1024)


S34_BEE_MODEL_SPECS: Mapping[str, ModelKVSpec] = {
    "Qwen/Qwen3.5-0.8B": ModelKVSpec("Qwen/Qwen3.5-0.8B", "qwen3.5", 24, 2, 64, 1024, 1000000.0, 32768),
    "Qwen/Qwen3.5-4B": ModelKVSpec("Qwen/Qwen3.5-4B", "qwen3.5", 36, 4, 128, 2560, 1000000.0, 32768),
    "HuggingFaceTB/SmolLM3-3B": ModelKVSpec("HuggingFaceTB/SmolLM3-3B", "smollm3", 32, 4, 128, 2048, 500000.0, 16384),
    "google/gemma-3-4b-it": ModelKVSpec("google/gemma-3-4b-it", "gemma3", 34, 4, 128, 2560, 1000000.0, 32768, is_vlm=True),
    "ibm-granite/granite-3.3-8b-instruct": ModelKVSpec(
        "ibm-granite/granite-3.3-8b-instruct", "granite3.3", 40, 8, 128, 4096, 100000.0, 32768
    ),
}


def compute_layer_group_slices(total_layers: int, n_blocks: int = 8) -> list[tuple[int, int]]:
    """Partition `total_layers` into `n_blocks` contiguous groups matching `CacheRendererV2`."""
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    if total_layers < n_blocks:
        raise ValueError(f"total_layers ({total_layers}) must be at least n_blocks ({n_blocks})")
    return [
        (total_layers * b // n_blocks, total_layers * (b + 1) // n_blocks)
        for b in range(n_blocks)
    ]


def compute_virtual_positions(
    sender_offsets: Sequence[tuple[int, int]],
    receiver_offsets: Sequence[tuple[int, int]],
    mode: str = "interp",
) -> np.ndarray:
    """Compute cross-tokenizer character-span RoPE positions `p_i` (`arXiv:2610.15095` Eq. 4).

    When `sender_offsets == receiver_offsets` with positive-width character spans, the midpoint
    of token `i` falls at `fraction = 0.5` inside receiver token `i`, yielding the exact
    positive-control identity `p_i = i + 0.5 - 0.5 = i`.
    """
    sender_count = len(sender_offsets)
    receiver_count = len(receiver_offsets)
    if sender_count == 0:
        return np.empty(0, dtype=np.float64)
    if receiver_count == 0:
        raise ValueError("receiver_offsets must be non-empty when sender_offsets is non-empty")

    if mode == "naive":
        return np.arange(sender_count, dtype=np.float64)
    if mode == "scaled":
        return np.arange(sender_count, dtype=np.float64) * (receiver_count / max(sender_count, 1))
    if mode != "interp":
        raise ValueError(f"unknown position mode {mode!r}")

    receiver_starts = [int(start) for start, _ in receiver_offsets]
    positions = np.empty(sender_count, dtype=np.float64)
    last_receiver = receiver_count - 1

    for j, (s_start, s_end) in enumerate(sender_offsets):
        midpoint = (float(s_start) + float(s_end)) / 2.0
        k = bisect_right(receiver_starts, midpoint) - 1
        k = min(max(k, 0), last_receiver)
        r_start, r_end = receiver_offsets[k]
        width = max(float(r_end - r_start), 1.0)
        fraction = min(max((midpoint - float(r_start)) / width, 0.0), 1.0)
        positions[j] = max(float(k) + fraction - 0.5, 0.0)
    return positions


def compute_cross_causal_limits(
    sender_offsets: Sequence[tuple[int, int]],
    receiver_offsets: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Return per-receiver-token sender prefix count `limit[j]` such that `b^S_i <= b^R_j`.

    Receiver token `r_j` may attend only to sender indices `i < limit[j]`, guaranteeing
    zero future-token leakage across mismatched BPE boundaries.
    """
    if not receiver_offsets:
        return np.empty(0, dtype=np.int64)
    if not sender_offsets:
        return np.zeros(len(receiver_offsets), dtype=np.int64)
    sender_ends = np.maximum.accumulate(np.asarray([float(end) for _, end in sender_offsets], dtype=np.float64))
    receiver_ends = np.asarray([float(end) for _, end in receiver_offsets], dtype=np.float64)
    return np.searchsorted(sender_ends, receiver_ends, side="right").astype(np.int64)


def cross_attention_with_causal_limits(
    receiver_queries: np.ndarray,
    sender_keys: np.ndarray,
    sender_values: np.ndarray,
    causal_limits: np.ndarray,
) -> np.ndarray:
    """Causal cross-attention where receiver query `j` only attends to sender tokens `< causal_limits[j]`.

    Shapes:
    - `receiver_queries`: `(seq_r, dim)`
    - `sender_keys`: `(seq_s, dim)`
    - `sender_values`: `(seq_s, dim)`
    - `causal_limits`: `(seq_r,)`
    """
    seq_r, dim = receiver_queries.shape
    seq_s = sender_keys.shape[0]
    out = np.zeros((seq_r, dim), dtype=np.float64)
    scale = 1.0 / math.sqrt(max(dim, 1))
    for j in range(seq_r):
        limit = int(min(max(int(causal_limits[j]), 0), seq_s))
        if limit == 0:
            continue
        k_allowed = sender_keys[:limit]
        v_allowed = sender_values[:limit]
        scores = (receiver_queries[j : j + 1] @ k_allowed.T) * scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), 1e-12)
        out[j] = (weights @ v_allowed)[0]
    return out


def bound_kv_to_rms_stats(
    pred_kv: np.ndarray,
    target_key_rms: np.ndarray,
    target_value_rms: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Rescale predicted pre-RoPE KV cache to match calibrated receiver per-layer, per-head RMS.

    `pred_kv` has shape `(batch, seq_len, num_layers, 2, num_kv_heads, head_dim)`.
    `target_key_rms` and `target_value_rms` have shape `(num_layers, num_kv_heads)`.
    """
    arr = np.asarray(pred_kv, dtype=np.float64)
    if arr.ndim != 6 or arr.shape[3] != 2:
        raise ValueError("pred_kv must have shape (batch, seq_len, num_layers, 2, num_kv_heads, head_dim)")
    rms = np.sqrt(np.mean(np.square(arr), axis=(1, 5)))
    rms = np.maximum(rms, float(eps))
    target = np.stack([np.asarray(target_key_rms, dtype=np.float64), np.asarray(target_value_rms, dtype=np.float64)], axis=1)
    scale = target[None, :, :, :] / rms
    return arr * scale[:, None, :, :, :, None]


def audit_latent_kv_safety(
    pred_kv: np.ndarray,
    target_key_rms: np.ndarray,
    target_value_rms: np.ndarray,
    max_rms_ratio: float = 1.35,
    min_rms_ratio: float = 0.65,
    refusal_direction: np.ndarray | None = None,
    min_refusal_cosine: float = -0.35,
) -> dict[str, Any]:
    """Screen an incoming latent KV cache (`HM-S34-06`) for RMS scale anomalies or refusal suppression."""
    arr = np.asarray(pred_kv, dtype=np.float64)
    reasons: list[str] = []
    if not np.all(np.isfinite(arr)):
        reasons.append("non_finite_kv_entries")
        return {
            "accepted": False,
            "reasons": reasons,
            "max_observed_rms_ratio": float("inf"),
            "min_observed_rms_ratio": 0.0,
            "refusal_cosine": 0.0,
        }

    rms = np.sqrt(np.mean(np.square(arr), axis=(1, 5)))
    target = np.stack([np.asarray(target_key_rms, dtype=np.float64), np.asarray(target_value_rms, dtype=np.float64)], axis=1)
    ratios = rms / np.maximum(target[None, :, :, :], 1e-8)
    max_ratio = float(np.max(ratios))
    min_ratio = float(np.min(ratios))

    if max_ratio > max_rms_ratio:
        reasons.append(f"rms_scale_inflation:{max_ratio:.3f}>{max_rms_ratio:.3f}")
    if min_ratio < min_rms_ratio:
        reasons.append(f"rms_scale_collapse:{min_ratio:.3f}<{min_rms_ratio:.3f}")

    refusal_cos = 0.0
    if refusal_direction is not None:
        ref_vec = np.asarray(refusal_direction, dtype=np.float64).reshape(-1)
        flat_mean = np.mean(arr, axis=(0, 1, 2, 3, 4)).reshape(-1)
        if ref_vec.shape == flat_mean.shape:
            denom = float(np.linalg.norm(ref_vec) * np.linalg.norm(flat_mean))
            if denom > 1e-12:
                refusal_cos = float(np.dot(flat_mean, ref_vec) / denom)
                if refusal_cos < min_refusal_cosine:
                    reasons.append(f"refusal_direction_suppressed:{refusal_cos:.3f}<{min_refusal_cosine:.3f}")

    return {
        "accepted": len(reasons) == 0,
        "reasons": reasons,
        "max_observed_rms_ratio": round(max_ratio, 6),
        "min_observed_rms_ratio": round(min_ratio, 6),
        "refusal_cosine": round(refusal_cos, 6),
    }


@dataclass
class UniversalHiveLatentHub:
    """Universal Shared Hive Latent KV Space (`HM-S34-03`: `2n` Hub Mappers vs `n(n-1)` Pairwise Mappers)."""

    n_blocks: int = 8
    d_model: int = 1024
    n_heads: int = 16
    registered_models: dict[str, ModelKVSpec] = field(default_factory=lambda: dict(S34_BEE_MODEL_SPECS))

    def register_model(self, spec: ModelKVSpec) -> None:
        self.registered_models[spec.model_id] = spec

    @property
    def num_families(self) -> int:
        return len(self.registered_models)

    @property
    def hub_mapper_count(self) -> int:
        return 2 * self.num_families

    @property
    def pairwise_mapper_count(self) -> int:
        n = self.num_families
        return n * max(0, n - 1)

    def complexity_report(self) -> dict[str, Any]:
        n = self.num_families
        hub_2n = self.hub_mapper_count
        pairwise = self.pairwise_mapper_count
        return {
            "num_models": n,
            "hub_mappers_2n": hub_2n,
            "pairwise_mappers_n_n_minus_1": pairwise,
            "mappers_saved": pairwise - hub_2n,
            "strictly_fewer_mappers": (hub_2n < pairwise) if n >= 4 else False,
            "canonical_hub_geometry": {
                "n_blocks": self.n_blocks,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
            },
        }


@dataclass
class SwaTTelepathyBridge:
    """Orchestrates prefill-free `SwaT` + `AttentionMatching` (`AM_U`) KV handoffs across Bee families."""

    hub: UniversalHiveLatentHub = field(default_factory=UniversalHiveLatentHub)
    n_blocks: int = 8

    @staticmethod
    def should_fallback_to_native_vlm(
        sender_spec: ModelKVSpec,
        receiver_spec: ModelKVSpec,
        ocr_text_density: float = 0.0,
        dense_ocr_threshold: float = 0.45,
    ) -> dict[str, Any]:
        """Enforce SwaT Table 21 boundary (`HM-S34-04`): fall back to native VLM on dense-OCR inputs."""
        if sender_spec.is_vlm and not receiver_spec.is_vlm and ocr_text_density >= dense_ocr_threshold:
            return {
                "use_swat_kv_transfer": False,
                "fallback_mode": "native_vlm_sender_decode",
                "reason": f"ocr_text_density={ocr_text_density:.2f}>={dense_ocr_threshold:.2f} (Table 21 DocVQA guard)",
            }
        return {
            "use_swat_kv_transfer": True,
            "fallback_mode": "swat_prefill_free_kv",
            "reason": "eligible_for_swat_transfer",
        }

    def plan_transfer(
        self,
        sender_model_id: str,
        receiver_model_id: str,
        sender_offsets: Sequence[tuple[int, int]],
        receiver_offsets: Sequence[tuple[int, int]],
        *,
        am_compaction_ratio: float = 5.0,
        rtx2070_used_vram_mib: float = 1447.0,
        t4_free_vram_mib: float = 8200.0,
        ocr_text_density: float = 0.0,
    ) -> dict[str, Any]:
        """Compute cross-family alignment, layer slices, compacted wire footprint, and VRAM guard compliance."""
        if sender_model_id not in self.hub.registered_models:
            raise KeyError(f"Unknown sender_model_id: {sender_model_id!r}")
        if receiver_model_id not in self.hub.registered_models:
            raise KeyError(f"Unknown receiver_model_id: {receiver_model_id!r}")
        if am_compaction_ratio < 1.0:
            raise ValueError("am_compaction_ratio must be >= 1.0")

        sender_spec = self.hub.registered_models[sender_model_id]
        receiver_spec = self.hub.registered_models[receiver_model_id]

        virt_pos = compute_virtual_positions(sender_offsets, receiver_offsets, mode="interp")
        causal_limits = compute_cross_causal_limits(sender_offsets, receiver_offsets)
        sender_slices = compute_layer_group_slices(sender_spec.num_layers, self.n_blocks)
        receiver_slices = compute_layer_group_slices(receiver_spec.num_layers, self.n_blocks)

        seq_r = len(receiver_offsets)
        compacted_tokens = max(1, int(math.ceil(seq_r / am_compaction_ratio)))
        raw_fp16_mib = receiver_spec.kv_cache_mib(seq_r, bytes_per_elem=2)
        compacted_fp8_mib = receiver_spec.kv_cache_mib(compacted_tokens, bytes_per_elem=1)

        projected_rtx2070_mib = float(rtx2070_used_vram_mib) + compacted_fp8_mib
        rtx2070_free_mib = 8192.0 - projected_rtx2070_mib
        wddm_ok = projected_rtx2070_mib < RTX2070_WDDM_CEILING_MIB
        rtx_headroom_ok = rtx2070_free_mib >= RTX2070_FREE_HEADROOM_GUARD_MIB
        t4_headroom_ok = float(t4_free_vram_mib) >= TESLA_T4_FREE_HEADROOM_GUARD_MIB

        vlm_gate = self.should_fallback_to_native_vlm(
            sender_spec, receiver_spec, ocr_text_density=ocr_text_density
        )
        manifest_seed = (
            f"{sender_model_id}->{receiver_model_id}:{len(sender_offsets)}->{seq_r}:"
            f"{compacted_tokens}:{virt_pos.tobytes().hex()[:32]}"
        )
        transfer_sha256 = hashlib.sha256(manifest_seed.encode("utf-8")).hexdigest()

        return {
            "schema": "hivemind.mesh.telepathy_kv.v1",
            "sender_model_id": sender_model_id,
            "receiver_model_id": receiver_model_id,
            "sender_tokens": len(sender_offsets),
            "receiver_tokens": seq_r,
            "compacted_tokens": compacted_tokens,
            "am_compaction_ratio": am_compaction_ratio,
            "virtual_positions": virt_pos.tolist(),
            "cross_causal_limits": causal_limits.tolist(),
            "sender_layer_slices": sender_slices,
            "receiver_layer_slices": receiver_slices,
            "raw_fp16_kv_mib": round(raw_fp16_mib, 4),
            "compacted_fp8_kv_mib": round(compacted_fp8_mib, 4),
            "vram_guards": {
                "rtx2070_projected_used_mib": round(projected_rtx2070_mib, 2),
                "rtx2070_projected_free_mib": round(rtx2070_free_mib, 2),
                "rtx2070_wddm_6656_ok": wddm_ok,
                "rtx2070_headroom_2048_ok": rtx_headroom_ok,
                "tesla_t4_free_4300_ok": t4_headroom_ok,
                "all_guards_passed": bool(wddm_ok and rtx_headroom_ok and t4_headroom_ok),
            },
            "vlm_routing_gate": vlm_gate,
            "hub_complexity": self.hub.complexity_report(),
            "transfer_sha256": transfer_sha256,
        }


def compute_cms_chunk_periods(
    frequencies: Sequence[float],
    max_chunk_tokens: int = 10000,
) -> list[int]:
    """Compute Continuum Memory System (`CMS`) chunk periods `C^(l) = floor(max C / f_l)` (`HM-S34-08`, `arXiv:2606.03979v2` §3.2).

    Requires positive frequencies in non-increasing order `f_1 >= f_2 >= ... >= f_k > 0`.
    """
    if not frequencies:
        raise ValueError("frequencies must be non-empty")
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    periods: list[int] = []
    prev_f = float("inf")
    for f in frequencies:
        fv = float(f)
        if fv <= 0.0:
            raise ValueError(f"CMS frequency must be positive, got {fv}")
        if fv > prev_f + 1e-12:
            raise ValueError("CMS frequencies must be ordered f_1 >= f_2 >= ... >= f_k")
        prev_f = fv
        periods.append(max(1, int(round(float(max_chunk_tokens) / fv))))
    return periods


def normalized_levenshtein_similarity(seq_a: Sequence[int] | str, seq_b: Sequence[int] | str) -> float:
    """Return `r_abs = 1 - edit_distance(seq_a, seq_b) / max(|seq_a|, |seq_b|)` in `[0, 1]` (`arXiv:2606.03979v2` §3.3)."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return 1.0
    denom = float(max(n, m))
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        a_i = seq_a[i - 1]
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a_i == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return max(0.0, 1.0 - float(dp[m]) / denom)


def compute_knowledge_seeding_loss(
    student_log_probs: np.ndarray,
    teacher_log_probs: np.ndarray,
    student_tokens: Sequence[int] | str,
    teacher_tokens: Sequence[int] | str,
    *,
    semantic_equivalence: float,
    lambda_gkd: float = 1.0,
    lambda_lti: float = 0.5,
    alpha_sem: float = 0.7,
    alpha_abs: float = 0.3,
) -> dict[str, float]:
    """Compute Stage 1 NREM Upward Distillation (`Knowledge Seeding`, `KS`) objective (`HM-S34-08`, `arXiv:2606.03979v2` §3.3).

    Combines On-Policy Generalized Knowledge Distillation (`GKD`, forward KL on student rollouts)
    with RL Learning-to-Imitate (`LTI`) rewarding both semantic equivalence (`r_sem`) and
    normalized Levenshtein prefix similarity (`r_abs`).
    """
    s_lp = np.asarray(student_log_probs, dtype=np.float64)
    t_lp = np.asarray(teacher_log_probs, dtype=np.float64)
    if s_lp.shape != t_lp.shape or s_lp.size == 0:
        raise ValueError("student_log_probs and teacher_log_probs must have identical non-empty shape")

    # Normalize log-probs over vocabulary axis (-1)
    s_lp = s_lp - np.log(np.maximum(np.sum(np.exp(s_lp - np.max(s_lp, axis=-1, keepdims=True)), axis=-1, keepdims=True), 1e-12)) - np.max(s_lp, axis=-1, keepdims=True)
    t_lp = t_lp - np.log(np.maximum(np.sum(np.exp(t_lp - np.max(t_lp, axis=-1, keepdims=True)), axis=-1, keepdims=True), 1e-12)) - np.max(t_lp, axis=-1, keepdims=True)
    t_prob = np.exp(t_lp)

    # Token-averaged KL(teacher || student)
    kl_per_token = np.sum(t_prob * (t_lp - s_lp), axis=-1)
    gkd_loss = float(np.mean(np.maximum(kl_per_token, 0.0)))

    r_sem = float(min(max(semantic_equivalence, 0.0), 1.0))
    r_abs = normalized_levenshtein_similarity(student_tokens, teacher_tokens)
    lti_reward = float(alpha_sem * r_sem + alpha_abs * r_abs)
    lti_loss = max(0.0, 1.0 - lti_reward)

    total_ks_loss = float(lambda_gkd * gkd_loss + lambda_lti * lti_loss)
    return {
        "ks_total_loss": round(total_ks_loss, 6),
        "gkd_loss": round(gkd_loss, 6),
        "lti_loss": round(lti_loss, 6),
        "lti_reward": round(lti_reward, 6),
        "r_sem": round(r_sem, 6),
        "r_abs": round(r_abs, 6),
    }


def select_rem_dreams_by_gradient_norm(
    dream_ids: Sequence[str],
    sft_gradient_norms: Sequence[float],
    injected_random_experts: Sequence[int],
    *,
    top_k: int = 12,
    random_b: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Select Top-`k` highest SFT gradient-norm dreams + `b` random diversity dreams (`HM-S34-09`, `arXiv:2606.03979v2` §3.4)."""
    m = len(dream_ids)
    if m == 0 or len(sft_gradient_norms) != m or len(injected_random_experts) != m:
        raise ValueError("dream_ids, sft_gradient_norms, and injected_random_experts must have equal non-empty length")
    if top_k < 0 or random_b < 0 or (top_k + random_b) == 0:
        raise ValueError("top_k + random_b must be positive")

    norms = np.asarray(sft_gradient_norms, dtype=np.float64)
    ranked_indices = np.argsort(-norms, kind="mergesort").tolist()
    k_eff = min(top_k, m)
    top_indices = ranked_indices[:k_eff]
    remaining = ranked_indices[k_eff:]

    rng = np.random.default_rng(seed)
    b_eff = min(random_b, len(remaining))
    if b_eff > 0:
        chosen_random = rng.choice(remaining, size=b_eff, replace=False).tolist()
    else:
        chosen_random = []

    selected_indices = top_indices + chosen_random
    speedup_vs_seal = float(m) / float(max(len(selected_indices), 1))
    return {
        "total_candidates_m": m,
        "selected_count": len(selected_indices),
        "top_k_indices": top_indices,
        "random_b_indices": chosen_random,
        "selected_dream_ids": [dream_ids[i] for i in selected_indices],
        "selected_injected_experts": [int(injected_random_experts[i]) for i in selected_indices],
        "mean_selected_grad_norm": round(float(np.mean(norms[selected_indices])), 6),
        "mean_rejected_grad_norm": round(float(np.mean(norms[remaining])) if remaining else 0.0, 6),
        "inner_loop_speedup_vs_seal": round(speedup_vs_seal, 3),
    }

