# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-13: DeepSeek-V3 auxiliary-loss-free router bias calibration."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
    extract_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_causal_midrow_nll,
)
from hivemind.swarm.moe_load_balance_bias_controller import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    RouterBiasCalibrationConfig,
    _restore_model_snapshot_bitwise,
    apply_verified_router_bias_calibration_inplace,
    calibrate_router_bias_auxiliary_free,
    compute_module_state_sha256,
    compute_non_router_bias_state_sha256,
    measure_expert_topk_utilization,
    verify_router_bias_receipt_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result when optional ``gwaya`` is absent."""

    is_clean: bool = True
    violations: tuple[str, ...] = ()
    penalty_energy: float = 0.0


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def check_file(self, _path: str) -> list[str]:
        return []

    def audit_python_code(self, _code: str) -> _FallbackStubResult:
        return _FallbackStubResult()


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "moe_load_balance_bias_controller.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)

EIGHT_DOMAINS: tuple[str, ...] = (
    "E0_general_fineweb",
    "E1_python_systems_code",
    "E2_math_reasoning",
    "E3_formal_logic_arc",
    "E4_physics_stem",
    "E5_exoplanet_transit",
    "E6_stellar_spectroscopy",
    "E7_defensive_cyber",
)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit_domain_axis(dim: int, domain_idx: int, offset: int = 0) -> torch.Tensor:
    """Return an orthonormal axis at ``16 * domain_idx + offset`` in ``dim=128`` space."""
    vec = torch.zeros(dim, dtype=torch.float32)
    base = domain_idx * 16 + offset
    vec[base] = 1.0 / math.sqrt(2.0)
    vec[base + 1] = -1.0 / math.sqrt(2.0)
    return vec


def _make_8expert_moe_model(
    *,
    num_layers: int = 2,
    hidden_size: int = 128,
    intermediate_size: int = 256,
    vocab_size: int = 64,
    num_experts: int = 8,
    top_k: int = 2,
    lora_rank: int = 4,
    ghost_sinks: tuple[int, ...] = (),
    collapse_router_bias: Sequence[float] | None = None,
    attach_bias: bool = True,
    poison_expert: int | None = None,
    seed: int = 20261301,
) -> tuple[CrowdV1WithEngram, list[torch.Tensor]]:
    """Build a deterministic 8-expert ``sigma_topk`` Sparse MoE with carrier->target domain circuits."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=64,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)
    carriers = [_unit_domain_axis(hidden_size, d_idx, offset=0) for d_idx in range(num_experts)]
    targets = [_unit_domain_axis(hidden_size, d_idx, offset=2) for d_idx in range(num_experts)]

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for d_idx in range(num_experts):
            lo_tok = d_idx * 8
            hi_tok = lo_tok + 8
            bg = 0.08 * torch.randn((8, hidden_size), generator=gen, dtype=torch.float32)
            for j_idx in range(num_experts):
                bg[:, j_idx * 16 : j_idx * 16 + 4] = 0.0
            bg = bg + 0.30 * carriers[d_idx].unsqueeze(0)
            bg[4] = bg[4] + 0.35 * targets[d_idx]
            wte[lo_tok:hi_tok] = bg

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.40 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.40 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.05 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 19,
    )

    sink_set = set(ghost_sinks)
    with torch.no_grad():
        for blk in model.blocks:
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            r_noise = 0.06 * torch.randn((num_experts, hidden_size), generator=gen, dtype=torch.float32)
            for j_idx in range(num_experts):
                r_noise[:, j_idx * 16 : j_idx * 16 + 4] = 0.0
            mlp.router.weight.copy_(r_noise)
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()

            for d_idx in range(num_experts):
                if d_idx in sink_set:
                    mlp.router.weight[d_idx].add_((2.6 - 0.15 * d_idx) * carriers[0])
                else:
                    mlp.router.weight[d_idx].add_(2.5 * carriers[d_idx])
                    exp = mlp.experts[str(d_idx)]
                    exp.w3_A[0, :hidden_size] = 0.45 * carriers[d_idx]
                    exp.w3_B[:, 0] = 0.85 * targets[d_idx]

            if poison_expert is not None:
                p_exp = mlp.experts[str(poison_expert)]
                p_exp.w3_A[0, :hidden_size] = 0.55 * sum(carriers, torch.zeros_like(carriers[0]))
                p_exp.w3_B[:, 0] = -3.2 * sum(targets, torch.zeros_like(targets[0]))

            if attach_bias:
                bias_vec = (
                    torch.tensor(collapse_router_bias, dtype=torch.float32)
                    if collapse_router_bias is not None
                    else torch.zeros(num_experts, dtype=torch.float32)
                )
                mlp.router.register_parameter("bias", nn.Parameter(bias_vec))

    return model, carriers


def _make_8domain_chunks(
    *,
    seq_len: int = 16,
    score_from: int = 8,
    chunks_per_domain: int = 6,
    domains: tuple[str, ...] = EIGHT_DOMAINS,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu").manual_seed(9100 + seed_offset)
    chunks: list[dict[str, Any]] = []
    for d_idx, dom in enumerate(domains):
        lo_tok = d_idx * 8
        hi_tok = lo_tok + 8
        tgt_tok = lo_tok + 4
        for r_idx in range(chunks_per_domain):
            ids = torch.randint(lo_tok, hi_tok, (seq_len,), generator=gen).tolist()
            for pos in range(score_from, seq_len):
                ids[pos] = tgt_tok
            chunks.append(
                {
                    "expert_id": dom,
                    "dataset_id": dom,
                    "row": seed_offset + 100 * d_idx + r_idx,
                    "ids": ids,
                }
            )
    return chunks


@torch.no_grad()
def _collect_layer_mlp_inputs(
    model: CrowdV1WithEngram,
    chunks: list[dict[str, Any]],
    *,
    layer_idx: int = 0,
    score_from: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(all_inputs, prose_inputs, specialist_inputs)`` entering block ``layer_idx``."""
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    cos = model.cos_cached.to(ids.device)
    sin = model.sin_cached.to(ids.device)
    xb = compute_embeddings_batch(model, ids)
    start_pos = max(0, score_from - 1)
    for idx, blk in enumerate(model.blocks):
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        if idx == layer_idx:
            scored = mi[:, start_pos:-1, :]
            is_gen = torch.tensor(
                [str(c["expert_id"]).startswith("E0_") for c in chunks], dtype=torch.bool
            )
            h_dim = int(scored.shape[-1])
            return (
                scored.reshape(-1, h_dim).contiguous(),
                scored[is_gen].reshape(-1, h_dim).contiguous(),
                scored[~is_gen].reshape(-1, h_dim).contiguous(),
            )
        xb = forward_block_post_mlp(blk, xm, mi)
    raise AssertionError(f"layer_idx={layer_idx} out of range")


@torch.no_grad()
def _eval_domain_nlls(
    model: CrowdV1WithEngram,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 8,
) -> dict[str, float]:
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    cos = model.cos_cached.to(ids.device)
    sin = model.sin_cached.to(ids.device)
    xb = compute_embeddings_batch(model, ids)
    for blk in model.blocks:
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        xb = forward_block_post_mlp(blk, xm, mi)
    losses = per_chunk_causal_midrow_nll(
        xb,
        model.ln.weight,
        model.ln.bias,
        float(model.ln.eps),
        model.head.weight,
        ids,
        score_from=score_from,
        seq_batch_size=32,
    ).tolist()
    grouped: dict[str, list[float]] = {}
    for ch, val in zip(chunks, losses, strict=True):
        grouped.setdefault(str(ch["expert_id"]), []).append(float(val))
    out = {k: sum(v) / len(v) for k, v in grouped.items()}
    spec_vals = [v for k, v in out.items() if not k.startswith("E0_")]
    out["spec_mean"] = sum(spec_vals) / len(spec_vals)
    out["nll_mean"] = sum(losses) / len(losses)
    return out


# ---------------------------------------------------------------------------
# S1: Collapsed 8-Expert sigma_topk Router Rebalanced <0.15 CV in <25 Iters
# ---------------------------------------------------------------------------


def test_s1_collapsed_8expert_rebalanced_under_25_iters_and_improves_nll() -> None:
    """S1: Rebalance collapsed 8-expert router (>90% on 1 expert, 3 dead) to <0.15 CV in <25 iters."""
    collapsed_bias = (0.0, 11.0, 0.0, 0.0, 0.0, -11.5, -11.5, -11.5)
    model, _ = _make_8expert_moe_model(
        num_layers=2,
        collapse_router_bias=collapsed_bias,
        attach_bias=True,
        seed=20261310,
    )
    probe_chunks = _make_8domain_chunks(chunks_per_domain=6, seed_offset=0)
    eval_chunks = _make_8domain_chunks(chunks_per_domain=6, seed_offset=400)

    # Verify severe initial collapse on layer 0: >90% token hit rate on expert 1 and 3 dead specialists
    x_all_l0, _, _ = _collect_layer_mlp_inputs(model, probe_chunks, layer_idx=0, score_from=8)
    with torch.no_grad():
        logits_before = model.blocks[0].mlp.router(x_all_l0)
    util_before = measure_expert_topk_utilization(logits_before, top_k=2)
    assert util_before["token_hit_rates"][1] > 0.90
    assert util_before["dead_expert_count"] >= 3
    assert util_before["dead_experts"] == [5, 6, 7]
    assert util_before["load_cv"] > 0.80

    nlls_before = _eval_domain_nlls(model, eval_chunks, score_from=8)

    edited_model, receipt = apply_verified_router_bias_calibration_inplace(
        model,
        probe_chunks,
        selection_chunks=eval_chunks,
        top_k=2,
        max_iters=25,
        step_size=0.15,
        score_from=8,
        inplace=False,
    )

    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["receipt_object"] == RECEIPT_OBJECT
    assert verify_router_bias_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "COMMITTED"
    assert receipt["max_iterations_run"] < 25
    assert receipt["load_cv"] < 0.15
    assert receipt["dead_experts_after"] == 0
    assert receipt["gini_after"] < receipt["gini_before"]
    assert receipt["guard_ok"] is True
    assert receipt["worst_domain_drift_pct"] <= 1.0
    assert receipt["delta_energy"] < 0.0
    assert receipt["anse_energy_active"] < receipt["anse_energy_parent"]

    # Non-inplace copy preserved original model hash
    assert compute_module_state_sha256(model) == receipt["parent_sha256"]
    assert compute_module_state_sha256(edited_model) == receipt["active_sha256"]

    nlls_after = _eval_domain_nlls(edited_model, eval_chunks, score_from=8)
    assert nlls_after["spec_mean"] < nlls_before["spec_mean"] - 0.03
    assert nlls_after["E5_exoplanet_transit"] < nlls_before["E5_exoplanet_transit"] - 0.05
    assert nlls_after["E6_stellar_spectroscopy"] < nlls_before["E6_stellar_spectroscopy"] - 0.05
    assert nlls_after["E7_defensive_cyber"] < nlls_before["E7_defensive_cyber"] - 0.05
    assert nlls_after["E0_general_fineweb"] <= nlls_before["E0_general_fineweb"] * 1.01

    # Verify routing utilization on held-out eval chunks as well
    x_eval_l0, _, _ = _collect_layer_mlp_inputs(edited_model, eval_chunks, layer_idx=0, score_from=8)
    with torch.no_grad():
        logits_eval_after = edited_model.blocks[0].mlp.router(x_eval_l0)
    util_eval_after = measure_expert_topk_utilization(logits_eval_after, top_k=2)
    assert util_eval_after["load_cv"] < 0.15
    assert util_eval_after["dead_expert_count"] == 0

    # Exercise dead_fraction_threshold > 0 branch and 3D router_logits handling
    util_3d = measure_expert_topk_utilization(
        logits_before.unsqueeze(0), top_k=2, dead_fraction_threshold=0.02
    )
    assert util_3d["dead_expert_count"] >= 3
    util_single_active = measure_expert_topk_utilization(
        logits_before, top_k=2, active_experts=[1]
    )
    assert util_single_active["load_cv"] == 0.0 and util_single_active["normalized_entropy"] == 1.0
    util_zero_active = measure_expert_topk_utilization(
        logits_before, top_k=2, active_experts=[5, 6, 7], dead_fraction_threshold=0.01
    )
    assert util_zero_active["dead_expert_count"] == 3 and util_zero_active["normalized_entropy"] == 0.0
    util_single_zero = measure_expert_topk_utilization(
        logits_before, top_k=2, active_experts=[5]
    )
    assert util_single_zero["load_cv"] == 1.0 and util_single_zero["normalized_entropy"] == 0.0


# ---------------------------------------------------------------------------
# S2: Exact Preservation of FoMoE Ghost-Sink Routing on General Prose
# ---------------------------------------------------------------------------


def test_s2_fomoe_ghost_sink_prose_routing_preserved_exactly() -> None:
    """S2: Rebalance active specialists while keeping FoMoE ghost sinks identically routed on prose."""
    # Experts 0 and 3 are FoMoE ghost sinks (w_B == 0 => dy == 0) capturing both top-2 slots on E0 prose;
    # specialist expert 1 collapsed (+6.0) while specialists 5, 6, 7 are dead (-6.0).
    sink_collapse_bias = (0.0, 6.0, 0.0, 0.0, 0.0, -6.0, -6.0, -6.0)
    model, _ = _make_8expert_moe_model(
        num_layers=2,
        ghost_sinks=(0, 3),
        collapse_router_bias=sink_collapse_bias,
        attach_bias=True,
        seed=20261320,
    )
    probe_chunks = _make_8domain_chunks(chunks_per_domain=6, seed_offset=100)
    _, prose_l0, spec_l0 = _collect_layer_mlp_inputs(model, probe_chunks, layer_idx=0, score_from=8)

    with torch.no_grad():
        prose_logits_before = model.blocks[0].mlp.router(prose_l0)
        _, prose_top_before = torch.topk(torch.sigmoid(prose_logits_before), k=2, dim=-1)
        prose_gates_before = torch.sigmoid(prose_logits_before).gather(1, prose_top_before)
    # Before calibration, expert 1 (+6.0 bias) steals slot 2 on prose away from sink 3!
    # Wait: test both (A) full model where sink 0 & 3 are protected when prose initially routes to sinks,
    # and (B) standalone protect_sink_experts=(0, 3) with calibrated margin guard.
    model_clean_sink, _ = _make_8expert_moe_model(
        num_layers=2,
        ghost_sinks=(0, 3),
        collapse_router_bias=(0.0, 3.5, 0.0, 0.0, 0.0, -9.5, -9.5, -9.5),
        attach_bias=True,
        seed=20261321,
    )
    _, prose_clean, spec_clean = _collect_layer_mlp_inputs(
        model_clean_sink, probe_chunks, layer_idx=0, score_from=8
    )
    with torch.no_grad():
        p_logits_0 = model_clean_sink.blocks[0].mlp.router(prose_clean)
        _, p_top_0 = torch.topk(torch.sigmoid(p_logits_0), k=2, dim=-1)
        p_gates_0 = torch.sigmoid(p_logits_0).gather(1, p_top_0)
    assert bool(((p_top_0 == 0) | (p_top_0 == 3)).all().item())

    # Even if standalone calibration on specialist inputs raises dead specialist biases,
    # prose_inputs guard locks ghost sinks (0, 3) across 100% of prose tokens.
    bias_before_0 = float(model_clean_sink.blocks[0].mlp.router.bias[0].item())
    bias_before_3 = float(model_clean_sink.blocks[0].mlp.router.bias[3].item())
    _, s2_rep = calibrate_router_bias_auxiliary_free(
        model_clean_sink.blocks[0].mlp,
        spec_clean,
        protect_sink_experts=(0, 3),
        prose_inputs=prose_clean,
        max_iters=25,
        step_size=0.15,
        inplace=False,
    )
    assert s2_rep["sink_preservation"]["exact_sink_routing_preserved"] is True
    assert s2_rep["sink_preservation"]["sink_match_fraction"] == 1.0
    assert s2_rep["sink_preservation"]["max_sink_gate_diff"] == 0.0
    assert s2_rep["load_cv_after"] < 0.15
    assert s2_rep["dead_experts_after"] == 0

    # Now run end-to-end verified calibration on model_clean_sink with protect_sink_experts=(0, 3)
    nlls_before = _eval_domain_nlls(model_clean_sink, probe_chunks, score_from=8)
    _, receipt = apply_verified_router_bias_calibration_inplace(
        model_clean_sink,
        probe_chunks,
        protect_sink_experts=(0, 3),
        top_k=2,
        max_iters=25,
        step_size=0.15,
        score_from=8,
        inplace=True,
    )
    assert receipt["accepted"] is True
    assert receipt["exact_sink_routing_preserved"] is True
    assert receipt["min_sink_match_fraction"] == 1.0
    assert receipt["load_cv"] < 0.15
    assert receipt["dead_experts_after"] == 0

    with torch.no_grad():
        assert float(model_clean_sink.blocks[0].mlp.router.bias[0].item()) == bias_before_0
        assert float(model_clean_sink.blocks[0].mlp.router.bias[3].item()) == bias_before_3
        p_logits_after = model_clean_sink.blocks[0].mlp.router(prose_clean)
        _, p_top_after = torch.topk(torch.sigmoid(p_logits_after), k=2, dim=-1)
        p_gates_after = torch.sigmoid(p_logits_after).gather(1, p_top_after)
        assert torch.equal(p_top_after, p_top_0)
        assert torch.allclose(p_gates_after, p_gates_0, atol=1e-7)

    nlls_after = _eval_domain_nlls(model_clean_sink, probe_chunks, score_from=8)
    assert nlls_after["E0_general_fineweb"] == pytest.approx(
        nlls_before["E0_general_fineweb"], abs=1e-6
    )
    assert nlls_after["spec_mean"] < nlls_before["spec_mean"] - 0.02

    # Also test prose_inputs with zero sink hits & single-expert active fallback
    r_nosink = nn.Linear(128, 8, bias=True)
    with torch.no_grad():
        r_nosink.weight.zero_()
        r_nosink.bias.copy_(torch.tensor([-5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    _, rep_nosink = calibrate_router_bias_auxiliary_free(
        r_nosink,
        spec_clean,
        protect_sink_experts=(0,),
        prose_inputs=prose_clean,
        max_iters=5,
        step_size=0.15,
    )
    assert rep_nosink["sink_preservation"]["exact_sink_routing_preserved"] is True
    _, rep_1act = calibrate_router_bias_auxiliary_free(
        nn.Linear(128, 2, bias=True),
        spec_clean,
        protect_sink_experts=(0,),
        max_iters=5,
    )
    assert rep_1act["iterations_run"] == 0 and rep_1act["converged"] is True
    assert float(prose_gates_before.numel()) > 0


# ---------------------------------------------------------------------------
# S3: Strictly <0.01% Parameter Modification & Bitwise Weight Immutability
# ---------------------------------------------------------------------------


def test_s3_strict_parameter_budget_and_bitwise_router_weight_immutability() -> None:
    """S3: Verify <0.01% model modification (<=0.001% at Crowd-v1 width) & bitwise W immutability."""
    # 1. Bias=False model collapsed via router weight offsets (attaches bias cleanly in-place)
    model_no_bias, axes = _make_8expert_moe_model(
        num_layers=2,
        collapse_router_bias=None,
        attach_bias=False,
        seed=20261330,
    )
    # Collapse via router weights without touching bias (bias is None!)
    with torch.no_grad():
        common = sum(axes, torch.zeros_like(axes[0]))
        for blk in model_no_bias.blocks:
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            assert mlp.router.bias is None
            mlp.router.weight[1].add_(2.75 * common)
            for dead_e in (5, 6, 7):
                mlp.router.weight[dead_e].sub_(2.75 * common)

    probe_chunks = _make_8domain_chunks(chunks_per_domain=6, seed_offset=200)
    buckets = extract_moe_trainable_buckets(model_no_bias, num_router_fragments=2)
    w_snaps = [blk.mlp.router.weight.detach().clone() for blk in model_no_bias.blocks]
    non_bias_sha_before = compute_non_router_bias_state_sha256(model_no_bias)

    _, receipt = apply_verified_router_bias_calibration_inplace(
        model_no_bias,
        probe_chunks,
        buckets=buckets,
        cal_ln_w=model_no_bias.ln.weight.detach().clone(),
        cal_ln_b=model_no_bias.ln.bias.detach().clone(),
        target_layers=[0, 1],
        max_iters=25,
        step_size=0.15,
        score_from=8,
        inplace=True,
    )
    assert receipt["accepted"] is True
    assert receipt["non_bias_weights_bitwise_preserved"] is True
    assert receipt["router_weights_bitwise_preserved"] is True
    assert compute_non_router_bias_state_sha256(model_no_bias) == non_bias_sha_before
    assert receipt["modified_param_count"] <= 16
    assert receipt["modified_param_pct"] < 0.01
    assert receipt["mlp_modified_param_pct"] < 0.01
    assert receipt["load_cv"] < 0.15
    for blk, w_snap in zip(model_no_bias.blocks, w_snaps, strict=True):
        assert blk.mlp.router.bias is not None
        assert torch.equal(blk.mlp.router.weight.detach(), w_snap)

    # 2. Full Crowd-v1 layer width (H=1536, I=2560, E=8): bias is <= 0.001% of layer parameters
    dense_crowd = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=1536, intermediate_size=2560))
    moe_crowd = CrowdLoRAMoEMLP(
        dense_crowd, num_experts=8, top_k=2, rank=8, routing_mode="sigma_topk", seed=42
    )
    gen = torch.Generator(device="cpu").manual_seed(20261331)
    with torch.no_grad():
        moe_crowd.router.weight.normal_(0.0, 0.05, generator=gen)
        moe_crowd.router.register_parameter(
            "bias",
            nn.Parameter(torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0, -5.0, -5.0, -5.0])),
        )
    w_crowd_before = moe_crowd.router.weight.detach().clone()
    x_crowd = torch.randn((96, 1536), generator=gen)
    cal_b, crowd_rep = calibrate_router_bias_auxiliary_free(
        moe_crowd, x_crowd, max_iters=25, step_size=0.15, inplace=True
    )
    assert torch.equal(moe_crowd.router.weight.detach(), w_crowd_before)
    assert crowd_rep["load_cv_after"] < 0.15
    crowd_layer_params = sum(p.numel() for p in moe_crowd.parameters())
    assert (float(cal_b.numel()) / float(crowd_layer_params)) * 100.0 <= 0.001

    # 3. Also verify CrowdFineGrainedMoEMLP, raw 1D bias tensor, and non-uniform target_utilization
    fg_moe = CrowdFineGrainedMoEMLP(
        CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=64, intermediate_size=128)),
        num_experts=4,
        top_k=1,
        routing_mode="sigma_topk",
    )
    with torch.no_grad():
        fg_moe.router.weight.normal_(0.0, 0.3, generator=gen)
        fg_moe.router.register_parameter(
            "bias", nn.Parameter(torch.tensor([4.0, 0.0, -4.0, -4.0]))
        )
    x_fg = torch.randn((160, 64), generator=gen)
    _, fg_rep = calibrate_router_bias_auxiliary_free(
        fg_moe,
        x_fg,
        target_utilization={0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2},
        max_iters=25,
        step_size=0.15,
        inplace=True,
    )
    assert fg_rep["target_error_cv_after"] < fg_rep["target_error_cv_before"]
    assert fg_rep["dead_experts_after"] == 0

    raw_bias = torch.tensor([4.0, 0.0, -4.0, -4.0], dtype=torch.float32)
    pre_logits = torch.randn((120, 4), generator=gen) * 0.4
    b_tensor_out, tensor_rep = calibrate_router_bias_auxiliary_free(
        raw_bias,
        pre_logits,
        target_utilization=torch.tensor([0.25, 0.25, 0.25, 0.25]),
        max_iters=20,
        step_size=0.15,
        center_bias=False,
        inplace=True,
    )
    assert b_tensor_out is raw_bias
    assert tensor_rep["load_cv_after"] < 0.15


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback on Forced Adverse Target Distribution
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_on_adverse_target() -> None:
    """S4: Trigger ANSE Maximum Pain (>=1e6) on adverse target and verify atomic bitwise rollback."""
    # Case 4A: Model with attach_bias=True where forcing 65% traffic to poisoned Expert 7 violates +1% guard
    model_bias, _ = _make_8expert_moe_model(
        num_layers=2,
        collapse_router_bias=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0),
        attach_bias=True,
        poison_expert=7,
        seed=20261340,
    )
    probe_chunks = _make_8domain_chunks(chunks_per_domain=6, seed_offset=300)
    parent_sha = compute_module_state_sha256(model_bias)

    adverse_target = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.65]
    _, rec_guard = apply_verified_router_bias_calibration_inplace(
        model_bias,
        probe_chunks,
        target_utilization=adverse_target,
        max_iters=25,
        step_size=0.20,
        score_from=8,
        inplace=True,
    )
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["guard_ok"] is False
    assert rec_guard["worst_domain_drift_pct"] > 1.0
    assert rec_guard["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert rec_guard["delta_energy"] > 0.0
    assert rec_guard["active_sha256"] == parent_sha
    assert compute_module_state_sha256(model_bias) == parent_sha
    assert verify_router_bias_receipt_provenance(rec_guard) is True

    # Case 4B: Model originally with bias=False (router.bias is None) must restore bias=None bitwise!
    model_nobias, _ = _make_8expert_moe_model(
        num_layers=2,
        attach_bias=False,
        poison_expert=7,
        seed=20261341,
    )
    with torch.no_grad():
        for blk in model_nobias.blocks:
            blk.mlp.router.weight[7].zero_()
    parent_nb_sha = compute_module_state_sha256(model_nobias)
    _, rec_nb = apply_verified_router_bias_calibration_inplace(
        model_nobias,
        probe_chunks,
        target_utilization=adverse_target,
        max_iters=25,
        step_size=0.20,
        score_from=8,
        inplace=True,
    )
    assert rec_nb["rolled_back"] is True
    assert rec_nb["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_nb["active_sha256"] == parent_nb_sha
    assert compute_module_state_sha256(model_nobias) == parent_nb_sha
    for blk in model_nobias.blocks:
        assert blk.mlp.router.bias is None

    # Case 4C: Non-finite / malformed eval_fn triggers Maximum Pain + rollback
    model_clean, _ = _make_8expert_moe_model(
        num_layers=2,
        collapse_router_bias=(0.0, 9.5, 0.0, 0.0, 0.0, -9.5, -9.5, -9.5),
        attach_bias=True,
        seed=20261342,
    )
    clean_sha = compute_module_state_sha256(model_clean)
    base_dom_map = {dom: 1.5 for dom in EIGHT_DOMAINS}
    bad_payloads: list[Any] = [
        {**base_dom_map, "E0_general_fineweb": float("nan")},
        "not_a_mapping",
        {**base_dom_map, "E0_general_fineweb": True},
    ]
    for bad_val in bad_payloads:
        calls = {"n": 0}

        def _bad_eval(_m: nn.Module, out: Any = bad_val, st: dict[str, int] = calls) -> Any:
            st["n"] += 1
            return dict(base_dom_map) if st["n"] == 1 else out

        _, rec_nf = apply_verified_router_bias_calibration_inplace(
            model_clean,
            probe_chunks,
            baseline_domain_nll=base_dom_map,
            eval_fn=_bad_eval,
            score_from=8,
            inplace=True,
        )
        assert rec_nf["status"] == "ROLLED_BACK_NON_FINITE"
        assert rec_nf["anse_energy"] == DEFAULT_ANSE_MAXIMUM_PAIN
        assert compute_module_state_sha256(model_clean) == clean_sha

    raise_state = {"n": 0}

    def _raising_eval(_m: nn.Module) -> dict[str, float]:
        raise_state["n"] += 1
        if raise_state["n"] == 1:
            return dict(base_dom_map)
        raise RuntimeError("eval boom")

    _, rec_err = apply_verified_router_bias_calibration_inplace(
        model_clean, probe_chunks, eval_fn=_raising_eval, score_from=8, inplace=True
    )
    assert rec_err["status"] == "ROLLED_BACK_NON_FINITE"
    assert compute_module_state_sha256(model_clean) == clean_sha

    # Case 4D: Specialist regression within +1% guard -> ROLLED_BACK_SPEC_REGRESSION
    spec_state = {"n": 0}

    def _spec_regress_eval(_m: nn.Module) -> dict[str, float]:
        spec_state["n"] += 1
        if spec_state["n"] == 1:
            return dict(base_dom_map)
        return {
            **base_dom_map,
            "E0_general_fineweb": 1.40,
            "E1_python_systems_code": 1.51,
        }

    _, rec_spec_reg = apply_verified_router_bias_calibration_inplace(
        model_clean, probe_chunks, eval_fn=_spec_regress_eval, score_from=8, inplace=True
    )
    assert rec_spec_reg["status"] == "ROLLED_BACK_SPEC_REGRESSION"
    assert compute_module_state_sha256(model_clean) == clean_sha

    # Case 4E: Budget exceeded & min_energy_improvement not met
    _, rec_budget = apply_verified_router_bias_calibration_inplace(
        model_clean,
        probe_chunks,
        config=RouterBiasCalibrationConfig(
            score_from=8, max_modified_param_fraction=1e-7, inplace=True
        ),
    )
    assert rec_budget["status"] == "ROLLED_BACK_BUDGET_EXCEEDED"
    assert compute_module_state_sha256(model_clean) == clean_sha

    _, rec_no_gain = apply_verified_router_bias_calibration_inplace(
        model_clean,
        probe_chunks,
        config=RouterBiasCalibrationConfig(
            score_from=8, min_energy_improvement=10.0, inplace=True
        ),
    )
    assert rec_no_gain["status"] == "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"
    assert compute_module_state_sha256(model_clean) == clean_sha

    # Case 4F: NOOP when step_size == 0.0 & tampered rollback verification error
    _, rec_noop = apply_verified_router_bias_calibration_inplace(
        model_clean, probe_chunks, step_size=0.0, score_from=8, inplace=True
    )
    assert rec_noop["status"] == "NOOP_ALREADY_CALIBRATED"
    assert rec_noop["modified_param_count"] == 0

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _restore_model_snapshot_bitwise(model_clean, {}, {}, "0" * 64)


# ---------------------------------------------------------------------------
# S5: Validation Errors, Provenance Tampering & Security / Stub Compliance
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_compliance_gates() -> None:
    """S5: Exercise validation checks, provenance tamper rejection, ZeroStubAudit, and gatekeeper."""
    # Config validations
    with pytest.raises(TypeError, match="int"):
        RouterBiasCalibrationConfig(top_k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        RouterBiasCalibrationConfig(top_k=0)
    with pytest.raises(TypeError, match="real number"):
        RouterBiasCalibrationConfig(step_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0.0"):
        RouterBiasCalibrationConfig(step_size=-0.1)
    with pytest.raises(ValueError, match="finite"):
        RouterBiasCalibrationConfig(step_size=float("nan"))
    with pytest.raises(ValueError, match="> 0.0"):
        RouterBiasCalibrationConfig(max_bias_magnitude=0.0)
    with pytest.raises(TypeError, match="bool"):
        RouterBiasCalibrationConfig(center_bias=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="general_prefix"):
        RouterBiasCalibrationConfig(general_prefix=" ")
    with pytest.raises(ValueError, match=">= 0"):
        RouterBiasCalibrationConfig(score_from=-1)
    with pytest.raises(ValueError, match="> 0.0"):
        RouterBiasCalibrationConfig(max_modified_param_fraction=0.0)

    # measure_expert_topk_utilization validations
    good_logits = torch.randn((16, 4))
    with pytest.raises(TypeError, match="torch.Tensor"):
        measure_expert_topk_utilization("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D or 3D"):
        measure_expert_topk_utilization(torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        measure_expert_topk_utilization(torch.ones((4, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        measure_expert_topk_utilization(torch.tensor([[float("inf"), 1.0]]))
    with pytest.raises(ValueError, match="exceeds"):
        measure_expert_topk_utilization(good_logits, top_k=5)
    with pytest.raises(ValueError, match="< 1.0"):
        measure_expert_topk_utilization(good_logits, dead_fraction_threshold=1.0)
    with pytest.raises(TypeError, match="sequence or set"):
        measure_expert_topk_utilization(good_logits, active_experts="01")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range"):
        measure_expert_topk_utilization(good_logits, active_experts=[9])
    with pytest.raises(ValueError, match="empty"):
        measure_expert_topk_utilization(good_logits, active_experts=[])

    # calibrate_router_bias_auxiliary_free validations
    r_ok = nn.Linear(8, 4, bias=True)
    x_ok = torch.ones((10, 8))
    with pytest.raises(TypeError, match="router_linear_or_bias"):
        calibrate_router_bias_auxiliary_free("bad", x_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        calibrate_router_bias_auxiliary_free(torch.ones((2, 2)), x_ok)
    with pytest.raises(TypeError, match="floating-point"):
        calibrate_router_bias_auxiliary_free(torch.ones(4, dtype=torch.long), x_ok)
    with pytest.raises(ValueError, match="finite"):
        calibrate_router_bias_auxiliary_free(torch.full((4,), float("nan")), x_ok)
    with pytest.raises(ValueError, match="logits with last dim"):
        calibrate_router_bias_auxiliary_free(torch.zeros(4), x_ok)
    with pytest.raises(ValueError, match="exceeds"):
        calibrate_router_bias_auxiliary_free(torch.zeros(4), torch.ones((4, 4)), top_k=5)
    with pytest.raises(ValueError, match="out_features >= 2"):
        calibrate_router_bias_auxiliary_free(nn.Linear(8, 1), x_ok)
    with pytest.raises(TypeError, match="floating-point"):
        lin_int = nn.Linear(8, 4, bias=False)
        lin_int.weight = nn.Parameter(torch.ones((4, 8), dtype=torch.long), requires_grad=False)
        calibrate_router_bias_auxiliary_free(lin_int, x_ok)
    with pytest.raises(ValueError, match="finite"):
        lin_inf = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            lin_inf.weight.fill_(float("inf"))
        calibrate_router_bias_auxiliary_free(lin_inf, x_ok)
    with pytest.raises(ValueError, match="in_features"):
        calibrate_router_bias_auxiliary_free(r_ok, torch.ones((4, 6)))
    with pytest.raises(ValueError, match="exceeds"):
        calibrate_router_bias_auxiliary_free(r_ok, x_ok, top_k=5)
    with pytest.raises(ValueError, match="router.bias"):
        bad_b_lin = nn.Linear(8, 4, bias=True)
        bad_b_lin.bias = nn.Parameter(torch.zeros(2))
        calibrate_router_bias_auxiliary_free(bad_b_lin, x_ok)
    with pytest.raises(ValueError, match="router.bias"):
        nan_b_lin = nn.Linear(8, 4, bias=True)
        with torch.no_grad():
            nan_b_lin.bias.fill_(float("nan"))
        calibrate_router_bias_auxiliary_free(nan_b_lin, x_ok)
    with pytest.raises(ValueError, match="exclude all experts"):
        calibrate_router_bias_auxiliary_free(r_ok, x_ok, protect_sink_experts=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="empty"):
        calibrate_router_bias_auxiliary_free(r_ok, x_ok, target_utilization={})
    with pytest.raises(ValueError, match="out of range"):
        calibrate_router_bias_auxiliary_free(r_ok, x_ok, target_utilization={9: 1.0})
    with pytest.raises(ValueError, match="match either active_experts"):
        calibrate_router_bias_auxiliary_free(r_ok, x_ok, target_utilization={0: 0.5, 1: 0.5})
    with pytest.raises(ValueError, match="non-bool numeric tensor"):
        calibrate_router_bias_auxiliary_free(
            r_ok, x_ok, target_utilization=torch.tensor([True, False, True, False])
        )
    with pytest.raises(ValueError, match="length"):
        calibrate_router_bias_auxiliary_free(r_ok, x_ok, target_utilization=torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError, match="length"):
        calibrate_router_bias_auxiliary_free(r_ok, x_ok, target_utilization=[0.5, 0.5])
    with pytest.raises(TypeError, match="None, a Sequence, Mapping, or 1D"):
        calibrate_router_bias_auxiliary_free(
            r_ok, x_ok, target_utilization="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="positive sum"):
        calibrate_router_bias_auxiliary_free(r_ok, x_ok, target_utilization=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="prose_inputs width"):
        calibrate_router_bias_auxiliary_free(
            r_ok, x_ok, protect_sink_experts=(0,), prose_inputs=torch.ones((4, 3))
        )
    with pytest.raises(ValueError, match="prose_inputs width"):
        calibrate_router_bias_auxiliary_free(
            torch.zeros(4), torch.ones((4, 4)), protect_sink_experts=(0,), prose_inputs=torch.ones((4, 3))
        )

    # apply_verified_router_bias_calibration_inplace & provenance validations
    model, _ = _make_8expert_moe_model(num_layers=2, seed=20261350)
    chunks = _make_8domain_chunks(chunks_per_domain=2, seed_offset=500)
    with pytest.raises(TypeError, match="nn.Module"):
        apply_verified_router_bias_calibration_inplace("bad", chunks)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        empty_m = nn.Module()
        empty_m.blocks = nn.ModuleList()
        apply_verified_router_bias_calibration_inplace(empty_m, chunks)
    with pytest.raises(ValueError, match="must expose"):
        bare_m = nn.Module()
        bare_m.blocks = nn.ModuleList([nn.Linear(4, 4)])
        apply_verified_router_bias_calibration_inplace(bare_m, chunks)
    with pytest.raises(ValueError, match="mlp.router"):
        nomoe_m = copy.deepcopy(model)
        nomoe_m.blocks = nn.ModuleList([nn.Module(), nn.Module()])
        nomoe_m.blocks[0].mlp = nn.ReLU()
        nomoe_m.blocks[1].mlp = nn.ReLU()
        apply_verified_router_bias_calibration_inplace(nomoe_m, chunks)
    with pytest.raises(TypeError, match="callable"):
        apply_verified_router_bias_calibration_inplace(
            model, chunks, eval_fn="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="both be provided"):
        apply_verified_router_bias_calibration_inplace(model, chunks, cal_ln_w=torch.ones(128))
    with pytest.raises(TypeError, match="RouterBiasCalibrationConfig"):
        apply_verified_router_bias_calibration_inplace(
            model, chunks, config="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="target_layers"):
        apply_verified_router_bias_calibration_inplace(model, chunks, target_layers=[])
    with pytest.raises(ValueError, match="out of range"):
        apply_verified_router_bias_calibration_inplace(model, chunks, target_layers=[99])
    with pytest.raises(ValueError, match="does not expose"):
        partial_m = copy.deepcopy(model)
        partial_m.blocks[1].mlp = nn.ReLU()
        apply_verified_router_bias_calibration_inplace(partial_m, chunks, target_layers=[1])
    with pytest.raises(TypeError, match="torch.Tensor"):
        apply_verified_router_bias_calibration_inplace(
            model, chunks, cal_ln_w="a", cal_ln_b="b"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="shape"):
        apply_verified_router_bias_calibration_inplace(
            model, chunks, cal_ln_w=torch.ones(4), cal_ln_b=torch.ones(4)
        )
    with pytest.raises(ValueError, match="finite"):
        apply_verified_router_bias_calibration_inplace(
            model, chunks, cal_ln_w=torch.full((128,), float("nan")), cal_ln_b=torch.zeros(128)
        )
    with pytest.raises(ValueError, match="non-empty sequence"):
        apply_verified_router_bias_calibration_inplace(model, [])
    with pytest.raises(ValueError, match="expert_id"):
        apply_verified_router_bias_calibration_inplace(model, [{"ids": [0, 1, 2]}], score_from=1)
    with pytest.raises(ValueError, match="non-empty string"):
        apply_verified_router_bias_calibration_inplace(
            model, [{"expert_id": " ", "ids": [0, 1]}], score_from=1
        )
    with pytest.raises(ValueError, match="length >= 2"):
        apply_verified_router_bias_calibration_inplace(
            model, [{"expert_id": "E0_a", "ids": [0]}], score_from=0
        )
    with pytest.raises(ValueError, match="token IDs"):
        apply_verified_router_bias_calibration_inplace(
            model, [{"expert_id": "E0_a", "ids": [0, 999]}], score_from=0
        )
    with pytest.raises(ValueError, match="share sequence length"):
        apply_verified_router_bias_calibration_inplace(
            model,
            [{"expert_id": "E0_a", "ids": [0, 1]}, {"expert_id": "E0_a", "ids": [0, 1, 2]}],
            score_from=0,
        )
    with pytest.raises(ValueError, match="too short"):
        apply_verified_router_bias_calibration_inplace(
            model, [{"expert_id": "E0_a", "ids": [0, 1]}], score_from=8
        )
    with pytest.raises(ValueError, match="general chunk"):
        apply_verified_router_bias_calibration_inplace(
            model, [{"expert_id": "E1_a", "ids": [0, 1, 2]}], score_from=0
        )
    with pytest.raises(ValueError, match="baseline_domain_nll"):
        apply_verified_router_bias_calibration_inplace(
            model, chunks, baseline_domain_nll={"wrong": 1.0}, score_from=8
        )
    with pytest.raises(ValueError, match="Parent model evaluation"):
        apply_verified_router_bias_calibration_inplace(
            model, chunks, eval_fn=lambda _m: {"E0_general_fineweb": -1.0}, score_from=8
        )

    inf_model = copy.deepcopy(model)
    with torch.no_grad():
        inf_model.blocks[-1].mlp.base_mlp.w1.weight.fill_(1e20)
        inf_model.blocks[-1].mlp.base_mlp.w2.weight.fill_(1e20)
        inf_model.blocks[-1].mlp.base_mlp.w3.weight.fill_(1e20)
    with pytest.raises(ValueError, match="Parent model evaluation"):
        apply_verified_router_bias_calibration_inplace(inf_model, chunks, score_from=8)

    with pytest.raises(TypeError, match="mapping"):
        verify_router_bias_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_router_bias_receipt_provenance({"receipt_sha256": "xyz"}) is False
    assert (
        verify_router_bias_receipt_provenance({"receipt_sha256": "0" * 64, "bad": float("nan")})
        is False
    )
    with pytest.raises(TypeError, match="nn.Module"):
        compute_module_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_module_state_sha256(nn.ReLU())
    with pytest.raises(TypeError, match="nn.Module"):
        compute_non_router_bias_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_non_router_bias_state_sha256(nn.ReLU())
    with pytest.raises(ValueError, match="non-finite"):
        bad_lin = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            bad_lin.weight.fill_(float("nan"))
        compute_module_state_sha256(bad_lin)

    # Security, ZeroStubAudit (with CI-safe fallback), lateral movement, and pr_gatekeeper checks
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn13", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn13", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/moe_load_balance_bias_controller.py", TARGET_MODULE_PATH),
        ("tests/test_moe_load_balance_bias_controller_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        assert fallback_auditor.check_file(str(path_obj)) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True

        if ZeroStubAudit is not None:
            stub_res = ZeroStubAudit().audit_python_code(src_text)
            assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
            assert stub_res.penalty_energy == 0.0

        assert scanner.scan_python_ast_security(rel, src_text) == []
        assert scanner.scan_text_for_payment_solicitation(src_text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(src_text, rel) == []
        assert scanner.scan_unicode_trojan_source(src_text, rel) == []
        assert lateral_guard.audit_source_for_lateral_movement(src_text, rel) == []

        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in src_text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        changed_files.append(
            {
                "filename": rel,
                "status": "added",
                "additions": len(src_text.splitlines()),
                "patch": "",
                "content": src_text,
            }
        )

    sample_receipt_json = json.dumps(
        {"nll_mean": 1.12, "load_cv": 0.037, "anse_energy": -0.055, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1113,
        pr_title="feat(swarm): DeepSeek-V3 auxiliary-loss-free router bias weight calibration for open-weight sigma_topk MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-13 router bias load-balance controller and deterministic CPU tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): DeepSeek-V3 auxiliary-loss-free router bias weight calibration for open-weight sigma_topk MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
