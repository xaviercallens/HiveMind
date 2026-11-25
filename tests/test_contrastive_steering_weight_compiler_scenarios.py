# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-31: contrastive steering vector to null-space weight compiler."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.contrastive_steering_weight_compiler import (
    ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    ContrastiveWeightCompilerConfig,
    build_prose_nullspace_projector,
    compile_steering_vector_into_nullspace_weight,
    compute_module_state_sha256,
    count_registered_runtime_hooks,
    execute_verified_contrastive_weight_compilation,
    extract_oof_contrastive_steering_subspace,
    verify_contrastive_compilation_receipt_provenance,
)
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.gcp_t4_crowd_diloco import CrowdLoRALinear
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "contrastive_steering_weight_compiler.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit_axis(dim: int, idx0: int, idx1: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    vec[idx0] = 1.0 / math.sqrt(2.0)
    vec[idx1] = -1.0 / math.sqrt(2.0)
    return vec


def _make_zero_mean_block(rows: int, cols: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn((rows, cols), generator=gen, dtype=torch.float32)
    raw = raw - raw.mean(dim=1, keepdim=True)
    return raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _build_contrastive_moe_model(
    *,
    split_mode: str = "lora_domain_experts",
    seed: int = 20261231,
) -> tuple[CrowdV1WithEngram, dict[str, Any]]:
    """Build a 3-layer CrowdV1WithEngram MoE with planted specialist thinking-loop contamination.

    Subspace layout (``H = 48, I = 96, vocab = 64``):
      - ``0..15``:  ``E0_general_fineweb`` prose tokens ``8..23``
      - ``16..31``: ``E1_python_systems_code`` prompt cues ``24..27`` + diverse concise targets ``28..35``
      - ``32..39``: Thinking-loop sink tokens ``0..3`` (carrier ``u_loop``)
    """
    hidden_size = 48
    intermediate_size = 96
    vocab_size = 64
    num_layers = 3
    num_experts = 4

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

    u_e0 = _unit_axis(hidden_size, 0, 1)
    u_e1 = _unit_axis(hidden_size, 16, 17)
    u_loop = _unit_axis(hidden_size, 32, 33)

    gen_tokens = list(range(8, 24))
    prompt_tokens = list(range(24, 28))
    concise_tokens = list(range(28, 36))
    loop_tokens = [0, 1, 2, 3]

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        gen_sub = _make_zero_mean_block(len(gen_tokens), 16, seed + 1) * 0.9 + 1.2 * u_e0[:16].unsqueeze(0)
        prompt_sub = _make_zero_mean_block(len(prompt_tokens), 16, seed + 2) * 0.4 + 0.9 * u_e1[16:32].unsqueeze(0)
        concise_sub = _make_zero_mean_block(len(concise_tokens), 16, seed + 4) * 1.3 + 1.1 * u_e1[16:32].unsqueeze(0)
        loop_sub = _make_zero_mean_block(len(loop_tokens), 8, seed + 3) * 0.25 + 2.4 * u_loop[32:40].unsqueeze(0)

        wte[8:24, :16] = gen_sub
        wte[24:28, 16:32] = prompt_sub
        wte[24:28, 32:40] = 0.45 * u_loop[32:40].unsqueeze(0)
        wte[28:36, 16:32] = concise_sub
        wte[0:4, 32:40] = loop_sub

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.85 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.85 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:16, :16].copy_(0.60 * torch.eye(16))
            blk.mlp.w3.weight[16:32, 16:32].copy_(0.45 * torch.eye(16))
            blk.mlp.w3.weight[32:40, 32:40].copy_(0.80 * torch.eye(8))

        if split_mode == "dense":
            model.blocks[-1].mlp.w3.weight[16:18, 16:18].zero_()

    if split_mode in {"lora_domain_experts", "fine_grained_width_split"}:
        upcycle_crowd_v1_to_sparse_moe(
            model,
            split_mode=split_mode,
            num_experts=num_experts,
            top_k=2,
            lora_rank=4,
            lora_alpha=8.0,
            routing_mode="sigma_topk",
            seed=seed + 17,
        )

    with torch.no_grad():
        if split_mode == "lora_domain_experts":
            for l_idx, blk in enumerate(model.blocks):
                moe = blk.mlp
                assert isinstance(moe, CrowdLoRAMoEMLP)
                moe.router.weight.zero_()
                for exp in moe.experts.values():
                    for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                        getattr(exp, attr).zero_()
                moe.router.weight[0] = 4.0 * u_e0
                moe.router.weight[1] = 4.0 * u_e1 + 3.5 * u_loop
                if l_idx == num_layers - 1:
                    exp1 = moe.experts["1"]
                    exp1.w3_A[0, 16:32] = 0.30 * u_e1[16:32]
                    exp1.w3_B[:, 0] = -0.60 * u_e1 + 0.85 * u_loop
                    exp1.w3_A[1, 32:40] = 0.45 * u_loop[32:40]
                    exp1.w3_B[:, 1] = 0.90 * u_loop
        elif split_mode == "fine_grained_width_split":
            for blk in model.blocks:
                moe_fg = blk.mlp
                assert isinstance(moe_fg, CrowdFineGrainedMoEMLP)
                moe_fg.router.weight.zero_()
                moe_fg.router.weight[0] = 4.0 * u_e0
                moe_fg.router.weight[1] = 4.0 * u_e1 + 3.5 * u_loop

    guard_chunks: list[dict[str, Any]] = []
    for idx in range(6):
        guard_chunks.append(
            {"expert_id": "E0_general_fineweb", "ids": [gen_tokens[idx % len(gen_tokens)]] * 10}
        )
        tok_c = concise_tokens[idx % len(concise_tokens)]
        guard_chunks.append(
            {
                "expert_id": "E1_python_systems_code",
                "ids": [prompt_tokens[idx % 4]] * 2 + [tok_c] * 8,
            }
        )

    pos_rows: list[list[int]] = []
    neg_rows: list[list[int]] = []
    for idx in range(12):
        tok_c = concise_tokens[idx % len(concise_tokens)]
        pos_rows.append([prompt_tokens[idx % 4]] * 3 + [tok_c] * 7)
        neg_rows.append(
            [prompt_tokens[idx % 4]] * 3 + [int(loop_tokens[(idx + j) % len(loop_tokens)]) for j in range(7)]
        )

    return model, {
        "guard_chunks": guard_chunks,
        "pos_ids": torch.tensor(pos_rows, dtype=torch.long),
        "neg_ids": torch.tensor(neg_rows, dtype=torch.long),
        "loop_tokens": loop_tokens,
        "u_e0": u_e0,
        "u_e1": u_e1,
        "u_loop": u_loop,
    }


def _extract_last_layer_trigger_acts(
    model: CrowdV1WithEngram,
    ids_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mlp_in_flat, block_out_flat)`` at the final transformer block."""
    cos, sin = model.cos_cached, model.sin_cached
    x = compute_embeddings_batch(model, ids_2d)
    for blk in model.blocks[:-1]:
        xm, mi = forward_block_pre_mlp(blk, x, cos, sin)
        x = forward_block_post_mlp(blk, xm, mi)
    xm_last, mi_last = forward_block_pre_mlp(model.blocks[-1], x, cos, sin)
    out_last = forward_block_post_mlp(model.blocks[-1], xm_last, mi_last)
    return (
        mi_last[:, :-1, :].reshape(-1, int(mi_last.shape[-1])).contiguous(),
        out_last[:, :-1, :].reshape(-1, int(out_last.shape[-1])).contiguous(),
    )


# ---------------------------------------------------------------------------
# S1: Static MoE Contrastive Weight Compilation (Lowers Spec NLL, Suppresses Loops, 0 Hooks)
# ---------------------------------------------------------------------------


def test_s1_static_moe_compilation_lowers_specialist_nll_suppresses_loops_zero_hooks() -> None:
    """S1: Compile concise-specialist vs thinking-loop activations into static MoE weights."""
    model, bundle = _build_contrastive_moe_model(split_mode="lora_domain_experts", seed=20261231)
    assert count_registered_runtime_hooks(model) == 0

    parent_sha = compute_module_state_sha256(model)
    compiled_model, receipt = execute_verified_contrastive_weight_compilation(
        model,
        pos_chunks_or_ids=bundle["pos_ids"],
        neg_chunks_or_ids=bundle["neg_ids"],
        guard_chunks=bundle["guard_chunks"],
        layer_indices=[2],
        expert_indices=[1],
        strength_alpha=0.60,
        ridge_lambda=1e-2,
        rank=2,
        min_oof_snr=0.25,
        repetition_token_ids=bundle["loop_tokens"],
    )

    assert compiled_model is model
    assert receipt["accepted"] is True
    assert receipt["rejected"] is False
    assert receipt["rolled_back"] is False
    assert receipt["weights_mutated"] is True
    assert receipt["status"] == "COMMITTED"
    assert receipt["active_sha256"] != parent_sha
    assert receipt["runtime_hook_count"] == 0
    assert count_registered_runtime_hooks(model) == 0
    assert receipt["guard_ok"] is True
    assert receipt["delta_energy"] < 0.0
    assert receipt["steering_snr"] >= 0.25
    assert receipt["specialist_nll_after"] < receipt["specialist_nll_before"] - 0.05
    assert receipt["loop_logit_mean_after"] < receipt["loop_logit_mean_before"] - 0.10
    assert receipt["loop_prob_mean_after"] < receipt["loop_prob_mean_before"]
    assert abs(float(receipt["general_drift_pct"])) <= 0.2
    assert verify_contrastive_compilation_receipt_provenance(receipt) is True

    # Tampering receipt fails provenance verification
    bad_receipt = dict(receipt)
    bad_receipt["nll_mean"] = float(receipt["nll_mean"]) + 0.01
    assert verify_contrastive_compilation_receipt_provenance(bad_receipt) is False

    # Also verify CrowdFineGrainedMoEMLP and dense CrowdRuntimeMLP compilation paths cleanly
    fg_model, fg_bundle = _build_contrastive_moe_model(split_mode="fine_grained_width_split", seed=20261232)
    _, fg_receipt = execute_verified_contrastive_weight_compilation(
        fg_model,
        pos_chunks_or_ids=fg_bundle["pos_ids"],
        neg_chunks_or_ids=fg_bundle["neg_ids"],
        guard_chunks=fg_bundle["guard_chunks"],
        layer_indices=[2],
        expert_indices=[1, 2],
        strength_alpha=0.55,
        repetition_token_ids=fg_bundle["loop_tokens"],
    )
    assert fg_receipt["accepted"] is True
    assert fg_receipt["runtime_hook_count"] == 0
    assert abs(float(fg_receipt["general_drift_pct"])) <= 0.2

    dense_model, dense_bundle = _build_contrastive_moe_model(split_mode="dense", seed=20261233)
    _, dense_receipt = execute_verified_contrastive_weight_compilation(
        dense_model,
        pos_chunks_or_ids=[{"ids": row} for row in dense_bundle["pos_ids"].tolist()],
        neg_chunks_or_ids=dense_bundle["neg_ids"].tolist(),
        guard_chunks=dense_bundle["guard_chunks"],
        layer_indices=[2],
        config=ContrastiveWeightCompilerConfig(strength_alpha=0.25, loop_energy_weight=0.02, score_from=1),
        repetition_token_ids=dense_bundle["loop_tokens"],
    )
    assert dense_receipt["accepted"] is True
    assert dense_receipt["specialist_nll_after"] < dense_receipt["specialist_nll_before"]
    assert dense_receipt["loop_logit_mean_after"] < dense_receipt["loop_logit_mean_before"]
    assert abs(float(dense_receipt["general_drift_pct"])) <= 0.2

    # Verify softmax_topk routing with layer_indices=None and automatic expert discovery
    sm_model, sm_bundle = _build_contrastive_moe_model(split_mode="dense", seed=20261234)
    upcycle_crowd_v1_to_sparse_moe(
        sm_model,
        split_mode="lora_domain_experts",
        num_experts=4,
        top_k=2,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode="softmax_topk",
        seed=20261235,
    )
    with torch.no_grad():
        moe_sm = sm_model.blocks[-1].mlp
        assert isinstance(moe_sm, CrowdLoRAMoEMLP)
        moe_sm.router.weight.zero_()
        moe_sm.router.weight[0] = 6.0 * sm_bundle["u_e0"]
        moe_sm.router.weight[1] = 6.0 * sm_bundle["u_e1"] + 5.0 * sm_bundle["u_loop"]
    sm_sha_before = compute_module_state_sha256(sm_model)
    _, sm_receipt = execute_verified_contrastive_weight_compilation(
        sm_model,
        pos_chunks_or_ids=sm_bundle["pos_ids"],
        neg_chunks_or_ids=sm_bundle["neg_ids"],
        eval_fn=lambda m: (
            {"E0_general": 1.15, "E1_spec": 1.25}
            if compute_module_state_sha256(m) == sm_sha_before
            else {"E0_general": 1.15, "E1_spec": 1.05}
        ),
        p_nulls={"blocks.2.act": torch.eye(96)},
        layer_indices=None,
        expert_indices=None,
        strength_alpha=0.45,
        repetition_token_ids=sm_bundle["loop_tokens"],
    )
    assert sm_receipt["accepted"] is True
    assert any(r["layer"] == 2 and r["expert"] == 1 for r in sm_receipt["compile_reports"])


# ---------------------------------------------------------------------------
# S2: Exact Preserved-Key Invariance (<1e-5 Drift) vs Unprojected CAA Bias
# ---------------------------------------------------------------------------


def test_s2_exact_preserved_key_invariance_vs_unprojected_caa_bias() -> None:
    """S2: Verify <1e-5 preserved-key invariance inside P_0 vs unprojected CAA prose corruption."""
    gen = torch.Generator(device="cpu").manual_seed(20261240)
    d_in, d_out = 24, 16
    # Preserved general prose keys lie strictly in first 8 dimensions; specialist trigger spans 8..16 + partial overlap
    prose_keys = torch.zeros((24, d_in), dtype=torch.float32)
    prose_keys[:, :8] = torch.randn((24, 8), generator=gen, dtype=torch.float32)

    p_null, p_info = build_prose_nullspace_projector(prose_keys, sv_threshold=1e-2, min_null_rank=4)
    assert p_info["active_modes"] == 8
    assert p_info["nullspace_dim"] == d_in - 8

    trig_keys = torch.zeros((20, d_in), dtype=torch.float32)
    trig_keys[:, :8] = 0.65 * torch.randn((20, 8), generator=gen, dtype=torch.float32)
    trig_keys[:, 8:16] = 1.25 * torch.randn((20, 8), generator=gen, dtype=torch.float32)

    pos_acts = torch.randn((20, d_out), generator=gen, dtype=torch.float32) * 0.25
    neg_acts = torch.randn((20, d_out), generator=gen, dtype=torch.float32) * 0.25
    u_0 = torch.zeros(d_out)
    u_0[0] = 1.0
    u_1 = torch.zeros(d_out)
    u_1[1] = 1.0
    pos_acts[:10] += 1.6 * u_0
    pos_acts[10:] += 1.3 * u_1

    sub_basis, sub_info = extract_oof_contrastive_steering_subspace(pos_acts, neg_acts, rank=2, min_oof_snr=0.25)
    assert sub_info["accepted"] is True
    assert sub_info["effective_rank"] == 2
    assert sub_basis.shape == (d_out, 2)

    # 1. Dense nn.Linear: P_0 keeps <1e-5 drift on prose_keys, while unprojected p_null=None corrupts prose_keys
    lin_base = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        lin_base.weight.copy_(0.2 * torch.randn((d_out, d_in), generator=gen, dtype=torch.float32))
    prose_out_ref = lin_base(prose_keys).detach().clone()

    lin_null, rep_null = compile_steering_vector_into_nullspace_weight(
        lin_base,
        trig_keys,
        sub_info["steering_vector"],
        p_null,
        strength_alpha=0.9,
        ridge_lambda=1e-2,
        inplace=False,
    )
    lin_unproj, rep_unproj = compile_steering_vector_into_nullspace_weight(
        lin_base,
        trig_keys,
        sub_info["steering_vector"],
        None,
        strength_alpha=0.9,
        ridge_lambda=1e-2,
        inplace=False,
    )
    prose_drift_null = float((lin_null(prose_keys) - prose_out_ref).abs().max().item())
    prose_drift_unproj = float((lin_unproj(prose_keys) - prose_out_ref).abs().max().item())
    assert rep_null["applied"] is True
    assert rep_null["actual_weight_nullspace_leakage_norm"] < 1e-5
    assert prose_drift_null < 1e-5
    assert prose_drift_unproj > 0.15
    assert rep_null["trigger_alignment_cosine"] > 0.95

    # 2. Zero-slot LoRA (CrowdLoRAExpertAdapter w3_B == 0)
    exp_zero = CrowdLoRAExpertAdapter(d_out, d_in, rank=4, alpha=8.0, seed=11)
    w_a0 = exp_zero.w3_A.detach().clone()
    w_b0 = exp_zero.w3_B.detach().clone()
    before_zero = F.linear(F.linear(prose_keys, w_a0), w_b0) * exp_zero.scaling
    _, rep_exp0 = compile_steering_vector_into_nullspace_weight(
        exp_zero, trig_keys, sub_basis, p_null, strength_alpha=0.7, weight_attr="w3"
    )
    after_zero = F.linear(F.linear(prose_keys, exp_zero.w3_A), exp_zero.w3_B) * exp_zero.scaling
    assert rep_exp0["lora_fold_mode_used"] == "zero_slot_exact"
    assert float((after_zero - before_zero).abs().max().item()) < 1e-5
    assert rep_exp0["effective_delta_rank"] == 2

    # 3. Full-rank pre-populated CrowdLoRALinear: partitioned_nullspace_svd & update_a_nullspace both keep <1e-5 drift!
    base_lin = nn.Linear(d_in, d_out, bias=False)
    lora_part = CrowdLoRALinear(base_lin, rank=4, alpha=8.0)
    with torch.no_grad():
        # Occupy 2 rank slots on prose keys (r_pres=2 < rank=4 -> partitioned_nullspace_svd)
        lora_part.lora_B.zero_()
        lora_part.lora_B[:, 0] = torch.randn(d_out, generator=gen)
        lora_part.lora_B[:, 1] = torch.randn(d_out, generator=gen)
        lora_part.lora_B[:, 2:] = 0.05 * torch.randn((d_out, 2), generator=gen)
        lora_part.lora_A.zero_()
        lora_part.lora_A[0, :4] = 0.8
        lora_part.lora_A[1, 4:8] = 0.8
        lora_part.lora_A[2:, 8:] = 0.2
    prose_part_before = lora_part(prose_keys).detach().clone()
    _, rep_part = compile_steering_vector_into_nullspace_weight(
        lora_part, trig_keys, sub_info, p_null, strength_alpha=0.6
    )
    prose_part_after = lora_part(prose_keys).detach().clone()
    assert rep_part["lora_fold_mode_used"] == "partitioned_nullspace_svd"
    assert float((prose_part_after - prose_part_before).abs().max().item()) < 1e-5

    # Saturate all 4 rank slots on prose keys -> triggers update_a_nullspace (still <1e-5 drift!)
    lora_full = CrowdLoRALinear(base_lin, rank=4, alpha=8.0)
    with torch.no_grad():
        lora_full.lora_B.copy_(torch.randn((d_out, 4), generator=gen))
        lora_full.lora_A.copy_(torch.randn((4, d_in), generator=gen))
    prose_full_before = lora_full(prose_keys).detach().clone()
    _, rep_full = compile_steering_vector_into_nullspace_weight(
        lora_full, trig_keys, sub_info["steering_vector"], p_null, strength_alpha=0.6
    )
    prose_full_after = lora_full(prose_keys).detach().clone()
    assert rep_full["lora_fold_mode_used"] == "update_a_nullspace"
    assert float((prose_full_after - prose_full_before).abs().max().item()) < 1e-5

    # Unconstrained qr_svd mode + tuple/mapping targets + max_norm_delta_ratio cap
    lora_qr = CrowdLoRALinear(base_lin, rank=4, alpha=8.0)
    with torch.no_grad():
        lora_qr.lora_B.copy_(torch.randn((d_out, 4), generator=gen))
    _, rep_qr = compile_steering_vector_into_nullspace_weight(
        lora_qr,
        trig_keys,
        sub_info["steering_vector"],
        None,
        strength_alpha=0.5,
        lora_fold_mode="qr_svd",
        max_norm_delta_ratio=0.01,
    )
    assert rep_qr["lora_fold_mode_used"] == "qr_svd_rank_preserving"
    assert rep_qr["budget_clamped"] is True

    # Also verify tuple, mapping, duck-typed LoRA, module.w3/weight, and (sub_basis, info) / per-token / empty targets
    b_tup = torch.zeros((d_out, 2))
    a_tup = torch.randn((2, d_in), generator=gen)
    (b_new, a_new), rep_tup = compile_steering_vector_into_nullspace_weight(
        (b_tup, a_tup), trig_keys, (sub_basis, sub_info), p_null, inplace=False
    )
    assert rep_tup["applied"] is True
    assert float(((b_new @ a_new - b_tup @ a_tup) @ prose_keys.T).abs().max().item()) < 1e-5

    # Collinear triggers + 2D subspace basis, 1D single trigger, zero (n_trig, d_out), zero (d_out, 2), empty (d_out, 0)
    col_trig = trig_keys[:1].expand(6, d_in).clone()
    _, rep_col = compile_steering_vector_into_nullspace_weight(lin_base, col_trig, sub_basis, p_null, inplace=False)
    assert rep_col["applied"] is True
    _, rep_1d = compile_steering_vector_into_nullspace_weight(lin_base, trig_keys[0], sub_basis, p_null, inplace=False)
    assert rep_1d["applied"] is True
    per_tok = torch.zeros((int(trig_keys.shape[0]), d_out), dtype=torch.float32)
    _, rep_pt0 = compile_steering_vector_into_nullspace_weight(lin_base, trig_keys, per_tok, p_null, inplace=False)
    assert rep_pt0["applied"] is False and rep_pt0["lora_fold_mode_used"] == "noop_zero_steering"
    _, rep_e0 = compile_steering_vector_into_nullspace_weight(lin_base, trig_keys, torch.zeros((d_out, 0)), p_null, inplace=False)
    assert rep_e0["applied"] is False
    _, rep_z2 = compile_steering_vector_into_nullspace_weight(lin_base, trig_keys, torch.zeros((d_out, 2)), p_null, inplace=False)
    assert rep_z2["applied"] is False

    # Build projector from Mapping and test explicit multi-target compilation via dotted module traversal + Tensor p_null
    p_map, _ = build_prose_nullspace_projector({"E0_general": prose_keys})
    assert p_map.shape == (d_in, d_in)

    class _DuckLoRA(nn.Module):
        def __init__(self, out_d: int, in_d: int) -> None:
            super().__init__()
            self.lora_B = nn.Parameter(torch.zeros((out_d, 2)))
            self.lora_A = nn.Parameter(torch.randn((2, in_d), generator=gen) * 0.1)

    class _WeightWrapper(nn.Module):
        def __init__(self, out_d: int, in_d: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros((out_d, in_d)))

    duck = _DuckLoRA(d_out, d_in)
    _, rep_duck = compile_steering_vector_into_nullspace_weight(duck, trig_keys, per_tok + 0.1, p_null)
    assert rep_duck["target_kind"] == "lora_linear" and rep_duck["applied"] is True

    wrap = _WeightWrapper(d_out, d_in)
    _, rep_wrap = compile_steering_vector_into_nullspace_weight(wrap, trig_keys, sub_info["steering_vector"], p_null)
    assert rep_wrap["target_kind"] == "module_weight" and rep_wrap["applied"] is True

    moe_model, _ = _build_contrastive_moe_model(split_mode="lora_domain_experts", seed=20261241)
    # Use a 2-key test: blocks.2.mlp.experts.1 (with prose_keys Mapping) and CrowdRuntimeMLP.w3
    exp_p = torch.randn((20, 48), generator=gen) * 0.1 + 1.5 * _unit_axis(48, 16, 17)
    exp_n = torch.randn((20, 48), generator=gen) * 0.1
    exp_trig = torch.cat([torch.zeros((20, 96)), torch.zeros((20, 96))], dim=0)
    exp_trig[:, 16:32] = 0.5
    exp_prose = torch.zeros((16, 96))
    exp_prose[:, :16] = torch.randn((16, 16), generator=gen)
    base_sha = compute_module_state_sha256(moe_model)
    _, exp_rec = execute_verified_contrastive_weight_compilation(
        moe_model,
        pos_activations={"blocks.2.mlp.experts.1": exp_p},
        neg_activations={"blocks.2.mlp.experts.1": exp_n},
        trigger_activations={"blocks.2.mlp.experts.1": exp_trig},
        prose_keys={"blocks.2.mlp.experts.1": exp_prose},
        eval_fn=lambda m: (
            {"E0_gen": 1.2, "E1_spec": 1.5}
            if compute_module_state_sha256(m) == base_sha
            else {"E0_gen": 1.2, "E1_spec": 1.1}
        ),
        strength_alpha=0.5,
    )
    assert exp_rec["accepted"] is True

    dense_mlp_t = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=48, intermediate_size=96, vocab_size=64))
    sha_dm = compute_module_state_sha256(dense_mlp_t)
    _, rep_dm = compile_steering_vector_into_nullspace_weight(
        dense_mlp_t, torch.randn((12, 96), generator=gen), torch.ones(48), None, weight_attr="w3"
    )
    assert rep_dm["target_kind"] == "dense_w3" and rep_dm["applied"] is True
    # Also test explicit tensor p_null and trigger_activations=None when d_out == d_in
    lin_sq = nn.Linear(48, 48, bias=False)
    sha_sq = compute_module_state_sha256(lin_sq)
    _, sq_rec = execute_verified_contrastive_weight_compilation(
        lin_sq,
        pos_activations={"default": exp_p},
        neg_activations={"default": exp_n},
        p_nulls=torch.eye(48),
        eval_fn=lambda m: (
            {"E0_gen": 1.2, "E1_spec": 1.5}
            if compute_module_state_sha256(m) == sha_sq
            else {"E0_gen": 1.2, "E1_spec": 1.1}
        ),
    )
    assert sq_rec["accepted"] is True
    assert compute_module_state_sha256(dense_mlp_t) != sha_dm


# ---------------------------------------------------------------------------
# S3: OOF SNR Anti-Overfitting Gate Rejecting Pure Gaussian Noise (Zero Mutation)
# ---------------------------------------------------------------------------


def test_s3_oof_snr_gate_rejects_random_gaussian_with_zero_mutation() -> None:
    """S3: Reject pure random Gaussian contrastive pairs with zero weight mutation."""
    for seed in (101, 202, 303):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pos_noise = torch.randn((48, 32), generator=gen, dtype=torch.float32)
        neg_noise = torch.randn((48, 32), generator=gen, dtype=torch.float32)

        subspace, info = extract_oof_contrastive_steering_subspace(
            pos_noise, neg_noise, rank=2, min_oof_snr=0.25
        )
        assert subspace.shape == (32, 0)
        assert info["effective_rank"] == 0
        assert info["accepted"] is False
        assert info["rejected"] is True
        assert info["zeroed_by_oof_gate"] is True
        assert info["oof_snr"] < 0.25
        assert float(info["steering_vector"].abs().max().item()) == 0.0

    # Rank-1 signal + isotropic noise with rank=2 should reject the noisy 2nd mode cleanly -> effective_rank == 1
    gen_r1 = torch.Generator(device="cpu").manual_seed(404)
    e0 = torch.zeros(32)
    e0[0] = 1.0
    pos_r1 = torch.randn((32, 32), generator=gen_r1) * 0.25 + 1.2 * e0
    neg_r1 = torch.randn((32, 32), generator=gen_r1) * 0.25
    sub_r1, info_r1 = extract_oof_contrastive_steering_subspace(pos_r1, neg_r1, rank=2, min_oof_snr=0.25)
    assert info_r1["accepted"] is True
    assert info_r1["effective_rank"] == 1
    assert sub_r1.shape == (32, 1)

    # Unpaired accepted samples, orthogonal-fold rejection, zero-mean fold rejection, dim=2 remaining_dim<=1, noiseless rank-1
    pos_unp_ok = torch.randn((14, 8), generator=gen_r1) * 0.1 + 1.5 * _unit_axis(8, 0, 1)
    neg_unp_ok = torch.randn((10, 8), generator=gen_r1) * 0.1
    sub_unp_ok, info_unp_ok = extract_oof_contrastive_steering_subspace(pos_unp_ok, neg_unp_ok, rank=2, min_oof_snr=0.25)
    assert info_unp_ok["accepted"] is True and sub_unp_ok.shape == (8, 1)

    # Zero mean contrast in fold 0 -> n_mu0 <= 1e-12
    p_z0 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    n_z0 = torch.zeros((4, 2))
    _, info_z0 = extract_oof_contrastive_steering_subspace(p_z0, n_z0, rank=2, min_oof_snr=0.25)
    assert info_z0["rejected"] is True

    # Remaining_dim <= 1 guard on dim=2, rank=3 + noiseless constant rank-1 SVD stop on dim=4, rank=2
    p_2d = torch.tensor([[1.0, 0.0], [1.1, 0.1], [1.2, -0.1], [1.05, 0.05]])
    n_2d = torch.zeros((4, 2))
    _, info_2d = extract_oof_contrastive_steering_subspace(p_2d, n_2d, rank=3, min_oof_snr=0.25)
    assert info_2d["accepted"] is True and info_2d["effective_rank"] == 1
    p_c4 = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 6)
    n_c4 = torch.zeros((6, 4))
    _, info_c4 = extract_oof_contrastive_steering_subspace(p_c4, n_c4, rank=2, min_oof_snr=0.25)
    assert info_c4["accepted"] is True and info_c4["effective_rank"] == 1

    # End-to-end compilation with pure Gaussian contrastive pairs leaves model bitwise unchanged
    model, bundle = _build_contrastive_moe_model(split_mode="lora_domain_experts", seed=20261250)
    parent_sha = compute_module_state_sha256(model)
    gen_m = torch.Generator(device="cpu").manual_seed(505)
    noisy_pos = torch.randn((40, 48), generator=gen_m, dtype=torch.float32)
    noisy_neg = torch.randn((40, 48), generator=gen_m, dtype=torch.float32)
    trig_k = torch.randn((20, 96), generator=gen_m, dtype=torch.float32)

    def _eval_clean(_m: nn.Module) -> dict[str, float]:
        return {"E0_general": 1.20, "E1_spec": 1.40}

    out_model, receipt = execute_verified_contrastive_weight_compilation(
        model,
        pos_activations={"blocks.2.mlp.experts.1": noisy_pos},
        neg_activations={"blocks.2.mlp.experts.1": noisy_neg},
        trigger_activations={"blocks.2.mlp.experts.1": trig_k},
        eval_fn=_eval_clean,
        min_oof_snr=0.25,
    )
    assert out_model is model
    assert receipt["accepted"] is False
    assert receipt["rejected"] is True
    assert receipt["weights_mutated"] is False
    assert receipt["status"] == "REJECTED_LOW_OOF_SNR"
    assert receipt["active_sha256"] == parent_sha
    assert compute_module_state_sha256(model) == parent_sha


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback on Over-Steered Alpha
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_on_oversteered_alpha() -> None:
    """S4: Over-steered alpha triggers ANSE Maximum Pain (1e6) and atomic bitwise rollback."""
    model, bundle = _build_contrastive_moe_model(split_mode="lora_domain_experts", seed=20261260)
    parent_sha = compute_module_state_sha256(model)
    parent_sd = {k: v.clone() for k, v in model.state_dict().items()}

    out_model, receipt = execute_verified_contrastive_weight_compilation(
        model,
        pos_chunks_or_ids=bundle["pos_ids"],
        neg_chunks_or_ids=bundle["neg_ids"],
        guard_chunks=bundle["guard_chunks"],
        layer_indices=[2],
        expert_indices=[1],
        strength_alpha=180.0,
        ridge_lambda=1e-2,
        repetition_token_ids=bundle["loop_tokens"],
    )
    assert out_model is model
    assert receipt["accepted"] is False
    assert receipt["rejected"] is True
    assert receipt["rolled_back"] is True
    assert receipt["weights_mutated"] is False
    assert receipt["guard_ok"] is False
    assert receipt["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert receipt["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert receipt["candidate_sha256"] != parent_sha
    assert receipt["active_sha256"] == parent_sha
    assert compute_module_state_sha256(model) == parent_sha
    for key, orig_t in parent_sd.items():
        assert torch.equal(model.state_dict()[key], orig_t)
    assert verify_contrastive_compilation_receipt_provenance(receipt) is True

    # Also test ROLLED_BACK_NO_ENERGY_IMPROVEMENT and ROLLED_BACK_NON_FINITE via custom eval_fn
    lin = nn.Linear(16, 16, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(16))
    lin_sha = compute_module_state_sha256(lin)
    gen_e = torch.Generator(device="cpu").manual_seed(606)
    e0 = torch.zeros(16)
    e0[0] = 1.0
    p_act = torch.randn((24, 16), generator=gen_e) * 0.1 + 1.0 * e0
    n_act = torch.randn((24, 16), generator=gen_e) * 0.1
    prose_k = torch.eye(16)[:4]
    trig_k = torch.cat([torch.eye(16)[8:12], torch.eye(16)[8:12]], dim=0)

    calls = {"n": 0}

    def _eval_no_improve(_m: nn.Module) -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"E0_general": 1.00, "E1_spec": 1.00}
        return {"E0_general": 1.005, "E1_spec": 1.005}

    _, rep_no_imp = execute_verified_contrastive_weight_compilation(
        lin,
        pos_activations={"default": p_act},
        neg_activations={"default": n_act},
        trigger_activations={"default": trig_k},
        prose_keys=prose_k,
        eval_fn=_eval_no_improve,
        strength_alpha=0.5,
    )
    assert rep_no_imp["rolled_back"] is True
    assert rep_no_imp["status"] == "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"
    assert compute_module_state_sha256(lin) == lin_sha

    calls_nan = {"n": 0}

    def _eval_nan(_m: nn.Module) -> dict[str, float]:
        calls_nan["n"] += 1
        if calls_nan["n"] == 1:
            return {"E0_general": 1.00, "E1_spec": 1.00}
        return {"E0_general": float("nan"), "E1_spec": 1.00}

    _, rep_nan = execute_verified_contrastive_weight_compilation(
        lin,
        pos_activations={"default": p_act},
        neg_activations={"default": n_act},
        trigger_activations={"default": trig_k},
        p_nulls={"default": torch.eye(16)},
        eval_fn=_eval_nan,
        strength_alpha=0.5,
    )
    assert rep_nan["rolled_back"] is True
    assert rep_nan["status"] == "ROLLED_BACK_NON_FINITE"
    assert rep_nan["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert compute_module_state_sha256(lin) == lin_sha

    # Overflowing alpha on real CrowdV1WithEngram model (both activation overflow 1e20 & weight overflow 1e38) roll back bitwise
    ov_model, ov_bundle = _build_contrastive_moe_model(split_mode="dense", seed=20261261)
    ov_sha = compute_module_state_sha256(ov_model)
    for extreme_alpha in (1e20, 1e38):
        _, rep_ov = execute_verified_contrastive_weight_compilation(
            ov_model,
            pos_chunks_or_ids=ov_bundle["pos_ids"],
            neg_chunks_or_ids=ov_bundle["neg_ids"],
            guard_chunks=ov_bundle["guard_chunks"],
            layer_indices=[2],
            strength_alpha=extreme_alpha,
            repetition_token_ids=ov_bundle["loop_tokens"],
        )
        assert rep_ov["rolled_back"] is True
        assert rep_ov["status"] == "ROLLED_BACK_NON_FINITE"
        assert compute_module_state_sha256(ov_model) == ov_sha

    # Bad post-compile eval_fn returns (exception, bad map, non-numeric, extra domain, missing general) all rollback cleanly
    for bad_ret in ("raises", "empty", "no_dom", "bad_val", "mismatch_dom", "no_gen"):
        c_b = {"n": 0}

        def _bad_eval(_m: nn.Module, kind: str = bad_ret, state: dict[str, int] = c_b) -> Any:
            state["n"] += 1
            if state["n"] == 1:
                return {"domain_nll": {"E0_gen": 1.0, "E1_spec": 1.0}, "nll_mean": 1.0}
            if kind == "raises":
                raise RuntimeError("boom")
            if kind == "empty":
                return {}
            if kind == "no_dom":
                return {"domain_nll": "bad"}
            if kind == "bad_val":
                return {"E0_gen": True, "E1_spec": 1.0}
            if kind == "mismatch_dom":
                return {"E0_gen": 1.0, "E2_other": 1.0}
            return {"spec_only": 1.0}

        if bad_ret == "no_gen":
            _, rep_b = execute_verified_contrastive_weight_compilation(
                lin,
                pos_activations={"default": p_act},
                neg_activations={"default": n_act},
                trigger_activations={"default": trig_k},
                eval_fn=lambda m: {"E0_gen": 1.0, "E1_spec": 1.0}
                if compute_module_state_sha256(m) == lin_sha
                else {"S1": 0.9, "S2": 0.9},
            )
        else:
            _, rep_b = execute_verified_contrastive_weight_compilation(
                lin,
                pos_activations={"default": p_act},
                neg_activations={"default": n_act},
                trigger_activations={"default": trig_k},
                eval_fn=_bad_eval,
            )
        assert rep_b["rolled_back"] is True and rep_b["status"] == "ROLLED_BACK_NON_FINITE"
        assert compute_module_state_sha256(lin) == lin_sha


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / Stub / Gatekeeper Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_compliance_gates() -> None:
    """S5: Exercise validation checks, edge branches, ZeroStubAudit, PRSecurityScanner, and gatekeeper."""
    # Config validation
    with pytest.raises(TypeError, match="int"):
        ContrastiveWeightCompilerConfig(rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        ContrastiveWeightCompilerConfig(rank=0)
    with pytest.raises(ValueError, match=">= 0"):
        ContrastiveWeightCompilerConfig(min_null_rank=-1)
    with pytest.raises(TypeError, match="real number"):
        ContrastiveWeightCompilerConfig(strength_alpha=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        ContrastiveWeightCompilerConfig(min_oof_snr=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        ContrastiveWeightCompilerConfig(strength_alpha=-0.5)
    with pytest.raises(ValueError, match="> 0.0"):
        ContrastiveWeightCompilerConfig(ridge_lambda=0.0)
    with pytest.raises(ValueError, match="> 0.0"):
        ContrastiveWeightCompilerConfig(max_norm_delta_ratio=0.0)
    with pytest.raises(ValueError, match="target_proj"):
        ContrastiveWeightCompilerConfig(target_proj="bad")
    with pytest.raises(ValueError, match="lora_fold_mode"):
        ContrastiveWeightCompilerConfig(lora_fold_mode="bad")
    with pytest.raises(ValueError, match="< 1.0"):
        ContrastiveWeightCompilerConfig(nullspace_sv_rel_tol=1.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        ContrastiveWeightCompilerConfig(max_norm_delta_ratio=1.5)
    with pytest.raises(ValueError, match="general_prefix"):
        ContrastiveWeightCompilerConfig(general_prefix=" ")
    with pytest.raises(TypeError, match="bool"):
        ContrastiveWeightCompilerConfig(inplace=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        count_registered_runtime_hooks("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sv_rel_tol"):
        build_prose_nullspace_projector(torch.ones((4, 8)), sv_rel_tol=1.0)
    with pytest.raises(ValueError, match="empty"):
        build_prose_nullspace_projector({})

    # Extractor validation & edge branches (p_null, unpaired, 3D, N=2 LOO)
    ok = torch.ones((6, 8))
    with pytest.raises(TypeError, match="Tensor"):
        extract_oof_contrastive_steering_subspace("bad", ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        extract_oof_contrastive_steering_subspace([], ok)
    with pytest.raises(TypeError, match="Tensor"):
        extract_oof_contrastive_steering_subspace(["bad"], ok)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-empty"):
        extract_oof_contrastive_steering_subspace(torch.ones(8), ok)
    with pytest.raises(TypeError, match="floating-point"):
        extract_oof_contrastive_steering_subspace(torch.ones((6, 8), dtype=torch.long), ok)
    with pytest.raises(ValueError, match="finite"):
        extract_oof_contrastive_steering_subspace(torch.full((6, 8), float("nan")), ok)
    with pytest.raises(ValueError, match="width"):
        extract_oof_contrastive_steering_subspace([torch.ones((3, 8)), torch.ones((3, 4))], ok)
    with pytest.raises(ValueError, match="at least 2"):
        extract_oof_contrastive_steering_subspace(torch.ones((1, 8)), ok)
    with pytest.raises(ValueError, match="width"):
        extract_oof_contrastive_steering_subspace(ok, torch.ones((6, 4)))

    # Unpaired + p_null + 3D + small N=2 paths
    g = torch.Generator().manual_seed(707)
    p_unp = torch.randn((3, 4, 8), generator=g) * 0.05 + torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    n_unp = torch.randn((2, 4, 8), generator=g) * 0.05
    p_eye = torch.eye(8)
    p_eye[0, 0] = 0.0
    sub_p, info_p = extract_oof_contrastive_steering_subspace(p_unp, n_unp, rank=2, min_oof_snr=0.1, p_null=p_eye)
    assert info_p["rejected"] is True and sub_p.shape == (8, 0)
    sub_2, info_2 = extract_oof_contrastive_steering_subspace(
        torch.ones((2, 4)) + torch.tensor([[0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]]),
        torch.zeros((2, 4)),
        rank=1,
        min_oof_snr=0.1,
    )
    assert info_2["accepted"] is True and sub_2.shape == (4, 1)

    # Compiler validations & no-op / mapping / MoE target paths
    w = nn.Linear(8, 6, bias=False)
    trig = torch.ones((4, 8))
    sv = torch.ones(6)
    with pytest.raises(TypeError, match="weight_or_lora"):
        compile_steering_vector_into_nullspace_weight("bad", trig, sv, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight_attr"):
        compile_steering_vector_into_nullspace_weight(w, trig, sv, None, weight_attr=" ")
    with pytest.raises(ValueError, match="non-empty 2D"):
        compile_steering_vector_into_nullspace_weight(torch.ones(4), trig, sv, None)
    with pytest.raises(ValueError, match="finite"):
        compile_steering_vector_into_nullspace_weight(torch.full((6, 8), float("nan")), trig, sv, None)
    with pytest.raises(ValueError, match="d_in=8"):
        compile_steering_vector_into_nullspace_weight(w, torch.ones((4, 5)), sv, None)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compile_steering_vector_into_nullspace_weight(w, trig, sv, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square 2D"):
        compile_steering_vector_into_nullspace_weight(w, trig, sv, torch.ones((8, 6)))
    with pytest.raises(TypeError, match="floating-point"):
        compile_steering_vector_into_nullspace_weight(w, trig, sv, torch.ones((8, 8), dtype=torch.long))
    with pytest.raises(ValueError, match="input dimension 8"):
        compile_steering_vector_into_nullspace_weight(w, trig, sv, torch.eye(4))
    with pytest.raises(ValueError, match="finite"):
        compile_steering_vector_into_nullspace_weight(w, trig, sv, torch.full((8, 8), float("nan")))
    with pytest.raises(ValueError, match="d_out=6"):
        compile_steering_vector_into_nullspace_weight(w, trig, torch.ones(5), None)
    with pytest.raises(ValueError, match="lora_fold_mode"):
        compile_steering_vector_into_nullspace_weight(w, trig, sv, None, lora_fold_mode="bad")
    with pytest.raises(ValueError, match="mapping"):
        compile_steering_vector_into_nullspace_weight(w, trig, {"bad": sv}, None)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compile_steering_vector_into_nullspace_weight(w, trig, "bad", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        compile_steering_vector_into_nullspace_weight(w, trig, torch.ones(6, dtype=torch.long), None)
    with pytest.raises(ValueError, match="1D or 2D"):
        compile_steering_vector_into_nullspace_weight(w, trig, torch.ones((1, 2, 6)), None)
    with pytest.raises(ValueError, match="finite"):
        compile_steering_vector_into_nullspace_weight(w, trig, torch.full((6,), float("nan")), None)
    with pytest.raises(ValueError, match="d_out=6"):
        compile_steering_vector_into_nullspace_weight(w, trig, torch.ones((5, 0)), None)
    with pytest.raises(ValueError, match="2D steering_vec"):
        compile_steering_vector_into_nullspace_weight(w, trig, torch.ones((5, 3)), None)

    # Zero steering / collapsed nullspace / underflow-delta / LoRA underflow no-ops
    _, rep_z = compile_steering_vector_into_nullspace_weight(w, trig, torch.zeros(6), None, strength_alpha=0.5)
    assert rep_z["applied"] is False and rep_z["lora_fold_mode_used"] == "noop_zero_steering"
    _, rep_c = compile_steering_vector_into_nullspace_weight(w, trig, sv, torch.zeros((8, 8)), strength_alpha=0.5)
    assert rep_c["applied"] is False and rep_c["lora_fold_mode_used"] == "noop_nullspace_collapsed"
    _, rep_tiny = compile_steering_vector_into_nullspace_weight(w, trig, sv, None, strength_alpha=1e-18)
    assert rep_tiny["applied"] is False and rep_tiny["lora_fold_mode_used"] == "noop_zero_delta"
    _, rep_tmat = compile_steering_vector_into_nullspace_weight(
        torch.ones((6, 8)), trig, sv, None, strength_alpha=0.1, inplace=False
    )
    assert rep_tmat["target_kind"] == "dense_tensor" and rep_tmat["applied"] is True
    exp_huge = CrowdLoRAExpertAdapter(6, 8, rank=2, alpha=2.0, seed=1)
    exp_huge.scaling = 1e6
    _, rep_lora_tiny = compile_steering_vector_into_nullspace_weight(
        exp_huge, trig, sv, None, strength_alpha=1e-14, weight_attr="w3"
    )
    assert rep_lora_tiny["lora_fold_mode_used"] == "noop_zero_delta"

    # Tuple / LoRA factor validation errors
    with pytest.raises(TypeError, match="torch.Tensor"):
        compile_steering_vector_into_nullspace_weight(("bad", torch.ones((2, 8))), trig, sv, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 2D"):
        compile_steering_vector_into_nullspace_weight((torch.ones(6), torch.ones((2, 8))), trig, sv, None)
    with pytest.raises(ValueError, match="rank mismatch"):
        compile_steering_vector_into_nullspace_weight((torch.ones((6, 2)), torch.ones((3, 8))), trig, sv, None)
    with pytest.raises(TypeError, match="floating-point"):
        compile_steering_vector_into_nullspace_weight(
            (torch.ones((6, 2), dtype=torch.long), torch.ones((2, 8))), trig, sv, None
        )
    with pytest.raises(ValueError, match="finite"):
        compile_steering_vector_into_nullspace_weight(
            (torch.full((6, 2), float("nan")), torch.ones((2, 8))), trig, sv, None
        )

    # Mapping LoRA target and MoE expert validation
    b_map = torch.zeros((6, 2))
    a_map = torch.ones((2, 8))
    _, rep_map = compile_steering_vector_into_nullspace_weight({"w3_B": b_map, "w3_A": a_map}, trig, sv, None)
    assert rep_map["applied"] is True
    _, rep_map_ba = compile_steering_vector_into_nullspace_weight({"B": b_map.clone(), "A": a_map.clone()}, trig, sv, None)
    assert rep_map_ba["target_kind"] == "lora_mapping"
    with pytest.raises(ValueError, match="Mapping target"):
        compile_steering_vector_into_nullspace_weight({"x": b_map}, trig, sv, None)

    dense_mlp = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=8, intermediate_size=16, vocab_size=16))
    moe_lora = CrowdLoRAMoEMLP(dense_mlp, num_experts=2, top_k=1, rank=2)
    with pytest.raises(ValueError, match="expert_index"):
        compile_steering_vector_into_nullspace_weight(moe_lora, torch.ones((4, 16)), torch.ones(8), None)
    with pytest.raises(ValueError, match="not found"):
        compile_steering_vector_into_nullspace_weight(moe_lora, torch.ones((4, 16)), torch.ones(8), None, expert_index=9)
    _, rep_moe = compile_steering_vector_into_nullspace_weight(
        moe_lora, torch.ones((4, 16)), torch.ones(8), None, expert_index=1
    )
    assert rep_moe["applied"] is True

    moe_fg = CrowdFineGrainedMoEMLP(dense_mlp, num_experts=2, top_k=1)
    with pytest.raises(ValueError, match="expert_index"):
        compile_steering_vector_into_nullspace_weight(moe_fg, torch.ones((4, 8)), torch.ones(8), None)
    with pytest.raises(ValueError, match="out of range"):
        compile_steering_vector_into_nullspace_weight(moe_fg, torch.ones((4, 8)), torch.ones(8), None, expert_index=5)
    with pytest.raises(ValueError, match="no experts_w9"):
        compile_steering_vector_into_nullspace_weight(moe_fg, torch.ones((4, 8)), torch.ones(8), None, weight_attr="w9", expert_index=0)
    with pytest.raises(ValueError, match="Unsupported weight_or_lora"):
        compile_steering_vector_into_nullspace_weight(nn.ReLU(), trig, sv, None)

    # Execution validation errors & inplace=False copy behavior
    with pytest.raises(ValueError, match="non-empty nn.Module"):
        execute_verified_contrastive_weight_compilation(nn.ReLU())
    with pytest.raises(TypeError, match="callable"):
        execute_verified_contrastive_weight_compilation(w, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ContrastiveWeightCompilerConfig"):
        execute_verified_contrastive_weight_compilation(w, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Both pos_activations"):
        execute_verified_contrastive_weight_compilation(w, pos_activations={"default": ok})
    with pytest.raises(ValueError, match="eval_fn or guard_chunks"):
        execute_verified_contrastive_weight_compilation(w, pos_activations={"default": ok}, neg_activations={"default": ok})
    with pytest.raises(ValueError, match="pos_chunks_or_ids"):
        execute_verified_contrastive_weight_compilation(w)
    with pytest.raises(ValueError, match="guard_chunks or eval_fn"):
        execute_verified_contrastive_weight_compilation(
            w, pos_chunks_or_ids=torch.zeros((2, 4), dtype=torch.long), neg_chunks_or_ids=torch.zeros((2, 4), dtype=torch.long)
        )
    with pytest.raises(ValueError, match="CrowdV1WithEngram-compatible"):
        execute_verified_contrastive_weight_compilation(
            w,
            pos_activations={"default": ok},
            neg_activations={"default": ok},
            guard_chunks=[{"expert_id": "E0_g", "ids": [0, 1]}],
        )
    with pytest.raises(ValueError, match="share at least one"):
        execute_verified_contrastive_weight_compilation(
            w, pos_activations={"a": ok}, neg_activations={"b": ok}, eval_fn=lambda _m: {"E0_g": 1.0}
        )
    with pytest.raises(ValueError, match="could not be resolved"):
        execute_verified_contrastive_weight_compilation(
            w,
            pos_activations={"blocks.99.mlp": torch.randn((12, 8), generator=g) * 0.1 + 1.0},
            neg_activations={"blocks.99.mlp": torch.randn((12, 8), generator=g) * 0.1},
            eval_fn=lambda _m: {"E0_g": 1.0},
            min_oof_snr=0.1,
        )
    with pytest.raises(ValueError, match="Parent model evaluation"):
        execute_verified_contrastive_weight_compilation(
            w, pos_activations={"default": ok}, neg_activations={"default": ok}, eval_fn=lambda _m: {"bad": -1.0}
        )

    # Token-ID, guard-chunk, repetition-token, and compatible-model validations
    m_val, b_val = _build_contrastive_moe_model(split_mode="dense", seed=20261271)
    good_pos = b_val["pos_ids"]
    good_neg = b_val["neg_ids"]
    good_gc = b_val["guard_chunks"]
    with pytest.raises(ValueError, match="non-empty 'blocks'"):
        execute_verified_contrastive_weight_compilation(
            w, pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, eval_fn=lambda _m: {"E0_g": 1.0}
        )

    class _NoHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4)])

    with pytest.raises(ValueError, match="wte"):
        execute_verified_contrastive_weight_compilation(
            _NoHead(), pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, eval_fn=lambda _m: {"E0_g": 1.0}
        )
    with pytest.raises(ValueError, match="layer_indices"):
        execute_verified_contrastive_weight_compilation(
            m_val, pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, guard_chunks=good_gc, layer_indices=[99]
        )
    with pytest.raises(ValueError, match="1D integer"):
        execute_verified_contrastive_weight_compilation(
            m_val, pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, guard_chunks=good_gc, repetition_token_ids=torch.zeros((2, 2), dtype=torch.long)
        )
    with pytest.raises(TypeError, match="sequence of ints"):
        execute_verified_contrastive_weight_compilation(
            m_val, pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, guard_chunks=good_gc, repetition_token_ids="12"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must not be empty"):
        execute_verified_contrastive_weight_compilation(
            m_val, pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, guard_chunks=good_gc, repetition_token_ids=[]
        )
    with pytest.raises(ValueError, match="lie in"):
        execute_verified_contrastive_weight_compilation(
            m_val, pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, guard_chunks=good_gc, repetition_token_ids=[9999]
        )
    for bad_gc in (
        [],
        [{"expert_id": "E0"}],
        [{"expert_id": "", "ids": [0, 1]}],
        [{"expert_id": "E0", "ids": [0]}],
        [{"expert_id": "E0", "ids": [0, 9999]}],
        [{"expert_id": "E0", "ids": [0, 1, 2]}, {"expert_id": "E1", "ids": [0, 1]}],
        [{"expert_id": "E1_spec", "ids": [0, 1, 2]}],
    ):
        with pytest.raises(ValueError):
            execute_verified_contrastive_weight_compilation(
                m_val, pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, guard_chunks=bad_gc  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="score_from"):
        execute_verified_contrastive_weight_compilation(
            m_val, pos_chunks_or_ids=good_pos, neg_chunks_or_ids=good_neg, guard_chunks=good_gc, config=ContrastiveWeightCompilerConfig(score_from=99)
        )
    for bad_ids in (
        torch.ones(4, dtype=torch.long),
        torch.ones((2, 4), dtype=torch.float32),
        [],
        [{"bad": [1, 2]}],
        [{"ids": [1]}, {"ids": [1]}],
        [torch.ones(4, dtype=torch.float32)],
        [[True, False]],
        ["bad"],
        [[1, 2, 3], [1, 2]],
        123,
        torch.ones((1, 4), dtype=torch.long),
        torch.tensor([[-1, 2, 3], [1, 2, 3]]),
        torch.tensor([[1, 9999, 3], [1, 2, 3]]),
    ):
        with pytest.raises((TypeError, ValueError)):
            execute_verified_contrastive_weight_compilation(
                m_val, pos_chunks_or_ids=bad_ids, neg_chunks_or_ids=good_neg, guard_chunks=good_gc  # type: ignore[arg-type]
            )
    # Valid Tensor-list batch too + score_from too large for pos_chunks_or_ids when eval_fn bypasses guard_chunks
    with pytest.raises(ValueError, match="score_from"):
        execute_verified_contrastive_weight_compilation(
            m_val,
            pos_chunks_or_ids=[row for row in good_pos[:2]],
            neg_chunks_or_ids=good_neg,
            eval_fn=lambda _m: {"E0_g": 1.0},
            config=ContrastiveWeightCompilerConfig(score_from=99),
        )
    with pytest.raises(TypeError, match="mapping"):
        verify_contrastive_compilation_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_contrastive_compilation_receipt_provenance({"receipt_sha256": "abc"}) is False
    assert (
        verify_contrastive_compilation_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": "wrong", "schema": SCHEMA_VERSION}
        )
        is False
    )
    assert (
        verify_contrastive_compilation_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": RECEIPT_OBJECT, "schema": SCHEMA_VERSION, "bad": float("nan")}
        )
        is False
    )

    # Hook detector check & inplace=False isolation
    hook_handle = w.register_forward_hook(lambda _m, _inp, out: out)
    try:
        assert count_registered_runtime_hooks(w) == 1
    finally:
        hook_handle.remove()
    assert count_registered_runtime_hooks(w) == 0

    m_orig, b_orig = _build_contrastive_moe_model(split_mode="lora_domain_experts", seed=20261270)
    sha_before = compute_module_state_sha256(m_orig)
    m_cloned, rep_cloned = execute_verified_contrastive_weight_compilation(
        m_orig,
        pos_chunks_or_ids=b_orig["pos_ids"],
        neg_chunks_or_ids=b_orig["neg_ids"],
        guard_chunks=b_orig["guard_chunks"],
        layer_indices=[2],
        expert_indices=[1],
        strength_alpha=0.8,
        inplace=False,
    )
    assert m_cloned is not m_orig
    assert rep_cloned["accepted"] is True
    assert compute_module_state_sha256(m_orig) == sha_before

    # Security, ZeroStubAudit (CI-safe try/except), lateral movement, and pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn31", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn31", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/contrastive_steering_weight_compiler.py", TARGET_MODULE_PATH),
        ("tests/test_contrastive_steering_weight_compiler_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 1.12, "steering_snr": 1.85, "anse_energy": -0.42, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1131,
        pr_title="feat(swarm): zero-overhead contrastive steering vector to null-space weight compiler for open-weight MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-31 contrastive steering to AlphaEdit null-space static weight compiler.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): zero-overhead contrastive steering vector to null-space weight compiler for open-weight MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
