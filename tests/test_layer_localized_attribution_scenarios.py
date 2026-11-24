# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-03: causal logit-lens layer-localized & LOEO expert attribution."""

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
from gwaya.ast_audit import ZeroStubAudit
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.layer_localized_attribution import (
    build_surgical_attribution_plan,
    compute_embeddings_batch,
    compute_expert_weight_norms,
    compute_leave_one_expert_out_attribution,
    compute_per_layer_moe_bypass_attribution,
    forward_block_dense_base_mlp,
    per_chunk_causal_midrow_nll,
    select_surgical_layer_mask,
    zero_expert_lora_b_inplace,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "layer_localized_attribution.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_subspace_separated_crowd_moe(
    *,
    vocab_size: int = 32,
    hidden_size: int = 32,
    num_layers: int = 6,
    intermediate_size: int = 64,
    num_experts: int = 4,
    top_k: int = 2,
    seed: int = 20261123,
) -> tuple[CrowdV1WithEngram, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic subspace-separated ``CrowdV1WithEngram`` + ``CrowdLoRAMoEMLP``."""
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
    half_v = vocab_size // 2
    half_h = hidden_size // 2
    half_i = intermediate_size // 2

    u_prose = torch.zeros(half_h, dtype=torch.float32)
    u_prose[0] = 1.0 / math.sqrt(2.0)
    u_prose[1] = -1.0 / math.sqrt(2.0)
    u_spec = torch.zeros(hidden_size - half_h, dtype=torch.float32)
    u_spec[0] = 1.0 / math.sqrt(2.0)
    u_spec[1] = -1.0 / math.sqrt(2.0)

    prose_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    prose_carrier[:half_h] = u_prose
    spec_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    spec_carrier[half_h:] = u_spec

    target_spec_vec = spec_carrier.clone()

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_raw = 0.25 * torch.randn((half_v, half_h), generator=gen, dtype=torch.float32)
        p_raw[:, :2] = 0.0
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True) + 2.2 * u_prose.unsqueeze(0)
        p_raw[5] = 2.6 * u_prose

        s_raw = 0.25 * torch.randn((vocab_size - half_v, hidden_size - half_h), generator=gen, dtype=torch.float32)
        s_raw[:, :2] = 0.0
        s_raw = s_raw - s_raw.mean(dim=1, keepdim=True) + 2.2 * u_spec.unsqueeze(0)
        s_raw[20 - half_v] = 2.6 * u_spec

        wte[:half_v, :half_h] = p_raw
        wte[half_v:, half_h:] = s_raw
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            # Identity-aligned SwiGLU base_mlp: positive channel 0 (E0) and channel half_h (specialist)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:half_h, :half_i].copy_(
                0.03 * torch.randn((half_h, half_i), generator=gen, dtype=torch.float32)
            )
            blk.mlp.w3.weight[half_h:, half_i:].copy_(
                0.03 * torch.randn((hidden_size - half_h, intermediate_size - half_i), generator=gen, dtype=torch.float32)
            )

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 17,
    )
    return model, prose_carrier, spec_carrier, target_spec_vec


def _make_domain_chunks(
    *,
    vocab_size: int = 32,
    seq_len: int = 16,
    score_from: int = 8,
    domains: tuple[str, ...] = ("E0_general_fineweb", "E1_code", "E2_math"),
    chunks_per_domain: int = 8,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Generate deterministic token chunks with general target token 5 and specialist target token 20."""
    gen = torch.Generator(device="cpu").manual_seed(7000 + seed_offset)
    half_v = vocab_size // 2
    chunks: list[dict[str, Any]] = []
    for d_idx, dom in enumerate(domains):
        is_gen = dom.startswith("E0_")
        for row_i in range(chunks_per_domain):
            if is_gen:
                ids = torch.randint(0, half_v, (seq_len,), generator=gen).tolist()
                for t in range(score_from, seq_len):
                    ids[t] = 5
            else:
                ids = torch.randint(half_v, vocab_size, (seq_len,), generator=gen).tolist()
                for t in range(score_from, seq_len):
                    ids[t] = 20
            chunks.append(
                {
                    "expert_id": dom,
                    "dataset_id": dom,
                    "row": seed_offset + 100 * d_idx + row_i,
                    "ids": ids,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# S1: Exact Recovery of Planted Late-Layer Interference & Early Syntax Safety
# ---------------------------------------------------------------------------


def test_s1_exact_recovery_of_planted_late_layer_interference_and_early_syntax_protection() -> None:
    model, prose_vec, spec_vec, target_spec = _build_subspace_separated_crowd_moe(
        num_layers=6, num_experts=4, top_k=2, seed=101
    )
    probe_chunks = _make_domain_chunks(
        seq_len=16, score_from=8, domains=("E0_general_fineweb", "E1_code", "E2_math"), chunks_per_domain=8, seed_offset=0
    )
    eval_chunks = _make_domain_chunks(
        seq_len=16, score_from=8, domains=("E0_general_fineweb", "E1_code", "E2_math"), chunks_per_domain=8, seed_offset=900
    )

    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            mlp.router.weight[1] = 2.0 * spec_vec
            # Layers 2..5 help specialists orthogonally on channel 16 (half_h)
            if l_idx >= 2:
                mlp.experts["1"].w1_A.zero_()
                mlp.experts["1"].w1_B.zero_()
                mlp.experts["1"].w2_A.zero_()
                mlp.experts["1"].w2_B.zero_()
                mlp.experts["1"].w3_A.zero_()
                mlp.experts["1"].w3_A[0, 16] = 0.06
                mlp.experts["1"].w3_B.zero_()
                mlp.experts["1"].w3_B[:, 0] = 0.35 * target_spec

            # Plant realistic-scale E0 prose interference strictly in LATE layers 4 & 5 via channel 0
            if l_idx in (4, 5):
                mlp.router.weight[2] = 2.6 * prose_vec
                mlp.experts["2"].w1_A.zero_()
                mlp.experts["2"].w1_B.zero_()
                mlp.experts["2"].w2_A.zero_()
                mlp.experts["2"].w2_B.zero_()
                mlp.experts["2"].w3_A.zero_()
                mlp.experts["2"].w3_A[0, 0] = 0.04
                mlp.experts["2"].w3_B.zero_()
                mlp.experts["2"].w3_B[:, 0] = -0.35 * prose_vec

    c2_buckets = extract_moe_trainable_buckets(model, num_router_fragments=2)

    attr = compute_per_layer_moe_bypass_attribution(
        model,
        probe_chunks,
        buckets=c2_buckets,
        score_from=8,
        max_general_probe_chunks=12,
        max_spec_probe_chunks=12,
        batch_size=4,
    )
    assert attr["num_layers"] == 6
    assert attr["num_general_chunks"] == 8
    assert attr["num_spec_chunks"] == 12
    assert attr["scored_positions"] == 8

    layers = attr["layers"]
    # Early syntax layers 0 & 1 and mid layers 2 & 3 have near-zero general harm; late layers 4 & 5 dominate
    for early_i in (0, 1):
        assert abs(layers[early_i]["general_harm_nats"]) < 1e-6
        assert layers[early_i]["moe_delta_rms_general"] < 1e-6
    for late_i in (4, 5):
        assert layers[late_i]["general_harm_nats"] > 0.05
        assert layers[late_i]["logit_lens_general_harm_nats"] > 0.01
        assert layers[late_i]["moe_delta_rms_general"] > 0.01
    assert set(attr["ranked_layers_by_general_harm"][:2]) == {4, 5}

    mask_res = select_surgical_layer_mask(
        attr,
        top_k_layers=2,
        min_harm_nats=0.01,
        protect_early_layers_below=2,
    )
    assert mask_res["selected_layers"] == [4, 5]
    assert mask_res["layer_mask"] == [False, False, False, False, True, True]
    assert mask_res["protected_layers"] == [0, 1]
    assert mask_res["harm_coverage_fraction"] > 0.99

    # Even if an adversarial perturbation touches layer 1, protect_early_layers_below=2 shields layers 0 & 1
    perturbed_layers = [dict(row) for row in layers]
    perturbed_layers[1]["general_harm_nats"] = 0.99
    guarded_mask = select_surgical_layer_mask(
        perturbed_layers,
        top_k_layers=2,
        min_harm_nats=0.01,
        protect_early_layers_below=2,
    )
    assert guarded_mask["selected_layers"] == [4, 5]
    assert 0 not in guarded_mask["selected_layers"]
    assert 1 not in guarded_mask["selected_layers"]
    assert guarded_mask["protected_layers"] == [0, 1]

    # Verify surgical edit ONLY on mask_res["selected_layers"] removes E0 harm while keeping layers 0 & 1 bitwise identical
    early_before = {
        (l_idx, name): p.detach().clone()
        for l_idx in (0, 1)
        for name, p in model.blocks[l_idx].named_parameters()
    }
    plan = build_surgical_attribution_plan(
        model,
        probe_chunks,
        buckets=c2_buckets,
        score_from=8,
        top_k_layers=2,
        min_harm_nats=0.01,
        protect_early_layers_below=2,
    )
    assert plan["selected_layers"] == [4, 5]
    assert 2 in plan["harmful_sink_experts"]

    for l_idx in plan["selected_layers"]:
        zero_expert_lora_b_inplace(model.blocks[l_idx].mlp, plan["recommended_sink_experts"])

    for (l_idx, name), orig_t in early_before.items():
        cur_p = dict(model.blocks[l_idx].named_parameters())[name]
        assert torch.equal(cur_p, orig_t)

    # Evaluate E0 and specialist NLL on held-out eval_chunks after surgical edit on layers 4 & 5
    e0_eval_ids = torch.tensor([c["ids"] for c in eval_chunks if c["expert_id"].startswith("E0_")], dtype=torch.long)
    sp_eval_ids = torch.tensor([c["ids"] for c in eval_chunks if not c["expert_id"].startswith("E0_")], dtype=torch.long)
    with torch.no_grad():
        def _eval_ids(ids_eval: torch.Tensor) -> float:
            xb_post = compute_embeddings_batch(model, ids_eval)
            for blk in model.blocks:
                blk_dt = blk.ln1.weight.dtype
                h_att, _ = blk.attn(blk.ln1(xb_post.to(blk_dt)), model.cos_cached, model.sin_cached, past_kv=None, use_cache=False)
                xm = xb_post + h_att
                xb_post = xm + blk.mlp(blk.ln2(xm).to(blk_dt)).to(torch.float32)
            return float(
                per_chunk_causal_midrow_nll(
                    xb_post, model.ln.weight, model.ln.bias, model.ln.eps, model.head.weight, ids_eval, score_from=8
                )
                .mean()
                .item()
            )
        nll_post = _eval_ids(e0_eval_ids)
        spec_nll_post = _eval_ids(sp_eval_ids)
    assert nll_post < attr["full_general_nll"] - 0.1
    assert spec_nll_post <= attr["full_spec_nll"] + 0.02


# ---------------------------------------------------------------------------
# S2: Detection of High-Norm Unscored Rogue Experts (||w1_B|| = 42.5)
# ---------------------------------------------------------------------------


def test_s2_detection_of_high_norm_unscored_rogue_experts() -> None:
    model, prose_vec, spec_vec, target_spec = _build_subspace_separated_crowd_moe(
        num_layers=4, num_experts=8, top_k=2, seed=202
    )
    # Note: E3 and E4 have ZERO chunks in probe_chunks (mirroring C2_final_spec.json)
    probe_chunks = _make_domain_chunks(
        seq_len=16,
        score_from=8,
        domains=("E0_general_fineweb", "E1_code", "E2_math", "E5_exo", "E6_stars", "E7_cyber"),
        chunks_per_domain=4,
        seed_offset=200,
    )

    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            mlp.router.weight[1] = 2.5 * spec_vec
            # Modest normal B norms on scored experts 0, 1, 2, 5, 6, 7 and normal unscored expert 4
            for e_idx in (0, 1, 2, 4, 5, 6, 7):
                mlp.experts[str(e_idx)].w1_B.fill_(0.02 * (1.0 + 0.1 * e_idx))
            mlp.experts["1"].w3_B[:, 0] = 0.25 * target_spec

            # Plant high-norm unscored rogue expert_3 with exact global ||w1_B|| = 42.5 across 4 layers!
            # Wait: 4 layers -> per-layer Frobenius norm = 42.5 / sqrt(4) = 21.25
            per_layer_norm = 42.5 / math.sqrt(4.0)
            w1_shape = mlp.experts["3"].w1_B.shape
            mlp.experts["3"].w1_B.fill_(per_layer_norm / math.sqrt(float(w1_shape[0] * w1_shape[1])))
            # Keep expert_3 unrouted on probe E0 initially (dormant rogue risk!)
            mlp.router.weight[3] = -12.0 * prose_vec - 12.0 * spec_vec
            if l_idx == 3:
                # High-norm SCORED expert_1 (||w1_B|| ~ 15.0) must NOT be misclassified as unscored rogue
                w1_s_shape = mlp.experts["1"].w1_B.shape
                mlp.experts["1"].w1_B.fill_(15.0 / math.sqrt(float(w1_s_shape[0] * w1_s_shape[1])))

    norms = compute_expert_weight_norms(model)
    assert math.isclose(norms[3]["w1_b_norm"], 42.5, rel_tol=1e-5)

    # Sub-case 2A: Dormant rogue expert_3 (delta_e0 == 0.0, unscored E3, ||w1_B|| == 42.5)
    loeo_dormant = compute_leave_one_expert_out_attribution(model, probe_chunks, score_from=8)
    e3_info = loeo_dormant["expert_loeo"]["expert_3"]
    assert math.isclose(e3_info["w1_b_norm"], 42.5, rel_tol=1e-5)
    assert e3_info["own_domain_chunk_count"] == 0
    assert e3_info["is_scored_domain"] is False
    assert e3_info["is_high_norm_outlier"] is True
    assert e3_info["is_rogue_high_norm"] is True
    assert e3_info["classification"] == "rogue_unscored_high_norm"
    assert loeo_dormant["rogue_high_norm_experts"] == [3]
    assert 3 in loeo_dormant["recommended_sink_experts"]

    # Normal-norm unscored expert_4 and high-norm SCORED expert_1 must NOT trigger false rogue detection
    assert 4 not in loeo_dormant["rogue_high_norm_experts"]
    assert loeo_dormant["expert_loeo"]["expert_4"]["is_rogue_high_norm"] is False
    assert 1 not in loeo_dormant["rogue_high_norm_experts"]
    assert loeo_dormant["expert_loeo"]["expert_1"]["is_scored_domain"] is True
    assert loeo_dormant["expert_loeo"]["expert_1"]["is_rogue_high_norm"] is False

    # Sub-case 2B: Active rogue expert_3 that ALSO leaks onto E0 prose
    with torch.no_grad():
        for blk in model.blocks:
            mlp = blk.mlp  # type: ignore[assignment]
            mlp.router.weight[3] = 3.0 * prose_vec
            mlp.experts["3"].w3_A.zero_()
            mlp.experts["3"].w3_A[0, 0] = 0.6
            mlp.experts["3"].w3_B.zero_()
            mlp.experts["3"].w3_B[:, 0] = -1.8 * prose_vec
    loeo_active = compute_leave_one_expert_out_attribution(model, probe_chunks, score_from=8)
    assert 3 in loeo_active["harmful_sink_experts"]
    assert 3 in loeo_active["rogue_high_norm_experts"]
    assert loeo_active["expert_loeo"]["expert_3"]["classification"] == "rogue_unscored_high_norm"

    # Sub-case 2C: Few non-zero experts (< 3 non-zero) still triggers absolute rogue_b_norm_threshold
    with torch.no_grad():
        for blk in model.blocks:
            zero_expert_lora_b_inplace(blk.mlp, [0, 1, 2, 4, 5, 6, 7])
    loeo_sparse = compute_leave_one_expert_out_attribution(model, probe_chunks, score_from=8)
    assert 3 in loeo_sparse["rogue_high_norm_experts"]


# ---------------------------------------------------------------------------
# S3: Regime-A Harmful expert_0 vs Regime-B Helpful expert_0 Classification
# ---------------------------------------------------------------------------


def test_s3_regime_a_harmful_expert0_vs_regime_b_helpful_expert0_classification() -> None:
    model, prose_vec, spec_vec, target_spec = _build_subspace_separated_crowd_moe(
        num_layers=2, num_experts=4, top_k=3, seed=303
    )
    probe_chunks = _make_domain_chunks(
        seq_len=16, score_from=8, domains=("E0_general_fineweb", "E1_code", "E2_math"), chunks_per_domain=6, seed_offset=300
    )

    # Neutral baseline first (all B == 0)
    loeo_neutral = compute_leave_one_expert_out_attribution(model, probe_chunks, score_from=8)
    assert loeo_neutral["expert0_regime"] == "neutral"
    assert loeo_neutral["protect_general_expert"] is False
    assert 0 in loeo_neutral["neutral_experts"]

    # Regime A: expert_0 harms E0; expert_1 helps E1 AND leaks onto E0; expert_2 harms E0 with 0 specialist benefit
    with torch.no_grad():
        for blk in model.blocks:
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            mlp.router.weight[0] = 2.8 * prose_vec
            mlp.router.weight[1] = 2.6 * spec_vec + 2.4 * prose_vec
            mlp.router.weight[2] = 2.6 * prose_vec
            mlp.router.weight[3] = -10.0 * prose_vec - 10.0 * spec_vec

            # expert_0 corrupts E0 token 5 via channel 0 at realistic LoRA residual scale
            mlp.experts["0"].w1_A.zero_()
            mlp.experts["0"].w1_B.zero_()
            mlp.experts["0"].w2_A.zero_()
            mlp.experts["0"].w2_B.zero_()
            mlp.experts["0"].w3_A.zero_()
            mlp.experts["0"].w3_A[0, 0] = 0.04
            mlp.experts["0"].w3_B.zero_()
            mlp.experts["0"].w3_B[:, 0] = -0.35 * prose_vec

            # expert_1 helps E1 token 20 strongly (channel 16) AND injects E0 interference (channel 0)
            mlp.experts["1"].w1_A.zero_()
            mlp.experts["1"].w1_B.zero_()
            mlp.experts["1"].w2_A.zero_()
            mlp.experts["1"].w2_B.zero_()
            mlp.experts["1"].w3_A.zero_()
            mlp.experts["1"].w3_A[0, 16] = 0.06
            mlp.experts["1"].w3_A[1, 0] = 0.04
            mlp.experts["1"].w3_B.zero_()
            mlp.experts["1"].w3_B[:, 0] = 0.40 * target_spec
            mlp.experts["1"].w3_B[:, 1] = -0.30 * prose_vec

            # expert_2 is purely off-target on E0
            mlp.experts["2"].w1_A.zero_()
            mlp.experts["2"].w1_B.zero_()
            mlp.experts["2"].w2_A.zero_()
            mlp.experts["2"].w2_B.zero_()
            mlp.experts["2"].w3_A.zero_()
            mlp.experts["2"].w3_A[0, 0] = 0.04
            mlp.experts["2"].w3_B.zero_()
            mlp.experts["2"].w3_B[:, 0] = -0.35 * prose_vec

    # Snapshot weights to verify LOEO leaves model bitwise intact
    snapshot_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    loeo_a = compute_leave_one_expert_out_attribution(model, probe_chunks, score_from=8)
    assert loeo_a["expert0_regime"] == "regime_a_harmful"
    assert loeo_a["protect_general_expert"] is False
    assert 0 in loeo_a["harmful_sink_experts"]
    assert 2 in loeo_a["harmful_sink_experts"]
    assert loeo_a["expert_loeo"]["expert_0"]["classification"] == "harmful_off_target"
    assert loeo_a["expert_loeo"]["expert_2"]["classification"] == "harmful_off_target"
    assert 1 in loeo_a["interfering_specialist_experts"]
    assert 1 not in loeo_a["harmful_sink_experts"]
    assert loeo_a["expert_loeo"]["expert_1"]["classification"] == "specialist_with_general_interference"

    for k, before_t in snapshot_before.items():
        assert torch.equal(model.state_dict()[k], before_t)

    # Regime B: expert_0 genuinely HELPS E0; expert_1 cleanly helps specialists; expert_2 harms E0
    with torch.no_grad():
        for blk in model.blocks:
            mlp = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            mlp.router.weight[0] = 2.8 * prose_vec
            mlp.router.weight[1] = 2.8 * spec_vec
            mlp.router.weight[2] = 2.6 * prose_vec

            # Flip expert_0 w3_B to +0.25 * prose_vec -> helps E0 prediction of token 5!
            mlp.experts["0"].w3_B[:, 0] = 0.25 * prose_vec

            # Clean specialist expert_1 (zero E0 leakage)
            mlp.experts["1"].w1_A[1] = 0.0
            mlp.experts["1"].w1_B[:, 1] = 0.0
            mlp.experts["1"].w3_A[1] = 0.0
            mlp.experts["1"].w3_B[:, 1] = 0.0

    loeo_b = compute_leave_one_expert_out_attribution(model, probe_chunks, score_from=8)
    assert loeo_b["expert0_regime"] == "regime_b_helpful"
    assert loeo_b["protect_general_expert"] is True
    assert 0 in loeo_b["helpful_general_experts"]
    assert 0 not in loeo_b["harmful_sink_experts"]
    assert loeo_b["expert_loeo"]["expert_0"]["delta_e0_when_zeroed"] > 0.01
    assert loeo_b["expert_loeo"]["expert_0"]["classification"] == "helpful_general"
    assert 1 in loeo_b["helpful_specialist_experts"]
    assert 2 in loeo_b["harmful_sink_experts"]


# ---------------------------------------------------------------------------
# S4: Causal Mid-Row Target Indexing (score_from - 1 .. T - 2 -> score_from .. T - 1)
# ---------------------------------------------------------------------------


def test_s4_causal_midrow_target_indexing_exact_boundary() -> None:
    gen = torch.Generator(device="cpu").manual_seed(404)
    n_chunks, seq_len, hidden_dim, vocab = 3, 20, 12, 17
    h_pre = torch.randn((n_chunks, seq_len, hidden_dim), generator=gen, dtype=torch.float32)
    ln_w = 0.8 + 0.2 * torch.rand(hidden_dim, generator=gen, dtype=torch.float32)
    ln_b = 0.1 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    head_w = torch.randn((vocab, hidden_dim), generator=gen, dtype=torch.float32) / math.sqrt(hidden_dim)
    ids = torch.randint(0, vocab, (n_chunks, seq_len), generator=gen, dtype=torch.long)

    for score_from in (0, 1, 8, 12):
        nlls = per_chunk_causal_midrow_nll(
            h_pre, ln_w, ln_b, 1e-5, head_w, ids, score_from=score_from, seq_batch_size=2
        )
        start_pos = max(0, score_from - 1)
        expected = torch.zeros(n_chunks, dtype=torch.float32)
        h_norm = F.layer_norm(h_pre, (hidden_dim,), ln_w, ln_b, eps=1e-5)
        full_logits = F.linear(h_norm, head_w)
        count = 0
        for t in range(start_pos, seq_len - 1):
            # Position t predicts target token t + 1
            expected += F.cross_entropy(full_logits[:, t, :], ids[:, t + 1], reduction="none")
            count += 1
        expected /= float(count)
        assert torch.allclose(nlls, expected, atol=1e-6, rtol=1e-6)

    # Strict boundary isolation test at score_from=10 (positions 9..18 predicting targets 10..19)
    sf = 10
    ref_nll = per_chunk_causal_midrow_nll(h_pre, ln_w, ln_b, 1e-5, head_w, ids, score_from=sf)

    # 1) Mutating prefix target tokens 0..9 MUST NOT alter NLL (bitwise equal)
    ids_mut_prefix = ids.clone()
    ids_mut_prefix[:, :sf] = (ids_mut_prefix[:, :sf] + 1) % vocab
    assert torch.equal(per_chunk_causal_midrow_nll(h_pre, ln_w, ln_b, 1e-5, head_w, ids_mut_prefix, score_from=sf), ref_nll)

    # 2) Mutating hidden states strictly before score_from-1 (0..8) AND final pos T-1 (19) MUST NOT alter NLL
    h_mut_outside = h_pre.clone()
    h_mut_outside[:, : sf - 1, :] += 5.0
    h_mut_outside[:, -1, :] -= 5.0
    assert torch.equal(per_chunk_causal_midrow_nll(h_mut_outside, ln_w, ln_b, 1e-5, head_w, ids, score_from=sf), ref_nll)

    # 3) Mutating exact first scored position (score_from - 1 = 9), last scored position (T - 2 = 18),
    #    first target (score_from = 10), or last target (T - 1 = 19) MUST change NLL
    for mut_h_pos in (sf - 1, seq_len - 2):
        h_mut = h_pre.clone()
        h_mut[:, mut_h_pos, 0] += 2.0
        assert not torch.equal(per_chunk_causal_midrow_nll(h_mut, ln_w, ln_b, 1e-5, head_w, ids, score_from=sf), ref_nll)
    for mut_t_pos in (sf, seq_len - 1):
        ids_mut = ids.clone()
        ids_mut[:, mut_t_pos] = (ids_mut[:, mut_t_pos] + 1) % vocab
        assert not torch.equal(per_chunk_causal_midrow_nll(h_pre, ln_w, ln_b, 1e-5, head_w, ids_mut, score_from=sf), ref_nll)

    # 4) End-to-end CrowdV1WithEngram.forward masked-label parity with compute_per_layer_moe_bypass_attribution
    e2e_model, _, _, _ = _build_subspace_separated_crowd_moe(num_layers=2, num_experts=4, top_k=2, seed=414)
    e2e_chunks = _make_domain_chunks(seq_len=16, score_from=8, domains=("E0_general_fineweb", "E1_code"), chunks_per_domain=4)
    e2e_attr = compute_per_layer_moe_bypass_attribution(e2e_model, e2e_chunks, score_from=8)
    e0_ids = torch.tensor([c["ids"] for c in e2e_chunks if c["expert_id"].startswith("E0_")], dtype=torch.long)
    masked_labels = e0_ids.clone()
    masked_labels[:, :8] = -100
    with torch.no_grad():
        _, forward_loss, _ = e2e_model(e0_ids, labels=masked_labels)
    assert forward_loss is not None
    assert abs(float(forward_loss.item()) - e2e_attr["full_general_nll"]) < 1e-5


# ---------------------------------------------------------------------------
# S5: Edge / Validation Branches & Security / Anti-Stub / Lateral-Movement Gates
# ---------------------------------------------------------------------------


def test_s5_edge_validation_branches_and_security_gates() -> None:
    model, _, _, _ = _build_subspace_separated_crowd_moe(num_layers=2, num_experts=4, top_k=2, seed=505)
    chunks = _make_domain_chunks(seq_len=12, score_from=6, domains=("E0_general_fineweb", "E1_code"), chunks_per_domain=3)
    buckets = extract_moe_trainable_buckets(model, num_router_fragments=2)

    # 1. Engram-enabled model + un-upcycled dense layer + generic base_mlp fallback
    engram_model = CrowdV1WithEngram(
        CrowdV1RuntimeConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            intermediate_size=16,
            max_position_embeddings=32,
            engram_buckets=32,
        )
    )
    emb = compute_embeddings_batch(engram_model, torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert emb.shape == (1, 3, 8)
    # Only upcycle layer 1 so layer 0 remains dense CrowdRuntimeMLP (covers un-upcycled layer path)
    upcycle_crowd_v1_to_sparse_moe(engram_model, num_experts=2, top_k=1, layer_indices=[1], seed=11)
    engram_chunks = [
        {"expert_id": "E0_gen", "ids": [0, 1, 2, 3, 4, 5, 6, 7]},
        {"expert_id": "E0_gen", "ids": [1, 2, 3, 4, 5, 6, 7, 8]},
    ]
    # General-only chunks (0 specialist chunks) -> num_spec_chunks == 0, full_spec_nll is None
    attr_gen_only = compute_per_layer_moe_bypass_attribution(engram_model, engram_chunks, score_from=4)
    assert attr_gen_only["num_spec_chunks"] == 0
    assert attr_gen_only["full_spec_nll"] is None
    assert attr_gen_only["layers"][0]["spec_benefit_nats"] is None
    # Also test generic base_mlp module without w1/w2/w3
    generic_blk = nn.Module()
    generic_blk.ln1 = nn.LayerNorm(8)
    generic_blk.mlp = nn.Module()
    generic_blk.mlp.base_mlp = nn.Linear(8, 8, bias=False)
    gen_out = forward_block_dense_base_mlp(generic_blk, torch.ones((2, 4, 8)))
    assert gen_out.shape == (2, 4, 8)
    bad_blk = nn.Module()
    bad_blk.ln1 = nn.LayerNorm(8)
    bad_blk.mlp = nn.Module()
    with pytest.raises(ValueError, match="base_mlp"):
        forward_block_dense_base_mlp(bad_blk, torch.ones((2, 4, 8)))

    # Non-E<digit> specialist domain names + explicit scored_expert_indices coverage
    custom_dom_chunks = [
        {"expert_id": "E0_gen", "ids": [0, 1, 2, 3, 4, 5, 6, 7]},
        {"expert_id": "custom_math", "ids": [8, 9, 10, 11, 12, 13, 14, 15]},
    ]
    loeo_custom = compute_leave_one_expert_out_attribution(engram_model, custom_dom_chunks, score_from=4)
    assert loeo_custom["expert_loeo"]["expert_1"]["is_scored_domain"] is True
    loeo_explicit = compute_leave_one_expert_out_attribution(
        engram_model, custom_dom_chunks, score_from=4, scored_expert_indices=[0]
    )
    assert loeo_explicit["expert_loeo"]["expert_1"]["is_scored_domain"] is False

    # 2. Custom load_buckets_fn + explicit cal_ln_w / cal_ln_b
    loeo_loaded = compute_leave_one_expert_out_attribution(
        model,
        chunks,
        buckets=buckets,
        load_buckets_fn=lambda m, b: load_moe_trainable_buckets(m, b, num_router_fragments=2),
        cal_ln_w=model.ln.weight.detach().clone(),
        cal_ln_b=model.ln.bias.detach().clone(),
        score_from=6,
    )
    assert loeo_loaded["eval_chunk_count"] == len(chunks)

    # 3. Validation error branches
    with pytest.raises(TypeError, match="nn.Module"):
        compute_expert_weight_norms("not_a_model")  # type: ignore[arg-type]
    dense_only = CrowdV1WithEngram(
        CrowdV1RuntimeConfig(
            vocab_size=16, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=4, intermediate_size=16, max_position_embeddings=16, engram_buckets=0,
        )
    )
    with pytest.raises(ValueError, match="experts"):
        compute_expert_weight_norms(dense_only)
    empty_blk_model = nn.Module()
    empty_blk_model.blocks = nn.ModuleList()
    with pytest.raises(ValueError, match="non-empty 'blocks'"):
        compute_expert_weight_norms(empty_blk_model)
    missing_attr_model = nn.Module()
    missing_attr_model.blocks = nn.ModuleList([nn.Linear(4, 4)])
    with pytest.raises(ValueError, match="must expose"):
        compute_expert_weight_norms(missing_attr_model)

    with pytest.raises(ValueError, match="both be provided"):
        compute_leave_one_expert_out_attribution(model, chunks, cal_ln_w=model.ln.weight.detach())
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_leave_one_expert_out_attribution(model, chunks, cal_ln_w="bad", cal_ln_b="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="LN params"):
        compute_leave_one_expert_out_attribution(model, chunks, cal_ln_w=torch.ones(3), cal_ln_b=torch.ones(3))
    with pytest.raises(ValueError, match="finite values"):
        bad_ln = torch.ones_like(model.ln.weight)
        bad_ln[0] = float("nan")
        compute_leave_one_expert_out_attribution(model, chunks, cal_ln_w=bad_ln, cal_ln_b=torch.zeros_like(bad_ln))

    orig_eps = model.ln.eps
    model.ln.eps = 0.0
    try:
        with pytest.raises(ValueError, match="ln_eps"):
            compute_leave_one_expert_out_attribution(model, chunks, score_from=6)
    finally:
        model.ln.eps = orig_eps

    with pytest.raises(ValueError, match="general_prefix"):
        compute_leave_one_expert_out_attribution(model, chunks, general_prefix="  ")
    with pytest.raises(ValueError, match="non-empty sequence"):
        compute_leave_one_expert_out_attribution(model, [])
    with pytest.raises(ValueError, match="mapping with 'expert_id'"):
        compute_leave_one_expert_out_attribution(model, [{"expert_id": "E0_a"}])
    with pytest.raises(ValueError, match="length >= 2"):
        compute_leave_one_expert_out_attribution(model, [{"expert_id": "E0_a", "ids": [1]}])
    with pytest.raises(ValueError, match="non-negative token IDs"):
        compute_leave_one_expert_out_attribution(model, [{"expert_id": "E0_a", "ids": [0, -1, 2]}])
    with pytest.raises(ValueError, match="sequence length"):
        compute_leave_one_expert_out_attribution(
            model, [{"expert_id": "E0_a", "ids": [0, 1, 2]}, {"expert_id": "E0_a", "ids": [0, 1, 2, 3]}]
        )
    with pytest.raises(ValueError, match="general chunk"):
        compute_leave_one_expert_out_attribution(model, [{"expert_id": "E1_code", "ids": [0, 1, 2, 3]}])
    with pytest.raises(TypeError, match="buckets must be a mapping"):
        compute_leave_one_expert_out_attribution(model, chunks, buckets="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="buckets must be a mapping"):
        compute_per_layer_moe_bypass_attribution(model, chunks, buckets="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="loeo_e0_harm_threshold"):
        compute_leave_one_expert_out_attribution(model, chunks, loeo_e0_harm_threshold=0.1)
    with pytest.raises(TypeError, match="must be an int"):
        compute_leave_one_expert_out_attribution(model, chunks, score_from=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        compute_leave_one_expert_out_attribution(model, chunks, score_from=-1)
    with pytest.raises(ValueError, match=">= 1"):
        compute_leave_one_expert_out_attribution(model, chunks, batch_size=0)
    with pytest.raises(TypeError, match="real number"):
        compute_leave_one_expert_out_attribution(model, chunks, rogue_b_norm_threshold=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and >= 0.0"):
        compute_leave_one_expert_out_attribution(model, chunks, rogue_b_norm_threshold=-1.0)
    with pytest.raises(TypeError, match="real number"):
        compute_leave_one_expert_out_attribution(model, chunks, loeo_e0_harm_threshold="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        compute_leave_one_expert_out_attribution(model, chunks, loeo_e0_harm_threshold=float("-inf"))
    with pytest.raises(ValueError, match="too short"):
        compute_per_layer_moe_bypass_attribution(model, chunks, score_from=12)

    # per_chunk_causal_midrow_nll and zero_expert_lora_b_inplace validation
    ok_h = torch.ones((2, 6, 8))
    ok_w = torch.ones(8)
    ok_b = torch.zeros(8)
    ok_hw = torch.ones((16, 8))
    ok_ids = torch.ones((2, 6), dtype=torch.long)
    with pytest.raises(TypeError, match="torch.Tensor"):
        per_chunk_causal_midrow_nll("bad", ok_w, ok_b, 1e-5, ok_hw, ok_ids)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="3D h_pre_ln"):
        per_chunk_causal_midrow_nll(torch.ones((2, 8)), ok_w, ok_b, 1e-5, ok_hw, ok_ids)
    with pytest.raises(ValueError, match="Mismatched"):
        per_chunk_causal_midrow_nll(ok_h, ok_w, ok_b, 1e-5, ok_hw, torch.ones((2, 5), dtype=torch.long))
    bad_h = ok_h.clone()
    bad_h[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite values"):
        per_chunk_causal_midrow_nll(bad_h, ok_w, ok_b, 1e-5, ok_hw, ok_ids)
    with pytest.raises(ValueError, match="LN params"):
        per_chunk_causal_midrow_nll(ok_h, torch.ones(4), ok_b, 1e-5, ok_hw, ok_ids)
    with pytest.raises(ValueError, match="head_weight"):
        per_chunk_causal_midrow_nll(ok_h, ok_w, ok_b, 1e-5, torch.ones((16, 4)), ok_ids)
    with pytest.raises(ValueError, match="token IDs"):
        bad_ids = ok_ids.clone()
        bad_ids[0, 0] = 99
        per_chunk_causal_midrow_nll(ok_h, ok_w, ok_b, 1e-5, ok_hw, bad_ids)
    with pytest.raises(ValueError, match="ln_eps"):
        per_chunk_causal_midrow_nll(ok_h, ok_w, ok_b, 0.0, ok_hw, ok_ids)
    with pytest.raises(ValueError, match="too short"):
        per_chunk_causal_midrow_nll(ok_h, ok_w, ok_b, 1e-5, ok_hw, ok_ids, score_from=6)

    with pytest.raises(TypeError, match="nn.Module"):
        zero_expert_lora_b_inplace("bad", [0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="experts"):
        zero_expert_lora_b_inplace(nn.Linear(4, 4), [0])
    assert zero_expert_lora_b_inplace(model.blocks[0].mlp, [99]) == 0

    # select_surgical_layer_mask edge cases + fallback e0_harm_nats + top_k=0/None + all-clean
    clean_layers = [{"layer": 0, "e0_harm_nats": -0.02}, {"layer": 1, "e0_harm_nats": 0.0}]
    clean_mask = select_surgical_layer_mask(clean_layers, top_k_layers=None, min_harm_nats=0.0, protect_early_layers_below=0)
    assert clean_mask["selected_layers"] == []
    assert clean_mask["harm_coverage_fraction"] == 0.0
    pos_layers = [{"layer": 0, "general_harm_nats": 0.1}, {"layer": 1, "general_harm_nats": 0.2}]
    zero_k_mask = select_surgical_layer_mask(pos_layers, top_k_layers=0)
    assert zero_k_mask["selected_layers"] == []

    with pytest.raises(ValueError, match="non-empty 'layers'"):
        select_surgical_layer_mask({"layers": []})
    with pytest.raises(ValueError, match="must be non-empty"):
        select_surgical_layer_mask([])
    with pytest.raises(TypeError, match="mapping or sequence"):
        select_surgical_layer_mask("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="containing 'layer'"):
        select_surgical_layer_mask([{"general_harm_nats": 0.1}])
    with pytest.raises(ValueError, match="Duplicate layer"):
        select_surgical_layer_mask([{"layer": 0, "general_harm_nats": 0.1}, {"layer": 0, "general_harm_nats": 0.2}])
    with pytest.raises(ValueError, match="general_harm_nats"):
        select_surgical_layer_mask([{"layer": 0}])
    with pytest.raises(ValueError, match="exceed max layer"):
        select_surgical_layer_mask({"num_layers": 1, "layers": [{"layer": 1, "general_harm_nats": 0.1}]})

    # 4. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement, and pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path("test_no_lateral_movement_ctrl03", REPO_ROOT / "tests" / "test_no_lateral_movement.py")
    gatekeeper = _load_module_from_path("pr_gatekeeper_ctrl03", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/layer_localized_attribution.py", TARGET_MODULE_PATH),
        ("tests/test_layer_localized_attribution_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

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
        changed_files.append({"filename": rel, "status": "added", "additions": len(src_text.splitlines()), "patch": "", "content": src_text})

    telemetry_json = json.dumps({"general_harm_nats": 0.123, "w1_b_norm": 42.5, "selected_layers": [4, 5]})
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}") == []

    pr_audit = scanner.audit_pull_request(
        pr_number=1103,
        pr_title="feat(swarm): causal logit-lens layer-localized and LOEO expert attribution for open-weight MoEs",
        pr_body="Implements NEURAL-CTRL-03 causal logit-lens layer-localized and LOEO expert attribution.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(swarm): causal logit-lens layer-localized and LOEO expert attribution for open-weight MoEs"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
