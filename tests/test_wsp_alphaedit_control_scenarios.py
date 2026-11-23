# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-01: WSP-AlphaEdit null-space filter, FoMoE ghost sink, and LN guard."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from gwaya.ast_audit import ZeroStubAudit
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.wsp_alphaedit_control import (
    ANSE_SAFE_E0_RATIO,
    DOMAIN_GUARD_MAX_RATIO,
    LN_GUARD_MAX_RATIO,
    WSPArmConfig,
    apply_layer_wsp_edit_inplace,
    calibrate_final_layernorm_and_sink_experts,
    compute_discriminative_alphaedit_filter,
    compute_embeddings_batch,
    compute_swiglu_intermediate_states,
    evaluate_wsp_guard_and_retention,
    fit_uncentered_prose_probe_with_oof,
    forward_model_pre_ln,
    interpolate_weight_buckets,
    per_chunk_nll_from_pre_ln,
    second_moment_and_mean,
    select_guard_optimal_lambda_ln,
    synthesize_wsp_arm_buckets,
    zero_expert_lora_b_inplace,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "wsp_alphaedit_control.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_orthogonal_basis(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((dim, dim), generator=gen, dtype=torch.float64)
    q_mat, r_mat = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r_mat))
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    return q_mat * signs.unsqueeze(0)


def _build_subspace_separated_crowd_model(
    *,
    vocab_size: int = 32,
    hidden_size: int = 32,
    num_layers: int = 2,
    intermediate_size: int = 64,
    seed: int = 20261123,
) -> CrowdV1WithEngram:
    """Build a tiny deterministic CrowdV1WithEngram with block-diagonal centered embeddings.

    Tokens ``0..vocab_size//2 - 1`` live exclusively in dims ``0..hidden_size//2 - 1`` (general prose),
    while tokens ``vocab_size//2..vocab_size - 1`` live exclusively in dims ``hidden_size//2..hidden_size - 1``
    (specialist domain), with zero row-mean per half so LayerNorm keeps both subspaces decoupled.
    """
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

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        u_prose = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(half_h)], dtype=torch.float32)
        u_prose = u_prose / u_prose.norm()
        u_spec = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(hidden_size - half_h)], dtype=torch.float32)
        u_spec = u_spec / u_spec.norm()
        p_raw = torch.randn((half_v, half_h), generator=gen, dtype=torch.float32)
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True)
        # Ensure all E0 embeddings share a clean zero-mean uncentered prose bias along u_prose
        p_raw = p_raw / p_raw.norm(dim=1, keepdim=True).clamp_min(1e-6) * 2.0 + 1.4 * u_prose.unsqueeze(0)
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True)
        s_raw = torch.randn((vocab_size - half_v, hidden_size - half_h), generator=gen, dtype=torch.float32)
        s_raw = s_raw - s_raw.mean(dim=1, keepdim=True)
        s_raw = s_raw / s_raw.norm(dim=1, keepdim=True).clamp_min(1e-6) * 2.0 + 1.4 * u_spec.unsqueeze(0)
        s_raw = s_raw - s_raw.mean(dim=1, keepdim=True)
        wte[:half_v, :half_h] = p_raw
        wte[half_v:, half_h:] = s_raw
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        half_i = intermediate_size // 2
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.q.weight[:half_h, :half_h].copy_(0.05 * torch.eye(half_h))
            blk.attn.k.weight[: half_h // 2, :half_h].copy_(0.05 * torch.eye(half_h // 2, half_h))
            blk.attn.v.weight[: half_h // 2, :half_h].copy_(0.05 * torch.eye(half_h // 2, half_h))
            blk.attn.o.weight[:half_h, :half_h].copy_(0.05 * torch.eye(half_h))
            # Block-diagonal SwiGLU base_mlp preserving subspace separation with active hidden units
            w1_p = torch.randn((half_i, half_h), generator=gen, dtype=torch.float32) * 0.20
            w2_p = torch.randn((half_i, half_h), generator=gen, dtype=torch.float32) * 0.20
            w3_p = torch.randn((half_h, half_i), generator=gen, dtype=torch.float32) * 0.12
            w3_p = w3_p - w3_p.mean(dim=0, keepdim=True)
            w1_s = torch.randn((intermediate_size - half_i, hidden_size - half_h), generator=gen, dtype=torch.float32) * 0.25
            w2_s = torch.randn((intermediate_size - half_i, hidden_size - half_h), generator=gen, dtype=torch.float32) * 0.25
            w3_s = torch.randn((hidden_size - half_h, intermediate_size - half_i), generator=gen, dtype=torch.float32) * 0.12
            w3_s = w3_s - w3_s.mean(dim=0, keepdim=True)
            blk.mlp.w1.weight[:half_i, :half_h].copy_(w1_p)
            blk.mlp.w2.weight[:half_i, :half_h].copy_(w2_p)
            blk.mlp.w3.weight[:half_h, :half_i].copy_(w3_p)
            blk.mlp.w1.weight[half_i:, half_h:].copy_(w1_s)
            blk.mlp.w2.weight[half_i:, half_h:].copy_(w2_s)
            blk.mlp.w3.weight[half_h:, half_i:].copy_(w3_s)

    return model


# ---------------------------------------------------------------------------
# S1: Exact Generalized Eigenmode Diagonalization & Minimum Specialist Distortion
# ---------------------------------------------------------------------------


def test_s1_generalized_eigenmode_diagonalization_and_zero_specialist_distortion() -> None:
    """Verify exact float64 generalized eigenmode shrinkage on prose modes and <1e-6 specialist distortion."""
    d = 12
    k_prose = 3
    q0 = _make_orthogonal_basis(d, seed=20260101)
    gen = torch.Generator(device="cpu").manual_seed(20260102)

    # Construct orthogonal prose (first 4 modes, 4th below tau) and specialist (last 8 modes) data
    n_samples = 256
    z_p = torch.randn((n_samples, d), generator=gen, dtype=torch.float64)
    z_p = z_p - z_p.mean(dim=0, keepdim=True)
    # Orthonormalize sample columns so empirical second moment has exact target eigenvalues
    u_p, _ = torch.linalg.qr(z_p, mode="reduced")
    p_scales = torch.zeros(d, dtype=torch.float64)
    p_scales[:k_prose] = torch.tensor([6.0, 4.0, 2.5], dtype=torch.float64)
    p_scales[k_prose] = 0.02  # sub-threshold prose mode (mu <= tau)
    x_prose = (u_p * math.sqrt(n_samples)) @ torch.diag(p_scales.sqrt()) @ q0.T

    z_s = torch.randn((n_samples, d), generator=gen, dtype=torch.float64)
    u_s, _ = torch.linalg.qr(z_s, mode="reduced")
    s_scales = torch.zeros(d, dtype=torch.float64)
    s_scales[k_prose + 1 :] = torch.linspace(0.6, 2.2, d - (k_prose + 1), dtype=torch.float64)
    x_spec = (u_s * math.sqrt(n_samples)) @ torch.diag(s_scales.sqrt()) @ q0.T

    m_prose, mean_prose = second_moment_and_mean(x_prose)
    m_spec, mean_spec = second_moment_and_mean(x_spec)
    assert m_prose.dtype == torch.float64 and m_prose.device.type == "cpu"
    assert m_spec.dtype == torch.float64 and mean_prose.shape == (d,)
    assert torch.allclose(m_prose, (x_prose.T @ x_prose) / n_samples, atol=1e-12)
    assert torch.allclose(mean_spec, x_spec.mean(dim=0), atol=1e-12)

    eta = 3.0
    tau = 1.5
    reg_spec = 0.10
    g_star, stats = compute_discriminative_alphaedit_filter(
        m_prose,
        m_spec,
        eta=eta,
        tau=tau,
        max_rank=8,
        reg_spec=reg_spec,
    )
    assert g_star.dtype == torch.float64
    assert stats["active_modes"] == k_prose
    assert stats["variance_ratio"] == pytest.approx(stats["max_ratio"], rel=1e-12)
    assert stats["min_selected_ratio"] > tau

    # Compute exact theoretical lam_spec and generalized eigenvalues mu_i for orthogonal modes
    lam_spec = max(reg_spec * float(torch.diagonal(m_spec).mean().item()), 1e-6)
    for i in range(k_prose):
        mu_i = float(p_scales[i].item()) / lam_spec
        expected_scale = 1.0 / (1.0 + eta * (mu_i - tau))
        vec = q0[:, i]
        out = g_star @ vec
        assert torch.allclose(out, expected_scale * vec, atol=1e-10)

    # Sub-threshold prose mode (index k_prose) and all specialist modes incur <1e-6 distortion
    w_a = torch.randn((4, d), generator=gen, dtype=torch.float64)
    w_a_filt = w_a @ g_star
    spec_delta = (w_a_filt - w_a) @ x_spec.T
    spec_ref = w_a @ x_spec.T
    rel_dist = float(spec_delta.norm().item() / spec_ref.norm().item())
    assert rel_dist < 1e-6

    sub_thresh_vec = q0[:, k_prose]
    assert torch.allclose(g_star @ sub_thresh_vec, sub_thresh_vec, atol=1e-12)

    # Part B: Non-orthogonal coupled SPD generalized eigenproblem
    f_p = torch.randn((d, d), generator=gen, dtype=torch.float64)
    f_s = torch.randn((d, d), generator=gen, dtype=torch.float64)
    m_p_gen = (f_p @ f_p.T) / d
    m_s_gen = (f_s @ f_s.T) / d
    g_gen, st_gen = compute_discriminative_alphaedit_filter(
        m_p_gen,
        m_s_gen,
        eta=2.5,
        tau=1.1,
        max_rank=4,
        reg_spec=0.10,
    )
    lam_gen = max(0.10 * float(torch.diagonal(m_s_gen).mean().item()), 1e-6)
    b_mat = m_s_gen + lam_gen * torch.eye(d, dtype=torch.float64)
    ev_b, vec_b = torch.linalg.eigh(b_mat)
    b_half = (vec_b * ev_b.sqrt().unsqueeze(0)) @ vec_b.T
    b_inv_half = (vec_b * ev_b.rsqrt().unsqueeze(0)) @ vec_b.T
    mu_asc, q_asc = torch.linalg.eigh(0.5 * (b_inv_half @ m_p_gen @ b_inv_half + (b_inv_half @ m_p_gen @ b_inv_half).T))
    mu_desc = mu_asc.flip(0)
    q_desc = q_asc.flip(1)
    k_gen = st_gen["active_modes"]
    assert 0 < k_gen <= 4
    for idx in range(d):
        dual_mode = b_half @ q_desc[:, idx]
        if idx < k_gen:
            expected_factor = 1.0 / (1.0 + 2.5 * (float(mu_desc[idx].item()) - 1.1))
        else:
            expected_factor = 1.0
        assert torch.allclose(g_gen @ dual_mode, expected_factor * dual_mode, atol=1e-10)

    # Global KKT identity at tau=0, max_rank=d: G^* (B + eta M_prose) == B
    g_full, _ = compute_discriminative_alphaedit_filter(
        m_p_gen,
        m_s_gen,
        eta=2.5,
        tau=0.0,
        max_rank=d,
        reg_spec=0.10,
    )
    assert torch.allclose(g_full @ (b_mat + 2.5 * m_p_gen), b_mat, atol=1e-10)


# ---------------------------------------------------------------------------
# S2: Bitwise Zero-Delta FoMoE Ghost-Sink Verification on SwiGLU LoRA Experts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("routing_mode", ["sigma_topk", "softmax_topk"])
def test_s2_bitwise_zero_delta_fomoe_ghost_sink_on_swiglu_lora(routing_mode: str) -> None:
    """Verify w1_B = w2_B = w3_B = 0 yields bitwise exact dy == 0 and CrowdLoRAMoEMLP == base_mlp."""
    gen = torch.Generator(device="cpu").manual_seed(20260201)
    cfg = CrowdV1RuntimeConfig(hidden_size=16, intermediate_size=32)
    dense = CrowdRuntimeMLP(cfg)
    with torch.no_grad():
        for p in dense.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.2)

    moe = CrowdLoRAMoEMLP(
        dense,
        num_experts=4,
        top_k=2,
        rank=4,
        alpha=8.0,
        routing_mode=routing_mode,
        resident_expert_indices=(0, 1, 2, 3),
        seed=20260202,
    )
    with torch.no_grad():
        moe.router.weight.copy_(torch.randn(moe.router.weight.shape, generator=gen, dtype=torch.float32) * 0.5)
        for e_key in moe.experts:
            exp = moe.experts[e_key]
            for attr in ("w1_B", "w2_B", "w3_B"):
                getattr(exp, attr).copy_(torch.randn(getattr(exp, attr).shape, generator=gen, dtype=torch.float32) * 0.3)

    x = torch.randn((3, 6, 16), generator=gen, dtype=torch.float32)
    flat_x = x.reshape(-1, 16)
    with torch.no_grad():
        y_dense = dense(x)
        y_before = moe(x)
        assert float((y_before - y_dense).abs().max().item()) > 1e-3

        # Zero experts 0 and 2 (including duplicate index 2 and non-resident index 99)
        zeroed_count = zero_expert_lora_b_inplace(moe, [0, 2, 2, 99])
        per_expert_b_elems = 32 * 4 + 32 * 4 + 16 * 4
        assert zeroed_count == 2 * per_expert_b_elems

        h1 = dense.w1(flat_x)
        h2 = dense.w2(flat_x)
        for ghost_idx in ("0", "2"):
            exp = moe.experts[ghost_idx]
            assert torch.equal(exp.w1_B, torch.zeros_like(exp.w1_B))
            assert torch.equal(exp.w2_B, torch.zeros_like(exp.w2_B))
            assert torch.equal(exp.w3_B, torch.zeros_like(exp.w3_B))
            dy_ghost = exp.forward_delta(flat_x, h1, h2, dense.w3.weight)
            assert torch.equal(dy_ghost, torch.zeros_like(dy_ghost))

        # Surviving experts still have non-zero delta, and intermediate states helper handles both modes + empty list
        dy_surv = moe.experts["1"].forward_delta(flat_x, h1, h2, dense.w3.weight)
        assert float(dy_surv.abs().max().item()) > 1e-4

        act_empty = compute_swiglu_intermediate_states(moe, flat_x, [])
        act_active = compute_swiglu_intermediate_states(moe, flat_x, [0, 1, 2, 3, 99])
        assert act_empty.shape == (18, 32)
        assert act_active.shape == (18, 32)
        assert float((act_active - act_empty).abs().max().item()) > 1e-4

        # Zeroing remaining surviving experts restores bitwise equality with dense base_mlp
        zero_expert_lora_b_inplace(moe, [1, 3])
        y_all_zero = moe(x)
        assert torch.equal(y_all_zero, y_dense)


# ---------------------------------------------------------------------------
# S3: Final LayerNorm Drift Recovery vs Minimum-Drift Fallback When Unfeasible
# ---------------------------------------------------------------------------


def test_s3_layernorm_guard_recovery_and_unfeasible_fallback() -> None:
    """Verify lambda_ln selects minimum NLL under <= +0.5% E0 guard and falls back to min-E0-drift when unfeasible."""
    gen = torch.Generator(device="cpu").manual_seed(20260301)
    n_e0, n_sp = 6, 6
    n_total = n_e0 + n_sp
    seq_len = 10
    h_dim = 16
    vocab = 20

    # Construct centered head weights: E0 uses dims 0..7, specialist uses dims 8..15
    head_w = torch.zeros((vocab, h_dim), dtype=torch.float32)
    h0 = torch.randn((10, 8), generator=gen, dtype=torch.float32)
    head_w[:10, :8] = h0 - h0.mean(dim=1, keepdim=True)
    h1 = torch.randn((10, 8), generator=gen, dtype=torch.float32)
    head_w[10:, 8:] = h1 - h1.mean(dim=1, keepdim=True)

    ids_e0 = torch.randint(0, 10, (n_e0, seq_len), generator=gen, dtype=torch.long)
    ids_sp = torch.randint(10, 20, (n_sp, seq_len), generator=gen, dtype=torch.long)
    ids_all = torch.cat([ids_e0, ids_sp], dim=0)
    is_e0 = np.array([True] * n_e0 + [False] * n_sp, dtype=bool)

    # Pre-LN hidden states aligned with next-token embedding at every position t -> target ids[:, t+1]
    h_pre = torch.zeros((n_total, seq_len, h_dim), dtype=torch.float32)
    for b in range(n_total):
        for t in range(seq_len - 1):
            tgt = int(ids_all[b, t + 1].item())
            h_pre[b, t] = 0.6 * head_w[tgt]

    start_ln_w = torch.ones(h_dim, dtype=torch.float32)
    start_ln_b = torch.zeros(h_dim, dtype=torch.float32)
    base_losses = per_chunk_nll_from_pre_ln(
        h_pre,
        start_ln_w,
        start_ln_b,
        1e-5,
        head_w,
        ids_all,
        score_from=2,
    ).numpy()

    # Tuned LN slightly boosts E0 scale (+15%) while adding a bias drift vector in E0 subspace, plus strong specialist gain.
    # At lambda=0.50, the scale boost balances bias drift (E0 stays <= +0.5% and spec drops); at lambda>=0.75, bias drift
    # dominates and pushes E0 well past +0.5%.
    drift_vec = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
    c2_ln_w = torch.cat([torch.full((8,), 1.15), torch.full((8,), 5.0)])
    c2_ln_b = torch.cat([0.55 * drift_vec, torch.zeros(8, dtype=torch.float32)])

    best_lam, grid = select_guard_optimal_lambda_ln(
        h_pre,
        start_ln_w,
        start_ln_b,
        c2_ln_w,
        c2_ln_b,
        1e-5,
        head_w,
        ids_all,
        base_losses,
        is_e0,
        score_from=2,
        target_e0_max_ratio=LN_GUARD_MAX_RATIO,
    )
    assert best_lam == pytest.approx(0.50)
    assert grid["0.50"]["e0_ratio"] <= LN_GUARD_MAX_RATIO
    assert grid["0.50"]["nll_mean"] < grid["0.00"]["nll_mean"]
    assert grid["1.00"]["e0_ratio"] > LN_GUARD_MAX_RATIO

    # Fallback mode: make even lambda=0.0 violate a tight guard (e.g. target_e0_max_ratio=0.50 on shifted base_losses)
    # Use c2_ln_w that strictly worsens E0 with increasing lambda so lambda=0.0 has minimum e0_ratio.
    c2_harm_w = torch.cat([torch.full((8,), -1.5), torch.full((8,), 1.0)])
    fallback_lam, fb_grid = select_guard_optimal_lambda_ln(
        h_pre,
        start_ln_w,
        start_ln_b,
        c2_harm_w,
        c2_ln_b,
        1e-5,
        head_w,
        ids_all,
        base_losses * 0.5,  # base artificially halved so all candidates have e0_ratio ~ 2.0 > 1.005
        is_e0,
        score_from=2,
        target_e0_max_ratio=LN_GUARD_MAX_RATIO,
    )
    min_ratio_key = min(fb_grid, key=lambda k: fb_grid[k]["e0_ratio"])
    assert fallback_lam == pytest.approx(float(min_ratio_key))
    assert fb_grid[min_ratio_key]["e0_ratio"] > LN_GUARD_MAX_RATIO


# ---------------------------------------------------------------------------
# S4: End-to-End Multi-Layer Synthetic Sparse MoE: WSP-AlphaEdit vs Uniform WiSE-FT
# ---------------------------------------------------------------------------


def test_s4_multilayer_sparse_moe_wsp_alphaedit_beats_wise_ft() -> None:
    """Prove WSP-AlphaEdit clears <= +1.0% E0 guard with >80% specialist gain retention where WiSE-FT loses >80%."""
    model = _build_subspace_separated_crowd_model(
        vocab_size=32,
        hidden_size=32,
        num_layers=2,
        intermediate_size=64,
        seed=20260401,
    )
    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=4,
        top_k=2,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=20260402,
    )
    start_buckets = {
        k: v.detach().cpu().float().clone()
        for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()
    }

    # Build deterministic disjoint Probe and Evaluation splits (16 E0 + 16 E1 chunks each)
    seq_len = 14
    score_from = 2

    def _make_split(n_per_domain: int, prose_shift: int, prefix_len: int, is_eval: bool) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for idx in range(n_per_domain):
            # Disjoint predictable general prose sequences spanning all 16 tokens in [0..15]
            t_a = idx % 16
            t_b = (idx + prose_shift) % 16
            p_seq = [t_a] * prefix_len + [t_b] * (seq_len - prefix_len)
            chunks.append({"expert_id": "E0_general_fineweb", "ids": p_seq, "row": idx + prose_shift * 100})
        for idx in range(n_per_domain):
            # Specialist tokens follow a deterministic cyclic rule mod 16 shifted by +16 with disjoint prefix
            start_tok = idx % 16
            if is_eval:
                s_seq = [16 + start_tok] + [16 + ((start_tok + t) % 16) for t in range(seq_len - 1)]
            else:
                s_seq = [16 + ((start_tok + t) % 16) for t in range(seq_len)]
            chunks.append({"expert_id": "E1_python_systems_code", "ids": s_seq, "row": idx + prefix_len * 100})
        return chunks

    probe_chunks = _make_split(16, prose_shift=3, prefix_len=7, is_eval=False)
    eval_chunks = _make_split(16, prose_shift=7, prefix_len=5, is_eval=True)
    assert {(c["expert_id"], tuple(c["ids"])) for c in probe_chunks}.isdisjoint(
        {(c["expert_id"], tuple(c["ids"])) for c in eval_chunks}
    )

    # Synthesize C2 checkpoint:
    # 1) Train LoRA experts 1, 2, 3 on specialist subspace [16..31] so C2 achieves huge specialist gain
    load_moe_trainable_buckets(model, {k: v.clone() for k, v in start_buckets.items()}, num_router_fragments=2)
    spec_train_ids = torch.tensor([c["ids"] for c in probe_chunks if c["expert_id"].startswith("E1_")], dtype=torch.long)
    u_prose = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(16)], dtype=torch.float32)
    u_prose = u_prose / u_prose.norm()
    u_spec = u_prose.clone()
    trainable_params = []
    with torch.no_grad():
        for blk in model.blocks:
            # Restrict base_A of experts 1..3 to specialist half [16:32] first so gradient training learns clean S patterns
            for e_id in ("1", "2", "3"):
                exp = blk.mlp.experts[e_id]
                exp.w1_A[:, :16].zero_()
                exp.w2_A[:, :16].zero_()
                exp.w3_A[:, :32].zero_()
                trainable_params.extend([exp.w1_B, exp.w2_B, exp.w3_B])
            # Route specialist tokens to experts 1, 2, 3 along u_spec and suppress expert 0 on S
            blk.mlp.router.weight[1:4, 16:32].copy_(1.5 * u_spec.unsqueeze(0).expand(3, -1))
            blk.mlp.router.weight[0, 16:32].copy_(-6.0 * u_spec)
    opt = torch.optim.SGD(trainable_params, lr=0.45)
    model.train()
    for _ in range(35):
        opt.zero_grad()
        _, loss_t, _ = model(spec_train_ids, labels=spec_train_ids)
        assert loss_t is not None
        loss_t.backward()
        opt.step()
    model.eval()

    # 2) Inject realistic E0 forgetting into C2:
    #    (a) Expert 0 acts as an E0-harmful ghost sink with zero specialist utility
    #    (b) Experts 1..3 pick up prose-subspace interference in w1_A[:, :16], w2_A[:, :16], w3_A[:, :32] and router[:, :16]
    #    (c) Final LN weight has mild drift on prose channels
    gen_noise = torch.Generator(device="cpu").manual_seed(20260404)
    with torch.no_grad():
        for blk in model.blocks:
            # Router aligns with uncentered prose direction u_prose so E0 tokens route into experts 0, 1, 2, 3
            blk.mlp.router.weight[0, :16].copy_(3.0 * u_prose)
            for e_id in (1, 2, 3):
                blk.mlp.router.weight[e_id, :16].copy_(2.2 * u_prose)
            # Harmful sink expert 0: reads prose dims 0..15 and pushes E0 logits away from targets
            exp0 = blk.mlp.experts["0"]
            exp0.w1_A.zero_()
            exp0.w2_A.zero_()
            exp0.w3_A.zero_()
            exp0.w1_A[:, :16].copy_(u_prose.unsqueeze(0).expand(4, -1) * 1.5)
            exp0.w2_A[:, :16].copy_(u_prose.unsqueeze(0).expand(4, -1) * 1.5)
            exp0.w1_B.fill_(0.8)
            exp0.w2_B.fill_(0.8)
            exp0.w3_A[:, :32].fill_(0.5)
            exp0.w3_B[:16, :].copy_(-1.5 * u_prose.unsqueeze(1).expand(-1, 4))
            exp0.w3_B[16:, :].zero_()

            # Surviving specialist experts 1..3 also have strong prose interference in w1_A[:, :16], w2_A[:, :16]
            for e_id in ("1", "2", "3"):
                exp = blk.mlp.experts[e_id]
                exp.w1_A[:, :16].copy_(torch.randn((4, 16), generator=gen_noise) * 0.9)
                exp.w2_A[:, :16].copy_(torch.randn((4, 16), generator=gen_noise) * 0.9)
                exp.w3_A[:, :32].copy_(torch.randn((4, 32), generator=gen_noise) * 0.8)
        model.ln.weight[:16].fill_(1.3)
        model.ln.bias[:16].copy_(0.3 * u_prose)

    c2_buckets = {
        k: v.detach().cpu().float().clone()
        for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()
    }

    def _eval_buckets(cand_buckets: dict[str, torch.Tensor], chunks: list[dict[str, Any]]) -> np.ndarray:
        load_moe_trainable_buckets(model, cand_buckets, num_router_fragments=2)
        model.eval()
        ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
        hp = forward_model_pre_ln(model, ids, batch_size=8)
        return (
            per_chunk_nll_from_pre_ln(
                hp,
                model.ln.weight,
                model.ln.bias,
                float(model.ln.eps),
                model.head.weight,
                ids,
                score_from=score_from,
            )
            .cpu()
            .numpy()
        )

    eval_is_e0 = np.array([c["expert_id"].startswith("E0_") for c in eval_chunks], dtype=bool)
    base_eval_losses = _eval_buckets(start_buckets, eval_chunks)
    c2_eval_losses = _eval_buckets(c2_buckets, eval_chunks)
    c2_metrics = evaluate_wsp_guard_and_retention(base_eval_losses, c2_eval_losses, eval_is_e0, tuned_losses=c2_eval_losses)
    assert c2_metrics["guard_ok"] is False
    assert c2_metrics["e0_pct_vs_base"] > 5.0
    assert c2_metrics["spec_gain_nats"] > 0.5
    assert c2_metrics["anse_energy"] > 1e5

    # Calibrate on Probe split and synthesize WSP-AlphaEdit buckets
    calibration = calibrate_final_layernorm_and_sink_experts(
        model,
        start_buckets,
        c2_buckets,
        probe_chunks,
        score_from=score_from,
        batch_size=8,
        loeo_chunks_per_domain=4,
    )
    assert 0 in calibration["harmful_sink_experts"]

    # Also exercise the non-sequential branch (sink + LN only) cleanly
    arm_sink_only = WSPArmConfig(
        name="WSP_sink_only",
        use_ln_calibration=True,
        use_ghost_sink=True,
        force_zero_expert0=True,
    )
    sink_buckets, sink_meta = synthesize_wsp_arm_buckets(
        model,
        start_buckets,
        c2_buckets,
        probe_chunks,
        arm_sink_only,
        calibration,
        score_from=score_from,
        batch_size=8,
    )
    assert 0 in sink_meta["sink_experts_applied"]
    assert len(sink_buckets) == len(c2_buckets)

    # Full WSP-AlphaEdit arm (Ghost sink + contrastive sigma_topk router surgery + dual-space AlphaEdit + LN guard)
    arm_wsp = WSPArmConfig(
        name="WSP_AlphaEdit_Hybrid",
        use_ln_calibration=True,
        use_ghost_sink=True,
        force_zero_expert0=False,
        router_beta_suppress=2.5,
        router_beta_sink=1.5,
        router_omega_spec=6.0,
        router_ridge_scale=0.75,
        alphaedit_eta_x=25.0,
        alphaedit_eta_act=12.0,
        alphaedit_tau=1.20,
        alphaedit_max_rank_x=32,
        alphaedit_max_rank_act=64,
        alphaedit_reg_spec=0.10,
    )
    wsp_buckets, wsp_meta = synthesize_wsp_arm_buckets(
        model,
        start_buckets,
        c2_buckets,
        probe_chunks,
        arm_wsp,
        calibration,
        score_from=score_from,
        batch_size=8,
    )
    assert len(wsp_meta["layers"]) == 2
    assert all(ly["alphaedit_x"]["active_modes"] > 0 for ly in wsp_meta["layers"])

    wsp_eval_losses = _eval_buckets(wsp_buckets, eval_chunks)
    wsp_metrics = evaluate_wsp_guard_and_retention(
        base_eval_losses,
        wsp_eval_losses,
        eval_is_e0,
        tuned_losses=c2_eval_losses,
    )
    assert wsp_metrics["guard_ok"] is True
    assert wsp_metrics["e0_ratio"] <= DOMAIN_GUARD_MAX_RATIO
    assert wsp_metrics["spec_gain_retained_frac"] > 0.80
    assert wsp_metrics["anse_energy"] < 0.0

    # Compare against uniform WiSE-FT sweep across alpha in {0.00, 0.02, 0.05, ..., 1.00}
    wise_alphas = [round(0.02 * i, 2) for i in range(51)]
    guard_passing_wise: list[dict[str, Any]] = []
    for alpha in wise_alphas:
        wb = interpolate_weight_buckets(start_buckets, c2_buckets, alpha)
        w_losses = _eval_buckets(wb, eval_chunks)
        wm = evaluate_wsp_guard_and_retention(
            base_eval_losses,
            w_losses,
            eval_is_e0,
            tuned_losses=c2_eval_losses,
        )
        if wm["guard_ok"]:
            guard_passing_wise.append({"alpha": alpha, **wm})

    assert guard_passing_wise, "WiSE-FT at alpha=0.0 must pass guard"
    best_wise_retained = max(item["spec_gain_retained_frac"] for item in guard_passing_wise)
    # Uniform WiSE-FT loses >80% of specialist gain (retains <20%) to clear the <= +1.0% general guard!
    assert best_wise_retained < 0.20
    assert wsp_metrics["spec_gain_retained_frac"] > 0.80

    # Confirm anse_safe_e0_ratio soft penalty branch works cleanly
    soft_pen = evaluate_wsp_guard_and_retention(
        np.array([1.0, 2.0]),
        np.array([ANSE_SAFE_E0_RATIO + 0.0005, 1.0]),
        np.array([True, False]),
        tuned_losses=np.array([1.2, 1.0]),
    )
    assert soft_pen["guard_ok"] is True
    assert soft_pen["anse_energy"] < 1.0


# ---------------------------------------------------------------------------
# S5: Input Validation, Edge Cases, PRSecurityScanner & ZeroStubAudit Cleanliness
# ---------------------------------------------------------------------------


def test_s5_input_validation_edge_cases_and_security_audits() -> None:
    """Validate defensive edge cases and enforce PRSecurityScanner (0 findings) + ZeroStubAudit cleanliness."""
    # 1. second_moment_and_mean edge cases
    with pytest.raises(TypeError, match="torch.Tensor"):
        second_moment_and_mean([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D tensor"):
        second_moment_and_mean(torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-empty"):
        second_moment_and_mean(torch.zeros((0, 4)))
    with pytest.raises(ValueError, match="finite"):
        second_moment_and_mean(torch.tensor([[1.0, float("nan")]]))

    # 2. compute_discriminative_alphaedit_filter edge cases (eta=0, max_rank=0, high tau, invalid inputs)
    m_id = torch.eye(4, dtype=torch.float64)
    g_eta0, st_eta0 = compute_discriminative_alphaedit_filter(m_id, m_id, eta=0.0)
    assert torch.equal(g_eta0, m_id)
    assert st_eta0["active_modes"] == 0 and st_eta0["mean_shrink"] == 0.0

    g_r0, st_r0 = compute_discriminative_alphaedit_filter(m_id, m_id, eta=2.0, max_rank=0)
    assert torch.equal(g_r0, m_id)
    assert st_r0["active_modes"] == 0

    g_tau_hi, st_tau_hi = compute_discriminative_alphaedit_filter(m_id, m_id, eta=2.0, tau=100.0)
    assert torch.equal(g_tau_hi, m_id)
    assert st_tau_hi["active_modes"] == 0

    with pytest.raises(ValueError, match="square"):
        compute_discriminative_alphaedit_filter(torch.ones((3, 4)), torch.ones((3, 4)), eta=1.0)
    with pytest.raises(ValueError, match=">="):
        compute_discriminative_alphaedit_filter(m_id, m_id, eta=-1.0)
    with pytest.raises(TypeError, match="int"):
        compute_discriminative_alphaedit_filter(m_id, m_id, eta=1.0, max_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">="):
        compute_discriminative_alphaedit_filter(m_id, m_id, eta=1.0, max_rank=-2)
    with pytest.raises(TypeError, match="real number"):
        compute_discriminative_alphaedit_filter(m_id, m_id, eta=False)  # type: ignore[arg-type]

    # 3. WSPArmConfig and OOF probe edge cases
    with pytest.raises(ValueError, match="non-empty string"):
        WSPArmConfig(name="  ")
    gen = torch.Generator(device="cpu").manual_seed(20260501)
    p0 = torch.randn((4, 6), generator=gen)
    p1 = torch.randn((4, 6), generator=gen)
    s0 = torch.randn((4, 6), generator=gen)
    s1 = torch.randn((4, 6), generator=gen)
    # Equal opposite mean so p_mean <= 1e-6 or small SNR hits both rel_weight branches
    u_zero, stats_zero = fit_uncentered_prose_probe_with_oof(
        torch.cat([p0, -p0], dim=0),
        torch.cat([p1, -p1], dim=0),
        s0,
        s1,
    )
    assert u_zero.shape == (6,)
    assert stats_zero["rel_weight"] == 0.0
    with pytest.raises(ValueError, match="feature dimension"):
        fit_uncentered_prose_probe_with_oof(p0, p1, s0, torch.ones((4, 5)))

    # 4. zero_expert_lora_b_inplace and compute_swiglu_intermediate_states edge cases
    with pytest.raises(TypeError, match="nn.Module"):
        zero_expert_lora_b_inplace("not_a_module", [0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="experts"):
        zero_expert_lora_b_inplace(nn.Linear(4, 4), [0])
    with pytest.raises(ValueError, match=">="):
        zero_expert_lora_b_inplace(CrowdLoRAMoEMLP(CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=8, intermediate_size=16))), [-1])
    with pytest.raises(TypeError, match="nn.Module"):
        compute_swiglu_intermediate_states("not_a_module", p0, [0])  # type: ignore[arg-type]
    moe_small = CrowdLoRAMoEMLP(CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=8, intermediate_size=16)))
    with pytest.raises(ValueError, match="feature width"):
        compute_swiglu_intermediate_states(moe_small, p0, [0])

    # 5. LayerNorm guard, pre-LN NLL, bucket interpolation, and synthesis validation
    h_pre = torch.ones((2, 6, 8))
    ids = torch.zeros((2, 6), dtype=torch.long)
    w_ln = torch.ones(8)
    b_ln = torch.zeros(8)
    head_w = torch.ones((10, 8))
    with pytest.raises(ValueError, match="3D"):
        per_chunk_nll_from_pre_ln(torch.ones((2, 8)), w_ln, b_ln, 1e-5, head_w, ids, score_from=1)
    with pytest.raises(ValueError, match="Mismatched"):
        per_chunk_nll_from_pre_ln(h_pre, w_ln, b_ln, 1e-5, head_w, torch.zeros((3, 6), dtype=torch.long), score_from=1)
    with pytest.raises(ValueError, match="LN params"):
        per_chunk_nll_from_pre_ln(h_pre, torch.ones(4), b_ln, 1e-5, head_w, ids, score_from=1)
    with pytest.raises(ValueError, match="head_weight"):
        per_chunk_nll_from_pre_ln(h_pre, w_ln, b_ln, 1e-5, torch.ones((10, 4)), ids, score_from=1)
    with pytest.raises(ValueError, match="ln_eps"):
        per_chunk_nll_from_pre_ln(h_pre, w_ln, b_ln, 0.0, head_w, ids, score_from=1)
    with pytest.raises(ValueError, match="too short"):
        per_chunk_nll_from_pre_ln(h_pre, w_ln, b_ln, 1e-5, head_w, ids, score_from=10)

    with pytest.raises(ValueError, match="base_losses"):
        select_guard_optimal_lambda_ln(h_pre, w_ln, b_ln, w_ln, b_ln, 1e-5, head_w, ids, [1.0], [True, False], score_from=1)
    with pytest.raises(ValueError, match="is_e0"):
        select_guard_optimal_lambda_ln(h_pre, w_ln, b_ln, w_ln, b_ln, 1e-5, head_w, ids, [1.0, 1.0], [False, False], score_from=1)
    with pytest.raises(ValueError, match="shapes must match"):
        select_guard_optimal_lambda_ln(h_pre, w_ln, b_ln, torch.ones(4), b_ln, 1e-5, head_w, ids, [1.0, 1.0], [True, False], score_from=1)
    with pytest.raises(ValueError, match="target_e0_max_ratio"):
        select_guard_optimal_lambda_ln(h_pre, w_ln, b_ln, w_ln, b_ln, 1e-5, head_w, ids, [1.0, 1.0], [True, False], score_from=1, target_e0_max_ratio=0.0)
    with pytest.raises(ValueError, match="lambda_grid"):
        select_guard_optimal_lambda_ln(h_pre, w_ln, b_ln, w_ln, b_ln, 1e-5, head_w, ids, [1.0, 1.0], [True, False], score_from=1, lambda_grid=[])
    with pytest.raises(ValueError, match="finite"):
        select_guard_optimal_lambda_ln(h_pre, w_ln, b_ln, w_ln, b_ln, 1e-5, head_w, ids, [1.0, 1.0], [True, False], score_from=1, lambda_grid=[float("nan")])
    # Torch tensor base_losses & is_e0 support
    lam_t, _ = select_guard_optimal_lambda_ln(
        h_pre, w_ln, b_ln, w_ln, b_ln, 1e-5, head_w, ids, torch.tensor([2.5, 2.5]), torch.tensor([True, True]), score_from=1
    )
    assert lam_t == 0.0

    with pytest.raises(ValueError, match="alpha"):
        interpolate_weight_buckets({"a": torch.ones(2)}, {"a": torch.ones(2)}, 1.5)
    with pytest.raises(ValueError, match="identical keys"):
        interpolate_weight_buckets({"a": torch.ones(2)}, {"b": torch.ones(2)}, 0.5)
    with pytest.raises(ValueError, match="Shape mismatch"):
        interpolate_weight_buckets({"a": torch.ones(2)}, {"a": torch.ones(3)}, 0.5)
    with pytest.raises(ValueError, match="1D shape"):
        evaluate_wsp_guard_and_retention([1.0], [1.0, 2.0], [True])
    with pytest.raises(ValueError, match="both general-domain"):
        evaluate_wsp_guard_and_retention([1.0, 2.0], [1.0, 2.0], [True, True])
    with pytest.raises(ValueError, match="tuned_losses"):
        evaluate_wsp_guard_and_retention([1.0, 2.0], [1.0, 2.0], [True, False], tuned_losses=[1.0])

    small_model = _build_subspace_separated_crowd_model(vocab_size=16, hidden_size=16, num_layers=1, intermediate_size=32)
    upcycle_crowd_v1_to_sparse_moe(small_model, num_experts=2, top_k=1, lora_rank=2)
    sb = extract_moe_trainable_buckets(small_model, num_router_fragments=1)
    # Exercise apply_layer_wsp_edit_inplace with force_zero_expert0=True and compute_embeddings_batch with engram_buckets>0
    layer_cfg = WSPArmConfig(name="force_e0_layer", use_ghost_sink=True, force_zero_expert0=True, alphaedit_eta_x=1.0)
    layer_res = apply_layer_wsp_edit_inplace(
        small_model.blocks[0].mlp,
        torch.randn((4, 16), generator=gen),
        torch.randn((4, 16), generator=gen),
        torch.randn((4, 16), generator=gen),
        torch.randn((4, 16), generator=gen),
        layer_cfg,
    )
    assert layer_res["sink_experts"] == [0]
    engram_model = CrowdV1WithEngram(
        CrowdV1RuntimeConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            intermediate_size=16,
            max_position_embeddings=16,
            engram_buckets=32,
        )
    )
    emb_out = compute_embeddings_batch(engram_model, torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert emb_out.shape == (1, 3, 8)
    with pytest.raises(ValueError, match="non-empty"):
        forward_model_pre_ln(small_model, torch.zeros((0, 6), dtype=torch.long))
    with pytest.raises(ValueError, match="non-empty"):
        calibrate_final_layernorm_and_sink_experts(small_model, sb, sb, [], score_from=1)
    with pytest.raises(ValueError, match="router_frag_"):
        calibrate_final_layernorm_and_sink_experts(small_model, {"expert_0": torch.ones(4)}, sb, [{"expert_id": "E0_a", "ids": [0, 1, 2]}], score_from=1)
    with pytest.raises(ValueError, match="too short"):
        calibrate_final_layernorm_and_sink_experts(small_model, {"router_frag_0": torch.ones(4)}, sb, [{"expert_id": "E0_a", "ids": [0, 1, 2]}], score_from=1)
    with pytest.raises(ValueError, match="at least 2 general"):
        synthesize_wsp_arm_buckets(
            small_model,
            sb,
            sb,
            [{"expert_id": "E0_a", "ids": [0, 1, 2, 3]}, {"expert_id": "E1_b", "ids": [8, 9, 10, 11]}],
            WSPArmConfig(name="too_few", alphaedit_eta_x=1.0),
            {"best_lambda_ln": 0.0, "harmful_sink_experts": []},
            score_from=1,
        )

    # 6. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement, and pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path("test_no_lateral_movement_ctrl01", REPO_ROOT / "tests" / "test_no_lateral_movement.py")
    gatekeeper = _load_module_from_path("pr_gatekeeper_ctrl01", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/wsp_alphaedit_control.py", TARGET_MODULE_PATH),
        ("tests/test_wsp_alphaedit_control_scenarios.py", TEST_FILE_PATH),
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

    # Verify serialized metric telemetry contains zero forbidden score+digit payloads
    telemetry_json = json.dumps({"nll_mean": 1.23, "variance_ratio": 4.56, "anse_energy": -0.42})
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}") == []

    pr_audit = scanner.audit_pull_request(
        pr_number=1101,
        pr_title="feat(swarm): discriminative AlphaEdit null-space weight filter, FoMoE ghost sink, and LayerNorm guard calibration",
        pr_body="Implements NEURAL-CTRL-01 discriminative AlphaEdit null-space weight filter and scenario suite.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(swarm): discriminative AlphaEdit null-space weight filter, FoMoE ghost sink, and LayerNorm guard calibration"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
