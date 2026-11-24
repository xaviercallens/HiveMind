# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-06: TIES/DARE sign-consensus & prose-nullspace MoE merger."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.ties_dare_moe_merger import (
    ANSE_MAXIMUM_PAIN,
    SCHEMA_VERSION,
    TIESDAREMergerConfig,
    apply_prose_projector_to_lora_pair,
    compute_buckets_sha256,
    compute_prose_nullspace_projector,
    dare_rescale_and_svd_compress_lora,
    merge_moe_checkpoints_with_prose_guard,
    merge_swiglu_lora_adapters_dare_svd,
    protect_router_frag_layernorm_tail,
    trim_and_elect_sign_ties,
)
from hivemind.swarm.wsp_alphaedit_control import (
    DOMAIN_GUARD_MAX_RATIO,
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
    ZeroStubAudit = None  # type: ignore[assignment,misc]

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "ties_dare_moe_merger.py"
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


def _make_orthogonal_basis(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((dim, dim), generator=gen, dtype=torch.float64)
    q_mat, r_mat = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r_mat))
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    return (q_mat * signs.unsqueeze(0)).contiguous()


def _build_5domain_crowd_moe_model(
    *,
    num_layers: int = 2,
    rank: int = 4,
    num_router_fragments: int = 2,
    seed: int = 20260601,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor], torch.Tensor]:
    """Build a 5-domain subspace-decoupled CrowdV1WithEngram + CrowdLoRAMoEMLP model.

    Domains occupy disjoint 8-dim zero-row-mean blocks of hidden_size=40:
      - E0_general_fineweb: tokens 0..7   -> dims 0..7   (expert 0)
      - E1_code:            tokens 8..15  -> dims 8..15  (expert 1)
      - E2_math:            tokens 16..23 -> dims 16..23 (expert 2)
      - E5_exoplanet:       tokens 24..31 -> dims 24..31 (expert 3)
      - E6_stellar:         tokens 32..39 -> dims 32..39 (expert 4)
    """
    vocab_size = 40
    hidden_size = 40
    intermediate_size = 80
    block_h = 8
    block_i = 16
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
    domain_dirs = torch.zeros((5, hidden_size), dtype=torch.float32)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for d_idx in range(5):
            h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
            t0, t1 = d_idx * 8, (d_idx + 1) * 8
            u_dom = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(block_h)], dtype=torch.float32)
            u_dom = u_dom / u_dom.norm()
            domain_dirs[d_idx, h0:h1] = u_dom

            raw = torch.randn((8, block_h), generator=gen, dtype=torch.float32)
            raw = raw - raw.mean(dim=1, keepdim=True)
            raw = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-6) * 1.8 + 1.2 * u_dom.unsqueeze(0)
            raw = raw - raw.mean(dim=1, keepdim=True)
            wte[t0:t1, h0:h1] = raw

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            # Gentle self-attention on prose subspace so repeated tokens yield low baseline E0 NLL
            blk.attn.q.weight[:block_h, :block_h].copy_(0.05 * torch.eye(block_h))
            blk.attn.k.weight[: block_h // 2, :block_h].copy_(0.05 * torch.eye(block_h // 2, block_h))
            blk.attn.v.weight[: block_h // 2, :block_h].copy_(0.05 * torch.eye(block_h // 2, block_h))
            blk.attn.o.weight[:block_h, :block_h].copy_(0.05 * torch.eye(block_h))
            for d_idx in range(5):
                h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
                i0, i1 = d_idx * block_i, (d_idx + 1) * block_i
                w1_b = torch.randn((block_i, block_h), generator=gen, dtype=torch.float32) * 0.12
                w2_b = torch.randn((block_i, block_h), generator=gen, dtype=torch.float32) * 0.12
                w3_b = torch.randn((block_h, block_i), generator=gen, dtype=torch.float32) * 0.08
                w3_b = w3_b - w3_b.mean(dim=0, keepdim=True)
                blk.mlp.w1.weight[i0:i1, h0:h1].copy_(w1_b)
                blk.mlp.w2.weight[i0:i1, h0:h1].copy_(w2_b)
                blk.mlp.w3.weight[h0:h1, i0:i1].copy_(w3_b)

        upcycle_crowd_v1_to_sparse_moe(
            model,
            num_experts=5,
            top_k=2,
            lora_rank=rank,
            lora_alpha=float(2 * rank),
            routing_mode="sigma_topk",
            seed=seed + 50,
        )
        # Initialize clean subspace-aligned router weights & zero-perturbation experts
        for blk in model.blocks:
            blk.mlp.router.weight.copy_(domain_dirs * 2.5 - 1.5)
            for e_idx in range(5):
                exp = blk.mlp.experts[str(e_idx)]
                exp.w1_A.zero_()
                exp.w2_A.zero_()
                exp.w3_A.zero_()
                exp.w1_B.zero_()
                exp.w2_B.zero_()
                exp.w3_B.zero_()
                h0, h1 = e_idx * block_h, (e_idx + 1) * block_h
                i0, i1 = e_idx * block_i, (e_idx + 1) * block_i
                exp.w1_A[:, h0:h1].copy_(torch.randn((rank, block_h), generator=gen) * 0.35)
                exp.w2_A[:, h0:h1].copy_(torch.randn((rank, block_h), generator=gen) * 0.35)
                exp.w3_A[:, i0:i1].copy_(torch.randn((rank, block_i), generator=gen) * 0.35)

    parent_buckets = {
        k: v.detach().cpu().float().clone()
        for k, v in extract_moe_trainable_buckets(model, num_router_fragments=num_router_fragments).items()
    }
    return model, parent_buckets, domain_dirs


def _make_domain_chunks(
    *,
    chunks_per_domain: int = 8,
    seq_len: int = 20,
    seed: int = 20260602,
) -> list[dict[str, Any]]:
    """Generate deterministic token sequences per domain with predictable next-token transitions."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    domain_specs = (
        ("E0_general_fineweb", 0),
        ("E1_python_systems_code", 8),
        ("E2_math_reasoning", 16),
        ("E5_exoplanet_transit", 24),
        ("E6_stellar_spectroscopy", 32),
    )
    chunks: list[dict[str, Any]] = []
    for dom_name, tok_base in domain_specs:
        for row_idx in range(chunks_per_domain):
            seq: list[int] = []
            offset = (row_idx + int(torch.randint(0, 8, (1,), generator=gen).item())) % 8
            if dom_name.startswith("E0_"):
                # Smooth repeatable prose patterns -> low baseline E0 NLL << ln(vocab)
                t_a = tok_base + (row_idx % 8)
                t_b = tok_base + ((row_idx + 3) % 8)
                half = seq_len // 2
                seq = [t_a] * half + [t_b] * (seq_len - half)
            else:
                for pos in range(seq_len):
                    # Periodic deterministic motif per domain so specialist adapters can reduce NLL
                    seq.append(tok_base + ((offset + pos) % 8))
            chunks.append(
                {
                    "expert_id": dom_name,
                    "dataset_id": f"synthetic_{dom_name}",
                    "row": row_idx,
                    "ids": seq,
                }
            )
    return chunks


def _eval_buckets_by_domain(
    model: CrowdV1WithEngram,
    buckets: dict[str, torch.Tensor],
    chunks: list[dict[str, Any]],
    *,
    num_router_fragments: int = 2,
    score_from: int = 4,
) -> dict[str, Any]:
    """Compute per-chunk and per-domain mean NLL for ``buckets``."""
    load_moe_trainable_buckets(model, buckets, num_router_fragments=num_router_fragments)
    model.eval()
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    h_pre = forward_model_pre_ln(model, ids, batch_size=16)
    nlls = (
        per_chunk_nll_from_pre_ln(
            h_pre,
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
    by_dom: dict[str, list[float]] = {}
    for ch, val in zip(chunks, nlls.tolist(), strict=True):
        by_dom.setdefault(str(ch["expert_id"]), []).append(float(val))
    dom_mean = {k: float(np.mean(v)) for k, v in sorted(by_dom.items())}
    spec_vals = [float(v) for ch, v in zip(chunks, nlls.tolist(), strict=True) if not str(ch["expert_id"]).startswith("E0_")]
    return {
        "nll_mean": float(nlls.mean()),
        "e0_nll": dom_mean["E0_general_fineweb"],
        "spec_nll": float(np.mean(spec_vals)),
        "per_domain": dom_mean,
    }


def _build_specialist_peer_buckets(
    model: CrowdV1WithEngram,
    parent_buckets: dict[str, torch.Tensor],
    domain_dirs: torch.Tensor,
    *,
    expert_idx: int,
    num_router_fragments: int = 2,
    spec_gain_strength: float = 0.75,
    prose_drift_strength: float = 0.90,
    cross_conflict_strength: float = 0.20,
    ln_drift_strength: float = 0.25,
    seed: int = 100,
) -> dict[str, torch.Tensor]:
    """Create one asynchronous specialist peer update with helpful target signal + prose/LN drift."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    load_moe_trainable_buckets(model, parent_buckets, num_router_fragments=num_router_fragments)
    block_h = 8
    block_i = 16
    tok_base = expert_idx * 8
    h0, h1 = expert_idx * block_h, (expert_idx + 1) * block_h
    i0, i1 = expert_idx * block_i, (expert_idx + 1) * block_i

    with torch.no_grad():
        # Construct directional specialist improvement aligning SwiGLU output toward next-token embedding
        wte_block = model.wte.weight[tok_base : tok_base + 8, h0:h1]
        next_target = (wte_block.roll(-1, dims=0) - wte_block).mean(dim=0)
        next_target = next_target / next_target.norm().clamp_min(1e-6)

        for blk in model.blocks:
            # Helpful router reinforcement on own expert + opposing off-domain noise on sibling experts
            blk.mlp.router.weight[expert_idx] += 2.0 * domain_dirs[expert_idx]
            # Contaminate router weights in general prose subspace (dims 0..7)
            blk.mlp.router.weight[expert_idx] += prose_drift_strength * domain_dirs[0]

            # Set helpful specialist LoRA factors on own expert
            exp = blk.mlp.experts[str(expert_idx)]
            exp.w1_A[:, h0:h1] += torch.randn((exp.rank, block_h), generator=gen) * 0.20
            exp.w2_A[:, h0:h1] += torch.randn((exp.rank, block_h), generator=gen) * 0.20
            exp.w1_B[i0:i1, :].copy_(torch.randn((block_i, exp.rank), generator=gen) * spec_gain_strength)
            exp.w2_B[i0:i1, :].copy_(torch.randn((block_i, exp.rank), generator=gen) * spec_gain_strength)
            exp.w3_B[h0:h1, :].copy_(
                next_target.unsqueeze(1) * spec_gain_strength
                + torch.randn((block_h, exp.rank), generator=gen) * 0.25
            )
            # Plant general-prose leakage inside specialist expert A-matrices (dims 0..7)
            exp.w1_A[:, :block_h] += torch.randn((exp.rank, block_h), generator=gen) * prose_drift_strength
            exp.w2_A[:, :block_h] += torch.randn((exp.rank, block_h), generator=gen) * prose_drift_strength
            exp.w3_A[:, :block_i] += torch.randn((exp.rank, block_i), generator=gen) * (0.6 * prose_drift_strength)
            exp.w3_B[:block_h, :] += torch.randn((block_h, exp.rank), generator=gen) * (0.5 * prose_drift_strength)

            # Plant opposing-sign gradient interference on sibling specialist experts (causes naive cancellation)
            for other_idx in range(1, 5):
                if other_idx == expert_idx:
                    continue
                other_exp = blk.mlp.experts[str(other_idx)]
                oh0, oh1 = other_idx * block_h, (other_idx + 1) * block_h
                oi0, oi1 = other_idx * block_i, (other_idx + 1) * block_i
                other_exp.w1_B[oi0:oi1, :].copy_(
                    torch.randn((block_i, other_exp.rank), generator=gen) * (-cross_conflict_strength)
                )
                other_exp.w3_B[oh0:oh1, :].copy_(
                    torch.randn((block_h, other_exp.rank), generator=gen) * (-cross_conflict_strength)
                )

        # Contaminate final LayerNorm along prose subspace
        model.ln.weight[:block_h] += ln_drift_strength * 1.4
        model.ln.bias[:block_h] += ln_drift_strength * domain_dirs[0, :block_h] * 2.2

    return {
        k: v.detach().cpu().float().clone()
        for k, v in extract_moe_trainable_buckets(model, num_router_fragments=num_router_fragments).items()
    }


# ---------------------------------------------------------------------------
# S1: 4-Peer Specialist MoE Merge Outperforming Naive Average Under Guard
# ---------------------------------------------------------------------------


def test_s1_four_peer_specialist_moe_merge_beats_naive_average_under_prose_guard() -> None:
    """Merge 4 specialist peers (code, math, exoplanet, stellar) beating naive avg under <= +1.0% guard."""
    model, parent_buckets, domain_dirs = _build_5domain_crowd_moe_model(
        num_layers=2, rank=4, num_router_fragments=2, seed=20260601
    )
    probe_chunks = _make_domain_chunks(chunks_per_domain=6, seq_len=20, seed=20260602)
    eval_chunks = _make_domain_chunks(chunks_per_domain=8, seq_len=20, seed=20260699)

    # Optimize each specialist peer's delta direction to genuinely lower its domain NLL on probe
    # by choosing helpful w3_B orientation against base hidden output
    peer_buckets_list: list[dict[str, torch.Tensor]] = []
    for p_offset, e_idx in enumerate((1, 2, 3, 4)):
        peer_b = _build_specialist_peer_buckets(
            model,
            parent_buckets,
            domain_dirs,
            expert_idx=e_idx,
            num_router_fragments=2,
            spec_gain_strength=0.45,
            prose_drift_strength=0.95,
            cross_conflict_strength=0.18,
            ln_drift_strength=0.30,
            seed=700 + p_offset * 31,
        )
        # Ensure each peer's LoRA delta reduces NLL on its own specialist domain vs parent
        # using one analytical coordinate sign check on w3_B
        base_dom_nll = _eval_buckets_by_domain(model, parent_buckets, probe_chunks, score_from=4)
        peer_dom_nll = _eval_buckets_by_domain(model, peer_b, probe_chunks, score_from=4)
        dom_key = ("", "E1_python_systems_code", "E2_math_reasoning", "E5_exoplanet_transit", "E6_stellar_spectroscopy")[e_idx]
        if peer_dom_nll["per_domain"][dom_key] >= base_dom_nll["per_domain"][dom_key]:
            # Flip w3_B sign on target expert so specialist delta reduces its target domain NLL
            load_moe_trainable_buckets(model, peer_b, num_router_fragments=2)
            with torch.no_grad():
                for blk in model.blocks:
                    blk.mlp.experts[str(e_idx)].w3_B.neg_()
            peer_b = {
                k: v.detach().cpu().float().clone()
                for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()
            }
        peer_buckets_list.append(peer_b)

    # Baseline parent evaluation on disjoint held-out eval_chunks
    parent_eval = _eval_buckets_by_domain(model, parent_buckets, eval_chunks, score_from=4)

    # Naive uniform averaging across the 4 peers
    naive_buckets = {
        k: (sum(p[k].to(torch.float32) for p in peer_buckets_list) / float(len(peer_buckets_list))).contiguous()
        for k in parent_buckets
    }
    naive_eval = _eval_buckets_by_domain(model, naive_buckets, eval_chunks, score_from=4)
    naive_e0_ratio = naive_eval["e0_nll"] / parent_eval["e0_nll"]
    assert naive_e0_ratio > DOMAIN_GUARD_MAX_RATIO, (
        f"Expected naive uniform merge to fail <= +1.0% general guard, got ratio={naive_e0_ratio:.4f}"
    )

    # Merge via TIES/DARE + Ledoit-Wolf Prose-Nullspace + Protected LN + ANSE Guard
    cfg = TIESDAREMergerConfig(
        density_top_p=0.45,
        drop_rate=0.10,
        seed=20261124,
        ln_tail_policy="guard_calibrated",
        prose_projection_mode="orthogonal_nullspace",
        prose_eigen_ratio_tau=1.15,
        prose_variance_threshold=0.999,
        prose_proj_strength=1.0,
        project_router_delta_orthogonal=True,
        enforce_all_domain_guard=True,
        score_from=4,
        batch_size=16,
    )
    merged_buckets, receipt = merge_moe_checkpoints_with_prose_guard(
        model,
        parent_buckets,
        peer_buckets_list,
        probe_chunks=probe_chunks,
        config=cfg,
        num_router_fragments=2,
    )
    _, receipt_repeat = merge_moe_checkpoints_with_prose_guard(
        model,
        parent_buckets,
        peer_buckets_list,
        probe_chunks=probe_chunks,
        config=cfg,
        num_router_fragments=2,
    )
    assert receipt_repeat["receipt_sha256"] == receipt["receipt_sha256"]
    assert receipt_repeat["final_sha256"] == receipt["final_sha256"]
    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["guard_ok"] is True
    assert receipt["delta_anse_energy"] < 0.0
    assert receipt["prose_nullspace_modes_projected"] > 0
    assert receipt["layernorm_tail_protected"] is True
    assert 0.0 < receipt["sign_consensus_ratio"] <= 1.0
    assert len(receipt["receipt_sha256"]) == 64

    # Evaluate on held-out eval_chunks (zero probe leakage!)
    merged_eval = _eval_buckets_by_domain(model, merged_buckets, eval_chunks, score_from=4)
    merged_e0_ratio = merged_eval["e0_nll"] / parent_eval["e0_nll"]
    assert merged_e0_ratio <= DOMAIN_GUARD_MAX_RATIO
    assert merged_eval["spec_nll"] < parent_eval["spec_nll"]
    assert merged_eval["spec_nll"] < naive_eval["spec_nll"]
    assert merged_eval["nll_mean"] < naive_eval["nll_mean"]


# ---------------------------------------------------------------------------
# S2: Exact Rank-r Preservation via QR-SVD Without Dense Materialization
# ---------------------------------------------------------------------------


def test_s2_exact_rank_r_preservation_qr_svd_without_dense_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merge K=4 rank-8 SwiGLU LoRA adapters via QR-SVD preserving rank=8 with zero dense (d_out x d_in)."""
    # Part A: Exact numerical equivalence with dense SVD on rectangular (64 x 48) adapters
    d_out, d_in, rank, k_peers = 64, 48, 8, 4
    u_ortho = _make_orthogonal_basis(d_out, seed=20260701)
    v_ortho = _make_orthogonal_basis(d_in, seed=20260702)

    # Disjoint rank-2 subspaces across 4 peers -> combined rank = 8 == target_rank -> zero truncation error!
    disjoint_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k in range(k_peers):
        s_slice = torch.linspace(2.5 - 0.3 * k, 1.8 - 0.2 * k, 2, dtype=torch.float64)
        u_sub = u_ortho[:, 2 * k : 2 * (k + 1)]
        v_sub = v_ortho[:, 2 * k : 2 * (k + 1)]
        b_full = torch.zeros((d_out, rank), dtype=torch.float64)
        a_full = torch.zeros((rank, d_in), dtype=torch.float64)
        b_full[:, :2] = u_sub * s_slice.sqrt().unsqueeze(0)
        a_full[:2, :] = s_slice.sqrt().unsqueeze(1) * v_sub.T
        disjoint_pairs.append((b_full, a_full))

    (b_exact, a_exact), exact_meta = dare_rescale_and_svd_compress_lora(
        disjoint_pairs, drop_rate=0.0, target_rank=rank, seed=42
    )
    assert b_exact.shape == (d_out, rank)
    assert a_exact.shape == (rank, d_in)
    assert exact_meta["preserved_exact_rank"] == rank
    assert exact_meta["input_rank_sum"] == k_peers * rank
    assert exact_meta["dense_matrix_materialized"] is False
    assert exact_meta["energy_retained_ratio"] == pytest.approx(1.0, abs=1e-12)

    dense_target = sum((bk @ ak) / float(k_peers) for bk, ak in disjoint_pairs)
    assert torch.allclose(b_exact @ a_exact, dense_target, atol=1e-11)

    # Full-rank-8 random adapters (total rank 32 -> truncated to exact rank 8) match dense truncated SVD
    gen = torch.Generator(device="cpu").manual_seed(20260703)
    full_pairs = [
        (
            torch.randn((d_out, rank), generator=gen, dtype=torch.float64) * 0.25,
            torch.randn((rank, d_in), generator=gen, dtype=torch.float64) * 0.25,
        )
        for _ in range(k_peers)
    ]
    (b_trunc, a_trunc), trunc_meta = dare_rescale_and_svd_compress_lora(
        full_pairs, drop_rate=0.0, target_rank=rank, seed=7
    )
    assert b_trunc.shape == (d_out, rank)
    assert a_trunc.shape == (rank, d_in)
    assert trunc_meta["core_max_dim"] <= k_peers * rank

    dense_sum = sum((bk @ ak) / float(k_peers) for bk, ak in full_pairs)
    u_d, s_d, vh_d = torch.linalg.svd(dense_sum, full_matrices=False)
    dense_rank8 = (u_d[:, :rank] * s_d[:rank].unsqueeze(0)) @ vh_d[:rank, :]
    assert torch.allclose(b_trunc @ a_trunc, dense_rank8, atol=1e-11)

    # Deterministic DARE rescaling + Monte Carlo unbiasedness check
    (b_d1, a_d1), _ = dare_rescale_and_svd_compress_lora(full_pairs, drop_rate=0.35, target_rank=rank, seed=123)
    (b_d2, a_d2), _ = dare_rescale_and_svd_compress_lora(full_pairs, drop_rate=0.35, target_rank=rank, seed=123)
    (b_d3, a_d3), _ = dare_rescale_and_svd_compress_lora(full_pairs, drop_rate=0.35, target_rank=rank, seed=124)
    assert torch.equal(b_d1, b_d2) and torch.equal(a_d1, a_d2)
    assert not torch.equal(b_d1, b_d3)

    # Unbiasedness check without truncation (target_rank = k_peers * rank)
    mc_acc = torch.zeros((d_out, d_in), dtype=torch.float64)
    n_mc = 48
    for s_idx in range(n_mc):
        (bm_mc, am_mc), _ = dare_rescale_and_svd_compress_lora(
            full_pairs, drop_rate=0.30, target_rank=k_peers * rank, seed=5000 + s_idx
        )
        mc_acc += bm_mc @ am_mc
    mc_mean = mc_acc / float(n_mc)
    rel_mc_err = float((mc_mean - dense_sum).norm().item() / dense_sum.norm().item())
    assert rel_mc_err < 0.15

    # Non-zero base_pair + TIES + target_rank > input_rank zero-padding + zero-norm fast path
    base_b = torch.randn((d_out, rank), generator=gen, dtype=torch.float64) * 0.1
    base_a = torch.randn((rank, d_in), generator=gen, dtype=torch.float64) * 0.1
    peers_rel = [(base_b + bk, base_a + ak) for bk, ak in full_pairs]
    (b_rel, a_rel), rel_meta = dare_rescale_and_svd_compress_lora(
        peers_rel,
        drop_rate=0.1,
        target_rank=rank,
        seed=99,
        base_pair=(base_b, base_a),
        scaling_lambda=0.8,
        ties_density_top_p=0.5,
        max_peer_norm_ratio=3.0,
    )
    assert b_rel.shape == (d_out, rank) and a_rel.shape == (rank, d_in)
    assert 0.0 < rel_meta["sign_consensus_ratio"] <= 1.0

    # scaling_lambda > 1.0 exercises negative residual base weight branch cleanly
    (b_extrap, a_extrap), _ = dare_rescale_and_svd_compress_lora(
        peers_rel, drop_rate=0.0, target_rank=rank, seed=99, base_pair=(base_b, base_a), scaling_lambda=1.25
    )
    assert b_extrap.shape == (d_out, rank) and a_extrap.shape == (rank, d_in)

    # Zero-norm base with frozen delta_A == 0 and TIES across B
    zero_da_peers = [(bk, base_a.clone()) for bk, _ in full_pairs]
    (b_za, a_za), _ = dare_rescale_and_svd_compress_lora(
        zero_da_peers, drop_rate=0.0, target_rank=rank, seed=12, base_pair=(base_b, base_a), ties_density_top_p=0.5
    )
    assert b_za.shape == (d_out, rank) and a_za.shape == (rank, d_in)

    # Zero-norm adapter fast path & target_rank padding
    (b_zero, a_zero), z_meta = dare_rescale_and_svd_compress_lora(
        [(torch.zeros((16, 4)), torch.zeros((4, 12)))], drop_rate=0.0, target_rank=8, seed=1
    )
    assert torch.equal(b_zero, torch.zeros((16, 8))) and torch.equal(a_zero, torch.zeros((8, 12)))
    assert z_meta["effective_rank"] == 0.0

    # Part B: Full Crowd-v1 SwiGLU dimensions (1536 -> 2560, rank=8) with SVD matrix size spy!
    svd_shapes: list[tuple[int, int]] = []
    real_svd = torch.linalg.svd

    def _spy_svd(mat: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        svd_shapes.append((int(mat.shape[0]), int(mat.shape[1])))
        assert max(mat.shape) <= k_peers * rank, f"Dense SVD materialized: {tuple(mat.shape)}"
        return real_svd(mat, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", _spy_svd)

    h_big, i_big, r_big = 1536, 2560, 8
    swiglu_peers: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
    for k in range(k_peers):
        g_k = torch.Generator(device="cpu").manual_seed(20260800 + k)
        swiglu_peers.append(
            {
                "w1": (
                    torch.randn((i_big, r_big), generator=g_k, dtype=torch.float32) * 0.05,
                    torch.randn((r_big, h_big), generator=g_k, dtype=torch.float32) * 0.05,
                ),
                "w2": (
                    torch.randn((i_big, r_big), generator=g_k, dtype=torch.float32) * 0.05,
                    torch.randn((r_big, h_big), generator=g_k, dtype=torch.float32) * 0.05,
                ),
                "w3": (
                    torch.randn((h_big, r_big), generator=g_k, dtype=torch.float32) * 0.05,
                    torch.randn((r_big, i_big), generator=g_k, dtype=torch.float32) * 0.05,
                ),
            }
        )

    merged_swiglu, sw_meta = merge_swiglu_lora_adapters_dare_svd(
        swiglu_peers, drop_rate=0.25, target_rank=r_big, seed=20260888, ties_density_top_p=0.40
    )
    assert len(svd_shapes) == 3
    assert all(max(sh) <= k_peers * r_big for sh in svd_shapes)
    assert sw_meta["dense_matrix_materialized"] is False
    assert merged_swiglu["w1"][0].shape == (i_big, r_big) and merged_swiglu["w1"][1].shape == (r_big, h_big)
    assert merged_swiglu["w2"][0].shape == (i_big, r_big) and merged_swiglu["w2"][1].shape == (r_big, h_big)
    assert merged_swiglu["w3"][0].shape == (h_big, r_big) and merged_swiglu["w3"][1].shape == (r_big, i_big)

    # Also test factored prose-nullspace projection on base_pair != 0 without dense materialization
    g_proj_big, _ = compute_prose_nullspace_projector(
        torch.randn((32, h_big), generator=gen, dtype=torch.float64),
        max_modes=4,
    )
    b_p, a_p = apply_prose_projector_to_lora_pair(
        merged_swiglu["w1"][0],
        merged_swiglu["w1"][1],
        g_proj_big,
        base_pair=swiglu_peers[0]["w1"],
        target_rank=r_big,
    )
    assert b_p.shape == (i_big, r_big) and a_p.shape == (r_big, h_big)


# ---------------------------------------------------------------------------
# S3: Byzantine / Noisy Outlier Resilience + ANSE Rollback on Poisoned Delta
# ---------------------------------------------------------------------------


def test_s3_byzantine_outlier_resilience_and_anse_rollback_on_poisoned_delta() -> None:
    """Verify TIES resilience against noisy/Byzantine peers + bitwise ANSE rollback on poisoned delta."""
    # Part A: TIES sign-election & outlier resilience on vector deltas
    gen = torch.Generator(device="cpu").manual_seed(20260901)
    dim = 200
    true_vec = torch.zeros(dim, dtype=torch.float64)
    true_vec[:40] = torch.linspace(1.0, 2.0, 40, dtype=torch.float64)

    honest_peers = [
        true_vec + 0.02 * torch.randn(dim, generator=gen, dtype=torch.float64) for _ in range(3)
    ]
    # Byzantine peer 1: sign-flip + low-magnitude uniform noise across all 200 coordinates
    byz_signflip = -1.5 * true_vec + 0.12 * torch.randn(dim, generator=gen, dtype=torch.float64)
    deltas_s3a = [*honest_peers, byz_signflip]

    naive_avg = sum(deltas_s3a) / 4.0
    ties_merged, ties_meta = trim_and_elect_sign_ties(deltas_s3a, density_top_p=0.25)

    cos_ties = float(F.cosine_similarity(ties_merged.unsqueeze(0), true_vec.unsqueeze(0)).item())
    mse_ties = float((ties_merged - true_vec).pow(2).mean().item())
    mse_naive = float((naive_avg - true_vec).pow(2).mean().item())
    assert cos_ties > 0.995
    assert mse_ties < mse_naive * 0.05
    assert 0.5 < ties_meta["sign_consensus_ratio"] < 1.0

    # Byzantine peer 2: extreme 100x magnitude poisoning neutralized by max_peer_norm_ratio + majority mode
    byz_spike = -100.0 * true_vec
    ties_clipped, clip_meta = trim_and_elect_sign_ties(
        [*honest_peers, byz_spike],
        density_top_p=0.25,
        sign_election_mode="majority",
        max_peer_norm_ratio=1.5,
    )
    assert clip_meta["norms_clipped"] == 1
    assert float(F.cosine_similarity(ties_clipped.unsqueeze(0), true_vec.unsqueeze(0)).item()) > 0.995
    assert float((ties_clipped - true_vec).pow(2).mean().item()) < 0.02

    # Part B: End-to-end ANSE rollback on poisoned MoE checkpoint violating <= +1.0% general guard
    model, parent_buckets, domain_dirs = _build_5domain_crowd_moe_model(
        num_layers=2, rank=4, num_router_fragments=2, seed=20260910
    )
    probe_chunks = _make_domain_chunks(chunks_per_domain=4, seq_len=16, seed=20260911)
    parent_sha = compute_buckets_sha256(parent_buckets)

    # Poisoned peer cohort corrupting general prose weights heavily (triggers guard > +1.0%)
    poisoned_peers: list[dict[str, torch.Tensor]] = []
    for _k in range(3):
        pb = {name: val.clone() for name, val in parent_buckets.items()}
        load_moe_trainable_buckets(model, pb, num_router_fragments=2)
        with torch.no_grad():
            for blk in model.blocks:
                blk.mlp.router.weight[0, :8] += 6.0 * domain_dirs[0, :8]
                exp0 = blk.mlp.experts["0"]
                exp0.w1_A[:, :8].copy_(domain_dirs[0, :8].unsqueeze(0).repeat(exp0.rank, 1))
                exp0.w2_A[:, :8].copy_(domain_dirs[0, :8].unsqueeze(0).repeat(exp0.rank, 1))
                exp0.w1_B[:16, :].fill_(3.5)
                exp0.w2_B[:16, :].fill_(3.5)
                exp0.w3_A[:, :16].fill_(0.5)
                exp0.w3_B[:8, :].copy_(domain_dirs[0, :8].unsqueeze(1).repeat(1, exp0.rank) * (-6.0))
                exp0.w3_B[8:16, :].fill_(4.0)
        poisoned_peers.append(
            {
                name: val.detach().cpu().float().clone()
                for name, val in extract_moe_trainable_buckets(model, num_router_fragments=2).items()
            }
        )

    cfg_rb = TIESDAREMergerConfig(
        density_top_p=0.5,
        drop_rate=0.0,
        ln_tail_policy="freeze_parent",
        prose_projection_mode="none",
        score_from=4,
        batch_size=16,
    )
    final_buckets, rb_receipt = merge_moe_checkpoints_with_prose_guard(
        model,
        parent_buckets,
        poisoned_peers,
        probe_chunks=probe_chunks,
        config=cfg_rb,
        num_router_fragments=2,
    )
    assert rb_receipt["status"] == "rolled_back"
    assert rb_receipt["accepted"] is False
    assert rb_receipt["rolled_back"] is True
    assert rb_receipt["rollback_bitwise_verified"] is True
    assert rb_receipt["reject_reason"] == "domain_guard_exceeded"
    assert rb_receipt["guard_ok"] is False
    assert rb_receipt["anse_energy"] == pytest.approx(rb_receipt["parent_anse_energy"])
    assert rb_receipt["candidate_anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rb_receipt["final_sha256"] == parent_sha
    assert rb_receipt["candidate_sha256"] != parent_sha
    for key, p_tensor in parent_buckets.items():
        assert torch.equal(final_buckets[key], p_tensor)

    # Also verify non-improving Delta E >= 0 (even when guard_ok=True) triggers atomic rollback
    def _nonimproving_eval(_m: nn.Module, cand_b: dict[str, torch.Tensor]) -> dict[str, Any]:
        if compute_buckets_sha256(cand_b) == parent_sha:
            return {"E0_general_fineweb": 2.0, "E1_code": 2.0}
        return {"E0_general_fineweb": 2.005, "E1_code": 2.05}

    final_ni, ni_receipt = merge_moe_checkpoints_with_prose_guard(
        model,
        parent_buckets,
        poisoned_peers,
        eval_fn=_nonimproving_eval,
        config=TIESDAREMergerConfig(
            density_top_p=0.5,
            drop_rate=0.0,
            ln_tail_policy="freeze_parent",
            prose_projection_mode="none",
            enforce_all_domain_guard=False,
        ),
        num_router_fragments=2,
    )
    assert ni_receipt["rolled_back"] is True
    assert ni_receipt["guard_ok"] is True
    assert ni_receipt["reject_reason"] == "non_improving_anse_energy"
    assert all(torch.equal(final_ni[k], parent_buckets[k]) for k in parent_buckets)

    # Also verify non-finite candidate output triggers atomic rollback via eval_fn
    def _nonfinite_eval(_m: nn.Module, cand_b: dict[str, torch.Tensor]) -> dict[str, Any]:
        if compute_buckets_sha256(cand_b) == parent_sha:
            return {"E0_general_fineweb": 2.0, "E1_code": 2.0}
        return {"domain_nlls": {"E0_general_fineweb": float("nan"), "E1_code": 1.5}, "non_finite_outputs": 1}

    final_nf, nf_receipt = merge_moe_checkpoints_with_prose_guard(
        model,
        parent_buckets,
        poisoned_peers,
        eval_fn=_nonfinite_eval,
        config=cfg_rb,
        num_router_fragments=2,
    )
    assert nf_receipt["rolled_back"] is True
    assert nf_receipt["reject_reason"] == "non_finite_candidate"
    assert nf_receipt["candidate_anse_energy"] == ANSE_MAXIMUM_PAIN
    assert nf_receipt["final_sha256"] == parent_sha
    assert all(torch.equal(final_nf[k], parent_buckets[k]) for k in parent_buckets)


# ---------------------------------------------------------------------------
# S4: router_frag LayerNorm Tail Protection Across Multi-Round Merges
# ---------------------------------------------------------------------------


def test_s4_router_frag_layernorm_tail_protection_multi_round_merges() -> None:
    """Prevent cumulative LayerNorm drift across 5 sequential OpenDiLoCo merge rounds."""
    model, round0_buckets, domain_dirs = _build_5domain_crowd_moe_model(
        num_layers=2, rank=4, num_router_fragments=2, seed=20261001
    )
    probe_chunks = _make_domain_chunks(chunks_per_domain=4, seq_len=16, seed=20261002)
    h_dim = int(model.ln.weight.numel())
    last_r_key = sorted(k for k in round0_buckets if k.startswith("router_frag_"))[-1]

    init_ln_tail = round0_buckets[last_r_key][-2 * h_dim :].clone()
    parent_eval0 = _eval_buckets_by_domain(model, round0_buckets, probe_chunks, score_from=4)

    # Simulate 5 sequential rounds where workers always inject +0.15 systematic LayerNorm drift
    cur_naive = {k: v.clone() for k, v in round0_buckets.items()}
    cur_freeze = {k: v.clone() for k, v in round0_buckets.items()}
    cur_guarded = {k: v.clone() for k, v in round0_buckets.items()}

    naive_ln_drifts: list[float] = []
    freeze_ln_drifts: list[float] = []
    guarded_ln_drifts: list[float] = []

    for r_idx in range(5):
        # Peers on cur_freeze & cur_guarded get router updates + systematic LN drift
        def _make_round_peers(base_b: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
            peers: list[dict[str, torch.Tensor]] = []
            for p_idx in range(3):
                pb = {k: v.clone() for k, v in base_b.items()}
                # Legitimate sparse router gate improvement on specialist dims
                pb["router_frag_0"][:h_dim] += 0.04 * domain_dirs[1]
                # Systematic harmful LayerNorm tail drift on router_frag_1[-2*h_dim:]
                pb[last_r_key][-2 * h_dim : -h_dim] += 0.16 * (1.0 + 0.05 * p_idx)
                pb[last_r_key][-h_dim:] += 0.22 * domain_dirs[0]
                peers.append(pb)
            return peers

        # 1. Naive unprotected round averaging accumulates LN drift monotonically!
        naive_peers = _make_round_peers(cur_naive)
        cur_naive = {
            k: (sum(p[k] for p in naive_peers) / 3.0).contiguous() for k in cur_naive
        }
        naive_ln_drifts.append(
            float((cur_naive[last_r_key][-2 * h_dim :] - init_ln_tail).norm().item())
        )

        # 2. Standalone protect_router_frag_layernorm_tail with freeze_parent
        freeze_peers = _make_round_peers(cur_freeze)
        protected_frags, prot_meta = protect_router_frag_layernorm_tail(
            cur_freeze,
            freeze_peers,
            hidden_size=h_dim,
            density_top_p=0.5,
            drop_rate=0.0,
            seed=20261010 + r_idx,
            ln_tail_policy="freeze_parent",
        )
        assert prot_meta["layernorm_tail_protected"] is True
        assert prot_meta["layernorm_drift_l2"] == 0.0
        cur_freeze.update(protected_frags)
        freeze_ln_drifts.append(
            float((cur_freeze[last_r_key][-2 * h_dim :] - init_ln_tail).norm().item())
        )

        # 3. Full merge_moe_checkpoints_with_prose_guard across 5 rounds with guard_calibrated
        guarded_peers = _make_round_peers(cur_guarded)
        # Add helpful specialist LoRA step on expert 1 so each round genuinely improves NLL
        for p_idx, pb in enumerate(guarded_peers):
            load_moe_trainable_buckets(model, pb, num_router_fragments=2)
            with torch.no_grad():
                for blk in model.blocks:
                    exp1 = blk.mlp.experts["1"]
                    exp1.w1_B[16:32, :] += 0.03 * (p_idx + 1)
                    exp1.w3_B[8:16, :] += 0.03 * domain_dirs[1, 8:16].unsqueeze(1)
            guarded_peers[p_idx] = {
                k: v.detach().cpu().float().clone()
                for k, v in extract_moe_trainable_buckets(model, num_router_fragments=2).items()
            }

        cfg_g = TIESDAREMergerConfig(
            density_top_p=0.5,
            drop_rate=0.0,
            ln_tail_policy="guard_calibrated",
            ln_max_step_norm=0.5,
            prose_projection_mode="condition_bounded_alphaedit",
            prose_eigen_ratio_tau=1.2,
            require_energy_improvement=False,
            score_from=4,
            batch_size=16,
        )
        cur_guarded, g_rec = merge_moe_checkpoints_with_prose_guard(
            model,
            cur_guarded,
            guarded_peers,
            probe_chunks=probe_chunks,
            config=cfg_g,
            num_router_fragments=2,
        )
        assert g_rec["guard_ok"] is True
        guarded_ln_drifts.append(
            float((cur_guarded[last_r_key][-2 * h_dim :] - init_ln_tail).norm().item())
        )

    # Naive LN drift grew ~5x across 5 rounds and exploded E0 NLL
    assert naive_ln_drifts[-1] > naive_ln_drifts[0] * 4.5
    naive_eval5 = _eval_buckets_by_domain(model, cur_naive, probe_chunks, score_from=4)
    assert naive_eval5["e0_nll"] / parent_eval0["e0_nll"] > DOMAIN_GUARD_MAX_RATIO

    # Protected freeze_parent had exact 0.0 cumulative LN drift while still updating router weights
    assert all(d == 0.0 for d in freeze_ln_drifts)
    assert not torch.equal(
        cur_freeze["router_frag_0"][:h_dim], round0_buckets["router_frag_0"][:h_dim]
    )

    # Guard-calibrated multi-round merge prevented LN explosion and preserved E0 <= +1.0%
    assert max(guarded_ln_drifts) < naive_ln_drifts[-1] * 0.25
    guarded_eval5 = _eval_buckets_by_domain(model, cur_guarded, probe_chunks, score_from=4)
    assert guarded_eval5["e0_nll"] / parent_eval0["e0_nll"] <= DOMAIN_GUARD_MAX_RATIO


# ---------------------------------------------------------------------------
# S5: Validation Errors, Edge Cases & Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_edge_cases_and_security_compliance_gates() -> None:
    """Validate defensive error branches, edge cases, and HiveMind security/stub compliance gates."""
    gen = torch.Generator(device="cpu").manual_seed(20261101)

    # 1. TIESDAREMergerConfig validation
    cfg_default = TIESDAREMergerConfig(max_peer_norm_ratio=2.5, router_density_top_p=0.5)
    assert cfg_default.density_top_p == 0.25
    assert cfg_default.max_peer_norm_ratio == 2.5
    assert cfg_default.router_density_top_p == 0.5
    with pytest.raises(TypeError, match="real number"):
        TIESDAREMergerConfig(density_top_p=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">="):
        TIESDAREMergerConfig(scaling_lambda=-1.0)
    with pytest.raises(ValueError, match="> 0.0"):
        TIESDAREMergerConfig(scaling_lambda=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        TIESDAREMergerConfig(density_top_p=1.5)
    with pytest.raises(ValueError, match=">= 0"):
        TIESDAREMergerConfig(prose_max_modes=-1)
    with pytest.raises(ValueError, match="density_top_p"):
        TIESDAREMergerConfig(density_top_p=0.0)
    with pytest.raises(ValueError, match="drop_rate"):
        TIESDAREMergerConfig(drop_rate=1.0)
    with pytest.raises(ValueError, match="target_rank"):
        TIESDAREMergerConfig(target_rank=0)
    with pytest.raises(TypeError, match="int"):
        TIESDAREMergerConfig(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sign_election_mode"):
        TIESDAREMergerConfig(sign_election_mode="median")
    with pytest.raises(ValueError, match="ln_tail_policy"):
        TIESDAREMergerConfig(ln_tail_policy="unknown")
    with pytest.raises(ValueError, match="prose_projection_mode"):
        TIESDAREMergerConfig(prose_projection_mode="bad")
    with pytest.raises(ValueError, match="alphaedit_max_operator_norm"):
        TIESDAREMergerConfig(alphaedit_max_operator_norm=0.9)
    with pytest.raises(ValueError, match="domain_guard_max_ratio"):
        TIESDAREMergerConfig(domain_guard_max_ratio=0.95)
    with pytest.raises(ValueError, match="anse_safe_e0_ratio"):
        TIESDAREMergerConfig(anse_safe_e0_ratio=0.99)
    with pytest.raises(TypeError, match="bool"):
        TIESDAREMergerConfig(enforce_all_domain_guard="yes")  # type: ignore[arg-type]

    # 2. trim_and_elect_sign_ties validation & edge cases
    with pytest.raises(ValueError, match="non-empty"):
        trim_and_elect_sign_ties([])
    with pytest.raises(TypeError, match="torch.Tensor"):
        trim_and_elect_sign_ties([[1.0, 2.0]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-empty"):
        trim_and_elect_sign_ties([torch.zeros((0, 4))])
    with pytest.raises(ValueError, match="finite"):
        trim_and_elect_sign_ties([torch.tensor([1.0, float("nan")])])
    with pytest.raises(ValueError, match="expected shape"):
        trim_and_elect_sign_ties([torch.ones(4), torch.ones(5)])
    with pytest.raises(ValueError, match="sign_election_mode"):
        trim_and_elect_sign_ties([torch.ones(4)], sign_election_mode="bad")
    with pytest.raises(TypeError, match="sequence"):
        trim_and_elect_sign_ties([torch.ones(4)], peer_weights="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="match peer count"):
        trim_and_elect_sign_ties([torch.ones(4)], peer_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="strictly positive sum"):
        trim_and_elect_sign_ties([torch.ones(4)], peer_weights=[0.0])

    zero_ties, zt_meta = trim_and_elect_sign_ties([torch.zeros(6), torch.zeros(6)], density_top_p=1.0, max_peer_norm_ratio=2.0)
    assert torch.equal(zero_ties, torch.zeros(6))
    assert zt_meta["sign_consensus_ratio"] == 1.0

    # 3. dare_rescale_and_svd_compress_lora & merge_swiglu_lora_adapters_dare_svd validation
    b_ok = torch.ones((6, 2))
    a_ok = torch.ones((2, 4))
    with pytest.raises(ValueError, match="non-empty"):
        dare_rescale_and_svd_compress_lora([], drop_rate=0.1, target_rank=2, seed=0)
    with pytest.raises(ValueError, match="drop_rate"):
        dare_rescale_and_svd_compress_lora([(b_ok, a_ok)], drop_rate=1.0, target_rank=2, seed=0)
    with pytest.raises(ValueError, match="pair"):
        dare_rescale_and_svd_compress_lora([(b_ok,)], drop_rate=0.1, target_rank=2, seed=0)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="2D tensor"):
        dare_rescale_and_svd_compress_lora([(torch.ones(6), a_ok)], drop_rate=0.1, target_rank=2, seed=0)
    with pytest.raises(ValueError, match="outer dims"):
        dare_rescale_and_svd_compress_lora(
            [(b_ok, a_ok), (torch.ones((5, 2)), a_ok)], drop_rate=0.1, target_rank=2, seed=0
        )
    with pytest.raises(ValueError, match="inner rank mismatch"):
        dare_rescale_and_svd_compress_lora(
            [(b_ok, torch.ones((3, 4)))], drop_rate=0.1, target_rank=2, seed=0
        )
    with pytest.raises(ValueError, match="base_pair"):
        dare_rescale_and_svd_compress_lora(
            [(b_ok, a_ok)], drop_rate=0.1, target_rank=2, seed=0, base_pair=(b_ok,)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="do not match"):
        dare_rescale_and_svd_compress_lora(
            [(b_ok, a_ok)], drop_rate=0.1, target_rank=2, seed=0, base_pair=(torch.ones((5, 2)), a_ok)
        )
    with pytest.raises(ValueError, match="sign_election_mode"):
        dare_rescale_and_svd_compress_lora(
            [(b_ok, a_ok)], drop_rate=0.1, target_rank=2, seed=0, sign_election_mode="bad"
        )
    with pytest.raises(ValueError, match="non-empty"):
        merge_swiglu_lora_adapters_dare_svd([], drop_rate=0.1, target_rank=2, seed=0)

    # 4. compute_prose_nullspace_projector & apply_prose_projector_to_lora_pair validation & edges
    xp = torch.randn((16, 8), generator=gen, dtype=torch.float64)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_prose_nullspace_projector([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        compute_prose_nullspace_projector(xp, mode="invalid")
    with pytest.raises(ValueError, match=">= 1.0"):
        compute_prose_nullspace_projector(xp, max_operator_norm=0.8)
    with pytest.raises(ValueError, match="spec_x must be provided"):
        compute_prose_nullspace_projector(xp, mode="condition_bounded_alphaedit")
    with pytest.raises(ValueError, match="width"):
        compute_prose_nullspace_projector(xp, spec_x=torch.ones((4, 6)))

    # Exercise max_modes / variance_threshold truncation without spec_x and with spec_x
    xp_rank = torch.zeros((16, 8), dtype=torch.float64)
    xp_rank[:, :4] = torch.randn((16, 4), generator=gen, dtype=torch.float64) * 3.0
    g_cap, meta_cap = compute_prose_nullspace_projector(
        xp_rank, eigen_ratio_tau=0.5, variance_threshold=0.5, max_modes=2
    )
    assert g_cap.shape == (8, 8)
    assert meta_cap["projected_modes"] <= 2

    g_cap_s, meta_caps = compute_prose_nullspace_projector(
        xp_rank,
        spec_x=torch.randn((16, 8), generator=gen, dtype=torch.float64) * 0.05,
        eigen_ratio_tau=1.0,
        variance_threshold=0.5,
        max_modes=2,
    )
    assert g_cap_s.shape == (8, 8)
    assert meta_caps["projected_modes"] <= 2

    g_none, meta_none = compute_prose_nullspace_projector(xp, mode="none")
    assert torch.equal(g_none, torch.eye(8, dtype=torch.float64))
    assert meta_none["projected_modes"] == 0

    g_zero_x, meta_zx = compute_prose_nullspace_projector(torch.zeros((8, 8)))
    assert torch.equal(g_zero_x, torch.eye(8, dtype=torch.float64))
    assert meta_zx["projected_modes"] == 0

    # Isotropic high tau -> no modes exceed tau * mean_eig
    g_hi_tau, meta_ht = compute_prose_nullspace_projector(xp, eigen_ratio_tau=100.0)
    assert torch.equal(g_hi_tau, torch.eye(8, dtype=torch.float64))
    assert meta_ht["projected_modes"] == 0

    # With spec_x dominating -> no prose modes exceed tau
    g_spec_dom, meta_sd = compute_prose_nullspace_projector(
        torch.randn((16, 8), generator=gen) * 0.1,
        spec_x=torch.randn((16, 8), generator=gen) * 5.0,
        eigen_ratio_tau=5.0,
    )
    assert torch.equal(g_spec_dom, torch.eye(8, dtype=torch.float64))
    assert meta_sd["projected_modes"] == 0

    with pytest.raises(ValueError, match="incompatible"):
        apply_prose_projector_to_lora_pair(torch.ones((4, 2)), torch.ones((2, 6)), torch.eye(8))

    # 5. protect_router_frag_layernorm_tail, compute_buckets_sha256, merge_moe_checkpoints_with_prose_guard validation
    with pytest.raises(ValueError, match="non-empty"):
        compute_buckets_sha256({})
    with pytest.raises(ValueError, match="ln_tail_policy"):
        protect_router_frag_layernorm_tail({"router_frag_0": torch.ones(8)}, [], hidden_size=2, ln_tail_policy="bad")
    with pytest.raises(ValueError, match="router_frag_"):
        protect_router_frag_layernorm_tail({"expert_0": torch.ones(8)}, [], hidden_size=2)
    with pytest.raises(ValueError, match="length"):
        protect_router_frag_layernorm_tail({"router_frag_0": torch.ones(2)}, [], hidden_size=4)
    with pytest.raises(ValueError, match="Shape mismatch"):
        protect_router_frag_layernorm_tail(
            {"router_frag_0": torch.ones(8)}, [{"router_frag_0": torch.ones(10)}], hidden_size=2
        )

    # Missing router_frag_0 in peer falls back cleanly, and ln_max_step_norm clamps guard_calibrated LN delta
    p_tail = {"router_frag_0": torch.ones(8), "router_frag_1": torch.ones(8)}
    c_tail = {"router_frag_1": torch.ones(8) + 2.0}
    fr_clamped, cl_meta = protect_router_frag_layernorm_tail(
        p_tail, [c_tail], hidden_size=2, ln_tail_policy="guard_calibrated", ln_max_step_norm=0.25, drop_rate=0.2
    )
    assert torch.equal(fr_clamped["router_frag_0"], p_tail["router_frag_0"])
    assert cl_meta["layernorm_drift_l2"] == pytest.approx(0.25, abs=1e-6)

    model, parent_b, _ = _build_5domain_crowd_moe_model(num_layers=1, rank=4, num_router_fragments=1, seed=20261110)
    probe_c = _make_domain_chunks(chunks_per_domain=2, seq_len=12, seed=20261111)
    with pytest.raises(TypeError, match="CrowdV1WithEngram"):
        merge_moe_checkpoints_with_prose_guard(nn.Linear(4, 4), parent_b, [parent_b], probe_chunks=probe_c)
    with pytest.raises(ValueError, match="non-empty"):
        merge_moe_checkpoints_with_prose_guard(model, {}, [parent_b], probe_chunks=probe_c)
    with pytest.raises(ValueError, match="non-empty"):
        merge_moe_checkpoints_with_prose_guard(model, parent_b, [], probe_chunks=probe_c)
    with pytest.raises(ValueError, match="probe_chunks or eval_fn"):
        merge_moe_checkpoints_with_prose_guard(model, parent_b, [parent_b])
    with pytest.raises(TypeError, match="TIESDAREMergerConfig"):
        merge_moe_checkpoints_with_prose_guard(
            model, parent_b, [parent_b], probe_chunks=probe_c, config="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="prose_activations"):
        merge_moe_checkpoints_with_prose_guard(
            model,
            parent_b,
            [parent_b],
            probe_chunks=probe_c,
            prose_activations=[],
            config=TIESDAREMergerConfig(score_from=4),
        )
    with pytest.raises(ValueError, match="spec_activations"):
        merge_moe_checkpoints_with_prose_guard(
            model,
            parent_b,
            [parent_b],
            probe_chunks=probe_c,
            prose_activations=[torch.ones((4, 40))],
            spec_activations=[],
            config=TIESDAREMergerConfig(score_from=4),
        )
    with pytest.raises(ValueError, match="non-empty"):
        merge_moe_checkpoints_with_prose_guard(model, parent_b, [{}], probe_chunks=probe_c)

    bad_peer = {k: v.clone() for k, v in parent_b.items()}
    bad_peer["expert_0"] = torch.ones(4)
    with pytest.raises(ValueError, match="Shape mismatch"):
        merge_moe_checkpoints_with_prose_guard(
            model, parent_b, [bad_peer], probe_chunks=probe_c, config=TIESDAREMergerConfig(score_from=4)
        )

    with pytest.raises(ValueError, match="non-finite evaluation NLL"):
        merge_moe_checkpoints_with_prose_guard(
            model,
            parent_b,
            [parent_b],
            eval_fn=lambda _m, _b: {"E0_general_fineweb": float("nan")},
        )

    # Exercise target_rank compression on single active peer + prose_activations path + missing expert in peer
    single_peer = {k: v.clone() for k, v in parent_b.items() if k != "expert_4"}
    single_peer["expert_1"] = single_peer["expert_1"] + 0.05
    _, single_rec = merge_moe_checkpoints_with_prose_guard(
        model,
        parent_b,
        [single_peer],
        eval_fn=lambda _m, b: {
            "E0_general_fineweb": 2.0,
            "E1_code": 2.0 if compute_buckets_sha256(b) == compute_buckets_sha256(parent_b) else 1.5,
        },
        prose_activations=[torch.randn((16, 40), generator=gen)],
        spec_activations=[torch.randn((16, 40), generator=gen)],
        config=TIESDAREMergerConfig(target_rank=2, ln_tail_policy="freeze_parent"),
    )
    assert single_rec["accepted"] is True
    assert single_rec["expert_ranks_preserved"]["expert_1"] == 2

    with pytest.raises(ValueError, match="target_rank"):
        merge_moe_checkpoints_with_prose_guard(
            model,
            parent_b,
            [single_peer],
            eval_fn=lambda _m, _b: {"E0_general_fineweb": 2.0, "E1_code": 1.5},
            config=TIESDAREMergerConfig(target_rank=8, ln_tail_policy="freeze_parent"),
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        merge_moe_checkpoints_with_prose_guard(
            model, parent_b, [single_peer], eval_fn=lambda _m, _b: {}
        )
    with pytest.raises(ValueError, match="at least one domain"):
        merge_moe_checkpoints_with_prose_guard(
            model, parent_b, [single_peer], eval_fn=lambda _m, _b: {"nll_mean": 1.5}
        )

    short_parent = {k: v.clone() for k, v in parent_b.items()}
    short_parent["expert_0"] = torch.ones(4)
    with pytest.raises(ValueError, match="smaller than model layout"):
        merge_moe_checkpoints_with_prose_guard(
            model,
            short_parent,
            [short_parent],
            eval_fn=lambda _m, _b: {"E0_general_fineweb": 2.0},
        )
    long_parent = {k: v.clone() for k, v in parent_b.items()}
    long_parent["expert_0"] = torch.cat([long_parent["expert_0"], torch.ones(4)])
    with pytest.raises(ValueError, match="does not match model layout"):
        merge_moe_checkpoints_with_prose_guard(
            model,
            long_parent,
            [long_parent],
            eval_fn=lambda _m, _b: {"E0_general_fineweb": 2.0},
        )

    # Partial-layer MoE model (layer 0 stays dense CrowdRuntimeMLP, layer 1 is CrowdLoRAMoEMLP)
    partial_cfg = CrowdV1RuntimeConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    partial_model = CrowdV1WithEngram(partial_cfg)
    upcycle_crowd_v1_to_sparse_moe(
        partial_model, num_experts=3, top_k=1, lora_rank=2, layer_indices=[1], seed=20261115
    )
    p_base = extract_moe_trainable_buckets(partial_model, num_router_fragments=1)
    p_peer = {k: v.clone() for k, v in p_base.items()}
    p_peer["expert_1"] = p_peer["expert_1"] + 0.05
    p_chunks = [
        {"expert_id": "E0_general_fineweb", "ids": [0, 1, 0, 1, 0, 1, 0, 1]},
        {"expert_id": "E1_code", "ids": [8, 9, 10, 11, 8, 9, 10, 11]},
    ]
    _, partial_rec = merge_moe_checkpoints_with_prose_guard(
        partial_model,
        p_base,
        [p_peer],
        probe_chunks=p_chunks,
        config=TIESDAREMergerConfig(score_from=2, ln_tail_policy="freeze_parent", require_energy_improvement=False),
        num_router_fragments=1,
    )
    assert partial_rec["accepted"] is True

    # 6. AGPL-3.0 header, ZeroStubAudit (if installed), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn06", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn06", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/ties_dare_moe_merger.py", TARGET_MODULE_PATH),
        ("tests/test_ties_dare_moe_merger_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

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

    telemetry_json = json.dumps(
        {"nll_mean": 1.12, "sign_consensus_ratio": 0.88, "anse_energy": -0.34}
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1112,
        pr_title="feat(swarm): TIES/DARE sign-consensus and prose-nullspace task-arithmetic merger for distributed MoE rounds",
        pr_body="Implements NEURAL-CTRL-DYN-06 TIES/DARE sign-consensus and prose-nullspace MoE merger with deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): TIES/DARE sign-consensus and prose-nullspace task-arithmetic merger for distributed MoE rounds"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
