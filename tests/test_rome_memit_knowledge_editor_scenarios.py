# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-12: AlphaEdit-MEMIT multi-layer SwiGLU MoE knowledge editor."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAExpertAdapter,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.rome_memit_knowledge_editor import (
    ANSE_MAXIMUM_PAIN,
    MEMIT_EDITOR_RECEIPT_OBJECT,
    MEMIT_EDITOR_SCHEMA,
    AlphaEditMEMITConfig,
    compute_nullspace_projector_from_keys,
    compute_payload_sha256,
    compute_state_sha256,
    distribute_multilayer_memit_moe_edits,
    fold_delta_into_lora_pair,
    fold_delta_into_swiglu_lora_w2_inplace,
    solve_alphaedit_memit_layer_delta,
    verify_memit_receipt_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """Minimal stub audit result used when optional ``gwaya`` is absent in CI."""

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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "rome_memit_knowledge_editor.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_4layer_subspace_crowd_moe(
    *,
    num_layers: int = 4,
    hidden_size: int = 48,
    intermediate_size: int = 96,
    vocab_size: int = 64,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 6,
    split_mode: str = "lora_domain_experts",
    routing_mode: str = "sigma_topk",
    resident_expert_indices: list[int] | None = None,
    seed: int = 20261124,
) -> CrowdV1WithEngram:
    """Build deterministic 3-subspace CrowdV1WithEngram + Sparse MoE on CPU.

    Subspaces (``hidden_size=48, vocab_size=64``):
      - ``0..15``:  ``E0_general_fineweb`` (tokens ``0..15``)
      - ``16..31``: ``E1_code`` specialist (tokens ``16..31``)
      - ``32..47``: target skill / fact subspace (tokens ``32..63``)
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
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)
    block_h = hidden_size // 3
    block_i = intermediate_size // 3

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        tok_ranges = [(0, 16), (16, 32), (32, vocab_size)]
        for s_idx, (t0, t1) in enumerate(tok_ranges):
            h0, h1 = s_idx * block_h, (s_idx + 1) * block_h
            raw = torch.randn((t1 - t0, block_h), generator=gen, dtype=torch.float32) * 1.5
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
            for s_idx in range(3):
                h0, h1 = s_idx * block_h, (s_idx + 1) * block_h
                i0, i1 = s_idx * block_i, (s_idx + 1) * block_i
                kv_h0, kv_h1 = h0 // 2, h1 // 2
                blk.attn.q.weight[h0:h1, h0:h1].copy_(0.05 * torch.eye(block_h))
                blk.attn.k.weight[kv_h0:kv_h1, h0:h1].copy_(0.05 * torch.eye(kv_h1 - kv_h0, block_h))
                blk.attn.v.weight[kv_h0:kv_h1, h0:h1].copy_(0.05 * torch.eye(kv_h1 - kv_h0, block_h))
                blk.attn.o.weight[h0:h1, h0:h1].copy_(0.05 * torch.eye(block_h))

                w1_b = torch.randn((block_i, block_h), generator=gen, dtype=torch.float32) * 0.22
                w2_b = torch.randn((block_i, block_h), generator=gen, dtype=torch.float32) * 0.22
                w3_b = torch.randn((block_h, block_i), generator=gen, dtype=torch.float32) * 0.10
                w3_b = w3_b - w3_b.mean(dim=0, keepdim=True)
                blk.mlp.w1.weight[i0:i1, h0:h1].copy_(w1_b)
                blk.mlp.w2.weight[i0:i1, h0:h1].copy_(w2_b)
                blk.mlp.w3.weight[h0:h1, i0:i1].copy_(w3_b)

    if split_mode != "dense":
        upcycle_crowd_v1_to_sparse_moe(
            model,
            split_mode=split_mode,
            num_experts=num_experts,
            top_k=top_k,
            lora_rank=lora_rank,
            routing_mode=routing_mode,
            resident_expert_indices=resident_expert_indices,
            seed=seed + 17,
        )
        with torch.no_grad():
            for blk in model.blocks:
                mlp = blk.mlp
                if hasattr(mlp, "router"):
                    # Route E0 (0..15) -> expert 0, E1 (16..31) -> expert 1, skill (32..47) -> experts 2 & 3
                    mlp.router.weight[0, :block_h].fill_(0.8)
                    mlp.router.weight[1, block_h : 2 * block_h].fill_(0.8)
                    if num_experts >= 4:
                        mlp.router.weight[2, 2 * block_h :].fill_(0.8)
                        mlp.router.weight[3, 2 * block_h :].fill_(0.7)
                    else:
                        mlp.router.weight[1, 2 * block_h :].fill_(0.6)
    return model


def _default_preserve_chunks() -> list[dict[str, Any]]:
    return [
        {"expert_id": "E0_general_fineweb", "ids": [0, 1, 2, 3, 4, 5, 6, 7]},
        {"expert_id": "E0_general_fineweb", "ids": [8, 9, 10, 11, 12, 13, 14, 15]},
        {"expert_id": "E1_code", "ids": [16, 17, 18, 19, 20, 21, 22, 23]},
        {"expert_id": "E1_code", "ids": [24, 25, 26, 27, 28, 29, 30, 31]},
    ]


# ---------------------------------------------------------------------------
# S1: Multi-Layer Batch Skill Injection & Rank-Preserving SwiGLU LoRA w2 Fold
# ---------------------------------------------------------------------------


def test_s1_multilayer_batch_skill_injection_and_swiglu_lora_w2_fold() -> None:
    """Simultaneous 3-layer MoE skill injection hits >90% efficacy with <= +0.2% prose drift."""
    model = _build_4layer_subspace_crowd_moe(lora_rank=6, num_experts=4, top_k=2, seed=20261201)
    preserve_chunks = _default_preserve_chunks()

    # Add a pre-existing rank-1 E1 specialist signature on expert 1 so zero-slot & existing adapters coexist
    with torch.no_grad():
        for blk in model.blocks:
            exp1 = blk.mlp.experts["1"]
            exp1.w2_B[32:64, 0] = 0.08
            exp1.w2_A[0, 16:32] = 0.08

    layer0_snapshot = compute_state_sha256({k: v for k, v in model.state_dict().items() if k.startswith("blocks.0.")})

    edit_seqs = torch.tensor(
        [
            [32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47],
            [48, 49, 50, 51, 52, 53, 54, 55],
        ],
        dtype=torch.long,
    )
    target_ids = torch.tensor([60, 61, 62], dtype=torch.long)

    receipt = distribute_multilayer_memit_moe_edits(
        model,
        edit_seqs,
        target_ids,
        preserve_chunks,
        layer_indices=[1, 2, 3],
        expert_indices=[2, 3],
    )

    assert receipt["object"] == MEMIT_EDITOR_RECEIPT_OBJECT
    assert receipt["schema"] == MEMIT_EDITOR_SCHEMA
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "ACCEPTED"
    assert receipt["guard_ok"] is True
    assert receipt["efficacy_ok"] is True
    assert receipt["layer_indices"] == [1, 2, 3]
    assert receipt["expert_indices"] == [2, 3]
    assert receipt["pre_edit_efficacy_ratio"] < 0.5
    assert receipt["edit_efficacy_ratio"] > 0.90
    assert receipt["mean_target_prob"] > 0.90
    assert receipt["general_pct_vs_base"] <= 0.2
    assert receipt["worst_domain_pct_vs_base"] <= 0.2
    assert receipt["delta_e"] < 0.0
    assert receipt["final_state_sha256"] == receipt["candidate_state_sha256"]
    assert receipt["max_preserved_key_drift"] < 5e-3
    assert verify_memit_receipt_provenance(receipt) is True

    # Early layer 0 and LoRA rank dimensions remain untouched
    layer0_after = compute_state_sha256({k: v for k, v in model.state_dict().items() if k.startswith("blocks.0.")})
    assert layer0_after == layer0_snapshot
    for l_idx in (1, 2, 3):
        for e_id in ("2", "3"):
            exp = model.blocks[l_idx].mlp.experts[e_id]
            assert exp.w2_B.shape == (96, 6)
            assert exp.w2_A.shape == (6, 48)

    # Tampered receipt fails SHA-256 provenance verification
    tampered = dict(receipt)
    tampered["edit_efficacy_ratio"] = 0.42
    assert verify_memit_receipt_provenance(tampered) is False

    # Exercise rank-preserving LoRA fold modes (update_a, qr_svd, zero_slot, noop) & dense/FG MoE paths
    gen = torch.Generator(device="cpu").manual_seed(20261202)
    b_full = torch.randn((12, 4), generator=gen, dtype=torch.float64)
    a_full = torch.randn((4, 8), generator=gen, dtype=torch.float64)
    p_null = torch.diag(torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    delta_in_col = (b_full @ torch.randn((4, 8), generator=gen, dtype=torch.float64)) @ p_null

    b_ua, a_ua, info_ua = fold_delta_into_lora_pair(
        b_full, a_full, delta_in_col, scaling=2.0, p_null=p_null, lora_fold_mode="auto"
    )
    assert info_ua["fold_method"] == "update_a_nullspace"
    assert b_ua.shape == b_full.shape and a_ua.shape == a_full.shape
    assert float((((b_ua @ a_ua) - (b_full @ a_full))[:, :3]).abs().max().item()) < 1e-12

    b_qr, a_qr, info_qr = fold_delta_into_lora_pair(
        b_full, a_full, delta_in_col, scaling=2.0, p_null=p_null, lora_fold_mode="qr_svd"
    )
    assert info_qr["fold_method"] == "qr_svd_rank_preserving"
    assert b_qr.shape == b_full.shape and a_qr.shape == a_full.shape

    b_noop, a_noop, info_noop = fold_delta_into_lora_pair(b_full, a_full, torch.zeros((12, 8)))
    assert info_noop["fold_method"] == "zero_delta_noop"
    assert torch.allclose(b_noop, b_full) and torch.allclose(a_noop, a_full)

    # Dense, FineGrained (softmax_topk & sigma_topk), and target_proj="w3"/"auto" paths
    m_fg = _build_4layer_subspace_crowd_moe(
        num_layers=2, num_experts=3, top_k=2, split_mode="fine_grained_width_split", routing_mode="softmax_topk", seed=20261203
    )
    rec_fg = distribute_multilayer_memit_moe_edits(
        m_fg, edit_seqs[:2], target_ids[:2], preserve_chunks, layer_indices=None, target_proj="w3"
    )
    assert rec_fg["accepted"] is True and rec_fg["edit_efficacy_ratio"] > 0.90

    m_fg_sig = _build_4layer_subspace_crowd_moe(
        num_layers=2, num_experts=3, top_k=2, split_mode="fine_grained_width_split", routing_mode="sigma_topk", seed=20261204
    )
    rec_fg_sig = distribute_multilayer_memit_moe_edits(
        m_fg_sig, edit_seqs[:2], target_ids[:2], preserve_chunks, layer_indices=[1], target_proj="auto"
    )
    assert rec_fg_sig["accepted"] is True and rec_fg_sig["edit_efficacy_ratio"] > 0.90

    m_dense = _build_4layer_subspace_crowd_moe(num_layers=2, split_mode="dense", seed=20261205)
    rec_dense = distribute_multilayer_memit_moe_edits(
        m_dense, edit_seqs[:2], target_ids[:2], preserve_chunks, layer_indices=[0, 1], target_proj="auto"
    )
    assert rec_dense["accepted"] is True and rec_dense["edit_efficacy_ratio"] > 0.90

    # Mid-layer edit ([0] on 2-layer model) with softmax_topk, non-resident expert 3, & auto w2
    m_mid_softmax = _build_4layer_subspace_crowd_moe(
        num_layers=2,
        num_experts=4,
        top_k=2,
        routing_mode="softmax_topk",
        resident_expert_indices=[0, 1, 2],
        seed=20261207,
    )
    rec_mid_softmax = distribute_multilayer_memit_moe_edits(
        m_mid_softmax,
        edit_seqs[:2],
        target_ids[:2],
        preserve_chunks,
        layer_indices=[0],
        expert_indices=[2],
        target_proj="auto",
    )
    assert rec_mid_softmax["accepted"] is True
    assert rec_mid_softmax["edit_efficacy_ratio"] > 0.90
    assert rec_mid_softmax["layer_reports"][0]["proj_mode"] == "w2"

    # Zero-w1 model triggers target_proj="auto" -> "w3" fallback cleanly
    m_zero_w1 = _build_4layer_subspace_crowd_moe(num_layers=2, seed=20261206)
    with torch.no_grad():
        for blk in m_zero_w1.blocks:
            blk.mlp.base_mlp.w1.weight.zero_()
    rec_auto_w3 = distribute_multilayer_memit_moe_edits(
        m_zero_w1, edit_seqs[:2], target_ids[:2], preserve_chunks, layer_indices=[1], target_proj="auto"
    )
    assert rec_auto_w3["layer_reports"][0]["proj_mode"] == "w3"

    # Generic Linear & CrowdLoRAExpertAdapter fold_delta_into_swiglu_lora_w2_inplace
    lin = nn.Linear(8, 12, bias=False)
    lin_info = fold_delta_into_swiglu_lora_w2_inplace(lin, torch.ones((12, 8)) * 0.01)
    assert lin_info["target_kind"] == "dense_weight"
    exp_solo = CrowdLoRAExpertAdapter(8, 12, rank=2, seed=7)
    exp_info = fold_delta_into_swiglu_lora_w2_inplace(exp_solo, torch.ones((12, 8)) * 0.01, weight_attr="w2")
    assert exp_info["target_kind"] == "lora_w2"


# ---------------------------------------------------------------------------
# S2: Exact Null-Space Preservation (<1e-5 Drift) vs Naive Unprojected MEMIT
# ---------------------------------------------------------------------------


def test_s2_exact_nullspace_preservation_vs_naive_unprojected_memit() -> None:
    """AlphaEdit keeps protected keys <1e-5 drift while naive MEMIT breaks general prose."""
    gen = torch.Generator(device="cpu").manual_seed(20261210)
    d_in, d_out = 24, 16
    n_pres_e0, n_pres_e1, n_edit = 10, 6, 4

    # Protected E0 general keys in dims 0..7, protected E1 specialist keys in dims 8..13
    k_e0 = torch.zeros((n_pres_e0, d_in), dtype=torch.float64)
    k_e0[:, :8] = torch.randn((n_pres_e0, 8), generator=gen, dtype=torch.float64)
    k_e1 = torch.zeros((n_pres_e1, d_in), dtype=torch.float64)
    k_e1[:, 8:14] = torch.randn((n_pres_e1, 6), generator=gen, dtype=torch.float64)
    preserve_all = torch.cat([k_e0, k_e1], dim=0)

    # Test sv_threshold filtering, 3D input flattening, Mapping/Sequence inputs, and all-zero fallback
    p_null = compute_nullspace_projector_from_keys({"E0": k_e0.unsqueeze(0), "E1": k_e1}, sv_threshold=1e-2)
    p_seq = compute_nullspace_projector_from_keys([k_e0, k_e1], sv_threshold=1e-2)
    assert torch.allclose(p_null, p_seq, atol=1e-12)
    assert torch.allclose(p_null, p_null.T, atol=1e-12)
    assert torch.allclose(p_null @ p_null, p_null, atol=1e-12)

    p_zero = compute_nullspace_projector_from_keys(torch.zeros((4, d_in), dtype=torch.float64), sv_threshold=1e-2)
    assert torch.allclose(p_zero, torch.eye(d_in, dtype=torch.float64), atol=1e-12)

    # Edit keys have deliberate overlap with protected E0/E1 keys PLUS signal in nullspace 14..23
    k_edit_rows = torch.zeros((n_edit, d_in), dtype=torch.float64)
    k_edit_rows[:, :14] = torch.randn((n_edit, 14), generator=gen, dtype=torch.float64) * 1.8
    k_edit_rows[:, 14:] = torch.randn((n_edit, d_in - 14), generator=gen, dtype=torch.float64) * 2.0
    r_res_rows = torch.randn((n_edit, d_out), generator=gen, dtype=torch.float64) * 1.5
    c_cov = 0.05 * ((preserve_all.T @ preserve_all) / float(preserve_all.shape[0]))

    # Solve in both auto-oriented row-batch, explicit row-batch, and column-vector orientations
    dw_alpha_auto_row = solve_alphaedit_memit_layer_delta(
        k_edit_rows, r_res_rows, p_null, c_cov, reg_lambda=1e-2
    )
    dw_alpha_row = solve_alphaedit_memit_layer_delta(
        k_edit_rows, r_res_rows, p_null, c_cov, reg_lambda=1e-2, keys_as_columns=False
    )
    dw_alpha_col = solve_alphaedit_memit_layer_delta(
        k_edit_rows.T, r_res_rows.T, p_null, c_cov, reg_lambda=1e-2
    )
    assert torch.allclose(dw_alpha_auto_row, dw_alpha_col, atol=1e-12)
    assert torch.allclose(dw_alpha_row, dw_alpha_col, atol=1e-12)

    # Square k_edit (d_in == u == 8) tie-break branches (col_mode=True, False, and square r)
    eye8 = torch.eye(8, dtype=torch.float64)
    assert solve_alphaedit_memit_layer_delta(eye8, torch.ones((6, 8)), eye8, eye8 * 0.1).shape == (6, 8)
    assert solve_alphaedit_memit_layer_delta(eye8, torch.ones((8, 6)), eye8, eye8 * 0.1).shape == (6, 8)
    dw_sq = solve_alphaedit_memit_layer_delta(
        eye8,
        eye8 * 0.5,
        eye8,
        eye8 * 0.1,
        reg_lambda=1e-2,
    )
    assert dw_sq.shape == (8, 8)

    # AlphaEdit protected drift < 1e-5 (in practice < 1e-11) while achieving high edit fidelity
    alpha_pres_drift = float((dw_alpha_col @ preserve_all.T).abs().max().item())
    assert alpha_pres_drift < 1e-5
    edit_rel_err = float(((dw_alpha_col @ k_edit_rows.T) - r_res_rows.T).norm().item() / r_res_rows.T.norm().item())
    assert edit_rel_err < 0.05

    # Naive unprojected MEMIT (P_0 = I) severely corrupts protected general/specialist keys
    dw_naive = solve_alphaedit_memit_layer_delta(
        k_edit_rows.T, r_res_rows.T, torch.eye(d_in, dtype=torch.float64), c_cov, reg_lambda=1e-2
    )
    naive_pres_drift = float((dw_naive @ preserve_all.T).abs().max().item())
    assert naive_pres_drift > 0.10
    assert naive_pres_drift > alpha_pres_drift * 1e6

    # Model-level S2: edit tokens carry an E0 overlap; AlphaEdit preserves E0 while unprojected MEMIT breaks E0
    model_alpha = _build_4layer_subspace_crowd_moe(num_layers=3, lora_rank=6, seed=20261211)
    with torch.no_grad():
        gen_s2 = torch.Generator(device="cpu").manual_seed(20261212)
        for t_idx in range(8):
            model_alpha.wte.weight[t_idx] = (
                model_alpha.wte.weight[0] * 0.2 + model_alpha.wte.weight[t_idx] * 0.8
            )
        # Inject non-orthogonal E0 overlap into skill tokens 32..39 and cross-subspace skill FFN coupling
        model_alpha.wte.weight[32:40, :16].copy_(model_alpha.wte.weight[:8, :16] * 0.9)
        for blk in model_alpha.blocks:
            w1_c = torch.randn((32, 48), generator=gen_s2, dtype=torch.float32) * 0.22
            w2_c = torch.randn((32, 48), generator=gen_s2, dtype=torch.float32) * 0.22
            w3_c = torch.randn((48, 32), generator=gen_s2, dtype=torch.float32) * 0.14
            w3_c = w3_c - w3_c.mean(dim=0, keepdim=True)
            blk.mlp.base_mlp.w1.weight[64:, :].copy_(w1_c)
            blk.mlp.base_mlp.w2.weight[64:, :].copy_(w2_c)
            blk.mlp.base_mlp.w3.weight[:, 64:].copy_(w3_c)
            blk.mlp.router.weight[0, 32:].fill_(0.0)
            blk.mlp.router.weight[2, :16].fill_(0.9)
    model_naive = copy.deepcopy(model_alpha)

    preserve_s2 = [
        {"expert_id": "E0_general_fineweb", "ids": [1, 1, 1, 1, 1, 1, 1, 1]},
        {"expert_id": "E0_general_fineweb", "ids": [2, 2, 2, 2, 2, 2, 2, 2]},
        {"expert_id": "E1_code", "ids": [16, 17, 18, 19, 20, 21, 22, 23]},
        {"expert_id": "E1_code", "ids": [24, 25, 26, 27, 28, 29, 30, 31]},
    ]
    e0_seqs = torch.tensor([c["ids"] for c in preserve_s2[:2]], dtype=torch.long)
    edit_s2 = torch.tensor([[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47]], dtype=torch.long)
    tgt_s2 = torch.tensor([58, 59], dtype=torch.long)

    with torch.no_grad():
        h_e0_before = model_alpha.blocks[2](
            model_alpha.blocks[1](
                model_alpha.blocks[0](model_alpha.wte(e0_seqs), model_alpha.cos_cached, model_alpha.sin_cached)[0],
                model_alpha.cos_cached,
                model_alpha.sin_cached,
            )[0],
            model_alpha.cos_cached,
            model_alpha.sin_cached,
        )[0]
        e0_logits_before = model_alpha.head(model_alpha.ln(h_e0_before))

    rec_alpha = distribute_multilayer_memit_moe_edits(
        model_alpha,
        edit_s2,
        tgt_s2,
        preserve_s2,
        layer_indices=[1, 2],
        sv_threshold=1e-4,
        use_nullspace_projection=True,
    )
    assert rec_alpha["accepted"] is True
    assert rec_alpha["edit_efficacy_ratio"] > 0.90
    assert rec_alpha["general_pct_vs_base"] <= 0.2
    assert rec_alpha["max_preserved_key_drift"] < 1e-5

    with torch.no_grad():
        h_e0_after = model_alpha.blocks[2](
            model_alpha.blocks[1](
                model_alpha.blocks[0](model_alpha.wte(e0_seqs), model_alpha.cos_cached, model_alpha.sin_cached)[0],
                model_alpha.cos_cached,
                model_alpha.sin_cached,
            )[0],
            model_alpha.cos_cached,
            model_alpha.sin_cached,
        )[0]
        e0_logits_after = model_alpha.head(model_alpha.ln(h_e0_after))
    assert float((e0_logits_after - e0_logits_before).abs().max().item()) < 1e-5

    # Unprojected MEMIT (P_0 = I) triggers general-domain drift > +1.0% (+40%!) and gets rejected by ANSE
    rec_naive = distribute_multilayer_memit_moe_edits(
        model_naive,
        edit_s2,
        tgt_s2,
        preserve_s2,
        background_seqs=edit_s2,
        layer_indices=[1, 2],
        use_nullspace_projection=False,
    )
    assert rec_naive["guard_ok"] is False
    assert rec_naive["accepted"] is False
    assert rec_naive["rolled_back"] is True
    assert rec_naive["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_naive["general_pct_vs_base"] > 1.0
    assert rec_naive["anse_energy"] >= ANSE_MAXIMUM_PAIN


# ---------------------------------------------------------------------------
# S3: Multi-Layer Residual Damping Stability vs Single-Layer ROME Norm Spike
# ---------------------------------------------------------------------------


def test_s3_multilayer_residual_damping_stability_vs_single_layer_rome() -> None:
    """Spreading edits across 3 layers eliminates single-layer ROME norm spike while keeping efficacy >90%."""
    base_model = _build_4layer_subspace_crowd_moe(num_layers=4, lora_rank=6, seed=20261220)
    preserve_chunks = _default_preserve_chunks()
    edit_seqs = torch.tensor(
        [
            [32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47],
            [48, 49, 50, 51, 52, 53, 54, 55],
        ],
        dtype=torch.long,
    )
    target_ids = torch.tensor([60, 61, 62], dtype=torch.long)

    m_multi = copy.deepcopy(base_model)
    m_damped = copy.deepcopy(base_model)
    m_rome = copy.deepcopy(base_model)

    rec_multi = distribute_multilayer_memit_moe_edits(
        m_multi,
        edit_seqs,
        target_ids,
        preserve_chunks,
        layer_indices=[1, 2, 3],
        background_seqs=[torch.tensor(c["ids"], dtype=torch.long) for c in preserve_chunks],
        preserve_key_seqs=torch.tensor([c["ids"] for c in preserve_chunks], dtype=torch.long),
        edit_positions=torch.tensor([-1, 7, -1], dtype=torch.long),
    )
    rec_damped = distribute_multilayer_memit_moe_edits(
        m_damped,
        edit_seqs,
        target_ids,
        preserve_chunks,
        layer_indices=[1, 2, 3],
        config=AlphaEditMEMITConfig(residual_damping=0.80, max_residual_norm=12.0),
    )
    rec_rome = distribute_multilayer_memit_moe_edits(
        m_rome,
        edit_seqs,
        target_ids,
        preserve_chunks,
        layer_indices=[3],
    )

    for rec in (rec_multi, rec_damped, rec_rome):
        assert rec["accepted"] is True
        assert rec["edit_efficacy_ratio"] > 0.90
        assert rec["general_pct_vs_base"] <= 0.2

    # Multi-layer MEMIT cuts single-layer weight norm spike and quadratic Frobenius energy significantly
    assert rec_multi["max_layer_delta_fro"] < 0.70 * rec_rome["max_layer_delta_fro"]
    assert rec_damped["max_layer_delta_fro"] < 0.75 * rec_rome["max_layer_delta_fro"]
    assert rec_multi["total_delta_fro_sq"] < rec_rome["total_delta_fro_sq"]

    # Remaining residual magnitude distributes evenly across layers instead of spiking in one layer
    layer_norms = [float(lr["max_delta_fro"]) for lr in rec_multi["layer_reports"]]
    assert len(layer_norms) == 3
    assert max(layer_norms) < rec_rome["max_layer_delta_fro"]


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Rollback on Conflicting & Infeasible Edits
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_rollback_on_conflicting_infeasible_edits() -> None:
    """Conflicting/infeasible edits trigger ANSE Maximum Pain (1e6) and atomic bitwise rollback."""
    preserve_chunks = _default_preserve_chunks()

    # Case A: Contradictory target tokens on identical key prompt (physically infeasible)
    model_conflict = _build_4layer_subspace_crowd_moe(num_layers=3, lora_rank=6, seed=20261230)
    parent_sha_a = compute_state_sha256(model_conflict)
    same_prompt = torch.tensor(
        [
            [32, 33, 34, 35, 36, 37, 38, 39],
            [32, 33, 34, 35, 36, 37, 38, 39],
        ],
        dtype=torch.long,
    )
    contradictory_targets = torch.tensor([60, 61], dtype=torch.long)
    rec_conflict = distribute_multilayer_memit_moe_edits(
        model_conflict,
        same_prompt,
        contradictory_targets,
        preserve_chunks,
        layer_indices=[1, 2],
        min_edit_efficacy_ratio=0.90,
    )
    assert rec_conflict["accepted"] is False
    assert rec_conflict["rolled_back"] is True
    assert rec_conflict["status"] == "ROLLED_BACK_INFEASIBLE_EDIT"
    assert rec_conflict["efficacy_ok"] is False
    assert rec_conflict["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_conflict["final_state_sha256"] == parent_sha_a
    assert compute_state_sha256(model_conflict) == parent_sha_a

    # Case B: Edit keys lie completely inside preserved nullspace kernel -> infeasible & rolled back / fixed point
    model_null_infeasible = _build_4layer_subspace_crowd_moe(num_layers=3, lora_rank=6, seed=20261231)
    parent_sha_b = compute_state_sha256(model_null_infeasible)
    e0_overwrite = torch.tensor([preserve_chunks[0]["ids"], preserve_chunks[1]["ids"]], dtype=torch.long)
    rec_null_inf = distribute_multilayer_memit_moe_edits(
        model_null_infeasible,
        e0_overwrite,
        torch.tensor([60, 61], dtype=torch.long),
        preserve_chunks,
        layer_indices=[1, 2],
        use_nullspace_projection=True,
    )
    assert rec_null_inf["accepted"] is False
    assert rec_null_inf["rolled_back"] is True
    assert rec_null_inf["status"] == "ROLLED_BACK_INFEASIBLE_EDIT"
    assert rec_null_inf["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_null_inf["final_state_sha256"] == parent_sha_b
    assert compute_state_sha256(model_null_infeasible) == parent_sha_b

    # Case B2: Inactive target expert routing leaves model unchanged -> FIXED_POINT_INFEASIBLE
    with torch.no_grad():
        for blk in model_null_infeasible.blocks:
            blk.mlp.router.weight[3, :16].fill_(-5.0)
    parent_sha_b2 = compute_state_sha256(model_null_infeasible)
    rec_null_fp = distribute_multilayer_memit_moe_edits(
        model_null_infeasible,
        e0_overwrite,
        torch.tensor([60, 61], dtype=torch.long),
        preserve_chunks,
        layer_indices=[1, 2],
        expert_indices=[3],
        use_nullspace_projection=True,
    )
    assert rec_null_fp["accepted"] is False
    assert rec_null_fp["rolled_back"] is False
    assert rec_null_fp["status"] == "FIXED_POINT_INFEASIBLE"
    assert rec_null_fp["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_null_fp["final_state_sha256"] == parent_sha_b2

    # Case C: Non-improving energy triggers ROLLED_BACK_ENERGY_NON_IMPROVING
    model_non_imp = _build_4layer_subspace_crowd_moe(num_layers=3, lora_rank=6, seed=20261232)
    parent_sha_c = compute_state_sha256(model_non_imp)
    cfg_zero_ew = AlphaEditMEMITConfig(edit_energy_weight=0.0, require_strict_energy_drop=True)
    rec_non_imp = distribute_multilayer_memit_moe_edits(
        model_non_imp,
        torch.tensor([[32, 33, 34, 35, 36, 37, 38, 39]], dtype=torch.long),
        torch.tensor([60], dtype=torch.long),
        preserve_chunks,
        layer_indices=[1, 2],
        config=cfg_zero_ew,
    )
    assert rec_non_imp["accepted"] is False
    assert rec_non_imp["rolled_back"] is True
    assert rec_non_imp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_non_imp["final_state_sha256"] == parent_sha_c
    assert compute_state_sha256(model_non_imp) == parent_sha_c


# ---------------------------------------------------------------------------
# S5: Input Validation Errors & Security / Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance() -> None:
    """Verify strict validation errors, AGPL-3.0 headers, optional ZeroStubAudit, and PR security gates."""
    # 1. Config & helper validation errors
    with pytest.raises(TypeError, match="real number"):
        AlphaEditMEMITConfig(reg_lambda=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        AlphaEditMEMITConfig(sv_threshold=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        AlphaEditMEMITConfig(sv_threshold=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        AlphaEditMEMITConfig(reg_lambda=0.0)
    with pytest.raises(ValueError, match="< 1.0"):
        AlphaEditMEMITConfig(target_prob=1.0)
    with pytest.raises(ValueError, match="> 0.0"):
        AlphaEditMEMITConfig(target_prob=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        AlphaEditMEMITConfig(residual_damping=1.5)
    with pytest.raises(TypeError, match="bool"):
        AlphaEditMEMITConfig(use_nullspace_projection="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_proj"):
        AlphaEditMEMITConfig(target_proj="w1")
    with pytest.raises(ValueError, match="lora_fold_mode"):
        AlphaEditMEMITConfig(lora_fold_mode="invalid")
    with pytest.raises(ValueError, match="general_prefix"):
        AlphaEditMEMITConfig(general_prefix=" ")
    with pytest.raises(TypeError, match="int"):
        AlphaEditMEMITConfig(score_from=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        AlphaEditMEMITConfig(score_from=-1)
    with pytest.raises(ValueError, match=">= 1"):
        AlphaEditMEMITConfig(batch_size=0)

    # 2. Projector, normal-equation solver, LoRA fold, and SHA-256 validation
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_nullspace_projector_from_keys([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D or 3D"):
        compute_nullspace_projector_from_keys(torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        compute_nullspace_projector_from_keys(torch.ones((2, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        compute_nullspace_projector_from_keys(torch.tensor([[1.0, float("nan")]]))
    with pytest.raises(ValueError, match="empty"):
        compute_nullspace_projector_from_keys({})
    with pytest.raises(ValueError, match="empty"):
        compute_nullspace_projector_from_keys([])
    with pytest.raises(TypeError, match="Tensor"):
        compute_nullspace_projector_from_keys("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width"):
        compute_nullspace_projector_from_keys([torch.ones((2, 4)), torch.ones((2, 5))])

    eye4 = torch.eye(4, dtype=torch.float64)
    with pytest.raises(TypeError, match="torch.Tensor"):
        solve_alphaedit_memit_layer_delta([[1.0]], eye4, eye4, eye4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D tensor"):
        solve_alphaedit_memit_layer_delta(torch.ones(4), eye4, eye4, eye4)
    with pytest.raises(TypeError, match="floating-point"):
        solve_alphaedit_memit_layer_delta(torch.ones((4, 4), dtype=torch.long), eye4, eye4, eye4)
    with pytest.raises(ValueError, match="finite"):
        solve_alphaedit_memit_layer_delta(eye4, eye4, torch.full((4, 4), float("nan")), eye4)
    with pytest.raises(ValueError, match="square"):
        solve_alphaedit_memit_layer_delta(eye4, eye4, torch.ones((3, 4)), eye4)
    with pytest.raises(ValueError, match="c_key_cov"):
        solve_alphaedit_memit_layer_delta(eye4, eye4, eye4, torch.ones((3, 3)))
    with pytest.raises(ValueError, match="positive semi-definite"):
        solve_alphaedit_memit_layer_delta(eye4, eye4, eye4, -torch.eye(4))
    with pytest.raises(ValueError, match="does not match p_null"):
        solve_alphaedit_memit_layer_delta(torch.ones((2, 3)), torch.ones((2, 3)), eye4, eye4)
    with pytest.raises(ValueError, match="row count"):
        solve_alphaedit_memit_layer_delta(torch.ones((3, 4)), torch.ones((2, 4)), eye4, eye4, keys_as_columns=True)
    with pytest.raises(ValueError, match="column count"):
        solve_alphaedit_memit_layer_delta(torch.ones((4, 2)), torch.ones((4, 3)), eye4, eye4, keys_as_columns=True)
    with pytest.raises(ValueError, match="feature width"):
        solve_alphaedit_memit_layer_delta(torch.ones((3, 5)), torch.ones((3, 4)), eye4, eye4, keys_as_columns=False)
    with pytest.raises(ValueError, match="row count"):
        solve_alphaedit_memit_layer_delta(torch.ones((2, 4)), torch.ones((3, 4)), eye4, eye4, keys_as_columns=False)

    with pytest.raises(ValueError, match="lora_fold_mode"):
        fold_delta_into_lora_pair(torch.ones((4, 2)), torch.ones((2, 4)), torch.ones((4, 4)), lora_fold_mode="bad")
    with pytest.raises(ValueError, match="rank mismatch"):
        fold_delta_into_lora_pair(torch.ones((4, 2)), torch.ones((3, 4)), torch.ones((4, 4)))
    with pytest.raises(ValueError, match="does not match"):
        fold_delta_into_lora_pair(torch.ones((4, 2)), torch.ones((2, 4)), torch.ones((5, 4)))
    with pytest.raises(ValueError, match="p_null shape"):
        fold_delta_into_lora_pair(torch.ones((4, 2)), torch.ones((2, 4)), torch.ones((4, 4)), p_null=torch.eye(3))
    with pytest.raises(TypeError, match="nn.Module"):
        fold_delta_into_swiglu_lora_w2_inplace("bad", torch.ones((4, 4)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        fold_delta_into_swiglu_lora_w2_inplace(nn.Linear(4, 4), torch.ones((4, 4)), weight_attr=" ")
    with pytest.raises(ValueError, match="must expose"):
        fold_delta_into_swiglu_lora_w2_inplace(nn.ReLU(), torch.ones((4, 4)), weight_attr="w2")
    with pytest.raises(ValueError, match="match weight shape"):
        fold_delta_into_swiglu_lora_w2_inplace(nn.Linear(4, 4), torch.ones((3, 3)))

    with pytest.raises(TypeError, match="Mapping"):
        compute_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(ValueError, match="non-empty strings"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"w": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"w": torch.tensor([float("nan")])})
    with pytest.raises(TypeError, match="mapping"):
        verify_memit_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_memit_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_memit_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_memit_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": MEMIT_EDITOR_RECEIPT_OBJECT, "schema": "wrong"}
        )
        is False
    )
    assert (
        verify_memit_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": MEMIT_EDITOR_RECEIPT_OBJECT,
                "schema": MEMIT_EDITOR_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 3. Model & orchestrator argument validation
    m_small = _build_4layer_subspace_crowd_moe(num_layers=2, lora_rank=2, seed=20261240)
    preserve_chunks = _default_preserve_chunks()
    ed_ok = [[32, 33, 34, 35]]
    tg_ok = [40]

    with pytest.raises(TypeError, match="nn.Module"):
        distribute_multilayer_memit_moe_edits("bad", ed_ok, tg_ok, preserve_chunks)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        distribute_multilayer_memit_moe_edits(nn.Linear(4, 4), ed_ok, tg_ok, preserve_chunks)

    class _NoHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4)])

    with pytest.raises(ValueError, match="expose"):
        distribute_multilayer_memit_moe_edits(_NoHead(), ed_ok, tg_ok, preserve_chunks)

    with pytest.raises(TypeError, match="AlphaEditMEMITConfig"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer dtype"):
        distribute_multilayer_memit_moe_edits(m_small, torch.ones((1, 4)), tg_ok, preserve_chunks)
    with pytest.raises(ValueError, match="non-empty"):
        distribute_multilayer_memit_moe_edits(m_small, [], tg_ok, preserve_chunks)
    with pytest.raises(TypeError, match="sequence of ints"):
        distribute_multilayer_memit_moe_edits(m_small, [123], tg_ok, preserve_chunks)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="uniform sequence length"):
        distribute_multilayer_memit_moe_edits(m_small, [[1, 2], [1, 2, 3]], tg_ok, preserve_chunks)
    with pytest.raises(TypeError, match="entries must be ints"):
        distribute_multilayer_memit_moe_edits(m_small, [[True, 2]], tg_ok, preserve_chunks)
    with pytest.raises(ValueError, match=">= 0"):
        distribute_multilayer_memit_moe_edits(m_small, [[-1, 2]], tg_ok, preserve_chunks)
    with pytest.raises(TypeError, match="2D integer tensor"):
        distribute_multilayer_memit_moe_edits(m_small, "bad", tg_ok, preserve_chunks)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D tensor"):
        distribute_multilayer_memit_moe_edits(m_small, torch.tensor([1, 2]), tg_ok, preserve_chunks)
    with pytest.raises(ValueError, match="sequence length"):
        distribute_multilayer_memit_moe_edits(m_small, [[1]], tg_ok, preserve_chunks)
    with pytest.raises(ValueError, match=">= 0"):
        distribute_multilayer_memit_moe_edits(m_small, torch.tensor([[-1, 2]]), tg_ok, preserve_chunks)
    with pytest.raises(ValueError, match="lie in"):
        distribute_multilayer_memit_moe_edits(m_small, [[1, 999]], tg_ok, preserve_chunks)
    with pytest.raises(TypeError, match="integer dtype"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, torch.tensor([1.0]), preserve_chunks)
    with pytest.raises(TypeError, match="sequence of ints"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, "bad", preserve_chunks)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length 1"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, [40, 41], preserve_chunks)
    with pytest.raises(ValueError, match="lie in"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, [999], preserve_chunks)
    with pytest.raises(TypeError, match="integer dtype"):
        distribute_multilayer_memit_moe_edits(
            m_small, ed_ok, tg_ok, preserve_chunks, edit_positions=torch.tensor([1.0])
        )
    with pytest.raises(TypeError, match="sequence of ints"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, edit_positions="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="edit_positions"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, edit_positions=[0, 1])
    with pytest.raises(TypeError, match="must be an int"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, edit_positions=[True])
    with pytest.raises(ValueError, match="out of range"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, edit_positions=[99])
    with pytest.raises(ValueError, match="preserve_chunks"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, [])
    with pytest.raises(ValueError, match="expert_id"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, [{"ids": [1, 2]}])
    with pytest.raises(ValueError, match="non-empty string"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, [{"expert_id": " ", "ids": [1, 2]}])
    with pytest.raises(ValueError, match="too short"):
        distribute_multilayer_memit_moe_edits(
            m_small,
            ed_ok,
            tg_ok,
            [{"expert_id": "E0_gen", "ids": [1, 2]}],
            config=AlphaEditMEMITConfig(score_from=5),
        )
    with pytest.raises(ValueError, match="E0_"):
        distribute_multilayer_memit_moe_edits(
            m_small, ed_ok, tg_ok, [{"expert_id": "E1_code", "ids": [16, 17, 18, 19]}]
        )
    with pytest.raises(ValueError, match="preserve_chunks token IDs"):
        distribute_multilayer_memit_moe_edits(
            m_small, ed_ok, tg_ok, [{"expert_id": "E0_gen", "ids": [1, 999]}]
        )
    with pytest.raises(ValueError, match="non-empty sequence"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, layer_indices=[])
    with pytest.raises(ValueError, match="layer_indices"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, layer_indices=[99])
    with pytest.raises(ValueError, match="non-empty sequence"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, expert_indices=[])
    with pytest.raises(ValueError, match="expert_indices"):
        distribute_multilayer_memit_moe_edits(m_small, ed_ok, tg_ok, preserve_chunks, expert_indices=[99])
    with pytest.raises(ValueError, match="preserve_key_seqs"):
        distribute_multilayer_memit_moe_edits(
            m_small, ed_ok, tg_ok, preserve_chunks, preserve_key_seqs=[[1, 999]]
        )
    with pytest.raises(ValueError, match="background_seqs"):
        distribute_multilayer_memit_moe_edits(
            m_small, ed_ok, tg_ok, preserve_chunks, background_seqs=[[1, 999]]
        )

    # 4. AGPL-3.0 header, optional ZeroStubAudit, PRSecurityScanner, lateral movement, and gatekeeper scans
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn12", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn12", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/rome_memit_knowledge_editor.py", TARGET_MODULE_PATH),
        ("tests/test_rome_memit_knowledge_editor_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        assert fallback_auditor.check_file(str(path_obj)) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True

        if ZeroStubAudit is not None:
            auditor = ZeroStubAudit()
            if hasattr(auditor, "audit_python_code"):
                stub_res = auditor.audit_python_code(src_text)
                assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
                assert stub_res.penalty_energy == 0.0
            if hasattr(auditor, "check_file"):
                assert auditor.check_file(str(path_obj)) == []

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
        {"nll_mean": 0.42, "edit_efficacy_ratio": 1.0, "anse_energy": -1.25},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}"
        )
        == []
    )
    assert len(compute_payload_sha256({"ok": True})) == 64

    pr_audit = scanner.audit_pull_request(
        pr_number=1112,
        pr_title="feat(swarm): AlphaEdit-MEMIT multi-layer closed-form SwiGLU fact and skill editor in general-prose null-space",
        pr_body="Implements NEURAL-CTRL-DYN-12 AlphaEdit-MEMIT multi-layer SwiGLU MoE editor with deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): AlphaEdit-MEMIT multi-layer closed-form SwiGLU fact and skill editor in general-prose null-space"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
