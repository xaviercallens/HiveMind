# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-19: lifelong sequential null-space basis tracker for WSP."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.sequential_nullspace_memory import (
    ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    NullspaceExhaustedError,
    SequentialNullspaceBasis,
    SequentialNullspaceConfig,
    canonical_json_bytes,
    commit_sequential_wsp_round_inplace,
    compute_payload_sha256,
    compute_state_sha256,
    solve_sequential_guarded_wsp_update,
    verify_sequential_wsp_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    compute_swiglu_intermediate_states,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "sequential_nullspace_memory.py"
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


def _build_orthogonal_basis(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((dim, dim), generator=gen, dtype=torch.float64)
    q_mat, r_mat = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r_mat))
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    return (q_mat * signs.unsqueeze(0)).contiguous()


def _build_subspace_crowd_moe_for_lifelong_wsp(
    *,
    num_layers: int = 2,
    hidden_size: int = 48,
    intermediate_size: int = 96,
    vocab_size: int = 64,
    num_experts: int = 2,
    top_k: int = 1,
    lora_rank: int = 8,
    split_mode: str = "lora_domain_experts",
    seed: int = 20261901,
) -> tuple[CrowdV1WithEngram, torch.Tensor, list[tuple[torch.Tensor, int]]]:
    """Build deterministic CPU CrowdV1WithEngram + Sparse MoE with decoupled prose & overlapping skills.

    Subspaces:
      - ``E0_general_fineweb`` (tokens ``0..15``): hidden dims ``0..15``, SwiGLU dims ``0..31``.
      - Skill prompts ``1..6`` (tokens ``16..31``) & skill target tokens (``32..63``):
        hidden dims ``16..47``, SwiGLU dims ``32..95``. Each skill prompt token shares
        substantial overlap along a common skill carrier ``c_shared`` (cosine ~0.49) plus a
        distinct orthogonal skill direction so stateless sequential editing overwrites rounds 1-3.
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
    h_prose = 16
    i_prose = 32
    h_skill = hidden_size - h_prose
    i_skill = intermediate_size - i_prose

    q_prose = _build_orthogonal_basis(h_prose, seed + 5).to(torch.float32)
    u_prose_carrier = q_prose[:, 0] - q_prose[:, 0].mean()
    u_prose_carrier = u_prose_carrier / u_prose_carrier.norm().clamp_min(1e-6)

    q_skill = _build_orthogonal_basis(h_skill, seed + 11).to(torch.float32)
    # Construct zero-mean vectors in the skill subspace so LayerNorm leaves both halves cleanly separated
    c_shared = q_skill[:, 0] - q_skill[:, 0].mean()
    c_shared = c_shared / c_shared.norm().clamp_min(1e-6)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        # E0 general prose tokens 0..15 in dims 0..15 with positive carrier alignment
        p_raw = torch.randn((16, h_prose), generator=gen, dtype=torch.float32) * 1.1
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True)
        p_raw = p_raw + 1.4 * u_prose_carrier.unsqueeze(0)
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True)
        wte[:16, :h_prose] = p_raw

        # Skill trigger tokens 16..21 (6 skills) sharing c_shared + orthogonal basis direction
        skill_prompts: list[tuple[torch.Tensor, int]] = []
        for idx in range(6):
            u_uniq = q_skill[:, idx + 1] - q_skill[:, idx + 1].mean()
            u_uniq = u_uniq / u_uniq.norm().clamp_min(1e-6)
            emb = 0.70 * c_shared + 0.714 * u_uniq
            emb = emb - emb.mean()
            emb = emb / emb.norm().clamp_min(1e-6) * 2.5
            wte[16 + idx, h_prose:] = emb
            # 4-token context ending on trigger token (16 + idx)
            seq = torch.tensor([[16 + ((idx + 1) % 6), 22 + idx, 16 + idx]], dtype=torch.long)
            target_tok = 32 + idx * 4
            skill_prompts.append((seq, target_tok))

        # Auxiliary skill context tokens 22..31
        ctx_raw = torch.randn((10, h_skill), generator=gen, dtype=torch.float32) * 0.8
        ctx_raw = ctx_raw - ctx_raw.mean(dim=1, keepdim=True)
        wte[22:32, h_prose:] = ctx_raw

        # Target vocabulary embeddings 32..63 placed on distinct well-separated skill directions
        for t_idx in range(32, vocab_size):
            col = q_skill[:, 8 + ((t_idx - 32) % (h_skill - 8))]
            col = col - col.mean()
            col = col / col.norm().clamp_min(1e-6) * 2.2
            if (t_idx - 32) // (h_skill - 8) > 0:
                col = -col
            wte[t_idx, h_prose:] = col

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.q.weight[:h_prose, :h_prose].copy_(0.04 * torch.eye(h_prose))
            blk.attn.k.weight[: h_prose // 2, :h_prose].copy_(0.04 * torch.eye(h_prose // 2, h_prose))
            blk.attn.v.weight[: h_prose // 2, :h_prose].copy_(0.04 * torch.eye(h_prose // 2, h_prose))
            blk.attn.o.weight[:h_prose, :h_prose].copy_(0.04 * torch.eye(h_prose))

            # Prose base SwiGLU block (dims 0..15 -> 0..31 -> 0..15)
            w1_p = torch.randn((i_prose, h_prose), generator=gen, dtype=torch.float32) * 0.20
            w2_p = torch.randn((i_prose, h_prose), generator=gen, dtype=torch.float32) * 0.20
            w3_p = torch.randn((h_prose, i_prose), generator=gen, dtype=torch.float32) * 0.12
            w3_p = w3_p - w3_p.mean(dim=0, keepdim=True)
            blk.mlp.w1.weight[:i_prose, :h_prose].copy_(w1_p)
            blk.mlp.w2.weight[:i_prose, :h_prose].copy_(w2_p)
            blk.mlp.w3.weight[:h_prose, :i_prose].copy_(w3_p)

            # Skill base SwiGLU projections map skill subspace orthogonally without bias toward target_tok
            proj_skill = torch.eye(i_skill, h_skill, dtype=torch.float32) * 0.85
            if i_skill > h_skill:
                proj_skill[h_skill:, :h_skill] = 0.25 * torch.randn((i_skill - h_skill, h_skill), generator=gen)
            blk.mlp.w1.weight[i_prose:, h_prose:].copy_(proj_skill)
            blk.mlp.w2.weight[i_prose:, h_prose:].copy_(proj_skill)
            blk.mlp.w3.weight[h_prose:, i_prose:].zero_()

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode=split_mode,
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        routing_mode="sigma_topk",
        seed=seed + 29,
    )
    with torch.no_grad():
        for blk in model.blocks:
            mlp = blk.mlp
            mlp.router.weight[0, :h_prose].copy_(3.0 * u_prose_carrier)
            mlp.router.weight[1, h_prose:].copy_(3.0 * c_shared)

    prose_chunks = [
        {"expert_id": "E0_general_fineweb", "ids": [0, 1, 2, 3, 4, 5, 6, 7]},
        {"expert_id": "E0_general_fineweb", "ids": [8, 9, 10, 11, 12, 13, 14, 15]},
        {"expert_id": "E0_general_fineweb", "ids": [1, 3, 5, 7, 9, 11, 13, 15]},
        {"expert_id": "E0_general_fineweb", "ids": [0, 2, 4, 6, 8, 10, 12, 14]},
    ]
    prose_ids = torch.tensor([c["ids"] for c in prose_chunks], dtype=torch.long)
    return model, prose_ids, skill_prompts


@torch.no_grad()
def _collect_block_swiglu_states(
    model: CrowdV1WithEngram,
    ids_2d: torch.Tensor,
    *,
    layer_idx: int,
    expert_idx: int,
    last_token_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect ``(x_mid_or_pre_ln, swiglu_act)`` at ``model.blocks[layer_idx]``."""
    cos = model.cos_cached.to("cpu")
    sin = model.sin_cached.to("cpu")
    xb = compute_embeddings_batch(model, ids_2d)
    for idx in range(layer_idx):
        blk = model.blocks[idx]
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        xb = forward_block_post_mlp(blk, xm, mi)
    target_blk = model.blocks[layer_idx]
    x_mid, mlp_in = forward_block_pre_mlp(target_blk, xb, cos, sin)
    mlp_sel = mlp_in[:, -1, :] if last_token_only else mlp_in.reshape(-1, int(mlp_in.shape[-1]))
    x_mid_sel = x_mid[:, -1, :] if last_token_only else x_mid.reshape(-1, int(x_mid.shape[-1]))
    act = compute_swiglu_intermediate_states(target_blk.mlp, mlp_sel, [expert_idx])
    return x_mid_sel, act


@torch.no_grad()
def _compute_last_block_w3_target_delta(
    model: CrowdV1WithEngram,
    seq_2d: torch.Tensor,
    target_tok: int,
    *,
    layer_idx: int,
    expert_idx: int,
    target_prob: float = 0.995,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(k_act, v_target_delta)`` for ``blocks[layer_idx].mlp.experts[expert_idx].w3``."""
    cos = model.cos_cached.to("cpu")
    sin = model.sin_cached.to("cpu")
    xb = compute_embeddings_batch(model, seq_2d)
    for idx in range(layer_idx):
        blk = model.blocks[idx]
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        xb = forward_block_post_mlp(blk, xm, mi)

    blk_t = model.blocks[layer_idx]
    x_mid, mlp_in = forward_block_pre_mlp(blk_t, xb, cos, sin)
    h_pre = forward_block_post_mlp(blk_t, x_mid, mlp_in)[:, -1, :].to(torch.float64)
    h_cand = h_pre.clone()

    mlp_last = mlp_in[:, -1, :]
    act = compute_swiglu_intermediate_states(blk_t.mlp, mlp_last, [expert_idx])
    gate_val = float(torch.sigmoid(blk_t.mlp.router(mlp_last.to(torch.float32)))[0, expert_idx].item())

    ln_w = model.ln.weight.detach().to(torch.float64)
    ln_b = model.ln.bias.detach().to(torch.float64)
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    head_w = model.head.weight.detach().to(torch.float64)
    h_dim = int(h_cand.shape[-1])

    for _ in range(32):
        h_norm = F.layer_norm(h_cand, (h_dim,), ln_w, ln_b, eps=ln_eps)
        logits = F.linear(h_norm, head_w)
        probs = F.softmax(logits, dim=-1)
        p_t = float(probs[0, target_tok].item())
        if p_t >= target_prob:
            break
        grad_dir = ln_w * (head_w[target_tok] - (probs[0] @ head_w))
        grad_dir = grad_dir - grad_dir.mean()
        # Keep update strictly within the skill hidden subspace (dims 16:)
        grad_dir[:16] = 0.0
        g_norm = float(grad_dir.norm().item())
        if g_norm > 1e-12:
            step = 0.8 if p_t < 0.5 else 0.4
            h_cand = h_cand + step * (grad_dir / g_norm).unsqueeze(0)

    # Scale by 1 / routing_gate so gate_val * (w3_lora @ act) delivers exact (h_cand - h_pre)
    v_delta = (h_cand - h_pre) / max(gate_val, 1e-4)
    return act.to(torch.float64), v_delta.to(torch.float64)


@torch.no_grad()
def _evaluate_model_prose_and_skills(
    model: CrowdV1WithEngram,
    prose_ids: torch.Tensor,
    skill_prompts: list[tuple[torch.Tensor, int]],
    active_skill_indices: list[int],
    *,
    current_round_pos: int | None = None,
) -> dict[str, Any]:
    """Evaluate exact CPU NLLs and top-1 skill accuracy across ``E0_general`` and active skills."""
    h_prose = forward_model_pre_ln(model, prose_ids, batch_size=4)
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    e0_nlls = per_chunk_nll_from_pre_ln(
        h_prose, model.ln.weight, model.ln.bias, ln_eps, model.head.weight, prose_ids, score_from=1
    )
    domain_nlls: dict[str, float] = {"E0_general_prose": float(e0_nlls.mean().item())}
    skill_efficacies: dict[str, float] = {}
    skill_probs: dict[str, float] = {}

    for s_idx in active_skill_indices:
        seq, tgt = skill_prompts[s_idx]
        h_s = forward_model_pre_ln(model, seq, batch_size=1)[:, -1, :]
        h_norm = F.layer_norm(h_s, (h_s.shape[-1],), model.ln.weight, model.ln.bias, eps=ln_eps)
        logits = F.linear(h_norm, model.head.weight)
        nll = float(F.cross_entropy(logits, torch.tensor([tgt], dtype=torch.long)).item())
        prob = float(F.softmax(logits, dim=-1)[0, tgt].item())
        top1 = int(logits.argmax(dim=-1).item()) == tgt
        s_key = f"skill_{s_idx + 1}"
        domain_nlls[s_key] = nll
        skill_efficacies[s_key] = 1.0 if top1 else prob
        skill_probs[s_key] = prob

    out: dict[str, Any] = {
        "domain_nlls": domain_nlls,
        "skill_probs": skill_probs,
        "skill_efficacies": skill_efficacies,
    }
    if current_round_pos is not None:
        cur_key = f"skill_{current_round_pos + 1}"
        out["new_skill_id"] = cur_key
        out["new_skill_efficacy"] = skill_efficacies[cur_key]
        out["historical_efficacies"] = {
            f"skill_{j + 1}": skill_efficacies[f"skill_{j + 1}"]
            for j in active_skill_indices
            if j != current_round_pos
        }
    return out


# ---------------------------------------------------------------------------
# S1: 6 Sequential WSP Skill Rounds Retaining >95% Across All 6 vs Stateless
# ---------------------------------------------------------------------------


def test_s1_six_sequential_wsp_skill_rounds_retain_all_skills_vs_stateless() -> None:
    """Inject 6 sequential WSP skills on CrowdV1WithEngram + CrowdLoRAMoEMLP, retaining >95% across all 6."""
    model, prose_ids, skill_prompts = _build_subspace_crowd_moe_for_lifelong_wsp(
        num_layers=2, hidden_size=48, intermediate_size=96, lora_rank=8, seed=20261901
    )
    stateless_model = copy.deepcopy(model)

    # Build initial U_0 from E0 general-prose SwiGLU intermediate states at block 1
    _, act_prose = _collect_block_swiglu_states(
        model, prose_ids, layer_idx=1, expert_idx=0, last_token_only=False
    )
    basis = SequentialNullspaceBasis.from_prose_keys(
        act_prose, sv_threshold=1e-2, min_null_rank=4
    )
    assert basis.ambient_dim == 96
    assert 1 <= basis.base_prose_rank <= 32
    assert basis.remaining_null_rank == 96 - basis.base_prose_rank

    base_eval = _evaluate_model_prose_and_skills(model, prose_ids, skill_prompts, [])
    base_e0_nll = base_eval["domain_nlls"]["E0_general_prose"]
    basis.set_baseline_general_nll(base_e0_nll)

    receipts: list[dict[str, Any]] = []
    for r_idx in range(6):
        seq_r, tgt_r = skill_prompts[r_idx]
        k_act, v_delta = _compute_last_block_w3_target_delta(
            model, seq_r, tgt_r, layer_idx=1, expert_idx=1
        )
        active_indices = list(range(r_idx + 1))

        def _eval_round(m: nn.Module, act_ix: list[int] = active_indices, cur: int = r_idx) -> dict[str, Any]:
            assert isinstance(m, CrowdV1WithEngram)
            return _evaluate_model_prose_and_skills(
                m, prose_ids, skill_prompts, act_ix, current_round_pos=cur
            )

        rec = commit_sequential_wsp_round_inplace(
            model,
            basis,
            k_act,
            v_delta,
            round_id=f"skill_{r_idx + 1}",
            layer_index=1,
            expert_index=1,
            weight_attr="w3",
            eval_fn=_eval_round,
            ridge_lambda=1e-3,
            min_null_rank=4,
        )
        assert rec["accepted"] is True, f"Round {r_idx + 1} failed: {rec['status']} ({rec['reason']})"
        assert rec["rolled_back"] is False
        assert rec["delta_e"] < 0.0
        assert rec["general_guard_ok"] is True
        assert rec["historical_guard_ok"] is True
        assert rec["max_historical_key_drift"] < 1e-5
        assert verify_sequential_wsp_receipt_provenance(rec) is True
        receipts.append(rec)

    assert basis.num_rounds == 6
    assert basis.orthogonality_error() < 1e-12
    assert basis.remaining_null_rank >= 4

    final_eval = _evaluate_model_prose_and_skills(model, prose_ids, skill_prompts, list(range(6)))
    final_e0_nll = final_eval["domain_nlls"]["E0_general_prose"]
    e0_pct_drift = 100.0 * (final_e0_nll / base_e0_nll - 1.0)
    assert e0_pct_drift <= 0.2, f"General prose drifted {e0_pct_drift:.4f}% > 0.2%"

    for r_idx in range(6):
        s_key = f"skill_{r_idx + 1}"
        assert final_eval["skill_efficacies"][s_key] > 0.95, f"{s_key} efficacy lost!"
        assert final_eval["skill_probs"][s_key] > 0.95, f"{s_key} prob dropped!"

    # Stateless control: reset basis to U_0 every round (forgetting U_{1..t}) -> overwrites rounds 1-3
    for r_idx in range(6):
        seq_r, tgt_r = skill_prompts[r_idx]
        k_act, v_delta = _compute_last_block_w3_target_delta(
            stateless_model, seq_r, tgt_r, layer_idx=1, expert_idx=1
        )
        stateless_basis = SequentialNullspaceBasis.from_prose_keys(
            act_prose, sv_threshold=1e-2, min_null_rank=4
        )
        delta_w, s_meta = solve_sequential_guarded_wsp_update(
            k_act, v_delta, stateless_basis, ridge_lambda=1e-3, min_null_rank=4
        )
        assert s_meta["accepted"] is True
        exp1 = stateless_model.blocks[1].mlp.experts["1"]
        with torch.no_grad():
            full_w3 = exp1.scaling * (exp1.w3_B.double() @ exp1.w3_A.double()) + delta_w
            u_s, s_s, vh_s = torch.linalg.svd(full_w3, full_matrices=False)
            rk = exp1.rank
            sq = s_s[:rk].clamp_min(0.0).sqrt()
            exp1.w3_B.copy_((u_s[:, :rk] * sq.unsqueeze(0)).float())
            exp1.w3_A.copy_((sq.unsqueeze(1) * vh_s[:rk, :]).float())

    stateless_eval = _evaluate_model_prose_and_skills(
        stateless_model, prose_ids, skill_prompts, list(range(6))
    )
    early_stateless_probs = [stateless_eval["skill_probs"][f"skill_{j + 1}"] for j in range(3)]
    assert min(early_stateless_probs) < 0.75, (
        f"Expected stateless sequential editing to overwrite rounds 1-3, got {early_stateless_probs}"
    )

    # Also test dense CrowdRuntimeMLP direct round commit (no eval_fn -> built-in residual eval)
    dense_mlp = CrowdRuntimeMLP(
        CrowdV1RuntimeConfig(
            vocab_size=32, hidden_size=24, num_hidden_layers=1, intermediate_size=48, engram_buckets=0
        )
    )
    gen = torch.Generator(device="cpu").manual_seed(20261902)
    q_d = _build_orthogonal_basis(24, seed=20261903)
    dense_basis = SequentialNullspaceBasis(u0_basis=q_d[:, :4], min_null_rank=4)
    k_d = (q_d[:, 4:6].T + 0.3 * q_d[:, :2].T).contiguous()
    v_goal = torch.randn((2, 48), generator=gen, dtype=torch.float64)
    rec_dense = commit_sequential_wsp_round_inplace(
        dense_mlp,
        dense_basis,
        k_d,
        v_goal,
        round_id="dense_w2_round1",
        weight_attr="w2",
        v_is_goal=True,
        ridge_lambda=1e-4,
    )
    assert rec_dense["accepted"] is True
    assert rec_dense["delta_e"] < 0.0
    assert rec_dense["max_historical_key_drift"] < 1e-6


# ---------------------------------------------------------------------------
# S2: Exact Orthogonality (<1e-5 Drift on Historical Round Keys After Subsequent Rounds)
# ---------------------------------------------------------------------------


def test_s2_exact_orthogonality_and_historical_key_invariance() -> None:
    """Verify exact float64 basis orthogonality and <1e-5 historical key drift over 8 sequential rounds."""
    d_in = 32
    d_out = 16
    q_all = _build_orthogonal_basis(d_in, seed=20261920)
    k_prose = q_all[:, :6].T.contiguous()  # 6 prose directions

    basis = SequentialNullspaceBasis.from_prose_keys(k_prose, sv_threshold=1e-2, min_null_rank=4)
    assert basis.base_prose_rank == 6
    assert basis.remaining_null_rank == 26
    assert basis.orthogonality_error() < 1e-12

    p0 = basis.projector()
    assert torch.allclose(p0 @ p0, p0, atol=1e-12)
    assert torch.allclose(p0, p0.T, atol=1e-12)
    assert torch.allclose(p0 @ basis.u0_basis, torch.zeros_like(basis.u0_basis), atol=1e-12)

    linear_layer = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
    gen = torch.Generator(device="cpu").manual_seed(20261921)
    with torch.no_grad():
        linear_layer.weight.copy_(torch.randn((d_out, d_in), generator=gen, dtype=torch.float64) * 0.25)

    c_bg = (q_all[:, 28:32] @ q_all[:, 28:32].T).contiguous()
    historical_keys: list[tuple[torch.Tensor, torch.Tensor]] = []

    for round_num in range(1, 9):
        # Each round's key intentionally mixes prior round directions + prose + 2 novel directions
        col_a = 6 + 2 * (round_num - 1)
        novel = q_all[:, col_a : col_a + 2]
        overlap_prose = 0.25 * q_all[:, :2]
        overlap_prev = 0.35 * q_all[:, 6 : 6 + max(2, 2 * (round_num - 1))][:, :2]
        k_round = (novel + overlap_prose + overlap_prev).T.contiguous()  # (2, d_in)
        v_target = torch.randn((2, d_out), generator=gen, dtype=torch.float64) * 1.5

        rec = commit_sequential_wsp_round_inplace(
            linear_layer,
            basis,
            k_round,
            v_target,
            round_id=f"ortho_r{round_num}",
            weight_attr="weight",
            ridge_lambda=1e-4,
            min_null_rank=4,
            c_key_cov=c_bg if round_num % 2 == 0 else None,
        )
        assert rec["accepted"] is True
        assert rec["max_historical_key_drift"] < 1e-12
        assert basis.orthogonality_error() < 1e-12

        # Check every historical key's exact linear output against its snapshot right after insertion
        with torch.no_grad():
            for h_idx, (hk, expected_out) in enumerate(historical_keys, start=1):
                actual_out = F.linear(hk, linear_layer.weight)
                drift = float((actual_out - expected_out).abs().max().item())
                assert drift < 1e-5, f"Round {round_num} drifted historical round {h_idx} by {drift:.3e}"

            historical_keys.append((k_round.clone(), F.linear(k_round, linear_layer.weight).clone()))

        # Prose output must remain invariant to machine precision
        with torch.no_grad():
            delta_total_on_prose = F.linear(k_prose, linear_layer.weight)
            assert delta_total_on_prose.shape == (6, d_out)

    # Verify basis slices, projector idempotence, and float32 CrowdLoRAExpertAdapter update_a path
    assert basis.num_rounds == 8
    assert basis.skill_rank == 16
    assert basis.remaining_null_rank == 10
    assert basis.skill_basis.shape == (d_in, 16)
    assert basis.get_round_basis("ortho_r3").shape == (d_in, 2)
    cap = basis.capacity_summary()
    assert cap["remaining_null_rank"] == 10
    assert cap["orthogonality_error"] < 1e-12

    p_final = basis.projector()
    assert float((p_final @ basis.u_basis).abs().max().item()) < 1e-12
    assert float((p_final @ p_final - p_final).abs().max().item()) < 1e-12

    # Exercise CrowdLoRAExpertAdapter in float32 with update_a after all zero slots are occupied
    lora_exp = CrowdLoRAExpertAdapter(d_in, d_out, rank=2, alpha=4.0, seed=77)
    with torch.no_grad():
        lora_exp.w2_B.copy_(torch.randn((d_out, 2), generator=gen, dtype=torch.float32))
    basis_f32 = SequentialNullspaceBasis.from_prose_keys(k_prose, min_null_rank=4)
    k_c = q_all[:, 10:12]  # column-oriented (d_in, 2)
    v_c = (lora_exp.w2_B.double() @ torch.randn((2, 2), generator=gen, dtype=torch.float64)).contiguous()
    delta_c, meta_c = solve_sequential_guarded_wsp_update(
        k_c, v_c, basis_f32, ridge_lambda=1e-4, keys_as_columns=True
    )
    assert meta_c["accepted"] is True
    assert float((delta_c @ k_prose.T).abs().max().item()) < 1e-12
    rec_lora = commit_sequential_wsp_round_inplace(
        lora_exp,
        basis_f32,
        k_c,
        v_c,
        weight_attr="w2",
        lora_fold_mode="update_a",
        keys_as_columns=True,
        ridge_lambda=1e-4,
    )
    assert rec_lora["accepted"] is True
    assert rec_lora["fold_info"]["fold_method"] == "update_a_nullspace"
    assert rec_lora["max_historical_key_drift"] < 1e-5


# ---------------------------------------------------------------------------
# S3: Null-Space Capacity Exhaustion Detection Halting Before Rank Collapse
# ---------------------------------------------------------------------------


def test_s3_nullspace_capacity_exhaustion_halts_cleanly_before_rank_collapse() -> None:
    """Verify clean refusal before remaining_null_rank collapses below min_null_rank."""
    d_in = 20
    d_out = 8
    q_all = _build_orthogonal_basis(d_in, seed=20261930)
    k_prose = q_all[:, :4].T.contiguous()

    basis = SequentialNullspaceBasis.from_prose_keys(k_prose, sv_threshold=1e-2, min_null_rank=4)
    assert basis.remaining_null_rank == 16
    assert basis.available_expandable_rank() == 12
    assert basis.is_exhausted() is False

    layer = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
    gen = torch.Generator(device="cpu").manual_seed(20261931)

    # Consume 4 dims per round across 3 rounds: 16 -> 12 -> 8 -> 4 (exact reserve boundary)
    for idx in range(3):
        c0 = 4 + idx * 4
        k_r = q_all[:, c0 : c0 + 4].T.contiguous()
        v_r = torch.randn((4, d_out), generator=gen, dtype=torch.float64)
        rec = commit_sequential_wsp_round_inplace(
            layer,
            basis,
            k_r,
            v_r,
            round_id=f"cap_r{idx + 1}",
            weight_attr="weight",
            min_null_rank=4,
            ridge_lambda=1e-3,
        )
        assert rec["accepted"] is True

    assert basis.remaining_null_rank == 4
    assert basis.available_expandable_rank() == 0
    assert basis.is_exhausted() is True

    pre_weights_sha = compute_state_sha256(layer)
    pre_basis_sha = basis.state_sha256()

    # Round 4 tries to use 2 of the last 4 dimensions -> would drop remaining to 2 < min_null_rank(4)
    k_overflow = q_all[:, 16:18].T.contiguous()
    v_overflow = torch.randn((2, d_out), generator=gen, dtype=torch.float64)

    delta_refused, meta_refused = solve_sequential_guarded_wsp_update(
        k_overflow, v_overflow, basis, ridge_lambda=1e-2, min_null_rank=4
    )
    assert meta_refused["accepted"] is False
    assert meta_refused["refused"] is True
    assert meta_refused["status"] == "REFUSED_NULLSPACE_EXHAUSTED"
    assert torch.equal(delta_refused, torch.zeros((d_out, d_in), dtype=torch.float64))

    with pytest.raises(NullspaceExhaustedError, match="rank collapse"):
        solve_sequential_guarded_wsp_update(
            k_overflow,
            v_overflow,
            basis,
            ridge_lambda=1e-2,
            min_null_rank=4,
            raise_on_exhaustion=True,
        )

    rec_refused = commit_sequential_wsp_round_inplace(
        layer,
        basis,
        k_overflow,
        v_overflow,
        round_id="cap_overflow_r4",
        weight_attr="weight",
        min_null_rank=4,
    )
    assert rec_refused["accepted"] is False
    assert rec_refused["rolled_back"] is False
    assert rec_refused["refused_nullspace_exhausted"] is True
    assert rec_refused["status"] == "REFUSED_NULLSPACE_EXHAUSTED"
    assert rec_refused["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert verify_sequential_wsp_receipt_provenance(rec_refused) is True
    assert basis.remaining_null_rank == 4
    assert compute_state_sha256(layer) == pre_weights_sha
    assert basis.state_sha256() == pre_basis_sha

    # Direct pre-exhausted basis (remaining_null_rank < min_null_rank) & direct append refusal
    low_basis = SequentialNullspaceBasis(u0_basis=q_all[:, :17], min_null_rank=4)
    assert low_basis.remaining_null_rank == 3
    delta_low, meta_low = solve_sequential_guarded_wsp_update(
        q_all[:, 17:18].T, torch.ones((1, d_out), dtype=torch.float64), low_basis, min_null_rank=4
    )
    assert meta_low["status"] == "REFUSED_NULLSPACE_EXHAUSTED"
    assert float(delta_low.norm().item()) == 0.0
    with pytest.raises(NullspaceExhaustedError, match="exhausted"):
        solve_sequential_guarded_wsp_update(
            q_all[:, 17:18].T,
            torch.ones((1, d_out), dtype=torch.float64),
            low_basis,
            min_null_rank=4,
            raise_on_exhaustion=True,
        )
    with pytest.raises(NullspaceExhaustedError, match="Cannot append"):
        low_basis.append_round_keys(q_all[:, 17:18].T, round_id="too_full")

    # Subspace collapse: candidate keys lie 100% inside protected U_t
    k_duplicate = k_prose[:2].clone()
    delta_dup, meta_dup = solve_sequential_guarded_wsp_update(
        k_duplicate, torch.ones((2, d_out), dtype=torch.float64), basis, min_null_rank=0
    )
    assert meta_dup["refused"] is True
    assert meta_dup["status"] == "REFUSED_INFEASIBLE_SUBSPACE"
    assert float(delta_dup.norm().item()) == 0.0
    with pytest.raises(NullspaceExhaustedError, match="collapse"):
        solve_sequential_guarded_wsp_update(
            k_duplicate,
            torch.ones((2, d_out), dtype=torch.float64),
            basis,
            min_null_rank=0,
            raise_on_exhaustion=True,
        )


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback Leaving Both Weights & U_t Untouched
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_leaves_weights_and_basis_untouched() -> None:
    """Verify ANSE Maximum Pain triggers bitwise rollback on both weights and basis U_t."""
    model, prose_ids, skill_prompts = _build_subspace_crowd_moe_for_lifelong_wsp(
        num_layers=2, hidden_size=48, intermediate_size=96, lora_rank=8, seed=20261940
    )
    _, act_prose = _collect_block_swiglu_states(
        model, prose_ids, layer_idx=1, expert_idx=0, last_token_only=False
    )
    basis = SequentialNullspaceBasis.from_prose_keys(act_prose, sv_threshold=1e-2, min_null_rank=4)
    base_eval = _evaluate_model_prose_and_skills(model, prose_ids, skill_prompts, [])
    basis.set_baseline_general_nll(base_eval["domain_nlls"]["E0_general_prose"])

    # Commit rounds 1 and 2 cleanly
    for r_idx in range(2):
        seq_r, tgt_r = skill_prompts[r_idx]
        k_act, v_delta = _compute_last_block_w3_target_delta(
            model, seq_r, tgt_r, layer_idx=1, expert_idx=1
        )
        rec = commit_sequential_wsp_round_inplace(
            model,
            basis,
            k_act,
            v_delta,
            round_id=f"skill_{r_idx + 1}",
            layer_index=1,
            expert_index=1,
            weight_attr="w3",
            eval_fn=lambda m, cur=r_idx: _evaluate_model_prose_and_skills(
                m, prose_ids, skill_prompts, list(range(cur + 1)), current_round_pos=cur  # type: ignore[arg-type]
            ),
            ridge_lambda=1e-3,
        )
        assert rec["accepted"] is True

    snap_weights_sha = compute_state_sha256(model)
    snap_basis_sha = basis.state_sha256()
    snap_rounds = basis.num_rounds
    snap_rem = basis.remaining_null_rank

    seq_3, tgt_3 = skill_prompts[2]
    k_act_3, v_delta_3 = _compute_last_block_w3_target_delta(
        model, seq_3, tgt_3, layer_idx=1, expert_idx=1
    )

    # 4A: Guard violation (> +1.0% on historical skill_1) triggers Maximum Pain + atomic rollback
    call_counter = {"n": 0}

    def _guard_breaching_eval(m: nn.Module) -> dict[str, Any]:
        call_counter["n"] += 1
        res = _evaluate_model_prose_and_skills(
            m, prose_ids, skill_prompts, [0, 1, 2], current_round_pos=2  # type: ignore[arg-type]
        )
        if call_counter["n"] % 2 == 0:
            res["domain_nlls"]["skill_1"] *= 1.05  # +5.0% breach vs > +1.0% ceiling
        return res

    rec_guard = commit_sequential_wsp_round_inplace(
        model,
        basis,
        k_act_3,
        v_delta_3,
        round_id="skill_3_guard_breach",
        layer_index=1,
        expert_index=1,
        weight_attr="w3",
        eval_fn=_guard_breaching_eval,
        ridge_lambda=1e-3,
    )
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["guard_ok"] is False
    assert rec_guard["historical_guard_ok"] is False
    assert rec_guard["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["final_weights_sha256"] == snap_weights_sha
    assert rec_guard["final_basis_sha256"] == snap_basis_sha
    assert compute_state_sha256(model) == snap_weights_sha
    assert basis.state_sha256() == snap_basis_sha
    assert basis.num_rounds == snap_rounds
    assert basis.remaining_null_rank == snap_rem
    assert verify_sequential_wsp_receipt_provenance(rec_guard) is True

    # 4B: Infeasible edit (0% new-skill efficacy) triggers Maximum Pain + atomic rollback
    def _infeasible_efficacy_eval(m: nn.Module) -> dict[str, Any]:
        res = _evaluate_model_prose_and_skills(
            m, prose_ids, skill_prompts, [0, 1, 2], current_round_pos=2  # type: ignore[arg-type]
        )
        res["new_skill_efficacy"] = 0.10  # below min_new_skill_efficacy=0.75
        return res

    rec_infeasible = commit_sequential_wsp_round_inplace(
        model,
        basis,
        k_act_3,
        v_delta_3,
        round_id="skill_3_infeasible",
        layer_index=1,
        expert_index=1,
        weight_attr="w3",
        eval_fn=_infeasible_efficacy_eval,
        ridge_lambda=1e-3,
    )
    assert rec_infeasible["accepted"] is False
    assert rec_infeasible["rolled_back"] is True
    assert rec_infeasible["status"] == "ROLLED_BACK_INFEASIBLE_EDIT"
    assert rec_infeasible["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert compute_state_sha256(model) == snap_weights_sha
    assert basis.state_sha256() == snap_basis_sha

    # 4C: Non-improving ANSE Delta E >= 0 triggers atomic rollback
    non_impr_calls = {"n": 0}

    def _non_improving_eval(m: nn.Module) -> dict[str, Any]:
        non_impr_calls["n"] += 1
        res = _evaluate_model_prose_and_skills(
            m, prose_ids, skill_prompts, [0, 1, 2], current_round_pos=2  # type: ignore[arg-type]
        )
        if non_impr_calls["n"] % 2 == 1:
            res["domain_nlls"]["skill_3"] = 0.05
        else:
            res["domain_nlls"]["skill_3"] = 0.04999
            res["domain_nlls"]["E0_general_prose"] *= 1.009  # hits soft_margin_penalty (+0.05) -> Delta E > 0
        return res

    rec_non_impr = commit_sequential_wsp_round_inplace(
        model,
        basis,
        k_act_3,
        v_delta_3,
        round_id="skill_3",
        layer_index=1,
        expert_index=1,
        weight_attr="w3",
        eval_fn=_non_improving_eval,
        ridge_lambda=1e-3,
    )
    assert rec_non_impr["accepted"] is False
    assert rec_non_impr["rolled_back"] is True
    assert rec_non_impr["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_non_impr["delta_e"] >= 0.0
    assert compute_state_sha256(model) == snap_weights_sha
    assert basis.state_sha256() == snap_basis_sha

    # 4D: Non-finite candidate NLL triggers Maximum Pain + atomic rollback
    nan_calls = {"n": 0}

    def _nan_eval(m: nn.Module) -> dict[str, Any]:
        nan_calls["n"] += 1
        res = _evaluate_model_prose_and_skills(
            m, prose_ids, skill_prompts, [0, 1, 2], current_round_pos=2  # type: ignore[arg-type]
        )
        if nan_calls["n"] % 2 == 0:
            res["domain_nlls"]["skill_3"] = float("nan")
        return res

    rec_nan = commit_sequential_wsp_round_inplace(
        model,
        basis,
        k_act_3,
        v_delta_3,
        round_id="skill_3",
        layer_index=1,
        expert_index=1,
        weight_attr="w3",
        eval_fn=_nan_eval,
        ridge_lambda=1e-3,
    )
    assert rec_nan["accepted"] is False
    assert rec_nan["rolled_back"] is True
    assert rec_nan["status"] == "ROLLED_BACK_NON_FINITE"
    assert rec_nan["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert compute_state_sha256(model) == snap_weights_sha
    assert basis.state_sha256() == snap_basis_sha

    # 4E: Conflicting key round lying inside U_t refused cleanly leaving weights & basis untouched
    rec_conflict = commit_sequential_wsp_round_inplace(
        model,
        basis,
        act_prose[:2].to(torch.float64),
        torch.ones((2, 48), dtype=torch.float64),
        round_id="conflict_round",
        layer_index=1,
        expert_index=1,
        weight_attr="w3",
    )
    assert rec_conflict["accepted"] is False
    assert rec_conflict["status"] == "REFUSED_INFEASIBLE_SUBSPACE"
    assert rec_conflict["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert compute_state_sha256(model) == snap_weights_sha
    assert basis.state_sha256() == snap_basis_sha

    # 4F: Subsequent valid Round 3 still commits cleanly after all rollbacks
    rec_ok = commit_sequential_wsp_round_inplace(
        model,
        basis,
        k_act_3,
        v_delta_3,
        round_id="skill_3",
        layer_index=1,
        expert_index=1,
        weight_attr="w3",
        eval_fn=lambda m: _evaluate_model_prose_and_skills(
            m, prose_ids, skill_prompts, [0, 1, 2], current_round_pos=2  # type: ignore[arg-type]
        ),
        ridge_lambda=1e-3,
    )
    assert rec_ok["accepted"] is True
    assert rec_ok["delta_e"] < 0.0
    assert basis.num_rounds == 3


# ---------------------------------------------------------------------------
# S5: Validation Errors, Provenance Tampering & Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_provenance_tampering_and_security_gates() -> None:
    """Exercise validation errors, edge-case code paths, provenance tampering, and HiveMind gates."""
    q8 = _build_orthogonal_basis(8, seed=20261950)

    # 1. Config & Basis construction / validation errors
    with pytest.raises(TypeError, match="real number"):
        SequentialNullspaceConfig(ridge_lambda=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="> 0.0"):
        SequentialNullspaceConfig(ridge_lambda=0.0)
    with pytest.raises(ValueError, match=">= 0.0"):
        SequentialNullspaceConfig(sv_threshold=-0.1)
    with pytest.raises(ValueError, match="finite"):
        SequentialNullspaceConfig(sv_threshold=float("nan"))
    with pytest.raises(TypeError, match="int"):
        SequentialNullspaceConfig(min_null_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        SequentialNullspaceConfig(min_null_rank=-1)
    with pytest.raises(ValueError, match="<= 1.0"):
        SequentialNullspaceConfig(min_projected_energy_ratio=1.5)
    with pytest.raises(ValueError, match="lora_fold_mode"):
        SequentialNullspaceConfig(lora_fold_mode="invalid_mode")
    with pytest.raises(ValueError, match="general_prefix"):
        SequentialNullspaceConfig(general_prefix=" ")
    with pytest.raises(TypeError, match="bool"):
        SequentialNullspaceConfig(require_strict_energy_drop="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at most one"):
        SequentialNullspaceBasis(prose_keys=q8[:, :2].T, u0_basis=q8[:, :2])
    with pytest.raises(ValueError, match="must be provided"):
        SequentialNullspaceBasis()
    with pytest.raises(ValueError, match=">= 1"):
        SequentialNullspaceBasis(ambient_dim=0)
    with pytest.raises(TypeError, match="torch.Tensor"):
        SequentialNullspaceBasis(u0_basis="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="match ambient_dim"):
        SequentialNullspaceBasis(ambient_dim=6, u0_basis=q8[:, :2])
    with pytest.raises(ValueError, match="cannot exceed"):
        SequentialNullspaceBasis(u0_basis=torch.ones((4, 5)))
    with pytest.raises(ValueError, match="orthonormal"):
        SequentialNullspaceBasis(u0_basis=torch.ones((4, 2)))
    capped_b = SequentialNullspaceBasis.from_prose_keys(q8[:, :4].T, max_prose_rank=2, min_null_rank=2)
    assert capped_b.base_prose_rank == 2
    assert capped_b.sv_threshold == pytest.approx(1e-2)
    assert capped_b.min_null_rank == 2
    with pytest.raises(ValueError, match="mapping must not be empty"):
        SequentialNullspaceBasis(prose_keys={})
    with pytest.raises(ValueError, match="sequence must not be empty"):
        SequentialNullspaceBasis(prose_keys=[])
    with pytest.raises(TypeError, match="torch.Tensor"):
        SequentialNullspaceBasis(prose_keys="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        SequentialNullspaceBasis(prose_keys=[123])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="2D or 3D"):
        SequentialNullspaceBasis(prose_keys=torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        SequentialNullspaceBasis(prose_keys=torch.ones((2, 4), dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        SequentialNullspaceBasis(prose_keys=torch.tensor([[1.0, float("nan")]]))
    with pytest.raises(ValueError, match="match ambient_dim"):
        SequentialNullspaceBasis(prose_keys=[torch.ones((2, 4)), torch.ones((2, 5))])

    # Empty basis & mapping/3D key collection + clone/snapshot/restore paths
    empty_b = SequentialNullspaceBasis(8, min_null_rank=0)
    assert empty_b.is_exhausted() is False
    assert empty_b.orthogonality_error() == 0.0
    assert torch.allclose(empty_b.projector(), torch.eye(8, dtype=torch.float64))
    assert torch.allclose(empty_b.project(torch.ones((2, 8))), torch.ones((2, 8), dtype=torch.float64))
    with pytest.raises(ValueError, match="row count"):
        empty_b.project(torch.ones((4, 2)), keys_as_columns=True)
    with pytest.raises(ValueError, match="width"):
        empty_b.project(torch.ones((2, 4)), keys_as_columns=False)

    map_basis = SequentialNullspaceBasis.from_prose_keys(
        {"b": q8[:, :2].T, "a": q8[:, :2].T.unsqueeze(0)}
    )
    assert map_basis.protected_rank == 2
    assert map_basis.utilization_ratio == pytest.approx(0.25)
    with pytest.raises(ValueError, match="row count"):
        map_basis.project(torch.ones((4, 2)), keys_as_columns=True)
    with pytest.raises(ValueError, match="width"):
        map_basis.project(torch.ones((2, 4)), keys_as_columns=False)
    with pytest.raises(KeyError, match="Unknown round_id"):
        map_basis.get_round_basis("missing_round")
    with pytest.raises(ValueError, match="non-empty"):
        map_basis.append_round_keys(q8[:, 2:3].T, round_id=" ")
    map_basis.append_round_keys(q8[:, 2:3].T, round_id="r1")
    with pytest.raises(ValueError, match="Duplicate"):
        map_basis.append_round_keys(q8[:, 3:4].T, round_id="r1")
    cloned = map_basis.clone()
    assert cloned.state_sha256() == map_basis.state_sha256()
    with pytest.raises(ValueError, match="Invalid"):
        cloned.restore({})
    with pytest.raises(ValueError, match="ambient_dim"):
        cloned.restore({"ambient_dim": 99, "u_basis": torch.zeros((99, 1))})
    with pytest.raises(ValueError, match="Invalid u_basis"):
        cloned.restore({"ambient_dim": 8, "u_basis": torch.zeros((4, 1))})

    # 2. Solver & target resolution validation errors
    with pytest.raises(TypeError, match="SequentialNullspaceBasis"):
        solve_sequential_guarded_wsp_update(torch.ones((2, 8)), torch.ones((2, 4)), "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 2D"):
        solve_sequential_guarded_wsp_update(torch.ones(8), torch.ones((1, 4)), map_basis)
    with pytest.raises(TypeError, match="floating-point"):
        solve_sequential_guarded_wsp_update(
            torch.ones((2, 8), dtype=torch.int64), torch.ones((2, 4)), map_basis
        )
    with pytest.raises(ValueError, match="finite"):
        solve_sequential_guarded_wsp_update(
            torch.tensor([[float("inf")] * 8]), torch.ones((1, 4)), map_basis
        )
    with pytest.raises(ValueError, match="incompatible"):
        solve_sequential_guarded_wsp_update(torch.ones((2, 5)), torch.ones((2, 4)), map_basis)
    with pytest.raises(ValueError, match="batch count"):
        solve_sequential_guarded_wsp_update(torch.ones((2, 8)), torch.ones((3, 4)), map_basis)
    with pytest.raises(ValueError, match="edit count"):
        solve_sequential_guarded_wsp_update(
            torch.ones((8, 2)), torch.ones((4, 3)), map_basis, keys_as_columns=True
        )
    with pytest.raises(ValueError, match="row dimension"):
        solve_sequential_guarded_wsp_update(
            torch.ones((7, 2)), torch.ones((4, 2)), map_basis, keys_as_columns=True
        )
    with pytest.raises(ValueError, match="feature width"):
        solve_sequential_guarded_wsp_update(
            torch.ones((2, 7)), torch.ones((2, 4)), map_basis, keys_as_columns=False
        )
    # Auto-detect column mode & square matrix orientation
    d_col_auto, m_col_auto = solve_sequential_guarded_wsp_update(
        q8[:, 4:6], torch.ones((4, 2), dtype=torch.float64), map_basis, min_null_rank=2
    )
    assert m_col_auto["accepted"] is True and d_col_auto.shape == (4, 8)
    d_sq_auto, m_sq_auto = solve_sequential_guarded_wsp_update(
        q8,
        torch.ones((4, 8), dtype=torch.float64),
        SequentialNullspaceBasis(8, min_null_rank=0),
        min_null_rank=0,
    )
    assert m_sq_auto["accepted"] is True and d_sq_auto.shape == (4, 8)
    with pytest.raises(ValueError, match="c_key_cov"):
        solve_sequential_guarded_wsp_update(
            q8[:, 3:5].T, torch.ones((2, 4)), map_basis, c_key_cov=torch.eye(4)
        )
    with pytest.raises(ValueError, match="positive semi-definite"):
        solve_sequential_guarded_wsp_update(
            q8[:, 3:5].T, torch.ones((2, 4)), map_basis, c_key_cov=-torch.eye(8)
        )

    # 3. commit_sequential_wsp_round_inplace validation + qr_svd + fine_grained + eval_fn edge cases
    lin8 = nn.Linear(8, 4, bias=False, dtype=torch.float64)
    with pytest.raises(TypeError, match="SequentialNullspaceBasis"):
        commit_sequential_wsp_round_inplace(lin8, "bad", torch.ones((1, 8)), torch.ones((1, 4)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        commit_sequential_wsp_round_inplace("bad", map_basis, torch.ones((1, 8)), torch.ones((1, 4)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight_attr"):
        commit_sequential_wsp_round_inplace(lin8, map_basis, torch.ones((1, 8)), torch.ones((1, 4)), weight_attr="bad")
    with pytest.raises(ValueError, match="blocks"):
        commit_sequential_wsp_round_inplace(lin8, map_basis, torch.ones((1, 8)), torch.ones((1, 4)), layer_index=0)
    with pytest.raises(ValueError, match="expert_index"):
        commit_sequential_wsp_round_inplace(lin8, map_basis, torch.ones((1, 8)), torch.ones((1, 4)), expert_index=0)
    with pytest.raises(TypeError, match="callable"):
        commit_sequential_wsp_round_inplace(lin8, map_basis, torch.ones((1, 8)), torch.ones((1, 4)), eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="match basis.ambient_dim"):
        commit_sequential_wsp_round_inplace(
            nn.Linear(6, 4, bias=False), map_basis, torch.ones((1, 6)), torch.ones((1, 4)), weight_attr="weight"
        )
    with pytest.raises(ValueError, match="output dim"):
        commit_sequential_wsp_round_inplace(
            lin8, map_basis, q8[:, 3:4].T, torch.ones((1, 5)), weight_attr="weight"
        )
    with pytest.raises(ValueError, match="already exists"):
        commit_sequential_wsp_round_inplace(
            lin8, map_basis, q8[:, 3:4].T, torch.ones((1, 4)), round_id="r1", weight_attr="weight"
        )

    # Test FineGrainedMoE, LoRA qr_svd fold, and eval_fn format errors
    fg_model, fg_prose_ids, _ = _build_subspace_crowd_moe_for_lifelong_wsp(
        num_layers=1,
        hidden_size=48,
        intermediate_size=96,
        num_experts=2,
        top_k=1,
        split_mode="fine_grained_width_split",
        seed=20261951,
    )
    q48 = _build_orthogonal_basis(48, seed=20261952)
    fg_basis = SequentialNullspaceBasis.from_prose_keys(q48[:, :4].T, min_null_rank=4)
    assert fg_prose_ids.shape[1] == 8
    rec_fg = commit_sequential_wsp_round_inplace(
        fg_model,
        fg_basis,
        q48[:, 4:6].T,
        torch.ones((2, 48), dtype=torch.float64) * 0.2,
        layer_index=0,
        expert_index=1,
        weight_attr="w2",
        ridge_lambda=1e-4,
    )
    assert rec_fg["accepted"] is True

    # Exercise qr_svd fold mode + zero-delta noop on CrowdLoRAExpertAdapter
    exp_qr = CrowdLoRAExpertAdapter(8, 4, rank=2, alpha=4.0, seed=19)
    with torch.no_grad():
        exp_qr.w2_B.copy_(torch.ones((4, 2)))
    rec_qr = commit_sequential_wsp_round_inplace(
        exp_qr,
        map_basis,
        q8[:, 4:5].T,
        torch.ones((1, 4), dtype=torch.float64),
        weight_attr="w2",
        lora_fold_mode="qr_svd",
        expand_basis_on_commit=False,
        ridge_lambda=1e-4,
    )
    assert rec_qr["accepted"] is True
    assert rec_qr["fold_info"]["fold_method"] == "qr_svd_rank_preserving"

    with pytest.raises(TypeError, match="mapping"):
        commit_sequential_wsp_round_inplace(
            lin8, map_basis, q8[:, 4:5].T, torch.ones((1, 4)), weight_attr="weight", eval_fn=lambda _m: []  # type: ignore[return-value]
        )
    with pytest.raises(ValueError, match="> 0.0"):
        commit_sequential_wsp_round_inplace(
            lin8, map_basis, q8[:, 4:5].T, torch.ones((1, 4)), weight_attr="weight", eval_fn=lambda _m: {"E0_gen": 0.0}
        )
    with pytest.raises(ValueError, match="non-empty domain"):
        commit_sequential_wsp_round_inplace(
            lin8, map_basis, q8[:, 4:5].T, torch.ones((1, 4)), weight_attr="weight", eval_fn=lambda _m: {"domain_nlls": {}}
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        commit_sequential_wsp_round_inplace(
            lin8, map_basis, q8[:, 4:5].T, torch.ones((1, 4)), weight_attr="weight", eval_fn=lambda _m: {" ": 1.0}
        )
    with pytest.raises(TypeError, match="numeric"):
        commit_sequential_wsp_round_inplace(
            lin8, map_basis, q8[:, 4:5].T, torch.ones((1, 4)), weight_attr="weight", eval_fn=lambda _m: {"E0_gen": "bad"}
        )
    with pytest.raises(TypeError, match="historical_efficacies"):
        commit_sequential_wsp_round_inplace(
            lin8,
            map_basis,
            q8[:, 4:5].T,
            torch.ones((1, 4)),
            weight_attr="weight",
            eval_fn=lambda _m: {"E0_gen": 1.0, "historical_efficacies": "bad"},
        )
    with pytest.raises(ValueError, match="out of range"):
        commit_sequential_wsp_round_inplace(
            fg_model, fg_basis, q48[:, 4:5].T, torch.ones((1, 48)), layer_index=9, expert_index=1
        )
    with pytest.raises(ValueError, match="expert_index must be specified"):
        commit_sequential_wsp_round_inplace(
            fg_model, fg_basis, q48[:, 4:5].T, torch.ones((1, 48)), layer_index=0
        )
    with pytest.raises(ValueError, match="out of range"):
        commit_sequential_wsp_round_inplace(
            fg_model, fg_basis, q48[:, 4:5].T, torch.ones((1, 48)), layer_index=0, expert_index=9
        )
    with pytest.raises(ValueError, match="requires weight_attr"):
        commit_sequential_wsp_round_inplace(
            exp_qr, map_basis, q8[:, 4:5].T, torch.ones((1, 4)), weight_attr="weight"
        )
    with pytest.raises(ValueError, match="Could not resolve"):
        commit_sequential_wsp_round_inplace(
            nn.Identity(), map_basis, q8[:, 4:5].T, torch.ones((1, 4)), weight_attr="weight"
        )

    # Multi-expert vs single-expert CrowdLoRAMoEMLP checks + zero-delta noop + per_domain_nll
    single_res_moe = CrowdLoRAMoEMLP(
        CrowdRuntimeMLP(CrowdV1RuntimeConfig(vocab_size=16, hidden_size=8, intermediate_size=16, engram_buckets=0)),
        num_experts=2,
        top_k=1,
        rank=2,
        resident_expert_indices=[1],
    )
    multi_res_moe = CrowdLoRAMoEMLP(
        CrowdRuntimeMLP(CrowdV1RuntimeConfig(vocab_size=16, hidden_size=8, intermediate_size=16, engram_buckets=0)),
        num_experts=2,
        top_k=1,
        rank=2,
    )
    with pytest.raises(ValueError, match="expert_index must be specified"):
        commit_sequential_wsp_round_inplace(
            multi_res_moe, map_basis, q8[:, 4:5].T, torch.ones((1, 16)), weight_attr="w2"
        )
    with pytest.raises(ValueError, match="not resident"):
        commit_sequential_wsp_round_inplace(
            single_res_moe, map_basis, q8[:, 4:5].T, torch.ones((1, 16)), expert_index=0, weight_attr="w2"
        )
    # Zero delta noop & per_domain_nll + retention_ratio + novel skill discovery
    p_calls = {"n": 0}

    def _per_domain_eval(_m: nn.Module) -> dict[str, Any]:
        p_calls["n"] += 1
        return {
            "per_domain_nll": {"E0_gen": 1.5, "custom_novel": 2.0 if p_calls["n"] == 1 else 0.8},
            "retention_ratio": 1.0,
        }

    rec_zero_delta = commit_sequential_wsp_round_inplace(
        single_res_moe,
        map_basis,
        q8[:, 4:5].T,
        torch.zeros((1, 16), dtype=torch.float64),
        weight_attr="w2",
        eval_fn=_per_domain_eval,
        baseline_general_nll=1.5,
        historical_baseline_nlls={"r1": 1.0},
        expand_basis_on_commit=False,
    )
    assert rec_zero_delta["accepted"] is True
    assert rec_zero_delta["fold_info"]["fold_method"] == "zero_delta_noop"

    # 4. Canonical JSON, state SHA-256 & receipt provenance tampering checks
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert len(compute_payload_sha256({"nll_mean": 1.2, "retention_ratio": 0.99, "anse_energy": -0.3})) == 64
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(TypeError, match="Mapping"):
        compute_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"a": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"a": torch.tensor([float("nan")])})
    with pytest.raises(TypeError, match="Mapping"):
        verify_sequential_wsp_receipt_provenance("not_a_dict")  # type: ignore[arg-type]
    assert verify_sequential_wsp_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert (
        verify_sequential_wsp_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": "wrong", "schema": SCHEMA_VERSION}
        )
        is False
    )
    tampered = dict(rec_qr)
    tampered["nll_mean"] = 99.0
    assert verify_sequential_wsp_receipt_provenance(tampered) is False
    nan_rec = dict(rec_qr)
    nan_rec["nll_mean"] = float("nan")
    assert verify_sequential_wsp_receipt_provenance(nan_rec) is False

    # 5. AGPL-3.0 header, CI-safe fallback + optional ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn19", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn19", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/sequential_nullspace_memory.py", TARGET_MODULE_PATH),
        ("tests/test_sequential_nullspace_memory_scenarios.py", TEST_FILE_PATH),
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

    sample_telemetry_json = json.dumps(
        {
            "object": RECEIPT_OBJECT,
            "nll_mean": 0.842,
            "retention_ratio": 0.994,
            "anse_energy": -0.615,
            "accepted": True,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {sample_telemetry_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1119,
        pr_title="feat(swarm): lifelong sequential null-space basis tracker for non-interfering multi-round WSP skill editing",
        pr_body="Implements NEURAL-CTRL-DYN-19 sequential null-space memory basis tracker and deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): lifelong sequential null-space basis tracker for non-interfering multi-round WSP skill editing"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
