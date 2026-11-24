# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-05: decision-energy rank-1 skill weight steering."""

from __future__ import annotations

import importlib.util
import json
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
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.decision_skill_weight_steering import (
    DEFAULT_ANSE_MAX_PAIN,
    MAX_PERTURBATION_RATIO_CEILING,
    DecisionSkillSteeringConfig,
    apply_verified_skill_edit_inplace,
    compute_contrastive_skill_direction,
    compute_target_state_sha256,
    inject_decision_skill_into_model_inplace,
    make_skill_energy_eval_fn,
    restore_target_state_inplace,
    search_decision_energy_rank1_edit,
    snapshot_target_state,
)
from hivemind.swarm.gcp_t4_crowd_diloco import CrowdLoRALinear

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover
    ZeroStubAudit = None  # type: ignore[assignment,misc]

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "decision_skill_weight_steering.py"
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


def _make_orthogonal_basis(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((dim, dim), generator=gen, dtype=torch.float64)
    q_mat, r_mat = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r_mat))
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    return (q_mat * signs.unsqueeze(0)).contiguous()


def _make_orthogonal_zero_mean_vectors(n_vecs: int, block_dim: int, seed: int) -> torch.Tensor:
    assert n_vecs < block_dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ones = torch.ones((block_dim, 1), dtype=torch.float64)
    mat = torch.cat(
        [ones, torch.randn((block_dim, block_dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    q_mat, _ = torch.linalg.qr(mat)
    return q_mat[:, 1 : 1 + n_vecs].T.to(torch.float32).contiguous()


def _build_tool_skill_crowd_model(
    *,
    use_moe: bool = True,
    num_layers: int = 4,
    hidden_size: int = 96,
    intermediate_size: int = 384,
    vocab_size: int = 64,
    seed: int = 20261124,
) -> tuple[CrowdV1WithEngram, dict[str, list[int]]]:
    """Build a 4-layer dense or Sparse MoE model lacking tool-call token preference pre-steering."""
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

    # Vocabulary partitions:
    # [0..4]: structured tool-call tokens (<tool_call>, {, "name", :, })
    # [5..8]: tool-intent trigger tokens
    # [9..35]: general prose tokens
    # [36..63]: specialist math/code tokens
    tool_tokens = [0, 1, 2, 3, 4]
    intent_tokens = [5, 6, 7, 8]
    gen_tokens = list(range(9, 36))
    spec_tokens = list(range(36, vocab_size))

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        gen_vecs = _make_orthogonal_zero_mean_vectors(len(gen_tokens), 32, seed=seed + 1) * 0.65
        spec_vecs = _make_orthogonal_zero_mean_vectors(len(spec_tokens), 30, seed=seed + 2) * 0.65
        intent_vecs = _make_orthogonal_zero_mean_vectors(len(intent_tokens), 16, seed=seed + 3) * 0.70

        # Tool tokens share a strong zero-mean direction u_tool in [80:96] plus orthogonal deltas
        tool_block = _make_orthogonal_zero_mean_vectors(len(tool_tokens) + 1, 16, seed=seed + 4)
        u_tool = tool_block[0]
        tool_vecs = (0.62 * u_tool.unsqueeze(0) + 0.14 * tool_block[1:])

        wte[9:36, 0:32] = gen_vecs
        wte[36:64, 32:62] = spec_vecs
        wte[5:9, 64:80] = intent_vecs
        wte[0:5, 80:96] = tool_vecs

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        # Populate realistic full-rank background weights across all 4 blocks
        for blk_idx, blk in enumerate(model.blocks):
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.q.weight[:16, :16].copy_(0.02 * torch.eye(16))
            blk.attn.k.weight[:8, :16].copy_(0.02 * torch.eye(8, 16))
            blk.attn.v.weight[:8, :16].copy_(0.02 * torch.eye(8, 16))
            blk.attn.o.weight[:16, :16].copy_(0.02 * torch.eye(16))

            if blk_idx < num_layers - 1:
                w3_bg = _make_orthogonal_basis(intermediate_size, seed=seed + 100 + blk_idx)[:hidden_size, :]
                blk.mlp.w3.weight.copy_(w3_bg.to(torch.float32) * 0.45)
                blk.mlp.w1.weight[:, :60].copy_(
                    torch.randn((intermediate_size, 60), generator=gen, dtype=torch.float32) * 0.02
                )
                blk.mlp.w2.weight[:, :60].copy_(
                    torch.randn((intermediate_size, 60), generator=gen, dtype=torch.float32) * 0.02
                )

        last_mlp = model.blocks[-1].mlp
        # Full-rank base w3 across all 384 intermediate units so erank is realistic (~90)
        q_w3 = _make_orthogonal_basis(intermediate_size, seed=seed + 200)[:hidden_size, :].to(torch.float32)
        last_mlp.w3.weight.copy_(q_w3 * 0.85)

        # General prose units [0..96] -> next prose token
        for idx, tok in enumerate(gen_tokens):
            next_tok = 9 + ((idx + 1) % len(gen_tokens))
            last_mlp.w1.weight[idx, 0:32].copy_(4.5 * wte[tok, 0:32])
            last_mlp.w2.weight[idx, 0:32].copy_(4.5 * wte[tok, 0:32])
            last_mlp.w3.weight[0:32, idx].copy_(1.35 * wte[next_tok, 0:32])

        # Specialist math/code units [128..192] -> next specialist token
        for idx, tok in enumerate(spec_tokens):
            next_tok = 36 + ((idx + 1) % len(spec_tokens))
            u_idx = 128 + idx
            last_mlp.w1.weight[u_idx, 32:62].copy_(4.5 * wte[tok, 32:62])
            last_mlp.w2.weight[u_idx, 32:62].copy_(4.5 * wte[tok, 32:62])
            last_mlp.w3.weight[32:62, u_idx].copy_(1.35 * wte[next_tok, 32:62])

        # Tool-intent trigger units [320..323] fire strongly on intent tokens,
        # but pre-steering w3 outputs neutral/zero tool-preference -> high tool NLL (~ln(64) = 4.16)
        for idx, tok in enumerate(intent_tokens):
            u_idx = 320 + idx
            last_mlp.w1.weight[u_idx, 64:80].copy_(14.0 * wte[tok, 64:80])
            last_mlp.w2.weight[u_idx, 64:80].copy_(14.0 * wte[tok, 64:80])
            last_mlp.w3.weight[:, u_idx].zero_()

    if use_moe:
        upcycle_crowd_v1_to_sparse_moe(
            model,
            split_mode="lora_domain_experts",
            num_experts=4,
            top_k=2,
            lora_rank=8,
            lora_alpha=16.0,
            routing_mode="sigma_topk",
            seed=seed + 50,
        )
        with torch.no_grad():
            last_moe = model.blocks[-1].mlp
            assert isinstance(last_moe, CrowdLoRAMoEMLP)
            last_moe.router.weight.zero_()
            u_gen_dir = wte[9:36, 0:32].mean(dim=0)
            u_spec_dir = wte[36:64, 32:62].mean(dim=0)
            u_int_dir = wte[5:9, 64:80].mean(dim=0)
            # Route general -> expert 0, math/code -> expert 2, tool-intent -> expert 1
            last_moe.router.weight[0, 0:32].copy_(4.0 * u_gen_dir)
            last_moe.router.weight[0, 64:80].copy_(-4.0 * u_int_dir)
            last_moe.router.weight[2, 32:62].copy_(4.0 * u_spec_dir)
            last_moe.router.weight[2, 64:80].copy_(-4.0 * u_int_dir)
            last_moe.router.weight[1, 64:80].copy_(5.0 * u_int_dir)
            last_moe.router.weight[1, 0:32].copy_(-4.0 * u_gen_dir)
            last_moe.router.weight[1, 32:62].copy_(-4.0 * u_spec_dir)
            last_moe.router.weight[3].fill_(-3.0)

    token_groups = {
        "tool": tool_tokens,
        "intent": intent_tokens,
        "general": gen_tokens,
        "specialist": spec_tokens,
    }
    return model, token_groups


# ---------------------------------------------------------------------------
# S1: Multi-Layer MoE & Dense Structured Tool-Calling Injection (>25% NLL Drop)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_moe", [True, False])
def test_s1_multilayer_moe_and_dense_tool_call_steering(use_moe: bool) -> None:
    """Training-free injection lowers tool-call NLL >25% with <=+0.5% prose drift and <0.02% perturbation."""
    model, tok_groups = _build_tool_skill_crowd_model(use_moe=use_moe, seed=20260101 + int(use_moe))
    model.eval()
    for p in model.parameters():
        assert p.grad is None

    tool_toks = tok_groups["tool"]
    intent_toks = tok_groups["intent"]
    gen_toks = tok_groups["general"]
    spec_toks = tok_groups["specialist"]

    # Tool-call sequences: [general_prefix, intent_token, target_tool_token]
    # Scoring from tool_score_from=1 evaluates predicting target_tool_token right after intent_token!
    pos_seqs = torch.tensor(
        [
            [gen_toks[(i * 2) % len(gen_toks)], intent_toks[i % len(intent_toks)], tool_toks[i % len(tool_toks)]]
            for i in range(12)
        ],
        dtype=torch.long,
    )
    neg_seqs = torch.tensor(
        [
            [gen_toks[(i + 0) % len(gen_toks)], gen_toks[(i + 1) % len(gen_toks)], gen_toks[(i + 2) % len(gen_toks)]]
            for i in range(12)
        ],
        dtype=torch.long,
    )
    spec_seqs = torch.tensor(
        [
            [spec_toks[(i + 0) % len(spec_toks)], spec_toks[(i + 1) % len(spec_toks)], spec_toks[(i + 2) % len(spec_toks)]]
            for i in range(12)
        ],
        dtype=torch.long,
    )
    preserve_map = {
        "E0_general_prose": neg_seqs,
        "E1_math_code": spec_seqs,
    }

    report = inject_decision_skill_into_model_inplace(
        model,
        pos_seqs,
        neg_seqs,
        preserve_map,
        target_token_ids=tool_toks,
        baseline_token_ids=gen_toks + spec_toks,
        layer_indices=[len(model.blocks) - 1],
        expert_idx=1 if use_moe else None,
        tool_score_from=1,
        general_score_from=0,
        batch_size=4,
    )

    # Verify zero gradient descent occurred
    for p in model.parameters():
        assert p.grad is None

    assert report["committed"] is True
    assert report["committed_layers"] == [len(model.blocks) - 1]
    assert report["rolled_back_layers"] == []
    assert report["tool_nll_gain_pct"] > 25.0
    assert report["general_drift_pct"] <= 0.5
    assert abs(report["domain_drifts_pct"]["E1_math_code"]) < 1e-2

    layer_rep = report["layer_reports"][0]
    assert layer_rep["committed"] is True
    assert layer_rep["rolled_back"] is False
    assert layer_rep["delta_energy"] < 0.0
    assert layer_rep["guard_ok"] is True
    assert layer_rep["perturbation_bound_respected"] is True
    assert 0.0 < layer_rep["norm_perturbation_ratio"] < MAX_PERTURBATION_RATIO_CEILING
    assert layer_rep["before_sha256"] != layer_rep["final_sha256"]
    assert len(layer_rep["receipt_sha256"]) == 64

    if use_moe:
        last_moe = model.blocks[-1].mlp
        assert isinstance(last_moe, CrowdLoRAMoEMLP)
        assert last_moe.experts["1"].rank == 8
        assert last_moe.experts["1"].w3_B.shape == (96, 8)
        assert last_moe.experts["1"].w3_A.shape == (8, 384)
        assert layer_rep["lora_rank_preserved"] is True
        assert layer_rep["lora_rank_before"] == 8
        assert layer_rep["lora_rank_after"] == 8
    else:
        assert layer_rep["effective_rank_perturbation_ratio"] is not None
        assert layer_rep["effective_rank_perturbation_ratio"] < MAX_PERTURBATION_RATIO_CEILING

    # Also verify MoE dense base_mlp.w3 branch (expert_idx=None on CrowdLoRAMoEMLP)
    if use_moe:
        moe_base_model, tok_g2 = _build_tool_skill_crowd_model(use_moe=True, seed=20260199)
        rep_base = inject_decision_skill_into_model_inplace(
            moe_base_model,
            pos_seqs,
            neg_seqs,
            neg_seqs,
            target_token_ids=tok_g2["tool"],
            expert_idx=None,
            tool_score_from=1,
            general_score_from=0,
        )
        assert rep_base["committed"] is True
        assert rep_base["tool_nll_gain_pct"] > 25.0
        assert rep_base["general_drift_pct"] <= 0.5


# ---------------------------------------------------------------------------
# S2: Exact Preserve-Subspace Orthogonality (<1e-5 Drift on Math / Code)
# ---------------------------------------------------------------------------


def test_s2_exact_preserve_subspace_orthogonality_on_math_and_code() -> None:
    """Verify <1e-5 activation drift on protected math/code states via exact null-space solver."""
    d_in, d_out, vocab_size = 48, 32, 40
    q_basis = _make_orthogonal_basis(d_in, seed=20260201)
    gen = torch.Generator(device="cpu").manual_seed(20260202)

    # Protected subspaces: general prose in dims 0..5, math in dims 6..11, code in dims 12..17
    gen_pres = (torch.randn((40, 6), generator=gen, dtype=torch.float64) @ q_basis[:, :6].T).float()
    math_pres = (torch.randn((32, 6), generator=gen, dtype=torch.float64) @ q_basis[:, 6:12].T).float()
    code_pres = (torch.randn((32, 6), generator=gen, dtype=torch.float64) @ q_basis[:, 12:18].T).float()

    # Positive skill states carry both protected-subspace background AND strong skill signal in dim 20
    pos_bg = torch.randn((48, 18), generator=gen, dtype=torch.float64) * 0.5
    pos_skill = 3.5 + 0.4 * torch.randn((48, 1), generator=gen, dtype=torch.float64)
    pos_hidden = (
        pos_bg @ q_basis[:, :18].T + pos_skill @ q_basis[:, 20:21].T
    ).float()

    # Negative states carry correlated protected-subspace background + negative/zero skill dim 20
    neg_bg = torch.randn((48, 18), generator=gen, dtype=torch.float64) * 0.5
    neg_skill = -1.0 + 0.4 * torch.randn((48, 1), generator=gen, dtype=torch.float64)
    neg_hidden = (
        neg_bg @ q_basis[:, :18].T + neg_skill @ q_basis[:, 20:21].T
    ).float()

    # Exercise both sequence and mapping inputs for preserve_hidden
    dir_seq = compute_contrastive_skill_direction(
        pos_hidden,
        neg_hidden,
        [gen_pres, math_pres, code_pres],
    )
    dir_map = compute_contrastive_skill_direction(
        pos_hidden,
        neg_hidden,
        {"general": gen_pres, "math": math_pres, "code": code_pres},
    )
    assert dir_seq["nullspace_mode_used"] == "exact_nullspace"
    assert dir_seq["nullspace_dim"] == d_in - 18
    assert dir_seq["passes_snr_threshold"] is True
    assert dir_seq["oof_snr"] > 3.0
    assert dir_seq["preserve_max_abs"] < 1e-6
    assert dir_seq["preserve_rms"] < 1e-6
    assert float((dir_seq["direction"] - dir_map["direction"]).abs().max().item()) < 1e-6

    # Build a dense target projection and unembed matrix preferring tool tokens [0, 1, 2]
    proj = nn.Linear(d_in, d_out, bias=False, dtype=torch.float32)
    with torch.no_grad():
        proj.weight.copy_(torch.randn((d_out, d_in), generator=gen, dtype=torch.float32) * 1.5)
    unembed_w = torch.randn((vocab_size, d_out), generator=gen, dtype=torch.float32) * 0.2
    unembed_w[:3, :] = 0.0
    unembed_w[:3, 0] = 3.0

    with torch.no_grad():
        math_before = proj(math_pres)
        code_before = proj(code_pres)
        gen_before = proj(gen_pres)

    def _eval_orth(_mod: Any) -> dict[str, Any]:
        with torch.no_grad():
            pos_logits = proj(pos_hidden) @ unembed_w.T
            tool_nll = float(
                torch.nn.functional.cross_entropy(
                    pos_logits, torch.zeros(pos_hidden.shape[0], dtype=torch.long)
                ).item()
            )
            gen_drift = float((proj(gen_pres) - gen_before).pow(2).mean().item())
            math_drift = float((proj(math_pres) - math_before).pow(2).mean().item())
            return {
                "tool_nll": tool_nll,
                "general_nll": 1.0 + gen_drift + math_drift,
            }

    search_res = search_decision_energy_rank1_edit(
        proj,
        dir_seq,
        unembed_w,
        _eval_orth,
        target_token_ids=[0, 1, 2],
    )
    assert search_res["best_alpha"] is not None
    assert search_res["delta_energy"] < 0.0

    commit_rec = apply_verified_skill_edit_inplace(proj, search_res, eval_energy_fn=_eval_orth)
    assert commit_rec["committed"] is True

    with torch.no_grad():
        math_after = proj(math_pres)
        code_after = proj(code_pres)
        gen_after = proj(gen_pres)
        pos_logits_after = proj(pos_hidden) @ unembed_w.T

    math_act_drift = float((math_after - math_before).abs().max().item())
    code_act_drift = float((code_after - code_before).abs().max().item())
    gen_act_drift = float((gen_after - gen_before).abs().max().item())
    assert math_act_drift < 1e-5
    assert code_act_drift < 1e-5
    assert gen_act_drift < 1e-5
    assert float(pos_logits_after[:, 0].mean().item()) > float((pos_hidden @ proj.weight.T @ unembed_w.T)[:, 0].mean().item()) or commit_rec["tool_nll_gain_pct"] > 0.0

    # Also verify soft preserve mode, no-preserve None, whitening=False, N<4 fallback, and strict empty nullspace
    dir_none = compute_contrastive_skill_direction(
        pos_hidden[:3],
        neg_hidden[:3],
        None,
        use_covariance_whitening=False,
    )
    assert dir_none["nullspace_dim"] == d_in
    assert dir_none["preserve_max_abs"] == 0.0

    full_rank_pres = torch.randn((64, d_in), generator=gen, dtype=torch.float32)
    dir_soft = compute_contrastive_skill_direction(pos_hidden, neg_hidden, full_rank_pres, nullspace_mode="soft")
    assert dir_soft["nullspace_mode_used"] == "soft_preserve_whitened"

    dir_strict_empty = compute_contrastive_skill_direction(
        pos_hidden, neg_hidden, full_rank_pres, nullspace_mode="strict"
    )
    assert dir_strict_empty["noop_reason"] == "empty_preserve_nullspace"
    assert float(dir_strict_empty["direction"].norm().item()) == 0.0

    # Pure identical zero-contrast noise fails SNR gate cleanly
    noise_pos = torch.randn((24, 16), generator=gen, dtype=torch.float32)
    noise_neg = torch.randn((24, 16), generator=gen, dtype=torch.float32)
    dir_noise = compute_contrastive_skill_direction(noise_pos, noise_neg, min_contrast_snr=5.0)
    assert dir_noise["passes_snr_threshold"] is False
    assert dir_noise["noop_reason"] == "contrast_snr_below_threshold"


# ---------------------------------------------------------------------------
# S3: ANSE Maximum Pain Rejection & Atomic Bitwise Rollback on >+1.0% Guard Violation
# ---------------------------------------------------------------------------


def test_s3_anse_maximum_pain_rejection_and_bitwise_rollback() -> None:
    """Reject and bitwise rollback whenever all candidates violate the +1.0% general-domain guard."""
    gen = torch.Generator(device="cpu").manual_seed(20260301)
    d_in, d_out = 32, 24
    layer = nn.Linear(d_in, d_out, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_((torch.randn((d_out, d_in), generator=gen, dtype=torch.float32) * 0.5).to(torch.bfloat16))

    before_sha = compute_target_state_sha256(layer)
    w_snap = layer.weight.detach().clone()

    v_in = torch.randn(d_in, generator=gen, dtype=torch.float64)
    v_in = v_in / v_in.norm()
    u_out = torch.randn(d_out, generator=gen, dtype=torch.float64)
    u_out = u_out / u_out.norm()

    # Every non-zero alpha lowers tool_nll slightly (-0.4) but causes +2.5% general_nll drift (> +1.0% guard!)
    def _toxic_eval_fn(_mod: nn.Module, cand_alpha: float) -> dict[str, Any]:
        if abs(cand_alpha) == 0.0:
            return {"tool_nll": 3.0, "general_nll": 2.0}
        return {"tool_nll": 2.6, "general_nll": 2.05}  # +2.5% drift!

    search_res = search_decision_energy_rank1_edit(
        layer,
        v_in,
        u_out,
        _toxic_eval_fn,
    )
    assert search_res["best_alpha"] is None
    assert search_res["all_rejected_maximum_pain"] is True
    assert search_res["rolled_back_during_search"] is True
    assert search_res["noop_reason"] == "maximum_pain_guard_violation"
    assert len(search_res["candidates"]) > 0
    for cand in search_res["candidates"]:
        assert cand["guard_ok"] is False
        assert cand["anse_energy"] >= DEFAULT_ANSE_MAX_PAIN
    assert torch.equal(layer.weight, w_snap)
    assert compute_target_state_sha256(layer) == before_sha

    # Committing the rejected search result leaves weights bitwise unchanged
    rec_noop = apply_verified_skill_edit_inplace(layer, search_res, eval_energy_fn=_toxic_eval_fn)
    assert rec_noop["committed"] is False
    assert rec_noop["rolled_back"] is True
    assert rec_noop["rollback_bitwise_verified"] is True
    assert rec_noop["final_sha256"] == before_sha
    assert torch.equal(layer.weight, w_snap)

    # Also test forced candidate commit against strict verifier -> applies -> detects >+1.0% -> atomic rollback!
    forced_alpha = 0.0001 * float(layer.weight.detach().float().norm().item())
    rec_forced_rb = apply_verified_skill_edit_inplace(
        layer,
        {"alpha": forced_alpha, "u_out": u_out, "v_in": v_in},
        eval_energy_fn=_toxic_eval_fn,
    )
    assert rec_forced_rb["committed"] is False
    assert rec_forced_rb["rolled_back"] is True
    assert rec_forced_rb["rollback_bitwise_verified"] is True
    assert rec_forced_rb["reject_reason"] == "maximum_pain_guard_violation"
    assert rec_forced_rb["candidate_sha256"] != before_sha
    assert rec_forced_rb["final_sha256"] == before_sha
    assert torch.equal(layer.weight, w_snap)

    # Non-finite child output and non-negative Delta E also trigger atomic rollback
    def _nan_eval(_mod: nn.Module, cand_alpha: float) -> dict[str, Any]:
        if cand_alpha == 0.0:
            return {"tool_nll": 3.0, "general_nll": 2.0}
        return {"tool_nll": float("nan"), "general_nll": 2.0}

    rec_nan = apply_verified_skill_edit_inplace(
        layer,
        {"alpha": forced_alpha, "u_out": u_out, "v_in": v_in},
        eval_energy_fn=_nan_eval,
    )
    assert rec_nan["committed"] is False
    assert rec_nan["rolled_back"] is True
    assert rec_nan["reject_reason"] == "non_finite_child_output"
    assert rec_nan["final_sha256"] == before_sha

    def _no_gain_eval(_mod: nn.Module, cand_alpha: float) -> float:
        return 2.0 if cand_alpha == 0.0 else 2.05

    rec_nogain = apply_verified_skill_edit_inplace(
        layer,
        {"alpha": forced_alpha, "u_out": u_out, "v_in": v_in},
        eval_energy_fn=_no_gain_eval,
    )
    assert rec_nogain["committed"] is False
    assert rec_nogain["rolled_back"] is True
    assert rec_nogain["reject_reason"] == "non_negative_delta_energy"
    assert rec_nogain["final_sha256"] == before_sha

    # Perturbation bound exceeded triggers rollback too
    too_big_alpha = 0.05 * float(layer.weight.detach().float().norm().item())
    rec_bound = apply_verified_skill_edit_inplace(
        layer,
        {"alpha": too_big_alpha, "u_out": u_out, "v_in": v_in},
        eval_energy_fn=lambda _m, a: 3.0 if a == 0.0 else 1.0,
    )
    assert rec_bound["committed"] is False
    assert rec_bound["rolled_back"] is True
    assert rec_bound["reject_reason"] == "perturbation_bound_exceeded"
    assert rec_bound["final_sha256"] == before_sha


# ---------------------------------------------------------------------------
# S4: LoRA Factor-Space Rank-1 Fold-In Without Increasing Adapter Rank r
# ---------------------------------------------------------------------------


def test_s4_lora_factor_space_rank1_fold_in_preserves_rank() -> None:
    """Fold rank-1 skill steering into CrowdLoRALinear and CrowdLoRAExpertAdapter without increasing r."""
    gen = torch.Generator(device="cpu").manual_seed(20260401)
    d_in, d_out, rank = 64, 48, 8

    # 1. Fresh CrowdLoRALinear (B=0) -> exact zero-slot rank-1 fold-in with base bitwise untouched
    base_lin = nn.Linear(d_in, d_out, bias=False, dtype=torch.float32)
    with torch.no_grad():
        base_lin.weight.copy_(torch.randn((d_out, d_in), generator=gen, dtype=torch.float32) * 0.4)
    lora_lin = CrowdLoRALinear(base_lin, rank=rank, alpha=16.0)
    base_snap = lora_lin.base.weight.detach().clone()

    q_in = _make_orthogonal_basis(d_in, seed=20260405)
    x_pres = (torch.randn((24, 12), generator=gen, dtype=torch.float64) @ q_in[:, :12].T).float()
    v_in = q_in[:, 15].clone()  # strictly orthogonal to x_pres
    u_out = torch.randn(d_out, generator=gen, dtype=torch.float64)
    u_out = u_out / u_out.norm()
    x_tool = (3.0 * v_in.unsqueeze(0).expand(16, -1)).float()

    def _eval_lora(mod: CrowdLoRALinear) -> dict[str, Any]:
        with torch.no_grad():
            out_tool = mod(x_tool)
            align = float((out_tool.double() @ u_out).mean().item())
            base_align = float((mod.base(x_tool).double() @ u_out).mean().item())
            pres_diff = float((mod(x_pres) - mod.base(x_pres)).abs().max().item())
            return {
                "tool_nll": 3.0 - 100.0 * (align - base_align),
                "general_nll": 1.0 + pres_diff,
            }

    rec_fresh = apply_verified_skill_edit_inplace(
        lora_lin,
        skill_dir=v_in,
        unembed_target_dir=u_out,
        eval_energy_fn=_eval_lora,
    )
    assert rec_fresh["committed"] is True
    assert rec_fresh["fold_method"] == "zero_slot_exact"
    assert rec_fresh["lora_rank_before"] == rank
    assert rec_fresh["lora_rank_after"] == rank
    assert rec_fresh["lora_rank_preserved"] is True
    assert lora_lin.rank == rank
    assert lora_lin.lora_B.shape == (d_out, rank)
    assert lora_lin.lora_A.shape == (rank, d_in)
    assert torch.equal(lora_lin.base.weight, base_snap)
    assert rec_fresh["norm_perturbation_ratio"] < MAX_PERTURBATION_RATIO_CEILING
    assert rec_fresh["effective_rank_perturbation_ratio"] is not None
    assert rec_fresh["effective_rank_perturbation_ratio"] < MAX_PERTURBATION_RATIO_CEILING

    # Verify exact rank-1 equivalence on zero-slot fold-in and exact preserve orthogonality
    with torch.no_grad():
        actual_delta = lora_lin.delta_weight_fp32().double()
        expected_delta = float(rec_fresh["alpha"]) * torch.outer(u_out, v_in)
        assert float((actual_delta - expected_delta).abs().max().item()) < 1e-6
        assert float((lora_lin(x_pres) - lora_lin.base(x_pres)).abs().max().item()) < 1e-5

    # 2. Trained full-rank LoRA (all r=8 modes populated!) -> preserve-safe rank-r fold-in
    base_lin2 = nn.Linear(d_in, d_out, bias=False, dtype=torch.float32)
    with torch.no_grad():
        base_lin2.weight.copy_(torch.randn((d_out, d_in), generator=gen, dtype=torch.float32) * 0.4)
    lora_full = CrowdLoRALinear(base_lin2, rank=rank, alpha=16.0)
    with torch.no_grad():
        lora_full.lora_B.copy_(torch.randn((d_out, rank), generator=gen, dtype=torch.float32) * 0.25)
        lora_full.lora_A.copy_(torch.randn((rank, d_in), generator=gen, dtype=torch.float32) * 0.25)
        pres_out_before = lora_full(x_pres).clone()

    def _eval_full_lora(mod: CrowdLoRALinear) -> dict[str, Any]:
        with torch.no_grad():
            tool_proj = float((mod(x_tool).double() @ u_out).mean().item())
            pres_drift = float((mod(x_pres) - pres_out_before).abs().max().item())
            return {
                "tool_nll": 4.0 - 50.0 * tool_proj,
                "general_nll": 1.0 + pres_drift,
            }

    rec_full = apply_verified_skill_edit_inplace(
        lora_full,
        skill_dir=v_in,
        unembed_target_dir=u_out,
        eval_energy_fn=_eval_full_lora,
    )
    assert rec_full["committed"] is True
    assert rec_full["lora_rank_preserved"] is True
    assert lora_full.lora_B.shape == (d_out, rank)
    assert lora_full.lora_A.shape == (rank, d_in)
    with torch.no_grad():
        assert float((lora_full(x_pres) - pres_out_before).abs().max().item()) < 1e-5

    # 3. Exercise rank-deficient qr_svd_exact, update_b_exact, qr_svd mode, and CrowdLoRAExpertAdapter
    exp_ad = CrowdLoRAExpertAdapter(d_in, d_in, rank=rank, alpha=16.0, seed=20260409)
    with torch.no_grad():
        exp_ad.w3_B.zero_()
        exp_ad.w3_B[:, :2].copy_(torch.randn((d_in, 2), generator=gen, dtype=torch.float32) * 0.2)
        exp_ad.w3_A[:2, :].copy_(torch.randn((2, d_in), generator=gen, dtype=torch.float32) * 0.2)

    u_exp = torch.randn(d_in, generator=gen, dtype=torch.float64)
    u_exp = u_exp / u_exp.norm()
    rec_qr = apply_verified_skill_edit_inplace(
        exp_ad,
        skill_dir=v_in,
        unembed_target_dir=u_exp,
        eval_energy_fn=lambda m: {"tool_nll": 3.0 - 80.0 * float((m.w3_B.double() @ m.w3_A.double() @ v_in.float().double() @ u_exp).item()), "general_nll": 1.0},
        weight_attr="w3",
        reference_weight_norm=100.0,
        lora_fold_mode="qr_svd",
    )
    assert rec_qr["committed"] is True
    assert rec_qr["fold_method"] == "qr_svd_exact"
    assert exp_ad.w3_B.shape == (d_in, rank)
    assert exp_ad.w3_A.shape == (rank, d_in)

    # update_b mode on full-rank LoRA where v_in lies in row(A)
    with torch.no_grad():
        lora_full.lora_A[0, :].copy_(v_in.float())
    rec_upb = apply_verified_skill_edit_inplace(
        lora_full,
        skill_dir=v_in,
        unembed_target_dir=u_out,
        eval_energy_fn=_eval_full_lora,
        lora_fold_mode="update_b",
    )
    assert rec_upb["committed"] is True
    assert rec_upb["fold_method"] in {"update_b_exact", "update_b_projected"}

    # Fallback qr_svd_truncated when u_out perp col(B) and v_in perp row(A)
    b_orth = torch.zeros((d_out, 4), dtype=torch.float32)
    b_orth[:4, :] = torch.eye(4)
    a_orth = torch.zeros((4, d_in), dtype=torch.float32)
    a_orth[:, :4] = torch.eye(4)

    class _SimpleLoRA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_B = nn.Parameter(b_orth.clone())
            self.lora_A = nn.Parameter(a_orth.clone())
            self.scaling = 1.0

    simp = _SimpleLoRA()
    u_perp = torch.zeros(d_out, dtype=torch.float64)
    u_perp[10] = 1.0
    v_perp = torch.zeros(d_in, dtype=torch.float64)
    v_perp[10] = 1.0
    s_orth = search_decision_energy_rank1_edit(
        simp,
        v_perp,
        u_perp,
        lambda m, a: {"tool_nll": 3.0 - abs(a) * 100.0, "general_nll": 1.0},
        reference_weight_norm=50.0,
        include_negative_alphas=True,
    )
    assert s_orth["best_alpha"] is not None
    assert s_orth["best_candidate"]["fold_method"] == "qr_svd_truncated"


# ---------------------------------------------------------------------------
# S5: Validation Errors, Edge Cases & Security / Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_and_security_stub_compliance_gates() -> None:
    """Exercise validation errors, edge branches, and all HiveMind security/stub/gatekeeper audits."""
    # 1. DecisionSkillSteeringConfig validation
    with pytest.raises(TypeError, match="real number"):
        DecisionSkillSteeringConfig(ridge_scale=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0.0"):
        DecisionSkillSteeringConfig(ridge_scale=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        DecisionSkillSteeringConfig(max_norm_perturbation_ratio=0.0)
    with pytest.raises(ValueError, match="max_norm_perturbation_ratio"):
        DecisionSkillSteeringConfig(max_norm_perturbation_ratio=0.001)
    with pytest.raises(ValueError, match="max_effective_rank_perturbation_ratio"):
        DecisionSkillSteeringConfig(max_effective_rank_perturbation_ratio=0.005)
    with pytest.raises(ValueError, match="nullspace_eigen_ratio"):
        DecisionSkillSteeringConfig(nullspace_eigen_ratio=1.0)
    with pytest.raises(ValueError, match="nullspace_mode"):
        DecisionSkillSteeringConfig(nullspace_mode="invalid")
    with pytest.raises(ValueError, match="lora_fold_mode"):
        DecisionSkillSteeringConfig(lora_fold_mode="invalid")
    with pytest.raises(ValueError, match="safe_drift_threshold"):
        DecisionSkillSteeringConfig(max_general_drift=0.01, safe_drift_threshold=0.02)
    with pytest.raises(TypeError, match="bool"):
        DecisionSkillSteeringConfig(use_covariance_whitening="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        DecisionSkillSteeringConfig(enforce_effective_rank_bound="no")  # type: ignore[arg-type]

    # 2. Target resolution & snapshot/restore validation
    w_ok = torch.randn((8, 4), dtype=torch.float32)
    with pytest.raises(TypeError, match="weight_attr"):
        compute_target_state_sha256(w_ok, weight_attr="w3")
    with pytest.raises(TypeError, match="nn.Module or 2D"):
        compute_target_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 2D"):
        compute_target_state_sha256(torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        compute_target_state_sha256(torch.ones((4, 4), dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        compute_target_state_sha256(torch.tensor([[1.0, float("nan")]]))
    with pytest.raises(ValueError, match="expert_idx"):
        compute_target_state_sha256(nn.Linear(4, 4), expert_idx=0)
    with pytest.raises(ValueError, match="2D weight"):
        compute_target_state_sha256(nn.ReLU())
    with pytest.raises(ValueError, match="no attribute"):
        compute_target_state_sha256(nn.Linear(4, 4), weight_attr="missing")
    with pytest.raises(ValueError, match="non-empty string"):
        compute_target_state_sha256(nn.Linear(4, 4), weight_attr="  ")
    with pytest.raises(ValueError, match="non-empty mapping"):
        restore_target_state_inplace(w_ok, {})
    snap_ok = snapshot_target_state(w_ok)
    assert restore_target_state_inplace(w_ok, snap_ok) == compute_target_state_sha256(w_ok)
    with pytest.raises(ValueError, match="do not match"):
        restore_target_state_inplace(w_ok, {"other": w_ok})
    with pytest.raises(ValueError, match="Snapshot mismatch"):
        restore_target_state_inplace(w_ok, {"weight": torch.zeros((2, 2))})

    # Custom nn.Module with .weight tensor, weight_attr, MoE error branches & CrowdRuntimeMLP weight_attr
    class _Holder(nn.Module):
        def __init__(self, tensor: torch.Tensor) -> None:
            super().__init__()
            self.weight = nn.Parameter(tensor)

    class _NoBiasHolder(nn.Module):
        def __init__(self, tensor: torch.Tensor) -> None:
            super().__init__()
            self.custom = nn.Parameter(tensor)

    class _BadLoRA(nn.Module):
        def __init__(self, *, scaling: float = 1.0, b_cols: int = 2, base_in: int = 4) -> None:
            super().__init__()
            self.lora_B = nn.Linear(b_cols, 4, bias=False)
            self.lora_A = nn.Linear(4, 3, bias=False)
            self.base = nn.Linear(base_in, 4, bias=False)
            self.scaling = scaling

    holder = _Holder(torch.randn((8, 4), dtype=torch.float64))
    assert len(compute_target_state_sha256(holder)) == 64
    holder_attr = _NoBiasHolder(torch.randn((8, 4), dtype=torch.float32))
    assert len(compute_target_state_sha256(holder_attr, weight_attr="custom")) == 64
    mlp_dense = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=8, intermediate_size=16))
    assert len(compute_target_state_sha256(mlp_dense, weight_attr="w3")) == 64
    with pytest.raises(ValueError, match="rank mismatch"):
        compute_target_state_sha256(_BadLoRA(b_cols=2))
    with pytest.raises(ValueError, match="scaling"):
        compute_target_state_sha256(_BadLoRA(scaling=0.0, b_cols=3))
    with pytest.raises(ValueError, match="base.weight shape"):
        compute_target_state_sha256(_BadLoRA(scaling=1.0, b_cols=3, base_in=5))
    moe_small = CrowdLoRAMoEMLP(mlp_dense, num_experts=2, top_k=1, rank=2)
    with pytest.raises(ValueError, match="not found"):
        compute_target_state_sha256(moe_small, expert_idx=9)
    with pytest.raises(ValueError, match="non-empty string"):
        compute_target_state_sha256(moe_small, expert_idx=0, weight_attr="  ")

    # 3. compute_contrastive_skill_direction validation & edge branches
    h_ok = torch.randn((6, 4), dtype=torch.float32)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_contrastive_skill_direction([[1.0]], h_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D or 3D"):
        compute_contrastive_skill_direction(torch.ones(4), h_ok)
    with pytest.raises(ValueError, match="finite"):
        compute_contrastive_skill_direction(torch.tensor([[1.0, float("nan"), 0.0, 0.0]]), h_ok)
    with pytest.raises(ValueError, match="must match pos_hidden width"):
        compute_contrastive_skill_direction(h_ok, torch.ones((6, 5)))
    with pytest.raises(ValueError, match="must not be empty"):
        compute_contrastive_skill_direction(h_ok, h_ok, [])
    with pytest.raises(ValueError, match="must not be empty"):
        compute_contrastive_skill_direction(h_ok, h_ok, {})
    with pytest.raises(TypeError, match="preserve_hidden"):
        compute_contrastive_skill_direction(h_ok, h_ok, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must match pos_hidden width"):
        compute_contrastive_skill_direction(h_ok, h_ok, torch.ones((4, 5)))
    with pytest.raises(ValueError, match="nullspace_mode"):
        compute_contrastive_skill_direction(h_ok, h_ok, nullspace_mode="bad")
    with pytest.raises(ValueError, match="nullspace_eigen_ratio"):
        compute_contrastive_skill_direction(h_ok, h_ok, nullspace_eigen_ratio=1.0)
    with pytest.raises(TypeError, match="bool"):
        compute_contrastive_skill_direction(h_ok, h_ok, use_covariance_whitening=1)  # type: ignore[arg-type]

    # Single-sample covariance branch (n <= 1) + 3D tensors + zero contrast mean + zero nullspace signal
    s1_dir = compute_contrastive_skill_direction(
        torch.tensor([[1.0, 2.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        use_covariance_whitening=True,
    )
    assert float(s1_dir["direction"].norm().item()) > 0.99
    z_3d = compute_contrastive_skill_direction(torch.ones((2, 3, 4)), torch.ones((2, 3, 4)))
    assert z_3d["noop_reason"] == "zero_contrast_mean"
    pres_all_signal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    z_ns = compute_contrastive_skill_direction(
        torch.tensor([[2.0, 0.0, 0.0, 0.0]] * 4),
        torch.tensor([[0.0, 0.0, 0.0, 0.0]] * 4),
        pres_all_signal,
    )
    assert z_ns["noop_reason"] == "zero_nullspace_signal"

    # 4. search_decision_energy_rank1_edit & apply_verified_skill_edit_inplace validation
    lin4 = nn.Linear(4, 8, bias=False)
    v4 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    u8 = torch.zeros(8, dtype=torch.float64)
    u8[0] = 1.0
    with pytest.raises(TypeError, match="callable"):
        search_decision_energy_rank1_edit(lin4, v4, u8, "not_callable")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        search_decision_energy_rank1_edit(lin4, v4, u8, lambda _m: 1.0, include_negative_alphas=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DecisionSkillSteeringConfig"):
        search_decision_energy_rank1_edit(lin4, v4, u8, lambda _m: 1.0, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="skill_dir mapping"):
        search_decision_energy_rank1_edit(lin4, {}, u8, lambda _m: 1.0)
    with pytest.raises(TypeError, match="torch.Tensor"):
        search_decision_energy_rank1_edit(lin4, "bad", u8, lambda _m: 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        search_decision_energy_rank1_edit(lin4, torch.tensor([1.0, float("nan"), 0.0, 0.0]), u8, lambda _m: 1.0)
    with pytest.raises(ValueError, match="length 4"):
        search_decision_energy_rank1_edit(lin4, torch.ones(3), u8, lambda _m: 1.0)
    with pytest.raises(ValueError, match="alpha_grid"):
        search_decision_energy_rank1_edit(lin4, v4, u8, lambda _m: 1.0, alpha_grid=[])
    with pytest.raises(TypeError, match="real number"):
        search_decision_energy_rank1_edit(lin4, v4, u8, lambda _m: 1.0, alpha_grid=[True])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite"):
        search_decision_energy_rank1_edit(lin4, v4, u8, lambda _m: 1.0, alpha_grid=[float("nan")])
    with pytest.raises(ValueError, match="zero norm"):
        search_decision_energy_rank1_edit(lin4, v4, torch.zeros(8), lambda _m: 1.0)
    with pytest.raises(ValueError, match="1D or 2D"):
        search_decision_energy_rank1_edit(lin4, v4, torch.ones((2, 2, 8)), lambda _m: 1.0)
    with pytest.raises(ValueError, match="shape"):
        search_decision_energy_rank1_edit(lin4, v4, torch.ones((4, 7)), lambda _m: 1.0)
    with pytest.raises(ValueError, match="finite"):
        search_decision_energy_rank1_edit(lin4, v4, torch.full((4, 8), float("nan")), lambda _m: 1.0)
    with pytest.raises(ValueError, match="must not be empty"):
        search_decision_energy_rank1_edit(lin4, v4, torch.ones((10, 8)), lambda _m: 1.0, target_token_ids=[])
    with pytest.raises(ValueError, match="must be 1D"):
        search_decision_energy_rank1_edit(lin4, v4, torch.ones((10, 8)), lambda _m: 1.0, target_token_ids=torch.zeros((2, 2), dtype=torch.long))
    with pytest.raises(ValueError, match=r"\[0, 10\)"):
        search_decision_energy_rank1_edit(lin4, v4, torch.ones((10, 8)), lambda _m: 1.0, target_token_ids=[99])
    with pytest.raises(TypeError, match="integer dtype"):
        search_decision_energy_rank1_edit(lin4, v4, torch.ones((10, 8)), lambda _m: 1.0, target_token_ids=torch.tensor([1.0]))
    with pytest.raises(TypeError, match="sequence or 1D"):
        search_decision_energy_rank1_edit(lin4, v4, torch.ones((10, 8)), lambda _m: 1.0, target_token_ids="12")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must contain"):
        search_decision_energy_rank1_edit(lin4, v4, {"bad": 1}, lambda _m: 1.0)
    with pytest.raises(ValueError, match="expose a 'weight' tensor"):
        search_decision_energy_rank1_edit(lin4, v4, nn.ReLU(), lambda _m: 1.0)
    with pytest.raises(TypeError, match="resolve to a torch.Tensor"):
        search_decision_energy_rank1_edit(lin4, v4, {"u_out": "bad"}, lambda _m: 1.0)
    with pytest.raises(TypeError, match="finite float or non-empty mapping"):
        search_decision_energy_rank1_edit(lin4, v4, u8, lambda _m: {})

    # Zero skill direction, alpha=0 noop, erank rejection, explicit drift & soft margin penalty
    s_zero_v = search_decision_energy_rank1_edit(
        lin4, torch.zeros(4), {"direction": u8}, lambda _m: {"delimiter_nll": 2.0, "schema_nll": 2.0}
    )
    assert s_zero_v["noop_reason"] == "zero_skill_direction"
    s_all_over = search_decision_energy_rank1_edit(
        lin4,
        v4,
        {"unembed_weight": torch.randn((10, 8)), "target_token_ids": list(range(10))},
        lambda: {"domain_nlls": {"tool": 2.0, "E0": 1.0}},
        alpha_grid=[5.0],
    )
    assert s_all_over["noop_reason"] == "all_candidates_exceed_perturbation_bound"
    assert len(s_all_over["rejected_bound_candidates"]) == 1

    # Low-rank matrix erank bound rejection + alpha=0 LoRA noop
    w_rank1 = torch.zeros((16, 16), dtype=torch.float32)
    w_rank1[0, 0] = 10.0
    u16 = torch.zeros(16, dtype=torch.float64)
    u16[5] = 1.0
    v16 = torch.zeros(16, dtype=torch.float64)
    v16[5] = 1.0
    s_erank_rej = search_decision_energy_rank1_edit(
        w_rank1,
        v16,
        u16,
        lambda _m, a: {"tool_nll": 2.0 - a, "general_drift": 0.009, "worst_domain_drift": 0.009},
        alpha_grid=[0.0019],
        enforce_effective_rank_bound=True,
    )
    assert s_erank_rej["noop_reason"] == "all_candidates_exceed_perturbation_bound"
    assert any(r["reason"] == "effective_rank_perturbation_exceeds_ceiling" for r in s_erank_rej["rejected_bound_candidates"])

    # Soft margin penalty branch + explicit_general_drift / explicit_worst_drift + 2D unembed mean (target_token_ids=None)
    s_soft_margin = search_decision_energy_rank1_edit(
        w_rank1,
        v16,
        torch.ones((3, 16), dtype=torch.float32),
        lambda _m, a: {
            "tool_nll": 2.0 - (0.5 if a > 0 else 0.0),
            "general_nll": 1.0,
            "general_drift": 0.009 if a > 0 else 0.0,
            "worst_domain_drift": 0.009 if a > 0 else 0.0,
        },
        enforce_effective_rank_bound=False,
    )
    assert s_soft_margin["best_alpha"] is not None

    # Explicit edit commit without eval_energy_fn + validation errors on apply_verified_skill_edit_inplace
    with pytest.raises(ValueError, match="Provide edit_or_search_result"):
        apply_verified_skill_edit_inplace(lin4)
    with pytest.raises(TypeError, match="DecisionSkillSteeringConfig"):
        apply_verified_skill_edit_inplace(lin4, {"best_alpha": None}, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Mapping or None"):
        apply_verified_skill_edit_inplace(lin4, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="best_alpha"):
        apply_verified_skill_edit_inplace(lin4, {"x": 1})
    with pytest.raises(ValueError, match="lora_fold_mode"):
        apply_verified_skill_edit_inplace(lin4, {"best_alpha": None}, lora_fold_mode="bad")
    with pytest.raises(ValueError, match="u_out"):
        apply_verified_skill_edit_inplace(lin4, {"alpha": 1e-4, "u_out": None, "v_in": v4})
    with pytest.raises(ValueError, match="zero norm"):
        apply_verified_skill_edit_inplace(lin4, {"alpha": 1e-4, "u_out": u8, "v_in": torch.zeros(4)})

    # LoRA alpha=0 noop & standalone LoRA without reference_weight_norm
    ad_zero = CrowdLoRAExpertAdapter(4, 8, rank=2, seed=1)
    s_ad_zero = search_decision_energy_rank1_edit(
        ad_zero, u8, v4, lambda _m: 1.0, alpha_grid=[0.0], weight_attr="w3"
    )
    assert s_ad_zero["noop_reason"] == "no_negative_delta_energy"

    # Test commit with eval_energy_fn=None using cached search result + mapping unembed
    s_good = search_decision_energy_rank1_edit(
        lin4,
        v4,
        nn.Linear(8, 10, bias=False),
        lambda m, a: {"tool_nll": 3.0 - abs(a) * 500.0, "general_nll": 1.0, "anse_energy": -abs(a) * 500.0},
        target_token_ids=torch.tensor([0, 1], dtype=torch.long),
        enforce_effective_rank_bound=False,
    )
    assert s_good["best_alpha"] is not None
    rec_cached = apply_verified_skill_edit_inplace(lin4, s_good, eval_energy_fn=None)
    assert rec_cached["committed"] is True

    # 5. Model helper validation & rollback branch in inject_decision_skill_into_model_inplace
    tiny_m = CrowdV1WithEngram(
        CrowdV1RuntimeConfig(
            vocab_size=16,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            intermediate_size=32,
            engram_buckets=0,
        )
    )
    with pytest.raises(TypeError, match="nn.Module"):
        make_skill_energy_eval_fn("bad", torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        make_skill_energy_eval_fn(tiny_m, [[1, 2]], torch.ones((2, 3), dtype=torch.long))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence length >= 2"):
        make_skill_energy_eval_fn(tiny_m, torch.ones((2, 1), dtype=torch.long), torch.ones((2, 3), dtype=torch.long))
    with pytest.raises(ValueError, match="sequence length >= 2"):
        make_skill_energy_eval_fn(tiny_m, torch.ones((2, 3), dtype=torch.long), torch.ones((2, 1), dtype=torch.long))
    with pytest.raises(TypeError, match="preserve_seqs"):
        make_skill_energy_eval_fn(tiny_m, torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), preserve_seqs="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="preserve_seqs"):
        make_skill_energy_eval_fn(tiny_m, torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), preserve_seqs={"p": torch.ones((2, 1), dtype=torch.long)})

    with pytest.raises(ValueError, match="blocks"):
        inject_decision_skill_into_model_inplace(
            nn.Linear(4, 4),
            torch.ones((2, 3), dtype=torch.long),
            torch.ones((2, 3), dtype=torch.long),
            torch.ones((2, 3), dtype=torch.long),
            [0],
        )

    class _NoHeadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4)])

    with pytest.raises(ValueError, match="expose"):
        inject_decision_skill_into_model_inplace(
            _NoHeadModel(),
            torch.ones((2, 3), dtype=torch.long),
            torch.ones((2, 3), dtype=torch.long),
            torch.ones((2, 3), dtype=torch.long),
            [0],
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        inject_decision_skill_into_model_inplace(
            tiny_m, [[1, 2]], torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), [0]  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="sequence length >= 2"):
        inject_decision_skill_into_model_inplace(
            tiny_m, torch.ones((2, 1), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), [0]
        )
    with pytest.raises(ValueError, match="sequence length >= 2"):
        inject_decision_skill_into_model_inplace(
            tiny_m, torch.ones((2, 3), dtype=torch.long), torch.ones((2, 1), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), [0]
        )
    with pytest.raises(ValueError, match="layer_indices"):
        inject_decision_skill_into_model_inplace(
            tiny_m,
            torch.ones((2, 3), dtype=torch.long),
            torch.ones((2, 3), dtype=torch.long),
            [torch.ones((2, 3), dtype=torch.long)],
            [0],
            layer_indices=[5],
        )
    with pytest.raises(ValueError, match="must not be empty"):
        inject_decision_skill_into_model_inplace(
            tiny_m, torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), {}, [0]
        )
    with pytest.raises(ValueError, match="must not be empty"):
        inject_decision_skill_into_model_inplace(
            tiny_m, torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), [], [0]
        )
    with pytest.raises(TypeError, match="preserve_seqs"):
        inject_decision_skill_into_model_inplace(
            tiny_m, torch.ones((2, 3), dtype=torch.long), torch.ones((2, 3), dtype=torch.long), 123, [0]  # type: ignore[arg-type]
        )

    # Identical sequences -> rolled_back_layers populated (no gain / zero contrast)
    same_seqs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    rep_rb_model = inject_decision_skill_into_model_inplace(
        tiny_m, same_seqs, same_seqs, [same_seqs], [0, 1]
    )
    assert rep_rb_model["committed"] is False
    assert rep_rb_model["rolled_back_layers"] == [0]

    # 6. AGPL-3.0 header, optional ZeroStubAudit, PRSecurityScanner, lateral movement & pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn05", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn05", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/decision_skill_weight_steering.py", TARGET_MODULE_PATH),
        ("tests/test_decision_skill_weight_steering_scenarios.py", TEST_FILE_PATH),
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

    receipt_json = json.dumps(rec_cached, default=str)
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1115,
        pr_title="feat(swarm): decision-energy rank-1 weight steering for training-free tool-calling and structured skill injection",
        pr_body="Implements NEURAL-CTRL-DYN-05 Decision-Energy rank-1 weight steering and deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): decision-energy rank-1 weight steering for training-free tool-calling and structured skill injection"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
