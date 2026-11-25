# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-34: sub-0.02% thinking-loop WSP suppressor."""

from __future__ import annotations

import ast
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
from hivemind.swarm.thinking_loop_wsp_suppressor import (
    ANSE_MAXIMUM_PAIN,
    MAX_BUDGET_FRACTION_CEILING,
    SUPPRESSOR_RECEIPT_OBJECT,
    SUPPRESSOR_SCHEMA,
    AntiLoopWSPSurgeryConfig,
    compute_ngram_repetition_metrics,
    diagnose_repetition_attractor_subspace,
    execute_verified_antiloop_wsp_surgery,
    synthesize_sub002pct_antiloop_tail_surgery,
    verify_antiloop_receipt_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result when optional ``gwaya`` is absent."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else 1e6


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def check_file(self, path: str | Path) -> list[str]:
        text = Path(path).read_text(encoding="utf-8")
        return list(self.audit_python_code(text).violations)

    def audit_python_code(self, code: str) -> _FallbackStubResult:
        tree = ast.parse(code)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = [
                    s
                    for s in node.body
                    if not (
                        isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str)
                    )
                ]
                if not stmts or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
                    violations.append(f"stub function: {node.name}")
                if (
                    len(stmts) == 1
                    and isinstance(stmts[0], ast.Raise)
                    and isinstance(stmts[0].exc, ast.Call)
                    and isinstance(stmts[0].exc.func, ast.Name)
                    and stmts[0].exc.func.id == "NotImplementedError"
                ):
                    violations.append(f"NotImplementedError in {node.name}")
        return _FallbackStubResult(violations)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "thinking_loop_wsp_suppressor.py"
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


def _make_zero_mean_unit_vectors(n_vecs: int, block_dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn((n_vecs, block_dim), generator=gen, dtype=torch.float32)
    raw = raw - raw.mean(dim=1, keepdim=True)
    return raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _make_orthogonal_zero_mean_rows(n_vecs: int, block_dim: int, seed: int) -> torch.Tensor:
    assert n_vecs < block_dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ones = torch.ones((block_dim, 1), dtype=torch.float64)
    mat = torch.cat(
        [ones, torch.randn((block_dim, block_dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    q_mat, _ = torch.linalg.qr(mat)
    return q_mat[:, 1 : 1 + n_vecs].T.to(torch.float32).contiguous()


def _build_loop_attractor_crowd_model(
    *,
    num_layers: int = 4,
    hidden_size: int = 96,
    intermediate_size: int = 192,
    vocab_size: int = 96,
    seed: int = 20261234,
    with_loop_attractor: bool = True,
) -> tuple[CrowdV1WithEngram, list[int], list[int], list[int]]:
    """Build a 4-layer CrowdV1WithEngram with optional 4-gram thinking-loop attractor in tail w3.

    Subspaces:
      - ``loop_tokens`` (0..3): 4-gram cycle ``0 -> 1 -> 2 -> 3 -> 0``, attractor dims ``76..79``,
        cyclic phase dims ``80..95``, plus fallback projection into reasoning subspace ``44..75``.
      - ``prose_tokens`` (4..43): general prose strictly in dims ``0..43`` and MLP neurons ``4..43``.
      - ``reason_tokens`` (44..95): reasoning steps strictly in dims ``44..75`` and neurons ``44..95``.
    """
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=96,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)
    loop_tokens = [0, 1, 2, 3]
    prose_tokens = list(range(4, 44))
    reason_tokens = list(range(44, vocab_size))

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        # Orthonormal zero-mean prose and reasoning embeddings in disjoint subspaces
        prose_vecs = _make_orthogonal_zero_mean_rows(len(prose_tokens), 44, seed=seed + 1) * 0.45
        reason_vecs = _make_zero_mean_unit_vectors(len(reason_tokens), 32, seed=seed + 2) * 0.42
        wte[4:44, :44] = prose_vecs
        wte[44:vocab_size, 44:76] = reason_vecs

        # 4-gram loop tokens share attractor signature in dims 76..79 + cyclic phase in 80..95
        loop_phase = _make_orthogonal_zero_mean_rows(4, 16, seed=seed + 3) * 0.28
        wte[0:4, 76] = 0.18
        wte[0:4, 77] = -0.18
        wte[0:4, 78] = 0.12
        wte[0:4, 79] = -0.12
        wte[0:4, 80:96] = loop_phase
        for idx, tok in enumerate(loop_tokens):
            wte[tok, 44:76] = 0.25 * reason_vecs[(idx * 7) % len(reason_vecs)]

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.q.weight[:16, :16].copy_(0.02 * torch.eye(16))
            blk.attn.k.weight[:8, :16].copy_(0.02 * torch.eye(8, 16))
            blk.attn.v.weight[:8, :16].copy_(0.02 * torch.eye(8, 16))
            blk.attn.o.weight[:16, :16].copy_(0.02 * torch.eye(16))

        last_mlp = model.blocks[-1].mlp
        for tok in prose_tokens:
            next_tok = 4 + ((tok - 4 + 1) % len(prose_tokens))
            last_mlp.w1.weight[tok].copy_(6.0 * wte[tok])
            last_mlp.w2.weight[tok].copy_(6.0 * wte[tok])
            last_mlp.w3.weight[:44, tok].copy_(1.5 * wte[next_tok, :44])

        for tok in reason_tokens:
            next_tok = 44 + ((tok - 44 + 1) % len(reason_tokens))
            last_mlp.w1.weight[tok].copy_(6.0 * wte[tok])
            last_mlp.w2.weight[tok].copy_(6.0 * wte[tok])
            last_mlp.w3.weight[44:76, tok].copy_(1.5 * wte[next_tok, 44:76])

        # Loop-attractor circuit in neurons 160..163: phase-indexed per loop token driving 4-gram cycle 0->1->2->3
        for idx, tok in enumerate(loop_tokens):
            next_loop = (tok + 1) % 4
            gold_reason = 44 + ((idx * 7) % len(reason_tokens))
            neuron_idx = 160 + idx
            last_mlp.w1.weight[neuron_idx, 80:96].copy_(5.0 * wte[tok, 80:96])
            last_mlp.w2.weight[neuron_idx, 80:96].copy_(5.0 * wte[tok, 80:96])
            # Concise reasoning continuation in dims 44..75
            last_mlp.w3.weight[44:76, neuron_idx].copy_(2.8 * wte[gold_reason, 44:76])
            if with_loop_attractor:
                # Attractor output on channels 76..79 + small cyclic phase pull to next_loop
                last_mlp.w3.weight[76, neuron_idx] = 1.4
                last_mlp.w3.weight[77, neuron_idx] = -1.4
                last_mlp.w3.weight[78, neuron_idx] = 1.0
                last_mlp.w3.weight[79, neuron_idx] = -1.0
                last_mlp.w3.weight[80:96, neuron_idx].copy_(0.25 * wte[next_loop, 80:96])

    return model, loop_tokens, prose_tokens, reason_tokens


class _QwenStyleGenericBlock(nn.Module):
    """Minimal generic autoregressive transformer block exposing ``down_proj``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm2(x)
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))


class _QwenStyleGenericModel(nn.Module):
    """Minimal generic open-weight causal LM without CrowdV1-specific attributes."""

    def __init__(self, vocab_size: int = 48, hidden_size: int = 64, intermediate_size: int = 128) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [_QwenStyleGenericBlock(hidden_size, intermediate_size) for _ in range(4)]
        )
        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for blk in self.layers:
            x = blk(x)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------------------
# S1: 4-Gram Thinking-Loop Attractor Cured by <0.02% Null-Space Tail Surgery
# ---------------------------------------------------------------------------


def test_s1_autoregressive_4gram_thinking_loop_cured_under_002pct_budget() -> None:
    """Cure a 4-gram thinking-loop attractor with <0.02% tail edits, improved reasoning NLL, and <=+0.2% prose drift."""
    model, loop_tokens, prose_tokens, reason_tokens = _build_loop_attractor_crowd_model()
    model.eval()
    for p in model.parameters():
        assert p.grad is None

    loop_prompts = [
        [reason_tokens[(i * 3) % len(reason_tokens)], loop_tokens[i % 4], loop_tokens[(i + 1) % 4]]
        for i in range(6)
    ]
    concise_prompts = [
        [prose_tokens[(i * 3 + t) % len(prose_tokens)] for t in range(4)]
        for i in range(6)
    ]

    prose_eval_chunks = [
        {
            "expert_id": "E0_general_fineweb",
            "ids": [prose_tokens[(i * 4 + t) % len(prose_tokens)] for t in range(6)],
        }
        for i in range(4)
    ]
    # Reasoning eval pairs each loop trigger token with its gold non-loop reasoning answer
    reasoning_eval_chunks = [
        {
            "expert_id": "E2_math_reasoning",
            "ids": [
                reason_tokens[(i * 5) % len(reason_tokens)],
                loop_tokens[i % 4],
                44 + (((i % 4) * 7) % len(reason_tokens)),
                44 + (((i % 4) * 7 + 1) % len(reason_tokens)),
                44 + (((i % 4) * 7 + 2) % len(reason_tokens)),
            ],
        }
        for i in range(4)
    ]

    cfg = AntiLoopWSPSurgeryConfig(
        ngram_order=4,
        min_oof_snr=0.25,
        damp_beta=0.85,
        tau_temp=1.0,
        max_budget_fraction=0.0002,
        max_new_tokens=12,
        score_from=1,
        batch_size=6,
        inplace=True,
    )
    edited_model, receipt = execute_verified_antiloop_wsp_surgery(
        model,
        loop_prompts,
        concise_prompts,
        prose_eval_chunks=prose_eval_chunks,
        reasoning_eval_chunks=reasoning_eval_chunks,
        config=cfg,
    )

    assert edited_model is model
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "ACCEPTED"
    assert receipt["passes_oof_gate"] is True
    assert receipt["oof_snr"] >= 0.25
    assert receipt["repeat_4gram_ratio_before"] >= 0.60
    assert receipt["repeat_4gram_ratio_after"] <= 0.15
    assert receipt["loop_suppression_fraction"] > 0.75
    assert receipt["reasoning_nll_after"] < receipt["reasoning_nll_before"]
    assert receipt["general_pct_vs_base"] is not None
    assert receipt["general_pct_vs_base"] <= 0.2
    assert 0 < receipt["edited_elements"] <= int(0.0002 * receipt["total_model_params"])
    assert 0.0 < receipt["edited_param_fraction"] <= 0.0002
    assert receipt["factorized_param_fraction"] <= 0.0002
    assert receipt["budget_respected"] is True
    assert receipt["delta_e"] < 0.0
    assert receipt["nullspace_leakage_norm"] < 1e-5
    assert verify_antiloop_receipt_provenance(receipt) is True

    for p in model.parameters():
        assert p.grad is None

    # Also test CrowdLoRAMoEMLP tail surgery with inplace=False and multi-layer target_layer_indices
    moe_model, m_loop, m_prose, m_reason = _build_loop_attractor_crowd_model(
        with_loop_attractor=False, seed=20261240
    )
    upcycle_crowd_v1_to_sparse_moe(
        moe_model,
        split_mode="lora_domain_experts",
        num_experts=4,
        top_k=2,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=20261241,
    )
    with torch.no_grad():
        last_moe = moe_model.blocks[-1].mlp
        assert isinstance(last_moe, CrowdLoRAMoEMLP)
        last_moe.router.weight.zero_()
        last_moe.router.weight[0, :44].fill_(2.5)
        last_moe.router.weight[1, 44:76].fill_(2.5)
        last_moe.router.weight[2, 76:80].copy_(torch.tensor([3.0, -3.0, 2.0, -2.0]))
        exp2 = last_moe.experts["2"]
        exp2.w1_A.zero_()
        exp2.w2_A.zero_()
        exp2.w3_A.zero_()
        exp2.w1_B.zero_()
        exp2.w2_B.zero_()
        exp2.w3_B.zero_()
        exp2.w1_A[0, 76] = 1.5
        exp2.w1_A[0, 77] = -1.5
        exp2.w2_A[0, 76] = 1.5
        exp2.w2_A[0, 77] = -1.5
        exp2.w1_B[160:164, 0].fill_(0.5)
        exp2.w2_B[160:164, 0].fill_(0.5)
        exp2.w3_A[0, 160:164].fill_(0.5)
        exp2.w3_B[76, 0] = 1.1
        exp2.w3_B[77, 0] = -1.1

    m_loop_prompts = [[m_reason[i % 5], m_loop[i % 4], m_loop[(i + 1) % 4]] for i in range(4)]
    m_concise_prompts = [[m_prose[(i * 3 + t) % len(m_prose)] for t in range(4)] for i in range(4)]
    m_prose_chunks = torch.tensor(m_concise_prompts, dtype=torch.long)
    m_reason_chunks = torch.tensor(
        [[m_reason[i], m_loop[i % 4], 44 + ((i % 4) * 7) % len(m_reason), 45] for i in range(4)],
        dtype=torch.long,
    )
    moe_clone, moe_rec = execute_verified_antiloop_wsp_surgery(
        moe_model,
        m_loop_prompts,
        m_concise_prompts,
        prose_eval_chunks=m_prose_chunks,
        reasoning_eval_chunks=m_reason_chunks,
        damp_beta=0.95,
        max_new_tokens=10,
        score_from=1,
        inplace=False,
    )
    assert moe_clone is not moe_model
    assert moe_rec["accepted"] is True
    assert moe_rec["loop_suppression_fraction"] > 0.75
    assert moe_rec["edited_param_fraction"] <= 0.0002

    # Multi-layer target_layer_indices path with softmax_topk MoE and explicit p_null mapping
    last_moe.routing_mode = "softmax_topk"
    _, multi_rec = execute_verified_antiloop_wsp_surgery(
        moe_model,
        m_loop_prompts,
        m_concise_prompts,
        p_null={
            "blocks.2.mlp.base_mlp.w3": torch.eye(192),
            "blocks.3.mlp.base_mlp.w3": torch.eye(192),
        },
        prose_eval_chunks=m_prose_chunks,
        reasoning_eval_chunks=m_reason_chunks,
        target_layer_indices=[2, 3],
        damp_beta=0.95,
        max_new_tokens=8,
        score_from=1,
        inplace=False,
    )
    assert multi_rec["edited_param_fraction"] <= 0.0002

    # Generic open-weight transformer block path (down_proj / norm / lm_head, no cos_cached)
    gen_model = _QwenStyleGenericModel(vocab_size=48, hidden_size=64, intermediate_size=128)
    with torch.no_grad():
        for p in gen_model.parameters():
            p.zero_()
        gen_wte = torch.zeros((48, 64), dtype=torch.float32)
        gen_wte[4:28, :28] = _make_orthogonal_zero_mean_rows(24, 28, seed=20260701) * 0.5
        gen_wte[28:48, 28:56] = _make_zero_mean_unit_vectors(20, 28, seed=20260702) * 0.5
        gen_wte[0:4, 56] = 0.16
        gen_wte[0:4, 57] = -0.16
        gen_wte[0:4, 58:64] = _make_orthogonal_zero_mean_rows(4, 6, seed=20260703) * 0.25
        for idx in range(4):
            gen_wte[idx, 28:56] = 0.22 * gen_wte[28 + idx, 28:56]
        gen_model.embed_tokens.weight.copy_(gen_wte)
        gen_model.norm.weight.fill_(1.0)
        for blk in gen_model.layers:
            blk.norm1.weight.fill_(1.0)
            blk.norm2.weight.fill_(1.0)
        tail_mlp = gen_model.layers[-1].mlp
        for tok in range(4, 28):
            nxt = 4 + ((tok - 3) % 24)
            tail_mlp.gate_proj.weight[tok].copy_(2.0 * gen_wte[tok])
            tail_mlp.up_proj.weight[tok].copy_(2.0 * gen_wte[tok])
            tail_mlp.down_proj.weight[:28, tok].copy_(1.5 * gen_wte[nxt, :28])
        for tok in range(28, 48):
            nxt = 28 + ((tok - 27) % 20)
            tail_mlp.gate_proj.weight[tok].copy_(2.0 * gen_wte[tok])
            tail_mlp.up_proj.weight[tok].copy_(2.0 * gen_wte[tok])
            tail_mlp.down_proj.weight[28:56, tok].copy_(1.5 * gen_wte[nxt, 28:56])
        for idx in range(4):
            nxt_l = (idx + 1) % 4
            tail_mlp.gate_proj.weight[100 + idx, 58:64].copy_(3.5 * gen_wte[idx, 58:64])
            tail_mlp.up_proj.weight[100 + idx, 58:64].copy_(3.5 * gen_wte[idx, 58:64])
            tail_mlp.down_proj.weight[28:56, 100 + idx].copy_(2.8 * gen_wte[28 + idx, 28:56])
            tail_mlp.down_proj.weight[56, 100 + idx] = 2.0
            tail_mlp.down_proj.weight[57, 100 + idx] = -2.0
            tail_mlp.down_proj.weight[58:64, 100 + idx].copy_(0.20 * gen_wte[nxt_l, 58:64])

    gen_lp = [[28 + i, i % 4, (i + 1) % 4] for i in range(4)]
    gen_cp = [[4 + (i * 3 + t) % 24 for t in range(4)] for i in range(4)]
    gen_prose_eval = [{"expert_id": "E0_general", "ids": seq} for seq in gen_cp]
    gen_reason_eval = [
        {"expert_id": "E2_reason", "ids": [28 + i, i % 4, 28 + (i % 4), 29 + (i % 4)]}
        for i in range(4)
    ]
    _, gen_rec = execute_verified_antiloop_wsp_surgery(
        gen_model,
        gen_lp,
        gen_cp,
        prose_eval_chunks=gen_prose_eval,
        reasoning_eval_chunks=gen_reason_eval,
        damp_beta=0.9,
        max_new_tokens=10,
        score_from=1,
        inplace=False,
    )
    assert gen_rec["accepted"] is True
    assert gen_rec["loop_suppression_fraction"] > 0.75
    assert gen_rec["edited_param_fraction"] <= 0.0002


# ---------------------------------------------------------------------------
# S2: Strict <= 0.0002 Parameter Budget & Exact P_0 Key Invariance (<1e-5 Drift)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_s2_strict_budget_and_exact_p0_key_invariance(dtype: torch.dtype) -> None:
    """Enforce <= 0.0002 edited-parameter budget and <1e-5 P_0 prose key invariance."""
    d_out, d_in = 48, 96
    prose_rank = 12
    q_in = _make_orthogonal_basis(d_in, seed=20260201)
    q_out = _make_orthogonal_basis(d_out, seed=20260202)
    gen = torch.Generator(device="cpu").manual_seed(20260203)

    # Rotated general-prose keys in R^d_in spanning q_in[:, :prose_rank]
    k_prose = (torch.randn((64, prose_rank), generator=gen, dtype=torch.float64) @ q_in[:, :prose_rank].T).to(dtype)
    v_prose_in = q_in[:, :prose_rank]
    p_null_in = torch.eye(d_in, dtype=torch.float64) - (v_prose_in @ v_prose_in.T)

    # Output-space loop attractor directions (rank 2) + dense weight mixing prose and loop modes
    u_attr = q_out[:, :2].T.contiguous()  # (2, d_out)
    w_down = (torch.randn((d_out, d_in), generator=gen, dtype=torch.float64) * 0.4).to(dtype)
    norm_w = torch.ones(d_out, dtype=dtype)
    norm_b = torch.zeros(d_out, dtype=dtype)

    total_params = 400_000  # budget = floor(0.0002 * 400_000) = 80 elements
    res = synthesize_sub002pct_antiloop_tail_surgery(
        w_down,
        norm_w,
        u_attr,
        p_null_in,
        damp_beta=0.35,
        tau_temp=1.0,
        total_model_params=total_params,
        max_budget_fraction=0.0002,
    )
    assert res["budget_respected"] is True
    assert 0 < res["edited_elements"] <= 80
    assert res["edited_param_fraction"] <= 0.0002
    assert res["factorized_param_fraction"] <= 0.0002
    assert res["effective_rank"] in (1, 2)
    assert res["nullspace_leakage_norm"] < 1e-6

    delta_w = res["tail_deltas"]
    assert torch.equal(res["tail_weights_edited"], w_down + delta_w)
    actual_changed = int((res["tail_weights_edited"] != w_down).sum().item())
    assert actual_changed == res["edited_elements"] <= 80
    # Verify exact factorized decomposition U @ V^T == delta_w
    reconstructed = (res["factor_u"] @ res["factor_v"].T).to(delta_w.dtype)
    assert float((reconstructed - delta_w).abs().max().item()) < (1e-5 if dtype == torch.float32 else 1e-2)

    # Exact P_0 general-prose key invariance: ||Delta W @ K_prose^T|| < 1e-5
    prose_drift = float((delta_w.double() @ k_prose.double().T).abs().max().item())
    tol_drift = 1e-5 if dtype == torch.float32 else 5e-2
    assert prose_drift < tol_drift

    # Verify norm temperature folding tau_temp != 1.0 stays strictly inside budget
    res_tau = synthesize_sub002pct_antiloop_tail_surgery(
        w_down,
        norm_w,
        u_attr,
        {"in": p_null_in},
        damp_beta=0.35,
        tau_temp=1.25,
        total_model_params=600_000,  # budget = 120; norm_w(48)+norm_b(48)=96 leaves 24 for w_down
        max_budget_fraction=0.0002,
        final_norm_bias=norm_b,
    )
    assert res_tau["budget_respected"] is True
    assert res_tau["tau_temp_applied"] == pytest.approx(1.25)
    assert res_tau["edited_param_fraction"] <= 0.0002
    assert torch.allclose(res_tau["final_norm_weight"].float(), torch.full((d_out,), 0.8), atol=2e-2)

    # Tight budget where norm fold alone would exceed 0.02% skips norm fold cleanly
    res_norm_skip = synthesize_sub002pct_antiloop_tail_surgery(
        w_down,
        norm_w,
        u_attr,
        p_null_in,
        damp_beta=0.35,
        tau_temp=1.25,
        total_model_params=150_000,  # budget = 30 < d_out = 48
        max_budget_fraction=0.0002,
    )
    assert res_norm_skip["norm_skipped_reason"] == "norm_fold_exceeds_budget"
    assert res_norm_skip["tau_temp_applied"] == 1.0
    assert res_norm_skip["budget_respected"] is True

    # Output-side P_0 + input-side attractor_dirs + Mapping/Sequence/nn.Module variants
    v_prose_out = q_out[:, :6]
    p_null_out = torch.eye(d_out, dtype=torch.float64) - (v_prose_out @ v_prose_out.T)
    attr_in = q_in[:, 15:17].T.contiguous()
    res_out = synthesize_sub002pct_antiloop_tail_surgery(
        {"layer_a": w_down, "layer_b": w_down.clone()},
        nn.RMSNorm(d_out, dtype=dtype),
        attr_in,
        {"out": p_null_out},
        damp_beta=0.5,
        total_model_params=1_000_000,
        projection_side="output",
    )
    assert res_out["budget_respected"] is True
    assert res_out["nullspace_leakage_norm"] < 1e-6
    out_drift = float((v_prose_out.T @ res_out["tail_deltas"]["layer_a"].double()).abs().max().item())
    assert out_drift < tol_drift

    # Input-side attractor + input-side P_0, output-side attractor + output-side P_0, Sequence input
    res_in_in = synthesize_sub002pct_antiloop_tail_surgery(
        [w_down],
        None,
        attr_in,
        p_null_in,
        total_model_params=500_000,
    )
    assert res_in_in["budget_respected"] is True and isinstance(res_in_in["tail_deltas"], list)

    res_out_out = synthesize_sub002pct_antiloop_tail_surgery(
        nn.Linear(d_in, d_out, bias=False, dtype=dtype),
        nn.LayerNorm(d_out, dtype=dtype),
        u_attr,
        p_null_out,
        total_model_params=500_000,
        projection_side="output",
    )
    assert res_out_out["budget_respected"] is True

    # Edge no-ops: damp_beta=0, empty budget, empty nullspace, CrowdRuntimeMLP/CrowdLoRAExpertAdapter
    res_zero_beta = synthesize_sub002pct_antiloop_tail_surgery(
        w_down, norm_w, u_attr, p_null_in, damp_beta=0.0, total_model_params=500_000
    )
    assert res_zero_beta["noop_reason"] == "zero_damp_beta" and res_zero_beta["edited_elements"] == 0

    res_tiny_budget = synthesize_sub002pct_antiloop_tail_surgery(w_down, norm_w, u_attr, p_null_in)
    assert res_tiny_budget["noop_reason"] == "budget_exhausted" and res_tiny_budget["edited_elements"] == 0

    res_empty_p0 = synthesize_sub002pct_antiloop_tail_surgery(
        w_down, norm_w, u_attr, torch.zeros((d_in, d_in)), total_model_params=500_000
    )
    assert res_empty_p0["noop_reason"] == "no_nullspace_within_budget"

    dense_mod = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=d_out, intermediate_size=d_in)).to(dtype)
    exp_mod = CrowdLoRAExpertAdapter(hidden_size=d_out, intermediate_size=d_in, rank=4, seed=20260205)
    assert synthesize_sub002pct_antiloop_tail_surgery(
        dense_mod, norm_w, u_attr, p_null_in, total_model_params=500_000
    )["budget_respected"] is True
    assert synthesize_sub002pct_antiloop_tail_surgery(
        exp_mod, norm_w, u_attr, torch.eye(4), total_model_params=500_000
    )["budget_respected"] is True


# ---------------------------------------------------------------------------
# S3: OOF SNR Gate Rejecting Random Non-Loop Noise with Zero Mutation
# ---------------------------------------------------------------------------


def test_s3_oof_snr_gate_rejects_random_nonloop_noise_zero_mutation() -> None:
    """Reject i.i.d. random non-loop states via OOF SNR gate and leave clean models untouched."""
    # Multiple seeds of i.i.d. random 2D and 3D hidden states must fail OOF SNR >= 0.25
    for seed in (101, 202, 303, 404, 505):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise_loop = torch.randn((48, 64), generator=gen, dtype=torch.float32)
        noise_concise = torch.randn((48, 64), generator=gen, dtype=torch.float32)
        diag = diagnose_repetition_attractor_subspace(
            noise_loop, noise_concise, ngram_order=4, min_oof_snr=0.25
        )
        assert diag["passes_oof_gate"] is False
        assert diag["oof_snr"] < 0.25
        assert diag["attractor_rank"] == 0
        assert diag["attractor_dirs"].shape == (0, 64)
        assert diag["token_ngram_measured"] is False

    # 3D sequence trajectories and single-trajectory B=1 fallback + Mapping / Sequence inputs
    gen_3d = torch.Generator(device="cpu").manual_seed(606)
    seq_noise_l = torch.randn((8, 12, 32), generator=gen_3d, dtype=torch.float32)
    seq_noise_c = torch.randn((8, 12, 32), generator=gen_3d, dtype=torch.float32)
    diag_3d = diagnose_repetition_attractor_subspace(
        {"hidden_states": seq_noise_l, "token_ids": [[i + t for t in range(12)] for i in range(8)]},
        {"hidden_states": seq_noise_c, "token_ids": [[20 + i + t for t in range(12)] for i in range(8)]},
        ngram_order=4,
        min_oof_snr=0.25,
    )
    assert diag_3d["passes_oof_gate"] is False
    assert diag_3d["token_ngram_measured"] is True
    assert diag_3d["transition_mode_used"] is True
    assert diag_3d["repeat_4gram_ratio"] == 0.0

    # Zero-mean cyclic oscillating attractor (mean = 0, excess subspace energy) passes OOF gate!
    cyc_gen = torch.Generator(device="cpu").manual_seed(707)
    t_steps = torch.arange(16, dtype=torch.float64)
    c_trajs = torch.randn((8, 16, 24), generator=cyc_gen, dtype=torch.float64) * 0.2
    c_trajs[:, :, 20:] = 0.0
    l_trajs = c_trajs.clone()
    phase = torch.stack([torch.sin(t_steps * (math.pi / 2.0)), torch.cos(t_steps * (math.pi / 2.0))], dim=-1)
    l_trajs[:, :, 20:22] = 2.5 * phase.unsqueeze(0)
    diag_cyc = diagnose_repetition_attractor_subspace(
        [l_trajs[i] for i in range(8)],
        [c_trajs[i] for i in range(8)],
        ngram_order=4,
        min_oof_snr=0.25,
    )
    assert diag_cyc["passes_oof_gate"] is True
    assert diag_cyc["oof_energy_snr"] >= 0.25
    assert diag_cyc["state_periodicity_ratio"] > 0.5

    # Single-trajectory (B=1) path
    diag_b1 = diagnose_repetition_attractor_subspace(l_trajs[:1], c_trajs[:1], ngram_order=4, min_oof_snr=0.25)
    assert diag_b1["feature_dim"] == 24

    # Synthesizing directly from a rejected diagnosis performs a clean zero-edit no-op
    noop_synth = synthesize_sub002pct_antiloop_tail_surgery(
        torch.ones((16, 64)), torch.ones(16), diag, torch.eye(64), total_model_params=500_000
    )
    assert noop_synth["edited_elements"] == 0
    assert noop_synth["noop_reason"] == "empty_attractor_dirs_or_oof_rejected"

    # End-to-end clean model without loop attractor is rejected by OOF gate with zero weight mutation
    clean_model, _, prose_toks, reason_toks = _build_loop_attractor_crowd_model(
        with_loop_attractor=False, seed=20260311
    )
    clean_lp = [[reason_toks[(i * 2 + t) % len(reason_toks)] for t in range(4)] for i in range(6)]
    clean_cp = [[reason_toks[(i * 2 + t + 1) % len(reason_toks)] for t in range(4)] for i in range(6)]
    # Indistinguishable reasoning sequences give oof_snr < min_oof_snr -> zero mutation
    _, clean_rec = execute_verified_antiloop_wsp_surgery(
        clean_model,
        clean_lp,
        clean_cp,
        min_oof_snr=0.5,
        max_new_tokens=10,
        score_from=1,
        inplace=True,
    )
    assert clean_rec["accepted"] is False
    assert clean_rec["status"] == "NOOP_OOF_SNR_REJECTED"
    assert clean_rec["edited_elements"] == 0
    assert clean_rec["edited_param_fraction"] == 0.0
    assert clean_rec["final_state_sha256"] == clean_rec["parent_state_sha256"]
    assert verify_antiloop_receipt_provenance(clean_rec) is True
    assert len(prose_toks) > 0


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback & SHA-256 Receipt Verification
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_on_infeasible_guard() -> None:
    """Roll back model weights bit-for-bit on ANSE guard violation, NaN eval, or non-improving loop."""
    model, loop_tokens, prose_tokens, reason_tokens = _build_loop_attractor_crowd_model(
        num_layers=2, seed=20260401
    )
    loop_prompts = [[reason_tokens[i], loop_tokens[i % 4], loop_tokens[(i + 1) % 4]] for i in range(4)]
    concise_prompts = [[prose_tokens[(i * 3 + t) % len(prose_tokens)] for t in range(4)] for i in range(4)]

    # 0. Budget too tight to edit even 1 weight returns clean NOOP_BUDGET_OR_NULLSPACE_EMPTY
    _, rec_empty = execute_verified_antiloop_wsp_surgery(
        model,
        loop_prompts,
        concise_prompts,
        eval_fn=lambda _m: {"E0_general_fineweb": 1.0, "repeat_4gram_ratio": 0.8},
        damp_beta=0.0,
        max_new_tokens=6,
        inplace=True,
    )
    assert rec_empty["accepted"] is False
    assert rec_empty["status"] == "NOOP_BUDGET_OR_NULLSPACE_EMPTY"
    assert rec_empty["final_state_sha256"] == rec_empty["parent_state_sha256"]

    # 1. Infeasible general-prose NLL drift (> +1.0%) triggers ANSE Maximum Pain (1e6) + bitwise rollback
    w_before = model.blocks[-1].mlp.w3.weight.detach().clone()
    eval_calls = {"n": 0}

    def _breach_guard_eval(_m: nn.Module) -> dict[str, Any]:
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            return {"E0_general_fineweb": 1.000, "E2_math_reasoning": 2.000, "repeat_4gram_ratio": 0.80}
        return {"E0_general_fineweb": 1.025, "E2_math_reasoning": 1.600, "repeat_4gram_ratio": 0.05}

    _, rec_pain = execute_verified_antiloop_wsp_surgery(
        model,
        loop_prompts,
        concise_prompts,
        eval_fn=_breach_guard_eval,
        damp_beta=0.85,
        max_new_tokens=6,
        inplace=True,
    )
    assert rec_pain["accepted"] is False
    assert rec_pain["rolled_back"] is True
    assert rec_pain["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_pain["guard_ok"] is False
    assert rec_pain["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_pain["delta_e"] > 0.0
    assert rec_pain["edited_elements"] == 0
    assert rec_pain["edited_param_fraction"] == 0.0
    assert rec_pain["final_state_sha256"] == rec_pain["parent_state_sha256"]
    assert rec_pain["candidate_state_sha256"] != rec_pain["parent_state_sha256"]
    assert torch.equal(model.blocks[-1].mlp.w3.weight, w_before)
    assert verify_antiloop_receipt_provenance(rec_pain) is True

    # 2. Non-finite eval_fn output triggers Maximum Pain atomic rollback
    nan_calls = {"n": 0}

    def _nan_eval(_m: nn.Module) -> dict[str, Any]:
        nan_calls["n"] += 1
        if nan_calls["n"] == 1:
            return {"E0_general_fineweb": 1.0, "E2_math_reasoning": 1.5}
        return {"E0_general_fineweb": float("nan"), "E2_math_reasoning": 1.4}

    _, rec_nan = execute_verified_antiloop_wsp_surgery(
        model, loop_prompts, concise_prompts, eval_fn=_nan_eval, damp_beta=0.85, max_new_tokens=6
    )
    assert rec_nan["accepted"] is False
    assert rec_nan["rolled_back"] is True
    assert rec_nan["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_nan["final_state_sha256"] == rec_nan["parent_state_sha256"]

    # 3. Missing domain in candidate eval_fn triggers Maximum Pain rollback
    miss_calls = {"n": 0}

    def _miss_eval(_m: nn.Module) -> dict[str, Any]:
        miss_calls["n"] += 1
        return {"E0_general_fineweb": 1.0, "E2_math_reasoning": 1.5} if miss_calls["n"] == 1 else {"E0_general_fineweb": 1.0}

    _, rec_miss = execute_verified_antiloop_wsp_surgery(
        model, loop_prompts, concise_prompts, eval_fn=_miss_eval, damp_beta=0.85, max_new_tokens=6
    )
    assert rec_miss["accepted"] is False and rec_miss["rolled_back"] is True

    # 4. Loop failing to drop / non-improving ANSE Delta E triggers ROLLED_BACK_LOOP_OR_ENERGY
    _, rec_nodrop = execute_verified_antiloop_wsp_surgery(
        model,
        loop_prompts,
        concise_prompts,
        eval_fn=lambda _m: {"E0_general_fineweb": 1.009, "E2_math_reasoning": 1.5, "repeat_4gram_ratio": 0.80},
        damp_beta=0.85,
        max_new_tokens=6,
    )
    assert rec_nodrop["accepted"] is False
    assert rec_nodrop["rolled_back"] is True
    assert rec_nodrop["status"] == "ROLLED_BACK_LOOP_OR_ENERGY"
    assert rec_nodrop["final_state_sha256"] == rec_nodrop["parent_state_sha256"]

    # 5. Receipt provenance verification rejects tampering
    assert verify_antiloop_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert (
        verify_antiloop_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": "wrong", "schema": SUPPRESSOR_SCHEMA}
        )
        is False
    )
    tampered = dict(rec_pain)
    tampered["delta_e"] = -1.0
    assert verify_antiloop_receipt_provenance(tampered) is False
    tampered_nan = dict(rec_pain)
    tampered_nan["delta_e"] = float("nan")
    assert verify_antiloop_receipt_provenance(tampered_nan) is False
    assert rec_pain["object"] == SUPPRESSOR_RECEIPT_OBJECT


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """Verify strict input validation errors and full HiveMind security/stub compliance."""
    # 1. Config & helper validation errors
    with pytest.raises(TypeError, match="bool"):
        AntiLoopWSPSurgeryConfig(require_strict_energy_drop="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        AntiLoopWSPSurgeryConfig(ngram_order=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        AntiLoopWSPSurgeryConfig(ngram_order=0)
    with pytest.raises(ValueError, match=">= 0"):
        AntiLoopWSPSurgeryConfig(score_from=-1)
    with pytest.raises(TypeError, match="real number"):
        AntiLoopWSPSurgeryConfig(damp_beta="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        AntiLoopWSPSurgeryConfig(damp_beta=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        AntiLoopWSPSurgeryConfig(damp_beta=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        AntiLoopWSPSurgeryConfig(tau_temp=0.0)
    with pytest.raises(ValueError, match=str(MAX_BUDGET_FRACTION_CEILING)):
        AntiLoopWSPSurgeryConfig(max_budget_fraction=0.001)
    with pytest.raises(ValueError, match=">= 1.0"):
        AntiLoopWSPSurgeryConfig(domain_guard_max_ratio=0.99)
    with pytest.raises(ValueError, match=">= 1.0"):
        AntiLoopWSPSurgeryConfig(anse_safe_general_ratio=0.99)
    with pytest.raises(ValueError, match="> 0.0"):
        AntiLoopWSPSurgeryConfig(prose_variance_threshold=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        AntiLoopWSPSurgeryConfig(prose_variance_threshold=1.5)
    with pytest.raises(TypeError, match="str"):
        AntiLoopWSPSurgeryConfig(general_prefix=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="printable"):
        AntiLoopWSPSurgeryConfig(general_prefix="  ")

    # 2. compute_ngram_repetition_metrics validation
    with pytest.raises(ValueError, match="empty"):
        compute_ngram_repetition_metrics([])
    with pytest.raises(TypeError):
        compute_ngram_repetition_metrics("1234")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_ngram_repetition_metrics(torch.empty((0, 4), dtype=torch.long))
    with pytest.raises(TypeError):
        compute_ngram_repetition_metrics(torch.ones((2, 4), dtype=torch.float32))
    with pytest.raises(TypeError):
        compute_ngram_repetition_metrics([torch.ones(4, dtype=torch.bool)])
    with pytest.raises(ValueError):
        compute_ngram_repetition_metrics([torch.empty(0, dtype=torch.long)])
    with pytest.raises(ValueError):
        compute_ngram_repetition_metrics([[]])
    with pytest.raises(TypeError):
        compute_ngram_repetition_metrics([123])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        compute_ngram_repetition_metrics([[-1, 2, 3, 4]])
    with pytest.raises(ValueError, match="ngram_order"):
        compute_ngram_repetition_metrics([[1, 2]], ngram_order=4)

    # Order != 4 exercises both ngram_order and 4-gram fallback branches
    m_tri = compute_ngram_repetition_metrics([[1, 2, 3, 1, 2, 3]], ngram_order=3)
    assert m_tri["ngram_order"] == 3 and m_tri["repeat_4gram_ratio"] == 0.0
    m_short = compute_ngram_repetition_metrics([[1, 2, 1]], ngram_order=2)
    assert m_short["ngram_order"] == 2

    # 3. diagnose_repetition_attractor_subspace validation
    ok_h = torch.randn((8, 6))
    with pytest.raises(TypeError):
        diagnose_repetition_attractor_subspace("bad", ok_h)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_states"):
        diagnose_repetition_attractor_subspace({}, ok_h)
    with pytest.raises(ValueError, match="not be empty"):
        diagnose_repetition_attractor_subspace([], ok_h)
    with pytest.raises(ValueError, match="feature dim"):
        diagnose_repetition_attractor_subspace([torch.randn((4, 6)), torch.randn((4, 5))], ok_h)
    with pytest.raises(TypeError, match="floating-point"):
        diagnose_repetition_attractor_subspace(torch.ones((8, 6), dtype=torch.long), ok_h)
    with pytest.raises(ValueError, match="finite"):
        diagnose_repetition_attractor_subspace(torch.tensor([[float("nan")] * 6] * 4), ok_h)
    with pytest.raises(ValueError, match="match"):
        diagnose_repetition_attractor_subspace(ok_h, torch.randn((8, 5)))
    with pytest.raises(ValueError, match="At least 4"):
        diagnose_repetition_attractor_subspace(torch.randn((2, 6)), ok_h)

    # 4. synthesize_sub002pct_antiloop_tail_surgery validation
    w_ok = torch.randn((8, 16))
    nw_ok = torch.ones(8)
    ad_ok = torch.ones((1, 8))
    p_ok = torch.eye(16)
    with pytest.raises(ValueError, match="resolve"):
        synthesize_sub002pct_antiloop_tail_surgery(nn.Module(), nw_ok, ad_ok, p_ok)
    with pytest.raises(TypeError):
        synthesize_sub002pct_antiloop_tail_surgery(123, nw_ok, ad_ok, p_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not be empty"):
        synthesize_sub002pct_antiloop_tail_surgery({}, nw_ok, ad_ok, p_ok)
    with pytest.raises(ValueError, match="not be empty"):
        synthesize_sub002pct_antiloop_tail_surgery([], nw_ok, ad_ok, p_ok)
    with pytest.raises(ValueError, match="cannot be provided"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, None, ad_ok, p_ok, final_norm_bias=nw_ok)
    with pytest.raises(ValueError, match="expose a 1D"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nn.Module(), ad_ok, p_ok)
    with pytest.raises(TypeError):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, "bad", ad_ok, p_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, p_ok, final_norm_bias=torch.zeros(4))
    with pytest.raises(ValueError, match="attractor_dirs"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, {}, p_ok)
    with pytest.raises(TypeError):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, "bad", p_ok)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, torch.ones((1, 8), dtype=torch.long), p_ok)
    with pytest.raises(ValueError, match="1D attractor_dirs"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, torch.empty(0), p_ok)
    with pytest.raises(ValueError, match="width"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, torch.empty((1, 0)), p_ok)
    with pytest.raises(ValueError, match="1D or 2D"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, torch.ones((1, 1, 8)), p_ok)
    with pytest.raises(ValueError, match="finite"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, torch.tensor([float("inf")] * 8), p_ok)
    with pytest.raises(ValueError, match="nullspace_eig_threshold"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, p_ok, nullspace_eig_threshold=1.0)
    with pytest.raises(ValueError, match="projection_side"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, p_ok, projection_side="bad")
    with pytest.raises(ValueError, match="square"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, torch.ones((4, 8)))
    with pytest.raises(ValueError, match="input dim"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, torch.eye(8), projection_side="input")
    with pytest.raises(ValueError, match="output dim"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, torch.eye(16), projection_side="output")
    with pytest.raises(ValueError, match="matches neither"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, torch.eye(5))
    with pytest.raises(TypeError, match="p_null"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No compatible"):
        synthesize_sub002pct_antiloop_tail_surgery(w_ok, nw_ok, ad_ok, {"other": torch.eye(5)})
    with pytest.raises(ValueError, match="matches neither d_out"):
        synthesize_sub002pct_antiloop_tail_surgery(
            w_ok, nw_ok, torch.ones((1, 5)), p_ok, total_model_params=500_000
        )

    # 5. execute_verified_antiloop_wsp_surgery & verify_antiloop_receipt_provenance validation
    m_val, lp_val, cp_val, _ = _build_loop_attractor_crowd_model(num_layers=1, seed=20260501)
    with pytest.raises(TypeError, match="mapping"):
        verify_antiloop_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AntiLoopWSPSurgeryConfig"):
        execute_verified_antiloop_wsp_surgery(m_val, [[1, 2, 3, 4]], [[5, 6, 7, 8]], config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_verified_antiloop_wsp_surgery(m_val, [[1, 2, 3, 4]], [[5, 6, 7, 8]], eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_antiloop_wsp_surgery("bad", [[1, 2]], [[3, 4]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        execute_verified_antiloop_wsp_surgery(nn.Linear(4, 4), [[1, 2]], [[3, 4]])
    with pytest.raises(ValueError, match="lie in"):
        execute_verified_antiloop_wsp_surgery(m_val, [[999, 1]], [[1, 2]])
    with pytest.raises(TypeError, match="target_layer_indices"):
        execute_verified_antiloop_wsp_surgery(m_val, [[1, 2]], [[3, 4]], target_layer_indices="0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not be empty"):
        execute_verified_antiloop_wsp_surgery(m_val, [[1, 2]], [[3, 4]], target_layer_indices=[])
    with pytest.raises(ValueError, match="lie in"):
        execute_verified_antiloop_wsp_surgery(m_val, [[1, 2]], [[3, 4]], target_layer_indices=[99])
    with pytest.raises(ValueError, match="eval_chunks"):
        execute_verified_antiloop_wsp_surgery(m_val, [[1, 2, 3, 4]] * 4, [[5, 6, 7, 8]] * 4, prose_eval_chunks=[])
    with pytest.raises(ValueError, match="lie in"):
        execute_verified_antiloop_wsp_surgery(
            m_val, [[1, 2, 3, 4]] * 4, [[5, 6, 7, 8]] * 4, prose_eval_chunks=torch.tensor([[999, 1]])
        )
    with pytest.raises(ValueError, match="expert_id"):
        execute_verified_antiloop_wsp_surgery(
            m_val, [[1, 2, 3, 4]] * 4, [[5, 6, 7, 8]] * 4, prose_eval_chunks=[{"expert_id": "E0"}]
        )
    with pytest.raises(ValueError, match="length >= 2"):
        execute_verified_antiloop_wsp_surgery(
            m_val, [[1, 2, 3, 4]] * 4, [[5, 6, 7, 8]] * 4, prose_eval_chunks=[{"expert_id": "E0", "ids": [1]}]
        )
    with pytest.raises(ValueError, match="identical sequence length"):
        execute_verified_antiloop_wsp_surgery(
            m_val,
            [[1, 2, 3, 4]] * 4,
            [[5, 6, 7, 8]] * 4,
            prose_eval_chunks=[{"expert_id": "E0", "ids": [1, 2]}, {"expert_id": "E0", "ids": [1, 2, 3]}],
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        execute_verified_antiloop_wsp_surgery(m_val, [[1, 2, 3, 4]] * 4, [[5, 6, 7, 8]] * 4, eval_fn=lambda _m: {})
    with pytest.raises(ValueError, match="domain NLL"):
        execute_verified_antiloop_wsp_surgery(
            m_val, [[1, 2, 3, 4]] * 4, [[5, 6, 7, 8]] * 4, eval_fn=lambda _m: {"repeat_4gram_ratio": 0.1}
        )
    with pytest.raises(ValueError, match="non_finite_outputs"):
        execute_verified_antiloop_wsp_surgery(
            m_val, [[1, 2, 3, 4]] * 4, [[5, 6, 7, 8]] * 4, eval_fn=lambda _m: {"non_finite_outputs": True}
        )
    assert len(lp_val) == 4 and len(cp_val) > 0

    # 6. AGPL-3.0 header, ZeroStubAudit (CI-safe try/except), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn34", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn34", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/thinking_loop_wsp_suppressor.py", TARGET_MODULE_PATH),
        ("tests/test_thinking_loop_wsp_suppressor_scenarios.py", TEST_FILE_PATH),
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

    telemetry_json = json.dumps(
        {
            "nll_mean": 1.12,
            "repeat_4gram_ratio": 0.0,
            "edited_param_fraction": 0.00014,
            "anse_energy": -0.78,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1134,
        pr_title="feat(swarm): sub-0.02pct thinking-loop and repetition-attractor weight-space suppressor for open-weight models",
        pr_body="Implements NEURAL-CTRL-DYN-34 sub-0.02% training-free thinking-loop and repetition-attractor weight-space suppressor.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): sub-0.02pct thinking-loop and repetition-attractor weight-space suppressor for open-weight models"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
