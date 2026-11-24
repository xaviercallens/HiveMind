# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-17: StreamingLLM attention-sink & GQA o_proj null-space surgeon."""

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
from hivemind.swarm.attention_sink_kv_surgeon import (
    ANSE_MAXIMUM_PAIN,
    SURGEON_RECEIPT_OBJECT,
    SURGEON_SCHEMA,
    AttentionSinkSurgeonConfig,
    canonical_json_bytes,
    compute_payload_sha256,
    compute_state_sha256,
    compute_tensor_sha256,
    detect_massive_sink_heads_and_channels,
    execute_verified_attention_sink_surgery,
    project_oproj_rogue_spike_into_prose_nullspace,
    verify_attention_sink_receipt_provenance,
)
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe

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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "attention_sink_kv_surgeon.py"
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


class _RMSNormBlock(nn.Module):
    """Minimal GQA block paired with HuggingFace-style ``o_proj`` and RMSNorm naming."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        head_dim = hidden_size // num_heads
        self.ln1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = _HFStyleGQAAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.ln2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )


class _HFStyleGQAAttention(nn.Module):
    """Attention module exposing ``num_attention_heads``, ``num_key_value_heads``, and ``o_proj``."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        _ = (cos[:, :, :t], sin[:, :, :t], past_kv)
        rep = self.num_attention_heads // self.num_key_value_heads
        out = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(rep, dim=1),
            v.repeat_interleave(rep, dim=1),
            is_causal=True,
        )
        projected = self.o_proj(out.transpose(1, 2).contiguous().reshape(b, t, -1))
        return projected, (k, v) if use_cache else None


class _HFStyleRMSModel(nn.Module):
    """Minimal RMSNorm + HF-style GQA model verifying norm & attribute compatibility."""

    def __init__(self, hidden_size: int = 16, vocab_size: int = 32) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([_RMSNormBlock(hidden_size, 4, 2)])
        self.ln = nn.RMSNorm(hidden_size, eps=1e-6)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.register_buffer("cos_cached", torch.ones((1, 1, 32, 4)), persistent=False)
        self.register_buffer("sin_cached", torch.zeros((1, 1, 32, 4)), persistent=False)


def _build_gqa_moe_spike_model(
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    vocab_size: int = 48,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 4,
    plant_mid_spike: bool = True,
    dual_role_bos_in_rogue_head: bool = False,
    bos_overlaps_rogue: bool = False,
    seed: int = 20261701,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor]]:
    """Build deterministic CPU GQA + Sparse MoE model with BOS sink & mid-sequence spike.

    Architecture: ``num_attention_heads=4``, ``num_key_value_heads=2``, ``head_dim=8``.
      - Token ``0..3``: BOS / StreamingLLM anchor sink tokens carrying dims ``(2, 3)``.
      - Token ``4..15``: ``E0_general_fineweb`` (target ``5``, dims ``0..11`` around ``u_e0``).
      - Token ``16..31``: ``E1_python_systems_code`` (target ``20``, dims ``12..19`` around ``u_e1``,
        plus mid-sequence spike trigger token ``25`` carrying dims ``(30, 31)``).
      - Token ``32..47``: ``E2_math_reasoning`` (target ``36``, dims ``20..27`` around ``u_e2``).
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

    u_e0 = _unit_axis(hidden_size, 0, 1)
    u_e1 = _unit_axis(hidden_size, 12, 13)
    u_e2 = _unit_axis(hidden_size, 20, 21)
    u_rogue = _unit_axis(hidden_size, 28, 29)
    u_bos_dual = _unit_axis(hidden_size, 24, 25)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)

        # BOS / StreamingLLM sink anchor tokens (0..3)
        for tok in range(4):
            wte[tok, 0:2] = 0.6 * u_e0[0:2]
            wte[tok, 2] = 3.0 / math.sqrt(2.0)
            wte[tok, 3] = -3.0 / math.sqrt(2.0)
            if dual_role_bos_in_rogue_head:
                wte[tok, 24] = 2.5 / math.sqrt(2.0)
                wte[tok, 25] = -2.5 / math.sqrt(2.0)
            if bos_overlaps_rogue:
                wte[tok, 30] = 2.8 / math.sqrt(2.0)
                wte[tok, 31] = -2.8 / math.sqrt(2.0)

        # E0 general prose (4..15), E1 code (16..31), E2 math (32..47)
        p_raw = 0.15 * torch.randn((12, 12), generator=gen, dtype=torch.float32)
        p_raw[:, :4] = 0.0
        wte[4:16, :12] = p_raw + 0.9 * u_e0[:12].unsqueeze(0)
        wte[5] = 2.6 * u_e0

        e1_raw = 0.15 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e1_raw[:, :2] = 0.0
        wte[16:32, 12:20] = e1_raw + 0.9 * u_e1[12:20].unsqueeze(0)
        wte[20] = 2.6 * u_e1
        # Mid-sequence trigger token 25 drives rogue massive activation in Head 3
        wte[25, 30] = 2.5 / math.sqrt(2.0)
        wte[25, 31] = -2.5 / math.sqrt(2.0)

        e2_raw = 0.15 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e2_raw[:, :2] = 0.0
        wte[32:48, 20:28] = e2_raw + 0.9 * u_e2[20:28].unsqueeze(0)
        wte[36] = 2.6 * u_e2

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.85 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.85 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:28, :28].copy_(0.15 * torch.eye(28))

        # Layer 0 GQA Attention (Q/K placed in low-frequency RoPE channels 2, 3, 6, 7 so phase stays ~0):
        # - KV Group 0 (Q heads 0, 1): Head 0 carries benign BOS sink at pos 0..3 (bos_ratio >= 7x, mid ~ 1x);
        #   Head 1 attends to regular mid tokens only (never BOS 0..3).
        # - KV Group 1 (Q heads 2, 3): Head 2 attends to regular tokens only (never token 25);
        #   Head 3 latches onto mid-sequence trigger token 25 -> massive spike in v_kv1[7] (col 31)!
        blk0 = model.blocks[0]
        # KV Group 0: Q/K channel 3 (low-freq) tracks BOS (2,3); Q/K channel 2 tracks regular tokens (u_e0/e1/e2)
        blk0.attn.q.weight[3, 2] = 3.5 / math.sqrt(2.0)
        blk0.attn.q.weight[3, 3] = -3.5 / math.sqrt(2.0)
        for q_row in (2, 10):
            for d0, d1 in ((0, 1), (12, 13), (20, 21)):
                blk0.attn.q.weight[q_row, d0] = 3.5 / math.sqrt(2.0)
                blk0.attn.q.weight[q_row, d1] = -3.5 / math.sqrt(2.0)
        # Head 1 strongly avoids BOS channel 3 keys on ALL mid-seq tokens (q_1[3] < 0, k_kv0[3] > 0)
        blk0.attn.q.weight[11, 2] = -4.0 / math.sqrt(2.0)
        blk0.attn.q.weight[11, 3] = 4.0 / math.sqrt(2.0)
        for d0, d1 in ((0, 1), (12, 13), (20, 21)):
            blk0.attn.q.weight[11, d0] = -6.0 / math.sqrt(2.0)
            blk0.attn.q.weight[11, d1] = 6.0 / math.sqrt(2.0)

        blk0.attn.k.weight[3, 2] = 3.5 / math.sqrt(2.0)
        blk0.attn.k.weight[3, 3] = -3.5 / math.sqrt(2.0)
        for d0, d1 in ((0, 1), (12, 13), (20, 21)):
            blk0.attn.k.weight[2, d0] = 3.5 / math.sqrt(2.0)
            blk0.attn.k.weight[2, d1] = -3.5 / math.sqrt(2.0)

        blk0.attn.v.weight[0, 2] = 8.0 / math.sqrt(2.0)
        blk0.attn.v.weight[0, 3] = -8.0 / math.sqrt(2.0)
        blk0.attn.v.weight[1, 0] = 0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[1, 1] = -0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[2, 12] = 0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[2, 13] = -0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[3, 20] = 0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[3, 21] = -0.85 / math.sqrt(2.0)

        # KV Group 1: Head 2 matches regular tokens via k_kv1[2] and strongly repels token 25 (k_kv1[7])!
        for d0, d1 in ((0, 1), (12, 13), (20, 21)):
            blk0.attn.q.weight[18, d0] = 4.0 / math.sqrt(2.0)
            blk0.attn.q.weight[18, d1] = -4.0 / math.sqrt(2.0)
            blk0.attn.q.weight[23, d0] = -5.0 / math.sqrt(2.0)
            blk0.attn.q.weight[23, d1] = 5.0 / math.sqrt(2.0)
            blk0.attn.k.weight[10, d0] = 3.5 / math.sqrt(2.0)
            blk0.attn.k.weight[10, d1] = -3.5 / math.sqrt(2.0)
        blk0.attn.q.weight[23, 30] = -4.0 / math.sqrt(2.0)
        blk0.attn.q.weight[23, 31] = 4.0 / math.sqrt(2.0)
        blk0.attn.k.weight[15, 30] = 4.0 / math.sqrt(2.0)
        blk0.attn.k.weight[15, 31] = -4.0 / math.sqrt(2.0)

        blk0.attn.v.weight[9, 12] = 0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[9, 13] = -0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[10, 20] = 0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[10, 21] = -0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[11, 0] = 0.85 / math.sqrt(2.0)
        blk0.attn.v.weight[11, 1] = -0.85 / math.sqrt(2.0)

        # Output projection: benign heads 0..2 carry substantial norm (~4.0) so Head 3 delta <= 3%
        blk0.attn.o.weight[:24, :24].copy_(0.85 * torch.eye(24))
        if dual_role_bos_in_rogue_head:
            # Head 3 ALSO carries a protected BOS anchor sink on channel 0 (col 24 -> dims 24..25)
            blk0.attn.q.weight[27, 24] = 3.5 / math.sqrt(2.0)
            blk0.attn.q.weight[27, 25] = -3.5 / math.sqrt(2.0)
            blk0.attn.q.weight[19, 24] = -4.0 / math.sqrt(2.0)
            blk0.attn.q.weight[19, 25] = 4.0 / math.sqrt(2.0)
            for d0, d1 in ((0, 1), (12, 13), (20, 21)):
                blk0.attn.q.weight[19, d0] = -6.0 / math.sqrt(2.0)
                blk0.attn.q.weight[19, d1] = 6.0 / math.sqrt(2.0)
            blk0.attn.k.weight[11, 24] = 3.5 / math.sqrt(2.0)
            blk0.attn.k.weight[11, 25] = -3.5 / math.sqrt(2.0)
            blk0.attn.v.weight[8, 24] = 9.5 / math.sqrt(2.0)
            blk0.attn.v.weight[8, 25] = -9.5 / math.sqrt(2.0)
            blk0.attn.o.weight[24:26, 24] = torch.tensor([0.75 / math.sqrt(2.0), -0.75 / math.sqrt(2.0)])

        if plant_mid_spike:
            # Head 3 attends sharply to trigger token 25 via k_kv1[7] -> massive activation in v_kv1[7] (col 31)
            blk0.attn.q.weight[31, 30] = 4.0 / math.sqrt(2.0)
            blk0.attn.q.weight[31, 31] = -4.0 / math.sqrt(2.0)
            blk0.attn.v.weight[15, 30] = 7.5 / math.sqrt(2.0)
            blk0.attn.v.weight[15, 31] = -7.5 / math.sqrt(2.0)
            # Column 31 has tiny Frobenius norm (0.11 / 4.16 ~= 0.026 <= 0.03) but 18 * 0.11 ~= 2.0 residual spike!
            blk0.attn.o.weight[28:30, 31] = 0.11 * u_rogue[28:30]

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 13,
    )

    with torch.no_grad():
        for l_idx, blk in enumerate(model.blocks):
            mlp = blk.mlp
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()

            # Clean domain routing + rogue spike router hijack on experts 0 and 3
            mlp.router.weight[0] = 3.5 * u_e0 + 2.5 * u_e1 + 8.0 * u_rogue
            mlp.router.weight[1] = 3.5 * u_e1
            mlp.router.weight[2] = 3.5 * u_e2
            mlp.router.weight[3] = -2.5 * u_e1 + 8.0 * u_rogue

            if l_idx == 0:
                # Expert 1 restores strong specialist E1 signal when correctly routed;
                # Rogue Expert 3 actively injects destructive off-target logit drift when hijacked!
                mlp.experts["1"].w3_A[0, 12:20] = 0.25 * u_e1[12:20]
                mlp.experts["1"].w3_B[:, 0] = 1.65 * u_e1
                mlp.experts["2"].w3_A[0, 20:28] = 0.25 * u_e2[20:28]
                mlp.experts["2"].w3_B[:, 0] = 1.40 * u_e2
                mlp.experts["3"].w3_A[0, :hidden_size] = 0.30 * (u_rogue + 0.5 * u_e1)
                mlp.experts["3"].w3_B[:, 0] = -2.20 * u_e1 + 1.50 * u_rogue

    return model, {"e0": u_e0, "e1": u_e1, "e2": u_e2, "rogue": u_rogue, "bos_dual": u_bos_dual}


def _make_streaming_chunks(
    *,
    seq_len: int = 16,
    chunks_per_domain: int = 6,
    include_long_context: bool = False,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Create deterministic chunks with BOS anchor positions ``0..3`` and mid-sequence content."""
    gen = torch.Generator(device="cpu").manual_seed(7700 + seed_offset)
    chunks: list[dict[str, Any]] = []
    bos_prefix = [0, 1, 2, 3]

    for r_idx in range(chunks_per_domain):
        tail_e0 = torch.randint(4, 16, (seq_len - 4,), generator=gen).tolist()
        for pos in range(2, len(tail_e0), 2):
            tail_e0[pos] = 5
        chunks.append(
            {
                "expert_id": "E0_general_fineweb",
                "row": seed_offset + r_idx,
                "ids": bos_prefix + tail_e0,
            }
        )

    for r_idx in range(chunks_per_domain):
        tail_e1 = torch.randint(16, 32, (seq_len - 4,), generator=gen).tolist()
        # Plant mid-sequence trigger token 25 right before gold target token 20
        for pos in range(0, len(tail_e1) - 1, 3):
            tail_e1[pos] = 25
            tail_e1[pos + 1] = 20
        chunks.append(
            {
                "expert_id": "E1_python_systems_code",
                "row": seed_offset + 100 + r_idx,
                "ids": bos_prefix + tail_e1,
            }
        )

    for r_idx in range(chunks_per_domain):
        tail_e2 = torch.randint(32, 48, (seq_len - 4,), generator=gen).tolist()
        for pos in range(2, len(tail_e2), 2):
            tail_e2[pos] = 36
        chunks.append(
            {
                "expert_id": "E2_math_reasoning",
                "row": seed_offset + 200 + r_idx,
                "ids": bos_prefix + tail_e2,
            }
        )

    if include_long_context:
        for r_idx in range(2):
            tail_long = torch.randint(4, 16, (seq_len - 4,), generator=gen).tolist()
            tail_long[-1] = 5
            chunks.append(
                {
                    "expert_id": "LONG_context_window",
                    "row": seed_offset + 300 + r_idx,
                    "ids": bos_prefix + tail_long,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# S1: Mid-Sequence GQA Spike Cured -> >80% Suppression, MoE Routing & NLL Gain
# ---------------------------------------------------------------------------


def test_s1_gqa_midseq_spike_misroutes_moe_and_nullspace_surgery_cures_it() -> None:
    """Mid-sequence rogue GQA spike hijacks MoE router; o_proj null-space surgery cures it (>80% drop)."""
    model, dirs = _build_gqa_moe_spike_model(plant_mid_spike=True, seed=20261701)
    probe_chunks = _make_streaming_chunks(chunks_per_domain=6, include_long_context=True, seed_offset=0)
    eval_chunks = _make_streaming_chunks(chunks_per_domain=6, include_long_context=True, seed_offset=500)
    long_chunks = [c for c in eval_chunks if c["expert_id"].startswith("E0_")][:3]

    # Measure pre-surgery Layer 0 MoE routing on E1 trigger tokens (token 25)
    e1_ids = torch.tensor([c["ids"] for c in eval_chunks if c["expert_id"].startswith("E1_")], dtype=torch.long)
    with torch.no_grad():
        x0 = model.wte(e1_ids).to(torch.float32)
        attn_out0, _ = model.blocks[0].attn(model.blocks[0].ln1(x0), model.cos_cached, model.sin_cached)
        mlp_in0 = model.blocks[0].ln2(x0 + attn_out0)
        trig_mask = e1_ids == 25
        pre_router_probs = torch.sigmoid(model.blocks[0].mlp.router(mlp_in0[trig_mask]))
        pre_top2 = torch.topk(pre_router_probs, k=2, dim=-1).indices
        pre_expert3_hijack_rate = float((pre_top2 == 3).any(dim=-1).float().mean().item())
        pre_expert1_hit_rate = float((pre_top2 == 1).any(dim=-1).float().mean().item())
    assert pre_expert3_hijack_rate > 0.80, "Pre-surgery rogue spike should hijack MoE router into expert 3"

    # Direct detector verification (GQA: 4 query heads -> 2 KV heads, Head 0 benign BOS, Head 3 rogue)
    synthetic_z = torch.randn((2, 12, 4, 8), generator=torch.Generator(device="cpu").manual_seed(17), dtype=torch.float32) * 0.25
    synthetic_z[:, :4, 0, 0] = 6.0  # Head 0 (KV head 0): benign BOS sink
    synthetic_z[:, 7, 3, 6] = 8.0   # Head 3 (KV head 1): mid-sequence rogue spike
    det_res = detect_massive_sink_heads_and_channels(
        synthetic_z,
        torch.arange(12),
        sink_positions=(0, 1, 2, 3),
        spike_ratio_threshold=5.0,
        num_kv_heads=2,
        o_proj_weight=model.blocks[0].attn.o.weight,
    )
    assert 0 in det_res["benign_bos_sink_heads"]
    assert 0 in det_res["benign_bos_sink_kv_heads"]
    assert det_res["rogue_spike_heads"] == [3]
    assert det_res["rogue_spike_kv_heads"] == [1]
    assert 6 in det_res["rogue_channels_by_head"][3]
    assert 30 in det_res["rogue_flat_channels"]
    assert det_res["rogue_head_slices"][3] == slice(24, 32)
    assert abs(float(det_res["rogue_head_directions"][3][6].item())) > 0.90

    # End-to-end verified surgery on the GQA + Sparse MoE model
    receipt = execute_verified_attention_sink_surgery(
        model,
        probe_chunks,
        eval_chunks=eval_chunks,
        long_context_chunks=long_chunks,
        score_from=4,
    )
    assert receipt["object"] == SURGEON_RECEIPT_OBJECT
    assert receipt["schema"] == SURGEON_SCHEMA
    assert verify_attention_sink_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "ACCEPTED"
    assert receipt["guard_ok"] is True
    assert receipt["delta_e"] < 0.0
    assert receipt["spike_suppression_ratio"] > 0.80
    assert receipt["general_pct_vs_base"] <= 0.30
    assert receipt["long_context_pct_vs_base"] <= 1.0
    assert receipt["per_domain"]["E1_python_systems_code"]["mean_diff"] < -0.05
    assert receipt["sink_norm_ratio"] >= 5.0
    assert 0 in receipt["detections_per_layer"][0]["benign_bos_sink_heads"]
    assert 3 in receipt["detections_per_layer"][0]["rogue_spike_heads"]

    # Post-surgery router check: E1 trigger tokens now route to Expert 1 instead of hijacked Expert 3!
    with torch.no_grad():
        x0_post = model.wte(e1_ids).to(torch.float32)
        attn_out0_post, _ = model.blocks[0].attn(model.blocks[0].ln1(x0_post), model.cos_cached, model.sin_cached)
        mlp_in0_post = model.blocks[0].ln2(x0_post + attn_out0_post)
        post_router_probs = torch.sigmoid(model.blocks[0].mlp.router(mlp_in0_post[trig_mask]))
        post_top2 = torch.topk(post_router_probs, k=2, dim=-1).indices
        post_expert3_hijack_rate = float((post_top2 == 3).any(dim=-1).float().mean().item())
        post_expert1_hit_rate = float((post_top2 == 1).any(dim=-1).float().mean().item())
    assert post_expert3_hijack_rate < 0.10
    assert post_expert1_hit_rate > pre_expert1_hit_rate
    assert float(dirs["rogue"].norm().item()) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# S2: Exact Preservation (<1e-5 Drift) of BOS Positions 0..3 Anchor Sinks
# ---------------------------------------------------------------------------


def test_s2_exact_preservation_of_bos_positions_0_to_3_anchor_representations() -> None:
    """Dual-role head carrying BOS sink + mid-sequence spike preserves BOS 0..3 within < 1e-5 drift."""
    model, dirs = _build_gqa_moe_spike_model(
        plant_mid_spike=True,
        dual_role_bos_in_rogue_head=True,
        seed=20261702,
    )
    probe_chunks = _make_streaming_chunks(chunks_per_domain=6, seed_offset=20)
    eval_chunks = _make_streaming_chunks(chunks_per_domain=6, seed_offset=520)

    # 1. Direct projection preserving BOS + prose activations in both head and residual space
    w_o = model.blocks[0].attn.o.weight.detach().clone()
    z_bos = torch.zeros((4, 8), dtype=torch.float32)
    z_bos[:, 0] = 2.5  # BOS anchor lives on channel 0 of Head 3
    z_prose = torch.zeros((6, 8), dtype=torch.float32)
    z_prose[:, 1] = 1.2
    preserve_head = torch.cat([z_bos, z_prose], dim=0)

    r_dir_head = torch.zeros(8, dtype=torch.float32)
    r_dir_head[0] = 0.25  # small accidental overlap with BOS channel 0
    r_dir_head[7] = 1.00  # dominant rogue spike on channel 7
    w_new_h, rep_h = project_oproj_rogue_spike_into_prose_nullspace(
        w_o,
        slice(24, 32),
        r_dir_head,
        preserve_head,
        damp_alpha=0.85,
        max_norm_delta_ratio=0.03,
    )
    assert rep_h["projection_mode"] == "head_input_nullspace"
    assert rep_h["nullspace_mode"] == "exact_nullspace"
    assert rep_h["preserved_rms_drift"] < 1e-6
    assert rep_h["spike_suppression_ratio"] > 0.80
    bos_diff_h = (z_bos @ w_new_h[:, 24:32].T) - (z_bos @ w_o[:, 24:32].T)
    assert float(bos_diff_h.abs().max().item()) < 1e-6

    # Residual-space call (rogue_dir in R^32, preserve_sink_acts in R^32) + sequence slice support
    y_bos = z_bos @ w_o[:, 24:32].T
    w_new_r, rep_r = project_oproj_rogue_spike_into_prose_nullspace(
        w_o,
        list(range(24, 32)),
        dirs["rogue"] + 0.3 * dirs["bos_dual"],
        y_bos,
        damp_alpha=0.85,
        max_norm_delta_ratio=0.03,
    )
    assert rep_r["projection_mode"] == "residual_output_nullspace"
    assert rep_r["nullspace_mode"] == "exact_nullspace"
    bos_diff_r = (z_bos @ w_new_r[:, 24:32].T) - y_bos
    assert float(bos_diff_r.abs().max().item()) < 1e-6

    # Mixed space: head rogue_dir (8,) with residual preserve_acts (N, 32)
    _, rep_mixed = project_oproj_rogue_spike_into_prose_nullspace(
        w_o,
        slice(24, 32),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        {"bos": y_bos},
        damp_alpha=0.85,
        max_norm_delta_ratio=0.03,
    )
    assert rep_mixed["projection_mode"] == "residual_output_nullspace"
    assert rep_mixed["preserved_rms_drift"] < 1e-6

    # 2. End-to-end verified surgery: dual-role head detected & BOS 0..3 drifted < 1e-5
    eval_ids = torch.tensor([c["ids"] for c in eval_chunks], dtype=torch.long)
    with torch.no_grad():
        x_bos_pre = model.wte(eval_ids).to(torch.float32)
        attn_pre, _ = model.blocks[0].attn(model.blocks[0].ln1(x_bos_pre), model.cos_cached, model.sin_cached)
        bos_pre_slice = attn_pre[:, :4, :].clone()

    receipt = execute_verified_attention_sink_surgery(
        model,
        probe_chunks,
        eval_chunks=eval_chunks,
        score_from=4,
        projection_space="residual",
    )
    assert receipt["accepted"] is True
    assert receipt["bos_guard_ok"] is True
    assert receipt["bos_anchor_max_abs_drift"] < 1e-5
    assert 3 in receipt["detections_per_layer"][0]["dual_role_heads"]
    assert receipt["spike_suppression_ratio"] > 0.80

    with torch.no_grad():
        attn_post, _ = model.blocks[0].attn(model.blocks[0].ln1(x_bos_pre), model.cos_cached, model.sin_cached)
        bos_post_slice = attn_post[:, :4, :]
        direct_bos_drift = float((bos_post_slice - bos_pre_slice).abs().max().item())
    assert direct_bos_drift < 1e-5


# ---------------------------------------------------------------------------
# S3: Norm-Delta Budget Enforcement & Zero KV-Cache Shape Change
# ---------------------------------------------------------------------------


def test_s3_norm_delta_budget_enforcement_and_zero_kv_cache_shape_change() -> None:
    """Enforce max_norm_delta_ratio across dtypes and verify KV-cache shapes & RoPE invariance."""
    gen = torch.Generator(device="cpu").manual_seed(20261703)
    w_large = torch.randn((32, 32), generator=gen, dtype=torch.float32)
    w_large[:, 24:32] *= 4.0  # dominant rogue slice forcing budget clamp at 0.03
    r_head = torch.zeros(8, dtype=torch.float32)
    r_head[7] = 1.0
    p_head = torch.zeros((4, 8), dtype=torch.float32)
    p_head[:, 0] = 1.0

    for dt in (torch.float32, torch.float64, torch.bfloat16):
        w_dt = w_large.to(dtype=dt)
        w_upd, rep = project_oproj_rogue_spike_into_prose_nullspace(
            w_dt,
            slice(24, 32),
            r_head,
            p_head,
            damp_alpha=0.85,
            max_norm_delta_ratio=0.03,
        )
        assert rep["budget_clamped"] is True
        assert rep["budget_scale"] < 1.0
        assert rep["applied_norm_delta_ratio"] <= 0.03 + 1e-6
        assert rep["non_target_columns_bitwise_unchanged"] is True
        assert torch.equal(w_upd[:, :24], w_dt[:, :24])

    # In-place mutation via nn.Linear wrapper + full KV-cache & RoPE check on GQA + MoE model
    model, _ = _build_gqa_moe_spike_model(plant_mid_spike=True, seed=20261703)
    probe_chunks = _make_streaming_chunks(chunks_per_domain=4, seed_offset=40)
    eval_ids = torch.tensor([c["ids"] for c in probe_chunks], dtype=torch.long)

    pre_cos_sha = compute_tensor_sha256(model.cos_cached)
    pre_sin_sha = compute_tensor_sha256(model.sin_cached)
    pre_kv_weights_sha = compute_state_sha256(
        {
            f"blk{i}_{p}": getattr(blk.attn, p).weight
            for i, blk in enumerate(model.blocks)
            for p in ("q", "k", "v")
        }
    )
    with torch.no_grad():
        _, _, pre_kvs = model(eval_ids, use_cache=True)
    assert pre_kvs is not None
    pre_kv_shapes = [(tuple(k.shape), tuple(v.shape)) for k, v in pre_kvs]

    receipt = execute_verified_attention_sink_surgery(
        model,
        probe_chunks,
        max_norm_delta_ratio=0.015,
        score_from=4,
        layer_indices=[0],
    )
    assert receipt["accepted"] is True
    assert receipt["kv_shape_unchanged"] is True
    assert receipt["rope_frequencies_unchanged"] is True
    assert all(r["applied_norm_delta_ratio"] <= 0.015 + 1e-6 for r in receipt["layer_reports"])

    post_kv_weights_sha = compute_state_sha256(
        {
            f"blk{i}_{p}": getattr(blk.attn, p).weight
            for i, blk in enumerate(model.blocks)
            for p in ("q", "k", "v")
        }
    )
    assert post_kv_weights_sha == pre_kv_weights_sha
    assert compute_tensor_sha256(model.cos_cached) == pre_cos_sha
    assert compute_tensor_sha256(model.sin_cached) == pre_sin_sha

    with torch.no_grad():
        _, _, post_kvs = model(eval_ids, use_cache=True)
    assert post_kvs is not None
    assert [(tuple(k.shape), tuple(v.shape)) for k, v in post_kvs] == pre_kv_shapes

    # Verify RMSNorm + HuggingFace-style o_proj compatibility path
    hf_model = _HFStyleRMSModel()
    hf_chunks = [
        {"expert_id": "E0_gen", "ids": [0, 1, 2, 3, 4, 5, 6, 7]},
        {"expert_id": "E1_spec", "ids": [0, 1, 2, 3, 8, 9, 10, 11]},
    ]
    hf_rec = execute_verified_attention_sink_surgery(hf_model, hf_chunks, score_from=4)
    assert hf_rec["status"] == "FIXED_POINT_UNCHANGED"
    assert verify_attention_sink_receipt_provenance(hf_rec) is True


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback on BOS / Domain Guard Violations
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_when_bos_or_domain_guard_violated() -> None:
    """Violating BOS anchor guard or general NLL guard triggers ANSE Maximum Pain & bitwise rollback."""
    # Case 4a: BOS shares rogue channel & protect_bos_sinks=False -> BOS anchor drift > 1e-5 -> rollback!
    model_bos, _ = _build_gqa_moe_spike_model(
        plant_mid_spike=True,
        bos_overlaps_rogue=True,
        seed=20261704,
    )
    chunks = _make_streaming_chunks(chunks_per_domain=5, seed_offset=60)
    parent_sha_bos = compute_state_sha256(model_bos.state_dict())

    rec_bos = execute_verified_attention_sink_surgery(
        model_bos,
        chunks,
        score_from=4,
        protect_bos_sinks=False,
        protect_general_prose=False,
        bos_anchor_max_drift=1e-5,
    )
    assert rec_bos["accepted"] is False
    assert rec_bos["rolled_back"] is True
    assert rec_bos["bos_guard_ok"] is False
    assert rec_bos["guard_ok"] is False
    assert rec_bos["status"] == "ROLLED_BACK_BOS_ANCHOR_GUARD_VIOLATION"
    assert rec_bos["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_bos["delta_e"] > 0.0
    assert rec_bos["final_state_sha256"] == parent_sha_bos
    assert compute_state_sha256(model_bos.state_dict()) == parent_sha_bos
    assert verify_attention_sink_receipt_provenance(rec_bos) is True

    # Case 4b: General / specialist domain drift guard violated (> +1.0%) -> Maximum Pain rollback!
    model_dom, _ = _build_gqa_moe_spike_model(plant_mid_spike=True, seed=20261705)
    with torch.no_grad():
        # Wire Head 3 to carry a helpful E1 signal without router hijack -> suppressing it violates <= +1.0% guard
        model_dom.blocks[0].mlp.router.weight[0] = 3.5 * _unit_axis(32, 0, 1)
        model_dom.blocks[0].mlp.router.weight[3].zero_()
        model_dom.blocks[0].attn.o.weight[:, 31].zero_()
        model_dom.blocks[0].attn.o.weight[12:14, 31] = 0.06 * _unit_axis(32, 12, 13)[12:14]
    parent_sha_dom = compute_state_sha256(model_dom.state_dict())
    rec_dom = execute_verified_attention_sink_surgery(
        model_dom,
        chunks,
        score_from=4,
        protect_general_prose=False,
        domain_guard_max_ratio=1.01,
        general_guard_max_ratio=1.01,
    )
    assert rec_dom["accepted"] is False
    assert rec_dom["rolled_back"] is True
    assert rec_dom["guard_ok"] is False
    assert rec_dom["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_dom["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_dom["final_state_sha256"] == parent_sha_dom

    # Case 4c: Guards pass, but Delta E >= 0 (non-improving energy) -> atomic rollback!
    model_ni, _ = _build_gqa_moe_spike_model(plant_mid_spike=True, seed=20261706)
    with torch.no_grad():
        # Remove router hijack & make o_proj spike mildly help E1 (< 1% guard) while spike_energy_weight=0.0
        model_ni.blocks[0].mlp.router.weight[0] = 3.5 * _unit_axis(32, 0, 1)
        model_ni.blocks[0].mlp.router.weight[3].zero_()
        model_ni.blocks[0].attn.o.weight[:, 31].zero_()
        model_ni.blocks[0].attn.o.weight[12:14, 31] = 0.0015 * _unit_axis(32, 12, 13)[12:14]
    parent_sha_ni = compute_state_sha256(model_ni.state_dict())
    rec_ni = execute_verified_attention_sink_surgery(
        model_ni,
        chunks,
        config=AttentionSinkSurgeonConfig(score_from=4, spike_energy_weight=0.0),
    )
    assert rec_ni["accepted"] is False
    assert rec_ni["rolled_back"] is True
    assert rec_ni["guard_ok"] is True
    assert rec_ni["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_ni["final_state_sha256"] == parent_sha_ni


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / Stub / Gatekeeper Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """Verify exhaustive input validation, edge cases, AGPL-3.0, ZeroStubAudit & PRSecurityScanner."""
    # 1. AttentionSinkSurgeonConfig & provenance validation
    with pytest.raises(TypeError, match="sequence"):
        AttentionSinkSurgeonConfig(sink_positions="0123")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        AttentionSinkSurgeonConfig(sink_positions=())
    with pytest.raises(TypeError, match="int"):
        AttentionSinkSurgeonConfig(sink_positions=(True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        AttentionSinkSurgeonConfig(sink_positions=(-1,))
    with pytest.raises(ValueError, match="> 1.0"):
        AttentionSinkSurgeonConfig(spike_ratio_threshold=1.0)
    with pytest.raises(TypeError, match="real number"):
        AttentionSinkSurgeonConfig(damp_alpha=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        AttentionSinkSurgeonConfig(damp_alpha=float("nan"))
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        AttentionSinkSurgeonConfig(damp_alpha=1.2)
    with pytest.raises(ValueError, match="<= 1.0"):
        AttentionSinkSurgeonConfig(max_norm_delta_ratio=1.5)
    with pytest.raises(ValueError, match=">= 0.0"):
        AttentionSinkSurgeonConfig(reg_preserve=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        AttentionSinkSurgeonConfig(max_norm_delta_ratio=0.0)
    with pytest.raises(ValueError, match="< 1.0"):
        AttentionSinkSurgeonConfig(nullspace_sv_rel_tol=1.0)
    with pytest.raises(TypeError, match="bool"):
        AttentionSinkSurgeonConfig(shrink_preserve="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1.0"):
        AttentionSinkSurgeonConfig(domain_guard_max_ratio=0.99)
    with pytest.raises(ValueError, match="general_prefix"):
        AttentionSinkSurgeonConfig(general_prefix=" ")
    with pytest.raises(ValueError, match="long_context_prefix"):
        AttentionSinkSurgeonConfig(long_context_prefix=" ")
    with pytest.raises(ValueError, match=">= 1"):
        AttentionSinkSurgeonConfig(batch_size=0)

    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_tensor_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-finite"):
        compute_tensor_sha256(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="non-empty Mapping"):
        compute_state_sha256({})
    with pytest.raises(ValueError, match="non-empty string"):
        compute_state_sha256({"": torch.ones(1)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"w": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"w": torch.tensor([float("inf")])})
    with pytest.raises(TypeError, match="mapping"):
        verify_attention_sink_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_attention_sink_receipt_provenance({"receipt_sha256": "xyz"}) is False
    assert verify_attention_sink_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_attention_sink_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": SURGEON_RECEIPT_OBJECT,
                "schema": SURGEON_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 2. detect_massive_sink_heads_and_channels validation & 3D / edge shapes
    with pytest.raises(TypeError, match="torch.Tensor"):
        detect_massive_sink_heads_and_channels([[1.0]], [0, 1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        detect_massive_sink_heads_and_channels(torch.empty((1, 0, 2, 4)), [0])
    with pytest.raises(TypeError, match="floating-point"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4), dtype=torch.long), torch.arange(6))
    with pytest.raises(ValueError, match="finite"):
        detect_massive_sink_heads_and_channels(torch.full((1, 6, 2, 4), float("nan")), torch.arange(6))
    with pytest.raises(ValueError, match="3 or 4 dims"):
        detect_massive_sink_heads_and_channels(torch.ones((2, 4)), torch.arange(2))
    with pytest.raises(ValueError, match="head_dim"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4)), torch.arange(6), head_dim=3)
    with pytest.raises(ValueError, match="divisible by head_dim"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 10)), torch.arange(6), head_dim=4)
    with pytest.raises(ValueError, match="spike_ratio_threshold"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4)), torch.arange(6), spike_ratio_threshold=1.0)
    with pytest.raises(ValueError, match="divisible by num_kv_heads"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 4, 4)), torch.arange(6), num_kv_heads=3)
    with pytest.raises(ValueError, match="o_proj_weight input width"):
        detect_massive_sink_heads_and_channels(
            torch.ones((1, 6, 2, 4)), torch.arange(6), o_proj_weight=torch.ones((8, 12))
        )
    with pytest.raises(TypeError, match="integer dtype"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4)), torch.ones(6, dtype=torch.float32))
    with pytest.raises(TypeError, match="torch.Tensor or sequence"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4)), "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4)), [])
    with pytest.raises(ValueError, match="non-negative"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4)), [-1, 0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="match seq_len"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4)), torch.arange(5))
    with pytest.raises(ValueError, match="2D token_positions"):
        detect_massive_sink_heads_and_channels(torch.ones((2, 6, 2, 4)), torch.ones((1, 6), dtype=torch.long))
    with pytest.raises(ValueError, match="1D or 2D"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 6, 2, 4)), torch.ones((1, 1, 6), dtype=torch.long))
    with pytest.raises(ValueError, match="non-sink mid-sequence"):
        detect_massive_sink_heads_and_channels(torch.ones((1, 4, 2, 4)), [0, 1, 2, 3])

    # Exercise 3D unbatched + 3D flattened + zero-sink-window + distributed-channel spike
    det_3d = detect_massive_sink_heads_and_channels(torch.ones((6, 2, 4)), [10, 11, 12, 13, 14, 15])
    assert det_3d["num_heads"] == 2 and det_3d["benign_bos_sink_heads"] == []
    z_flat_3d = torch.ones((1, 6, 8)) * 0.2
    z_flat_3d[0, 5, 4:8] = 4.0
    det_flat = detect_massive_sink_heads_and_channels(z_flat_3d, torch.arange(6), head_dim=4)
    assert det_flat["rogue_spike_heads"] == [1]

    # 3. project_oproj_rogue_spike_into_prose_nullspace validation & noop branches
    w8 = torch.eye(8)
    r4 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="<= 1.0"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, None, max_norm_delta_ratio=1.5)
    with pytest.raises(ValueError, match="< 1.0"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, None, nullspace_sv_rel_tol=1.0)
    with pytest.raises(TypeError, match="torch.Tensor or nn.Linear"):
        project_oproj_rogue_spike_into_prose_nullspace("bad", slice(0, 4), r4, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 2D"):
        project_oproj_rogue_spike_into_prose_nullspace(torch.ones(4), slice(0, 4), r4, None)
    with pytest.raises(TypeError, match="floating-point"):
        project_oproj_rogue_spike_into_prose_nullspace(torch.ones((4, 4), dtype=torch.long), slice(0, 4), r4, None)
    with pytest.raises(ValueError, match="finite"):
        project_oproj_rogue_spike_into_prose_nullspace(torch.full((4, 4), float("nan")), slice(0, 4), r4, None)
    with pytest.raises(ValueError, match="step must be 1"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 8, 2), r4, None)
    with pytest.raises(TypeError, match="slice or sequence"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, 0, r4, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty column"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(4, 4), r4, None)
    with pytest.raises(ValueError, match="duplicate"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, [0, 0, 1, 2], r4, None)
    with pytest.raises(ValueError, match="lie in"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, [0, 99], torch.ones(2), None)
    with pytest.raises(TypeError, match="torch.Tensor"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), [1.0, 0.0, 0.0, 0.0], None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 1D"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), torch.ones((2, 2)), None)
    with pytest.raises(TypeError, match="floating-point"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), torch.ones(4, dtype=torch.long), None)
    with pytest.raises(ValueError, match="finite"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), torch.tensor([float("nan"), 0.0, 0.0, 0.0]), None)
    with pytest.raises(ValueError, match="must match head slice"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), torch.ones(3), None)
    with pytest.raises(ValueError, match="must not be empty"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, {})
    with pytest.raises(ValueError, match="must not be empty"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, [])
    with pytest.raises(TypeError, match="preserve_sink_acts"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, ["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="2D or 3D"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, torch.ones((2, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, torch.full((2, 4), float("nan")))
    with pytest.raises(ValueError, match="width"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, [torch.ones((2, 4)), torch.ones((2, 3))])
    with pytest.raises(ValueError, match="preserve_sink_acts width"):
        project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, torch.ones((2, 5)))

    # Exercise noop, full-rank generalized_rayleigh, and nn.Linear inplace branches
    _, noop_zero_w = project_oproj_rogue_spike_into_prose_nullspace(torch.zeros((8, 8)), slice(0, 4), r4, None)
    assert noop_zero_w["noop_reason"] == "zero_weight_norm"
    _, noop_alpha = project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, None, damp_alpha=0.0)
    assert noop_alpha["noop_reason"] == "zero_damp_alpha"
    _, noop_dir = project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), torch.zeros(4), None)
    assert noop_dir["noop_reason"] == "zero_rogue_dir"
    _, noop_span = project_oproj_rogue_spike_into_prose_nullspace(
        w8, slice(0, 4), r4, torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    assert noop_span["noop_reason"] == "rogue_dir_in_preserved_span"
    _, noop_zero_pres = project_oproj_rogue_spike_into_prose_nullspace(w8, slice(0, 4), r4, torch.zeros((2, 4)))
    assert noop_zero_pres["nullspace_mode"] == "unconstrained"

    # Full-rank generalized Rayleigh (non-orthogonal preserved matrix with shrink_preserve=False and True)
    full_pres = torch.eye(4) + 0.05 * torch.randn((4, 4), generator=torch.Generator(device="cpu").manual_seed(9))
    _, rep_ray = project_oproj_rogue_spike_into_prose_nullspace(
        w8, slice(0, 4), r4, full_pres, shrink_preserve=True
    )
    assert rep_ray["nullspace_mode"] == "generalized_rayleigh"

    lin_mod = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        lin_mod.weight.copy_(w8)
    _, rep_lin = project_oproj_rogue_spike_into_prose_nullspace(
        lin_mod, slice(0, 4), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), torch.ones((2, 8)), inplace=True
    )
    assert rep_lin["projection_mode"] == "residual_output_nullspace"

    # 4. execute_verified_attention_sink_surgery validation checks
    m_small, _ = _build_gqa_moe_spike_model(plant_mid_spike=False, seed=20261709)
    ok_chunks = _make_streaming_chunks(chunks_per_domain=2, seed_offset=80)
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_attention_sink_surgery("bad", ok_chunks)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        execute_verified_attention_sink_surgery(nn.Linear(4, 4), ok_chunks)
    with pytest.raises(ValueError, match="must expose"):
        bad_m = nn.Module()
        bad_m.blocks = nn.ModuleList([nn.Linear(4, 4)])
        execute_verified_attention_sink_surgery(bad_m, ok_chunks)
    with pytest.raises(ValueError, match="projection_space"):
        execute_verified_attention_sink_surgery(m_small, ok_chunks, projection_space="invalid")
    with pytest.raises(TypeError, match="AttentionSinkSurgeonConfig"):
        execute_verified_attention_sink_surgery(m_small, ok_chunks, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        execute_verified_attention_sink_surgery(m_small, ok_chunks, layer_indices="0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty subset"):
        execute_verified_attention_sink_surgery(m_small, ok_chunks, layer_indices=[99])
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_attention_sink_surgery(m_small, [])
    with pytest.raises(ValueError, match="expert_id"):
        execute_verified_attention_sink_surgery(m_small, [{"ids": [0, 1, 2, 3, 4]}], score_from=4)
    with pytest.raises(ValueError, match="non-empty string"):
        execute_verified_attention_sink_surgery(m_small, [{"expert_id": " ", "ids": [0, 1, 2, 3, 4]}], score_from=4)
    with pytest.raises(TypeError, match="integer dtype"):
        execute_verified_attention_sink_surgery(
            m_small, [{"expert_id": "E0_a", "ids": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}], score_from=4
        )
    with pytest.raises(TypeError, match="sequence or 1D"):
        execute_verified_attention_sink_surgery(m_small, [{"expert_id": "E0_a", "ids": 123}], score_from=4)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="length >= 2"):
        execute_verified_attention_sink_surgery(m_small, [{"expert_id": "E0_a", "ids": [0]}], score_from=0)
    with pytest.raises(ValueError, match="token IDs must lie"):
        execute_verified_attention_sink_surgery(m_small, [{"expert_id": "E0_a", "ids": [0, 999]}], score_from=0)
    with pytest.raises(ValueError, match="too short"):
        execute_verified_attention_sink_surgery(m_small, [{"expert_id": "E0_a", "ids": [0, 1, 2]}], score_from=4)
    with pytest.raises(ValueError, match="share sequence length"):
        execute_verified_attention_sink_surgery(
            m_small,
            [{"expert_id": "E0_a", "ids": [0, 1, 2, 3, 4, 5]}, {"expert_id": "E1_b", "ids": [0, 1, 2, 3, 4]}],
            score_from=4,
        )
    with pytest.raises(ValueError, match="both general"):
        execute_verified_attention_sink_surgery(
            m_small, [{"expert_id": "E0_a", "ids": [0, 1, 2, 3, 4, 5]}], score_from=4
        )

    # 5. AGPL-3.0, optional ZeroStubAudit, PRSecurityScanner, lateral movement & pr_gatekeeper checks
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn17", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn17", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/attention_sink_kv_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_attention_sink_kv_surgeon_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 0.42, "sink_norm_ratio": 9.8, "anse_energy": -0.18},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}"
        )
        == []
    )
    assert len(canonical_json_bytes({"ok": True})) > 0
    assert len(compute_payload_sha256({"ok": True})) == 64

    pr_audit = scanner.audit_pull_request(
        pr_number=1117,
        pr_title="feat(swarm): StreamingLLM attention-sink & GQA value-output null-space stabilizer for open-weight models",
        pr_body="Implements NEURAL-CTRL-DYN-17 attention-sink & GQA o_proj null-space stabilizer with deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): StreamingLLM attention-sink & GQA value-output null-space stabilizer for open-weight models"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
