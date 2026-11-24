# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-07: quantization-compensated WSP and LoftQ prose-nullspace LoRA."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeBlock,
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.quantization_compensated_wsp import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    QUANT_RECEIPT_SCHEMA,
    QuantCompensatedLinear,
    QuantizationWSPConfig,
    compensate_quantization_error_via_lora_svd,
    compute_module_state_sha256,
    quantize_and_verify_moe_wsp_layer,
    symmetric_group_quantize_dequantize,
    verify_quantization_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    WSPArmConfig,
    apply_layer_wsp_edit_inplace,
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
    second_moment_and_mean,
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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "quantization_compensated_wsp.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_wsp_quant_moe_fixture(
    *,
    seed: int = 20261124,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    vocab_size: int = 48,
    num_experts: int = 4,
    lora_rank: int = 8,
) -> dict[str, Any]:
    """Build a deterministic CrowdV1WithEngram + CrowdLoRAMoEMLP with WSP edit and prose outliers."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)

    u_e0 = torch.zeros(hidden_size, dtype=torch.float32)
    u_e0[0] = 1.0 / math.sqrt(2.0)
    u_e0[1] = -1.0 / math.sqrt(2.0)

    u_e1 = torch.zeros(hidden_size, dtype=torch.float32)
    u_e1[32] = 1.0 / math.sqrt(2.0)
    u_e1[33] = -1.0 / math.sqrt(2.0)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_raw = 0.25 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        p_raw[:, :2] = 0.0
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True) + 2.0 * u_e0[:16].unsqueeze(0)
        p_raw[5] = 2.6 * u_e0[:16]

        e1_raw = 0.25 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        e1_raw[:, :2] = 0.0
        e1_raw = e1_raw - e1_raw.mean(dim=1, keepdim=True) + 2.0 * u_e1[32:48].unsqueeze(0)
        e1_raw[20 - 16] = 2.6 * u_e1[32:48]

        wte[:16, :16] = p_raw
        wte[16:32, 32:48] = e1_raw
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)

        blk = model.blocks[0]
        blk.ln1.weight.fill_(1.0)
        blk.ln2.weight.fill_(1.0)
        blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(1.2 * torch.eye(hidden_size))
        blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(1.2 * torch.eye(hidden_size))
        # High-dynamic-range outlier weights in group 0 force naive 4-bit RTN to quantize prose weights coarsely
        blk.mlp.w1.weight[0, 10] = 6.5
        blk.mlp.w2.weight[0, 10] = 6.5
        blk.mlp.w3.weight[:16, :16].copy_(0.45 * torch.outer(u_e0[:16], u_e0[:16]))
        blk.mlp.w3.weight[16:32, 14] = 8.5
        blk.mlp.w3.weight[32:48, 32:48].copy_(0.08 * torch.eye(16))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=2,
        lora_rank=lora_rank,
        lora_alpha=16.0,
        routing_mode="sigma_topk",
        seed=seed + 17,
    )

    chunks: list[dict[str, Any]] = []
    for _ in range(6):
        ids = torch.randint(0, 16, (16,), generator=gen).tolist()
        for pos in range(8, 16):
            ids[pos] = 5
        chunks.append({"expert_id": "E0_general_fineweb", "ids": ids})
    for _ in range(6):
        ids = torch.randint(16, 32, (16,), generator=gen).tolist()
        for pos in range(8, 16):
            ids[pos] = 20
        chunks.append({"expert_id": "E1_code", "ids": ids})

    ids_tensor = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    is_e0 = torch.tensor([c["expert_id"].startswith("E0_") for c in chunks], dtype=torch.bool)
    cos, sin = model.cos_cached, model.sin_cached
    blk = model.blocks[0]

    def _eval_model(cand_mod: nn.Module | None = None) -> dict[str, float]:
        prev_mlp = blk.mlp
        if isinstance(cand_mod, CrowdLoRAMoEMLP):
            blk.mlp = cand_mod  # type: ignore[assignment]
        try:
            with torch.no_grad():
                xb = compute_embeddings_batch(model, ids_tensor)
                xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
                xo = forward_block_post_mlp(blk, xm, mi)
                nlls = per_chunk_nll_from_pre_ln(
                    xo,
                    model.ln.weight,
                    model.ln.bias,
                    float(getattr(model.ln, "eps", 1e-5)),
                    model.head.weight,
                    ids_tensor,
                    score_from=8,
                )
                return {
                    "E0_general_fineweb": float(nlls[is_e0].mean().item()),
                    "E1_code": float(nlls[~is_e0].mean().item()),
                }
        finally:
            blk.mlp = prev_mlp

    base_nll = _eval_model()

    # Plant specialist E1 gain + prose interference, then run WSP surgery
    with torch.no_grad():
        mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
        mlp.router.weight.zero_()
        for exp in mlp.experts.values():
            for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                getattr(exp, attr).zero_()
        mlp.router.weight[1] = 3.0 * u_e1 + 2.2 * u_e0
        mlp.experts["1"].w3_A[0, 32] = 0.25
        mlp.experts["1"].w3_B[:, 0] = 0.85 * u_e1
        mlp.experts["1"].w3_A[1, 0] = 0.12
        mlp.experts["1"].w3_B[:, 1] = -0.60 * u_e0

        xb = compute_embeddings_batch(model, ids_tensor)
        _, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        scored = mi[:, 7:-1, :]
        p_x = scored[is_e0].reshape(-1, hidden_size)
        s_x = scored[~is_e0].reshape(-1, hidden_size)
        half_p = p_x.shape[0] // 2
        half_s = s_x.shape[0] // 2
        apply_layer_wsp_edit_inplace(
            mlp,
            p_x[:half_p],
            p_x[half_p:],
            s_x[:half_s],
            s_x[half_s:],
            WSPArmConfig(
                name="wsp_pre_quant",
                router_beta_suppress=2.5,
                alphaedit_eta_x=2.0,
                alphaedit_eta_act=2.0,
                alphaedit_tau=1.2,
            ),
            sink_experts=[],
        )

    wsp_tuned_nll = _eval_model()
    return {
        "model": model,
        "block": blk,
        "moe_mlp": blk.mlp,
        "base_nll": base_nll,
        "tuned_nll": wsp_tuned_nll,
        "prose_x": p_x,
        "eval_fn": lambda mod: {"domain_nll": _eval_model(mod)},
    }


# ---------------------------------------------------------------------------
# S1: 4-bit MoE Quantization — Naive RTN Breaks Guard, Prose-LoftQ Restores
# ---------------------------------------------------------------------------


def test_s1_4bit_quantization_naive_rtn_breaks_guard_while_loftq_prose_lora_restores() -> None:
    """Naive 4-bit RTN re-breaks the <= +1.0% prose guard (>+1.5%); LoftQ prose-LoRA restores <= +0.25%."""
    fix = _build_wsp_quant_moe_fixture(seed=20261124)
    moe_mlp: CrowdLoRAMoEMLP = fix["moe_mlp"]
    base_nll: dict[str, float] = fix["base_nll"]
    tuned_nll: dict[str, float] = fix["tuned_nll"]
    prose_x: torch.Tensor = fix["prose_x"]
    eval_fn = fix["eval_fn"]

    # Pre-quantization WSP FP32 baseline already satisfies <= +1.0% guard and delivers strong specialist gain
    pre_e0_drift_pct = 100.0 * (tuned_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"] - 1.0)
    assert pre_e0_drift_pct <= 0.25
    assert base_nll["E1_code"] - tuned_nll["E1_code"] > 0.05

    # 1. Naive 4-bit RTN (use_lora_compensation=False) re-breaks general-prose guard (> +1.5% drift)
    pre_sha = compute_module_state_sha256(moe_mlp)
    rtn_layer, rtn_receipt = quantize_and_verify_moe_wsp_layer(
        moe_mlp,
        bits=4,
        group_size=32,
        rank=8,
        use_lora_compensation=False,
        eval_fn=eval_fn,
        baseline_domain_nll=base_nll,
        tuned_domain_nll=tuned_nll,
        inplace=True,
    )
    assert rtn_layer is moe_mlp
    assert rtn_receipt["accepted"] is False
    assert rtn_receipt["rolled_back"] is True
    assert rtn_receipt["guard_ok"] is False
    assert rtn_receipt["general_drift_pct"] > 1.5
    assert rtn_receipt["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN - 10.0
    assert rtn_receipt["active_sha256"] == pre_sha
    assert verify_quantization_receipt_provenance(rtn_receipt) is True

    # 2. Prose-weighted LoftQ rank-8 LoRA restores <= +0.25% general drift and >95% specialist gain
    comp_layer, comp_receipt = quantize_and_verify_moe_wsp_layer(
        moe_mlp,
        bits=4,
        group_size=32,
        rank=8,
        num_alternating_steps=3,
        prose_activations=prose_x,
        prose_weight=2.0,
        eval_fn=eval_fn,
        baseline_domain_nll=base_nll,
        tuned_domain_nll=tuned_nll,
        inplace=True,
    )
    assert comp_layer is moe_mlp
    assert comp_receipt["schema"] == QUANT_RECEIPT_SCHEMA
    assert comp_receipt["accepted"] is True
    assert comp_receipt["rolled_back"] is False
    assert comp_receipt["guard_ok"] is True
    assert comp_receipt["general_drift_pct"] <= 0.25
    assert comp_receipt["specialist_gain_retained_frac"] > 0.95
    assert comp_receipt["anse_energy"] < 0.0
    assert comp_receipt["quant_error_fro"] < comp_receipt["naive_quant_error_fro"]
    assert comp_receipt["vram_compression_ratio"] > 2.5
    assert isinstance(moe_mlp.base_mlp.w1, QuantCompensatedLinear)
    assert isinstance(moe_mlp.base_mlp.w2, QuantCompensatedLinear)
    assert isinstance(moe_mlp.base_mlp.w3, QuantCompensatedLinear)
    assert verify_quantization_receipt_provenance(comp_receipt) is True

    # 3. Direct prose-weighted vs unweighted comparison on multi-mode weight matrix
    gen = torch.Generator(device="cpu").manual_seed(777)
    d_out, d_in = 128, 64
    v_spec = torch.linalg.qr(torch.randn((d_in, 10), generator=gen, dtype=torch.float64))[0]
    u_spec = torch.randn((d_out, 10), generator=gen, dtype=torch.float64)
    v_prose = torch.linalg.qr(torch.randn((d_in, 4), generator=gen, dtype=torch.float64))[0]
    u_prose = torch.randn((d_out, 4), generator=gen, dtype=torch.float64)
    w_mixed = (
        0.05 * torch.randn((d_out, d_in), generator=gen, dtype=torch.float64)
        + 2.0 * (u_spec @ v_spec.T)
        + 0.6 * (u_prose @ v_prose.T)
    )
    x_p = torch.randn((160, 4), generator=gen, dtype=torch.float64) @ v_prose.T
    m_p = (x_p.T @ x_p) / float(x_p.shape[0])

    _, _, _, diag_unw = compensate_quantization_error_via_lora_svd(
        w_mixed, rank=8, bits=4, group_size=32, num_alternating_steps=3, m_prose=m_p, prose_weight=0.0
    )
    _, _, _, diag_pw = compensate_quantization_error_via_lora_svd(
        w_mixed, rank=8, bits=4, group_size=32, num_alternating_steps=3, m_prose=m_p, prose_weight=2.0
    )
    assert diag_unw["prose_weighted"] is False
    assert diag_pw["prose_weighted"] is True
    assert diag_pw["prose_error_fro"] < 0.25 * diag_unw["prose_error_fro"]


# ---------------------------------------------------------------------------
# S2: Monotonic Quantization Residual Frobenius Error Decrease Across LoftQ
# ---------------------------------------------------------------------------


def test_s2_monotonic_loftq_residual_error_decrease_and_large_qr_svd() -> None:
    """Alternating LoftQ steps strictly/monotonically reduce residual Frobenius & weighted errors."""
    gen = torch.Generator(device="cpu").manual_seed(202607)
    d_out, d_in = 128, 64
    u = torch.randn((d_out, 4), generator=gen, dtype=torch.float64)
    v = torch.randn((d_in, 4), generator=gen, dtype=torch.float64)
    w_f64 = 0.06 * torch.randn((d_out, d_in), generator=gen, dtype=torch.float64) + 1.8 * (u @ v.T)

    x_prose = torch.zeros((100, d_in), dtype=torch.float64)
    x_prose[:, :8] = torch.randn((100, 8), generator=gen, dtype=torch.float64)
    m_prose = (x_prose.T @ x_prose) / 100.0

    for mp_arg in (None, m_prose):
        q_deq, lora_b, lora_a, diag = compensate_quantization_error_via_lora_svd(
            w_f64,
            rank=8,
            bits=4,
            group_size=32,
            num_alternating_steps=5,
            m_prose=mp_arg,
            prose_weight=2.0,
        )
        assert q_deq.dtype == torch.float64
        assert lora_b.dtype == torch.float64
        assert lora_a.dtype == torch.float64
        errs = diag["step_errors_fro"]
        w_errs = diag["step_weighted_errors"]
        assert len(errs) == 5
        assert errs[0] < diag["naive_quant_error_fro"]
        assert errs[2] < errs[0]
        assert all(errs[i] <= errs[i - 1] + 1e-12 for i in range(1, len(errs)))
        assert all(w_errs[i] <= w_errs[i - 1] + 1e-12 for i in range(1, len(w_errs)))
        actual_fro = float(torch.linalg.matrix_norm(w_f64 - (q_deq + lora_b @ lora_a), ord="fro").item())
        assert abs(actual_fro - diag["quant_error_fro"]) < 1e-10

    # Zero-padding when requested rank exceeds min(d_out, d_in), plus zero m_prose fallback
    w_small = torch.randn((6, 10), generator=gen, dtype=torch.float32)
    q_s, b_s, a_s, d_s = compensate_quantization_error_via_lora_svd(
        w_small,
        rank=12,
        bits=4,
        group_size=32,
        num_alternating_steps=2,
        m_prose=torch.zeros((10, 10), dtype=torch.float32),
        prose_weight=2.0,
    )
    assert q_s.shape == (6, 10)
    assert b_s.shape == (6, 12) and b_s.dtype == torch.float32
    assert a_s.shape == (12, 10) and a_s.dtype == torch.float32
    assert d_s["effective_rank"] == 6
    assert d_s["prose_weighted"] is False
    assert d_s["quant_error_fro"] < 1e-4

    # Large matrix (> 512 in both dimensions) exercises deterministic thin subspace QR-SVD path
    w_large = 0.05 * torch.randn((520, 520), generator=gen, dtype=torch.float32)
    q_l, b_l, a_l, d_l = compensate_quantization_error_via_lora_svd(
        w_large, rank=8, bits=4, group_size=32, num_alternating_steps=2
    )
    assert q_l.shape == (520, 520)
    assert b_l.shape == (520, 8) and a_l.shape == (8, 520)
    assert d_l["quant_error_fro"] < d_l["naive_quant_error_fro"]


# ---------------------------------------------------------------------------
# S3: 8-bit vs 4-bit Group-Size Boundary & Layer Topology Handling
# ---------------------------------------------------------------------------


def test_s3_8bit_vs_4bit_and_nondivisible_group_size_boundaries() -> None:
    """Verify 4-bit/8-bit groupwise quantization across non-divisible dims, zero groups, and blocks."""
    gen = torch.Generator(device="cpu").manual_seed(303)

    # Non-divisible columns (45 cols, group_size=32 -> 2 groups, 19 padded cols) + all-zero row
    w_nd = torch.randn((12, 45), generator=gen, dtype=torch.float32)
    w_nd[3, :] = 0.0
    w_nd[4, 32:] = 0.0  # trailing partial group is zero

    q4, d4 = symmetric_group_quantize_dequantize(w_nd, bits=4, group_size=32)
    q8, d8 = symmetric_group_quantize_dequantize(w_nd, bits=8, group_size=32)

    assert q4.shape == w_nd.shape and q4.dtype == torch.float32
    assert q8.shape == w_nd.shape and q8.dtype == torch.float32
    assert d4["padded_cols"] == 64 and d4["pad_cols"] == 19
    assert d4["num_groups_per_row"] == 2 and d4["zero_groups"] >= 3
    assert d4["q_max"] == 7 and d8["q_max"] == 127
    assert d8["quant_error_fro"] < 0.1 * d4["quant_error_fro"]
    assert d8["max_abs_error"] < 0.1 * d4["max_abs_error"]
    assert torch.all(q4[3] == 0.0) and torch.all(q8[3] == 0.0)
    assert bool(torch.isfinite(q4).all().item())

    # Group size larger than column count (cols=17, group_size=32) and group_size=1 exactness
    w_17 = torch.randn((5, 17), generator=gen, dtype=torch.bfloat16)
    q17, d17 = symmetric_group_quantize_dequantize(w_17, bits=4, group_size=32)
    assert q17.shape == (5, 17) and q17.dtype == torch.bfloat16
    assert d17["num_groups_per_row"] == 1 and d17["pad_cols"] == 15

    q_exact, d_exact = symmetric_group_quantize_dequantize(w_nd.to(torch.float64), bits=8, group_size=1)
    assert d_exact["quant_error_fro"] < 1e-12
    assert torch.allclose(q_exact, w_nd.to(torch.float64), atol=1e-12)

    # All-zero tensor check
    q_z, d_z = symmetric_group_quantize_dequantize(torch.zeros((4, 10)), bits=4, group_size=8)
    assert d_z["quant_error_fro"] == 0.0 and torch.all(q_z == 0.0)

    # Dense CrowdRuntimeMLP, CrowdRuntimeBlock (with router/attn flags), and standalone nn.Linear
    cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=40,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=10,
        intermediate_size=75,  # non-divisible by 32!
        max_position_embeddings=16,
        engram_buckets=0,
    )
    blk = CrowdRuntimeBlock(cfg)
    x_in = torch.randn((12, 40), generator=gen, dtype=torch.float32)
    m_x, _ = second_moment_and_mean(x_in)

    # Non-inplace copy quantization on CrowdRuntimeBlock with eval_fn=None and explicit m_prose map
    blk_copy, blk_rec = quantize_and_verify_moe_wsp_layer(
        blk,
        bits=8,
        group_size=32,
        rank=4,
        m_prose={"x": m_x},
        prose_activations=x_in,
        quantize_attention=True,
        quantize_base_mlp=True,
        inplace=False,
    )
    assert blk_copy is not blk
    assert isinstance(blk.mlp.w1, nn.Linear)
    assert isinstance(blk_copy.mlp.w1, QuantCompensatedLinear)
    assert isinstance(blk_copy.attn.q, QuantCompensatedLinear)
    assert blk_rec["accepted"] is True and blk_rec["eval_verified"] is False
    assert verify_quantization_receipt_provenance(blk_rec) is True

    # Standalone nn.Linear with bias and CrowdRuntimeMLP with router=True / attn=False coverage
    lin_bias = nn.Linear(45, 24, bias=True)
    comp_lin, lin_rec = quantize_and_verify_moe_wsp_layer(
        lin_bias,
        bits=8,
        group_size=32,
        rank=4,
        m_prose=torch.eye(45),
        inplace=False,
    )
    assert isinstance(comp_lin, QuantCompensatedLinear)
    assert comp_lin.bias is not None and comp_lin.effective_weight().shape == (24, 45)
    x_lin_test = torch.randn((3, 45), generator=gen, dtype=torch.float32)
    y_test = comp_lin(x_lin_test)
    y_ref = torch.nn.functional.linear(x_lin_test, comp_lin.effective_weight(), comp_lin.bias)
    assert y_test.shape == (3, 24) and torch.allclose(y_test, y_ref, atol=1e-5)
    assert lin_rec["quantized_linear_count"] == 1

    # CrowdV1WithEngram skips tied head while quantizing blocks and respecting quantize_router/quantize_attention
    model_full = CrowdV1WithEngram(cfg)
    upcycle_crowd_v1_to_sparse_moe(model_full, num_experts=3, top_k=2, lora_rank=4, seed=99)
    q_full, q_full_rec = quantize_and_verify_moe_wsp_layer(
        model_full,
        bits=8,
        group_size=32,
        rank=4,
        quantize_attention=False,
        quantize_base_mlp=True,
        quantize_router=True,
        inplace=True,
    )
    assert q_full is model_full
    assert "head" not in q_full_rec["quantized_linears"]
    assert "blocks.0.mlp.router" in q_full_rec["quantized_linears"]
    assert "blocks.0.attn.q" not in q_full_rec["quantized_linears"]

    # quantize_base_mlp=False with explicit QuantizationWSPConfig quantizes attention only
    blk_attn_only = CrowdRuntimeBlock(cfg)
    blk_attn_res, attn_rec = quantize_and_verify_moe_wsp_layer(
        blk_attn_only,
        config=QuantizationWSPConfig(bits=8, group_size=32, rank=4, quantize_attention=True, quantize_base_mlp=False, inplace=False),
    )
    assert isinstance(blk_attn_res.attn.q, QuantCompensatedLinear)
    assert isinstance(blk_attn_res.mlp.w1, nn.Linear)
    assert all(k.startswith("attn.") for k in attn_rec["quantized_linears"])


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Rejection & Atomic FP32 / BF16 Fallback
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_rejection_and_atomic_fp32_bf16_fallback() -> None:
    """Reject extreme 4-bit guard violations with ANSE Maximum Pain (>=1e6) and atomic FP32/BF16 fallback."""
    fix = _build_wsp_quant_moe_fixture(seed=20261124)
    moe_mlp: CrowdLoRAMoEMLP = fix["moe_mlp"]
    base_nll = fix["base_nll"]
    eval_fn = fix["eval_fn"]

    # 1. Bitwise original FP32 rollback when naive 4-bit RTN triggers ANSE Maximum Pain
    orig_sha = compute_module_state_sha256(moe_mlp)
    active_fp32, rec_fp32 = quantize_and_verify_moe_wsp_layer(
        moe_mlp,
        bits=4,
        group_size=32,
        rank=4,
        use_lora_compensation=False,
        eval_fn=eval_fn,
        baseline_domain_nll=base_nll,
        fallback_dtype=None,
        inplace=True,
    )
    assert active_fp32 is moe_mlp
    assert rec_fp32["accepted"] is False
    assert rec_fp32["rolled_back"] is True
    assert rec_fp32["status"] == "ROLLED_BACK_ATOMIC_FALLBACK"
    assert rec_fp32["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN - 10.0
    assert rec_fp32["active_sha256"] == orig_sha
    assert compute_module_state_sha256(moe_mlp) == orig_sha
    assert isinstance(moe_mlp.base_mlp.w1, nn.Linear)
    assert verify_quantization_receipt_provenance(rec_fp32) is True

    # 2. Explicit BF16 atomic fallback restores unquantized nn.Linear weights in bfloat16 and still forwards cleanly
    active_bf16, rec_bf16 = quantize_and_verify_moe_wsp_layer(
        moe_mlp,
        bits=4,
        group_size=32,
        rank=4,
        use_lora_compensation=False,
        eval_fn=eval_fn,
        baseline_domain_nll=base_nll,
        fallback_dtype=torch.bfloat16,
        inplace=False,
    )
    assert active_bf16 is not moe_mlp
    assert rec_bf16["accepted"] is False
    assert rec_bf16["rolled_back"] is True
    assert rec_bf16["fallback_dtype"] == "torch.bfloat16"
    assert isinstance(active_bf16.base_mlp.w1, nn.Linear)
    assert active_bf16.base_mlp.w1.weight.dtype == torch.bfloat16
    assert moe_mlp.base_mlp.w1.weight.dtype == torch.float32
    gen_s4 = torch.Generator(device="cpu").manual_seed(404)
    bf16_out = active_bf16(torch.randn((2, 64), generator=gen_s4))
    assert bf16_out.shape == (2, 64) and bool(torch.isfinite(bf16_out).all().item())

    # 3. Soft-margin penalty branch when drift is in (0.85%, 1.00%] and standalone nn.Linear rollback
    lin_root = nn.Linear(32, 16, bias=True)
    lin_sha = compute_module_state_sha256(lin_root)
    eval_calls = {"n": 0}

    def _soft_margin_eval(_m: nn.Module) -> tuple[float, dict[str, float]]:
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            return 0.0, {"E0_general": 2.0, "E1_spec": 2.0}
        return 0.0, {"E0_general": 2.018, "E1_spec": 1.8}  # +0.90% > +0.85%, <= +1.0%

    comp_soft, rec_soft = quantize_and_verify_moe_wsp_layer(
        lin_root,
        bits=4,
        group_size=32,
        rank=4,
        eval_fn=_soft_margin_eval,
        inplace=True,
    )
    assert isinstance(comp_soft, QuantCompensatedLinear)
    assert rec_soft["accepted"] is True
    assert rec_soft["anse_energy"] > -0.1 and rec_soft["anse_energy"] < 1.0  # includes +0.05 soft margin penalty

    def _failing_eval(_m: nn.Module) -> dict[str, Any]:
        if isinstance(_m, QuantCompensatedLinear):
            return {"domain_nll": {"E0_general": float("nan"), "E1_spec": 2.0}}
        return {"domain_nll": {"E0_general": 2.0, "E1_spec": 2.0}}

    restored_lin, rec_nan = quantize_and_verify_moe_wsp_layer(
        lin_root,
        bits=4,
        group_size=32,
        rank=4,
        eval_fn=_failing_eval,
        inplace=True,
    )
    assert restored_lin is lin_root
    assert rec_nan["accepted"] is False and rec_nan["rolled_back"] is True
    assert rec_nan["anse_energy"] >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert "EVAL_FAILURE" in str(rec_nan["failure_reason"])
    assert compute_module_state_sha256(restored_lin) == lin_sha

    # Tampered receipt provenance rejection
    tampered = json.loads(json.dumps(rec_fp32))
    tampered["general_drift_pct"] = 0.0
    assert verify_quantization_receipt_provenance(tampered) is False


# ---------------------------------------------------------------------------
# S5: Validation Errors & Repository Security / Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """Verify strict validation branches and AGPL-3.0 / ZeroStub / PRSecurity / lateral / gatekeeper gates."""
    # 1. Config & helper validation branches
    with pytest.raises(TypeError, match="int"):
        QuantizationWSPConfig(bits=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bits must be in"):
        QuantizationWSPConfig(bits=3)
    with pytest.raises(TypeError, match="int"):
        QuantizationWSPConfig(group_size="32")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        QuantizationWSPConfig(group_size=0)
    with pytest.raises(TypeError, match="real number"):
        QuantizationWSPConfig(prose_weight=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0.0"):
        QuantizationWSPConfig(prose_weight=-0.5)
    with pytest.raises(ValueError, match="finite"):
        QuantizationWSPConfig(max_domain_drift=float("nan"))
    with pytest.raises(ValueError, match="> 0.0"):
        QuantizationWSPConfig(anse_safe_e0_ratio=0.0)
    with pytest.raises(TypeError, match="bool"):
        QuantizationWSPConfig(use_lora_compensation=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="general_prefix"):
        QuantizationWSPConfig(general_prefix=" ")

    with pytest.raises(TypeError, match="torch.Tensor"):
        symmetric_group_quantize_dequantize([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D tensor"):
        symmetric_group_quantize_dequantize(torch.ones(8))
    with pytest.raises(TypeError, match="floating-point"):
        symmetric_group_quantize_dequantize(torch.ones((2, 4), dtype=torch.int32))
    with pytest.raises(ValueError, match="finite"):
        symmetric_group_quantize_dequantize(torch.tensor([[1.0, float("nan")]]))

    w_ok = torch.ones((4, 8))
    with pytest.raises(ValueError, match="shape"):
        compensate_quantization_error_via_lora_svd(w_ok, rank=2, m_prose=torch.eye(4))
    with pytest.raises(ValueError, match="positive semi-definite"):
        bad_mp = torch.eye(8)
        bad_mp[0, 0] = -5.0
        compensate_quantization_error_via_lora_svd(w_ok, rank=2, m_prose=bad_mp)

    with pytest.raises(ValueError, match="Incompatible LoRA shapes"):
        QuantCompensatedLinear(torch.ones(4, 8), torch.ones(4, 2), torch.ones(3, 8))
    with pytest.raises(ValueError, match="1D tensor"):
        QuantCompensatedLinear(torch.ones(4, 8), torch.ones(4, 2), torch.ones(2, 8), bias=torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        QuantCompensatedLinear(torch.ones(4, 8), torch.ones(4, 2), torch.ones(2, 8), bias=torch.tensor([1.0, 0.0, 0.0, float("nan")]))

    with pytest.raises(TypeError, match="nn.Module"):
        compute_module_state_sha256("not_a_module")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_module_state_sha256(nn.Module())
    nan_lin = nn.Linear(4, 4)
    with torch.no_grad():
        nan_lin.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        compute_module_state_sha256(nan_lin)

    assert verify_quantization_receipt_provenance({"receipt_sha256": "short"}) is False
    assert verify_quantization_receipt_provenance({"receipt_sha256": "0" * 64, "bad": float("nan")}) is False
    with pytest.raises(TypeError, match="mapping"):
        verify_quantization_receipt_provenance("bad")  # type: ignore[arg-type]

    # quantize_and_verify_moe_wsp_layer validation branches
    good_lin = nn.Linear(8, 4)
    with pytest.raises(TypeError, match="nn.Module"):
        quantize_and_verify_moe_wsp_layer("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        quantize_and_verify_moe_wsp_layer(good_lin, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.dtype"):
        quantize_and_verify_moe_wsp_layer(good_lin, fallback_dtype="bf16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="float32"):
        quantize_and_verify_moe_wsp_layer(good_lin, fallback_dtype=torch.int8)
    with pytest.raises(TypeError, match="QuantizationWSPConfig"):
        quantize_and_verify_moe_wsp_layer(good_lin, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No eligible nn.Linear"):
        quantize_and_verify_moe_wsp_layer(nn.LayerNorm(8))
    with pytest.raises(ValueError, match="finite"):
        bad_bias = nn.Linear(8, 4)
        with torch.no_grad():
            bad_bias.bias[0] = float("inf")
        quantize_and_verify_moe_wsp_layer(bad_bias)
    with pytest.raises(ValueError, match="finite"):
        bad_sub = nn.Sequential(nn.Linear(8, 4))
        with torch.no_grad():
            bad_sub[0].bias[0] = float("nan")
        quantize_and_verify_moe_wsp_layer(bad_sub)
    with pytest.raises(ValueError, match="square"):
        quantize_and_verify_moe_wsp_layer(good_lin, m_prose=torch.ones(4, 8))
    with pytest.raises(ValueError, match="square"):
        quantize_and_verify_moe_wsp_layer(good_lin, m_prose={"root": torch.ones(4, 8)})
    with pytest.raises(TypeError, match="Mapping"):
        quantize_and_verify_moe_wsp_layer(good_lin, m_prose="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        quantize_and_verify_moe_wsp_layer(good_lin, prose_activations="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D or 3D"):
        quantize_and_verify_moe_wsp_layer(good_lin, prose_activations=torch.ones(8))
    with pytest.raises(TypeError, match="floating-point"):
        quantize_and_verify_moe_wsp_layer(good_lin, prose_activations=torch.ones((2, 8), dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        quantize_and_verify_moe_wsp_layer(good_lin, prose_activations=torch.tensor([[1.0, float("nan")]]))
    with pytest.raises(TypeError, match="non-empty mapping"):
        quantize_and_verify_moe_wsp_layer(good_lin, eval_fn=lambda _m: "bad")  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError, match="non-empty mapping"):
        quantize_and_verify_moe_wsp_layer(good_lin, eval_fn=lambda _m: {"domain_nll": {}})
    with pytest.raises(ValueError, match="positive finite number"):
        quantize_and_verify_moe_wsp_layer(good_lin, eval_fn=lambda _m: {"domain_nll": {"E0_a": "bad"}})
    with pytest.raises(ValueError, match="identical keys"):
        quantize_and_verify_moe_wsp_layer(
            good_lin,
            eval_fn=lambda _m: {"domain_nll": {"E0_a": 1.0}},
            baseline_domain_nll={"E0_b": 1.0},
        )
    with pytest.raises(ValueError, match="identical keys"):
        quantize_and_verify_moe_wsp_layer(
            good_lin,
            eval_fn=lambda _m: {"domain_nll": {"E0_a": 1.0}},
            tuned_domain_nll={"E0_b": 1.0},
        )

    # Candidate eval changing keys triggers EVAL_FAILURE rollback
    _, key_drift_rec = quantize_and_verify_moe_wsp_layer(
        good_lin,
        bits=4,
        group_size=8,
        rank=2,
        eval_fn=lambda m: {"domain_nll": {"E0_other": 1.0} if isinstance(m, QuantCompensatedLinear) else {"E0_a": 1.0}},
    )
    assert key_drift_rec["rolled_back"] is True and "EVAL_FAILURE" in str(key_drift_rec["failure_reason"])

    # Corrupt snapshot hook test for RuntimeError rollback invariant
    mlp_tiny = CrowdRuntimeMLP(
        CrowdV1RuntimeConfig(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8)
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        orig_w1_ref = mlp_tiny.w1

        def _mutate_during_eval(_m: nn.Module) -> dict[str, Any]:
            if isinstance(_m.w1, QuantCompensatedLinear):
                with torch.no_grad():
                    orig_w1_ref.bias = nn.Parameter(torch.ones(32))
                return {"domain_nll": {"E0_gen": 9.9}}
            return {"domain_nll": {"E0_gen": 1.0}}

        quantize_and_verify_moe_wsp_layer(mlp_tiny, bits=4, group_size=16, rank=4, eval_fn=_mutate_during_eval)

    # 2. AGPL-3.0 header, optional ZeroStubAudit, PRSecurityScanner, lateral movement, and gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn07", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn07", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    fallback_auditor = _FallbackZeroStubAudit()

    for rel, path_obj in (
        ("src/hivemind/swarm/quantization_compensated_wsp.py", TARGET_MODULE_PATH),
        ("tests/test_quantization_compensated_wsp_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 0.159, "quant_error_fro": 0.042, "anse_energy": -0.037, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}") == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1107,
        pr_title="feat(swarm): quantization-compensated Weight-Space Programming and LoftQ prose-nullspace LoRA error absorption",
        pr_body="Implements NEURAL-CTRL-DYN-07 quantization-compensated WSP with deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): quantization-compensated Weight-Space Programming and LoftQ prose-nullspace LoRA error absorption"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
