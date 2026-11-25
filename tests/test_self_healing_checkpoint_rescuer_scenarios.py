# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-35: self-healing discarded checkpoint rescuer."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    ANSE_MAXIMUM_PAIN,
    RESCUER_RECEIPT_OBJECT,
    RESCUER_SCHEMA,
    CheckpointRescueConfig,
    apply_checkpoint_blocks_inplace,
    compute_checkpoint_sha256,
    execute_verified_checkpoint_rescue,
    extract_checkpoint_blocks,
    repair_drifted_blocks_in_nullspace,
    triage_drifted_checkpoint_blocks,
    verify_rescue_receipt_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result used when optional ``gwaya`` is absent."""

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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "self_healing_checkpoint_rescuer.py"
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


def _make_zero_mean_ortho_rows(n_rows: int, dim: int, seed: int) -> torch.Tensor:
    """Create ``n_rows`` orthonormal vectors in ``R^dim`` orthogonal to the all-ones vector."""
    assert n_rows < dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ones = torch.ones((dim, 1), dtype=torch.float64)
    mat = torch.cat(
        [ones, torch.randn((dim, dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    q_mat, _ = torch.linalg.qr(mat)
    return q_mat[:, 1 : 1 + n_rows].T.to(torch.float32).contiguous()


def _build_t3_style_moe_pair(
    *,
    seed: int = 20261235,
    hidden_size: int = 48,
    intermediate_size: int = 96,
    num_layers: int = 3,
    vocab_size: int = 64,
    num_experts: int = 4,
    lora_rank: int = 4,
) -> tuple[
    CrowdV1WithEngram,
    CrowdV1WithEngram,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, torch.Tensor],
]:
    """Build deterministic CPU Base and T3-style drifted MoE models with disjoint probe/eval splits.

    Geometry:
      - Tokens ``0..31`` (``E0_general_fineweb``): zero-mean subspace inside ``dims 0..23``.
      - Tokens ``32..63`` (``E6_stellar_spectroscopy``): zero-mean subspace inside ``dims 24..47``.
      - Layer 0 Expert 1: clean specialist anchor (strong E6 gain, zero E0 harm).
      - Layer 1 Expert 1 & Router: dual-active (strong E6 gain + E0 interference).
      - Layer 2 Expert 0 & Router: pure prose-corrupting sink (heavy E0 regression, zero E6 gain).
    """
    half_v = vocab_size // 2
    half_h = hidden_size // 2
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
        tie_word_embeddings=False,
    )
    base_model = CrowdV1WithEngram(cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # Zero-mean unit directions in prose (0..23) and astronomy specialist (24..47) subspaces
    p_basis = _make_zero_mean_ortho_rows(4, half_h, seed=seed + 11)
    s_basis = _make_zero_mean_ortho_rows(4, hidden_size - half_h, seed=seed + 23)
    u_p_in, u_p_out = p_basis[0], p_basis[1]
    u_s_in, u_s_out = s_basis[0], s_basis[1]

    with torch.no_grad():
        for p in base_model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_rows = _make_zero_mean_ortho_rows(16, half_h, seed=seed + 31)
        s_rows = _make_zero_mean_ortho_rows(16, hidden_size - half_h, seed=seed + 37)
        for i in range(half_v):
            row = 0.25 * p_rows[i % 16] + 0.55 * u_p_in
            wte[i, :half_h] = row - row.mean()
        for i in range(vocab_size - half_v):
            row = 0.25 * s_rows[i % 16] + 0.55 * u_s_in
            wte[half_v + i, half_h:] = row - row.mean()
        base_model.wte.weight.copy_(wte)

        head_w = 0.08 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)
        head_w[:, :half_h] = head_w[:, :half_h] - head_w[:, :half_h].mean(dim=1, keepdim=True)
        head_w[:, half_h:] = head_w[:, half_h:] - head_w[:, half_h:].mean(dim=1, keepdim=True)
        # Base knows general prose target token 3 moderately well, and astronomy token 40 weakly
        head_w[3, :half_h] = 0.25 * u_p_in + 0.35 * u_p_out
        head_w[40, half_h:] = 0.22 * u_s_in + 0.22 * u_s_out
        base_model.head.weight.copy_(head_w)

        base_model.ln.weight.fill_(1.0)
        base_model.ln.bias.zero_()
        for blk in base_model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(1.1 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(1.1 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.25 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        base_model,
        num_experts=num_experts,
        top_k=2,
        lora_rank=lora_rank,
        routing_mode="sigma_topk",
        seed=seed + 53,
    )
    with torch.no_grad():
        for blk in base_model.blocks:
            blk.mlp.router.weight.zero_()
            blk.mlp.router.weight[1, half_h:] = 2.5 * u_s_in
            blk.mlp.router.weight[0, :half_h] = 2.5 * u_p_in

    drifted_model = copy.deepcopy(base_model)
    with torch.no_grad():
        # Layer 0 Expert 1: Clean specialist anchor (purely inside astronomy subspace)
        exp01 = drifted_model.blocks[0].mlp.experts["1"]
        for p in exp01.parameters():
            p.zero_()
        exp01.w1_A[0, half_h:] = 1.4 * u_s_in
        exp01.w1_B[half_h:hidden_size, 0] = 0.40 * u_s_out
        exp01.w2_A[0, half_h:] = 1.4 * u_s_in
        exp01.w2_B[half_h:hidden_size, 0] = 0.40 * u_s_out
        exp01.w3_A[0, half_h:hidden_size] = 1.1 * u_s_out
        exp01.w3_B[half_h:, 0] = 0.35 * u_s_out
        drifted_model.blocks[0].mlp.router.weight[1, half_h:] += 1.5 * u_s_in

        # Layer 1 Expert 1: Salvageable dual-active block (strong E6 gain + E0 interference)
        exp11 = drifted_model.blocks[1].mlp.experts["1"]
        for p in exp11.parameters():
            p.zero_()
        # Mode 0: astronomy specialist feature
        exp11.w1_A[0, half_h:] = 1.4 * u_s_in
        exp11.w1_B[half_h:hidden_size, 0] = 0.40 * u_s_out
        exp11.w2_A[0, half_h:] = 1.4 * u_s_in
        exp11.w2_B[half_h:hidden_size, 0] = 0.40 * u_s_out
        exp11.w3_A[0, half_h:hidden_size] = 1.1 * u_s_out
        exp11.w3_B[half_h:, 0] = 0.35 * u_s_out
        # Mode 1: spurious general-prose interference inside prose subspace
        exp11.w1_A[1, :half_h] = 1.0 * u_p_in
        exp11.w1_B[:half_h, 1] = 0.27 * u_p_out
        exp11.w2_A[1, :half_h] = 1.0 * u_p_in
        exp11.w2_B[:half_h, 1] = 0.27 * u_p_out
        exp11.w3_A[1, :half_h] = 1.0 * u_p_out
        exp11.w3_B[:half_h, 1] = -0.58 * u_p_out - 0.33 * u_p_in
        drifted_model.blocks[1].mlp.router.weight[1, :half_h] += 3.0 * u_p_in
        drifted_model.blocks[1].mlp.router.weight[1, half_h:] += 1.5 * u_s_in

        # Layer 2 Expert 0 & Router: Pure prose-corrupting sink (heavy E0 drift, zero E6 gain)
        exp20 = drifted_model.blocks[2].mlp.experts["0"]
        for p in exp20.parameters():
            p.zero_()
        exp20.w1_A[0, :half_h] = 1.1 * u_p_in
        exp20.w1_B[:half_h, 0] = 0.27 * u_p_out
        exp20.w2_A[0, :half_h] = 1.1 * u_p_in
        exp20.w2_B[:half_h, 0] = 0.27 * u_p_out
        exp20.w3_A[0, :half_h] = 1.1 * u_p_out
        exp20.w3_B[:half_h, 0] = -0.66 * u_p_out - 0.37 * u_p_in
        drifted_model.blocks[2].mlp.router.weight[0, :half_h] += 2.0 * u_p_in

        # Mild final LN drift calibrated cleanly by fold_ln_lambda=0.8
        drifted_model.ln.bias[:half_h] -= 0.008 * u_p_out
        drifted_model.ln.bias[half_h:] += 0.020 * u_s_out

    # Disjoint probe (rows 0..15) and eval (rows 100..115) chunks
    def _make_chunks(offset: int, per_dom: int) -> list[dict[str, Any]]:
        c_gen = torch.Generator(device="cpu").manual_seed(seed + 500 + offset)
        rows: list[dict[str, Any]] = []
        for idx in range(per_dom):
            e0_ids = torch.randint(0, half_v, (12,), generator=c_gen, dtype=torch.long).tolist()
            for t in range(6, 12):
                e0_ids[t] = 3
            rows.append({"expert_id": "E0_general_fineweb", "row": offset + idx, "ids": e0_ids})

            e6_ids = torch.randint(half_v, vocab_size, (12,), generator=c_gen, dtype=torch.long).tolist()
            for t in range(6, 12):
                e6_ids[t] = 40
            rows.append({"expert_id": "E6_stellar_spectroscopy", "row": offset + idx, "ids": e6_ids})
        return rows

    probe_chunks = _make_chunks(0, 8)
    eval_chunks = _make_chunks(100, 12)
    geom = {
        "u_p_in": u_p_in,
        "u_p_out": u_p_out,
        "u_s_in": u_s_in,
        "u_s_out": u_s_out,
        "p_basis": p_basis,
        "s_basis": s_basis,
    }
    return base_model, drifted_model, probe_chunks, eval_chunks, geom


# ---------------------------------------------------------------------------
# S1: T3-Style Discarded MoE Checkpoint Rescue (+17.5% E0, -20% E6 -> <= +0.4% E0, >80% retained)
# ---------------------------------------------------------------------------


def test_s1_t3_style_discarded_moe_checkpoint_rescue() -> None:
    """Rescue a T3-style checkpoint with >+10% E0 drift & ~-20% specialist gain."""
    base_model, drifted_model, probe_chunks, eval_chunks, _ = _build_t3_style_moe_pair()
    cfg = CheckpointRescueConfig(
        score_from=6,
        batch_size=16,
        fold_ln_lambda=0.8,
        inplace=False,
    )

    rescued_model, receipt = execute_verified_checkpoint_rescue(
        base_model,
        drifted_model,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )

    assert receipt["object"] == RESCUER_RECEIPT_OBJECT
    assert receipt["schema"] == RESCUER_SCHEMA
    assert verify_rescue_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "RESCUED_COMMITTED"

    # Measure exact raw E0 & E6 pct vs base from deterministic eval_chunks
    from hivemind.swarm.wsp_alphaedit_control import forward_model_pre_ln, per_chunk_nll_from_pre_ln

    eval_ids = torch.tensor([c["ids"] for c in eval_chunks], dtype=torch.long)
    e0_mask = torch.tensor([c["expert_id"].startswith("E0_") for c in eval_chunks], dtype=torch.bool)

    def _domain_nll(mod: CrowdV1WithEngram) -> tuple[float, float]:
        with torch.no_grad():
            hp = forward_model_pre_ln(mod, eval_ids, batch_size=16)
            nlls = per_chunk_nll_from_pre_ln(
                hp, mod.ln.weight, mod.ln.bias, float(mod.ln.eps), mod.head.weight, eval_ids, score_from=6
            )
        return float(nlls[e0_mask].mean().item()), float(nlls[~e0_mask].mean().item())

    base_e0, base_e6 = _domain_nll(base_model)
    raw_e0, raw_e6 = _domain_nll(drifted_model)
    res_e0, res_e6 = _domain_nll(rescued_model)

    raw_e0_drift_pct = 100.0 * (raw_e0 / base_e0 - 1.0)
    raw_e6_gain_pct = 100.0 * (raw_e6 / base_e6 - 1.0)
    rescued_e0_drift_pct = 100.0 * (res_e0 / base_e0 - 1.0)
    rescued_e6_gain_pct = 100.0 * (res_e6 / base_e6 - 1.0)

    # Raw mimics T3: severe E0 prose regression (> +10%, ~ +17.5%) alongside ~ -20% specialist gain
    assert raw_e0_drift_pct >= 12.0
    assert raw_e6_gain_pct <= -15.0

    # Rescued clears tight <= +0.4% general drift while retaining > 80% of specialist gain
    assert rescued_e0_drift_pct <= 0.4
    assert receipt["general_drift_ratio"] <= 1.004
    assert receipt["specialist_retention_ratio"] > 0.80
    assert rescued_e6_gain_pct <= -12.0

    # ANSE Delta E < 0 strictly vs both base and raw drifted checkpoint
    assert receipt["delta_e_vs_base"] < 0.0
    assert receipt["delta_e_vs_raw"] < 0.0
    assert receipt["raw_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert receipt["anse_energy"] < receipt["base_anse_energy"]
    assert receipt["anse_energy"] < receipt["raw_anse_energy"]

    # Triage populated anchors, dual-active blocks, and pure corruptors
    assert "blocks.0.mlp.experts.1" in receipt["protected_specialist_anchors"]
    assert "blocks.1.mlp.experts.1" in receipt["salvageable_dual_active_blocks"]
    assert "blocks.2.mlp.experts.0" in receipt["prose_corrupting_blocks"]
    assert receipt["max_nullspace_leakage_norm"] < 1e-5


# ---------------------------------------------------------------------------
# S2: Bitwise Preservation of Clean Specialist Anchors & Exact P_0 Invariance
# ---------------------------------------------------------------------------


def test_s2_bitwise_anchor_preservation_and_exact_p0_invariance() -> None:
    """Verify clean anchors stay bitwise identical and dual-active deltas satisfy exact P_0 invariance."""
    gen = torch.Generator(device="cpu").manual_seed(20260002)
    dim = 16
    # Orthogonal split: dims 0..5 = prose subspace, dims 6..15 = specialist subspace
    prose_keys = torch.zeros((12, dim), dtype=torch.float64)
    prose_keys[:, :6] = torch.randn((12, 6), generator=gen, dtype=torch.float64)
    spec_vec = torch.zeros((dim, 1), dtype=torch.float64)
    spec_vec[6:] = torch.randn((10, 1), generator=gen, dtype=torch.float64)
    spec_vec = spec_vec / spec_vec.norm()

    _, _, vh_p = torch.linalg.svd(prose_keys, full_matrices=False)
    q_prose = vh_p[:6, :].T.contiguous()  # (16, 6) spanning dims 0..5 only
    p_null = torch.eye(dim, dtype=torch.float64) - q_prose @ q_prose.T

    # Build mixed dense + LoRA + non-zero-base LoRA + 1D + final LN blocks
    w_anchor_base = torch.randn((dim, dim), generator=gen, dtype=torch.float32)
    w_anchor_drift = w_anchor_base + 0.35 * torch.randn((dim, dim), generator=gen, dtype=torch.float32)

    w_dual_base = torch.randn((dim, dim), generator=gen, dtype=torch.float32)
    delta_prose_part = torch.randn((dim, dim), generator=gen, dtype=torch.float32) @ q_prose.float() @ q_prose.float().T
    delta_spec_part = torch.randn((dim, 1), generator=gen, dtype=torch.float32) @ spec_vec.T.float()
    w_dual_drift = w_dual_base + delta_prose_part + delta_spec_part

    w_corrupt_base = torch.randn((dim, dim), generator=gen, dtype=torch.float32)
    w_corrupt_drift = w_corrupt_base + 2.5 * torch.randn((dim, dim), generator=gen, dtype=torch.float32)

    # Continual non-zero-base LoRA block
    lora_b0 = torch.randn((dim, 4), generator=gen, dtype=torch.float64) * 0.2
    lora_a0 = torch.randn((4, dim), generator=gen, dtype=torch.float64) * 0.2
    lora_b1 = lora_b0 + 0.1 * torch.randn((dim, 4), generator=gen, dtype=torch.float64)
    lora_a1 = lora_a0 + 0.3 * torch.randn((4, dim), generator=gen, dtype=torch.float64)

    ln_w_base = torch.ones(dim, dtype=torch.float32)
    ln_w_drift = ln_w_base + 0.5 * torch.ones(dim, dtype=torch.float32)
    bias_base = torch.zeros(dim, dtype=torch.float64)
    bias_drift = (q_prose[:, 0] * 1.5 + spec_vec[:, 0] * 2.0).contiguous()

    base_blocks = {
        "blocks.0.mlp": w_anchor_base,
        "blocks.1.mlp": w_dual_base,
        "blocks.1.bias_vec": bias_base,
        "blocks.1.lora": {
            "w1_A": lora_a0,
            "w1_B": lora_b0,
            "w2_A": lora_a0.clone(),
            "w2_B": lora_b0.clone(),
            "w3_A": lora_a0.clone(),
            "w3_B": lora_b0.clone(),
        },
        "blocks.2.mlp": w_corrupt_base,
        "final_ln": {"weight": ln_w_base},
    }
    drift_blocks = {
        "blocks.0.mlp": w_anchor_drift,
        "blocks.1.mlp": w_dual_drift,
        "blocks.1.bias_vec": bias_drift,
        "blocks.1.lora": {
            "w1_A": lora_a1,
            "w1_B": lora_b1,
            "w2_A": lora_a1.clone(),
            "w2_B": lora_b1.clone(),
            "w3_A": lora_a1.clone(),
            "w3_B": lora_b1.clone(),
        },
        "blocks.2.mlp": w_corrupt_drift,
        "final_ln": {"weight": ln_w_drift},
    }

    harm_nats = {
        "blocks.0.mlp": -0.025,       # helpful general + clean specialist anchor
        "blocks.1.mlp": 0.045,        # dual-active
        "blocks.1.bias_vec": 0.020,   # dual-active 1D vector
        "blocks.1.lora": 0.035,       # dual-active non-zero base LoRA
        "blocks.2.mlp": 0.090,        # pure corruptor
    }
    gain_nats = {
        "blocks.0.mlp": 0.22,
        "blocks.1.mlp": 0.19,
        "blocks.1.bias_vec": 0.08,
        "blocks.1.lora": 0.15,
        "blocks.2.mlp": 0.001,
    }

    triage = triage_drifted_checkpoint_blocks(base_blocks, drift_blocks, harm_nats, gain_nats)
    assert triage["protected_specialist_anchors"] == ["blocks.0.mlp"]
    assert set(triage["salvageable_dual_active_blocks"]) == {
        "blocks.1.mlp",
        "blocks.1.bias_vec",
        "blocks.1.lora",
    }
    assert triage["prose_corrupting_blocks"] == ["blocks.2.mlp"]
    assert triage["layernorm_blocks"] == ["final_ln"]

    # Also pass an unnormalized projector (eigenvalues != {0, 1}) to test binarization path
    p_soft = 0.85 * p_null + 0.15 * (q_prose @ q_prose.T)
    repaired, rep_report = repair_drifted_blocks_in_nullspace(
        base_blocks,
        drift_blocks,
        triage,
        {"blocks.1": {"x": p_null, "act": p_soft}, "default": p_null},
        fold_ln_lambda=0.8,
    )

    # 1. Clean specialist anchor preserved bitwise
    assert torch.equal(repaired["blocks.0.mlp"], w_anchor_drift)
    # 2. Pure prose corruptor reverted bitwise to base
    assert torch.equal(repaired["blocks.2.mlp"], w_corrupt_base)
    # 3. Exact P_0 invariance on dual-active 2D & 1D deltas
    rep_delta_2d = (repaired["blocks.1.mlp"] - w_dual_base).to(torch.float64)
    prose_response = rep_delta_2d @ q_prose
    assert float(prose_response.abs().max().item()) < 1e-6
    spec_expected = delta_spec_part.to(torch.float64) @ spec_vec
    spec_actual = rep_delta_2d @ spec_vec
    assert float((spec_actual - spec_expected).abs().max().item()) < 1e-6

    rep_bias = repaired["blocks.1.bias_vec"] - bias_base
    assert float((q_prose.T @ rep_bias).abs().max().item()) < 1e-12
    assert abs(float((spec_vec[:, 0] @ rep_bias).item()) - 2.0) < 1e-12

    # 4. Non-zero-base LoRA keeps rank and zero prose delta leakage
    lora_rep = repaired["blocks.1.lora"]
    delta_lora = lora_rep["w1_B"] @ lora_rep["w1_A"] - lora_b0 @ lora_a0
    assert float((delta_lora @ q_prose).abs().max().item()) < 1e-11
    assert rep_report["max_nullspace_leakage_norm"] < 1e-5

    # 5. Final LayerNorm interpolation lambda = 0.8
    expected_ln = ln_w_base + 0.8 * (ln_w_drift - ln_w_base)
    assert torch.allclose(repaired["final_ln"]["weight"], expected_ln, atol=1e-6)

    # 6. Output-side projection path on LoRA w3 + clean-perp continual LoRA + dense output-side + zero projector fallback
    lora_b0_clean = lora_b0.clone()
    lora_a0_clean = lora_a0 @ p_null  # base has zero component in ker(P_0)
    lora_b0_out_clean = p_null @ lora_b0
    lora_a0_out_clean = lora_a0.clone()
    triage_extra_s2 = triage_drifted_checkpoint_blocks(
        {
            "blocks.0.lora_clean_in": {
                "w1_A": lora_a0_clean, "w1_B": lora_b0_clean,
                "w2_A": lora_a0_clean, "w2_B": lora_b0_clean,
                "w3_A": lora_a0_clean, "w3_B": lora_b0_clean,
            },
            "blocks.0.lora_clean_out": {
                "w1_A": lora_a0_out_clean, "w1_B": lora_b0_out_clean,
                "w2_A": lora_a0_out_clean, "w2_B": lora_b0_out_clean,
                "w3_A": lora_a0_out_clean, "w3_B": lora_b0_out_clean,
            },
            "blocks.0.lora_zero_b": {
                "w1_A": lora_a0, "w1_B": torch.zeros_like(lora_b0),
                "w2_A": lora_a0, "w2_B": torch.zeros_like(lora_b0),
                "w3_A": lora_a0, "w3_B": torch.zeros_like(lora_b0),
            },
            "blocks.0.dense_out": w_dual_base,
            "final_ln": {"weight": ln_w_base},
        },
        {
            "blocks.0.lora_clean_in": {
                "w1_A": lora_a1, "w1_B": lora_b1,
                "w2_A": lora_a1, "w2_B": lora_b1,
                "w3_A": lora_a1, "w3_B": lora_b1,
            },
            "blocks.0.lora_clean_out": {
                "w1_A": lora_a1, "w1_B": lora_b1,
                "w2_A": lora_a1, "w2_B": lora_b1,
                "w3_A": lora_a1, "w3_B": lora_b1,
            },
            "blocks.0.lora_zero_b": {
                "w1_A": lora_a1, "w1_B": lora_b1,
                "w2_A": lora_a1, "w2_B": lora_b1,
                "w3_A": lora_a1, "w3_B": lora_b1,
            },
            "blocks.0.dense_out": w_dual_drift,
            "final_ln": {"weight": ln_w_drift},
        },
        [0.04, 0.04, 0.04, 0.04],  # sequence lookup with final_ln omitted
        [0.15, 0.15, 0.15, 0.15],
    )
    rep_out, _ = repair_drifted_blocks_in_nullspace(
        {
            "blocks.0.lora_clean_in": {
                "w1_A": lora_a0_clean, "w1_B": lora_b0_clean,
                "w2_A": lora_a0_clean, "w2_B": lora_b0_clean,
                "w3_A": lora_a0_clean, "w3_B": lora_b0_clean,
            },
            "blocks.0.lora_clean_out": {
                "w1_A": lora_a0_out_clean, "w1_B": lora_b0_out_clean,
                "w2_A": lora_a0_out_clean, "w2_B": lora_b0_out_clean,
                "w3_A": lora_a0_out_clean, "w3_B": lora_b0_out_clean,
            },
            "blocks.0.lora_zero_b": {
                "w1_A": lora_a0, "w1_B": torch.zeros_like(lora_b0),
                "w2_A": lora_a0, "w2_B": torch.zeros_like(lora_b0),
                "w3_A": lora_a0, "w3_B": torch.zeros_like(lora_b0),
            },
            "blocks.0.dense_out": w_dual_base,
            "final_ln": {"weight": ln_w_base},
        },
        {
            "blocks.0.lora_clean_in": {
                "w1_A": lora_a1, "w1_B": lora_b1,
                "w2_A": lora_a1, "w2_B": lora_b1,
                "w3_A": lora_a1, "w3_B": lora_b1,
            },
            "blocks.0.lora_clean_out": {
                "w1_A": lora_a1, "w1_B": lora_b1,
                "w2_A": lora_a1, "w2_B": lora_b1,
                "w3_A": lora_a1, "w3_B": lora_b1,
            },
            "blocks.0.lora_zero_b": {
                "w1_A": lora_a1, "w1_B": lora_b1,
                "w2_A": lora_a1, "w2_B": lora_b1,
                "w3_A": lora_a1, "w3_B": lora_b1,
            },
            "blocks.0.dense_out": w_dual_drift,
            "final_ln": {"weight": ln_w_drift},
        },
        triage_extra_s2,
        {"blocks.0.lora_clean_in": p_null, "blocks.0.lora_zero_b": p_null, "default": p_null},
        projection_side="output",
    )
    delta_out = rep_out["blocks.0.lora_clean_out"]["w3_B"] @ rep_out["blocks.0.lora_clean_out"]["w3_A"] - lora_b0_out_clean @ lora_a0_out_clean
    assert float((q_prose.T @ delta_out).abs().max().item()) < 1e-11
    delta_zb_out = rep_out["blocks.0.lora_zero_b"]["w3_B"] @ rep_out["blocks.0.lora_zero_b"]["w3_A"]
    assert float((q_prose.T @ delta_zb_out).abs().max().item()) < 1e-11
    delta_dense_out = (rep_out["blocks.0.dense_out"] - w_dual_base).to(torch.float64)
    assert float((q_prose.T @ delta_dense_out).abs().max().item()) < 1e-6

    # Also check input-side on blocks.0.lora_clean_in and zero-eigenvalue fallback projector
    rep_in_clean, _ = repair_drifted_blocks_in_nullspace(
        {
            "blocks.0.lora_clean_in": {
                "w1_A": lora_a0_clean, "w1_B": lora_b0_clean,
                "w2_A": lora_a0_clean, "w2_B": lora_b0_clean,
                "w3_A": lora_a0_clean, "w3_B": lora_b0_clean,
            }
        },
        {
            "blocks.0.lora_clean_in": {
                "w1_A": lora_a1, "w1_B": lora_b1,
                "w2_A": lora_a1, "w2_B": lora_b1,
                "w3_A": lora_a1, "w3_B": lora_b1,
            }
        },
        triage_drifted_checkpoint_blocks(
            {
                "blocks.0.lora_clean_in": {
                    "w1_A": lora_a0_clean, "w1_B": lora_b0_clean,
                    "w2_A": lora_a0_clean, "w2_B": lora_b0_clean,
                    "w3_A": lora_a0_clean, "w3_B": lora_b0_clean,
                }
            },
            {
                "blocks.0.lora_clean_in": {
                    "w1_A": lora_a1, "w1_B": lora_b1,
                    "w2_A": lora_a1, "w2_B": lora_b1,
                    "w3_A": lora_a1, "w3_B": lora_b1,
                }
            },
            {"blocks.0.lora_clean_in": 0.04},
            {"blocks.0.lora_clean_in": 0.15},
        ),
        p_null,
        projection_side="input",
    )
    d_in_c = rep_in_clean["blocks.0.lora_clean_in"]["w1_B"] @ rep_in_clean["blocks.0.lora_clean_in"]["w1_A"] - lora_b0_clean @ lora_a0_clean
    assert float((d_in_c @ q_prose).abs().max().item()) < 1e-11

    rep_zero_p, _ = repair_drifted_blocks_in_nullspace(
        {"blocks.0.w": w_dual_base},
        {"blocks.0.w": w_dual_drift},
        triage_drifted_checkpoint_blocks({"blocks.0.w": w_dual_base}, {"blocks.0.w": w_dual_drift}, {"blocks.0.w": 0.05}, {"blocks.0.w": 0.15}),
        0.1 * torch.eye(dim, dtype=torch.float64),
        nullspace_eig_threshold=0.5,
    )
    assert torch.allclose(rep_zero_p["blocks.0.w"], w_dual_base)


# ---------------------------------------------------------------------------
# S3: Detection and NaN / Spectral-Explosion Block Quarantine
# ---------------------------------------------------------------------------


def test_s3_nan_and_spectral_explosion_block_quarantine() -> None:
    """Detect and quarantine NaN, Inf, bit-flip, 1D norm explosion, and spectral-spike blocks."""
    base_model, drifted_model, probe_chunks, eval_chunks, _ = _build_t3_style_moe_pair(seed=20260003)

    # Inject multiple corruptions across blocks:
    # 1. NaN in blocks.2.mlp.experts.2.w3_B
    # 2. +Inf in blocks.2.mlp.experts.3.w1_A
    # 3. Rank-1 spectral explosion in blocks.0.mlp.experts.2 (LoRA w1_B/A)
    with torch.no_grad():
        drifted_model.blocks[2].mlp.experts["2"].w3_B[0, 0] = float("nan")
        drifted_model.blocks[2].mlp.experts["3"].w1_A[0, 0] = float("inf")
        exp_boom = drifted_model.blocks[0].mlp.experts["2"]
        exp_boom.w1_A.zero_()
        exp_boom.w1_B.zero_()
        exp_boom.w1_A[0, 0] = 12.0
        exp_boom.w1_B[0, 0] = 12.0  # singular value 144 > single_mode_spike_norm=50
        # Also make a 2nd multi-mode spectral spike + bit-flip in synthetic dense dict check

    # Standalone triage checks all corruption classes without crashing
    base_blocks = extract_checkpoint_blocks(base_model)
    drift_blocks = extract_checkpoint_blocks(drifted_model)
    base_d = torch.eye(8, dtype=torch.float32)
    spike_d = base_d.clone()
    spike_d[0, 0] = 120.0  # spectral inflation > 20x & spike ratio > 25x
    bitflip_d = base_d.clone()
    bitflip_d[0, 0] = 600.0  # bitflip outlier
    extreme_d = base_d.clone()
    extreme_d[0, 0] = 2e4  # extreme magnitude
    vec_b = torch.ones(8, dtype=torch.float32)
    vec_boom = torch.full((8,), 80.0, dtype=torch.float32)

    # Non-zero base LoRA with multi-mode spike & inflation + nested NaN 2D tensor
    la0 = torch.eye(4, 8, dtype=torch.float32) * 0.2
    lb0 = torch.eye(8, 4, dtype=torch.float32) * 0.2
    la_boom = torch.eye(4, 8, dtype=torch.float32)
    lb_boom = torch.diag(torch.tensor([40.0, 0.5, 0.5, 0.5], dtype=torch.float32))
    lb_boom_full = torch.cat([lb_boom, torch.zeros((4, 4), dtype=torch.float32)], dim=0)
    nan_2d = torch.eye(4, dtype=torch.float32)
    nan_2d[0, 0] = float("nan")

    triage_extra = triage_drifted_checkpoint_blocks(
        {
            "b_spike": base_d,
            "b_bitflip": base_d,
            "b_extreme": base_d,
            "b_vec_boom": vec_b,
            "b_neg_spec": base_d,
            "b_lora_boom": {"w1_A": la0, "w1_B": lb0, "w2_A": la0, "w2_B": lb0, "w3_A": la0, "w3_B": lb0},
            "b_nan_nested": {"w": torch.eye(4, dtype=torch.float32)},
        },
        {
            "b_spike": spike_d,
            "b_bitflip": bitflip_d,
            "b_extreme": extreme_d,
            "b_vec_boom": vec_boom,
            "b_neg_spec": base_d + 0.1,
            "b_lora_boom": {"w1_A": la_boom, "w1_B": lb_boom_full, "w2_A": la0, "w2_B": lb0, "w3_A": la0, "w3_B": lb0},
            "b_nan_nested": {"w": nan_2d},
        },
        [0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0],  # sequence attribution lookup
        [0.1, 0.1, 0.1, 0.1, -0.05, 0.1, 0.1],  # b_neg_spec hurts specialist with no general benefit
    )
    assert set(triage_extra["quarantined_blocks"]) == {
        "b_spike",
        "b_bitflip",
        "b_extreme",
        "b_vec_boom",
        "b_lora_boom",
        "b_nan_nested",
    }
    assert triage_extra["prose_corrupting_blocks"] == ["b_neg_spec"]
    assert any("spectral" in r for r in triage_extra["blocks"]["b_spike"]["quarantine_reasons"])
    assert any("bitflip" in r for r in triage_extra["blocks"]["b_bitflip"]["quarantine_reasons"])
    assert any("extreme" in r for r in triage_extra["blocks"]["b_extreme"]["quarantine_reasons"])
    assert any("vector_norm" in r for r in triage_extra["blocks"]["b_vec_boom"]["quarantine_reasons"])
    assert any("lora_spectral" in r for r in triage_extra["blocks"]["b_lora_boom"]["quarantine_reasons"])
    assert any("non_finite" in r for r in triage_extra["blocks"]["b_nan_nested"]["quarantine_reasons"])

    # End-to-end rescue quarantines all 3 corrupted MoE blocks and rescues T3 cleanly
    rescued_model, receipt = execute_verified_checkpoint_rescue(
        base_model,
        drifted_model,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=CheckpointRescueConfig(score_from=6, batch_size=16, inplace=False),
    )

    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert set(receipt["quarantined_blocks"]) >= {
        "blocks.2.mlp.experts.2",
        "blocks.2.mlp.experts.3",
        "blocks.0.mlp.experts.2",
    }
    assert receipt["raw_anse_energy"] == ANSE_MAXIMUM_PAIN
    assert receipt["delta_e_vs_base"] < 0.0
    assert receipt["delta_e_vs_raw"] < 0.0
    assert receipt["general_drift_ratio"] <= 1.004
    assert receipt["specialist_retention_ratio"] > 0.80

    # Quarantined blocks in rescued_model are bitwise restored to base_model (zero NaN/Inf)
    rescued_blocks = extract_checkpoint_blocks(rescued_model)
    for q_bid in ("blocks.2.mlp.experts.2", "blocks.2.mlp.experts.3", "blocks.0.mlp.experts.2"):
        for sk, t_base in base_blocks[q_bid].items():
            assert torch.equal(rescued_blocks[q_bid][sk], t_base)
    assert len(drift_blocks) == len(rescued_blocks)


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback on Collapse or Guard Failure
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_and_provenance() -> None:
    """Verify atomic bitwise rollback and ANSE Maximum Pain on collapse, guard breach, or no-trigger."""
    base_model, drifted_model, probe_chunks, eval_chunks, geom = _build_t3_style_moe_pair(seed=20260004)
    base_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(base_model))
    raw_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(drifted_model))

    # 1. Specialist gain collapses (<70% retained) because dual-active features are inside ker(P_0)
    # Build a block dictionary + eval_fn where null-space projection destroys specialist gain
    b_dict = {"blocks.0.w": torch.eye(8, dtype=torch.float32)}
    d_dict = {"blocks.0.w": torch.eye(8, dtype=torch.float32) * 2.0}
    p_collapse = torch.zeros((8, 8), dtype=torch.float64)  # ker(P_0) = R^8 -> zero delta retained!

    def _collapse_eval(blocks: dict[str, torch.Tensor]) -> dict[str, float]:
        delta_norm = float((blocks["blocks.0.w"] - b_dict["blocks.0.w"]).abs().sum().item())
        if delta_norm > 6.0:  # raw_drifted
            return {"E0_general_fineweb": 2.40, "E6_stellar_spectroscopy": 1.60}
        return {"E0_general_fineweb": 2.00, "E6_stellar_spectroscopy": 2.00}

    rb_blocks, rec_collapse = execute_verified_checkpoint_rescue(
        b_dict,
        d_dict,
        eval_fn=_collapse_eval,
        layer_general_harm_nats={"blocks.0.w": 0.15},
        layer_specialist_gain_nats={"blocks.0.w": 0.40},
        p_null_by_layer=p_collapse,
    )
    assert rec_collapse["accepted"] is False
    assert rec_collapse["rolled_back"] is True
    assert rec_collapse["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_collapse["retention_ok"] is False
    assert rec_collapse["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_collapse["rollback_bitwise_verified"] is True
    assert compute_checkpoint_sha256(rb_blocks) == rec_collapse["base_state_sha256"]
    assert verify_rescue_receipt_provenance(rec_collapse) is True

    # 2. General guard failure after rescue (e.g., eval_fn shows +2.5% E0 drift on candidate)
    calls = {"idx": 0}

    def _guard_fail_eval(_mod: Any) -> dict[str, float]:
        calls["idx"] += 1
        if calls["idx"] == 1:  # base
            return {"E0_general_fineweb": 2.00, "E6_stellar_spectroscopy": 2.50}
        if calls["idx"] == 2:  # raw drifted (+20% E0 drift, -20% E6 gain)
            return {"E0_general_fineweb": 2.40, "E6_stellar_spectroscopy": 2.00}
        # candidate still +2.5% E0 drift (> +1.0% guard!)
        return {"E0_general_fineweb": 2.05, "E6_stellar_spectroscopy": 2.05}

    work_drifted = copy.deepcopy(drifted_model)
    _, rec_guard = execute_verified_checkpoint_rescue(
        base_model,
        work_drifted,
        probe_chunks=probe_chunks,
        eval_fn=_guard_fail_eval,
        config=CheckpointRescueConfig(score_from=6, batch_size=16, rollback_target="drifted", inplace=True),
    )
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_guard["general_guard_ok"] is False
    assert rec_guard["rollback_target"] == "drifted"
    assert rec_guard["final_state_sha256"] == raw_sha
    assert compute_checkpoint_sha256(extract_checkpoint_blocks(work_drifted)) == raw_sha

    # 3. Non-finite candidate eval_fn triggers ANSE Maximum Pain rollback to base
    nan_calls = {"idx": 0}

    def _nan_cand_eval(_mod: Any) -> dict[str, Any]:
        nan_calls["idx"] += 1
        if nan_calls["idx"] == 1:
            return {"E0_general_fineweb": 2.00, "E6_stellar_spectroscopy": 2.50}
        if nan_calls["idx"] == 2:
            return {"E0_general_fineweb": 2.40, "E6_stellar_spectroscopy": 2.00}
        return {"E0_general_fineweb": float("nan"), "E6_stellar_spectroscopy": 2.00}

    work_base_rb = copy.deepcopy(drifted_model)
    _, rec_nan = execute_verified_checkpoint_rescue(
        base_model,
        work_base_rb,
        probe_chunks=probe_chunks,
        eval_fn=_nan_cand_eval,
        config=CheckpointRescueConfig(score_from=6, batch_size=16, rollback_target="base", inplace=True),
    )
    assert rec_nan["accepted"] is False
    assert rec_nan["rolled_back"] is True
    assert rec_nan["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_nan["final_state_sha256"] == base_sha
    assert compute_checkpoint_sha256(extract_checkpoint_blocks(work_base_rb)) == base_sha

    # 4. Raw drift <= +10% trigger threshold does not commit rescue
    mild_calls = {"idx": 0}

    def _mild_drift_eval(_mod: Any) -> dict[str, float]:
        mild_calls["idx"] += 1
        if mild_calls["idx"] == 1:
            return {"E0_general_fineweb": 2.00, "E6_stellar_spectroscopy": 2.50}
        if mild_calls["idx"] == 2:
            return {"E0_general_fineweb": 2.10, "E6_stellar_spectroscopy": 2.00}  # +5% <= +10%
        return {"E0_general_fineweb": 2.002, "E6_stellar_spectroscopy": 2.02}

    _, rec_mild = execute_verified_checkpoint_rescue(
        base_model,
        drifted_model,
        probe_chunks=probe_chunks,
        eval_fn=_mild_drift_eval,
        config=CheckpointRescueConfig(score_from=6, batch_size=16, inplace=False),
    )
    assert rec_mild["accepted"] is False
    assert rec_mild["rolled_back"] is True
    assert rec_mild["status"] == "ROLLED_BACK_RAW_DRIFT_BELOW_TRIGGER"
    assert rec_mild["raw_drift_triggered"] is False

    # 5. Tampered receipt fails cryptographic provenance check
    assert verify_rescue_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_rescue_receipt_provenance({"receipt_sha256": "0" * 64, "object": "x", "schema": RESCUER_SCHEMA}) is False
    tampered = dict(rec_collapse)
    tampered["accepted"] = True
    assert verify_rescue_receipt_provenance(tampered) is False
    tampered_nan = dict(rec_collapse)
    tampered_nan["anse_energy"] = float("nan")
    assert verify_rescue_receipt_provenance(tampered_nan) is False
    assert "u_p_in" in geom and len(eval_chunks) > 0


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """Exercise input validation error paths and verify HiveMind security & zero-stub compliance."""
    # 1. Config & receipt validation errors
    with pytest.raises(TypeError, match="bool"):
        CheckpointRescueConfig(inplace="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        CheckpointRescueConfig(score_from=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        CheckpointRescueConfig(score_from=-1)
    with pytest.raises(ValueError, match=">= 1"):
        CheckpointRescueConfig(batch_size=0)
    with pytest.raises(TypeError, match="real number"):
        CheckpointRescueConfig(fold_ln_lambda="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        CheckpointRescueConfig(anchor_floor_nats=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        CheckpointRescueConfig(soft_margin_penalty=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        CheckpointRescueConfig(maximum_pain_energy=0.0)
    with pytest.raises(ValueError, match="> 0.0"):
        CheckpointRescueConfig(min_specialist_retention_ratio=0.0)
    with pytest.raises(ValueError, match="<= corruption_ceiling_nats"):
        CheckpointRescueConfig(anchor_floor_nats=0.02, corruption_ceiling_nats=0.005)
    with pytest.raises(ValueError, match="<= 1.0"):
        CheckpointRescueConfig(fold_ln_lambda=1.5)
    with pytest.raises(ValueError, match=">= 1.0"):
        CheckpointRescueConfig(max_general_drift_ratio=0.95)
    with pytest.raises(ValueError, match=">= 1.0"):
        CheckpointRescueConfig(anse_safe_general_ratio=0.99)
    with pytest.raises(ValueError, match="< 1.0"):
        CheckpointRescueConfig(nullspace_eig_threshold=1.0)
    with pytest.raises(ValueError, match="projection_side"):
        CheckpointRescueConfig(projection_side="bad")
    with pytest.raises(ValueError, match="rollback_target"):
        CheckpointRescueConfig(rollback_target="bad")
    with pytest.raises(ValueError, match="general_prefix"):
        CheckpointRescueConfig(general_prefix="  ")
    assert CheckpointRescueConfig(specialist_gain_floor_nats=0.05).specialist_gain_floor_nats == 0.05
    with pytest.raises(TypeError, match="mapping"):
        verify_rescue_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        compute_checkpoint_sha256(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        compute_checkpoint_sha256({})
    with pytest.raises(ValueError, match="empty"):
        compute_checkpoint_sha256(nn.ReLU())
    class _MixedDtypeModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("u8", torch.ones(2, dtype=torch.uint8))
            self.register_buffer("bf16", torch.ones(2, dtype=torch.bfloat16))
            self.register_buffer("f64", torch.ones(2, dtype=torch.float64))

    assert len(compute_checkpoint_sha256(_MixedDtypeModule())) == 64

    # 2. Block extraction / application / triage / repair validation errors
    with pytest.raises(TypeError):
        extract_checkpoint_blocks("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        extract_checkpoint_blocks(nn.Linear(4, 4))
    bad_blk_m = nn.Module()
    bad_blk_m.blocks = nn.ModuleList([nn.ReLU()])  # type: ignore[assignment]
    with pytest.raises(ValueError, match="supported MoE or MLP"):
        extract_checkpoint_blocks(bad_blk_m)
    with pytest.raises(TypeError):
        apply_checkpoint_blocks_inplace("bad", {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        apply_checkpoint_blocks_inplace(nn.Linear(4, 4), {})
    with pytest.raises(ValueError, match="no 'ln'"):
        apply_checkpoint_blocks_inplace(bad_blk_m, {"final_ln": {"weight": torch.ones(4)}})
    with pytest.raises(ValueError, match="out-of-range"):
        apply_checkpoint_blocks_inplace(bad_blk_m, {"blocks.9.mlp": {"w1": torch.ones(4, 4)}})
    with pytest.raises(ValueError, match="Unsupported block path"):
        apply_checkpoint_blocks_inplace(bad_blk_m, {"blocks.0.weird": {"w": torch.ones(4, 4)}})

    b_ok = {"blocks.0.w": torch.eye(4)}
    d_ok = {"blocks.0.w": torch.eye(4)}
    with pytest.raises(TypeError, match="non-empty Mapping"):
        triage_drifted_checkpoint_blocks([], d_ok, {"blocks.0.w": 0.0}, {"blocks.0.w": 0.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        triage_drifted_checkpoint_blocks({"": torch.eye(4)}, d_ok, {"": 0.0}, {"": 0.0})
    with pytest.raises(ValueError, match="floating-point"):
        triage_drifted_checkpoint_blocks({"b": torch.ones(4, dtype=torch.long)}, {"b": torch.ones(4, dtype=torch.long)}, {"b": 0.0}, {"b": 0.0})
    with pytest.raises(ValueError, match="sub-keys must be non-empty"):
        triage_drifted_checkpoint_blocks({"b": {"": torch.eye(2)}}, {"b": {"": torch.eye(2)}}, {"b": 0.0}, {"b": 0.0})
    with pytest.raises(ValueError, match="floating-point"):
        triage_drifted_checkpoint_blocks({"b": {"w": torch.ones(2, dtype=torch.long)}}, {"b": {"w": torch.ones(2, dtype=torch.long)}}, {"b": 0.0}, {"b": 0.0})
    with pytest.raises(ValueError, match="non-finite"):
        triage_drifted_checkpoint_blocks({"b": {"w": torch.tensor([float("nan"), 1.0])}}, {"b": {"w": torch.ones(2)}}, {"b": 0.0}, {"b": 0.0})
    with pytest.raises(TypeError, match="non-empty Mapping"):
        triage_drifted_checkpoint_blocks({"b": "bad"}, {"b": "bad"}, {"b": 0.0}, {"b": 0.0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="Sub-keys mismatch"):
        triage_drifted_checkpoint_blocks({"b": {"x": torch.eye(2)}}, {"b": {"y": torch.eye(2)}}, {"b": 0.0}, {"b": 0.0})
    with pytest.raises(ValueError, match="non-finite"):
        triage_drifted_checkpoint_blocks({"blocks.0.w": torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])}, {"blocks.0.w": torch.eye(2)}, {"blocks.0.w": 0.0}, {"blocks.0.w": 0.0})
    with pytest.raises(ValueError, match="anchor_floor_nats"):
        triage_drifted_checkpoint_blocks(b_ok, d_ok, {"blocks.0.w": 0.0}, {"blocks.0.w": 0.0}, anchor_floor_nats=0.05, corruption_ceiling_nats=0.01)
    with pytest.raises(ValueError, match="must match"):
        triage_drifted_checkpoint_blocks(b_ok, {"blocks.1.w": torch.eye(4)}, {"blocks.0.w": 0.0}, {"blocks.0.w": 0.0})
    with pytest.raises(ValueError, match="Shape mismatch"):
        triage_drifted_checkpoint_blocks(b_ok, {"blocks.0.w": torch.eye(3)}, {"blocks.0.w": 0.0}, {"blocks.0.w": 0.0})
    with pytest.raises(ValueError, match="missing attribution"):
        triage_drifted_checkpoint_blocks(b_ok, d_ok, {}, {"blocks.0.w": 0.0})
    with pytest.raises(TypeError, match="Mapping or Sequence"):
        triage_drifted_checkpoint_blocks(b_ok, d_ok, "bad", {"blocks.0.w": 0.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence length"):
        triage_drifted_checkpoint_blocks(b_ok, d_ok, [], {"blocks.0.w": 0.0})

    triage_dual = triage_drifted_checkpoint_blocks(b_ok, d_ok, {"blocks.0.w": 0.05}, {"blocks.0.w": 0.05})
    with pytest.raises(TypeError, match="triage_map"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, {}, torch.eye(4))
    with pytest.raises(ValueError, match="projection_side"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, torch.eye(4), projection_side="bad")
    with pytest.raises(ValueError, match="nullspace_eig_threshold"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, torch.eye(4), nullspace_eig_threshold=1.0)
    with pytest.raises(TypeError, match="2D Tensor or non-empty Mapping"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, {"blocks.0.w": "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="square 2D"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, torch.ones(2, 4))
    with pytest.raises(TypeError, match="floating-point"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, torch.ones((4, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, torch.tensor([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(TypeError, match="2D Tensor or non-empty Mapping"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, {})
    with pytest.raises(ValueError, match="No compatible projector"):
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, triage_dual, torch.eye(9))
    with pytest.raises(ValueError, match="No compatible projector"):
        repair_drifted_blocks_in_nullspace(
            {"v": torch.ones(4)},
            {"v": torch.ones(4)},
            triage_drifted_checkpoint_blocks({"v": torch.ones(4)}, {"v": torch.ones(4)}, {"v": 0.05}, {"v": 0.05}),
            torch.eye(9),
        )
    with pytest.raises(ValueError, match="missing entry"):
        bad_triage_miss = copy.deepcopy(triage_dual)
        bad_triage_miss["blocks"] = {}
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, bad_triage_miss, torch.eye(4))
    with pytest.raises(ValueError, match="Unsupported block classification"):
        bad_triage = copy.deepcopy(triage_dual)
        bad_triage["blocks"]["blocks.0.w"]["classification"] = "unknown_cls"
        repair_drifted_blocks_in_nullspace(b_ok, d_ok, bad_triage, torch.eye(4))

    # 3. Execute validation errors & dense / FineGrained / target_model / eval_fn edge paths
    with pytest.raises(TypeError, match="CheckpointRescueConfig"):
        execute_verified_checkpoint_rescue(b_ok, d_ok, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_verified_checkpoint_rescue(b_ok, d_ok, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_model"):
        execute_verified_checkpoint_rescue(b_ok, d_ok, target_model="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eval_fn"):
        execute_verified_checkpoint_rescue(b_ok, d_ok)
    with pytest.raises(ValueError, match="layer_general_harm_nats"):
        execute_verified_checkpoint_rescue(b_ok, d_ok, eval_fn=lambda _x: {"E0_a": 1.0, "E6_b": 1.0})
    with pytest.raises(ValueError, match="p_null_by_layer"):
        execute_verified_checkpoint_rescue(
            b_ok, d_ok, eval_fn=lambda _x: {"E0_a": 1.0, "E6_b": 1.0},
            layer_general_harm_nats={"blocks.0.w": 0.1}, layer_specialist_gain_nats={"blocks.0.w": 0.1},
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        execute_verified_checkpoint_rescue(
            b_ok, d_ok, eval_fn=lambda _x: {},
            layer_general_harm_nats={"blocks.0.w": 0.1}, layer_specialist_gain_nats={"blocks.0.w": 0.1}, p_null_by_layer=torch.eye(4),
        )
    with pytest.raises(TypeError, match="numeric"):
        execute_verified_checkpoint_rescue(
            b_ok, d_ok, eval_fn=lambda _x: {"E0_a": "bad", "E6_b": 1.0},  # type: ignore[dict-item]
            layer_general_harm_nats={"blocks.0.w": 0.1}, layer_specialist_gain_nats={"blocks.0.w": 0.1}, p_null_by_layer=torch.eye(4),
        )
    with pytest.raises(ValueError, match="general .* and specialist"):
        execute_verified_checkpoint_rescue(
            b_ok, d_ok, eval_fn=lambda _x: {"E0_a": 1.0},
            layer_general_harm_nats={"blocks.0.w": 0.1}, layer_specialist_gain_nats={"blocks.0.w": 0.1}, p_null_by_layer=torch.eye(4),
        )
    with pytest.raises(ValueError, match="non-finite evaluation NLL"):
        execute_verified_checkpoint_rescue(
            b_ok, d_ok, eval_fn=lambda _x: {"E0_a": 1.0, "E6_b": 1.0, "non_finite_outputs": True},
            layer_general_harm_nats={"blocks.0.w": 0.1}, layer_specialist_gain_nats={"blocks.0.w": 0.1}, p_null_by_layer=torch.eye(4),
        )
    # Test FineGrainedMoEMLP and dense CrowdRuntimeMLP extraction/application + error checks + auto rescue on dense MLP
    fg_cfg = CrowdV1RuntimeConfig(
        vocab_size=16, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, intermediate_size=16, max_position_embeddings=16,
        engram_buckets=0, tie_word_embeddings=True,
    )
    dense_m = CrowdV1WithEngram(fg_cfg)
    dense_m.blocks[0].mlp.w1.bias = None
    dense_blocks = extract_checkpoint_blocks(dense_m)
    apply_checkpoint_blocks_inplace(dense_m, dense_blocks)

    with pytest.raises(ValueError, match="eval_chunks"):
        execute_verified_checkpoint_rescue(
            dense_m, dense_m, eval_chunks=[{"expert_id": "E0_a", "ids": [0, 1]}],
        )

    with pytest.raises(ValueError, match="Shape/attribute mismatch"):
        apply_checkpoint_blocks_inplace(dense_m, {"final_ln": {"weight": torch.ones(3)}})
    with pytest.raises(ValueError, match="MLP weight mismatch"):
        apply_checkpoint_blocks_inplace(dense_m, {"blocks.0.mlp": {"w1": torch.ones(2, 2)}})
    with pytest.raises(ValueError, match="MoE experts"):
        apply_checkpoint_blocks_inplace(dense_m, {"blocks.0.mlp.experts.0": {"w1": torch.ones(2, 2)}})
    with pytest.raises(ValueError, match="probe_chunks"):
        execute_verified_checkpoint_rescue(dense_m, dense_m, probe_chunks=[])
    with pytest.raises(ValueError, match="mapping containing"):
        execute_verified_checkpoint_rescue(dense_m, dense_m, probe_chunks=[{"expert_id": "E0"}])
    with pytest.raises(ValueError, match="non-empty"):
        execute_verified_checkpoint_rescue(dense_m, dense_m, probe_chunks=[{"expert_id": "  ", "ids": [0, 1]}])
    with pytest.raises(ValueError, match="length >= 2"):
        execute_verified_checkpoint_rescue(dense_m, dense_m, probe_chunks=[{"expert_id": "E0", "ids": [0]}])
    with pytest.raises(ValueError, match="lie in"):
        execute_verified_checkpoint_rescue(dense_m, dense_m, probe_chunks=[{"expert_id": "E0", "ids": [99, 1]}])
    with pytest.raises(ValueError, match="sequence length"):
        execute_verified_checkpoint_rescue(
            dense_m, dense_m, probe_chunks=[{"expert_id": "E0", "ids": [0, 1]}, {"expert_id": "E6", "ids": [0, 1, 2]}]
        )
    with pytest.raises(ValueError, match="general .* and specialist"):
        execute_verified_checkpoint_rescue(dense_m, dense_m, probe_chunks=[{"expert_id": "E0", "ids": [0, 1, 2]}])
    with pytest.raises(ValueError, match="probe_chunks/eval_chunks or eval_fn"):
        execute_verified_checkpoint_rescue(dense_m, dense_m)

    # Dense CrowdRuntimeMLP attribution + target_model + keyword overrides check
    dense_drift = copy.deepcopy(dense_m)
    with torch.no_grad():
        dense_drift.blocks[0].mlp.w3.weight.add_(0.05)
    dense_probe = [
        {"expert_id": "E0_gen", "ids": [0, 1, 2, 3, 4, 5]},
        {"expert_id": "E6_spec", "ids": [8, 9, 10, 11, 12, 13]},
    ]
    _, dense_rec = execute_verified_checkpoint_rescue(
        dense_blocks,
        extract_checkpoint_blocks(dense_drift),
        target_model=dense_m,
        probe_chunks=dense_probe,
        config=CheckpointRescueConfig(score_from=2, min_raw_general_drift_ratio=5.0),
        fold_ln_lambda=0.8,
        min_raw_general_drift_ratio=5.0,
        max_general_drift_ratio=1.01,
        min_specialist_retention_ratio=0.70,
        anchor_floor_nats=-0.01,
        corruption_ceiling_nats=0.005,
        rollback_target="base",
        inplace=False,
    )
    assert dense_rec["rolled_back"] is True
    assert verify_rescue_receipt_provenance(dense_rec) is True

    # Fine-grained MoE + LoRA router/expert error checks
    fg_m = copy.deepcopy(dense_m)
    upcycle_crowd_v1_to_sparse_moe(fg_m, split_mode="fine_grained_width_split", num_experts=2, top_k=1)
    fg_blocks = extract_checkpoint_blocks(fg_m)
    apply_checkpoint_blocks_inplace(fg_m, fg_blocks)
    with pytest.raises(ValueError, match="Fine-grained expert mismatch"):
        apply_checkpoint_blocks_inplace(fg_m, {"blocks.0.mlp.experts.9": {"w1": torch.ones(8, 8)}})
    with pytest.raises(ValueError, match="Shape mismatch"):
        apply_checkpoint_blocks_inplace(fg_m, {"blocks.0.mlp.experts.0": {"w1": torch.ones(2, 2)}})

    lora_m = copy.deepcopy(dense_m)
    upcycle_crowd_v1_to_sparse_moe(lora_m, split_mode="lora_domain_experts", num_experts=2, top_k=1, lora_rank=2)
    lora_m.blocks[0].mlp.router.bias = nn.Parameter(torch.zeros(2))
    lora_blocks = extract_checkpoint_blocks(lora_m)
    apply_checkpoint_blocks_inplace(lora_m, lora_blocks)
    with pytest.raises(ValueError, match="Router attribute/shape mismatch"):
        apply_checkpoint_blocks_inplace(lora_m, {"blocks.0.mlp.router": {"weight": torch.ones(9, 9)}})
    with pytest.raises(ValueError, match="missing in layer"):
        apply_checkpoint_blocks_inplace(lora_m, {"blocks.0.mlp.experts.9": {"w1_A": torch.ones(2, 8)}})
    with pytest.raises(ValueError, match="LoRA expert mismatch"):
        apply_checkpoint_blocks_inplace(lora_m, {"blocks.0.mlp.experts.0": {"w1_A": torch.ones(9, 9)}})
    with pytest.raises(ValueError, match="non-finite"):
        lora_bad = copy.deepcopy(lora_m)
        with torch.no_grad():
            lora_bad.wte.weight[0, 0] = float("nan")
            lora_bad.blocks[0].mlp.experts["0"].w1_A[0, 0] = float("nan")
        execute_verified_checkpoint_rescue(lora_bad, lora_bad, probe_chunks=dense_probe)

    # 4. AGPL-3.0 header, ZeroStubAudit (CI-safe try/except), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn35", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn35", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/self_healing_checkpoint_rescuer.py", TARGET_MODULE_PATH),
        ("tests/test_self_healing_checkpoint_rescuer_scenarios.py", TEST_FILE_PATH),
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
            "nll_mean": 1.85,
            "general_drift_ratio": 1.003,
            "specialist_retention_ratio": 0.92,
            "anse_energy": -0.45,
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
        pr_number=1235,
        pr_title="feat(swarm): self-healing discarded checkpoint rescuer via causal layer attribution and null-space repair",
        pr_body="Implements NEURAL-CTRL-DYN-35 self-healing checkpoint rescuer with causal attribution, P_0 null-space projection, and ANSE rollback.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): self-healing discarded checkpoint rescuer via causal layer attribution and null-space repair"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
    assert math.isfinite(float(telemetry_json.count(".")))
