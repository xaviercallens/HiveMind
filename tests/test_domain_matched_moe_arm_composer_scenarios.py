# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-38: domain-matched multi-arm MoE composer & 2-matmul LN."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.domain_matched_moe_arm_composer import (
    ANSE_MAXIMUM_PAIN,
    COMPOSER_RECEIPT_OBJECT,
    COMPOSER_SCHEMA,
    MultiArmComposerConfig,
    compute_checkpoint_sha256,
    execute_verified_multi_arm_moe_composition,
    extract_checkpoint_blocks,
    fast_affine_layernorm_temperature_grid,
    select_domain_winning_experts_across_arms,
    verify_composer_receipt_provenance,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "domain_matched_moe_arm_composer.py"
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


def _install_domain_lora_delta(
    exp_mod: Any,
    in_dir: torch.Tensor,
    out_dir: torch.Tensor,
    *,
    strength: float,
    inter_slice: slice,
    seed: int,
) -> None:
    """Configure a ``CrowdLoRAExpertAdapter`` to emit ``strength * out_dir`` along ``in_dir``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rank = int(exp_mod.w1_A.shape[0])
    scaling = float(exp_mod.scaling)
    b_scale = math.sqrt(abs(strength)) / max(math.sqrt(scaling), 1e-6)
    sign_b = 1.0 if strength >= 0.0 else -1.0

    exp_mod.w1_A.zero_()
    exp_mod.w1_B.zero_()
    exp_mod.w2_A.zero_()
    exp_mod.w2_B.zero_()
    exp_mod.w3_A.zero_()
    exp_mod.w3_B.zero_()

    # Activate SwiGLU intermediate channels inside ``inter_slice`` when ``x @ in_dir > 0``
    i_start = inter_slice.start
    i_width = inter_slice.stop - inter_slice.start
    exp_mod.w1_A[0] = in_dir
    exp_mod.w2_A[0] = in_dir
    exp_mod.w1_B[i_start : i_start + min(rank, i_width), 0] = 2.5
    exp_mod.w2_B[i_start : i_start + min(rank, i_width), 0] = 2.5

    # Project activated intermediate slice onto ``out_dir``
    n_active = min(rank, i_width)
    exp_mod.w3_A[0, i_start : i_start + n_active] = 1.0 / float(n_active)
    exp_mod.w3_B[:, 0] = sign_b * b_scale * out_dir
    if rank > 1:
        exp_mod.w3_A[1] = 1e-4 * torch.randn(exp_mod.w3_A.shape[1], generator=gen)


def _build_c3_multi_domain_arm_pair(
    *,
    seed: int = 20261238,
    hidden_size: int = 48,
    intermediate_size: int = 96,
    num_layers: int = 2,
    vocab_size: int = 64,
    num_experts: int = 8,
    lora_rank: int = 4,
) -> tuple[
    CrowdV1WithEngram,
    CrowdV1WithEngram,
    CrowdV1WithEngram,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build deterministic C3-style Base, Arm A (code winner), and Arm B (math/stars winner) MoEs.

    Subspaces (4 pairwise orthogonal blocks of 12 hidden dims each):
      - E0_general_fineweb (tokens 0..15): dims 0..11
      - E1_python_systems_code (tokens 16..31): dims 12..23 (Arm A strong win)
      - E2_math_reasoning (tokens 32..47): dims 24..35 (Arm B strong win)
      - E6_stellar_spectroscopy (tokens 48..63): dims 36..47 (Arm B strong win)
    """
    block_h = hidden_size // 4
    block_v = vocab_size // 4
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

    domain_specs = (
        (0, "E0_general_fineweb", 0, 0, slice(0, block_h)),
        (1, "E1_python_systems_code", 1, block_v, slice(block_h, 2 * block_h)),
        (2, "E2_math_reasoning", 2, 2 * block_v, slice(2 * block_h, 3 * block_h)),
        (6, "E6_stellar_spectroscopy", 3, 3 * block_v, slice(3 * block_h, 4 * block_h)),
    )

    axes_in: dict[int, torch.Tensor] = {}
    axes_out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for p in base_model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        head_w = 0.04 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)

        for e_slot, _, b_idx, v_off, h_slice in domain_specs:
            basis = _make_zero_mean_ortho_rows(4, block_h, seed=seed + 100 + b_idx * 17)
            u_in, u_out = basis[0], basis[1]
            axes_in[e_slot] = u_in
            axes_out[e_slot] = u_out
            rows = _make_zero_mean_ortho_rows(8, block_h, seed=seed + 200 + b_idx * 29)
            for t_i in range(block_v):
                vec = 0.20 * rows[t_i % 8] + 0.65 * u_in
                wte[v_off + t_i, h_slice] = vec - vec.mean()
            head_w[:, h_slice] = head_w[:, h_slice] - head_w[:, h_slice].mean(dim=1, keepdim=True)
            for tok in (v_off + 2, v_off + 3, v_off + 4):
                head_w[tok, h_slice] = 0.12 * u_in + 0.55 * u_out

        base_model.wte.weight.copy_(wte)
        base_model.head.weight.copy_(head_w)
        base_model.ln.weight.fill_(1.0)
        base_model.ln.bias.zero_()
        for blk in base_model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.35 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.35 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.15 * torch.eye(hidden_size))

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
            for e_slot, _, _, _, h_slice in domain_specs:
                blk.mlp.router.weight[e_slot, h_slice] = 2.5 * axes_in[e_slot]

    arm_a = copy.deepcopy(base_model)
    arm_b = copy.deepcopy(base_model)

    with torch.no_grad():
        for l_idx in range(num_layers):
            # Arm A: deep specialist E1_code gain, plus slight E0 prose drift on expert_2 & LN
            exp1_a = arm_a.blocks[l_idx].mlp.experts["1"]
            full_in1 = torch.zeros(hidden_size, dtype=torch.float32)
            full_out1 = torch.zeros(hidden_size, dtype=torch.float32)
            full_in1[block_h : 2 * block_h] = axes_in[1]
            full_out1[block_h : 2 * block_h] = axes_out[1]
            _install_domain_lora_delta(
                exp1_a, full_in1, full_out1, strength=6.5, inter_slice=slice(0, 12), seed=seed + 10 + l_idx
            )
            # Arm A carries slight prose drift on expert_2 (weak on math) & LN
            exp2_a = arm_a.blocks[l_idx].mlp.experts["2"]
            prose_in = torch.zeros(hidden_size, dtype=torch.float32)
            prose_in[:block_h] = axes_in[0]
            prose_out = torch.zeros(hidden_size, dtype=torch.float32)
            prose_out[:block_h] = -axes_out[0]
            _install_domain_lora_delta(
                exp2_a, prose_in, prose_out, strength=1.2, inter_slice=slice(12, 24), seed=seed + 20 + l_idx
            )

            # Arm B: deep specialist E2_math & E6_stars gains, plus router/expert prose leakage
            for e_slot, h_slice, i_slice, str_val in (
                (2, slice(2 * block_h, 3 * block_h), slice(24, 36), 6.2),
                (6, slice(3 * block_h, 4 * block_h), slice(36, 48), 7.5),
            ):
                exp_b = arm_b.blocks[l_idx].mlp.experts[str(e_slot)]
                f_in = torch.zeros(hidden_size, dtype=torch.float32)
                f_out = torch.zeros(hidden_size, dtype=torch.float32)
                f_in[h_slice] = axes_in[e_slot]
                f_out[h_slice] = axes_out[e_slot]
                _install_domain_lora_delta(
                    exp_b, f_in, f_out, strength=str_val, inter_slice=i_slice, seed=seed + 30 + e_slot + l_idx
                )
                # Inject prose leakage into Arm B's expert_2 / expert_6 and router
                exp_b.w1_A[1, :block_h] = 0.8 * axes_in[0]
                exp_b.w1_B[:4, 1] = 1.5
                exp_b.w2_A[1, :block_h] = 0.8 * axes_in[0]
                exp_b.w2_B[:4, 1] = 1.5
                exp_b.w3_B[:block_h, 0] = -0.35 * axes_out[0]

            # Arm B's unfrozen router misfires on E0 prose
            arm_b.blocks[l_idx].mlp.router.weight[2, :block_h] = 2.8 * axes_in[0]
            arm_b.blocks[l_idx].mlp.router.weight[6, :block_h] = 2.6 * axes_in[0]

        # Arm A & B slightly perturb final LN
        arm_a.ln.weight.mul_(1.04)
        arm_b.ln.weight.mul_(1.12)

    def _make_chunks(seed_off: int, n_per_dom: int) -> list[dict[str, Any]]:
        c_gen = torch.Generator(device="cpu").manual_seed(seed + 500 + seed_off)
        chunks: list[dict[str, Any]] = []
        seq_len = 16
        for e_slot, dom_name, _, v_off, _ in domain_specs:
            target_pool = (v_off + 2, v_off + 3, v_off + 4)
            for r_i in range(n_per_dom):
                ids = torch.randint(v_off, v_off + block_v, (seq_len,), generator=c_gen).tolist()
                for pos in range(4, seq_len):
                    ids[pos] = target_pool[(pos + r_i + e_slot) % len(target_pool)]
                chunks.append({"expert_id": dom_name, "ids": ids})
        return chunks

    return base_model, arm_a, arm_b, _make_chunks(1, 6), _make_chunks(2, 6)


# ---------------------------------------------------------------------------
# S1: Synthetic C3 Arm A (Code) + Arm B (Math/Science) Composition
# ---------------------------------------------------------------------------


def test_s1_domain_matched_composition_beats_both_donor_arms_within_prose_guard() -> None:
    """S1: Domain-matched composition of C3 Arm A (code) + Arm B (math/stars) beats both arms at <=+0.3% E0."""
    base_model, arm_a, arm_b, probe_chunks, eval_chunks = _build_c3_multi_domain_arm_pair()
    base_sha = compute_checkpoint_sha256(base_model)

    cfg = MultiArmComposerConfig(
        score_from=4,
        batch_size=8,
        inplace=False,
    )
    composed_model, receipt = execute_verified_multi_arm_moe_composition(
        base_model,
        {"arm_a": arm_a, "arm_b": arm_b},
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )

    assert compute_checkpoint_sha256(base_model) == base_sha  # inplace=False preserved base
    assert receipt["object"] == COMPOSER_RECEIPT_OBJECT
    assert receipt["schema"] == COMPOSER_SCHEMA
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "MULTI_ARM_COMPOSITION_COMMITTED"
    assert receipt["cand_guard_ok"] is True
    assert receipt["beats_all_donors"] is True
    assert receipt["anse_energy"] < 0.0
    assert receipt["delta_e_vs_base"] < 0.0

    # Strictly beats both Arm A and Arm B overall NLL and ANSE energy
    arm_a_m = receipt["donor_arm_metrics"]["arm_a"]
    arm_b_m = receipt["donor_arm_metrics"]["arm_b"]
    assert receipt["nll_mean"] < arm_a_m["nll_mean"]
    assert receipt["nll_mean"] < arm_b_m["nll_mean"]
    assert receipt["anse_energy"] < arm_a_m["anse_energy"]
    assert receipt["anse_energy"] < arm_b_m["anse_energy"]
    assert receipt["delta_e_vs_donors"]["arm_a"] < 0.0
    assert receipt["delta_e_vs_donors"]["arm_b"] < 0.0

    # General prose drift strictly <= +0.3% (well under +1.0% guard)
    assert receipt["cand_general_drift_ratio"] is not None
    assert receipt["cand_general_drift_ratio"] <= 1.003

    # Verified P_0 projection eliminated prose leakage & 2-matmul LN calibration accelerated 25x
    assert receipt["selection_summary"]["max_nullspace_leak_before"] > 0.1
    assert receipt["selection_summary"]["max_nullspace_leak_after"] <= 1e-8
    assert receipt["ln_calibration_summary"]["executed"] is True
    assert receipt["matmul_speedup_factor"] >= 20.0
    assert verify_composer_receipt_provenance(receipt) is True

    # Verify composed model weights differ from base & match SHA-256 receipt
    composed_blocks = extract_checkpoint_blocks(composed_model)
    assert compute_checkpoint_sha256(composed_blocks) == receipt["final_state_sha256"]


# ---------------------------------------------------------------------------
# S2: Exact 2-Matmul Equivalence vs Naive 50-Matmul LayerNorm Calibration
# ---------------------------------------------------------------------------


def test_s2_fast_affine_layernorm_grid_matches_naive_50_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: 2-matmul affine LayerNorm search matches naive 50-matmul NLLs < 1e-5 and picks identical (lam, tau)."""
    gen = torch.Generator(device="cpu").manual_seed(20261239)
    batch_sz, seq_len, hidden_dim, vocab_size = 12, 10, 32, 24

    h_pre = 1.6 * torch.randn((batch_sz, seq_len, hidden_dim), generator=gen, dtype=torch.float32)
    base_w = 1.0 + 0.05 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    base_b = 0.03 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    cand_w = base_w * 1.18 + 0.04 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    cand_b = base_b - 0.08 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    head_w = 0.35 * torch.randn((vocab_size, hidden_dim), generator=gen, dtype=torch.float32)

    targets = torch.randint(0, vocab_size, (batch_sz, seq_len), generator=gen, dtype=torch.long)
    targets[0, -1] = -100  # exercise ignore_index handling
    is_general = torch.tensor([i < 4 for i in range(batch_sz)], dtype=torch.bool)
    domain_labels = [("E0_general" if i < 4 else ("E1_code" if i < 8 else "E2_math")) for i in range(batch_sz)]

    lam_grid = tuple(round(0.05 * i, 2) for i in range(11, 21))  # 10 points
    tau_grid = (0.96, 0.98, 1.00, 1.02, 1.04)  # 5 points => 50 grid evaluations
    assert len(lam_grid) * len(tau_grid) == 50

    call_counter = {"count": 0}
    orig_linear = F.linear

    def _counted_linear(inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        call_counter["count"] += 1
        return orig_linear(inp, weight, bias)

    monkeypatch.setattr(F, "linear", _counted_linear)
    fast_res = fast_affine_layernorm_temperature_grid(
        h_pre,
        base_w,
        base_b,
        cand_w,
        cand_b,
        head_w,
        targets,
        is_general,
        eps=1e-5,
        lambda_grid=lam_grid,
        tau_grid=tau_grid,
        max_general_drift_rel=0.006,
        seq_batch_size=12,
        domain_labels=domain_labels,
        max_domain_drift_rel=0.05,
    )
    monkeypatch.undo()

    assert call_counter["count"] == 2
    assert fast_res["head_matmuls_executed"] == 2
    assert fast_res["naive_head_matmuls"] == 50
    assert fast_res["matmul_speedup_factor"] == pytest.approx(25.0)

    # Naive 50-matmul ground truth evaluation in float32
    valid_m = targets != -100
    gen_m = is_general.unsqueeze(1).expand_as(targets) & valid_m
    base_logits_32 = orig_linear(
        F.layer_norm(h_pre, (hidden_dim,), base_w, base_b, eps=1e-5), head_w
    )
    base_tok_32 = F.cross_entropy(
        base_logits_32.reshape(-1, vocab_size)[valid_m.reshape(-1)],
        targets.reshape(-1)[valid_m.reshape(-1)],
        reduction="none",
    )
    ref_total = float(base_tok_32.mean().item())
    ref_gen = float(
        F.cross_entropy(
            base_logits_32.reshape(-1, vocab_size)[gen_m.reshape(-1)],
            targets.reshape(-1)[gen_m.reshape(-1)],
            reduction="mean",
        ).item()
    )

    naive_grid = np.empty((len(lam_grid), len(tau_grid)), dtype=np.float64)
    naive_feasible: list[tuple[float, float, float, float, float, float]] = []
    naive_fallback: list[tuple[float, float, float, float, float, float]] = []
    for i_l, lam in enumerate(lam_grid):
        w_l = base_w + lam * (cand_w - base_w)
        b_l = base_b + lam * (cand_b - base_b)
        raw_logits = orig_linear(F.layer_norm(h_pre, (hidden_dim,), w_l, b_l, eps=1e-5), head_w)
        for j_t, tau in enumerate(tau_grid):
            scaled = raw_logits / tau
            flat_s = scaled.reshape(-1, vocab_size)
            mean_nll = float(
                F.cross_entropy(flat_s[valid_m.reshape(-1)], targets.reshape(-1)[valid_m.reshape(-1)]).item()
            )
            gen_nll = float(
                F.cross_entropy(flat_s[gen_m.reshape(-1)], targets.reshape(-1)[gen_m.reshape(-1)]).item()
            )
            naive_grid[i_l, j_t] = mean_nll
            gen_drift = (gen_nll / ref_gen) - 1.0
            if gen_nll <= 1.006 * ref_gen + 1e-12:
                naive_feasible.append((mean_nll - ref_total, gen_drift, abs(tau - 1.0), abs(lam - 1.0), lam, tau))
            naive_fallback.append((gen_drift, mean_nll - ref_total, abs(tau - 1.0), abs(lam - 1.0), lam, tau))

    _, _, _, _, naive_best_lam, naive_best_tau = min(naive_feasible) if naive_feasible else min(naive_fallback)
    max_nll_diff = float(np.max(np.abs(np.asarray(fast_res["nll_grid"]) - naive_grid)))
    assert max_nll_diff < 1e-5
    assert fast_res["best_lambda_ln"] == naive_best_lam
    assert fast_res["best_tau_temp"] == naive_best_tau

    # Verify folded norm parameters produce identical logits to explicit / tau
    w_int = base_w + naive_best_lam * (cand_w - base_w)
    b_int = base_b + naive_best_lam * (cand_b - base_b)
    logits_folded = orig_linear(
        F.layer_norm(
            h_pre, (hidden_dim,), fast_res["folded_ln_weight"], fast_res["folded_ln_bias"], eps=1e-5
        ),
        head_w,
    )
    logits_explicit = orig_linear(F.layer_norm(h_pre, (hidden_dim,), w_int, b_int, eps=1e-5), head_w) / naive_best_tau
    assert float((logits_folded - logits_explicit).abs().max().item()) < 1e-5

    # Also cover biasless LayerNorm, zero-delta shortcut, 2D tensor, and fallback infeasible guard
    h_2d = h_pre.reshape(-1, hidden_dim)[:16]
    t_2d = targets.reshape(-1)[:16]
    t_2d[0] = 1
    g_2d = torch.tensor([i < 6 for i in range(16)], dtype=torch.bool)
    res_nodelta = fast_affine_layernorm_temperature_grid(
        h_2d,
        base_w,
        None,
        base_w.clone(),
        None,
        head_w,
        t_2d,
        g_2d,
        lambda_grid=(0.5, 1.0),
        tau_grid=(1.0, 1.05),
        max_general_drift_rel=0.0,
        seq_batch_size=8,
        base_general_nll_override=1e-4,  # forces fallback path
        base_total_nll_override=1.0,
    )
    assert res_nodelta["folded_ln_bias"] is None
    assert res_nodelta["guard_feasible"] is False
    assert res_nodelta["head_matmuls_executed"] == 2  # 1 matmul per batch across 2 batches


# ---------------------------------------------------------------------------
# S3: P_0 Null-Space Projection of Prose-Leaking Donor Experts
# ---------------------------------------------------------------------------


def test_s3_p_null_projection_eliminates_prose_leakage_across_layouts() -> None:
    """S3: Domain-matched expert selection assigns Arm A/B winners and eliminates I - P_0 leakage."""
    gen = torch.Generator(device="cpu").manual_seed(20261240)
    hidden_dim, inter_dim = 24, 48

    q_full, _ = torch.linalg.qr(torch.randn((hidden_dim, hidden_dim), generator=gen, dtype=torch.float64))
    u_prose = q_full[:, :4]
    u_spec = q_full[:, 4:8]
    p_null = (torch.eye(hidden_dim, dtype=torch.float64) - u_prose @ u_prose.T).to(torch.float32)
    # Perturb lightly to verify _binarize_projector restores exact idempotence
    p_noisy = p_null + 5e-4 * (u_prose @ u_prose.T).to(torch.float32)

    q_act, _ = torch.linalg.qr(torch.randn((inter_dim, inter_dim), generator=gen, dtype=torch.float64))
    u_act_p = q_act[:, :6]
    p_act = (torch.eye(inter_dim, dtype=torch.float64) - u_act_p @ u_act_p.T).to(torch.float32)

    # 1. Flat 2D dense deltas + LoRA expert dict
    spec_d1 = (u_spec[:, :2] @ torch.tensor([[1.5, -0.5], [0.8, 1.2]], dtype=torch.float64) @ u_spec[:, :2].T).to(torch.float32)
    leak_d1 = (u_prose[:, :2] @ torch.tensor([[3.0, 1.0], [-1.0, 2.0]], dtype=torch.float64) @ u_prose[:, :2].T).to(torch.float32)
    spec_d2 = (u_spec[:, 2:4] @ torch.tensor([[2.1, 0.3], [-0.4, 1.9]], dtype=torch.float64) @ u_spec[:, 2:4].T).to(torch.float32)
    leak_d2 = (u_prose[:, 2:4] @ torch.tensor([[2.4, -0.9], [1.1, 2.7]], dtype=torch.float64) @ u_prose[:, 2:4].T).to(torch.float32)

    arms_dense = {
        "arm_a": {
            0: torch.zeros((hidden_dim, hidden_dim)),
            1: spec_d1 + leak_d1,  # Arm A wins E1
            2: 0.1 * spec_d2,
            3: torch.ones((hidden_dim, hidden_dim)),  # sink slot
            4: leak_d1.clone(),  # protected slot
            5: 0.05 * spec_d1,  # unhelpful slot
        },
        "arm_b": {
            0: 0.01 * spec_d1,
            1: 0.1 * spec_d1,
            2: spec_d2 + leak_d2,  # Arm B wins E2
            3: torch.ones((hidden_dim, hidden_dim)),
            4: 2.0 * leak_d1,
            5: 0.05 * spec_d1,
        },
    }
    probe_losses = {
        "base": {"E0_general": 3.00, "E1_code": 3.50, "E2_math": 4.00, "E4_prot": 3.10, "E5_other": 3.20},
        "arm_a": {"E0_general": 3.01, "E1_code": 2.90, "E2_math": 3.95, "E4_prot": 2.80, "E5_other": 3.19},
        "arm_b": {"E0_general": 3.02, "E1_code": 3.40, "E2_math": 3.20, "E4_prot": 3.05, "E5_other": 3.22},
    }

    composed_dense, rep_dense = select_domain_winning_experts_across_arms(
        arms_dense,
        probe_losses,
        p_noisy,
        sink_experts={3},
        protected_experts={4},
        min_domain_gain_nats=0.05,
        revert_harmful_to_zero=True,
    )
    assert rep_dense["expert_assignments"]["expert_1"]["winning_arm"] == "arm_a"
    assert rep_dense["expert_assignments"]["expert_2"]["winning_arm"] == "arm_b"
    assert rep_dense["expert_assignments"]["expert_3"]["action"] == "ghost_sink_pruned"
    assert rep_dense["expert_assignments"]["expert_5"]["action"] == "reverted_unhelpful_slot"
    assert torch.equal(composed_dense[4], arms_dense["arm_a"][4])  # protected bitwise
    assert torch.count_nonzero(composed_dense[3]).item() == 0
    assert torch.count_nonzero(composed_dense[5]).item() == 0

    # Verify zero prose leakage and exact specialist preservation
    x_prose = u_prose[:, 0].to(torch.float32)
    x_spec = u_spec[:, 0].to(torch.float32)
    assert float(torch.linalg.vector_norm(composed_dense[1] @ x_prose).item()) <= 1e-5
    assert float(torch.linalg.vector_norm(composed_dense[2] @ x_prose).item()) <= 1e-5
    assert float(torch.linalg.vector_norm(composed_dense[1] @ x_spec - spec_d1 @ x_spec).item()) <= 1e-5
    assert rep_dense["max_nullspace_leak_before"] > 1.0
    assert rep_dense["max_nullspace_leak_after"] <= 1e-10

    # 2. Nested SwiGLU layout with per-layer {x, act} projectors & nonzero LoRA base
    def _make_swiglu(scale_p: float, scale_s: float) -> dict[str, torch.Tensor]:
        w1 = (scale_p * torch.outer(torch.ones(inter_dim), u_prose[:, 0].float()) + scale_s * torch.outer(torch.ones(inter_dim), u_spec[:, 0].float()))
        w2 = w1.clone()
        w3 = (scale_p * torch.outer(u_prose[:, 1].float(), u_act_p[:, 0].float()) + scale_s * torch.outer(u_spec[:, 1].float(), q_act[:, 10].float()))
        return {"w1": w1, "w2": w2, "w3": w3}

    arms_nested = {
        "arm_a": {"blocks.0": {1: _make_swiglu(1.2, 0.9), 2: _make_swiglu(0.5, 0.1)}},
        "arm_b": {"blocks.0": {1: _make_swiglu(0.4, 0.1), 2: _make_swiglu(1.8, 1.3)}},
    }
    composed_nested, rep_nested = select_domain_winning_experts_across_arms(
        arms_nested,
        {
            "arm_a": {"E1_code": {"base_nll": 3.5, "moe_nll": 2.9}, "E2_math": {"base_nll": 4.0, "moe_nll": 3.9}},
            "arm_b": {"E1_code": {"base_nll": 3.5, "moe_nll": 3.4}, "E2_math": {"base_nll": 4.0, "moe_nll": 3.1}},
        },
        {0: {"x": p_null, "act": p_act}},
        expert_domain_map={1: "E1_code", 2: "E2_math"},
    )
    assert rep_nested["expert_assignments"]["expert_1"]["winning_arm"] == "arm_a"
    assert rep_nested["expert_assignments"]["expert_2"]["winning_arm"] == "arm_b"
    assert float((composed_nested["blocks.0"][1]["w1"] @ u_prose[:, 0].float()).abs().max().item()) <= 1e-5
    assert float((composed_nested["blocks.0"][2]["w3"] @ u_act_p[:, 0].float()).abs().max().item()) <= 1e-5

    # 3. Execute dict+eval_fn composition with explicit base_experts & nonzero LoRA base
    lora_base = {
        "blocks.0.mlp.router": {"weight": torch.zeros((2, hidden_dim))},
        "blocks.0.mlp.experts.0": {
            "w1_A": torch.randn((2, hidden_dim), generator=gen) * 0.1,
            "w1_B": torch.randn((inter_dim, 2), generator=gen) * 0.1,
            "w2_A": torch.randn((2, hidden_dim), generator=gen) * 0.1,
            "w2_B": torch.randn((inter_dim, 2), generator=gen) * 0.1,
            "w3_A": torch.randn((2, inter_dim), generator=gen) * 0.1,
            "w3_B": torch.randn((hidden_dim, 2), generator=gen) * 0.1,
        },
        "blocks.0.mlp.experts.1": {
            "w1_A": torch.randn((2, hidden_dim), generator=gen) * 0.1,
            "w1_B": torch.randn((inter_dim, 2), generator=gen) * 0.1,
            "w2_A": torch.randn((2, hidden_dim), generator=gen) * 0.1,
            "w2_B": torch.randn((inter_dim, 2), generator=gen) * 0.1,
            "w3_A": torch.randn((2, inter_dim), generator=gen) * 0.1,
            "w3_B": torch.randn((hidden_dim, 2), generator=gen) * 0.1,
        },
        "final_ln": {"weight": torch.ones(hidden_dim), "bias": torch.zeros(hidden_dim)},
    }
    arm_1 = copy.deepcopy(lora_base)
    arm_2 = copy.deepcopy(lora_base)
    p_null_cpu = p_null.to(torch.float64)
    # Keep w1_A_base in P_0 for clean rank-r LoRA delta + inject prose leakage in arm_1/arm_2
    for e_k in ("blocks.0.mlp.experts.0", "blocks.0.mlp.experts.1"):
        lora_base[e_k]["w1_A"].copy_(lora_base[e_k]["w1_A"] @ p_null.to(torch.float32))
        lora_base[e_k]["w3_B"].copy_(p_null.to(torch.float32) @ lora_base[e_k]["w3_B"])
    arm_1 = copy.deepcopy(lora_base)
    arm_2 = copy.deepcopy(lora_base)
    arm_1["blocks.0.mlp.experts.0"]["w1_B"].add_(0.25)
    arm_1["blocks.0.mlp.experts.0"]["w1_A"].add_(0.20 * u_spec[:, :2].T.to(torch.float32) + 0.30 * u_prose[:, :2].T.to(torch.float32))
    arm_2["blocks.0.mlp.experts.1"]["w1_B"].add_(0.25)
    arm_2["blocks.0.mlp.experts.1"]["w1_A"].add_(0.20 * u_spec[:, 2:4].T.to(torch.float32) + 0.30 * u_prose[:, 2:4].T.to(torch.float32))

    # Also test nonzero-base LoRA input/output projection when base has prose component
    nz_base = {0: copy.deepcopy(lora_base["blocks.0.mlp.experts.0"]), 1: copy.deepcopy(lora_base["blocks.0.mlp.experts.1"])}
    nz_base[0]["w1_A"].add_(0.05 * u_prose[:, :2].T.to(torch.float32))
    nz_base[0]["w3_B"].add_(0.05 * u_prose[:, :2].to(torch.float32))
    nz_arm1 = {0: copy.deepcopy(nz_base[0]), 1: copy.deepcopy(nz_base[1])}
    nz_arm2 = {0: copy.deepcopy(nz_base[0]), 1: copy.deepcopy(nz_base[1])}
    nz_arm1[0]["w1_A"].add_(0.2 * u_spec[:, :2].T.to(torch.float32) + 0.2 * u_prose[:, :2].T.to(torch.float32))
    nz_arm1[0]["w3_B"].add_(0.2 * u_spec[:, :2].to(torch.float32) + 0.2 * u_prose[:, :2].to(torch.float32))
    nz_comp, nz_rep = select_domain_winning_experts_across_arms(
        {"a": nz_arm1, "b": nz_arm2},
        {"a": {"E0": 2.9, "E1": 3.5}, "b": {"E0": 3.1, "E1": 3.1}},
        p_null,
        base_experts=nz_base,
    )
    assert nz_rep["max_nullspace_leak_after"] <= 1e-8
    assert nz_comp[0]["w1_A"].shape == nz_base[0]["w1_A"].shape

    base_w0 = (lora_base["blocks.0.mlp.experts.0"]["w1_B"].double() @ lora_base["blocks.0.mlp.experts.0"]["w1_A"].double()) @ p_null_cpu
    base_w1 = (lora_base["blocks.0.mlp.experts.1"]["w1_B"].double() @ lora_base["blocks.0.mlp.experts.1"]["w1_A"].double()) @ p_null_cpu

    def _dict_eval(blocks: Mapping[str, Any]) -> dict[str, float]:
        cur_w0 = (blocks["blocks.0.mlp.experts.0"]["w1_B"].double() @ blocks["blocks.0.mlp.experts.0"]["w1_A"].double()) @ p_null_cpu
        cur_w1 = (blocks["blocks.0.mlp.experts.1"]["w1_B"].double() @ blocks["blocks.0.mlp.experts.1"]["w1_A"].double()) @ p_null_cpu
        e0_delta = float(torch.linalg.matrix_norm(cur_w0 - base_w0).item())
        e1_delta = float(torch.linalg.matrix_norm(cur_w1 - base_w1).item())
        return {
            "E0_general": 3.00,
            "E1_code": 3.50 - 0.10 * min(1.0, e0_delta) - 0.25 * min(1.0, e1_delta),
        }

    out_blk, dict_rec = execute_verified_multi_arm_moe_composition(
        lora_base,
        {"arm_1": arm_1, "arm_2": arm_2},
        eval_fn=_dict_eval,
        prose_hidden_states=u_prose.T.to(torch.float32),
        router_blend_weights={"arm_1": 0.4, "arm_2": 0.6},
        config=MultiArmComposerConfig(fold_ln_lambda=0.5, fold_tau_temp=1.0),
    )
    assert dict_rec["accepted"] is True
    assert "final_ln" in out_blk


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Bitwise Rollback on Infeasible Guard / Non-Beat
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_on_infeasible_guard_and_non_beat() -> None:
    """S4: Infeasible guard, non-finite outputs, and donor non-beat trigger atomic bitwise rollback."""
    base_model, arm_a, arm_b, probe_chunks, eval_chunks = _build_c3_multi_domain_arm_pair()
    base_sha = compute_checkpoint_sha256(base_model)
    arm_a_sha = compute_checkpoint_sha256(arm_a)

    # 4a: Composed model violates general-prose guard (> +1.0%) -> ANSE Maximum Pain rollback to base
    work_a = copy.deepcopy(base_model)

    def _infeasible_eval(target: Any) -> dict[str, Any]:
        if isinstance(target, nn.Module):
            cur_sha = compute_checkpoint_sha256(target)
        else:
            cur_sha = compute_checkpoint_sha256(target)
        if cur_sha == base_sha:
            return {"E0_general_fineweb": 3.00, "E1_code": 3.50, "E2_math": 4.00}
        if cur_sha == arm_a_sha:
            return {"E0_general_fineweb": 3.05, "E1_code": 3.20, "E2_math": 3.95}  # +1.67% fail
        return {"E0_general_fineweb": 3.12, "E1_code": 3.10, "E2_math": 3.50}  # +4.0% fail

    _, rec_fail = execute_verified_multi_arm_moe_composition(
        base_model,
        {"arm_a": arm_a, "arm_b": arm_b},
        target_model=work_a,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        eval_fn=_infeasible_eval,
        probe_domain_losses_by_arm={
            "arm_a": {"E0_general_fineweb": 3.05, "E1_code": 3.20, "E2_math": 3.95},
            "arm_b": {"E0_general_fineweb": 3.12, "E1_code": 3.90, "E2_math": 3.40},
        },
        base_probe_domain_losses={"E0_general_fineweb": 3.00, "E1_code": 3.50, "E2_math": 4.00},
        config=MultiArmComposerConfig(score_from=4),
    )
    assert rec_fail["accepted"] is False
    assert rec_fail["rolled_back"] is True
    assert rec_fail["rollback_target"] == "base"
    assert rec_fail["rollback_bitwise_verified"] is True
    assert rec_fail["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_fail["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_fail["final_state_sha256"] == rec_fail["base_state_sha256"]
    assert compute_checkpoint_sha256(work_a) == base_sha
    assert verify_composer_receipt_provenance(rec_fail) is True

    # 4b: Non-finite evaluation triggers ANSE Maximum Pain rollback
    work_b = copy.deepcopy(base_model)

    def _nan_cand_eval(target: Any) -> dict[str, Any]:
        cur_sha = compute_checkpoint_sha256(target)
        if cur_sha in (base_sha, arm_a_sha, compute_checkpoint_sha256(arm_b)):
            return {"E0_general_fineweb": 3.00, "E1_code": 3.50, "E2_math": 4.00}
        return {"non_finite_outputs": True}

    _, rec_nan = execute_verified_multi_arm_moe_composition(
        base_model,
        {"arm_a": arm_a, "arm_b": arm_b},
        target_model=work_b,
        probe_chunks=probe_chunks,
        eval_fn=_nan_cand_eval,
        probe_domain_losses_by_arm={
            "arm_a": {"E1_code": -0.2, "E2_math": 0.0, "E0_general_fineweb": 0.0},
            "arm_b": {"E1_code": 0.0, "E2_math": -0.3, "E0_general_fineweb": 0.0},
        },
        config=MultiArmComposerConfig(score_from=4),
    )
    assert rec_nan["rolled_back"] is True
    assert rec_nan["cand_anse_energy"] == ANSE_MAXIMUM_PAIN
    assert compute_checkpoint_sha256(work_b) == base_sha

    # 4c: Donor arm A beats composed candidate -> rollback_target="best_feasible_donor" restores arm_a
    work_c = copy.deepcopy(base_model)

    def _donor_wins_eval(target: Any) -> dict[str, Any]:
        cur_sha = compute_checkpoint_sha256(target)
        if cur_sha == base_sha:
            return {"E0_general_fineweb": 3.00, "E1_code": 3.50, "E2_math": 4.00}
        if cur_sha == arm_a_sha:
            return {"E0_general_fineweb": 3.00, "E1_code": 2.70, "E2_math": 3.50}
        if cur_sha == compute_checkpoint_sha256(arm_b):
            return {"E0_general_fineweb": 3.02, "E1_code": 3.40, "E2_math": 3.60}
        return {"E0_general_fineweb": 3.01, "E1_code": 3.05, "E2_math": 3.70}

    _, rec_rb_donor = execute_verified_multi_arm_moe_composition(
        base_model,
        {"arm_a": arm_a, "arm_b": arm_b},
        target_model=work_c,
        probe_chunks=probe_chunks,
        eval_fn=_donor_wins_eval,
        config=MultiArmComposerConfig(score_from=4, rollback_target="best_feasible_donor"),
    )
    assert rec_rb_donor["accepted"] is False
    assert rec_rb_donor["rolled_back"] is True
    assert rec_rb_donor["rollback_target"] == "arm_a"
    assert rec_rb_donor["final_state_sha256"] == rec_rb_donor["donor_state_sha256"]["arm_a"]
    assert compute_checkpoint_sha256(work_c) == arm_a_sha


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / Stub / Guardrail Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_gates() -> None:
    """S5: Input validation errors, provenance tampering, ZeroStubAudit, PRSecurityScanner & OwnerOrderGuardrail."""
    # 1. Config & provenance validation
    with pytest.raises(ValueError, match="max_general_drift_ratio"):
        MultiArmComposerConfig(max_general_drift_ratio=0.95)
    with pytest.raises(ValueError, match="anse_safe_general_ratio"):
        MultiArmComposerConfig(anse_safe_general_ratio=0.99)
    with pytest.raises(ValueError, match="nullspace_eig_threshold"):
        MultiArmComposerConfig(nullspace_eig_threshold=1.0)
    with pytest.raises(TypeError, match="bool"):
        MultiArmComposerConfig(inplace="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        MultiArmComposerConfig(score_from=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        MultiArmComposerConfig(score_from=-1)
    with pytest.raises(ValueError, match=">= 1"):
        MultiArmComposerConfig(batch_size=0)
    with pytest.raises(TypeError, match="real number"):
        MultiArmComposerConfig(min_energy_improvement="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        MultiArmComposerConfig(min_energy_improvement=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        MultiArmComposerConfig(min_domain_gain_nats=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        MultiArmComposerConfig(fold_tau_temp=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        MultiArmComposerConfig(fold_ln_lambda=1.5)

    assert verify_composer_receipt_provenance({"receipt_sha256": "short"}) is False
    assert verify_composer_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    with pytest.raises(TypeError, match="mapping"):
        verify_composer_receipt_provenance("bad")  # type: ignore[arg-type]

    # 2. fast_affine_layernorm_temperature_grid validation
    h_ok = torch.ones((4, 8))
    w_ok = torch.ones(8)
    b_ok = torch.zeros(8)
    head_ok = torch.ones((6, 8))
    t_ok = torch.zeros(4, dtype=torch.long)
    g_ok = torch.tensor([True, True, False, False])
    with pytest.raises(TypeError, match="Tensor"):
        fast_affine_layernorm_temperature_grid("bad", w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, g_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ndim"):
        fast_affine_layernorm_temperature_grid(torch.ones(8), w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, g_ok)
    with pytest.raises(ValueError, match="non-empty"):
        fast_affine_layernorm_temperature_grid(torch.empty((0, 8)), w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, g_ok)
    with pytest.raises(TypeError, match="floating-point"):
        fast_affine_layernorm_temperature_grid(torch.ones((4, 8), dtype=torch.long), w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, g_ok)
    with pytest.raises(ValueError, match="finite"):
        bad_h = h_ok.clone()
        bad_h[0, 0] = float("nan")
        fast_affine_layernorm_temperature_grid(bad_h, w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, g_ok)
    with pytest.raises(ValueError, match="both be provided or both be None"):
        fast_affine_layernorm_temperature_grid(h_ok, w_ok, b_ok, w_ok, None, head_ok, t_ok, g_ok)
    with pytest.raises(ValueError, match="hidden dim"):
        fast_affine_layernorm_temperature_grid(h_ok, w_ok, b_ok, w_ok, b_ok, torch.ones((6, 9)), t_ok, g_ok)
    with pytest.raises(TypeError, match="integer"):
        fast_affine_layernorm_temperature_grid(h_ok, w_ok, b_ok, w_ok, b_ok, head_ok, torch.zeros(4), g_ok)
    with pytest.raises(ValueError, match="non-empty"):
        fast_affine_layernorm_temperature_grid(h_ok, w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, g_ok, lambda_grid=[])
    with pytest.raises(TypeError, match="bool"):
        fast_affine_layernorm_temperature_grid(h_ok, w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, [1, 0, 1, 0])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-ignored token"):
        fast_affine_layernorm_temperature_grid(h_ok, w_ok, b_ok, w_ok, b_ok, head_ok, torch.full_like(t_ok, -100), g_ok)
    with pytest.raises(ValueError, match="token IDs"):
        fast_affine_layernorm_temperature_grid(h_ok, w_ok, b_ok, w_ok, b_ok, head_ok, torch.full_like(t_ok, 99), g_ok)
    with pytest.raises(ValueError, match="general token"):
        fast_affine_layernorm_temperature_grid(h_ok, w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, [False, False, False, False])

    # 3. select_domain_winning_experts_across_arms & execute_verified_multi_arm_moe_composition validation
    with pytest.raises(ValueError, match="at least 2"):
        select_domain_winning_experts_across_arms({"only": {0: torch.ones((2, 2))}}, {"only": {"E0": 1.0}})
    with pytest.raises(ValueError, match="both sink_experts and protected_experts"):
        select_domain_winning_experts_across_arms(
            {"a": {0: torch.ones((2, 2))}, "b": {0: torch.ones((2, 2))}},
            {"a": {"E0": 1.0}, "b": {"E0": 2.0}},
            sink_experts={0},
            protected_experts={0},
        )
    with pytest.raises(ValueError, match="symmetric"):
        select_domain_winning_experts_across_arms(
            {"a": {0: torch.ones((2, 2))}, "b": {0: torch.ones((2, 2))}},
            {"a": {"E0": 1.0}, "b": {"E0": 2.0}},
            torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="at least 2 donor arms"):
        execute_verified_multi_arm_moe_composition({"x": {"w": torch.ones(2)}}, {"only": {}})
    with pytest.raises(ValueError, match="rollback_target"):
        execute_verified_multi_arm_moe_composition(
            {"x": {"w": torch.ones(2)}},
            {"a": {}, "b": {}},
            config=MultiArmComposerConfig(rollback_target="nonexistent"),
        )

    # 4. AGPL-3.0, ZeroStubAudit (CI-safe), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn38", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn38", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/domain_matched_moe_arm_composer.py", TARGET_MODULE_PATH),
        ("tests/test_domain_matched_moe_arm_composer_scenarios.py", TEST_FILE_PATH),
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

    for doc_rel in (
        "docs/NEURO_CONTROLLER_ROADMAP.md",
        "docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md",
    ):
        doc_path = REPO_ROOT / doc_rel
        doc_text = doc_path.read_text(encoding="utf-8")
        assert scanner.scan_text_for_payment_solicitation(doc_text, doc_rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(doc_text, doc_rel) == []
        assert scanner.scan_unicode_trojan_source(doc_text, doc_rel) == []
        intent = classify_request_intent(
            doc_text, changed_files=[doc_rel], source_channel="pull_request"
        )
        assert intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION, (
            f"OwnerOrderGuardrail directive trigger in {doc_rel}: {intent.matched_triggers}"
        )

    telemetry_json = json.dumps(
        {
            "nll_mean": 3.1204,
            "domain_gain_nats": 0.3421,
            "matmul_speedup_factor": 25.0,
            "anse_energy": -0.2185,
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
        pr_number=1238,
        pr_title="feat(swarm): domain-matched multi-arm MoE specialist composer and affine 2-matmul LayerNorm calibrator",
        pr_body="Implements NEURAL-CTRL-DYN-38 domain-matched multi-arm MoE specialist composer and 2-matmul affine LayerNorm calibrator with P_0 null-space projection, ANSE ratchet, and SHA-256 provenance.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): domain-matched multi-arm MoE specialist composer and affine 2-matmul LayerNorm calibrator"
        ],
        author_email=None,
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []


def test_s6_layout_projector_and_validation_branches() -> None:
    """S6: Cover checkpoint_blocks/nested_layers layouts, SwiGLU/1D projectors, and validation branches."""
    # Projector binarization + Tensor/Mapping resolution + 1D & SwiGLU dense projection
    p_non_idem = torch.diag(torch.tensor([0.9, 0.8, 0.05, 0.0], dtype=torch.float64))
    p_zero = torch.diag(torch.tensor([0.05, 0.02, 0.0, 0.0], dtype=torch.float64))
    p_inter = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64))

    # 1. Nested layer -> expert SwiGLU layout with mean_diff / delta_nll & revert_harmful_to_zero
    nested_arms = {
        "arm_a": {
            "blocks.0": {
                "expert_0": {"w1": torch.ones((6, 4)), "w2": torch.ones((6, 4)), "w3": torch.ones((4, 6)), "scaling": 1.0},
                "expert_1": {"w1": torch.full((6, 4), 2.0), "w2": torch.ones((6, 4)), "w3": torch.ones((4, 6))},
                "expert_2": {"w1": torch.ones((6, 4)), "w2": torch.ones((6, 4)), "w3": torch.ones((4, 6))},
            }
        },
        "arm_b": {
            "blocks.0": {
                "expert_0": {"w1": torch.ones((6, 4)), "w2": torch.ones((6, 4)), "w3": torch.ones((4, 6))},
                "expert_1": {"w1": torch.ones((6, 4)), "w2": torch.ones((6, 4)), "w3": torch.ones((4, 6))},
                "expert_2": {"w1": torch.full((6, 4), 3.0), "w2": torch.ones((6, 4)), "w3": torch.ones((4, 6))},
            }
        },
    }
    nested_base = {
        "blocks.0": {
            "expert_0": {"w1": torch.zeros((6, 4)), "w2": torch.zeros((6, 4)), "w3": torch.zeros((4, 6))},
            "expert_1": {"w1": torch.zeros((6, 4)), "w2": torch.zeros((6, 4)), "w3": torch.zeros((4, 6))},
            "expert_2": {"w1": torch.zeros((6, 4)), "w2": torch.zeros((6, 4)), "w3": torch.zeros((4, 6))},
        }
    }
    probe_losses_dict = {
        "base": {"E0_gen": 3.0, "E1_code": 3.0, "E2_math": 3.0},
        "arm_a": {
            "E0_gen": {"mean_diff": -0.01, "moe_nll": 2.99},
            "E1_code": {"delta_nll": -0.25, "arm_nll": 2.75},
            "E2_math": {"moe_nll": 3.05, "base_nll": 3.0},
        },
        "arm_b": {
            "E0_gen": {"mean_diff": 0.0, "moe_nll": 3.0},
            "E1_code": {"delta_nll": -0.05, "arm_nll": 2.95},
            "E2_math": {"moe_nll": 3.02, "base_nll": 3.0},
        },
    }
    comp_nested, rep_nested = select_domain_winning_experts_across_arms(
        nested_arms,
        probe_losses_dict,
        {"blocks.0": {"x": p_non_idem, "act": p_inter, "expert_1": p_non_idem}},
        base_experts=nested_base,
        sink_experts=[0],
        min_domain_gain_nats=0.10,
        revert_harmful_to_zero=True,
    )
    assert "blocks.0" in comp_nested
    assert rep_nested["expert_assignments"]["expert_0"]["action"] == "ghost_sink_pruned"
    assert rep_nested["expert_assignments"]["expert_2"]["action"] == "reverted_unhelpful_slot"
    assert rep_nested["expert_assignments"]["expert_1"]["winning_arm"] == "arm_a"

    # 2. checkpoint_blocks layout with 1D vector deltas + p_null_by_expert Tensor / zero projector / x-act top-level keys
    ckpt_arms = {
        "arm_a": {
            "blocks.0.mlp.experts.0": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "blocks.0.mlp.experts.1": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        },
        "arm_b": {
            "blocks.0.mlp.experts.0": torch.tensor([0.5, 0.5, 0.5, 0.5]),
            "blocks.0.mlp.experts.1": torch.tensor([2.0, -1.0, 0.5, 1.5]),
        },
    }
    comp_1d, _ = select_domain_winning_experts_across_arms(
        ckpt_arms,
        {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
        p_zero,
        nullspace_leak_tol=1e-6,
    )
    assert comp_1d["blocks.0.mlp.experts.1"].shape == (4,)
    comp_xkey, _ = select_domain_winning_experts_across_arms(
        ckpt_arms,
        {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
        {"x": p_non_idem},
        nullspace_leak_tol=100.0,
    )
    assert comp_xkey["blocks.0.mlp.experts.0"].shape == (4,)

    # 3. Output-side zero-base & non-zero-base LoRA projection (when px only, no pact)
    lora_arms = {
        "arm_a": {
            0: {
                "w1_A": torch.ones((2, 4)),
                "w1_B": torch.ones((6, 2)),
                "w2_A": torch.ones((2, 4)),
                "w2_B": torch.ones((6, 2)),
                "w3_A": torch.ones((2, 6)),
                "w3_B": torch.ones((4, 2)),
            }
        },
        "arm_b": {
            0: {
                "w1_A": torch.full((2, 4), 1.5),
                "w1_B": torch.ones((6, 2)),
                "w2_A": torch.full((2, 4), 1.5),
                "w2_B": torch.ones((6, 2)),
                "w3_A": torch.ones((2, 6)),
                "w3_B": torch.full((4, 2), 2.0),
            }
        },
    }
    lora_base = {
        0: {
            "w1_A": torch.full((2, 4), 0.2),
            "w1_B": torch.full((6, 2), 0.2),
            "w2_A": torch.full((2, 4), 0.2),
            "w2_B": torch.full((6, 2), 0.2),
            "w3_A": torch.full((2, 6), 0.2),
            "w3_B": torch.full((4, 2), 0.2),
        }
    }
    select_domain_winning_experts_across_arms(
        lora_arms,
        {"arm_a": {"E0_gen": 2.8, "E1_code": 2.7}, "arm_b": {"E0_gen": 2.7, "E1_code": 2.5}},
        p_non_idem,
    )
    select_domain_winning_experts_across_arms(
        lora_arms,
        {"arm_a": {"E0_gen": 2.8, "E1_code": 2.7}, "arm_b": {"E0_gen": 2.7, "E1_code": 2.5}},
        p_non_idem,
        base_experts=lora_base,
        sink_experts=[0],
    )
    select_domain_winning_experts_across_arms(
        lora_arms,
        {"arm_a": {"E0_gen": 2.8, "E1_code": 2.7}, "arm_b": {"E0_gen": 2.7, "E1_code": 2.5}},
        p_inter,
    )

    # 4. Remaining fast validation branches across helpers & execute_verified_multi_arm_moe_composition
    with pytest.raises(ValueError, match="matches neither"):
        select_domain_winning_experts_across_arms(
            ckpt_arms,
            {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
            torch.eye(9, dtype=torch.float64),
        )
    with pytest.raises(TypeError, match="p_null_by_expert"):
        select_domain_winning_experts_across_arms(
            ckpt_arms,
            {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
            "invalid",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="expert_domain_map"):
        select_domain_winning_experts_across_arms(
            ckpt_arms,
            {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
            expert_domain_map={1: "E9_missing"},
        )
    with pytest.raises(ValueError, match="missing donor arm"):
        select_domain_winning_experts_across_arms(ckpt_arms, {"arm_a": {"E0_gen": 2.9}})
    with pytest.raises(ValueError, match="no common domain keys"):
        select_domain_winning_experts_across_arms(
            ckpt_arms, {"arm_a": {"E0_gen": 2.9}, "arm_b": {"E1_code": 2.5}}
        )
    with pytest.raises(ValueError, match="identical layer and expert slot layouts"):
        select_domain_winning_experts_across_arms(
            {
                "arm_a": {"blocks.0.mlp.experts.0": torch.ones((2, 2))},
                "arm_b": {"blocks.0.mlp.experts.1": torch.ones((2, 2))},
            },
            {"arm_a": {"E0_gen": 2.9}, "arm_b": {"E0_gen": 2.8}},
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        select_domain_winning_experts_across_arms(
            {
                "arm_a": {"blocks.0.mlp.experts.0": torch.ones((2, 2))},
                "arm_b": {"blocks.0.mlp.experts.0": torch.ones((2, 3))},
            },
            {"arm_a": {"E0_gen": 2.9}, "arm_b": {"E0_gen": 2.8}},
        )
    with pytest.raises(ValueError, match="Mixed block key"):
        select_domain_winning_experts_across_arms(
            {
                "arm_a": {"blocks.0.mlp.experts.0": torch.ones((2, 2)), "bad": torch.ones((2, 2))},
                "arm_b": {"blocks.0.mlp.experts.0": torch.ones((2, 2)), "bad": torch.ones((2, 2))},
            },
            {"arm_a": {"E0_gen": 2.9}, "arm_b": {"E0_gen": 2.8}},
        )
    with pytest.raises(ValueError, match="Cannot parse expert slot"):
        select_domain_winning_experts_across_arms(
            {"arm_a": {"bad_slot": torch.ones((2, 2))}, "arm_b": {"bad_slot": torch.ones((2, 2))}},
            {"arm_a": {"E0_gen": 2.9}, "arm_b": {"E0_gen": 2.8}},
        )
    with pytest.raises(ValueError, match="Cannot parse layer index"):
        select_domain_winning_experts_across_arms(
            {
                "arm_a": {"bad_layer": {"expert_0": {"w": torch.ones((2, 2))}}},
                "arm_b": {"bad_layer": {"expert_0": {"w": torch.ones((2, 2))}}},
            },
            {"arm_a": {"E0_gen": 2.9}, "arm_b": {"E0_gen": 2.8}},
        )

    base_blk = {
        "blocks.0.mlp.router": {"weight": torch.ones((2, 4))},
        "blocks.0.mlp.experts.0": {"delta": torch.zeros((4, 4))},
        "blocks.0.mlp.experts.1": {"delta": torch.zeros((4, 4))},
    }
    donors_blk = {
        "arm_a": {
            "blocks.0.mlp.router": {"weight": torch.ones((2, 4)) * 1.1},
            "blocks.0.mlp.experts.0": {"delta": torch.ones((4, 4)) * 0.2},
            "blocks.0.mlp.experts.1": {"delta": torch.ones((4, 4)) * 0.1},
        },
        "arm_b": {
            "blocks.0.mlp.router": {"weight": torch.ones((2, 4)) * 1.2},
            "blocks.0.mlp.experts.0": {"delta": torch.ones((4, 4)) * 0.1},
            "blocks.0.mlp.experts.1": {"delta": torch.ones((4, 4)) * 0.3},
        },
    }
    # Rollback to base when rollback_target='best_feasible_donor' but all donors violate general guard
    _, rec_fb = execute_verified_multi_arm_moe_composition(
        base_blk,
        donors_blk,
        prose_hidden_states={"0": torch.randn((8, 4), generator=torch.Generator().manual_seed(42))},
        eval_fn=lambda b: (
            {"E0_gen": 2.0, "E1_code": 3.0}
            if float(b["blocks.0.mlp.experts.0"]["delta"].abs().sum().item()) == 0.0
            else {"E0_gen": 2.5, "E1_code": 2.6}
        ),
        config=MultiArmComposerConfig(rollback_target="best_feasible_donor", inplace=False),
    )
    assert rec_fb["accepted"] is False
    assert rec_fb["rollback_target"] == "base"
    assert verify_composer_receipt_provenance(rec_fb) is True

    with pytest.raises(TypeError, match="MultiArmComposerConfig"):
        execute_verified_multi_arm_moe_composition(base_blk, donors_blk, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="eval_fn"):
        execute_verified_multi_arm_moe_composition(base_blk, donors_blk, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_model"):
        execute_verified_multi_arm_moe_composition(base_blk, donors_blk, target_model="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eval_fn must be provided"):
        execute_verified_multi_arm_moe_composition(base_blk, donors_blk)
    with pytest.raises(ValueError, match="MoE '.router' and '.experts.'"):
        execute_verified_multi_arm_moe_composition(
            {"only_ln": {"w": torch.ones(4)}},
            {"a": {"only_ln": {"w": torch.ones(4)}}, "b": {"only_ln": {"w": torch.ones(4)}}},
            eval_fn=lambda _: {"E0_gen": 2.0, "E1_code": 2.0},
        )
    with pytest.raises(ValueError, match="non-finite"):
        execute_verified_multi_arm_moe_composition(
            base_blk,
            donors_blk,
            eval_fn=lambda _: {"E0_gen": float("nan"), "E1_code": 2.0},
        )

    # 5. Additional fast mask/grid/block/eval_fn/rollback validation branches (>=95% standalone coverage)
    h_b = torch.randn((4, 3, 4), generator=torch.Generator().manual_seed(7))
    w0_b = torch.ones(4)
    w1_b = torch.tensor([1.2, 0.8, 1.1, 0.9])
    hd_b = torch.randn((5, 4), generator=torch.Generator().manual_seed(8))
    tgt_b = torch.tensor([[0, 1, 2], [-100, -100, -100], [1, 2, 3], [2, 3, 4]], dtype=torch.long)
    fast_affine_layernorm_temperature_grid(
        h_b,
        w0_b,
        None,
        w1_b,
        None,
        hd_b,
        tgt_b,
        np.array([True, True, False, False]),
        seq_batch_size=1,
        domain_labels=["E0_a", "E0_a", "E1_b", "E2_c"],
        max_domain_drift_rel=0.0001,
    )
    for bad_mask, exc_t in (
        (torch.ones((4, 3), dtype=torch.long), TypeError),
        (np.array([1, 0, 1, 0]), TypeError),
        ([], ValueError),
        ("bad", TypeError),
        ([True, False], ValueError),
    ):
        with pytest.raises(exc_t):
            fast_affine_layernorm_temperature_grid(h_b, w0_b, None, w1_b, None, hd_b, tgt_b, bad_mask)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="domain_labels"):
        fast_affine_layernorm_temperature_grid(
            h_b, w0_b, None, w1_b, None, hd_b, tgt_b, [True, True, False, False], domain_labels=["E0"]
        )
    with pytest.raises(ValueError, match="length"):
        fast_affine_layernorm_temperature_grid(h_b, torch.ones(3), None, w1_b, None, hd_b, tgt_b, [True, True, False, False])
    with pytest.raises(ValueError, match="ndim"):
        fast_affine_layernorm_temperature_grid(h_b, w0_b, None, w1_b, None, torch.ones(4), tgt_b, [True, True, False, False])
    with pytest.raises(TypeError, match="Tensor"):
        fast_affine_layernorm_temperature_grid(h_b, w0_b, None, w1_b, None, hd_b, [[0, 1, 2]] * 4, [True, True, False, False])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        fast_affine_layernorm_temperature_grid(h_b, w0_b, None, w1_b, None, hd_b, torch.zeros((2, 3), dtype=torch.long), [True, True, False, False])
    with pytest.raises(TypeError, match="sequence"):
        fast_affine_layernorm_temperature_grid(h_b, w0_b, None, w1_b, None, hd_b, tgt_b, [True, True, False, False], lambda_grid="bad")  # type: ignore[arg-type]

    # Empty p_null mapping -> (None, None) & projector non-square / dim checks & sink all-slots uniform router blend
    select_domain_winning_experts_across_arms(
        ckpt_arms,
        {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
        {},
    )
    select_domain_winning_experts_across_arms(
        ckpt_arms,
        {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
        {"blocks.99": p_non_idem},
    )
    with pytest.raises(ValueError, match="square"):
        select_domain_winning_experts_across_arms(
            ckpt_arms,
            {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
            torch.ones((2, 3)),
        )
    with pytest.raises(TypeError, match="sink_experts"):
        select_domain_winning_experts_across_arms(
            ckpt_arms,
            {"arm_a": {"E0_gen": 2.9, "E1_code": 2.8}, "arm_b": {"E0_gen": 2.95, "E1_code": 2.5}},
            sink_experts="bad",  # type: ignore[arg-type]
        )

    # Named donor rollback + router_blend_weights + block/eval_fn validation
    _, rec_named_rb = execute_verified_multi_arm_moe_composition(
        base_blk,
        donors_blk,
        sink_experts=[0, 1],
        eval_fn=lambda _: {"E0_gen": 2.5, "E1_code": 2.5, "all_finite": True},
        config=MultiArmComposerConfig(rollback_target="arm_b"),
        inplace=False,
    )
    assert rec_named_rb["rollback_target"] == "arm_b"

    for bad_w in ({"arm_a": 1.0}, {"arm_a": 0.0, "arm_b": 0.0}):
        with pytest.raises(ValueError, match="router_blend_weights"):
            execute_verified_multi_arm_moe_composition(
                base_blk,
                donors_blk,
                router_blend_weights=bad_w,
                eval_fn=lambda _: {"E0_gen": 2.0, "E1_code": 2.0},
            )
    for bad_ret, exc_e in (
        ({}, ValueError),
        ({"E0_gen": True, "E1_code": 2.0}, TypeError),
        ({"E0_gen": 2.0}, ValueError),
    ):
        with pytest.raises(exc_e):
            execute_verified_multi_arm_moe_composition(base_blk, donors_blk, eval_fn=lambda _, r=bad_ret: r)
    for bad_base, exc_b in (
        ("bad", TypeError),
        ({"": {"w": torch.ones(2)}}, ValueError),
        ({"b": {}}, TypeError),
        ({"b": {"w": torch.ones(2, dtype=torch.long)}}, ValueError),
        ({"b": {"w": torch.tensor([float("nan")])}}, ValueError),
    ):
        with pytest.raises(exc_b):
            execute_verified_multi_arm_moe_composition(bad_base, donors_blk, eval_fn=lambda _: {"E0_gen": 2.0, "E1_code": 2.0})  # type: ignore[arg-type]
    for bad_d in (
        {"arm_a": {"blocks.0.mlp.router": {"weight": torch.ones((2, 4))}}, "arm_b": donors_blk["arm_b"]},
        {
            "arm_a": {
                "blocks.0.mlp.router": {"other": torch.ones((2, 4))},
                "blocks.0.mlp.experts.0": {"delta": torch.zeros((4, 4))},
                "blocks.0.mlp.experts.1": {"delta": torch.zeros((4, 4))},
            },
            "arm_b": donors_blk["arm_b"],
        },
        {
            "arm_a": {
                "blocks.0.mlp.router": {"weight": torch.ones((2, 5))},
                "blocks.0.mlp.experts.0": {"delta": torch.zeros((4, 4))},
                "blocks.0.mlp.experts.1": {"delta": torch.zeros((4, 4))},
            },
            "arm_b": donors_blk["arm_b"],
        },
    ):
        with pytest.raises(ValueError):
            execute_verified_multi_arm_moe_composition(base_blk, bad_d, eval_fn=lambda _: {"E0_gen": 2.0, "E1_code": 2.0})
