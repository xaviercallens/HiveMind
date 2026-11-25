# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-36: post-DiLoCo cross-arm router-freeze synergy surgeon."""

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
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAExpertAdapter,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.router_freeze_cross_arm_surgeon import (
    ANSE_MAXIMUM_PAIN,
    RECEIPT_OBJECT,
    SCHEMA_VERSION,
    CrossArmSurgeryConfig,
    apply_grafted_state_inplace,
    compute_checkpoint_sha256,
    decompose_router_vs_expert_prose_drift,
    execute_verified_cross_arm_moe_surgery,
    extract_checkpoint_blocks,
    graft_specialist_experts_onto_safe_router,
    verify_cross_arm_receipt_provenance,
)
from hivemind.swarm.tangent_task_vector_surgeon import compute_prose_nullspace_projector_scaled

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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "router_freeze_cross_arm_surgeon.py"
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


def _build_campaign3_cross_arm_trio(
    *,
    seed: int = 20261236,
    hidden_size: int = 48,
    intermediate_size: int = 96,
    num_layers: int = 3,
    vocab_size: int = 64,
    num_experts: int = 8,
    lora_rank: int = 4,
    include_sink_expert: bool = True,
) -> tuple[
    CrowdV1WithEngram,
    CrowdV1WithEngram,
    CrowdV1WithEngram,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, torch.Tensor],
]:
    """Build deterministic 8-expert Base, Arm D (frozen router), and Arm B (unfrozen router) MoEs.

    Geometry mirrors Campaign-3 ``E-MOE-C3``:
      - Tokens ``0..31`` (``E0_general_fineweb``): zero-mean subspace in ``dims 0..23``.
      - Tokens ``32..63`` (``E6_stellar_spectroscopy``): zero-mean subspace in ``dims 24..47``.
      - Arm D (router-frozen): router anchored at base (``E0 <= +0.2%``) with modest ``expert_6`` gain.
      - Arm B (router-unfrozen): deep ``-40%``+ ``expert_6`` specialist gain AND unfrozen router
        boost on ``E6``, PLUS router-induced prose misfires (``E0 >= +12%``) and optional
        prose-corrupting sink ``expert_3``.
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

    p_basis = _make_zero_mean_ortho_rows(4, half_h, seed=seed + 11)
    s_basis = _make_zero_mean_ortho_rows(4, hidden_size - half_h, seed=seed + 23)
    u_p_in, u_p_out = p_basis[0], p_basis[1]
    u_s_in, u_s_out = s_basis[0], s_basis[1]

    with torch.no_grad():
        for param in base_model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_rows = _make_zero_mean_ortho_rows(16, half_h, seed=seed + 31)
        s_rows = _make_zero_mean_ortho_rows(16, hidden_size - half_h, seed=seed + 37)
        for i in range(half_v):
            row = 0.22 * p_rows[i % 16] + 0.58 * u_p_in
            wte[i, :half_h] = row - row.mean()
        for i in range(vocab_size - half_v):
            row = 0.22 * s_rows[i % 16] + 0.58 * u_s_in
            wte[half_v + i, half_h:] = row - row.mean()
        base_model.wte.weight.copy_(wte)

        head_w = 0.08 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)
        head_w[:, :half_h] = head_w[:, :half_h] - head_w[:, :half_h].mean(dim=1, keepdim=True)
        head_w[:, half_h:] = head_w[:, half_h:] - head_w[:, half_h:].mean(dim=1, keepdim=True)
        for tok_p in (3, 4, 5):
            head_w[tok_p, :half_h] = 0.25 * u_p_in + 0.35 * u_p_out
        for tok_s in (40, 41, 42):
            head_w[tok_s, half_h:] = 0.08 * u_s_in + 0.42 * u_s_out
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
            # Base / frozen router gives moderate gate to expert 0 on prose & mild gate to expert 6 on astronomy
            blk.mlp.router.weight[0, :half_h] = 2.4 * u_p_in
            blk.mlp.router.weight[6, half_h:] = 0.05 * u_s_in

    arm_d_model = copy.deepcopy(base_model)
    arm_b_model = copy.deepcopy(base_model)

    with torch.no_grad():
        # Arm D (router-frozen): router identical to base; expert 6 has modest specialist improvement
        for l_idx in range(num_layers):
            exp_d6 = arm_d_model.blocks[l_idx].mlp.experts["6"]
            for p in exp_d6.parameters():
                p.zero_()
            exp_d6.w1_A[0, half_h:] = 0.45 * u_s_in
            exp_d6.w1_B[half_h:hidden_size, 0] = 0.22 * u_s_out
            exp_d6.w2_A[0, half_h:] = 0.45 * u_s_in
            exp_d6.w2_B[half_h:hidden_size, 0] = 0.22 * u_s_out
            exp_d6.w3_A[0, half_h:hidden_size] = 0.45 * u_s_out
            exp_d6.w3_B[half_h:, 0] = 0.12 * u_s_out

        # Arm B (router-unfrozen): deep 3x specialist expert 6 + unfrozen router boost on E6,
        # paired with severe router-induced prose misfires (and optional sink expert 3)
        for l_idx in range(num_layers):
            exp_b6 = arm_b_model.blocks[l_idx].mlp.experts["6"]
            for p in exp_b6.parameters():
                p.zero_()
            # Mode 0: deep specialist astronomy feature (-40%+ E6 NLL drop)
            exp_b6.w1_A[0, half_h:] = 1.15 * u_s_in
            exp_b6.w1_B[half_h:hidden_size, 0] = 0.48 * u_s_out
            exp_b6.w2_A[0, half_h:] = 1.15 * u_s_in
            exp_b6.w2_B[half_h:hidden_size, 0] = 0.48 * u_s_out
            exp_b6.w3_A[0, half_h:hidden_size] = 0.95 * u_s_out
            exp_b6.w3_B[half_h:, 0] = 0.52 * u_s_out

            # Mode 1: mild dual-active prose leakage inside expert 6 (removed by P_0)
            exp_b6.w1_A[1, :half_h] = 0.9 * u_p_in
            exp_b6.w1_B[:half_h, 1] = 0.24 * u_p_out
            exp_b6.w2_A[1, :half_h] = 0.9 * u_p_in
            exp_b6.w2_B[:half_h, 1] = 0.24 * u_p_out
            exp_b6.w3_A[1, :half_h] = 0.9 * u_p_out
            exp_b6.w3_B[:half_h, 1] = -0.52 * u_p_out - 0.32 * u_p_in

            # Unfrozen router boosts expert 6 on specialist tokens AND leaks heavily onto prose tokens
            arm_b_model.blocks[l_idx].mlp.router.weight[6, half_h:] += 2.5 * u_s_in
            arm_b_model.blocks[l_idx].mlp.router.weight[6, :half_h] += 3.8 * u_p_in

            if include_sink_expert:
                exp_b3 = arm_b_model.blocks[l_idx].mlp.experts["3"]
                for p in exp_b3.parameters():
                    p.zero_()
                exp_b3.w1_A[0, :half_h] = 1.2 * u_p_in
                exp_b3.w1_B[:half_h, 0] = 0.30 * u_p_out
                exp_b3.w2_A[0, :half_h] = 1.2 * u_p_in
                exp_b3.w2_B[:half_h, 0] = 0.30 * u_p_out
                exp_b3.w3_A[0, :half_h] = 1.0 * u_p_out
                exp_b3.w3_B[:half_h, 0] = -0.60 * u_p_out - 0.38 * u_p_in
                arm_b_model.blocks[l_idx].mlp.router.weight[3, :half_h] += 3.5 * u_p_in

    def _make_chunks(chunk_seed: int, n_per_domain: int) -> list[dict[str, Any]]:
        g = torch.Generator(device="cpu").manual_seed(chunk_seed)
        chunks: list[dict[str, Any]] = []
        for idx in range(n_per_domain):
            seq_p = torch.randint(0, half_v, (12,), generator=g).tolist()
            for pos_p in range(6, 12):
                seq_p[pos_p] = 3 + (pos_p % 3)
            chunks.append(
                {"expert_id": "E0_general_fineweb", "dataset_id": "fineweb", "row": chunk_seed + idx, "ids": seq_p}
            )
            seq_s = torch.randint(half_v, vocab_size, (12,), generator=g).tolist()
            for pos_s in range(6, 12):
                seq_s[pos_s] = 40 + (pos_s % 3)
            chunks.append(
                {"expert_id": "E6_stellar_spectroscopy", "dataset_id": "sdss", "row": chunk_seed + idx, "ids": seq_s}
            )
        return chunks

    probe_chunks = _make_chunks(seed + 1000, n_per_domain=8)
    eval_chunks = _make_chunks(seed + 2000, n_per_domain=8)
    directions = {"u_p_in": u_p_in, "u_p_out": u_p_out, "u_s_in": u_s_in, "u_s_out": u_s_out}
    return base_model, arm_d_model, arm_b_model, probe_chunks, eval_chunks, directions


def _eval_domain_nlls(model: CrowdV1WithEngram, chunks: list[dict[str, Any]], *, score_from: int = 6) -> dict[str, float]:
    """Compute exact per-domain mid-row NLLs on ``chunks`` via full model forward."""
    model.eval()
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    with torch.no_grad():
        logits, _, _ = model(ids)
        start = max(0, score_from - 1)
        shift_logits = logits[:, start:-1, :]
        shift_targets = ids[:, start + 1 :]
        per_tok = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_targets.reshape(-1),
            reduction="none",
        ).view(ids.shape[0], -1)
        per_chunk = per_tok.mean(dim=1).tolist()
    by_dom: dict[str, list[float]] = {}
    for ch, val in zip(chunks, per_chunk, strict=True):
        by_dom.setdefault(str(ch["expert_id"]), []).append(float(val))
    return {k: float(sum(v) / len(v)) for k, v in sorted(by_dom.items())}


def test_s1_campaign3_arm_d_and_arm_b_cross_arm_synergy() -> None:
    """S1: Cross-arm grafting beats BOTH Arm D (frozen router) and Arm B (unfrozen router) at <= +0.3% prose drift."""
    base_m, arm_d_m, arm_b_m, probe_chunks, eval_chunks, _ = _build_campaign3_cross_arm_trio()

    # Verify synthetic Campaign-3 donor behavior on disjoint eval split
    base_dom = _eval_domain_nlls(base_m, eval_chunks)
    arm_d_dom = _eval_domain_nlls(arm_d_m, eval_chunks)
    arm_b_dom = _eval_domain_nlls(arm_b_m, eval_chunks)

    arm_d_e0_drift = (arm_d_dom["E0_general_fineweb"] / base_dom["E0_general_fineweb"]) - 1.0
    arm_b_e0_drift = (arm_b_dom["E0_general_fineweb"] / base_dom["E0_general_fineweb"]) - 1.0
    arm_b_e6_rel = (arm_b_dom["E6_stellar_spectroscopy"] / base_dom["E6_stellar_spectroscopy"]) - 1.0
    arm_d_e6_rel = (arm_d_dom["E6_stellar_spectroscopy"] / base_dom["E6_stellar_spectroscopy"]) - 1.0

    assert abs(arm_d_e0_drift) <= 0.003
    assert -0.25 <= arm_d_e6_rel <= -0.05
    assert arm_b_e0_drift >= 0.11
    assert arm_b_e6_rel <= -0.38

    cfg = CrossArmSurgeryConfig(score_from=6, gate_blend_alpha=0.15, inplace=False)
    hybrid_m, receipt = execute_verified_cross_arm_moe_surgery(
        base_m,
        arm_d_m,
        arm_b_m,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )

    assert receipt["object"] == RECEIPT_OBJECT
    assert receipt["schema"] == SCHEMA_VERSION
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "CROSS_ARM_SYNERGY_COMMITTED"
    assert verify_cross_arm_receipt_provenance(receipt) is True

    # Beats BOTH donor arms under ANSE Delta E < 0 and keeps prose drift <= +0.3%
    assert receipt["general_drift_ratio"] <= 1.003
    assert receipt["delta_e_vs_base"] < 0.0
    assert receipt["delta_e_vs_frozen_arm"] < 0.0
    assert receipt["delta_e_vs_unfrozen_arm"] < 0.0
    assert receipt["anse_energy"] < receipt["frozen_arm_anse_energy"]
    assert receipt["anse_energy"] < receipt["unfrozen_arm_anse_energy"]
    assert receipt["router_kl_drift"] > 0.05
    assert receipt["specialist_gain_nats"] >= 2.0 * receipt["frozen_arm_spec_gain_nats"]

    hybrid_dom = _eval_domain_nlls(hybrid_m, eval_chunks)
    hybrid_e6_rel = (hybrid_dom["E6_stellar_spectroscopy"] / base_dom["E6_stellar_spectroscopy"]) - 1.0
    hybrid_e0_drift = (hybrid_dom["E0_general_fineweb"] / base_dom["E0_general_fineweb"]) - 1.0
    assert hybrid_e0_drift <= 0.003
    assert hybrid_e6_rel <= -0.35
    assert hybrid_dom["E6_stellar_spectroscopy"] < arm_d_dom["E6_stellar_spectroscopy"]
    assert receipt["final_state_sha256"] == compute_checkpoint_sha256(
        extract_checkpoint_blocks(hybrid_m)
    )
    assert torch.equal(hybrid_m.wte.weight, base_m.wte.weight)
    assert "score" not in receipt


def test_s2_router_vs_expert_kl_decomposition_and_p0_invariance() -> None:
    """S2: Exact router-vs-expert KL/output/NLL attribution & P_0 invariance with OOF gating."""
    gen = torch.Generator(device="cpu").manual_seed(20260201)
    dim_h, dim_i, num_e, rank = 16, 24, 4, 2
    p_basis = _make_zero_mean_ortho_rows(4, 8, seed=101)
    s_basis = _make_zero_mean_ortho_rows(4, 8, seed=202)
    u_p = torch.cat([p_basis[0], torch.zeros(8)])
    u_p_out = torch.cat([p_basis[1], torch.zeros(8)])
    u_s = torch.cat([torch.zeros(8), s_basis[0]])
    u_s_out = torch.cat([torch.zeros(8), s_basis[1]])

    x_prose = torch.stack(
        [
            ((0.7 + 0.3 * ((i % 4) - 1.5)) * u_p + 0.25 * torch.cat([p_basis[1 + (i % 3)], torch.zeros(8)]))
            for i in range(16)
        ],
        dim=0,
    )
    x_spec = torch.stack(
        [
            ((0.8 + 0.2 * ((i % 4) - 1.5)) * u_s + 0.20 * torch.cat([torch.zeros(8), s_basis[1 + (i % 3)]]))
            for i in range(16)
        ],
        dim=0,
    )

    r_base = torch.zeros((num_e, dim_h), dtype=torch.float32)
    r_base[0] = 2.0 * u_p
    r_base[1] = 1.0 * u_s

    # Pure router shift vs pure expert shift decomposition
    r_unf = r_base.clone()
    r_unf[1] = 1.0 * u_s + 3.5 * u_p + 2.5 * u_s

    exp_clean = CrowdLoRAExpertAdapter(dim_h, dim_i, rank=rank, seed=11)
    with torch.no_grad():
        for p in exp_clean.parameters():
            p.zero_()
        exp_clean.w1_A[0] = 1.2 * u_s
        exp_clean.w1_B[:dim_h, 0] = 0.5 * u_s_out
        exp_clean.w2_A[0] = 1.2 * u_s
        exp_clean.w2_B[:dim_h, 0] = 0.5 * u_s_out
        exp_clean.w3_A[0, :dim_h] = 1.0 * u_s_out
        exp_clean.w3_B[:, 0] = 0.4 * u_s_out

    exp_leaking = CrowdLoRAExpertAdapter(dim_h, dim_i, rank=rank, seed=22)
    with torch.no_grad():
        for p in exp_leaking.parameters():
            p.zero_()
        # Mode 0: specialist signal in range(P_0); Mode 1: prose leak in ker(P_0)
        exp_leaking.w1_A[0] = 1.3 * u_s
        exp_leaking.w1_B[:dim_h, 0] = 0.5 * u_s_out
        exp_leaking.w2_A[0] = 1.3 * u_s
        exp_leaking.w2_B[:dim_h, 0] = 0.5 * u_s_out
        exp_leaking.w3_A[0, :dim_h] = 1.1 * u_s_out
        exp_leaking.w3_B[:, 0] = 0.5 * u_s_out

        exp_leaking.w1_A[1] = 1.1 * u_p
        exp_leaking.w1_B[:dim_h, 1] = 0.4 * u_p_out
        exp_leaking.w2_A[1] = 1.1 * u_p
        exp_leaking.w2_B[:dim_h, 1] = 0.4 * u_p_out
        exp_leaking.w3_A[1, :dim_h] = 1.0 * u_p_out
        exp_leaking.w3_B[:, 1] = -0.6 * u_p_out

    experts_zero = {
        e: CrowdLoRAExpertAdapter(dim_h, dim_i, rank=rank, seed=e + 10) for e in range(num_e)
    }
    experts_mixed = copy.deepcopy(experts_zero)
    experts_mixed[1] = exp_leaking

    # A. Pure router drift (expert 1 is purely in ker(P_0) or active only when unfrozen router fires on prose)
    experts_router_only = copy.deepcopy(experts_zero)
    experts_router_only[1] = exp_leaking
    # Because frozen router assigns negative/zero weight on prose? Let's make frozen router keep expert 1 out of top-2 on prose!
    r_frz_strict = r_base.clone()
    r_frz_strict[0] = 3.0 * u_p
    r_frz_strict[2] = 2.5 * u_p  # top-2 on prose are {0, 2}, NOT 1!
    r_frz_strict[3] = 2.0 * u_p
    r_base_strict = r_frz_strict.clone()
    r_unf_strict = r_frz_strict.clone()
    r_unf_strict[1] = 6.0 * u_p + 3.0 * u_s  # unfrozen router forces expert 1 into top-2 on prose!

    decomp_router = decompose_router_vs_expert_prose_drift(
        r_base_strict,
        r_frz_strict,
        r_unf_strict,
        experts_router_only,
        x_prose,
        routing_mode="sigma_topk",
        top_k=2,
        spec_hidden_states=x_spec,
    )
    assert decomp_router["router_kl_drift"] > 0.1
    assert decomp_router["frozen_vs_base_router_kl_drift"] == pytest.approx(0.0, abs=1e-12)
    assert decomp_router["router_drift_share"] > 0.99
    assert decomp_router["expert_drift_share"] < 0.01
    assert decomp_router["dominant_prose_drift_source"] == "router_gate_shift"
    assert decomp_router["telescoping_max_abs_error"] < 1e-12
    assert decomp_router["per_expert_attribution"]["expert_1"]["gate_shift_on_prose"] > 0.3
    assert decomp_router["per_expert_attribution"]["expert_1"]["gate_shift_on_spec"] > 0.0

    # B. Pure expert delta drift (routers frozen identically across base, frz, unf) + exact NLL attribution
    r_frz_exp = r_base.clone()
    r_frz_exp[1] = 2.2 * u_p + 1.0 * u_s  # expert 1 is active on prose already at frozen router
    head_w = 0.2 * torch.randn((12, dim_h), generator=gen, dtype=torch.float32)
    prose_targets = torch.randint(0, 12, (x_prose.shape[0],), generator=gen)
    ln_params = {"final_ln": {"weight": torch.ones(dim_h), "bias": torch.zeros(dim_h)}}
    decomp_exp = decompose_router_vs_expert_prose_drift(
        r_frz_exp,
        r_frz_exp,
        r_frz_exp,
        experts_mixed,
        x_prose,
        routing_mode="softmax_topk",
        top_k=2,
        lm_head_weight=head_w,
        prose_targets=prose_targets,
        final_ln_params=ln_params,
    )
    assert decomp_exp["router_kl_drift"] == pytest.approx(0.0, abs=1e-12)
    assert decomp_exp["expert_drift_share"] == pytest.approx(1.0, abs=1e-12)
    assert decomp_exp["dominant_prose_drift_source"] == "expert_mlp_delta"
    nll_attr = decomp_exp["nll_attribution"]
    assert nll_attr is not None
    assert nll_attr["router_induced_nll_drift_nats"] == pytest.approx(0.0, abs=1e-12)
    assert (
        nll_attr["router_induced_nll_drift_nats"]
        + nll_attr["expert_induced_nll_drift_nats"]
        + nll_attr["frozen_anchor_nll_drift_nats"]
    ) == pytest.approx(nll_attr["total_prose_nll_drift_nats"], abs=1e-12)

    # Also test prose_nll_eval_fn branch and zero-drift branch
    decomp_fn = decompose_router_vs_expert_prose_drift(
        [r_base],
        [r_base],
        [r_base],
        [experts_zero],
        [x_prose],
        prose_nll_eval_fn=lambda state: 2.0 + float(state["routers"][0]["weight"].abs().mean().item()) * 0.0,
    )
    assert decomp_fn["dominant_prose_drift_source"] == "none"
    assert decomp_fn["nll_attribution"]["total_prose_nll_drift_nats"] == pytest.approx(0.0, abs=1e-12)

    # C. Exact P_0 invariance & OOF selective gating in graft_specialist_experts_onto_safe_router
    p_null, _ = compute_prose_nullspace_projector_scaled(x_prose, sv_rel_tol=1e-6, variance_threshold=1.0)
    q_perp = torch.eye(dim_h, dtype=torch.float64) - p_null

    # Non-zero base for expert 2 + clean specialist expert 0 + leaking expert 1
    experts_base_nz = copy.deepcopy(experts_zero)
    with torch.no_grad():
        experts_base_nz[2].w1_B[:dim_h, 0] = 0.2 * u_s_out
        experts_base_nz[2].w1_A[0] = 0.5 * u_s + 0.1 * u_p
        experts_mixed[2] = copy.deepcopy(experts_base_nz[2])
        experts_mixed[2].w1_A[0] = 0.9 * u_s + 0.7 * u_p
        experts_mixed[0] = copy.deepcopy(exp_clean)

    # Router row 1 has specialist + prose shift (passes OOF); row 2 has ONLY prose shift (rejected by P_0/OOF)
    r_unf_test = r_frz_strict.clone()
    r_unf_test[1] = r_frz_strict[1] + 3.0 * u_p + 2.5 * u_s
    r_unf_test[2] = r_frz_strict[2] + 2.5 * u_p  # pure prose shift -> must stay frozen!

    grafted, g_report = graft_specialist_experts_onto_safe_router(
        r_frz_strict,
        r_unf_test,
        experts_mixed,
        p_null,
        gate_blend_alpha=0.25,
        base_experts=experts_base_nz,
        prose_hidden_states=x_prose,
        spec_hidden_states=x_spec,
        protected_experts=(0,),
    )
    r_graft = grafted["routers"].to(torch.float64)
    dr_graft = r_graft - r_frz_strict.to(torch.float64)

    # Exact P_0 invariance on router delta and expert 1 delta
    assert float((dr_graft @ q_perp).abs().max().item()) < 1e-6
    assert g_report["max_router_nullspace_leak"] < 1e-6
    assert g_report["max_expert_nullspace_leak"] < 1e-6
    # Prose logits unchanged; specialist logits boosted for verified expert 1; row 2 untouched
    assert float((x_prose.to(torch.float64) @ dr_graft.T).abs().max().item()) < 1e-6
    assert float((x_spec.to(torch.float64) @ dr_graft[1]).mean().item()) > 0.2
    assert torch.equal(grafted["routers"][2], r_frz_strict[2])
    assert 1 in g_report["oof_verified_router_experts"]
    assert 1 in g_report["projected_experts"]
    assert g_report["per_layer"][0]["expert_actions"][0] == "preserved_anchor_bitwise"

    exp1_graft = grafted["experts"][1]
    assert float((exp1_graft.w1_A.to(torch.float64) @ q_perp).abs().max().item()) < 1e-6
    assert float((q_perp @ exp1_graft.w3_B.to(torch.float64)).abs().max().item()) < 1e-6

    # OOF rejection when specialist tokens contradict across folds
    x_spec_noisy = torch.cat([3.0 * u_s.unsqueeze(0), -3.0 * u_s.unsqueeze(0)] * 4, dim=0)
    grafted_rej, rej_rep = graft_specialist_experts_onto_safe_router(
        r_frz_strict,
        r_unf_test,
        experts_mixed,
        p_null,
        gate_blend_alpha=0.25,
        prose_hidden_states=x_prose,
        spec_hidden_states=x_spec_noisy,
    )
    assert rej_rep["oof_verified_router_experts"] == []
    assert torch.equal(grafted_rej["routers"], r_frz_strict)

    # Probe-less nullspace alignment check + explicit router projector + non-zero w3 base (output side)
    with torch.no_grad():
        experts_base_nz[2].w3_B[:, 0] = 0.25 * u_s_out + 0.12 * u_p_out
        experts_base_nz[2].w3_A[0, :dim_h] = 0.5 * u_s_out
        experts_mixed[2].w3_B[:, 0] = 0.45 * u_s_out + 0.35 * u_p_out
        experts_mixed[2].w3_A[0, :dim_h] = 0.8 * u_s_out
    p_non_idem = 0.9 * p_null + 0.05 * q_perp
    grafted_noprobe, np_rep = graft_specialist_experts_onto_safe_router(
        {"weight": r_frz_strict},
        {"weight": r_unf_test},
        experts_mixed,
        {"x": p_non_idem, "router": p_null},
        base_experts=experts_base_nz,
        gate_blend_alpha=0.20,
    )
    assert 1 in np_rep["oof_verified_router_experts"]
    assert 2 in np_rep["projected_experts"]
    assert torch.equal(grafted_noprobe["routers"]["weight"][2], r_frz_strict[2])


def test_s3_selective_ghost_sink_pruning_of_prose_corruptors() -> None:
    """S3: Selective ghost-sink pruning zeroes pure prose-corrupting experts while keeping specialists."""
    base_m, arm_d_m, arm_b_m, probe_chunks, eval_chunks, _ = _build_campaign3_cross_arm_trio(
        seed=20263333, include_sink_expert=True
    )

    # 1. Explicit sink_experts=(3,) on graft_specialist_experts_onto_safe_router
    grafted, g_rep = graft_specialist_experts_onto_safe_router(
        arm_d_m,
        arm_b_m,
        arm_b_m,
        None,
        gate_blend_alpha=0.15,
        sink_experts=(3,),
    )
    assert g_rep["sink_experts_pruned"] == [3]
    for l_idx in range(len(arm_b_m.blocks)):
        sink_exp = grafted["experts"][l_idx][3]
        assert torch.count_nonzero(sink_exp.w1_B).item() == 0
        assert torch.count_nonzero(sink_exp.w2_B).item() == 0
        assert torch.count_nonzero(sink_exp.w3_B).item() == 0
        assert torch.equal(
            grafted["routers"][l_idx]["weight"][3],
            arm_d_m.blocks[l_idx].mlp.router.weight[3],
        )
        # Specialist expert 6 stays non-zero!
        spec_exp = grafted["experts"][l_idx][6]
        assert torch.count_nonzero(spec_exp.w3_B).item() > 0

    # Verify apply_grafted_state_inplace updates a target model cleanly
    work_clone = copy.deepcopy(arm_d_m)
    apply_grafted_state_inplace(work_clone, grafted)
    assert torch.count_nonzero(work_clone.blocks[0].mlp.experts["3"].w3_B).item() == 0

    # 2. Automatic LOEO sink detection in execute_verified_cross_arm_moe_surgery
    # Make expert 3 active even under Arm D's router so LOEO catches it under unfrozen/frozen evaluation
    with torch.no_grad():
        for blk in arm_b_m.blocks:
            blk.mlp.router.weight[0, :24] = 1.0 * arm_b_m.blocks[0].mlp.router.weight[0, :24].sign()
    hybrid_m, receipt = execute_verified_cross_arm_moe_surgery(
        base_m,
        arm_d_m,
        arm_b_m,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        sink_experts=None,
        config=CrossArmSurgeryConfig(score_from=6, sink_prose_harm_floor_nats=0.001, inplace=False),
    )
    assert receipt["accepted"] is True
    assert 3 in receipt["graft_summary"]["sink_experts_pruned"]
    assert 6 not in receipt["graft_summary"]["sink_experts_pruned"]
    for blk in hybrid_m.blocks:
        assert torch.count_nonzero(blk.mlp.experts["3"].w3_B).item() == 0
        assert torch.count_nonzero(blk.mlp.experts["6"].w3_B).item() > 0


def test_s4_anse_maximum_pain_atomic_rollback_on_infeasible_guard() -> None:
    """S4: ANSE Maximum Pain forces atomic bitwise rollback when guard fails or donors are not beaten."""
    base_m, arm_d_m, arm_b_m, probe_chunks, eval_chunks, _ = _build_campaign3_cross_arm_trio(
        seed=20264444
    )
    frozen_sha = compute_checkpoint_sha256(arm_d_m)
    base_sha = compute_checkpoint_sha256(base_m)
    unfrozen_sha = compute_checkpoint_sha256(arm_b_m)

    # Case 1: Infeasible guard via custom eval_fn returning > +1.0% prose drift on candidate
    target_work = copy.deepcopy(arm_d_m)

    def _infeasible_eval(candidate: Any) -> dict[str, float]:
        doms = _eval_domain_nlls(candidate, eval_chunks, score_from=6)
        # If candidate differs from base, arm_d, and arm_b, simulate residual unmitigated E0 failure
        cand_sha = compute_checkpoint_sha256(candidate)
        if cand_sha not in (base_sha, frozen_sha, unfrozen_sha):
            doms["E0_general_fineweb"] = doms["E0_general_fineweb"] * 1.05
        return doms

    out_m, rec_fail = execute_verified_cross_arm_moe_surgery(
        base_m,
        arm_d_m,
        arm_b_m,
        target_model=target_work,
        probe_chunks=probe_chunks,
        eval_fn=_infeasible_eval,
        config=CrossArmSurgeryConfig(score_from=6, rollback_target="frozen_arm", inplace=True),
    )
    assert out_m is target_work
    assert rec_fail["accepted"] is False
    assert rec_fail["rolled_back"] is True
    assert rec_fail["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_fail["rollback_bitwise_verified"] is True
    assert rec_fail["cand_guard_ok"] is False
    assert rec_fail["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_fail["final_state_sha256"] == rec_fail["frozen_arm_state_sha256"]
    assert compute_checkpoint_sha256(target_work) == frozen_sha
    assert verify_cross_arm_receipt_provenance(rec_fail) is True

    # Tampering receipt invalidates SHA-256 provenance
    tampered = dict(rec_fail)
    tampered["accepted"] = True
    assert verify_cross_arm_receipt_provenance(tampered) is False

    # Case 2: Non-finite candidate output triggers Maximum Pain & rollback to base
    base_blk_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(base_m))
    frz_blk_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(arm_d_m))
    unf_blk_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(arm_b_m))
    _, rec_nan = execute_verified_cross_arm_moe_surgery(
        extract_checkpoint_blocks(base_m),
        extract_checkpoint_blocks(arm_d_m),
        extract_checkpoint_blocks(arm_b_m),
        eval_fn=lambda blocks: (
            {"E0_general": 2.0, "E6_spec": 2.5}
            if compute_checkpoint_sha256(blocks) == base_blk_sha
            else (
                {"E0_general": 2.0, "E6_spec": 2.2}
                if compute_checkpoint_sha256(blocks) == frz_blk_sha
                else (
                    {"E0_general": 2.5, "E6_spec": 1.4}
                    if compute_checkpoint_sha256(blocks) == unf_blk_sha
                    else {"E0_general": 2.0, "E6_spec": float("nan")}
                )
            )
        ),
        rollback_target="base",
        fold_ln_lambda=0.5,
    )
    assert rec_nan["accepted"] is False
    assert rec_nan["rolled_back"] is True
    assert rec_nan["rollback_target"] == "base"
    assert rec_nan["final_state_sha256"] == base_blk_sha
    assert rec_nan["cand_anse_energy"] == ANSE_MAXIMUM_PAIN

    # Case 3: Candidate passes guard, yet fails to beat Arm D (frozen_arm) -> rollback to unfrozen_arm check
    _, rec_tie = execute_verified_cross_arm_moe_surgery(
        extract_checkpoint_blocks(base_m),
        extract_checkpoint_blocks(arm_d_m),
        extract_checkpoint_blocks(arm_b_m),
        eval_fn=lambda blocks: (
            {"E0_general": 2.0, "E6_spec": 2.5}
            if compute_checkpoint_sha256(blocks) == base_blk_sha
            else (
                {"E0_general": 2.0, "E6_spec": 1.6}
                if compute_checkpoint_sha256(blocks) == frz_blk_sha
                else (
                    {"E0_general": 2.4, "E6_spec": 1.5}
                    if compute_checkpoint_sha256(blocks) == unf_blk_sha
                    else {"E0_general": 2.0, "E6_spec": 1.9}
                )
            )
        ),
        rollback_target="unfrozen_arm",
    )
    assert rec_tie["accepted"] is False
    assert rec_tie["delta_e_vs_frozen_arm"] > 0.0
    assert rec_tie["final_state_sha256"] == unf_blk_sha


def test_s5_validation_errors_multi_formats_and_security_stub_gates() -> None:
    """S5: Input validation, FineGrained/dense adapter paths, and security/stub compliance gates."""
    # 1. CrossArmSurgeryConfig & provenance validation
    with pytest.raises(ValueError, match="<= 1.0"):
        CrossArmSurgeryConfig(gate_blend_alpha=1.5)
    with pytest.raises(TypeError, match="real number"):
        CrossArmSurgeryConfig(gate_blend_alpha=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="routing_mode"):
        CrossArmSurgeryConfig(routing_mode="bad")
    with pytest.raises(ValueError, match=">= 1"):
        CrossArmSurgeryConfig(top_k=0)
    with pytest.raises(TypeError, match="bool"):
        CrossArmSurgeryConfig(inplace="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        CrossArmSurgeryConfig(top_k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        CrossArmSurgeryConfig(score_from=-1)
    with pytest.raises(ValueError, match=">= 1"):
        CrossArmSurgeryConfig(batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        CrossArmSurgeryConfig(min_oof_snr=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        CrossArmSurgeryConfig(soft_margin_penalty=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        CrossArmSurgeryConfig(maximum_pain_energy=0.0)
    with pytest.raises(ValueError, match="> 0.0"):
        CrossArmSurgeryConfig(nullspace_variance_threshold=0.0)
    with pytest.raises(ValueError, match="nullspace_eig_threshold"):
        CrossArmSurgeryConfig(nullspace_eig_threshold=1.0)
    with pytest.raises(ValueError, match="n_folds"):
        CrossArmSurgeryConfig(n_folds=1)
    with pytest.raises(ValueError, match=">= 1.0"):
        CrossArmSurgeryConfig(max_general_drift_ratio=0.9)
    with pytest.raises(ValueError, match="rollback_target"):
        CrossArmSurgeryConfig(rollback_target="invalid")
    with pytest.raises(ValueError, match="general_prefix"):
        CrossArmSurgeryConfig(general_prefix="   ")
    assert CrossArmSurgeryConfig(fold_ln_lambda=0.5).fold_ln_lambda == 0.5

    with pytest.raises(TypeError, match="mapping"):
        verify_cross_arm_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_cross_arm_receipt_provenance({"receipt_sha256": "short"}) is False
    assert verify_cross_arm_receipt_provenance({"receipt_sha256": "0" * 64, "object": "x"}) is False
    assert (
        verify_cross_arm_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": RECEIPT_OBJECT,
                "schema": SCHEMA_VERSION,
                "val": float("nan"),
            }
        )
        is False
    )

    # 2. Decompose & graft validation + FineGrained / SwiGLU / linear tensor / dict paths
    r_ok = torch.eye(4)[:2, :]
    x_ok = torch.ones((4, 4))
    exp_dict_ok = {0: {"w": torch.eye(4)}, 1: {"w": torch.eye(4)}}
    with pytest.raises(ValueError, match="routing_mode"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, x_ok, routing_mode="bad")
    with pytest.raises(TypeError):
        decompose_router_vs_expert_prose_drift(123, r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(TypeError, match="bool"):
        decompose_router_vs_expert_prose_drift({True: r_ok}, {0: r_ok}, {0: r_ok}, {0: exp_dict_ok}, x_ok)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="Invalid layer key"):
        decompose_router_vs_expert_prose_drift({"bad_layer": r_ok}, {0: r_ok}, {0: r_ok}, {0: exp_dict_ok}, x_ok)
    with pytest.raises(ValueError, match="empty"):
        decompose_router_vs_expert_prose_drift({}, r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(ValueError, match="empty"):
        decompose_router_vs_expert_prose_drift([], r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(ValueError, match="non-empty 2D"):
        decompose_router_vs_expert_prose_drift(torch.ones((1, 4)), r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(ValueError, match="finite"):
        bad_r = r_ok.clone()
        bad_r[0, 0] = float("nan")
        decompose_router_vs_expert_prose_drift(bad_r, r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(ValueError, match="bias shape"):
        decompose_router_vs_expert_prose_drift(
            {"weight": r_ok, "bias": torch.ones(5)}, r_ok, r_ok, exp_dict_ok, x_ok
        )
    with pytest.raises(ValueError, match="bias presence"):
        decompose_router_vs_expert_prose_drift(
            {"weight": r_ok, "bias": torch.zeros(2)}, r_ok, r_ok, exp_dict_ok, x_ok
        )
    with pytest.raises(ValueError, match="must match"):
        decompose_router_vs_expert_prose_drift({0: r_ok}, {1: r_ok}, {0: r_ok}, {0: exp_dict_ok}, x_ok)
    with pytest.raises(ValueError, match="shape mismatch"):
        decompose_router_vs_expert_prose_drift(r_ok, torch.eye(4)[:3, :], r_ok, exp_dict_ok, x_ok)
    with pytest.raises(ValueError, match="feature width"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, torch.ones((4, 8)))
    with pytest.raises(ValueError, match="non-empty 2D or 3D"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, torch.ones((0, 4)))
    with pytest.raises(TypeError, match="floating-point"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, torch.ones((4, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        decompose_router_vs_expert_prose_drift(
            r_ok, r_ok, r_ok, exp_dict_ok, torch.tensor([[float("nan"), 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        )
    with pytest.raises(ValueError, match="missing hidden states"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, {9: x_ok})
    with pytest.raises(ValueError, match="sequence length"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, [x_ok, x_ok])
    with pytest.raises(TypeError, match="Tensor"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lm_head_weight"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, x_ok, lm_head_weight=torch.ones(5, 4))
    with pytest.raises(TypeError, match="callable"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, x_ok, prose_nll_eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length"):
        decompose_router_vs_expert_prose_drift(
            r_ok, r_ok, r_ok, exp_dict_ok, x_ok, lm_head_weight=torch.ones(5, 4), prose_targets=torch.zeros(2, dtype=torch.long)
        )

    # Single-layer CrowdLoRAMoEMLP, nested layer maps, SwiGLU nn.Module, linear tensors, explicit base_mlps
    class _SwiGLUMod(nn.Module):
        def __init__(self, d_in: int = 4, d_mid: int = 6) -> None:
            super().__init__()
            self.w1 = nn.Linear(d_in, d_mid, bias=False)
            self.w2 = nn.Linear(d_in, d_mid, bias=False)
            self.w3 = nn.Linear(d_mid, d_in, bias=False)

    sw_m = _SwiGLUMod()
    dec_lin = decompose_router_vs_expert_prose_drift(
        {"layer_0": r_ok},
        {"layer_0": r_ok},
        {"layer_0": r_ok},
        {"layer_0": {0: sw_m, 1: torch.eye(4)}},
        {"blocks.0": x_ok},
        base_experts={"layer_0": {0: _SwiGLUMod(), 1: torch.zeros((4, 4))}},
        base_mlps={"layer_0": sw_m},
    )
    assert dec_lin["num_experts"] == 2

    # Fine-grained SwiGLU MoE + LoRA MoE MLP direct calls
    fg_cfg = CrowdV1RuntimeConfig(
        vocab_size=16, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, intermediate_size=16, max_position_embeddings=16,
        engram_buckets=0, tie_word_embeddings=True,
    )
    fg_m = CrowdV1WithEngram(fg_cfg)
    upcycle_crowd_v1_to_sparse_moe(fg_m, split_mode="fine_grained_width_split", num_experts=2, top_k=1)
    fg_unf = copy.deepcopy(fg_m)
    with torch.no_grad():
        fg_unf.blocks[0].mlp.router.weight[1, 4:] += 1.5
        fg_unf.blocks[0].mlp.experts_w3[1].weight[:4, :] += 0.2
    x_fg = torch.randn((6, 8), generator=torch.Generator(device="cpu").manual_seed(9))
    dec_fg = decompose_router_vs_expert_prose_drift(fg_m, fg_m, fg_unf, fg_unf, x_fg, top_k=1)
    assert dec_fg["num_experts"] == 2

    lora_single = CrowdV1WithEngram(fg_cfg)
    upcycle_crowd_v1_to_sparse_moe(lora_single, split_mode="lora_domain_experts", num_experts=2, top_k=1, lora_rank=2)
    mlp_s = lora_single.blocks[0].mlp
    dec_single = decompose_router_vs_expert_prose_drift(mlp_s, mlp_s, mlp_s, mlp_s, x_fg, top_k=1)
    assert dec_single["router_kl_drift"] == 0.0

    p_fg = torch.diag(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
    gr_fg, gr_fg_rep = graft_specialist_experts_onto_safe_router(
        fg_m, fg_unf, fg_unf, {0: {1: p_fg}}, base_experts=fg_m, sink_experts=(0,)
    )
    assert 1 in gr_fg_rep["projected_experts"]
    assert gr_fg["experts"][0][0]["w1"].shape == (8, 8)

    # Nested layer mapping + SwiGLU module + 1D/2D tensor expert grafting + linear routers
    lin_r = nn.Linear(4, 2, bias=True)
    gr_lin, _ = graft_specialist_experts_onto_safe_router(
        [lin_r],
        [lin_r],
        [{0: torch.ones((4, 4)), 1: {"v": torch.ones(4), "w_rect": torch.ones((2, 4))}}],
        p_fg[:4, :4],
        base_experts=[{0: torch.zeros((4, 4)), 1: {"v": torch.zeros(4), "w_rect": torch.zeros((2, 4))}}],
    )
    assert isinstance(gr_lin["routers"], list)
    gr_nested, _ = graft_specialist_experts_onto_safe_router(
        {"blocks.0.mlp.router": r_ok},
        {"blocks.0.mlp.router": r_ok},
        {"blocks.0": {"expert_0": sw_m, "expert_1": torch.ones((4, 2))}},
        {"blocks.0.mlp.experts.0": {"x": p_fg[:4, :4], "act": torch.eye(6)}, "default": p_fg[:4, :4]},
        base_experts={"blocks.0": {"expert_0": _SwiGLUMod(), "expert_1": torch.zeros((4, 2))}},
    )
    assert "blocks.0" in gr_nested["experts"]

    with pytest.raises(ValueError, match="sink_experts and protected_experts"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, None, sink_experts=(0,), protected_experts=(0,))
    with pytest.raises(TypeError, match="sequence or set"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, None, sink_experts="0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both be provided"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, None, prose_hidden_states=x_ok)
    with pytest.raises(ValueError, match="Projector shape"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, torch.eye(9))
    with pytest.raises(TypeError, match="p_null_by_expert"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="symmetric"):
        graft_specialist_experts_onto_safe_router(
            r_ok, r_ok, exp_dict_ok, torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        )
    with pytest.raises(TypeError, match="grafted_state"):
        apply_grafted_state_inplace(fg_m, {})

    # 3. Execute validation & eval_fn error paths
    blocks_fg = extract_checkpoint_blocks(fg_m)
    with pytest.raises(TypeError, match="CrossArmSurgeryConfig"):
        execute_verified_cross_arm_moe_surgery(blocks_fg, blocks_fg, blocks_fg, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_verified_cross_arm_moe_surgery(blocks_fg, blocks_fg, blocks_fg, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_model"):
        execute_verified_cross_arm_moe_surgery(blocks_fg, blocks_fg, blocks_fg, target_model="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Either an nn.Module"):
        execute_verified_cross_arm_moe_surgery(blocks_fg, blocks_fg, blocks_fg)
    with pytest.raises(TypeError, match="nn.Module or non-empty block Mapping"):
        execute_verified_cross_arm_moe_surgery("bad", blocks_fg, blocks_fg, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layouts must match"):
        execute_verified_cross_arm_moe_surgery(blocks_fg, {"other": {"w": torch.eye(2)}}, blocks_fg, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})
    with pytest.raises(ValueError, match="non-empty"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m, eval_fn=lambda _: {})
    with pytest.raises(TypeError, match="numeric"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m, eval_fn=lambda _: {"E0": "bad", "E1": 1.0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite evaluation NLL"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0, "non_finite_outputs": True})
    with pytest.raises(ValueError, match="probe_chunks"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m, probe_chunks=[])
    with pytest.raises(ValueError, match="mapping containing"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m, probe_chunks=[{"expert_id": "E0"}])
    with pytest.raises(ValueError, match="non-empty"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m, probe_chunks=[{"expert_id": " ", "ids": [0, 1]}])
    with pytest.raises(ValueError, match="length >= 2"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m, probe_chunks=[{"expert_id": "E0", "ids": [0]}])
    with pytest.raises(ValueError, match="lie in"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m, probe_chunks=[{"expert_id": "E0", "ids": [99, 1]}])
    with pytest.raises(ValueError, match="sequence length"):
        execute_verified_cross_arm_moe_surgery(
            fg_m, fg_m, fg_m, probe_chunks=[{"expert_id": "E0", "ids": [0, 1]}, {"expert_id": "E1", "ids": [0, 1, 2]}]
        )
    with pytest.raises(ValueError, match="too short"):
        execute_verified_cross_arm_moe_surgery(
            fg_m, fg_m, fg_m, probe_chunks=[{"expert_id": "E0", "ids": [0, 1]}, {"expert_id": "E1", "ids": [2, 3]}]
        )
    with pytest.raises(ValueError, match="general .* and specialist"):
        execute_verified_cross_arm_moe_surgery(
            fg_m, fg_m, fg_m, eval_fn=lambda _: {"E0_only": 1.0}
        )
    with pytest.raises(ValueError, match="probe_chunks/eval_chunks or eval_fn"):
        execute_verified_cross_arm_moe_surgery(fg_m, fg_m, fg_m)

    # 3b. Biased linear routers, missing/rectangular/1D experts, non-zero P_0 LoRA base, LN/LOEO/NaN execute branches
    dec_biased = decompose_router_vs_expert_prose_drift(
        lin_r,
        lin_r,
        lin_r,
        {0: {"w": torch.ones((2, 4)), "w_t": torch.ones((4, 2)), "vec": torch.ones(4)}},
        x_ok,
        base_experts={0: {"w": torch.zeros((2, 4)), "w_t": torch.zeros((4, 2)), "vec": torch.zeros(4)}},
    )
    assert dec_biased["num_experts"] == 2

    p4_half = torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    lora_b0 = {
        "w1_A": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        "w1_B": torch.ones((6, 1)),
        "w2_A": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "w2_B": torch.ones((6, 1)),
        "w3_A": torch.ones((1, 6)),
        "w3_B": torch.tensor([[0.0], [0.0], [1.0], [0.5]]),
    }
    lora_u1 = {
        "w1_A": torch.tensor([[1.0, 0.5, 1.0, 0.0]]),
        "w1_B": torch.ones((6, 1)),
        "w2_A": torch.tensor([[0.5, 1.0, 0.0, 1.0]]),
        "w2_B": torch.ones((6, 1)),
        "w3_A": torch.ones((1, 6)),
        "w3_B": torch.tensor([[0.7], [0.4], [1.0], [0.5]]),
    }
    gr_nonzero_b0, rep_nz = graft_specialist_experts_onto_safe_router(
        r_ok, r_ok, {0: lora_u1, 1: exp_dict_ok[1]}, {0: p4_half, 1: torch.eye(4)}, base_experts={0: lora_b0, 1: exp_dict_ok[1]}
    )
    assert 0 in rep_nz["projected_experts"]
    assert "experts" in gr_nonzero_b0

    r_ok3 = torch.eye(4)[:3, :]
    graft_specialist_experts_onto_safe_router(
        r_ok3,
        r_ok3,
        {0: lora_u1, 1: sw_m, 2: {"v": torch.tensor([0.0, 0.0, 0.2, 0.1])}},
        {"blocks.0.mlp.experts.0": {"act": torch.eye(6)}, "blocks.0.mlp.experts.2": p4_half},
        base_experts={0: lora_b0, 1: _SwiGLUMod(), 2: {"v": torch.zeros(4)}},
        nullspace_leak_tol=1.0,
    )
    graft_specialist_experts_onto_safe_router(
        r_ok[:2, :], r_ok[:2, :], {0: lora_u1, 1: sw_m}, {}
    )
    graft_specialist_experts_onto_safe_router(
        r_ok[:2, :], r_ok[:2, :], {0: lora_u1, 1: sw_m}, torch.zeros((4, 4)), nullspace_eig_threshold=0.5
    )

    fg_probes = [
        {"expert_id": "E0_general", "ids": [0, 1, 2, 3, 4, 5]},
        {"expert_id": "E0_general", "ids": [5, 4, 3, 2, 1, 0]},
        {"expert_id": "E1_spec", "ids": [0, 2, 4, 1, 3, 5]},
        {"expert_id": "E1_spec", "ids": [1, 3, 5, 0, 2, 4]},
    ]
    fg_exec_cfg = CrossArmSurgeryConfig(
        score_from=1,
        batch_size=2,
        n_folds=2,
        max_general_drift_ratio=1.001,
        require_beat_both_donors=False,
    )
    _, rec_fg_exec = execute_verified_cross_arm_moe_surgery(
        fg_m,
        blocks_fg,
        fg_unf,
        probe_chunks=fg_probes,
        sink_experts=None,
        config=fg_exec_cfg,
    )
    assert rec_fg_exec["schema"] == SCHEMA_VERSION

    _, rec_unf_mod = execute_verified_cross_arm_moe_surgery(
        blocks_fg,
        blocks_fg,
        fg_unf,
        probe_chunks=fg_probes,
        sink_experts=(0,),
        config=fg_exec_cfg,
        eval_fn=(lambda m, _st={"c": 0}: (
            _st.__setitem__("c", _st["c"] + 1)
            or ({"E0_general": 1.0, "E1_spec": 1.0} if _st["c"] == 1 else {"E0_general": 1.0, "E1_spec": 1.0, "all_finite": False})
        )),
    )
    assert rec_unf_mod["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"

    blocks_no_ln = {k: v for k, v in blocks_fg.items() if k != "final_ln"}
    _, rec_no_ln = execute_verified_cross_arm_moe_surgery(
        blocks_no_ln,
        blocks_no_ln,
        blocks_no_ln,
        target_model=fg_m,
        sink_experts=(0,),
        eval_fn=lambda _: {"E0_general": 1.0, "E1_spec": 1.0},
    )
    assert rec_no_ln["fold_ln_lambda"] == 0.0

    # Non-finite hidden / NLL candidate (triggers fallback LN calibration & NaN guards) and mismatched domain keys
    huge_hp_unf = copy.deepcopy(fg_unf)
    with torch.no_grad():
        for e_ix in range(2):
            huge_hp_unf.blocks[0].mlp.experts_w1[e_ix].weight.fill_(1e20)
            huge_hp_unf.blocks[0].mlp.experts_w2[e_ix].weight.fill_(1e20)
            huge_hp_unf.blocks[0].mlp.experts_w3[e_ix].weight.fill_(1e20)
    _, rec_huge_hp = execute_verified_cross_arm_moe_surgery(
        fg_m,
        fg_m,
        huge_hp_unf,
        probe_chunks=fg_probes,
        sink_experts=(0,),
        fold_ln_lambda=None,
        config=fg_exec_cfg,
    )
    assert rec_huge_hp["unfrozen_arm_anse_energy"] == 1_000_000.0

    huge_ln_target = copy.deepcopy(fg_m)
    huge_ln_unf = copy.deepcopy(fg_unf)
    with torch.no_grad():
        huge_ln_target.head.weight[:, ::2].fill_(10.0)
        huge_ln_target.head.weight[:, 1::2].fill_(-10.0)
        huge_ln_unf.ln.weight.fill_(1e38)
    _, rec_huge_ln = execute_verified_cross_arm_moe_surgery(
        fg_m,
        fg_m,
        huge_ln_unf,
        target_model=huge_ln_target,
        probe_chunks=fg_probes,
        sink_experts=(0,),
        fold_ln_lambda=1.0,
        config=fg_exec_cfg,
    )
    assert rec_huge_ln["unfrozen_arm_anse_energy"] == 1_000_000.0

    _, rec_mismatched = execute_verified_cross_arm_moe_surgery(
        fg_m,
        fg_m,
        fg_unf,
        eval_fn=(lambda m, _st={"c": 0}: (
            _st.__setitem__("c", _st["c"] + 1)
            or ({"E0_general": 1.0, "E1_spec": 1.0} if _st["c"] == 1 else {"E0_general": 1.0, "E2_spec": 1.0})
        )),
    )
    assert rec_mismatched["frozen_arm_anse_energy"] == 1_000_000.0

    bad_ln_blocks = copy.deepcopy(blocks_fg)
    bad_ln_blocks["final_ln"]["weight"].fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        execute_verified_cross_arm_moe_surgery(blocks_fg, bad_ln_blocks, blocks_fg, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})

    # Additional validator paths across module helpers
    class _EmptyBlocks(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList()

    class _NoRouterBlk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4)])

    with pytest.raises(TypeError, match="torch.Tensor"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, {"blocks.0": "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="empty"):
        decompose_router_vs_expert_prose_drift(_EmptyBlocks(), r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(ValueError, match="exposes no 'router'"):
        decompose_router_vs_expert_prose_drift(_NoRouterBlk(), r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(TypeError, match="exposes no blocks or router"):
        decompose_router_vs_expert_prose_drift(sw_m, r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(ValueError, match="Duplicate layer index"):
        decompose_router_vs_expert_prose_drift({0: r_ok, "0": r_ok}, r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(TypeError, match="bias"):
        decompose_router_vs_expert_prose_drift({"weight": r_ok, "bias": "bad"}, r_ok, r_ok, exp_dict_ok, x_ok)
    with pytest.raises(TypeError, match="entries must be"):
        decompose_router_vs_expert_prose_drift(["bad"], r_ok, r_ok, exp_dict_ok, x_ok)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="bias.*finite"):
        decompose_router_vs_expert_prose_drift(
            {"weight": r_ok, "bias": torch.tensor([0.0, float("nan")])},
            {"weight": r_ok, "bias": torch.zeros(2)},
            {"weight": r_ok, "bias": torch.zeros(2)},
            exp_dict_ok,
            x_ok,
        )
    with pytest.raises(TypeError, match="[Ee]xposes no MoE experts"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, sw_m, x_ok)
    with pytest.raises(TypeError, match="[Ee]xposes no MoE experts"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, _NoRouterBlk(), x_ok)
    with pytest.raises(ValueError, match="layer count"):
        decompose_router_vs_expert_prose_drift({0: r_ok, 1: r_ok}, {0: r_ok, 1: r_ok}, {0: r_ok, 1: r_ok}, fg_m, x_ok)
    with pytest.raises(ValueError, match="Single-layer"):
        decompose_router_vs_expert_prose_drift({0: r_ok, 1: r_ok}, {0: r_ok, 1: r_ok}, {0: r_ok, 1: r_ok}, mlp_s, x_fg[:4, :4])
    with pytest.raises(ValueError, match="empty"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {}, x_ok)
    with pytest.raises(ValueError, match="layer count"):
        decompose_router_vs_expert_prose_drift({0: r_ok, 1: r_ok}, {0: r_ok, 1: r_ok}, {0: r_ok, 1: r_ok}, {"blocks.0.mlp.experts.0": torch.eye(4)}, x_ok)
    with pytest.raises(ValueError, match="Single-layer"):
        decompose_router_vs_expert_prose_drift({0: r_ok, 1: r_ok}, {0: r_ok, 1: r_ok}, {0: r_ok, 1: r_ok}, exp_dict_ok, x_ok)
    with pytest.raises(ValueError, match="num_experts"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {5: torch.eye(4)}, x_ok)
    with pytest.raises(TypeError, match="non-empty expert mapping"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {"layer_0": {}}, x_ok)
    with pytest.raises(ValueError, match="num_experts"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {"layer_0": {9: {"w": torch.eye(4)}}}, x_ok)
    with pytest.raises(ValueError, match="layer count"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {"layer_0": exp_dict_ok, "layer_1": exp_dict_ok}, x_ok)
    with pytest.raises(ValueError, match="sequence length"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, [exp_dict_ok, exp_dict_ok], x_ok)
    with pytest.raises(TypeError, match="non-empty mapping of experts"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, [{}], x_ok)
    with pytest.raises(TypeError, match="MoE module, Mapping, or Sequence"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, "bad", x_ok)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {True: torch.eye(4)}, x_ok)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="Invalid expert key"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {"bad_exp": torch.eye(4)}, x_ok)
    with pytest.raises(TypeError, match="LoRA or SwiGLU"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: lin_r}, x_ok)
    with pytest.raises(TypeError, match="torch.Tensor"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: {0: {"scaling": 2.0, "w": "bad"}}}, x_ok)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-empty strings"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: {"": torch.eye(4)}}, x_ok)
    with pytest.raises(ValueError, match="at least one parameter"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: {0: {"scaling": 2.0}}}, x_ok)
    with pytest.raises(ValueError, match="1D/2D tensor"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: torch.ones((2, 2, 2))}, x_ok)
    with pytest.raises(ValueError, match="finite and floating-point"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: torch.ones((4, 4), dtype=torch.long)}, x_ok)
    with pytest.raises(ValueError, match="input dim"):
        bad_lora_a = dict(lora_b0, w1_A=torch.ones((1, 5)))
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: bad_lora_a}, x_ok)
    with pytest.raises(ValueError, match="output dim"):
        bad_lora_b = dict(lora_b0, w3_B=torch.ones((5, 1)))
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: bad_lora_b}, x_ok)
    with pytest.raises(ValueError, match="input dim"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: dict(w1=torch.ones((6, 5)), w2=torch.ones((6, 4)), w3=torch.ones((4, 6)))}, x_ok)
    with pytest.raises(ValueError, match="output dim"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: dict(w1=torch.ones((6, 4)), w2=torch.ones((6, 4)), w3=torch.ones((5, 6)))}, x_ok)
    with pytest.raises(ValueError, match="1D length"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: torch.ones(9)}, x_ok)
    with pytest.raises(ValueError, match="incompatible with hidden_size"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, {0: torch.ones((5, 5))}, x_ok)
    with pytest.raises(TypeError, match="Module or Mapping"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, x_ok, base_mlps={0: "bad"})
    with pytest.raises(TypeError, match="base_mlps must be"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, x_ok, base_mlps="bad")
    with pytest.raises(ValueError, match="lm_head_weight must be a 2D"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, x_ok, lm_head_weight=torch.ones(4), prose_targets=torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="lm_head_weight width"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, x_ok, lm_head_weight=torch.ones(5, 5), prose_targets=torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="1D integer tensor"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, x_ok, lm_head_weight=torch.ones(5, 4), prose_targets=torch.zeros((2, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="empty"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, {})
    with pytest.raises(ValueError, match="empty"):
        decompose_router_vs_expert_prose_drift(r_ok, r_ok, r_ok, exp_dict_ok, [])
    with pytest.raises(ValueError, match="square 2D"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, {"default": torch.ones((2, 4))})
    with pytest.raises(TypeError, match="floating-point"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, {"default": torch.eye(4, dtype=torch.long)})
    with pytest.raises(ValueError, match="shape"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, {"default": {"x": torch.eye(5)}})
    with pytest.raises(ValueError, match="finite"):
        bad_p = torch.eye(4)
        bad_p[0, 0] = float("nan")
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, bad_p)
    with pytest.raises(ValueError, match="nullspace_eig_threshold"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, None, nullspace_eig_threshold=1.0)
    with pytest.raises(ValueError, match="n_folds"):
        graft_specialist_experts_onto_safe_router(r_ok, r_ok, exp_dict_ok, None, n_folds=1)
    with pytest.raises(ValueError, match="n_folds"):
        graft_specialist_experts_onto_safe_router(
            r_ok, 2.0 * r_ok, exp_dict_ok, None, prose_hidden_states=x_ok[:1], spec_hidden_states=x_ok[:1], n_folds=2
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        execute_verified_cross_arm_moe_surgery({"": {"w": torch.eye(2)}}, blocks_fg, blocks_fg, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})
    with pytest.raises(TypeError, match="non-empty Mapping"):
        execute_verified_cross_arm_moe_surgery({"b": {}}, blocks_fg, blocks_fg, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})
    with pytest.raises(ValueError, match="floating-point Tensor"):
        execute_verified_cross_arm_moe_surgery({"b": {"w": torch.ones(2, dtype=torch.long)}}, blocks_fg, blocks_fg, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})
    with pytest.raises(ValueError, match="Checkpoints must expose MoE"):
        b_no_moe = {"only.w": {"weight": torch.eye(2)}}
        execute_verified_cross_arm_moe_surgery(b_no_moe, b_no_moe, b_no_moe, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})
    with pytest.raises(ValueError, match="Sub-key mismatch"):
        b2 = copy.deepcopy(blocks_fg)
        b2["final_ln"]["extra"] = torch.ones(2)
        execute_verified_cross_arm_moe_surgery(blocks_fg, b2, blocks_fg, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})
    with pytest.raises(ValueError, match="shape mismatch"):
        b3 = copy.deepcopy(blocks_fg)
        b3["final_ln"]["weight"] = torch.ones(16)
        execute_verified_cross_arm_moe_surgery(blocks_fg, b3, blocks_fg, eval_fn=lambda _: {"E0": 1.0, "E1": 1.0})
    with pytest.raises(ValueError, match="general .* and specialist"):
        execute_verified_cross_arm_moe_surgery(
            fg_m,
            fg_m,
            fg_m,
            probe_chunks=[{"expert_id": "E0_general", "ids": [0, 1, 2, 3]}, {"expert_id": "E0_general", "ids": [3, 2, 1, 0]}],
            config=CrossArmSurgeryConfig(score_from=1),
        )

    # 4. AGPL-3.0 header, ZeroStubAudit (CI-safe), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn36", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn36", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/router_freeze_cross_arm_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_router_freeze_cross_arm_surgeon_scenarios.py", TEST_FILE_PATH),
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
            "nll_mean": 1.82,
            "router_kl_drift": 0.24,
            "specialist_gain_nats": 1.35,
            "anse_energy": -0.31,
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
        pr_number=1236,
        pr_title="feat(swarm): post-DiLoCo router-unfrozen vs router-frozen cross-arm synergy surgeon for open-weight MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-36 post-DiLoCo cross-arm synergy surgeon with router-vs-expert KL decomposition, P_0 grafting, OOF selective gating, and ANSE rollback.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): post-DiLoCo router-unfrozen vs router-frozen cross-arm synergy surgeon for open-weight MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
    assert math.isfinite(float(telemetry_json.count(".")))
